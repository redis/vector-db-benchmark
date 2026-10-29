//! Streaming upload: feed an engine's parallel uploaders from disk, a batch at
//! a time, instead of materialising the corpus in client RAM (#342).
//!
//! # Which reader an upload uses
//!
//! [`UploadReader`] is the `--upload-reader` flag. [`plan`] turns it and the
//! dataset into a decision:
//!
//! | flag | compound (`tar`) dataset | anything else |
//! |---|---|---|
//! | `memory` | in-memory reader | in-memory reader |
//! | `stream` | streaming reader (error if `vectors.npy` is not streamable) | error |
//! | `auto` (default) | streaming only when the in-memory reader could not hold the corpus (see [`auto_streams`]), else in-memory | in-memory reader |
//!
//! `auto` streams a compound corpus only when loading it is not an option:
//! when its vectors reach [`AUTO_STREAM_MIN_BYTES`] (128 GiB — above the 10M
//! MS MARCO variant's 41 GB, below the 100M's ~410 GB), or when the in-memory
//! reader's measured peak, [`IN_MEMORY_PEAK_FACTOR`] x the vector bytes, exceeds
//! the host's `MemAvailable`. Everything else — every HDF5/JSONL dataset, the
//! synthetic fixtures, and the 100K, 1M and 10M MS MARCO variants on a host
//! that can load them — takes exactly the code path it took before this module
//! existed, so their `upload_time` keeps measuring what it measured: on the
//! streaming path the disk read and decode run inside the upload window, on the
//! in-memory path they are `read_time`. The two readers deliver the same rows
//! (see `readers::stream_reader`), so the switch changes client memory and the
//! read/upload time split, never what is uploaded. Only engines that implement the streaming path (Redis,
//! OpenSearch) honour `stream`; the others refuse it rather than silently
//! loading the corpus (`Dataset::read_vectors`).
//!
//! # Client memory bound
//!
//! [`upload_stream`] runs `parallel` workers. Each one, in a loop, takes the
//! stream lock, reads ONE raw batch (`batch_size` rows of `vectors.npy` bytes
//! and `payloads.jsonl` lines), releases the lock, decodes it, hands it to the
//! engine, and drops it before reading the next. So at most `parallel` batches
//! exist at any time, and the corpus size does not appear in the bound:
//!
//! ```text
//! peak ≈ parallel × batch_size × per_row + 2 × STREAM_READ_BUFFER
//! per_row = 4·dim            raw vector bytes (freed by decode)
//!         + 4·dim + 24       decoded Vec<f32>
//!         + 16·dim           engine wire encoding (JSON floats, worst case)
//!         + 4·payload_bytes  raw line + parsed MetadataItem + encoded fields
//! ```
//!
//! [`memory_bound_bytes`] computes it. For MS MARCO (dim 1024, ~1.8 KB payload
//! lines) per_row is ~32 KB, so `parallel 32 × batch 1000` is ~1.0 GB and
//! `parallel 16 × batch 500` is ~260 MB, at 100K passages or 100M. The
//! `streaming_memory` integration suite measures the reader + driver with a
//! counting allocator and fails if the peak grows with the corpus.

use std::path::Path;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::readers::stream_reader::{
    f32_npy_layout, streamable_layout, Batch, CompoundStream, F32NpyLayout, STREAM_READ_BUFFER,
};

/// `--upload-reader`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UploadReader {
    /// Stream compound datasets the in-memory reader could not hold
    /// ([`auto_streams`]).
    #[default]
    Auto,
    /// Always stream (compound datasets only; refused elsewhere).
    Stream,
    /// Always load the corpus into memory (the pre-#342 behaviour).
    Memory,
}

impl UploadReader {
    pub fn as_str(self) -> &'static str {
        match self {
            UploadReader::Auto => "auto",
            UploadReader::Stream => "stream",
            UploadReader::Memory => "memory",
        }
    }
}

impl FromStr for UploadReader {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(UploadReader::Auto),
            "stream" => Ok(UploadReader::Stream),
            "memory" => Ok(UploadReader::Memory),
            other => Err(format!(
                "unknown upload reader {other:?}; expected auto, stream or memory"
            )),
        }
    }
}

impl std::fmt::Display for UploadReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Vector bytes at and above which `auto` always streams: 128 GiB. The 100K
/// (0.4 GB), 1M (4.1 GB) and 10M (41 GB) MS MARCO variants are under it, so
/// their published upload numbers keep the in-memory reader; the 100M
/// (~410 GB) is over it on any host.
pub const AUTO_STREAM_MIN_BYTES: u64 = 128 << 30;

/// The in-memory reader's peak over the vector bytes: `read_npy_vectors`
/// holds the `Array2` and the `Vec<Vec<f32>>` it converts into at once
/// (measured: 80.5 GB for the 10M's 41 GB of vectors).
pub const IN_MEMORY_PEAK_FACTOR: u64 = 2;

/// Memory this process can still use, and how that number was arrived at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableMemory {
    /// `None`: unknown (no `/proc/meminfo`, e.g. macOS) — never a refusal.
    pub bytes: Option<u64>,
    /// Every input consulted and what was made of it, for the upload result.
    pub source: String,
}

/// The raw files [`available_memory`] reads; `None` for a file that is absent.
#[derive(Debug, Clone, Default)]
pub struct MemoryFiles {
    /// `/proc/meminfo`.
    pub meminfo: Option<String>,
    /// cgroup v2 `/sys/fs/cgroup/memory.max` and `memory.current`.
    pub cgroup2_max: Option<String>,
    pub cgroup2_current: Option<String>,
    /// cgroup v2 `memory.stat` (its `inactive_file` is reclaimable cache).
    pub cgroup2_stat: Option<String>,
    /// cgroup v1 `/sys/fs/cgroup/memory/memory.limit_in_bytes` and
    /// `memory.usage_in_bytes`.
    pub cgroup1_limit: Option<String>,
    pub cgroup1_usage: Option<String>,
    /// cgroup v1 `memory.stat` (its `total_inactive_file`).
    pub cgroup1_stat: Option<String>,
}

impl MemoryFiles {
    /// Read them from this host.
    pub fn read() -> Self {
        let r = |p: &str| std::fs::read_to_string(p).ok();
        MemoryFiles {
            meminfo: r("/proc/meminfo"),
            cgroup2_max: r("/sys/fs/cgroup/memory.max"),
            cgroup2_current: r("/sys/fs/cgroup/memory.current"),
            cgroup2_stat: r("/sys/fs/cgroup/memory.stat"),
            cgroup1_limit: r("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            cgroup1_usage: r("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            cgroup1_stat: r("/sys/fs/cgroup/memory/memory.stat"),
        }
    }
}

/// This host's [`available_memory`].
pub fn mem_available() -> AvailableMemory {
    available_memory(&MemoryFiles::read())
}

/// `MemAvailable` capped by the cgroup's headroom: `/proc/meminfo` reports the
/// host, not a container's limit, so inside a 16 GiB container on a 512 GiB
/// host it alone would let the in-memory reader start and be OOM-killed. The
/// result is `min(MemAvailable, limit - usage)` over the cgroup v2 pair
/// (`memory.max` of `max` = unlimited) and, failing that, the cgroup v1 pair.
/// `usage` excludes the cgroup's reclaimable page cache — `inactive_file` from
/// `memory.stat` (v1: `total_inactive_file`), as kubelet and `docker stats` do
/// — so a container that just read a large corpus is not judged full of it.
/// A value that does not parse is ignored and named in `source`.
pub fn available_memory(f: &MemoryFiles) -> AvailableMemory {
    let mut notes: Vec<String> = Vec::new();
    let mut best: Option<u64> = None;
    let mut take = |bytes: u64, what: String, notes: &mut Vec<String>| {
        notes.push(what);
        best = Some(best.map_or(bytes, |b| b.min(bytes)));
    };

    match f.meminfo.as_deref().map(parse_mem_available) {
        Some(Some(b)) => take(b, format!("MemAvailable {b}"), &mut notes),
        Some(None) => notes.push("/proc/meminfo has no parsable MemAvailable (ignored)".into()),
        None => notes.push("no /proc/meminfo".into()),
    }

    let num = |s: &str| s.trim().parse::<u64>().ok();
    let headroom = |label: &str,
                    limit: Option<&str>,
                    usage: Option<&str>,
                    stat: Option<&str>,
                    stat_key: &str,
                    notes: &mut Vec<String>|
     -> Option<u64> {
        let limit = limit?.trim();
        if limit == "max" {
            notes.push(format!("{label} limit max (unlimited)"));
            return None;
        }
        let Some(l) = num(limit) else {
            notes.push(format!("{label} limit {limit:?} is not a number (ignored)"));
            return None;
        };
        let used = match usage.map(|u| (u.trim(), num(u))) {
            Some((_, Some(u))) => u,
            Some((raw, None)) => {
                notes.push(format!("{label} usage {raw:?} is not a number (ignored)"));
                return None;
            }
            None => 0,
        };
        let cache = match stat {
            None => {
                notes.push(format!(
                    "{label}: no memory.stat, page cache not subtracted"
                ));
                0
            }
            Some(text) => {
                let line = text
                    .lines()
                    .find(|l| l.split_whitespace().next() == Some(stat_key));
                match line.and_then(|l| l.split_whitespace().nth(1)).map(num) {
                    Some(Some(c)) => {
                        notes.push(format!("{label}: {stat_key} {c} subtracted from usage"));
                        c
                    }
                    _ => {
                        notes.push(format!(
                            "{label}: memory.stat has no parsable {stat_key}, page cache not \
                             subtracted"
                        ));
                        0
                    }
                }
            }
        };
        Some(l.saturating_sub(used.saturating_sub(cache)))
    };

    let v2 = headroom(
        "cgroup v2 memory.max",
        f.cgroup2_max.as_deref(),
        f.cgroup2_current.as_deref(),
        f.cgroup2_stat.as_deref(),
        "inactive_file",
        &mut notes,
    );
    let cgroup = match v2 {
        Some(h) => Some((h, "cgroup v2 memory.max - (memory.current - inactive_file)")),
        None if f.cgroup2_max.is_none() => headroom(
            "cgroup v1 memory.limit_in_bytes",
            f.cgroup1_limit.as_deref(),
            f.cgroup1_usage.as_deref(),
            f.cgroup1_stat.as_deref(),
            "total_inactive_file",
            &mut notes,
        )
        .map(|h| {
            (
                h,
                "cgroup v1 limit_in_bytes - (usage_in_bytes - total_inactive_file)",
            )
        }),
        None => None,
    };
    if let Some((h, what)) = cgroup {
        take(h, format!("{what} {h}"), &mut notes);
    }
    AvailableMemory {
        bytes: best,
        source: format!(
            "{}; using {}",
            notes.join("; "),
            best.map_or("unknown".to_string(), |b| b.to_string())
        ),
    }
}

fn parse_mem_available(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

/// Would the in-memory reader fit? `Err` explains why not. An unknown
/// `MemAvailable` is not a refusal. This compares only the reader's own peak
/// with memory available when the dataset was opened: an engine on the same
/// host that grows by its index during the upload is not accounted for, so a
/// pass means "the reader alone fits", not "the run will not OOM".
pub fn in_memory_fits(vector_bytes: u64, mem_available: Option<u64>) -> Result<(), String> {
    let need = vector_bytes.saturating_mul(IN_MEMORY_PEAK_FACTOR);
    match mem_available {
        Some(avail) if need > avail => Err(format!(
            "loading {vector_bytes} vector bytes into memory peaks at ~{need} bytes \
             ({IN_MEMORY_PEAK_FACTOR}x), more than the {avail} bytes available (MemAvailable, \
             capped by the cgroup limit)"
        )),
        _ => Ok(()),
    }
}

/// `auto`'s rule for a streamable compound corpus: `Ok(reason)` to stream,
/// `Err(reason)` to keep the in-memory reader.
pub fn auto_streams(vector_bytes: u64, mem_available: Option<u64>) -> Result<String, String> {
    if vector_bytes >= AUTO_STREAM_MIN_BYTES {
        return Ok(format!(
            "{vector_bytes} vector bytes, at or above the {AUTO_STREAM_MIN_BYTES}-byte \
             auto-stream threshold"
        ));
    }
    match in_memory_fits(vector_bytes, mem_available) {
        Err(e) => Ok(e),
        Ok(()) => Err(format!(
            "{vector_bytes} vector bytes, under the {AUTO_STREAM_MIN_BYTES}-byte auto-stream \
             threshold and loadable (available memory {})",
            mem_available.map_or("unknown".to_string(), |m| m.to_string())
        )),
    }
}

/// What [`plan`] decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UploadPlan {
    InMemory {
        reason: String,
    },
    Stream {
        layout: F32NpyLayout,
        reason: String,
    },
}

impl UploadPlan {
    pub fn streams(&self) -> bool {
        matches!(self, UploadPlan::Stream { .. })
    }
}

/// Decide the reader. `compound_dir` is the dataset directory when the dataset
/// is compound (`type: "tar"`), `None` otherwise; `mem_available` is the host's
/// available memory ([`mem_available`]`.bytes`), used only by `auto`.
pub fn plan(
    mode: UploadReader,
    compound_dir: Option<&Path>,
    mem_available: Option<u64>,
) -> Result<UploadPlan, String> {
    match (mode, compound_dir) {
        (UploadReader::Memory, _) => Ok(UploadPlan::InMemory {
            reason: "--upload-reader memory".into(),
        }),
        (UploadReader::Stream, None) => Err(
            "--upload-reader stream needs a compound dataset (type \"tar\": vectors.npy + \
             payloads.jsonl); this dataset is read by another reader"
                .into(),
        ),
        (UploadReader::Stream, Some(dir)) => Ok(UploadPlan::Stream {
            layout: f32_npy_layout(&dir.join("vectors.npy"))?,
            reason: "--upload-reader stream".into(),
        }),
        (UploadReader::Auto, None) => Ok(UploadPlan::InMemory {
            reason: "not a compound dataset".into(),
        }),
        (UploadReader::Auto, Some(dir)) => match streamable_layout(&dir.join("vectors.npy")) {
            Some(layout) => match auto_streams(layout.data_bytes(), mem_available) {
                Ok(reason) => Ok(UploadPlan::Stream { layout, reason }),
                Err(reason) => Ok(UploadPlan::InMemory { reason }),
            },
            None => Ok(UploadPlan::InMemory {
                reason: "vectors.npy is not a C-order '<f4' array".into(),
            }),
        },
    }
}

/// The documented client-memory bound of a streaming upload, in bytes (see
/// the module docs).
pub fn memory_bound_bytes(
    parallel: usize,
    batch_size: usize,
    dim: usize,
    payload_bytes: u64,
) -> u64 {
    let per_row = 4 * dim as u64 + (4 * dim as u64 + 24) + 16 * dim as u64 + 4 * payload_bytes;
    parallel.max(1) as u64 * batch_size.max(1) as u64 * per_row + 2 * STREAM_READ_BUFFER as u64
}

/// Average `payloads.jsonl` line length, for [`memory_bound_bytes`]; 0 when
/// there is no payload file.
pub fn mean_payload_bytes(dir: &Path, rows: u64) -> u64 {
    std::fs::metadata(dir.join("payloads.jsonl"))
        .map(|m| m.len() / rows.max(1))
        .unwrap_or(0)
}

/// Run a streamed ingest on ONE [`IngestClock`] (#344 semantics, #342 path):
/// the clock starts immediately before `upload` (which streams and sends
/// every batch), `upload_done` when it returns, then `wait` (the engine's
/// index-ready wait, given the uploaded row count) runs, and `index_ready`
/// closes the phases. `read_time` is whatever the engine did before the first
/// batch (a payload-only validation pass, at most).
///
/// On the streaming path the disk read and decode run INSIDE the upload
/// window, so `upload_time` covers read + decode + send (the in-memory path
/// reports the read as `read_time`); `results.upload_reader` records which
/// reader ran so the two are never compared blind (#362).
///
/// [`IngestClock`]: crate::ingest::IngestClock
pub fn ingest_streamed(
    read_time: f64,
    index_wait: crate::ingest::IndexWait,
    upload: impl FnOnce() -> Result<StreamStats, String>,
    wait: impl FnOnce(usize) -> Result<(), String>,
) -> Result<(StreamStats, crate::ingest::IngestPhases), String> {
    let mut clock = crate::ingest::IngestClock::start();
    let stats = upload()?;
    clock.upload_done();
    let documents = stats.rows as usize;
    wait(documents)?;
    let phases = clock.index_ready(documents, read_time, index_wait)?;
    Ok((stats, phases))
}

/// What a streaming upload did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamStats {
    pub rows: u64,
    pub batches: u64,
    /// Most batches decoded-or-uploading at once; never more than `parallel`.
    pub max_in_flight: usize,
}

/// Upload a compound corpus from disk with `parallel` workers, `batch_size`
/// rows per batch (see the module docs for the memory bound).
///
/// `connect` runs once per worker; `upload` is called with each decoded batch
/// (mutable, so an engine can normalize payload text in place) and owns the
/// engine's retry policy. The first error stops every worker and is returned.
/// Batches are handed out in ascending id order, though they complete in any
/// order. Succeeds only when every row of `vectors.npy` was uploaded AND the
/// stream confirmed `payloads.jsonl` has exactly as many rows.
pub fn upload_stream<C, Connect, Upload>(
    stream: CompoundStream,
    parallel: usize,
    batch_size: usize,
    connect: Connect,
    upload: Upload,
    progress: &(dyn Fn(u64) + Sync),
) -> Result<StreamStats, String>
where
    Connect: Fn() -> Result<C, String> + Sync,
    Upload: Fn(&mut C, &mut Batch) -> Result<(), String> + Sync,
{
    if batch_size == 0 {
        return Err("batch_size must be at least 1".into());
    }
    let expected = stream.rows();
    let stream = Mutex::new(stream);
    let error: Mutex<Option<String>> = Mutex::new(None);
    let uploaded = AtomicU64::new(0);
    let batches = AtomicU64::new(0);
    let in_flight = AtomicUsize::new(0);
    let max_in_flight = AtomicUsize::new(0);
    let exhausted = std::sync::atomic::AtomicBool::new(false);
    let fail = |e: String| {
        let mut slot = error.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some(e);
        }
    };
    let failed = || error.lock().map(|s| s.is_some()).unwrap_or(true);

    std::thread::scope(|s| {
        for _ in 0..parallel.max(1) {
            s.spawn(|| {
                let mut conn = match connect() {
                    Ok(c) => c,
                    Err(e) => return fail(e),
                };
                loop {
                    if failed() {
                        return;
                    }
                    // The only serialized step: sequential reads of the two
                    // files. Decoding and uploading happen outside the lock.
                    let raw = {
                        let mut guard = match stream.lock() {
                            Ok(g) => g,
                            Err(_) => return fail("stream lock poisoned".into()),
                        };
                        match guard.next_raw(batch_size) {
                            Ok(Some(raw)) => raw,
                            Ok(None) => {
                                exhausted.store(true, Ordering::SeqCst);
                                return;
                            }
                            Err(e) => return fail(e),
                        }
                    };
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    max_in_flight.fetch_max(now, Ordering::SeqCst);
                    let result = raw.decode().and_then(|mut batch| {
                        let n = batch.len() as u64;
                        upload(&mut conn, &mut batch).map(|()| n)
                    });
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    match result {
                        Ok(n) => {
                            uploaded.fetch_add(n, Ordering::SeqCst);
                            batches.fetch_add(1, Ordering::SeqCst);
                            progress(n);
                        }
                        Err(e) => return fail(e),
                    }
                }
            });
        }
    });

    if let Some(e) = error.into_inner().unwrap_or_else(|p| p.into_inner()) {
        return Err(e);
    }
    let rows = uploaded.load(Ordering::SeqCst);
    // Belt and braces: a worker only stops cleanly on `Ok(None)`, which the
    // stream returns once every row is out and the payloads are exhausted.
    if !exhausted.load(Ordering::SeqCst) || rows != expected {
        return Err(format!(
            "streaming upload sent {rows} of {expected} rows without an error; refusing to \
             report a partial corpus as complete"
        ));
    }
    Ok(StreamStats {
        rows,
        batches: batches.load(Ordering::SeqCst),
        max_in_flight: max_in_flight.load(Ordering::SeqCst),
    })
}

#[cfg(test)]
mod tests {
    /// #342 x #344: a streamed ingest reports its phases on one clock — the
    /// upload phase is the streaming upload, the index wait follows it, the
    /// phases pass `check_legacy_fields` against the `upload_time` /
    /// `total_time` the engine reports, and a failure in either step is
    /// returned (the wait never runs after a failed upload).
    #[test]
    fn a_streamed_ingest_reports_one_clock_phases() {
        use crate::ingest::IndexWait;
        use std::time::Duration;
        let stats = super::StreamStats {
            rows: 1234,
            batches: 7,
            max_in_flight: 2,
        };
        let waited = std::cell::Cell::new(0usize);
        let (got, phases) = super::ingest_streamed(
            0.25,
            IndexWait::RedisPercentIndexed,
            || {
                std::thread::sleep(Duration::from_millis(30));
                Ok(stats.clone())
            },
            |n| {
                waited.set(n);
                std::thread::sleep(Duration::from_millis(20));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(got, stats);
        assert_eq!(
            waited.get(),
            1234,
            "the wait is told the uploaded row count"
        );
        assert_eq!(phases.documents, 1234);
        assert_eq!(phases.read_time, 0.25);
        assert!(phases.upload_time >= 0.03, "{phases:?}");
        assert!(phases.index_wait_time >= 0.02, "{phases:?}");
        assert!(phases.index_wait_time < phases.total_ingest_time);
        // What the engines report from it must pass #344's invariant.
        phases
            .check_legacy_fields(
                phases.upload_time,
                phases.read_time + phases.total_ingest_time,
            )
            .unwrap();
        assert_eq!(phases.index_wait, IndexWait::RedisPercentIndexed);

        let called = std::cell::Cell::new(false);
        let e = super::ingest_streamed(
            0.0,
            IndexWait::OpenSearchForceMergeRefresh,
            || Err("disk gone".to_string()),
            |_| {
                called.set(true);
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(e, "disk gone");
        assert!(!called.get(), "waited for an index after a failed upload");
        let e = super::ingest_streamed(
            0.0,
            IndexWait::OpenSearchForceMergeRefresh,
            || Ok(stats.clone()),
            |_| Err("index timeout".to_string()),
        )
        .unwrap_err();
        assert_eq!(e, "index timeout");
    }
    use super::*;
    use crate::readers::write_npy_vectors;
    use std::collections::BTreeMap;

    fn corpus(rows: usize, dim: usize, with_payloads: bool) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let vectors: Vec<Vec<f32>> = (0..rows)
            .map(|r| (0..dim).map(|c| (r * 31 + c) as f32).collect())
            .collect();
        write_npy_vectors(dir.path().join("vectors.npy").to_str().unwrap(), &vectors).unwrap();
        if with_payloads {
            let text: String = (0..rows).map(|i| format!("{{\"row\":{i}}}\n")).collect();
            std::fs::write(dir.path().join("payloads.jsonl"), text).unwrap();
        }
        dir
    }

    #[test]
    fn the_flag_parses_and_defaults_to_auto() {
        assert_eq!(UploadReader::default(), UploadReader::Auto);
        for m in [
            UploadReader::Auto,
            UploadReader::Stream,
            UploadReader::Memory,
        ] {
            assert_eq!(m.as_str().parse::<UploadReader>().unwrap(), m);
        }
        assert!("streaming".parse::<UploadReader>().is_err());
    }

    /// The decision table of the module docs, row by row. `auto` must keep a
    /// small compound corpus in memory (the pre-#342 path), and `stream` must
    /// refuse a dataset it cannot stream rather than quietly loading it.
    #[test]
    fn plan_follows_the_decision_table() {
        let dir = corpus(10, 4, true);
        let p = dir.path();
        let mem = Some(1 << 30);
        assert!(!plan(UploadReader::Memory, Some(p), Some(1))
            .unwrap()
            .streams());
        assert!(!plan(UploadReader::Memory, None, mem).unwrap().streams());
        assert!(plan(UploadReader::Stream, None, mem).is_err());
        assert_eq!(
            plan(UploadReader::Stream, Some(p), mem).unwrap(),
            UploadPlan::Stream {
                layout: f32_npy_layout(&p.join("vectors.npy")).unwrap(),
                reason: "--upload-reader stream".into(),
            }
        );
        assert!(!plan(UploadReader::Auto, None, Some(1)).unwrap().streams());
        // 160 bytes of vectors: far under the threshold, so in memory.
        for m in [mem, None] {
            match plan(UploadReader::Auto, Some(p), m).unwrap() {
                UploadPlan::InMemory { reason } => assert!(reason.contains("160 vector bytes")),
                other => panic!("{other:?}"),
            }
        }
        // ... unless the host could not load even that.
        assert!(plan(UploadReader::Auto, Some(p), Some(319))
            .unwrap()
            .streams());
        assert!(!plan(UploadReader::Auto, Some(p), Some(320))
            .unwrap()
            .streams());
        // A non-<f4 file: auto keeps the in-memory reader, stream refuses.
        let npy = p.join("vectors.npy");
        let arr = ndarray::Array2::<f64>::zeros((2, 3));
        ndarray_npy::WriteNpyExt::write_npy(&arr, std::fs::File::create(&npy).unwrap()).unwrap();
        assert!(!plan(UploadReader::Auto, Some(p), Some(1))
            .unwrap()
            .streams());
        assert!(plan(UploadReader::Stream, Some(p), mem).is_err());
    }

    /// A header declaring >= the threshold makes `auto` stream. The header is
    /// all `plan` reads, so a sparse file stands in for 128 GiB of vectors.
    /// The host memory is generous, so only the size threshold decides.
    #[test]
    fn auto_streams_at_the_threshold_and_not_one_row_below() {
        let dir = tempfile::tempdir().unwrap();
        let npy = dir.path().join("vectors.npy");
        let dim = 1024usize;
        let at = AUTO_STREAM_MIN_BYTES / (4 * dim as u64);
        for (rows, streams) in [(at, true), (at - 1, false)] {
            let header = crate::msmarco::npy_f32_header(rows, dim);
            std::fs::write(&npy, &header).unwrap();
            let f = std::fs::OpenOptions::new().write(true).open(&npy).unwrap();
            f.set_len(header.len() as u64 + rows * 4 * dim as u64)
                .unwrap();
            assert_eq!(
                plan(UploadReader::Auto, Some(dir.path()), Some(u64::MAX))
                    .unwrap()
                    .streams(),
                streams,
                "{rows} rows"
            );
        }
    }

    /// The reviewed requirement (#361): `auto` keeps every MS MARCO variant up
    /// to the 10M on the in-memory reader on a host that can load it, so the
    /// published 1M/10M `upload_time` numbers stay comparable; the 100M streams
    /// on any host; a corpus the host cannot load streams instead of OOMing.
    #[test]
    fn auto_streams_only_what_memory_cannot_hold() {
        let gib = 1u64 << 30;
        let big_host = Some(1024 * gib);
        for (name, rows) in [("100K", 99_964u64), ("1M", 1_000_044), ("10M", 9_999_959)] {
            let bytes = rows * 4096;
            assert!(
                auto_streams(bytes, big_host).is_err(),
                "{name} streams on a 1 TiB host"
            );
            assert!(
                auto_streams(bytes, None).is_err(),
                "{name} streams on an unknown host"
            );
        }
        let hundred_m = 100_000_027u64 * 4096;
        for host in [big_host, None, Some(u64::MAX)] {
            assert!(
                auto_streams(hundred_m, host).is_ok(),
                "100M loaded on {host:?}"
            );
        }
        // The 10M on a 64 GiB host: 82 GB would not fit, so it streams.
        assert!(auto_streams(9_999_959 * 4096, Some(64 * gib)).is_ok());
        // The fit rule is 2x, at the boundary.
        assert!(in_memory_fits(100, Some(200)).is_ok());
        assert!(in_memory_fits(100, Some(199)).is_err());
        assert!(in_memory_fits(u64::MAX, None).is_ok());
        assert_eq!(
            parse_mem_available("MemTotal: 10 kB\nMemAvailable:    2048 kB\n"),
            Some(2 << 20)
        );
        assert_eq!(parse_mem_available("MemTotal: 10 kB\n"), None);
    }

    /// #361 review: `MemAvailable` is the host's; a container's cgroup limit
    /// must cap it, `max` is unlimited, and an unparsable file is ignored with
    /// the reason recorded.
    #[test]
    fn available_memory_is_capped_by_the_cgroup() {
        let gib = 1u64 << 30;
        let meminfo = Some(format!(
            "MemTotal: 1 kB\nMemAvailable: {} kB\n",
            64 * gib / 1024
        ));
        // A numeric v2 limit below MemAvailable wins, minus current usage.
        let m = available_memory(&MemoryFiles {
            meminfo: meminfo.clone(),
            cgroup2_max: Some(format!("{}\n", 16 * gib)),
            cgroup2_current: Some(format!("{}\n", 2 * gib)),
            ..Default::default()
        });
        assert_eq!(m.bytes, Some(14 * gib), "{}", m.source);
        assert!(m.source.contains("cgroup v2"), "{}", m.source);
        // A limit above MemAvailable does not raise it.
        let m = available_memory(&MemoryFiles {
            meminfo: meminfo.clone(),
            cgroup2_max: Some(format!("{}", 512 * gib)),
            cgroup2_current: Some("0".into()),
            ..Default::default()
        });
        assert_eq!(m.bytes, Some(64 * gib));
        // `max` is unlimited.
        let m = available_memory(&MemoryFiles {
            meminfo: meminfo.clone(),
            cgroup2_max: Some("max\n".into()),
            cgroup2_current: Some("123".into()),
            ..Default::default()
        });
        assert_eq!(m.bytes, Some(64 * gib));
        assert!(m.source.contains("unlimited"), "{}", m.source);
        // Garbage is ignored, and says so.
        let m = available_memory(&MemoryFiles {
            meminfo: meminfo.clone(),
            cgroup2_max: Some("lots".into()),
            ..Default::default()
        });
        assert_eq!(m.bytes, Some(64 * gib));
        assert!(
            m.source.contains("\"lots\" is not a number (ignored)"),
            "{}",
            m.source
        );
        // cgroup v1 when there is no v2 file.
        let m = available_memory(&MemoryFiles {
            meminfo: meminfo.clone(),
            cgroup1_limit: Some(format!("{}", 8 * gib)),
            cgroup1_usage: Some(format!("{}", gib)),
            ..Default::default()
        });
        assert_eq!(m.bytes, Some(7 * gib), "{}", m.source);
        // #361 review: page cache charged to the cgroup is reclaimable. A
        // 64 GiB container that just read the 10M holds 55 GiB of inactive
        // file cache; the 1M (2 x 4.1 GB in memory) must still load.
        let one_m = 1_000_044u64 * 4096;
        let cache_heavy = |stat: Option<&str>| {
            available_memory(&MemoryFiles {
                meminfo: Some(format!("MemAvailable: {} kB\n", 200 * gib / 1024)),
                cgroup2_max: Some(format!("{}", 64 * gib)),
                cgroup2_current: Some(format!("{}", 60 * gib)),
                cgroup2_stat: stat.map(str::to_string),
                ..Default::default()
            })
        };
        let stat = format!(
            "anon 5368709120\ninactive_file {}\nactive_file 1\n",
            55 * gib
        );
        let m = cache_heavy(Some(&stat));
        assert_eq!(m.bytes, Some(59 * gib), "{}", m.source);
        assert!(
            auto_streams(one_m, m.bytes).is_err(),
            "1M streamed: {}",
            m.source
        );
        assert!(m.source.contains("inactive_file"), "{}", m.source);
        // Without a usable memory.stat nothing is subtracted, and it says so.
        for (stat, why) in [
            (None, "no memory.stat"),
            (Some("anon 1\n"), "no parsable inactive_file"),
            (Some("inactive_file lots\n"), "no parsable inactive_file"),
        ] {
            let m = cache_heavy(stat);
            assert_eq!(m.bytes, Some(4 * gib), "{}", m.source);
            assert!(m.source.contains(why), "{}", m.source);
        }
        // v1 reads total_inactive_file.
        let m = available_memory(&MemoryFiles {
            cgroup1_limit: Some(format!("{}", 8 * gib)),
            cgroup1_usage: Some(format!("{}", 6 * gib)),
            cgroup1_stat: Some(format!("cache 9\ntotal_inactive_file {}\n", 3 * gib)),
            ..Default::default()
        });
        assert_eq!(m.bytes, Some(5 * gib), "{}", m.source);

        // A cgroup with no meminfo still bounds; nothing at all is unknown.
        let m = available_memory(&MemoryFiles {
            cgroup2_max: Some(format!("{}", 4 * gib)),
            ..Default::default()
        });
        assert_eq!(m.bytes, Some(4 * gib));
        assert_eq!(available_memory(&MemoryFiles::default()).bytes, None);
    }

    /// #361 review: a lost `\` line continuation put a run of spaces into
    /// `results.upload_reader.reason`. Every reason string is one line with
    /// single spaces.
    #[test]
    fn reason_strings_have_no_runs_of_spaces() {
        let gib = 1u64 << 30;
        let reasons = [
            auto_streams(200 * gib, Some(u64::MAX)).unwrap(),
            auto_streams(10 * gib, Some(gib)).unwrap(),
            auto_streams(gib, Some(u64::MAX)).unwrap_err(),
            auto_streams(gib, None).unwrap_err(),
            in_memory_fits(10, Some(1)).unwrap_err(),
            available_memory(&MemoryFiles {
                meminfo: Some("MemAvailable: 4 kB\n".into()),
                cgroup2_max: Some("junk".into()),
                ..Default::default()
            })
            .source,
        ];
        for r in reasons {
            assert!(!r.contains("  ") && !r.contains('\n'), "{r:?}");
        }
    }

    /// Every row is delivered exactly once, with its own payload, whatever the
    /// parallelism; never more than `parallel` batches are alive at once; and
    /// the reported count is the corpus size.
    #[test]
    fn every_row_is_uploaded_once_with_bounded_in_flight_batches() {
        let dir = corpus(1003, 3, true);
        for parallel in [1, 2, 8] {
            let seen: Mutex<BTreeMap<i64, (f32, String)>> = Mutex::new(BTreeMap::new());
            let progressed = AtomicU64::new(0);
            let stats = upload_stream(
                CompoundStream::open(dir.path(), false).unwrap(),
                parallel,
                50,
                || Ok(()),
                |_, b| {
                    assert!(b.len() <= 50);
                    let mut s = seen.lock().unwrap();
                    for i in 0..b.len() {
                        let m = format!("{:?}", b.metadata[i]);
                        assert!(s.insert(b.ids[i], (b.vectors[i][0], m)).is_none());
                    }
                    Ok(())
                },
                &|n| {
                    progressed.fetch_add(n, Ordering::SeqCst);
                },
            )
            .unwrap();
            assert_eq!(stats.rows, 1003);
            assert_eq!(stats.batches, 21);
            assert!(stats.max_in_flight <= parallel, "{stats:?}");
            assert_eq!(progressed.load(Ordering::SeqCst), 1003);
            let s = seen.into_inner().unwrap();
            assert_eq!(s.len(), 1003);
            for (id, (v0, m)) in s {
                assert_eq!(v0, (id * 31) as f32);
                assert!(m.contains(&format!("Int({id})")), "{id}: {m}");
            }
        }
    }

    /// The first failure stops the upload and is what is returned; a bad
    /// payload line mid-corpus is an error, not a skipped row.
    #[test]
    fn an_upload_or_read_error_stops_the_run() {
        let dir = corpus(500, 2, true);
        let calls = AtomicU64::new(0);
        let e = upload_stream(
            CompoundStream::open(dir.path(), false).unwrap(),
            4,
            10,
            || Ok(()),
            |_, b| {
                calls.fetch_add(1, Ordering::SeqCst);
                if b.ids[0] == 100 {
                    Err("engine said no".into())
                } else {
                    Ok(())
                }
            },
            &|_| {},
        )
        .unwrap_err();
        assert_eq!(e, "engine said no");
        assert!(calls.load(Ordering::SeqCst) < 50, "workers kept going");

        let e = upload_stream(
            CompoundStream::open(dir.path(), false).unwrap(),
            2,
            10,
            || Err::<(), _>("cannot connect".to_string()),
            |_, _| Ok(()),
            &|_| {},
        )
        .unwrap_err();
        assert_eq!(e, "cannot connect");

        let mut text: String = (0..500).map(|i| format!("{{\"row\":{i}}}\n")).collect();
        text.push_str("{\"extra\":1}\n");
        std::fs::write(dir.path().join("payloads.jsonl"), &text).unwrap();
        let e = upload_stream(
            CompoundStream::open(dir.path(), false).unwrap(),
            3,
            64,
            || Ok(()),
            |_, _| Ok(()),
            &|_| {},
        )
        .unwrap_err();
        assert!(e.contains("Metadata count (501)"), "{e}");
        assert!(upload_stream(
            CompoundStream::open(dir.path(), false).unwrap(),
            1,
            0,
            || Ok(()),
            |_, _| Ok(()),
            &|_| {}
        )
        .is_err());
    }

    #[test]
    fn the_memory_bound_scales_with_batches_not_corpus() {
        let b = memory_bound_bytes(32, 1000, 1024, 1843);
        // ~1.0 GB for MS MARCO at parallel 32 x batch 1000 (module docs).
        assert!((1_000_000_000..1_100_000_000).contains(&b), "{b}");
        assert_eq!(
            memory_bound_bytes(1, 1, 0, 0),
            24 + 2 * STREAM_READ_BUFFER as u64
        );
        let dir = corpus(10, 4, true);
        assert_eq!(mean_payload_bytes(dir.path(), 10), 10); // `{"row":N}\n`
        assert_eq!(mean_payload_bytes(&dir.path().join("nope"), 10), 0);
    }
}
