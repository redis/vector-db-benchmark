//! `prepare-msmarco` — materialise the MS MARCO v2.1 / Cohere `embed-english-v3`
//! corpus (vectors **and** metadata) on disk in the compound layout the
//! benchmark's `type: "tar"` reader expects.
//!
//! All six registered variants ship prepared tarballs in S3 and auto-download
//! like any other dataset; this binary is what BUILT them, and what rebuilds
//! them from upstream if you would rather re-derive than trust the artifact.
//! Upstream itself is a 60-shard Hugging Face dataset of 113.5M passages with no
//! tarball of its own and no sane default size, which is why a preparation step
//! exists at all. Run:
//!
//! ```text
//! cargo run --release --bin prepare-msmarco -- --dataset msmarco-cohere-1024-100K-cosine
//! cargo run --release --bin prepare-msmarco -- --dataset msmarco-cohere-1024-1M-cosine --out-dir /data/ds
//! ```
//!
//! What it writes, under `<out-dir>/<the variant's dir>`:
//!
//! * `vectors.npy`    — N x 1024 `float32`, the first N passages of the global
//!   order, converted from the upstream `float16`.
//! * `payloads.jsonl` — one JSON object per vector: `docid`, `url`, `title`,
//!   `headings`, `segment`, `start_char`, `end_char`.
//! * `tests.jsonl`    — the 1677 TREC-DL 2021-2023 queries, each with its
//!   embedding and a brute-forced top-1000 over *this* prefix.
//! * `PREPARED.json`  — provenance: source repo, sizes, and the verification
//!   statistics from the pass described below.
//!
//! # The size is fixed by the dataset name
//!
//! There is deliberately no `--limit`. Two runs that both report
//! `msmarco-cohere-1024-1M-cosine` must have uploaded the same corpus, so the
//! passage count comes from [`msmarco::Variant`], which is also what
//! `datasets.json` declares as `vector_count` and what the benchmark's own
//! corpus-completeness gate checks against.
//!
//! # Ground truth is brute-forced, then cross-checked against HF's
//!
//! HF ships a global top-1000 per query. Truncating it to a prefix is NOT the
//! prefix's top-k (see `msmarco`'s module docs), so the ranking here is computed
//! by brute force over the prepared vectors — read back from the file that was
//! just written, so it is computed on exactly the bytes the engines will read.
//!
//! The shipped list is then used as an independent oracle: every in-prefix
//! global-top-1k hit must appear, in order and with a matching cosine, at the
//! head of our ranking, and the payload at that offset must name the same
//! passage. That one assertion covers the brute force, the npy-row-to-jsonl-line
//! alignment, and the cross-shard offset arithmetic. It is not optional — a
//! failure aborts before anything is registered as usable.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use flate2::read::MultiGzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use vector_db_benchmark::msmarco::{
    self, in_prefix_hits, normalize_in_place, parse_f16_npy_header, payload_from_passage, test_row,
    verify_head_against_shipped, NpyF32Writer, ShardTake, TopK, Variant,
};

/// Passages converted / scored per block. 8192 x 1024 x 4 B = 32 MiB, small
/// enough to keep preparation's peak RSS flat at any corpus size.
const BLOCK_ROWS: usize = 8192;

/// Tolerance when matching our cosines against HF's, and the window inside which
/// a differing id at the same rank is accepted as a tie rather than a wrong
/// ranking.
///
/// 2e-4, not the 2e-3 this started at, because the looser value was ~40x the
/// largest disagreement any real build produces and was doing double duty as the
/// tie window. Measured with `--verify` over the three prepared corpora at
/// decreasing tolerances:
///
/// | corpus | positions | max delta | ties @2e-3 | @2e-4 | @1e-4 | @5e-5 |
/// |--------|-----------|-----------|-----------|-------|-------|-------|
/// | 100K   | 2,134     | 4.63e-5   | 0         | 0     | 0     | 0     |
/// | 1M     | 17,523    | 4.85e-5   | 82        | 82    | 82    | 82    |
/// | 10M    | 197,898   | 5.96e-5   | 5,406     | 5,406 | 5,406 | —     |
///
/// Every tie is a genuine one: the counts do not move as the window shrinks by
/// 40x, so nothing was being waved through by a loose bound. 2e-4 keeps ~3.4x
/// headroom over the worst observed delta (5.96e-5 on the 10M) — enough for
/// float rounding on another machine, while making the oracle 10x stricter than
/// before.
const COSINE_TOLERANCE: f32 = 2e-4;

/// HTTP read timeout. Generous: these are multi-GB streams over the public CDN.
const HTTP_TIMEOUT: Duration = Duration::from_secs(600);

/// Metadata shards read at once during `--discover-crc32`. A single stream from
/// the CDN measured ~12 MB/s, so the scan is bandwidth-bound per connection, not
/// CPU-bound; 16 keeps the host busy without hammering the origin.
const DISCOVER_CONCURRENCY: usize = 16;

#[derive(Parser, Debug)]
#[command(
    name = "prepare-msmarco",
    about = "Download and prepare an MS MARCO v2.1 + Cohere embed-v3 dataset (vectors + metadata)."
)]
struct Args {
    /// Registered dataset name to build, e.g. `msmarco-cohere-1024-1M-cosine`.
    /// Not needed with `--discover-crc32`.
    #[arg(long, required_unless_present = "discover_crc32")]
    dataset: Option<String>,

    /// Base directory holding datasets (the variant gets its registered subdir).
    #[arg(long, default_value = "datasets")]
    out_dir: PathBuf,

    /// Where to cache the shared query file (~59 MB). Defaults to
    /// `<out-dir>/.msmarco-cache`.
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Rebuild even if the output directory already holds a complete dataset.
    #[arg(long)]
    force: bool,

    /// Re-run the shipped-top-1k cross-check against an ALREADY prepared
    /// dataset, without rebuilding it.
    ///
    /// Everything the check needs is already on disk — `tests.jsonl` holds the
    /// brute-forced ranking, `payloads.jsonl` holds the docids — so a corpus can
    /// be re-validated after download, or re-validated at a different
    /// `--tolerance`, in a couple of minutes instead of a full rebuild. Matches
    /// shipped hits by DOCID rather than by offset, so it works for both
    /// sampling modes.
    #[arg(long)]
    verify: bool,

    /// Cosine tolerance for `--verify`. Defaults to the value preparation uses.
    #[arg(long)]
    tolerance: Option<f32>,

    /// Scan the whole corpus's metadata and report, for each target size, the
    /// crc32 threshold that selects closest to it. Writes no dataset.
    ///
    /// This is the discovery step for the uniformly-sampled variants: selection
    /// depends on `docid`, so the realized size of a threshold cannot be known
    /// without reading every docid. Streams only `passages_jsonl` (26.9 GB
    /// gzipped) — never the 232.5 GB of embeddings — and one scan sizes every
    /// possible threshold at once.
    #[arg(long)]
    discover_crc32: bool,

    /// Hugging Face token, for rate-limited or gated access. Also read from
    /// `HF_TOKEN`.
    #[arg(long, env = "HF_TOKEN", hide_env_values = true)]
    hf_token: Option<String>,
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args) {
        eprintln!("prepare-msmarco: error: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), String> {
    if args.discover_crc32 {
        return discover_crc32(args);
    }
    if args.verify {
        return verify_prepared(args);
    }
    let dataset = args
        .dataset
        .as_deref()
        .ok_or_else(|| "--dataset is required".to_string())?;
    let variant = msmarco::variant(dataset).ok_or_else(|| {
        format!(
            "unknown dataset {:?}. Registered MS MARCO variants: {}",
            dataset,
            msmarco::VARIANTS
                .iter()
                .map(|v| v.dataset_name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;

    let dir = args.out_dir.join(variant.dir);
    if !args.force && is_complete(&dir, variant) {
        println!(
            "{} is already prepared at {} ({} passages). Pass --force to rebuild.",
            variant.dataset_name,
            dir.display(),
            variant.limit
        );
        return Ok(());
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {}", dir.display(), e))?;
    clear_staged(&dir);

    let cache = args
        .cache_dir
        .clone()
        .unwrap_or_else(|| args.out_dir.join(".msmarco-cache"));
    std::fs::create_dir_all(&cache).map_err(|e| format!("create {}: {}", cache.display(), e))?;

    let client = http_client()?;
    let started = Instant::now();

    println!(
        "Preparing {} ({} passages, {}-dim cosine) into {}",
        variant.dataset_name,
        variant.limit,
        msmarco::DIM,
        dir.display()
    );

    let mut queries = load_queries(&client, &cache, args.hf_token.as_deref(), variant)?;

    // WHICH offsets need a docid remembered depends on the sampling mode, and
    // getting it wrong is not harmless: a prefix build handed offsets beyond its
    // prefix fails its own completeness check on a correct corpus. The rule
    // lives in `msmarco::offsets_to_track`, under test, rather than here.
    let wanted_offsets: std::collections::HashSet<u64> = queries
        .iter()
        .flat_map(|q| {
            let shipped: Vec<i64> = q.shipped.iter().map(|(off, _, _)| *off).collect();
            msmarco::offsets_to_track(&shipped, variant.sampling, variant.limit)
        })
        .collect();

    let mut plan: Vec<ShardTake> = Vec::new();
    let (docid_at, max_field_bytes) = if variant.sampling.needs_full_scan() {
        let shard_rows = all_shard_rows(&client, args.hf_token.as_deref())?;
        let tmp = dir.join("parts.tmp");
        let corpus = write_corpus_sampled(
            &client,
            args.hf_token.as_deref(),
            variant.sampling,
            &shard_rows,
            &dir,
            &tmp,
            &wanted_offsets,
        )?;
        if corpus.count != variant.limit {
            return Err(format!(
                "{} selected {} passages but the registry declares {}. The threshold's realized \
                 count is measured, not predicted, so a mismatch means the upstream export \
                 changed — rerun --discover-crc32 and update the variant.",
                variant.dataset_name, corpus.count, variant.limit
            ));
        }
        // Ids are positions in the SAMPLE, not global offsets, so each query's
        // shipped hits are remapped through the selection before they can act as
        // an oracle. A shipped hit that was not selected simply drops out.
        for q in queries.iter_mut() {
            q.hits = msmarco::InPrefixHits {
                hits: q
                    .shipped
                    .iter()
                    .filter_map(|(off, pid, cos)| {
                        corpus
                            .offset_to_local
                            .get(&(*off as u64))
                            .map(|local| (*local, pid.clone(), *cos))
                    })
                    .collect(),
            };
        }
        let with_hits = queries.iter().filter(|q| !q.hits.hits.is_empty()).count();
        let total_hits: usize = queries.iter().map(|q| q.hits.hits.len()).sum();
        println!(
            "After sampling, {with_hits} queries retain {total_hits} of their shipped top-1k hits."
        );
        msmarco::check_coverage(variant.limit, with_hits, total_hits)?;
        (corpus.docid_at, corpus.max_field_bytes)
    } else {
        plan = plan_shards(&client, args.hf_token.as_deref(), variant)?;
        let wanted: std::collections::HashSet<i64> =
            wanted_offsets.iter().map(|o| *o as i64).collect();
        write_corpus(
            &client,
            args.hf_token.as_deref(),
            variant,
            &plan,
            &dir,
            &wanted,
        )?
    };

    let ranked = brute_force_ground_truth(&dir, variant, &queries)?;
    let stats = verify(variant.limit, &queries, &ranked, &docid_at)?;
    write_tests(&dir, &queries, &ranked)?;
    write_manifest(&dir, variant, &plan, &stats, &queries, &max_field_bytes)?;
    publish(&dir)?;

    println!(
        "Done in {:.1}s. Register-ready at {} — run with --datasets {}",
        started.elapsed().as_secs_f64(),
        dir.display(),
        variant.dataset_name
    );
    Ok(())
}

/// The four files a finished dataset directory holds. Every one is written to
/// `<name>.part` first and renamed only once verification has passed, so a
/// directory either holds a complete, cross-checked corpus or is untouched —
/// which is what makes [`is_complete`]'s claim true rather than aspirational.
/// (A `--force` rebuild that dies halfway now also leaves the previous good
/// corpus intact instead of overwriting it with a partial one.)
/// Written last, after every other file is in place, and required by
/// [`is_complete`].
///
/// Four renames are four separate operations: dying between the second and the
/// fourth leaves a new `PREPARED.json` and `tests.jsonl` over an old
/// `payloads.jsonl` and `vectors.npy`, which the previous "do all three exist
/// and does the row count match" check accepted. A single file that only exists
/// once the set is whole turns "complete or untouched" from a description of the
/// happy path into something the code enforces.
const COMPLETE_MARKER: &str = "COMPLETE";

const OUTPUTS: [&str; 4] = [
    "vectors.npy",
    "payloads.jsonl",
    "tests.jsonl",
    "PREPARED.json",
];

/// Path a file is written to before it is published.
fn staged(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.part"))
}

/// Publish every staged file, replacing whatever was there. Renames within one
/// directory, so each is atomic; the set as a whole is not, which is why
/// `vectors.npy` (the one [`is_complete`] measures) goes last.
fn publish(dir: &Path) -> Result<(), String> {
    // Clear any previous marker FIRST: from here until the last rename the
    // directory is a mix of old and new files and must not read as complete.
    let marker = dir.join(COMPLETE_MARKER);
    let _ = std::fs::remove_file(&marker);
    for name in OUTPUTS.iter().rev() {
        let from = staged(dir, name);
        if !from.exists() {
            return Err(format!("{} was never written", from.display()));
        }
        std::fs::rename(&from, dir.join(name))
            .map_err(|e| format!("publish {}: {}", from.display(), e))?;
    }
    std::fs::write(&marker, "ok\n").map_err(|e| format!("write {}: {}", marker.display(), e))
}

/// Remove any `.part` files left by an earlier interrupted run.
fn clear_staged(dir: &Path) {
    for name in OUTPUTS {
        let _ = std::fs::remove_file(staged(dir, name));
    }
    let _ = std::fs::remove_file(dir.join(COMPLETE_MARKER));
}

/// A prepared directory is complete when all three files exist and `vectors.npy`
/// declares exactly the variant's passage count. A half-written corpus from an
/// interrupted run must not be mistaken for a finished one.
fn is_complete(dir: &Path, variant: &Variant) -> bool {
    let (v, p, t) = (
        dir.join("vectors.npy"),
        dir.join("payloads.jsonl"),
        dir.join("tests.jsonl"),
    );
    // The marker is written only after all four renames land, so a run
    // interrupted mid-publish leaves a directory that does NOT read as complete.
    if !dir.join(COMPLETE_MARKER).exists() {
        return false;
    }
    if !(v.exists() && p.exists() && t.exists()) {
        return false;
    }
    matches!(
        vector_db_benchmark::readers::npy_row_count(v.to_str().unwrap_or_default()),
        Ok(rows) if rows == variant.limit
    )
}

fn http_client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .build()
        .map_err(|e| format!("build HTTP client: {e}"))
}

fn get(
    client: &reqwest::blocking::Client,
    url: &str,
    token: Option<&str>,
    range: Option<(u64, u64)>,
) -> Result<reqwest::blocking::Response, String> {
    let mut req = client.get(url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    if let Some((start, end_inclusive)) = range {
        req = req.header(
            reqwest::header::RANGE,
            format!("bytes={start}-{end_inclusive}"),
        );
    }
    let resp = req.send().map_err(|e| format!("GET {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("GET {url}: HTTP {status}"));
    }
    // A ranged request that comes back 200 means the server ignored the Range
    // header and is about to stream the whole multi-GB file from byte 0 — which
    // would silently give the caller the wrong bytes at the wrong offsets.
    if range.is_some() && status != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(format!(
            "GET {url}: asked for a byte range but the server answered {status} \
             (expected 206 Partial Content); it is serving the whole object, \
             so the offsets would not line up"
        ));
    }
    Ok(resp)
}

/// Stream every `passages_jsonl` shard and histogram the docid buckets.
///
/// Reads metadata only — the embeddings are never touched, which is what keeps
/// this to 26.9 GB instead of the full 259 GB. The result sizes every possible
/// threshold, so the sampled variants' constants come from one scan.
fn discover_crc32(args: &Args) -> Result<(), String> {
    let client = http_client()?;
    let token = args.hf_token.as_deref();
    let mut hist = msmarco::BucketHistogram::new();
    let started = Instant::now();

    println!(
        "Scanning {} metadata shards to size the crc32 thresholds (embeddings are NOT fetched), \
         {} shards at a time...",
        msmarco::SHARDS,
        DISCOVER_CONCURRENCY
    );
    let bar = progress(msmarco::TOTAL_PASSAGES, "docids");

    // One stream per shard tops out around 12 MB/s against the CDN, which would
    // make a single-threaded scan of ~100 GB take hours. The histogram is
    // order-independent — every shard's counts merge by addition — so the shards
    // are read concurrently and merged at the end. The build path cannot do this
    // so freely: there, ids are assigned in corpus order.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(DISCOVER_CONCURRENCY)
        .build()
        .map_err(|e| format!("build scan pool: {e}"))?;

    let per_shard: Vec<Result<(usize, u64, msmarco::BucketHistogram), String>> =
        pool.install(|| {
            (0..msmarco::SHARDS)
                .into_par_iter()
                .map(|shard| {
                    let url = msmarco::jsonl_url(shard);
                    let mut reader = BufReader::with_capacity(
                        1 << 22,
                        MultiGzDecoder::new(BufReader::with_capacity(
                            1 << 22,
                            get(&client, &url, token, None)?,
                        )),
                    );
                    let mut local = msmarco::BucketHistogram::new();
                    let mut line = String::new();
                    let mut in_shard: u64 = 0;
                    loop {
                        line.clear();
                        let n = reader
                            .read_line(&mut line)
                            .map_err(|e| format!("{url}: line {in_shard}: {e}"))?;
                        if n == 0 {
                            break;
                        }
                        if line.trim().is_empty() {
                            continue;
                        }
                        let docid = msmarco::extract_docid(line.trim_end())
                            .ok_or_else(|| format!("{url}: line {in_shard} has no docid"))?;
                        local.add(&docid);
                        in_shard += 1;
                        if in_shard.is_multiple_of(100_000) {
                            bar.inc(100_000);
                        }
                    }
                    Ok((shard, in_shard, local))
                })
                .collect()
        });
    bar.finish_and_clear();

    let mut lines_total: u64 = 0;
    let mut counts_by_shard = vec![0u64; msmarco::SHARDS];
    for r in per_shard {
        let (shard, n, local) = r?;
        counts_by_shard[shard] = n;
        lines_total += n;
        hist.merge(&local);
    }
    for (shard, n) in counts_by_shard.iter().enumerate() {
        println!("  shard {shard:02}: {n} passages");
    }

    if lines_total != msmarco::TOTAL_PASSAGES {
        eprintln!(
            "\tWARNING: counted {lines_total} passages, the dataset card says {} — the upstream \
             export has changed, so any threshold derived here is for THIS corpus, not the one \
             the pinned revision describes.",
            msmarco::TOTAL_PASSAGES
        );
    }

    let targets = [100_000u64, 1_000_000, 10_000_000];
    let report = hist.to_json(&targets);
    println!(
        "\n{}",
        serde_json::to_string_pretty(&report).unwrap_or_default()
    );
    println!(
        "Scanned {} passages in {:.1}s.",
        lines_total,
        started.elapsed().as_secs_f64()
    );
    let out = args.out_dir.join("crc32-thresholds.json");
    std::fs::create_dir_all(&args.out_dir).ok();
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&report).unwrap_or_default(),
    )
    .map_err(|e| format!("write {}: {}", out.display(), e))?;
    println!("Written to {}", out.display());
    Ok(())
}

/// Re-verify a prepared dataset in place.
fn verify_prepared(args: &Args) -> Result<(), String> {
    let dataset = args
        .dataset
        .as_deref()
        .ok_or_else(|| "--dataset is required with --verify".to_string())?;
    let variant =
        msmarco::variant(dataset).ok_or_else(|| format!("unknown dataset {dataset:?}"))?;
    let dir = args.out_dir.join(variant.dir);
    let tol = args.tolerance.unwrap_or(COSINE_TOLERANCE);

    let client = http_client()?;
    let cache = args
        .cache_dir
        .clone()
        .unwrap_or_else(|| args.out_dir.join(".msmarco-cache"));
    std::fs::create_dir_all(&cache).map_err(|e| format!("create {}: {}", cache.display(), e))?;
    let queries = load_queries(&client, &cache, args.hf_token.as_deref(), variant)?;

    // Only the passages some query's top-1k actually names need locating.
    let wanted: std::collections::HashSet<&str> = queries
        .iter()
        .flat_map(|q| q.shipped.iter().map(|(_, pid, _)| pid.as_str()))
        .collect();

    // docid -> local id, by position in payloads.jsonl. That file is written in
    // local-id order by construction, which is the same thing `vectors.npy`'s
    // rows mean, so the position IS the id.
    let payloads_path = dir.join("payloads.jsonl");
    let pf = File::open(&payloads_path).map_err(|e| {
        format!(
            "open {}: {} (is the dataset prepared?)",
            payloads_path.display(),
            e
        )
    })?;
    let mut local_of: HashMap<String, i64> = HashMap::new();
    let mut docid_at: HashMap<i64, String> = HashMap::new();
    let mut id: i64 = 0;
    let bar = progress(variant.limit, "payloads");
    for line in BufReader::with_capacity(1 << 22, pf).lines() {
        let line = line.map_err(|e| format!("read {}: {e}", payloads_path.display()))?;
        if let Some(docid) = msmarco::extract_docid(&line) {
            if wanted.contains(docid.as_str()) {
                local_of.insert(docid.clone(), id);
                docid_at.insert(id, docid);
            }
        }
        id += 1;
        if id % 100_000 == 0 {
            bar.inc(100_000);
        }
    }
    bar.finish_and_clear();
    if id as u64 != variant.limit {
        return Err(format!(
            "{} holds {} payload lines but the registry declares {}",
            payloads_path.display(),
            id,
            variant.limit
        ));
    }

    // The brute-forced ranking, straight out of tests.jsonl.
    let tests_path = dir.join("tests.jsonl");
    let tf =
        File::open(&tests_path).map_err(|e| format!("open {}: {}", tests_path.display(), e))?;
    let mut ranked: Vec<Vec<(f32, i64)>> = Vec::with_capacity(queries.len());
    for line in BufReader::with_capacity(1 << 20, tf).lines() {
        let line = line.map_err(|e| format!("read {}: {e}", tests_path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| format!("parse tests.jsonl: {e}"))?;
        let ids = row["closest_ids"]
            .as_array()
            .ok_or("tests.jsonl: no closest_ids")?;
        let scores = row["closest_scores"]
            .as_array()
            .ok_or("tests.jsonl: no closest_scores")?;
        ranked.push(
            ids.iter()
                .zip(scores)
                .filter_map(|(i, sc)| Some((sc.as_f64()? as f32, i.as_i64()?)))
                .collect(),
        );
    }
    if ranked.len() != queries.len() {
        return Err(format!(
            "{} holds {} rows but {} queries were loaded",
            tests_path.display(),
            ranked.len(),
            queries.len()
        ));
    }

    let mut stats = VerifyStats {
        queries_checked: 0,
        positions_compared: 0,
        max_score_delta: 0.0,
        tie_swaps: 0,
    };
    for (i, q) in queries.iter().enumerate() {
        let hits = msmarco::InPrefixHits {
            hits: q
                .shipped
                .iter()
                .filter_map(|(_, pid, cos)| local_of.get(pid).map(|l| (*l, pid.clone(), *cos)))
                .collect(),
        };
        if hits.hits.is_empty() {
            continue;
        }
        let check = verify_head_against_shipped(&ranked[i], &hits, &docid_at, tol)
            .map_err(|e| format!("query {} ({:?}): {e}", q.id, q.text))?;
        stats.queries_checked += 1;
        stats.positions_compared += check.compared;
        stats.tie_swaps += check.tie_swaps;
        stats.max_score_delta = stats.max_score_delta.max(check.max_score_delta);
    }

    msmarco::check_coverage(
        variant.limit,
        stats.queries_checked,
        stats.positions_compared,
    )?;
    println!(
        "{} verified at tolerance {:.0e}: {} queries / {} ranking positions, max cosine delta \
         {:.2e}, {} tie reorderings.",
        variant.dataset_name,
        tol,
        stats.queries_checked,
        stats.positions_compared,
        stats.max_score_delta,
        stats.tie_swaps
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

struct Query {
    id: String,
    text: String,
    trec_year: Option<i64>,
    /// Normalized, `DIM` long.
    emb: Vec<f32>,
    /// The shipped global top-1k entries that land inside this variant's
    /// selection. For a prefix this is fixed at load time; for a sample it is
    /// rebuilt once the selection is known.
    hits: msmarco::InPrefixHits,
    /// The full shipped top-1k, kept so a sampled variant can remap it.
    shipped: Vec<(i64, String, f32)>,
}

fn load_queries(
    client: &reqwest::blocking::Client,
    cache: &Path,
    token: Option<&str>,
    variant: &Variant,
) -> Result<Vec<Query>, String> {
    let path = cache.join("queries.jsonl.gz");
    if !path.exists() {
        println!("Downloading queries ({})", msmarco::queries_url());
        let mut resp = get(client, &msmarco::queries_url(), token, None)?;
        let tmp = path.with_extension("gz.part");
        let mut out = File::create(&tmp).map_err(|e| format!("create {}: {}", tmp.display(), e))?;
        std::io::copy(&mut resp, &mut out).map_err(|e| format!("download queries: {e}"))?;
        out.flush().map_err(|e| format!("flush queries: {e}"))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("rename {}: {}", tmp.display(), e))?;
    }

    let file = File::open(&path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let reader = BufReader::new(MultiGzDecoder::new(BufReader::new(file)));

    let mut queries = Vec::with_capacity(msmarco::QUERY_COUNT);
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("read queries line {}: {e}", lineno + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("parse queries line {}: {e}", lineno + 1))?;

        let mut emb = f32_array(&v, "emb", lineno)?;
        if emb.len() != msmarco::DIM {
            return Err(format!(
                "query on line {} has a {}-dim embedding, expected {}",
                lineno + 1,
                emb.len(),
                msmarco::DIM
            ));
        }
        normalize_in_place(&mut emb);

        let offsets: Vec<i64> = v
            .get("top1k_offsets")
            .and_then(|x| x.as_array())
            .ok_or_else(|| format!("query on line {} has no top1k_offsets", lineno + 1))?
            .iter()
            .map(|x| {
                x.as_i64()
                    .ok_or_else(|| format!("non-integer top1k offset on line {}", lineno + 1))
            })
            .collect::<Result<_, _>>()?;
        let passage_ids: Vec<String> = v
            .get("top1k_passage_ids")
            .and_then(|x| x.as_array())
            .ok_or_else(|| format!("query on line {} has no top1k_passage_ids", lineno + 1))?
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("non-string passage id on line {}", lineno + 1))
            })
            .collect::<Result<_, _>>()?;
        let cossims = f32_array(&v, "top1k_cossim", lineno)?;

        let hits = in_prefix_hits(&offsets, &passage_ids, &cossims, variant.limit)
            .map_err(|e| format!("query on line {}: {e}", lineno + 1))?;

        let shipped: Vec<(i64, String, f32)> = (0..offsets.len())
            .map(|i| (offsets[i], passage_ids[i].clone(), cossims[i]))
            .collect();
        queries.push(Query {
            id: v
                .get("_id")
                .map(json_scalar_to_string)
                .unwrap_or_else(|| format!("line{}", lineno + 1)),
            text: v
                .get("text")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string(),
            trec_year: v.get("trec-year").and_then(|x| x.as_i64()),
            emb,
            hits,
            shipped,
        });
    }

    if queries.len() != msmarco::QUERY_COUNT {
        return Err(format!(
            "expected {} queries in queries.jsonl.gz, found {} — the cached file at {} is \
             truncated or the upstream set changed; delete it and retry",
            msmarco::QUERY_COUNT,
            queries.len(),
            path.display()
        ));
    }

    let with_hits = queries.iter().filter(|q| !q.hits.hits.is_empty()).count();
    let total_hits: usize = queries.iter().map(|q| q.hits.hits.len()).sum();
    if variant.sampling.needs_full_scan() {
        // `hits` is still the PREFIX mapping here — meaningless for a sampled
        // variant, which cannot know its selection until the corpus is read. It
        // is rebuilt, reported and gated after `write_corpus_sampled`, so say
        // nothing about coverage yet rather than print a number that is not the
        // one being used.
        println!("Loaded {} queries.", queries.len());
    } else {
        println!(
            "Loaded {} queries. {} of them have at least one of their global top-1000 inside this \
             {}-passage prefix ({} hits total) — those are what the ground truth is cross-checked \
             against.",
            queries.len(),
            with_hits,
            variant.limit,
            total_hits
        );
    }
    // Fail here, not after the download and the brute force: these two numbers
    // already bound what the cross-check can reach (review of #319, item 4).
    if !variant.sampling.needs_full_scan() {
        msmarco::check_coverage_upper_bound(variant.limit, with_hits, total_hits)?;
    }
    Ok(queries)
}

/// `_id` is a string in the query file, but MS MARCO topic ids are numeric and a
/// future export could type them as numbers; keep the exact digits either way.
fn json_scalar_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn f32_array(v: &serde_json::Value, key: &str, lineno: usize) -> Result<Vec<f32>, String> {
    v.get(key)
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("query on line {} has no {} array", lineno + 1, key))?
        .iter()
        .map(|x| {
            x.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| format!("non-numeric {} entry on line {}", key, lineno + 1))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Shard plan
// ---------------------------------------------------------------------------

/// Fetch each shard's row count (a 512-byte ranged GET, never the multi-GB
/// payload) until the variant's prefix is covered, then hand the arithmetic to
/// the library's [`msmarco::plan_shards`], which is unit-tested across shard
/// boundaries — the case no build below 10M reaches.
fn plan_shards(
    client: &reqwest::blocking::Client,
    token: Option<&str>,
    variant: &Variant,
) -> Result<Vec<ShardTake>, String> {
    let mut rows = Vec::new();
    let mut so_far = 0u64;
    for shard in 0..msmarco::SHARDS {
        if so_far >= variant.limit {
            break;
        }
        // The header is 128 bytes on every shipped shard; ask for 512 so a
        // longer one still parses in a single round trip.
        let url = msmarco::npy_url(shard);
        let mut resp = get(client, &url, token, Some((0, 511)))?;
        let mut head = Vec::new();
        resp.read_to_end(&mut head)
            .map_err(|e| format!("read header of {url}: {e}"))?;
        let header = parse_f16_npy_header(&head).map_err(|e| format!("{url}: {e}"))?;
        if header.cols != msmarco::DIM as u64 {
            return Err(format!(
                "{url}: {}-dim vectors, expected {}",
                header.cols,
                msmarco::DIM
            ));
        }
        rows.push(header.rows);
        so_far += header.rows;
    }

    let plan = msmarco::plan_shards(&rows, variant.limit)?;
    println!(
        "Plan: {} shard(s), {} passages ({} from shard {} onwards)",
        plan.len(),
        variant.limit,
        plan.last().map(|p| p.take).unwrap_or(0),
        plan.last().map(|p| p.shard).unwrap_or(0)
    );
    Ok(plan)
}

// ---------------------------------------------------------------------------
// Pass A — vectors.npy + payloads.jsonl
// ---------------------------------------------------------------------------

/// What [`write_corpus`] reports back: the docids of the offsets the
/// verification pass will ask about, and the longest value seen per string
/// field.
type Corpus = (HashMap<i64, String>, HashMap<String, usize>);

/// Stream the planned shards into `vectors.npy` and `payloads.jsonl`, returning
/// the docids of the offsets the verification pass will ask about.
///
/// The embeddings and the metadata are two separate files that are only related
/// positionally, so they are consumed in lockstep: one npy row, one json.gz
/// line. Whenever a shard is consumed in full, its line count is checked to be
/// exactly its row count — a longer or shorter metadata file means the pairing
/// is off and every passage would carry a neighbour's metadata.
fn write_corpus(
    client: &reqwest::blocking::Client,
    token: Option<&str>,
    variant: &Variant,
    plan: &[ShardTake],
    dir: &Path,
    wanted_docids: &std::collections::HashSet<i64>,
) -> Result<Corpus, String> {
    let vectors_path = staged(dir, "vectors.npy");
    let payloads_path = staged(dir, "payloads.jsonl");
    let mut npy = NpyF32Writer::create(&vectors_path, variant.limit, msmarco::DIM)?;
    let mut payloads = BufWriter::with_capacity(
        1 << 20,
        File::create(&payloads_path)
            .map_err(|e| format!("create {}: {}", payloads_path.display(), e))?,
    );

    let mut docid_at: HashMap<i64, String> = HashMap::with_capacity(wanted_docids.len());
    // Longest value seen per string field, in BYTES. Engines with typed string
    // columns declare a length cap (Milvus' VarChar `max_length`, Elasticsearch's
    // 32 766-byte `keyword` term limit), and a payload that exceeds it is
    // rejected at insert time — so the measurement belongs in the manifest
    // rather than in a surprised bug report.
    let mut max_field_bytes: HashMap<String, usize> = HashMap::new();
    let bar = progress(variant.limit, "vectors+payloads");

    for take in plan {
        let npy_url = msmarco::npy_url(take.shard);
        let header_probe = {
            let mut resp = get(client, &npy_url, token, Some((0, 511)))?;
            let mut head = Vec::new();
            resp.read_to_end(&mut head)
                .map_err(|e| format!("read header of {npy_url}: {e}"))?;
            parse_f16_npy_header(&head)?
        };
        if header_probe.rows != take.shard_rows {
            return Err(format!(
                "{npy_url}: row count changed between planning ({}) and reading ({}) — the \
                 upstream dataset was updated mid-run",
                take.shard_rows, header_probe.rows
            ));
        }

        let row_bytes = (msmarco::DIM * 2) as u64;
        let start = header_probe.row_offset(0);
        let end_inclusive = start + take.take * row_bytes - 1;
        let mut vec_stream = BufReader::with_capacity(
            1 << 22,
            get(client, &npy_url, token, Some((start, end_inclusive)))?,
        );

        let meta_url = msmarco::jsonl_url(take.shard);
        let mut meta_stream = BufReader::with_capacity(
            1 << 22,
            MultiGzDecoder::new(BufReader::with_capacity(
                1 << 22,
                get(client, &meta_url, token, None)?,
            )),
        );

        let mut raw = vec![0u8; BLOCK_ROWS * msmarco::DIM * 2];
        let mut row_in_shard = 0u64;
        let mut line = String::new();

        while row_in_shard < take.take {
            let rows = ((take.take - row_in_shard) as usize).min(BLOCK_ROWS);
            let want = rows * msmarco::DIM * 2;
            vec_stream
                .read_exact(&mut raw[..want])
                .map_err(|e| format!("{npy_url}: read rows at {}: {e}", row_in_shard))?;
            let block = msmarco::decode_f16_block(&raw[..want])?;
            msmarco::ensure_finite(&block, (take.first_id + row_in_shard) as i64, msmarco::DIM)?;

            for r in 0..rows {
                npy.write_row(&block[r * msmarco::DIM..(r + 1) * msmarco::DIM])?;

                line.clear();
                let n = read_json_line(&mut meta_stream, &mut line).map_err(|e| {
                    format!(
                        "{meta_url}: reading metadata line {}: {e}",
                        row_in_shard + r as u64
                    )
                })?;
                if n == 0 {
                    return Err(format!(
                        "{meta_url} ran out of lines at row {} of shard {}, which the NPY says \
                         has {} rows — the metadata file is shorter than the embeddings, so the \
                         two cannot be paired",
                        row_in_shard + r as u64,
                        take.shard,
                        take.shard_rows
                    ));
                }
                let record: serde_json::Value = serde_json::from_str(line.trim_end())
                    .map_err(|e| format!("{meta_url}: line {}: {e}", row_in_shard + r as u64))?;
                let payload = payload_from_passage(&record)?;

                for (field, value) in payload.as_object().into_iter().flatten() {
                    if let Some(text) = value.as_str() {
                        let e = max_field_bytes.entry(field.clone()).or_insert(0);
                        *e = (*e).max(text.len());
                    }
                }

                let id = (take.first_id + row_in_shard + r as u64) as i64;
                if wanted_docids.contains(&id) {
                    if let Some(d) = payload.get("docid").and_then(|d| d.as_str()) {
                        docid_at.insert(id, d.to_string());
                    }
                }

                serde_json::to_writer(&mut payloads, &payload)
                    .map_err(|e| format!("write payload: {e}"))?;
                payloads
                    .write_all(b"\n")
                    .map_err(|e| format!("write payload: {e}"))?;
            }

            row_in_shard += rows as u64;
            bar.inc(rows as u64);
        }

        // Only a fully consumed shard can have its tail checked; the last shard
        // of a prefix is cut short on purpose.
        if take.take == take.shard_rows {
            line.clear();
            let n = read_json_line(&mut meta_stream, &mut line)
                .map_err(|e| format!("{meta_url}: reading past the last row: {e}"))?;
            if n != 0 && !line.trim().is_empty() {
                return Err(format!(
                    "{meta_url} has more lines than shard {} has NPY rows ({}) — the metadata \
                     file is longer than the embeddings, so the two cannot be paired",
                    take.shard, take.shard_rows
                ));
            }
        }
    }

    bar.finish_and_clear();
    npy.finish()?;
    payloads
        .flush()
        .map_err(|e| format!("flush {}: {}", payloads_path.display(), e))?;

    let violations = msmarco::cap_violations(&max_field_bytes);
    for line in &violations {
        eprintln!("\t⚠ WARNING: {line}");
    }
    if !violations.is_empty() {
        eprintln!(
            "\t  The corpus is still correct and every other engine takes it; the cap is that \
             engine's own hard ceiling, so no setting on our side admits these values. \
             Recorded under `engine_cap_violations` in PREPARED.json."
        );
    }

    let missing = wanted_docids.len() - docid_at.len();
    if missing > 0 {
        return Err(format!(
            "{missing} of the {} offsets named by the shipped top-1k were never given a docid \
             while writing the corpus — the offset arithmetic is wrong",
            wanted_docids.len()
        ));
    }
    println!(
        "Wrote {} vectors and {} payload lines; captured {} docids for verification.",
        variant.limit,
        variant.limit,
        docid_at.len()
    );
    Ok((docid_at, max_field_bytes))
}

/// Row count of every shard, from its NPY header (a 512-byte ranged GET each,
/// run concurrently). The sampled path needs all 60 to know where each shard
/// starts in the global order.
fn all_shard_rows(
    client: &reqwest::blocking::Client,
    token: Option<&str>,
) -> Result<Vec<u64>, String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(DISCOVER_CONCURRENCY)
        .build()
        .map_err(|e| format!("build header pool: {e}"))?;
    let rows: Vec<Result<u64, String>> = pool.install(|| {
        (0..msmarco::SHARDS)
            .into_par_iter()
            .map(|shard| {
                let url = msmarco::npy_url(shard);
                let mut resp = get(client, &url, token, Some((0, 511)))?;
                let mut head = Vec::new();
                resp.read_to_end(&mut head)
                    .map_err(|e| format!("read header of {url}: {e}"))?;
                let h = parse_f16_npy_header(&head).map_err(|e| format!("{url}: {e}"))?;
                if h.cols != msmarco::DIM as u64 {
                    return Err(format!("{url}: {}-dim vectors", h.cols));
                }
                Ok(h.rows)
            })
            .collect()
    });
    let mut out = Vec::with_capacity(msmarco::SHARDS);
    for r in rows {
        out.push(r?);
    }
    let total: u64 = out.iter().sum();
    if total != msmarco::TOTAL_PASSAGES {
        return Err(format!(
            "the 60 shards hold {total} passages, the pinned revision should have {}",
            msmarco::TOTAL_PASSAGES
        ));
    }
    Ok(out)
}

/// Build a uniformly-sampled corpus: every passage whose `docid` hashes below
/// the threshold, from the WHOLE corpus.
///
/// Unlike the prefix path there is no cheap answer — selection depends on every
/// `docid`, so all 60 shards are read in full. Shards are processed
/// concurrently (a single CDN stream measured ~12 MB/s against ~265 MB/s for 16
/// of them) into per-shard temp files, then concatenated **in shard order** so
/// local ids still follow corpus order. That ordering is load-bearing twice
/// over: `TopK::add_block` requires ascending ids, and the ids are what the
/// ground truth is keyed by.
///
/// Returns the realized row count alongside the usual verification maps. The
/// count cannot be predicted, which is why `vectors.npy` is written through
/// [`DeferredNpyF32Writer`].
#[allow(clippy::too_many_arguments)]
fn write_corpus_sampled(
    client: &reqwest::blocking::Client,
    token: Option<&str>,
    sampling: msmarco::Sampling,
    shard_rows: &[u64],
    dir: &Path,
    tmp: &Path,
    wanted_offsets: &std::collections::HashSet<u64>,
) -> Result<SampledCorpus, String> {
    let first_id: Vec<u64> = shard_rows
        .iter()
        .scan(0u64, |acc, n| {
            let base = *acc;
            *acc += n;
            Some(base)
        })
        .collect();

    std::fs::create_dir_all(tmp).map_err(|e| format!("create {}: {}", tmp.display(), e))?;
    let bar = progress(shard_rows.iter().sum::<u64>(), "scan+select");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(DISCOVER_CONCURRENCY)
        .build()
        .map_err(|e| format!("build scan pool: {e}"))?;

    let results: Vec<Result<ShardSelection, String>> = pool.install(|| {
        (0..shard_rows.len())
            .into_par_iter()
            .map(|shard| {
                select_from_shard(
                    client,
                    token,
                    sampling,
                    shard,
                    shard_rows[shard],
                    first_id[shard],
                    tmp,
                    wanted_offsets,
                    &bar,
                )
            })
            .collect()
    });
    bar.finish_and_clear();

    let mut selections: Vec<ShardSelection> = Vec::with_capacity(shard_rows.len());
    for r in results {
        selections.push(r?);
    }
    selections.sort_by_key(|s| s.shard);

    // Local ids are assigned by walking shards in order, so a shard's ids start
    // after every earlier shard's selections.
    let mut base = 0i64;
    let mut offset_to_local: HashMap<u64, i64> = HashMap::new();
    let mut docid_at: HashMap<i64, String> = HashMap::new();
    let mut max_field_bytes: HashMap<String, usize> = HashMap::new();
    for sel in &selections {
        for (offset, within) in &sel.wanted {
            let local = base + *within;
            offset_to_local.insert(*offset, local);
            if let Some(d) = sel.docids.get(offset) {
                docid_at.insert(local, d.clone());
            }
        }
        for (k, v) in &sel.max_field_bytes {
            let e = max_field_bytes.entry(k.clone()).or_insert(0);
            *e = (*e).max(*v);
        }
        base += sel.kept as i64;
    }
    let total = base as u64;
    if total == 0 {
        return Err("the sample selected no passages at all".to_string());
    }

    // Concatenate in shard order into the real outputs.
    println!(
        "Concatenating {} shard parts ({} passages)...",
        selections.len(),
        total
    );
    let mut npy = msmarco::DeferredNpyF32Writer::create(&staged(dir, "vectors.npy"), msmarco::DIM)?;
    let payloads_path = staged(dir, "payloads.jsonl");
    let mut payloads = BufWriter::with_capacity(
        1 << 22,
        File::create(&payloads_path)
            .map_err(|e| format!("create {}: {}", payloads_path.display(), e))?,
    );
    let mut row = vec![0f32; msmarco::DIM];
    for sel in &selections {
        let vpath = tmp.join(format!("shard_{:02}.f32", sel.shard));
        let mut vf = BufReader::with_capacity(
            1 << 22,
            File::open(&vpath).map_err(|e| format!("open {}: {}", vpath.display(), e))?,
        );
        let mut raw = vec![0u8; msmarco::DIM * 4];
        for _ in 0..sel.kept {
            vf.read_exact(&mut raw)
                .map_err(|e| format!("{}: {e}", vpath.display()))?;
            for (i, c) in raw.as_chunks::<4>().0.iter().enumerate() {
                row[i] = f32::from_le_bytes(*c);
            }
            npy.write_row(&row)?;
        }
        let ppath = tmp.join(format!("shard_{:02}.jsonl", sel.shard));
        let mut pf = File::open(&ppath).map_err(|e| format!("open {}: {}", ppath.display(), e))?;
        std::io::copy(&mut pf, &mut payloads)
            .map_err(|e| format!("concat {}: {e}", ppath.display()))?;
        let _ = std::fs::remove_file(&vpath);
        let _ = std::fs::remove_file(&ppath);
    }
    let written = npy.finish()?;
    payloads
        .flush()
        .map_err(|e| format!("flush {}: {}", payloads_path.display(), e))?;
    if written != total {
        return Err(format!(
            "wrote {written} vectors but the shard selections total {total}"
        ));
    }
    let _ = std::fs::remove_dir(tmp);

    for line in msmarco::cap_violations(&max_field_bytes) {
        eprintln!("\t⚠ WARNING: {line}");
    }
    println!(
        "Selected {} of {} passages ({:.4}%); captured {} docids for verification.",
        total,
        shard_rows.iter().sum::<u64>(),
        total as f64 / shard_rows.iter().sum::<u64>() as f64 * 100.0,
        docid_at.len()
    );

    Ok(SampledCorpus {
        count: total,
        offset_to_local,
        docid_at,
        max_field_bytes,
    })
}

/// What [`write_corpus_sampled`] needs back from one shard.
struct ShardSelection {
    shard: usize,
    /// Passages kept from this shard.
    kept: u64,
    /// For each wanted global offset that was kept, its index WITHIN this shard.
    wanted: Vec<(u64, i64)>,
    docids: HashMap<u64, String>,
    max_field_bytes: HashMap<String, usize>,
}

struct SampledCorpus {
    count: u64,
    offset_to_local: HashMap<u64, i64>,
    docid_at: HashMap<i64, String>,
    max_field_bytes: HashMap<String, usize>,
}

/// Stream one shard's embeddings and metadata in lockstep, writing the selected
/// rows to per-shard temp files.
#[allow(clippy::too_many_arguments)]
fn select_from_shard(
    client: &reqwest::blocking::Client,
    token: Option<&str>,
    sampling: msmarco::Sampling,
    shard: usize,
    rows: u64,
    first_id: u64,
    tmp: &Path,
    wanted_offsets: &std::collections::HashSet<u64>,
    bar: &ProgressBar,
) -> Result<ShardSelection, String> {
    let npy_url = msmarco::npy_url(shard);
    let header = {
        let mut resp = get(client, &npy_url, token, Some((0, 511)))?;
        let mut head = Vec::new();
        resp.read_to_end(&mut head)
            .map_err(|e| format!("read header of {npy_url}: {e}"))?;
        parse_f16_npy_header(&head)?
    };
    if header.rows != rows {
        return Err(format!(
            "{npy_url}: row count changed between planning ({rows}) and reading ({})",
            header.rows
        ));
    }

    let row_bytes = (msmarco::DIM * 2) as u64;
    let start = header.row_offset(0);
    let end = start + rows * row_bytes - 1;
    let mut vec_stream =
        BufReader::with_capacity(1 << 22, get(client, &npy_url, token, Some((start, end)))?);
    let meta_url = msmarco::jsonl_url(shard);
    let mut meta_stream = BufReader::with_capacity(
        1 << 22,
        MultiGzDecoder::new(BufReader::with_capacity(
            1 << 22,
            get(client, &meta_url, token, None)?,
        )),
    );

    let mut vout = BufWriter::with_capacity(
        1 << 22,
        File::create(tmp.join(format!("shard_{shard:02}.f32")))
            .map_err(|e| format!("create shard {shard} vector part: {e}"))?,
    );
    let mut pout = BufWriter::with_capacity(
        1 << 22,
        File::create(tmp.join(format!("shard_{shard:02}.jsonl")))
            .map_err(|e| format!("create shard {shard} payload part: {e}"))?,
    );

    let mut raw = vec![0u8; BLOCK_ROWS * msmarco::DIM * 2];
    let mut line = String::new();
    let mut done: u64 = 0;
    let mut kept: i64 = 0;
    let mut wanted = Vec::new();
    let mut docids = HashMap::new();
    let mut max_field_bytes: HashMap<String, usize> = HashMap::new();

    while done < rows {
        let n = ((rows - done) as usize).min(BLOCK_ROWS);
        let want = n * msmarco::DIM * 2;
        vec_stream
            .read_exact(&mut raw[..want])
            .map_err(|e| format!("{npy_url}: rows at {done}: {e}"))?;
        let block = msmarco::decode_f16_block(&raw[..want])?;
        msmarco::ensure_finite(&block, (first_id + done) as i64, msmarco::DIM)?;

        for r in 0..n {
            line.clear();
            let got = meta_stream
                .read_line(&mut line)
                .map_err(|e| format!("{meta_url}: line {}: {e}", done + r as u64))?;
            if got == 0 {
                return Err(format!(
                    "{meta_url} ran out of lines at row {} of shard {shard}, which the NPY says \
                     has {rows} rows — the metadata is shorter than the embeddings",
                    done + r as u64
                ));
            }
            let record: serde_json::Value = serde_json::from_str(line.trim_end())
                .map_err(|e| format!("{meta_url}: line {}: {e}", done + r as u64))?;
            let payload = payload_from_passage(&record)?;
            let docid = payload
                .get("docid")
                .and_then(|d| d.as_str())
                .ok_or_else(|| format!("{meta_url}: row {} has no docid", done + r as u64))?;

            let offset = first_id + done + r as u64;
            if !sampling.keeps(offset, docid, u64::MAX) {
                continue;
            }

            for v in &block[r * msmarco::DIM..(r + 1) * msmarco::DIM] {
                vout.write_all(&v.to_le_bytes())
                    .map_err(|e| format!("write shard {shard} vector: {e}"))?;
            }
            for (field, value) in payload.as_object().into_iter().flatten() {
                if let Some(text) = value.as_str() {
                    let e = max_field_bytes.entry(field.clone()).or_insert(0);
                    *e = (*e).max(text.len());
                }
            }
            if wanted_offsets.contains(&offset) {
                wanted.push((offset, kept));
                docids.insert(offset, docid.to_string());
            }
            serde_json::to_writer(&mut pout, &payload)
                .map_err(|e| format!("write shard {shard} payload: {e}"))?;
            pout.write_all(b"\n")
                .map_err(|e| format!("write shard {shard} payload: {e}"))?;
            kept += 1;
        }
        done += n as u64;
        bar.inc(n as u64);
    }

    vout.flush()
        .map_err(|e| format!("flush shard {shard} vectors: {e}"))?;
    pout.flush()
        .map_err(|e| format!("flush shard {shard} payloads: {e}"))?;

    Ok(ShardSelection {
        shard,
        kept: kept as u64,
        wanted,
        docids,
        max_field_bytes,
    })
}

/// Read one newline-terminated line into `line`, returning bytes read (0 at
/// EOF). Thin wrapper so the two error sites above read the same way.
fn read_json_line<R: BufRead>(reader: &mut R, line: &mut String) -> Result<usize, std::io::Error> {
    reader.read_line(line)
}

// ---------------------------------------------------------------------------
// Pass B — brute-force ground truth
// ---------------------------------------------------------------------------

/// Brute-force the top-[`msmarco::NEIGHBOURS`] for every query, reading back the
/// `vectors.npy` just written.
///
/// Reading our own output (rather than keeping the corpus in memory, or scoring
/// during the download) is deliberate: it makes the ground truth a function of
/// the exact file the engines will upload, so a bug in the writer shows up as a
/// ground-truth mismatch instead of hiding.
fn brute_force_ground_truth(
    dir: &Path,
    variant: &Variant,
    queries: &[Query],
) -> Result<Vec<Vec<(f32, i64)>>, String> {
    let path = staged(dir, "vectors.npy");
    let mut file = File::open(&path).map_err(|e| format!("open {}: {}", path.display(), e))?;

    let mut head = vec![0u8; 512];
    file.read_exact(&mut head)
        .map_err(|e| format!("read {} header: {e}", path.display()))?;
    let data_start = f32_npy_data_start(&head, variant.limit)?;
    file = File::open(&path).map_err(|e| format!("reopen {}: {}", path.display(), e))?;
    let mut reader = BufReader::with_capacity(1 << 22, file);
    std::io::copy(&mut (&mut reader).take(data_start), &mut std::io::sink())
        .map_err(|e| format!("skip {} header: {e}", path.display()))?;

    let flat_queries: Vec<f32> = queries.iter().flat_map(|q| q.emb.iter().copied()).collect();
    let mut acc = TopK::new(flat_queries, msmarco::DIM, msmarco::NEIGHBOURS)?;

    let bar = progress(variant.limit, "ground truth");
    let mut raw = vec![0u8; BLOCK_ROWS * msmarco::DIM * 4];
    let mut done = 0u64;
    while done < variant.limit {
        let rows = ((variant.limit - done) as usize).min(BLOCK_ROWS);
        let want = rows * msmarco::DIM * 4;
        reader
            .read_exact(&mut raw[..want])
            .map_err(|e| format!("read {} at row {}: {e}", path.display(), done))?;

        let mut block: Vec<f32> = raw[..want]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        msmarco::ensure_finite(&block, done as i64, msmarco::DIM)?;
        // The engines see normalized vectors (cosine distance), so the ranking
        // must be built from normalized vectors too.
        for row in block.as_chunks_mut::<{ msmarco::DIM }>().0 {
            normalize_in_place(row);
        }

        acc.add_block(&block, done as i64)?;
        done += rows as u64;
        bar.inc(rows as u64);
    }
    bar.finish_and_clear();

    Ok(acc.finish())
}

/// Locate the payload offset of the `'<f4'` NPY just written, and confirm it
/// still declares the expected row count.
fn f32_npy_data_start(head: &[u8], expect_rows: u64) -> Result<u64, String> {
    let expected = msmarco::npy_f32_header(expect_rows, msmarco::DIM);
    if head.len() < expected.len() || head[..expected.len()] != expected[..] {
        return Err(
            "vectors.npy does not start with the header it was written with — the file was \
             modified or truncated between the two passes"
                .to_string(),
        );
    }
    Ok(expected.len() as u64)
}

// ---------------------------------------------------------------------------
// Verification + outputs
// ---------------------------------------------------------------------------

struct VerifyStats {
    queries_checked: usize,
    positions_compared: usize,
    max_score_delta: f32,
    tie_swaps: usize,
}

fn verify(
    limit: u64,
    queries: &[Query],
    ranked: &[Vec<(f32, i64)>],
    docid_at: &HashMap<i64, String>,
) -> Result<VerifyStats, String> {
    let mut stats = VerifyStats {
        queries_checked: 0,
        positions_compared: 0,
        max_score_delta: 0.0,
        tie_swaps: 0,
    };
    for (i, q) in queries.iter().enumerate() {
        if q.hits.hits.is_empty() {
            continue;
        }
        let check = verify_head_against_shipped(&ranked[i], &q.hits, docid_at, COSINE_TOLERANCE)
            .map_err(|e| format!("query {} ({:?}): {e}", q.id, q.text))?;
        stats.queries_checked += 1;
        stats.positions_compared += check.compared;
        stats.tie_swaps += check.tie_swaps;
        stats.max_score_delta = stats.max_score_delta.max(check.max_score_delta);
    }
    // The floor, not just the report: this cross-check IS the correctness
    // argument for the corpus, so coverage that quietly collapsed must abort
    // rather than print a reassuring number (review of #319, item 1).
    msmarco::check_coverage(limit, stats.queries_checked, stats.positions_compared)?;
    let (min_queries, min_positions) = msmarco::coverage_floor(limit);
    println!(
        "Verified {} queries / {} ranking positions against the shipped global top-1k: \
         max cosine delta {:.2e}, {} tie reorderings (floor: {}/{}).",
        stats.queries_checked,
        stats.positions_compared,
        stats.max_score_delta,
        stats.tie_swaps,
        min_queries,
        min_positions
    );
    Ok(stats)
}

fn write_tests(dir: &Path, queries: &[Query], ranked: &[Vec<(f32, i64)>]) -> Result<(), String> {
    let path = staged(dir, "tests.jsonl");
    let mut out = BufWriter::with_capacity(
        1 << 20,
        File::create(&path).map_err(|e| format!("create {}: {}", path.display(), e))?,
    );
    for (i, q) in queries.iter().enumerate() {
        let row = test_row(&q.emb, &q.id, &q.text, q.trec_year, &ranked[i]);
        serde_json::to_writer(&mut out, &row).map_err(|e| format!("write tests.jsonl: {e}"))?;
        out.write_all(b"\n")
            .map_err(|e| format!("write tests.jsonl: {e}"))?;
    }
    out.flush()
        .map_err(|e| format!("flush {}: {}", path.display(), e))?;
    // Report the published name, not the `.part` it is staged under.
    println!(
        "Wrote {} queries to {}",
        queries.len(),
        dir.join("tests.jsonl").display()
    );
    Ok(())
}

fn write_manifest(
    dir: &Path,
    variant: &Variant,
    plan: &[ShardTake],
    stats: &VerifyStats,
    queries: &[Query],
    max_field_bytes: &HashMap<String, usize>,
) -> Result<(), String> {
    let manifest = serde_json::json!({
        "dataset": variant.dataset_name,
        "source": format!("https://huggingface.co/datasets/{}", msmarco::HF_REPO),
        "passages": variant.limit,
        "dimensions": msmarco::DIM,
        "distance": "cosine",
        "neighbours": msmarco::NEIGHBOURS,
        "queries": queries.len(),
        "sampling": match variant.sampling {
            msmarco::Sampling::Prefix => serde_json::json!({
                "mode": "prefix",
                "rule": "the first N passages of the corpus order",
                "note": "the corpus is docid-ordered, which tracks URL, so this is an \
                         alphabetically bounded slice — representative geometrically \
                         (measured), NOT representative in metadata",
            }),
            msmarco::Sampling::Crc32 { threshold } => serde_json::json!({
                "mode": "crc32",
                "rule": format!(
                    "keep where zlib.crc32(docid) % {} < {}",
                    msmarco::CRC32_BUCKETS, threshold
                ),
                "threshold": threshold,
                "buckets": msmarco::CRC32_BUCKETS,
                "note": "uniform over the whole corpus in metadata as well as geometry; \
                         the same selection function the Redis Enterprise MS MARCO suite uses",
            }),
        },
        "shards_consumed": plan
            .iter()
            .map(|p| serde_json::json!({
                "shard": p.shard,
                "stem": msmarco::shard_stem(p.shard),
                "shard_rows": p.shard_rows,
                "rows_taken": p.take,
                "first_global_offset": p.first_id,
            }))
            .collect::<Vec<_>>(),
        "ground_truth": {
            "method": "brute force over this prefix, on the normalized float32 vectors in vectors.npy",
            "why_not_shipped_top1k":
                "the shipped top1k_offsets rank the full 113,520,750-passage corpus; restricted \
                 to a prefix it is neither complete nor correctly ordered past the few hits it \
                 retains",
            "cross_checked_against_shipped_top1k": {
                "queries": stats.queries_checked,
                "positions": stats.positions_compared,
                "max_cosine_delta": stats.max_score_delta,
                "tie_reorderings": stats.tie_swaps,
                "tolerance": COSINE_TOLERANCE,
            },
        },
        // Machine-readable, because the console warning scrolls away and this is
        // what decides whether an engine can ingest the corpus at all. Empty on
        // the 100K prefix; the 1M one exceeds Milvus' ceiling (see below).
        "engine_cap_violations": msmarco::cap_violations(max_field_bytes),
        "payload_fields": msmarco::PAYLOAD_FIELDS
            .iter()
            .map(|(name, ty)| serde_json::json!({
                "field": name,
                "schema_type": ty,
                "max_bytes": max_field_bytes.get(*name),
            }))
            .collect::<Vec<_>>(),
        "prepared_at": chrono::Utc::now().to_rfc3339(),
    });
    let path = staged(dir, "PREPARED.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write {}: {}", path.display(), e))
}

fn progress(total: u64, what: &str) -> ProgressBar {
    let bar = ProgressBar::new(total);
    bar.set_style(
        ProgressStyle::with_template(&format!(
            "{{spinner}} {what} [{{bar:40}}] {{pos}}/{{len}} ({{eta}})"
        ))
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=> "),
    );
    bar
}
