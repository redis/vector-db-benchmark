//! MS MARCO v2.1 (TREC-RAG 2024) + Cohere `embed-english-v3` dataset preparation.
//!
//! The upstream corpus lives on Hugging Face as
//! [`CohereLabs/msmarco-v2.1-embed-english-v3`][hf]: 113,520,750 passages, each
//! with a 1024-dim Cohere Embed v3 English embedding (cosine), **and** the real
//! MS MARCO document metadata (`docid`, `url`, `title`, `headings`, `segment`
//! text, `start_char`, `end_char`). That metadata is the point of registering it
//! here — every other text-embedding dataset we ship is vectors only, so none of
//! them can exercise filtered / full-text search on a *real* corpus.
//!
//! [hf]: https://huggingface.co/datasets/CohereLabs/msmarco-v2.1-embed-english-v3
//!
//! # Layout on HF
//!
//! * `passages_npy/msmarco_v2.1_doc_segmented_{00..59}.npy` — the embeddings, as
//!   **float16** (`'<f2'`) matrices. Row `r` of shard `s` is the passage at
//!   global offset `sum(rows of shards < s) + r`.
//! * `passages_jsonl/msmarco_v2.1_doc_segmented_{00..59}.json.gz` — the metadata,
//!   one JSON object per line, in the SAME order as the npy rows. This 1:1
//!   row/line correspondence is load-bearing: get it wrong and every passage
//!   carries another passage's metadata, which no recall number would reveal.
//!   [`verify_head_against_shipped`] is the guard.
//! * `queries_jsonl/queries.jsonl.gz` — 1677 TREC-DL 2021-2023 queries with their
//!   embedding and a brute-forced global `top1k` (offsets, passage ids, cosines).
//!
//! # Why the shipped ground truth cannot be reused for a subset
//!
//! `top1k_offsets` is the top-1000 over the **whole** 113.5M corpus. Truncating
//! it to the offsets that happen to land inside an N-passage prefix does NOT give
//! the top-k of that prefix: a passage ranked 1001st globally can easily be in
//! the prefix and outrank everything that survived truncation. A 1M prefix keeps
//! only ~9 of any query's global top-1000, so the truncated list would be both
//! too short and, past those ~9, plain wrong — a textbook silently-wrong ground
//! truth. So subsets get a genuine brute-force ranking ([`TopK`]).
//!
//! The shipped list is still worth everything as a **cross-check**, and that is
//! what it is used for here. Every in-prefix global-top-1k hit outranks every
//! in-prefix passage that is *not* in the global top-1k (the latter all score
//! below the 1000th global cosine, which the former by definition do not). So the
//! first `m` entries of our brute-forced ranking must equal, id for id, the `m`
//! in-prefix shipped hits — where `m` is however many of those there are. That
//! single assertion simultaneously proves the brute force, the npy↔jsonl row
//! alignment, and the cross-shard offset arithmetic.

use std::collections::HashMap;

use rayon::prelude::*;

/// The Hugging Face repo the corpus is fetched from.
pub const HF_REPO: &str = "CohereLabs/msmarco-v2.1-embed-english-v3";

/// The exact upstream commit every file is fetched from.
///
/// **Not `main`.** The claim these datasets rest on is that two runs reporting
/// `msmarco-cohere-1024-1M-cosine` uploaded the same corpus. Against a moving
/// ref that holds for the passage *count* and nothing else: a re-export upstream
/// would silently redefine what the name means, and no prepared directory would
/// record which export it came from. Each directory is internally consistent
/// either way — ground truth is brute-forced over whatever bytes arrived — so
/// this is not a wrong-recall bug. It is a cross-run comparability bug, which is
/// the thing this harness exists to prevent.
///
/// `write_corpus` already refuses a shard whose row count changed between
/// planning and reading; this is the same guard stretched across runs.
pub const HF_REVISION: &str = "e78737fe92ac1b783211b705c12207ca75fcc9b7";

/// Embedding dimensionality of Cohere `embed-english-v3.0`.
pub const DIM: usize = 1024;

/// Number of `passages_npy` / `passages_jsonl` shards.
pub const SHARDS: usize = 60;

/// Passages in the full corpus, per the dataset card.
pub const TOTAL_PASSAGES: u64 = 113_520_750;

/// Ground-truth width written into `tests.jsonl`.
///
/// 1000, not the 100 the other compound datasets carry, so it matches the depth
/// of the upstream `top1k_*` lists this corpus is built from: recall@k is then
/// answerable for any k up to 1000 without re-preparing, and the full depth of
/// the shipped oracle stays comparable against our own ranking.
///
/// Consequences worth knowing. It is what `metrics_schema.ground_truth` reports
/// as the row width, and a config that sets `top` above its own ground truth's
/// width can never reach recall 1.0 — at width 1000 that ceiling is simply far
/// away. It also makes `tests.jsonl` roughly three times larger (~84 MB for the
/// 1677 queries) and gives the brute force a 1000-deep list to maintain per
/// query instead of 100.
pub const NEIGHBOURS: usize = 1000;

/// Queries in `queries_jsonl/queries.jsonl.gz` (TREC-DL 2021 + 2022 + 2023).
pub const QUERY_COUNT: usize = 1677;

/// Payload fields carried over from the MS MARCO segmented-document records,
/// with the `schema` type each is registered under in `datasets/datasets.json`.
/// Kept here so the prepared payloads and the registered schema cannot drift.
pub const PAYLOAD_FIELDS: &[(&str, &str)] = &[
    ("docid", "keyword"),
    ("url", "keyword"),
    ("title", "text"),
    ("headings", "text"),
    ("segment", "text"),
    ("start_char", "int"),
    ("end_char", "int"),
];

/// Number of hash buckets the uniform sampler partitions the corpus into.
///
/// One million, not the 100 the Redis Enterprise MS MARCO suite uses, because
/// the smallest variant here is 0.088% of the corpus — at 100 buckets the
/// coarsest selectable fraction is 1%, which overshoots it by more than 11x.
pub const CRC32_BUCKETS: u32 = 1_000_000;

/// Which bucket a passage falls in, from its `docid`.
///
/// CRC-32 (ISO-HDLC), bit-identical to Python's `zlib.crc32` — the same function
/// the Redis Enterprise MS MARCO suite samples its corpus with
/// (`crc32(docid) % 100 < PCT`). Matching it exactly is deliberate: it makes the
/// two benchmarks' corpora comparable by construction, and it means the
/// selection can be reproduced from a three-line Python script by anyone
/// checking our work.
pub fn crc32_bucket(docid: &str) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(docid.as_bytes());
    h.finalize() % CRC32_BUCKETS
}

/// Pull `docid` out of a `passages_jsonl` line without parsing the whole object.
///
/// The discovery scan reads every one of the 113,520,750 lines and needs nothing
/// but this one field; a full `serde_json` parse of ~180 GB of JSON to reach the
/// first key would dominate the run. Falls back to a real parse when the fast
/// path does not match, so a change in upstream field order degrades to slow
/// rather than to wrong.
pub fn extract_docid(line: &str) -> Option<String> {
    if let Some(rest) = line.strip_prefix("{\"docid\":") {
        let rest = rest.trim_start();
        if let Some(rest) = rest.strip_prefix('"') {
            if let Some(end) = rest.find('"') {
                return Some(rest[..end].to_string());
            }
        }
    }
    serde_json::from_str::<serde_json::Value>(line)
        .ok()?
        .get("docid")?
        .as_str()
        .map(str::to_string)
}

/// A count of passages per [`crc32_bucket`], over the whole corpus.
///
/// One scan produces the realized sample size for EVERY possible threshold (a
/// prefix sum), so thresholds can be chosen without rescanning — which matters
/// because the scan is ~100 GB of gzipped metadata.
pub struct BucketHistogram {
    counts: Vec<u64>,
}

impl Default for BucketHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl BucketHistogram {
    pub fn new() -> Self {
        Self {
            counts: vec![0; CRC32_BUCKETS as usize],
        }
    }

    pub fn add(&mut self, docid: &str) {
        self.counts[crc32_bucket(docid) as usize] += 1;
    }

    /// Fold another shard's counts in. Addition, so shards can be scanned in
    /// any order or concurrently.
    pub fn merge(&mut self, other: &BucketHistogram) {
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += b;
        }
    }

    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Exactly how many passages a threshold selects.
    pub fn count_below(&self, threshold: u32) -> u64 {
        self.counts[..(threshold as usize).min(self.counts.len())]
            .iter()
            .sum()
    }

    /// The threshold whose realized count is closest to `target`, with that
    /// count. Searched by binary search over the prefix sums, which are
    /// monotone.
    pub fn threshold_for(&self, target: u64) -> (u32, u64) {
        let (mut lo, mut hi) = (0u32, CRC32_BUCKETS);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.count_below(mid) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // `lo` is the smallest threshold reaching `target`; the one below it may
        // be closer from underneath.
        let up = (lo, self.count_below(lo));
        if lo == 0 {
            return up;
        }
        let down = (lo - 1, self.count_below(lo - 1));
        if target.abs_diff(down.1) <= target.abs_diff(up.1) {
            down
        } else {
            up
        }
    }

    pub fn to_json(&self, targets: &[u64]) -> serde_json::Value {
        serde_json::json!({
            "buckets": CRC32_BUCKETS,
            "total_passages": self.total(),
            "targets": targets.iter().map(|t| {
                let (th, n) = self.threshold_for(*t);
                serde_json::json!({"target": t, "threshold": th, "realized": n})
            }).collect::<Vec<_>>(),
        })
    }
}

/// Whether a passage is in the sample: its `docid` hashes below `threshold`.
///
/// Selection is by HASH, never by position. A position-based prefix was tried
/// and removed: the corpus is ordered by `docid`, which tracks URL, so the first
/// N passages are an alphabetically bounded slice — the first 100,000 span only
/// `0-60.reviews` to `acqnotes.com`, and `url` contains "nih" zero times in
/// them. That does not distort the vector geometry (a 20k block at the head of
/// shard 00 measures mean nearest-neighbour cosine 0.8825 and mean random-pair
/// cosine 0.1442, against 0.8759 / 0.1499 for a sample spread over all 60
/// shards), so prefix recall numbers were not wrong — but this corpus exists
/// for its metadata, and a default that skews the metadata is the wrong default.
///
/// It also means selection cannot be known from any prefix of the input: every
/// build reads the whole corpus.
pub fn keeps(docid: &str, threshold: u32) -> bool {
    crc32_bucket(docid) < threshold
}

/// A registered size of the dataset. Each maps to one entry in
/// `datasets/datasets.json`, and `limit` is that entry's `vector_count`.
///
/// The size is a property of the NAME, not a command-line knob: two runs that
/// both say `msmarco-cohere-1024-1M-cosine` must have uploaded
/// byte-identical corpora, or the results are not comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variant {
    /// Name in `datasets/datasets.json`.
    pub dataset_name: &'static str,
    /// Directory under `datasets/`, matching the entry's `path`.
    ///
    /// Named by THRESHOLD (`crc32-t8805`), not by a round size. A path like
    /// `1M-crc32` would advertise 1,000,000 points the corpus does not have,
    /// which `config.rs`'s #224 guard rejects on precisely the reasoning that
    /// makes it wrong: a path naming its own size is treated as authoritative.
    pub dir: &'static str,
    /// Exactly how many passages the variant holds — the REALIZED count of its
    /// threshold, measured by the `--discover-crc32` scan over all 113,520,750
    /// docids. A hash threshold cannot be made to land on a round number, so the
    /// round number is in the name and the true count is here.
    pub limit: u64,
    /// `crc32_bucket(docid) < threshold` selects this variant's passages.
    pub threshold: u32,
}

/// Every registered size, smallest first.
///
/// One family: every passage whose `docid` hashes below a threshold, i.e. a
/// uniform sample of the whole 113.5M-passage corpus. A prefix family existed
/// briefly and was removed — see [`Sampling`] for why position-based selection
/// is the wrong default for a corpus whose point is its metadata.
///
/// The thresholds nest (886 < 8805 < 88075), so each variant is a strict subset
/// of the larger ones and results stay comparable across sizes.
pub const VARIANTS: &[Variant] = &[
    // Realized counts measured by `--discover-crc32` over all 113,520,750
    // docids at revision e78737fe: 99,964 / 1,000,044 / 9,999,959.
    Variant {
        dataset_name: "msmarco-cohere-1024-100K-cosine",
        dir: "msmarco-cohere-1024/crc32-t886",
        limit: 99_964,
        threshold: 886,
    },
    Variant {
        dataset_name: "msmarco-cohere-1024-1M-cosine",
        dir: "msmarco-cohere-1024/crc32-t8805",
        limit: 1_000_044,
        threshold: 8_805,
    },
    Variant {
        dataset_name: "msmarco-cohere-1024-10M-cosine",
        dir: "msmarco-cohere-1024/crc32-t88075",
        limit: 9_999_959,
        threshold: 88_075,
    },
];

/// Look a variant up by its registered dataset name.
pub fn variant(dataset_name: &str) -> Option<&'static Variant> {
    VARIANTS.iter().find(|v| v.dataset_name == dataset_name)
}

/// `msmarco_v2.1_doc_segmented_07` — the stem both the npy and the json.gz shard
/// share.
pub fn shard_stem(shard: usize) -> String {
    format!("msmarco_v2.1_doc_segmented_{:02}", shard)
}

/// URL of one embedding shard (float16 npy).
pub fn npy_url(shard: usize) -> String {
    format!(
        "https://huggingface.co/datasets/{}/resolve/{}/passages_npy/{}.npy",
        HF_REPO,
        HF_REVISION,
        shard_stem(shard)
    )
}

/// URL of one metadata shard (gzipped JSONL).
pub fn jsonl_url(shard: usize) -> String {
    format!(
        "https://huggingface.co/datasets/{}/resolve/{}/passages_jsonl/{}.json.gz",
        HF_REPO,
        HF_REVISION,
        shard_stem(shard)
    )
}

/// URL of the query file (gzipped JSONL, ~59 MB).
pub fn queries_url() -> String {
    format!(
        "https://huggingface.co/datasets/{}/resolve/{}/queries_jsonl/queries.jsonl.gz",
        HF_REPO, HF_REVISION
    )
}

// ---------------------------------------------------------------------------
// float16 NPY header
// ---------------------------------------------------------------------------

/// The shape and data offset of a `'<f2'` NPY file, parsed from its header
/// alone — the point being that a 3.6 GB shard's row count costs a 256-byte
/// ranged GET, not a download.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F16NpyHeader {
    pub rows: u64,
    pub cols: u64,
    /// Byte offset where the f16 payload starts.
    pub data_start: u64,
}

impl F16NpyHeader {
    /// Byte offset of row `row`.
    pub fn row_offset(&self, row: u64) -> u64 {
        self.data_start + row * self.cols * 2
    }
}

/// Parse a `'<f2'` NPY header out of the first bytes of the file.
///
/// Rejects anything that is not little-endian float16, C-order and 2-D: the
/// caller decodes the payload by raw byte arithmetic, so a silently different
/// dtype or a Fortran-order matrix would transpose or garble every vector rather
/// than fail.
pub fn parse_f16_npy_header(bytes: &[u8]) -> Result<F16NpyHeader, String> {
    if bytes.len() < 10 {
        return Err("NPY header: fewer than 10 bytes available".to_string());
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err("NPY header: bad magic (not an NPY file)".to_string());
    }
    let major = bytes[6];
    let (dict_len, dict_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("NPY header: v2/v3 length prefix truncated".to_string());
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12usize,
            )
        }
        other => return Err(format!("NPY header: unsupported major version {}", other)),
    };
    let dict_end = dict_start
        .checked_add(dict_len)
        .ok_or_else(|| "NPY header: length overflow".to_string())?;
    if bytes.len() < dict_end {
        return Err(format!(
            "NPY header: need {} bytes, only {} available",
            dict_end,
            bytes.len()
        ));
    }
    let dict = std::str::from_utf8(&bytes[dict_start..dict_end])
        .map_err(|e| format!("NPY header: not UTF-8: {}", e))?;

    if !(dict.contains("'descr': '<f2'") || dict.contains("\"descr\": \"<f2\"")) {
        return Err(format!(
            "NPY header: expected little-endian float16 ('<f2'), got: {}",
            dict.trim()
        ));
    }
    if !(dict.contains("'fortran_order': False") || dict.contains("\"fortran_order\": false")) {
        return Err("NPY header: Fortran-order arrays are not supported".to_string());
    }

    let open = dict
        .find("'shape'")
        .or_else(|| dict.find("\"shape\""))
        .and_then(|i| dict[i..].find('(').map(|o| i + o))
        .ok_or_else(|| "NPY header: no 'shape' tuple".to_string())?;
    let close = dict[open..]
        .find(')')
        .map(|c| open + c)
        .ok_or_else(|| "NPY header: unterminated 'shape' tuple".to_string())?;
    let dims: Vec<u64> = dict[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u64>()
                .map_err(|_| format!("NPY header: non-integer shape entry {:?}", s))
        })
        .collect::<Result<_, _>>()?;
    if dims.len() != 2 {
        return Err(format!(
            "NPY header: expected a 2-D shape, got {} dimension(s)",
            dims.len()
        ));
    }

    Ok(F16NpyHeader {
        rows: dims[0],
        cols: dims[1],
        data_start: dict_end as u64,
    })
}

/// Decode a block of little-endian float16 values into `f32`.
///
/// `bytes.len()` must be even; a trailing half value means the caller's ranged
/// GET was cut mid-element, which would shift every subsequent vector by one
/// byte, so it is an error rather than a truncation.
pub fn decode_f16_block(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "float16 block has an odd byte length ({}), so it ends mid-value",
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::f16::from_le_bytes(*c).to_f32())
        .collect())
}

/// Scale `v` to unit length in place. A zero vector is left untouched (there is
/// no meaningful direction to give it), matching `read_npy_vectors`.
///
/// Ground truth MUST be computed on normalized vectors, because that is exactly
/// what the engines are handed: `Dataset::needs_normalization()` is true for
/// `cosine`, so `read_npy_vectors` normalizes on the way in. Ranking by raw dot
/// product instead would produce a different — and, against a cosine index,
/// wrong — ordering.
pub fn normalize_in_place(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming f32 NPY writer
// ---------------------------------------------------------------------------

/// Build the version-1.0 header for a C-order `'<f4'` array of `rows` x `cols`.
pub fn npy_f32_header(rows: u64, cols: usize) -> Vec<u8> {
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        rows, cols
    );
    // Magic (6) + version (2) + u16 length (2) + dict + padding + '\n' must be a
    // multiple of 64.
    let unpadded = 10 + dict.len() + 1;
    let padding = (64 - (unpadded % 64)) % 64;
    let dict_len = dict.len() + padding + 1;

    let mut out = Vec::with_capacity(10 + dict_len);
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(dict_len as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat_n(b' ', padding));
    out.push(b'\n');
    out
}

/// Writes a 2-D `'<f4'` NPY whose row count is only known once the stream ends.
///
/// The uniform sampler cannot state its row count up front — selection depends
/// on every `docid` in the corpus, so the total is not known until the last
/// shard is read. Rather than buffer 40 GB or copy the file afterwards, this
/// reserves a fixed-size header, streams the rows, then seeks back and writes
/// the real one. The reservation is safe because the header is padded to a
/// 64-byte boundary: every row count this corpus can produce renders to the same
/// 128 bytes, which [`DeferredNpyF32Writer::finish`] asserts rather than assumes.
pub struct DeferredNpyF32Writer {
    file: std::fs::File,
    cols: usize,
    rows_written: u64,
    header_len: usize,
}

impl DeferredNpyF32Writer {
    pub fn create(path: &std::path::Path, cols: usize) -> Result<Self, String> {
        use std::io::Write as _;
        let mut file =
            std::fs::File::create(path).map_err(|e| format!("create {}: {}", path.display(), e))?;
        // Reserve exactly what a plausible final count will render to.
        let header_len = npy_f32_header(1, cols).len();
        file.write_all(&vec![0u8; header_len])
            .map_err(|e| format!("reserve NPY header: {e}"))?;
        Ok(Self {
            file,
            cols,
            rows_written: 0,
            header_len,
        })
    }

    pub fn write_row(&mut self, row: &[f32]) -> Result<(), String> {
        use std::io::Write as _;
        if row.len() != self.cols {
            return Err(format!(
                "NPY row has {} values, expected {}",
                row.len(),
                self.cols
            ));
        }
        let mut buf = Vec::with_capacity(row.len() * 4);
        for v in row {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        self.file
            .write_all(&buf)
            .map_err(|e| format!("write NPY row: {e}"))?;
        self.rows_written += 1;
        Ok(())
    }

    pub fn rows_written(&self) -> u64 {
        self.rows_written
    }

    /// Patch in the real header and return the row count.
    ///
    /// Errors if the real header is not the reserved size — which would mean the
    /// data starts at the wrong offset, shifting every vector. That is the one
    /// way this design can go wrong, so it is checked, not assumed.
    pub fn finish(mut self) -> Result<u64, String> {
        use std::io::{Seek, SeekFrom, Write as _};
        let header = npy_f32_header(self.rows_written, self.cols);
        if header.len() != self.header_len {
            return Err(format!(
                "NPY header for {} rows renders to {} bytes but {} were reserved — the payload \
                 would start at the wrong offset",
                self.rows_written,
                header.len(),
                self.header_len
            ));
        }
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| format!("seek to NPY header: {e}"))?;
        self.file
            .write_all(&header)
            .map_err(|e| format!("write NPY header: {e}"))?;
        self.file
            .flush()
            .map_err(|e| format!("flush NPY file: {e}"))?;
        Ok(self.rows_written)
    }
}

// ---------------------------------------------------------------------------
// Brute-force ground truth
// ---------------------------------------------------------------------------

/// Dot product of two equal-length vectors, accumulated in 8 lanes.
///
/// The obvious `zip().map().sum()` is a single dependent chain of float adds,
/// which LLVM may not reorder (float addition is not associative) and so cannot
/// vectorise. Eight independent accumulators give it eight lanes to fill. The
/// lane layout depends only on the dimensionality, so the result is bit-for-bit
/// reproducible across runs and thread counts — which matters, because this is
/// what the published ground truth is built from.
///
/// A length mismatch makes the shorter slice win via `chunks_exact`; callers
/// (only [`TopK::add_block`]) have already checked both are `dim` wide.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for (x, y) in a_rest.iter().zip(b_rest) {
        sum += x * y;
    }
    sum
}

/// Per-query running top-k over the corpus, fed one block of passages at a time.
///
/// Held as a sorted-descending `Vec` per query rather than a heap: the
/// overwhelmingly common case is "this candidate loses to the current worst",
/// which is one comparison here, and only a winner pays the insert. That still
/// holds at [`NEIGHBOURS`] = 1000 — the measured cost of going from 100 to 1000
/// was 29.7 s to 39.7 s on the 100K build — because the fast path's frequency
/// rises with `k`, not its cost.
pub struct TopK {
    k: usize,
    dim: usize,
    /// Flattened, already-normalized query matrix (`queries * dim`).
    queries: Vec<f32>,
    /// Per query, `(score, id)` sorted by descending score.
    best: Vec<Vec<(f32, i64)>>,
    /// Lowest id the next block may start at; see the `debug_assert` in
    /// [`TopK::add_block`].
    next_expected_id: i64,
}

impl TopK {
    /// `queries` is a flattened `n_queries x dim` matrix, normalized by the
    /// caller.
    pub fn new(queries: Vec<f32>, dim: usize, k: usize) -> Result<Self, String> {
        if dim == 0 || !queries.len().is_multiple_of(dim) {
            return Err(format!(
                "query matrix of {} values is not a multiple of dim {}",
                queries.len(),
                dim
            ));
        }
        let n = queries.len() / dim;
        Ok(Self {
            k,
            dim,
            queries,
            best: vec![Vec::with_capacity(k + 1); n],
            next_expected_id: i64::MIN,
        })
    }

    pub fn n_queries(&self) -> usize {
        self.best.len()
    }

    /// Score one block of normalized passage vectors against every query.
    ///
    /// `block` is a flattened `rows x dim` matrix whose first row has id
    /// `first_id`. Parallelised across queries, so each rayon task owns one
    /// query's `best` list exclusively and the block is read-shared.
    pub fn add_block(&mut self, block: &[f32], first_id: i64) -> Result<(), String> {
        // A returned error, NOT a debug_assert: every suite and the preparer itself
        // run `--release`, where a debug assertion is compiled out — so the guard
        // would have been absent from exactly the binary that writes published
        // ground truth. The tie-break below only delivers its documented
        // "ascending by id" ordering because blocks arrive in ascending id order,
        // so parallelising pass B over blocks has to fail here rather than
        // silently change what gets published.
        if first_id < self.next_expected_id {
            return Err(format!(
                "blocks must arrive in ascending id order: got a block starting at {first_id} \
                 after one ending at {}. The id tie-break depends on this, so accepting it \
                 would change the published ranking without changing any recall number.",
                self.next_expected_id
            ));
        }
        self.next_expected_id = first_id + (block.len() / self.dim.max(1)) as i64;
        if !block.len().is_multiple_of(self.dim) {
            return Err(format!(
                "passage block of {} values is not a multiple of dim {}",
                block.len(),
                self.dim
            ));
        }
        let dim = self.dim;
        let k = self.k;
        let queries = &self.queries;
        self.best.par_iter_mut().enumerate().for_each(|(qi, best)| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            for (row, doc) in block.chunks_exact(dim).enumerate() {
                // Cosine on unit vectors is the plain dot product.
                let score = dot(q, doc);
                if best.len() == k && score <= best[k - 1].0 {
                    continue;
                }
                let id = first_id + row as i64;
                let pos = best
                    .binary_search_by(|probe| {
                        // Descending by score; ascending by id on a tie so
                        // the ranking is deterministic across runs and
                        // thread counts. NOTE: the `score <= worst` fast path
                        // above skips this comparison entirely, so the id
                        // tie-break is only actually delivered because blocks
                        // arrive in ascending id order. Parallelising pass B
                        // over blocks would silently change published ground
                        // truth, which is why `add_block` rejects an
                        // out-of-order block outright.
                        score
                            .partial_cmp(&probe.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(probe.1.cmp(&id))
                    })
                    .unwrap_or_else(|e| e);
                best.insert(pos, (score, id));
                best.truncate(k);
            }
        });
        Ok(())
    }

    /// The finished ranking: per query, `(score, id)` descending.
    pub fn finish(self) -> Vec<Vec<(f32, i64)>> {
        self.best
    }
}

// ---------------------------------------------------------------------------
// Records -> on-disk rows
// ---------------------------------------------------------------------------

/// Project one `passages_jsonl` record onto the payload we store, keeping only
/// [`PAYLOAD_FIELDS`] and in that order.
///
/// A record missing `docid` is an error, not a `null` payload: `docid` is the
/// key [`verify_head_against_shipped`] matches against `top1k_passage_ids`, and
/// without it the npy↔jsonl alignment is unverifiable.
pub fn payload_from_passage(record: &serde_json::Value) -> Result<serde_json::Value, String> {
    let obj = record
        .as_object()
        .ok_or_else(|| "passage record is not a JSON object".to_string())?;
    if !obj.contains_key("docid") {
        return Err("passage record has no 'docid' field".to_string());
    }
    let mut out = serde_json::Map::with_capacity(PAYLOAD_FIELDS.len());
    for (field, _) in PAYLOAD_FIELDS {
        if let Some(v) = obj.get(*field) {
            out.insert((*field).to_string(), v.clone());
        }
    }
    Ok(serde_json::Value::Object(out))
}

/// Build one `tests.jsonl` row.
///
/// `query` and `closest_ids` / `closest_scores` are what the compound reader
/// consumes; `query_id`, `query_text` and `trec_year` are carried along so a
/// result can be traced back to the TREC topic it came from. The reader ignores
/// unknown keys, and notably there is no `conditions` key — these are pure-KNN
/// queries, and inventing a filter for them would make the ground truth a
/// fiction.
pub fn test_row(
    query: &[f32],
    query_id: &str,
    query_text: &str,
    trec_year: Option<i64>,
    ranked: &[(f32, i64)],
) -> serde_json::Value {
    serde_json::json!({
        "query": query,
        "closest_ids": ranked.iter().map(|(_, id)| *id).collect::<Vec<i64>>(),
        "closest_scores": ranked.iter().map(|(s, _)| *s).collect::<Vec<f32>>(),
        "query_id": query_id,
        "query_text": query_text,
        "trec_year": trec_year,
    })
}

// ---------------------------------------------------------------------------
// Cross-check against the shipped global top-1k
// ---------------------------------------------------------------------------

/// One query's shipped global top-1000 hits that land inside the prepared
/// prefix, in shipped rank order.
#[derive(Debug, Clone, Default)]
pub struct InPrefixHits {
    /// `(global offset == our id, passage id, cosine)`.
    pub hits: Vec<(i64, String, f32)>,
}

/// Keep the entries of one query's shipped global top-1k whose offset falls
/// inside a `limit`-passage prefix. Order is preserved, which is shipped rank
/// order (descending cosine).
pub fn in_prefix_hits(
    offsets: &[i64],
    passage_ids: &[String],
    cossims: &[f32],
    limit: u64,
) -> Result<InPrefixHits, String> {
    if offsets.len() != passage_ids.len() || offsets.len() != cossims.len() {
        return Err(format!(
            "shipped top1k arrays disagree in length: {} offsets, {} passage ids, {} cosines",
            offsets.len(),
            passage_ids.len(),
            cossims.len()
        ));
    }
    let mut hits = Vec::new();
    for i in 0..offsets.len() {
        let off = offsets[i];
        if off >= 0 && (off as u64) < limit {
            hits.push((off, passage_ids[i].clone(), cossims[i]));
        }
    }
    Ok(InPrefixHits { hits })
}

/// Depth of the shipped per-query ranking (`top1k_*`). Distinct from
/// [`NEIGHBOURS`], which is how deep OUR brute force goes: this one is fixed by
/// the upstream export and is what the coverage floor below is derived from.
pub const SHIPPED_TOP_DEPTH: u64 = 1000;

/// The minimum cross-check coverage a prepared variant must reach, as
/// `(queries, ranking positions)`.
///
/// Without a floor the verification degrades silently. `load_queries` only
/// aborts when *no* query has an in-prefix hit, so a changed upstream export, a
/// future smaller variant, or an off-by-one in [`in_prefix_hits`] that dropped
/// most offsets would still print a reassuring "Verified N queries / M
/// positions" and abort nothing — with the guard that IS the correctness
/// argument for these datasets effectively gone.
///
/// The position floor is derived, not magic. The shipped list is 1000 deep per
/// query over [`TOTAL_PASSAGES`], so a uniformly-distributed prefix of `limit`
/// passages would retain
/// `SHIPPED_TOP_DEPTH * QUERY_COUNT * limit / TOTAL_PASSAGES` of them. Real hits
/// cluster well above uniform (the 100K prefix retains 2134 against a uniform
/// 1477), so requiring a QUARTER of the uniform expectation leaves large
/// headroom on every registered size while still catching a collapse. The
/// absolute minimum of 100 stops a hypothetical tiny variant from passing on a
/// trivially small expectation, and the query floor catches the case where
/// coverage concentrates into a handful of queries.
pub fn coverage_floor(limit: u64) -> (usize, usize) {
    let uniform = SHIPPED_TOP_DEPTH
        .saturating_mul(QUERY_COUNT as u64)
        .saturating_mul(limit)
        / TOTAL_PASSAGES.max(1);
    let positions = ((uniform / 4) as usize).max(100);
    let queries = QUERY_COUNT / 20; // 5%
    (queries, positions)
}

/// Enforce [`coverage_floor`] on what the cross-check actually compared.
pub fn check_coverage(
    limit: u64,
    queries_checked: usize,
    positions_compared: usize,
) -> Result<(), String> {
    let (min_queries, min_positions) = coverage_floor(limit);
    if queries_checked < min_queries || positions_compared < min_positions {
        return Err(format!(
            "ground-truth cross-check covered only {queries_checked} queries / \
             {positions_compared} ranking positions, below the floor of {min_queries} / \
             {min_positions} for a {limit}-passage prefix. The shipped top-1k is the only \
             independent check on this ground truth, so publishing a corpus it barely \
             touched would mean publishing an unverified one. Most likely the upstream \
             top1k export changed shape, or the offsets are being mapped wrongly."
        ));
    }
    Ok(())
}

/// Which global offsets the corpus writer must remember a `docid` for.
///
/// All of them: selection depends on `docid`, so which offsets end up in the
/// sample is unknown until the corpus has been read, and any shipped offset
/// might turn out to be one. Negative offsets are dropped rather than cast into
/// enormous `u64`s.
pub fn offsets_to_track(shipped_offsets: &[i64]) -> Vec<u64> {
    shipped_offsets
        .iter()
        .filter(|off| **off >= 0)
        .map(|off| *off as u64)
        .collect()
}

/// Reject a block containing a non-finite value.
///
/// A NaN defeats the `score <= worst` fast path in [`TopK::add_block`] (every
/// comparison against NaN is false), so it always enters the insert path, where
/// `partial_cmp(..).unwrap_or(Equal)` places it at an arbitrary rank. Queries
/// that have in-prefix shipped hits would abort loudly on the mismatch, but a
/// query with none has nothing checking it and would publish a polluted head in
/// silence. One scan per block closes that off at the source.
pub fn ensure_finite(block: &[f32], first_id: i64, dim: usize) -> Result<(), String> {
    if let Some(pos) = block.iter().position(|v| !v.is_finite()) {
        let row = pos.checked_div(dim).unwrap_or(0);
        let component = pos.checked_rem(dim).unwrap_or(pos);
        return Err(format!(
            "passage {} holds a non-finite embedding value ({}) at component {} — the \
             upstream float16 data is corrupt or was decoded with the wrong byte order",
            first_id + row as i64,
            block[pos],
            component
        ));
    }
    Ok(())
}

/// Engine-enforced caps on stored string length, in UTF-8 bytes, paired with the
/// dataset schema types each applies to.
///
/// These are server-side REJECTION thresholds, so a payload above one does not
/// degrade — that engine refuses the insert. `PREPARED.json` records the
/// measured per-field maxima; [`cap_violations`] is what turns those numbers
/// from a record into a warning at preparation time, rather than a surprise the
/// next time someone points a new engine at a corpus with real prose in it.
pub const ENGINE_STRING_CAPS: &[(&str, usize, &[&str])] = &[
    // Milvus VarChar `max_length`, its own documented maximum.
    ("Milvus VarChar", 65_535, &["keyword", "text", "uuid"]),
    // Elasticsearch/OpenSearch refuse a `keyword` term above this; `text` is
    // analyzed into terms and is not subject to it.
    (
        "Elasticsearch/OpenSearch keyword term",
        32_766,
        &["keyword"],
    ),
];

/// Which measured field maxima exceed an engine's cap, as human-readable lines.
pub fn cap_violations(max_field_bytes: &HashMap<String, usize>) -> Vec<String> {
    let mut out = Vec::new();
    for (field, schema_type) in PAYLOAD_FIELDS {
        let Some(&measured) = max_field_bytes.get(*field) else {
            continue;
        };
        for (engine, cap, applies_to) in ENGINE_STRING_CAPS {
            if applies_to.contains(schema_type) && measured > *cap {
                out.push(format!(
                    "{field} ({schema_type}) reaches {measured} bytes, above the {engine} \
                     cap of {cap} — that engine will REJECT inserts for this dataset"
                ));
            }
        }
    }
    out
}

/// Outcome of the cross-check for one query.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadCheck {
    /// How many leading positions were comparable (`min(k, in-prefix hits)`).
    pub compared: usize,
    /// Largest `|our cosine - shipped cosine|` over the compared positions.
    pub max_score_delta: f32,
    /// Positions where the ids differed but both cosines agreed within
    /// tolerance, i.e. a legitimate tie reordering rather than a wrong ranking.
    pub tie_swaps: usize,
}

/// Assert our brute-forced ranking agrees with the shipped global top-1k on
/// every position the shipped list can speak to.
///
/// Every in-prefix global-top-1k hit outranks every in-prefix passage that is not
/// in the global top-1k, so our first `m` results must be exactly those `m` hits,
/// in order. A differing id is accepted only when the two cosines are within
/// `tol` of each other — a genuine tie, which float16 storage makes routine —
/// and is counted in [`HeadCheck::tie_swaps`]. Anything else is an error: it
/// means the brute force, the row alignment or the offset arithmetic is wrong.
///
/// `tol` also bounds the score comparison itself, catching the case where the
/// ids line up but we are scoring a different vector than the one HF scored.
pub fn verify_head_against_shipped(
    ranked: &[(f32, i64)],
    hits: &InPrefixHits,
    docid_at: &HashMap<i64, String>,
    tol: f32,
) -> Result<HeadCheck, String> {
    let compared = ranked.len().min(hits.hits.len());
    let mut max_score_delta = 0.0f32;
    let mut tie_swaps = 0usize;

    for (i, &(our_score, our_id)) in ranked.iter().enumerate().take(compared) {
        let (shipped_id, ref shipped_passage_id, shipped_score) = hits.hits[i];

        let delta = (our_score - shipped_score).abs();

        if our_id != shipped_id {
            if delta <= tol {
                tie_swaps += 1;
            } else {
                return Err(format!(
                    "rank {}: brute force put id {} (cosine {:.6}) where the shipped global \
                     top-1k has id {} (cosine {:.6}); the gap of {:.6} exceeds the {:.6} \
                     tolerance, so this is a wrong ranking, not a tie",
                    i, our_id, our_score, shipped_id, shipped_score, delta, tol
                ));
            }
        } else if delta > tol {
            return Err(format!(
                "rank {}: id {} scores {:.6} here but {:.6} in the shipped top-1k \
                 (delta {:.6} > {:.6}) — the vector at that offset is not the one HF scored",
                i, our_id, our_score, shipped_score, delta, tol
            ));
        }

        if delta > max_score_delta {
            max_score_delta = delta;
        }

        // The alignment check proper: the metadata we attached to this offset
        // must name the same passage the shipped list names for it.
        if let Some(docid) = docid_at.get(&shipped_id) {
            if docid != shipped_passage_id {
                return Err(format!(
                    "offset {} carries payload docid {:?} but the shipped top-1k calls it {:?} \
                     — the npy rows and the json.gz lines are misaligned, so every passage \
                     holds another passage's metadata",
                    shipped_id, docid, shipped_passage_id
                ));
            }
        }
    }

    Ok(HeadCheck {
        compared,
        max_score_delta,
        tie_swaps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_are_registered_under_their_own_names_and_dirs() {
        assert_eq!(VARIANTS.len(), 3);
        // One family only. A position-based prefix family existed and was
        // removed: selection must be by hash, or the metadata distribution is an
        // artefact of how the corpus happens to be ordered.
        assert!(VARIANTS.iter().all(|v| v.threshold > 0));
        for v in VARIANTS {
            assert_eq!(variant(v.dataset_name).map(|f| f.limit), Some(v.limit));
        }
        assert!(variant("msmarco-sparse-1M").is_none());
        // Distinct names, distinct dirs, distinct sizes — a shared dir would let
        // one size overwrite another's corpus under a different name.
        let names: std::collections::HashSet<_> = VARIANTS.iter().map(|v| v.dataset_name).collect();
        let dirs: std::collections::HashSet<_> = VARIANTS.iter().map(|v| v.dir).collect();
        let limits: std::collections::HashSet<_> = VARIANTS.iter().map(|v| v.limit).collect();
        assert_eq!(names.len(), VARIANTS.len());
        assert_eq!(dirs.len(), VARIANTS.len());
        assert_eq!(limits.len(), VARIANTS.len());
        // The crc32 thresholds must nest, so each sampled variant is a strict
        // subset of the larger ones and results stay comparable across sizes.
        let mut thresholds: Vec<u32> = VARIANTS.iter().map(|v| v.threshold).collect();
        let sorted = {
            let mut t = thresholds.clone();
            t.sort_unstable();
            t
        };
        assert_eq!(
            thresholds, sorted,
            "crc32 variants must be listed ascending"
        );
        thresholds.dedup();
        assert_eq!(
            thresholds.len(),
            VARIANTS.len(),
            "thresholds must be distinct"
        );
        // A bigger threshold must mean a bigger realized count.
        let mut sampled: Vec<(u32, u64)> =
            VARIANTS.iter().map(|v| (v.threshold, v.limit)).collect();
        sampled.sort_unstable();
        for w in sampled.windows(2) {
            assert!(w[0].1 < w[1].1, "{:?} then {:?}", w[0], w[1]);
        }
        // No variant may claim more passages than the corpus has.
        assert!(VARIANTS.iter().all(|v| v.limit <= TOTAL_PASSAGES));
    }

    /// The registry and this module are two copies of the same facts. If they
    /// drift, `prepare-msmarco` writes `datasets/A/` while the benchmark reads
    /// `datasets/B/`, or writes N vectors under a name that declares M — and the
    /// corpus-completeness gate would then reject a corpus that is actually fine
    /// (or, with a too-small declared count, accept a truncated one).
    #[test]
    fn every_variant_matches_its_datasets_json_entry() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("datasets")
            .join("datasets.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let registry: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap();

        for v in VARIANTS {
            let entry = registry
                .iter()
                .find(|e| e["name"] == serde_json::json!(v.dataset_name))
                .unwrap_or_else(|| panic!("{} is not in datasets.json", v.dataset_name));

            assert_eq!(
                entry["path"],
                serde_json::json!(v.dir),
                "{}",
                v.dataset_name
            );
            assert_eq!(
                entry["vector_count"],
                serde_json::json!(v.limit),
                "{} declares a different passage count than the preparer writes",
                v.dataset_name
            );
            assert_eq!(
                entry["vector_size"],
                serde_json::json!(DIM),
                "{}",
                v.dataset_name
            );
            assert_eq!(
                entry["distance"],
                serde_json::json!("cosine"),
                "{}",
                v.dataset_name
            );
            // Compound layout: vectors.npy + payloads.jsonl + tests.jsonl.
            assert_eq!(
                entry["type"],
                serde_json::json!("tar"),
                "{}",
                v.dataset_name
            );
            // A `link`, when present, points at the prepared tarball published to
            // S3. It must name THIS variant: an entry pointing at another size's
            // artifact would fetch a corpus of the wrong length under this name.
            // There is no upstream tarball to fall back on, so a bogus URL is
            // equally fatal — it sends the auto-downloader somewhere that cannot
            // exist.
            if let Some(link) = entry.get("link").and_then(|l| l.as_str()) {
                assert!(
                    link.ends_with(&format!("/{}.tgz", v.dataset_name)),
                    "{}: link {link:?} does not name this variant's tarball",
                    v.dataset_name
                );
                assert!(
                    link.contains("/vecsim/msmarco-cohere-1024/"),
                    "{}: link {link:?} is outside the published prefix",
                    v.dataset_name
                );
            }

            // The schema must name exactly the payload fields the preparer
            // writes, with the same types: a field in the payload but not the
            // schema is never indexed (so filters on it silently match nothing),
            // and a field in the schema but not the payload makes engines build
            // a column that is always null.
            let schema = entry["schema"]
                .as_object()
                .unwrap_or_else(|| panic!("{} has no schema object", v.dataset_name));
            assert_eq!(
                schema.len(),
                PAYLOAD_FIELDS.len(),
                "{} schema has {} fields, the payload writes {}",
                v.dataset_name,
                schema.len(),
                PAYLOAD_FIELDS.len()
            );
            for (field, ty) in PAYLOAD_FIELDS {
                assert_eq!(
                    schema.get(*field),
                    Some(&serde_json::json!(ty)),
                    "{}: schema field {field:?} should be {ty:?}",
                    v.dataset_name
                );
            }
        }
    }

    /// The sampler must be bit-identical to Python's `zlib.crc32`, which is what
    /// the Redis Enterprise MS MARCO suite selects with. Vectors taken from zlib
    /// directly — if this ever drifts, the two corpora stop being comparable and
    /// nothing else would notice.
    #[test]
    fn crc32_bucket_matches_python_zlib() {
        for (docid, full, bucket) in [
            ("hello", 907_060_870u32, 60_870u32),
            ("msmarco_v2.1_doc_00_0#0_0", 3_951_548_037, 548_037),
            (
                "msmarco_v2.1_doc_09_894803720#6_1517223658",
                1_891_268_743,
                268_743,
            ),
            ("", 0, 0),
        ] {
            assert_eq!(
                full % CRC32_BUCKETS,
                bucket,
                "test vector is self-consistent"
            );
            assert_eq!(crc32_bucket(docid), bucket, "docid {docid:?}");
        }
    }

    #[test]
    fn docid_extraction_handles_the_real_line_and_falls_back_safely() {
        // The exact shape upstream emits, space after the colon included.
        let real = r#"{"docid": "msmarco_v2.1_doc_00_0#0_0", "url": "http://x/", "title": "T"}"#;
        assert_eq!(
            extract_docid(real).as_deref(),
            Some("msmarco_v2.1_doc_00_0#0_0")
        );
        // No space.
        assert_eq!(
            extract_docid(r#"{"docid":"abc","url":"u"}"#).as_deref(),
            Some("abc")
        );
        // Field order changed: the fast path misses, the parser catches it.
        assert_eq!(
            extract_docid(r#"{"url": "u", "docid": "later"}"#).as_deref(),
            Some("later")
        );
        // Genuinely absent, and malformed.
        assert!(extract_docid(r#"{"url": "u"}"#).is_none());
        assert!(extract_docid("not json").is_none());
    }

    #[test]
    fn histogram_sizes_a_sample_for_any_threshold() {
        let mut h = BucketHistogram::new();
        // 60k synthetic docids, uniformly bucketed by construction of crc32.
        let n = 60_000u32;
        for i in 0..n {
            h.add(&format!("msmarco_v2.1_doc_{:02}_{}#0_0", i % 60, i));
        }
        assert_eq!(h.total(), n as u64);
        assert_eq!(h.count_below(0), 0);
        assert_eq!(h.count_below(CRC32_BUCKETS), n as u64);
        // Prefix sums are monotone.
        let (a, b) = (h.count_below(1000), h.count_below(2000));
        assert!(a <= b);

        // threshold_for lands on the closest realized count, and it really is
        // the closest — neither neighbour beats it.
        for target in [100u64, 1_000, 30_000] {
            let (th, got) = h.threshold_for(target);
            let err = target.abs_diff(got);
            if th > 0 {
                assert!(
                    target.abs_diff(h.count_below(th - 1)) >= err,
                    "target {target}"
                );
            }
            if th < CRC32_BUCKETS {
                assert!(
                    target.abs_diff(h.count_below(th + 1)) >= err,
                    "target {target}"
                );
            }
        }
        // Asking for everything gives everything.
        assert_eq!(h.threshold_for(n as u64).1, n as u64);
    }

    /// Merging must be exactly equivalent to one sequential scan — that is what
    /// makes the concurrent discovery scan legitimate.
    #[test]
    fn merged_shard_histograms_equal_a_single_scan() {
        let docids: Vec<String> = (0..5_000)
            .map(|i| format!("msmarco_v2.1_doc_{:02}_{}#0_0", i % 60, i))
            .collect();

        let mut sequential = BucketHistogram::new();
        for d in &docids {
            sequential.add(d);
        }

        // Same docids, split across "shards" and merged in a different order.
        let mut merged = BucketHistogram::new();
        let mut parts: Vec<BucketHistogram> = Vec::new();
        for chunk in docids.chunks(700) {
            let mut h = BucketHistogram::new();
            for d in chunk {
                h.add(d);
            }
            parts.push(h);
        }
        parts.reverse();
        for h in &parts {
            merged.merge(h);
        }

        assert_eq!(merged.total(), sequential.total());
        for t in [0u32, 1, 137, 9_999, CRC32_BUCKETS] {
            assert_eq!(
                merged.count_below(t),
                sequential.count_below(t),
                "threshold {t}"
            );
        }
    }

    /// Thresholds must nest, so a smaller variant is a strict subset of a larger
    /// one — the property that makes results across sizes comparable.
    #[test]
    fn crc32_thresholds_nest() {
        for i in 0..5_000u32 {
            let docid = format!("msmarco_v2.1_doc_00_{i}#0_0");
            if keeps(&docid, 881) {
                assert!(
                    keeps(&docid, 8_810),
                    "{docid} is in the small sample but not the large one"
                );
            }
        }
    }

    /// The bucket distribution has to be near-uniform or the threshold would not
    /// predict the sample size.
    #[test]
    fn crc32_buckets_are_uniform_enough_to_size_a_sample() {
        let n = 200_000u32;
        let hits = (0..n)
            .filter(|i| {
                let docid = format!("msmarco_v2.1_doc_{:02}_{}#0_0", i % 60, i);
                crc32_bucket(&docid) < CRC32_BUCKETS / 100 // nominal 1%
            })
            .count();
        let rate = hits as f64 / n as f64;
        assert!(
            (0.008..0.012).contains(&rate),
            "1% threshold selected {rate:.4} — not uniform enough to size a sample"
        );
    }

    #[test]
    fn shard_urls_are_zero_padded_and_share_a_stem() {
        assert_eq!(shard_stem(0), "msmarco_v2.1_doc_segmented_00");
        assert_eq!(shard_stem(59), "msmarco_v2.1_doc_segmented_59");
        assert!(npy_url(7).ends_with("passages_npy/msmarco_v2.1_doc_segmented_07.npy"));
        assert!(jsonl_url(7).ends_with("passages_jsonl/msmarco_v2.1_doc_segmented_07.json.gz"));
        assert!(queries_url().ends_with("queries_jsonl/queries.jsonl.gz"));
    }

    /// Every fetch must be pinned to one immutable commit. A `main` anywhere here
    /// means the dataset name no longer identifies a fixed set of bytes.
    #[test]
    fn every_upstream_url_is_pinned_to_a_commit_not_a_branch() {
        assert_eq!(HF_REVISION.len(), 40, "expected a full 40-char commit sha");
        assert!(HF_REVISION.chars().all(|c| c.is_ascii_hexdigit()));
        for url in [npy_url(0), npy_url(59), jsonl_url(0), queries_url()] {
            assert!(
                url.contains(&format!("/resolve/{HF_REVISION}/")),
                "not pinned: {url}"
            );
            assert!(!url.contains("/resolve/main/"), "still on a branch: {url}");
        }
    }

    /// The ordering guard has to be a real error: every suite and the preparer
    /// run `--release`, where a `debug_assert!` does not exist.
    #[test]
    fn out_of_order_blocks_are_rejected_in_release_builds_too() {
        let mut acc = TopK::new(flat(&[[1.0, 0.0]]), 2, 3).unwrap();
        acc.add_block(&flat(&[[1.0, 0.0], [0.9, 0.1]]), 10).unwrap();
        // Ascending is fine...
        acc.add_block(&flat(&[[0.8, 0.2]]), 12).unwrap();
        // ...going backwards is not.
        let e = acc.add_block(&flat(&[[0.7, 0.3]]), 5).unwrap_err();
        assert!(e.contains("ascending id order"), "{e}");
        // Run this suite under `--release` as well as debug: that is what proves
        // the guard is not compiled out in the profile the preparer uses.
    }

    /// The real shard-00 header, byte for byte.
    fn real_f16_header() -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(b"\x93NUMPY\x01\x00\x76\x00");
        let dict = "{'descr': '<f2', 'fortran_order': False, 'shape': (1760180, 1024), }";
        h.extend_from_slice(dict.as_bytes());
        while h.len() < 127 {
            h.push(b' ');
        }
        h.push(b'\n');
        h
    }

    #[test]
    fn parses_the_real_float16_shard_header() {
        let h = parse_f16_npy_header(&real_f16_header()).unwrap();
        assert_eq!(h.rows, 1_760_180);
        assert_eq!(h.cols, DIM as u64);
        assert_eq!(h.data_start, 128);
        assert_eq!(h.row_offset(0), 128);
        assert_eq!(h.row_offset(2), 128 + 2 * 1024 * 2);
    }

    #[test]
    fn rejects_dtypes_and_orders_that_would_garble_the_payload() {
        // Byte-level substitution: the leading magic byte (0x93) is not UTF-8,
        // so the header cannot be round-tripped through `String`.
        let swap = |from: &str, to: &str| {
            let h = real_f16_header();
            let (from, to) = (from.as_bytes(), to.as_bytes());
            let at = h
                .windows(from.len())
                .position(|w| w == from)
                .expect("substring present in the header");
            let mut out = h[..at].to_vec();
            out.extend_from_slice(to);
            out.extend_from_slice(&h[at + from.len()..]);
            // Re-pad to the original 128 bytes: the declared header length must
            // keep matching, or the test would be asserting on a truncation
            // error instead of on the dtype/order/shape rejection it is about.
            while out.len() < h.len() {
                out.insert(out.len() - 1, b' ');
            }
            while out.len() > h.len() {
                out.remove(out.len() - 2);
            }
            out
        };
        // float32 would halve the row count and shift every vector.
        let e = parse_f16_npy_header(&swap("'<f2'", "'<f4'")).unwrap_err();
        assert!(e.contains("float16"), "{e}");
        // Big-endian float16 is a different byte order, not a different width.
        let e = parse_f16_npy_header(&swap("'<f2'", "'>f2'")).unwrap_err();
        assert!(e.contains("float16"), "{e}");
        // Fortran order would transpose the matrix.
        let e = parse_f16_npy_header(&swap("False", "True")).unwrap_err();
        assert!(e.contains("Fortran"), "{e}");
        // A 1-D or 3-D shape has no meaningful row/col split.
        let e = parse_f16_npy_header(&swap("(1760180, 1024)", "(1760180,)")).unwrap_err();
        assert!(e.contains("2-D"), "{e}");

        assert!(parse_f16_npy_header(b"not npy at all")
            .unwrap_err()
            .contains("magic"));
        assert!(parse_f16_npy_header(b"\x93NUM")
            .unwrap_err()
            .contains("10 bytes"));
    }

    #[test]
    fn decodes_float16_and_refuses_a_block_cut_mid_value() {
        let vals = [1.0f32, -2.5, 0.0, 65504.0];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        assert_eq!(decode_f16_block(&bytes).unwrap(), vals.to_vec());

        let e = decode_f16_block(&bytes[..bytes.len() - 1]).unwrap_err();
        assert!(e.contains("odd byte length"), "{e}");
    }

    #[test]
    fn normalization_makes_unit_vectors_and_leaves_zero_alone() {
        let mut v = vec![3.0f32, 4.0];
        normalize_in_place(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);

        let mut z = vec![0.0f32, 0.0];
        normalize_in_place(&mut z);
        assert_eq!(z, vec![0.0, 0.0]);
    }

    #[test]
    fn npy_header_is_64_byte_aligned_and_declares_the_row_count() {
        for (rows, cols) in [(1u64, 1usize), (100_000, 1024), (10_000_000, 1024)] {
            let h = npy_f32_header(rows, cols);
            assert_eq!(
                h.len() % 64,
                0,
                "header not 64-byte aligned for {rows}x{cols}"
            );
            assert_eq!(&h[0..6], b"\x93NUMPY");
            assert_eq!(h[6], 1);
            let declared = u16::from_le_bytes([h[8], h[9]]) as usize;
            assert_eq!(10 + declared, h.len());
            assert_eq!(*h.last().unwrap(), b'\n');
            let dict = std::str::from_utf8(&h[10..]).unwrap();
            assert!(dict.contains(&format!("({}, {})", rows, cols)), "{dict}");
            assert!(dict.contains("'<f4'"));
        }
    }

    /// The deferred writer's whole premise: patch the header afterwards and the
    /// production reader still reads it.
    #[test]
    fn deferred_npy_round_trips_and_pins_the_header_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.npy");
        let rows: Vec<Vec<f32>> = (0..7)
            .map(|i| (0..3).map(|j| i as f32 * 2.0 + j as f32).collect())
            .collect();

        let mut w = DeferredNpyF32Writer::create(&path, 3).unwrap();
        for r in &rows {
            w.write_row(r).unwrap();
        }
        assert_eq!(w.finish().unwrap(), 7);

        let (ids, read) = crate::readers::read_npy_vectors(path.to_str().unwrap(), false).unwrap();
        assert_eq!(ids.len(), 7);
        assert_eq!(read, rows);

        // The reservation only works because every count the corpus can produce
        // renders to the same header size. Pin that across the whole range.
        let reserved = npy_f32_header(1, DIM).len();
        for n in [0u64, 1, 99_999, 100_000, 1_000_000, 10_000_000, 113_520_750] {
            assert_eq!(
                npy_f32_header(n, DIM).len(),
                reserved,
                "row count {n} renders a different header size"
            );
        }
        // A wrong-width row is still rejected.
        let mut w = DeferredNpyF32Writer::create(&dir.path().join("b.npy"), 3).unwrap();
        assert!(w.write_row(&[1.0, 2.0]).is_err());
    }

    fn flat(rows: &[[f32; 2]]) -> Vec<f32> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }

    /// The 8-lane dot must agree with the naive one, including on a length that
    /// is not a multiple of 8 (the remainder tail).
    #[test]
    fn eight_lane_dot_matches_the_naive_sum() {
        for len in [1usize, 7, 8, 9, 16, 1024] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).cos()).collect();
            let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!(
                (dot(&a, &b) - naive).abs() < 1e-4,
                "len {len}: {} vs {naive}",
                dot(&a, &b)
            );
        }
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    fn topk_ranks_by_cosine_across_block_boundaries() {
        // Two queries: +x and +y.
        let queries = flat(&[[1.0, 0.0], [0.0, 1.0]]);
        let mut acc = TopK::new(queries, 2, 2).unwrap();
        assert_eq!(acc.n_queries(), 2);

        // ids 0,1 in the first block; 2,3 in the second.
        acc.add_block(&flat(&[[0.9, 0.1], [0.1, 0.9]]), 0).unwrap();
        acc.add_block(&flat(&[[1.0, 0.0], [0.0, 1.0]]), 2).unwrap();

        let out = acc.finish();
        assert_eq!(
            out[0].iter().map(|(_, i)| *i).collect::<Vec<_>>(),
            vec![2, 0]
        );
        assert_eq!(
            out[1].iter().map(|(_, i)| *i).collect::<Vec<_>>(),
            vec![3, 1]
        );
        assert!((out[0][0].0 - 1.0).abs() < 1e-6);
    }

    /// Ties must break on the id, or two runs of the same preparation could
    /// publish different ground truth for the same dataset name.
    #[test]
    fn topk_breaks_ties_deterministically_on_the_id() {
        let queries = flat(&[[1.0, 0.0]]);
        let mut acc = TopK::new(queries, 2, 3).unwrap();
        // Four identical vectors: every score ties.
        acc.add_block(&flat(&[[1.0, 0.0], [1.0, 0.0]]), 10).unwrap();
        acc.add_block(&flat(&[[1.0, 0.0], [1.0, 0.0]]), 12).unwrap();
        let out = acc.finish();
        assert_eq!(
            out[0].iter().map(|(_, i)| *i).collect::<Vec<_>>(),
            vec![10, 11, 12]
        );
    }

    #[test]
    fn topk_rejects_ragged_query_and_block_matrices() {
        assert!(TopK::new(vec![1.0, 2.0, 3.0], 2, 5).is_err());
        assert!(TopK::new(vec![1.0], 0, 5).is_err());
        let mut acc = TopK::new(flat(&[[1.0, 0.0]]), 2, 5).unwrap();
        let e = acc.add_block(&[1.0, 2.0, 3.0], 0).unwrap_err();
        assert!(e.contains("not a multiple of dim"), "{e}");
    }

    #[test]
    fn payload_keeps_only_the_registered_fields_and_needs_a_docid() {
        let rec = serde_json::json!({
            "docid": "msmarco_v2.1_doc_00_0#0_0",
            "url": "http://example.com/",
            "title": "T",
            "headings": "H",
            "segment": "body text",
            "start_char": 0,
            "end_char": 1278,
            "extra_field_we_do_not_store": 42,
        });
        let p = payload_from_passage(&rec).unwrap();
        let obj = p.as_object().unwrap();
        assert_eq!(obj.len(), PAYLOAD_FIELDS.len());
        assert!(!obj.contains_key("extra_field_we_do_not_store"));
        assert_eq!(obj["end_char"], serde_json::json!(1278));
        // Numeric fields stay numeric — stringifying them would make the `int`
        // schema entry a lie and break range filters.
        assert!(obj["start_char"].is_i64());

        let e = payload_from_passage(&serde_json::json!({"url": "u"})).unwrap_err();
        assert!(e.contains("docid"), "{e}");
        assert!(payload_from_passage(&serde_json::json!("not an object")).is_err());
    }

    /// Every payload field must be declared in the dataset schema with the type
    /// the engines will build a column/index for.
    #[test]
    fn payload_fields_declare_a_schema_type_each() {
        for (name, ty) in PAYLOAD_FIELDS {
            assert!(
                ["keyword", "text", "int", "float", "bool", "datetime"].contains(ty),
                "{name} declares unknown schema type {ty}"
            );
        }
        assert!(PAYLOAD_FIELDS.iter().any(|(n, _)| *n == "docid"));
    }

    #[test]
    fn test_row_carries_no_conditions_key() {
        let row = test_row(
            &[0.1, 0.2],
            "787021",
            "what is produced by muscle",
            Some(2021),
            &[(0.9, 7), (0.8, 3)],
        );
        assert_eq!(row["closest_ids"], serde_json::json!([7, 3]));
        // Serializing f32 widens to the exact f64 with the same value, so the
        // round trip back to f32 is lossless.
        assert_eq!(
            row["query"],
            serde_json::json!([0.1f32 as f64, 0.2f32 as f64])
        );
        assert_eq!(row["query_id"], serde_json::json!("787021"));
        // A `conditions` key would declare a filter these queries do not have.
        assert!(row.get("conditions").is_none());
    }

    #[test]
    fn in_prefix_hits_keeps_shipped_order_and_drops_out_of_range_offsets() {
        let offsets = vec![50, 5_000_000, 7, 900];
        let ids: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let cos = vec![0.9f32, 0.8, 0.7, 0.6];

        let h = in_prefix_hits(&offsets, &ids, &cos, 1000).unwrap();
        assert_eq!(
            h.hits.iter().map(|(o, _, _)| *o).collect::<Vec<_>>(),
            vec![50, 7, 900]
        );
        assert_eq!(h.hits[0].1, "a");

        // Ragged shipped arrays are a corrupt query file, not something to
        // silently zip.
        assert!(in_prefix_hits(&offsets, &ids[..2], &cos, 1000).is_err());
    }

    fn docids(pairs: &[(i64, &str)]) -> HashMap<i64, String> {
        pairs.iter().map(|(i, s)| (*i, s.to_string())).collect()
    }

    #[test]
    fn coverage_floor_admits_the_real_variants_and_rejects_a_collapse() {
        // Measured on the real 100K build: 562 queries / 2134 positions.
        let (min_q, min_p) = coverage_floor(100_000);
        assert!(
            min_q <= 562,
            "query floor {min_q} would fail the real 100K build"
        );
        assert!(
            min_p <= 2134,
            "position floor {min_p} would fail the real 100K build"
        );
        check_coverage(100_000, 562, 2134).unwrap();

        // It must still be a real bar, not a formality: a collapse to a handful
        // of positions has to fail.
        let e = check_coverage(100_000, 562, 3).unwrap_err();
        assert!(e.contains("below the floor"), "{e}");
        // …and so does coverage concentrated into a few queries.
        assert!(check_coverage(100_000, 4, 2134).is_err());

        // The floor scales with the prefix, so a bigger variant cannot pass on
        // the small variant's bar.
        let (_, p_1m) = coverage_floor(1_000_000);
        let (_, p_10m) = coverage_floor(10_000_000);
        assert!(p_1m > min_p && p_10m > p_1m, "{min_p} {p_1m} {p_10m}");
        assert!(check_coverage(10_000_000, 1677, 2134).is_err());

        // Never below the absolute minimum, however tiny the prefix.
        assert_eq!(coverage_floor(1).1, 100);
        assert_eq!(coverage_floor(0).1, 100);
    }

    #[test]
    fn non_finite_embeddings_are_rejected_with_the_offending_passage() {
        let mut block = vec![0.5f32; 8];
        assert!(ensure_finite(&block, 100, 4).is_ok());

        block[5] = f32::NAN;
        let e = ensure_finite(&block, 100, 4).unwrap_err();
        // Row 1 of the block => id 101, component 1.
        assert!(e.contains("passage 101"), "{e}");
        assert!(e.contains("component 1"), "{e}");

        block[5] = f32::INFINITY;
        assert!(ensure_finite(&block, 100, 4).is_err());
    }

    /// A NaN would otherwise sail past the `score <= worst` fast path (every
    /// comparison with NaN is false) and land at an arbitrary rank.
    #[test]
    fn a_nan_would_corrupt_the_ranking_if_it_reached_topk() {
        let queries = flat(&[[1.0, 0.0]]);
        let mut acc = TopK::new(queries, 2, 2).unwrap();
        acc.add_block(&flat(&[[1.0, 0.0], [f32::NAN, 0.0]]), 0)
            .unwrap();
        let out = acc.finish();
        // The NaN row is present in the published head — which is exactly why
        // `ensure_finite` runs before any block reaches here.
        assert!(out[0].iter().any(|(s, _)| s.is_nan()));
    }

    #[test]
    fn cap_violations_flag_only_the_engines_whose_limit_applies() {
        let m = |pairs: &[(&str, usize)]| -> HashMap<String, usize> {
            pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
        };

        // The real 100K maxima: everything fits, so nothing is flagged.
        let ok = m(&[
            ("docid", 41),
            ("url", 190),
            ("title", 588),
            ("headings", 25_840),
            ("segment", 28_581),
        ]);
        assert!(cap_violations(&ok).is_empty(), "{:?}", cap_violations(&ok));

        // `segment` is `text`: past Milvus' cap it is flagged, but the
        // Elasticsearch `keyword` term limit does not apply to it.
        let big_text = m(&[("segment", 70_000)]);
        let v = cap_violations(&big_text);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(v[0].contains("Milvus"), "{}", v[0]);

        // `url` is `keyword`: a 40 KB one breaks Elasticsearch but not Milvus.
        let big_kw = m(&[("url", 40_000)]);
        let v = cap_violations(&big_kw);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(v[0].contains("keyword term"), "{}", v[0]);

        // An unmeasured field is skipped rather than assumed zero.
        assert!(cap_violations(&HashMap::new()).is_empty());
    }

    #[test]
    fn head_check_passes_when_the_head_matches_the_shipped_hits() {
        let ranked = vec![(0.9f32, 50), (0.7, 7), (0.6, 900), (0.55, 42)];
        let hits = InPrefixHits {
            hits: vec![
                (50, "a".into(), 0.9),
                (7, "c".into(), 0.7),
                (900, "d".into(), 0.6),
            ],
        };
        let map = docids(&[(50, "a"), (7, "c"), (900, "d")]);
        let c = verify_head_against_shipped(&ranked, &hits, &map, 1e-3).unwrap();
        // Only the three shipped hits are comparable; id 42 is beyond them.
        assert_eq!(c.compared, 3);
        assert_eq!(c.tie_swaps, 0);
        assert!(c.max_score_delta < 1e-6);
    }

    #[test]
    fn head_check_rejects_a_genuinely_wrong_ranking() {
        // Our #1 is a passage the shipped global top-1k ranks below another
        // in-prefix passage — impossible unless the brute force is wrong.
        let ranked = vec![(0.7f32, 7), (0.9, 50)];
        let hits = InPrefixHits {
            hits: vec![(50, "a".into(), 0.9), (7, "c".into(), 0.7)],
        };
        let map = docids(&[(50, "a"), (7, "c")]);
        let e = verify_head_against_shipped(&ranked, &hits, &map, 1e-3).unwrap_err();
        assert!(e.contains("wrong ranking"), "{e}");
    }

    /// float16 storage makes exact ties common; a swap between two positions
    /// whose cosines agree is not an error, but it is counted.
    #[test]
    fn head_check_tolerates_only_real_ties() {
        let hits = InPrefixHits {
            hits: vec![(50, "a".into(), 0.90000), (7, "c".into(), 0.90001)],
        };
        let map = docids(&[(50, "a"), (7, "c")]);

        let tied = vec![(0.90001f32, 7), (0.90000, 50)];
        let c = verify_head_against_shipped(&tied, &hits, &map, 1e-3).unwrap();
        assert_eq!(c.tie_swaps, 2);

        // Same swap, but the cosines are far apart: not a tie.
        let hits_apart = InPrefixHits {
            hits: vec![(50, "a".into(), 0.9), (7, "c".into(), 0.2)],
        };
        let e = verify_head_against_shipped(&tied, &hits_apart, &map, 1e-3).unwrap_err();
        assert!(e.contains("wrong ranking"), "{e}");
    }

    /// The misalignment guard: same ids, same scores, but the payload at that
    /// offset names a different passage. Nothing downstream would catch this.
    #[test]
    fn head_check_catches_npy_jsonl_misalignment() {
        let ranked = vec![(0.9f32, 50)];
        let hits = InPrefixHits {
            hits: vec![(50, "expected-docid".into(), 0.9)],
        };
        let shifted = docids(&[(50, "the-next-passages-docid")]);
        let e = verify_head_against_shipped(&ranked, &hits, &shifted, 1e-3).unwrap_err();
        assert!(e.contains("misaligned"), "{e}");
    }

    /// Ids and ranks agree, but we are scoring a different vector than HF did —
    /// e.g. a float16 decode that read the wrong byte order.
    #[test]
    fn head_check_catches_a_scored_vector_that_is_not_the_shipped_one() {
        let ranked = vec![(0.42f32, 50)];
        let hits = InPrefixHits {
            hits: vec![(50, "a".into(), 0.9)],
        };
        let e =
            verify_head_against_shipped(&ranked, &hits, &docids(&[(50, "a")]), 1e-3).unwrap_err();
        assert!(e.contains("not the one HF scored"), "{e}");
    }

    /// A query whose global top-1000 misses the prefix entirely gives nothing to
    /// check — that must be reported as zero comparisons, never as a pass.
    #[test]
    fn head_check_reports_zero_comparisons_when_no_shipped_hit_is_in_prefix() {
        let ranked = vec![(0.5f32, 1), (0.4, 2)];
        let c =
            verify_head_against_shipped(&ranked, &InPrefixHits::default(), &HashMap::new(), 1e-3)
                .unwrap();
        assert_eq!(c.compared, 0);
    }
}
