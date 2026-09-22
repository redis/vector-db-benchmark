//! `prepare-msmarco` — materialise the MS MARCO v2.1 / Cohere `embed-english-v3`
//! corpus (vectors **and** metadata) on disk in the compound layout the
//! benchmark's `type: "tar"` reader expects.
//!
//! The upstream corpus is a 60-shard Hugging Face dataset of 113.5M passages —
//! there is no tarball to point `datasets.json`'s `link` at, and no sane default
//! size, so the registered entries have no download link and this binary builds
//! them. Run:
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

use vector_db_benchmark::msmarco::{
    self, in_prefix_hits, normalize_in_place, parse_f16_npy_header, payload_from_passage, test_row,
    verify_head_against_shipped, NpyF32Writer, ShardTake, TopK, Variant,
};

/// Passages converted / scored per block. 8192 x 1024 x 4 B = 32 MiB, small
/// enough to keep preparation's peak RSS flat at any corpus size.
const BLOCK_ROWS: usize = 8192;

/// Tolerance when matching our cosines against HF's. The embeddings are stored
/// as float16 (~3 decimal digits), and HF computed their cosines in float32, so
/// a few 1e-4 of disagreement is expected; 1e-2 would hide a real error.
const COSINE_TOLERANCE: f32 = 2e-3;

/// HTTP read timeout. Generous: these are multi-GB streams over the public CDN.
const HTTP_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Parser, Debug)]
#[command(
    name = "prepare-msmarco",
    about = "Download and prepare an MS MARCO v2.1 + Cohere embed-v3 dataset (vectors + metadata)."
)]
struct Args {
    /// Registered dataset name to build, e.g. `msmarco-cohere-1024-1M-cosine`.
    #[arg(long)]
    dataset: String,

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
    let variant = msmarco::variant(&args.dataset).ok_or_else(|| {
        format!(
            "unknown dataset {:?}. Registered MS MARCO variants: {}",
            args.dataset,
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

    let queries = load_queries(&client, &cache, args.hf_token.as_deref(), variant)?;
    let plan = plan_shards(&client, args.hf_token.as_deref(), variant)?;

    // Only the offsets some query's shipped top-1k actually names need their
    // docid remembered for the alignment check — a few thousand strings, not a
    // parallel copy of the whole corpus.
    let wanted_docids: std::collections::HashSet<i64> = queries
        .iter()
        .flat_map(|q| q.hits.hits.iter().map(|(off, _, _)| *off))
        .collect();

    let (docid_at, max_field_bytes) = write_corpus(
        &client,
        args.hf_token.as_deref(),
        variant,
        &plan,
        &dir,
        &wanted_docids,
    )?;

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
    for name in OUTPUTS.iter().rev() {
        let from = staged(dir, name);
        if !from.exists() {
            return Err(format!("{} was never written", from.display()));
        }
        std::fs::rename(&from, dir.join(name))
            .map_err(|e| format!("publish {}: {}", from.display(), e))?;
    }
    Ok(())
}

/// Remove any `.part` files left by an earlier interrupted run.
fn clear_staged(dir: &Path) {
    for name in OUTPUTS {
        let _ = std::fs::remove_file(staged(dir, name));
    }
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

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

struct Query {
    id: String,
    text: String,
    trec_year: Option<i64>,
    /// Normalized, `DIM` long.
    emb: Vec<f32>,
    /// The shipped global top-1k entries that land inside this variant's prefix.
    hits: msmarco::InPrefixHits,
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
    println!(
        "Loaded {} queries. {} of them have at least one of their global top-1000 inside this \
         {}-passage prefix ({} hits total) — those are what the ground truth is cross-checked \
         against.",
        queries.len(),
        with_hits,
        variant.limit,
        total_hits
    );
    if with_hits == 0 {
        return Err(
            "no query has a single shipped top-1k hit inside this prefix, so the brute-forced \
             ground truth could not be verified against anything"
                .to_string(),
        );
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
        "prefix_of_global_order": true,
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
