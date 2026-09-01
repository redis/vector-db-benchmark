//! `generate-dataset` — emit small, deterministic synthetic datasets on disk in
//! the exact layouts the benchmark's sparse / hybrid / compound(tar) readers
//! expect, so the sparse, hybrid and filter-datatype code paths can be run
//! end-to-end from a registered dataset name (not just from the integration
//! tests).
//!
//! The datasets are registered in `datasets/datasets.json` with LOCAL paths and
//! NO download link — they do not exist until this binary writes them. Run:
//!
//! ```text
//! cargo run --release --bin generate-dataset          # writes into ./datasets
//! cargo run --release --bin generate-dataset -- --out-dir /tmp/ds --only sparse
//! ```
//!
//! What it writes (all sizes are tiny so generation + a smoke run are fast):
//!   * `synthetic-sparse-300/`  — CSR `data.csr` + `queries.csr` +
//!     `neighbours.jsonl` (sparse, dot/MIPS).
//!   * `synthetic-hybrid-16/`   — dense `vectors.npy`/`queries.npy` + sparse
//!     `data.csr`/`queries.csr` + shared `neighbours.jsonl` (hybrid, RRF fusion).
//!   * `synthetic-filter-32/`   — compound `vectors.npy` + `payloads.jsonl`
//!     (keyword/int/bool/datetime fields) + `tests.jsonl` (per-query `conditions`
//!     filter + filtered ground truth). `type: "tar"`.
//!   * `synthetic-selectivity-32/` — compound `vectors.npy` + `payloads.jsonl`
//!     (int `rank` field) + `tests.jsonl` sweeping a `rank < K` range filter
//!     across a selectivity ladder (1%..90%), each row annotated with its
//!     `selectivity`. `type: "tar"`.
//!   * `synthetic-multivector-16/` — `data.mvec`/`queries.mvec` (ColBERT-style
//!     ragged token-vector rows) + `neighbours.jsonl` (multivector,
//!     brute-forced MaxSim ground truth). `type: "multivector"`.

use std::path::{Path, PathBuf};

use chrono::{Duration, TimeZone, Utc};
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use vector_db_benchmark::readers::{
    write_multivector_matrix, write_npy_vectors, write_sparse_matrix,
};
use vector_db_benchmark::synthetic::{
    generate_hybrid, generate_multivector, generate_sparse, write_neighbours_jsonl,
    MultiVectorGenData, SparseData, SYNTHETIC_MULTIVECTOR_DIM, SYNTHETIC_MULTIVECTOR_ROWS,
    SYNTHETIC_SPARSE_DIM, SYNTHETIC_SPARSE_ROWS,
};

/// Dataset names as registered in `datasets/datasets.json`.
const SPARSE_NAME: &str = "synthetic-sparse-300";
const HYBRID_NAME: &str = "synthetic-hybrid-16";
const FILTER_NAME: &str = "synthetic-filter-32";
const SELECTIVITY_NAME: &str = "synthetic-selectivity-32";
const MULTIVECTOR_NAME: &str = "synthetic-multivector-16";

#[derive(Parser, Debug)]
#[command(
    name = "generate-dataset",
    about = "Generate small deterministic synthetic datasets (sparse / hybrid / filter / selectivity) on disk."
)]
struct Args {
    /// Base directory to write datasets into (each dataset gets its own subdir).
    #[arg(long, default_value = "datasets")]
    out_dir: PathBuf,

    /// Generate only one dataset kind. Omit to generate all five.
    #[arg(long, value_parser = ["sparse", "hybrid", "filter", "selectivity", "multivector"])]
    only: Option<String>,
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args) {
        eprintln!("generate-dataset: error: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), String> {
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|e| format!("create {}: {}", args.out_dir.display(), e))?;

    let want = |kind: &str| args.only.as_deref().map(|o| o == kind).unwrap_or(true);

    if want("sparse") {
        gen_sparse(&args.out_dir)?;
    }
    if want("hybrid") {
        gen_hybrid(&args.out_dir)?;
    }
    if want("filter") {
        gen_filter(&args.out_dir)?;
    }
    if want("selectivity") {
        gen_selectivity(&args.out_dir)?;
    }
    if want("multivector") {
        gen_multivector(&args.out_dir)?;
    }
    println!("\nDone. Register/select these datasets by the names printed above.");
    Ok(())
}

/// Report a written file with its on-disk size.
fn note(path: &Path) -> Result<(), String> {
    let len = std::fs::metadata(path)
        .map_err(|e| format!("stat {}: {}", path.display(), e))?
        .len();
    println!("    {:<20} {:>8} bytes", file_name(path), len);
    Ok(())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// ── sparse ──────────────────────────────────────────────────────────────────

fn gen_sparse(base: &Path) -> Result<(), String> {
    // nnz=10, 10 queries, top-10 GT. Fixed seed → reproducible. The corpus size
    // and dimensionality come from shared constants so the `datasets.json` guard
    // can check the declared vector_count against the number actually written.
    let SparseData {
        data,
        queries,
        neighbours,
    } = generate_sparse(
        0x5A5A_5EED,
        SYNTHETIC_SPARSE_DIM,
        10,
        SYNTHETIC_SPARSE_ROWS,
        10,
        10,
    );

    let dir = base.join(SPARSE_NAME);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    println!(
        "[sparse]  {SPARSE_NAME}/  ({} docs, {} queries)",
        data.len(),
        queries.len()
    );

    let data_csr = dir.join("data.csr");
    let queries_csr = dir.join("queries.csr");
    let nb = dir.join("neighbours.jsonl");
    write_sparse_matrix(data_csr.to_str().ok_or("bad path")?, &data)?;
    write_sparse_matrix(queries_csr.to_str().ok_or("bad path")?, &queries)?;
    write_neighbours_jsonl(&nb, &neighbours)?;
    note(&data_csr)?;
    note(&queries_csr)?;
    note(&nb)?;
    Ok(())
}

// ── hybrid ──────────────────────────────────────────────────────────────────

fn gen_hybrid(base: &Path) -> Result<(), String> {
    let h = generate_hybrid(0xB19_1DEA);

    let dir = base.join(HYBRID_NAME);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    println!(
        "[hybrid]  {HYBRID_NAME}/  ({} docs, {} queries, dim {}, top {})",
        h.dense.len(),
        h.dense_queries.len(),
        h.dim,
        h.top
    );

    let vectors_npy = dir.join("vectors.npy");
    let queries_npy = dir.join("queries.npy");
    let data_csr = dir.join("data.csr");
    let queries_csr = dir.join("queries.csr");
    let nb = dir.join("neighbours.jsonl");
    write_npy_vectors(vectors_npy.to_str().ok_or("bad path")?, &h.dense)?;
    write_npy_vectors(queries_npy.to_str().ok_or("bad path")?, &h.dense_queries)?;
    write_sparse_matrix(data_csr.to_str().ok_or("bad path")?, &h.sparse)?;
    write_sparse_matrix(queries_csr.to_str().ok_or("bad path")?, &h.sparse_queries)?;
    write_neighbours_jsonl(&nb, &h.neighbours)?;
    for p in [&vectors_npy, &queries_npy, &data_csr, &queries_csr, &nb] {
        note(p)?;
    }
    Ok(())
}

// ── multivector (ColBERT-style / MaxSim) ────────────────────────────────────

fn gen_multivector(base: &Path) -> Result<(), String> {
    // 4-8 tokens/doc, 10 queries, top-10 GT, real brute-force MaxSim (see
    // `generate_multivector`'s doc comment for why this is NOT planted like
    // `generate_hybrid`). Corpus size and dim come from shared constants so a
    // future `datasets.json` guard can check the declared vector_count.
    let MultiVectorGenData {
        data,
        queries,
        neighbours,
    } = generate_multivector(
        0xC0FFEE_u64,
        SYNTHETIC_MULTIVECTOR_DIM,
        4,
        8,
        SYNTHETIC_MULTIVECTOR_ROWS,
        10,
        10,
    );

    let dir = base.join(MULTIVECTOR_NAME);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    println!(
        "[mvector] {MULTIVECTOR_NAME}/  ({} docs, {} queries, dim {})",
        data.len(),
        queries.len(),
        SYNTHETIC_MULTIVECTOR_DIM
    );

    let data_mvec = dir.join("data.mvec");
    let queries_mvec = dir.join("queries.mvec");
    let nb = dir.join("neighbours.jsonl");
    write_multivector_matrix(data_mvec.to_str().ok_or("bad path")?, &data)?;
    write_multivector_matrix(queries_mvec.to_str().ok_or("bad path")?, &queries)?;
    write_neighbours_jsonl(&nb, &neighbours)?;
    note(&data_mvec)?;
    note(&queries_mvec)?;
    note(&nb)?;
    Ok(())
}

// ── filter (compound / tar) ─────────────────────────────────────────────────

/// Keyword values assigned round-robin by `id % 4`.
const COLORS: [&str; 4] = ["red", "green", "blue", "dark blue"];

fn color_for(id: usize) -> &'static str {
    COLORS[id % COLORS.len()]
}
fn size_for(id: usize) -> i64 {
    (id % 5) as i64 + 1
}
fn flag_for(id: usize) -> bool {
    id.is_multiple_of(2)
}

/// A per-query filter: the `conditions` JSON attached to the query and the
/// predicate deciding which document ids satisfy it (used to brute-force the
/// filtered ground truth). Rotating these across queries exercises the keyword,
/// int, bool and datetime filter datatypes in a single dataset.
struct QueryFilter {
    conditions: serde_json::Value,
    matches: Box<dyn Fn(usize) -> bool>,
}

fn gen_filter(base: &Path) -> Result<(), String> {
    const N: usize = 400;
    const N_QUERIES: usize = 12;
    const DIM: usize = 32;
    const TOP: usize = 10;

    // Fixed seed → reproducible vectors, queries and ground truth.
    let mut rng = StdRng::seed_from_u64(0xF117E_u64);
    let gen_vec =
        |rng: &mut StdRng| -> Vec<f32> { (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..N).map(|_| gen_vec(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERIES).map(|_| gen_vec(&mut rng)).collect();

    // datetime field: one ISO-8601 timestamp per doc, one day apart from base.
    let base_ts = Utc.timestamp_opt(1_609_459_200, 0).unwrap(); // 2021-01-01T00:00:00Z
    let iso_for = move |day: i64| (base_ts + Duration::days(day)).to_rfc3339();
    let ts_gte = iso_for(100);
    let ts_lt = iso_for(300);

    // Per-query filter, rotating through the four datatypes.
    let filter_for = |q: usize| -> QueryFilter {
        match q % 4 {
            0 => QueryFilter {
                // keyword match_any: color IN {red, blue}
                conditions: serde_json::json!({
                    "and": [ { "color": { "match": { "any": ["red", "blue"] } } } ]
                }),
                matches: Box::new(|id| ["red", "blue"].contains(&color_for(id))),
            },
            1 => QueryFilter {
                // int match_any: size IN {1, 2, 3}
                conditions: serde_json::json!({
                    "and": [ { "size": { "match": { "any": [1, 2, 3] } } } ]
                }),
                matches: Box::new(|id| [1, 2, 3].contains(&size_for(id))),
            },
            2 => QueryFilter {
                // bool: flag == true
                conditions: serde_json::json!({
                    "and": [ { "flag": { "match": { "value": true } } } ]
                }),
                matches: Box::new(flag_for),
            },
            _ => {
                // datetime range: ts IN [day 100, day 300)
                let gte = ts_gte.clone();
                let lt = ts_lt.clone();
                QueryFilter {
                    conditions: serde_json::json!({
                        "and": [ { "ts": { "range": { "gte": gte, "lt": lt } } } ]
                    }),
                    matches: Box::new(|id| (100..300).contains(&id)),
                }
            }
        }
    };

    // Ground truth: top-TOP nearest by L2 over ONLY the docs matching the filter.
    let filtered_gt = |q_vec: &[f32], matches: &dyn Fn(usize) -> bool| -> Vec<i64> {
        let l2 = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                .sum()
        };
        let mut scored: Vec<(i64, f64)> = (0..N)
            .filter(|id| matches(*id))
            .map(|id| (id as i64, l2(q_vec, &vectors[id])))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.iter().take(TOP).map(|(id, _)| *id).collect()
    };

    let dir = base.join(FILTER_NAME);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    println!(
        "[filter]  {FILTER_NAME}/  ({N} docs, {N_QUERIES} queries, dim {DIM}, \
         fields color/size/flag/ts)"
    );

    // vectors.npy (implicit ids 0..N).
    let vectors_npy = dir.join("vectors.npy");
    write_npy_vectors(vectors_npy.to_str().ok_or("bad path")?, &vectors)?;

    // payloads.jsonl: keyword/int/bool/datetime per doc.
    let payloads: String = (0..N)
        .map(|id| {
            serde_json::json!({
                "color": color_for(id),
                "size": size_for(id),
                "flag": flag_for(id),
                "ts": iso_for(id as i64),
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let payloads_path = dir.join("payloads.jsonl");
    std::fs::write(&payloads_path, payloads).map_err(|e| e.to_string())?;

    // tests.jsonl: query + conditions + filtered ground truth.
    let tests: String = queries
        .iter()
        .enumerate()
        .map(|(q, q_vec)| {
            let f = filter_for(q);
            serde_json::json!({
                "query": q_vec.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                "conditions": f.conditions,
                "closest_ids": filtered_gt(q_vec, &f.matches),
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let tests_path = dir.join("tests.jsonl");
    std::fs::write(&tests_path, tests).map_err(|e| e.to_string())?;

    note(&vectors_npy)?;
    note(&payloads_path)?;
    note(&tests_path)?;
    Ok(())
}

// ── selectivity ladder ───────────────────────────────────────────────────────

/// Target filter selectivities (fraction of the corpus that matches) for the
/// selectivity-ladder dataset, from highly restrictive (1%) to barely filtered
/// (90%). The #1 methodology idea shared by VectorDBBench, Pinecone VSB and
/// qdrant's vector-db-benchmark: filter recall/latency must be reported as a
/// FUNCTION of selectivity, because a restrictive filter and a permissive one
/// exercise very different engine paths — naive post-filtering HNSW can COLLAPSE
/// at low selectivity, while a pre-filtering engine stays correct.
const SELECTIVITY_LADDER: [f64; 7] = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.90];

/// Emit `synthetic-selectivity-32`: an int `rank` = doc id on every doc and one
/// `rank < K` range query per rung of [`SELECTIVITY_LADDER`], so a SINGLE dataset
/// sweeps the same numeric-range filter across selectivities. Each test row also
/// carries `selectivity` and `n_matching` (extra keys the reader ignores) so a
/// downstream analysis can bucket recall/latency by selectivity. The corpus is
/// larger than the other synthetic sets (N=2000) so that, run against a real
/// HNSW index with `ef << N`, the restrictive rungs actually stress the
/// post-filter regime rather than degenerating to exhaustive search.
fn gen_selectivity(base: &Path) -> Result<(), String> {
    const N: usize = 2000;
    const DIM: usize = 32;
    const TOP: usize = 10;

    // Fixed seed → reproducible vectors, queries and ground truth.
    let mut rng = StdRng::seed_from_u64(0x5E1EC_u64);
    let gen_vec =
        |rng: &mut StdRng| -> Vec<f32> { (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..N).map(|_| gen_vec(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..SELECTIVITY_LADDER.len())
        .map(|_| gen_vec(&mut rng))
        .collect();

    // Number of matching docs for rung `q`: the K lowest ranks (ids 0..K).
    let k_for = |q: usize| -> usize {
        let k = (SELECTIVITY_LADDER[q] * N as f64).round() as usize;
        k.clamp(TOP, N) // keep >= TOP so recall is well-defined
    };

    // Ground truth: top-TOP nearest by L2 over ONLY the docs matching rung `q`
    // (rank < K, i.e. ids 0..K).
    let filtered_gt = |q_vec: &[f32], k: usize| -> Vec<i64> {
        let l2 = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                .sum()
        };
        let mut scored: Vec<(i64, f64)> = (0..k)
            .map(|id| (id as i64, l2(q_vec, &vectors[id])))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.iter().take(TOP).map(|(id, _)| *id).collect()
    };

    let dir = base.join(SELECTIVITY_NAME);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    println!(
        "[select]  {SELECTIVITY_NAME}/  ({N} docs, {} queries, dim {DIM}, \
         selectivity ladder {SELECTIVITY_LADDER:?})",
        SELECTIVITY_LADDER.len()
    );

    // vectors.npy (implicit ids 0..N).
    let vectors_npy = dir.join("vectors.npy");
    write_npy_vectors(vectors_npy.to_str().ok_or("bad path")?, &vectors)?;

    // payloads.jsonl: one int `rank` per doc.
    let payloads: String = (0..N)
        .map(|id| serde_json::json!({ "rank": id as i64 }).to_string())
        .collect::<Vec<_>>()
        .join("\n");
    let payloads_path = dir.join("payloads.jsonl");
    std::fs::write(&payloads_path, payloads).map_err(|e| e.to_string())?;

    // tests.jsonl: query + range condition + filtered ground truth, annotated
    // with the rung's target selectivity and actual match count.
    let tests: String = queries
        .iter()
        .enumerate()
        .map(|(q, q_vec)| {
            let k = k_for(q);
            serde_json::json!({
                "query": q_vec.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                "conditions": { "and": [ { "rank": { "range": { "lt": k as i64 } } } ] },
                "closest_ids": filtered_gt(q_vec, k),
                "selectivity": SELECTIVITY_LADDER[q],
                "n_matching": k,
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let tests_path = dir.join("tests.jsonl");
    std::fs::write(&tests_path, tests).map_err(|e| e.to_string())?;

    note(&vectors_npy)?;
    note(&payloads_path)?;
    note(&tests_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_lines(path: &Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    // The selectivity dataset must (1) emit exactly one query per ladder rung,
    // (2) size each rung's match count to its target selectivity (clamped to
    // >= TOP), (3) tie the `rank < K` filter bound to that match count, and
    // (4) draw the whole filtered ground truth from inside the surviving set.
    #[test]
    fn selectivity_dataset_matches_the_ladder() {
        const N: usize = 2000;
        const TOP: usize = 10;
        let tmp = tempfile::tempdir().unwrap();
        gen_selectivity(tmp.path()).unwrap();
        let dir = tmp.path().join(SELECTIVITY_NAME);

        // payloads: one `rank` per doc, rank == id.
        let payloads = read_lines(&dir.join("payloads.jsonl"));
        assert_eq!(payloads.len(), N, "payload row per doc");
        for (id, row) in payloads.iter().enumerate() {
            assert_eq!(row["rank"].as_i64().unwrap(), id as i64, "rank == id");
        }

        // tests: one row per ladder rung, fully consistent with its selectivity.
        let rows = read_lines(&dir.join("tests.jsonl"));
        assert_eq!(rows.len(), SELECTIVITY_LADDER.len(), "one query per rung");
        for (q, row) in rows.iter().enumerate() {
            let sel = SELECTIVITY_LADDER[q];
            let expected_k = ((sel * N as f64).round() as usize).clamp(TOP, N);

            assert_eq!(
                row["selectivity"].as_f64().unwrap(),
                sel,
                "rung {q} selectivity annotation"
            );
            let n_matching = row["n_matching"].as_u64().unwrap() as usize;
            assert_eq!(
                n_matching, expected_k,
                "rung {q} match count == round(sel*N)"
            );

            // filter bound `rank < K` must equal the match count.
            let lt = row["conditions"]["and"][0]["rank"]["range"]["lt"]
                .as_i64()
                .unwrap();
            assert_eq!(lt as usize, expected_k, "rung {q} filter bound ties to K");

            // ground truth: TOP ids, all inside the surviving set (id < K), distinct.
            let ids: Vec<i64> = row["closest_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            assert_eq!(ids.len(), TOP, "rung {q} returns TOP ground-truth ids");
            for &id in &ids {
                assert!(
                    (id as usize) < expected_k,
                    "rung {q} GT id {id} must be inside the filtered set (< {expected_k})"
                );
            }
            let mut uniq = ids.clone();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(uniq.len(), ids.len(), "rung {q} GT ids must be distinct");
        }

        // The tightest rung must still yield >= TOP matches (recall well-defined).
        assert!(
            rows[0]["n_matching"].as_u64().unwrap() as usize >= TOP,
            "tightest rung keeps >= TOP matches"
        );
    }

    // Fixed seeds → byte-identical output across runs (reproducible ground truth).
    #[test]
    fn selectivity_dataset_is_deterministic() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        gen_selectivity(a.path()).unwrap();
        gen_selectivity(b.path()).unwrap();
        for f in ["payloads.jsonl", "tests.jsonl", "vectors.npy"] {
            let ra = std::fs::read(a.path().join(SELECTIVITY_NAME).join(f)).unwrap();
            let rb = std::fs::read(b.path().join(SELECTIVITY_NAME).join(f)).unwrap();
            assert_eq!(ra, rb, "{f} must be identical across runs");
        }
    }
}
