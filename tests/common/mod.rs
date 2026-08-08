//! Shared helpers for `match_any` filter integration tests.
//!
//! Builds a small compound-format dataset (`vectors.npy` + `payloads.jsonl` +
//! `tests.jsonl`) whose queries each carry a `match_any` condition on a keyword
//! field, drives the real benchmark binary against a running engine, and reads
//! back the reported recall.
//!
//! Ground truth is brute-forced over ONLY the documents that satisfy the
//! filter, so a high recall proves the engine actually applied `match_any`
//! (it returned the OR-set's nearest neighbours, not the whole corpus's). An
//! engine that ignores the filter, or matches the wrong set, scores low recall.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vector_db_benchmark::readers::{write_gt_neighbours, write_npy_vectors, write_sparse_matrix};
use vector_db_benchmark::synthetic::{generate_hybrid, generate_sparse, HybridData, SparseData};

/// Keyword values assigned round-robin to documents by `id % 4`.
///
/// The last value is intentionally MULTI-WORD ("dark blue"): keyword matching
/// must be whole-value/exact, so `match_any ["red","blue"]` must NOT select a
/// "dark blue" doc. This makes the recall test sensitive to an engine that
/// tokenizes keyword fields (e.g. a regression to Weaviate's default `word`
/// tokenization, under which `Equal "blue"` would wrongly match "dark blue").
const COLORS: [&str; 4] = ["red", "green", "blue", "dark blue"];
/// The `match_any` set every query filters on (COLORS indices 0 and 2).
pub const MATCH_ANY_COLORS: [&str; 2] = ["red", "blue"];

/// The `match_any` set every labels-query filters on, for the MULTI-VALUED
/// keyword field `labels` (issue #88). Each doc carries a 2-element array; the
/// filter must match a doc that shares ANY element with this set — impossible if
/// the engine stored the array as one joined scalar, so recall discriminates the
/// fix from the bug. See [`labels_for`] for the ANY-vs-ALL discrimination.
pub const MATCH_ANY_LABELS: [&str; 2] = ["red", "blue"];

/// Each MATCHING doc carries exactly ONE query label (`red` XOR `blue`) plus a
/// non-query tag; non-matching docs carry neither. With the query set
/// {red, blue}, `id%4 ∈ {0,2}` match; `{1,3}` do not. Because no matching doc
/// holds BOTH query labels, the fixture distinguishes three behaviors:
/// contains-ANY (correct → 200 docs), contains-ALL (→ 0 docs, recall 0), and
/// the joined-scalar bug (whole-string `"red;green" == "red"` → 0, recall 0).
fn labels_for(id: usize) -> Vec<&'static str> {
    match id % 4 {
        0 => vec!["red", "green"],    // matches via `red` only
        2 => vec!["blue", "yellow"],  // matches via `blue` only
        _ => vec!["green", "yellow"], // no query label
    }
}

fn matches_labels_filter(id: usize) -> bool {
    let l = labels_for(id);
    MATCH_ANY_LABELS.iter().any(|q| l.contains(q))
}

const N_DOCS: usize = 400;
const N_QUERIES: usize = 10;
const TOP: usize = 10;

pub struct MatchAnyProject {
    /// Temp project root (leaked; lives for the process). Passed as cwd.
    pub root: PathBuf,
    pub dataset_name: String,
    pub top: usize,
    /// Number of documents satisfying the filter (sanity bound: >> top).
    pub matching_docs: usize,
}

fn color_for(id: usize) -> &'static str {
    COLORS[id % COLORS.len()]
}

fn matches_filter(id: usize) -> bool {
    MATCH_ANY_COLORS.contains(&color_for(id))
}

/// Ground-truth distance metric for the brute-forced neighbours. Engines that
/// rank by L2 (Redis/Valkey FT, pgvector, …) use `L2`; VectorSets ranks by
/// cosine similarity intrinsically (VADD/VSIM take no metric), so its fixtures
/// must declare `cosine` and brute-force cosine ground truth — otherwise even a
/// perfectly-applied filter scores low recall against an L2 ranking.
#[derive(Clone, Copy)]
pub enum GtMetric {
    L2,
    Cosine,
}

impl GtMetric {
    /// datasets.json `distance` string.
    fn name(self) -> &'static str {
        match self {
            GtMetric::L2 => "l2",
            GtMetric::Cosine => "cosine",
        }
    }

    /// Distance (smaller = closer) between two vectors under this metric. For
    /// cosine we return `1 - cosine_similarity`; it is scale-invariant, so it
    /// matches VSIM's cosine ranking whether or not the vectors are normalized.
    fn dist(self, a: &[f32], b: &[f32]) -> f64 {
        match self {
            GtMetric::L2 => a
                .iter()
                .zip(b)
                .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                .sum(),
            GtMetric::Cosine => {
                let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
                let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                if na * nb > 0.0 {
                    1.0 - dot / (na * nb)
                } else {
                    1.0
                }
            }
        }
    }
}

/// Build a full temp project (datasets + config + results dir) for a
/// `match_any` benchmark and return its root. `engine_configs_json` is the
/// verbatim contents of `experiments/configurations/test.json` (a JSON array
/// of engine configs). `dim` is the vector dimensionality.
pub fn write_match_any_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> MatchAnyProject {
    write_match_any_project_metric(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::L2,
        N_QUERIES,
    )
}

/// Cosine-ground-truth variant of [`write_match_any_project`] for engines that
/// rank by cosine similarity (VectorSets).
pub fn write_match_any_cosine_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> MatchAnyProject {
    write_match_any_project_metric(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::Cosine,
        N_QUERIES,
    )
}

/// [`write_match_any_project`] with an explicit query count. The mixed/filter
/// harnesses cap `num_to_run` at the number of queries in the fixture, so a
/// larger count is needed to exercise the multi-worker join-merge (and, for
/// mixed, to reliably drive updates) at `parallel >= 4`.
pub fn write_match_any_project_n(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    n_queries: usize,
) -> MatchAnyProject {
    write_match_any_project_metric(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::L2,
        n_queries,
    )
}

/// Cosine variant of [`write_match_any_project_n`] (VectorSets).
pub fn write_match_any_cosine_project_n(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    n_queries: usize,
) -> MatchAnyProject {
    write_match_any_project_metric(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::Cosine,
        n_queries,
    )
}

fn write_match_any_project_metric(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    metric: GtMetric,
    n_queries: usize,
) -> MatchAnyProject {
    // Deterministic data/queries so ground truth is reproducible across engines.
    let mut rng = StdRng::seed_from_u64(0xA11CE);
    let gen_vec =
        |rng: &mut StdRng| -> Vec<f32> { (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..N_DOCS).map(|_| gen_vec(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| gen_vec(&mut rng)).collect();

    // Nearest neighbours computed over the FILTERED corpus only.
    let filtered_gt = |q: &[f32]| -> Vec<i64> {
        let mut scored: Vec<(i64, f64)> = (0..N_DOCS)
            .filter(|id| matches_filter(*id))
            .map(|id| (id as i64, metric.dist(q, &vectors[id])))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.iter().take(TOP).map(|(id, _)| *id).collect()
    };

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp); // keep the dir alive for the subprocess

    let ds_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&ds_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    // Data vectors -> vectors.npy (implicit ids 0..N_DOCS).
    write_npy_vectors(ds_dir.join("vectors.npy").to_str().unwrap(), &vectors).unwrap();

    // Per-document metadata -> payloads.jsonl (keyword `color`, int `size`).
    let payloads: String = (0..N_DOCS)
        .map(|id| {
            serde_json::json!({ "color": color_for(id), "size": (id % 3) as i64 + 1 }).to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("payloads.jsonl"), payloads).unwrap();

    // Queries + match_any condition + filtered ground truth -> tests.jsonl.
    let any_vals: Vec<serde_json::Value> = MATCH_ANY_COLORS
        .iter()
        .map(|c| serde_json::json!(c))
        .collect();
    let tests: String = queries
        .iter()
        .map(|q| {
            serde_json::json!({
                "query": q.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                "conditions": { "and": [ { "color": { "match": { "any": any_vals } } } ] },
                "closest_ids": filtered_gt(q),
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("tests.jsonl"), tests).unwrap();

    let datasets_json = serde_json::json!([{
        "name": dataset_name,
        "type": "tar",
        "path": format!("{}/", dataset_name),
        "distance": metric.name(),
        "vector_size": dim,
        "vector_count": N_DOCS,
        "schema": { "color": "keyword", "size": "int" },
    }]);
    fs::write(
        root.join("datasets/datasets.json"),
        serde_json::to_string_pretty(&datasets_json).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();

    MatchAnyProject {
        root,
        dataset_name: dataset_name.to_string(),
        top: TOP,
        matching_docs: (0..N_DOCS).filter(|id| matches_filter(*id)).count(),
    }
}

/// Like [`write_match_any_project`], but the `match_any` filter is on a
/// MULTI-VALUED keyword field (`labels`, a per-doc 2-element array) instead of a
/// scalar `color`. Proves contains-any array semantics end-to-end: an engine
/// that stores the array as a joined scalar (the pre-#88 Milvus/Weaviate bug)
/// tests whole-value equality and scores ~0 recall. `metric` follows the engine
/// (L2 for most; cosine for VectorSets).
pub fn write_match_any_labels_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    metric: GtMetric,
) -> MatchAnyProject {
    let mut rng = StdRng::seed_from_u64(0xB0BA);
    let gen_vec =
        |rng: &mut StdRng| -> Vec<f32> { (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..N_DOCS).map(|_| gen_vec(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERIES).map(|_| gen_vec(&mut rng)).collect();

    // Ground truth over ONLY the docs whose labels array intersects the set.
    let filtered_gt = |q: &[f32]| -> Vec<i64> {
        let mut scored: Vec<(i64, f64)> = (0..N_DOCS)
            .filter(|id| matches_labels_filter(*id))
            .map(|id| (id as i64, metric.dist(q, &vectors[id])))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.iter().take(TOP).map(|(id, _)| *id).collect()
    };

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let ds_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&ds_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    write_npy_vectors(ds_dir.join("vectors.npy").to_str().unwrap(), &vectors).unwrap();

    // Per-document metadata -> payloads.jsonl (multi-valued keyword `labels`).
    let payloads: String = (0..N_DOCS)
        .map(|id| serde_json::json!({ "labels": labels_for(id) }).to_string())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("payloads.jsonl"), payloads).unwrap();

    let any_vals: Vec<serde_json::Value> = MATCH_ANY_LABELS
        .iter()
        .map(|c| serde_json::json!(c))
        .collect();
    let tests: String = queries
        .iter()
        .map(|q| {
            serde_json::json!({
                "query": q.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                "conditions": { "and": [ { "labels": { "match": { "any": any_vals } } } ] },
                "closest_ids": filtered_gt(q),
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("tests.jsonl"), tests).unwrap();

    let datasets_json = serde_json::json!([{
        "name": dataset_name,
        "type": "tar",
        "path": format!("{}/", dataset_name),
        "distance": metric.name(),
        "vector_size": dim,
        "vector_count": N_DOCS,
        "schema": { "labels": "keyword" },
    }]);
    fs::write(
        root.join("datasets/datasets.json"),
        serde_json::to_string_pretty(&datasets_json).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();

    MatchAnyProject {
        root,
        dataset_name: dataset_name.to_string(),
        top: TOP,
        matching_docs: (0..N_DOCS).filter(|id| matches_labels_filter(*id)).count(),
    }
}

/// Distinct `int` `size` values assigned round-robin to documents by `id % 5`
/// (values 1..=5). The `match_any` int filter selects a STRICT SUBSET of these
/// (see `MATCH_ANY_SIZES`), so an engine that ignores the filter — or that
/// compares the int filter against string-typed storage (the HIGH bug this
/// test guards) — returns whole-corpus nearest neighbours and scores low recall.
fn size_for(id: usize) -> i64 {
    (id % 5) as i64 + 1
}

/// The int `match_any` set every query filters on: `size IN {1, 2}` — a strict
/// subset of the 5 possible sizes (~40% of docs match).
pub const MATCH_ANY_SIZES: [i64; 2] = [1, 2];

fn matches_int_filter(id: usize) -> bool {
    MATCH_ANY_SIZES.contains(&size_for(id))
}

/// Like `write_match_any_project`, but attaches the `match_any` filter to the
/// INT `size` field (`{size: {match: {any: [1, 2]}}}` → Mongo `{size:{$in:[…]}}`).
/// Ground truth is brute-forced over ONLY documents whose `size` is in the
/// IN-set, so high recall proves the engine applied a NUMERIC `$in` that matched
/// natively-stored integers. A filter-ignoring engine — or one that emits an
/// integer `$in` against string-stored sizes — scores low recall.
pub fn write_match_any_int_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> MatchAnyProject {
    write_match_any_int_project_metric(dataset_name, engine_configs_json, dim, GtMetric::L2)
}

/// Cosine-ground-truth variant of [`write_match_any_int_project`] for engines
/// that rank by cosine similarity (VectorSets).
pub fn write_match_any_int_cosine_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> MatchAnyProject {
    write_match_any_int_project_metric(dataset_name, engine_configs_json, dim, GtMetric::Cosine)
}

fn write_match_any_int_project_metric(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    metric: GtMetric,
) -> MatchAnyProject {
    // Deterministic data/queries so ground truth is reproducible across engines.
    let mut rng = StdRng::seed_from_u64(0x5133_u64);
    let gen_vec =
        |rng: &mut StdRng| -> Vec<f32> { (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..N_DOCS).map(|_| gen_vec(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERIES).map(|_| gen_vec(&mut rng)).collect();

    // Nearest neighbours computed over the size-FILTERED corpus only.
    let filtered_gt = |q: &[f32]| -> Vec<i64> {
        let mut scored: Vec<(i64, f64)> = (0..N_DOCS)
            .filter(|id| matches_int_filter(*id))
            .map(|id| (id as i64, metric.dist(q, &vectors[id])))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.iter().take(TOP).map(|(id, _)| *id).collect()
    };

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp); // keep the dir alive for the subprocess

    let ds_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&ds_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    write_npy_vectors(ds_dir.join("vectors.npy").to_str().unwrap(), &vectors).unwrap();

    // Per-document metadata: keyword `color` + int `size`.
    let payloads: String = (0..N_DOCS)
        .map(|id| serde_json::json!({ "color": color_for(id), "size": size_for(id) }).to_string())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("payloads.jsonl"), payloads).unwrap();

    // Queries + int match_any condition + filtered ground truth -> tests.jsonl.
    let any_vals: Vec<serde_json::Value> = MATCH_ANY_SIZES
        .iter()
        .map(|s| serde_json::json!(s))
        .collect();
    let tests: String = queries
        .iter()
        .map(|q| {
            serde_json::json!({
                "query": q.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                "conditions": { "and": [ { "size": { "match": { "any": any_vals } } } ] },
                "closest_ids": filtered_gt(q),
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("tests.jsonl"), tests).unwrap();

    let datasets_json = serde_json::json!([{
        "name": dataset_name,
        "type": "tar",
        "path": format!("{}/", dataset_name),
        "distance": metric.name(),
        "vector_size": dim,
        "vector_count": N_DOCS,
        "schema": { "color": "keyword", "size": "int" },
    }]);
    fs::write(
        root.join("datasets/datasets.json"),
        serde_json::to_string_pretty(&datasets_json).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();

    MatchAnyProject {
        root,
        dataset_name: dataset_name.to_string(),
        top: TOP,
        matching_docs: (0..N_DOCS).filter(|id| matches_int_filter(*id)).count(),
    }
}

// ── Generic filter-datatype fixtures (bool / uuid / full-text / datetime) ──
//
// Mirrors `write_match_any_project` but is parameterised by the filter under
// test. Each builds a compound (`tar`) dataset (`vectors.npy` +
// `payloads.jsonl` + `tests.jsonl`) whose queries carry a fixed `conditions`
// filter, with ground truth brute-forced over ONLY the documents that satisfy
// the filter. A high recall therefore proves the engine actually applied the
// filter (returned the filtered nearest neighbours, not the whole corpus's).

/// A built filter-benchmark project (same shape as `MatchAnyProject`).
pub struct FilterProject {
    pub root: PathBuf,
    pub dataset_name: String,
    pub top: usize,
    /// Number of documents satisfying the filter (sanity bound: >> top).
    pub matching_docs: usize,
}

/// Core builder: `schema` is the datasets.json `schema` object, `payload_for`
/// returns the per-document payload object, `condition` is the (shared) filter
/// JSON attached to every query, and `matches` decides whether a document id
/// satisfies the filter (used to brute-force the filtered ground truth).
#[allow(clippy::too_many_arguments)]
fn write_filter_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    metric: GtMetric,
    schema: serde_json::Value,
    payload_for: impl Fn(usize) -> serde_json::Value,
    condition: serde_json::Value,
    matches: impl Fn(usize) -> bool,
) -> FilterProject {
    // Single shared filter → every query gets the same condition/predicate.
    write_filter_project_multi(
        dataset_name,
        engine_configs_json,
        dim,
        metric,
        N_QUERIES,
        schema,
        payload_for,
        move |_q| condition.clone(),
        move |_q, id| matches(id),
    )
}

/// Generalised core builder that allows EACH query to carry its own `condition`
/// and its own `matches` predicate (used for multi-tenancy, where every query
/// is scoped to a different tenant). `condition_for(q)` is the filter JSON for
/// query `q`; `matches_for(q, id)` decides whether document `id` satisfies query
/// `q`'s filter (used to brute-force that query's tenant-local ground truth).
///
/// `matching_docs` is reported as the MINIMUM per-query match count, so the
/// caller's `matching_docs >= top` sanity check bounds the smallest tenant.
#[allow(clippy::too_many_arguments)]
fn write_filter_project_multi(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    metric: GtMetric,
    n_queries: usize,
    schema: serde_json::Value,
    payload_for: impl Fn(usize) -> serde_json::Value,
    condition_for: impl Fn(usize) -> serde_json::Value,
    matches_for: impl Fn(usize, usize) -> bool,
) -> FilterProject {
    let mut rng = StdRng::seed_from_u64(0xF117E);
    let gen_vec =
        |rng: &mut StdRng| -> Vec<f32> { (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..N_DOCS).map(|_| gen_vec(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| gen_vec(&mut rng)).collect();

    // Nearest neighbours for query `q`, computed over ONLY the docs that satisfy
    // query `q`'s filter (its tenant/subset).
    let filtered_gt = |q_idx: usize, q: &[f32]| -> Vec<i64> {
        let mut scored: Vec<(i64, f64)> = (0..N_DOCS)
            .filter(|id| matches_for(q_idx, *id))
            .map(|id| (id as i64, metric.dist(q, &vectors[id])))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.iter().take(TOP).map(|(id, _)| *id).collect()
    };

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let ds_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&ds_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    write_npy_vectors(ds_dir.join("vectors.npy").to_str().unwrap(), &vectors).unwrap();

    let payloads: String = (0..N_DOCS)
        .map(|id| payload_for(id).to_string())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("payloads.jsonl"), payloads).unwrap();

    let tests: String = queries
        .iter()
        .enumerate()
        .map(|(q_idx, q)| {
            serde_json::json!({
                "query": q.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                "conditions": condition_for(q_idx),
                "closest_ids": filtered_gt(q_idx, q),
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("tests.jsonl"), tests).unwrap();

    let datasets_json = serde_json::json!([{
        "name": dataset_name,
        "type": "tar",
        "path": format!("{}/", dataset_name),
        "distance": metric.name(),
        "vector_size": dim,
        "vector_count": N_DOCS,
        "schema": schema,
    }]);
    fs::write(
        root.join("datasets/datasets.json"),
        serde_json::to_string_pretty(&datasets_json).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();

    // Smallest per-query match count (bounds the smallest tenant).
    let matching_docs = (0..n_queries)
        .map(|q_idx| (0..N_DOCS).filter(|id| matches_for(q_idx, *id)).count())
        .min()
        .unwrap_or(0);

    FilterProject {
        root,
        dataset_name: dataset_name.to_string(),
        top: TOP,
        matching_docs,
    }
}

/// bool filter: field `flag` (schema type `bool`), value `id % 2 == 0`. The
/// query filters `flag == true`, so half the corpus matches.
pub fn write_bool_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_bool_project_metric(dataset_name, engine_configs_json, dim, GtMetric::L2)
}

/// Cosine-ground-truth variant of [`write_bool_project`] for engines that rank by
/// cosine similarity (VectorSets).
pub fn write_bool_cosine_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_bool_project_metric(dataset_name, engine_configs_json, dim, GtMetric::Cosine)
}

fn write_bool_project_metric(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    metric: GtMetric,
) -> FilterProject {
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        metric,
        serde_json::json!({ "flag": "bool" }),
        |id| serde_json::json!({ "flag": id % 2 == 0 }),
        serde_json::json!({ "and": [ { "flag": { "match": { "value": true } } } ] }),
        |id| id % 2 == 0,
    )
}

/// UUID values assigned round-robin by `id % 4`. The query filters on the first
/// UUID (exact keyword/TAG match), so a quarter of the corpus matches.
pub const UUIDS: [&str; 4] = [
    "550e8400-e29b-41d4-a716-446655440000",
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
    "6ba7b812-9dad-11d1-80b4-00c04fd430c8",
];

/// uuid filter: field `uid` (schema type `uuid`), exact match on `UUIDS[0]`.
pub fn write_uuid_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::L2,
        serde_json::json!({ "uid": "uuid" }),
        |id| serde_json::json!({ "uid": UUIDS[id % UUIDS.len()] }),
        serde_json::json!({ "and": [ { "uid": { "match": { "value": UUIDS[0] } } } ] }),
        |id| id % UUIDS.len() == 0,
    )
}

/// full-text filter: field `body` (schema type `text`). Even docs contain the
/// term "quick"; odd docs do not. The query is a single-term full-text match on
/// "quick" (works on Redis TEXT and on Valkey's degraded tokenised-TAG path).
pub fn write_fulltext_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::L2,
        serde_json::json!({ "body": "text" }),
        |id| {
            let body = if id % 2 == 0 {
                "the quick brown fox"
            } else {
                "lazy dog sleeps here"
            };
            serde_json::json!({ "body": body })
        },
        serde_json::json!({ "and": [ { "body": { "match": { "text": "quick" } } } ] }),
        |id| id % 2 == 0,
    )
}

/// datetime filter: field `ts` (schema type `datetime`), one ISO-8601 timestamp
/// per doc spaced one day apart from 2021-01-01. The query is an ISO range
/// `[day 100, day 300)`, selecting ids 100..=299 (200 docs).
pub fn write_datetime_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_datetime_project_metric(dataset_name, engine_configs_json, dim, GtMetric::L2)
}

/// Cosine-ground-truth variant of [`write_datetime_project`] for engines that
/// rank by cosine similarity (VectorSets).
pub fn write_datetime_cosine_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_datetime_project_metric(dataset_name, engine_configs_json, dim, GtMetric::Cosine)
}

fn write_datetime_project_metric(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    metric: GtMetric,
) -> FilterProject {
    use chrono::{Duration, TimeZone, Utc};
    let base = Utc.timestamp_opt(1_609_459_200, 0).unwrap(); // 2021-01-01T00:00:00Z
    let iso_for = move |day: i64| (base + Duration::days(day)).to_rfc3339();
    let gte = iso_for(100);
    let lt = iso_for(300);
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        metric,
        serde_json::json!({ "ts": "datetime" }),
        move |id| serde_json::json!({ "ts": iso_for(id as i64) }),
        serde_json::json!({ "and": [ { "ts": { "range": { "gte": gte, "lt": lt } } } ] }),
        |id| (100..300).contains(&id),
    )
}

/// Same ISO-8601 timestamps as [`write_datetime_project`], but the schema
/// declares `ts` as **`keyword`**, not `datetime`, and the query carries a
/// **one-sided** `lt` bound.
///
/// This is the fixture for the storage-vs-filter DISAGREEMENT (PR #230 review
/// M2): an engine that decides "is this a datetime?" from the SCHEMA on the
/// storage side but from the VALUE on the filter side stores an ISO string here
/// while comparing against an epoch number. VectorSets coerces a non-numeric
/// attribute to `0` in a numeric comparison, so `.ts < <epoch>` then matches
/// EVERY document — the query runs effectively unfiltered, with exit code 0 and
/// no warning. Only recall detects it (measured live on redis 8.8: 0.800
/// schema-gated vs 1.000 once both halves agree).
///
/// The one-sided bound is deliberate: a two-sided range degrades to zero hits,
/// which is far more likely to be noticed. This is the quiet direction.
pub fn write_datetime_keyword_schema_cosine_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    use chrono::{Duration, TimeZone, Utc};
    let base = Utc.timestamp_opt(1_609_459_200, 0).unwrap(); // 2021-01-01T00:00:00Z
    let iso_for = move |day: i64| (base + Duration::days(day)).to_rfc3339();
    let lt = iso_for(200);
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::Cosine,
        // NOT "datetime" — the whole point of the fixture.
        serde_json::json!({ "ts": "keyword" }),
        move |id| serde_json::json!({ "ts": iso_for(id as i64) }),
        serde_json::json!({ "and": [ { "ts": { "range": { "lt": lt } } } ] }),
        |id| id < 200,
    )
}

/// Great-circle distance in metres (haversine, R=6_371_000 m). Used to
/// brute-force geo-radius ground truth. The ~55 m margin baked into
/// [`write_geo_project`] keeps every doc clearly inside or outside the radius
/// despite tiny differences vs each engine's own earth model.
fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6_371_000.0;
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dphi = (lat2 - lat1).to_radians();
    let dlambda = (lon2 - lon1).to_radians();
    let a = (dphi / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dlambda / 2.0).sin().powi(2);
    2.0 * R * a.sqrt().asin()
}

/// geo-radius filter: field `location` (schema `geo`), one point per doc along a
/// meridian ~111 m apart from (lat 40.0, lon -74.0). The query is a radius around
/// doc 0's location selecting the nearest ~198 docs; ground truth is brute-forced
/// with [`haversine_m`]. The reader parses geo as `{"lon":..,"lat":..}`; the
/// query condition is `{geo:{lat,lon,radius}}` with radius in METERS.
pub fn write_geo_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    let (lat0, lon0) = (40.0_f64, -74.0_f64);
    let loc = |id: usize| (lat0 + id as f64 * 0.001, lon0);
    // ~111 m per 0.001 deg latitude; 22 km ≈ the nearest 198 docs, and the
    // radius falls ~55 m between doc 197 (inside) and doc 198 (outside).
    let radius = 22_000.0_f64;
    let (q_lat, q_lon) = loc(0);
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::L2,
        serde_json::json!({ "location": "geo" }),
        move |id| {
            let (lat, lon) = loc(id);
            serde_json::json!({ "location": { "lon": lon, "lat": lat } })
        },
        serde_json::json!({ "and": [ { "location": { "geo": { "lat": q_lat, "lon": q_lon, "radius": radius } } } ] }),
        move |id| {
            let (lat, lon) = loc(id);
            haversine_m(lat, lon, q_lat, q_lon) <= radius
        },
    )
}

/// Multi-condition AND filter: a keyword match AND a numeric range in one query
/// (`color == "red" AND size >= 50`). Every other fixture puts a SINGLE condition
/// under `and`; this exercises that engines correctly COMPOSE (intersect) two
/// clauses of different types. `color` is `id % 2 == 0 ? "red" : "blue"` and
/// `size` is `id % 100`, so the ground truth is the even ids whose `id % 100 >=
/// 50` — an AND that neither clause alone selects.
pub fn write_and_filter_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::L2,
        serde_json::json!({ "color": "keyword", "size": "int" }),
        move |id| {
            serde_json::json!({
                "color": if id % 2 == 0 { "red" } else { "blue" },
                "size": (id % 100) as i64,
            })
        },
        serde_json::json!({ "and": [
            { "color": { "match": { "value": "red" } } },
            { "size": { "range": { "gte": 50 } } },
        ] }),
        move |id| id % 2 == 0 && (id % 100) as i64 >= 50,
    )
}

/// Multi-condition OR fixture: same `color`/`size` payload as the AND fixture,
/// but the query is a top-level `{or: [...]}` UNION — `color == "red" OR size >=
/// 90`. The two arms overlap only partially (all reds, plus the blue docs with
/// size in 90..99), so the union (~220 docs) is strictly larger than either arm
/// and strictly larger than their intersection. An engine that mis-handles OR —
/// treating it as AND, or dropping an arm — searches a much smaller doc set, so
/// its nearest neighbours diverge from the union's and recall collapses.
pub fn write_or_filter_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::L2,
        serde_json::json!({ "color": "keyword", "size": "int" }),
        move |id| {
            serde_json::json!({
                "color": if id % 2 == 0 { "red" } else { "blue" },
                "size": (id % 100) as i64,
            })
        },
        serde_json::json!({ "or": [
            { "color": { "match": { "value": "red" } } },
            { "size": { "range": { "gte": 90 } } },
        ] }),
        move |id| id % 2 == 0 || (id % 100) as i64 >= 90,
    )
}

/// Nested/grouped boolean fixture: `(color == "red" AND size >= 50) OR
/// (color == "blue" AND size < 10)`. The condition is a top-level `or` whose two
/// arms are themselves `and` GROUPS — a genuine two-level tree that CANNOT be
/// flattened to top-level and/or without changing its meaning. A parser that
/// mis-flattens it (the historical behaviour) matches a wildly different doc set
/// — an OR-of-all-leaves matches ~everything, an AND-of-all-leaves matches
/// nothing (color can't be both red and blue) — so either way its nearest
/// neighbours diverge from the ~120-doc nested set and recall collapses.
pub fn write_nested_filter_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_nested_filter_project_metric(dataset_name, engine_configs_json, dim, GtMetric::L2)
}

/// Cosine-ground-truth variant of [`write_nested_filter_project`] for engines
/// that rank by cosine similarity (VectorSets).
pub fn write_nested_filter_cosine_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_nested_filter_project_metric(dataset_name, engine_configs_json, dim, GtMetric::Cosine)
}

fn write_nested_filter_project_metric(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    metric: GtMetric,
) -> FilterProject {
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        metric,
        serde_json::json!({ "color": "keyword", "size": "int" }),
        move |id| {
            serde_json::json!({
                "color": if id % 2 == 0 { "red" } else { "blue" },
                "size": (id % 100) as i64,
            })
        },
        serde_json::json!({ "or": [
            { "and": [
                { "color": { "match": { "value": "red" } } },
                { "size": { "range": { "gte": 50 } } },
            ] },
            { "and": [
                { "color": { "match": { "value": "blue" } } },
                { "size": { "range": { "lt": 10 } } },
            ] },
        ] }),
        move |id| {
            let size = (id % 100) as i64;
            (id % 2 == 0 && size >= 50) || (id % 2 == 1 && size < 10)
        },
    )
}

// ── Selectivity-ladder fixture ──────────────────────────────────────────────
//
// The #1 methodology idea shared by VectorDBBench, Pinecone VSB and qdrant's
// vector-db-benchmark: filter recall/latency must be measured as a FUNCTION of
// filter selectivity, not at a single point. A restrictive filter (few matching
// docs) and a permissive one exercise very different engine code paths — and
// naive post-filtering HNSW can COLLAPSE at low selectivity, because the graph
// walk rarely reaches a matching node, whereas a pre-filtering (or
// brute-force-below-threshold) engine stays correct.
//
// This fixture puts an int field `rank` = doc id (0..N_DOCS) on every doc and
// emits ONE query per rung of `SELECTIVITY_LADDER`, each a `rank < K` range that
// selects exactly the K lowest ranks (selectivity K/N_DOCS). So a single dataset
// sweeps the same range-filter path from ~3% to ~99% selectivity. Ground truth
// is brute-forced over only the surviving docs per rung, so an engine that drops
// the filter — or whose recall collapses at the restrictive end — scores low.
//
// SCOPE NOTE: with only N_DOCS=400 and a high ef, search is near-exhaustive, so
// this asserts range-filter CORRECTNESS across selectivity boundaries (each rung
// a distinct range extent) rather than reproducing at-scale post-filter collapse
// (which needs ef << corpus). It is the local, deterministic counterpart to the
// large selectivity-graded datasets those external tools ship.

/// Filter-match counts (out of `N_DOCS` = 400) for each selectivity rung, from
/// highly restrictive (~3%) to barely filtered (~99%). The tightest rung keeps
/// `>= TOP` matches so recall is well-defined at every point.
pub const SELECTIVITY_LADDER: [usize; 8] = [12, 20, 40, 100, 200, 300, 360, 396];

/// selectivity-ladder filter: field `rank` (schema type `int`) = doc id. Query
/// `q` filters `rank < SELECTIVITY_LADDER[q]`, sweeping selectivity across the
/// ladder, with per-rung ground truth brute-forced over only the matching docs.
pub fn write_selectivity_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_filter_project_multi(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::L2,
        SELECTIVITY_LADDER.len(),
        serde_json::json!({ "rank": "int" }),
        |id| serde_json::json!({ "rank": id as i64 }),
        move |q| {
            let k = SELECTIVITY_LADDER[q] as i64;
            serde_json::json!({ "and": [ { "rank": { "range": { "lt": k } } } ] })
        },
        move |q, id| (id as i64) < SELECTIVITY_LADDER[q] as i64,
    )
}

// ── Multi-tenancy fixture ───────────────────────────────────────────────────
//
// Mirrors upstream qdrant/vector-db-benchmark's `random-768-*-tenants` scenario:
// MANY tenants share ONE index; every search is scoped to a single tenant via a
// keyword-equality filter on a `tenant` field, and recall is measured against
// the nearest neighbours WITHIN that tenant only. It reuses the existing
// keyword-TAG filter path (no new engine code) — the ONLY difference from the
// other filter fixtures is that each query targets a DIFFERENT tenant.
//
// Because the ground truth is tenant-local, recall is also a leakage detector:
// any cross-tenant document an engine wrongly returns is absent from that
// query's ground truth, so it cannot count toward recall AND it displaces a
// correct tenant-local neighbour — a leaking engine therefore scores low recall.

/// Number of tenants sharing the single index. With `N_DOCS` docs assigned
/// round-robin, each tenant owns `N_DOCS / N_TENANTS` docs (400 / 25 = 16 > TOP).
pub const N_TENANTS: usize = 25;

/// Tenant label for document / query index `k` (round-robin by `k % N_TENANTS`).
pub fn tenant_for(k: usize) -> String {
    format!("tenant_{}", k % N_TENANTS)
}

/// multi-tenancy filter: field `tenant` (schema type `keyword`), one tenant per
/// doc round-robin. Query `q` is scoped to tenant `q % N_TENANTS` via an exact
/// keyword match, with ground truth brute-forced over ONLY that tenant's docs.
pub fn write_tenant_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    // One query per tenant (N_TENANTS queries) so EVERY tenant label — including
    // the two-digit ones — is exercised as a query scope, not just as documents.
    write_filter_project_multi(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::L2,
        N_TENANTS,
        serde_json::json!({ "tenant": "keyword" }),
        |id| serde_json::json!({ "tenant": tenant_for(id) }),
        |q| {
            serde_json::json!({
                "and": [ { "tenant": { "match": { "value": tenant_for(q) } } } ]
            })
        },
        |q, id| id % N_TENANTS == q % N_TENANTS,
    )
}

// ── Sparse-vector fixture ───────────────────────────────────────────────────
//
// Builds a small sparse (`type: "sparse"`) dataset: `data.csr` + `queries.csr`
// + `neighbours.jsonl`. Ground truth is brute-forced by sparse DOT PRODUCT and
// sorted DESCENDING (sparse similarity is MIPS — larger dot = more similar), so
// a high recall proves the engine ran a real sparse-index search. Sorting the
// wrong way (ascending, as if it were an L2 distance) would pick the least
// similar docs and silently zero out recall — hence the explicit `b.cmp a`.

/// A built sparse-benchmark project.
pub struct SparseProject {
    pub root: PathBuf,
    pub dataset_name: String,
    pub top: usize,
}

/// Build a temp project with a deterministic random sparse dataset and its
/// dot-product (descending) ground truth. `engine_configs_json` is the verbatim
/// `experiments/configurations/test.json`.
pub fn write_sparse_project(dataset_name: &str, engine_configs_json: &str) -> SparseProject {
    write_sparse_project_with_gt(dataset_name, engine_configs_json, false)
}

/// Same fixture, but the ground truth is written as the BINARY `results.gt`
/// block the public `msmarco-sparse-*` datasets ship, with no `neighbours.jsonl`
/// at all.
///
/// This is the only end-to-end exercise of the `results.gt` branch. Without it
/// nothing proves the ids inside `results.gt` are 0-based row indices matching
/// the ids the uploader assigns from `data.csr` row order — if they were 1-based
/// (or document ids), recall on a published `msmarco-sparse-1M` run would be
/// near zero and every other test in this repo would still be green.
pub fn write_sparse_project_gt(dataset_name: &str, engine_configs_json: &str) -> SparseProject {
    write_sparse_project_with_gt(dataset_name, engine_configs_json, true)
}

fn write_sparse_project_with_gt(
    dataset_name: &str,
    engine_configs_json: &str,
    binary_gt: bool,
) -> SparseProject {
    const DIM: usize = 300;
    const NNZ: usize = 10;
    const N: usize = 150;
    const Q: usize = 10;
    const TOP: usize = 10;

    // Fixed seed → reproducible data/queries/ground-truth across engines & runs.
    // Generation is shared with the `generate-dataset` binary via
    // `vector_db_benchmark::synthetic` so both produce byte-identical datasets.
    let SparseData {
        data,
        queries,
        neighbours: neighbors,
    } = generate_sparse(0x5A5A_5EED, DIM, NNZ, N, Q, TOP);

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let ds_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&ds_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();
    write_sparse_matrix(ds_dir.join("data.csr").to_str().unwrap(), &data).unwrap();
    write_sparse_matrix(ds_dir.join("queries.csr").to_str().unwrap(), &queries).unwrap();
    if binary_gt {
        write_gt_neighbours(ds_dir.join("results.gt").to_str().unwrap(), &neighbors).unwrap();
    } else {
        write_neighbours(&ds_dir, &neighbors);
    }

    let datasets_json = serde_json::json!([{
        "name": dataset_name, "type": "sparse", "path": dataset_name,
        "distance": "dot", "vector_size": DIM, "vector_count": N,
    }]);
    fs::write(
        root.join("datasets/datasets.json"),
        serde_json::to_string_pretty(&datasets_json).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();

    SparseProject {
        root,
        dataset_name: dataset_name.to_string(),
        top: TOP,
    }
}

// ── Hybrid (dense + sparse) fixture ─────────────────────────────────────────
//
// Builds a `type: "hybrid"` dataset: dense `vectors.npy`/`queries.npy` (L2) +
// sparse `data.csr`/`queries.csr` (dot/MIPS) + a SHARED `neighbours.jsonl`.
//
// GROUND-TRUTH / RECALL-FLOOR CHOICE — the ground truth R is recoverable ONLY
// via fusion; NEITHER modality alone reaches the floor.
//
// We deliberately do NOT brute-force the exact RRF order (its constant `k` is a
// server detail). Instead we PLANT, per query, K ground-truth docs split into
// two halves and two rings of single-modality distractors:
//   * R_dense (K/2 docs): dense ranks 0..K/2 (nearest by L2), but only MODERATE
//     sparse dot → in the sparse ranking they land at ranks K..3K/2 (below both
//     R_sparse and the sparse distractors).
//   * R_sparse (K/2 docs): sparse ranks 0..K/2 (highest dot), but only MODERATE
//     dense distance → in the dense ranking they land at ranks K..3K/2.
//   * D_d (K/2 dense-only distractors): dense ranks K/2..K (just past R_dense),
//     ~zero sparse dot → absent from the meaningful sparse list.
//   * D_s (K/2 sparse-only distractors): sparse ranks K/2..K (just below
//     R_sparse), dense-far → absent from the meaningful dense list.
//
// Consequence:
//   * dense-only top-K  = R_dense + D_d  → recall(R) ≈ 0.5
//   * sparse-only top-K = R_sparse + D_s → recall(R) ≈ 0.5
//   * fused (RRF) top-K = R (all K)      → recall(R) ≈ 1.0
// Under RRF every R doc appears in BOTH prefetches (its "off" modality ranks it
// at K..3K/2, still inside the prefetch depth), so it collects TWO 1/(k+rank)
// terms — and one of them has rank < K/2. Every distractor appears in only ONE
// prefetch with rank ≥ K/2, so its best (only meaningful) term is ≤ 1/(k+K/2) <
// 1/(k+K/2−1). Thus every R doc outscores every distractor for ANY k ≥ 0, and
// the fused top-K is exactly R. We assert a 0.9 FLOOR on the fused recall to
// absorb ANN slack, and the companion `*-dense` view (registered below) drives
// the SAME data through a plain dense search as a NEGATIVE CONTROL that MUST
// score < 0.6 — proving the dataset genuinely requires fusion. An inverted
// sparse orientation (ascending, as if L2), a dropped sparse prefetch, or a
// broken `Fusion::Rrf` all collapse the fused result toward one modality and
// fail the floor.

/// A built hybrid-benchmark project. `dataset_name` is the `type:"hybrid"`
/// dataset; `dense_dataset_name` is a dense-only (`type:"jsonl"`) VIEW over the
/// SAME dense vectors + SAME ground truth, used as a negative control.
pub struct HybridProject {
    pub root: PathBuf,
    pub dataset_name: String,
    pub dense_dataset_name: String,
    pub top: usize,
}

/// Build a temp project with a deterministic planted hybrid dataset whose fused
/// (RRF) top-K ground truth is recoverable ONLY by combining both modalities.
pub fn write_hybrid_project(dataset_name: &str, engine_configs_json: &str) -> HybridProject {
    // The planted dataset is generated by the shared `synthetic::generate_hybrid`
    // (same code path as the `generate-dataset` binary), so the fixture and the
    // registered dataset are byte-identical. See that function for the full
    // fused-only-recoverable planting rationale.
    let HybridData {
        dense,
        dense_queries: dense_q,
        sparse,
        sparse_queries: sparse_q,
        neighbours,
        dim: dense_dim,
        top: k,
    } = generate_hybrid(0xB19_1DEA);
    let n = dense.len();

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let ds_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&ds_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    write_npy_vectors(ds_dir.join("vectors.npy").to_str().unwrap(), &dense).unwrap();
    write_npy_vectors(ds_dir.join("queries.npy").to_str().unwrap(), &dense_q).unwrap();
    write_sparse_matrix(ds_dir.join("data.csr").to_str().unwrap(), &sparse).unwrap();
    write_sparse_matrix(ds_dir.join("queries.csr").to_str().unwrap(), &sparse_q).unwrap();
    write_neighbours(&ds_dir, &neighbours);

    // Dense-only VIEW (negative control): same dense vectors + same ground truth
    // as a plain jsonl dataset, so an ordinary dense search can be run on it.
    let dense_dataset_name = format!("{dataset_name}-dense");
    let dv_dir = root.join("datasets").join(&dense_dataset_name);
    fs::create_dir_all(&dv_dir).unwrap();
    write_jsonl_vectors(&dv_dir.join("vectors.jsonl"), &dense);
    write_jsonl_vectors(&dv_dir.join("queries.jsonl"), &dense_q);
    write_neighbours(&dv_dir, &neighbours);

    let datasets_json = serde_json::json!([
        {
            "name": dataset_name, "type": "hybrid", "path": dataset_name,
            "distance": "l2", "vector_size": dense_dim, "vector_count": n,
        },
        {
            "name": dense_dataset_name, "type": "jsonl", "path": format!("{dense_dataset_name}/"),
            "distance": "l2", "vector_size": dense_dim, "vector_count": n,
        },
    ]);
    fs::write(
        root.join("datasets/datasets.json"),
        serde_json::to_string_pretty(&datasets_json).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();

    HybridProject {
        root,
        dataset_name: dataset_name.to_string(),
        dense_dataset_name,
        top: k,
    }
}

/// Write `vectors` as a `.jsonl` file (delegates to the shared serializer in
/// `vector_db_benchmark::synthetic`, so there is a single source of truth).
fn write_jsonl_vectors(path: &Path, vectors: &[Vec<f32>]) {
    vector_db_benchmark::synthetic::write_jsonl_vectors(path, vectors).unwrap();
}

/// Write `neighbours.jsonl` (one JSON id-array per line) into `ds_dir`
/// (delegates to the shared serializer in `vector_db_benchmark::synthetic`).
fn write_neighbours(ds_dir: &Path, neighbours: &[Vec<i64>]) {
    vector_db_benchmark::synthetic::write_neighbours_jsonl(
        &ds_dir.join("neighbours.jsonl"),
        neighbours,
    )
    .unwrap();
}

/// Path to the compiled binary under test. Cargo exports
/// `CARGO_BIN_EXE_vector-db-benchmark` to integration tests automatically.
pub fn binary_path() -> PathBuf {
    if let Some(p) = std::env::var_os("CARGO_BIN_EXE_vector-db-benchmark") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/release/vector-db-benchmark")
}

/// Run the benchmark binary once for `engine`/`dataset`, with extra env vars
/// (engine host/port overrides). Returns whether it exited successfully;
/// prints stdout/stderr on failure.
pub fn run_binary(
    root: &Path,
    engine: &str,
    dataset: &str,
    host: &str,
    envs: &[(&str, &str)],
) -> bool {
    let mut cmd = std::process::Command::new(binary_path());
    cmd.args([
        "--engines",
        engine,
        "--datasets",
        dataset,
        "--host",
        host,
        "--skip-if-exists",
        "false",
    ])
    .current_dir(root);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run vector-db-benchmark");
    if !out.status.success() {
        eprintln!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    out.status.success()
}

/// Read `results.mean_recall` from the engine's search result JSON.
pub fn read_recall(root: &Path, engine: &str) -> f64 {
    let pattern = format!("{}-*-search-*.json", engine);
    let dir = root.join("results");
    let path = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            glob::Pattern::new(&pattern)
                .unwrap()
                .matches(&p.file_name().unwrap().to_string_lossy())
        })
        .unwrap_or_else(|| panic!("no search result for {}", engine));
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    v["results"]["mean_recall"].as_f64().unwrap()
}

/// Read the per-query `results.recalls` array from the engine's search result
/// JSON. Each entry is one query's recall vs its (tenant-local) ground truth, so
/// asserting a floor on EVERY entry catches a single tenant that leaked or was
/// mis-scoped — stronger than only checking the mean.
pub fn read_recalls(root: &Path, engine: &str) -> Vec<f64> {
    let pattern = format!("{}-*-search-*.json", engine);
    let dir = root.join("results");
    let path = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            glob::Pattern::new(&pattern)
                .unwrap()
                .matches(&p.file_name().unwrap().to_string_lossy())
        })
        .unwrap_or_else(|| panic!("no search result for {}", engine));
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    v["results"]["recalls"]
        .as_array()
        .expect("recalls array")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// Like [`run_binary`] but appends `extra` CLI args (e.g. `--skip-vector-index`
/// for the filter-only harness, or `--update-search-ratio 1:5` for mixed).
pub fn run_binary_extra(
    root: &Path,
    engine: &str,
    dataset: &str,
    host: &str,
    envs: &[(&str, &str)],
    extra: &[&str],
) -> bool {
    let mut cmd = std::process::Command::new(binary_path());
    cmd.args([
        "--engines",
        engine,
        "--datasets",
        dataset,
        "--host",
        host,
        "--skip-if-exists",
        "false",
    ]);
    cmd.args(extra);
    cmd.current_dir(root);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run vector-db-benchmark");
    if !out.status.success() {
        eprintln!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    out.status.success()
}

/// Read the whole `results` object from an engine's search result JSON, so a
/// test can assert on any field (percentiles, requested/failed_queries,
/// update_* metrics, the mean_precisions sentinel, …). `engine` is the result
/// filename prefix — note `--skip-vector-index` renames the engine to
/// `<engine_type>-no-vector`, so pass that prefix for filter-only runs.
pub fn read_results_obj(root: &Path, engine: &str) -> serde_json::Value {
    let pattern = format!("{}-*-search-*.json", engine);
    let dir = root.join("results");
    let path = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            glob::Pattern::new(&pattern)
                .unwrap()
                .matches(&p.file_name().unwrap().to_string_lossy())
        })
        .unwrap_or_else(|| panic!("no search result for {}", engine));
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    v["results"].clone()
}
