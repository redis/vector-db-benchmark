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
use vector_db_benchmark::readers::{
    write_gt_neighbours, write_multivector_matrix, write_npy_vectors, write_sparse_matrix,
};
use vector_db_benchmark::synthetic::{
    generate_hybrid, generate_multivector, generate_sparse, HybridData, MultiVectorGenData,
    SparseData,
};

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

pub const N_DOCS: usize = 400;
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

/// Earth radius these fixtures' ground truth is computed on. Matches Milvus'
/// `ST_DWITHIN` refine step and `engine::geo::EARTH_RADIUS_M` exactly; other
/// engines' models differ by up to ~0.11 % (see `engine/geo.rs`), which is why
/// every fixture except `write_geo_edge_project` keeps a wide boundary margin.
const GT_EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Great-circle distance in metres (haversine, R=[`GT_EARTH_RADIUS_M`]). Used to
/// brute-force geo-radius ground truth. The margin baked into
/// [`write_geo_project`] keeps every doc clearly inside or outside the radius
/// despite tiny differences vs each engine's own earth model; the figure lives
/// with the layout that sets it, in `write_geo_project`.
fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = GT_EARTH_RADIUS_M;
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
///
/// **What it proves, and what it does not.** Every document lies on ONE
/// meridian, so a lat/lon BOUNDING BOX and the great-circle CIRCLE of the same
/// radius select the identical set. Measured: swapping an engine's geo predicate
/// for a real lat/lon polygon still scores **1.000** here, and **0.240** on
/// [`write_geo_corner_project`]. So this fixture shows that *a* geo-ish filter
/// was applied — not that its shape is a circle. Prefer the corner fixture for
/// anything new; this one is kept because several engines' geo tests already
/// reference it.
pub fn write_geo_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    let (lat0, lon0) = (40.0_f64, -74.0_f64);
    let loc = |id: usize| (lat0 + id as f64 * 0.001, lon0);
    // ~111 m per 0.001 deg latitude; 22 km ≈ the nearest 198 docs. Measured, the
    // radius falls between doc 197 (94.6 m INSIDE) and doc 198 (16.6 m OUTSIDE),
    // so the tightest margin is 16.6 m — an earlier comment said "~55 m" here.
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

// ── Bounding-box-discriminating geo fixture (issue #223) ────────────────────
//
// [`write_geo_project`] puts every document on ONE meridian, so a lat/lon
// BOUNDING BOX and a great-circle CIRCLE of the same radius select the identical
// set: it proves a geo filter was applied, but it cannot tell a correct radius
// from a box that merely contains it. Every "cheap" geo implementation (a
// bounding box, an equirectangular approximation) passes it.
//
// This fixture is built to fail those. 400 documents sit in the lat/lon box of
// half-width `radius` around the query point:
//
//   * 100 within 0.71 · radius of the centre — INSIDE both the circle and the box;
//   * 300 in the four corner regions, each at least 1.10 · radius away —
//     OUTSIDE the circle, INSIDE the box.
//
// The two groups are INTERLEAVED by `id % 4`, not split at id 100, so the
// in-circle set is never also a contiguous upload batch (`batch_size: 100` in
// the geo configs). Without that, a failure that dropped every batch but the
// first would score 1.000 with no filter applied.
//
// Ground truth is the top-10 over the 100 in-circle documents only (25 %
// selectivity). An engine applying the true radius scores ~1.0; an engine
// applying the bounding box, or no filter at all, searches all 400 and lands
// roughly a quarter of the correct neighbours — measured at 0.240-0.280, far
// below the 0.9 floor every filter test in this repo asserts.
//
// WHAT IT IS BLIND TO, stated so nobody over-reads it. The guard band is
// [0.706544407377, 1.101281084372] * radius, so a radius anywhere in that band
// selects the identical 100 documents — a wrong radius there is a provably
// equivalent mutant here (measured: x1.101 -> 100 docs, x1.11 -> 104,
// x1.15 -> 128, x1.40 -> 400). Both bounds are INCLUSIVE and must be quoted to
// full precision, not rounded: at x0.706544 the fixture already selects 98, and
// x1.101281 still selects 100 — a 4e-7 rounding in either direction misstates it
// in exactly the way this note warns about.
//
// Two more blind spots, both undisclosed until review. (a) A lat/lon SWAP
// applied to both the storage and the query side selects the identical 100
// documents — self-consistent, so no recall fixture can see it; the axis order
// is pinned instead by unit tests on each engine's emitted string AND on
// milvus' stored WKT (`geo_wkt_point`), and partially by the edge fixture, where
// a swapped 179.9 is not a valid latitude. (b) All 10 queries share ONE centre
// and ONE radius, so a hard-coded centre passes here; `write_geo_edge_project`
// is the fixture that varies the centre.
// So is `>` vs `>=` (nothing sits within 10 % of the boundary), and so is an
// equirectangular approximation (~1e-5 relative error at 20 km, deep inside the
// band). Those are caught by the exact-string unit pins on each engine's emitted
// filter and by `engine::geo`'s poles-and-antimeridian haversine grid, NOT by
// this fixture. It is blind in both directions, which is also why it cannot see
// a prune box SMALLER than the cap — see `write_geo_edge_project` for that.
//
// `tests/integration_redis.rs` runs it against RediSearch's native
// `@f:[lon lat r m]` as a control, so a failure here is the engine, not the
// fixture.

/// Fraction of `radius` at which the four corner clusters sit, per axis. Each
/// component is ≤ 0.98 (inside the box) and the pair is ≥ √1.21 = 1.10 · radius
/// from the centre (outside the circle), so no document is near the boundary.
const GEO_CORNER_MIN: f64 = 0.78;
const GEO_CORNER_MAX: f64 = 0.98;

/// L2 variant of the bounding-box-discriminating geo fixture.
pub fn write_geo_corner_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_geo_corner_project_metric(dataset_name, engine_configs_json, dim, GtMetric::L2)
}

/// Cosine variant, for engines that rank by cosine intrinsically (VectorSets).
pub fn write_geo_corner_cosine_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_geo_corner_project_metric(dataset_name, engine_configs_json, dim, GtMetric::Cosine)
}

/// Degrees of latitude / longitude per metre at `lat`, used only to PLACE the
/// documents. Every in/out decision is made with [`haversine_m`], so this local
/// flat approximation cannot make the ground truth wrong — only the layout
/// slightly uneven, which the wide margins absorb.
fn geo_corner_offsets(lat: f64, radius_m: f64) -> (f64, f64) {
    const M_PER_DEG_LAT: f64 = 111_320.0;
    (
        radius_m / M_PER_DEG_LAT,
        radius_m / (M_PER_DEG_LAT * lat.to_radians().cos()),
    )
}

fn write_geo_corner_project_metric(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    metric: GtMetric,
) -> FilterProject {
    let (q_lat, q_lon) = (40.0_f64, -74.0_f64);
    let radius = 20_000.0_f64;
    let (d_lat, d_lon) = geo_corner_offsets(q_lat, radius);

    // (lat, lon) for document `id`. In-circle is `id % 4 == 0` — INTERLEAVED, not
    // the first 100 — so the in-circle set is never also an upload batch.
    //
    // It used to be ids 0..99, and `batch_size` is 100 in the Milvus and MongoDB
    // geo configs, so "the answer to every query" and "the first upload batch"
    // were the same 100 documents. Any failure that left only the first batch
    // resident, or that correlated with insertion order, would then score 1.000
    // with no geo filter applied at all. Interleaving removes the coincidence:
    // no contiguous run of 100 ids is the answer to anything here.
    let loc = move |id: usize| -> (f64, f64) {
        if id.is_multiple_of(4) {
            let k = id / 4; // 0..99
                            // 10x10 grid over [-0.5, 0.5]^2 of the box half-widths: at most
                            // 0.707 · radius from the centre.
            let (u, v) = (-0.5 + (k % 10) as f64 / 9.0, -0.5 + (k / 10) as f64 / 9.0);
            (q_lat + v * d_lat, q_lon + u * d_lon)
        } else {
            // 0..299 over the ids that are NOT multiples of 4.
            let k = id - id / 4 - 1;
            let span = GEO_CORNER_MAX - GEO_CORNER_MIN;
            let (u, v) = (
                GEO_CORNER_MIN + span * (k % 15) as f64 / 14.0,
                GEO_CORNER_MIN + span * ((k / 15) % 5) as f64 / 4.0,
            );
            let (su, sv) = ((k / 75) % 2, (k / 150) % 2);
            let u = if su == 0 { u } else { -u };
            let v = if sv == 0 { v } else { -v };
            (q_lat + v * d_lat, q_lon + u * d_lon)
        }
    };

    let proj = write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        metric,
        serde_json::json!({ "location": "geo" }),
        move |id| {
            let (lat, lon) = loc(id);
            serde_json::json!({ "location": { "lon": lon, "lat": lat } })
        },
        serde_json::json!({ "and": [ { "location": { "geo": {
            "lat": q_lat, "lon": q_lon, "radius": radius } } } ] }),
        move |id| {
            let (lat, lon) = loc(id);
            haversine_m(lat, lon, q_lat, q_lon) <= radius
        },
    );

    // The fixture's whole value is in these three properties, so assert them
    // here rather than trusting the arithmetic above: an off-by-one in the
    // layout would silently turn this back into a test a bounding box passes.
    let mut inside = 0usize;
    let mut in_box_outside_circle = 0usize;
    for id in 0..N_DOCS {
        let (lat, lon) = loc(id);
        let d = haversine_m(lat, lon, q_lat, q_lon);
        let in_box =
            (lat - q_lat).abs() <= d_lat * 1.000_001 && (lon - q_lon).abs() <= d_lon * 1.000_001;
        assert!(in_box, "doc {id} escaped the bounding box — it would be filtered out by BOTH a box and a circle, weakening the fixture");
        // No document within 5 % of the boundary, so no engine's earth model
        // can move one across it.
        assert!(
            d <= radius * 0.95 || d >= radius * 1.05,
            "doc {id} sits {d} m out, too close to the {radius} m boundary"
        );
        if d <= radius {
            inside += 1;
        } else {
            in_box_outside_circle += 1;
        }
    }
    assert_eq!(inside, 100, "in-circle count");
    assert_eq!(in_box_outside_circle, 300, "box-minus-circle count");
    assert_eq!(proj.matching_docs, inside);
    proj
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

// ── Antimeridian / near-boundary geo fixture (issue #223, Milvus RTREE) ─────
//
// `write_geo_corner_project` above defeats a bounding box that is LARGER than
// the circle. It cannot see the opposite bug — a prune box that is SMALLER —
// because its in-circle documents all sit at <= 0.71 * radius, comfortably
// inside any such box, and its out-of-circle documents are meant to be excluded
// anyway.
//
// Milvus has exactly that bug when a `Geometry` column carries an `RTREE`:
// `create_bounding_box_for_dwithin` divides by `111320.0` m/degree where the
// truth is 111 194.93 (0.112 % high), and it does not wrap the antimeridian at
// all. The two axes are NOT affected equally, which is why only some bearings
// prune: modelling the box as lat +- d/111320 and lon +- d/(111320 cos phi), at
// centre (81, 10) with r = 200 km the in-cap cutoff is x0.998876 on bearings
// 0/180 (so N/S documents past that are dropped) but x1.0121 on 90/270 (so E/W
// documents are NOT). At (0, 179.9) the bearing-90 cutoff is exactly x0.5 —
// every document across +180 is dropped. Anything inside the cap but outside that box is
// discarded before the exact `ST_DWITHIN` refine ever runs. On the shipped
// dataset that silently caps recall at ~0.935.
//
// This fixture puts documents exactly where those three defects bite:
//
//   * query 0 — centre (81 N, 10 E), the high-latitude case, with in-cap
//     documents on all four cardinal bearings at 0.9990-0.9997 * radius: inside
//     the cap, outside a box 0.11 % short;
//   * query 1 — centre (0 N, 179.9 E), with in-cap documents 30-150 km east and
//     west, so half of them sit ACROSS the antimeridian at negative longitudes,
//     which an unwrapped box drops outright.
//
// Measured first-hand against `milvusdb/milvus:v2.6.19`, identical rows in two
// collections that differ only by the presence of the RTREE:
//
//   centre (81, 10), r = 200 km : truth 20 docs — RTREE returned 14, dropping
//     the six at 0.9990-0.9995 * radius due NORTH and SOUTH (the latitude
//     half-width is the one the 111 320 divisor shortens);
//   centre (0, 179.9), r = 200 km : truth 8 docs — RTREE returned 4, dropping
//     every one of the four that crossed +180.
//
// Without the index both cases return the full truth.
//
// HOW SEVERE, AND WHY THE HARNESS DOES NOT SEE IT. Measured on this exact
// 400-document layout, two collections with identical rows differing only by the
// RTREE, using the engine's own emitted `ST_DWITHIN` filter:
//
//   q0 (81 N)        truth 150 -> RTREE returns  76  (49 % of the in-cap set gone)
//   q1 (antimeridian) truth 150 -> RTREE returns 112  (25 % gone)
//
// So the fixture IS a strong discriminator — those are not a handful of
// documents, and several of each query's ground-truth top-10 are inside the
// pruned set. It simply never gets the chance, and the reason is INDEX CREATION
// ORDER, not scale and not the query path:
//
//   index in `collections/create`'s `indexParams` (before insert) -> pruned
//     (q0 76/150, q1 112/150), on BOTH `entities/query` and `entities/search`;
//   index via `indexes/create` AFTER insert + flush             -> not pruned
//     (150/150 on both queries, both paths).
//
// This engine creates scalar indexes after insert, which is why re-adding the
// RTREE and running the integration test still scored 1.000. That is an artifact
// of ordering at this size, not a guarantee — which is exactly why the column is
// left unindexed rather than relying on it. Two earlier explanations were wrong
// and are recorded here so they are not re-derived: "the ~6 pruned documents
// rarely reach the top-10" (the pruned set is 74 and 38 documents, not ~6) and
// "the vector-search path ignores the prune" (it honours it — see the ordering
// result above).
//
// CALIBRATION WARNING. The query-0 band is only 0.1 % wide, which is NARROWER
// than the spread between engines' earth radii (pgvector's `earthdistance` uses
// 6 378 168 m, 0.11 % above the 6 371 000 m this fixture's ground truth uses, so
// it would legitimately exclude those documents). Use this fixture only with an
// engine whose radius is 6 371 000 m — Milvus' `ST_DWITHIN` is. The query-1
// antimeridian half has no such caveat: those documents are 30-150 km inside a
// 200 km cap under any earth model, so that half discriminates robustly.

/// Destination point from `(lat, lon)` after travelling `dist_m` on `bearing`,
/// on a sphere of [`GT_EARTH_RADIUS_M`]. Longitude is normalised to [-180, 180],
/// which is what makes the antimeridian cases actually cross it.
fn dest_point(lat: f64, lon: f64, bearing_deg: f64, dist_m: f64) -> (f64, f64) {
    let ang = dist_m / GT_EARTH_RADIUS_M;
    let (phi1, lam1, theta) = (lat.to_radians(), lon.to_radians(), bearing_deg.to_radians());
    let phi2 = (phi1.sin() * ang.cos() + phi1.cos() * ang.sin() * theta.cos()).asin();
    let lam2 =
        lam1 + (theta.sin() * ang.sin() * phi1.cos()).atan2(ang.cos() - phi1.sin() * phi2.sin());
    let mut lon2 = lam2.to_degrees();
    // Normalise into [-180, 180) so a point past +180 comes back as negative.
    while lon2 >= 180.0 {
        lon2 -= 360.0;
    }
    while lon2 < -180.0 {
        lon2 += 360.0;
    }
    (phi2.to_degrees(), lon2)
}

/// The two query centres and their radius. Query 0 is the high-latitude
/// near-boundary case, query 1 the antimeridian case.
const GEO_EDGE_CENTRES: [(f64, f64); 2] = [(81.0, 10.0), (0.0, 179.9)];
const GEO_EDGE_RADIUS_M: f64 = 200_000.0;

/// Where document `id` sits. 0..199 belong to centre 0, 200..399 to centre 1;
/// within each half the first 150 are inside the cap and the last 50 outside.
fn geo_edge_loc(id: usize) -> (f64, f64) {
    let half = id / 200;
    let k = id % 200;
    let (clat, clon) = GEO_EDGE_CENTRES[half];
    // East and west first — those are the bearings the longitude defect hits.
    let bearing = [90.0, 270.0, 0.0, 180.0][k % 4];
    if k < 150 {
        let frac = if half == 0 {
            // 0.9990 .. 0.9997 — inside the cap, outside a 0.11 %-short box.
            0.9990 + 0.0007 * (k / 4) as f64 / 37.0
        } else {
            // 0.15 .. 0.75 of the radius: 30-150 km out, robustly inside the cap
            // under any earth model, but across the antimeridian going east.
            0.15 + 0.60 * (k / 4) as f64 / 37.0
        };
        dest_point(clat, clon, bearing, GEO_EDGE_RADIUS_M * frac)
    } else {
        // Outside: 1.05 .. 1.40 of the radius.
        let frac = 1.05 + 0.35 * ((k - 150) / 4) as f64 / 12.0;
        dest_point(clat, clon, bearing, GEO_EDGE_RADIUS_M * frac)
    }
}

/// Antimeridian / near-boundary geo fixture. Two queries, each a geo-radius on
/// the `location` field, with ground truth brute-forced per query.
pub fn write_geo_edge_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    let matches = |q: usize, id: usize| {
        let (lat, lon) = geo_edge_loc(id);
        let (clat, clon) = GEO_EDGE_CENTRES[q];
        haversine_m(lat, lon, clat, clon) <= GEO_EDGE_RADIUS_M
    };

    let proj = write_filter_project_multi(
        dataset_name,
        engine_configs_json,
        dim,
        GtMetric::L2,
        GEO_EDGE_CENTRES.len(),
        serde_json::json!({ "location": "geo" }),
        move |id| {
            let (lat, lon) = geo_edge_loc(id);
            serde_json::json!({ "location": { "lon": lon, "lat": lat } })
        },
        move |q| {
            let (clat, clon) = GEO_EDGE_CENTRES[q];
            serde_json::json!({ "and": [ { "location": { "geo": {
                "lat": clat, "lon": clon, "radius": GEO_EDGE_RADIUS_M } } } ] })
        },
        matches,
    );

    // The fixture is worthless unless it actually contains the two shapes it
    // claims, so assert them rather than trusting the layout arithmetic.
    let mut near_boundary = 0usize; // in-cap, but outside a 0.11 %-short box
    let mut across_antimeridian = 0usize; // in-cap for q1, opposite lon sign
    for id in 0..N_DOCS {
        let (lat, lon) = geo_edge_loc(id);
        if matches(0, id) {
            let d = haversine_m(lat, lon, GEO_EDGE_CENTRES[0].0, GEO_EDGE_CENTRES[0].1);
            if d > GEO_EDGE_RADIUS_M * 0.99888 {
                near_boundary += 1;
            }
        }
        if matches(1, id) && lon < 0.0 {
            across_antimeridian += 1;
        }
    }
    // Counts in-cap documents past 0.99888 * radius on ANY bearing. Only the
    // ~74 on bearings 0/180 are actually outside Milvus' prune box (see the
    // per-axis cutoffs in the header); this is the superset, so the message says
    // what it computes rather than what it implies.
    assert!(
        near_boundary >= 20,
        "fixture must hold in-cap documents past 0.99888 * radius (the latitude \
         cutoff of a 0.112 %-high m/degree divisor), got {near_boundary}"
    );
    assert!(
        across_antimeridian >= 20,
        "fixture must hold in-cap documents on the far side of the antimeridian, \
         got {across_antimeridian}"
    );
    assert!(
        proj.matching_docs >= proj.top,
        "every query needs >= top matching docs, smallest is {}",
        proj.matching_docs
    );
    proj
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

// ── Multi-vector (ColBERT-style / MaxSim) fixture ───────────────────────────

/// A built multivector-benchmark project.
pub struct MultiVectorProject {
    pub root: PathBuf,
    pub dataset_name: String,
    pub top: usize,
    /// Per-token dimensionality — exposed so callers that need to re-register
    /// this SAME corpus under a different `datasets.json` (e.g. to flip
    /// `distance` while keeping the same `.mvec` files/collection) don't have
    /// to hardcode a literal that can silently drift from `DIM`.
    pub dim: usize,
}

/// Build a temp project with a deterministic random multi-vector dataset and
/// its brute-force MaxSim (descending) ground truth. Shares
/// `vector_db_benchmark::synthetic::generate_multivector` with the
/// `generate-dataset` binary, so the fixture and the registered dataset are
/// byte-identical.
pub fn write_multivector_project(
    dataset_name: &str,
    engine_configs_json: &str,
) -> MultiVectorProject {
    write_multivector_project_with_distance(dataset_name, engine_configs_json, "dot")
}

/// Same fixture as [`write_multivector_project`], but with a caller-chosen
/// `distance`. NOTE: as of #316, this parameter is only ever exercised with
/// `"dot"` (via [`write_multivector_project`]) — the live cosine-guard
/// integration test needs to re-register an ALREADY-UPLOADED corpus under a
/// different distance mid-test, which this function can't do (it always
/// allocates a fresh corpus/tempdir), so that test hand-writes its own
/// second `datasets.json` instead of calling this with `"cosine"`.
pub fn write_multivector_project_with_distance(
    dataset_name: &str,
    engine_configs_json: &str,
    distance: &str,
) -> MultiVectorProject {
    const DIM: usize = 16;
    const MIN_TOKENS: usize = 4;
    const MAX_TOKENS: usize = 8;
    const N: usize = 150;
    const Q: usize = 10;
    const TOP: usize = 10;

    let MultiVectorGenData {
        data,
        queries,
        neighbours,
    } = generate_multivector(0xC0FFEE, DIM, MIN_TOKENS, MAX_TOKENS, N, Q, TOP);

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let ds_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&ds_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();
    write_multivector_matrix(ds_dir.join("data.mvec").to_str().unwrap(), &data).unwrap();
    write_multivector_matrix(ds_dir.join("queries.mvec").to_str().unwrap(), &queries).unwrap();
    write_neighbours(&ds_dir, &neighbours);

    let datasets_json = serde_json::json!([{
        "name": dataset_name, "type": "multivector", "path": dataset_name,
        "distance": distance, "vector_size": DIM,
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

    MultiVectorProject {
        root,
        dataset_name: dataset_name.to_string(),
        top: TOP,
        dim: DIM,
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

/// A finished `vector-db-benchmark` run: its exit status plus stdout+stderr.
pub struct BinaryRun {
    pub ok: bool,
    pub combined: String,
}

/// Like [`run_binary`], but hands back the output instead of only a bool.
///
/// For the tests that assert an engine REFUSES a dataset (issue #223: Chroma and
/// Turbopuffer cannot express a geo radius). "The run failed" on its own is a
/// weak assertion — a crash, a typo in the config or an unreachable server all
/// satisfy it — so those tests need the message to check WHICH failure it was.
pub fn run_binary_capture(
    root: &Path,
    engine: &str,
    dataset: &str,
    host: &str,
    envs: &[(&str, &str)],
) -> BinaryRun {
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
    BinaryRun {
        ok: out.status.success(),
        combined: format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    }
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
/// update_* metrics, the mean_precision_at_returned sentinel, …). `engine` is the result
/// filename prefix — note `--skip-vector-index` renames the engine to
/// `<engine_type>-no-vector`, so pass that prefix for filter-only runs.
/// Read the `params` block from the engine's search result JSON.
///
/// Used to assert on `params.corpus_reuse` — the recorded `--skip-upload`
/// verdict (#238), which is the only thing distinguishing a verified run from a
/// waived one in the artifact.
pub fn read_params_obj(root: &Path, engine: &str) -> serde_json::Value {
    read_result_doc(root, engine)["params"].clone()
}

/// The whole search-result document for `engine`.
pub fn read_result_doc(root: &Path, engine: &str) -> serde_json::Value {
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
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

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

// ---------------------------------------------------------------------------
// Destructive-suite instance ownership (issue #292)
// ---------------------------------------------------------------------------
//
// Four integration suites (`integration_redis`, `integration_valkey`,
// `integration_dragonfly`, `integration_kividb`) share one `flush_db()` shape:
// drop every index `FT._LIST` reports, then `FLUSHALL`. That destroys the WHOLE
// server — every database, not just db 0 — and each suite's default port is a
// shared container from `tests/docker-compose.test.yml`. Two incidents were
// caused by a port override that looked applied but was not, so the run silently
// destroyed a server it did not own.
//
// The guard below makes the destructive suites refuse to touch a server unless
// they can POSITIVELY establish it is safe. It is deliberately placed behind the
// suites' `test_port()`, because EVERY path that reaches the server — direct
// `redis::Client` connections and the `REDIS_PORT`-style env vars handed to
// spawned benchmark binaries alike — resolves its port there.
//
// Design rule, learned the hard way: **every way the probe can fail must refuse.**
// The first version coerced an unreachable server, a denied command and an
// unsupported command all to "0 keys", i.e. to `Fresh`, which fails OPEN in the
// one function whose job is to refuse.

/// Env var an operator sets to waive the ownership check.
pub const ALLOW_DIRTY_ENV: &str = "VDBB_TEST_ALLOW_DIRTY";

/// How long [`claim_resp_instance`] waits for the server to become reachable.
///
/// Must be >= the longest `wait_for_*()` deadline in the guarded suites (30 s in
/// `integration_valkey` / `_dragonfly` / `_kividb`, 10 s in `integration_redis`).
/// If it were shorter, the documented "bring the container up, then run the
/// suite" workflow would race past the guard: the probe would give up, the
/// suite's own wait would succeed, and — because the claim runs once per process
/// behind a `OnceLock` — it would never be retried.
const CLAIM_REACHABLE_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// Per-attempt connect timeout inside that window.
const CLAIM_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// What the pre-flight probe managed to learn about the target server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Probe {
    /// The server answered every question the guard asked.
    Reachable {
        /// Keys summed over ALL databases (`INFO keyspace`), not just db 0.
        /// `DBSIZE` was wrong here: it counts one database while `FLUSHALL`
        /// destroys all of them, so keys parked in db 1+ read as "empty".
        keys: i64,
        /// Number of entries `FT._LIST` returned.
        index_count: usize,
        /// Per-start server identity; see [`server_identity`]. May be empty.
        server_id: String,
    },
    /// The guard could not establish that the server is safe to destroy —
    /// unreachable, still loading, or a probe command that errored. Carries a
    /// short reason for the message.
    Inconclusive(String),
}

/// Verdict of the ownership check a FLUSHALL-issuing suite runs before it
/// touches a server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Claim {
    /// Server holds nothing, so this process takes it over.
    Fresh,
    /// Server is non-empty, but either the operator waived the check or the
    /// server's identity matches the claim this target directory recorded.
    Reused,
    /// Not established as safe. Carries the message to show the operator.
    Refuse(String),
}

/// Everything [`claim_verdict`] depends on, so the decision is unit-testable
/// with no live server.
#[derive(Debug, Clone)]
pub struct ClaimInputs<'a> {
    /// Cargo test target, e.g. `"integration_redis"`.
    pub target: &'a str,
    /// Env var that selects this suite's port, e.g. `"REDIS_TEST_PORT"`.
    pub port_env: &'a str,
    pub host: &'a str,
    pub port: u16,
    /// What the probe learned.
    pub probe: &'a Probe,
    /// Server identity this target directory recorded the last time it claimed
    /// `host:port`, if any.
    pub prior_claim: Option<&'a str>,
    /// Where that claim is stored, for the message.
    pub claim_path: &'a str,
    /// Port this engine's server listens on INSIDE its container, so the
    /// `docker run -p …` line in the refusal is runnable. NOT always 6379 —
    /// kividb listens on 6380 (`tests/docker-compose.test.yml` maps
    /// `6386:6380`), and an operator who followed a hardcoded `:6379` would get
    /// a container the suite cannot reach and then a second refusal from this
    /// very guard.
    pub container_port: u16,
    /// Operator set [`ALLOW_DIRTY_ENV`].
    pub forced: bool,
}

/// Advice appended to every refusal.
fn refusal_footer(i: &ClaimInputs<'_>) -> String {
    format!(
        "Point the suite at a container of your own:\n\
         \n\
         \x20   docker run -d --rm -p <your-port>:{container} --name <your-name> <image>\n\
         \x20   {port_env}=<your-port> cargo test --test {target} -- --test-threads=1\n\
         \n\
         `{port_env}` is the supported way to move the suite. Do not edit the port in\n\
         tests/{target}.rs instead: tests/harness_invariants.rs pins that default to the\n\
         `{target}` mapping in tests/docker-compose.test.yml, so an edit fails the build,\n\
         and it would move the default for everyone rather than just for you.\n\
         \n\
         If the server really is yours and its contents are disposable, re-run with\n\
         {allow}=1 to waive this check.\n",
        container = i.container_port,
        port_env = i.port_env,
        target = i.target,
        allow = ALLOW_DIRTY_ENV,
    )
}

/// Decide whether a destructive suite may run against the server described by
/// `i`. Pure: no I/O, no env reads, no clock.
pub fn claim_verdict(i: &ClaimInputs<'_>) -> Claim {
    if i.forced {
        return Claim::Reused;
    }

    let (keys, index_count, server_id) = match i.probe {
        Probe::Inconclusive(why) => {
            return Claim::Refuse(format!(
                "\n\n\
                 REFUSING TO RUN `{target}` AGAINST {host}:{port}\n\
                 \n\
                 The safety check could not establish that this server is safe to destroy:\n\
                 \x20   {why}\n\
                 \n\
                 Every test in `{target}` calls `flush_db()`, which drops every index\n\
                 `FT._LIST` reports and then issues `FLUSHALL` — destroying EVERY database on\n\
                 the server. The check refuses rather than guess, because guessing wrong is\n\
                 unrecoverable.\n\
                 \n\
                 {footer}",
                target = i.target,
                host = i.host,
                port = i.port,
                why = why,
                footer = refusal_footer(i),
            ));
        }
        Probe::Reachable {
            keys,
            index_count,
            server_id,
        } => (*keys, *index_count, server_id.as_str()),
    };

    if keys <= 0 && index_count == 0 {
        return Claim::Fresh;
    }
    if !server_id.is_empty() && i.prior_claim == Some(server_id) {
        return Claim::Reused;
    }

    // Non-empty and unclaimed. Say WHY, because the two cases need opposite
    // actions from the operator.
    let diagnosis = match i.prior_claim {
        None => format!(
            "This target directory has NO claim recorded for {host}:{port}.\n\
             \x20   (looked in {claim_path})\n\
             \n\
             Either the server is not yours, or your claim was removed — `cargo clean`, a\n\
             fresh CARGO_TARGET_DIR, a new worktree, or `--target <triple>` (which relocates\n\
             the claim under <target>/<triple>/) all lose it. A previous run that was killed\n\
             mid-suite also leaves the server populated with no claim.",
            host = i.host,
            port = i.port,
            claim_path = i.claim_path,
        ),
        Some(prior) if server_id.is_empty() => format!(
            "A claim IS recorded for {host}:{port} ({claim_path}, instance {prior}), but this\n\
             server reports neither `run_id` nor `master_replid`, so nothing can be matched\n\
             against it. An unidentifiable server is never auto-authorised.",
            host = i.host,
            port = i.port,
            claim_path = i.claim_path,
            prior = display_id(prior),
        ),
        Some(prior) => format!(
            "A claim IS recorded for {host}:{port} ({claim_path}), but it does not match:\n\
             \x20   claimed instance: {prior}\n\
             \x20   instance now:     {now}\n\
             \n\
             The server was restarted or replaced. NOTE: `docker restart` of your OWN\n\
             container changes both `run_id` and `master_replid` while an RDB-persisted\n\
             dataset survives — so if this is your container, this refusal is a false alarm.\n\
             Delete {claim_path} (or set {allow}=1) and re-run.",
            host = i.host,
            port = i.port,
            claim_path = i.claim_path,
            prior = display_id(prior),
            now = display_id(server_id),
            allow = ALLOW_DIRTY_ENV,
        ),
    };

    Claim::Refuse(format!(
        "\n\n\
         REFUSING TO RUN `{target}` AGAINST {host}:{port}\n\
         \n\
         That server holds {keys} key(s) across all databases and {indexes} search index(es)\n\
         that this target directory has no claim for. Every test in `{target}` calls\n\
         `flush_db()`, which drops every index `FT._LIST` reports and then issues `FLUSHALL`\n\
         — destroying EVERY database on the server, not just db 0.\n\
         \n\
         {diagnosis}\n\
         \n\
         {footer}",
        target = i.target,
        host = i.host,
        port = i.port,
        keys = keys,
        indexes = index_count,
        diagnosis = diagnosis,
        footer = refusal_footer(i),
    ))
}

/// Render a possibly-empty instance id for a message.
fn display_id(id: &str) -> &str {
    if id.is_empty() {
        "<none reported>"
    } else {
        id
    }
}

/// Sum `keys=` across every `dbN:` line of an `INFO keyspace` section, or `None`
/// when the section cannot be trusted.
///
/// Redis omits empty databases entirely, so a section that has the `# Keyspace`
/// header and no `dbN:` lines legitimately means zero keys. What must NOT read as
/// zero is a reply that never was a keyspace section, or a `dbN:` line whose
/// `keys=` is missing, unparseable or negative — the earlier `-> i64` version
/// collapsed all of those to `0`, i.e. to `Fresh`, which is this guard's own bug
/// class one notch in. `None` becomes [`Probe::Inconclusive`], which refuses.
///
/// Field ORDER and the extra fields differ per engine — verified live on
/// redis:8.8.0 (`db0:keys=1,expires=0,avg_ttl=0,subexpiry=0`), valkey-bundle
/// (`...,keys_with_volatile_items=0`), dragonfly df-v1.40.1
/// (`db0:keys=1,expires=0,hits=0,misses=0,hit_ratio=0.00,avg_ttl=-1`) and kividb
/// v1.0.2-full (`db0:keys=1,expires=0,avg_ttl=0`) — so parse by name. All four
/// emit the `# Keyspace` header (measured).
pub fn sum_keyspace_keys(info: &str) -> Option<i64> {
    if !info
        .lines()
        .any(|l| l.trim().eq_ignore_ascii_case("# keyspace"))
    {
        return None;
    }
    let mut total: i64 = 0;
    for line in info.lines() {
        let line = line.trim();
        if !line.starts_with("db") {
            continue;
        }
        let (_db, fields) = line.split_once(':')?;
        let keys: i64 = fields
            .split(',')
            .find_map(|kv| kv.trim().strip_prefix("keys="))?
            .trim()
            .parse()
            .ok()?;
        if keys < 0 {
            return None;
        }
        total = total.checked_add(keys)?;
    }
    Some(total)
}

/// Pull `field:` out of an `INFO` section body.
pub fn info_field(info: &str, field: &str) -> String {
    let prefix = format!("{field}:");
    info.lines()
        .find_map(|l| l.trim().strip_prefix(prefix.as_str()))
        .unwrap_or("")
        .trim()
        .to_string()
}

/// A per-START identity for the server, so a claim recorded against one server
/// is not honoured for a different one that later took the port.
///
/// `run_id` from `INFO server` where available (redis:8.8.0, valkey-bundle,
/// kividb v1.0.2-full), falling back to `master_replid` from `INFO replication`
/// (dragonfly df-v1.40.1 reports no `run_id`; all four checked live). Empty when
/// neither is present — an empty identity never matches a prior claim.
///
/// LIMITS, both measured:
///   * `master_replid` identifies a REPLICATION GROUP, not an instance: a
///     Dragonfly replica reports the same value as its primary. A claim recorded
///     against a primary would therefore also authorise a replica of it that
///     later occupied the same host:port.
///   * Both values are re-generated on restart, so `docker restart` of your own
///     container invalidates its claim even though an RDB-persisted dataset
///     survives. That direction is safe (it refuses), and the refusal message
///     names it explicitly.
fn server_identity(conn: &mut redis::Connection) -> String {
    let server: String = redis::cmd("INFO")
        .arg("server")
        .query(conn)
        .unwrap_or_default();
    let run_id = info_field(&server, "run_id");
    if !run_id.is_empty() {
        return run_id;
    }
    let repl: String = redis::cmd("INFO")
        .arg("replication")
        .query(conn)
        .unwrap_or_default();
    info_field(&repl, "master_replid")
}

/// Probe the server. EVERY failure path yields [`Probe::Inconclusive`], which
/// [`claim_verdict`] refuses on. `Probe::Reachable` is constructed at exactly
/// one place: the final statement, after `INFO keyspace` AND `FT._LIST` both
/// returned `Ok`.
///
/// `pub` and `wait`-parameterised so `tests/harness_invariants.rs` can assert the
/// unreachable case directly against a closed port with a zero wait. Without
/// that, the whole producer side was untested: a mutant returning
/// `Reachable { keys: 0, index_count: 0 }` for an unreachable server — the exact
/// regression INV-P4's docstring cites — passed every invariant.
pub fn probe_server(host: &str, port: u16, wait: std::time::Duration) -> Probe {
    let url = format!("redis://{host}:{port}/");
    let Ok(client) = redis::Client::open(url.as_str()) else {
        return Probe::Inconclusive(format!("could not parse the connection URL {url}"));
    };

    // Wait for the server the same way the suite's own `wait_for_*()` does. A
    // container that is still starting must NOT skip the check: that was the
    // whole bug — the probe gave up, the suite's 10-30 s wait then succeeded,
    // and the `OnceLock` meant the claim was never retried.
    let deadline = std::time::Instant::now() + wait;
    let mut conn = loop {
        match client.get_connection_with_timeout(CLAIM_CONNECT_TIMEOUT) {
            Ok(c) => break c,
            Err(e) => {
                if std::time::Instant::now() >= deadline {
                    return Probe::Inconclusive(format!(
                        "could not reach {host}:{port} within {}s ({e})",
                        wait.as_secs()
                    ));
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };

    // A server replaying its RDB/AOF answers INFO but reports a keyspace that is
    // still filling, so "0 keys" would be a lie.
    //
    // Best-effort by design: an engine whose `INFO persistence` errors, or that
    // omits `loading` entirely (kividb v1.0.2-full does — measured), simply gets
    // no loading check. Making a missing section fatal would refuse those engines
    // outright, and a server so locked down that INFO fails is caught two lines
    // below by `INFO keyspace` anyway.
    if let Ok(persistence) = redis::cmd("INFO")
        .arg("persistence")
        .query::<String>(&mut conn)
    {
        if info_field(&persistence, "loading") == "1" {
            return Probe::Inconclusive(format!(
                "{host}:{port} is still loading its dataset (INFO persistence reports \
                 loading:1), so its keyspace cannot be trusted yet"
            ));
        }
    }

    let keyspace = match redis::cmd("INFO")
        .arg("keyspace")
        .query::<String>(&mut conn)
    {
        Ok(s) => s,
        Err(e) => {
            return Probe::Inconclusive(format!(
                "`INFO keyspace` failed on {host}:{port} ({e}), so the number of keys at \
                 risk is unknown"
            ));
        }
    };
    let Some(keys) = sum_keyspace_keys(&keyspace) else {
        return Probe::Inconclusive(format!(
            "could not parse the `INFO keyspace` reply from {host}:{port} (no `# Keyspace` \
             header, or a `dbN:` line without a usable `keys=`), so the number of keys at \
             risk is unknown; the reply was: {keyspace:?}"
        ));
    };

    let index_count = match redis::cmd("FT._LIST").query::<Vec<String>>(&mut conn) {
        Ok(v) => v.len(),
        Err(e) => {
            return Probe::Inconclusive(format!(
                "`FT._LIST` failed on {host}:{port} ({e}); `flush_db()` drops every index it \
                 reports, so the indexes at risk are unknown"
            ));
        }
    };

    Probe::Reachable {
        keys,
        index_count,
        server_id: server_identity(&mut conn),
    }
}

/// Path of the file recording which server identity this target directory last
/// claimed for `host:port`. Lives beside the test binaries (derived from
/// `current_exe()`: `<target>/<profile>/deps/<bin>` -> `<target>`), so a session
/// running with its own `CARGO_TARGET_DIR` has its own claims.
fn claim_file(target: &str, host: &str, port: u16) -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let target_dir = exe.parent()?.parent()?.parent()?;
    Some(
        target_dir
            .join("vdbb-test-claims")
            .join(format!("{target}-{host}-{port}.claim")),
    )
}

/// Claim `host:port` for a destructive suite, or panic with the refusal message.
///
/// Called from the suite's `test_port()`. The WORK happens once per process; the
/// panic is replayed cheaply for every later test.
///
/// That split matters. The suites memoize the port with
/// `OnceLock::get_or_init`, and `get_or_init` does NOT memoize an initializer
/// that panics — so putting the panic inside it made EVERY test re-run the whole
/// 30 s reachability loop. Measured against a dead port, the same two tests took
/// 60.21 s with the panic inside the initializer and 30.07 s with it outside; the
/// 44-test suite would have needed ~22 min (44 x 30 s) to report "server not
/// available" and now finishes in 30.17 s. Here the memoized value is the VERDICT
/// (`None` = allowed, `Some(msg)` = refuse) and the panic happens outside the
/// initializer, so only the first test pays the loop.
static CLAIM_OUTCOME: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

pub fn claim_resp_instance(
    target: &str,
    port_env: &str,
    host: &str,
    port: u16,
    container_port: u16,
) {
    if let Some(msg) =
        CLAIM_OUTCOME.get_or_init(|| claim_outcome(target, port_env, host, port, container_port))
    {
        panic!("{msg}");
    }
}

/// The decision, as a value. Never panics, so [`CLAIM_OUTCOME`] memoizes it.
///
/// One `OnceLock` per process is enough because a test binary is one suite
/// against one server: `target`/`host`/`port` are identical on every call.
fn claim_outcome(
    target: &str,
    port_env: &str,
    host: &str,
    port: u16,
    container_port: u16,
) -> Option<String> {
    let forced = std::env::var(ALLOW_DIRTY_ENV)
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);
    if forced {
        // Never silent: a stale `export` in a shell rc would otherwise disable
        // the guard permanently with no signal at all.
        //
        // Written straight to the process's stderr handle, NOT via `eprintln!`:
        // libtest redirects the print macros into its per-test capture buffer and
        // only replays it for FAILING tests, so an `eprintln!` here is invisible
        // in exactly the case that matters — a passing run whose guard is off.
        use std::io::Write as _;
        let _ = writeln!(
            std::io::stderr(),
            "WARNING: {ALLOW_DIRTY_ENV} is set, so `{target}` will NOT verify that it owns \
             {host}:{port} before `flush_db()` drops every FT._LIST index and FLUSHALLs the \
             server. Unset {ALLOW_DIRTY_ENV} to restore the #292 safety check."
        );
    }

    let probe = probe_server(host, port, CLAIM_REACHABLE_WAIT);

    let path = claim_file(target, host, port);
    let path_display = path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<claim path unavailable>".to_string());
    let prior = path
        .as_ref()
        .and_then(|p| fs::read_to_string(p).ok())
        .map(|s| s.trim().to_string());

    let verdict = claim_verdict(&ClaimInputs {
        target,
        port_env,
        host,
        port,
        probe: &probe,
        prior_claim: prior.as_deref(),
        claim_path: &path_display,
        container_port,
        forced,
    });

    match verdict {
        Claim::Refuse(msg) => Some(msg),
        Claim::Fresh | Claim::Reused => {
            if let Probe::Reachable { server_id, .. } = &probe {
                if let Some(p) = path {
                    if let Some(dir) = p.parent() {
                        let _ = fs::create_dir_all(dir);
                    }
                    let _ = fs::write(&p, server_id);
                }
            }
            None
        }
    }
}
