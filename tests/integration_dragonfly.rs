//! Integration tests for the Dragonfly engine (Dragonfly Search FT.* KNN).
//!
//! Requires Dragonfly running on port 6385.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d dragonfly
//! Run with:   DRAGONFLY_PORT=6385 cargo test --test integration_dragonfly -- --test-threads=1
//!
//! DESTRUCTIVE. Every test calls `flush_db()`, which drops every index
//! `FT._LIST` reports and then issues `FLUSHALL`. `DRAGONFLY_PORT` is the ONLY
//! supported way to point the suite at your own container — do NOT edit the port
//! in this file (`tests/harness_invariants.rs` rejects a second port literal).
//! `test_port()` also claims the instance on first use and refuses to run if the
//! server holds state it has no recorded claim for. Any probe it cannot complete
//! (unreachable, still loading, a denied or unsupported command) also refuses.
//!
//! Scope: vector KNN (whole-corpus COSINE ground truth, so recall reflects index
//! quality alone) PLUS metadata filtering — Dragonfly Search supports hybrid
//! filtered KNN, so the `test_binary_dragonfly_*` tests exercise the filter path
//! (bool/uuid/datetime/full-text/AND/OR/nested) against filtered ground truth.
//! GEO is unsupported (Dragonfly's geo parser rejects the builder's `$param`s).

use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use redis::Connection;

mod common;

/// (vectors, queries, cosine-ground-truth neighbours) produced by [`make_data`].
type KnnData = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<i64>>);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Dragonfly port under test. `DRAGONFLY_PORT` is the ONLY supported way to
/// move this suite off the shared default — these tests call `flush_db()`,
/// which `FLUSHALL`s the whole server. The first call also claims the instance
/// (see `common::claim_resp_instance`), so a server holding state this harness
/// has no recorded claim for is refused instead of destroyed.
fn test_port() -> u16 {
    static PORT: std::sync::OnceLock<u16> = std::sync::OnceLock::new();
    *PORT.get_or_init(|| {
        let port = std::env::var("DRAGONFLY_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6385);
        common::claim_resp_instance("integration_dragonfly", "DRAGONFLY_PORT", TEST_HOST, port);
        port
    })
}

const TEST_HOST: &str = "127.0.0.1";

fn get_test_connection() -> Connection {
    let url = format!("redis://{}:{}/", TEST_HOST, test_port());
    let client = redis::Client::open(url.as_str()).expect("Failed to create Dragonfly client");
    client
        .get_connection()
        .expect("Failed to connect to Dragonfly. Is dragonfly running on the test port?")
}

fn wait_for_dragonfly() {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let url = format!("redis://{}:{}/", TEST_HOST, test_port());
        if let Ok(client) = redis::Client::open(url.as_str()) {
            if let Ok(mut conn) = client.get_connection() {
                let pong: Result<String, _> = redis::cmd("PING").query(&mut conn);
                if pong.is_ok() {
                    return;
                }
            }
        }
        if Instant::now() > deadline {
            panic!("Dragonfly not available on port {} after 30s", test_port());
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn flush_db(conn: &mut Connection) {
    // Drop all FT indexes first, then flush leftover keys.
    if let Ok(indexes) = redis::cmd("FT._LIST").query::<Vec<String>>(conn) {
        for idx_name in indexes {
            let _ = redis::cmd("FT.DROPINDEX").arg(&idx_name).query::<()>(conn);
        }
    }
    let _: () = redis::cmd("FLUSHALL").query(conn).unwrap();
}

/// Cosine distance `1 - cos_sim` (scale-invariant; matches DISTANCE_METRIC COSINE
/// ranking whether or not the vectors are normalized).
fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - dot / (na.sqrt() * nb.sqrt())
}

fn brute_force_cosine_neighbors(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
    let mut dists: Vec<(i64, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as i64, cosine_distance(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(top).map(|(id, _)| *id).collect()
}

/// Write a self-contained pure-KNN project (vectors.jsonl + queries.jsonl +
/// neighbours.jsonl, cosine distance) and return its root. Ground truth is
/// whole-corpus cosine NN — no filters — so recall measures index quality only.
#[allow(clippy::too_many_arguments)]
fn create_knn_project(
    dataset_name: &str,
    engine_configs_json: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    neighbors: &[Vec<i64>],
    dim: usize,
) -> PathBuf {
    let tmp = tempfile::tempdir().expect("Failed to create temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp); // keep alive for the subprocess

    let dataset_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&dataset_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    let mut vecs_content = String::new();
    for v in vectors {
        let line: Vec<f64> = v.iter().map(|x| *x as f64).collect();
        vecs_content.push_str(&serde_json::to_string(&line).unwrap());
        vecs_content.push('\n');
    }
    fs::write(dataset_dir.join("vectors.jsonl"), &vecs_content).unwrap();

    let mut queries_content = String::new();
    for q in queries {
        let line: Vec<f64> = q.iter().map(|x| *x as f64).collect();
        queries_content.push_str(&serde_json::to_string(&line).unwrap());
        queries_content.push('\n');
    }
    fs::write(dataset_dir.join("queries.jsonl"), &queries_content).unwrap();

    let mut neighbors_content = String::new();
    for n in neighbors {
        neighbors_content.push_str(&serde_json::to_string(n).unwrap());
        neighbors_content.push('\n');
    }
    fs::write(dataset_dir.join("neighbours.jsonl"), &neighbors_content).unwrap();

    let datasets_json = serde_json::json!([{
        "name": dataset_name,
        "type": "jsonl",
        "path": format!("{}/", dataset_name),
        "distance": "cosine",
        "vector_size": dim,
        "vector_count": vectors.len(),
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

    root
}

/// Deterministic vectors/queries + cosine ground truth for a KNN run.
fn make_data(n_docs: usize, n_queries: usize, dim: usize, top: usize) -> KnnData {
    let mut rng = StdRng::seed_from_u64(0xD6A6);
    let gen =
        |rng: &mut StdRng| -> Vec<f32> { (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..n_docs).map(|_| gen(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| gen(&mut rng)).collect();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_cosine_neighbors(q, &vectors, top))
        .collect();
    (vectors, queries, neighbors)
}

fn run_knn_recall_test(engine_name: &str, dataset_name: &str, parallel: u64) -> f64 {
    wait_for_dragonfly();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let n_docs = 2000;
    let n_queries = 100;
    let top = 10;

    let (vectors, queries, neighbors) = make_data(n_docs, n_queries, dim, top);

    let engine_config = serde_json::json!([{
        "name": engine_name,
        "engine": "dragonfly",
        "connection_params": {},
        "collection_params": {
            "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 }
        },
        "search_params": [
            { "parallel": parallel, "top": top, "search_params": { "ef": 256 } }
        ],
        "upload_params": { "parallel": if parallel > 1 { parallel } else { 1 }, "batch_size": 64 }
    }]);

    let root = create_knn_project(
        dataset_name,
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        dim,
    );

    let port = test_port().to_string();
    let ok = common::run_binary(
        &root,
        engine_name,
        dataset_name,
        TEST_HOST,
        &[("DRAGONFLY_PORT", port.as_str())],
    );
    assert!(ok, "benchmark binary run failed for {}", engine_name);

    let recall = common::read_recall(&root, engine_name);
    println!(
        "dragonfly KNN recall (parallel={}) = {:.3}",
        parallel, recall
    );
    recall
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// End-to-end KNN recall via the binary at parallel=1. Asserts recall >= 0.9
/// against whole-corpus cosine ground truth (proves upload + FT.SEARCH KNN work).
#[test]
fn test_dragonfly_knn_recall() {
    let recall = run_knn_recall_test("dragonfly-knn-p1", "dragonfly-knn-p1", 1);
    assert!(recall >= 0.9, "dragonfly KNN recall {:.3} < 0.9", recall);
}

/// Same KNN recall check at parallel=4 — exercises the multi-worker search path
/// (per-thread sample buffers merged on join) and concurrent connections.
#[test]
fn test_dragonfly_knn_recall_parallel() {
    let recall = run_knn_recall_test("dragonfly-knn-p4", "dragonfly-knn-p4", 4);
    assert!(
        recall >= 0.9,
        "dragonfly KNN recall (parallel=4) {:.3} < 0.9",
        recall
    );
}

/// Run one metadata-filter fixture end-to-end and assert recall >= 0.9 vs the
/// filtered ground truth — proving Dragonfly Search applies the filter (a dropped
/// or mis-built filter runs kNN over the wrong doc set and recall collapses).
fn run_dragonfly_filter_test(
    name: &str,
    dataset: &str,
    build: impl Fn(&str, &str, usize) -> common::FilterProject,
) {
    wait_for_dragonfly();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": name, "engine": "dragonfly", "algorithm": "hnsw",
        "collection_params": { "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 } },
        "search_params": [{ "parallel": 1, "search_params": { "ef": 256 } }],
        "upload_params": { "parallel": 1, "batch_size": 64 }
    }]);
    let proj = build(dataset, &serde_json::to_string(&configs).unwrap(), dim);
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );
    let port = test_port().to_string();
    assert!(
        common::run_binary(
            &proj.root,
            name,
            dataset,
            TEST_HOST,
            &[("DRAGONFLY_PORT", port.as_str())],
        ),
        "dragonfly {} run failed",
        name
    );
    let recall = common::read_recall(&proj.root, name);
    println!("dragonfly {} recall={:.3}", name, recall);
    assert!(
        recall >= 0.9,
        "dragonfly {} recall {:.3} < 0.9",
        name,
        recall
    );
}

/// Bool-equality filter — Dragonfly now indexes `bool` as a TAG and applies the
/// `@flag:{true}` prefilter (previously KNN-only).
#[test]
fn test_binary_dragonfly_bool() {
    run_dragonfly_filter_test("dragonfly-bool", "bool-test", common::write_bool_project);
}

/// UUID exact-match (TAG).
#[test]
fn test_binary_dragonfly_uuid() {
    run_dragonfly_filter_test("dragonfly-uuid", "uuid-test", common::write_uuid_project);
}

/// Datetime range — stored as NUMERIC epoch seconds, filtered with an ISO range.
#[test]
fn test_binary_dragonfly_datetime() {
    run_dragonfly_filter_test("dragonfly-dt", "dt-test", common::write_datetime_project);
}

/// Full-text — a `text` field indexed as TEXT, `@body:quick` prefilter.
#[test]
fn test_binary_dragonfly_fulltext() {
    run_dragonfly_filter_test(
        "dragonfly-text",
        "text-test",
        common::write_fulltext_project,
    );
}

// NOTE: geo-radius is intentionally NOT tested — Dragonfly Search's geo-query
// parser rejects the `$param` placeholders the shared RediSearch filter builder
// emits (verified live), so geo filtering is unsupported here (like Chroma/Milvus).

/// Multi-condition AND (keyword AND numeric range).
#[test]
fn test_binary_dragonfly_and_filter() {
    run_dragonfly_filter_test(
        "dragonfly-and",
        "and-test",
        common::write_and_filter_project,
    );
}

/// Multi-condition OR (union).
#[test]
fn test_binary_dragonfly_or_filter() {
    run_dragonfly_filter_test("dragonfly-or", "or-test", common::write_or_filter_project);
}

/// Nested boolean `(A AND B) OR (C AND D)`.
#[test]
fn test_binary_dragonfly_nested_filter() {
    run_dragonfly_filter_test(
        "dragonfly-nested",
        "nested-test",
        common::write_nested_filter_project,
    );
}

/// A `"algorithm":"flat"` config must run end-to-end: EF_RUNTIME is HNSW-only,
/// so the engine must NOT emit it (or bind the EF param) for a FLAT index — else
/// every FT.SEARCH is a syntax error and the run fails with "No searches
/// completed". FLAT is exact, so recall should be ~1.0. Guards Fix 3 (HNSW gating
/// of EF_RUNTIME / EF param / PARAMS count).
#[test]
fn test_dragonfly_flat_algorithm_works() {
    wait_for_dragonfly();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let n_docs = 1000;
    let n_queries = 50;
    let top = 10;
    let (vectors, queries, neighbors) = make_data(n_docs, n_queries, dim, top);

    let engine_config = serde_json::json!([{
        "name": "dragonfly-flat",
        "engine": "dragonfly",
        "algorithm": "flat",
        "connection_params": {},
        "collection_params": {},
        "search_params": [
            { "parallel": 1, "top": top, "search_params": { "ef": 256 } }
        ],
        "upload_params": { "parallel": 1, "batch_size": 64 }
    }]);

    let root = create_knn_project(
        "dragonfly-flat",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        dim,
    );

    let port = test_port().to_string();
    let ok = common::run_binary(
        &root,
        "dragonfly-flat",
        "dragonfly-flat",
        TEST_HOST,
        &[("DRAGONFLY_PORT", port.as_str())],
    );
    assert!(
        ok,
        "FLAT-algorithm binary run failed (EF_RUNTIME not gated?)"
    );

    let recall = common::read_recall(&root, "dragonfly-flat");
    println!("dragonfly FLAT recall = {:.3}", recall);
    assert!(recall >= 0.9, "dragonfly FLAT recall {:.3} < 0.9", recall);
}

/// Direct FT.CREATE + HSET + FT.SEARCH sanity check: the query vector must be
/// its own nearest neighbour. Guards the FLOAT32 encoding + KNN wiring without
/// going through the binary.
#[test]
fn test_dragonfly_knn_self_neighbor() {
    wait_for_dragonfly();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 200;
    let (vectors, _q, _n) = make_data(count, 1, dim, 1);

    let _: () = redis::cmd("FT.CREATE")
        .arg("idx")
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg("1")
        .arg("")
        .arg("SCHEMA")
        .arg("vector")
        .arg("VECTOR")
        .arg("HNSW")
        .arg(10)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        .arg("M")
        .arg(16)
        .arg("EF_CONSTRUCTION")
        .arg(128)
        .query(&mut conn)
        .expect("FT.CREATE");

    for (i, v) in vectors.iter().enumerate() {
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let _: () = redis::cmd("HSET")
            .arg(i.to_string())
            .arg("vector")
            .arg(&bytes[..])
            .query(&mut conn)
            .expect("HSET");
    }

    let query_bytes: Vec<u8> = vectors[0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN 5 @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(5)
        .arg("PARAMS")
        .arg(4)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("EF")
        .arg("64")
        .arg("DIALECT")
        .arg(2)
        .query(&mut conn)
        .expect("FT.SEARCH");

    assert!(!response.is_empty(), "expected search results");
    let top_id = match &response[1] {
        redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
        redis::Value::SimpleString(s) => s.clone(),
        other => panic!("unexpected id value: {:?}", other),
    };
    assert_eq!(
        top_id, "0",
        "query vector should be its own nearest neighbor"
    );

    let _ = redis::cmd("FT.DROPINDEX").arg("idx").query::<()>(&mut conn);
}

// ---------------------------------------------------------------------------
// EF_RUNTIME behaviour: parsing + effect on recall
// ---------------------------------------------------------------------------

/// Create an HNSW COSINE index named `idx` and HSET every vector (FLOAT32 LE).
fn build_hnsw_index(conn: &mut Connection, vectors: &[Vec<f32>], dim: usize, m: i64, efc: i64) {
    let _: () = redis::cmd("FT.CREATE")
        .arg("idx")
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg("1")
        .arg("")
        .arg("SCHEMA")
        .arg("vector")
        .arg("VECTOR")
        .arg("HNSW")
        .arg(10)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        .arg("M")
        .arg(m)
        .arg("EF_CONSTRUCTION")
        .arg(efc)
        .query(conn)
        .expect("FT.CREATE");

    for (i, v) in vectors.iter().enumerate() {
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let _: () = redis::cmd("HSET")
            .arg(i.to_string())
            .arg("vector")
            .arg(&bytes[..])
            .query(conn)
            .expect("HSET");
    }
}

/// Run an HNSW KNN query with an explicit `EF_RUNTIME` and return the result ids
/// in rank order (parses the RESP2 `[count, id, fields, ...]` shape).
fn knn_ids_at_ef(conn: &mut Connection, query: &[f32], k: usize, ef: i64) -> Vec<i64> {
    let bytes: Vec<u8> = query.iter().flat_map(|f| f.to_le_bytes()).collect();
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(k)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("PARAMS")
        .arg(6)
        .arg("vec_param")
        .arg(&bytes[..])
        .arg("K")
        .arg(k.to_string())
        .arg("EF")
        .arg(ef.to_string())
        .arg("DIALECT")
        .arg(2)
        .query(conn)
        .expect("FT.SEARCH");

    // [count, id, [fields], id, [fields], ...] — ids sit at odd indices.
    let mut ids = Vec::new();
    let mut i = 1;
    while i < response.len() {
        let id = match &response[i] {
            redis::Value::BulkString(s) => String::from_utf8_lossy(s).parse::<i64>().ok(),
            redis::Value::SimpleString(s) => s.parse::<i64>().ok(),
            _ => None,
        };
        if let Some(id) = id {
            ids.push(id);
        }
        i += 2;
    }
    ids
}

/// Mean recall@k over `queries` at a fixed `EF_RUNTIME` against brute-force
/// cosine ground truth.
fn mean_recall_at_ef(
    conn: &mut Connection,
    queries: &[Vec<f32>],
    neighbors: &[Vec<i64>],
    k: usize,
    ef: i64,
) -> f64 {
    let mut total = 0.0f64;
    for (q, gt) in queries.iter().zip(neighbors.iter()) {
        let got = knn_ids_at_ef(conn, q, k, ef);
        let hits = got.iter().filter(|id| gt.contains(id)).count();
        total += hits as f64 / k as f64;
    }
    total / queries.len() as f64
}

/// The headline EF_RUNTIME proof: on a harder corpus, a SMALL `ef` must yield
/// materially LOWER recall than a LARGE `ef` — against the SAME graph, so only
/// the per-query `EF_RUNTIME` differs. A dropped/ignored EF_RUNTIME would make
/// both runs identical (gap ~ 0); a real one moves recall. Guards the engine's
/// headline feature with teeth the tiny/trivial fixtures lack.
#[test]
fn test_dragonfly_ef_runtime_recall_gap() {
    wait_for_dragonfly();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    // Harder corpus: enough docs + dimensionality that low-ef HNSW is imperfect.
    let dim = 48;
    let n_docs = 10_000;
    let n_queries = 100;
    let top = 10;
    let (m, efc) = (6, 24); // small M / ef_construction => a sparser graph

    let (vectors, queries, neighbors) = make_data(n_docs, n_queries, dim, top);
    build_hnsw_index(&mut conn, &vectors, dim, m, efc);

    let ef_low = 16;
    let ef_high = 512;
    let recall_low = mean_recall_at_ef(&mut conn, &queries, &neighbors, top, ef_low);
    let recall_high = mean_recall_at_ef(&mut conn, &queries, &neighbors, top, ef_high);
    let gap = recall_high - recall_low;
    println!(
        "dragonfly EF_RUNTIME recall gap: ef={} -> {:.3}, ef={} -> {:.3} (gap {:.3})",
        ef_low, recall_low, ef_high, recall_high, gap
    );

    assert!(
        recall_high >= recall_low,
        "higher ef must not reduce recall (low={:.3}, high={:.3})",
        recall_low,
        recall_high
    );
    assert!(
        recall_low < 0.99,
        "corpus not hard enough: low-ef recall {:.3} is already ~1.0, gap has no teeth",
        recall_low
    );
    assert!(
        gap > 0.02,
        "EF_RUNTIME had no effect: recall gap {:.3} <= 0.02 (high-ef must beat low-ef)",
        gap
    );
    assert!(
        recall_high >= 0.9,
        "high-ef recall {:.3} < 0.9 (graph quality regression?)",
        recall_high
    );

    let _ = redis::cmd("FT.DROPINDEX").arg("idx").query::<()>(&mut conn);
}

/// EF_RUNTIME is genuinely parsed/bound, not ignored: an FT.SEARCH whose `$EF`
/// param is non-numeric MUST error. (If Dragonfly silently ignored EF_RUNTIME,
/// this would succeed — a false green.)
#[test]
fn test_dragonfly_non_numeric_ef_rejected() {
    wait_for_dragonfly();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let (vectors, _q, _n) = make_data(100, 1, dim, 1);
    build_hnsw_index(&mut conn, &vectors, dim, 16, 128);

    let query_bytes: Vec<u8> = vectors[0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let res: Result<redis::Value, _> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("LIMIT")
        .arg(0)
        .arg(5)
        .arg("PARAMS")
        .arg(6)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg("5")
        .arg("EF")
        .arg("not_a_number")
        .arg("DIALECT")
        .arg(2)
        .query(&mut conn);

    assert!(
        res.is_err(),
        "non-numeric EF_RUNTIME must be rejected (proves EF_RUNTIME is parsed, not ignored); got {:?}",
        res
    );

    let _ = redis::cmd("FT.DROPINDEX").arg("idx").query::<()>(&mut conn);
}

/// Count keys under a glob (KEYS is fine on the small test keyspace).
fn count_keys(conn: &mut Connection, pattern: &str) -> usize {
    redis::cmd("KEYS")
        .arg(pattern)
        .query::<Vec<String>>(conn)
        .map(|k| k.len())
        .unwrap_or(0)
}

/// Delete only `*-search-*.json` result files under `root/results`.
fn delete_search_result_files(root: &std::path::Path) {
    let dir = root.join("results");
    if let Ok(rd) = fs::read_dir(&dir) {
        for entry in rd.filter_map(|e| e.ok()) {
            if entry.file_name().to_string_lossy().contains("-search-") {
                fs::remove_file(entry.path()).ok();
            }
        }
    }
}

/// #151-4 regression (dragonfly mirror): "upload all, then --skip-upload search
/// each" gives every config its OWN graph. Two configs (dense high-ef vs sparse
/// low-ef) coexist on one server via disjoint `idx:<config>` indexes + `<config>:`
/// keyspaces; pre-fix they shared `idx` + keyspace → identical recall on the sweep.
#[test]
fn test_dragonfly_coexistence_skip_upload() {
    wait_for_dragonfly();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let n_docs = 2000;
    let n_queries = 100;
    let top = 10;
    let (vectors, queries, neighbors) = make_data(n_docs, n_queries, dim, top);

    let engine_config = serde_json::json!([
        {
            "name": "dragonfly-co-a",
            "engine": "dragonfly",
            "connection_params": {},
            "collection_params": { "hnsw_config": { "M": 64, "EF_CONSTRUCTION": 200 } },
            "search_params": [{ "parallel": 1, "top": top, "search_params": { "ef": 256 } }],
            "upload_params": { "parallel": 1, "batch_size": 64 }
        },
        {
            "name": "dragonfly-co-b",
            "engine": "dragonfly",
            "connection_params": {},
            "collection_params": { "hnsw_config": { "M": 4, "EF_CONSTRUCTION": 8 } },
            "search_params": [{ "parallel": 1, "top": top, "search_params": { "ef": 10 } }],
            "upload_params": { "parallel": 1, "batch_size": 64 }
        }
    ]);

    let root = create_knn_project(
        "dragonfly-co",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        dim,
    );
    let port = test_port().to_string();

    // Phase 1: upload + search BOTH, KEEPING data for the skip-upload phase.
    assert!(
        common::run_binary_extra(
            &root,
            "dragonfly-co-*",
            "dragonfly-co",
            TEST_HOST,
            &[("DRAGONFLY_PORT", port.as_str())],
            &["--keep-data"],
        ),
        "dragonfly coexistence phase 1 failed"
    );

    let base_a = common::read_recall(&root, "dragonfly-co-a");
    let base_b = common::read_recall(&root, "dragonfly-co-b");

    // Deterministic coexistence: `n_docs` keys under EACH per-config prefix, and
    // both disjoint indexes exist.
    assert_eq!(
        count_keys(&mut conn, "dragonfly-co-a:*"),
        n_docs,
        "dragonfly-co-a keyspace"
    );
    assert_eq!(
        count_keys(&mut conn, "dragonfly-co-b:*"),
        n_docs,
        "dragonfly-co-b keyspace"
    );
    assert!(
        redis::cmd("FT.INFO")
            .arg("idx:dragonfly-co-a")
            .query::<redis::Value>(&mut conn)
            .is_ok(),
        "idx:dragonfly-co-a must exist"
    );
    assert!(
        redis::cmd("FT.INFO")
            .arg("idx:dragonfly-co-b")
            .query::<redis::Value>(&mut conn)
            .is_ok(),
        "idx:dragonfly-co-b must exist"
    );

    delete_search_result_files(&root);

    // Phase 2: --skip-upload search of both against the coexisting indexes.
    assert!(
        common::run_binary_extra(
            &root,
            "dragonfly-co-*",
            "dragonfly-co",
            TEST_HOST,
            &[("DRAGONFLY_PORT", port.as_str())],
            &["--skip-upload", "--keep-data"],
        ),
        "dragonfly coexistence phase 2 (--skip-upload) failed"
    );

    let rec_a = common::read_recall(&root, "dragonfly-co-a");
    let rec_b = common::read_recall(&root, "dragonfly-co-b");

    assert!(
        (rec_a - base_a).abs() < 1e-9,
        "dragonfly-co-a skip-upload recall {} != baseline {}",
        rec_a,
        base_a
    );
    assert!(
        (rec_b - base_b).abs() < 1e-9,
        "dragonfly-co-b skip-upload recall {} != baseline {}",
        rec_b,
        base_b
    );
    assert!(
        (rec_a - rec_b).abs() > 1e-9,
        "coexisting configs must have distinct recall: a={} b={}",
        rec_a,
        rec_b
    );
    assert!(
        rec_a > rec_b,
        "dense high-ef graph (a={}) should out-recall the sparse one (b={})",
        rec_a,
        rec_b
    );

    fs::remove_dir_all(&root).ok();
}

/// #151-4 negative (dragonfly mirror): `--skip-upload` with NO prior upload must
/// FAIL LOUDLY (index-existence guard), never writing a recall-0.0 result file.
#[test]
fn test_dragonfly_skip_upload_without_prior_upload_errors() {
    wait_for_dragonfly();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let n_docs = 500;
    let n_queries = 20;
    let top = 10;
    let (vectors, queries, neighbors) = make_data(n_docs, n_queries, dim, top);

    let engine_config = serde_json::json!([{
        "name": "dragonfly-noupload",
        "engine": "dragonfly",
        "connection_params": {},
        "collection_params": { "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 } },
        "search_params": [{ "parallel": 1, "top": top, "search_params": { "ef": 64 } }],
        "upload_params": { "parallel": 1, "batch_size": 64 }
    }]);

    let root = create_knn_project(
        "dragonfly-noupload",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        dim,
    );
    let port = test_port().to_string();

    let ok = common::run_binary_extra(
        &root,
        "dragonfly-noupload",
        "dragonfly-noupload",
        TEST_HOST,
        &[("DRAGONFLY_PORT", port.as_str())],
        &["--skip-upload"],
    );
    assert!(
        !ok,
        "--skip-upload with no prior upload must fail loudly, but exited 0"
    );
    let wrote_search = fs::read_dir(root.join("results"))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .any(|e| e.file_name().to_string_lossy().contains("-search-"))
        })
        .unwrap_or(false);
    assert!(
        !wrote_search,
        "guard must prevent any search result file from being written"
    );

    fs::remove_dir_all(&root).ok();
}
