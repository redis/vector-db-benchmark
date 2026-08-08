//! Integration tests for the Milvus engine.
//!
//! Requires Milvus (v2.6.19, per tests/docker-compose.test.yml) running on port
//! 19531 (standalone mode with etcd + minio).
//! Start with: docker compose -f tests/docker-compose.test.yml up -d milvus --wait
//! Run with:   MILVUS_PORT=19531 cargo test --test integration_milvus --release -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MILVUS_PORT: u16 = 19531;
const MILVUS_HOST: &str = "127.0.0.1";
const TEST_COLLECTION: &str = "bench_test";

fn milvus_base_url() -> String {
    let port: u16 = std::env::var("MILVUS_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MILVUS_PORT);
    format!("http://{}:{}", MILVUS_HOST, port)
}

fn http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
}

fn wait_for_milvus() {
    let client = http_client();
    let url = format!("{}/v2/vectordb/collections/list", milvus_base_url());
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if let Ok(resp) = client.post(&url).json(&serde_json::json!({})).send() {
            if resp.status().is_success() {
                let body: serde_json::Value = resp.json().unwrap_or_default();
                if body.get("code").and_then(|c| c.as_i64()).unwrap_or(-1) == 0 {
                    return;
                }
            }
        }
        if Instant::now() > deadline {
            panic!("Milvus not available on port {} after 120s", MILVUS_PORT);
        }
        thread::sleep(Duration::from_millis(1000));
    }
}

fn drop_test_collection() {
    drop_collection(TEST_COLLECTION);
}

/// Best-effort drop of a named collection (the index read-back tests run with
/// `--keep-data`, so they must clean up their own collection afterwards).
fn drop_collection(name: &str) {
    let client = http_client();
    let url = format!("{}/v2/vectordb/collections/drop", milvus_base_url());
    let _ = client
        .post(&url)
        .json(&serde_json::json!({"collectionName": name}))
        .send();
}

/// What the server reports for one index: its type, build state, and — crucially
/// — how many rows it actually covers.
#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexInfo {
    index_type: String,
    state: String,
    indexed_rows: i64,
    pending_rows: i64,
    total_rows: i64,
}

/// Read the collection's indexes back FROM THE SERVER, keyed by field name.
///
/// `indexes/list` returns index NAMES only, so each one is then described to
/// recover which field it covers, its type, and its row coverage. Reading this
/// back from the server (rather than trusting the create call's 200) is the
/// whole point: the create path already returned success while creating nothing
/// but the vector index (issue #218).
///
/// `indexedRows`/`totalRows` are load-bearing, not decoration. Before the
/// collection is flushed every index reports `state: "Finished"` with
/// `indexedRows = totalRows = 0` — "finished indexing nothing" — and queries
/// brute-force the growing segments regardless. An assertion that only checked
/// `state` would pass on a collection where no index covers a single row.
fn read_back_indexes(collection: &str) -> std::collections::HashMap<String, IndexInfo> {
    let client = http_client();
    let resp = client
        .post(format!("{}/v2/vectordb/indexes/list", milvus_base_url()))
        .json(&serde_json::json!({ "collectionName": collection }))
        .send()
        .expect("indexes/list request failed");
    let body: serde_json::Value = resp.json().expect("indexes/list not JSON");
    assert_eq!(
        body.get("code").and_then(|c| c.as_i64()),
        Some(0),
        "indexes/list failed for '{}': {:?}",
        collection,
        body
    );

    let mut out = std::collections::HashMap::new();
    for name in body
        .get("data")
        .and_then(|d| d.as_array())
        .cloned()
        .unwrap_or_default()
    {
        let name = name.as_str().unwrap_or_default();
        let resp = client
            .post(format!(
                "{}/v2/vectordb/indexes/describe",
                milvus_base_url()
            ))
            .json(&serde_json::json!({"collectionName": collection, "indexName": name}))
            .send()
            .expect("indexes/describe request failed");
        let body: serde_json::Value = resp.json().expect("indexes/describe not JSON");
        for row in body
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default()
        {
            let s = |k: &str| {
                row.get(k)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string()
            };
            let n = |k: &str| row.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
            out.insert(
                s("fieldName"),
                IndexInfo {
                    index_type: s("indexType"),
                    state: s("indexState"),
                    indexed_rows: n("indexedRows"),
                    pending_rows: n("pendingRows"),
                    total_rows: n("totalRows"),
                },
            );
        }
    }
    out
}

/// The scalar `indexType` the engine must create for a dataset schema type —
/// the test's INDEPENDENT copy of `milvus_field_kind`'s decision, so a silent
/// change to the engine's mapping fails here rather than passing by circularity.
/// `None` = the type is not materialised as a column (only `geo`, issue #223).
fn expected_index_type(field_name: &str, schema_type: &str) -> Option<&'static str> {
    if schema_type == "geo" {
        return None;
    }
    if field_name == "labels" && (schema_type == "keyword" || schema_type == "text") {
        return Some("INVERTED"); // Array(VarChar)
    }
    match schema_type {
        "bool" => Some("BITMAP"),
        "int" | "float" | "datetime" | "keyword" | "text" | "uuid" => Some("INVERTED"),
        _ => None,
    }
}

/// Assert one index is genuinely usable: built, nothing pending, and covering
/// EVERY row of the collection.
///
/// `state == "Finished"` on its own proves nothing — before the collection is
/// flushed, every index reports `Finished` with `indexedRows = totalRows = 0`,
/// so a state-only assertion passes while the server brute-forces the entire
/// corpus. That is the exact hole this check closes.
fn assert_index_covers_all_rows(collection: &str, field: &str, info: &IndexInfo) {
    assert_eq!(
        info.state, "Finished",
        "index on '{}' in '{}' is not built (state={})",
        field, collection, info.state
    );
    // NB: `pendingRows` is intentionally not asserted. On v2.6.19 a fully
    // indexed collection still reports a large `pendingRows` while background
    // compaction re-queues segments (measured: indexed = total = 1000000 with
    // pendingRows = 847872). Coverage, not the pending counter, is what says
    // the index is usable.
    assert!(
        info.total_rows > 0,
        "index on '{}' in '{}' covers ZERO rows (totalRows=0) — the collection was not \
         flushed, so every query brute-forces the growing segments and the index is inert",
        field,
        collection
    );
    assert_eq!(
        info.indexed_rows, info.total_rows,
        "index on '{}' in '{}' covers only {}/{} rows",
        field, collection, info.indexed_rows, info.total_rows
    );
}

/// **Issue #218 guard.** Assert that EVERY field of the dataset schema has a
/// scalar index on the server, of the expected type, built, and covering every
/// row — plus the `vector` ANN index.
///
/// This cannot be replaced by a recall assertion: an unindexed scalar column
/// returns exactly the same rows as an indexed one, so recall is identical and
/// only latency differs. That is precisely why a Milvus collection ran for this
/// long with no scalar index at all. The property must be read back from the
/// server's own index catalogue.
///
/// Row coverage is asserted too, not just existence and state: an index that
/// exists but covers zero rows is exactly as useless as no index, and Milvus
/// reports that state as `Finished` (see `assert_index_covers_all_rows`).
fn assert_all_schema_fields_indexed(collection: &str, schema: &serde_json::Value) {
    // Read the catalogue and drop the collection BEFORE asserting, so a failing
    // assertion cannot leak the collection into the shared test server.
    let found = read_back_indexes(collection);
    drop_collection(collection);
    println!("milvus index read-back for '{}': {:?}", collection, found);

    let vector = found.get("vector").unwrap_or_else(|| {
        panic!(
            "collection '{}' has no vector index: {:?}",
            collection, found
        )
    });
    assert_eq!(
        vector.index_type, "HNSW",
        "vector index type unexpected in '{}': {:?}",
        collection, found
    );
    // The ANN index is subject to the same zero-row trap as the scalar ones.
    assert_index_covers_all_rows(collection, "vector", vector);

    let mut expected_fields = std::collections::BTreeSet::new();
    for (field, ty) in schema.as_object().expect("schema must be an object") {
        let schema_type = ty.as_str().unwrap_or_default();
        let Some(expected) = expected_index_type(field, schema_type) else {
            continue;
        };
        expected_fields.insert(field.clone());
        let info = found.get(field).unwrap_or_else(|| {
            panic!(
                "schema field '{}' ({}) has NO index in collection '{}' — a filter on it is a \
                 brute-force scalar scan (issue #218). Indexes present: {:?}",
                field, schema_type, collection, found
            )
        });
        assert_eq!(
            info.index_type, expected,
            "schema field '{}' ({}) is indexed as {} in '{}', expected {}",
            field, schema_type, info.index_type, collection, expected
        );
        assert_index_covers_all_rows(collection, field, info);
    }
    assert!(
        !expected_fields.is_empty(),
        "fixture schema declared no indexable field — the assertion would be vacuous"
    );

    // Exact set equality, not just containment: an index on the server that the
    // caller's schema literal does not mention means this test's copy of the
    // fixture schema has drifted, and the "every field is indexed" claim would
    // be checked against the wrong field list.
    let actual_fields: std::collections::BTreeSet<String> = found
        .keys()
        .filter(|f| f.as_str() != "vector")
        .cloned()
        .collect();
    assert_eq!(
        actual_fields, expected_fields,
        "scalar indexes on '{}' do not match the declared schema",
        collection
    );
}

fn generate_test_vectors(count: usize, dim: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let ids: Vec<i64> = (0..count as i64).collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (ids, vectors)
}

fn create_collection(dim: usize, metric_type: &str) {
    let client = http_client();
    let body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "schema": {
            "fields": [
                {
                    "fieldName": "id",
                    "dataType": "Int64",
                    "isPrimary": true,
                },
                {
                    "fieldName": "vector",
                    "dataType": "FloatVector",
                    "elementTypeParams": {
                        "dim": dim.to_string(),
                    }
                }
            ],
            "enableDynamicField": false,
        }
    });

    let url = format!("{}/v2/vectordb/collections/create", milvus_base_url());
    let resp = client.post(&url).json(&body).send().unwrap();
    assert!(
        resp.status().is_success(),
        "Failed to create collection: {}",
        resp.text().unwrap_or_default()
    );

    // Create index
    let index_body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "indexParams": [{
            "fieldName": "vector",
            "indexName": "vector_index",
            "metricType": metric_type,
            "indexType": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 200,
            }
        }]
    });
    let url = format!("{}/v2/vectordb/indexes/create", milvus_base_url());
    let resp = client.post(&url).json(&index_body).send().unwrap();
    assert!(resp.status().is_success());

    // Load collection
    let load_body = serde_json::json!({"collectionName": TEST_COLLECTION});
    let url = format!("{}/v2/vectordb/collections/load", milvus_base_url());
    let _ = client.post(&url).json(&load_body).send();

    // Wait for load state
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let url = format!(
            "{}/v2/vectordb/collections/get_load_state",
            milvus_base_url()
        );
        if let Ok(resp) = client
            .post(&url)
            .json(&serde_json::json!({"collectionName": TEST_COLLECTION}))
            .send()
        {
            let body: serde_json::Value = resp.json().unwrap_or_default();
            if let Some(data) = body.get("data") {
                let state = data.get("loadState").and_then(|s| s.as_str()).unwrap_or("");
                if state == "LoadStateLoaded" {
                    break;
                }
            }
        }
        if Instant::now() > deadline {
            break;
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn insert_vectors(ids: &[i64], vectors: &[Vec<f32>]) {
    let client = http_client();

    let data: Vec<serde_json::Value> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(&id, vec)| {
            serde_json::json!({
                "id": id,
                "vector": vec,
            })
        })
        .collect();

    let body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "data": data,
    });

    let url = format!("{}/v2/vectordb/entities/insert", milvus_base_url());
    let resp = client.post(&url).json(&body).send().unwrap();
    assert!(
        resp.status().is_success(),
        "Insert failed: {}",
        resp.text().unwrap_or_default()
    );
    let resp_body: serde_json::Value = resp.json().unwrap();
    let code = resp_body.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
    assert_eq!(code, 0, "Insert error: {:?}", resp_body);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_milvus_collection_crud() {
    wait_for_milvus();
    drop_test_collection();

    let client = http_client();

    // Create
    create_collection(4, "L2");

    // Verify exists
    let url = format!("{}/v2/vectordb/collections/has", milvus_base_url());
    let resp = client
        .post(&url)
        .json(&serde_json::json!({"collectionName": TEST_COLLECTION}))
        .send()
        .unwrap();
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(
        body.get("data")
            .and_then(|d| d.get("has"))
            .and_then(|h| h.as_bool()),
        Some(true)
    );

    // Drop
    drop_test_collection();

    // Verify gone
    let resp = client
        .post(&url)
        .json(&serde_json::json!({"collectionName": TEST_COLLECTION}))
        .send()
        .unwrap();
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(
        body.get("data")
            .and_then(|d| d.get("has"))
            .and_then(|h| h.as_bool()),
        Some(false)
    );
}

#[test]
fn test_milvus_insert_and_search() {
    wait_for_milvus();
    drop_test_collection();
    create_collection(4, "L2");

    // Insert known vectors
    let ids = vec![0i64, 1, 2, 3, 4];
    let vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    insert_vectors(&ids, &vectors);

    // Wait for flush
    thread::sleep(Duration::from_secs(1));

    // Search for [1, 0, 0, 0]
    let client = http_client();
    let search_body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "data": [[1.0, 0.0, 0.0, 0.0]],
        "limit": 3,
        "outputFields": ["id"],
        "annsField": "vector",
    });
    let url = format!("{}/v2/vectordb/entities/search", milvus_base_url());
    let resp = client.post(&url).json(&search_body).send().unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();

    let code = body.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
    assert_eq!(code, 0, "Search error: {:?}", body);

    let data = body.get("data").and_then(|d| d.as_array()).unwrap();
    assert!(!data.is_empty(), "Expected search results");

    // First result should be id=0 (L2 distance = 0)
    let first_id = data[0].get("id").and_then(|v| v.as_i64()).unwrap();
    assert_eq!(first_id, 0, "First result should be exact match");

    drop_test_collection();
}

#[test]
fn test_milvus_precision_l2() {
    wait_for_milvus();
    drop_test_collection();

    let dim = 8;
    let n = 200;
    let k = 10;

    create_collection(dim, "L2");

    let (ids, vectors) = generate_test_vectors(n, dim);

    // Insert in batches to avoid too large payloads
    for chunk_start in (0..n).step_by(100) {
        let chunk_end = (chunk_start + 100).min(n);
        insert_vectors(
            &ids[chunk_start..chunk_end],
            &vectors[chunk_start..chunk_end],
        );
    }

    // Wait for flush
    thread::sleep(Duration::from_secs(2));

    // Compute brute-force ground truth
    let query = &vectors[0];
    let mut distances: Vec<(i64, f64)> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(&id, v)| {
            let dist: f64 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
                .sum();
            (id, dist)
        })
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ground_truth: std::collections::HashSet<i64> =
        distances.iter().take(k).map(|(id, _)| *id).collect();

    // Search
    let client = http_client();
    let search_body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "data": [query],
        "limit": k,
        "outputFields": ["id"],
        "annsField": "vector",
        "searchParams": {
            "params": {"ef": 256},
        }
    });
    let url = format!("{}/v2/vectordb/entities/search", milvus_base_url());
    let resp = client.post(&url).json(&search_body).send().unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(body.get("code").and_then(|c| c.as_i64()), Some(0));

    let data = body.get("data").and_then(|d| d.as_array()).unwrap();
    let found: std::collections::HashSet<i64> = data
        .iter()
        .filter_map(|item| item.get("id").and_then(|v| v.as_i64()))
        .collect();

    let overlap = ground_truth.intersection(&found).count();
    let precision = overlap as f64 / k as f64;
    println!(
        "Milvus L2 precision@{}: {:.2} ({}/{})",
        k, precision, overlap, k
    );
    assert!(
        precision >= 0.8,
        "Expected precision >= 0.80, got {:.2}",
        precision
    );

    drop_test_collection();
}

#[test]
fn test_milvus_full_cycle() {
    wait_for_milvus();
    drop_test_collection();

    let dim = 4;

    // Create + index + load
    create_collection(dim, "L2");

    // Insert
    let (ids, vectors) = generate_test_vectors(20, dim);
    insert_vectors(&ids, &vectors);
    thread::sleep(Duration::from_secs(1));

    // Search
    let client = http_client();
    let search_body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "data": [vectors[0]],
        "limit": 5,
        "outputFields": ["id"],
        "annsField": "vector",
    });
    let url = format!("{}/v2/vectordb/entities/search", milvus_base_url());
    let resp = client.post(&url).json(&search_body).send().unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(body.get("code").and_then(|c| c.as_i64()), Some(0));
    let data = body.get("data").and_then(|d| d.as_array()).unwrap();
    assert_eq!(data.len(), 5);

    // Delete collection
    drop_test_collection();

    // Verify gone
    let url = format!("{}/v2/vectordb/collections/has", milvus_base_url());
    let resp = client
        .post(&url)
        .json(&serde_json::json!({"collectionName": TEST_COLLECTION}))
        .send()
        .unwrap();
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(
        body.get("data")
            .and_then(|d| d.get("has"))
            .and_then(|h| h.as_bool()),
        Some(false)
    );
}

/// End-to-end `match_any`: filter a keyword field to an OR-set and assert the
/// engine returns the filtered nearest neighbours (recall vs ground truth
/// brute-forced over only the matching docs). Proves the `in [...]` expr arm.
#[test]
fn test_binary_milvus_match_any() {
    wait_for_milvus();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-ma", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj = common::write_match_any_project(
        "match-any-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    assert!(
        common::run_binary(
            &proj.root,
            "milvus-ma",
            "match-any-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_matchany"),
            ],
        ),
        "milvus match_any run failed"
    );

    let recall = common::read_recall(&proj.root, "milvus-ma");
    println!("milvus match_any recall={:.3}", recall);
    assert!(recall >= 0.9, "milvus match_any recall {:.3} < 0.9", recall);
}

/// Bool-field equality filter end-to-end. Regression: `"bool"` hit the schema
/// `_ => continue` arm so no column was created, while the filter emitted a
/// native `flag == true` against the missing column. Now `bool` -> native Bool
/// column (upload converts the reader's "true"/"false" string to a JSON bool).
#[test]
fn test_binary_milvus_bool() {
    wait_for_milvus();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-bool", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj =
        common::write_bool_project("bool-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary_extra(
            &proj.root,
            "milvus-bool",
            "bool-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_bool")
            ],
            &["--keep-data"],
        ),
        "milvus bool run failed"
    );
    assert_all_schema_fields_indexed("bench_bool", &serde_json::json!({"flag": "bool"}));
    let recall = common::read_recall(&proj.root, "milvus-bool");
    println!("milvus bool recall={:.3}", recall);
    assert!(recall >= 0.9, "milvus bool recall {:.3} < 0.9", recall);
}

/// UUID exact-match filter end-to-end. Regression: `uuid` hit the schema map's
/// `_ => continue`, so no field was created and the filter silently broke. Now
/// `uuid` -> VarChar (no analyzer, exact match), and `uid == UUIDS[0]` selects
/// the quarter of docs it should.
#[test]
fn test_binary_milvus_uuid() {
    wait_for_milvus();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-uuid", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj =
        common::write_uuid_project("uuid-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary_extra(
            &proj.root,
            "milvus-uuid",
            "uuid-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_uuid")
            ],
            &["--keep-data"],
        ),
        "milvus uuid run failed"
    );
    assert_all_schema_fields_indexed("bench_uuid", &serde_json::json!({"uid": "uuid"}));
    let recall = common::read_recall(&proj.root, "milvus-uuid");
    println!("milvus uuid recall={:.3}", recall);
    assert!(recall >= 0.9, "milvus uuid recall {:.3} < 0.9", recall);
}

/// Multi-condition AND (keyword match AND numeric range) — verifies Milvus
/// composes two conditions of different types into one boolean-expr `&&`
/// (`color == "red" && size >= 50`), not just a single clause.
#[test]
fn test_binary_milvus_and_filter() {
    wait_for_milvus();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-and", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj = common::write_and_filter_project(
        "and-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary_extra(
            &proj.root,
            "milvus-and",
            "and-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_and")
            ],
            &["--keep-data"],
        ),
        "milvus and-filter run failed"
    );
    assert_all_schema_fields_indexed(
        "bench_and",
        &serde_json::json!({"color": "keyword", "size": "int"}),
    );
    let recall = common::read_recall(&proj.root, "milvus-and");
    println!("milvus and-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "milvus and-filter recall {:.3} < 0.9",
        recall
    );
}

/// Multi-condition OR (`color == "red" OR size >= 90`) — verifies Milvus unions
/// two clauses into a boolean-expr `||` (`color == "red" || size >= 90`).
#[test]
fn test_binary_milvus_or_filter() {
    wait_for_milvus();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-or", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj =
        common::write_or_filter_project("or-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "milvus-or",
            "or-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_or")
            ],
        ),
        "milvus or-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "milvus-or");
    println!("milvus or-filter recall={:.3}", recall);
    assert!(recall >= 0.9, "milvus or-filter recall {:.3} < 0.9", recall);
}

/// Datetime range filter end-to-end. Regression: `"datetime"` was dropped from
/// the schema and the range builder inlined the quoted ISO string. Now
/// `datetime` -> Int64 epoch column; upload and the range filter both convert
/// ISO-8601 to epoch seconds, so the `[day 100, day 300)` window is selected.
#[test]
fn test_binary_milvus_datetime() {
    wait_for_milvus();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-dt", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj =
        common::write_datetime_project("dt-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary_extra(
            &proj.root,
            "milvus-dt",
            "dt-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_dt")
            ],
            &["--keep-data"],
        ),
        "milvus datetime run failed"
    );
    assert_all_schema_fields_indexed("bench_dt", &serde_json::json!({"ts": "datetime"}));
    let recall = common::read_recall(&proj.root, "milvus-dt");
    println!("milvus datetime recall={:.3}", recall);
    assert!(recall >= 0.9, "milvus datetime recall {:.3} < 0.9", recall);
}

/// Full-text filter end-to-end. Regression: a `{match:{text}}` clause was dropped
/// (the match arm required `value`/`any`), so the search ran UNFILTERED. Now the
/// `text` VarChar column is created with enable_analyzer/enable_match and the
/// filter uses `TEXT_MATCH(body, 'quick')`, selecting docs containing the token.
#[test]
fn test_binary_milvus_fulltext() {
    wait_for_milvus();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-ft", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj =
        common::write_fulltext_project("ft-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary_extra(
            &proj.root,
            "milvus-ft",
            "ft-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_ft")
            ],
            &["--keep-data"],
        ),
        "milvus fulltext run failed"
    );
    assert_all_schema_fields_indexed("bench_ft", &serde_json::json!({"body": "text"}));
    let recall = common::read_recall(&proj.root, "milvus-ft");
    println!("milvus fulltext recall={:.3}", recall);
    assert!(recall >= 0.9, "milvus fulltext recall {:.3} < 0.9", recall);
}

/// Nested/grouped boolean filter end-to-end:
/// `(color == "red" && size >= 50) || (color == "blue" && size < 10)`. Verifies
/// the Milvus builder RECURSES into each `{and:[...]}` group and emits a
/// PARENTHESISED sub-expression combined by `||`, instead of mis-flattening the
/// nested tree (which drops the group clauses → returns every row → recall
/// collapses). A live recall >= 0.9 proves the grouped union is correct.
#[test]
fn test_binary_milvus_nested_filter() {
    wait_for_milvus();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-nested", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj = common::write_nested_filter_project(
        "nested-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "milvus-nested",
            "nested-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_nested")
            ],
        ),
        "milvus nested-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "milvus-nested");
    println!("milvus nested-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "milvus nested-filter recall {:.3} < 0.9",
        recall
    );
}

/// End-to-end `match_any` on a MULTI-VALUED keyword field (`labels`, #88).
/// Milvus stores it as an Array(VarChar) and filters with
/// `array_contains_any`; before the fix it was a comma-joined VarChar tested
/// with whole-string `in`, which cannot match a single element (recall ~0).
#[test]
fn test_binary_milvus_match_any_labels() {
    wait_for_milvus();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-mal", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj = common::write_match_any_labels_project(
        "match-any-labels-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
        common::GtMetric::L2,
    );
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    assert!(
        common::run_binary_extra(
            &proj.root,
            "milvus-mal",
            "match-any-labels-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_matchany_labels"),
            ],
            &["--keep-data"],
        ),
        "milvus match_any labels run failed"
    );

    assert_all_schema_fields_indexed(
        "bench_matchany_labels",
        &serde_json::json!({"labels": "keyword"}),
    );
    let recall = common::read_recall(&proj.root, "milvus-mal");
    println!("milvus match_any labels recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "milvus multi-valued labels match_any recall {:.3} < 0.9",
        recall
    );
}
