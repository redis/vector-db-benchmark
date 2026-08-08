//! Integration tests for the Elasticsearch engine.
//!
//! Requires Elasticsearch 8.10.2 running on port 9201.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d
//! Run with:   cargo test --test integration_elasticsearch -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ES_PORT: u16 = 9201;
const ES_HOST: &str = "127.0.0.1";
const ES_INDEX: &str = "bench_test";

fn es_base_url() -> String {
    format!("http://{}:{}", ES_HOST, ES_PORT)
}

fn es_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .danger_accept_invalid_certs(true)
        .build()
        .expect("Failed to create HTTP client")
}

fn wait_for_elasticsearch() {
    let client = es_client();
    let url = format!("{}/_cluster/health", es_base_url());
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Ok(resp) = client.get(&url).send() {
            if resp.status().is_success() {
                return;
            }
        }
        if Instant::now() > deadline {
            panic!("Elasticsearch not available on port {} after 60s", ES_PORT);
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn delete_test_index() {
    let client = es_client();
    let url = format!("{}/{}", es_base_url(), ES_INDEX);
    let _ = client.delete(&url).send();
}

fn generate_test_vectors(count: usize, dim: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let ids: Vec<i64> = (0..count as i64).collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (ids, vectors)
}

fn id_to_uuid_hex(id: i64) -> String {
    uuid::Uuid::from_u128(id as u128).as_simple().to_string()
}

fn get_index_doc_count() -> usize {
    let client = es_client();
    // Refresh first to make sure all docs are searchable
    let _ = client
        .post(format!("{}/{}/_refresh", es_base_url(), ES_INDEX))
        .send();
    let url = format!("{}/{}/_count", es_base_url(), ES_INDEX);
    let resp = client.get(&url).send().expect("Failed to get doc count");
    let body: serde_json::Value = resp.json().expect("Failed to parse count response");
    body.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize
}

fn get_index_settings() -> serde_json::Value {
    let client = es_client();
    let url = format!("{}/{}", es_base_url(), ES_INDEX);
    let resp = client.get(&url).send().expect("Failed to get index");
    resp.json().expect("Failed to parse index response")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_es_create_index_l2() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();
    let url = format!("{}/{}", es_base_url(), ES_INDEX);

    let body = serde_json::json!({
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                // Mirrors the engine's production settings (#240): no periodic
                // refresh during ingest, matching OpenSearch and upstream.
                "refresh_interval": -1
            }
        },
        "mappings": {
            "_source": { "excludes": ["vector"] },
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": 4,
                    "index": true,
                    "similarity": "l2_norm",
                    "index_options": {
                        "type": "hnsw",
                        "m": 16,
                        "ef_construction": 100
                    }
                }
            }
        }
    });

    let resp = client
        .put(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .expect("Failed to create index");

    assert!(
        resp.status().is_success(),
        "Index creation failed: {}",
        resp.text().unwrap_or_default()
    );

    // Verify index exists
    let settings = get_index_settings();
    assert!(settings.get(ES_INDEX).is_some(), "Index should exist");

    delete_test_index();
}

#[test]
fn test_es_create_index_cosine() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();
    let url = format!("{}/{}", es_base_url(), ES_INDEX);

    let body = serde_json::json!({
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": 8,
                    "index": true,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "hnsw",
                        "m": 32,
                        "ef_construction": 256
                    }
                }
            }
        }
    });

    let resp = client
        .put(&url)
        .json(&body)
        .send()
        .expect("Failed to create index");
    assert!(resp.status().is_success());

    delete_test_index();
}

#[test]
fn test_es_bulk_upload_and_count() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();

    // Create index
    let create_body = serde_json::json!({
        "settings": { "index": { "number_of_shards": 1, "number_of_replicas": 0 } },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": 4,
                    "index": true,
                    "similarity": "l2_norm",
                    "index_options": { "type": "hnsw", "m": 16, "ef_construction": 100 }
                }
            }
        }
    });
    let resp = client
        .put(format!("{}/{}", es_base_url(), ES_INDEX))
        .json(&create_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Bulk upload 20 vectors
    let (ids, vectors) = generate_test_vectors(20, 4);
    let mut bulk_body = String::new();
    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        bulk_body.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let vec_json: Vec<String> = vectors[i].iter().map(|f| f.to_string()).collect();
        bulk_body.push_str(&format!("{{\"vector\":[{}]}}\n", vec_json.join(",")));
    }

    let resp = client
        .post(format!("{}/{}/_bulk", es_base_url(), ES_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(bulk_body)
        .send()
        .expect("Bulk upload failed");
    assert!(resp.status().is_success());

    let resp_body: serde_json::Value = resp.json().unwrap();
    assert_eq!(
        resp_body.get("errors").and_then(|v| v.as_bool()),
        Some(false),
        "Bulk upload should have no errors"
    );

    // Verify document count
    let count = get_index_doc_count();
    assert_eq!(count, 20, "Should have 20 documents");

    delete_test_index();
}

#[test]
fn test_es_uuid_id_format() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();

    // Create index
    let create_body = serde_json::json!({
        "settings": { "index": { "number_of_shards": 1, "number_of_replicas": 0 } },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": 2,
                    "index": true,
                    "similarity": "l2_norm",
                    "index_options": { "type": "hnsw", "m": 16, "ef_construction": 100 }
                }
            }
        }
    });
    client
        .put(format!("{}/{}", es_base_url(), ES_INDEX))
        .json(&create_body)
        .send()
        .unwrap();

    // Upload a single document with ID 42
    let uuid_hex = id_to_uuid_hex(42);
    let bulk_body = format!(
        "{{\"index\":{{\"_id\":\"{}\"}}}}\n{{\"vector\":[1.0, 2.0]}}\n",
        uuid_hex
    );
    client
        .post(format!("{}/{}/_bulk", es_base_url(), ES_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(bulk_body)
        .send()
        .unwrap();

    // Refresh and retrieve by UUID
    client
        .post(format!("{}/{}/_refresh", es_base_url(), ES_INDEX))
        .send()
        .unwrap();

    let resp = client
        .get(format!("{}/{}/_doc/{}", es_base_url(), ES_INDEX, uuid_hex))
        .send()
        .unwrap();
    assert!(
        resp.status().is_success(),
        "Document should be retrievable by UUID hex"
    );

    let doc: serde_json::Value = resp.json().unwrap();
    assert_eq!(
        doc.get("_id").and_then(|v| v.as_str()),
        Some(uuid_hex.as_str())
    );

    delete_test_index();
}

#[test]
fn test_es_delete_index() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();

    // Create index
    client
        .put(format!("{}/{}", es_base_url(), ES_INDEX))
        .json(&serde_json::json!({
            "mappings": {
                "properties": {
                    "vector": { "type": "dense_vector", "dims": 2, "index": true, "similarity": "l2_norm" }
                }
            }
        }))
        .send()
        .unwrap();

    // Delete it
    let resp = client
        .delete(format!("{}/{}", es_base_url(), ES_INDEX))
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Verify gone (should return 404)
    let resp = client
        .get(format!("{}/{}", es_base_url(), ES_INDEX))
        .send()
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}

#[test]
fn test_es_delete_nonexistent_index() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();

    // Delete non-existent index should return 404 (not crash)
    let resp = client
        .delete(format!("{}/{}", es_base_url(), ES_INDEX))
        .send()
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}

#[test]
fn test_es_force_merge() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();

    // Create index and upload some data
    client
        .put(format!("{}/{}", es_base_url(), ES_INDEX))
        .json(&serde_json::json!({
            "settings": { "index": { "number_of_shards": 1, "number_of_replicas": 0 } },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector", "dims": 4, "index": true,
                        "similarity": "l2_norm",
                        "index_options": { "type": "hnsw", "m": 16, "ef_construction": 100 }
                    }
                }
            }
        }))
        .send()
        .unwrap();

    // Upload a small batch
    let (ids, vectors) = generate_test_vectors(10, 4);
    let mut bulk_body = String::new();
    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        bulk_body.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let vec_json: Vec<String> = vectors[i].iter().map(|f| f.to_string()).collect();
        bulk_body.push_str(&format!("{{\"vector\":[{}]}}\n", vec_json.join(",")));
    }
    client
        .post(format!("{}/{}/_bulk", es_base_url(), ES_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(bulk_body)
        .send()
        .unwrap();

    // Force merge
    let resp = client
        .post(format!(
            "{}/{}/_forcemerge?wait_for_completion=true&max_num_segments=1",
            es_base_url(),
            ES_INDEX
        ))
        .send()
        .unwrap();
    assert!(
        resp.status().is_success(),
        "Force merge failed: {}",
        resp.text().unwrap_or_default()
    );

    // Verify data is still there after merge
    let count = get_index_doc_count();
    assert_eq!(count, 10, "Should still have 10 documents after merge");

    delete_test_index();
}

#[test]
fn test_es_schema_field_mapping() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();

    // Create index with schema fields (int->long, geo->geo_point)
    let body = serde_json::json!({
        "settings": { "index": { "number_of_shards": 1, "number_of_replicas": 0 } },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector", "dims": 4, "index": true,
                    "similarity": "l2_norm",
                    "index_options": { "type": "hnsw", "m": 16, "ef_construction": 100 }
                },
                "price": { "type": "long", "index": true },
                "location": { "type": "geo_point", "index": true }
            }
        }
    });

    let resp = client
        .put(format!("{}/{}", es_base_url(), ES_INDEX))
        .json(&body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Verify mappings
    let resp = client
        .get(format!("{}/{}/_mapping", es_base_url(), ES_INDEX))
        .send()
        .unwrap();
    let mappings: serde_json::Value = resp.json().unwrap();
    let props = &mappings[ES_INDEX]["mappings"]["properties"];

    assert_eq!(props["price"]["type"], "long");
    assert_eq!(props["location"]["type"], "geo_point");
    assert_eq!(props["vector"]["type"], "dense_vector");

    delete_test_index();
}

// ---------------------------------------------------------------------------
// KNN Search Tests (Stories 005, 006, 007)
// ---------------------------------------------------------------------------

fn uuid_hex_to_int(hex: &str) -> i64 {
    uuid::Uuid::parse_str(hex).unwrap().as_u128() as i64
}

fn create_test_index_with_vectors(
    client: &reqwest::blocking::Client,
    dim: usize,
    ids: &[i64],
    vectors: &[Vec<f32>],
    similarity: &str,
) {
    // Create index
    let body = serde_json::json!({
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1"
            }
        },
        "mappings": {
            "_source": { "excludes": ["vector"] },
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": true,
                    "similarity": similarity,
                    "index_options": {
                        "type": "hnsw",
                        "m": 16,
                        "ef_construction": 200
                    }
                }
            }
        }
    });
    let resp = client
        .put(format!("{}/{}", es_base_url(), ES_INDEX))
        .json(&body)
        .send()
        .expect("Failed to create index");
    assert!(
        resp.status().is_success(),
        "Index creation failed: {}",
        resp.text().unwrap_or_default()
    );

    // Bulk upload
    let mut bulk_body = String::new();
    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        bulk_body.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({ "vector": vectors[i] });
        bulk_body.push_str(&serde_json::to_string(&doc).unwrap());
        bulk_body.push('\n');
    }
    let resp = client
        .post(format!("{}/{}/_bulk", es_base_url(), ES_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(bulk_body)
        .send()
        .expect("Bulk upload failed");
    assert!(resp.status().is_success());

    // Refresh to make docs searchable
    client
        .post(format!("{}/{}/_refresh", es_base_url(), ES_INDEX))
        .send()
        .expect("Refresh failed");
}

fn es_knn_search(
    client: &reqwest::blocking::Client,
    query_vector: &[f32],
    top: usize,
    num_candidates: usize,
    filter: Option<serde_json::Value>,
) -> Vec<(i64, f64)> {
    let mut knn = serde_json::json!({
        "field": "vector",
        "query_vector": query_vector,
        "k": top,
        "num_candidates": num_candidates,
    });
    if let Some(f) = filter {
        knn.as_object_mut().unwrap().insert("filter".to_string(), f);
    }
    let body = serde_json::json!({ "knn": knn, "size": top });

    let resp = client
        .post(format!("{}/{}/_search", es_base_url(), ES_INDEX))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .expect("KNN search request failed");
    assert!(
        resp.status().is_success(),
        "KNN search failed: {}",
        resp.text().unwrap_or_default()
    );

    let resp_body: serde_json::Value = resp.json().unwrap();
    let hits = resp_body["hits"]["hits"]
        .as_array()
        .expect("Missing hits.hits");

    hits.iter()
        .map(|hit| {
            let id_hex = hit["_id"].as_str().unwrap();
            let score = hit["_score"].as_f64().unwrap_or(0.0);
            (uuid_hex_to_int(id_hex), score)
        })
        .collect()
}

/// Brute-force L2 nearest neighbors for precision validation.
fn brute_force_neighbors_l2(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
    let mut dists: Vec<(i64, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f64 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            (i as i64, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(top).map(|(id, _)| *id).collect()
}

#[test]
fn test_es_knn_search_basic() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();
    let dim = 4;
    let count = 50;
    let top = 5;
    let (ids, vectors) = generate_test_vectors(count, dim);

    create_test_index_with_vectors(&client, dim, &ids, &vectors, "l2_norm");

    // Search with the first vector as query — it should be its own nearest neighbor
    let results = es_knn_search(&client, &vectors[0], top, 50, None);

    assert!(!results.is_empty(), "KNN search should return results");
    assert_eq!(
        results[0].0, 0,
        "Query vector (id=0) should be its own nearest neighbor, got id={}",
        results[0].0
    );
    assert!(
        results.len() <= top,
        "Should return at most {} results",
        top
    );

    delete_test_index();
}

#[test]
fn test_es_knn_precision_exact() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();
    let dim = 8;
    let count = 100;
    let top = 10;
    let (ids, vectors) = generate_test_vectors(count, dim);

    create_test_index_with_vectors(&client, dim, &ids, &vectors, "l2_norm");

    // Pick a query vector and compute brute-force ground truth
    let query_idx = 42;
    let expected = brute_force_neighbors_l2(&vectors[query_idx], &vectors, top);

    // KNN search with high num_candidates for near-exact recall
    let results = es_knn_search(&client, &vectors[query_idx], top, count, None);
    let result_ids: Vec<i64> = results.iter().map(|(id, _)| *id).collect();

    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let found_set: std::collections::HashSet<i64> = result_ids.into_iter().collect();
    let hits = expected_set.intersection(&found_set).count();
    let precision = hits as f64 / top as f64;

    assert!(
        (precision - 1.0).abs() < f64::EPSILON,
        "Precision should be 1.0 for small exact dataset with high num_candidates, got {}",
        precision
    );

    delete_test_index();
}

#[test]
fn test_es_knn_search_cosine() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();
    let dim = 4;
    let (ids, vectors) = generate_test_vectors(20, dim);

    create_test_index_with_vectors(&client, dim, &ids, &vectors, "cosine");

    let results = es_knn_search(&client, &vectors[0], 5, 20, None);
    assert!(!results.is_empty(), "Cosine KNN should return results");
    assert_eq!(results[0].0, 0, "Self should be top-1 for cosine");

    delete_test_index();
}

#[test]
fn test_es_knn_search_with_match_filter() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();
    let dim = 4;

    // Create index with vector + category field
    let body = serde_json::json!({
        "settings": { "index": { "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1" } },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector", "dims": dim, "index": true,
                    "similarity": "l2_norm",
                    "index_options": { "type": "hnsw", "m": 16, "ef_construction": 200 }
                },
                "category": { "type": "keyword", "index": true }
            }
        }
    });
    client
        .put(format!("{}/{}", es_base_url(), ES_INDEX))
        .json(&body)
        .send()
        .unwrap();

    // Upload 20 vectors: even IDs get category "A", odd IDs get "B"
    let (ids, vectors) = generate_test_vectors(20, dim);
    let mut bulk_body = String::new();
    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        let category = if ids[i] % 2 == 0 { "A" } else { "B" };
        bulk_body.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({ "vector": vectors[i], "category": category });
        bulk_body.push_str(&serde_json::to_string(&doc).unwrap());
        bulk_body.push('\n');
    }
    client
        .post(format!("{}/{}/_bulk", es_base_url(), ES_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(bulk_body)
        .send()
        .unwrap();

    client
        .post(format!("{}/{}/_refresh", es_base_url(), ES_INDEX))
        .send()
        .unwrap();

    // Search with filter: only category "A" (even IDs)
    let filter = serde_json::json!({
        "bool": { "must": [{ "match": { "category": "A" } }] }
    });
    let results = es_knn_search(&client, &vectors[0], 10, 20, Some(filter));

    // All returned IDs should be even
    for (id, _) in &results {
        assert!(
            id % 2 == 0,
            "Filtered search should only return category A (even IDs), got id={}",
            id
        );
    }
    assert!(
        !results.is_empty(),
        "Filtered search should return at least one result"
    );

    delete_test_index();
}

#[test]
fn test_es_knn_search_with_range_filter() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();
    let dim = 4;

    // Create index with vector + price field
    let body = serde_json::json!({
        "settings": { "index": { "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1" } },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector", "dims": dim, "index": true,
                    "similarity": "l2_norm",
                    "index_options": { "type": "hnsw", "m": 16, "ef_construction": 200 }
                },
                "price": { "type": "long", "index": true }
            }
        }
    });
    client
        .put(format!("{}/{}", es_base_url(), ES_INDEX))
        .json(&body)
        .send()
        .unwrap();

    // Upload 20 vectors with price = id * 10
    let (ids, vectors) = generate_test_vectors(20, dim);
    let mut bulk_body = String::new();
    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        bulk_body.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({ "vector": vectors[i], "price": ids[i] * 10 });
        bulk_body.push_str(&serde_json::to_string(&doc).unwrap());
        bulk_body.push('\n');
    }
    client
        .post(format!("{}/{}/_bulk", es_base_url(), ES_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(bulk_body)
        .send()
        .unwrap();

    client
        .post(format!("{}/{}/_refresh", es_base_url(), ES_INDEX))
        .send()
        .unwrap();

    // Search with range filter: price >= 100 (ids >= 10)
    let filter = serde_json::json!({
        "bool": { "must": [{ "range": { "price": { "gte": 100 } } }] }
    });
    let results = es_knn_search(&client, &vectors[0], 10, 20, Some(filter));

    for (id, _) in &results {
        assert!(
            *id >= 10,
            "Range-filtered search should only return price >= 100 (id >= 10), got id={}",
            id
        );
    }
    assert!(
        !results.is_empty(),
        "Range-filtered search should return at least one result"
    );

    delete_test_index();
}

#[test]
fn test_es_knn_search_no_filter() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();
    let dim = 4;
    let (ids, vectors) = generate_test_vectors(10, dim);

    create_test_index_with_vectors(&client, dim, &ids, &vectors, "l2_norm");

    // Search without filter — should return results from entire dataset
    let results = es_knn_search(&client, &vectors[0], 5, 10, None);
    assert_eq!(
        results.len(),
        5,
        "Should return exactly top-5 from 10 vectors"
    );

    delete_test_index();
}

#[test]
fn test_es_full_cycle_configure_upload_search_delete() {
    wait_for_elasticsearch();
    delete_test_index();

    let client = es_client();
    let dim = 8;
    let count = 50;
    let top = 5;
    let (ids, vectors) = generate_test_vectors(count, dim);

    // 1. Configure (create index)
    create_test_index_with_vectors(&client, dim, &ids, &vectors, "l2_norm");

    // 2. Verify upload
    let doc_count = get_index_doc_count();
    assert_eq!(doc_count, count, "All vectors should be uploaded");

    // 3. Search
    let expected = brute_force_neighbors_l2(&vectors[0], &vectors, top);
    let results = es_knn_search(&client, &vectors[0], top, count, None);
    let result_ids: Vec<i64> = results.iter().map(|(id, _)| *id).collect();

    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let found_set: std::collections::HashSet<i64> = result_ids.into_iter().collect();
    let hits = expected_set.intersection(&found_set).count();
    let precision = hits as f64 / top as f64;
    assert!(
        (precision - 1.0).abs() < f64::EPSILON,
        "Full-cycle precision should be 1.0, got {}",
        precision
    );

    // 4. Delete
    let resp = client
        .delete(format!("{}/{}", es_base_url(), ES_INDEX))
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Verify deleted
    let resp = client
        .get(format!("{}/{}", es_base_url(), ES_INDEX))
        .send()
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}

/// End-to-end `match_any`: filter a keyword field to an OR-set and assert the
/// engine returns the filtered nearest neighbours (recall vs ground truth
/// brute-forced over only the matching docs). Proves the `terms` filter arm.
#[test]
fn test_binary_elasticsearch_match_any() {
    wait_for_elasticsearch();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "es-ma", "engine": "elasticsearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
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
            "es-ma",
            "match-any-test",
            "127.0.0.1",
            &[
                ("ELASTIC_PORT", "9201"),
                ("ELASTIC_INDEX", "bench_matchany")
            ],
        ),
        "es match_any run failed"
    );

    let recall = common::read_recall(&proj.root, "es-ma");
    println!("es match_any recall={:.3}", recall);
    assert!(recall >= 0.9, "es match_any recall {:.3} < 0.9", recall);
}

/// Bool-field equality filter end-to-end. Regression for the schema-type bug:
/// the canonical schema names the field "bool", which is NOT a valid ES type —
/// forwarding it verbatim made index creation reject the whole mapping. With
/// "bool" -> "boolean" ES coerces the reader's "true"/"false" string and the
/// `{flag:{match:{value:true}}}` filter selects the even ids.
#[test]
fn test_binary_elasticsearch_bool() {
    wait_for_elasticsearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "es-bool", "engine": "elasticsearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_bool_project("bool-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "es-bool",
            "bool-test",
            "127.0.0.1",
            &[("ELASTIC_PORT", "9201"), ("ELASTIC_INDEX", "bench_bool")],
        ),
        "es bool run failed"
    );
    let recall = common::read_recall(&proj.root, "es-bool");
    println!("es bool recall={:.3}", recall);
    assert!(recall >= 0.9, "es bool recall {:.3} < 0.9", recall);
}

/// Datetime range filter end-to-end. Regression for the schema-type bug:
/// "datetime" is not a valid ES type; with "datetime" -> "date" ES parses the
/// reader's ISO-8601 strings and the `{ts:{range:{gte,lt}}}` ISO bounds select
/// the [day 100, day 300) window.
#[test]
fn test_binary_elasticsearch_datetime() {
    wait_for_elasticsearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "es-dt", "engine": "elasticsearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_datetime_project("dt-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "es-dt",
            "dt-test",
            "127.0.0.1",
            &[("ELASTIC_PORT", "9201"), ("ELASTIC_INDEX", "bench_dt")],
        ),
        "es datetime run failed"
    );
    let recall = common::read_recall(&proj.root, "es-dt");
    println!("es datetime recall={:.3}", recall);
    assert!(recall >= 0.9, "es datetime recall {:.3} < 0.9", recall);
}

/// Geo-radius filter end-to-end (previously untested for ES). `geo` -> geo_point
/// mapping + `geo_distance` query; recall vs haversine ground truth.
#[test]
fn test_binary_elasticsearch_geo() {
    wait_for_elasticsearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "es-geo", "engine": "elasticsearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_geo_project("geo-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "es-geo",
            "geo-test",
            "127.0.0.1",
            &[("ELASTIC_PORT", "9201"), ("ELASTIC_INDEX", "bench_geo")],
        ),
        "es geo run failed"
    );
    let recall = common::read_recall(&proj.root, "es-geo");
    println!("es geo recall={:.3}", recall);
    assert!(recall >= 0.9, "es geo recall {:.3} < 0.9", recall);
}

/// Multi-condition AND (keyword match AND numeric range) — verifies ES composes
/// two clauses of different types into one `bool.must`.
#[test]
fn test_binary_elasticsearch_and_filter() {
    wait_for_elasticsearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "es-and", "engine": "elasticsearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_and_filter_project(
        "and-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "es-and",
            "and-test",
            "127.0.0.1",
            &[("ELASTIC_PORT", "9201"), ("ELASTIC_INDEX", "bench_and")],
        ),
        "es and-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "es-and");
    println!("es and-filter recall={:.3}", recall);
    assert!(recall >= 0.9, "es and-filter recall {:.3} < 0.9", recall);
}

/// Multi-condition OR (`color == "red" OR size >= 90`) — verifies ES unions two
/// clauses into a `bool.should` (minimum_should_match 1), searching the union.
#[test]
fn test_binary_elasticsearch_or_filter() {
    wait_for_elasticsearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "es-or", "engine": "elasticsearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_or_filter_project("or-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "es-or",
            "or-test",
            "127.0.0.1",
            &[("ELASTIC_PORT", "9201"), ("ELASTIC_INDEX", "bench_or")],
        ),
        "es or-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "es-or");
    println!("es or-filter recall={:.3}", recall);
    assert!(recall >= 0.9, "es or-filter recall {:.3} < 0.9", recall);
}

/// Nested/grouped boolean filter: `(color==red AND size>=50) OR (color==blue
/// AND size<10)`. Verifies ES nests each AND-group as its OWN `bool.must`
/// inside the parent `bool.should` (minimum_should_match 1) instead of
/// flattening the four leaves — a mis-flattened builder searches a wildly
/// different set and recall collapses, so recall >= 0.9 proves native nesting.
#[test]
fn test_binary_elasticsearch_nested_filter() {
    wait_for_elasticsearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "es-nested", "engine": "elasticsearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
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
            "es-nested",
            "nested-test",
            "127.0.0.1",
            &[("ELASTIC_PORT", "9201"), ("ELASTIC_INDEX", "bench_nested")],
        ),
        "es nested-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "es-nested");
    println!("es nested-filter recall={:.3}", recall);
    assert!(recall >= 0.9, "es nested-filter recall {:.3} < 0.9", recall);
}

/// Selectivity ladder: one `rank < K` range query per rung, sweeping filter
/// selectivity from ~3% to ~99% in a single dataset. Verifies ES's pre-filtered
/// kNN keeps recall across the whole selectivity range (recall vs per-rung
/// ground truth), not just at one operating point.
#[test]
fn test_binary_elasticsearch_selectivity() {
    wait_for_elasticsearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "es-sel", "engine": "elasticsearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_selectivity_project(
        "sel-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "es-sel",
            "sel-test",
            "127.0.0.1",
            &[("ELASTIC_PORT", "9201"), ("ELASTIC_INDEX", "bench_sel")],
        ),
        "es selectivity run failed"
    );
    let recall = common::read_recall(&proj.root, "es-sel");
    println!("es selectivity recall={:.3}", recall);
    assert!(recall >= 0.9, "es selectivity recall {:.3} < 0.9", recall);
}

/// UUID exact-match filter end-to-end. Regression for the schema-type bug:
/// "uuid" is not a valid ES type; forwarding it verbatim made index creation
/// reject the whole mapping (silently breaking the filter). With "uuid" ->
/// "keyword" ES does an exact term match, selecting the quarter of docs whose
/// `uid` equals UUIDS[0].
#[test]
fn test_binary_elasticsearch_uuid() {
    wait_for_elasticsearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "es-uuid", "engine": "elasticsearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_uuid_project("uuid-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "es-uuid",
            "uuid-test",
            "127.0.0.1",
            &[("ELASTIC_PORT", "9201"), ("ELASTIC_INDEX", "bench_uuid")],
        ),
        "es uuid run failed"
    );
    let recall = common::read_recall(&proj.root, "es-uuid");
    println!("es uuid recall={:.3}", recall);
    assert!(recall >= 0.9, "es uuid recall {:.3} < 0.9", recall);
}

/// End-to-end full-text filter (#120): the query carries a single
/// `{"body":{"match":{"text":"quick"}}}` condition and ground truth is
/// brute-forced over only the docs whose body CONTAINS "quick". Before the fix,
/// ES dropped the text clause and ran the kNN query UNFILTERED, so recall was
/// scored against the filtered ground truth and collapsed. A high recall here
/// proves the analyzed `match` filter arm is applied end-to-end.
#[test]
fn test_binary_elasticsearch_fulltext() {
    wait_for_elasticsearch();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "es-text", "engine": "elasticsearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_fulltext_project("text-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    assert!(
        common::run_binary(
            &proj.root,
            "es-text",
            "text-test",
            "127.0.0.1",
            &[
                ("ELASTIC_PORT", "9201"),
                ("ELASTIC_INDEX", "bench_fulltext")
            ],
        ),
        "es fulltext run failed"
    );

    let recall = common::read_recall(&proj.root, "es-text");
    println!("es fulltext recall={:.3}", recall);
    assert!(recall >= 0.9, "es fulltext recall {:.3} < 0.9", recall);
}
