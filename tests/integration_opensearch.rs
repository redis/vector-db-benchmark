//! Integration tests for the OpenSearch engine.
//!
//! Requires OpenSearch 2.19.2 running on port 9202 (security disabled).
//! Start with: docker compose -f tests/docker-compose.test.yml up -d opensearch --wait
//! Run with:   OPENSEARCH_PORT=9202 cargo test --test integration_opensearch --release -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const OS_PORT: u16 = 9202;
const OS_HOST: &str = "127.0.0.1";
const OS_INDEX: &str = "bench_test";

fn os_base_url() -> String {
    let port: u16 = std::env::var("OPENSEARCH_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(OS_PORT);
    format!("http://{}:{}", OS_HOST, port)
}

fn os_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .danger_accept_invalid_certs(true)
        .build()
        .expect("Failed to create HTTP client")
}

fn wait_for_opensearch() {
    let client = os_client();
    let url = format!("{}/_cluster/health", os_base_url());
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Ok(resp) = client.get(&url).send() {
            if resp.status().is_success() {
                return;
            }
        }
        if Instant::now() > deadline {
            panic!("OpenSearch not available on port {} after 60s", OS_PORT);
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn delete_test_index() {
    let client = os_client();
    let url = format!("{}/{}", os_base_url(), OS_INDEX);
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
    let client = os_client();
    // Refresh first to make sure all docs are searchable
    let _ = client
        .post(format!("{}/{}/_refresh", os_base_url(), OS_INDEX))
        .send();
    let url = format!("{}/{}/_count", os_base_url(), OS_INDEX);
    let resp = client.get(&url).send().expect("Failed to get doc count");
    let body: serde_json::Value = resp.json().expect("Failed to parse count response");
    body.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_opensearch_create_knn_index() {
    wait_for_opensearch();
    delete_test_index();

    let client = os_client();
    let body = serde_json::json!({
        "settings": {
            "index": {
                "knn": true,
            }
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 4,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "l2",
                        "parameters": {
                            "m": 16,
                            "ef_construction": 100,
                        }
                    }
                }
            }
        }
    });

    let url = format!("{}/{}", os_base_url(), OS_INDEX);
    let resp = client
        .put(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .expect("Failed to create index");
    assert!(
        resp.status().is_success(),
        "Failed to create index: {}",
        resp.text().unwrap_or_default()
    );

    // Verify the index exists
    let resp = client
        .get(format!("{}/{}", os_base_url(), OS_INDEX))
        .send()
        .expect("Failed to get index");
    assert!(resp.status().is_success());

    delete_test_index();
}

#[test]
fn test_opensearch_bulk_upload() {
    wait_for_opensearch();
    delete_test_index();

    let client = os_client();
    // Create index
    let body = serde_json::json!({
        "settings": {"index": {"knn": true}},
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 4,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "l2",
                    }
                }
            }
        }
    });
    let resp = client
        .put(format!("{}/{}", os_base_url(), OS_INDEX))
        .json(&body)
        .send()
        .expect("Failed to create index");
    assert!(resp.status().is_success());

    // Upload vectors
    let (ids, vectors) = generate_test_vectors(50, 4);
    let mut ndjson = String::new();
    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        ndjson.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({"vector": vectors[i]});
        ndjson.push_str(&serde_json::to_string(&doc).unwrap());
        ndjson.push('\n');
    }

    let resp = client
        .post(format!("{}/{}/_bulk", os_base_url(), OS_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .expect("Failed to bulk upload");
    assert!(resp.status().is_success());
    let resp_body: serde_json::Value = resp.json().unwrap();
    assert!(!resp_body["errors"].as_bool().unwrap_or(true));

    // Verify count
    assert_eq!(get_index_doc_count(), 50);

    delete_test_index();
}

#[test]
fn test_opensearch_knn_search() {
    wait_for_opensearch();
    delete_test_index();

    let client = os_client();
    // Create index with cosine similarity
    let body = serde_json::json!({
        "settings": {"index": {"knn": true}},
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 4,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                    }
                }
            }
        }
    });
    let resp = client
        .put(format!("{}/{}", os_base_url(), OS_INDEX))
        .json(&body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Upload known vectors
    let vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];

    let mut ndjson = String::new();
    for (i, vec) in vectors.iter().enumerate() {
        let uuid_hex = id_to_uuid_hex(i as i64);
        ndjson.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({"vector": vec});
        ndjson.push_str(&serde_json::to_string(&doc).unwrap());
        ndjson.push('\n');
    }

    let resp = client
        .post(format!("{}/{}/_bulk", os_base_url(), OS_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Refresh
    let _ = client
        .post(format!("{}/{}/_refresh", os_base_url(), OS_INDEX))
        .send();

    // Search for vector closest to [1, 0, 0, 0]
    let search_body = serde_json::json!({
        "query": {
            "knn": {
                "vector": {
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "k": 3,
                }
            }
        },
        "size": 3,
    });

    let resp = client
        .post(format!("{}/{}/_search", os_base_url(), OS_INDEX))
        .json(&search_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();

    let hits = body["hits"]["hits"].as_array().unwrap();
    assert!(!hits.is_empty(), "Expected search results");

    // First result should be id=0 (exact match)
    let first_id = hits[0]["_id"].as_str().unwrap();
    let expected_id = id_to_uuid_hex(0);
    assert_eq!(
        first_id, expected_id,
        "First result should be the exact match vector"
    );

    delete_test_index();
}

#[test]
fn test_opensearch_precision_l2() {
    wait_for_opensearch();
    delete_test_index();

    let client = os_client();
    let dim = 8;
    let n = 200;
    let k = 10;

    // Create index
    let body = serde_json::json!({
        "settings": {"index": {"knn": true}},
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "l2",
                        "parameters": {"m": 32, "ef_construction": 200}
                    }
                }
            }
        }
    });
    let resp = client
        .put(format!("{}/{}", os_base_url(), OS_INDEX))
        .json(&body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Upload random vectors
    let (ids, vectors) = generate_test_vectors(n, dim);
    let mut ndjson = String::new();
    for i in 0..n {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        ndjson.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({"vector": vectors[i]});
        ndjson.push_str(&serde_json::to_string(&doc).unwrap());
        ndjson.push('\n');
    }
    let resp = client
        .post(format!("{}/{}/_bulk", os_base_url(), OS_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Refresh
    let _ = client
        .post(format!("{}/{}/_refresh", os_base_url(), OS_INDEX))
        .send();

    // Compute ground truth: brute-force L2 distances for query = vectors[0]
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

    // KNN search
    let search_body = serde_json::json!({
        "query": {
            "knn": {
                "vector": {
                    "vector": query,
                    "k": k,
                }
            }
        },
        "size": k,
    });
    let resp = client
        .post(format!("{}/{}/_search", os_base_url(), OS_INDEX))
        .json(&search_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    let hits = body["hits"]["hits"].as_array().unwrap();

    let found: std::collections::HashSet<i64> = hits
        .iter()
        .filter_map(|h| {
            let hex = h["_id"].as_str()?;
            let uuid = uuid::Uuid::parse_str(hex).ok()?;
            Some(uuid.as_u128() as i64)
        })
        .collect();

    let overlap = ground_truth.intersection(&found).count();
    let precision = overlap as f64 / k as f64;
    println!(
        "OpenSearch L2 precision@{}: {:.2} ({}/{})",
        k, precision, overlap, k
    );
    assert!(
        precision >= 0.8,
        "Expected precision >= 0.80, got {:.2}",
        precision
    );

    delete_test_index();
}

/// Regression test for qdrant/vector-db-benchmark#167: filtered kNN search must
/// use *efficient* filtering (filter pushed inside the `knn` clause) rather than
/// post-filtering (kNN wrapped in an outer `bool.must` + `filter`).
///
/// With a selective filter, post-filtering retrieves the `k` global nearest
/// neighbours first and only *then* drops those outside the filter, so it returns
/// far fewer than `k` matches and collapses precision. Efficient filtering
/// retrieves the `k` nearest neighbours *within* the filter. This test asserts the
/// efficient form scores high, and asserts the post-filtering form does measurably
/// worse on the same data — so the fix cannot silently regress.
#[test]
fn test_opensearch_filtered_precision() {
    wait_for_opensearch();
    delete_test_index();

    let client = os_client();
    let dim = 8;
    let n = 500;
    let k = 10;
    let num_categories = 5; // target category is ~20% of the corpus → selective
    let target_category = 3;

    // Create index with a knn_vector plus an integer `category` payload field.
    let body = serde_json::json!({
        "settings": {"index": {"knn": true}},
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "l2",
                        "parameters": {"m": 32, "ef_construction": 200}
                    }
                },
                "category": {"type": "integer"}
            }
        }
    });
    let resp = client
        .put(format!("{}/{}", os_base_url(), OS_INDEX))
        .json(&body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Upload random vectors, round-robin category assignment.
    let (ids, vectors) = generate_test_vectors(n, dim);
    let categories: Vec<i64> = (0..n as i64).map(|i| i % num_categories).collect();
    let mut ndjson = String::new();
    for i in 0..n {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        ndjson.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({"vector": vectors[i], "category": categories[i]});
        ndjson.push_str(&serde_json::to_string(&doc).unwrap());
        ndjson.push('\n');
    }
    let resp = client
        .post(format!("{}/{}/_bulk", os_base_url(), OS_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let _ = client
        .post(format!("{}/{}/_refresh", os_base_url(), OS_INDEX))
        .send();

    // Ground truth: k nearest by L2 *among docs in the target category*.
    let query = &vectors[0];
    let mut distances: Vec<(i64, f64)> = ids
        .iter()
        .zip(vectors.iter())
        .zip(categories.iter())
        .filter(|((_, _), &cat)| cat == target_category)
        .map(|((&id, v), _)| {
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
    assert_eq!(
        ground_truth.len(),
        k,
        "test setup: need >= k docs in category"
    );

    let filter = serde_json::json!({"bool": {"must": [{"match": {"category": target_category}}]}});

    // Helper: run a search body and return the set of returned ids.
    let run = |search_body: &serde_json::Value| -> Vec<(i64, i64)> {
        let resp = client
            .post(format!("{}/{}/_search", os_base_url(), OS_INDEX))
            .json(search_body)
            .send()
            .unwrap();
        assert!(resp.status().is_success());
        let body: serde_json::Value = resp.json().unwrap();
        body["hits"]["hits"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|h| {
                let hex = h["_id"].as_str()?;
                let id = uuid::Uuid::parse_str(hex).ok()?.as_u128() as i64;
                let cat = h["_source"]["category"].as_i64()?;
                Some((id, cat))
            })
            .collect()
    };

    // Efficient filtering (the form our engine now produces): filter inside knn.
    let efficient_body = serde_json::json!({
        "query": {"knn": {"vector": {"vector": query, "k": k, "filter": filter}}},
        "size": k,
    });
    let efficient_hits = run(&efficient_body);
    let efficient_found: std::collections::HashSet<i64> =
        efficient_hits.iter().map(|(id, _)| *id).collect();
    let efficient_overlap = ground_truth.intersection(&efficient_found).count();
    let efficient_precision = efficient_overlap as f64 / k as f64;

    // Post-filtering (the old buggy form): kNN wrapped in an outer bool + filter.
    let post_body = serde_json::json!({
        "query": {"bool": {
            "must": [{"knn": {"vector": {"vector": query, "k": k}}}],
            "filter": filter,
        }},
        "size": k,
    });
    let post_hits = run(&post_body);
    let post_found: std::collections::HashSet<i64> = post_hits.iter().map(|(id, _)| *id).collect();
    let post_precision = post_found.intersection(&ground_truth).count() as f64 / k as f64;

    println!(
        "OpenSearch filtered precision@{k}: efficient={:.2} ({} hits), post-filter={:.2} ({} hits)",
        efficient_precision,
        efficient_hits.len(),
        post_precision,
        post_hits.len(),
    );

    // Every returned doc must satisfy the filter.
    for (id, cat) in &efficient_hits {
        assert_eq!(
            *cat, target_category,
            "efficient filtering returned doc {id} outside target category"
        );
    }

    // Efficient filtering returns a full page of high-precision results ...
    assert_eq!(
        efficient_hits.len(),
        k,
        "efficient filtering should return k in-filter neighbours"
    );
    assert!(
        efficient_precision >= 0.8,
        "efficient filtering precision {:.2} < 0.80",
        efficient_precision
    );

    // ... and it must clearly beat the old post-filtering behaviour, which loses
    // most of its candidates to the selective filter.
    assert!(
        efficient_precision > post_precision,
        "efficient filtering ({:.2}) should beat post-filtering ({:.2})",
        efficient_precision,
        post_precision
    );

    delete_test_index();
}

#[test]
fn test_opensearch_full_cycle() {
    wait_for_opensearch();
    delete_test_index();

    let client = os_client();
    let dim = 4;

    // Create
    let body = serde_json::json!({
        "settings": {"index": {"knn": true}},
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "l2",
                    }
                }
            }
        }
    });
    let resp = client
        .put(format!("{}/{}", os_base_url(), OS_INDEX))
        .json(&body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Upload
    let (ids, vectors) = generate_test_vectors(20, dim);
    let mut ndjson = String::new();
    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        ndjson.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({"vector": vectors[i]});
        ndjson.push_str(&serde_json::to_string(&doc).unwrap());
        ndjson.push('\n');
    }
    let resp = client
        .post(format!("{}/{}/_bulk", os_base_url(), OS_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    assert_eq!(get_index_doc_count(), 20);

    // Search
    let _ = client
        .post(format!("{}/{}/_refresh", os_base_url(), OS_INDEX))
        .send();
    let search_body = serde_json::json!({
        "query": {
            "knn": {
                "vector": {
                    "vector": vectors[0],
                    "k": 5,
                }
            }
        },
        "size": 5,
    });
    let resp = client
        .post(format!("{}/{}/_search", os_base_url(), OS_INDEX))
        .json(&search_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    let hits = body["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 5);

    // Delete
    delete_test_index();
    let resp = client
        .get(format!("{}/{}", os_base_url(), OS_INDEX))
        .send()
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}

/// End-to-end `match_any`: filter a keyword field to an OR-set and assert the
/// engine returns the filtered nearest neighbours (recall vs ground truth
/// brute-forced over only the matching docs). Proves the `terms` filter arm.
#[test]
fn test_binary_opensearch_match_any() {
    wait_for_opensearch();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-ma", "engine": "opensearch",
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
            "os-ma",
            "match-any-test",
            // Explicit http:// scheme: the OpenSearch engine defaults to https
            // for a bare host, but the CI container runs plaintext http (security
            // plugin disabled).
            "http://127.0.0.1",
            &[
                ("OPENSEARCH_PORT", "9202"),
                ("OPENSEARCH_INDEX", "bench_matchany"),
            ],
        ),
        "opensearch match_any run failed"
    );

    let recall = common::read_recall(&proj.root, "os-ma");
    println!("opensearch match_any recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "opensearch match_any recall {:.3} < 0.9",
        recall
    );
}

/// Bool-field equality filter end-to-end. Regression for the schema-type bug:
/// "bool" is not a valid OS type; with "bool" -> "boolean" OS coerces the
/// reader's "true"/"false" string and selects the even ids.
#[test]
fn test_binary_opensearch_bool() {
    wait_for_opensearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-bool", "engine": "opensearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_bool_project("bool-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "os-bool",
            "bool-test",
            "http://127.0.0.1",
            &[
                ("OPENSEARCH_PORT", "9202"),
                ("OPENSEARCH_INDEX", "bench_bool")
            ],
        ),
        "opensearch bool run failed"
    );
    let recall = common::read_recall(&proj.root, "os-bool");
    println!("opensearch bool recall={:.3}", recall);
    assert!(recall >= 0.9, "opensearch bool recall {:.3} < 0.9", recall);
}

/// Multi-condition AND (keyword match AND numeric range) — verifies OpenSearch
/// composes two conditions of different types into one `bool.filter` array
/// (term on `color` AND range on `size`), not just a single clause.
#[test]
fn test_binary_opensearch_and_filter() {
    wait_for_opensearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-and", "engine": "opensearch",
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
            "os-and",
            "and-test",
            "http://127.0.0.1",
            &[
                ("OPENSEARCH_PORT", "9202"),
                ("OPENSEARCH_INDEX", "bench_and")
            ],
        ),
        "opensearch and-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "os-and");
    println!("opensearch and-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "opensearch and-filter recall {:.3} < 0.9",
        recall
    );
}

/// Multi-condition OR (`color == "red" OR size >= 90`) — verifies OpenSearch
/// unions two clauses into a `bool.should` (minimum_should_match 1).
#[test]
fn test_binary_opensearch_or_filter() {
    wait_for_opensearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-or", "engine": "opensearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_or_filter_project("or-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "os-or",
            "or-test",
            "http://127.0.0.1",
            &[
                ("OPENSEARCH_PORT", "9202"),
                ("OPENSEARCH_INDEX", "bench_or")
            ],
        ),
        "opensearch or-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "os-or");
    println!("opensearch or-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "opensearch or-filter recall {:.3} < 0.9",
        recall
    );
}

/// Nested/grouped boolean filter end-to-end:
/// `(color == "red" AND size >= 50) OR (color == "blue" AND size < 10)`.
/// Verifies OpenSearch nests each OR arm as its own `{bool:{must:[...]}}` inside
/// the outer `should` (native bool nesting) rather than flattening the tree into
/// one clause list — a mis-flattened builder matches a wildly different doc set,
/// so a high recall vs the nested-union ground truth proves correct nesting.
#[test]
fn test_binary_opensearch_nested_filter() {
    wait_for_opensearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-nested", "engine": "opensearch",
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
            "os-nested",
            "nested-test",
            "http://127.0.0.1",
            &[
                ("OPENSEARCH_PORT", "9202"),
                ("OPENSEARCH_INDEX", "bench_nested")
            ],
        ),
        "opensearch nested-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "os-nested");
    println!("opensearch nested-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "opensearch nested-filter recall {:.3} < 0.9",
        recall
    );
}

/// UUID exact-match filter end-to-end. Regression for the schema-type bug:
/// "uuid" is not a valid OS type; forwarding it verbatim made index creation
/// reject the whole mapping. With "uuid" -> "keyword" OS does an exact term
/// match, selecting the quarter of docs whose `uid` equals UUIDS[0].
#[test]
fn test_binary_opensearch_uuid() {
    wait_for_opensearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-uuid", "engine": "opensearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_uuid_project("uuid-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "os-uuid",
            "uuid-test",
            "http://127.0.0.1",
            &[
                ("OPENSEARCH_PORT", "9202"),
                ("OPENSEARCH_INDEX", "bench_uuid")
            ],
        ),
        "opensearch uuid run failed"
    );
    let recall = common::read_recall(&proj.root, "os-uuid");
    println!("opensearch uuid recall={:.3}", recall);
    assert!(recall >= 0.9, "opensearch uuid recall {:.3} < 0.9", recall);
}

/// Datetime range filter end-to-end. Regression for the schema-type bug:
/// "datetime" is not a valid OS type; with "datetime" -> "date" OS parses the
/// reader's ISO-8601 strings and selects the [day 100, day 300) window.
#[test]
fn test_binary_opensearch_datetime() {
    wait_for_opensearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-dt", "engine": "opensearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_datetime_project("dt-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "os-dt",
            "dt-test",
            "http://127.0.0.1",
            &[
                ("OPENSEARCH_PORT", "9202"),
                ("OPENSEARCH_INDEX", "bench_dt")
            ],
        ),
        "opensearch datetime run failed"
    );
    let recall = common::read_recall(&proj.root, "os-dt");
    println!("opensearch datetime recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "opensearch datetime recall {:.3} < 0.9",
        recall
    );
}

/// Geo-radius filter end-to-end (previously untested for OS). `geo` -> geo_point
/// mapping + `geo_distance` query; recall vs haversine ground truth.
#[test]
fn test_binary_opensearch_geo() {
    wait_for_opensearch();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-geo", "engine": "opensearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_geo_project("geo-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "os-geo",
            "geo-test",
            "http://127.0.0.1",
            &[
                ("OPENSEARCH_PORT", "9202"),
                ("OPENSEARCH_INDEX", "bench_geo")
            ],
        ),
        "opensearch geo run failed"
    );
    let recall = common::read_recall(&proj.root, "os-geo");
    println!("opensearch geo recall={:.3}", recall);
    assert!(recall >= 0.9, "opensearch geo recall {:.3} < 0.9", recall);
}

/// End-to-end full-text filter (#120): the query carries a single
/// `{"body":{"match":{"text":"quick"}}}` condition and ground truth is
/// brute-forced over only the docs whose body CONTAINS "quick". Before the fix,
/// OpenSearch dropped the text clause and ran the kNN query UNFILTERED, so
/// recall was scored against the filtered ground truth and collapsed. A high
/// recall here proves the analyzed `match` filter arm is applied end-to-end.
#[test]
fn test_binary_opensearch_fulltext() {
    wait_for_opensearch();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-text", "engine": "opensearch",
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
            "os-text",
            "text-test",
            // Explicit http:// scheme: the OpenSearch engine defaults to https
            // for a bare host, but the CI container runs plaintext http (security
            // plugin disabled).
            "http://127.0.0.1",
            &[
                ("OPENSEARCH_PORT", "9202"),
                ("OPENSEARCH_INDEX", "bench_fulltext"),
            ],
        ),
        "opensearch fulltext run failed"
    );

    let recall = common::read_recall(&proj.root, "os-text");
    println!("opensearch fulltext recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "opensearch fulltext recall {:.3} < 0.9",
        recall
    );
}

/// #210: after a run, exactly ONE **searchable** segment per shard must remain.
///
/// This is a comparability invariant, not a tidiness one. A segment is one HNSW
/// graph; a k-NN query searches every segment of every shard and merges the
/// per-segment result lists, so segment count moves both recall and latency.
/// Before the fix the engine force-merged without `max_num_segments`, leaving
/// whatever Lucene's merge policy happened to settle on — a number that depends
/// on ingest timing, batch size and worker count, so it varies between runs of
/// the same config and differs from Elasticsearch's pinned 1.
///
/// `"search": true` is the load-bearing part of the assertion, not a detail. A
/// force merge commits the new segment but only refreshes Lucene's INTERNAL
/// reader; the searcher that answers queries keeps the pre-merge segments until
/// an external refresh. An implementation that merges without refreshing
/// afterwards leaves a `_segments` response that *contains* a single big
/// segment while every query still runs against the old ones — so counting all
/// segments, or trusting the merge response, would both pass while the numbers
/// stayed exactly as wrong as before.
///
/// The upload deliberately uses several parallel workers and small batches: each
/// worker fills its own Lucene indexing buffer, so the pre-merge index has more
/// than one segment and the assertion below has something to prove. (Verified by
/// reverting `max_num_segments(1)`, which makes this test fail.)
#[test]
fn test_binary_opensearch_force_merge_single_segment() {
    wait_for_opensearch();

    const INDEX: &str = "bench_forcemerge";
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-fm", "engine": "opensearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 8, "batch_size": 20}
    }]);
    let proj = common::write_match_any_project(
        "force-merge-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );

    assert!(
        common::run_binary_extra(
            &proj.root,
            "os-fm",
            "force-merge-test",
            // Explicit http:// scheme: the OpenSearch engine defaults to https
            // for a bare host, but the CI container runs plaintext http (security
            // plugin disabled).
            "http://127.0.0.1",
            &[("OPENSEARCH_PORT", "9202"), ("OPENSEARCH_INDEX", INDEX)],
            // Keep the index so its segment layout can be inspected; without this
            // the engine drops it at the end of the run.
            &["--keep-data"],
        ),
        "opensearch force-merge run failed"
    );

    let client = os_client();
    let resp = client
        .get(format!("{}/{}/_segments", os_base_url(), INDEX))
        .send()
        .expect("_segments request");
    assert!(resp.status().is_success(), "_segments failed: {:?}", resp);
    let body: serde_json::Value = resp.json().unwrap();

    let shards = body["indices"][INDEX]["shards"]
        .as_object()
        .unwrap_or_else(|| panic!("no shards in _segments response: {}", body));
    assert!(!shards.is_empty(), "index reported zero shards: {}", body);

    for (shard_id, replicas) in shards {
        for replica in replicas.as_array().expect("shard entry is an array") {
            let segments = replica["segments"].as_object().expect("segments object");
            let searchable: Vec<&String> = segments
                .iter()
                .filter(|(_, s)| s["search"].as_bool().unwrap_or(false))
                .map(|(name, _)| name)
                .collect();
            assert_eq!(
                searchable.len(),
                1,
                "shard {} serves queries from {} searchable segments, expected 1 \
                 (force merge did not pin max_num_segments=1, or did not refresh \
                 afterwards so the searcher still holds the pre-merge segments); \
                 response: {}",
                shard_id,
                searchable.len(),
                body
            );
        }
    }

    // Clean up: this test is the only one that keeps its index.
    let _ = client.delete(format!("{}/{}", os_base_url(), INDEX)).send();
}
