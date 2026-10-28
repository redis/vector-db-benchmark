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

    // Query-level effort must be accepted by the real Lucene engine while
    // the returned result count stays at K, not the larger candidate budget.
    let tuned = vector_db_benchmark::opensearch_query::build_knn_body(
        &[1.0, 0.0, 0.0, 0.0],
        3,
        None,
        Some(128),
    )
    .unwrap();
    let response = client
        .post(format!("{}/{}/_search", os_base_url(), OS_INDEX))
        .json(&tuned)
        .send()
        .unwrap();
    assert!(
        response.status().is_success(),
        "{}",
        response.text().unwrap_or_default()
    );
    let tuned_body: serde_json::Value = response.json().unwrap();
    assert_eq!(tuned_body["hits"]["hits"].as_array().unwrap().len(), 3);
    assert_eq!(
        tuned_body["hits"]["hits"][0]["fields"]["_id"][0],
        id_to_uuid_hex(0)
    );

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
/// **This test would be a coin flip without the merge-stats guard below.** The
/// pre-merge segment count is not a property of the fixture, it is however many
/// Lucene indexing buffers happened to be live at the single end-of-upload
/// refresh — the upload is far too small to fill one. Measured on the engine's
/// own upload shape: 8 concurrent workers give 2-6 segments, but a run that ends
/// up effectively serialised gives exactly **1**, and then a bare `_forcemerge`
/// is a no-op that also leaves 1, so `searchable == 1` holds on unfixed code too.
/// A 2-vCPU CI runner reaches that state often. So the assertion alone would go
/// green on roughly half of all reverts, which is worse than no guard at all,
/// because it looks like protection.
///
/// The fix is to check that the force merge actually *had* something to merge.
/// `merges.total_docs` is exact and deterministic here: measured 400 (the whole
/// corpus, one merge) whenever the pre-merge index had >1 segment, and **0**
/// whenever it had exactly 1. So the run either proves the invariant or fails
/// loudly saying it could not — it never passes vacuously.
///
/// `batch_size` is small so the upload spans many bulk requests and the workers
/// overlap for as long as possible; that raises the odds of a multi-segment
/// pre-merge state, but the guard is what makes the test *correct* regardless.
#[test]
fn test_binary_opensearch_force_merge_single_segment() {
    wait_for_opensearch();

    const INDEX: &str = "bench_forcemerge";
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-fm", "engine": "opensearch",
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 8, "batch_size": 5}
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

    // The invariant held — but did it hold because the fix works, or because
    // there was only ever one segment and nothing to merge? Checked AFTER the
    // assertion above, deliberately: on genuinely unfixed code both conditions
    // are true at once, and the segment-count failure is the accurate diagnosis,
    // so it must be the one that fires.
    //
    // `merges.total_docs` counts docs rewritten by merges. Merging the corpus
    // into one segment moves all of them; a merge with nothing to do moves none.
    // Measured: 400 whenever the pre-merge index had >1 segment, 0 when it had 1.
    let doc_count = client
        .get(format!("{}/{}/_count", os_base_url(), INDEX))
        .send()
        .expect("_count request")
        .json::<serde_json::Value>()
        .unwrap()["count"]
        .as_u64()
        .expect("count");
    assert!(doc_count > 0, "fixture uploaded no documents");

    let merge_stats = client
        .get(format!("{}/{}/_stats/merge", os_base_url(), INDEX))
        .send()
        .expect("_stats/merge request")
        .json::<serde_json::Value>()
        .unwrap();
    let merged_docs = merge_stats["_all"]["primaries"]["merges"]["total_docs"]
        .as_u64()
        .unwrap_or_else(|| panic!("no merges.total_docs in {}", merge_stats));

    // Clean up before the last assertion so a failure does not leak the index.
    let _ = client.delete(format!("{}/{}", os_base_url(), INDEX)).send();

    assert!(
        merged_docs >= doc_count,
        "INCONCLUSIVE, not a pass: the index ended with one searchable segment, \
         but merges rewrote only {} of {} docs — so the upload produced a \
         single-segment index and this run proved nothing (the same result would \
         hold without max_num_segments(1)). Raise upload_params.parallel, lower \
         batch_size, or grow the fixture until the upload produces a \
         multi-segment index on this machine.",
        merged_docs,
        doc_count
    );
}

/// Shard count reaches the server, end to end (#211).
///
/// Every link in the chain is unit-covered on its own — the shipped configs are
/// asserted to pin the key, `parse_number_of_shards` is asserted to accept and
/// reject, `build_index_settings` is asserted to emit it — but nothing proved
/// the value actually survives the trip. `collection_params` is a
/// `#[serde(flatten)]` catch-all: an undeclared or wrongly-placed key parses
/// cleanly and is dropped in silence, and the index that results looks entirely
/// healthy. The server's own `_settings` is the only evidence, and only a live
/// run produces it, so the read-back lives here rather than in a PR comment that
/// expires.
///
/// Uses **3** deliberately: no cluster defaults to it (open-source OpenSearch
/// defaults to 1, legacy Amazon domains to 5), so a dropped setting cannot pass
/// by coincidence the way asserting 1 against a 1-defaulting cluster would.
/// Also asserts the value is echoed into the results JSON, which is what makes a
/// published run auditable after the index is gone.
#[test]
fn test_binary_opensearch_number_of_shards_reaches_the_server() {
    wait_for_opensearch();

    const SHARDS: i64 = 3;
    const INDEX: &str = "bench_shards";
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "os-shards", "engine": "opensearch",
        "collection_params": {
            "number_of_shards": SHARDS,
            "method": { "parameters": { "m": 16, "ef_construction": 100 } }
        },
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_bool_project(
        "shards-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );

    let client = os_client();
    let index_url = format!("{}/{}", os_base_url(), INDEX);
    // Start from a clean slate: an index left by a previous run keeps its
    // original shard count (it is fixed at creation), which would make this test
    // report the old value.
    let _ = client.delete(&index_url).send();

    // --keep-data so the index outlives the run and its settings can be read.
    let ok = common::run_binary_extra(
        &proj.root,
        "os-shards",
        "shards-test",
        // Explicit http:// scheme: the OpenSearch engine defaults to https for a
        // bare host, but the CI container runs plaintext http.
        "http://127.0.0.1",
        &[("OPENSEARCH_PORT", "9202"), ("OPENSEARCH_INDEX", INDEX)],
        &["--keep-data"],
    );
    assert!(ok, "opensearch number_of_shards run failed");

    let settings: serde_json::Value = client
        .get(format!("{}/_settings", index_url))
        .send()
        .expect("read index settings")
        .json()
        .expect("settings response is JSON");
    // OpenSearch stringifies index settings.
    let stored = settings[INDEX]["settings"]["index"]["number_of_shards"]
        .as_str()
        .map(|s| s.to_string());
    // Count the primaries that actually materialized, not just the setting.
    let cat: serde_json::Value = client
        .get(format!(
            "{}/_cat/shards/{}?format=json&h=prirep,state",
            os_base_url(),
            INDEX
        ))
        .send()
        .expect("read _cat/shards")
        .json()
        .expect("_cat/shards response is JSON");
    let primaries = cat
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter(|r| r["prirep"].as_str() == Some("p"))
                .count()
        })
        .unwrap_or(0);

    // Clean up before asserting so a failure does not leak the index into the
    // next test's cluster.
    let _ = client.delete(&index_url).send();

    assert_eq!(
        stored.as_deref(),
        Some(SHARDS.to_string().as_str()),
        "index {INDEX} must be created with number_of_shards={SHARDS}; got {stored:?}. \
         collection_params is a serde(flatten) catch-all, so a dropped key leaves a \
         perfectly healthy index at the cluster default"
    );
    assert_eq!(
        primaries, SHARDS as usize,
        "expected {SHARDS} primary shards to materialize, saw {primaries}"
    );

    // The run must also be self-describing: an auditor reading a published
    // result file should not have to infer the shard count from the config name.
    let params = read_params_obj(&proj.root, "os-shards");
    assert_eq!(
        params["number_of_shards"],
        serde_json::json!(SHARDS),
        "search results must record params.number_of_shards; got {params}"
    );
}

/// Read the `params` object from an engine's search result JSON.
fn read_params_obj(root: &std::path::Path, engine: &str) -> serde_json::Value {
    let pattern = format!("{}-*-search-*.json", engine);
    let dir = root.join("results");
    let path = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            glob::Pattern::new(&pattern)
                .unwrap()
                .matches(&p.file_name().unwrap().to_string_lossy())
        })
        .unwrap_or_else(|| panic!("no search result for {}", engine));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    v["params"].clone()
}

#[test]
fn test_binary_opensearch_filter_only_contract() {
    wait_for_opensearch();
    let _fielddata = DisabledIdFielddata::new();
    let config = serde_json::json!([{"name":"scalar-os","engine":"opensearch","collection_params":{"number_of_shards":1},"upload_params":{"parallel":1},"search_params":[{"parallel":4,"top":3}]}]);
    let (proj, total, returned) = common::write_filter_only_project(&config.to_string());
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    let index = "scalar-contract";
    assert!(common::run_binary_extra(
        &proj.root,
        "scalar-os",
        "scalar-test",
        "http://127.0.0.1",
        &[("OPENSEARCH_PORT", &port), ("OPENSEARCH_INDEX", index)],
        &[
            "--skip-vector-index",
            "--queries",
            "90",
            "--keep-data",
            "--fail-on-dropped-queries"
        ]
    ));
    common::assert_filter_only_results(&proj.root, "opensearch-no-vector", total, returned);
    let mapping: serde_json::Value = os_client()
        .get(format!("{}/{index}/_mapping", os_base_url()))
        .send()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(
        mapping[index]["mappings"]["properties"]["price"]["type"],
        "double"
    );
    let vector = &mapping[index]["mappings"]["properties"]["vector"];
    assert_eq!(vector["type"], "float");
    assert_eq!(vector["index"], false);
    assert_eq!(vector["doc_values"], false);
    let settings: serde_json::Value = os_client()
        .get(format!("{}/{index}/_settings", os_base_url()))
        .send()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(settings[index]["settings"]["index"]["knn"], "false");
    common::assert_filter_only_predicates(
        &proj.root,
        "scalar-os",
        "opensearch-no-vector",
        "http://127.0.0.1",
        &[("OPENSEARCH_PORT", &port), ("OPENSEARCH_INDEX", index)],
    );
    // The reuse path must keep the scalar schema and corpus intact.
    assert!(common::run_binary_extra(
        &proj.root,
        "scalar-os",
        "scalar-test",
        "http://127.0.0.1",
        &[("OPENSEARCH_PORT", &port), ("OPENSEARCH_INDEX", index)],
        &[
            "--skip-vector-index",
            "--queries",
            "90",
            "--skip-upload",
            "--keep-data",
            "--fail-on-dropped-queries"
        ]
    ));
    common::assert_filter_only_results(&proj.root, "opensearch-no-vector", total, returned);
    os_client()
        .delete(format!("{}/{index}", os_base_url()))
        .send()
        .unwrap();
    std::fs::remove_dir_all(proj.root).unwrap();
}

/// Temporarily forbid loading _id fielddata and restore the previous setting.
/// This suite is destructive and requires the dedicated disposable compose
/// cluster; OPENSEARCH_PORT must never point at a shared or production cluster.
struct DisabledIdFielddata(serde_json::Value);
impl DisabledIdFielddata {
    fn new() -> Self {
        let old: serde_json::Value = os_client()
            .get(format!("{}/_cluster/settings", os_base_url()))
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        let previous = old["transient"]["indices"]["id_field_data"]["enabled"].clone();
        os_client()
            .put(format!("{}/_cluster/settings", os_base_url()))
            .json(&serde_json::json!({"transient":{"indices.id_field_data.enabled":false}}))
            .send()
            .unwrap()
            .error_for_status()
            .unwrap();
        Self(previous)
    }
}
impl Drop for DisabledIdFielddata {
    fn drop(&mut self) {
        let _ = os_client()
            .put(format!("{}/_cluster/settings", os_base_url()))
            .json(&serde_json::json!({"transient":{"indices.id_field_data.enabled":self.0}}))
            .send();
    }
}

#[test]
fn test_binary_opensearch_filter_only_dot_high_dimension_and_default_limit() {
    wait_for_opensearch();
    let config = serde_json::json!([{"name":"scalar-wide","engine":"opensearch","collection_params":{"number_of_shards":1},"upload_params":{"parallel":1},"search_params":[{"parallel":1}]}]);
    let (proj, total, _) = common::write_filter_only_project(&config.to_string());
    common::resize_filter_only_vectors(&proj.root, 4096, "dot");
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    assert!(common::run_binary_extra(
        &proj.root,
        "scalar-wide",
        "scalar-test",
        "http://127.0.0.1",
        &[
            ("OPENSEARCH_PORT", &port),
            ("OPENSEARCH_INDEX", "scalar-wide")
        ],
        &["--skip-vector-index", "--fail-on-dropped-queries"]
    ));
    let doc = common::read_result_doc(&proj.root, "opensearch-no-vector");
    assert_eq!(doc["results"]["requested_queries"], 9);
    assert_eq!(doc["results"]["filter_only"]["result_limit"], 10);
    assert_eq!(
        doc["results"]["filter_only"]["matched_documents_sum"],
        total
    );
    assert_eq!(
        doc["results"]["filter_only"]["returned_documents_sum"],
        common::filter_only_expected_counts()
            .iter()
            .map(|n| (*n).min(10))
            .sum::<u64>()
    );
    std::fs::remove_dir_all(proj.root).unwrap();
}

#[test]
fn test_binary_opensearch_filter_only_rejects_vector_index_reuse() {
    wait_for_opensearch();
    let config = serde_json::json!([{"name":"scalar-reuse","engine":"opensearch","collection_params":{"number_of_shards":1},"upload_params":{"parallel":1},"search_params":[{"parallel":1,"top":3}]}]);
    let (proj, _, _) = common::write_filter_only_project(&config.to_string());
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    let index = "scalar-vector-reuse";
    let envs = [
        ("OPENSEARCH_PORT", port.as_str()),
        ("OPENSEARCH_INDEX", index),
    ];
    assert!(common::run_binary_extra(
        &proj.root,
        "scalar-reuse",
        "scalar-test",
        "http://127.0.0.1",
        &envs,
        &["--skip-search", "--keep-data"]
    ));
    common::clear_result_files(&proj.root);
    let out = std::process::Command::new(common::binary_path())
        .current_dir(&proj.root)
        .envs(envs)
        .args([
            "--engines",
            "scalar-reuse",
            "--datasets",
            "scalar-test",
            "--host",
            "http://127.0.0.1",
            "--skip-vector-index",
            "--skip-upload",
            "--queries",
            "1",
            "--fail-on-dropped-queries",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("filter-only requires index.knn=false"),
        "{stderr}"
    );
    assert!(std::fs::read_dir(proj.root.join("results"))
        .unwrap()
        .all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("-search-")));
    os_client()
        .delete(format!("{}/{index}", os_base_url()))
        .send()
        .unwrap()
        .error_for_status()
        .unwrap();
    std::fs::remove_dir_all(proj.root).unwrap();
}

/// #343: `--dump-results` on the OpenSearch KNN path. Ids come back as the
/// engine's uuid `_id`s and must be dumped as corpus rows; the dump's recall
/// must equal the published `mean_recall`.
#[test]
fn test_binary_opensearch_dump_results_vector() {
    wait_for_opensearch();
    let configs = serde_json::json!([{
        "name": "os-dump-vec", "engine": "opensearch",
        "search_params": [{"parallel": 2, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project("dump-vec-test", &configs.to_string(), 8);
    let dump = proj.root.join("dumps");
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    assert!(common::run_binary_extra(
        &proj.root,
        "os-dump-vec",
        "dump-vec-test",
        "http://127.0.0.1",
        &[
            ("OPENSEARCH_PORT", &port),
            ("OPENSEARCH_INDEX", "bench_dump_vec")
        ],
        &[
            "--dump-results",
            dump.to_str().unwrap(),
            "--fail-on-dropped-queries"
        ]
    ));
    common::assert_vector_dump_matches_run(&proj.root, &dump, "opensearch", "os-dump-vec");
}

/// #343 + #344: declared `"workload": "filter-only"` on OpenSearch, dumped
/// and self-compared.
#[test]
fn test_binary_opensearch_dump_results_declared_filter_only() {
    wait_for_opensearch();
    let _fielddata = DisabledIdFielddata::new();
    let config = serde_json::json!([{
        "name": "dump-fo-os", "engine": "opensearch",
        "collection_params": {"number_of_shards": 1}, "upload_params": {"parallel": 1},
        "search_params": [{"parallel": 2, "top": 3, "workload": "filter-only"}]
    }]);
    let (proj, total, returned) = common::write_filter_only_project(&config.to_string());
    let dump = proj.root.join("dumps");
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    assert!(common::run_binary_extra(
        &proj.root,
        "dump-fo-os",
        "scalar-test",
        "http://127.0.0.1",
        &[("OPENSEARCH_PORT", &port), ("OPENSEARCH_INDEX", "dump-fo")],
        &[
            "--queries",
            "90",
            "--dump-results",
            dump.to_str().unwrap(),
            "--fail-on-dropped-queries"
        ]
    ));
    common::assert_filter_only_results(&proj.root, "dump-fo-os", total, returned);
    common::assert_filter_only_dump(&dump, "opensearch", 90);
    let (ok, report) = common::run_compare(&dump, &dump, "opensearch");
    assert!(ok, "self-comparison must pass");
    let counts = &report["pairs"][0]["shapes"][0]["count_agreement"];
    assert_eq!(counts["fraction"], 1.0);
    assert_eq!(counts["complete_page_set_agreement"], 1.0);
}

/// #339: the ranked full-text workload end to end on OpenSearch — boosts,
/// the porter2 analyzer (a Porter analyzer would also return "generation"
/// for "generously"), the skipped termless query, per-query docids + scores.
/// See `common::assert_bm25_run` for every claim.
#[test]
fn test_binary_opensearch_fulltext_bm25() {
    wait_for_opensearch();
    let configs = serde_json::json!([{
        "name": "os-bm25", "engine": "opensearch",
        "collection_params": {"number_of_shards": 1}, "upload_params": {"parallel": 1},
        "search_params": common::bm25_search_params()
    }]);
    let proj = common::write_bm25_project(&configs.to_string());
    let dump = proj.root.join("dumps");
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    assert!(common::run_binary_extra(
        &proj.root,
        "os-bm25",
        common::BM25_DATASET,
        "http://127.0.0.1",
        &[
            ("OPENSEARCH_PORT", &port),
            ("OPENSEARCH_INDEX", "bench_bm25")
        ],
        &[
            "--dump-results",
            dump.to_str().unwrap(),
            "--fail-on-dropped-queries"
        ]
    ));
    common::assert_bm25_run(&proj.root, &dump, "opensearch", "os-bm25");
    std::fs::remove_dir_all(&proj.root).ok();
}

/// #339: the fulltext arm refuses, before the timed window, an index whose
/// text fields use the default analyzer (built by a vector-only config) and an
/// index missing a searched text field — both reached through `--skip-upload`.
#[test]
fn test_binary_opensearch_fulltext_refuses_unpinned_or_incomplete_index() {
    wait_for_opensearch();
    let name = "os-bm25-guard";
    let index = "bench_bm25_guard";
    let write_config = |root: &std::path::Path, search_params: serde_json::Value| {
        std::fs::write(
            root.join("experiments/configurations/test.json"),
            serde_json::json!([{"name": name, "engine": "opensearch",
                "collection_params": {"number_of_shards": 1},
                "upload_params": {"parallel": 1}, "search_params": search_params}])
            .to_string(),
        )
        .unwrap();
    };
    let fulltext = |fields: serde_json::Value| {
        serde_json::json!([{"parallel": 1, "top": 5, "workload": "fulltext",
            "fulltext": {"fields": fields}}])
    };
    let proj = common::write_bm25_project("[]");
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    let envs = [
        ("OPENSEARCH_PORT", port.as_str()),
        ("OPENSEARCH_INDEX", index),
    ];
    let run = |extra: &[&str]| {
        common::run_binary_output(
            &proj.root,
            name,
            common::BM25_DATASET,
            "http://127.0.0.1",
            &envs,
            extra,
        )
    };
    let refused = |out: std::process::Output, needle: &str| {
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "must refuse: {stderr}");
        assert!(stderr.contains(needle), "expected {needle:?} in: {stderr}");
    };

    write_config(&proj.root, serde_json::json!([{"parallel": 1}]));
    assert!(run(&["--skip-search", "--keep-data"]).status.success());
    write_config(&proj.root, fulltext(serde_json::json!({"title": 1.0})));
    refused(
        run(&["--skip-upload", "--keep-data", "--exit-on-error"]),
        "expected the pinned vdbb_english_porter2",
    );

    assert!(run(&["--skip-search", "--keep-data"]).status.success());
    common::write_bm25_registry(&proj.root, true);
    write_config(
        &proj.root,
        fulltext(serde_json::json!({"title": 1.0, "body": 1.0})),
    );
    refused(
        run(&["--skip-upload", "--keep-data", "--exit-on-error"]),
        "'body' is not in the OpenSearch mapping",
    );
    write_config(&proj.root, fulltext(serde_json::json!({"title": 1.0})));
    let ok = run(&["--skip-upload", "--exit-on-error"]);
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );

    // #339 review: a 2-shard index (per-shard BM25 statistics) is refused
    // unless the config opts in; the opt-in records the live shard count.
    common::write_bm25_registry(&proj.root, false);
    let two_shards = |search_params: serde_json::Value| {
        std::fs::write(
            proj.root.join("experiments/configurations/test.json"),
            serde_json::json!([{"name": name, "engine": "opensearch",
                "collection_params": {"number_of_shards": 2},
                "upload_params": {"parallel": 1}, "search_params": search_params}])
            .to_string(),
        )
        .unwrap();
    };
    two_shards(fulltext(serde_json::json!({"title": 1.0})));
    refused(run(&["--keep-data", "--exit-on-error"]), "2 shards");
    common::clear_result_files(&proj.root);
    two_shards(
        serde_json::json!([{"parallel": 1, "top": 5, "workload": "fulltext",
        "fulltext": {"fields": {"title": 1.0}, "allow_multi_shard": true}}]),
    );
    let ok = run(&["--skip-upload", "--exit-on-error"]);
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    let doc = common::read_result_doc(&proj.root, name);
    assert_eq!(doc["params"]["fulltext"]["number_of_shards"], 2);
    assert_eq!(doc["params"]["fulltext"]["allow_multi_shard"], true);
    let _ = os_client()
        .delete(format!("{}/{index}", os_base_url()))
        .send();
    std::fs::remove_dir_all(&proj.root).ok();
}

/// #339 follow-up: `analysis: parity` end to end on OpenSearch — the nostem
/// analyzer, whitespace normalization at upload (verified on `_source`), only
/// the searched fields indexed, and refusal of an index uploaded in the other
/// mode when reused through `--skip-upload`, in both directions.
#[test]
fn test_binary_opensearch_fulltext_parity() {
    wait_for_opensearch();
    let name = "os-bm25-parity";
    let index = "bench_bm25_parity";
    let write_config = |root: &std::path::Path, search_params: serde_json::Value| {
        std::fs::write(
            root.join("experiments/configurations/test.json"),
            serde_json::json!([{"name": name, "engine": "opensearch",
                "collection_params": {"number_of_shards": 1},
                "upload_params": {"parallel": 1}, "search_params": search_params}])
            .to_string(),
        )
        .unwrap();
    };
    let proj = common::write_bm25_project("[]");
    write_config(&proj.root, common::bm25_parity_search_params());
    let dump = proj.root.join("dumps");
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    let envs = [
        ("OPENSEARCH_PORT", port.as_str()),
        ("OPENSEARCH_INDEX", index),
    ];
    assert!(common::run_binary_extra(
        &proj.root,
        name,
        common::BM25_DATASET,
        "http://127.0.0.1",
        &envs,
        &[
            "--dump-results",
            dump.to_str().unwrap(),
            "--fail-on-dropped-queries",
            "--keep-data"
        ]
    ));
    common::assert_bm25_parity_run(&proj.root, &dump, "opensearch", name);
    let hit: serde_json::Value = os_client()
        .post(format!("{}/{index}/_search", os_base_url()))
        .json(&serde_json::json!({"query": {"term": {"docid": "d4"}}}))
        .send()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(
        hit["hits"]["hits"][0]["_source"]["headings"], "alpha beta",
        "{hit}"
    );
    let run = |extra: &[&str]| {
        common::run_binary_output(
            &proj.root,
            name,
            common::BM25_DATASET,
            "http://127.0.0.1",
            &envs,
            extra,
        )
    };
    let refused = |out: std::process::Output, needle: &str| {
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "must refuse: {stderr}");
        assert!(stderr.contains(needle), "expected {needle:?} in: {stderr}");
    };
    let english = serde_json::json!([{"parallel": 1, "top": 5, "workload": "fulltext",
        "fulltext": {"fields": {"headings": 1.0, "segment": 1.0}}}]);
    write_config(&proj.root, english);
    refused(
        run(&["--skip-upload", "--keep-data", "--exit-on-error"]),
        "the index records analysis",
    );
    common::clear_result_files(&proj.root);
    assert!(run(&["--skip-search", "--keep-data"]).status.success());
    write_config(&proj.root, common::bm25_parity_search_params());
    refused(
        run(&["--skip-upload", "--keep-data", "--exit-on-error"]),
        "without whitespace normalization",
    );
    // A schema text field that is indexed but not searched is refused in
    // parity. Build the parity index while the schema does not declare
    // `title` (OpenSearch then maps the payload key dynamically, as indexed
    // text), then reuse it once the schema declares `title` as text.
    let registry_path = proj.root.join("datasets/datasets.json");
    let mut registry: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&registry_path).unwrap()).unwrap();
    registry[0]["schema"]
        .as_object_mut()
        .unwrap()
        .remove("title");
    std::fs::write(&registry_path, registry.to_string()).unwrap();
    assert!(run(&["--skip-search", "--keep-data"]).status.success());
    common::write_bm25_registry(&proj.root, false);
    refused(
        run(&["--skip-upload", "--keep-data", "--exit-on-error"]),
        "indexed text field 'title' that is not searched",
    );
    let _ = os_client()
        .delete(format!("{}/{index}", os_base_url()))
        .send();
    std::fs::remove_dir_all(&proj.root).ok();
}

/// #340 query side on OpenSearch, over an MS MARCO-shaped corpus: the prepared
/// `filter_queries.jsonl` runs, every answer passes the per-query count gate,
/// records carry the prepared shape, the mapping holds exactly the referenced
/// derived keyword/long fields with `dynamic: false` (text and `url` stay in
/// `_source`, unindexed), reuse revalidates, and a SUM-PRESERVING error in two
/// expected counts fails the run.
#[test]
fn test_binary_opensearch_msmarco_filter_only_per_query_gate() {
    wait_for_opensearch();
    let _fielddata = DisabledIdFielddata::new();
    let config = serde_json::json!([{"name":"msm-os","engine":"opensearch","collection_params":{"number_of_shards":1},"upload_params":{"parallel":2},"search_params":[{"parallel":4,"top":5}]}]);
    let p = common::write_msmarco_filter_project(&config.to_string());
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    let index = "msm-filter";
    let envs = [
        ("OPENSEARCH_PORT", port.as_str()),
        ("OPENSEARCH_INDEX", index),
    ];
    let dump = p.root.join("dumps");
    let run = |extra: &[&str]| {
        let mut args = vec![
            "--skip-vector-index",
            "--queries",
            "27",
            "--repetitions",
            "1",
            "--keep-data",
            "--fail-on-dropped-queries",
        ];
        args.extend_from_slice(extra);
        common::run_binary_extra(
            &p.root,
            "msm-os",
            common::MSM_DATASET,
            "http://127.0.0.1",
            &envs,
            &args,
        )
    };
    assert!(run(&["--dump-results", dump.to_str().unwrap()]));
    common::assert_msmarco_filter_run(&p, "opensearch-no-vector", 27, &dump);

    let mapping: serde_json::Value = os_client()
        .get(format!("{}/{index}/_mapping", os_base_url()))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let m = &mapping[index]["mappings"];
    assert_eq!(m["dynamic"], "false", "{m}");
    let mut fields: Vec<&String> = m["properties"].as_object().unwrap().keys().collect();
    fields.sort();
    assert_eq!(
        fields,
        ["category", "doc_len", "rand_key", "tier", "vector"],
        "filter-only maps exactly the fields its queries reference: {m}"
    );
    assert_eq!(m["properties"]["category"]["type"], "keyword");
    assert_eq!(m["properties"]["rand_key"]["type"], "long");
    // Unmapped fields are stored in _source, not indexed: a query on one
    // matches nothing although every document holds it.
    let hit: serde_json::Value = os_client()
        .get(format!("{}/{index}/_search?size=1", os_base_url()))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let source = &hit["hits"]["hits"][0]["_source"];
    assert!(source["segment"].as_str().unwrap().starts_with("Passage"));
    assert!(source["url"].as_str().unwrap().contains("fx.example"));
    let unindexed: serde_json::Value = os_client()
        .post(format!("{}/{index}/_count", os_base_url()))
        .json(&serde_json::json!({"query": {"exists": {"field": "segment"}}}))
        .send()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(unindexed["count"], 0, "{unindexed}");

    // Reuse: the realized mapping and the local corpus are revalidated.
    common::clear_result_files(&p.root);
    assert!(run(&["--skip-upload"]));
    let r = common::read_results_obj(&p.root, "opensearch-no-vector");
    assert_eq!(r["filter_only"]["per_query_counts_checked"], 27);

    // Two compensating errors keep the sum; the per-query gate still fails.
    common::clear_result_files(&p.root);
    p.tamper_expected_counts(&[(0, 1), (3, -1)]);
    let stderr = common::run_binary_expect_failure(
        &p.root,
        "msm-os",
        common::MSM_DATASET,
        "http://127.0.0.1",
        &envs,
        &[
            "--skip-vector-index",
            "--skip-upload",
            "--keep-data",
            "--repetitions",
            "1",
        ],
    );
    assert!(stderr.contains("per-query count gate"), "{stderr}");
    assert!(stderr.contains("fx-filter-000 (tag_eq)"), "{stderr}");
    assert!(stderr.contains("fx-filter-003 (numeric_point)"), "{stderr}");

    os_client()
        .delete(format!("{}/{index}", os_base_url()))
        .send()
        .unwrap();
    std::fs::remove_dir_all(&p.root).ok();
}

/// #340 x #344: filter-only with MS MARCO prepared queries on the shared
/// `full` index. One engine config mixes a vector search config with the
/// filter-only one, so the index is a k-NN index mapping every declared field
/// (the vector workload may filter on any of them); filter-only keeps #340's
/// contract on it: the realized mapping, the metadata domain and its
/// predicates are checked against the narrowed field set (its numeric fields
/// mapped `long` / `double`, `url` free to hold `;`), the prepared set runs,
/// and the per-query count gate passes — and still fails a SUM-PRESERVING
/// error in two expected counts.
#[test]
fn test_binary_opensearch_msmarco_filter_only_on_the_full_layout() {
    wait_for_opensearch();
    // No `DisabledIdFielddata` here: the vector config's k-NN query
    // tie-breaks on `_id`. The no-vector MS MARCO test above keeps it.
    let config = serde_json::json!([{"name":"msm-os-full","engine":"opensearch","collection_params":{"number_of_shards":1},"upload_params":{"parallel":2},
        "search_params":[{"parallel":1,"top":1},{"parallel":4,"top":5,"workload":"filter-only"}]}]);
    let p = common::write_msmarco_filter_project(&config.to_string());
    let n = p.queries.len();
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    let index = "msm-filter-full";
    let envs = [
        ("OPENSEARCH_PORT", port.as_str()),
        ("OPENSEARCH_INDEX", index),
    ];
    let dump = p.root.join("dumps");
    let out = common::run_binary_output(
        &p.root,
        "msm-os-full",
        common::MSM_DATASET,
        "http://127.0.0.1",
        &envs,
        &[
            "--keep-data",
            "--fail-on-dropped-queries",
            "--repetitions",
            "1",
            "--dump-results",
            dump.to_str().unwrap(),
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        stdout.matches("Experiment stage: Upload").count(),
        1,
        "{stdout}"
    );
    common::assert_msmarco_full_run(&p, n, &dump);

    let mapping: serde_json::Value = os_client()
        .get(format!("{}/{index}/_mapping", os_base_url()))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let props = &mapping[index]["mappings"]["properties"];
    assert_eq!(props["vector"]["type"], "knn_vector", "{props}");
    assert_eq!(props["category"]["type"], "keyword", "{props}");
    assert_eq!(props["rand_key"]["type"], "long", "{props}");
    assert_eq!(props["url"]["type"], "keyword", "{props}");

    // Two compensating errors keep the sum; the per-query gate still fails
    // on `full`, naming the queries.
    common::clear_result_files(&p.root);
    p.tamper_expected_counts(&[(0, 1), (3, -1)]);
    let out = common::run_binary_output(
        &p.root,
        "msm-os-full",
        common::MSM_DATASET,
        "http://127.0.0.1",
        &envs,
        &["--skip-upload", "--keep-data", "--repetitions", "1"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(stderr.contains("per-query count gate"), "{stderr}");
    assert!(stderr.contains("fx-filter-000 (tag_eq)"), "{stderr}");
    assert!(stderr.contains("fx-filter-003 (numeric_point)"), "{stderr}");

    let _ = os_client()
        .delete(format!("{}/{index}", os_base_url()))
        .send();
    std::fs::remove_dir_all(&p.root).ok();
}

/// #340 review: without a prepared query set (H&M, synthetic) filter-only maps
/// every declared field as before #340 — `text` included — and leaves dynamic
/// mapping alone. Only a prepared set narrows the layout.
#[test]
fn test_binary_opensearch_filter_only_without_prepared_set_keeps_the_declared_layout() {
    wait_for_opensearch();
    let config = serde_json::json!([{"name":"scalar-layout-os","engine":"opensearch","collection_params":{"number_of_shards":1},"upload_params":{"parallel":1},"search_params":[{"parallel":1,"top":3}]}]);
    let (proj, _, _) = common::write_filter_only_project(&config.to_string());
    let declared = common::add_text_field(&proj.root);
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    let index = "scalar-layout";
    assert!(common::run_binary_extra(
        &proj.root,
        "scalar-layout-os",
        "scalar-test",
        "http://127.0.0.1",
        &[("OPENSEARCH_PORT", &port), ("OPENSEARCH_INDEX", index)],
        &["--skip-vector-index", "--skip-search", "--keep-data"],
    ));
    let mapping: serde_json::Value = os_client()
        .get(format!("{}/{index}/_mapping", os_base_url()))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let m = &mapping[index]["mappings"];
    assert!(m.get("dynamic").is_none(), "{m}");
    let mut fields: Vec<String> = m["properties"]
        .as_object()
        .unwrap()
        .keys()
        .filter(|k| *k != "vector")
        .cloned()
        .collect();
    fields.sort();
    assert_eq!(fields, declared, "{m}");
    assert_eq!(m["properties"]["note"]["type"], "text", "{m}");
    // Mapped by the harness, not by dynamic mapping (which would add a
    // `.keyword` multi-field to every string): the layout is the declared one.
    for f in &declared {
        assert!(
            m["properties"][f].get("fields").is_none(),
            "{f} dynamic: {m}"
        );
    }
    os_client()
        .delete(format!("{}/{index}", os_base_url()))
        .send()
        .unwrap();
    std::fs::remove_dir_all(&proj.root).ok();
}

/// #344: one engine config runs vector, filter-only and fulltext against ONE
/// OpenSearch index built once in the shared `full` layout — a k-NN index
/// whose `float` fields are mapped `double`, so the filter-only range on
/// `price` (values 1e-8 apart) counts exactly. See
/// `common::shared_index::assert_run` for every claim.
#[test]
fn test_binary_opensearch_one_ingest_many_workloads() {
    use common::shared_index;
    wait_for_opensearch();
    // No `DisabledIdFielddata` here: the vector arm's KNN query tie-breaks on
    // `_id`. The filter-only arm's independence from `_id` fielddata is pinned
    // by the filter-only suites above.
    let config = "os-shared";
    let index = "bench_shared_full";
    let configs = serde_json::json!([{
        "name": config, "engine": "opensearch",
        "collection_params": {"number_of_shards": 1}, "upload_params": {"parallel": 2},
        "search_params": shared_index::search_params()
    }]);
    let proj = shared_index::write_project(&configs.to_string());
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    let out = common::run_binary_output(
        &proj.root,
        config,
        shared_index::DATASET,
        "http://127.0.0.1",
        &[("OPENSEARCH_PORT", &port), ("OPENSEARCH_INDEX", index)],
        &["--fail-on-dropped-queries", "--keep-data"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        stdout.matches("Experiment stage: Upload").count(),
        1,
        "one ingest for three workloads: {stdout}"
    );
    shared_index::assert_run(&proj, config, "opensearch_force_merge_refresh");

    // The realized index is the union: knn on, an indexed knn_vector, text,
    // keyword, long and double — holding the corpus once.
    let mapping: serde_json::Value = os_client()
        .get(format!("{}/{index}/_mapping", os_base_url()))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let props = &mapping[index]["mappings"]["properties"];
    assert_eq!(props["vector"]["type"], "knn_vector", "{props}");
    assert_eq!(props["price"]["type"], "double", "{props}");
    assert_eq!(props["size"]["type"], "long", "{props}");
    assert_eq!(props["color"]["type"], "keyword", "{props}");
    assert_eq!(props["title"]["type"], "text", "{props}");
    let settings: serde_json::Value = os_client()
        .get(format!("{}/{index}/_settings", os_base_url()))
        .send()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(settings[index]["settings"]["index"]["knn"], "true");
    let count: serde_json::Value = os_client()
        .get(format!("{}/{index}/_count", os_base_url()))
        .send()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(count["count"], shared_index::DOCS);
    let _ = os_client()
        .delete(format!("{}/{index}", os_base_url()))
        .send();
    std::fs::remove_dir_all(&proj.root).ok();
}

/// #344: the realized index is checked against the planned layout ONCE,
/// before any search config runs. A `--skip-upload` run of the mixed config —
/// vector FIRST — over an index a vector-only config built (`price` mapped
/// `float`, which collapses 1e-8-apart values) is refused before any result is
/// written.
#[test]
fn test_binary_opensearch_full_layout_refuses_a_reused_vector_layout_index() {
    os_reuse_refused(
        "os-shared-reuse",
        "bench_shared_reuse",
        common::shared_index::vector_only(),
        common::shared_index::search_params(),
        "index layout check (full): filter-only requires double mapping for price",
    );
}

/// #344, the reverse: a vector-only config over the `full` index a mixed
/// config built (`price` mapped `double`) would be labelled `vector`. Refused
/// before any result is written.
#[test]
fn test_binary_opensearch_vector_layout_refuses_a_reused_full_index() {
    os_reuse_refused(
        "os-shared-reverse",
        "bench_shared_reverse",
        common::shared_index::search_params(),
        common::shared_index::vector_only(),
        "index layout check (vector): the index is not a `vector`-layout index",
    );
}

/// #344 review: on the shared `full` index, a corpus token outside
/// filter-only's metadata domain is refused before the upload.
#[test]
fn test_binary_opensearch_full_layout_refuses_bad_metadata_before_upload() {
    use common::shared_index;
    wait_for_opensearch();
    let config = "os-shared-domain";
    let index = "bench_shared_domain";
    let configs = serde_json::json!([{
        "name": config, "engine": "opensearch",
        "collection_params": {"number_of_shards": 1},
        "upload_params": {"parallel": 1},
        "search_params": shared_index::search_params()
    }]);
    let proj = shared_index::write_project(&configs.to_string());
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    shared_index::assert_bad_metadata_refused_before_upload(
        &proj,
        config,
        "http://127.0.0.1",
        &[
            ("OPENSEARCH_PORT", port.as_str()),
            ("OPENSEARCH_INDEX", index),
        ],
    );
    let _ = os_client()
        .delete(format!("{}/{index}", os_base_url()))
        .send();
    std::fs::remove_dir_all(&proj.root).ok();
}

fn os_reuse_refused(
    config: &str,
    index: &str,
    build: serde_json::Value,
    reuse: serde_json::Value,
    expected: &str,
) {
    use common::shared_index;
    wait_for_opensearch();
    let proj = shared_index::write_project("[]");
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    shared_index::assert_reuse_refused(
        &proj,
        &serde_json::json!({"name": config, "engine": "opensearch",
            "collection_params": {"number_of_shards": 1},
            "upload_params": {"parallel": 1}}),
        "http://127.0.0.1",
        &[
            ("OPENSEARCH_PORT", port.as_str()),
            ("OPENSEARCH_INDEX", index),
        ],
        build,
        reuse,
        expected,
    );
    let _ = os_client()
        .delete(format!("{}/{index}", os_base_url()))
        .send();
    std::fs::remove_dir_all(&proj.root).ok();
}

/// #341: the hybrid workload on OpenSearch — lucene `knn` with the predicate
/// as its `filter` and an explicit `k`, one timed pass per selectivity bucket,
/// recall against the brute-forced FILTERED truth per bucket. The k-NN
/// plugin's filtered-request counter must have grown by at least one per
/// query in every bucket (the filter reached the k-NN query), and the report
/// states that Lucene's exact-search fallback is not observable.
#[test]
fn test_binary_opensearch_hybrid_per_bucket_recall() {
    wait_for_opensearch();
    let configs = serde_json::json!([{
        "name": "os-hyb", "engine": "opensearch",
        "collection_params": {"number_of_shards": 1},
        "search_params": [{"parallel": 2, "top": 10, "workload": "hybrid",
                           "search_params": {"ef": 200}}],
        "upload_params": {"parallel": 2, "batch_size": 500}
    }]);
    let proj = common::write_msmarco_hybrid_project("hybrid-test", &configs.to_string());
    let dump = proj.root.join("dumps");
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    assert!(common::run_binary_extra(
        &proj.root,
        "os-hyb",
        "hybrid-test",
        "http://127.0.0.1",
        &[
            ("OPENSEARCH_PORT", &port),
            ("OPENSEARCH_INDEX", "bench_hybrid")
        ],
        &[
            "--dump-results",
            dump.to_str().unwrap(),
            "--fail-on-dropped-queries"
        ]
    ));
    let evidence = common::assert_hybrid_run(&proj, "os-hyb", &dump, 0.8, "vector");
    for e in &evidence {
        // Exact on this dedicated single-node container: 12 queries plus one
        // prime per worker (parallel 2). Production only requires >= answered.
        assert_eq!(e["knn_query_with_filter_requests_delta"], 14, "{e}");
    }
    let results = common::read_results_obj(&proj.root, "os-hyb");
    assert!(results["hybrid"]["engine"]["exact_search_fallback"]
        .as_str()
        .unwrap()
        .starts_with("not observable"));
    std::fs::remove_dir_all(proj.root).unwrap();
}

/// #341 x #344: hybrid and filter-only in ONE engine config share one index,
/// built once in the `full` layout. Hybrid's filter still reaches the k-NN
/// query in every bucket; filter-only passes its per-query count gate (and its
/// requests carry no vector clause, checked before timing); both search files
/// and the hybrid dump record `full`.
#[test]
fn test_binary_opensearch_hybrid_and_filter_only_share_one_full_index() {
    wait_for_opensearch();
    let config = "os-hyb-fo";
    let index = "bench_hybrid_full";
    let configs = serde_json::json!([{
        "name": config, "engine": "opensearch",
        "collection_params": {"number_of_shards": 1},
        "search_params": [
            {"parallel": 2, "top": 10, "workload": "hybrid", "search_params": {"ef": 200}},
            {"parallel": 2, "top": 10, "workload": "filter-only"}],
        "upload_params": {"parallel": 2, "batch_size": 500}
    }]);
    let proj = common::write_msmarco_hybrid_project("hybrid-test", &configs.to_string());
    let counts = common::add_hybrid_filter_queries(&proj);
    let dump = proj.root.join("dumps");
    let port = std::env::var("OPENSEARCH_PORT").unwrap_or_else(|_| OS_PORT.to_string());
    let out = common::run_binary_output(
        &proj.root,
        config,
        "hybrid-test",
        "http://127.0.0.1",
        &[("OPENSEARCH_PORT", &port), ("OPENSEARCH_INDEX", index)],
        &[
            "--dump-results",
            dump.to_str().unwrap(),
            "--fail-on-dropped-queries",
            "--keep-data",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let evidence = common::assert_hybrid_run(&proj, config, &dump, 0.8, "full");
    for e in &evidence {
        assert!(
            e["knn_query_with_filter_requests_delta"].as_u64().unwrap() >= 12,
            "{e}"
        );
    }
    common::assert_hybrid_mix_filter_only(&proj, config, &stdout, &counts);
    let mapping: serde_json::Value = os_client()
        .get(format!("{}/{index}/_mapping", os_base_url()))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let props = &mapping[index]["mappings"]["properties"];
    assert_eq!(props["vector"]["type"], "knn_vector", "{props}");
    assert_eq!(props["title"]["type"], "text", "{props}");
    assert_eq!(props["category"]["type"], "keyword", "{props}");
    assert_eq!(props["tier"]["type"], "long", "{props}");
    let _ = os_client()
        .delete(format!("{}/{index}", os_base_url()))
        .send();
    std::fs::remove_dir_all(&proj.root).ok();
}
