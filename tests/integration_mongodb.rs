//! Integration tests for the MongoDB engine.
//!
//! Requires MongoDB 8.x with Atlas Search running on port 27018 (replica set).
//! Start with: docker compose -f tests/docker-compose.test.yml up -d mongodb-search --wait
//! Run with:   MONGODB_PORT=27018 cargo test --test integration_mongodb --release -- --test-threads=1

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use mongodb::bson::{doc, Document};
use mongodb::sync::Client;

mod common;
use rand::Rng;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MONGODB_PORT: u16 = 27018;
const MONGODB_HOST: &str = "127.0.0.1";
const TEST_DB: &str = "bench_test";
const TEST_COLLECTION: &str = "vectors";
const TEST_INDEX: &str = "vector_index";

fn mongodb_uri() -> String {
    let port: u16 = std::env::var("MONGODB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MONGODB_PORT);
    format!("mongodb://{}:{}/?directConnection=true", MONGODB_HOST, port)
}

fn mongodb_client() -> Client {
    Client::with_uri_str(mongodb_uri()).expect("Failed to create MongoDB client")
}

fn wait_for_mongodb() {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if let Ok(client) = Client::with_uri_str(mongodb_uri()) {
            let db = client.database("admin");
            if db.run_command(doc! { "ping": 1 }).run().is_ok() {
                return;
            }
        }
        if Instant::now() > deadline {
            panic!("MongoDB not available on port {} after 120s", MONGODB_PORT);
        }
        thread::sleep(Duration::from_millis(1000));
    }
}

/// Drop the search index (if any), wait for it to disappear, then drop the
/// collection and wait for it to be gone.  Mirrors the engine's configure()
/// cleanup so tests exercise the same Atlas-safe path.
fn drop_test_collection() {
    let client = mongodb_client();
    let db = client.database(TEST_DB);

    // Drop search index explicitly
    let _ = db
        .run_command(doc! {
            "dropSearchIndex": TEST_COLLECTION,
            "name": TEST_INDEX,
        })
        .run();

    // Wait for index to disappear
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let cmd = doc! { "listSearchIndexes": TEST_COLLECTION };
        let index_exists = db.run_command(cmd).run().ok().is_some_and(|result| {
            result
                .get_document("cursor")
                .ok()
                .and_then(|c| c.get_array("firstBatch").ok())
                .is_some_and(|batch| {
                    batch.iter().any(|idx| {
                        idx.as_document().and_then(|d| d.get_str("name").ok()) == Some(TEST_INDEX)
                    })
                })
        });
        if !index_exists {
            break;
        }
        if Instant::now() > deadline {
            eprintln!("Warning: search index still exists after 60s");
            break;
        }
        thread::sleep(Duration::from_secs(2));
    }

    // Drop collection
    let coll = db.collection::<Document>(TEST_COLLECTION);
    let _ = coll.drop().run();

    // Wait for collection to disappear
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let names = db.list_collection_names().run().unwrap_or_default();
        if !names.contains(&TEST_COLLECTION.to_string()) {
            break;
        }
        if Instant::now() > deadline {
            break;
        }
        thread::sleep(Duration::from_secs(2));
    }
}

fn generate_test_vectors(count: usize, dim: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let ids: Vec<i64> = (0..count as i64).collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (ids, vectors)
}

fn insert_vectors(client: &Client, ids: &[i64], vectors: &[Vec<f32>]) {
    let coll = client
        .database(TEST_DB)
        .collection::<Document>(TEST_COLLECTION);

    let docs: Vec<Document> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(&id, vec)| {
            let bson_vec: Vec<mongodb::bson::Bson> = vec
                .iter()
                .map(|&f| mongodb::bson::Bson::Double(f as f64))
                .collect();
            doc! { "_id": id, "vector": bson_vec }
        })
        .collect();

    coll.insert_many(docs).run().expect("Insert failed");
}

fn create_vector_index(client: &Client, dim: usize, similarity: &str) {
    let db = client.database(TEST_DB);

    // Create collection if not exists
    let _ = db.create_collection(TEST_COLLECTION).run();

    // Insert a dummy doc so index has data
    let coll = db.collection::<Document>(TEST_COLLECTION);
    let dummy: Vec<mongodb::bson::Bson> =
        (0..dim).map(|_| mongodb::bson::Bson::Double(0.0)).collect();
    let _ = coll
        .insert_one(doc! { "_id": -1i64, "vector": dummy })
        .run();

    // Create search index
    let index_def = doc! {
        "name": TEST_INDEX,
        "type": "vectorSearch",
        "definition": {
            "fields": [{
                "type": "vector",
                "path": "vector",
                "numDimensions": dim as i32,
                "similarity": similarity,
            }]
        }
    };

    let cmd = doc! {
        "createSearchIndexes": TEST_COLLECTION,
        "indexes": [index_def],
    };

    db.run_command(cmd)
        .run()
        .expect("Failed to create vector search index");

    // Wait for index readiness
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let cmd = doc! { "listSearchIndexes": TEST_COLLECTION };
        if let Ok(result) = db.run_command(cmd).run() {
            if let Ok(cursor) = result.get_document("cursor") {
                if let Ok(batch) = cursor.get_array("firstBatch") {
                    for index in batch {
                        if let Some(index_doc) = index.as_document() {
                            let name = index_doc.get_str("name").unwrap_or("");
                            let status = index_doc.get_str("status").unwrap_or("");
                            let queryable = index_doc.get_bool("queryable").unwrap_or(false);
                            if name == TEST_INDEX
                                && (status == "READY" || status == "ACTIVE")
                                && queryable
                            {
                                // Remove dummy
                                let _ = coll.delete_one(doc! { "_id": -1i64 }).run();
                                return;
                            }
                        }
                    }
                }
            }
        }
        if Instant::now() > deadline {
            panic!("Vector search index did not become ready within 120s");
        }
        thread::sleep(Duration::from_secs(1));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_mongodb_collection_crud() {
    wait_for_mongodb();
    drop_test_collection();

    let client = mongodb_client();
    let db = client.database(TEST_DB);

    // Create collection
    db.create_collection(TEST_COLLECTION)
        .run()
        .expect("Failed to create collection");

    // Insert a document
    let coll = db.collection::<Document>(TEST_COLLECTION);
    coll.insert_one(doc! { "_id": 1i64, "value": "test" })
        .run()
        .expect("Failed to insert");

    // Count documents
    let count = coll
        .count_documents(doc! {})
        .run()
        .expect("Failed to count");
    assert_eq!(count, 1);

    // Drop
    drop_test_collection();

    // Verify empty
    let count = coll.count_documents(doc! {}).run().unwrap_or(0);
    assert_eq!(count, 0);
}

#[test]
fn test_mongodb_insert_and_search() {
    wait_for_mongodb();
    drop_test_collection();

    let client = mongodb_client();
    let dim = 4;

    create_vector_index(&client, dim, "euclidean");

    // Insert known vectors
    let ids = vec![0i64, 1, 2, 3, 4];
    let vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    insert_vectors(&client, &ids, &vectors);

    // Wait for indexing
    thread::sleep(Duration::from_secs(2));

    // Vector search for [1, 0, 0, 0]
    let coll = client
        .database(TEST_DB)
        .collection::<Document>(TEST_COLLECTION);

    let pipeline = vec![
        doc! {
            "$vectorSearch": {
                "index": TEST_INDEX,
                "path": "vector",
                "queryVector": [1.0f64, 0.0, 0.0, 0.0],
                "numCandidates": 20i64,
                "limit": 3i64,
            }
        },
        doc! {
            "$project": {
                "_id": 1,
                "score": { "$meta": "vectorSearchScore" },
            }
        },
    ];

    let cursor = coll.aggregate(pipeline).run().expect("Search failed");
    let results: Vec<Document> = cursor.filter_map(|r| r.ok()).collect();
    assert!(!results.is_empty(), "Expected search results");

    // First result should be id=0 (exact match)
    let first_id = results[0].get_i64("_id").unwrap();
    assert_eq!(first_id, 0, "First result should be exact match");

    drop_test_collection();
}

#[test]
fn test_mongodb_precision() {
    wait_for_mongodb();
    drop_test_collection();

    let client = mongodb_client();
    let dim = 8;
    let n = 200;
    let k = 10;

    create_vector_index(&client, dim, "euclidean");

    let (ids, vectors) = generate_test_vectors(n, dim);
    insert_vectors(&client, &ids, &vectors);

    // Wait for indexing
    thread::sleep(Duration::from_secs(3));

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

    // Vector search
    let coll = client
        .database(TEST_DB)
        .collection::<Document>(TEST_COLLECTION);

    let bson_query: Vec<mongodb::bson::Bson> = query
        .iter()
        .map(|&f| mongodb::bson::Bson::Double(f as f64))
        .collect();

    let pipeline = vec![
        doc! {
            "$vectorSearch": {
                "index": TEST_INDEX,
                "path": "vector",
                "queryVector": bson_query,
                "numCandidates": (k * 20) as i64,
                "limit": k as i64,
            }
        },
        doc! {
            "$project": {
                "_id": 1,
                "score": { "$meta": "vectorSearchScore" },
            }
        },
    ];

    let cursor = coll.aggregate(pipeline).run().expect("Search failed");
    let results: Vec<Document> = cursor.filter_map(|r| r.ok()).collect();

    let found: std::collections::HashSet<i64> = results
        .iter()
        .filter_map(|doc| doc.get_i64("_id").ok())
        .collect();

    let overlap = ground_truth.intersection(&found).count();
    let precision = overlap as f64 / k as f64;
    println!(
        "MongoDB euclidean precision@{}: {:.2} ({}/{})",
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
fn test_mongodb_full_cycle() {
    wait_for_mongodb();
    drop_test_collection();

    let client = mongodb_client();
    let dim = 4;

    // Create + index
    create_vector_index(&client, dim, "euclidean");

    // Upload
    let (ids, vectors) = generate_test_vectors(20, dim);
    insert_vectors(&client, &ids, &vectors);
    thread::sleep(Duration::from_secs(2));

    // Search
    let coll = client
        .database(TEST_DB)
        .collection::<Document>(TEST_COLLECTION);
    let bson_query: Vec<mongodb::bson::Bson> = vectors[0]
        .iter()
        .map(|&f| mongodb::bson::Bson::Double(f as f64))
        .collect();

    let pipeline = vec![
        doc! {
            "$vectorSearch": {
                "index": TEST_INDEX,
                "path": "vector",
                "queryVector": bson_query,
                "numCandidates": 50i64,
                "limit": 5i64,
            }
        },
        doc! {
            "$project": {
                "_id": 1,
                "score": { "$meta": "vectorSearchScore" },
            }
        },
    ];
    let cursor = coll.aggregate(pipeline).run().expect("Search failed");
    let results: Vec<Document> = cursor.filter_map(|r| r.ok()).collect();
    assert_eq!(results.len(), 5);

    // Delete
    drop_test_collection();

    let count = coll.count_documents(doc! {}).run().unwrap_or(0);
    assert_eq!(count, 0);
}

/// Two back-to-back benchmark cycles with different dimensions.
/// Verifies that index cleanup between runs is correct — the second run
/// must create a fresh index with a different dimension and still return
/// accurate results.
#[test]
fn test_mongodb_multi_dataset_runs() {
    wait_for_mongodb();
    drop_test_collection();

    let client = mongodb_client();

    // ── Run 1: dim=4, euclidean, 20 vectors ───────────────────────
    println!("=== Run 1: dim=4, euclidean ===");
    {
        let dim = 4;
        create_vector_index(&client, dim, "euclidean");

        let (ids, vectors) = generate_test_vectors(20, dim);
        insert_vectors(&client, &ids, &vectors);
        thread::sleep(Duration::from_secs(2));

        let coll = client
            .database(TEST_DB)
            .collection::<Document>(TEST_COLLECTION);
        let bson_query: Vec<mongodb::bson::Bson> = vectors[0]
            .iter()
            .map(|&f| mongodb::bson::Bson::Double(f as f64))
            .collect();

        let pipeline = vec![
            doc! {
                "$vectorSearch": {
                    "index": TEST_INDEX,
                    "path": "vector",
                    "queryVector": bson_query,
                    "numCandidates": 50i64,
                    "limit": 5i64,
                }
            },
            doc! {
                "$project": {
                    "_id": 1,
                    "score": { "$meta": "vectorSearchScore" },
                }
            },
        ];
        let cursor = coll.aggregate(pipeline).run().expect("Run 1 search failed");
        let results: Vec<Document> = cursor.filter_map(|r| r.ok()).collect();
        assert_eq!(results.len(), 5, "Run 1: expected 5 results");
        let first_id = results[0].get_i64("_id").unwrap();
        assert_eq!(
            first_id, ids[0],
            "Run 1: first result should be query vector"
        );
    }

    // ── Cleanup between runs (mirrors engine configure()) ─────────
    println!("=== Cleanup between runs ===");
    drop_test_collection();

    // ── Run 2: dim=8, cosine, 50 vectors ──────────────────────────
    println!("=== Run 2: dim=8, cosine ===");
    {
        let dim = 8;
        create_vector_index(&client, dim, "cosine");

        let (ids, vectors) = generate_test_vectors(50, dim);
        insert_vectors(&client, &ids, &vectors);
        thread::sleep(Duration::from_secs(2));

        let coll = client
            .database(TEST_DB)
            .collection::<Document>(TEST_COLLECTION);
        let bson_query: Vec<mongodb::bson::Bson> = vectors[0]
            .iter()
            .map(|&f| mongodb::bson::Bson::Double(f as f64))
            .collect();

        let pipeline = vec![
            doc! {
                "$vectorSearch": {
                    "index": TEST_INDEX,
                    "path": "vector",
                    "queryVector": bson_query,
                    "numCandidates": 100i64,
                    "limit": 10i64,
                }
            },
            doc! {
                "$project": {
                    "_id": 1,
                    "score": { "$meta": "vectorSearchScore" },
                }
            },
        ];
        let cursor = coll.aggregate(pipeline).run().expect("Run 2 search failed");
        let results: Vec<Document> = cursor.filter_map(|r| r.ok()).collect();
        assert_eq!(results.len(), 10, "Run 2: expected 10 results");

        // Verify doc count is from run 2 only (no leftover from run 1)
        let count = coll.count_documents(doc! {}).run().expect("count failed");
        assert_eq!(
            count, 50,
            "Run 2: should have exactly 50 docs, not leftovers from run 1"
        );
    }

    drop_test_collection();
}

// ---------------------------------------------------------------------------
// Binary end-to-end tests
// ---------------------------------------------------------------------------

/// Find the release binary path
fn binary_path() -> PathBuf {
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    path.push("vector-db-benchmark");
    if path.exists() {
        return path;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/release/vector-db-benchmark")
}

/// Create a temporary project directory with dataset + engine config.
fn create_test_project(
    dataset_name: &str,
    engine_configs_json: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    neighbors: &[Vec<i64>],
    distance: &str,
    dim: usize,
) -> PathBuf {
    let tmp = tempfile::tempdir().expect("Failed to create temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

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
        "distance": distance,
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

/// Parse a search result JSON and return mean_precision_at_returned
fn read_search_precision(results_dir: &PathBuf, engine_name: &str) -> f64 {
    let pattern = format!("{}-*-search-*.json", engine_name);
    let mut found = Vec::new();
    for entry in fs::read_dir(results_dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if glob::Pattern::new(&pattern).unwrap().matches(&name) {
            found.push(entry.path());
        }
    }
    assert!(
        !found.is_empty(),
        "No search result files found matching '{}'",
        pattern
    );

    let content = fs::read_to_string(&found[0]).unwrap();
    let result: serde_json::Value = serde_json::from_str(&content).unwrap();
    result["results"]["mean_precision_at_returned"]
        .as_f64()
        .expect("mean_precision_at_returned not found in result JSON")
}

/// Brute-force L2 nearest neighbors for building ground truth.
fn brute_force_neighbors_l2(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
    let mut dists: Vec<(i64, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f64 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
                .sum();
            (i as i64, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(top).map(|(id, _)| *id).collect()
}

/// Run the binary against MongoDB and return (stdout, stderr, success).
fn run_benchmark(
    project_root: &PathBuf,
    engine_name: &str,
    dataset_name: &str,
    port: u16,
) -> (String, String, bool) {
    let bin = binary_path();
    assert!(
        bin.exists(),
        "Binary not found at {:?}. Run `cargo build --release` first.",
        bin
    );

    let output = Command::new(&bin)
        .args([
            "--engines",
            engine_name,
            "--datasets",
            dataset_name,
            "--host",
            MONGODB_HOST,
            "--skip-if-exists",
            "false",
        ])
        .env("MONGODB_PORT", port.to_string())
        .env("MONGODB_DB", TEST_DB)
        .env("MONGODB_COLLECTION", TEST_COLLECTION)
        .env("MONGODB_INDEX_NAME", TEST_INDEX)
        .current_dir(project_root)
        .output()
        .expect("Failed to run vector-db-benchmark");

    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
        output.status.success(),
    )
}

/// End-to-end test: runs the actual vector-db-benchmark binary against MongoDB
/// with two different datasets back-to-back, verifying clean index recreation.
#[test]
fn test_binary_mongodb_multi_dataset() {
    wait_for_mongodb();
    drop_test_collection();

    let port: u16 = std::env::var("MONGODB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MONGODB_PORT);

    let engine_name = "test-mongodb";
    let engine_config = serde_json::json!([{
        "name": engine_name,
        "engine": "mongodb",
        "connection_params": {},
        "collection_params": {},
        "search_params": [{
            "parallel": 1,
            "num_candidates": 20,
        }],
        "upload_params": {
            "parallel": 1,
            "batch_size": 50
        }
    }]);
    let engine_json = serde_json::to_string_pretty(&engine_config).unwrap();

    // ── Run 1: 50 vectors, dim=8, euclidean ─────────────────────
    println!("=== Binary run 1: dim=8, euclidean ===");
    let dim1 = 8;
    let count1 = 50;
    let top = 5;
    let (_, vectors1) = generate_test_vectors(count1, dim1);
    let queries1: Vec<Vec<f32>> = vectors1[..5].to_vec();
    let neighbors1: Vec<Vec<i64>> = queries1
        .iter()
        .map(|q| brute_force_neighbors_l2(q, &vectors1, top))
        .collect();

    let project1 = create_test_project(
        "test-euclidean",
        &engine_json,
        &vectors1,
        &queries1,
        &neighbors1,
        "l2",
        dim1,
    );

    let (stdout, stderr, success) = run_benchmark(&project1, engine_name, "test-euclidean", port);
    println!("stdout:\n{}", stdout);
    if !stderr.is_empty() {
        println!("stderr:\n{}", stderr);
    }
    assert!(
        success,
        "Run 1 failed.\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    let precision1 = read_search_precision(&project1.join("results"), engine_name);
    println!("Run 1 precision: {:.4}", precision1);
    assert!(
        precision1 >= 0.8,
        "Run 1 precision should be >= 0.8, got {:.4}",
        precision1
    );

    // ── Run 2: 80 vectors, dim=16, cosine (different dataset, same engine) ──
    // This exercises full cleanup: drop index → wait → drop collection → wait → recreate
    println!("\n=== Binary run 2: dim=16, cosine ===");
    let dim2 = 16;
    let count2 = 80;
    let (_, vectors2) = generate_test_vectors(count2, dim2);
    let queries2: Vec<Vec<f32>> = vectors2[..5].to_vec();
    let neighbors2: Vec<Vec<i64>> = queries2
        .iter()
        .map(|q| brute_force_neighbors_l2(q, &vectors2, top))
        .collect();

    let project2 = create_test_project(
        "test-cosine",
        &engine_json,
        &vectors2,
        &queries2,
        &neighbors2,
        "cosine",
        dim2,
    );

    let (stdout, stderr, success) = run_benchmark(&project2, engine_name, "test-cosine", port);
    println!("stdout:\n{}", stdout);
    if !stderr.is_empty() {
        println!("stderr:\n{}", stderr);
    }
    assert!(
        success,
        "Run 2 failed.\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    // Verify the collection has exactly count2 documents (no leftovers from run 1)
    let client = mongodb_client();
    let coll = client
        .database(TEST_DB)
        .collection::<Document>(TEST_COLLECTION);
    // Collection may already be dropped by engine.delete(), which is fine
    let doc_count = coll.count_documents(doc! {}).run().unwrap_or(0);
    assert!(
        doc_count == 0 || doc_count == count2 as u64,
        "Expected 0 (deleted) or {} docs, got {} — stale data from run 1?",
        count2,
        doc_count
    );

    drop_test_collection();

    // Cleanup temp dirs
    let _ = fs::remove_dir_all(&project1);
    let _ = fs::remove_dir_all(&project2);
}

/// Read a specific field from the search results JSON.
fn read_search_result_field(
    results_dir: &PathBuf,
    engine_name: &str,
    field: &str,
) -> Option<serde_json::Value> {
    let pattern = format!("{}-*-search-*.json", engine_name);
    for entry in fs::read_dir(results_dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if glob::Pattern::new(&pattern).unwrap().matches(&name) {
            let content = fs::read_to_string(entry.path()).unwrap();
            let result: serde_json::Value = serde_json::from_str(&content).unwrap();
            return result["results"].get(field).cloned();
        }
    }
    None
}

/// End-to-end test: runs the binary with --update-search-ratio against MongoDB.
#[test]
fn test_binary_mongodb_mixed_benchmark() {
    wait_for_mongodb();
    drop_test_collection();

    let port: u16 = std::env::var("MONGODB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MONGODB_PORT);

    let dim = 16;
    let count = 100;
    let top = 5;
    let (_, vectors) = generate_test_vectors(count, dim);

    let queries: Vec<Vec<f32>> = vectors[..10].to_vec();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors_l2(q, &vectors, top))
        .collect();

    let engine_name = "test-mongodb-mixed";
    let engine_config = serde_json::json!([{
        "name": engine_name,
        "engine": "mongodb",
        "connection_params": {},
        "collection_params": {},
        // parallel: 1 — this test asserts an EXACT update_count (2), which only
        // holds single-threaded (the mixed loop's `break 'outer` makes the update
        // count interleaving-dependent at parallel > 1). The multi-worker
        // join-merge is covered by test_binary_mongodb_mixed_parallel.
        "search_params": [{
            "parallel": 1,
            "num_candidates": 20,
            "top": top,
        }],
        "upload_params": {
            "parallel": 1,
            "batch_size": 50
        }
    }]);

    let project_root = create_test_project(
        "test-mixed",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        "l2",
        dim,
    );

    let bin = binary_path();
    assert!(bin.exists(), "Binary not found at {:?}", bin);

    // Run with --update-search-ratio 1:5 (10 queries → 2 update cycles)
    let output = Command::new(&bin)
        .args([
            "--engines",
            engine_name,
            "--datasets",
            "test-mixed",
            "--host",
            MONGODB_HOST,
            "--update-search-ratio",
            "1:5",
        ])
        .env("MONGODB_PORT", port.to_string())
        .env("MONGODB_DB", TEST_DB)
        .env("MONGODB_COLLECTION", TEST_COLLECTION)
        .env("MONGODB_INDEX_NAME", TEST_INDEX)
        .current_dir(&project_root)
        .output()
        .expect("Failed to run vector-db-benchmark");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("stdout:\n{}", stdout);
    if !stderr.is_empty() {
        println!("stderr:\n{}", stderr);
    }

    assert!(
        output.status.success(),
        "Binary failed.\nstdout: {}\nstderr: {}",
        stdout,
        stderr
    );

    // Verify stdout mentions mixed mode
    assert!(
        stdout.contains("Mixed Search+Update"),
        "Expected 'Mixed Search+Update' in output.\nstdout: {}",
        stdout,
    );

    // Verify results JSON has update metrics
    let results_dir = project_root.join("results");

    let update_count = read_search_result_field(&results_dir, engine_name, "update_count");
    assert_eq!(
        update_count,
        Some(serde_json::json!(2)),
        "Expected update_count=2 in results JSON, got {:?}",
        update_count
    );

    let ratio = read_search_result_field(&results_dir, engine_name, "update_search_ratio");
    assert_eq!(
        ratio,
        Some(serde_json::json!("1:5")),
        "Expected update_search_ratio='1:5' in results JSON, got {:?}",
        ratio
    );

    let update_rps = read_search_result_field(&results_dir, engine_name, "update_rps");
    assert!(
        update_rps.is_some() && update_rps.unwrap().as_f64().unwrap() > 0.0,
        "Expected update_rps > 0 in results JSON"
    );

    // Precision should still be valid
    let precision = read_search_precision(&results_dir, engine_name);
    assert!(
        precision >= 0.8,
        "Mixed benchmark precision should be >= 0.8, got {}",
        precision
    );

    drop_test_collection();
    fs::remove_dir_all(&project_root).ok();
}

/// End-to-end MIXED harness at `parallel: 4` over a 2000-query fixture, so many
/// full search phases (and updates) run and the per-worker thread-local sample
/// buffers are merged across threads (the join-merge path — the actual rewrite).
/// Complements `test_binary_mongodb_mixed_benchmark` (parallel: 1, exact
/// update_count): here we assert recall/precision are intact, updates ran
/// (`update_count > 0`, `update_rps > 0`), and search percentiles are monotone.
#[test]
fn test_binary_mongodb_mixed_parallel() {
    wait_for_mongodb();
    drop_test_collection();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "mongo-mx", "engine": "mongodb",
        "connection_params": {}, "collection_params": {},
        "search_params": [{"parallel": 4, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project_n(
        "mongo-mx-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
        2000,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = std::env::var("MONGODB_PORT").unwrap_or_else(|_| MONGODB_PORT.to_string());
    assert!(
        common::run_binary_extra(
            &proj.root,
            "mongo-mx",
            "mongo-mx-test",
            MONGODB_HOST,
            &[
                ("MONGODB_PORT", port.as_str()),
                ("MONGODB_DB", TEST_DB),
                ("MONGODB_COLLECTION", TEST_COLLECTION),
                ("MONGODB_INDEX_NAME", TEST_INDEX),
            ],
            &["--update-search-ratio", "1:5", "--repetitions", "1"],
        ),
        "mongodb mixed (parallel) run failed"
    );

    let r = common::read_results_obj(&proj.root, "mongo-mx");
    let recall = r["mean_recall"].as_f64().unwrap();
    let precision = r["mean_precision_at_returned"].as_f64().unwrap();
    let update_count = r["update_count"].as_u64().unwrap();
    let update_rps = r["update_rps"].as_f64().unwrap();
    let p50 = r["p50_time"].as_f64().unwrap();
    let p95 = r["p95_time"].as_f64().unwrap();
    let p99 = r["p99_time"].as_f64().unwrap();
    println!(
        "mongodb mixed (parallel=4): recall={recall:.3} precision={precision:.3} \
         update_count={update_count} update_rps={update_rps:.1} p50={p50} p95={p95} p99={p99}"
    );
    assert!(precision >= 0.8, "mixed precision {precision} < 0.8");
    assert!(recall >= 0.9, "mixed recall {recall} < 0.9");
    assert!(update_count > 0, "mixed run performed no updates");
    assert!(update_rps > 0.0, "update_rps should be positive");
    assert!(
        p50 <= p95 && p95 <= p99,
        "percentiles must be monotone: p50={p50} p95={p95} p99={p99}"
    );
    drop_test_collection();
    fs::remove_dir_all(&proj.root).ok();
}

/// End-to-end FILTER-ONLY harness (`--skip-vector-index`) at `parallel: 4` with
/// `--queries 1000`: MongoDB has no `check_commandstats` backstop, so this is the
/// primary guard that failed `$vectorSearch`/`find` calls are counted (not folded
/// into RPS/percentiles). Asserts the filter-only sentinel (`mean_precision_at_returned ==
/// -1`), full query accounting (requested == succeeded, failed == 0) on a healthy
/// run, positive RPS, and monotone linear percentiles, with the per-worker sample
/// buffers merged across threads.
#[test]
fn test_binary_mongodb_filter_only() {
    wait_for_mongodb();
    drop_test_collection();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "mongo-fo", "engine": "mongodb",
        "connection_params": {}, "collection_params": {},
        "search_params": [{"parallel": 4, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "mongo-fo-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = std::env::var("MONGODB_PORT").unwrap_or_else(|_| MONGODB_PORT.to_string());
    assert!(
        common::run_binary_extra(
            &proj.root,
            "mongo-fo",
            "mongo-fo-test",
            MONGODB_HOST,
            &[
                ("MONGODB_PORT", port.as_str()),
                ("MONGODB_DB", TEST_DB),
                ("MONGODB_COLLECTION", TEST_COLLECTION),
                ("MONGODB_INDEX_NAME", TEST_INDEX),
            ],
            &["--skip-vector-index", "--queries", "1000"],
        ),
        "mongodb filter-only run failed"
    );

    let r = common::read_results_obj(&proj.root, "mongodb-no-vector");
    let mp = r["mean_precision_at_returned"].as_f64().unwrap();
    let rps = r["rps"].as_f64().unwrap();
    let p50 = r["p50_time"].as_f64().unwrap();
    let p95 = r["p95_time"].as_f64().unwrap();
    let p99 = r["p99_time"].as_f64().unwrap();
    let requested = r["requested_queries"].as_u64().unwrap();
    let succeeded = r["succeeded_queries"].as_u64().unwrap();
    let failed = r["failed_queries"].as_u64().unwrap();
    println!(
        "mongodb filter-only: mean_precision_at_returned={mp} rps={rps:.1} p50={p50} p95={p95} p99={p99} \
         requested={requested} succeeded={succeeded} failed={failed}"
    );
    assert_eq!(mp, -1.0, "filter-only sentinel lost");
    assert_eq!(requested, 1000, "requested_queries");
    assert_eq!(failed, 0, "healthy run must have no failed queries");
    assert_eq!(succeeded, 1000, "all queries should succeed");
    assert!(rps > 0.0, "rps should be positive");
    assert!(
        p50 <= p95 && p95 <= p99,
        "percentiles must be monotone: p50={p50} p95={p95} p99={p99}"
    );
    drop_test_collection();
    fs::remove_dir_all(&proj.root).ok();
}

/// End-to-end `match_any`: filter a keyword field to an OR-set and assert the
/// engine returns the filtered nearest neighbours (recall vs ground truth
/// brute-forced over only the matching docs). Proves the `$in` filter arm.
#[test]
fn test_binary_mongodb_match_any() {
    wait_for_mongodb();
    drop_test_collection();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "mongo-ma", "engine": "mongodb",
        "connection_params": {}, "collection_params": {},
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

    let port = std::env::var("MONGODB_PORT").unwrap_or_else(|_| MONGODB_PORT.to_string());
    assert!(
        common::run_binary(
            &proj.root,
            "mongo-ma",
            "match-any-test",
            MONGODB_HOST,
            &[
                ("MONGODB_PORT", port.as_str()),
                ("MONGODB_DB", TEST_DB),
                ("MONGODB_COLLECTION", TEST_COLLECTION),
                ("MONGODB_INDEX_NAME", TEST_INDEX),
            ],
        ),
        "mongodb match_any run failed"
    );

    let recall = common::read_recall(&proj.root, "mongo-ma");
    println!("mongodb match_any recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "mongodb match_any recall {:.3} < 0.9",
        recall
    );
}

/// UUID exact-match filter end-to-end. MongoDB stores `uuid` values as plain
/// BSON strings (no special schema type), so a `uid == UUIDS[0]` `$vectorSearch`
/// filter is a straight string equality. This confirms uuid works out of the box
/// (unlike ES/OS/pgvector/weaviate/milvus, which needed a schema-type fix).
#[test]
fn test_binary_mongodb_uuid() {
    wait_for_mongodb();
    drop_test_collection();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "mongo-uuid", "engine": "mongodb",
        "connection_params": {}, "collection_params": {},
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_uuid_project("uuid-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);

    let port = std::env::var("MONGODB_PORT").unwrap_or_else(|_| MONGODB_PORT.to_string());
    assert!(
        common::run_binary(
            &proj.root,
            "mongo-uuid",
            "uuid-test",
            MONGODB_HOST,
            &[
                ("MONGODB_PORT", port.as_str()),
                ("MONGODB_DB", TEST_DB),
                ("MONGODB_COLLECTION", TEST_COLLECTION),
                ("MONGODB_INDEX_NAME", TEST_INDEX),
            ],
        ),
        "mongodb uuid run failed"
    );

    let recall = common::read_recall(&proj.root, "mongo-uuid");
    println!("mongodb uuid recall={:.3}", recall);
    assert!(recall >= 0.9, "mongodb uuid recall {:.3} < 0.9", recall);
    drop_test_collection();
    fs::remove_dir_all(&proj.root).ok();
}

/// Multi-condition AND (keyword match AND numeric range) — verifies MongoDB
/// composes two conditions of different types into one `$vectorSearch` filter
/// (`color == "red"` AND `size >= 50` via `$and`/`$gte`), not just a single
/// clause. Recall is brute-forced over only the intersecting docs, so an engine
/// that drops or mis-joins either clause scores low.
#[test]
fn test_binary_mongodb_and_filter() {
    wait_for_mongodb();
    drop_test_collection();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "mongo-and", "engine": "mongodb",
        "connection_params": {}, "collection_params": {},
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_and_filter_project(
        "and-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = std::env::var("MONGODB_PORT").unwrap_or_else(|_| MONGODB_PORT.to_string());
    assert!(
        common::run_binary(
            &proj.root,
            "mongo-and",
            "and-test",
            MONGODB_HOST,
            &[
                ("MONGODB_PORT", port.as_str()),
                ("MONGODB_DB", TEST_DB),
                ("MONGODB_COLLECTION", TEST_COLLECTION),
                ("MONGODB_INDEX_NAME", TEST_INDEX),
            ],
        ),
        "mongodb and-filter run failed"
    );

    let recall = common::read_recall(&proj.root, "mongo-and");
    println!("mongodb and-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "mongodb and-filter recall {:.3} < 0.9",
        recall
    );
    drop_test_collection();
    fs::remove_dir_all(&proj.root).ok();
}

/// Multi-condition OR (`color == "red" OR size >= 90`) — verifies MongoDB unions
/// two clauses into a `$vectorSearch` filter `$or`, searching the union (not the
/// intersection). Recall is brute-forced over the union, so an engine that
/// mis-joins or drops an arm scores low.
#[test]
fn test_binary_mongodb_or_filter() {
    wait_for_mongodb();
    drop_test_collection();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "mongo-or", "engine": "mongodb",
        "connection_params": {}, "collection_params": {},
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_or_filter_project("or-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);

    let port = std::env::var("MONGODB_PORT").unwrap_or_else(|_| MONGODB_PORT.to_string());
    assert!(
        common::run_binary(
            &proj.root,
            "mongo-or",
            "or-test",
            MONGODB_HOST,
            &[
                ("MONGODB_PORT", port.as_str()),
                ("MONGODB_DB", TEST_DB),
                ("MONGODB_COLLECTION", TEST_COLLECTION),
                ("MONGODB_INDEX_NAME", TEST_INDEX),
            ],
        ),
        "mongodb or-filter run failed"
    );

    let recall = common::read_recall(&proj.root, "mongo-or");
    println!("mongodb or-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "mongodb or-filter recall {:.3} < 0.9",
        recall
    );
    drop_test_collection();
    fs::remove_dir_all(&proj.root).ok();
}

/// Nested/grouped boolean filter — `(color==red AND size>=50) OR (color==blue
/// AND size<10)` — verifies MongoDB builds the nested `{$or:[{$and:...},
/// {$and:...}]}` filter natively rather than mis-flattening the two AND groups.
/// Ground truth is brute-forced over the nested union, so a builder that
/// flattens the tree matches a wildly different set and recall collapses.
#[test]
fn test_binary_mongodb_nested_filter() {
    wait_for_mongodb();
    drop_test_collection();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "mongo-nested", "engine": "mongodb",
        "connection_params": {}, "collection_params": {},
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_nested_filter_project(
        "nested-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = std::env::var("MONGODB_PORT").unwrap_or_else(|_| MONGODB_PORT.to_string());
    assert!(
        common::run_binary(
            &proj.root,
            "mongo-nested",
            "nested-test",
            MONGODB_HOST,
            &[
                ("MONGODB_PORT", port.as_str()),
                ("MONGODB_DB", TEST_DB),
                ("MONGODB_COLLECTION", TEST_COLLECTION),
                ("MONGODB_INDEX_NAME", TEST_INDEX),
            ],
        ),
        "mongodb nested-filter run failed"
    );

    let recall = common::read_recall(&proj.root, "mongo-nested");
    println!("mongodb nested-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "mongodb nested-filter recall {:.3} < 0.9",
        recall
    );
    drop_test_collection();
    fs::remove_dir_all(&proj.root).ok();
}

/// End-to-end `match_any` on the INT `size` field. Proves the numeric `$in`
/// filter matches natively-stored integers end-to-end. Ground truth is
/// brute-forced over ONLY the docs whose `size` is in the IN-set (a strict
/// subset), so an engine that ignores the filter — or that emits an integer
/// `$in` against string-stored sizes (the HIGH bug) — scores low recall.
#[test]
fn test_binary_mongodb_match_any_int() {
    wait_for_mongodb();
    drop_test_collection();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "mongo-ma-int", "engine": "mongodb",
        "connection_params": {}, "collection_params": {},
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_int_project(
        "match-any-int-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    let port = std::env::var("MONGODB_PORT").unwrap_or_else(|_| MONGODB_PORT.to_string());
    assert!(
        common::run_binary(
            &proj.root,
            "mongo-ma-int",
            "match-any-int-test",
            MONGODB_HOST,
            &[
                ("MONGODB_PORT", port.as_str()),
                ("MONGODB_DB", TEST_DB),
                ("MONGODB_COLLECTION", TEST_COLLECTION),
                ("MONGODB_INDEX_NAME", TEST_INDEX),
            ],
        ),
        "mongodb match_any int run failed"
    );

    let recall = common::read_recall(&proj.root, "mongo-ma-int");
    println!("mongodb match_any INT recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "mongodb match_any int recall {:.3} < 0.9",
        recall
    );
}

// ---------------------------------------------------------------------------
// Issue #216: hnsw_config must actually reach the server
// ---------------------------------------------------------------------------

/// Read the vector index definition back from the server.
///
/// This is the only assertion that can prove a build-time knob took effect.
/// Recall/latency assertions cannot: HNSW is accurate enough on a small corpus
/// that a default index and a tuned index produce identical results, which is
/// precisely why issue #216 (all 96 MongoDB sweep rows measuring one default
/// index) survived undetected — "recall looks plausible throughout".
fn read_back_index_definition() -> Document {
    let client = mongodb_client();
    let db = client.database(TEST_DB);

    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let result = db
            .run_command(doc! { "listSearchIndexes": TEST_COLLECTION })
            .run();
        if let Ok(result) = result {
            let batch = result
                .get_document("cursor")
                .ok()
                .and_then(|c| c.get_array("firstBatch").ok())
                .cloned()
                .unwrap_or_default();
            for idx in batch {
                let Some(idx) = idx.as_document() else {
                    continue;
                };
                if idx.get_str("name").ok() != Some(TEST_INDEX) {
                    continue;
                }
                if let Ok(def) = idx.get_document("latestDefinition") {
                    return def.clone();
                }
            }
        }
        assert!(
            Instant::now() < deadline,
            "index '{}' never appeared in listSearchIndexes on {}.{}",
            TEST_INDEX,
            TEST_DB,
            TEST_COLLECTION
        );
        thread::sleep(Duration::from_secs(2));
    }
}

/// Pull the `hnswOptions` sub-document out of the vector field of a definition.
fn hnsw_options_of(definition: &Document) -> Option<Document> {
    definition
        .get_array("fields")
        .ok()?
        .iter()
        .filter_map(|f| f.as_document())
        .find(|f| f.get_str("type").ok() == Some("vector"))?
        .get_document("hnswOptions")
        .ok()
        .cloned()
}

/// Run the binary once with the given `collection_params`, keeping the index
/// alive (`--keep-data`) so the definition can be read back afterwards.
fn run_and_read_back_definition(
    engine_name: &str,
    collection_params: serde_json::Value,
    port: u16,
) -> (Document, PathBuf) {
    let engine_config = serde_json::json!([{
        "name": engine_name,
        "engine": "mongodb",
        "connection_params": {},
        "collection_params": collection_params,
        "search_params": [{ "parallel": 1, "num_candidates": 20 }],
        "upload_params": { "parallel": 1, "batch_size": 50 }
    }]);

    let dim = 8;
    let (_, vectors) = generate_test_vectors(40, dim);
    let queries: Vec<Vec<f32>> = vectors[..3].to_vec();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors_l2(q, &vectors, 5))
        .collect();

    let project = create_test_project(
        "test-hnswopts",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        "l2",
        dim,
    );

    let bin = binary_path();
    assert!(bin.exists(), "Binary not found at {:?}", bin);
    let output = Command::new(&bin)
        .args([
            "--engines",
            engine_name,
            "--datasets",
            "test-hnswopts",
            "--host",
            MONGODB_HOST,
            "--skip-if-exists",
            "false",
            // Keep the index so it can be inspected after the run; the default
            // cleanup drops it and takes the evidence with it.
            "--keep-data",
        ])
        .env("MONGODB_PORT", port.to_string())
        .env("MONGODB_DB", TEST_DB)
        .env("MONGODB_COLLECTION", TEST_COLLECTION)
        .env("MONGODB_INDEX_NAME", TEST_INDEX)
        .current_dir(&project)
        .output()
        .expect("Failed to run vector-db-benchmark");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        output.status.success(),
        "run for {} failed.\nstdout: {}\nstderr: {}",
        engine_name,
        stdout,
        stderr
    );

    (read_back_index_definition(), project)
}

/// Issue #216: `collection_params.hnsw_config` must reach the server as
/// `hnswOptions.{maxEdges,numEdgeCandidates}`.
///
/// The test asserts on the definition MongoDB itself reports, and pairs it with
/// a negative control (a config with no `hnsw_config` must produce an index with
/// NO `hnswOptions`). Without the control the assertion could pass against a
/// server that always reports some default, proving nothing.
///
/// IMPORTANT — why the tuned values are 32/200 and not 16/100:
/// **the server elides default-valued options from the definition it reports.**
/// The defaults are `maxEdges=16`, `numEdgeCandidates=100`, so `{16, 100}` reads
/// back with no `hnswOptions` at all, and `{16, 200}` reads back with only
/// `numEdgeCandidates`. "Absent from the read-back" therefore means "not sent OR
/// sent at the default" — the control distinguishes those two only because the
/// positive case deliberately uses NON-DEFAULT values on both knobs.
///
/// Consequence for anyone reparameterising this test: dropping to `M: 16` makes
/// the `maxEdges` assertion below fail against a perfectly correct engine. Keep
/// both values off their defaults.
#[test]
fn test_binary_mongodb_hnsw_config_reaches_server() {
    wait_for_mongodb();

    let port: u16 = std::env::var("MONGODB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MONGODB_PORT);

    // ── Negative control: no hnsw_config → no hnswOptions on the server ──
    drop_test_collection();
    let (default_def, project_default) =
        run_and_read_back_definition("test-mongodb-hnsw-default", serde_json::json!({}), port);
    println!("default index definition: {:?}", default_def);
    assert!(
        hnsw_options_of(&default_def).is_none(),
        "a config without hnsw_config must not send hnswOptions, but the server \
         reports {:?} — the positive assertion below would then prove nothing",
        default_def
    );

    // ── Positive: hnsw_config → hnswOptions, verified by read-back ──
    drop_test_collection();
    let (tuned_def, project_tuned) = run_and_read_back_definition(
        "test-mongodb-hnsw-tuned",
        serde_json::json!({ "hnsw_config": { "M": 32, "EF_CONSTRUCTION": 200 } }),
        port,
    );
    println!("tuned index definition: {:?}", tuned_def);

    let opts = hnsw_options_of(&tuned_def).unwrap_or_else(|| {
        panic!(
            "hnsw_config was declared but the server reports no hnswOptions: {:?} \
             — the knob is being silently discarded (issue #216)",
            tuned_def
        )
    });

    // Present only because 32 is not the default (16); see the note above.
    let max_edges = opts
        .get_i32("maxEdges")
        .map(|v| v as i64)
        .or_else(|_| opts.get_i64("maxEdges"))
        .expect("maxEdges missing from hnswOptions");
    let num_edge_candidates = opts
        .get_i32("numEdgeCandidates")
        .map(|v| v as i64)
        .or_else(|_| opts.get_i64("numEdgeCandidates"))
        .expect("numEdgeCandidates missing from hnswOptions");

    assert_eq!(
        max_edges, 32,
        "hnsw_config.M=32 must arrive as hnswOptions.maxEdges=32, server reports {:?}",
        opts
    );
    assert_eq!(
        num_edge_candidates, 200,
        "hnsw_config.EF_CONSTRUCTION=200 must arrive as \
         hnswOptions.numEdgeCandidates=200, server reports {:?}",
        opts
    );

    drop_test_collection();
    let _ = fs::remove_dir_all(&project_default);
    let _ = fs::remove_dir_all(&project_tuned);
}

/// MongoDB enforces `hnswOptions` bounds server-side (`maxEdges` in `[16..64]`,
/// `numEdgeCandidates` in `[100..3200]`). The engine forwards values verbatim
/// rather than clamping them, so an out-of-range config must FAIL LOUDLY instead
/// of quietly benchmarking a different index than the one requested.
#[test]
fn test_binary_mongodb_out_of_range_hnsw_config_fails_loudly() {
    wait_for_mongodb();
    drop_test_collection();

    let port: u16 = std::env::var("MONGODB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MONGODB_PORT);

    let engine_name = "test-mongodb-hnsw-oob";
    // EF_CONSTRUCTION=64 was in the ORIGINAL shipped sweep and is below
    // MongoDB's documented minimum of 100 — further proof those numbers were
    // never sent anywhere.
    let engine_config = serde_json::json!([{
        "name": engine_name,
        "engine": "mongodb",
        "connection_params": {},
        "collection_params": { "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 64 } },
        "search_params": [{ "parallel": 1, "num_candidates": 20 }],
        "upload_params": { "parallel": 1, "batch_size": 50 }
    }]);

    let dim = 8;
    let (_, vectors) = generate_test_vectors(40, dim);
    let queries: Vec<Vec<f32>> = vectors[..3].to_vec();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors_l2(q, &vectors, 5))
        .collect();

    let project = create_test_project(
        "test-hnsw-oob",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        "l2",
        dim,
    );

    let (stdout, stderr, success) = {
        let bin = binary_path();
        let output = Command::new(&bin)
            .args([
                "--engines",
                engine_name,
                "--datasets",
                "test-hnsw-oob",
                "--host",
                MONGODB_HOST,
                "--skip-if-exists",
                "false",
            ])
            .env("MONGODB_PORT", port.to_string())
            .env("MONGODB_DB", TEST_DB)
            .env("MONGODB_COLLECTION", TEST_COLLECTION)
            .env("MONGODB_INDEX_NAME", TEST_INDEX)
            .current_dir(&project)
            .output()
            .expect("Failed to run vector-db-benchmark");
        (
            String::from_utf8_lossy(&output.stdout).to_string(),
            String::from_utf8_lossy(&output.stderr).to_string(),
            output.status.success(),
        )
    };

    assert!(
        !success,
        "EF_CONSTRUCTION=64 is below MongoDB's minimum of 100; the run must fail \
         rather than silently benchmark a default index.\nstdout: {}\nstderr: {}",
        stdout, stderr
    );
    let combined = format!("{}{}", stdout, stderr);
    assert!(
        combined.contains("numEdgeCandidates"),
        "failure must name the rejected knob so the cause is obvious.\nstdout: {}\nstderr: {}",
        stdout,
        stderr
    );

    drop_test_collection();
    let _ = fs::remove_dir_all(&project);
}

/// Geo-radius end-to-end (issue #223).
///
/// Before this, `build_mongo_filter_entry`'s `_ => {}` dropped the geo leaf, so
/// `$vectorSearch` ran with NO `filter` while recall was scored against
/// geo-filtered ground truth; since #251 the same input is a hard error, so the
/// shipped `random-geo-radius-*-angular-filters` were unrunnable on MongoDB.
///
/// The fix is a different STAGE, not a different operator. `$vectorSearch`'s
/// `filter` is MQL restricted to ten comparison operators and rejects
/// `$geoWithin` outright (verified live: `"filter.loc" at least one of [$gt,
/// $gte, $lt, $lte, $eq, $ne, $in, $nin, $exists, $not] must be present`), and a
/// `vectorSearch`-type index has no geo field type. MongoDB's geo-capable vector
/// pre-filter is the `vectorSearch` OPERATOR inside a `$search` stage over a
/// `search`-type index, whose `filter` takes `geoWithin` with a `circle`. The
/// engine switches to that path for — and only for — a dataset whose schema
/// declares a `geo` field, so no other MongoDB number moves.
///
/// The fixture is the bounding-box-discriminating one: a box (or no filter at
/// all) scores ~0.25 against its ground truth, so ≥ 0.9 can only come from a
/// real radius. It is a PRE-filter, not a `$match` after the stage — a
/// post-filter would shrink the k results and could not reach 0.9 here.
#[test]
fn test_binary_mongodb_geo() {
    wait_for_mongodb();
    drop_test_collection();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "mongo-geo", "engine": "mongodb",
        "connection_params": {}, "collection_params": {},
        "search_params": [{"parallel": 1, "num_candidates": 400}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_geo_corner_project(
        "mongo-geo-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = std::env::var("MONGODB_PORT").unwrap_or_else(|_| MONGODB_PORT.to_string());
    // `--keep-data` so the index survives the run and can be read back below;
    // the test drops the collection itself at the end.
    assert!(
        common::run_binary_extra(
            &proj.root,
            "mongo-geo",
            "mongo-geo-test",
            MONGODB_HOST,
            &[
                ("MONGODB_PORT", port.as_str()),
                ("MONGODB_DB", TEST_DB),
                ("MONGODB_COLLECTION", TEST_COLLECTION),
                ("MONGODB_INDEX_NAME", TEST_INDEX),
            ],
            &["--keep-data"],
        ),
        "mongodb geo run failed"
    );

    // The index the engine actually built must be the `search`-type one — a
    // `vectorSearch`-type index here would mean the geo filter never had a
    // chance, and only the recall assertion below would (eventually) notice.
    let client = mongodb_client();
    let indexes: Vec<mongodb::bson::Document> = client
        .database(TEST_DB)
        .collection::<mongodb::bson::Document>(TEST_COLLECTION)
        .aggregate(vec![doc! { "$listSearchIndexes": {} }])
        .run()
        .expect("listSearchIndexes")
        .filter_map(Result::ok)
        .collect();
    let ours = indexes
        .iter()
        .find(|i| i.get_str("name").unwrap_or("") == TEST_INDEX)
        .unwrap_or_else(|| panic!("no index named {TEST_INDEX}: {indexes:?}"));
    assert_eq!(
        ours.get_str("type").unwrap_or(""),
        "search",
        "a geo dataset must build the search-type index: {ours:?}"
    );

    let recall = common::read_recall(&proj.root, "mongo-geo");
    println!("mongodb geo recall={recall:.3}");
    assert!(recall >= 0.9, "mongodb geo recall {recall:.3} < 0.9");
    drop_test_collection();
}
