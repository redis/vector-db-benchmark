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

/// The server reply the #293 guard reads on MongoDB.
///
/// `update_one_doc` returns "did this write match nothing" straight from
/// `UpdateResult::matched_count`, and `gate_update_attribution` rejects a mixed
/// run whose updates matched nothing. That guard is only as good as the driver
/// and server reporting `matched_count` honestly for a no-upsert update — a
/// server behaviour no unit test can pin, and one that would make the guard
/// silently vacuous if it regressed.
///
/// This test does NOT exercise the benchmark's mixed path; it pins the single
/// server fact that path depends on.
#[test]
fn test_update_one_matched_count_distinguishes_a_missing_document_from_an_update() {
    wait_for_mongodb();
    let client = mongodb_client();
    // Own collection, so this never races the shared `vectors` corpus.
    let coll = client
        .database(TEST_DB)
        .collection::<Document>("probe_293_matched_count");
    let _ = coll.drop().run();

    // POSITIVE CONTROL: with nothing inserted, the update must report a MISS.
    // If matched_count were always >= 1 the guard could never fire, and the
    // assertion below would pass vacuously.
    let missed = coll
        .update_one(
            doc! { "_id": 1i64 },
            doc! { "$set": { "vector": [1.0, 2.0] } },
        )
        .run()
        .expect("update_one against an absent _id must succeed, not error");
    assert_eq!(
        missed.matched_count, 0,
        "update_one without upsert must match 0 documents when the _id is absent"
    );
    assert_eq!(
        coll.count_documents(doc! {}).run().unwrap(),
        0,
        "a non-upsert update must not have created the document"
    );

    // The mixed-workload case: the document exists, so the write lands on it.
    coll.insert_one(doc! { "_id": 1i64, "vector": [0.0, 0.0] })
        .run()
        .unwrap();
    let matched = coll
        .update_one(
            doc! { "_id": 1i64 },
            doc! { "$set": { "vector": [1.0, 2.0] } },
        )
        .run()
        .unwrap();
    assert_eq!(
        matched.matched_count, 1,
        "update_one must match the existing document — this 1-vs-0 is the whole \
         #293 signal for MongoDB"
    );

    let _ = coll.drop().run();
}

/// The collection the ENGINE addresses for config `engine_name` when the run is
/// given `MONGODB_COLLECTION=TEST_COLLECTION` (#306).
///
/// Deliberately spelled out rather than imported: the engine lives in a binary
/// crate this test cannot link against, so this literal is the independent
/// statement of the naming contract. If the engine's derivation changes, these
/// tests must fail — that is the point.
fn engine_collection(engine_name: &str) -> String {
    format!("{TEST_COLLECTION}:{engine_name}")
}

/// The search index the ENGINE builds for config `engine_name` when the run is
/// given `MONGODB_INDEX_NAME=TEST_INDEX` (#306).
///
/// The separator is `_`, mirroring `derive_search_index_name`: MongoDB Atlas
/// rejects a `:` in a search index name with
/// `BadValue: invalid index name`, even though the
/// `mongodb/mongodb-atlas-local` image these tests run against accepts it. That
/// permissiveness gap is why the colon survived to a real Atlas run, so keep
/// this helper in step with production and do not "fix" it back to `:`.
///
/// It has to duplicate the production rule rather than call it: the engine lives
/// in a bin target, so an integration test cannot import `derive_search_index_name`.
/// The unit test `search_index_names_are_atlas_legal` is what actually guards the
/// character set.
fn engine_index(engine_name: &str) -> String {
    format!("{TEST_INDEX}_{engine_name}")
}

/// Count documents in a specific collection of the test database, server-side.
fn count_docs(collection: &str) -> u64 {
    mongodb_client()
        .database(TEST_DB)
        .collection::<Document>(collection)
        .count_documents(doc! {})
        .run()
        .expect("countDocuments")
}

/// Drop the search index (if any), wait for it to disappear, then drop the
/// collection and wait for it to be gone.  Mirrors the engine's configure()
/// cleanup so tests exercise the same Atlas-safe path.
///
/// #306: the engine no longer writes to the bare `TEST_COLLECTION`; each config
/// gets `TEST_COLLECTION:<config>`. Those are dropped too — scoped to this
/// file's own `vectors:` namespace rather than wiping the database, so a
/// concurrently running suite on another branch is not collateral.
fn drop_test_collection() {
    let client = mongodb_client();
    let db = client.database(TEST_DB);

    let per_config_prefix = format!("{TEST_COLLECTION}:");
    for name in db.list_collection_names().run().unwrap_or_default() {
        if name.starts_with(&per_config_prefix) {
            let _ = db.collection::<Document>(&name).drop().run();
        }
    }

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
        .collection::<Document>(&engine_collection(engine_name));
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
    // #293: recall and update_count are both blind to whether the updates landed
    // on the documents the search reads. The engine folds `matched_count` in and
    // publishes the attribution tier it achieved.
    // NOT corpus_row: `matched_count` describes the update's FILTER rather than
    // the payload, and the collection lags the Atlas vector index — two ways
    // weaker than the Redis-wire engines, so it must not share their label.
    assert_eq!(
        r["update_attribution"].as_str(),
        Some("matched_row"),
        "MongoDB's tier must say matched_row, not borrow the stronger corpus_row"
    );
    assert!(
        r["update_attribution_detail"]
            .as_str()
            .is_some_and(|d| d.contains("matched_count") && d.contains("FILTER")),
        "the artifact must carry the mechanism, not just the grade: {:?}",
        r["update_attribution_detail"]
    );
    assert_eq!(r["update_failures"].as_u64(), Some(0));
    assert_eq!(
        r["update_unattributed"].as_u64(),
        Some(0),
        "a healthy mixed run must match an existing document for every update"
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
fn read_back_index_definition(engine_name: &str) -> Document {
    let client = mongodb_client();
    let db = client.database(TEST_DB);
    // #306: per-config collection + per-config index name.
    let collection = engine_collection(engine_name);
    let index = engine_index(engine_name);

    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let result = db
            .run_command(doc! { "listSearchIndexes": &collection })
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
                if idx.get_str("name").ok() != Some(index.as_str()) {
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
            index,
            TEST_DB,
            collection
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

    (read_back_index_definition(engine_name), project)
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
    let geo_index = engine_index("mongo-geo");
    let indexes: Vec<mongodb::bson::Document> = client
        .database(TEST_DB)
        .collection::<mongodb::bson::Document>(&engine_collection("mongo-geo"))
        .aggregate(vec![doc! { "$listSearchIndexes": {} }])
        .run()
        .expect("listSearchIndexes")
        .filter_map(Result::ok)
        .collect();
    let ours = indexes
        .iter()
        .find(|i| i.get_str("name").unwrap_or("") == geo_index)
        .unwrap_or_else(|| panic!("no index named {geo_index}: {indexes:?}"));
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

// ---------------------------------------------------------------------------
// Issue #238 — `--skip-upload` must reuse the corpus, not destroy it
// ---------------------------------------------------------------------------

/// MongoDB's destruction path is a third shape: `collection.drop()`, which is
/// unbounded — unlike Redis/Valkey, whose `DD`/UNLINK are scoped to this config's
/// namespace, it removes the entire benchmark collection.
///
/// RED on the pre-fix branch: `countDocuments` 400 -> 0 while the run printed
/// QPS lines and exited 0.
#[test]
fn test_binary_mongodb_skip_upload_skip_vector_index_preserves_corpus() {
    wait_for_mongodb();
    drop_test_collection();

    let port: u16 = std::env::var("MONGODB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MONGODB_PORT);
    let port_s = port.to_string();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "cfg238mg", "engine": "mongodb",
        "search_params": [{"parallel": 1, "num_candidates": 20}],
        "upload_params": {"parallel": 1}
    }]);
    let proj = common::write_match_any_project(
        "cfg238mg-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    let envs: [(&str, &str); 4] = [
        ("MONGODB_PORT", port_s.as_str()),
        ("MONGODB_DB", TEST_DB),
        ("MONGODB_COLLECTION", TEST_COLLECTION),
        ("MONGODB_INDEX_NAME", TEST_INDEX),
    ];

    assert!(
        common::run_binary_extra(
            &proj.root,
            "cfg238mg",
            "cfg238mg-test",
            MONGODB_HOST,
            &envs,
            &["--skip-vector-index", "--keep-data", "--skip-search"],
        ),
        "phase 1 (upload with --skip-vector-index) failed"
    );
    // `--skip-vector-index` rewrites the config name to `<engine>-no-vector`
    // (experiment::run), so the collection the engine addresses is derived from
    // THAT name, not from `cfg238mg`.
    let before = count_docs(&engine_collection("mongodb-no-vector"));
    assert_eq!(before, common::N_DOCS as u64, "phase 1 corpus");

    assert!(
        common::run_binary_extra(
            &proj.root,
            "cfg238mg",
            "cfg238mg-test",
            MONGODB_HOST,
            &envs,
            &["--skip-upload", "--skip-vector-index", "--keep-data"],
        ),
        "phase 2 (--skip-upload --skip-vector-index) failed"
    );

    let after = count_docs(&engine_collection("mongodb-no-vector"));
    assert_eq!(
        after, before,
        "--skip-upload --skip-vector-index dropped the collection it was told to \
         reuse ({before} -> {after} docs) — issue #238"
    );

    drop_test_collection();
    fs::remove_dir_all(&proj.root).ok();
}

/// MongoDB's `corpus_row_count()` must be a live `countDocuments`, proven by
/// deleting documents behind the tool's back and watching the number follow.
#[test]
fn test_binary_mongodb_skip_upload_short_corpus_is_fatal() {
    wait_for_mongodb();
    drop_test_collection();

    let port: u16 = std::env::var("MONGODB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MONGODB_PORT);
    let port_s = port.to_string();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "cfg238mgshort", "engine": "mongodb",
        "search_params": [{"parallel": 1, "num_candidates": 20}],
        "upload_params": {"parallel": 1}
    }]);
    let proj = common::write_match_any_project(
        "cfg238mgshort-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    let envs: [(&str, &str); 4] = [
        ("MONGODB_PORT", port_s.as_str()),
        ("MONGODB_DB", TEST_DB),
        ("MONGODB_COLLECTION", TEST_COLLECTION),
        ("MONGODB_INDEX_NAME", TEST_INDEX),
    ];

    assert!(
        common::run_binary_extra(
            &proj.root,
            "cfg238mgshort",
            "cfg238mgshort-test",
            MONGODB_HOST,
            &envs,
            &["--keep-data", "--skip-search"],
        ),
        "phase 1 (upload) failed"
    );
    assert_eq!(
        count_docs(&engine_collection("cfg238mgshort")),
        common::N_DOCS as u64,
        "phase 1 corpus"
    );

    let half = (common::N_DOCS / 2) as i64;
    mongodb_client()
        .database(TEST_DB)
        .collection::<Document>(&engine_collection("cfg238mgshort"))
        .delete_many(doc! { "_id": { "$lt": half } })
        .run()
        .expect("delete_many");
    assert_eq!(
        count_docs(&engine_collection("cfg238mgshort")),
        half as u64,
        "half the corpus should remain"
    );

    let out = Command::new(binary_path())
        .args([
            "--engines",
            "cfg238mgshort",
            "--datasets",
            "cfg238mgshort-test",
            "--host",
            MONGODB_HOST,
            "--skip-if-exists",
            "false",
            "--skip-upload",
            "--keep-data",
        ])
        .env("MONGODB_PORT", &port_s)
        .env("MONGODB_DB", TEST_DB)
        .env("MONGODB_COLLECTION", TEST_COLLECTION)
        .env("MONGODB_INDEX_NAME", TEST_INDEX)
        .current_dir(&proj.root)
        .output()
        .expect("run vector-db-benchmark");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "--skip-upload against a half-deleted MongoDB collection must be a hard error.\n{combined}"
    );
    assert!(
        combined.contains(&format!("holds {half} of the {} rows", common::N_DOCS)),
        "the count must track the deletion, proving it is a live countDocuments.\n{combined}"
    );

    drop_test_collection();
    fs::remove_dir_all(&proj.root).ok();
}

/// #293 end to end: every mixed update matches nothing → hard error;
/// `--allow-partial-corpus` → honest zeros.
///
/// Without this the `Ok(true) => ut.unattributed += 1` arm in `search_mixed` is
/// dead in every run, so reading `matched_count` and discarding it are
/// indistinguishable and the fix could be reverted at the source with the suite
/// green. The mixed tests above only pin the healthy branch.
///
/// Fixture: shift every `_id` behind the tool's back. The collection keeps its
/// exact document count, so `--skip-upload`'s reuse check (`countDocuments`)
/// still passes and is NOT what fires.
#[test]
fn test_binary_mongodb_mixed_updates_that_miss_the_corpus_are_fatal() {
    wait_for_mongodb();
    drop_test_collection();

    let port: u16 = std::env::var("MONGODB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MONGODB_PORT);
    let port_s = port.to_string();

    let cfg = "cfg293mgmiss";
    let ds = "cfg293mgmiss-test";
    let configs = serde_json::json!([{
        "name": cfg, "engine": "mongodb",
        "search_params": [{"parallel": 1, "num_candidates": 20}],
        "upload_params": {"parallel": 1}
    }]);
    let proj =
        common::write_match_any_project_n(ds, &serde_json::to_string(&configs).unwrap(), 8, 500);
    let envs: [(&str, &str); 4] = [
        ("MONGODB_PORT", port_s.as_str()),
        ("MONGODB_DB", TEST_DB),
        ("MONGODB_COLLECTION", TEST_COLLECTION),
        ("MONGODB_INDEX_NAME", TEST_INDEX),
    ];

    // Rebuilt before EACH run: the gate rejects the numbers after the timed
    // window. MongoDB's update is a no-upsert `update_one`, so unlike the
    // Redis-wire engines a rejected run writes nothing — but rebuild anyway so
    // this test does not silently depend on that difference.
    let build_shifted_corpus = || {
        assert!(
            common::run_binary_extra(
                &proj.root,
                cfg,
                ds,
                MONGODB_HOST,
                &envs,
                &["--keep-data", "--skip-search"],
            ),
            "fixture upload failed"
        );
        let coll = mongodb_client()
            .database(TEST_DB)
            .collection::<Document>(&engine_collection(cfg));
        // `_id` is immutable, so re-key by copy-then-delete.
        let originals: Vec<Document> = coll
            .find(doc! {})
            .run()
            .unwrap()
            .map(|d| d.unwrap())
            .collect();
        assert_eq!(originals.len(), common::N_DOCS, "fixture corpus size");
        let shifted: Vec<Document> = originals
            .iter()
            .map(|d| {
                let mut c = d.clone();
                let id = c.get_i64("_id").unwrap();
                c.insert("_id", 90_000i64 + id);
                c
            })
            .collect();
        coll.delete_many(doc! {}).run().unwrap();
        coll.insert_many(shifted).run().unwrap();
        assert_eq!(
            count_docs(&engine_collection(cfg)),
            common::N_DOCS as u64,
            "the shifted corpus must keep its document count, so the reuse check \
             passes and this test exercises the #293 gate rather than #238's"
        );
        let results_dir = proj.root.join("results");
        if let Ok(entries) = fs::read_dir(&results_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.contains("-search-"))
                {
                    fs::remove_file(path).ok();
                }
            }
        }
    };

    let run = |extra: &[&str]| {
        let mut cmd = Command::new(common::binary_path());
        cmd.args([
            "--engines",
            cfg,
            "--datasets",
            ds,
            "--host",
            MONGODB_HOST,
            "--skip-if-exists",
            "false",
            "--skip-upload",
            "--keep-data",
            "--update-search-ratio",
            "1:5",
            "--repetitions",
            "1",
        ]);
        cmd.args(extra);
        for (k, v) in &envs {
            cmd.env(k, v);
        }
        cmd.current_dir(&proj.root)
            .output()
            .expect("run vector-db-benchmark")
    };

    build_shifted_corpus();
    let out = run(&[]);
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "a mixed run whose every update matched 0 documents must be a hard error \
         (#293), but the run succeeded.\n{combined}"
    );
    assert!(
        combined.contains("reported that the row each one addressed did not already exist")
            && combined.contains("Signal read: update_one reports matched_count"),
        "the error must be the #293 gate quoting the matched_count signal.\n{combined}"
    );

    build_shifted_corpus();
    let out2 = run(&["--allow-partial-corpus"]);
    assert!(
        out2.status.success(),
        "--allow-partial-corpus must downgrade the #293 gate to a warning.\n{}{}",
        String::from_utf8_lossy(&out2.stdout),
        String::from_utf8_lossy(&out2.stderr)
    );
    let r = common::read_results_obj(&proj.root, cfg);
    let update_count = r["update_count"].as_u64().unwrap();
    let unattributed = r["update_unattributed"].as_u64().unwrap();
    println!("mongodb #293 waived: update_count={update_count} update_unattributed={unattributed}");

    // POSITIVE evidence that the reuse check was satisfied and this test really
    // is exercising the #293 gate. The `!contains("incomplete")` check on the
    // rejected arm above is a negative; this reads the verdict the run recorded.
    let reuse = common::read_params_obj(&proj.root, cfg)["corpus_reuse"].clone();
    assert_eq!(
        reuse["status"], "verified",
        "the shifted corpus must verify on row count, or this fixture is testing \
         the #238 reuse gate instead: {reuse}"
    );
    assert!(unattributed > 0, "the missed updates must be recorded");
    assert_eq!(
        update_count, 0,
        "not one update matched, so the count must be 0"
    );

    drop_test_collection();
    fs::remove_dir_all(&proj.root).ok();
}

// ---------------------------------------------------------------------------
// #305: the ANN ceiling that made the capped catch-up gate unfixable in place
// ---------------------------------------------------------------------------

/// The three server facts the #305 fix stands on, at a corpus size no other
/// test in this suite reaches.
///
/// Every other mongodb integration corpus here is 20-500 vectors, so the branch
/// that broke — a catch-up gate that stopped waiting at 10_000 documents — was
/// unreachable from the whole suite; a regression re-capping it would leave the
/// file green. This uses 10_001 documents, the smallest corpus the old gate got
/// wrong, and pins:
///
/// 1. the APPROXIMATE probe cannot see the 10_001st document even at its
///    maximum settings (so the old gate released at 10_000/10_001 — the
///    positive control that the bug is real, and that the exhaustive assertion
///    below is not passing for free);
/// 2. the ceiling is the SERVER's, not a choice: `numCandidates: 10_001` is
///    rejected outright, so no ANN parameterisation can count the corpus;
/// 3. the EXHAUSTIVE probe the fix uses has no such ceiling and returns all
///    10_001 — the claim `wait_for_index_catchup` now depends on.
///
/// This does not exercise the benchmark's own code path (the probe builder is
/// private to the binary and unit-tested there); it pins the server behaviour
/// that path assumes, in the manner of the #293 `matched_count` test above.
#[test]
fn test_exhaustive_probe_counts_past_the_ann_10000_document_ceiling() {
    // One document past the old cap: below this the bug is invisible.
    const N: i64 = 10_001;
    const DIM: usize = 4;
    const COLL: &str = "probe_305_ann_ceiling";
    const INDEX: &str = "probe_305_index";

    wait_for_mongodb();
    let client = mongodb_client();
    let db = client.database(TEST_DB);
    let coll = db.collection::<Document>(COLL);

    // Own collection and own index name, so this never races the shared corpus.
    let _ = db
        .run_command(doc! { "dropSearchIndex": COLL, "name": INDEX })
        .run();
    let _ = coll.drop().run();

    let mut docs = Vec::with_capacity(N as usize);
    for i in 0..N {
        let f = i as f64;
        docs.push(doc! {
            "_id": i,
            "vector": [ (f * 0.001).sin(), (f * 0.001).cos(), (i % 97) as f64 / 97.0, 0.5 ],
        });
    }
    coll.insert_many(&docs)
        .run()
        .expect("failed to insert the 10_001-document probe corpus");
    assert_eq!(
        coll.count_documents(doc! {}).run().unwrap(),
        N as u64,
        "the corpus must be one document past the old 10_000 cap"
    );

    db.run_command(doc! {
        "createSearchIndexes": COLL,
        "indexes": [ {
            "name": INDEX,
            "type": "vectorSearch",
            "definition": { "fields": [ {
                "type": "vector",
                "path": "vector",
                "numDimensions": DIM as i32,
                "similarity": "cosine",
            } ] },
        } ],
    })
    .run()
    .expect("failed to create the probe vector search index");

    let query: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
    let enn = vec![
        doc! { "$vectorSearch": {
            "index": INDEX,
            "path": "vector",
            "queryVector": query.clone(),
            "limit": N,
            "exact": true,
        } },
        doc! { "$count": "n" },
    ];

    // FACT 3, and the wait: the exhaustive probe must eventually report the
    // WHOLE corpus. If it plateaus below N this fails with the number it
    // plateaued at — which is precisely the shape of the bug.
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut last = 0i64;
    let mut last_err = String::from("none");
    loop {
        match coll.aggregate(enn.clone()).run() {
            Ok(cursor) => {
                last = cursor
                    .filter_map(|r| r.ok())
                    .filter_map(|d| d.get_i32("n").map(i64::from).ok())
                    .last()
                    .unwrap_or(0);
                if last >= N {
                    break;
                }
            }
            Err(e) => last_err = e.to_string(),
        }
        assert!(
            Instant::now() < deadline,
            "exhaustive probe never reported the whole corpus: {last} / {N} after \
             300s (last error: {last_err}). If `limit` above 10_000 is now \
             rejected, the catch-up gate in mongodb_engine.rs has lost its only \
             corpus-complete signal and must not be allowed to pass silently.",
        );
        thread::sleep(Duration::from_secs(2));
    }
    assert_eq!(
        last, N,
        "exhaustive probe must count every indexed document"
    );

    // FACT 1 (positive control): the approximate probe at its MAXIMUM settings
    // sees exactly 10_000 — one short — so the old `results.len() >= want` gate
    // with `want = min(expected, 10_000)` released here, on a corpus that is
    // fully indexed only by luck of timing at this size and 0.85% indexed on
    // glove-100. Without this control the exhaustive assertion above could pass
    // on a server with no ceiling at all and prove nothing.
    let ann = coll
        .aggregate(vec![
            doc! { "$vectorSearch": {
                "index": INDEX,
                "path": "vector",
                "queryVector": query.clone(),
                "limit": 10_000i64,
                "numCandidates": 10_000i64,
            } },
            doc! { "$count": "n" },
        ])
        .run()
        .expect("the maximal approximate probe must still be a legal query");
    let ann_n = ann
        .filter_map(|r| r.ok())
        .filter_map(|d| d.get_i32("n").map(i64::from).ok())
        .last()
        .unwrap_or(0);
    assert_eq!(
        ann_n, 10_000,
        "the maximal approximate probe must top out at 10_000 — if it did not, \
         this test is not exercising the ceiling #305 is about"
    );
    assert!(
        ann_n < N,
        "the approximate probe cannot see the whole corpus, which is why the \
         capped gate could never be made correct"
    );

    // FACT 2: the ceiling is the server's. No ANN parameterisation reaches N.
    let rejected = coll
        .aggregate(vec![doc! { "$vectorSearch": {
            "index": INDEX,
            "path": "vector",
            "queryVector": query,
            "limit": N,
            "numCandidates": N,
        } }])
        .run()
        .and_then(|c| c.collect::<Result<Vec<_>, _>>());
    let err = rejected
        .expect_err("numCandidates above 10_000 must be rejected, not silently clamped")
        .to_string();
    assert!(
        err.contains("numCandidates"),
        "expected the server to name numCandidates as out of bounds, got: {err}"
    );

    let _ = db
        .run_command(doc! { "dropSearchIndex": COLL, "name": INDEX })
        .run();
    let _ = coll.drop().run();
}
// Issue #306 — every config addresses its OWN collection and search index
// ---------------------------------------------------------------------------

/// The headline failure of #306, end to end.
///
/// Before the fix every config in a sweep wrote to the one literal
/// `bench.vectors` with the one literal `vector_index`. `--skip-upload` skips
/// `configure()`, which is the only caller of `create_vector_index`, so config B
/// searched config A's HNSW graph, `countDocuments` returned A's row count, the
/// reuse check printed "holds N of N", and the run published B's `M` /
/// `EF_CONSTRUCTION` label over A's measurement and exited 0.
///
/// Phase 1 uploads under config A. Phase 2 runs config B with `--skip-upload`
/// and must be REJECTED: B's collection does not exist, so the reuse
/// precondition sees 0 of N.
///
/// RED ON REVERT: with the shared collection, phase 2's `countDocuments` finds
/// A's N documents, the precondition passes, and `run_binary_extra` returns
/// true — the first assertion fires. The follow-up assertions pin that the
/// rejection is the reuse gate (not an incidental connection error) and that
/// A's corpus was left untouched.
#[test]
fn test_binary_mongodb_skip_upload_against_another_configs_corpus_is_fatal() {
    wait_for_mongodb();
    drop_test_collection();

    let port: u16 = std::env::var("MONGODB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MONGODB_PORT);
    let port_s = port.to_string();

    let cfg_a = "cfg306a";
    let cfg_b = "cfg306b";
    let ds = "cfg306-test";
    let configs = serde_json::json!([
        {
            "name": cfg_a, "engine": "mongodb",
            "search_params": [{"parallel": 1, "num_candidates": 20}],
            "upload_params": {"parallel": 1}
        },
        {
            "name": cfg_b, "engine": "mongodb",
            "search_params": [{"parallel": 1, "num_candidates": 20}],
            "upload_params": {"parallel": 1}
        },
    ]);
    let proj = common::write_match_any_project(ds, &serde_json::to_string(&configs).unwrap(), 8);
    let envs: [(&str, &str); 4] = [
        ("MONGODB_PORT", port_s.as_str()),
        ("MONGODB_DB", TEST_DB),
        ("MONGODB_COLLECTION", TEST_COLLECTION),
        ("MONGODB_INDEX_NAME", TEST_INDEX),
    ];

    // Phase 1: config A populates ITS collection.
    assert!(
        common::run_binary_extra(
            &proj.root,
            cfg_a,
            ds,
            MONGODB_HOST,
            &envs,
            &["--keep-data", "--skip-search"],
        ),
        "phase 1 (config A upload) failed"
    );
    assert_eq!(
        count_docs(&engine_collection(cfg_a)),
        common::N_DOCS as u64,
        "config A must populate '{}' — if this is 0 the engine wrote somewhere else",
        engine_collection(cfg_a)
    );

    // Phase 2: config B reuses "the corpus". There is no corpus of B's.
    let out = Command::new(binary_path())
        .args([
            "--engines",
            cfg_b,
            "--datasets",
            ds,
            "--host",
            MONGODB_HOST,
            "--skip-if-exists",
            "false",
            "--skip-upload",
            "--keep-data",
        ])
        .envs(envs.iter().map(|(k, v)| (*k, *v)))
        .current_dir(&proj.root)
        .output()
        .expect("run vector-db-benchmark");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "config B has no corpus of its own; --skip-upload must abort rather than \
         measure config A's index and label it B (#306):\n{combined}"
    );
    assert!(
        combined.contains(&format!("holds 0 of the {} rows", common::N_DOCS)),
        "the rejection must be the reuse precondition seeing an empty corpus, not \
         an incidental failure:\n{combined}"
    );
    assert!(
        combined.contains("is empty or missing"),
        "config B's collection must read as absent, not merely short:\n{combined}"
    );

    // And B must not have destroyed A on its way out.
    assert_eq!(
        count_docs(&engine_collection(cfg_a)),
        common::N_DOCS as u64,
        "config B's aborted run touched config A's collection"
    );

    drop_test_collection();
    fs::remove_dir_all(&proj.root).ok();
}

/// #306 × #151-4 — MongoDB must PARTICIPATE in the startup collision guard.
///
/// The derivation alone leaves one way back into the shared-collection world:
/// `MONGODB_COLLECTION_EXACT=1` drops the per-config suffix, so N configs
/// resolve to one verbatim collection again and `configure()`'s
/// `collection.drop()` is back to wiping the sibling. The Redis-family engines
/// have that case rejected at startup; the guard did not know the string
/// `"mongodb"` at all.
///
/// Asserted on the ERROR TEXT, not merely a non-zero exit: without a reachable
/// server the run would fail anyway, so a bare exit-code assertion would pass
/// against a removed guard. This test needs no server — the guard runs before
/// any engine is constructed.
#[test]
fn test_binary_mongodb_exact_pin_with_two_configs_is_rejected_at_startup() {
    let configs = serde_json::json!([
        {
            "name": "mongo-guard-a", "engine": "mongodb",
            "search_params": [{"parallel": 1, "num_candidates": 20}],
        },
        {
            "name": "mongo-guard-b", "engine": "mongodb",
            "search_params": [{"parallel": 1, "num_candidates": 20}],
        },
    ]);
    let proj = common::write_match_any_project(
        "mongo-guard",
        &serde_json::to_string(&configs).unwrap(),
        8,
    );

    let out = Command::new(binary_path())
        .args([
            "--engines",
            "mongo-guard-*",
            "--datasets",
            "mongo-guard",
            "--host",
            MONGODB_HOST,
            "--skip-if-exists",
            "false",
        ])
        .current_dir(&proj.root)
        .env("MONGODB_DB", TEST_DB)
        .env("MONGODB_COLLECTION", "shared-coll")
        .env("MONGODB_COLLECTION_EXACT", "1")
        .output()
        .expect("run vector-db-benchmark");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "exact-pinned sweep of 2 mongodb configs must be rejected, not run: {combined}"
    );
    assert!(
        combined.contains("derive the same index namespace"),
        "the failure must be the #151-4 collision guard, not an incidental error: {combined}"
    );
    assert!(
        combined.contains("MONGODB_COLLECTION_EXACT is set"),
        "the guard must name the exact-pin as the cause so the fix is obvious: {combined}"
    );
    assert!(
        combined.contains("the mongodb collection"),
        "the guard must name the object that would be overwritten: {combined}"
    );
    fs::remove_dir_all(&proj.root).ok();
}

/// `--keep-data` across a multi-config sweep is documented (experiment.rs) as
/// keeping EVERY config's data resident simultaneously. That was false for
/// MongoDB — config B's `configure()` dropped config A's collection — and #306
/// makes it true. This pins the documented behaviour.
///
/// RED ON REVERT: with one shared collection, config B's `configure()` drops it,
/// so after the sweep only ONE collection exists and A's count is 0.
#[test]
fn test_binary_mongodb_keep_data_keeps_every_configs_corpus() {
    wait_for_mongodb();
    drop_test_collection();

    let port: u16 = std::env::var("MONGODB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MONGODB_PORT);
    let port_s = port.to_string();

    let ds = "cfg306keep-test";
    let cfg_a = "cfg306keepa";
    let cfg_b = "cfg306keepb";
    let configs = serde_json::json!([
        {
            "name": cfg_a, "engine": "mongodb",
            "search_params": [{"parallel": 1, "num_candidates": 20}],
            "upload_params": {"parallel": 1}
        },
        {
            "name": cfg_b, "engine": "mongodb",
            "search_params": [{"parallel": 1, "num_candidates": 20}],
            "upload_params": {"parallel": 1}
        },
    ]);
    let proj = common::write_match_any_project(ds, &serde_json::to_string(&configs).unwrap(), 8);
    let envs: [(&str, &str); 4] = [
        ("MONGODB_PORT", port_s.as_str()),
        ("MONGODB_DB", TEST_DB),
        ("MONGODB_COLLECTION", TEST_COLLECTION),
        ("MONGODB_INDEX_NAME", TEST_INDEX),
    ];

    // One invocation, both configs (`cfg306keep*`), --keep-data.
    assert!(
        common::run_binary_extra(
            &proj.root,
            "cfg306keep*",
            ds,
            MONGODB_HOST,
            &envs,
            &["--keep-data", "--skip-search"],
        ),
        "two-config --keep-data sweep failed"
    );

    for cfg in [cfg_a, cfg_b] {
        assert_eq!(
            count_docs(&engine_collection(cfg)),
            common::N_DOCS as u64,
            "--keep-data must leave '{}' resident; config '{}' lost its corpus to \
             a sibling's configure() (#306)",
            engine_collection(cfg),
            cfg
        );
    }

    drop_test_collection();
    fs::remove_dir_all(&proj.root).ok();
}
