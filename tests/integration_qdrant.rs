//! Integration tests for the Qdrant engine.
//!
//! Requires Qdrant running on gRPC port 6335 and REST port 6334.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d qdrant
//! Run with:   QDRANT_GRPC_PORT=6335 cargo test --test integration_qdrant -- --test-threads=1

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::thread;
use std::time::{Duration, Instant};

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const QDRANT_REST_PORT: u16 = 6334;
const QDRANT_GRPC_PORT: u16 = 6335;
const QDRANT_HOST: &str = "127.0.0.1";
const COLLECTION: &str = "bench_test";

fn rest_url() -> String {
    format!("http://{}:{}", QDRANT_HOST, QDRANT_REST_PORT)
}

fn grpc_url() -> String {
    format!("http://{}:{}", QDRANT_HOST, QDRANT_GRPC_PORT)
}

fn rest_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
}

fn wait_for_qdrant() {
    let client = rest_client();
    let url = format!("{}/collections", rest_url());
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Ok(resp) = client.get(&url).send() {
            if resp.status().is_success() {
                return;
            }
        }
        if Instant::now() > deadline {
            panic!(
                "Qdrant not available on port {} after 60s",
                QDRANT_REST_PORT
            );
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn delete_collection() {
    let client = rest_client();
    let _ = client
        .delete(format!("{}/collections/{}", rest_url(), COLLECTION))
        .send();
}

fn generate_test_vectors(count: usize, dim: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let ids: Vec<i64> = (0..count as i64).collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (ids, vectors)
}

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

/// Like `common::run_binary`, but hands back the combined stdout+stderr so a
/// test asserting a FAILURE can pin *why* it failed. `!run_binary(...)` on its
/// own is satisfied by a downed container or a broken fixture just as happily as
/// by the rejection under test, and `run_binary` prints its diagnostics only on
/// the success path — so a regression there would be silent.
fn run_binary_capture(root: &std::path::Path, engine: &str, dataset: &str) -> (bool, String) {
    let mut cmd = std::process::Command::new(common::binary_path());
    cmd.args([
        "--engines",
        engine,
        "--datasets",
        dataset,
        "--host",
        "localhost",
        "--skip-if-exists",
        "false",
    ])
    .current_dir(root)
    .env("QDRANT_GRPC_PORT", QDRANT_GRPC_PORT.to_string())
    .env("QDRANT_REST_PORT", QDRANT_REST_PORT.to_string());
    let out = cmd.output().expect("run vector-db-benchmark");
    let combined = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), combined)
}

/// A rejected filter must leave NO benchmark number behind — that is the whole
/// point of failing rather than degrading into an unfiltered run.
fn assert_no_search_result(root: &std::path::Path, engine: &str) {
    let dir = root.join("results");
    let found: Vec<String> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with(&format!("{}-", engine)) && n.contains("-search-"))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        found.is_empty(),
        "a rejected filter must not publish a recall number, but found {found:?}"
    );
}

fn create_grpc_client() -> (tokio::runtime::Runtime, qdrant_client::Qdrant) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    // `from_url(...).build()` is synchronous and returns a Result, not a future.
    let client = qdrant_client::Qdrant::from_url(&grpc_url())
        .build()
        .unwrap();
    (rt, client)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_qdrant_collection_crud() {
    wait_for_qdrant();
    delete_collection();

    let (rt, client) = create_grpc_client();

    // Create collection
    use qdrant_client::qdrant::{
        vectors_config::Config, CreateCollectionBuilder, Distance, VectorParamsBuilder,
        VectorsConfig,
    };

    rt.block_on(
        client.create_collection(CreateCollectionBuilder::new(COLLECTION).vectors_config(
            VectorsConfig {
                config: Some(Config::Params(
                    VectorParamsBuilder::new(4, Distance::Euclid).build(),
                )),
            },
        )),
    )
    .expect("Failed to create collection");

    // Verify exists
    let info = rt
        .block_on(client.collection_info(COLLECTION))
        .expect("Failed to get collection info");
    assert!(info.result.is_some(), "Collection should exist");

    // Delete
    rt.block_on(
        client.delete_collection(qdrant_client::qdrant::DeleteCollectionBuilder::new(
            COLLECTION,
        )),
    )
    .expect("Failed to delete collection");
}

#[test]
fn test_qdrant_upsert_and_search() {
    wait_for_qdrant();
    delete_collection();

    let (rt, client) = create_grpc_client();

    use qdrant_client::qdrant::{
        vectors_config::Config, CreateCollectionBuilder, Distance, PointStruct,
        SearchPointsBuilder, VectorParamsBuilder, VectorsConfig,
    };

    // Create collection
    rt.block_on(
        client.create_collection(CreateCollectionBuilder::new(COLLECTION).vectors_config(
            VectorsConfig {
                config: Some(Config::Params(
                    VectorParamsBuilder::new(4, Distance::Euclid).build(),
                )),
            },
        )),
    )
    .unwrap();

    // Upsert points
    let (ids, vectors) = generate_test_vectors(50, 4);
    let points: Vec<PointStruct> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(id, vec)| PointStruct::new(*id as u64, vec.clone(), qdrant_client::Payload::new()))
        .collect();

    rt.block_on(client.upsert_points(
        qdrant_client::qdrant::UpsertPointsBuilder::new(COLLECTION, points).wait(true),
    ))
    .expect("Failed to upsert points");

    // Search
    let results = rt
        .block_on(client.search_points(
            SearchPointsBuilder::new(COLLECTION, vectors[0].clone(), 5).with_payload(false),
        ))
        .expect("Failed to search");

    assert!(!results.result.is_empty(), "Search should return results");

    // First result should be the query vector itself (id=0)
    if let Some(first) = results.result.first() {
        if let Some(id) = &first.id {
            if let Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) =
                &id.point_id_options
            {
                assert_eq!(*n, 0, "Query vector should be its own nearest neighbor");
            }
        }
    }

    delete_collection();
}

#[test]
fn test_qdrant_precision() {
    wait_for_qdrant();
    delete_collection();

    let (rt, client) = create_grpc_client();

    use qdrant_client::qdrant::{
        vectors_config::Config, CreateCollectionBuilder, Distance, PointStruct,
        SearchPointsBuilder, VectorParamsBuilder, VectorsConfig,
    };

    let dim = 8;
    let count = 100;
    let top = 10;

    rt.block_on(
        client.create_collection(CreateCollectionBuilder::new(COLLECTION).vectors_config(
            VectorsConfig {
                config: Some(Config::Params(
                    VectorParamsBuilder::new(dim as u64, Distance::Euclid).build(),
                )),
            },
        )),
    )
    .unwrap();

    let (ids, vectors) = generate_test_vectors(count, dim);
    let points: Vec<PointStruct> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(id, vec)| PointStruct::new(*id as u64, vec.clone(), qdrant_client::Payload::new()))
        .collect();

    rt.block_on(client.upsert_points(
        qdrant_client::qdrant::UpsertPointsBuilder::new(COLLECTION, points).wait(true),
    ))
    .unwrap();

    // Wait for indexing
    thread::sleep(Duration::from_secs(2));

    let query_idx = 42;
    let expected = brute_force_neighbors_l2(&vectors[query_idx], &vectors, top);

    let results = rt
        .block_on(
            client.search_points(
                SearchPointsBuilder::new(COLLECTION, vectors[query_idx].clone(), top as u64)
                    .with_payload(false),
            ),
        )
        .unwrap();

    let found: std::collections::HashSet<i64> = results
        .result
        .iter()
        .filter_map(|p| {
            if let Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) =
                &p.id.as_ref().and_then(|id| id.point_id_options.as_ref())
            {
                Some(*n as i64)
            } else {
                None
            }
        })
        .collect();

    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let hits = expected_set.intersection(&found).count();
    let precision = hits as f64 / top as f64;

    assert!(
        precision >= 0.9,
        "Precision should be >= 0.9 for small dataset, got {}",
        precision
    );

    delete_collection();
}

#[test]
fn test_qdrant_payload_filter() {
    wait_for_qdrant();
    delete_collection();

    let (rt, client) = create_grpc_client();

    use qdrant_client::qdrant::{
        vectors_config::Config, Condition, CreateCollectionBuilder, Distance, FieldType, Filter,
        PointStruct, SearchPointsBuilder, VectorParamsBuilder, VectorsConfig,
    };

    rt.block_on(
        client.create_collection(CreateCollectionBuilder::new(COLLECTION).vectors_config(
            VectorsConfig {
                config: Some(Config::Params(
                    VectorParamsBuilder::new(4, Distance::Euclid).build(),
                )),
            },
        )),
    )
    .unwrap();

    // Create field index
    rt.block_on(client.create_field_index(
        qdrant_client::qdrant::CreateFieldIndexCollectionBuilder::new(
            COLLECTION,
            "category",
            FieldType::Keyword,
        ),
    ))
    .unwrap();

    // Upsert with payload
    let (ids, vectors) = generate_test_vectors(20, 4);
    let points: Vec<PointStruct> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(id, vec)| {
            let mut payload = qdrant_client::Payload::new();
            payload.insert("category", if *id % 2 == 0 { "A" } else { "B" });
            PointStruct::new(*id as u64, vec.clone(), payload)
        })
        .collect();

    rt.block_on(client.upsert_points(
        qdrant_client::qdrant::UpsertPointsBuilder::new(COLLECTION, points).wait(true),
    ))
    .unwrap();

    // Search with filter: only category "A"
    let filter = Filter {
        must: vec![Condition::matches("category", "A".to_string())],
        ..Default::default()
    };

    let results = rt
        .block_on(
            client.search_points(
                SearchPointsBuilder::new(COLLECTION, vectors[0].clone(), 10)
                    .filter(filter)
                    .with_payload(true),
            ),
        )
        .unwrap();

    assert!(
        !results.result.is_empty(),
        "Filtered search should return results"
    );

    // Verify all results have even IDs (category A)
    for p in &results.result {
        if let Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) =
            &p.id.as_ref().and_then(|id| id.point_id_options.as_ref())
        {
            assert!(
                *n % 2 == 0,
                "Filtered search should only return category A (even IDs), got id={}",
                n
            );
        }
    }

    delete_collection();
}

// ---------------------------------------------------------------------------
// Binary-level coverage: run the real engine end-to-end via the CLI.
// Covers the query_points migration and prefetch (search_params.prefetch).
// ---------------------------------------------------------------------------

fn binary_path() -> std::path::PathBuf {
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
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/release/vector-db-benchmark")
}

/// Write a temp project (datasets + configs + results) and return its root.
fn write_dense_project(
    dataset_name: &str,
    engine_configs_json: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    neighbors: &[Vec<i64>],
    dim: usize,
) -> std::path::PathBuf {
    use std::fs;
    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let dataset_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&dataset_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    let jsonl = |rows: &[Vec<f32>]| -> String {
        rows.iter()
            .map(|v| {
                serde_json::to_string(&v.iter().map(|x| *x as f64).collect::<Vec<_>>()).unwrap()
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    fs::write(dataset_dir.join("vectors.jsonl"), jsonl(vectors)).unwrap();
    fs::write(dataset_dir.join("queries.jsonl"), jsonl(queries)).unwrap();
    let nb = neighbors
        .iter()
        .map(|n| serde_json::to_string(n).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(dataset_dir.join("neighbours.jsonl"), nb).unwrap();

    let datasets_json = serde_json::json!([{
        "name": dataset_name, "type": "jsonl", "path": format!("{}/", dataset_name),
        "distance": "l2", "vector_size": dim, "vector_count": vectors.len(),
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

fn run_qdrant_binary(root: &std::path::Path, engine: &str, dataset: &str) -> bool {
    let out = std::process::Command::new(binary_path())
        .args([
            "--engines",
            engine,
            "--datasets",
            dataset,
            "--host",
            "localhost",
            "--skip-if-exists",
            "false",
        ])
        .env("QDRANT_GRPC_PORT", QDRANT_GRPC_PORT.to_string())
        .env("QDRANT_REST_PORT", QDRANT_REST_PORT.to_string())
        .current_dir(root)
        .output()
        .expect("run vector-db-benchmark");
    if !out.status.success() {
        eprintln!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    out.status.success()
}

fn read_precision(root: &std::path::Path, engine: &str) -> f64 {
    use std::fs;
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
    v["results"]["mean_precision_at_returned"].as_f64().unwrap()
}

/// End-to-end via the real engine: a plain search (covers the query_points
/// migration) and a prefetch/two-stage search both return high-recall results.
#[test]
fn test_binary_qdrant_query_points_and_prefetch() {
    wait_for_qdrant();

    let dim = 8;
    let (_ids, vectors) = generate_test_vectors(200, dim);
    let queries: Vec<Vec<f32>> = vectors[..10].to_vec();
    let top = 10;
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors_l2(q, &vectors, top))
        .collect();

    let configs = serde_json::json!([
        {
            "name": "qdrant-qp", "engine": "qdrant",
            "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
            "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
            "upload_params": {"parallel": 1, "batch_size": 100}
        },
        {
            "name": "qdrant-pf", "engine": "qdrant",
            "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
            "search_params": [{"parallel": 1, "search_params": {
                "hnsw_ef": 128, "prefetch": {"limit": 50, "params": {"hnsw_ef": 256}}
            }}],
            "upload_params": {"parallel": 1, "batch_size": 100}
        }
    ]);
    let root = write_dense_project(
        "qp-test",
        &serde_json::to_string(&configs).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        dim,
    );

    assert!(
        run_qdrant_binary(&root, "qdrant-qp", "qp-test"),
        "plain run failed"
    );
    let p_plain = read_precision(&root, "qdrant-qp");
    assert!(
        p_plain >= 0.9,
        "query_points precision {:.3} < 0.9",
        p_plain
    );

    assert!(
        run_qdrant_binary(&root, "qdrant-pf", "qp-test"),
        "prefetch run failed"
    );
    let p_pf = read_precision(&root, "qdrant-pf");
    assert!(p_pf >= 0.9, "prefetch precision {:.3} < 0.9", p_pf);
    println!(
        "qdrant query_points precision={:.3}, prefetch precision={:.3}",
        p_plain, p_pf
    );
}

/// End-to-end sparse-vector coverage: build a small sparse dataset (via the
/// shared `write_sparse_project` fixture), run the real engine (sparse collection,
/// upsert, and a `query_points` search using the named "sparse" vector), then
/// assert recall against brute-force dot-product (descending / MIPS) ground truth.
#[test]
fn test_binary_qdrant_sparse() {
    wait_for_qdrant();

    let configs = serde_json::json!([{
        "name": "qdrant-sparse-cov", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1}], "upload_params": {"parallel": 1, "batch_size": 50}
    }]);
    let proj =
        common::write_sparse_project("sparse-cov", &serde_json::to_string(&configs).unwrap());

    assert!(
        run_qdrant_binary(&proj.root, "qdrant-sparse-cov", "sparse-cov"),
        "sparse run failed"
    );
    let precision = read_precision(&proj.root, "qdrant-sparse-cov");
    println!(
        "qdrant sparse precision={:.3} (top={})",
        precision, proj.top
    );
    assert!(precision >= 0.9, "sparse precision {:.3} < 0.9", precision);
}

/// End-to-end HYBRID (dense + sparse) coverage WITH a negative control.
///
/// The planted dataset's ground truth is recoverable ONLY by fusing both
/// modalities (see `write_hybrid_project`). We assert two things against live
/// qdrant:
///   1. the HYBRID engine (named "dense" + "sparse" vectors, upsert of both, a
///      query fusing a dense prefetch and a sparse prefetch via RRF) clears the
///      0.9 recall floor, and
///   2. a NEGATIVE CONTROL — a plain dense search over the SAME dense vectors +
///      SAME ground truth (the `*-dense` jsonl view) — stays strictly LOW
///      (< 0.6). Together these prove the dataset genuinely requires fusion and
///      the hybrid path is doing real work, not silently collapsing to one
///      modality.
#[test]
fn test_binary_qdrant_hybrid() {
    wait_for_qdrant();

    let configs = serde_json::json!([
        {
            "name": "qdrant-hybrid-cov", "engine": "qdrant",
            "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
            // prefetch.limit sets the per-modality candidate depth fused by RRF
            // (>= 2*top so each ground-truth doc is visible in both prefetches).
            "search_params": [{"parallel": 1, "search_params": {"prefetch": {"limit": 32}}}],
            "upload_params": {"parallel": 1, "batch_size": 50}
        },
        {
            "name": "qdrant-hybrid-dense-neg", "engine": "qdrant",
            "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
            "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
            "upload_params": {"parallel": 1, "batch_size": 50}
        }
    ]);
    let proj =
        common::write_hybrid_project("hybrid-cov", &serde_json::to_string(&configs).unwrap());

    // 1. Fused hybrid recall must clear the floor.
    assert!(
        run_qdrant_binary(&proj.root, "qdrant-hybrid-cov", &proj.dataset_name),
        "hybrid run failed"
    );
    let recall = common::read_recall(&proj.root, "qdrant-hybrid-cov");
    println!("qdrant hybrid recall={:.3} (top={})", recall, proj.top);
    assert!(recall >= 0.9, "hybrid recall {:.3} < 0.9", recall);

    // 2. Negative control: plain dense search over the SAME data must be LOW,
    //    proving the ground truth is unreachable without the sparse modality.
    assert!(
        run_qdrant_binary(
            &proj.root,
            "qdrant-hybrid-dense-neg",
            &proj.dense_dataset_name
        ),
        "dense-only negative-control run failed"
    );
    let dense_recall = common::read_recall(&proj.root, "qdrant-hybrid-dense-neg");
    println!("qdrant hybrid dense-only negative-control recall={dense_recall:.3}");
    assert!(
        dense_recall < 0.6,
        "negative control recall {dense_recall:.3} >= 0.6 — dataset does NOT require fusion",
    );
    assert!(
        recall > dense_recall + 0.3,
        "fusion ({recall:.3}) must beat dense-only ({dense_recall:.3}) by a wide margin",
    );
}

/// End-to-end `match_any` coverage. Qdrant already supports `match_any`, so this
/// doubles as validation that the shared fixture + harness are correct: build a
/// dataset whose queries filter a keyword field to an OR-set, with ground truth
/// brute-forced over ONLY the matching documents, then assert the engine's
/// recall is high (it applied the filter and returned the filtered NNs).
#[test]
fn test_binary_qdrant_match_any() {
    wait_for_qdrant();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "qdrant-ma", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
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

    let grpc = QDRANT_GRPC_PORT.to_string();
    let rest = QDRANT_REST_PORT.to_string();
    assert!(
        common::run_binary(
            &proj.root,
            "qdrant-ma",
            "match-any-test",
            "localhost",
            &[
                ("QDRANT_GRPC_PORT", grpc.as_str()),
                ("QDRANT_REST_PORT", rest.as_str()),
            ],
        ),
        "qdrant match_any run failed"
    );

    let recall = common::read_recall(&proj.root, "qdrant-ma");
    println!("qdrant match_any recall={:.3}", recall);
    assert!(recall >= 0.9, "qdrant match_any recall {:.3} < 0.9", recall);
}

/// #222, server-side premise: an EMPTY `Filter` is MATCH-ALL, not "no matches"
/// and not an error. This is why `parse_qdrant_conditions` must return `None`
/// rather than `Some(Filter{must:[],should:[]})` when every leaf of an `and`
/// group drops — sending that filter runs the query completely unconstrained
/// while the code believes it filtered. Live-asserted so the premise cannot
/// silently change under a Qdrant upgrade.
#[test]
fn test_qdrant_empty_filter_is_match_all() {
    wait_for_qdrant();
    delete_collection();

    let (rt, client) = create_grpc_client();
    use qdrant_client::qdrant::{
        vectors_config::Config, Condition, CreateCollectionBuilder, Distance, FieldType, Filter,
        PointStruct, SearchPointsBuilder, VectorParamsBuilder, VectorsConfig,
    };

    rt.block_on(
        client.create_collection(CreateCollectionBuilder::new(COLLECTION).vectors_config(
            VectorsConfig {
                config: Some(Config::Params(
                    VectorParamsBuilder::new(4, Distance::Euclid).build(),
                )),
            },
        )),
    )
    .unwrap();
    rt.block_on(client.create_field_index(
        qdrant_client::qdrant::CreateFieldIndexCollectionBuilder::new(
            COLLECTION,
            "category",
            FieldType::Keyword,
        ),
    ))
    .unwrap();

    let (ids, vectors) = generate_test_vectors(20, 4);
    let points: Vec<PointStruct> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(id, vec)| {
            let mut payload = qdrant_client::Payload::new();
            payload.insert("category", if *id % 2 == 0 { "A" } else { "B" });
            PointStruct::new(*id as u64, vec.clone(), payload)
        })
        .collect();
    rt.block_on(client.upsert_points(
        qdrant_client::qdrant::UpsertPointsBuilder::new(COLLECTION, points).wait(true),
    ))
    .unwrap();

    let hits = |filter: Option<Filter>| -> usize {
        let mut b = SearchPointsBuilder::new(COLLECTION, vectors[0].clone(), 100);
        if let Some(f) = filter {
            b = b.filter(f);
        }
        rt.block_on(client.search_points(b)).unwrap().result.len()
    };

    let unfiltered = hits(None);
    let empty_filter = hits(Some(Filter::default()));
    let real_filter = hits(Some(Filter {
        must: vec![Condition::matches("category", "A".to_string())],
        ..Default::default()
    }));

    assert_eq!(unfiltered, 20, "fixture should hold 20 points");
    assert_eq!(
        empty_filter, unfiltered,
        "an empty Filter must be understood as MATCH-ALL ({} vs {}) — the whole \
         point of the #222 guard",
        empty_filter, unfiltered
    );
    assert_eq!(real_filter, 10, "a real filter must actually constrain");

    delete_collection();
}

/// #222 defect 2, server-side premise + UTC correctness of the datetime arm.
///
/// Two things the unit tests cannot establish, both live:
///
/// 1. **An all-`None` `DatetimeRange` is MATCH-ALL over gRPC.** The decision to
///    hard-error on an unparseable datetime bound rests entirely on this: if a
///    vacuous range were an error or matched nothing, dropping the bound would
///    be harmless. It is not — the request goes out looking filtered and
///    returns the whole collection.
/// 2. **A client-converted `Timestamp` selects exactly what the server selects
///    for the equivalent datetime string.** `parse_rfc3339_timestamp` maps the
///    RFC-3339, naive, date-only and epoch-seconds spellings onto the SAME
///    `prost_types::Timestamp` (pinned by the unit tests), and `DatetimeRange`'s
///    bounds are `Timestamp`, not strings — so this asserts the whole family
///    lands on the same documents as Qdrant's own string parsing, i.e. the
///    fallback is not a type substitution onto some other index.
#[test]
fn test_qdrant_datetime_range_is_utc_correct_and_vacuous_range_is_match_all() {
    wait_for_qdrant();
    delete_collection();

    let (rt, client) = create_grpc_client();
    use qdrant_client::qdrant::{
        vectors_config::Config, Condition, CreateCollectionBuilder, DatetimeRange, Distance,
        FieldType, Filter, PointStruct, SearchPointsBuilder, Timestamp, VectorParamsBuilder,
        VectorsConfig,
    };

    rt.block_on(
        client.create_collection(CreateCollectionBuilder::new(COLLECTION).vectors_config(
            VectorsConfig {
                config: Some(Config::Params(
                    VectorParamsBuilder::new(4, Distance::Euclid).build(),
                )),
            },
        )),
    )
    .unwrap();
    rt.block_on(client.create_field_index(
        qdrant_client::qdrant::CreateFieldIndexCollectionBuilder::new(
            COLLECTION,
            "ts",
            FieldType::Datetime,
        ),
    ))
    .unwrap();

    // 8 points, one per day: id N carries ts = 2023-01-0N T00:00:00Z.
    let (ids, vectors) = generate_test_vectors(8, 4);
    let points: Vec<PointStruct> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(id, vec)| {
            let mut payload = qdrant_client::Payload::new();
            payload.insert("ts", format!("2023-01-{:02}T00:00:00Z", id + 1));
            PointStruct::new(*id as u64, vec.clone(), payload)
        })
        .collect();
    rt.block_on(client.upsert_points(
        qdrant_client::qdrant::UpsertPointsBuilder::new(COLLECTION, points).wait(true),
    ))
    .unwrap();

    let hit_ids = |filter: Option<Filter>| -> Vec<u64> {
        let mut b = SearchPointsBuilder::new(COLLECTION, vectors[0].clone(), 100);
        if let Some(f) = filter {
            b = b.filter(f);
        }
        let mut out: Vec<u64> = rt
            .block_on(client.search_points(b))
            .unwrap()
            .result
            .iter()
            .filter_map(|p| match p.id.as_ref()?.point_id_options.as_ref()? {
                qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => Some(*n),
                _ => None,
            })
            .collect();
        out.sort_unstable();
        out
    };

    // (1) A DatetimeRange with every bound None is MATCH-ALL — byte-identical to
    //     an empty filter and to sending no filter at all.
    let vacuous = hit_ids(Some(Filter {
        must: vec![Condition::datetime_range("ts", DatetimeRange::default())],
        ..Default::default()
    }));
    let unfiltered = hit_ids(None);
    assert_eq!(unfiltered.len(), 8, "fixture should hold 8 points");
    assert_eq!(
        vacuous, unfiltered,
        "an all-None DatetimeRange must be MATCH-ALL — this is why an unparseable \
         bound has to fail the run instead of being dropped"
    );

    // (2) 2023-01-05T00:00:00Z as a client-side Timestamp — the value
    //     `parse_rfc3339_timestamp` produces for the RFC-3339, naive
    //     ("2023-01-05 00:00:00"), date-only ("2023-01-05") and epoch-seconds
    //     ("1672876800") spellings alike.
    const JAN5_2023_UTC: i64 = 1_672_876_800;
    let via_timestamp = hit_ids(Some(Filter {
        must: vec![Condition::datetime_range(
            "ts",
            DatetimeRange {
                gte: Some(Timestamp {
                    seconds: JAN5_2023_UTC,
                    nanos: 0,
                }),
                ..Default::default()
            },
        )],
        ..Default::default()
    }));
    assert_eq!(
        via_timestamp,
        vec![4, 5, 6, 7],
        "gte 2023-01-05T00:00:00Z must select exactly the 2023-01-05..08 points \
         (UTC, no off-by-one-day and no timezone drift)"
    );

    // …and the server's OWN parsing of the equivalent string picks the same set,
    // so the client-side conversion is not a substitution onto another index.
    let resp: serde_json::Value = rest_client()
        .post(format!(
            "{}/collections/{}/points/scroll",
            rest_url(),
            COLLECTION
        ))
        .json(&serde_json::json!({
            "limit": 100,
            "filter": {"must": [{"key": "ts", "range": {"gte": "2023-01-05T00:00:00Z"}}]}
        }))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let mut via_string: Vec<u64> = resp["result"]["points"]
        .as_array()
        .unwrap_or_else(|| panic!("unexpected scroll response: {resp}"))
        .iter()
        .map(|p| p["id"].as_u64().unwrap())
        .collect();
    via_string.sort_unstable();
    assert_eq!(
        via_string, via_timestamp,
        "server-parsed datetime string and client-converted Timestamp must select \
         the same documents"
    );

    delete_collection();
}

/// #222 defect 3, end-to-end: a float `match.any` used to have its members
/// deleted by `filter_map(as_str)`, leaving an EMPTY `MatchAny` — which Qdrant
/// happily evaluates to zero hits, so the run reported **recall 0 as a
/// benchmark result**. It must now fail the run instead. Built by rewriting the
/// conditions of the standard match_any fixture, so everything else about the
/// project is a known-good configuration.
#[test]
fn test_binary_qdrant_float_match_any_fails_instead_of_reporting_zero_recall() {
    wait_for_qdrant();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "qdrant-ma-float", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "match-any-float-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );

    // Rewrite every query's condition to a FLOAT match_any (unrepresentable in
    // Qdrant's Match, which has keyword/integer variants only).
    let tests_path = proj
        .root
        .join("datasets")
        .join("match-any-float-test")
        .join("tests.jsonl");
    let rewritten: Vec<String> = std::fs::read_to_string(&tests_path)
        .unwrap()
        .lines()
        .map(|line| {
            let mut v: serde_json::Value = serde_json::from_str(line).unwrap();
            v["conditions"] =
                serde_json::json!({"and": [{"size": {"match": {"any": [1.5, 2.5]}}}]});
            v.to_string()
        })
        .collect();
    std::fs::write(&tests_path, rewritten.join("\n")).unwrap();

    let (ok, output) = run_binary_capture(&proj.root, "qdrant-ma-float", "match-any-float-test");
    assert!(
        !ok,
        "a float match_any must FAIL the run, not report a recall number"
    );
    // `!ok` alone would also pass if the container were down or the fixture were
    // broken, so pin WHY it failed and that no number reached disk.
    assert!(
        output.contains("size"),
        "the failure must name the offending field, not fail for some unrelated \
         reason; got:\n{output}"
    );
    assert_no_search_result(&proj.root, "qdrant-ma-float");
}

/// #222 defect 1, end-to-end — the one the original fix did NOT close.
///
/// Measured on the first revision of this branch, with every leaf rewritten to
/// drop: `run_succeeded = true, reported_mean_recall = Some(0.46)`. Returning
/// `None` for a filter that evaporated is indistinguishable from "this query has
/// no filter", so the run went out unconstrained and published 0.46 as a Qdrant
/// recall against FILTERED ground truth. Only failing the run separates the two.
#[test]
fn test_binary_qdrant_unbuildable_filter_fails_instead_of_publishing_a_recall() {
    wait_for_qdrant();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "qdrant-drop", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "drop-filter-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );

    let tests_path = proj
        .root
        .join("datasets")
        .join("drop-filter-test")
        .join("tests.jsonl");
    let rewritten: Vec<String> = std::fs::read_to_string(&tests_path)
        .unwrap()
        .lines()
        .map(|line| {
            let mut v: serde_json::Value = serde_json::from_str(line).unwrap();
            // Every leaf un-buildable: pre-fix an EMPTY Filter (match-all), then
            // `None` (also match-all). The ground truth stays filtered either way.
            v["conditions"] = serde_json::json!({"and": [{"size": {"nosuchop": {"value": 1}}}]});
            v.to_string()
        })
        .collect();
    std::fs::write(&tests_path, rewritten.join("\n")).unwrap();

    let (ok, output) = run_binary_capture(&proj.root, "qdrant-drop", "drop-filter-test");
    assert!(
        !ok,
        "a filter that builds no condition must FAIL the run — running it \
         unfiltered against filtered ground truth publishes a plausible wrong number"
    );
    assert!(
        output.contains("nosuchop"),
        "the failure must name the offending operator; got:\n{output}"
    );
    assert_no_search_result(&proj.root, "qdrant-drop");
}

/// Geo-radius filter end-to-end (previously untested for qdrant). `geo` -> a Geo
/// payload index + `Condition::geo_radius`; recall vs haversine ground truth.
#[test]
fn test_binary_qdrant_geo() {
    wait_for_qdrant();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "qdrant-geo", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_geo_project("geo-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    let grpc = QDRANT_GRPC_PORT.to_string();
    let rest = QDRANT_REST_PORT.to_string();
    assert!(
        common::run_binary(
            &proj.root,
            "qdrant-geo",
            "geo-test",
            "localhost",
            &[
                ("QDRANT_GRPC_PORT", grpc.as_str()),
                ("QDRANT_REST_PORT", rest.as_str()),
            ],
        ),
        "qdrant geo run failed"
    );
    let recall = common::read_recall(&proj.root, "qdrant-geo");
    println!("qdrant geo recall={:.3}", recall);
    assert!(recall >= 0.9, "qdrant geo recall {:.3} < 0.9", recall);
}

/// Multi-condition AND (keyword match AND numeric range) — verifies qdrant
/// composes two conditions into one Filter.must (intersection).
#[test]
fn test_binary_qdrant_and_filter() {
    wait_for_qdrant();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "qdrant-and", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_and_filter_project(
        "and-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);
    let grpc = QDRANT_GRPC_PORT.to_string();
    let rest = QDRANT_REST_PORT.to_string();
    assert!(
        common::run_binary(
            &proj.root,
            "qdrant-and",
            "and-test",
            "localhost",
            &[
                ("QDRANT_GRPC_PORT", grpc.as_str()),
                ("QDRANT_REST_PORT", rest.as_str()),
            ],
        ),
        "qdrant and-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "qdrant-and");
    println!("qdrant and-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "qdrant and-filter recall {:.3} < 0.9",
        recall
    );
}

/// Multi-condition OR (`color == "red" OR size >= 90`) — verifies qdrant unions
/// two clauses into `Filter.should` (not `must`), searching the whole union.
#[test]
fn test_binary_qdrant_or_filter() {
    wait_for_qdrant();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "qdrant-or", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_or_filter_project("or-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    let grpc = QDRANT_GRPC_PORT.to_string();
    let rest = QDRANT_REST_PORT.to_string();
    assert!(
        common::run_binary(
            &proj.root,
            "qdrant-or",
            "or-test",
            "localhost",
            &[
                ("QDRANT_GRPC_PORT", grpc.as_str()),
                ("QDRANT_REST_PORT", rest.as_str()),
            ],
        ),
        "qdrant or-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "qdrant-or");
    println!("qdrant or-filter recall={:.3}", recall);
    assert!(recall >= 0.9, "qdrant or-filter recall {:.3} < 0.9", recall);
}

/// Nested/grouped boolean filter — `(color == "red" AND size >= 50) OR
/// (color == "blue" AND size < 10)`. The condition is a top-level `or` whose two
/// arms are themselves `and` GROUPS, so it can only be answered by nesting each
/// group as its OWN sub-Filter (Filter.must) inside the parent Filter.should. A
/// builder that mis-flattens the sub-trees matches a wildly different doc set, so
/// recall >= 0.9 proves the native nesting is correct.
#[test]
fn test_binary_qdrant_nested_filter() {
    wait_for_qdrant();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "qdrant-nested", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_nested_filter_project(
        "nested-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);
    let grpc = QDRANT_GRPC_PORT.to_string();
    let rest = QDRANT_REST_PORT.to_string();
    assert!(
        common::run_binary(
            &proj.root,
            "qdrant-nested",
            "nested-test",
            "localhost",
            &[
                ("QDRANT_GRPC_PORT", grpc.as_str()),
                ("QDRANT_REST_PORT", rest.as_str()),
            ],
        ),
        "qdrant nested-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "qdrant-nested");
    println!("qdrant nested-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "qdrant nested-filter recall {:.3} < 0.9",
        recall
    );
}

/// UUID exact-match filter end-to-end. Qdrant maps the `uuid` schema type to its
/// dedicated `FieldType::Uuid` payload index (distinct from keyword), which was
/// otherwise untested — this proves an exact `uid == UUIDS[0]` match selects the
/// quarter of docs it should through that special index.
#[test]
fn test_binary_qdrant_uuid() {
    wait_for_qdrant();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "qdrant-uuid", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_uuid_project("uuid-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    let grpc = QDRANT_GRPC_PORT.to_string();
    let rest = QDRANT_REST_PORT.to_string();
    assert!(
        common::run_binary(
            &proj.root,
            "qdrant-uuid",
            "uuid-test",
            "localhost",
            &[
                ("QDRANT_GRPC_PORT", grpc.as_str()),
                ("QDRANT_REST_PORT", rest.as_str()),
            ],
        ),
        "qdrant uuid run failed"
    );
    let recall = common::read_recall(&proj.root, "qdrant-uuid");
    println!("qdrant uuid recall={:.3}", recall);
    assert!(recall >= 0.9, "qdrant uuid recall {:.3} < 0.9", recall);
}

/// Selectivity ladder: one `rank < K` range query per rung, sweeping filter
/// selectivity from ~3% to ~99% in a single dataset. Verifies qdrant's
/// filterable HNSW keeps recall across the whole selectivity range (recall vs
/// per-rung ground truth), not just at one operating point.
#[test]
fn test_binary_qdrant_selectivity() {
    wait_for_qdrant();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "qdrant-sel", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_selectivity_project(
        "sel-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);
    let grpc = QDRANT_GRPC_PORT.to_string();
    let rest = QDRANT_REST_PORT.to_string();
    assert!(
        common::run_binary(
            &proj.root,
            "qdrant-sel",
            "sel-test",
            "localhost",
            &[
                ("QDRANT_GRPC_PORT", grpc.as_str()),
                ("QDRANT_REST_PORT", rest.as_str()),
            ],
        ),
        "qdrant selectivity run failed"
    );
    let recall = common::read_recall(&proj.root, "qdrant-sel");
    println!("qdrant selectivity recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "qdrant selectivity recall {:.3} < 0.9",
        recall
    );
}

/// Control for the multi-valued `labels` fixture (#88). Qdrant already stores
/// `labels` as a native list payload and matches per element, so it must clear
/// 0.9 recall. If this fails alongside the Milvus/Weaviate/pgvector labels
/// tests, the fixture (not an engine fix) is at fault.
#[test]
fn test_binary_qdrant_match_any_labels() {
    wait_for_qdrant();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "qdrant-mal", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
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

    let grpc = QDRANT_GRPC_PORT.to_string();
    let rest = QDRANT_REST_PORT.to_string();
    assert!(
        common::run_binary(
            &proj.root,
            "qdrant-mal",
            "match-any-labels-test",
            "localhost",
            &[
                ("QDRANT_GRPC_PORT", grpc.as_str()),
                ("QDRANT_REST_PORT", rest.as_str()),
            ],
        ),
        "qdrant match_any labels run failed"
    );

    let recall = common::read_recall(&proj.root, "qdrant-mal");
    println!("qdrant match_any labels recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "qdrant multi-valued labels match_any recall {:.3} < 0.9",
        recall
    );
}

// ---------------------------------------------------------------------------
// Quantization coverage: SCALAR (int8), BINARY, and PRODUCT quantization all
// run end-to-end through the real CLI against live qdrant on the SAME
// FIXED-SEED dataset.
//
// Quantization is LOSSY, so we use a realistic dimensionality (dim=64) and
// enable `rescore:true` + generous `oversampling` in search_params: the
// quantized index picks an oversampled candidate set, then qdrant re-ranks it
// against the FULL-PRECISION vectors. With rescore on, recall recovers to a
// high floor.
//
// Because rescore re-ranks against full-precision vectors, a high rescored
// recall alone does NOT prove quantization was applied — qdrant silently
// ignores unused quantization search params, so a run whose quantization_config
// was dropped would still score ~1.0. The teeth therefore come from a
// NO-RESCORE negative control (`test_binary_qdrant_quantization_is_applied`):
// binary search WITHOUT rescore reads the 1-bit-quantized vectors directly and
// must be MATERIALLY LOSSIER than the full-precision baseline — which can only
// happen if the quantization_config was genuinely applied to the collection.
// (Read-back of the collection config is impossible: the CLI drops the
// collection at the end of every run, see experiment.rs `engine.delete()`.)
// ---------------------------------------------------------------------------

/// The one shared, DETERMINISTIC dataset (data vectors + queries + brute-force
/// ground truth) that every quantization mode is run against. Built with a
/// fixed-seed `StdRng` so the corpus, queries and recall are identical every
/// run (matching the tests/common fixtures' seeding convention).
struct QuantDataset {
    vectors: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    neighbors: Vec<Vec<i64>>,
    dim: usize,
}

/// Fixed-seed UNIFORM vectors in [-1, 1] (NOT gaussian). Deterministic so the
/// dataset — and therefore the recall floors — are reproducible across runs.
fn seeded_vectors(seed: u64, count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect()
}

/// Build the shared quantization dataset: 1000 docs + 20 DISTINCT queries (NOT
/// copies of stored points, so quantization loss actually matters), dim=64.
fn build_quant_dataset() -> QuantDataset {
    let dim = 64;
    let top = 10;
    // Two different seeds → queries are distinct from the corpus.
    let vectors = seeded_vectors(0x0DE1, 1000, dim);
    let queries = seeded_vectors(0x0DE2, 20, dim);
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors_l2(q, &vectors, top))
        .collect();
    QuantDataset {
        vectors,
        queries,
        neighbors,
        dim,
    }
}

/// Run one config end-to-end and return the reported RECALL (`mean_recall`).
/// `quantization_config` is the `collection_params` quantization object (None =
/// full-precision baseline); `search_quant` is the `search_params.quantization`
/// object (None = plain search, no rescore/oversampling).
fn run_quantization_mode(
    engine_name: &str,
    dataset: &str,
    quantization_config: Option<serde_json::Value>,
    search_quant: Option<serde_json::Value>,
    data: &QuantDataset,
) -> f64 {
    let mut collection_params = serde_json::json!({ "timeout": 120 });
    if let Some(qc) = quantization_config {
        collection_params["quantization_config"] = qc;
    }
    let mut search_params = serde_json::json!({ "hnsw_ef": 256 });
    if let Some(sq) = search_quant {
        search_params["quantization"] = sq;
    }
    let configs = serde_json::json!([{
        "name": engine_name, "engine": "qdrant",
        "connection_params": {"timeout": 120},
        "collection_params": collection_params,
        "search_params": [{"parallel": 1, "search_params": search_params}],
        "upload_params": {"parallel": 1, "batch_size": 256}
    }]);
    let root = write_dense_project(
        dataset,
        &serde_json::to_string(&configs).unwrap(),
        &data.vectors,
        &data.queries,
        &data.neighbors,
        data.dim,
    );
    assert!(
        run_qdrant_binary(&root, engine_name, dataset),
        "{engine_name} run failed (collection did not build/search)"
    );
    // `mean_recall` (recall@K = hits/K), matching the "recall" label used below.
    common::read_recall(&root, engine_name)
}

/// End-to-end SCALAR / BINARY / PRODUCT quantization coverage on one shared
/// fixed-seed dataset. Each mode must build its quantized collection, search
/// with rescore, and clear a recall floor tuned against live qdrant.
#[test]
fn test_binary_qdrant_quantization_modes() {
    wait_for_qdrant();

    let data = build_quant_dataset();

    // SCALAR int8: rescore recovers near-exact recall with modest oversampling.
    let sq = run_quantization_mode(
        "qdrant-quant-sq",
        "quant-sq",
        Some(serde_json::json!({"scalar": {"type": "int8", "always_ram": true}})),
        Some(serde_json::json!({"rescore": true, "oversampling": 4.0})),
        &data,
    );
    println!("qdrant scalar(int8) quantization recall={sq:.3}");

    // PRODUCT x16: coarser than scalar, still recovers well under rescore.
    let pq = run_quantization_mode(
        "qdrant-quant-pq",
        "quant-pq",
        Some(serde_json::json!({"product": {"compression": "x16", "always_ram": true}})),
        Some(serde_json::json!({"rescore": true, "oversampling": 4.0})),
        &data,
    );
    println!("qdrant product(x16) quantization recall={pq:.3}");

    // BINARY: 1-bit-per-dim is the lossiest mode, so on undifferentiated uniform
    // data the binary index needs the highest oversampling to surface the true
    // neighbours into the rescore candidate set. Oversampling only widens the
    // full-precision rescore candidate pool (limit * oversampling), so a larger
    // value trades a little search time for a higher, MORE STABLE recall without
    // touching the no-rescore negative control in
    // `test_binary_qdrant_quantization_is_applied`. At oversampling=8 the binary
    // arm occasionally dipped just under the 0.9 floor (observed ~0.895) because
    // binary is the lossiest mode and qdrant's HNSW construction is run-to-run
    // nondeterministic; oversampling=20 rescores 200 full-precision candidates
    // (top=10) out of 1000 docs, which keeps recall comfortably above the floor
    // on every run.
    let bq = run_quantization_mode(
        "qdrant-quant-bq",
        "quant-bq",
        Some(serde_json::json!({"binary": {"always_ram": true}})),
        Some(serde_json::json!({"rescore": true, "oversampling": 20.0})),
        &data,
    );
    println!("qdrant binary quantization recall={bq:.3}");

    // Floors tuned against the FIXED-SEED dataset (fully reproducible).
    // Observed: scalar=1.000, product=1.000, binary=1.000. All floors set to
    // 0.9 — a meaningful bar with margin. (This test proves the quantized
    // collections BUILD and SEARCH; that quantization is actually APPLIED to
    // the read path is proven by the no-rescore control below.)
    assert!(sq >= 0.9, "scalar(int8) quantization recall {sq:.3} < 0.9");
    assert!(pq >= 0.9, "product(x16) quantization recall {pq:.3} < 0.9");
    assert!(bq >= 0.9, "binary quantization recall {bq:.3} < 0.9");
}

/// PROOF-OF-APPLICATION (teeth): prove quantization is genuinely on the read
/// path, not silently dropped. Binary quantization searched WITHOUT rescore
/// reads the 1-bit-quantized vectors directly, so on dim-64 data it must be
/// MATERIALLY LOSSIER than the full-precision baseline. If the
/// quantization_config were ignored/dropped, the "binary" collection would be
/// plain full precision and this gap would vanish — so the gap assertion fails
/// closed. (Mirrors the negative-control pattern of `test_binary_qdrant_hybrid`.)
#[test]
fn test_binary_qdrant_quantization_is_applied() {
    wait_for_qdrant();

    let data = build_quant_dataset();

    // Full-precision baseline: NO quantization at all → upper-bound recall.
    let baseline =
        run_quantization_mode("qdrant-quant-baseline", "quant-baseline", None, None, &data);
    println!("qdrant full-precision baseline recall={baseline:.3}");

    // BINARY, NO rescore (oversampling 1.0): searches purely on the 1-bit
    // quantized vectors. Only reaches this (much lower) recall if quantization
    // was actually applied to the collection.
    let bq_no_rescore = run_quantization_mode(
        "qdrant-quant-bq-norescore",
        "quant-bq-norescore",
        Some(serde_json::json!({"binary": {"always_ram": true}})),
        Some(serde_json::json!({"rescore": false, "oversampling": 1.0})),
        &data,
    );
    println!("qdrant binary NO-rescore recall={bq_no_rescore:.3}");

    // The baseline must itself be high (sanity: the harness works).
    assert!(
        baseline >= 0.9,
        "full-precision baseline recall {baseline:.3} < 0.9 (harness broken?)"
    );
    // TEETH: quantization must materially degrade recall when rescore is off.
    // A dropped/ignored quantization_config would leave binary == full precision
    // and this margin would collapse to ~0.
    let margin = baseline - bq_no_rescore;
    println!("qdrant quantization-applied margin (baseline - binary_no_rescore) = {margin:.3}");
    assert!(
        margin > 0.1,
        "binary NO-rescore recall {bq_no_rescore:.3} is not materially below \
         baseline {baseline:.3} (margin {margin:.3} <= 0.1) — quantization does \
         not appear to be applied to the read path"
    );
}

// ---------------------------------------------------------------------------
// generate-dataset binary coverage (issue #122): prove that a dataset written
// by the `generate-dataset` binary is consumable end-to-end by the benchmark.
//
// This is stronger than the fixture-based `test_binary_qdrant_sparse`: it runs
// the SHIPPED binary (its own CLI, its own on-disk writes into a fresh
// `datasets/` dir), registers the result exactly as `datasets/datasets.json`
// does (local `path`, no download link), and runs the real engine against it.
// ---------------------------------------------------------------------------

/// Path to the compiled `generate-dataset` binary (Cargo exports this env var
/// to integration tests automatically).
fn generate_dataset_bin() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_generate-dataset"))
}

#[test]
fn test_generate_dataset_binary_sparse_end_to_end() {
    wait_for_qdrant();

    // Fresh temp project: <root>/datasets, <root>/experiments/..., <root>/results.
    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    let datasets_dir = root.join("datasets");
    std::fs::create_dir_all(&datasets_dir).unwrap();
    std::fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    std::fs::create_dir_all(root.join("results")).unwrap();

    // 1. Run the REAL generator binary, writing only the sparse dataset.
    let ds_name = "synthetic-sparse-300";
    let out = std::process::Command::new(generate_dataset_bin())
        .args([
            "--out-dir",
            datasets_dir.to_str().unwrap(),
            "--only",
            "sparse",
        ])
        .output()
        .expect("run generate-dataset");
    assert!(
        out.status.success(),
        "generate-dataset failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Layout check: the sparse reader needs exactly these three files.
    for f in ["data.csr", "queries.csr", "neighbours.jsonl"] {
        assert!(
            datasets_dir.join(ds_name).join(f).exists(),
            "generator did not write {f}"
        );
    }

    // 2. Register the generated dataset (local path, NO link) + an engine config,
    //    exactly as datasets/datasets.json does for the shipped entry.
    let datasets_json = serde_json::json!([{
        "name": ds_name, "type": "sparse", "path": ds_name,
        "distance": "dot", "vector_size": 300,
    }]);
    std::fs::write(
        datasets_dir.join("datasets.json"),
        serde_json::to_string_pretty(&datasets_json).unwrap(),
    )
    .unwrap();
    let configs = serde_json::json!([{
        "name": "qdrant-gen-sparse", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1}], "upload_params": {"parallel": 1, "batch_size": 50}
    }]);
    std::fs::write(
        root.join("experiments/configurations/test.json"),
        serde_json::to_string(&configs).unwrap(),
    )
    .unwrap();

    // 3. Run the benchmark against the GENERATED dataset and check recall.
    assert!(
        run_qdrant_binary(&root, "qdrant-gen-sparse", ds_name),
        "benchmark run over generated sparse dataset failed"
    );
    let precision = read_precision(&root, "qdrant-gen-sparse");
    println!("generated sparse dataset precision={precision:.3}");
    assert!(
        precision >= 0.9,
        "generated sparse precision {precision:.3} < 0.9"
    );
}

// ---------------------------------------------------------------------------
// Collection-parameter passthrough (upstream parity): the on-disk / tenant
// knobs used by upstream's on-disk experiments must reach the SERVER, not just
// parse.
//
// These are exactly the knobs that fail silently: a dropped
// `hnsw_config.on_disk` still produces a working collection with plausible
// recall — it just benchmarks an in-memory graph while the config, the result
// file and the run name all say "on disk". Behavioural teeth (as used for
// quantization) cannot separate those two, so this test READS THE COLLECTION
// CONFIG BACK from a live Qdrant instead.
//
// To make read-back possible the CLI is run with `--keep-data` (which skips the
// usual teardown) into its own collection name.
// ---------------------------------------------------------------------------

/// Like `run_qdrant_binary`, but keeps the collection after the run and points
/// the engine at `collection` so the caller can inspect the result.
fn run_qdrant_binary_keep_collection(
    root: &std::path::Path,
    engine: &str,
    dataset: &str,
    collection: &str,
) -> bool {
    let out = std::process::Command::new(binary_path())
        .args([
            "--engines",
            engine,
            "--datasets",
            dataset,
            "--host",
            "localhost",
            "--skip-if-exists",
            "false",
            "--keep-data",
        ])
        .env("QDRANT_GRPC_PORT", QDRANT_GRPC_PORT.to_string())
        .env("QDRANT_REST_PORT", QDRANT_REST_PORT.to_string())
        .env("QDRANT_COLLECTION_NAME", collection)
        .current_dir(root)
        .output()
        .expect("run vector-db-benchmark");
    if !out.status.success() {
        eprintln!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    out.status.success()
}

#[test]
fn test_binary_qdrant_on_disk_and_tenant_collection_params() {
    wait_for_qdrant();

    let collection = "bench_on_disk_params";
    let engine = "qdrant-on-disk-params";
    let dataset = "tenant-on-disk";

    // `search_params` is spelled `config` here on purpose: that is upstream's
    // spelling, so this proves a config written that way runs end-to-end and
    // still returns correct results. (It does NOT prove `hnsw_ef` reached the
    // server — that only affects search quality/latency; the recall floor below
    // is what would catch a search broken by this collection layout.)
    let configs = serde_json::json!([{
        "name": engine,
        "engine": "qdrant",
        "connection_params": {"timeout": 120},
        "collection_params": {
            "timeout": 120,
            "vectors_config": { "on_disk": true, "datatype": "float16" },
            // m: 0 + payload_m: build graphs per tenant value only — the
            // multi-tenancy layout upstream benchmarks.
            "hnsw_config": { "m": 0, "ef_construct": 64, "on_disk": true, "payload_m": 16,
                             "inline_storage": true },
            // DELIBERATELY `false`. Qdrant v1.18.2 DEFAULTS on_disk_payload to
            // true (verified live), so asserting `true` here would pass even with
            // the forwarding deleted — a vacuous assertion. Only a real
            // forwarding path can produce `false`.
            "on_disk_payload": false,
            "payload_index_params": { "tenant": { "is_tenant": true, "on_disk": true } }
        },
        // with_payload exercises the search knob added alongside these collection
        // params; recall must stay correct with payloads coming back.
        "search_params": [{ "parallel": 1, "config": { "hnsw_ef": 64, "with_payload": true } }],
        "upload_params": {"parallel": 1, "batch_size": 128}
    }]);
    let proj = common::write_tenant_project(dataset, &serde_json::to_string(&configs).unwrap(), 16);

    let ok = run_qdrant_binary_keep_collection(&proj.root, engine, dataset, collection);

    // Read the collection config back BEFORE asserting, so the collection is
    // always cleaned up even when an assertion fails below.
    let (rt, client) = create_grpc_client();
    let info = if ok {
        rt.block_on(client.collection_info(collection)).ok()
    } else {
        None
    };
    let _ = rt.block_on(client.delete_collection(collection));

    assert!(ok, "{engine} run failed");
    let info = info.expect("collection_info").result.expect("info result");

    let config = info.config.expect("collection config");
    let params = config.params.expect("collection params");
    assert!(
        !params.on_disk_payload,
        "on_disk_payload: false was not applied — the collection sits on Qdrant's \
         own default (true), so the configured value never reached the server"
    );

    let hnsw = config.hnsw_config.expect("hnsw config");
    assert_eq!(hnsw.m, Some(0), "hnsw_config.m: 0 did not reach the server");
    assert_eq!(hnsw.ef_construct, Some(64));
    assert_eq!(
        hnsw.inline_storage,
        Some(true),
        "hnsw_config.inline_storage did not reach the server"
    );
    assert_eq!(
        hnsw.on_disk,
        Some(true),
        "hnsw_config.on_disk did not reach the server — the run would have \
         benchmarked an IN-MEMORY graph while claiming to be on disk"
    );
    assert_eq!(
        hnsw.payload_m,
        Some(16),
        "hnsw_config.payload_m did not reach the server"
    );

    // vectors_config: on_disk + float16 storage.
    let vectors = params.vectors_config.expect("vectors config").config;
    match vectors {
        Some(qdrant_client::qdrant::vectors_config::Config::Params(vp)) => {
            assert_eq!(vp.on_disk, Some(true), "vectors_config.on_disk not applied");
            assert_eq!(
                vp.datatype,
                Some(qdrant_client::qdrant::Datatype::Float16 as i32),
                "vectors_config.datatype float16 not applied — storage would be \
                 full float32"
            );
        }
        other => panic!("expected single dense vector params, got {other:?}"),
    }

    // payload_index_params: the keyword index on `tenant` must be a TENANT index
    // kept on disk.
    let schema = info
        .payload_schema
        .get("tenant")
        .expect("tenant payload index should exist");
    match schema.params.as_ref().and_then(|p| p.index_params.as_ref()) {
        Some(qdrant_client::qdrant::payload_index_params::IndexParams::KeywordIndexParams(k)) => {
            assert_eq!(
                k.is_tenant,
                Some(true),
                "payload_index_params.is_tenant not applied — the index would not \
                 be tenant-grouped"
            );
            assert_eq!(k.on_disk, Some(true), "payload index on_disk not applied");
        }
        other => panic!("expected keyword index params on `tenant`, got {other:?}"),
    }

    // The layout above is the one most likely to break SEARCH rather than
    // creation: `m: 0` means there is no global HNSW graph, so results can only
    // come back via the per-tenant payload graphs, and float16 storage is lossy.
    // Asserting the config alone would pass at recall 0. Ground truth here is
    // tenant-local and exact (write_tenant_project), so this also catches a
    // dropped tenant filter.
    let recall = common::read_recall(&proj.root, engine);
    println!("qdrant tenant/on-disk recall={recall:.3}");
    assert!(
        recall > 0.9,
        "recall {recall:.3} <= 0.9 — the on-disk/tenant collection built, but search \
         over it is broken (m:0 + payload_m graphs, float16 storage, or the tenant \
         filter)"
    );
}

/// REGRESSION: an OMITTED `vectors_config.on_disk` must NOT be sent as an
/// explicit `false`.
///
/// Behavioural assertions cannot tell an mmap'd vector storage from an in-memory
/// one — which is precisely why this survived — so this reads the collection
/// config BACK from a live Qdrant and asserts the field is absent.
///
/// Why absence matters (verified on qdrant v1.18.2 at `memmap_threshold: 1` by
/// inspecting the segment layout): omitting `on_disk` lets the optimizer mmap
/// the vector storage (`vector_storage/matrix.dat`), while an explicit `false`
/// OVERRIDES the threshold and pins the vectors in RAM
/// (`vector_storage/vectors/chunk_0.mmap`). Sending `unwrap_or(false)` therefore
/// silently disabled mmap for `qdrant-on-disk-default` and for all six
/// `qdrant-mmap-*` configurations, whose names, configs and results all claimed
/// on-disk storage.
#[test]
fn test_binary_qdrant_omitted_vectors_on_disk_stays_omitted() {
    wait_for_qdrant();

    let collection = "bench_omitted_on_disk";
    let engine = "qdrant-omitted-on-disk";
    let dataset = "omitted-on-disk";

    // NOTE: no `vectors_config` at all — mmap is meant to be driven purely by
    // optimizers_config.memmap_threshold, exactly like the qdrant-mmap-* configs.
    let configs = serde_json::json!([{
        "name": engine,
        "engine": "qdrant",
        "connection_params": {"timeout": 120},
        "collection_params": {
            "optimizers_config": { "memmap_threshold": 10000 },
            "hnsw_config": { "m": 16, "ef_construct": 64 }
        },
        "search_params": [{ "parallel": 1, "config": { "hnsw_ef": 64 } }],
        "upload_params": {"parallel": 1, "batch_size": 128}
    }]);
    let proj = common::write_tenant_project(dataset, &serde_json::to_string(&configs).unwrap(), 16);

    let ok = run_qdrant_binary_keep_collection(&proj.root, engine, dataset, collection);

    let (rt, client) = create_grpc_client();
    let info = if ok {
        rt.block_on(client.collection_info(collection)).ok()
    } else {
        None
    };
    let _ = rt.block_on(client.delete_collection(collection));

    assert!(ok, "{engine} run failed");
    let info = info.expect("collection_info").result.expect("info result");
    let config = info.config.expect("collection config");
    let params = config.params.expect("collection params");

    match params.vectors_config.expect("vectors config").config {
        Some(qdrant_client::qdrant::vectors_config::Config::Params(vp)) => {
            assert_eq!(
                vp.on_disk, None,
                "an omitted vectors_config.on_disk was sent as an explicit value \
                 ({:?}); an explicit `false` overrides memmap_threshold and pins \
                 the vectors in RAM",
                vp.on_disk
            );
        }
        other => panic!("expected single dense vector params, got {other:?}"),
    }

    // The threshold itself must still have arrived, otherwise this test would
    // pass vacuously on a collection that was never mmap-eligible to begin with.
    let opt = config.optimizer_config.expect("optimizer config");
    assert_eq!(opt.memmap_threshold, Some(10000));
}

/// End-to-end run over BINARY `results.gt` ground truth — the layout the public
/// `msmarco-sparse-*` datasets ship, and the only branch of `read_sparse_queries`
/// that no other test exercises (every other fixture writes `neighbours.jsonl`).
///
/// What this pins that a reader unit test cannot: the ids inside `results.gt`
/// must be 0-based row indices lining up with the ids the uploader assigns from
/// `data.csr` row order. If they were 1-based, or MS MARCO document ids, recall
/// on a published `msmarco-sparse-1M` run would collapse to ~0 and every other
/// test in this repo would still be green.
#[test]
fn test_binary_qdrant_sparse_binary_ground_truth_end_to_end() {
    wait_for_qdrant();

    let configs = serde_json::json!([{
        "name": "qdrant-sparse-gt", "engine": "qdrant",
        "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
        "search_params": [{"parallel": 1}], "upload_params": {"parallel": 1, "batch_size": 50}
    }]);
    let proj =
        common::write_sparse_project_gt("sparse-gt", &serde_json::to_string(&configs).unwrap());

    // The fixture must carry ONLY the binary layout, otherwise this silently
    // falls back to the jsonl branch and pins nothing.
    let ds_dir = proj.root.join("datasets").join(&proj.dataset_name);
    assert!(ds_dir.join("results.gt").exists(), "results.gt not written");
    assert!(
        !ds_dir.join("neighbours.jsonl").exists(),
        "neighbours.jsonl would take precedence and hide the results.gt path"
    );

    assert!(
        run_qdrant_binary(&proj.root, "qdrant-sparse-gt", "sparse-gt"),
        "sparse results.gt run failed"
    );
    let precision = read_precision(&proj.root, "qdrant-sparse-gt");
    println!(
        "qdrant sparse (results.gt) precision={:.3} (top={})",
        precision, proj.top
    );
    assert!(
        precision >= 0.9,
        "sparse results.gt precision {:.3} < 0.9 — the binary ground-truth ids do \
         not line up with the point ids the uploader assigns",
        precision
    );
}
