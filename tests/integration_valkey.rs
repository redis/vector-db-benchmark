//! Integration tests for Valkey engine (Valkey Search FT.* commands).
//!
//! Requires valkey/valkey-bundle running on port 6380.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d valkey
//! Run with:   VALKEY_PORT=6380 cargo test --test integration_valkey -- --test-threads=1

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;
use redis::Connection;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_port() -> u16 {
    std::env::var("VALKEY_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6380)
}

const TEST_HOST: &str = "127.0.0.1";

fn get_test_connection() -> Connection {
    let url = format!("redis://{}:{}/", TEST_HOST, test_port());
    let client = redis::Client::open(url.as_str()).expect("Failed to create Valkey client");
    client
        .get_connection()
        .expect("Failed to connect to Valkey. Is valkey-bundle running?")
}

fn wait_for_valkey() {
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
            panic!("Valkey not available on port {} after 30s", test_port());
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn flush_db(conn: &mut Connection) {
    // Drop all FT indexes first (Valkey Search does not support DD flag)
    if let Ok(indexes) = redis::cmd("FT._LIST").query::<Vec<String>>(conn) {
        for idx_name in indexes {
            let _ = redis::cmd("FT.DROPINDEX").arg(&idx_name).query::<()>(conn);
        }
    }
    let _: () = redis::cmd("FLUSHALL").query(conn).unwrap();
}

fn generate_test_vectors(count: usize, dim: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let ids: Vec<i64> = (0..count as i64).collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (ids, vectors)
}

fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn brute_force_neighbors(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
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

fn parse_ft_search_ids(response: &[redis::Value]) -> Vec<i64> {
    let mut ids = Vec::new();
    let mut i = 1;
    while i < response.len() {
        let id_str = match &response[i] {
            redis::Value::BulkString(data) => Some(String::from_utf8_lossy(data).to_string()),
            redis::Value::SimpleString(s) => Some(s.clone()),
            _ => None,
        };
        if let Some(s) = id_str {
            if let Ok(id) = s.parse::<i64>() {
                ids.push(id);
            }
        }
        i += 2;
    }
    ids
}

fn extract_ft_info_value(info: &redis::Value, key: &str) -> Option<i64> {
    fn value_to_string(v: &redis::Value) -> Option<String> {
        match v {
            redis::Value::BulkString(s) => Some(String::from_utf8_lossy(s).to_string()),
            redis::Value::SimpleString(s) => Some(s.clone()),
            _ => None,
        }
    }
    fn value_to_i64(v: &redis::Value) -> Option<i64> {
        match v {
            redis::Value::Int(n) => Some(*n),
            redis::Value::BulkString(s) => String::from_utf8_lossy(s).parse::<i64>().ok(),
            redis::Value::SimpleString(s) => s.parse::<i64>().ok(),
            redis::Value::Double(f) => Some(*f as i64),
            _ => None,
        }
    }
    match info {
        redis::Value::Map(pairs) => {
            for (k, v) in pairs {
                if value_to_string(k).as_deref() == Some(key) {
                    return value_to_i64(v);
                }
            }
            None
        }
        redis::Value::Array(items) => {
            for i in 0..items.len().saturating_sub(1) {
                if value_to_string(&items[i]).as_deref() == Some(key) {
                    return value_to_i64(&items[i + 1]);
                }
            }
            None
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Valkey (Valkey Search FT.*) Integration Tests
// ---------------------------------------------------------------------------

/// Verify parallel pipelined uploads (matching the engine's threaded path)
/// persist every vector and that DBSIZE / FT.INFO num_docs agree.
#[test]
fn test_valkey_parallel_pipeline_upload_keyspace() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 500;
    let batch_size = 64;
    let num_threads = 8;
    let (ids, vectors) = generate_test_vectors(count, dim);

    // Create index
    redis::cmd("FT.CREATE")
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
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    // Build batches
    let batches: Vec<(usize, usize)> = (0..ids.len())
        .step_by(batch_size)
        .map(|start| (start, (start + batch_size).min(ids.len())))
        .collect();

    let batch_idx = std::sync::atomic::AtomicUsize::new(0);
    let total_batches = batches.len();
    let port = test_port();

    // Parallel upload using std::thread::scope (same as engine code)
    std::thread::scope(|s| {
        for _ in 0..num_threads {
            let batches = &batches;
            let batch_idx = &batch_idx;
            let ids = &ids;
            let vectors = &vectors;

            s.spawn(move || {
                let url = format!("redis://127.0.0.1:{}/", port);
                let client = redis::Client::open(url.as_str()).unwrap();
                let mut t_conn = client.get_connection().unwrap();

                loop {
                    let idx = batch_idx.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if idx >= total_batches {
                        break;
                    }
                    let (start, end) = batches[idx];
                    let mut pipe = redis::pipe();
                    for i in start..end {
                        let key = ids[i].to_string();
                        let vec_bytes = vec_to_bytes(&vectors[i]);
                        let mut cmd = redis::cmd("HSET");
                        cmd.arg(key.as_str()).arg("vector").arg(&vec_bytes[..]);
                        pipe.add_command(cmd).ignore();
                    }
                    pipe.query::<()>(&mut t_conn)
                        .unwrap_or_else(|e| panic!("Pipeline batch {} failed: {}", idx, e));
                }
            });
        }
    });

    thread::sleep(Duration::from_millis(1000));

    // Verify DBSIZE
    let dbsize: i64 = redis::cmd("DBSIZE").query(&mut conn).unwrap();
    assert_eq!(
        dbsize, count as i64,
        "DBSIZE ({}) must equal uploaded count ({})",
        dbsize, count
    );

    // Verify FT.INFO num_docs
    let info: redis::Value = redis::cmd("FT.INFO")
        .arg("idx")
        .query(&mut conn)
        .expect("FT.INFO failed");
    let num_docs = extract_ft_info_value(&info, "num_docs").unwrap_or(-1);
    assert_eq!(
        num_docs, count as i64,
        "FT.INFO num_docs ({}) must equal uploaded count ({})",
        num_docs, count
    );

    // Cleanup
    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .query(&mut conn)
        .unwrap();
}

/// Verify that pipelined uploads (matching the engine's code path) actually
/// persist every vector and that DBSIZE / FT.INFO num_docs agree.
#[test]
fn test_valkey_pipeline_upload_keyspace() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 500;
    let batch_size = 64; // matches valkey-docker-test config
    let (ids, vectors) = generate_test_vectors(count, dim);

    // Create index
    redis::cmd("FT.CREATE")
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
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    // Upload using the same pattern as the engine: pipe.add_command(cmd).ignore()
    for chunk_start in (0..ids.len()).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(ids.len());
        let mut pipe = redis::pipe();
        for i in chunk_start..chunk_end {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            let mut cmd = redis::cmd("HSET");
            cmd.arg(key.as_str()).arg("vector").arg(&vec_bytes[..]);
            pipe.add_command(cmd).ignore();
        }
        pipe.query::<()>(&mut conn)
            .unwrap_or_else(|e| panic!("Pipeline batch at offset {} failed: {}", chunk_start, e));
    }

    thread::sleep(Duration::from_millis(1000));

    // Verify DBSIZE
    let dbsize: i64 = redis::cmd("DBSIZE").query(&mut conn).unwrap();
    assert_eq!(
        dbsize, count as i64,
        "DBSIZE ({}) must equal uploaded count ({})",
        dbsize, count
    );

    // Verify FT.INFO num_docs
    let info: redis::Value = redis::cmd("FT.INFO")
        .arg("idx")
        .query(&mut conn)
        .expect("FT.INFO failed");
    let num_docs = extract_ft_info_value(&info, "num_docs").unwrap_or(-1);
    assert_eq!(
        num_docs, count as i64,
        "FT.INFO num_docs ({}) must equal uploaded count ({})",
        num_docs, count
    );

    // Cleanup
    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_valkey_upload_and_retrieve() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 4;
    let (ids, vectors) = generate_test_vectors(10, dim);

    // Create index
    redis::cmd("FT.CREATE")
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
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    // Upload vectors via HSET
    for i in 0..ids.len() {
        let key = ids[i].to_string();
        let vec_bytes = vec_to_bytes(&vectors[i]);
        redis::cmd("HSET")
            .arg(&key)
            .arg("vector")
            .arg(&vec_bytes[..])
            .query::<()>(&mut conn)
            .expect("HSET failed");
    }

    // Verify vectors stored
    for id in &ids {
        let key = id.to_string();
        let exists: bool = redis::cmd("EXISTS").arg(&key).query(&mut conn).unwrap();
        assert!(exists, "Key {} should exist", key);
    }

    // Cleanup
    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_valkey_knn_search() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 100;
    let top = 10;
    let (ids, vectors) = generate_test_vectors(count, dim);

    redis::cmd("FT.CREATE")
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
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    // Upload via pipeline (batched)
    for chunk_start in (0..ids.len()).step_by(50) {
        let chunk_end = (chunk_start + 50).min(ids.len());
        let mut pipe = redis::pipe();
        for i in chunk_start..chunk_end {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            pipe.cmd("HSET")
                .arg(key)
                .arg("vector")
                .arg(vec_bytes)
                .ignore();
        }
        pipe.query::<()>(&mut conn).expect("Pipeline HSET failed");
    }

    thread::sleep(Duration::from_millis(500));

    // Verify keyspace matches expected vector count
    let dbsize: i64 = redis::cmd("DBSIZE").query(&mut conn).unwrap();
    assert_eq!(
        dbsize, count as i64,
        "DBSIZE should match uploaded vector count"
    );

    let info: redis::Value = redis::cmd("FT.INFO").arg("idx").query(&mut conn).unwrap();
    let num_docs = extract_ft_info_value(&info, "num_docs").unwrap_or(0);
    assert_eq!(
        num_docs, count as i64,
        "FT.INFO num_docs should match uploaded vector count"
    );

    let query_vec = &vectors[0];
    let query_bytes = vec_to_bytes(query_vec);

    // Valkey Search: DIALECT 2 only, no EF_RUNTIME, no SORTBY on computed fields
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN $K @vector $vec_param AS vector_score]")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(2)
        .arg("PARAMS")
        .arg(4)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .query(&mut conn)
        .expect("FT.SEARCH failed");

    assert!(!response.is_empty(), "Should get search results");

    if let redis::Value::BulkString(data) = &response[1] {
        let top_id: i64 = String::from_utf8_lossy(data).parse().unwrap();
        assert_eq!(top_id, 0, "Query vector should be its own nearest neighbor");
    }

    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_valkey_knn_precision() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 200;
    let top = 5;
    let (ids, vectors) = generate_test_vectors(count, dim);

    redis::cmd("FT.CREATE")
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
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    for chunk_start in (0..ids.len()).step_by(50) {
        let chunk_end = (chunk_start + 50).min(ids.len());
        let mut pipe = redis::pipe();
        for i in chunk_start..chunk_end {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            pipe.cmd("HSET")
                .arg(key)
                .arg("vector")
                .arg(vec_bytes)
                .ignore();
        }
        pipe.query::<()>(&mut conn).expect("Pipeline HSET failed");
    }

    thread::sleep(Duration::from_millis(500));

    let query_idx = 42;
    let query_vec = &vectors[query_idx];
    let expected = brute_force_neighbors(query_vec, &vectors, top);

    let query_bytes = vec_to_bytes(query_vec);
    // Valkey Search: DIALECT 2 only, no EF_RUNTIME
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN $K @vector $vec_param AS vector_score]")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(2)
        .arg("PARAMS")
        .arg(4)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .query(&mut conn)
        .expect("FT.SEARCH failed");

    let mut result_ids: Vec<i64> = Vec::new();
    let mut i = 1;
    while i < response.len() {
        if let redis::Value::BulkString(data) = &response[i] {
            if let Ok(id) = String::from_utf8_lossy(data).parse::<i64>() {
                result_ids.push(id);
            }
        }
        i += 2;
    }

    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let found_set: std::collections::HashSet<i64> = result_ids.iter().copied().collect();
    let hits = expected_set.intersection(&found_set).count();
    let precision = hits as f64 / top as f64;

    assert!(
        precision >= 0.8,
        "Precision should be >= 0.8, got {} (expected {:?}, found {:?})",
        precision,
        expected_set,
        found_set
    );

    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_valkey_filtered_knn_search() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 20;
    let top = 5;
    let (ids, vectors) = generate_test_vectors(count, dim);

    // Create index with TAG + NUMERIC metadata fields
    // Note: Valkey Search does not support SORTABLE keyword
    redis::cmd("FT.CREATE")
        .arg("idx_filter")
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg("1")
        .arg("")
        .arg("SCHEMA")
        .arg("vector")
        .arg("VECTOR")
        .arg("HNSW")
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .arg("category")
        .arg("TAG")
        .arg("SEPARATOR")
        .arg(";")
        .arg("price")
        .arg("NUMERIC")
        .query::<()>(&mut conn)
        .expect("FT.CREATE with metadata failed");

    // Upload: even IDs = "electronics", odd = "clothing"; price = 10 * id
    {
        let mut pipe = redis::pipe();
        for i in 0..ids.len() {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            let category = if i % 2 == 0 {
                "electronics"
            } else {
                "clothing"
            };
            let price = (i * 10).to_string();
            pipe.cmd("HSET")
                .arg(key)
                .arg("vector")
                .arg(vec_bytes)
                .arg("category")
                .arg(category)
                .arg("price")
                .arg(price)
                .ignore();
        }
        pipe.query::<()>(&mut conn)
            .expect("Pipeline HSET with metadata failed");
    }

    thread::sleep(Duration::from_millis(500));

    let query_vec = &vectors[0];
    let query_bytes = vec_to_bytes(query_vec);

    // TAG filter (electronics only) - Valkey Search compatible
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx_filter")
        .arg("@category:{electronics}=>[KNN $K @vector $vec_param AS vector_score]")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(2)
        .arg("PARAMS")
        .arg(4)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .query(&mut conn)
        .expect("FT.SEARCH with TAG filter failed");

    let tag_ids = parse_ft_search_ids(&response);
    assert!(!tag_ids.is_empty(), "TAG filter should return results");
    for id in &tag_ids {
        assert_eq!(
            *id % 2,
            0,
            "TAG filter for electronics should only return even IDs, got {}",
            id
        );
    }

    // NUMERIC range filter (price < 100) - Valkey Search compatible
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx_filter")
        .arg("@price:[-inf (100]=>[KNN $K @vector $vec_param AS vector_score]")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(2)
        .arg("PARAMS")
        .arg(4)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .query(&mut conn)
        .expect("FT.SEARCH with NUMERIC filter failed");

    let num_ids = parse_ft_search_ids(&response);
    assert!(
        !num_ids.is_empty(),
        "NUMERIC range filter should return results"
    );
    for id in &num_ids {
        assert!(
            *id < 10,
            "NUMERIC filter price<100 should only return ids<10, got {}",
            id
        );
    }

    // Cleanup
    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx_filter")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_valkey_ft_info() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 4;
    let count = 50;

    redis::cmd("FT.CREATE")
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
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    let (ids, vectors) = generate_test_vectors(count, dim);
    {
        let mut pipe = redis::pipe();
        for i in 0..ids.len() {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            pipe.cmd("HSET")
                .arg(key)
                .arg("vector")
                .arg(vec_bytes)
                .ignore();
        }
        pipe.query::<()>(&mut conn).expect("Pipeline HSET failed");
    }

    thread::sleep(Duration::from_millis(500));

    let info: redis::Value = redis::cmd("FT.INFO")
        .arg("idx")
        .query(&mut conn)
        .expect("FT.INFO failed");

    let num_docs = extract_ft_info_value(&info, "num_docs");
    assert!(num_docs.is_some(), "FT.INFO should contain num_docs field");
    assert_eq!(
        num_docs.unwrap(),
        count as i64,
        "FT.INFO num_docs should match uploaded count"
    );

    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .query(&mut conn)
        .unwrap();
}

/// Find the release binary path (same pattern as Redis integration tests)
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

/// Create a temporary project directory with inline dataset and engine config
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

    // vectors.jsonl
    let mut vecs_content = String::new();
    for v in vectors {
        let line: Vec<f64> = v.iter().map(|x| *x as f64).collect();
        vecs_content.push_str(&serde_json::to_string(&line).unwrap());
        vecs_content.push('\n');
    }
    fs::write(dataset_dir.join("vectors.jsonl"), &vecs_content).unwrap();

    // queries.jsonl
    let mut queries_content = String::new();
    for q in queries {
        let line: Vec<f64> = q.iter().map(|x| *x as f64).collect();
        queries_content.push_str(&serde_json::to_string(&line).unwrap());
        queries_content.push('\n');
    }
    fs::write(dataset_dir.join("queries.jsonl"), &queries_content).unwrap();

    // neighbours.jsonl
    let mut neighbors_content = String::new();
    for n in neighbors {
        neighbors_content.push_str(&serde_json::to_string(n).unwrap());
        neighbors_content.push('\n');
    }
    fs::write(dataset_dir.join("neighbours.jsonl"), &neighbors_content).unwrap();

    // datasets.json
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

    // Engine configs
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();

    root
}

#[test]
fn test_valkey_full_cycle_via_binary() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 10;
    let count = 50;
    let top = 5;
    let mut rng = rand::thread_rng();

    // Generate vectors & queries
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..10)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors(q, &vectors, top))
        .collect();

    let engine_config = serde_json::json!([{
        "name": "test-valkey-l2",
        "engine": "valkey",
        "connection_params": {},
        "collection_params": {
            "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 }
        },
        "search_params": [
            { "parallel": 1, "search_params": { "ef": 128 } }
        ],
        "upload_params": { "parallel": 1 }
    }]);

    let project_root = create_test_project(
        "test-l2",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        "l2",
        dim,
    );

    let bin = binary_path();
    assert!(
        bin.exists(),
        "Binary not found at {:?}. Run `cargo build --release` first.",
        bin
    );

    let output = Command::new(&bin)
        .args([
            "--engines",
            "test-valkey-l2",
            "--datasets",
            "test-l2",
            "--host",
            TEST_HOST,
        ])
        .env("VALKEY_PORT", test_port().to_string())
        .current_dir(&project_root)
        .output()
        .expect("Failed to run benchmark binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "Benchmark should complete successfully. Exit code: {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        stdout,
        stderr
    );

    // Verify output contains expected phases
    assert!(
        stdout.contains("Configure") || stdout.contains("configure"),
        "Output should mention Configure phase"
    );
    assert!(
        stdout.contains("Upload") || stdout.contains("upload"),
        "Output should mention Upload phase"
    );
    assert!(
        stdout.contains("Search") || stdout.contains("search") || stdout.contains("QPS"),
        "Output should mention Search phase or QPS"
    );

    // Verify stdout reports uploading the expected number of vectors
    // (DBSIZE will be 0 after the benchmark because it calls delete())
    let expected_count_msg = format!("Read {} vectors", count);
    assert!(
        stdout.contains(&expected_count_msg),
        "Output should report reading {} vectors, got:\n{}",
        count,
        stdout
    );
}

/// Verify sub-batched pipeline upload with high-dimensional vectors.
/// With dim=100 (400 bytes/vector), 64 HSETs would be ~28KB RESP data,
/// exceeding the 16KB sub-batch limit. This test confirms the sub-batching
/// pattern from the engine code works correctly.
#[test]
fn test_valkey_sub_batched_pipeline_upload_high_dim() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 100;
    let count = 200;
    let batch_size = 64;
    let max_pipe_bytes: usize = 4_096;
    let (ids, vectors) = generate_test_vectors(count, dim);

    redis::cmd("FT.CREATE")
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
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    // Upload using the same sub-batching pattern as the engine code
    for chunk_start in (0..ids.len()).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(ids.len());

        let mut pipe = redis::pipe();
        let mut pipe_bytes: usize = 0;

        for i in chunk_start..chunk_end {
            // Mirror the engine's now-prefixed key (#151-4) so this byte-accounting
            // test's RESP wire-size formula and 4096-byte flush boundaries still
            // track the real sub-batch flush path (a bare key would silently stop
            // guarding the changed path).
            let key = format!("valkey-sb:{}", ids[i]);
            let vec_bytes = vec_to_bytes(&vectors[i]);

            // Estimate RESP wire size (same formula as engine code)
            let num_args = 4; // HSET key vector <bytes>
            let cmd_bytes = format!("*{}\r\n", num_args).len()
                + 10 // $4\r\nHSET\r\n
                + format!("${}\r\n", key.len()).len() + key.len() + 2
                + 12 // $6\r\nvector\r\n
                + format!("${}\r\n", vec_bytes.len()).len() + vec_bytes.len() + 2;

            if pipe_bytes > 0 && pipe_bytes + cmd_bytes > max_pipe_bytes {
                pipe.query::<()>(&mut conn)
                    .unwrap_or_else(|e| panic!("Sub-batch flush failed at offset {}: {}", i, e));
                pipe = redis::pipe();
                pipe_bytes = 0;
            }

            let mut hset_cmd = redis::cmd("HSET");
            hset_cmd.arg(key.as_str()).arg("vector").arg(&vec_bytes[..]);
            pipe.add_command(hset_cmd).ignore();
            pipe_bytes += cmd_bytes;
        }

        if pipe_bytes > 0 {
            pipe.query::<()>(&mut conn).unwrap_or_else(|e| {
                panic!("Final sub-batch failed at chunk {}: {}", chunk_start, e)
            });
        }
    }

    thread::sleep(Duration::from_millis(1000));

    // Verify DBSIZE
    let dbsize: i64 = redis::cmd("DBSIZE").query(&mut conn).unwrap();
    assert_eq!(
        dbsize, count as i64,
        "DBSIZE ({}) must equal uploaded count ({})",
        dbsize, count
    );

    // Verify FT.INFO num_docs
    let info: redis::Value = redis::cmd("FT.INFO")
        .arg("idx")
        .query(&mut conn)
        .expect("FT.INFO failed");
    let num_docs = extract_ft_info_value(&info, "num_docs").unwrap_or(-1);
    assert_eq!(
        num_docs, count as i64,
        "FT.INFO num_docs ({}) must equal uploaded count ({})",
        num_docs, count
    );

    // Cleanup
    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .query(&mut conn)
        .unwrap();
}

/// End-to-end `match_any`: filter a keyword (TAG) field to an OR-set and assert
/// the engine returns the filtered nearest neighbours (recall vs ground truth
/// brute-forced over only the matching docs). Proves the inlined TAG-OR arm.
#[test]
fn test_binary_valkey_match_any() {
    wait_for_valkey();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "valkey-ma", "engine": "valkey",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
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

    let port = test_port().to_string();
    assert!(
        common::run_binary(
            &proj.root,
            "valkey-ma",
            "match-any-test",
            "127.0.0.1",
            &[("VALKEY_PORT", port.as_str())],
        ),
        "valkey match_any run failed"
    );

    let recall = common::read_recall(&proj.root, "valkey-ma");
    println!("valkey match_any recall={:.3}", recall);
    assert!(recall >= 0.9, "valkey match_any recall {:.3} < 0.9", recall);
}

/// Same filtered (`match_any`) search as above, but over the **RESP3** protocol
/// (`VALKEY_PROTOCOL=resp3`). Valkey Search returns FT.SEARCH results as a RESP3
/// map rather than the RESP2 array, so this exercises the RESP3 branch of the
/// response parser end-to-end. Recall must match the RESP2 path.
#[test]
fn test_binary_valkey_match_any_resp3() {
    wait_for_valkey();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "valkey-ma-r3", "engine": "valkey",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "match-any-test-r3",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = test_port().to_string();
    assert!(
        common::run_binary(
            &proj.root,
            "valkey-ma-r3",
            "match-any-test-r3",
            "127.0.0.1",
            &[("VALKEY_PORT", port.as_str()), ("VALKEY_PROTOCOL", "resp3")],
        ),
        "valkey match_any (RESP3) run failed"
    );

    let recall = common::read_recall(&proj.root, "valkey-ma-r3");
    println!("valkey match_any RESP3 recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "valkey RESP3 match_any recall {:.3} < 0.9 — RESP3 FT.SEARCH parsing broken?",
        recall
    );
}

/// End-to-end FILTER-ONLY harness (`--skip-vector-index`) at `parallel: 4` with
/// `--queries 1000`, so the per-worker thread-local latency buffers are merged
/// across threads (the join-merge path). Asserts the filter-only sentinel
/// (`mean_precision_at_returned == -1`), full query accounting (requested == succeeded,
/// failed == 0) on a healthy run, positive RPS, and monotone linear percentiles.
#[test]
fn test_binary_valkey_filter_only() {
    wait_for_valkey();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "valkey-fo", "engine": "valkey",
        "search_params": [{"parallel": 4, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "valkey-fo-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = test_port().to_string();
    assert!(
        common::run_binary_extra(
            &proj.root,
            "valkey-fo",
            "valkey-fo-test",
            "127.0.0.1",
            &[("VALKEY_PORT", port.as_str())],
            &["--skip-vector-index", "--queries", "1000"],
        ),
        "valkey filter-only run failed"
    );

    let r = common::read_results_obj(&proj.root, "valkey-no-vector");
    let mp = r["mean_precision_at_returned"].as_f64().unwrap();
    let rps = r["rps"].as_f64().unwrap();
    let p50 = r["p50_time"].as_f64().unwrap();
    let p95 = r["p95_time"].as_f64().unwrap();
    let p99 = r["p99_time"].as_f64().unwrap();
    let requested = r["requested_queries"].as_u64().unwrap();
    let succeeded = r["succeeded_queries"].as_u64().unwrap();
    let failed = r["failed_queries"].as_u64().unwrap();
    println!(
        "valkey filter-only: mean_precision_at_returned={mp} rps={rps:.1} p50={p50} p95={p95} p99={p99} \
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
    fs::remove_dir_all(&proj.root).ok();
}

/// End-to-end MIXED harness (`--update-search-ratio`) at `parallel: 4`: mirrors
/// `test_binary_redis_mixed_benchmark` but exercises the valkey mixed path with a
/// real multi-worker join-merge. Asserts search recall/precision are intact,
/// updates ran (`update_count > 0`, `update_rps > 0`), and the search percentiles
/// are monotone.
#[test]
fn test_binary_valkey_mixed_benchmark() {
    wait_for_valkey();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "valkey-mx", "engine": "valkey",
        "search_params": [{"parallel": 4, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    // 2000 queries so that at parallel: 4 the mixed loop reliably completes many
    // full search phases (and thus updates), and merges a large per-worker sample
    // set across threads.
    let proj = common::write_match_any_project_n(
        "valkey-mx-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
        2000,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = test_port().to_string();
    assert!(
        common::run_binary_extra(
            &proj.root,
            "valkey-mx",
            "valkey-mx-test",
            "127.0.0.1",
            &[("VALKEY_PORT", port.as_str())],
            &["--update-search-ratio", "1:5", "--repetitions", "1"],
        ),
        "valkey mixed run failed"
    );

    let r = common::read_results_obj(&proj.root, "valkey-mx");
    let recall = r["mean_recall"].as_f64().unwrap();
    let precision = r["mean_precision_at_returned"].as_f64().unwrap();
    let update_count = r["update_count"].as_u64().unwrap();
    let update_rps = r["update_rps"].as_f64().unwrap();
    let p50 = r["p50_time"].as_f64().unwrap();
    let p95 = r["p95_time"].as_f64().unwrap();
    let p99 = r["p99_time"].as_f64().unwrap();
    println!(
        "valkey mixed: recall={recall:.3} precision={precision:.3} update_count={update_count} \
         update_rps={update_rps:.1} p50={p50} p95={p95} p99={p99}"
    );
    assert!(precision >= 0.8, "mixed precision {precision} < 0.8");
    assert!(recall >= 0.9, "mixed recall {recall} < 0.9");
    assert!(update_count > 0, "mixed run performed no updates");
    assert!(update_rps > 0.0, "update_rps should be positive");
    assert!(
        p50 <= p95 && p95 <= p99,
        "percentiles must be monotone: p50={p50} p95={p95} p99={p99}"
    );
    fs::remove_dir_all(&proj.root).ok();
}

// ── New filter datatypes: bool / uuid / full-text / datetime ────────────────
//
// Each drives the real binary against a compound dataset whose queries carry a
// single filter type, with ground truth brute-forced over only the matching
// docs. Valkey has no TEXT field type, so the full-text case exercises the
// DEGRADED path (text tokenised to a multi-value TAG on upload + single-term TAG
// match); single-term recall must still clear the bar.

/// Shared driver: build the project with `build`, run the binary, assert recall.
fn run_filter_recall_test(
    name: &str,
    dataset: &str,
    build: impl Fn(&str, &str, usize) -> common::FilterProject,
) {
    wait_for_valkey();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": name, "engine": "valkey",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
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
            "127.0.0.1",
            &[("VALKEY_PORT", port.as_str())],
        ),
        "valkey {} run failed",
        name
    );

    let recall = common::read_recall(&proj.root, name);
    println!("valkey {} recall={:.3}", name, recall);
    assert!(recall >= 0.9, "valkey {} recall {:.3} < 0.9", name, recall);
}

#[test]
fn test_binary_valkey_bool() {
    run_filter_recall_test("valkey-bool", "bool-test", common::write_bool_project);
}

#[test]
fn test_binary_valkey_uuid() {
    run_filter_recall_test("valkey-uuid", "uuid-test", common::write_uuid_project);
}

#[test]
fn test_binary_valkey_fulltext() {
    run_filter_recall_test("valkey-text", "text-test", common::write_fulltext_project);
}

#[test]
fn test_binary_valkey_datetime() {
    run_filter_recall_test("valkey-dt", "dt-test", common::write_datetime_project);
}

/// Multi-condition AND (keyword match AND numeric range) — verifies Valkey Search
/// composes two conditions of different types into one intersecting FT.SEARCH
/// filter (`@color:{red} @size:[50 +inf]`), not just a single clause.
#[test]
fn test_binary_valkey_and_filter() {
    run_filter_recall_test("valkey-and", "and-test", common::write_and_filter_project);
}

/// Selectivity ladder: one `rank < K` range query per rung, sweeping filter
/// selectivity from ~3% to ~99% in a single dataset. Verifies the numeric-range
/// filter path stays correct across the whole selectivity range.
#[test]
fn test_binary_valkey_selectivity() {
    run_filter_recall_test("valkey-sel", "sel-test", common::write_selectivity_project);
}

/// Multi-condition OR: `color == "red" OR size >= 90` — verifies Valkey Search
/// UNIONs two clauses (top-level `{or:[...]}` → `@color:{red} | @size:[90 +inf]`),
/// not an intersection.
#[test]
fn test_binary_valkey_or_filter() {
    run_filter_recall_test("valkey-or", "or-test", common::write_or_filter_project);
}

/// Nested boolean: `(color==red AND size>=50) OR (color==blue AND size<10)` —
/// verifies the parser recurses into grouped sub-conditions rather than
/// flattening the tree (which would match a wildly different doc set).
#[test]
fn test_binary_valkey_nested_filter() {
    run_filter_recall_test(
        "valkey-nested",
        "nested-test",
        common::write_nested_filter_project,
    );
}

/// Multi-tenancy: many tenants share ONE index and every query is scoped to a
/// single tenant via a keyword-equality filter on a `tenant` field, with ground
/// truth brute-forced over ONLY that tenant's docs. Reuses the keyword-TAG
/// filter arm — no new engine code.
///
/// STRONGER than the other filter recall tests: the ground truth is tenant-local,
/// so a cross-tenant document that leaked into a result cannot count toward recall
/// AND displaces a correct neighbour. Search is exact (ef high over ~16 docs/tenant)
/// and there is one query per tenant, so a correct engine scores EXACTLY 1.0 on
/// every query. We therefore assert (a) mean recall == 1.0 and (b) EVERY per-query
/// recall == 1.0 — any single leaked or mis-scoped tenant fails. (The saved result
/// JSON records per-query recalls but not the raw returned ids, so the exact
/// per-query recall is the strongest available tenant-isolation check without
/// changing engine result serialization.)
#[test]
fn test_binary_valkey_tenancy() {
    wait_for_valkey();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "valkey-tenancy", "engine": "valkey",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_tenant_project(
        "tenancy-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(
        proj.matching_docs >= proj.top,
        "each tenant must have >= top docs (smallest tenant has {})",
        proj.matching_docs
    );

    let port = test_port().to_string();
    assert!(
        // --dump-raw-latencies: the per-tenant leakage check below reads the
        // full per-query `recalls` array, only emitted under this flag (results
        // otherwise carry compact HDR/quality digests).
        common::run_binary_extra(
            &proj.root,
            "valkey-tenancy",
            "tenancy-test",
            "127.0.0.1",
            &[("VALKEY_PORT", port.as_str())],
            &["--dump-raw-latencies"],
        ),
        "valkey tenancy run failed"
    );

    let recall = common::read_recall(&proj.root, "valkey-tenancy");
    let per_query = common::read_recalls(&proj.root, "valkey-tenancy");
    println!(
        "valkey tenancy mean recall={:.3} per-query={:?}",
        recall, per_query
    );
    // Exact per-query recall (see redis test): tenant-local ground truth means a
    // single leaked cross-tenant doc drops recall below 1.0.
    assert!(
        recall > 0.999,
        "valkey tenancy mean recall {:.4} != 1.0 — cross-tenant leakage or mis-scope?",
        recall
    );
    for (q, r) in per_query.iter().enumerate() {
        assert!(
            *r > 0.999,
            "valkey tenancy query {} recall {:.4} != 1.0 — cross-tenant leakage?",
            q,
            r
        );
    }
}

/// Count keys under a glob (small test keyspace).
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

/// #151-4 regression (valkey mirror): "upload all, then --skip-upload search each"
/// gives every config its OWN graph. Two configs (dense high-ef vs sparse low-ef)
/// coexist via disjoint `idx:<config>` indexes + `<config>:` keyspaces. Pre-fix
/// valkey's create_index FLUSHALL + shared `idx` collapsed both to one graph.
#[test]
fn test_valkey_coexistence_skip_upload() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 32;
    let count = 2000;
    let top = 10;
    let (_, vectors) = generate_test_vectors(count, dim);
    let queries: Vec<Vec<f32>> = vectors[..20].to_vec();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors(q, &vectors, top))
        .collect();

    let engine_config = serde_json::json!([
        {
            "name": "valkey-co-a",
            "engine": "valkey",
            "algorithm": "hnsw",
            "collection_params": { "hnsw_config": { "M": 64, "EF_CONSTRUCTION": 200 } },
            "search_params": [{ "parallel": 1, "search_params": { "ef": 256 }, "top": top }],
            "upload_params": { "data_type": "FLOAT32", "parallel": 1, "batch_size": 64 }
        },
        {
            "name": "valkey-co-b",
            "engine": "valkey",
            "algorithm": "hnsw",
            "collection_params": { "hnsw_config": { "M": 4, "EF_CONSTRUCTION": 8 } },
            "search_params": [{ "parallel": 1, "search_params": { "ef": 10 }, "top": top }],
            "upload_params": { "data_type": "FLOAT32", "parallel": 1, "batch_size": 64 }
        }
    ]);

    let root = create_test_project(
        "valkey-co",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        "l2",
        dim,
    );
    let port = test_port().to_string();

    // Phase 1: upload + search BOTH, KEEPING data for the skip-upload phase.
    assert!(
        common::run_binary_extra(
            &root,
            "valkey-co-*",
            "valkey-co",
            TEST_HOST,
            &[("VALKEY_PORT", port.as_str())],
            &["--keep-data"],
        ),
        "valkey coexistence phase 1 failed"
    );

    let base_a = common::read_recall(&root, "valkey-co-a");
    let base_b = common::read_recall(&root, "valkey-co-b");

    // Deterministic coexistence: `count` keys under EACH per-config prefix, and
    // both disjoint indexes exist.
    assert_eq!(
        count_keys(&mut conn, "valkey-co-a:*"),
        count,
        "valkey-co-a keyspace"
    );
    assert_eq!(
        count_keys(&mut conn, "valkey-co-b:*"),
        count,
        "valkey-co-b keyspace"
    );
    assert!(
        redis::cmd("FT.INFO")
            .arg("idx:valkey-co-a")
            .query::<redis::Value>(&mut conn)
            .is_ok(),
        "idx:valkey-co-a must exist"
    );
    assert!(
        redis::cmd("FT.INFO")
            .arg("idx:valkey-co-b")
            .query::<redis::Value>(&mut conn)
            .is_ok(),
        "idx:valkey-co-b must exist"
    );

    delete_search_result_files(&root);

    // Phase 2: --skip-upload search of both against the coexisting indexes.
    assert!(
        common::run_binary_extra(
            &root,
            "valkey-co-*",
            "valkey-co",
            TEST_HOST,
            &[("VALKEY_PORT", port.as_str())],
            &["--skip-upload", "--keep-data"],
        ),
        "valkey coexistence phase 2 (--skip-upload) failed"
    );

    let rec_a = common::read_recall(&root, "valkey-co-a");
    let rec_b = common::read_recall(&root, "valkey-co-b");

    assert!(
        (rec_a - base_a).abs() < 1e-9,
        "valkey-co-a skip-upload recall {} != baseline {}",
        rec_a,
        base_a
    );
    assert!(
        (rec_b - base_b).abs() < 1e-9,
        "valkey-co-b skip-upload recall {} != baseline {}",
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

/// #151-4 negative (valkey mirror): `--skip-upload` with NO prior upload must FAIL
/// LOUDLY (index-existence guard), never writing a recall-0.0 result file.
#[test]
fn test_valkey_skip_upload_without_prior_upload_errors() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 200;
    let top = 5;
    let (_, vectors) = generate_test_vectors(count, dim);
    let queries: Vec<Vec<f32>> = vectors[..10].to_vec();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors(q, &vectors, top))
        .collect();

    let engine_config = serde_json::json!([{
        "name": "valkey-noupload",
        "engine": "valkey",
        "algorithm": "hnsw",
        "collection_params": { "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 } },
        "search_params": [{ "parallel": 1, "search_params": { "ef": 64 }, "top": top }],
        "upload_params": { "data_type": "FLOAT32", "parallel": 1, "batch_size": 64 }
    }]);

    let root = create_test_project(
        "valkey-noupload",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        "l2",
        dim,
    );
    let port = test_port().to_string();

    let ok = common::run_binary_extra(
        &root,
        "valkey-noupload",
        "valkey-noupload",
        TEST_HOST,
        &[("VALKEY_PORT", port.as_str())],
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
