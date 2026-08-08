//! Integration tests for Redis and VectorSets engines.
//!
//! Requires redis:8.6.0 running on port 6399.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d
//! Run with:   cargo test --test integration_redis -- --test-threads=1

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;
use redis::Connection;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Redis port under test. Overridable so a session can run against its OWN
/// container instead of the shared 6399 one — these tests call `flush_db()`, so
/// two sessions sharing an instance clobber each other.
fn test_port() -> u16 {
    std::env::var("REDIS_TEST_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6399)
}
const TEST_HOST: &str = "127.0.0.1";

fn get_test_connection() -> Connection {
    let url = format!("redis://{}:{}/", TEST_HOST, test_port());
    let client = redis::Client::open(url.as_str()).expect("Failed to create Redis client");
    client
        .get_connection()
        .expect("Failed to connect to Redis. Is redis:8.6.0 running on port 6399?")
}

fn wait_for_redis() {
    let deadline = Instant::now() + Duration::from_secs(10);
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
            panic!("Redis not available on port {} after 10s", test_port());
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn flush_db(conn: &mut Connection) {
    // Drop all FT indexes first (FLUSHALL does NOT remove them in Redis 8)
    if let Ok(indexes) = redis::cmd("FT._LIST").query::<Vec<String>>(conn) {
        for idx_name in indexes {
            let _ = redis::cmd("FT.DROPINDEX")
                .arg(&idx_name)
                .arg("DD")
                .query::<()>(conn);
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

/// Compute brute-force nearest neighbors (by L2 distance) for a query against a set of vectors.
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

/// Compute brute-force nearest neighbors by cosine distance (1 - cosine_similarity).
fn brute_force_neighbors_cosine(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
    let q_norm: f64 = query
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let mut dists: Vec<(i64, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f64 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let v_norm: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let cos_sim = if q_norm * v_norm > 0.0 {
                dot / (q_norm * v_norm)
            } else {
                0.0
            };
            (i as i64, 1.0 - cos_sim)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(top).map(|(id, _)| *id).collect()
}

/// Parse FT.SEARCH result IDs from response (skipping field-value pairs).
fn parse_ft_search_ids(response: &[redis::Value]) -> Vec<i64> {
    let mut ids = Vec::new();
    let mut i = 1; // skip total count at index 0
    while i < response.len() {
        // Doc IDs may be BulkString (RESP2) or SimpleString (Redis 8)
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
        i += 2; // skip field values
    }
    ids
}

/// Extract a numeric value from an FT.INFO response by key name.
/// Handles both RESP2 (flat Array of key/value pairs) and RESP3 (Map).
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
        // RESP3 Map: Vec<(Value, Value)>
        redis::Value::Map(pairs) => {
            for (k, v) in pairs {
                if value_to_string(k).as_deref() == Some(key) {
                    return value_to_i64(v);
                }
            }
            None
        }
        // RESP2 flat array: [key, value, key, value, ...]
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
// Redis (RediSearch FT.*) Integration Tests
// ---------------------------------------------------------------------------

#[test]
fn test_redis_upload_and_retrieve() {
    wait_for_redis();
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
        .arg("DD")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_redis_knn_search() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 100;
    let top = 10;
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

    // Upload via pipeline
    {
        let mut pipe = redis::pipe();
        for i in 0..ids.len() {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            let mut cmd = redis::cmd("HSET");
            cmd.arg(key).arg("vector").arg(&vec_bytes[..]);
            pipe.add_command(cmd);
        }
        pipe.query::<()>(&mut conn).expect("Pipeline HSET failed");
    }

    // Wait for indexing
    thread::sleep(Duration::from_millis(500));

    // Search using first vector as query
    let query_vec = &vectors[0];
    let query_bytes = vec_to_bytes(query_vec);

    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(4)
        .arg("PARAMS")
        .arg(6)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .arg("EF")
        .arg(128)
        .query(&mut conn)
        .expect("FT.SEARCH failed");

    // First element is total count
    assert!(!response.is_empty(), "Should get search results");

    // The query vector itself should be the top-1 result (distance ~0)
    if let redis::Value::BulkString(data) = &response[1] {
        let top_id: i64 = String::from_utf8_lossy(data).parse().unwrap();
        assert_eq!(top_id, 0, "Query vector should be its own nearest neighbor");
    }

    // Cleanup
    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .arg("DD")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_redis_knn_precision() {
    wait_for_redis();
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

    {
        let mut pipe = redis::pipe();
        for i in 0..ids.len() {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            let mut cmd = redis::cmd("HSET");
            cmd.arg(key).arg("vector").arg(&vec_bytes[..]);
            pipe.add_command(cmd);
        }
        pipe.query::<()>(&mut conn).expect("Pipeline HSET failed");
    }

    thread::sleep(Duration::from_millis(500));

    // Pick a query vector from the dataset
    let query_idx = 42;
    let query_vec = &vectors[query_idx];
    let expected = brute_force_neighbors(query_vec, &vectors, top);

    let query_bytes = vec_to_bytes(query_vec);
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(4)
        .arg("PARAMS")
        .arg(6)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .arg("EF")
        .arg(256)
        .query(&mut conn)
        .expect("FT.SEARCH failed");

    // Parse result IDs
    let mut result_ids: Vec<i64> = Vec::new();
    let mut i = 1;
    while i < response.len() {
        if let redis::Value::BulkString(data) = &response[i] {
            if let Ok(id) = String::from_utf8_lossy(data).parse::<i64>() {
                result_ids.push(id);
            }
        }
        i += 2; // skip field values
    }

    // Compute precision
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
        .arg("DD")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_redis_metadata_upload() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let vec_bytes = vec_to_bytes(&[1.0f32, 2.0, 3.0, 4.0]);

    // Upload a hash with vector + metadata fields
    redis::cmd("HSET")
        .arg("meta_test_0")
        .arg("vector")
        .arg(&vec_bytes[..])
        .arg("category")
        .arg("electronics")
        .arg("tags")
        .arg("sale;new")
        .query::<()>(&mut conn)
        .expect("HSET with metadata failed");

    // Verify metadata stored
    let category: String = redis::cmd("HGET")
        .arg("meta_test_0")
        .arg("category")
        .query(&mut conn)
        .unwrap();
    assert_eq!(category, "electronics");

    let tags: String = redis::cmd("HGET")
        .arg("meta_test_0")
        .arg("tags")
        .query(&mut conn)
        .unwrap();
    assert_eq!(tags, "sale;new");
}

#[test]
fn test_redis_filtered_knn_search() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 20;
    let top = 5;
    let (ids, vectors) = generate_test_vectors(count, dim);

    // Create index with vector + TAG + NUMERIC metadata fields
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
        .arg("SORTABLE")
        .arg("price")
        .arg("NUMERIC")
        .arg("SORTABLE")
        .query::<()>(&mut conn)
        .expect("FT.CREATE with metadata failed");

    // Upload vectors with metadata: even IDs = "electronics", odd = "clothing"
    // price = 10 * id (0, 10, 20, ..., 190)
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
            let mut cmd = redis::cmd("HSET");
            cmd.arg(key)
                .arg("vector")
                .arg(&vec_bytes[..])
                .arg("category")
                .arg(category)
                .arg("price")
                .arg(&price);
            pipe.add_command(cmd);
        }
        pipe.query::<()>(&mut conn)
            .expect("Pipeline HSET with metadata failed");
    }

    thread::sleep(Duration::from_millis(500));

    let query_vec = &vectors[0];
    let query_bytes = vec_to_bytes(query_vec);

    // --- Test 1: TAG filter (electronics only) ---
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx_filter")
        .arg("@category:{electronics}=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(4)
        .arg("PARAMS")
        .arg(6)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .arg("EF")
        .arg(128)
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

    // --- Test 2: NUMERIC range filter (price 50..100 → IDs 5-10) ---
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx_filter")
        .arg("@price:[50 100]=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(4)
        .arg("PARAMS")
        .arg(6)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .arg("EF")
        .arg(128)
        .query(&mut conn)
        .expect("FT.SEARCH with NUMERIC filter failed");

    let range_ids = parse_ft_search_ids(&response);
    assert!(
        !range_ids.is_empty(),
        "NUMERIC range filter should return results"
    );
    for id in &range_ids {
        let price = *id * 10;
        assert!(
            (50..=100).contains(&price),
            "NUMERIC filter should only return IDs with price in [50,100], got id={} price={}",
            id,
            price
        );
    }

    // --- Test 3: AND condition (electronics AND price 0..80) ---
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx_filter")
        .arg(
            "(@category:{electronics} @price:[0 80])=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]",
        )
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(4)
        .arg("PARAMS")
        .arg(6)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .arg("EF")
        .arg(128)
        .query(&mut conn)
        .expect("FT.SEARCH with AND filter failed");

    let and_ids = parse_ft_search_ids(&response);
    assert!(!and_ids.is_empty(), "AND filter should return results");
    for id in &and_ids {
        assert_eq!(
            *id % 2,
            0,
            "AND filter: should be electronics (even), got {}",
            id
        );
        let price = *id * 10;
        assert!(
            (0..=80).contains(&price),
            "AND filter: price should be in [0,80], got id={} price={}",
            id,
            price
        );
    }

    // --- Test 4: No filter (baseline) ---
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx_filter")
        .arg("*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(4)
        .arg("PARAMS")
        .arg(6)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .arg("EF")
        .arg(128)
        .query(&mut conn)
        .expect("FT.SEARCH unfiltered failed");

    let all_ids = parse_ft_search_ids(&response);
    assert_eq!(
        all_ids.len(),
        top,
        "Unfiltered search should return exactly top={} results",
        top
    );

    assert!(
        tag_ids.len() <= all_ids.len(),
        "Filtered results should not exceed unfiltered results"
    );

    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx_filter")
        .arg("DD")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_redis_cosine_precision() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 200;
    let top = 5;
    let (ids, vectors) = generate_test_vectors(count, dim);

    redis::cmd("FT.CREATE")
        .arg("idx_cos")
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
        .expect("FT.CREATE with COSINE failed");

    {
        let mut pipe = redis::pipe();
        for i in 0..ids.len() {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            let mut cmd = redis::cmd("HSET");
            cmd.arg(key).arg("vector").arg(&vec_bytes[..]);
            pipe.add_command(cmd);
        }
        pipe.query::<()>(&mut conn).expect("Pipeline HSET failed");
    }

    thread::sleep(Duration::from_millis(500));

    let query_idx = 42;
    let query_vec = &vectors[query_idx];
    let expected = brute_force_neighbors_cosine(query_vec, &vectors, top);

    let query_bytes = vec_to_bytes(query_vec);
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx_cos")
        .arg("*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(4)
        .arg("PARAMS")
        .arg(6)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .arg("EF")
        .arg(256)
        .query(&mut conn)
        .expect("FT.SEARCH COSINE failed");

    let result_ids = parse_ft_search_ids(&response);

    // Query vector should be its own nearest neighbor
    assert!(
        !result_ids.is_empty() && result_ids[0] == query_idx as i64,
        "Query vector should be top-1 result, got {:?}",
        result_ids
    );

    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let found_set: std::collections::HashSet<i64> = result_ids.iter().copied().collect();
    let hits = expected_set.intersection(&found_set).count();
    let precision = hits as f64 / top as f64;

    assert!(
        precision >= 0.8,
        "COSINE precision should be >= 0.8, got {} (expected {:?}, found {:?})",
        precision,
        expected_set,
        found_set
    );

    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx_cos")
        .arg("DD")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_redis_parallel_upload_search() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 500;
    let top = 5;
    let num_threads = 4;
    let (ids, vectors) = generate_test_vectors(count, dim);

    redis::cmd("FT.CREATE")
        .arg("idx_par")
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

    // Parallel upload across 4 threads
    let chunk_size = count.div_ceil(num_threads);
    thread::scope(|s| {
        for chunk_idx in 0..num_threads {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(count);
            let ids_slice = &ids[start..end];
            let vecs_slice = &vectors[start..end];

            s.spawn(move || {
                let mut t_conn = get_test_connection();
                let mut pipe = redis::pipe();
                for i in 0..ids_slice.len() {
                    let key = ids_slice[i].to_string();
                    let vec_bytes = vec_to_bytes(&vecs_slice[i]);
                    let mut cmd = redis::cmd("HSET");
                    cmd.arg(key).arg("vector").arg(&vec_bytes[..]);
                    pipe.add_command(cmd);
                }
                pipe.query::<()>(&mut t_conn)
                    .expect("Parallel pipeline HSET failed");
            });
        }
    });

    thread::sleep(Duration::from_millis(1000));

    // Verify document count via FT.INFO
    let info: redis::Value = redis::cmd("FT.INFO")
        .arg("idx_par")
        .query(&mut conn)
        .expect("FT.INFO failed");

    // Extract num_docs from FT.INFO response.
    // Handles RESP2 (flat array of key/value) and RESP3 (Map).
    let num_docs = extract_ft_info_value(&info, "num_docs");

    assert_eq!(
        num_docs,
        Some(count as i64),
        "All {} documents should be indexed after parallel upload",
        count
    );

    // Parallel search across 4 threads.
    // Store results indexed by query number so precision check uses the right entry.
    let num_queries = 20;
    let query_idx = Arc::new(AtomicUsize::new(0));
    #[allow(clippy::type_complexity)]
    let results: Arc<Mutex<Vec<(usize, Vec<i64>)>>> = Arc::new(Mutex::new(Vec::new()));

    thread::scope(|s| {
        for _ in 0..num_threads {
            let qi = Arc::clone(&query_idx);
            let res = Arc::clone(&results);
            let vecs = &vectors;

            s.spawn(move || {
                let mut t_conn = get_test_connection();
                loop {
                    let idx = qi.fetch_add(1, Ordering::SeqCst);
                    if idx >= num_queries {
                        break;
                    }
                    let qv = &vecs[idx];
                    let qb = vec_to_bytes(qv);

                    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
                        .arg("idx_par")
                        .arg("*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
                        .arg("SORTBY")
                        .arg("vector_score")
                        .arg("ASC")
                        .arg("LIMIT")
                        .arg(0)
                        .arg(top)
                        .arg("RETURN")
                        .arg(1)
                        .arg("vector_score")
                        .arg("DIALECT")
                        .arg(4)
                        .arg("PARAMS")
                        .arg(6)
                        .arg("vec_param")
                        .arg(&qb[..])
                        .arg("K")
                        .arg(top)
                        .arg("EF")
                        .arg(128)
                        .query(&mut t_conn)
                        .expect("Parallel FT.SEARCH failed");

                    let r_ids = parse_ft_search_ids(&response);
                    res.lock().unwrap().push((idx, r_ids));
                }
            });
        }
    });

    let all_results = results.lock().unwrap();
    assert_eq!(
        all_results.len(),
        num_queries,
        "All {} parallel queries should produce results",
        num_queries
    );
    for (qi, r) in all_results.iter() {
        assert!(
            !r.is_empty(),
            "Query {} should return at least 1 result",
            qi
        );
    }

    // Precision check on query 0
    let query0_result = all_results.iter().find(|(qi, _)| *qi == 0).unwrap();
    let expected = brute_force_neighbors(&vectors[0], &vectors, top);
    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let found_set: std::collections::HashSet<i64> = query0_result.1.iter().copied().collect();
    let hits = expected_set.intersection(&found_set).count();
    let precision = hits as f64 / top as f64;
    assert!(
        precision >= 0.6,
        "Parallel search precision should be >= 0.6, got {}",
        precision
    );

    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx_par")
        .arg("DD")
        .query(&mut conn)
        .unwrap();
}

// ---------------------------------------------------------------------------
// VectorSets (VADD/VSIM) Integration Tests
// ---------------------------------------------------------------------------

#[test]
fn test_vectorsets_vadd_and_vsim() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
    ];

    // Upload vectors using VADD
    for (i, vec) in vectors.iter().enumerate() {
        let vec_bytes = vec_to_bytes(vec);
        redis::cmd("VADD")
            .arg("vset_basic")
            .arg("FP32")
            .arg(&vec_bytes[..])
            .arg(i.to_string())
            .arg("NOQUANT")
            .arg("CAS")
            .query::<()>(&mut conn)
            .unwrap_or_else(|e| panic!("VADD failed for vector {}: {}", i, e));
    }

    // Search for a vector close to [1,0,0,0]
    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let query_bytes = vec_to_bytes(&query);

    let response: Vec<redis::Value> = redis::cmd("VSIM")
        .arg("vset_basic")
        .arg("FP32")
        .arg(&query_bytes[..])
        .arg("WITHSCORES")
        .arg("COUNT")
        .arg(3)
        .query(&mut conn)
        .expect("VSIM failed");

    // Should get results back
    assert!(
        response.len() >= 2,
        "Expected at least one result pair, got {:?}",
        response
    );

    // Parse first result
    let top_id = match &response[0] {
        redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<i64>().unwrap(),
        redis::Value::Int(n) => *n,
        _ => panic!("Unexpected response type: {:?}", response[0]),
    };

    // ID 0 ([1,0,0,0]) should be most similar to query [1,0,0,0]
    assert_eq!(top_id, 0, "Vector 0 should be the closest match");

    // Cleanup
    let _: () = redis::cmd("DEL")
        .arg("vset_basic")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_vectorsets_score_conversion() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    // Upload two orthogonal unit vectors
    let v0 = vec![1.0f32, 0.0];
    let v1 = vec![0.0f32, 1.0];

    for (i, vec) in [&v0, &v1].iter().enumerate() {
        let vec_bytes = vec_to_bytes(vec);
        redis::cmd("VADD")
            .arg("vset_scores")
            .arg("FP32")
            .arg(&vec_bytes[..])
            .arg(i.to_string())
            .arg("NOQUANT")
            .arg("CAS")
            .query::<()>(&mut conn)
            .unwrap();
    }

    // Search with v0 as query
    let query_bytes = vec_to_bytes(&v0);
    let response: Vec<redis::Value> = redis::cmd("VSIM")
        .arg("vset_scores")
        .arg("FP32")
        .arg(&query_bytes[..])
        .arg("WITHSCORES")
        .arg("COUNT")
        .arg(2)
        .query(&mut conn)
        .expect("VSIM failed");

    // Parse scores: VectorSets returns similarity (1 = identical, 0 = orthogonal)
    assert!(response.len() >= 4, "Expected 2 result pairs");

    let score0 = match &response[1] {
        redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<f64>().unwrap(),
        redis::Value::Double(f) => *f,
        _ => panic!("Unexpected score type"),
    };
    let score1 = match &response[3] {
        redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<f64>().unwrap(),
        redis::Value::Double(f) => *f,
        _ => panic!("Unexpected score type"),
    };

    // After 1-score conversion: identical vector -> 0, orthogonal -> ~1
    let dist0 = 1.0 - score0;
    let dist1 = 1.0 - score1;

    assert!(
        dist0 < 0.01,
        "Self-similarity distance should be ~0, got {}",
        dist0
    );
    assert!(
        dist1 > 0.4,
        "Orthogonal distance should be large, got {}",
        dist1
    );

    let _: () = redis::cmd("DEL")
        .arg("vset_scores")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_vectorsets_pipeline_batch() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 50;
    let (ids, vectors) = generate_test_vectors(count, dim);

    // Upload via pipeline (matching engine implementation)
    let mut pipe = redis::pipe();
    for i in 0..ids.len() {
        let vec_bytes = vec_to_bytes(&vectors[i]);
        let mut cmd = redis::cmd("VADD");
        cmd.arg("vset_pipe")
            .arg("FP32")
            .arg(&vec_bytes[..])
            .arg(ids[i].to_string())
            .arg("NOQUANT")
            .arg("M")
            .arg(16)
            .arg("EF")
            .arg(200)
            .arg("CAS");
        pipe.add_command(cmd);
    }
    pipe.query::<()>(&mut conn).expect("Pipeline VADD failed");

    // Verify we can search and get results
    let query_bytes = vec_to_bytes(&vectors[0]);
    let response: Vec<redis::Value> = redis::cmd("VSIM")
        .arg("vset_pipe")
        .arg("FP32")
        .arg(&query_bytes[..])
        .arg("WITHSCORES")
        .arg("COUNT")
        .arg(5)
        .arg("EF")
        .arg(64)
        .query(&mut conn)
        .expect("VSIM failed");

    assert!(
        response.len() >= 2,
        "Should get at least 1 result from VSIM"
    );

    // The query vector (id=0) should appear in results
    let top_id = match &response[0] {
        redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<i64>().unwrap(),
        redis::Value::Int(n) => *n,
        _ => panic!("Unexpected type"),
    };
    assert_eq!(top_id, 0, "Query vector should be its own nearest neighbor");

    let _: () = redis::cmd("DEL").arg("vset_pipe").query(&mut conn).unwrap();
}

#[test]
fn test_vectorsets_knn_precision() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 200;
    let top = 5;
    let (_, vectors) = generate_test_vectors(count, dim);

    // Upload via pipeline with NOQUANT, M=16, EF=200
    let mut pipe = redis::pipe();
    for (i, vec) in vectors.iter().enumerate() {
        let vec_bytes = vec_to_bytes(vec);
        let mut cmd = redis::cmd("VADD");
        cmd.arg("vset_precision")
            .arg("FP32")
            .arg(&vec_bytes[..])
            .arg(i.to_string())
            .arg("NOQUANT")
            .arg("M")
            .arg(16)
            .arg("EF")
            .arg(200)
            .arg("CAS");
        pipe.add_command(cmd);
    }
    pipe.query::<()>(&mut conn).expect("Pipeline VADD failed");

    // Query with vector[42]
    let query_idx = 42;
    let query_vec = &vectors[query_idx];
    let expected = brute_force_neighbors_cosine(query_vec, &vectors, top);

    let query_bytes = vec_to_bytes(query_vec);
    let response: Vec<redis::Value> = redis::cmd("VSIM")
        .arg("vset_precision")
        .arg("FP32")
        .arg(&query_bytes[..])
        .arg("WITHSCORES")
        .arg("COUNT")
        .arg(top)
        .arg("EF")
        .arg(64)
        .query(&mut conn)
        .expect("VSIM precision query failed");

    // Parse results: alternating [id, score, id, score, ...]
    assert!(
        response.len() >= 2,
        "Should get at least 1 result from VSIM"
    );

    let mut result_ids: Vec<i64> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    let mut i = 0;
    while i + 1 < response.len() {
        let id = match &response[i] {
            redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<i64>().unwrap(),
            redis::Value::Int(n) => *n,
            _ => panic!("Unexpected ID type at index {}: {:?}", i, response[i]),
        };
        let score = match &response[i + 1] {
            redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<f64>().unwrap(),
            redis::Value::Double(f) => *f,
            _ => panic!(
                "Unexpected score type at index {}: {:?}",
                i + 1,
                response[i + 1]
            ),
        };
        result_ids.push(id);
        scores.push(score);
        i += 2;
    }

    // Query vector should be top-1
    assert_eq!(
        result_ids[0], query_idx as i64,
        "Query vector should be its own nearest neighbor, got {}",
        result_ids[0]
    );

    // Self-distance should be ~0 (score ~1, distance = 1 - score)
    let self_dist = 1.0 - scores[0];
    assert!(
        self_dist.abs() < 0.01,
        "Self-distance should be ~0, got {}",
        self_dist
    );

    // Precision check against brute-force cosine neighbors
    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let found_set: std::collections::HashSet<i64> = result_ids.iter().copied().collect();
    let hits = expected_set.intersection(&found_set).count();
    let precision = hits as f64 / top as f64;

    assert!(
        precision >= 0.8,
        "VectorSets precision should be >= 0.8, got {} (expected {:?}, found {:?})",
        precision,
        expected_set,
        found_set
    );

    let _: () = redis::cmd("DEL")
        .arg("vset_precision")
        .query(&mut conn)
        .unwrap();
}

// ---------------------------------------------------------------------------
// Subprocess tests — run the actual vector-db-benchmark binary
// ---------------------------------------------------------------------------

/// Create a temporary project layout that the binary can discover:
///   tmp/v0/datasets/datasets.json
///   tmp/datasets/<name>/vectors.jsonl + queries.jsonl + neighbours.jsonl
///   tmp/experiments/configurations/<engine>.json
///   tmp/results/
///
/// Returns the temp dir path.
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
    // Leak the TempDir so the directory persists until explicit cleanup
    std::mem::forget(tmp);

    // Create directory structure
    let dataset_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&dataset_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    // Write vectors.jsonl
    let mut vecs_content = String::new();
    for v in vectors {
        let line: Vec<f64> = v.iter().map(|x| *x as f64).collect();
        vecs_content.push_str(&serde_json::to_string(&line).unwrap());
        vecs_content.push('\n');
    }
    fs::write(dataset_dir.join("vectors.jsonl"), &vecs_content).unwrap();

    // Write queries.jsonl
    let mut queries_content = String::new();
    for q in queries {
        let line: Vec<f64> = q.iter().map(|x| *x as f64).collect();
        queries_content.push_str(&serde_json::to_string(&line).unwrap());
        queries_content.push('\n');
    }
    fs::write(dataset_dir.join("queries.jsonl"), &queries_content).unwrap();

    // Write neighbours.jsonl
    let mut neighbors_content = String::new();
    for n in neighbors {
        neighbors_content.push_str(&serde_json::to_string(n).unwrap());
        neighbors_content.push('\n');
    }
    fs::write(dataset_dir.join("neighbours.jsonl"), &neighbors_content).unwrap();

    // Write datasets.json
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

    // Write engine configs
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();

    root
}

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
    // Fallback: look relative to manifest dir
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/release/vector-db-benchmark")
}

/// Parse the search result JSON and return mean_precision_at_returned
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

    // Read the first matching result file
    let content = fs::read_to_string(&found[0]).unwrap();
    let result: serde_json::Value = serde_json::from_str(&content).unwrap();
    result["results"]["mean_precision_at_returned"]
        .as_f64()
        .expect("mean_precision_at_returned not found in result JSON")
}

#[test]
fn test_binary_redis_end_to_end() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 100;
    let top = 5;
    let (_, vectors) = generate_test_vectors(count, dim);

    // Use a subset as queries and compute ground truth
    let queries: Vec<Vec<f32>> = vectors[..10].to_vec();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors(q, &vectors, top))
        .collect();

    let engine_config = serde_json::json!([{
        "name": "test-redis-l2",
        "engine": "redis",
        "algorithm": "hnsw",
        "collection_params": {
            "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 }
        },
        "search_params": [{
            "parallel": 1,
            "search_params": { "ef": 256 },
            "top": top,
        }],
        "upload_params": {
            "data_type": "FLOAT32",
            "parallel": 1,
            "batch_size": 64
        }
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
            "test-redis-l2",
            "--datasets",
            "test-l2",
            "--host",
            "localhost",
        ])
        .env("REDIS_PORT", test_port().to_string())
        .current_dir(&project_root)
        .output()
        .expect("Failed to run vector-db-benchmark");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "Binary failed.\nstdout: {}\nstderr: {}",
        stdout,
        stderr
    );

    // Verify result file exists and precision is reasonable
    let results_dir = project_root.join("results");
    let precision = read_search_precision(&results_dir, "test-redis-l2");
    assert!(
        precision >= 0.8,
        "Binary Redis L2 precision should be >= 0.8, got {}",
        precision
    );

    // Cleanup temp dir
    fs::remove_dir_all(&project_root).ok();
}

/// SVS-VAMANA (Intel Scalable Vector Search) end-to-end via the binary.
/// Guards the two paths that HNSW-only code broke for `algorithm: svs-vamana`:
///   1. `create_index` must emit GRAPH_MAX_DEGREE/CONSTRUCTION_WINDOW_SIZE, NOT
///      M/EF_CONSTRUCTION (which SVS-VAMANA rejects at FT.CREATE), and
///   2. the query must use SEARCH_WINDOW_SIZE $EF (not EF_RUNTIME), with the
///      swept `ef` bound as the $EF PARAM.
///
/// Requires Redis 8.2+ (SVS-VAMANA support); the CI test image is 8.x.
#[test]
fn test_binary_redis_svs_vamana() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 100;
    let top = 5;
    let (_, vectors) = generate_test_vectors(count, dim);
    let queries: Vec<Vec<f32>> = vectors[..10].to_vec();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors(q, &vectors, top))
        .collect();

    let engine_config = serde_json::json!([{
        "name": "test-redis-svs",
        "engine": "redis",
        "algorithm": "svs-vamana",
        // The same M / EF_CONSTRUCTION config drives GRAPH_MAX_DEGREE /
        // CONSTRUCTION_WINDOW_SIZE for SVS.
        "collection_params": {
            "hnsw_config": { "M": 32, "EF_CONSTRUCTION": 200 }
        },
        // Two swept ef values exercise distinct SEARCH_WINDOW_SIZE settings; both
        // are large enough to be exact over 100 docs so the recall floor holds
        // regardless of which result file read_search_precision samples first.
        "search_params": [
            { "parallel": 1, "search_params": { "ef": 200 }, "top": top },
            { "parallel": 1, "search_params": { "ef": 64 }, "top": top }
        ],
        "upload_params": {
            "data_type": "FLOAT32",
            "parallel": 1,
            "batch_size": 64
        }
    }]);

    let project_root = create_test_project(
        "test-svs",
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
            "test-redis-svs",
            "--datasets",
            "test-svs",
            "--host",
            "localhost",
        ])
        .env("REDIS_PORT", test_port().to_string())
        .current_dir(&project_root)
        .output()
        .expect("Failed to run vector-db-benchmark");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "SVS-VAMANA binary run failed.\nstdout: {}\nstderr: {}",
        stdout,
        stderr
    );

    let results_dir = project_root.join("results");
    let precision = read_search_precision(&results_dir, "test-redis-svs");
    assert!(
        precision >= 0.8,
        "SVS-VAMANA precision should be >= 0.8, got {}",
        precision
    );

    fs::remove_dir_all(&project_root).ok();
}

#[test]
fn test_binary_redis_open_loop_fixed_qps() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 100;
    let top = 5;
    let (_, vectors) = generate_test_vectors(count, dim);
    let queries: Vec<Vec<f32>> = vectors[..10].to_vec();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors(q, &vectors, top))
        .collect();
    let engine_config = serde_json::json!([{
        "name": "test-redis-open-loop",
        "engine": "redis",
        "algorithm": "hnsw",
        "collection_params": {
            "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 }
        },
        "search_params": [{
            "parallel": 4,
            "search_params": { "ef": 256 },
            "top": top
        }],
        "upload_params": {
            "data_type": "FLOAT32",
            "parallel": 1,
            "batch_size": 64
        }
    }]);
    let project_root = create_test_project(
        "test-open-loop",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        "l2",
        dim,
    );

    let output = Command::new(binary_path())
        .args([
            "--engines",
            "test-redis-open-loop",
            "--datasets",
            "test-open-loop",
            "--host",
            "localhost",
            "--target-qps",
            "100",
            "--search-duration",
            "0.4",
            "--max-lateness-ms",
            "1000",
            "--repetitions",
            "1",
        ])
        .env("REDIS_PORT", test_port().to_string())
        .current_dir(&project_root)
        .output()
        .expect("Failed to run open-loop benchmark");
    assert!(
        output.status.success(),
        "open-loop binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let results_dir = project_root.join("results");
    assert_eq!(
        read_search_result_field(&results_dir, "test-redis-open-loop", "target_qps"),
        Some(serde_json::json!(100.0))
    );
    assert_eq!(
        read_search_result_field(&results_dir, "test-redis-open-loop", "offered_queries"),
        Some(serde_json::json!(40))
    );
    assert_eq!(
        read_search_result_field(&results_dir, "test-redis-open-loop", "succeeded_queries"),
        Some(serde_json::json!(40))
    );
    assert_eq!(
        read_search_result_field(&results_dir, "test-redis-open-loop", "dropped_queries"),
        Some(serde_json::json!(0))
    );
    assert!(read_search_result_field(
        &results_dir,
        "test-redis-open-loop",
        "schedule_delay_p95_time"
    )
    .and_then(|v| v.as_f64())
    .is_some());
    assert!(
        read_search_result_field(&results_dir, "test-redis-open-loop", "end_to_end_p95_time")
            .and_then(|v| v.as_f64())
            .is_some()
    );

    fs::remove_dir_all(&project_root).ok();
}

#[test]
fn test_binary_vectorsets_end_to_end() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 100;
    let top = 5;
    let (_, vectors) = generate_test_vectors(count, dim);

    let queries: Vec<Vec<f32>> = vectors[..10].to_vec();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors_cosine(q, &vectors, top))
        .collect();

    let engine_config = serde_json::json!([{
        "name": "test-vectorsets",
        "engine": "vectorsets",
        "search_params": [{
            "parallel": 1,
            "search_params": { "ef": 64 },
            "top": top,
        }],
        "upload_params": {
            "hnsw_config": {
                "quant": "NOQUANT",
                "M": 16,
                "EF_CONSTRUCTION": 200
            },
            "CAS": true,
            "parallel": 1,
            "batch_size": 64
        }
    }]);

    let project_root = create_test_project(
        "test-cosine",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        "cosine",
        dim,
    );

    let bin = binary_path();
    assert!(bin.exists(), "Binary not found at {:?}", bin);

    let output = Command::new(&bin)
        .args([
            "--engines",
            "test-vectorsets",
            "--datasets",
            "test-cosine",
            "--host",
            "localhost",
        ])
        .env("REDIS_PORT", test_port().to_string())
        .current_dir(&project_root)
        .output()
        .expect("Failed to run vector-db-benchmark");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "Binary failed.\nstdout: {}\nstderr: {}",
        stdout,
        stderr
    );

    let results_dir = project_root.join("results");
    let precision = read_search_precision(&results_dir, "test-vectorsets");
    assert!(
        precision >= 0.8,
        "Binary VectorSets precision should be >= 0.8, got {}",
        precision
    );

    fs::remove_dir_all(&project_root).ok();
}

/// Parse INFO COMMANDSTATS output and return call count for a given command.
fn commandstats_calls(conn: &mut Connection, cmd_name: &str) -> u64 {
    let info: String = redis::cmd("INFO").arg("COMMANDSTATS").query(conn).unwrap();
    // Lines look like: cmdstat_FT.SEARCH:calls=10,usec=1234,...
    let needle = format!("cmdstat_{}:", cmd_name);
    for line in info.lines() {
        if line.starts_with(&needle) {
            if let Some(rest) = line.strip_prefix(&needle) {
                for part in rest.split(',') {
                    if let Some(val) = part.strip_prefix("calls=") {
                        return val.parse().unwrap_or(0);
                    }
                }
            }
        }
    }
    0
}

/// Read a search result JSON and return a specific field from "results".
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

#[test]
fn test_binary_redis_mixed_benchmark() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 100;
    let top = 5;
    let (_, vectors) = generate_test_vectors(count, dim);

    let queries: Vec<Vec<f32>> = vectors[..10].to_vec();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors(q, &vectors, top))
        .collect();

    let engine_config = serde_json::json!([{
        "name": "test-redis-mixed",
        "engine": "redis",
        "algorithm": "hnsw",
        "collection_params": {
            "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 }
        },
        // parallel: 1 — this test asserts EXACT FT.SEARCH/HSET call counts, which
        // only hold single-threaded (the mixed loop's `break 'outer` skips the
        // update phase for whichever worker draws the out-of-range search index,
        // so the update count is interleaving-dependent at parallel > 1). The
        // multi-worker join-merge is covered by test_binary_redis_mixed_parallel.
        "search_params": [{
            "parallel": 1,
            "search_params": { "ef": 256 },
            "top": top,
        }],
        "upload_params": {
            "data_type": "FLOAT32",
            "parallel": 1,
            "batch_size": 64
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

    // Reset commandstats before running
    let _: String = redis::cmd("CONFIG")
        .arg("RESETSTAT")
        .query(&mut conn)
        .unwrap();

    // Run with --update-search-ratio 1:5 (10 queries → 2 update cycles).
    // --repetitions 1: this test asserts the exact per-run FT.SEARCH/HSET call
    // counts, so it must measure a single pass (the default is 3 warm reps).
    let output = Command::new(&bin)
        .args([
            "--engines",
            "test-redis-mixed",
            "--datasets",
            "test-mixed",
            "--host",
            "localhost",
            "--update-search-ratio",
            "1:5",
            "--repetitions",
            "1",
        ])
        .env("REDIS_PORT", test_port().to_string())
        .current_dir(&project_root)
        .output()
        .expect("Failed to run vector-db-benchmark");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

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

    // Check INFO COMMANDSTATS
    let ft_search_calls = commandstats_calls(&mut conn, "FT.SEARCH");
    let hset_calls = commandstats_calls(&mut conn, "hset");

    // 10 queries → FT.SEARCH should have exactly 10 calls
    assert_eq!(
        ft_search_calls, 10,
        "Expected 10 FT.SEARCH calls, got {}",
        ft_search_calls
    );

    // 100 uploads + 2 mixed updates (10 queries / 5 per cycle = 2 updates)
    assert_eq!(
        hset_calls, 102,
        "Expected 102 HSET calls (100 upload + 2 mixed updates), got {}",
        hset_calls
    );

    // Verify results JSON has update metrics
    let results_dir = project_root.join("results");

    let update_count = read_search_result_field(&results_dir, "test-redis-mixed", "update_count");
    assert_eq!(
        update_count,
        Some(serde_json::json!(2)),
        "Expected update_count=2 in results JSON, got {:?}",
        update_count
    );

    let ratio = read_search_result_field(&results_dir, "test-redis-mixed", "update_search_ratio");
    assert_eq!(
        ratio,
        Some(serde_json::json!("1:5")),
        "Expected update_search_ratio='1:5' in results JSON, got {:?}",
        ratio
    );

    let update_rps = read_search_result_field(&results_dir, "test-redis-mixed", "update_rps");
    assert!(
        update_rps.is_some() && update_rps.unwrap().as_f64().unwrap() > 0.0,
        "Expected update_rps > 0 in results JSON"
    );

    // Precision should still be valid
    let precision = read_search_precision(&results_dir, "test-redis-mixed");
    assert!(
        precision >= 0.8,
        "Mixed benchmark precision should be >= 0.8, got {}",
        precision
    );

    fs::remove_dir_all(&project_root).ok();
}

/// End-to-end FILTER-ONLY harness (`--skip-vector-index`): uploads vectors
/// without indexing and runs pure metadata-filter queries through
/// `search_filter_only`, driven at `parallel: 4` with `--queries 1000` so the
/// thread-local per-worker latency buffers are genuinely merged across threads
/// (the join-merge path — the actual rewrite). Asserts the filter-only sentinel
/// (`mean_precision_at_returned == -1`), that every dispatched query is accounted for
/// (`requested == succeeded`, `failed == 0`) on a healthy run, positive RPS, and
/// monotone linear percentiles (p50 <= p95 <= p99).
#[test]
fn test_binary_redis_filter_only() {
    wait_for_redis();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "redis-fo", "engine": "redis",
        "search_params": [{"parallel": 4, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "redis-fo-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = test_port().to_string();
    assert!(
        common::run_binary_extra(
            &proj.root,
            "redis-fo",
            "redis-fo-test",
            "localhost",
            &[("REDIS_PORT", port.as_str())],
            &["--skip-vector-index", "--queries", "1000"],
        ),
        "redis filter-only run failed"
    );

    // --skip-vector-index renames the engine to "<type>-no-vector".
    let r = common::read_results_obj(&proj.root, "redis-no-vector");
    let mp = r["mean_precision_at_returned"].as_f64().unwrap();
    let rps = r["rps"].as_f64().unwrap();
    let p50 = r["p50_time"].as_f64().unwrap();
    let p95 = r["p95_time"].as_f64().unwrap();
    let p99 = r["p99_time"].as_f64().unwrap();
    let requested = r["requested_queries"].as_u64().unwrap();
    let succeeded = r["succeeded_queries"].as_u64().unwrap();
    let failed = r["failed_queries"].as_u64().unwrap();
    println!(
        "redis filter-only: mean_precision_at_returned={mp} rps={rps:.1} p50={p50} p95={p95} p99={p99} \
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

/// End-to-end MIXED harness at `parallel: 4` over a 2000-query fixture, so many
/// full search phases (and updates) run and the per-worker thread-local sample
/// buffers are merged across threads (the join-merge path — the actual rewrite).
/// Complements `test_binary_redis_mixed_benchmark` (parallel: 1, exact counts):
/// here we assert the search recall/precision are intact, updates ran
/// (`update_count > 0`, `update_rps > 0`), and search percentiles are monotone.
#[test]
fn test_binary_redis_mixed_parallel() {
    wait_for_redis();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "redis-mx", "engine": "redis",
        "search_params": [{"parallel": 4, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project_n(
        "redis-mx-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
        2000,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = test_port().to_string();
    assert!(
        common::run_binary_extra(
            &proj.root,
            "redis-mx",
            "redis-mx-test",
            "localhost",
            &[("REDIS_PORT", port.as_str())],
            &["--update-search-ratio", "1:5", "--repetitions", "1"],
        ),
        "redis mixed (parallel) run failed"
    );

    let r = common::read_results_obj(&proj.root, "redis-mx");
    let recall = r["mean_recall"].as_f64().unwrap();
    let precision = r["mean_precision_at_returned"].as_f64().unwrap();
    let update_count = r["update_count"].as_u64().unwrap();
    let update_rps = r["update_rps"].as_f64().unwrap();
    let p50 = r["p50_time"].as_f64().unwrap();
    let p95 = r["p95_time"].as_f64().unwrap();
    let p99 = r["p99_time"].as_f64().unwrap();
    let succeeded = r["succeeded_queries"].as_u64().unwrap();
    println!(
        "redis mixed (parallel=4): recall={recall:.3} precision={precision:.3} \
         succeeded={succeeded} update_count={update_count} update_rps={update_rps:.1} \
         p50={p50} p95={p95} p99={p99}"
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

/// End-to-end `match_any`: filter a keyword (TAG) field to an OR-set and assert
/// the engine returns the filtered nearest neighbours (recall vs ground truth
/// brute-forced over only the matching docs). Proves the TAG-OR filter arm.
#[test]
fn test_binary_redis_match_any() {
    wait_for_redis();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "redis-ma", "engine": "redis",
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

    assert!(
        common::run_binary(
            &proj.root,
            "redis-ma",
            "match-any-test",
            "localhost",
            &[("REDIS_PORT", &test_port().to_string())],
        ),
        "redis match_any run failed"
    );

    let recall = common::read_recall(&proj.root, "redis-ma");
    println!("redis match_any recall={:.3}", recall);
    assert!(recall >= 0.9, "redis match_any recall {:.3} < 0.9", recall);
}

/// Same filtered (`match_any`) search as above, but forces the **RESP3** protocol
/// via `REDIS_URI=…?protocol=resp3`. RESP3 returns FT.SEARCH results as a map
/// (`{results:[{id, extra_attributes:{vector_score}}]}`) rather than the RESP2
/// flat array, so this exercises the RESP3 branch of the response parser
/// end-to-end. Recall must match the RESP2 path.
#[test]
fn test_binary_redis_match_any_resp3() {
    wait_for_redis();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "redis-ma-r3", "engine": "redis",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "match-any-test-r3",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);

    let resp3_uri = format!("redis://localhost:{}/?protocol=resp3", test_port());
    assert!(
        common::run_binary(
            &proj.root,
            "redis-ma-r3",
            "match-any-test-r3",
            "localhost",
            &[("REDIS_URI", &resp3_uri)],
        ),
        "redis match_any (RESP3) run failed"
    );

    let recall = common::read_recall(&proj.root, "redis-ma-r3");
    println!("redis match_any RESP3 recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "redis RESP3 match_any recall {:.3} < 0.9 — RESP3 FT.SEARCH parsing broken?",
        recall
    );
}

// ── New filter datatypes: bool / uuid / full-text / datetime ────────────────
//
// Each drives the real binary against a compound dataset whose queries carry a
// single filter type, with ground truth brute-forced over only the matching
// docs — a high recall proves the corresponding filter arm (TAG bool, TAG uuid,
// TEXT full-text, NUMERIC datetime range) is applied end-to-end.

/// Shared driver: build the project with `build`, run the binary, assert recall.
fn run_filter_recall_test(
    name: &str,
    dataset: &str,
    build: impl Fn(&str, &str, usize) -> common::FilterProject,
) {
    wait_for_redis();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": name, "engine": "redis",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = build(dataset, &serde_json::to_string(&configs).unwrap(), dim);
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    assert!(
        common::run_binary(
            &proj.root,
            name,
            dataset,
            "localhost",
            &[("REDIS_PORT", &test_port().to_string())],
        ),
        "redis {} run failed",
        name
    );

    let recall = common::read_recall(&proj.root, name);
    println!("redis {} recall={:.3}", name, recall);
    assert!(recall >= 0.9, "redis {} recall {:.3} < 0.9", name, recall);
}

#[test]
fn test_binary_redis_bool() {
    run_filter_recall_test("redis-bool", "bool-test", common::write_bool_project);
}

#[test]
fn test_binary_redis_uuid() {
    run_filter_recall_test("redis-uuid", "uuid-test", common::write_uuid_project);
}

#[test]
fn test_binary_redis_fulltext() {
    run_filter_recall_test("redis-text", "text-test", common::write_fulltext_project);
}

#[test]
fn test_binary_redis_datetime() {
    run_filter_recall_test("redis-dt", "dt-test", common::write_datetime_project);
}

/// Geo-radius filter: points along a meridian, query a radius selecting the
/// nearest ~198. Exercises RediSearch's `@location:[lon lat radius m]` GEO filter
/// against haversine ground truth (new fixture — geo was previously untested).
#[test]
fn test_binary_redis_geo() {
    run_filter_recall_test("redis-geo", "geo-test", common::write_geo_project);
}

/// The bounding-box-DISCRIMINATING geo fixture, run against RediSearch's native
/// `@location:[lon lat r m]` as a CONTROL for issue #223.
///
/// `write_geo_project` above puts every document on one meridian, where a box
/// and a circle select the same set. This one puts 300 of 400 documents in the
/// corners of the bounding box, outside the circle: a correct radius scores
/// ~1.0, a bounding box or no filter scores ~0.25. Running it on an engine whose
/// geo filter was already correct is what proves 1.0 is reachable, so a failure
/// on the engines fixed in #223 is the engine and not the fixture.
#[test]
fn test_binary_redis_geo_corner() {
    run_filter_recall_test(
        "redis-geo-corner",
        "geo-corner-test",
        common::write_geo_corner_project,
    );
}

/// Multi-condition AND: `color == "red" AND size >= 50` in one query — verifies
/// the engine intersects two clauses of different types (every other fixture
/// filters on a single condition).
#[test]
fn test_binary_redis_and_filter() {
    run_filter_recall_test("redis-and", "and-test", common::write_and_filter_project);
}

/// Multi-condition OR: `color == "red" OR size >= 90` in one query — verifies the
/// engine UNIONs two clauses (top-level `{or:[...]}`). An engine that treats it
/// as AND, or drops an arm, searches a strict subset and recall collapses.
#[test]
fn test_binary_redis_or_filter() {
    run_filter_recall_test("redis-or", "or-test", common::write_or_filter_project);
}

/// Regression for #184: a multi-config sweep with `--keep-data
/// --reset-between-configs` must keep only the LAST config's data resident, not
/// accumulate every config's copy (which OOMs a memory-bounded server). Runs a
/// 2-config sweep with both flags and asserts exactly ONE FT index survives (the
/// last config's) and the keyspace stays at one config's size. (Without
/// --reset-between-configs, --keep-data keeps both configs coexisting — see
/// test_binary_redis_coexistence_skip_upload.)
#[test]
fn test_binary_redis_keep_data_sweep_frees_prior_configs() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    // Two configs over the SAME dataset — a minimal M×EF sweep. Distinct names →
    // distinct per-config index/keyspace (idx:redis-sweepa / idx:redis-sweepb).
    let configs = serde_json::json!([
        {
            "name": "redis-sweepa", "engine": "redis",
            "search_params": [{"parallel": 1, "search_params": {"ef": 128}}],
            "upload_params": {"parallel": 1, "batch_size": 200}
        },
        {
            "name": "redis-sweepb", "engine": "redis",
            "search_params": [{"parallel": 1, "search_params": {"ef": 128}}],
            "upload_params": {"parallel": 1, "batch_size": 200}
        }
    ]);
    let proj =
        common::write_bool_project("sweep-test", &serde_json::to_string(&configs).unwrap(), dim);

    // `redis-sweep*` matches both configs; `--keep-data` would (pre-fix) leave
    // both configs' data behind.
    assert!(
        common::run_binary_extra(
            &proj.root,
            "redis-sweep*",
            "sweep-test",
            "localhost",
            &[("REDIS_PORT", &test_port().to_string())],
            &["--keep-data", "--reset-between-configs"],
        ),
        "redis keep-data sweep run failed"
    );

    // Exactly one FT index must remain (the last config's); the first config's
    // was torn down despite --keep-data. Pre-fix this was 2.
    let indexes: Vec<String> = redis::cmd("FT._LIST").query(&mut conn).unwrap_or_default();
    let sweep_indexes: Vec<&String> = indexes
        .iter()
        .filter(|i| i.contains("sweepa") || i.contains("sweepb"))
        .collect();
    println!("surviving sweep indexes: {:?}", sweep_indexes);
    assert_eq!(
        sweep_indexes.len(),
        1,
        "expected exactly 1 surviving sweep index (only the last config kept), got {:?}",
        sweep_indexes
    );

    // And the keyspace holds one config's docs, not two (the memory symptom).
    let dbsize: i64 = redis::cmd("DBSIZE").query(&mut conn).unwrap_or(0);
    println!("dbsize after keep-data sweep: {}", dbsize);
    assert!(
        dbsize <= 600,
        "keyspace should hold ~one config's 400 docs, not two configs' 800 (got {dbsize})"
    );

    flush_db(&mut conn);
}

/// #188 upload-once/build-many: with `REDIS_KEY_PREFIX` set, all sweep configs
/// share ONE corpus keyspace, so a config after the first must SKIP the (identical)
/// re-upload and just build its index over the shared corpus. Runs a 2-config
/// sweep with the shared prefix and asserts (a) the second config logs the
/// "skipping re-upload" path, (b) BOTH configs' indexes return correct results
/// over the shared corpus (recall >= 0.9), and (c) the corpus exists once under
/// the shared prefix. Default (no REDIS_KEY_PREFIX) keeps per-config isolation —
/// covered by test_binary_redis_coexistence_skip_upload.
#[test]
fn test_binary_redis_shared_corpus_upload_once() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let configs = serde_json::json!([
        {
            "name": "redis-sc-a", "engine": "redis",
            "search_params": [{"parallel": 1, "search_params": {"ef": 128}}],
            "upload_params": {"parallel": 1, "batch_size": 100}
        },
        {
            "name": "redis-sc-b", "engine": "redis",
            "search_params": [{"parallel": 1, "search_params": {"ef": 128}}],
            "upload_params": {"parallel": 1, "batch_size": 100}
        }
    ]);
    let proj =
        common::write_bool_project("sc-test", &serde_json::to_string(&configs).unwrap(), dim);

    let out = Command::new(binary_path())
        .args([
            "--engines",
            "redis-sc*",
            "--datasets",
            "sc-test",
            "--host",
            "localhost",
            "--keep-data",
            "--skip-if-exists",
            "false",
        ])
        .env("REDIS_PORT", test_port().to_string())
        .env("REDIS_KEY_PREFIX", "shared_sc:")
        .current_dir(&proj.root)
        .output()
        .expect("run shared-corpus sweep");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "shared-corpus sweep failed.\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&out.stderr)
    );

    // The second config must have SKIPPED the re-upload (the #188 win).
    assert!(
        stdout.contains("skipping re-upload"),
        "expected an upload-once skip for the 2nd config; stdout:\n{}",
        stdout
    );

    // ...but it must still have WAITED for its own index to backfill. Its index
    // is created over an already-populated keyspace, so RediSearch fills it
    // asynchronously: FT.INFO reports ~1% indexed for a beat after FT.CREATE.
    // A skipping config that returned without waiting searched a half-built
    // index and reported low recall with no error — visible only as a flaky
    // `rb` below, which is a silent wrong result on a real sweep. Asserting the
    // wait ran is deterministic, unlike the recall threshold it protects.
    assert!(
        stdout.contains("Index time (backfill over the shared corpus)"),
        "the config that skipped the re-upload must still wait for its index to \
         backfill before searching; stdout:\n{}",
        stdout
    );

    // Both configs' indexes work over the shared corpus.
    let ra = common::read_recall(&proj.root, "redis-sc-a");
    let rb = common::read_recall(&proj.root, "redis-sc-b");
    println!("shared-corpus recall a={:.3} b={:.3}", ra, rb);
    assert!(ra >= 0.9, "redis-sc-a recall {:.3} < 0.9", ra);
    assert!(rb >= 0.9, "redis-sc-b recall {:.3} < 0.9", rb);

    // The corpus exists once under the shared prefix (400 docs), not per-config.
    let keys: i64 = redis::cmd("EVAL")
        .arg("return #redis.call('keys', KEYS[1])")
        .arg(1)
        .arg("shared_sc:*")
        .query(&mut conn)
        .unwrap();
    println!("shared corpus keys={}", keys);
    assert!(
        keys >= 400,
        "shared corpus should hold the 400-doc corpus (got {keys})"
    );

    flush_db(&mut conn);
}

/// Nested boolean: `(color==red AND size>=50) OR (color==blue AND size<10)` —
/// verifies the parser recurses into grouped sub-conditions (parenthesised
/// RediSearch clause) instead of flattening the tree. A flattening parser matches
/// a wildly different doc set, so recall collapses.
#[test]
fn test_binary_redis_nested_filter() {
    run_filter_recall_test(
        "redis-nested",
        "nested-test",
        common::write_nested_filter_project,
    );
}

/// Selectivity ladder: one `rank < K` range query per rung, sweeping filter
/// selectivity from ~3% to ~99% in a single dataset. Verifies the numeric-range
/// filter path stays correct (recall vs per-rung ground truth) across the whole
/// selectivity range, not just at one operating point.
#[test]
fn test_binary_redis_selectivity() {
    run_filter_recall_test("redis-sel", "sel-test", common::write_selectivity_project);
}

/// Multi-tenancy: many tenants share ONE index and every query is scoped to a
/// single tenant via a keyword-equality filter on a `tenant` field, with ground
/// truth brute-forced over ONLY that tenant's docs. Reuses the keyword-TAG
/// filter arm — no new engine code.
///
/// STRONGER than the other filter recall tests: the ground truth is tenant-local,
/// so a cross-tenant document that leaked into a result cannot count toward recall
/// AND displaces a correct neighbour. Search is exact (ef=400 over ~16 docs/tenant)
/// and there is one query per tenant, so a correct engine scores EXACTLY 1.0 on
/// every query. We therefore assert (a) mean recall == 1.0 and (b) EVERY per-query
/// recall == 1.0 — any single leaked or mis-scoped tenant fails. (The saved result
/// JSON records per-query recalls but not the raw returned ids, so the exact
/// per-query recall is the strongest available tenant-isolation check without
/// changing engine result serialization.)
#[test]
fn test_binary_redis_tenancy() {
    wait_for_redis();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "redis-tenancy", "engine": "redis",
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

    assert!(
        // --dump-raw-latencies: the per-tenant leakage check below reads the
        // full per-query `recalls` array, which is only emitted under this flag
        // (results otherwise carry compact HDR/quality digests).
        common::run_binary_extra(
            &proj.root,
            "redis-tenancy",
            "tenancy-test",
            "localhost",
            &[("REDIS_PORT", &test_port().to_string())],
            &["--dump-raw-latencies"],
        ),
        "redis tenancy run failed"
    );

    let recall = common::read_recall(&proj.root, "redis-tenancy");
    let per_query = common::read_recalls(&proj.root, "redis-tenancy");
    println!(
        "redis tenancy mean recall={:.3} per-query={:?}",
        recall, per_query
    );
    // Search here is exact (ef=400 over ~16 docs/tenant), so a correct engine
    // scores 1.0. Assert EXACT per-query recall: the ground truth is tenant-local,
    // so a single leaked cross-tenant doc displaces a correct neighbour and drops
    // recall below 1.0 — the strongest isolation check without id-level result
    // serialization.
    assert!(
        recall > 0.999,
        "redis tenancy mean recall {:.4} != 1.0 — cross-tenant leakage or mis-scope?",
        recall
    );
    for (q, r) in per_query.iter().enumerate() {
        assert!(
            *r > 0.999,
            "redis tenancy query {} recall {:.4} != 1.0 — cross-tenant leakage?",
            q,
            r
        );
    }
}

/// Read `results.mean_recall` for `engine_name` from its search result JSON.
fn read_search_mean_recall(results_dir: &PathBuf, engine_name: &str) -> f64 {
    read_search_result_field(results_dir, engine_name, "mean_recall")
        .and_then(|v| v.as_f64())
        .unwrap_or_else(|| panic!("no mean_recall for {}", engine_name))
}

/// FT.INFO num_docs for `index`, or -1 if the index is missing.
fn ft_info_num_docs(conn: &mut Connection, index: &str) -> i64 {
    match redis::cmd("FT.INFO").arg(index).query::<redis::Value>(conn) {
        Ok(info) => extract_ft_info_value(&info, "num_docs").unwrap_or(-1),
        Err(_) => -1,
    }
}

/// Delete only the `*-search-*.json` files under `results_dir` (keeps upload files).
fn delete_search_result_files(results_dir: &PathBuf) {
    for entry in fs::read_dir(results_dir).unwrap() {
        let p = entry.unwrap().path();
        if p.file_name()
            .unwrap()
            .to_string_lossy()
            .contains("-search-")
        {
            fs::remove_file(p).ok();
        }
    }
}

/// #151-4 regression: "upload all, then --skip-upload search each" must give every
/// config its OWN graph. Two configs (dense high-ef vs sparse low-ef) coexist on
/// one server; pre-fix they shared index `idx` + keyspace, so the skip-upload
/// sweep read the last-uploaded graph for BOTH → identical recall. Post-fix they
/// address disjoint `idx:<config>` indexes + `<config>:` keyspaces → distinct.
#[test]
fn test_binary_redis_coexistence_skip_upload() {
    wait_for_redis();
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
            "name": "redis-co-a",
            "engine": "redis",
            "algorithm": "hnsw",
            "collection_params": { "hnsw_config": { "M": 64, "EF_CONSTRUCTION": 200 } },
            "search_params": [{ "parallel": 1, "search_params": { "ef": 256 }, "top": top }],
            "upload_params": { "data_type": "FLOAT32", "parallel": 1, "batch_size": 64 }
        },
        {
            "name": "redis-co-b",
            "engine": "redis",
            "algorithm": "hnsw",
            "collection_params": { "hnsw_config": { "M": 4, "EF_CONSTRUCTION": 8 } },
            "search_params": [{ "parallel": 1, "search_params": { "ef": 10 }, "top": top }],
            "upload_params": { "data_type": "FLOAT32", "parallel": 1, "batch_size": 64 }
        }
    ]);

    let root = create_test_project(
        "test-co",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        "l2",
        dim,
    );
    let bin = binary_path();
    let port = test_port().to_string();
    let results_dir = root.join("results");

    // Phase 1: upload + search BOTH configs, KEEPING data for the skip-upload phase.
    let out1 = Command::new(&bin)
        .args([
            "--engines",
            "redis-co-*",
            "--datasets",
            "test-co",
            "--host",
            "localhost",
            "--keep-data",
            "--skip-if-exists",
            "false",
        ])
        .env("REDIS_PORT", &port)
        .current_dir(&root)
        .output()
        .expect("run phase 1");
    assert!(
        out1.status.success(),
        "phase 1 failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out1.stdout),
        String::from_utf8_lossy(&out1.stderr)
    );

    let base_a = read_search_mean_recall(&results_dir, "redis-co-a");
    let base_b = read_search_mean_recall(&results_dir, "redis-co-b");

    // Deterministic coexistence proof: two DISJOINT indexes each hold all `count`
    // docs, and the keyspace carries `count` keys under EACH per-config prefix.
    assert_eq!(
        ft_info_num_docs(&mut conn, "idx:redis-co-a"),
        count as i64,
        "idx:redis-co-a must hold all docs"
    );
    assert_eq!(
        ft_info_num_docs(&mut conn, "idx:redis-co-b"),
        count as i64,
        "idx:redis-co-b must hold all docs"
    );
    let keys_a: i64 = redis::cmd("EVAL")
        .arg("return #redis.call('keys', KEYS[1])")
        .arg(1)
        .arg("redis-co-a:*")
        .query(&mut conn)
        .unwrap();
    let keys_b: i64 = redis::cmd("EVAL")
        .arg("return #redis.call('keys', KEYS[1])")
        .arg(1)
        .arg("redis-co-b:*")
        .query(&mut conn)
        .unwrap();
    assert_eq!(keys_a, count as i64, "redis-co-a: keyspace");
    assert_eq!(keys_b, count as i64, "redis-co-b: keyspace");

    // Delete ONLY the search result files; keep upload files + the server data.
    delete_search_result_files(&results_dir);

    // Phase 2: --skip-upload search of both configs against the coexisting indexes.
    let out2 = Command::new(&bin)
        .args([
            "--engines",
            "redis-co-*",
            "--datasets",
            "test-co",
            "--host",
            "localhost",
            "--skip-upload",
            "--keep-data",
            "--skip-if-exists",
            "false",
        ])
        .env("REDIS_PORT", &port)
        .current_dir(&root)
        .output()
        .expect("run phase 2");
    assert!(
        out2.status.success(),
        "phase 2 (--skip-upload) failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out2.stdout),
        String::from_utf8_lossy(&out2.stderr)
    );

    let rec_a = read_search_mean_recall(&results_dir, "redis-co-a");
    let rec_b = read_search_mean_recall(&results_dir, "redis-co-b");

    // Each config read its OWN graph → recall equals its single-invocation baseline.
    assert!(
        (rec_a - base_a).abs() < 1e-9,
        "redis-co-a skip-upload recall {} != baseline {}",
        rec_a,
        base_a
    );
    assert!(
        (rec_b - base_b).abs() < 1e-9,
        "redis-co-b skip-upload recall {} != baseline {}",
        rec_b,
        base_b
    );
    // The two graphs are distinct (pre-fix these would be identical — last writer
    // wins). The dense high-ef graph out-recalls the sparse low-ef one.
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

/// #151-4 negative: `--skip-upload` with NO prior upload must FAIL LOUDLY (the
/// per-search index-existence guard), never silently writing a recall-0.0 file.
#[test]
fn test_binary_redis_skip_upload_without_prior_upload_errors() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 100;
    let top = 5;
    let (_, vectors) = generate_test_vectors(count, dim);
    let queries: Vec<Vec<f32>> = vectors[..10].to_vec();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors(q, &vectors, top))
        .collect();

    let engine_config = serde_json::json!([{
        "name": "redis-noupload",
        "engine": "redis",
        "algorithm": "hnsw",
        "collection_params": { "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 } },
        "search_params": [{ "parallel": 1, "search_params": { "ef": 64 }, "top": top }],
        "upload_params": { "data_type": "FLOAT32", "parallel": 1, "batch_size": 64 }
    }]);

    let root = create_test_project(
        "test-noupload",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        "l2",
        dim,
    );
    let bin = binary_path();

    // --skip-upload against a server with no matching index. exit_on_error defaults
    // true, so the guard's hard error must fail the process.
    let out = Command::new(&bin)
        .args([
            "--engines",
            "redis-noupload",
            "--datasets",
            "test-noupload",
            "--host",
            "localhost",
            "--skip-upload",
            "--skip-if-exists",
            "false",
        ])
        .env("REDIS_PORT", test_port().to_string())
        .current_dir(&root)
        .output()
        .expect("run skip-upload-no-index");

    assert!(
        !out.status.success(),
        "--skip-upload with no prior upload must fail loudly, but exited 0"
    );
    // And it must NOT have written a recall-0.0 search result file.
    let wrote_search = fs::read_dir(root.join("results"))
        .map(|rd| {
            rd.filter_map(|e| e.ok()).any(|e| {
                e.file_name().to_string_lossy().contains("redis-noupload-")
                    && e.file_name().to_string_lossy().contains("-search-")
            })
        })
        .unwrap_or(false);
    assert!(
        !wrote_search,
        "guard must prevent any search result file from being written"
    );

    fs::remove_dir_all(&root).ok();
}

// ---------------------------------------------------------------------------
// Issue #238 — `--skip-upload` must NOT destroy the corpus it was told to reuse
// ---------------------------------------------------------------------------

/// Count keys under a literal prefix, server-side.
fn count_prefix(conn: &mut Connection, prefix: &str) -> i64 {
    redis::cmd("EVAL")
        .arg("return #redis.call('keys', KEYS[1])")
        .arg(1)
        .arg(format!("{prefix}*"))
        .query(conn)
        .unwrap()
}

/// Regression test for #238: `--skip-upload --skip-vector-index` destroyed the
/// corpus and then measured the empty index, reporting a QPS number and exiting
/// 0.
///
/// The assertions read state back **off the live server** (`FT.INFO num_docs`
/// plus a keyspace count), because the run's own output cannot detect this: on
/// the unfixed binary phase 2 prints a perfectly ordinary QPS line.
///
/// RED on unfixed code: `configure()` ran on the skip-upload path and issued
/// `FT.DROPINDEX idx:redis-no-vector DD`, so both counts come back 0 instead of
/// 400 (verified live on redis 8.2: DBSIZE 400 -> 0).
#[test]
fn test_binary_redis_skip_upload_skip_vector_index_preserves_corpus() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "redis-238-keep", "engine": "redis",
        "search_params": [{"parallel": 1, "search_params": {"ef": 64}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "redis-238-keep-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    let port = test_port().to_string();
    let envs: [(&str, &str); 1] = [("REDIS_PORT", port.as_str())];

    // Phase 1: build the filter-only corpus the second phase is told to reuse.
    // `--skip-vector-index` rewrites the config name to `<engine>-no-vector`, so
    // the index/keyspace phase 2 addresses is the one phase 1 wrote.
    assert!(
        common::run_binary_extra(
            &proj.root,
            "redis-238-keep",
            "redis-238-keep-test",
            "localhost",
            &envs,
            &["--skip-vector-index", "--keep-data", "--skip-search"],
        ),
        "phase 1 (upload with --skip-vector-index) failed"
    );

    let docs_before = ft_info_num_docs(&mut conn, "idx:redis-no-vector");
    let keys_before = count_prefix(&mut conn, "redis-no-vector:");
    assert_eq!(
        docs_before,
        common::N_DOCS as i64,
        "phase 1 must leave a complete corpus indexed"
    );
    assert_eq!(keys_before, common::N_DOCS as i64, "phase 1 keyspace");

    // Phase 2: the flag combination that means "do not upload, do not build an
    // index, just use what is there".
    assert!(
        common::run_binary_extra(
            &proj.root,
            "redis-238-keep",
            "redis-238-keep-test",
            "localhost",
            &envs,
            &["--skip-upload", "--skip-vector-index", "--keep-data"],
        ),
        "phase 2 (--skip-upload --skip-vector-index) failed"
    );

    let docs_after = ft_info_num_docs(&mut conn, "idx:redis-no-vector");
    let keys_after = count_prefix(&mut conn, "redis-no-vector:");
    assert_eq!(
        docs_after, docs_before,
        "--skip-upload --skip-vector-index destroyed the indexed corpus \
         ({docs_before} -> {docs_after} docs) — issue #238"
    );
    assert_eq!(
        keys_after, keys_before,
        "--skip-upload --skip-vector-index destroyed the keyspace \
         ({keys_before} -> {keys_after} keys) — issue #238"
    );

    flush_db(&mut conn);
    fs::remove_dir_all(&proj.root).ok();
}

/// Regression test for #238's second half: a `--skip-upload` run against a
/// SHORT corpus must not quietly measure it.
///
/// Deleting half the docs leaves an index that answers every query without
/// error, so no behavioural assertion catches it — and recall is not a reliable
/// detector either. Whether a truncated corpus shows up in recall depends on
/// WHICH rows went: on `random-100` (9 queries, one ground-truth neighbour each,
/// ids 0-8) deleting the upper half of a Qdrant collection leaves
/// `mean_recall: 1.0`, while deleting the lower half gives `0.0`. A metric that
/// is a coin flip on the deletion pattern cannot be the guard; a server-side row
/// count is not.
///
/// RED on unfixed code: phase 2 exits 0 and writes a search result file.
#[test]
fn test_binary_redis_skip_upload_short_corpus_is_fatal() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "cfg238short", "engine": "redis",
        "search_params": [{"parallel": 1, "search_params": {"ef": 64}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "cfg238short-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    let port = test_port().to_string();
    let envs: [(&str, &str); 1] = [("REDIS_PORT", port.as_str())];
    let results_dir = proj.root.join("results");

    assert!(
        common::run_binary_extra(
            &proj.root,
            "cfg238short",
            "cfg238short-test",
            "localhost",
            &envs,
            &["--keep-data", "--skip-search"],
        ),
        "phase 1 (upload) failed"
    );
    assert_eq!(
        ft_info_num_docs(&mut conn, "idx:cfg238short"),
        common::N_DOCS as i64,
        "phase 1 must leave a complete corpus"
    );

    // Amputate half the corpus behind the tool's back.
    let half = common::N_DOCS / 2;
    for id in 0..half {
        let _: () = redis::cmd("UNLINK")
            .arg(format!("cfg238short:{id}"))
            .query(&mut conn)
            .unwrap();
    }
    let remaining = ft_info_num_docs(&mut conn, "idx:cfg238short");
    assert_eq!(remaining, half as i64, "half the corpus should remain");

    delete_search_result_files(&results_dir);

    let bin = binary_path();
    let run = |extra: &[&str]| {
        let mut cmd = Command::new(&bin);
        cmd.args([
            "--engines",
            "cfg238short",
            "--datasets",
            "cfg238short-test",
            "--host",
            "localhost",
            "--skip-if-exists",
            "false",
            "--skip-upload",
            "--keep-data",
        ]);
        cmd.args(extra);
        cmd.env("REDIS_PORT", &port)
            .current_dir(&proj.root)
            .output()
            .expect("run vector-db-benchmark")
    };

    let out = run(&[]);
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "--skip-upload against a half-deleted corpus must be a hard error \
         (issue #238), but the run succeeded.\n{combined}"
    );
    assert!(
        combined.contains("the corpus you asked to reuse is incomplete")
            && combined.contains(&format!("holds {half} of the {} rows", common::N_DOCS)),
        "the error must name the shortfall it found.\n{combined}"
    );
    let wrote_results = fs::read_dir(&results_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().contains("-search-"));
    assert!(
        !wrote_results,
        "a rejected --skip-upload run must not leave a search result file behind"
    );

    // The guard must be deliberately overridable — and must never have been a
    // reason to touch the data.
    let out2 = run(&["--allow-partial-corpus"]);
    assert!(
        out2.status.success(),
        "--allow-partial-corpus must downgrade the guard to a warning.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out2.stdout),
        String::from_utf8_lossy(&out2.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out2.stderr).contains("WARNING: --skip-upload"),
        "the override must still warn"
    );
    assert_eq!(
        ft_info_num_docs(&mut conn, "idx:cfg238short"),
        half as i64,
        "the reuse check must never modify the corpus"
    );

    // The waived run must SAY it was waived. A verified run, an unverified run
    // and a waived run over a half-deleted corpus produced structurally
    // identical artifacts before #238, so the verdict travels in the file.
    let reuse = common::read_params_obj(&proj.root, "cfg238short")["corpus_reuse"].clone();
    assert_eq!(reuse["status"], "short", "recorded verdict: {reuse}");
    assert_eq!(reuse["expected_rows"], common::N_DOCS as u64);
    assert_eq!(reuse["actual_rows"], half as u64);
    assert_eq!(reuse["waived_by_allow_partial_corpus"], true);
    assert_eq!(reuse["actual_is_estimate"], false);

    flush_db(&mut conn);
    fs::remove_dir_all(&proj.root).ok();
}

/// The SURPLUS arm of the guard, exercised end to end (issue #238).
///
/// The classifier unit test only proves the variant is returned; this proves the
/// handler *warns and continues* rather than aborting — the arm that decides
/// warn-vs-abort is the one that matters, and it had no runtime coverage.
#[test]
fn test_binary_redis_skip_upload_surplus_warns_and_continues() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "cfg238surplus", "engine": "redis",
        "search_params": [{"parallel": 1, "search_params": {"ef": 64}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "cfg238surplus-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    let port = test_port().to_string();
    let envs: [(&str, &str); 1] = [("REDIS_PORT", port.as_str())];

    assert!(
        common::run_binary_extra(
            &proj.root,
            "cfg238surplus",
            "cfg238surplus-test",
            "localhost",
            &envs,
            &["--keep-data", "--skip-search"],
        ),
        "phase 1 (upload) failed"
    );

    // Five extra indexed docs under this config's prefix: a real superset corpus.
    for id in 10_000..10_005 {
        let _: () = redis::cmd("HSET")
            .arg(format!("cfg238surplus:{id}"))
            .arg("vector")
            .arg(vec_to_bytes(&vec![0.0f32; dim]))
            .query(&mut conn)
            .unwrap();
    }
    let expected_surplus = common::N_DOCS as i64 + 5;
    for _ in 0..50 {
        if ft_info_num_docs(&mut conn, "idx:cfg238surplus") == expected_surplus {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(
        ft_info_num_docs(&mut conn, "idx:cfg238surplus"),
        expected_surplus,
        "the extra docs must be indexed for this to be a surplus"
    );

    delete_search_result_files(&proj.root.join("results"));

    let out = Command::new(binary_path())
        .args([
            "--engines",
            "cfg238surplus",
            "--datasets",
            "cfg238surplus-test",
            "--host",
            "localhost",
            "--skip-if-exists",
            "false",
            "--skip-upload",
            "--keep-data",
        ])
        .env("REDIS_PORT", &port)
        .current_dir(&proj.root)
        .output()
        .expect("run vector-db-benchmark");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        out.status.success(),
        "a surplus corpus must WARN and continue, not abort.\nstdout: {}\nstderr: {stderr}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        stderr.contains("WARNING: --skip-upload")
            && stderr.contains(&format!("holds {expected_surplus} rows")),
        "the surplus warning must name the count it found.\nstderr: {stderr}"
    );
    let reuse = common::read_params_obj(&proj.root, "cfg238surplus")["corpus_reuse"].clone();
    assert_eq!(reuse["status"], "surplus", "recorded verdict: {reuse}");
    assert_eq!(reuse["actual_rows"], expected_surplus as u64);

    flush_db(&mut conn);
    fs::remove_dir_all(&proj.root).ok();
}

/// `--skip-upload` with NO `--keep-data` must not delete the corpus it reused
/// (issue #238, second destructive path).
///
/// `--keep-data` defaults to false and `engine.delete()` runs at the end of every
/// successful experiment, so the plainest form of the flag used to benchmark a
/// corpus and then destroy it — exit 0, no warning. The other #238 regression
/// tests all pass `--keep-data`, so none of them covered this.
///
/// RED before the fix: `DBSIZE` 400 -> 0.
#[test]
fn test_binary_redis_skip_upload_without_keep_data_preserves_corpus() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "cfg238nokeep", "engine": "redis",
        "search_params": [{"parallel": 1, "search_params": {"ef": 64}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "cfg238nokeep-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    let port = test_port().to_string();
    let envs: [(&str, &str); 1] = [("REDIS_PORT", port.as_str())];

    assert!(
        common::run_binary_extra(
            &proj.root,
            "cfg238nokeep",
            "cfg238nokeep-test",
            "localhost",
            &envs,
            &["--keep-data", "--skip-search"],
        ),
        "phase 1 (upload) failed"
    );
    let before = ft_info_num_docs(&mut conn, "idx:cfg238nokeep");
    assert_eq!(before, common::N_DOCS as i64, "phase 1 corpus");

    // Deliberately NO --keep-data.
    assert!(
        common::run_binary_extra(
            &proj.root,
            "cfg238nokeep",
            "cfg238nokeep-test",
            "localhost",
            &envs,
            &["--skip-upload"],
        ),
        "phase 2 (--skip-upload, no --keep-data) failed"
    );

    assert_eq!(
        ft_info_num_docs(&mut conn, "idx:cfg238nokeep"),
        before,
        "--skip-upload without --keep-data deleted the corpus it reused — issue #238"
    );
    assert_eq!(
        count_prefix(&mut conn, "cfg238nokeep:"),
        common::N_DOCS as i64,
        "--skip-upload without --keep-data unlinked the keyspace it reused — issue #238"
    );

    flush_db(&mut conn);
    fs::remove_dir_all(&proj.root).ok();
}

// ---------------------------------------------------------------------------
// Issue #290 — `--skip-upload` must not run against a corpus it cannot VERIFY
// ---------------------------------------------------------------------------

/// A `--skip-upload` run whose dataset offers no expected row count must abort,
/// not proceed with a printed note (issue #290).
///
/// Before this fix the guard compared an `Option<u64>` expected count against
/// the server-side count, and `None` short-circuited to a printed "Reuse check —
/// SKIPPED" and `Ok`. `None` is reachable without any user error: the count is
/// measured from the corpus FILE, and `--skip-upload` exists precisely for the
/// case where the corpus lives on the server and not on this machine.
///
/// This test reaches that state the way the tool's own doc comment describes it
/// — "one whose big `vectors.npy` was deleted to reclaim disk while
/// `tests.jsonl` stayed behind so queries still work" — by deleting
/// `vectors.npy` from the fixture. Queries and ground truth are in
/// `tests.jsonl`, so the run still searches and still publishes.
///
/// What the assertions rest on: the server-side row count read back with
/// `FT.INFO` (`ft_info_num_docs`), before and after each run. Recall is asserted
/// once, at the very end, only to show what the waiver publishes — it is NOT the
/// detector, and could not be: on a truncated corpus recall is a coin flip on
/// which rows went.
///
/// Structure: a POSITIVE CONTROL first (everything present → the run must
/// succeed and record `verified`), then the guard with the server corpus INTACT
/// (proving it fires on unverifiability itself, not on missing data), then with
/// the server corpus EMPTY (the published-recall-0.0 case from the issue), then
/// the `--allow-partial-corpus` waiver, then the second `None` path: a corpus
/// file whose header cannot be READ. That last phase is the one that covers the
/// swallowed-`Err` half of #290 — `reuse_expected_rows` already returned `Err`
/// for it on master; it was the caller that turned that into "cannot verify".
///
/// RED on unfixed master `cff7e3a`: every guarded run here exits 0 and writes a
/// search result file; the empty-corpus one publishes `mean_recall` 0.0 with
/// `corpus_reuse.status` `"unverified"` and `failed_queries` 0.
#[test]
fn test_binary_redis_skip_upload_unverifiable_corpus_is_fatal() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "cfg290unver", "engine": "redis",
        "search_params": [{"parallel": 1, "search_params": {"ef": 64}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "cfg290unver-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    let port = test_port().to_string();
    let envs: [(&str, &str); 1] = [("REDIS_PORT", port.as_str())];
    let results_dir = proj.root.join("results");
    let vectors_npy = proj
        .root
        .join("datasets")
        .join("cfg290unver-test")
        .join("vectors.npy");

    // Phase 1: build the corpus the later runs are told to reuse.
    assert!(
        common::run_binary_extra(
            &proj.root,
            "cfg290unver",
            "cfg290unver-test",
            "localhost",
            &envs,
            &["--keep-data", "--skip-search"],
        ),
        "phase 1 (upload) failed"
    );
    assert_eq!(
        ft_info_num_docs(&mut conn, "idx:cfg290unver"),
        common::N_DOCS as i64,
        "phase 1 must leave a complete corpus"
    );
    delete_search_result_files(&results_dir);

    let bin = binary_path();
    let run = |extra: &[&str]| {
        let mut cmd = Command::new(&bin);
        cmd.args([
            "--engines",
            "cfg290unver",
            "--datasets",
            "cfg290unver-test",
            "--host",
            "localhost",
            "--skip-if-exists",
            "false",
            "--skip-upload",
            "--keep-data",
        ]);
        cmd.args(extra);
        cmd.env("REDIS_PORT", &port)
            .current_dir(&proj.root)
            .output()
            .expect("run vector-db-benchmark")
    };
    let combined = |out: &std::process::Output| {
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    };
    let wrote_search_result = || {
        fs::read_dir(&results_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().contains("-search-"))
    };

    // POSITIVE CONTROL. Corpus file present, server corpus complete: the guard
    // must let this through, or the test could pass by rejecting everything.
    let control = run(&[]);
    assert!(
        control.status.success(),
        "--skip-upload over a corpus that IS verifiable and complete must succeed.\n{}",
        combined(&control)
    );
    let reuse = common::read_params_obj(&proj.root, "cfg290unver")["corpus_reuse"].clone();
    assert_eq!(reuse["status"], "verified", "control verdict: {reuse}");
    assert_eq!(reuse["expected_rows"], common::N_DOCS as u64);
    assert_eq!(reuse["actual_rows"], common::N_DOCS as u64);
    delete_search_result_files(&results_dir);

    // Take the corpus FILE away. tests.jsonl (queries + ground truth) and
    // payloads.jsonl stay, so nothing else about the run changes: it can still
    // search, and the dataset directory still exists so no download is
    // attempted. Only the expected row count is now unobtainable.
    fs::remove_file(&vectors_npy).expect("remove vectors.npy");

    // (a) Server corpus INTACT. The number this run would publish is in fact
    // correct — and the tool has no way to know that. Asserting the FT.INFO
    // count here is what proves the abort is about unverifiability and not
    // about missing data.
    assert_eq!(
        ft_info_num_docs(&mut conn, "idx:cfg290unver"),
        common::N_DOCS as i64,
        "the server-side corpus must still be complete at this point"
    );
    let intact = run(&[]);
    let intact_out = combined(&intact);
    assert!(
        !intact.status.success(),
        "--skip-upload must abort when the dataset's expected row count cannot be \
         determined (issue #290), but the run succeeded.\n{intact_out}"
    );
    assert!(
        intact_out.contains("has nothing to compare against")
            && intact_out.contains("config 'cfg290unver'")
            && intact_out.contains("could not be measured on disk"),
        "the error must name the missing side of the comparison.\n{intact_out}"
    );
    assert!(
        !wrote_search_result(),
        "a rejected --skip-upload run must not leave a search result file behind"
    );
    assert_eq!(
        ft_info_num_docs(&mut conn, "idx:cfg290unver"),
        common::N_DOCS as i64,
        "the reuse check must never modify the corpus"
    );

    // (b) Server corpus EMPTY — the case from the issue. Index kept so the probe
    // still succeeds and reports an exact 0; dropping it would instead exercise
    // the (already fatal) probe-failure branch.
    for id in 0..common::N_DOCS {
        let _: () = redis::cmd("UNLINK")
            .arg(format!("cfg290unver:{id}"))
            .query(&mut conn)
            .unwrap();
    }
    assert_eq!(
        ft_info_num_docs(&mut conn, "idx:cfg290unver"),
        0,
        "server-side ground truth for this phase: the corpus is gone"
    );
    let empty = run(&[]);
    let empty_out = combined(&empty);
    assert!(
        !empty.status.success(),
        "--skip-upload over an EMPTY corpus it cannot verify must abort.\n{empty_out}"
    );
    assert!(
        !wrote_search_result(),
        "the rejected run must not publish a result file for an empty corpus"
    );

    // (c) The waiver. It must work, must warn, and must be recorded.
    let waived = run(&["--allow-partial-corpus"]);
    assert!(
        waived.status.success(),
        "--allow-partial-corpus must downgrade the guard to a warning.\n{}",
        combined(&waived)
    );
    assert!(
        String::from_utf8_lossy(&waived.stderr).contains("WARNING: --skip-upload"),
        "the override must still warn.\n{}",
        combined(&waived)
    );
    let reuse = common::read_params_obj(&proj.root, "cfg290unver")["corpus_reuse"].clone();
    assert_eq!(
        reuse["status"], "corpus_size_unknown",
        "the waived verdict must be distinguishable from an engine-side \
         'unverified' in the artifact: {reuse}"
    );
    assert_eq!(reuse["expected_rows"], serde_json::Value::Null);
    assert_eq!(reuse["actual_rows"], 0);
    assert_eq!(reuse["waived_by_allow_partial_corpus"], true);
    assert_eq!(
        ft_info_num_docs(&mut conn, "idx:cfg290unver"),
        0,
        "the reuse check must never modify the corpus, waived or not"
    );

    // Illustrative only: this is the number the default path used to publish
    // with exit 0. It is NOT the detector — see the doc comment.
    assert_eq!(
        common::read_recall(&proj.root, "cfg290unver"),
        0.0,
        "an empty corpus scores recall 0.0 — the number #290 is about"
    );
    assert_eq!(
        common::read_results_obj(&proj.root, "cfg290unver")["failed_queries"],
        0,
        "and it scores it with no failed queries, which is why nothing else catches it"
    );

    // (d) The OTHER #290 path, and the widest one: the corpus file IS present
    // but its header cannot be read. `reuse_expected_rows` already returned Err
    // here before this fix — it was the CALL SITE that degraded that Err to
    // `None`, leaving the guard inert for the run. So this phase, not the unit
    // tests, is what covers the swallowed-Err half of the issue.
    delete_search_result_files(&results_dir);
    fs::write(&vectors_npy, b"not an npy file at all").expect("write junk vectors.npy");
    let junk = run(&[]);
    let junk_out = combined(&junk);
    assert!(
        !junk.status.success(),
        "a failed corpus measurement must abort the run, not be swallowed into \
         'cannot verify' (issue #290).\n{junk_out}"
    );
    assert!(
        junk_out.contains("measuring the corpus on disk failed"),
        "the error must say the measurement FAILED, not that the layout has no \
         count — they are different conditions.\n{junk_out}"
    );
    assert!(
        !wrote_search_result(),
        "the rejected run must not publish a result file after a failed measurement"
    );

    flush_db(&mut conn);
    fs::remove_dir_all(&proj.root).ok();
}
