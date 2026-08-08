//! Integration tests for the VectorSets engine's FILTER path (VADD/VSIM).
//!
//! VectorSets is Redis Vector Sets — same `redis:8.8.0` image as the other Redis
//! tests, but VSIM's FILTER expression grammar (bare-bool syntax error, numeric
//! coercion, value-`in`-field membership) is only observable against a LIVE
//! server, which is exactly why the earlier bool bug slipped past the
//! string-equality unit tests. These tests drive the real benchmark binary
//! end-to-end with filtered ground truth, so a high recall proves the FILTER was
//! actually applied (and, for bool, that it did not error the whole query).
//!
//! VectorSets ranks by COSINE similarity intrinsically (VADD/VSIM take no metric
//! arg), so the fixtures use cosine ground truth (`*_cosine_project`).
//!
//! Requires redis:8.8.0 (Vector Sets) reachable on `VECTORSETS_PORT` (default
//! 6398 — kept distinct from the redis:8.6.0 tests on 6399 because bool FILTER
//! grammar needs 8.8+). Start with:
//!   docker run -d --rm -p 6398:6379 redis:8.8.0
//! Run with:
//!   VECTORSETS_PORT=6398 cargo test --test integration_vectorsets -- --test-threads=1

use std::time::{Duration, Instant};

mod common;

const TEST_HOST: &str = "127.0.0.1";

/// Port of the live Redis 8.8 (Vector Sets) server. The engine reads `REDIS_PORT`
/// (see `engine::build_redis_url`), so we forward this value under that name.
fn test_port() -> u16 {
    std::env::var("VECTORSETS_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6398)
}

/// Block until the server answers PING (or panic after 10s), and verify it
/// actually supports Vector Sets (VADD) so a misconfigured image fails loudly.
fn wait_for_vectorsets() {
    let port = test_port();
    let url = format!("redis://{}:{}/", TEST_HOST, port);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(client) = redis::Client::open(url.as_str()) {
            if let Ok(mut conn) = client.get_connection() {
                if redis::cmd("PING").query::<String>(&mut conn).is_ok() {
                    // VADD arity error ("wrong number of arguments") proves the
                    // command EXISTS; "unknown command" means no Vector Sets.
                    let probe: Result<redis::Value, redis::RedisError> =
                        redis::cmd("VADD").query(&mut conn);
                    if let Err(e) = probe {
                        let msg = e.to_string().to_lowercase();
                        assert!(
                            !msg.contains("unknown command"),
                            "server on port {port} lacks Vector Sets (VADD). Use redis:8.8.0."
                        );
                    }
                    return;
                }
            }
        }
        if Instant::now() > deadline {
            panic!("Redis (Vector Sets) not available on port {port} after 10s");
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn vectorsets_config(name: &str) -> String {
    let configs = serde_json::json!([{
        "name": name,
        "engine": "vectorsets",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {
            "hnsw_config": {"quant": "NOQUANT", "M": 16, "EF_CONSTRUCTION": 200},
            "CAS": true,
            "parallel": 1,
            "batch_size": 100
        }
    }]);
    serde_json::to_string(&configs).unwrap()
}

/// End-to-end keyword `match_any`: filter `color IN {red, blue}` and assert the
/// engine returns the filtered nearest neighbours. Pins the `"<v>" in .field`
/// value-in-field emission (equality on a scalar keyword attribute).
#[test]
fn test_binary_vectorsets_match_any() {
    wait_for_vectorsets();

    let name = "vectorsets-ma";
    let proj = common::write_match_any_cosine_project("vs-match-any", &vectorsets_config(name), 8);
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
            "vs-match-any",
            TEST_HOST,
            &[("REDIS_PORT", port.as_str())],
        ),
        "vectorsets match_any run failed"
    );

    let recall = common::read_recall(&proj.root, name);
    println!("vectorsets match_any (keyword) recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "vectorsets keyword match_any recall {:.3} < 0.9",
        recall
    );
}

/// End-to-end integer `match_any`: filter `size IN {1, 2}`. Pins the numeric
/// `.field == N` arm (sizes are stored as JSON strings; VSIM coerces "1" == 1).
#[test]
fn test_binary_vectorsets_match_any_int() {
    wait_for_vectorsets();

    let name = "vectorsets-ma-int";
    let proj =
        common::write_match_any_int_cosine_project("vs-match-any-int", &vectorsets_config(name), 8);
    assert!(proj.matching_docs >= proj.top);

    let port = test_port().to_string();
    assert!(
        common::run_binary(
            &proj.root,
            name,
            "vs-match-any-int",
            TEST_HOST,
            &[("REDIS_PORT", port.as_str())],
        ),
        "vectorsets int match_any run failed"
    );

    let recall = common::read_recall(&proj.root, name);
    println!("vectorsets match_any (int) recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "vectorsets int match_any recall {:.3} < 0.9",
        recall
    );
}

/// End-to-end BOOL filter: `flag == true`. This pins FIX 1 end-to-end — a bare
/// `.flag == true` FILTER expression is a SYNTAX ERROR on a live VSIM server and
/// fails the WHOLE query; the fix quotes it (`.flag == "true"`, matching the
/// JSON-string storage), so the query succeeds and returns the filtered NNs.
/// Reverting the quoting turns this test RED (query error → run fails / no
/// result), which is the intended regression teeth.
#[test]
fn test_binary_vectorsets_bool_filter() {
    wait_for_vectorsets();

    let name = "vectorsets-bool";
    let proj = common::write_bool_cosine_project("vs-bool", &vectorsets_config(name), 8);
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
            "vs-bool",
            TEST_HOST,
            &[("REDIS_PORT", port.as_str())],
        ),
        "vectorsets bool filter run failed (bare `.flag == true` is a VSIM syntax error?)"
    );

    let recall = common::read_recall(&proj.root, name);
    println!("vectorsets bool filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "vectorsets bool filter recall {:.3} < 0.9",
        recall
    );
}

/// End-to-end DATETIME range filter: `ts IN [day 100, day 300)` over ISO-8601
/// bounds (issue #220). This is the regression test for a LIVE WRONG NUMBER: the
/// range builder rejected every non-numeric bound, so both ISO bounds were
/// dropped, the clause became `None`, and VSIM ran with **no FILTER argument** —
/// a full UNFILTERED search scored against datetime-filtered ground truth. That
/// failure returns plenty of results and no error, so this asserts RECALL (a
/// "query succeeded" assertion would have passed against the bug — measured on a
/// live redis:8.8.0: recall 0.520 unfiltered vs 1.000 fixed).
///
/// It pins BOTH halves of the fix, which must agree on the stored representation:
/// upload writes the `datetime`-typed attribute as epoch seconds, and the filter
/// emits epoch-second bounds. Reverting either half turns this test red.
#[test]
fn test_binary_vectorsets_datetime() {
    wait_for_vectorsets();

    let name = "vectorsets-dt";
    let proj = common::write_datetime_cosine_project("vs-dt", &vectorsets_config(name), 8);
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
            "vs-dt",
            TEST_HOST,
            &[("REDIS_PORT", port.as_str())],
        ),
        "vectorsets datetime run failed"
    );

    let recall = common::read_recall(&proj.root, name);
    println!("vectorsets datetime range recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "vectorsets datetime range recall {:.3} < 0.9 (ISO bounds dropped → unfiltered VSIM?)",
        recall
    );
}

/// End-to-end NESTED boolean group: `(color == red AND size >= 50) OR
/// (color == blue AND size < 10)`. `build_clauses` had no `and`/`or` recursion
/// branch, so each `{"and":[…]}` entry matched no field leaf, every clause was
/// dropped and — exactly as with the datetime bug — VSIM ran UNFILTERED. Recall
/// is the only detector: the unfiltered nearest neighbours differ from the
/// ~120-doc nested set (measured live: 0.260 broken vs 1.000 fixed).
#[test]
fn test_binary_vectorsets_nested_filter() {
    wait_for_vectorsets();

    let name = "vectorsets-nested";
    let proj = common::write_nested_filter_cosine_project("vs-nested", &vectorsets_config(name), 8);
    assert!(proj.matching_docs >= proj.top);

    let port = test_port().to_string();
    assert!(
        common::run_binary(
            &proj.root,
            name,
            "vs-nested",
            TEST_HOST,
            &[("REDIS_PORT", port.as_str())],
        ),
        "vectorsets nested filter run failed"
    );

    let recall = common::read_recall(&proj.root, name);
    println!("vectorsets nested filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "vectorsets nested filter recall {:.3} < 0.9 (nested group dropped → unfiltered VSIM?)",
        recall
    );
}

/// End-to-end MIXED harness (`--update-search-ratio`) at `parallel: 4`: drives
/// the VectorSets mixed path (VSIM search + VADD update) with a real multi-worker
/// join-merge of the thread-local sample buffers. Cosine ground truth (VectorSets
/// ranks by cosine). Asserts search recall/precision are intact, updates ran
/// (`update_count > 0`, `update_rps > 0`), and search percentiles are monotone.
#[test]
fn test_binary_vectorsets_mixed_benchmark() {
    wait_for_vectorsets();

    let name = "vsets-mx";
    let configs = serde_json::json!([{
        "name": name,
        "engine": "vectorsets",
        "search_params": [{"parallel": 4, "search_params": {"ef": 400}}],
        "upload_params": {
            "hnsw_config": {"quant": "NOQUANT", "M": 16, "EF_CONSTRUCTION": 200},
            "CAS": true,
            "parallel": 1,
            "batch_size": 100
        }
    }]);
    // 2000 queries so that at parallel: 4 the mixed loop reliably completes many
    // full search phases (and thus updates), and merges a large per-worker sample
    // set across threads.
    let proj = common::write_match_any_cosine_project_n(
        "vs-mx",
        &serde_json::to_string(&configs).unwrap(),
        8,
        2000,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = test_port().to_string();
    assert!(
        common::run_binary_extra(
            &proj.root,
            name,
            "vs-mx",
            TEST_HOST,
            &[("REDIS_PORT", port.as_str())],
            &["--update-search-ratio", "1:5", "--repetitions", "1"],
        ),
        "vectorsets mixed run failed"
    );

    let r = common::read_results_obj(&proj.root, name);
    let recall = r["mean_recall"].as_f64().unwrap();
    let precision = r["mean_precisions"].as_f64().unwrap();
    let update_count = r["update_count"].as_u64().unwrap();
    let update_rps = r["update_rps"].as_f64().unwrap();
    let p50 = r["p50_time"].as_f64().unwrap();
    let p95 = r["p95_time"].as_f64().unwrap();
    let p99 = r["p99_time"].as_f64().unwrap();
    println!(
        "vectorsets mixed: recall={recall:.3} precision={precision:.3} update_count={update_count} \
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
    std::fs::remove_dir_all(&proj.root).ok();
}
