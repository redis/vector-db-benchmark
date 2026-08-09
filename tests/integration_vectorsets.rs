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
//!   VECTORSETS_PORT=6398 cargo test --test integration_vectorsets
//!
//! The suite is parallel-safe since #236: each config's corpus lives in its own
//! `idx:<config-name>` vector set, so tests no longer share one `idx` key that
//! each `configure()` deleted out from under its neighbours. (Before that fix the
//! `--test-threads=1` flag was mandatory, not merely advisable — without it most
//! of these tests failed at recall 0.000.) Every test below therefore MUST use a
//! distinct engine-config name; that name is the isolation boundary.

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

/// End-to-end DATETIME range filter whose schema declares `ts` as **`keyword`**,
/// with a **one-sided** `lt` bound (PR #230 review M2).
///
/// This is the regression test for the storage/filter DISAGREEMENT, a second
/// silent-wrong route into issue #220's failure class. The first fix for #220
/// decided "is this field a datetime?" from `dataset.config.schema` on the
/// STORAGE side but from the VALUE on the FILTER side. Whenever the schema does
/// not say `datetime` the two disagree: upload writes the ISO string, the filter
/// compares against an epoch number, and VSIM coerces the non-numeric attribute
/// to `0` — so `.ts < <epoch>` matches EVERY document and the query is
/// effectively UNFILTERED. Exit code 0, no error, no warning; recall is the only
/// detector (measured live on redis:8.8.0: 0.800 schema-gated vs 1.000 once both
/// halves derive the representation from the value alone).
///
/// The one-sided bound is the point. A two-sided range collapses to zero hits,
/// which is loud; `lt`/`lte` alone silently returns the whole corpus.
#[test]
fn test_binary_vectorsets_datetime_keyword_schema() {
    wait_for_vectorsets();

    let name = "vectorsets-dt-kw";
    let proj = common::write_datetime_keyword_schema_cosine_project(
        "vs-dt-kw",
        &vectorsets_config(name),
        8,
    );
    assert!(proj.matching_docs >= proj.top);

    let port = test_port().to_string();
    assert!(
        common::run_binary(
            &proj.root,
            name,
            "vs-dt-kw",
            TEST_HOST,
            &[("REDIS_PORT", port.as_str())],
        ),
        "vectorsets keyword-schema datetime run failed"
    );

    let recall = common::read_recall(&proj.root, name);
    println!("vectorsets keyword-schema datetime recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "vectorsets keyword-schema datetime recall {:.3} < 0.9 \
         (storage stored ISO strings while the filter compared epochs → `.ts < N` matched everything?)",
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
            // --keep-data suppresses the teardown `engine.delete()`, so the
            // server-side assertions at the end of this test have a corpus to
            // read. Without it `VCARD idx:vsets-mx` is 0 by the time the binary
            // exits and the read proves nothing. The key is removed below.
            &[
                "--update-search-ratio",
                "1:5",
                "--repetitions",
                "1",
                "--keep-data",
            ],
        ),
        "vectorsets mixed run failed"
    );

    let r = common::read_results_obj(&proj.root, name);
    let recall = r["mean_recall"].as_f64().unwrap();
    let precision = r["mean_precision_at_returned"].as_f64().unwrap();
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

    // --- #293: none of the four assertions above can see whether the updates
    // reached the corpus that was searched. recall/precision are measured on a
    // corpus that is complete either way, and update_count/update_rps were
    // counts of client-side loop iterations.
    //
    // What actually catches the #293 mutation (updates pointed at another key)
    // is NOT an assertion here — it is the in-run gate, which rejects the point,
    // so `run_binary` above returns false and this test fails there. The failing
    // branch itself is driven end to end by
    // `test_binary_vectorsets_mixed_updates_that_miss_the_corpus_are_fatal`.
    //
    // `update_unattributed` is the gate's actual output; the other two are the
    // tier label (a per-engine constant) and the unrelated failure count.
    assert_eq!(
        r["update_unattributed"].as_u64(),
        Some(0),
        "a healthy mixed run must land every update on an element already in the \
         searched set"
    );
    assert_eq!(
        r["update_attribution"].as_str(),
        Some("corpus_row"),
        "VectorSets must publish that every counted update was confirmed by the \
         server to have overwritten an element already in the searched set"
    );
    assert_eq!(
        r["update_failures"].as_u64(),
        Some(0),
        "no update should have failed in this run"
    );

    let url = format!("redis://{}:{}/", TEST_HOST, test_port());
    let client = redis::Client::open(url.as_str()).unwrap();
    let mut conn = client.get_connection().unwrap();
    // #236: this config's corpus is the single key `idx:<config-name>`.
    let searched_key = format!("idx:{name}");
    let vcard: i64 = redis::cmd("VCARD")
        .arg(&searched_key)
        .query(&mut conn)
        .unwrap();
    // This test ran with --keep-data, so the corpus is ours to clean up. Do it
    // BEFORE the assertions below: a panic between the read and the DEL would
    // otherwise leak `idx:<config>` onto the shared server for every later test.
    let _: () = redis::cmd("DEL")
        .arg(&searched_key)
        .query(&mut conn)
        .unwrap();

    // Positive control: without this, the equality below could be satisfied by
    // a server that lost the corpus entirely (0 == 0 against a broken expected).
    assert!(
        vcard > 0,
        "searched key {searched_key} is empty after the mixed run — the corpus \
         assertion below would be checking nothing"
    );
    // A mixed update overwrites elements that are already present, so the
    // cardinality must be exactly the uploaded corpus — an update that ADDED an
    // element to the searched set would push this past 400.
    //
    // LIMIT, stated because it is easy to over-read: this equality does NOT
    // prove the updates landed. Under the #293 mutation the searched set keeps
    // all 400 elements untouched and this assertion passes; the guard inside the
    // run is what fails there.
    assert_eq!(
        vcard, 400,
        "the searched vector set must still hold the whole 400-doc corpus"
    );
    std::fs::remove_dir_all(&proj.root).ok();
}

/// The server reply that makes `update_count` server-attributable (#293).
///
/// `vadd_single` now returns "did this write CREATE something" straight from
/// `VADD`'s integer reply, which `finalize_update_stats` records and
/// `experiment::gate_update_attribution` then rejects the run over. That guard is only as good as
/// the reply, and the reply is a server behaviour no unit test can pin — if a
/// future Redis returned a constant, the guard would silently become vacuous
/// while every test stayed green. So assert it against the live server.
///
/// This test does NOT exercise the benchmark's mixed path; it pins the single
/// server fact that path depends on.
#[test]
fn test_vadd_reply_distinguishes_a_new_element_from_an_overwrite() {
    wait_for_vectorsets();

    let url = format!("redis://{}:{}/", TEST_HOST, test_port());
    let client = redis::Client::open(url.as_str()).unwrap();
    let mut conn = client.get_connection().unwrap();

    // Own key, outside the `idx:` namespace every other test in this file uses,
    // so this stays parallel-safe. Torn down at the end.
    let key = "vsets-293-vadd-reply-probe";
    let _: () = redis::cmd("DEL").arg(key).query(&mut conn).unwrap();

    // POSITIVE CONTROL: a genuinely new element reports 1. If VADD always
    // reported 0, the overwrite assertion below would pass vacuously and the
    // shipped guard could never fire.
    // Built in the SHAPE `vadd_single` actually sends: FP32 bytes, the quant
    // token, M / EF, CAS (true in the shipped test configs) and SETATTR. A bare
    // `VADD key VALUES ...` would pin a weaker command than the engine issues,
    // and this reply is what the whole guard reads.
    let vadd = |conn: &mut redis::Connection, vec: [f32; 3], elem: &str| -> i64 {
        let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        redis::cmd("VADD")
            .arg(key)
            .arg("FP32")
            .arg(&bytes[..])
            .arg(elem)
            .arg("NOQUANT")
            .arg("M")
            .arg(16)
            .arg("EF")
            .arg(200)
            .arg("CAS")
            .arg("SETATTR")
            .arg(r#"{"color":"red"}"#)
            .query(conn)
            .unwrap()
    };

    let created: i64 = vadd(&mut conn, [1.0, 0.0, 0.0], "e1");
    assert_eq!(created, 1, "VADD of a new element must reply 1");

    // The mixed-workload case: same element id, different vector.
    let overwritten: i64 = vadd(&mut conn, [0.0, 1.0, 0.0], "e1");
    assert_eq!(
        overwritten, 0,
        "VADD overwriting an existing element must reply 0 — the whole #293 guard \
         reads this value to tell an in-corpus update from a write that landed \
         somewhere the search never looks"
    );

    // A second distinct element is a creation again, and the set grew by
    // exactly the two creations — not by the overwrite.
    let created2: i64 = vadd(&mut conn, [0.0, 0.0, 1.0], "e2");
    assert_eq!(created2, 1);
    let card: i64 = redis::cmd("VCARD").arg(key).query(&mut conn).unwrap();
    assert_eq!(card, 2, "two creations, one overwrite => cardinality 2");

    let _: () = redis::cmd("DEL").arg(key).query(&mut conn).unwrap();
}

/// Geo-radius end-to-end (issue #223).
///
/// Before this, `geo` hit `build_clause`'s `_ => None`: with the geo leaf the
/// ONLY leaf, VSIM ran with no `FILTER` argument at all and its recall was
/// scored against geo-filtered ground truth. Since #251 the same input is a hard
/// error, so the shipped `random-geo-radius-*-angular-filters` were unrunnable.
///
/// VSIM `FILTER` has no geo type and no function calls, but it does have `*`,
/// `+` and `>=` over several top-level attributes at once — enough for an EXACT
/// great-circle test once the point is stored as its unit vector on the sphere
/// (`engine::geo`). The fixture is the bounding-box-discriminating one: 300 of
/// 400 documents sit in the corners of the query's lat/lon box but outside its
/// circle, so a box (or no filter) scores ~0.25 and only the true radius scores
/// ~1.0. Cosine ground truth, because VSIM ranks by cosine intrinsically.
#[test]
fn test_binary_vectorsets_geo() {
    wait_for_vectorsets();

    let name = "vectorsets-geo";
    let proj = common::write_geo_corner_cosine_project("vs-geo", &vectorsets_config(name), 8);
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
            "vs-geo",
            TEST_HOST,
            &[("REDIS_PORT", port.as_str())],
        ),
        "vectorsets geo run failed"
    );

    let recall = common::read_recall(&proj.root, name);
    println!("vectorsets geo recall={recall:.3}");
    assert!(recall >= 0.9, "vectorsets geo recall {recall:.3} < 0.9");
    std::fs::remove_dir_all(&proj.root).ok();
}

// ── #236: per-config key namespacing ─────────────────────────────────────────

/// Open a connection to the live server used by these tests.
fn conn() -> redis::Connection {
    let url = format!("redis://{}:{}/", TEST_HOST, test_port());
    redis::Client::open(url.as_str())
        .expect("redis client")
        .get_connection()
        .expect("redis connection")
}

/// `VCARD <key>` — element count of a vector set, `0` when the key is absent.
/// This is the load-bearing read-back: it is answered by the SERVER about the
/// SERVER's state, so it cannot be satisfied by a plausible-looking recall.
fn vcard(conn: &mut redis::Connection, key: &str) -> i64 {
    redis::cmd("VCARD").arg(key).query::<i64>(conn).unwrap_or(0)
}

/// `VDIM <key>` — dimensionality of the vectors stored in a vector set, `0` when
/// the key is absent. Two configs uploaded at different dims therefore have
/// distinguishable *contents*, not merely distinguishable counts.
fn vdim(conn: &mut redis::Connection, key: &str) -> i64 {
    redis::cmd("VDIM").arg(key).query::<i64>(conn).unwrap_or(0)
}

/// `vector_count` as written into the fixture's `datasets/datasets.json`, i.e.
/// exactly how many vectors a completed upload must leave on the server.
fn expected_count(root: &std::path::Path) -> i64 {
    let raw = std::fs::read_to_string(root.join("datasets/datasets.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    v[0]["vector_count"].as_i64().expect("vector_count")
}

/// Issue #236 — two VectorSets configs sharing one server must own two DISJOINT
/// vector sets.
///
/// Before this fix every config addressed the literal key `idx`, and
/// `configure()` opened with `DEL idx`. So starting config B did not merely
/// interleave with config A — it **deleted A's entire corpus**, then rebuilt the
/// same key with its own. That is the #151-4 hazard the Redis family was fixed
/// for, applied to a single key instead of an index + keyspace.
///
/// The assertions are deliberately all server-side (`VCARD`/`VDIM`/`EXISTS`),
/// never recall. Two same-shaped corpora produce a perfectly plausible recall
/// whichever one actually answered the query, which is precisely why this bug
/// survived a suite full of recall assertions — see the repo's silent-wrong
/// bug class. The two configs upload at DIFFERENT dimensionalities (8 and 16),
/// so a surviving key can be attributed to its owner by content and not only by
/// count.
///
/// RED on master: after run B, `VCARD idx:vectorsets-iso-a` is 0 — A's corpus is
/// gone and only the shared `idx` exists.
#[test]
fn test_vectorsets_two_configs_do_not_clobber_each_other() {
    wait_for_vectorsets();

    let name_a = "vectorsets-iso-a";
    let name_b = "vectorsets-iso-b";
    // Must match `index_naming::derive_index_name("VECTORSETS_INDEX_NAME", "idx", …)`.
    // Spelled out literally rather than imported: the point of the test is to pin
    // the on-the-wire key, so it must fail if the derivation changes shape.
    let key_a = format!("idx:{name_a}");
    let key_b = format!("idx:{name_b}");
    const LEGACY_KEY: &str = "idx";

    let proj_a = common::write_match_any_cosine_project("vs-iso-a", &vectorsets_config(name_a), 8);
    let proj_b = common::write_match_any_cosine_project("vs-iso-b", &vectorsets_config(name_b), 16);
    let n_a = expected_count(&proj_a.root);
    let n_b = expected_count(&proj_b.root);

    let mut c = conn();
    // Clean slate for exactly the three keys this test reasons about (never
    // FLUSHALL — the suite shares one server with the other VectorSets tests).
    for k in [key_a.as_str(), key_b.as_str(), LEGACY_KEY] {
        let _ = redis::cmd("DEL").arg(k).query::<i64>(&mut c);
    }

    let port = test_port().to_string();
    let env = [("REDIS_PORT", port.as_str())];

    // ── Run config A, keeping its data resident ──
    assert!(
        common::run_binary_extra(
            &proj_a.root,
            name_a,
            "vs-iso-a",
            TEST_HOST,
            &env,
            &["--keep-data"]
        ),
        "config A run failed"
    );
    // Snapshot rather than assert here: every assertion is deferred to the end so
    // that a regression prints the WHOLE before/after picture (including what
    // landed in the legacy shared key) instead of aborting at the first symptom.
    let card_a_after_a = vcard(&mut c, &key_a);
    println!(
        "#236 read-back after config A: VCARD {key_a}={card_a_after_a} (VDIM {}) | \
         legacy VCARD {LEGACY_KEY}={} (VDIM {})",
        vdim(&mut c, &key_a),
        vcard(&mut c, LEGACY_KEY),
        vdim(&mut c, LEGACY_KEY),
    );

    // ── Run config B against the SAME server ──
    assert!(
        common::run_binary_extra(
            &proj_b.root,
            name_b,
            "vs-iso-b",
            TEST_HOST,
            &env,
            &["--keep-data"]
        ),
        "config B run failed"
    );

    // Both corpora coexist, each with its own count …
    let card_a = vcard(&mut c, &key_a);
    let card_b = vcard(&mut c, &key_b);
    println!(
        "#236 read-back after config B: VCARD {key_a}={card_a} (VDIM {}), \
         VCARD {key_b}={card_b} (VDIM {}) | legacy VCARD {LEGACY_KEY}={} (VDIM {})",
        vdim(&mut c, &key_a),
        vdim(&mut c, &key_b),
        vcard(&mut c, LEGACY_KEY),
        vdim(&mut c, LEGACY_KEY),
    );
    assert_eq!(
        card_a_after_a, n_a,
        "config A must upload into its own key '{key_a}' (#236)"
    );
    assert_eq!(
        card_a, n_a,
        "config B clobbered config A's corpus (#236): VCARD {key_a} = {card_a}, expected {n_a}"
    );
    assert_eq!(
        card_b, n_b,
        "config B did not land in its own key (#236): VCARD {key_b} = {card_b}, expected {n_b}"
    );

    // … and neither key holds the other's vectors: the dims are the ones each
    // config uploaded, so no key was rebuilt from the other's corpus.
    assert_eq!(
        vdim(&mut c, &key_a),
        8,
        "'{key_a}' does not hold A's 8-d vectors"
    );
    assert_eq!(
        vdim(&mut c, &key_b),
        16,
        "'{key_b}' does not hold B's 16-d vectors"
    );

    // And nothing was written to the old shared key at all.
    let legacy_exists: i64 = redis::cmd("EXISTS")
        .arg(LEGACY_KEY)
        .query(&mut c)
        .unwrap_or(0);
    assert_eq!(
        legacy_exists, 0,
        "the hardcoded shared key '{LEGACY_KEY}' must no longer be written (#236)"
    );

    // Teardown: this test opted into --keep-data, so it owns the cleanup.
    for k in [key_a.as_str(), key_b.as_str(), LEGACY_KEY] {
        let _ = redis::cmd("DEL").arg(k).query::<i64>(&mut c);
    }
    std::fs::remove_dir_all(&proj_a.root).ok();
    std::fs::remove_dir_all(&proj_b.root).ok();
}

/// Issue #236, part 2 — VectorSets must PARTICIPATE in the #151-4 startup
/// collision guard, not just derive a good default.
///
/// The derivation alone leaves one way back into the shared-key world: the
/// `VECTORSETS_INDEX_NAME_EXACT` pin drops the per-config suffix, so N configs
/// resolve to one verbatim key again. The Redis-family engines have that case
/// rejected at startup; before this fix `experiment::run`'s guard did not know
/// the string `"vectorsets"` at all, so the pin silently re-armed the bug.
///
/// Asserted on the ERROR TEXT, not merely a non-zero exit: without a live
/// server (or with any other misconfiguration) the run would fail anyway, and a
/// bare exit-code assertion would pass against a removed guard.
#[test]
fn test_vectorsets_exact_pin_with_two_configs_is_rejected_at_startup() {
    let configs = serde_json::json!([
        {
            "name": "vectorsets-guard-a",
            "engine": "vectorsets",
            "search_params": [{"parallel": 1, "search_params": {"ef": 64}}],
        },
        {
            "name": "vectorsets-guard-b",
            "engine": "vectorsets",
            "search_params": [{"parallel": 1, "search_params": {"ef": 64}}],
        },
    ]);
    let proj = common::write_match_any_cosine_project(
        "vs-guard",
        &serde_json::to_string(&configs).unwrap(),
        8,
    );

    let out = std::process::Command::new(common::binary_path())
        .args([
            "--engines",
            "vectorsets-guard-*",
            "--datasets",
            "vs-guard",
            "--host",
            TEST_HOST,
            "--skip-if-exists",
            "false",
        ])
        .current_dir(&proj.root)
        .env("REDIS_PORT", test_port().to_string())
        .env("VECTORSETS_INDEX_NAME", "shared-vset")
        .env("VECTORSETS_INDEX_NAME_EXACT", "1")
        .output()
        .expect("run vector-db-benchmark");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "exact-pinned sweep of 2 vectorsets configs must be rejected, not run: {combined}"
    );
    assert!(
        combined.contains("derive the same index namespace"),
        "the failure must be the #151-4 collision guard, not an incidental error: {combined}"
    );
    assert!(
        combined.contains("VECTORSETS_INDEX_NAME_EXACT is set"),
        "the guard must name the exact-pin as the cause so the fix is obvious: {combined}"
    );
    std::fs::remove_dir_all(&proj.root).ok();
}

/// Issue #236 × #271 — the migration case the fix creates, and the guard that
/// makes it loud.
///
/// A corpus written by a pre-#236 binary sits under the bare key `idx`. A
/// current binary addresses `idx:<config>`, so `--skip-upload` finds nothing —
/// and `VSIM` against a MISSING key returns an **empty array with no error**, so
/// no amount of downstream inspection can tell "no data" from "no matches". Left
/// unguarded this publishes a `mean_recall: 0.0` result file and exits 0: the
/// repo's silent-wrong-result class, in the exact scenario this PR's migration
/// note describes.
///
/// It is guarded, and by the mechanism that already exists rather than a second
/// one: `--skip-upload` runs `check_corpus_reuse_precondition`, which calls
/// `Engine::corpus_row_count` (#271 — `VCARD <key>` for VectorSets) and hard-errors
/// when the server holds fewer rows than the dataset declares.
///
/// Where the teeth actually are, stated precisely because an earlier version of
/// this comment got it wrong. The legacy key is seeded with exactly the expected
/// row count, which *looks* like the load-bearing part — as if a probe hardcoded
/// to `idx` would read 400 of 400 and wave the run through. It would not: this
/// test runs over a PRIVATE base (`BASE` below, not the literal `idx`, for the
/// isolation reason given there), so a hardcoded probe reads an empty `idx`, gets
/// 0 of 400, and hard-errors — passing the negative half for the wrong reason.
///
/// The half that actually fails against a wrong-key probe is the **positive
/// control** at the end: `--skip-upload` against the corpus this config really
/// uploaded must SUCCEED, and a probe reading `idx` sees 0 there and rejects it.
/// Verified by counterfactual — with `corpus_row_count` patched back to
/// `.arg("idx")`, this test fails at the positive control, not at any assertion
/// above it. (`test_vectorsets_corpus_row_count_tracks_the_live_key` fails too,
/// and its `holds 200 of the 400 rows` assertion IS a direct wrong-key detector.)
///
/// The seeding still earns its place: it makes the scenario the realistic one (a
/// legacy corpus present, not merely absent) and proves the run is rejected even
/// when a full-size corpus is sitting right there on the server.
#[test]
fn test_vectorsets_skip_upload_hard_errors_on_a_pre_236_corpus() {
    wait_for_vectorsets();

    let name = "vectorsets-migrate";
    // The migration is "un-suffixed base key" → "<base>:<config>". Reproduced here
    // over a PRIVATE base rather than the real `idx`, for two reasons: the shape is
    // identical (that is exactly what `VECTORSETS_INDEX_NAME_EXACT` writes), and the
    // literal `idx` is global state that
    // `test_vectorsets_two_configs_do_not_clobber_each_other` asserts is never
    // written — two tests mutating it would race at default test threads, which is
    // the very property #236 restored.
    const BASE: &str = "idx236mig";
    let key = format!("{BASE}:{name}");
    const LEGACY_KEY: &str = BASE;

    let proj = common::write_match_any_cosine_project("vs-migrate", &vectorsets_config(name), 8);
    let n = expected_count(&proj.root);

    let mut c = conn();
    for k in [key.as_str(), LEGACY_KEY] {
        let _ = redis::cmd("DEL").arg(k).query::<i64>(&mut c);
    }

    // Seed the pre-#236 corpus: `n` vectors under the bare shared key, i.e. the
    // exact count a probe on the WRONG key would happily accept.
    for id in 0..n {
        let mut cmd = redis::cmd("VADD");
        cmd.arg(LEGACY_KEY).arg("VALUES").arg(8);
        for d in 0..8 {
            cmd.arg((id as f64 * 0.001 + d as f64).to_string());
        }
        cmd.arg(id.to_string());
        cmd.query::<i64>(&mut c).expect("seed legacy corpus");
    }
    assert_eq!(
        vcard(&mut c, LEGACY_KEY),
        n,
        "legacy corpus must be seeded to exactly the expected row count"
    );
    assert_eq!(
        vcard(&mut c, &key),
        0,
        "the per-config key must start empty"
    );

    let out = std::process::Command::new(common::binary_path())
        .args([
            "--engines",
            name,
            "--datasets",
            "vs-migrate",
            "--host",
            TEST_HOST,
            "--skip-if-exists",
            "false",
            "--skip-upload",
        ])
        .current_dir(&proj.root)
        .env("REDIS_PORT", test_port().to_string())
        .env("VECTORSETS_INDEX_NAME", BASE)
        .output()
        .expect("run vector-db-benchmark");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(
        !out.status.success(),
        "--skip-upload against a pre-#236 corpus must FAIL, not publish recall 0.0: {combined}"
    );
    assert!(
        combined.contains("the corpus you asked to reuse is empty or missing"),
        "the failure must be the corpus-reuse guard reading THIS config's key, \
         not an incidental error: {combined}"
    );
    // The whole point is that no number gets published.
    let results = std::fs::read_dir(proj.root.join("results"))
        .map(|d| d.filter_map(|e| e.ok()).count())
        .unwrap_or(0);
    assert_eq!(results, 0, "a rejected run must publish no result file");

    // ── Positive control: the same flags against a corpus this config actually
    // uploaded must succeed. Without this the assertions above would also pass
    // against a guard that rejects unconditionally.
    let _ = redis::cmd("DEL").arg(LEGACY_KEY).query::<i64>(&mut c);
    let port = test_port().to_string();
    let env = [
        ("REDIS_PORT", port.as_str()),
        ("VECTORSETS_INDEX_NAME", BASE),
    ];
    assert!(
        common::run_binary_extra(
            &proj.root,
            name,
            "vs-migrate",
            TEST_HOST,
            &env,
            &["--keep-data"]
        ),
        "upload run failed"
    );
    assert_eq!(vcard(&mut c, &key), n, "upload must land in '{key}'");
    assert!(
        common::run_binary_extra(
            &proj.root,
            name,
            "vs-migrate",
            TEST_HOST,
            &env,
            &["--skip-upload", "--keep-data"]
        ),
        "--skip-upload against this config's OWN corpus must succeed"
    );

    for k in [key.as_str(), LEGACY_KEY] {
        let _ = redis::cmd("DEL").arg(k).query::<i64>(&mut c);
    }
    std::fs::remove_dir_all(&proj.root).ok();
}

/// Issue #238/#271 family test, the VectorSets member that #271 did not ship:
/// `corpus_row_count()` must be a LIVE read of THIS config's key, not a constant
/// and not the old shared `idx`. Amputate the corpus behind the tool's back and
/// the reported number has to follow.
///
/// kividb / mongodb / qdrant / valkey each got this test in #271; VectorSets did
/// not, which is exactly why CI could not see that its probe was hardcoded to
/// `idx` while #236 moved every other site to `idx:<config>`.
#[test]
fn test_vectorsets_corpus_row_count_tracks_the_live_key() {
    wait_for_vectorsets();

    let name = "vectorsets-rowcount";
    let key = format!("idx:{name}");
    let proj = common::write_match_any_cosine_project("vs-rowcount", &vectorsets_config(name), 8);
    let n = expected_count(&proj.root);

    let mut c = conn();
    let _ = redis::cmd("DEL").arg(&key).query::<i64>(&mut c);

    let port = test_port().to_string();
    let env = [("REDIS_PORT", port.as_str())];
    assert!(
        common::run_binary_extra(
            &proj.root,
            name,
            "vs-rowcount",
            TEST_HOST,
            &env,
            &["--keep-data", "--skip-search"]
        ),
        "phase 1 (upload) failed"
    );
    assert_eq!(vcard(&mut c, &key), n, "upload must land in '{key}'");

    // Amputate half the corpus with VREM, straight on the server.
    let half = n / 2;
    for id in 0..half {
        let removed: i64 = redis::cmd("VREM")
            .arg(&key)
            .arg(id.to_string())
            .query(&mut c)
            .expect("VREM");
        assert_eq!(removed, 1, "VREM must actually remove element {id}");
    }
    assert_eq!(vcard(&mut c, &key), n - half);

    let out = std::process::Command::new(common::binary_path())
        .args([
            "--engines",
            name,
            "--datasets",
            "vs-rowcount",
            "--host",
            TEST_HOST,
            "--skip-if-exists",
            "false",
            "--skip-upload",
            "--keep-data",
        ])
        .current_dir(&proj.root)
        .env("REDIS_PORT", &port)
        .output()
        .expect("run vector-db-benchmark");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(
        !out.status.success(),
        "--skip-upload against a half-deleted vectorsets corpus must be a hard error.\n{combined}"
    );
    assert!(
        combined.contains(&format!("holds {} of the {n} rows", n - half)),
        "the count must track the amputation, proving it is a live read of '{key}' \
         and not a constant or the old shared 'idx'.\n{combined}"
    );

    let _ = redis::cmd("DEL").arg(&key).query::<i64>(&mut c);
    std::fs::remove_dir_all(&proj.root).ok();
}

/// Per-config memory attribution (#236 follow-on).
///
/// `get_memory_usage` reports server-wide `INFO memory:used_memory`. That number
/// was *correct* as a per-config figure before this PR — the hardcoded key plus a
/// destructive `DEL` in `configure()` guaranteed exactly one resident corpus. Now
/// that configs coexist it is the SUM over all of them, so every config after the
/// first over-reports (2× here, N× on an N-config sweep — and the coexistence
/// test above is precisely that scenario).
///
/// `VINFO` carries no memory field, but the corpus is a single key, so
/// `MEMORY USAGE <key>` attributes it exactly. This asserts the per-config figure
/// does NOT grow when a sibling config lands on the same server, while the global
/// `used_memory` does — i.e. that the two numbers now mean different things and
/// the per-config one is the honest one.
#[test]
fn test_vectorsets_index_memory_is_per_config_not_server_wide() {
    wait_for_vectorsets();

    let name_a = "vectorsets-mem-a";
    let name_b = "vectorsets-mem-b";
    let key_a = format!("idx:{name_a}");
    let key_b = format!("idx:{name_b}");

    let proj_a = common::write_match_any_cosine_project("vs-mem-a", &vectorsets_config(name_a), 8);
    let proj_b = common::write_match_any_cosine_project("vs-mem-b", &vectorsets_config(name_b), 8);
    let n_b = expected_count(&proj_b.root);

    let mut c = conn();
    for k in [key_a.as_str(), key_b.as_str()] {
        let _ = redis::cmd("DEL").arg(k).query::<i64>(&mut c);
    }

    let port = test_port().to_string();
    let env = [("REDIS_PORT", port.as_str())];

    assert!(
        common::run_binary_extra(
            &proj_a.root,
            name_a,
            "vs-mem-a",
            TEST_HOST,
            &env,
            &["--keep-data", "--skip-search"]
        ),
        "config A upload failed"
    );
    let (a_alone_index, a_alone_used) = read_upload_memory(&proj_a.root, name_a);

    assert!(
        common::run_binary_extra(
            &proj_b.root,
            name_b,
            "vs-mem-b",
            TEST_HOST,
            &env,
            &["--keep-data", "--skip-search"]
        ),
        "config B upload failed"
    );
    let (b_with_a_index, b_with_a_used) = read_upload_memory(&proj_b.root, name_b);

    println!(
        "#236 memory attribution: A alone index={a_alone_index} used_memory={a_alone_used} | \
         B with A resident index={b_with_a_index} used_memory={b_with_a_used} | \
         MEMORY USAGE {key_a}={} {key_b}={}",
        memory_usage(&mut c, &key_a),
        memory_usage(&mut c, &key_b),
    );

    // The per-config figure is attributable: B's own corpus is the same shape as
    // A's, so B's index memory must stay in A's ballpark even though the SERVER
    // now holds both. A 1.5× ceiling is far below the 2× a summed figure gives.
    assert!(
        b_with_a_index > 0 && a_alone_index > 0,
        "per-config index_memory_bytes must be reported for both configs \
         (A={a_alone_index}, B={b_with_a_index})"
    );
    assert!(
        (b_with_a_index as f64) < 1.5 * a_alone_index as f64,
        "index_memory_bytes must be THIS config's corpus, not the server sum: \
         A alone={a_alone_index}, B with A resident={b_with_a_index}"
    );
    // …and it tracks the actual key, within the slack of the engine's own
    // bookkeeping between the upload snapshot and this read-back.
    let live_b = memory_usage(&mut c, &key_b);
    assert!(
        live_b > 0 && (b_with_a_index as f64) >= 0.5 * live_b as f64,
        "index_memory_bytes ({b_with_a_index}) must track MEMORY USAGE {key_b} ({live_b})"
    );
    // …and it SCALES with the corpus rather than reporting a constant. Every
    // assertion above is satisfied by a fixed per-key overhead figure, which is
    // the plausible way `MEMORY USAGE` could be meaningless here (a module type
    // with no `mem_usage` callback reports bare key overhead). Compare against a
    // one-element vector set of the same dimensionality on the same server: 400
    // elements must cost an order of magnitude more than 1.
    let tiny_key = "idx:vs-mem-tiny";
    let _ = redis::cmd("DEL").arg(tiny_key).query::<i64>(&mut c);
    let mut seed = redis::cmd("VADD");
    seed.arg(tiny_key).arg("VALUES").arg(8);
    for d in 0..8 {
        seed.arg(d.to_string());
    }
    seed.arg("solo");
    seed.query::<i64>(&mut c)
        .expect("seed 1-element vector set");
    let tiny = memory_usage(&mut c, tiny_key);
    let _ = redis::cmd("DEL").arg(tiny_key).query::<i64>(&mut c);
    assert!(
        tiny > 0 && b_with_a_index > 10 * tiny,
        "index_memory_bytes must scale with the corpus, not report a constant: \
         {n_b} elements = {b_with_a_index} B vs 1 element = {tiny} B"
    );
    // And the two numbers demonstrably mean different things: the server-wide
    // figure accounts for far more than this config's corpus, which is exactly
    // why it cannot serve as the per-config one. Asserted as a floor rather than
    // as growth between the two runs — `used_memory` is global state that any
    // concurrently-running test moves in either direction, and a flaky assertion
    // about the number we are de-emphasising would be its own kind of wrong. The
    // inequality only gets safer as other corpora land.
    assert!(
        b_with_a_used > 2 * b_with_a_index,
        "server-wide used_memory ({b_with_a_used}) should dwarf this config's own \
         corpus ({b_with_a_index}); if they are the same number, the per-config \
         figure is just the server total again"
    );
    let _ = a_alone_used; // reported above; not asserted (see note)

    for k in [key_a.as_str(), key_b.as_str()] {
        let _ = redis::cmd("DEL").arg(k).query::<i64>(&mut c);
    }
    std::fs::remove_dir_all(&proj_a.root).ok();
    std::fs::remove_dir_all(&proj_b.root).ok();
}

/// `MEMORY USAGE <key>`, or 0 when the key is absent.
fn memory_usage(conn: &mut redis::Connection, key: &str) -> i64 {
    redis::cmd("MEMORY")
        .arg("USAGE")
        .arg(key)
        .query::<Option<i64>>(conn)
        .ok()
        .flatten()
        .unwrap_or(0)
}

/// `(index_memory_bytes, used_memory)` from an engine's upload result JSON.
fn read_upload_memory(root: &std::path::Path, engine: &str) -> (i64, i64) {
    let pattern = format!("{engine}-*-upload-*.json");
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
        .unwrap_or_else(|| panic!("no upload result for {engine}"));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mem = &v["results"]["memory_usage"];
    (
        mem["index_memory_bytes"].as_i64().unwrap_or(-1),
        mem["used_memory"].as_i64().unwrap_or(-1),
    )
}

/// #293 end to end: every mixed update creates a NEW element instead of
/// overwriting one → hard error; `--allow-partial-corpus` → honest zeros.
///
/// Without this the `Ok(true) => ut.unattributed += 1` arm in `search_mixed` is
/// dead in every run, so "read the VADD reply" and "ignore it and return
/// `Ok(false)`" are indistinguishable — the whole fix could be reverted at the
/// source with the suite green. The mixed test above only pins the healthy
/// branch.
///
/// Fixture: the corpus keeps its exact cardinality but none of its element ids,
/// so `--skip-upload`'s reuse check (VCARD vs the declared row count) still
/// passes and is NOT what fires.
#[test]
fn test_binary_vectorsets_mixed_updates_that_miss_the_corpus_are_fatal() {
    wait_for_vectorsets();

    let name = "vs293miss";
    let ds = "vs293miss-test";
    let configs = serde_json::json!([{
        "name": name,
        "engine": "vectorsets",
        "search_params": [{"parallel": 1, "search_params": {"ef": 64}}],
        "upload_params": {
            "hnsw_config": {"quant": "NOQUANT", "M": 16, "EF_CONSTRUCTION": 200},
            "CAS": true, "parallel": 1, "batch_size": 100
        }
    }]);
    let proj = common::write_match_any_cosine_project_n(
        ds,
        &serde_json::to_string(&configs).unwrap(),
        8,
        500,
    );
    let port = test_port().to_string();
    let key = format!("idx:{name}");

    let url = format!("redis://{}:{}/", TEST_HOST, port);
    let client = redis::Client::open(url.as_str()).unwrap();
    let mut conn = client.get_connection().unwrap();

    // Rebuilt before EACH run: the gate rejects the numbers after the timed
    // window, so a rejected run has already added the elements it complained
    // were missing. Reusing that state would silently test the healthy path.
    let build_shifted_corpus = |conn: &mut redis::Connection| {
        let _: () = redis::cmd("DEL").arg(&key).query(conn).unwrap();
        // A vector set of the right cardinality whose element ids are all
        // outside the dataset's 0..N_DOCS range.
        for i in 0..common::N_DOCS {
            let v: Vec<u8> = (0..8u32)
                .flat_map(|d| ((i + d as usize) as f32).to_le_bytes())
                .collect();
            let _: i64 = redis::cmd("VADD")
                .arg(&key)
                .arg("FP32")
                .arg(&v[..])
                .arg(format!("{}", 90_000 + i))
                .arg("NOQUANT")
                .query(conn)
                .unwrap();
        }
        let card: i64 = redis::cmd("VCARD").arg(&key).query(conn).unwrap();
        assert_eq!(
            card,
            common::N_DOCS as i64,
            "the shifted corpus must keep its cardinality, so the reuse check \
             passes and this test exercises the #293 gate rather than #238's"
        );
        // Drop prior search result files so `read_results_obj` below cannot pick
        // up a stale one from the rejected run.
        let results_dir = proj.root.join("results");
        if let Ok(entries) = std::fs::read_dir(&results_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.contains("-search-"))
                {
                    std::fs::remove_file(path).ok();
                }
            }
        }
    };

    let run = |extra: &[&str]| {
        let mut cmd = std::process::Command::new(common::binary_path());
        cmd.args([
            "--engines",
            name,
            "--datasets",
            ds,
            "--host",
            TEST_HOST,
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
        cmd.current_dir(&proj.root)
            .env("REDIS_PORT", &port)
            .output()
            .expect("run vector-db-benchmark")
    };

    build_shifted_corpus(&mut conn);
    let out = run(&[]);
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "a mixed run whose every VADD created a new element must be a hard error \
         (#293), but the run succeeded.\n{combined}"
    );
    assert!(
        combined.contains("reported that the row each one addressed did not already exist")
            && combined.contains("VADD replied 1"),
        "the error must be the #293 gate quoting the VADD signal.\n{combined}"
    );

    build_shifted_corpus(&mut conn);
    let out2 = run(&["--allow-partial-corpus"]);
    assert!(
        out2.status.success(),
        "--allow-partial-corpus must downgrade the #293 gate to a warning.\n{}{}",
        String::from_utf8_lossy(&out2.stdout),
        String::from_utf8_lossy(&out2.stderr)
    );
    let r = common::read_results_obj(&proj.root, name);
    let update_count = r["update_count"].as_u64().unwrap();
    let unattributed = r["update_unattributed"].as_u64().unwrap();
    println!(
        "vectorsets #293 waived: update_count={update_count} update_unattributed={unattributed}"
    );
    assert!(unattributed > 0, "the missed updates must be recorded");
    assert_eq!(
        update_count, 0,
        "not one update overwrote an existing element, so update_count must be 0"
    );

    let _: () = redis::cmd("DEL").arg(&key).query(&mut conn).unwrap();
    std::fs::remove_dir_all(&proj.root).ok();
}
