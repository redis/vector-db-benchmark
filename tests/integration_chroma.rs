//! Integration tests for the Chroma engine.
//!
//! Requires Chroma running on port 8003 (container port 8000).
//! Start with: docker compose -f tests/docker-compose.test.yml up -d chroma --wait
//! Run with:   CHROMA_PORT=8003 cargo test --test integration_chroma -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

mod common;

const CHROMA_PORT: u16 = 8003;
const CHROMA_HOST: &str = "127.0.0.1";

fn chroma_port() -> u16 {
    std::env::var("CHROMA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CHROMA_PORT)
}

fn wait_for_chroma() {
    let client = reqwest::blocking::Client::new();
    let url = format!("http://{}:{}/api/v2/heartbeat", CHROMA_HOST, chroma_port());
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Ok(resp) = client.get(&url).send() {
            if resp.status().is_success() {
                return;
            }
        }
        if Instant::now() > deadline {
            panic!("Chroma not available on port {} after 60s", chroma_port());
        }
        thread::sleep(Duration::from_millis(500));
    }
}

/// Run one filter fixture end-to-end and assert recall >= 0.9 vs filtered ground
/// truth. `build` is a `common::write_*_project` fixture.
fn run_filter_recall_test(
    name: &str,
    dataset: &str,
    build: impl Fn(&str, &str, usize) -> common::FilterProject,
) {
    wait_for_chroma();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": name, "engine": "chroma",
        "search_params": [{"parallel": 1}],
        "upload_params": {"parallel": 1, "batch_size": 200}
    }]);
    let proj = build(dataset, &serde_json::to_string(&configs).unwrap(), dim);
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );
    let port = chroma_port().to_string();
    assert!(
        common::run_binary(
            &proj.root,
            name,
            dataset,
            CHROMA_HOST,
            &[("CHROMA_PORT", port.as_str()), ("CHROMA_COLLECTION", name)],
        ),
        "chroma {} run failed",
        name
    );
    let recall = common::read_recall(&proj.root, name);
    println!("chroma {} recall={:.3}", name, recall);
    assert!(recall >= 0.9, "chroma {} recall {:.3} < 0.9", name, recall);
}

#[test]
fn test_binary_chroma_bool() {
    run_filter_recall_test("chroma-bool", "bool-test", common::write_bool_project);
}

#[test]
fn test_binary_chroma_datetime() {
    run_filter_recall_test("chroma-dt", "dt-test", common::write_datetime_project);
}

#[test]
fn test_binary_chroma_uuid() {
    run_filter_recall_test("chroma-uuid", "uuid-test", common::write_uuid_project);
}

/// `match_any` on an int field → Chroma `$in` over the IN-set.
#[test]
fn test_binary_chroma_match_any_int() {
    wait_for_chroma();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "chroma-ma-int", "engine": "chroma",
        "search_params": [{"parallel": 1}],
        "upload_params": {"parallel": 1, "batch_size": 200}
    }]);
    let proj = common::write_match_any_int_project(
        "ma-int-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);
    let port = chroma_port().to_string();
    assert!(
        common::run_binary(
            &proj.root,
            "chroma-ma-int",
            "ma-int-test",
            CHROMA_HOST,
            &[
                ("CHROMA_PORT", port.as_str()),
                ("CHROMA_COLLECTION", "chroma-ma-int"),
            ],
        ),
        "chroma match_any_int run failed"
    );
    let recall = common::read_recall(&proj.root, "chroma-ma-int");
    println!("chroma match_any_int recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "chroma match_any_int recall {:.3} < 0.9",
        recall
    );
}

/// Full-text: `{match:{text}}` → Chroma `where_document` `$contains` over the
/// record's uploaded `document` (the text field's value). Even docs contain
/// "quick"; recall vs the docs that contain the token.
#[test]
fn test_binary_chroma_fulltext() {
    run_filter_recall_test("chroma-text", "text-test", common::write_fulltext_project);
}

#[test]
fn test_binary_chroma_and_filter() {
    run_filter_recall_test("chroma-and", "and-test", common::write_and_filter_project);
}

#[test]
fn test_binary_chroma_or_filter() {
    run_filter_recall_test("chroma-or", "or-test", common::write_or_filter_project);
}

#[test]
fn test_binary_chroma_nested_filter() {
    run_filter_recall_test(
        "chroma-nested",
        "nested-test",
        common::write_nested_filter_project,
    );
}

#[test]
fn test_binary_chroma_selectivity() {
    run_filter_recall_test("chroma-sel", "sel-test", common::write_selectivity_project);
}

/// Chroma REFUSES a geo dataset, loudly, and that is the intended end state
/// (issue #223) — not a step on the way to supporting it.
///
/// Chroma's `where` DSL is a closed enum of `field OP literal` comparisons
/// (`$eq $ne $gt $gte $lt $lte $in $nin $and $or`) over scalar metadata, with no
/// geospatial operator, no attribute-vs-attribute comparison and no arithmetic.
/// A spherical cap is not an axis-aligned box in any query-independent
/// coordinate system, so no conjunction of ranges expresses it, and the linear
/// `x*qx + y*qy + z*qz >= cos(r/R)` form VectorSets uses needs cross-field
/// arithmetic Chroma does not have. A bounding box would admit points up to
/// √2·r away — a filter that LOOKS applied and under-constrains, which is what
/// #219 exists to stop.
///
/// So the assertion is on the failure, and on the failure being the right one:
/// a nonzero exit with the #219 message, never a completed run with a plausible
/// recall. Asserting "the run fails" alone would pass for a crash or a typo, so
/// the sibling filter tests above (which must still succeed) are what prove the
/// refusal is specific to geo.
#[test]
fn test_binary_chroma_geo_is_refused_loudly() {
    wait_for_chroma();
    let dim = 8;
    let name = "chroma-geo";
    let configs = serde_json::json!([{
        "name": name, "engine": "chroma",
        "search_params": [{"parallel": 1}],
        "upload_params": {"parallel": 1, "batch_size": 200}
    }]);
    let proj = common::write_geo_corner_project(
        "chroma-geo-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    let port = chroma_port().to_string();
    let out = common::run_binary_capture(
        &proj.root,
        name,
        "chroma-geo-test",
        CHROMA_HOST,
        &[("CHROMA_PORT", port.as_str()), ("CHROMA_COLLECTION", name)],
    );
    assert!(
        !out.ok,
        "chroma must REFUSE a geo dataset, not report a recall for a filter it \
         never applied:\n{}",
        out.combined
    );
    assert!(
        out.combined.contains("UNFILTERED") && out.combined.contains("#219"),
        "the refusal must be the #219 dropped-filter error, not some other \
         failure:\n{}",
        out.combined
    );
    assert!(
        out.combined.contains("Chroma cannot express the filter"),
        "the error must name the engine:\n{}",
        out.combined
    );
}
