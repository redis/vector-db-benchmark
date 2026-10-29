//! One ingest, many workloads (#344): what the upload and search files say
//! about the shared index and the ingest phases.
use super::{build_search_result_json, build_upload_result_json, check_upload_phases};
use super::{stamp_shared_index, WorkloadPlan};
use crate::config::SearchParams;
use crate::engine::{SearchResults, UploadStats};
use serde_json::json;
use vector_db_benchmark::ingest::{IndexWait, IngestPhases};
use vector_db_benchmark::workload::{plan, IndexLayout, Workload};

fn phases() -> IngestPhases {
    IngestPhases::from_offsets(
        1_800_000_000.0,
        10.0,
        16.0,
        400,
        2.0,
        IndexWait::RedisPercentIndexed,
    )
}

fn stats_with(phases: Option<IngestPhases>, upload_time: f64, total_time: f64) -> UploadStats {
    UploadStats {
        upload_time,
        total_time,
        upload_count: 400,
        ingest_phases: phases,
        ..Default::default()
    }
}

#[test]
fn upload_file_adds_phase_keys_next_to_the_legacy_ones() {
    let stats = stats_with(Some(phases()), 10.0, 18.0);
    check_upload_phases(&stats).unwrap();
    let doc = build_upload_result_json("redis-x", "ds", &stats, None, &json!({}));
    let r = &doc["results"];
    // Legacy keys keep their values and meaning.
    assert_eq!(r["upload_time"], 10.0);
    assert_eq!(r["total_time"], 18.0);
    assert_eq!(r["upload_count"], 400);
    // New keys.
    assert_eq!(r["read_time"], 2.0);
    assert_eq!(r["index_wait_time"], 6.0);
    assert_eq!(r["total_ingest_time"], 16.0);
    assert_eq!(r["upload_docs_per_sec"], 40.0);
    assert_eq!(r["ingest_docs_per_sec"], 25.0);
    assert_eq!(
        r["ingest_phases"]["index_wait"],
        "redis_ft_info_percent_indexed"
    );
    let b = &r["ingest_phases"];
    let wall = b["index_ready_at"].as_f64().unwrap() - b["upload_started_at"].as_f64().unwrap();
    assert!((wall - r["total_ingest_time"].as_f64().unwrap()).abs() < 1e-3);

    // An engine that does not report phases gets none of the new keys.
    let legacy = build_upload_result_json(
        "qdrant-x",
        "ds",
        &stats_with(None, 10.0, 18.0),
        None,
        &json!({}),
    );
    for key in [
        "read_time",
        "index_wait_time",
        "total_ingest_time",
        "upload_docs_per_sec",
        "ingest_docs_per_sec",
        "ingest_phases",
    ] {
        assert!(legacy["results"].get(key).is_none(), "{key}");
    }
    check_upload_phases(&stats_with(None, 10.0, 18.0)).unwrap();
}

#[test]
fn phases_that_disagree_with_the_legacy_keys_are_not_published() {
    // upload_time measured separately from the phase clock.
    assert!(check_upload_phases(&stats_with(Some(phases()), 9.0, 18.0))
        .unwrap_err()
        .contains("upload_time"));
    // total_time that is not read + upload + index wait.
    assert!(check_upload_phases(&stats_with(Some(phases()), 10.0, 16.0))
        .unwrap_err()
        .contains("total_time"));
}

#[test]
fn upload_file_names_the_shared_index_and_the_workloads_it_serves() {
    let mut doc = json!({"params": {}});
    let p: WorkloadPlan = plan(
        "redis",
        &[
            None,
            Some(Workload::FilterOnly),
            Some(Workload::Vector),
            Some(Workload::Fulltext),
        ],
        false,
    )
    .unwrap();
    stamp_shared_index(&mut doc, &p);
    assert_eq!(doc["params"]["index_layout"], "full");
    assert_eq!(
        doc["params"]["index_workloads"],
        json!(["vector", "filter-only", "fulltext"])
    );
    let mut doc = json!({"params": {}});
    stamp_shared_index(&mut doc, &plan("redis", &[None], false).unwrap());
    assert_eq!(doc["params"]["index_layout"], "vector");
    assert_eq!(doc["params"]["index_workloads"], json!(["vector"]));
}

#[test]
fn search_file_records_the_index_layout_it_ran_on() {
    let params: SearchParams = serde_json::from_value(json!({"parallel": 1, "top": 3})).unwrap();
    let build = |layout: Option<IndexLayout>| {
        let results = SearchResults {
            top: 3,
            workload: Some(Workload::Vector),
            index_layout: layout,
            ..Default::default()
        };
        build_search_result_json(
            "test",
            "fixture",
            &params,
            &results,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            &json!({}),
        )
    };
    assert_eq!(
        build(Some(IndexLayout::Full))["params"]["index_layout"],
        "full"
    );
    assert_eq!(
        build(Some(IndexLayout::NoVector))["params"]["index_layout"],
        "no-vector"
    );
    assert!(build(None)["params"].get("index_layout").is_none());
}
