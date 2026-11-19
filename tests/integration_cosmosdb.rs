//! Cloud-backed CLI smoke test. Set AZURE_COSMOS_ENDPOINT and AZURE_COSMOS_KEY
//! and run `make integration-test-cosmosdb`.

mod common;

use std::fs;

use serde_json::json;

/// Write the random-100 fixture and one Flat engine config under `root`.
/// The engine name is unique to the tempdir, so the test owns its container.
fn write_fixture(
    root: &std::path::Path,
    collection_params: serde_json::Value,
    search_params: serde_json::Value,
) -> String {
    let engine_name = format!(
        "cosmosdb-it-{}",
        root.file_name().unwrap().to_string_lossy()
    );
    let fixture = root.join("datasets/random-100");
    fs::create_dir_all(&fixture).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/random-100");
    for file in ["vectors.jsonl", "queries.jsonl", "neighbours.jsonl"] {
        fs::copy(source.join(file), fixture.join(file)).unwrap();
    }
    let datasets = json!([{
        "name": "random-100",
        "type": "jsonl",
        "path": "random-100/",
        "distance": "cosine",
        "vector_size": 100,
        "vector_count": 100,
    }]);
    fs::write(root.join("datasets/datasets.json"), datasets.to_string()).unwrap();
    let config = json!([{
        "name": engine_name,
        "engine": "cosmosdb",
        "algorithm": "flat",
        "collection_params": collection_params,
        "upload_params": { "parallel": 4 },
        "search_params": [search_params],
    }]);
    fs::write(
        root.join("experiments/configurations/cosmosdb.json"),
        config.to_string(),
    )
    .unwrap();
    engine_name
}

#[test]
fn flat_search_matches_the_fixture_and_cleans_up_the_container() {
    if std::env::var_os("AZURE_COSMOS_ENDPOINT").is_none() {
        eprintln!("SKIP integration_cosmosdb: set AZURE_COSMOS_ENDPOINT and AZURE_COSMOS_KEY");
        return;
    }

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let engine_name = write_fixture(root, json!({}), json!({ "parallel": 2, "top": 1 }));

    assert!(common::run_binary(
        root,
        &engine_name,
        "random-100",
        "localhost",
        &[]
    ));
    let uploaded = common::read_result_doc(root, &engine_name);
    let recall = uploaded["results"]["mean_recall"].as_f64().unwrap();
    println!("Cosmos DB Flat random-100 recall@1: {recall}");
    assert_eq!(recall, 1.0);
    assert_eq!(uploaded["results"]["failed_queries"], 0);
    let diagnostics = &uploaded["results"]["cosmosdb_search_diagnostics"];
    assert!(diagnostics["request_charge_ru"].as_f64().unwrap() > 0.0);
    assert!(
        diagnostics["request_attempts"].as_u64().unwrap()
            >= uploaded["results"]["succeeded_queries"].as_u64().unwrap()
    );
    assert_eq!(diagnostics["queries_without_request_diagnostics"], 0);
    let metadata = &uploaded["server_metadata"]["after"]["cosmosdb"];
    assert_eq!(
        metadata["vector_indexing_policy"]["vectorIndexes"][0]["type"],
        "flat"
    );
    assert_eq!(
        metadata["vector_embedding_policy"]["vectorEmbeddings"][0]["dimensions"],
        100
    );

    // The normal run cleaned up the benchmark container. A second upload with
    // the same name can therefore create a fresh corpus.
    assert!(common::run_binary(
        root,
        &engine_name,
        "random-100",
        "localhost",
        &[]
    ));
}

/// Feed-range fan-out on a container with several physical partitions. It
/// provisions 12,000 manual RU/s for the duration of the run, so it needs its
/// own opt-in on top of the account credentials.
#[test]
fn flat_fanout_matches_the_stock_query_across_feed_ranges() {
    if std::env::var_os("AZURE_COSMOS_ENDPOINT").is_none()
        || std::env::var_os("AZURE_COSMOS_TEST_FANOUT").is_none()
    {
        eprintln!(
            "SKIP integration_cosmosdb fan-out: set AZURE_COSMOS_ENDPOINT, AZURE_COSMOS_KEY and AZURE_COSMOS_TEST_FANOUT=1"
        );
        return;
    }

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let engine_name = write_fixture(
        root,
        json!({ "ru_per_second": 12000 }),
        json!({
            "parallel": 2,
            "top": 1,
            "search_params": { "partition_concurrency": "auto" },
        }),
    );

    assert!(common::run_binary(
        root,
        &engine_name,
        "random-100",
        "localhost",
        &[]
    ));
    let result = common::read_result_doc(root, &engine_name);
    assert_eq!(result["results"]["mean_recall"].as_f64().unwrap(), 1.0);
    assert_eq!(result["results"]["failed_queries"], 0);
    let fanout = &result["results"]["cosmosdb_search_diagnostics"]["partition_fanout"];
    let feed_ranges = fanout["feed_ranges"].as_u64().unwrap();
    println!("Cosmos DB Flat fan-out over {feed_ranges} feed ranges: {fanout}");
    assert!(
        feed_ranges >= 2,
        "the container has one feed range, so fan-out was not exercised"
    );
    assert_eq!(fanout["per_query_concurrency"], feed_ranges);
    // Flat is exact, so both routes must return the same IDs.
    assert!(fanout["parity"]["samples"].as_u64().unwrap() > 0);
    assert_eq!(fanout["parity"]["stock_repeat_missing_ids"], 0);
    assert_eq!(fanout["parity"]["cross_mode_missing_ids"], 0);
}
