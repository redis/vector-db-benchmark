//! Cloud-backed CLI smoke test. Set AZURE_COSMOS_ENDPOINT and AZURE_COSMOS_KEY
//! and run `make integration-test-cosmosdb`.

mod common;

use std::fs;

use serde_json::json;

#[test]
fn flat_search_matches_the_fixture_and_cleans_up_the_container() {
    if std::env::var_os("AZURE_COSMOS_ENDPOINT").is_none() {
        eprintln!("SKIP integration_cosmosdb: set AZURE_COSMOS_ENDPOINT and AZURE_COSMOS_KEY");
        return;
    }

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
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
        "upload_params": { "parallel": 4 },
        "search_params": [{ "parallel": 2, "top": 1 }],
    }]);
    fs::write(
        root.join("experiments/configurations/cosmosdb.json"),
        config.to_string(),
    )
    .unwrap();

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
