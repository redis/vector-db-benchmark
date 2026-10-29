//! CLI smoke tests — the exact commands the docker-build validation job runs
//! (`--help`, `--describe datasets`, `--describe engines`), promoted into the
//! blocking test suite so a regression fails CI without waiting on the
//! non-blocking docker-build job. No engine/container required.

use std::process::Command;

mod common;

/// Run the built binary with `args`, returning (success, stdout, stderr).
fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(common::binary_path())
        .args(args)
        .output()
        .expect("failed to spawn benchmark binary");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// #361 review (#219 rule): an `INDEXING_WAIT_SECS` that is set but does not
/// parse refuses the run at startup, before any download or connection,
/// instead of running with the computed budget it silently replaced.
#[test]
fn an_unparsable_indexing_wait_is_refused_before_download_or_connection() {
    use serde_json::json;
    let root = tempfile::tempdir().unwrap();
    let datasets = root.path().join("datasets");
    std::fs::create_dir(&datasets).unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port().to_string();
    drop(listener);
    std::fs::write(
        datasets.join("datasets.json"),
        json!([{
            "name":"wait-fixture", "type":"tar", "path":"missing-corpus",
            "vector_size":1,"distance":"l2",
            "link":format!("http://127.0.0.1:{port}/must-not-download.tar.gz")
        }])
        .to_string(),
    )
    .unwrap();
    let engines_file = root.path().join("engines.json");
    std::fs::write(
        &engines_file,
        json!([{"name":"wait-cfg","engine":"redis","search_params":[{}]}]).to_string(),
    )
    .unwrap();
    for bad in ["100000s", "-1"] {
        let output = Command::new(common::binary_path())
            .current_dir(root.path())
            .args([
                "--engines",
                "wait-cfg",
                "--datasets",
                "wait-fixture",
                "--host",
                "127.0.0.1",
                "--engines-file",
            ])
            .arg(&engines_file)
            .env("REDIS_PORT", &port)
            .env("INDEXING_WAIT_SECS", bad)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!output.status.success(), "{bad}: {stdout}");
        assert!(
            stderr.contains("INDEXING_WAIT_SECS") && stderr.contains(bad),
            "{bad}: {stderr}"
        );
        assert!(!stdout.contains("Experiment stage:"), "{bad}: {stdout}");
        assert!(!datasets.join("missing-corpus").exists());
    }
}

#[test]
fn help_exits_zero_and_lists_key_flags() {
    let (ok, stdout, _stderr) = run(&["--help"]);
    assert!(ok, "--help should exit 0");
    // clap renders usage + the flags the docker smoke and users rely on.
    assert!(stdout.contains("Usage"), "help missing Usage: {stdout}");
    for flag in ["--engines", "--datasets", "--describe"] {
        assert!(stdout.contains(flag), "help missing {flag}");
    }
}

#[test]
fn describe_datasets_exits_zero_and_lists_a_known_dataset() {
    let (ok, stdout, stderr) = run(&["--describe", "datasets"]);
    assert!(ok, "--describe datasets should exit 0; stderr={stderr}");
    // random-100 ships in the image and is what the docker smoke benchmarks.
    assert!(
        stdout.contains("random-100"),
        "expected random-100 in --describe datasets output:\n{stdout}"
    );
}

#[test]
fn describe_engines_exits_zero_and_lists_known_engines() {
    let (ok, stdout, stderr) = run(&["--describe", "engines"]);
    assert!(ok, "--describe engines should exit 0; stderr={stderr}");
    assert!(
        stdout.contains("Available engines"),
        "missing header:\n{stdout}"
    );
    // pin a couple of Redis-family engines incl. the newest (dragonfly).
    for engine in ["redis", "dragonfly"] {
        assert!(
            stdout.contains(engine),
            "expected engine '{engine}' in --describe engines output:\n{stdout}"
        );
    }
}

#[test]
fn describe_unknown_option_errors_out() {
    let (ok, _stdout, stderr) = run(&["--describe", "bogus"]);
    assert!(!ok, "--describe bogus should exit non-zero");
    assert!(
        stderr.contains("Unknown describe option"),
        "expected a helpful error on stderr:\n{stderr}"
    );
}

/// Registry selection must fail before touching a missing corpus or a server.
/// Exercise distinct configs whose differences the old first-config retain lost.
#[test]
fn scalar_duplicate_configs_fail_before_download_or_connection() {
    use serde_json::json;
    for (engine, first, second) in [
        (
            "redis",
            json!({"search_params":[{"top":3}]}),
            json!({"search_params":[{"top":10}]}),
        ),
        (
            "opensearch",
            json!({"collection_params":{"number_of_shards":1}}),
            json!({"collection_params":{"number_of_shards":4}}),
        ),
    ] {
        let root = tempfile::tempdir().unwrap();
        let datasets = root.path().join("datasets");
        std::fs::create_dir(&datasets).unwrap();
        // A closed ephemeral loopback endpoint makes a broken guard fail quickly
        // instead of contacting a real server or downloading a real dataset.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port().to_string();
        drop(listener);
        std::fs::write(
            datasets.join("datasets.json"),
            json!([{
                "name":"scalar-selection-fixture", "type":"tar", "path":"missing-corpus",
                "vector_size":1,"distance":"l2","schema":{"tag":"keyword"},
                "link":format!("http://127.0.0.1:{port}/must-not-download.tar.gz")
            }])
            .to_string(),
        )
        .unwrap();
        let configs: Vec<_> = [first, second]
            .into_iter()
            .enumerate()
            .map(|(i, mut config)| {
                config["name"] = json!(format!("candidate-{i}"));
                config["engine"] = json!(engine);
                config
            })
            .collect();
        let engines_file = root.path().join("engines.json");
        std::fs::write(&engines_file, json!(configs).to_string()).unwrap();
        let output = Command::new(common::binary_path())
            .current_dir(root.path())
            .args([
                "--skip-vector-index",
                "--engines",
                "candidate-*",
                "--datasets",
                "scalar-selection-fixture",
                "--host",
                "127.0.0.1",
                "--engines-file",
            ])
            .arg(&engines_file)
            .env("REDIS_PORT", &port)
            .env("OPENSEARCH_PORT", &port)
            .env("OPENSEARCH_TIMEOUT", "1")
            .env("OPENSEARCH_INDEX_OP_MAX_RETRIES", "0")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!output.status.success());
        for detail in [
            "one config per engine type",
            engine,
            "candidate-0",
            "candidate-1",
        ] {
            assert!(
                stderr.contains(detail),
                "{engine}: missing {detail:?}: {stderr} / {stdout}"
            );
        }
        assert!(!stdout.contains("Experiment stage:"), "{stdout}");
        assert!(!stdout.contains("Downloading"), "{stdout}");
        assert!(!datasets.join("missing-corpus").exists());
    }
}

/// Ignored adapters must not participate in scalar config or namespace guards.
#[test]
fn scalar_unsupported_duplicates_are_skipped_before_selection_and_exact_collision_guards() {
    use serde_json::json;
    let root = tempfile::tempdir().unwrap();
    let datasets = root.path().join("datasets");
    std::fs::create_dir(&datasets).unwrap();
    std::fs::write(
        datasets.join("datasets.json"),
        json!([{
            "name":"scalar-selection-fixture", "type":"tar", "path":"missing-corpus",
            "vector_size":1,"distance":"l2","schema":{"tag":"keyword"}
        }])
        .to_string(),
    )
    .unwrap();
    let configs = json!([
        {"name":"candidate-redis","engine":"redis"},
        {"name":"candidate-qdrant-1","engine":"qdrant"},
        {"name":"candidate-qdrant-2","engine":"qdrant"},
        {"name":"candidate-vectorsets-1","engine":"vectorsets"},
        {"name":"candidate-vectorsets-2","engine":"vectorsets"}
    ]);
    let engines_file = root.path().join("engines.json");
    std::fs::write(&engines_file, configs.to_string()).unwrap();
    // A closed loopback port proves selection reaches Redis without a server.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port().to_string();
    drop(listener);
    let output = Command::new(common::binary_path())
        .current_dir(root.path())
        .args([
            "--skip-vector-index",
            "--engines",
            "candidate-*",
            "--datasets",
            "scalar-selection-fixture",
            "--host",
            "127.0.0.1",
            "--engines-file",
        ])
        .arg(&engines_file)
        .env("REDIS_PORT", port)
        .env("VECTORSETS_INDEX_NAME_EXACT", "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "closed Redis port must fail");
    assert!(
        stdout.contains("Found 1 datasets, 1 engines"),
        "{stdout} / {stderr}"
    );
    assert!(
        stdout.contains("Experiment stage: Configure"),
        "{stdout} / {stderr}"
    );
    for engine in ["qdrant", "vectorsets"] {
        assert!(
            stderr.contains(&format!("not implemented for engine '{engine}', skipping")),
            "{stderr}"
        );
    }
    assert!(!stderr.contains("one config per engine type"), "{stderr}");
    assert!(!stderr.contains("same index namespace"), "{stderr}");
}

/// #344: a workload the engine has no arm for is refused at startup, before
/// any download or connection, and is never run as the vector workload instead.
#[test]
fn unimplemented_or_mixed_workloads_fail_before_download_or_connection() {
    use serde_json::json;
    let cases = [
        (
            "valkey",
            json!([{"workload":"fulltext"}]),
            false,
            "not implemented",
        ),
        // A config mixing filter-only with vector-index workloads is no longer
        // refused on Redis/OpenSearch: it builds the shared `full` index (#344,
        // covered by the engines' integration suites). Elsewhere filter-only is
        // not implemented, which is refused first.
        // #339: a `fulltext` block is refused on another workload, and a
        // misspelled key or a non-positive boost is refused before connecting.
        (
            "redis",
            json!([{"fulltext":{"fields":{"title":2.0}}}]),
            false,
            "only valid on a fulltext search config",
        ),
        (
            "redis",
            json!([{"workload":"fulltext","fulltext":{"feilds":{"title":2.0}}}]),
            false,
            "invalid `fulltext` block",
        ),
        (
            "opensearch",
            json!([{"workload":"fulltext","fulltext":{"fields":{"title":0}}}]),
            false,
            "finite number > 0",
        ),
        // #339 review: an OpenSearch-only key on another engine is refused,
        // not accepted and ignored.
        (
            "redis",
            json!([{"workload":"fulltext","fulltext":{"allow_multi_shard":true}}]),
            false,
            "only applies to OpenSearch",
        ),
        // Hybrid exists on Redis and OpenSearch (#341), not on Valkey.
        (
            "valkey",
            json!([{"workload":"hybrid"}]),
            false,
            "not implemented",
        ),
        (
            "qdrant",
            json!([{"workload":"filter-only"}]),
            false,
            "not implemented",
        ),
        // Legacy filter-only harnesses stay flag-only, as on master.
        (
            "valkey",
            json!([{"workload":"filter-only"}]),
            false,
            "not implemented",
        ),
        (
            "mongodb",
            json!([{"workload":"filter-only"}]),
            false,
            "not implemented",
        ),
        (
            "redis",
            json!([{"workload":"vector"}]),
            true,
            "--skip-vector-index",
        ),
    ];
    for (engine, search_params, skip_vector_index, detail) in cases {
        let root = tempfile::tempdir().unwrap();
        let datasets = root.path().join("datasets");
        std::fs::create_dir(&datasets).unwrap();
        // A closed loopback endpoint: a broken guard fails fast instead of
        // reaching a real server or downloading a real dataset.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port().to_string();
        drop(listener);
        std::fs::write(
            datasets.join("datasets.json"),
            json!([{
                "name":"workload-fixture", "type":"tar", "path":"missing-corpus",
                "vector_size":1,"distance":"l2","schema":{"tag":"keyword"},
                "link":format!("http://127.0.0.1:{port}/must-not-download.tar.gz")
            }])
            .to_string(),
        )
        .unwrap();
        let engines_file = root.path().join("engines.json");
        std::fs::write(
            &engines_file,
            json!([{"name":"wl","engine":engine,"search_params":search_params}]).to_string(),
        )
        .unwrap();
        let mut cmd = Command::new(common::binary_path());
        cmd.current_dir(root.path())
            .args([
                "--engines",
                "wl",
                "--datasets",
                "workload-fixture",
                "--host",
                "127.0.0.1",
                "--engines-file",
            ])
            .arg(&engines_file)
            .env("REDIS_PORT", &port)
            .env("OPENSEARCH_PORT", &port)
            .env("QDRANT_PORT", &port);
        if skip_vector_index {
            cmd.arg("--skip-vector-index");
        }
        let output = cmd.output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            !output.status.success(),
            "{engine} {search_params}: {stdout}"
        );
        assert!(
            stderr.contains(detail),
            "{engine} {search_params}: missing {detail:?}: {stderr}"
        );
        assert!(!stdout.contains("Experiment stage:"), "{stdout}");
        assert!(!datasets.join("missing-corpus").exists());
    }
}

/// #343 acceptance through the real subcommand: a dump compared with itself
/// passes with perfect agreement; a candidate that dropped a query the
/// reference answered exits non-zero and names it.
#[test]
fn compare_results_self_passes_and_a_one_sided_answer_fails() {
    use vector_db_benchmark::results_dump::{
        write_dump, DumpHeader, IdSpace, QueryRecord, DUMP_SCHEMA_VERSION,
    };
    use vector_db_benchmark::workload::Workload;
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, engine: &str, drop_query_1: bool| {
        let header = DumpHeader {
            schema_version: DUMP_SCHEMA_VERSION,
            engine: engine.into(),
            config: format!("{engine}-cfg"),
            dataset: "ds".into(),
            workload: Workload::Fulltext,
            parallel: 4,
            top: 3,
            search_id: 0,
            id_space: IdSpace::Field {
                field: "docid".into(),
            },
            index_layout: None,
        };
        let records: Vec<QueryRecord> = (0..3)
            .map(|q| QueryRecord {
                query_id: q,
                seq: q,
                shape: None,
                ids: if drop_query_1 && q == 1 {
                    vec![]
                } else {
                    vec!["d1".into(), "d2".into(), "d3".into()]
                },
                scores: None,
                total: None,
                latency: 0.001,
                error: (drop_query_1 && q == 1).then(|| "timeout".to_string()),
                expected: None,
            })
            .collect();
        let path = dir.path().join(name);
        write_dump(std::fs::File::create(&path).unwrap(), &header, &records).unwrap();
        path
    };
    let reference = write("os.jsonl", "opensearch", false);
    let candidate = write("redis.jsonl", "redis", true);
    let compare = |a: &std::path::Path, b: &std::path::Path| {
        Command::new(common::binary_path())
            .arg("compare-results")
            .arg(a)
            .arg(b)
            .args(["--reference", "opensearch"])
            .output()
            .unwrap()
    };
    let out = compare(&reference, &reference);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{stdout}");
    assert!(
        stdout.contains("Reference engine: **opensearch**"),
        "{stdout}"
    );
    assert!(stdout.contains("| 1.0000 | 1.0000 | 1.0000 |"), "{stdout}");

    // Reference given second: still the reference, and the drop fails the run.
    let out = compare(&candidate, &reference);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stdout}");
    assert!(
        stdout.contains("query 1 answered only by the reference"),
        "{stdout}"
    );
    assert!(stderr.contains("answered by one engine"), "{stderr}");
}

/// #343 review: a non-empty `--dump-results` directory is refused at startup,
/// before any download or connection, instead of after the whole timed run.
#[test]
fn a_non_empty_dump_dir_is_refused_before_download_or_connection() {
    use serde_json::json;
    let root = tempfile::tempdir().unwrap();
    let datasets = root.path().join("datasets");
    std::fs::create_dir(&datasets).unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port().to_string();
    drop(listener);
    std::fs::write(
        datasets.join("datasets.json"),
        json!([{
            "name":"dump-fixture", "type":"tar", "path":"missing-corpus",
            "vector_size":1,"distance":"l2","schema":{"tag":"keyword"},
            "link":format!("http://127.0.0.1:{port}/must-not-download.tar.gz")
        }])
        .to_string(),
    )
    .unwrap();
    let engines_file = root.path().join("engines.json");
    std::fs::write(
        &engines_file,
        json!([{"name":"dump-cfg","engine":"redis","search_params":[{}]}]).to_string(),
    )
    .unwrap();
    let dump = root.path().join("dumps");
    std::fs::create_dir(&dump).unwrap();
    std::fs::write(dump.join("previous-run.jsonl"), "{}").unwrap();
    let output = Command::new(common::binary_path())
        .current_dir(root.path())
        .args([
            "--engines",
            "dump-cfg",
            "--datasets",
            "dump-fixture",
            "--host",
            "127.0.0.1",
            "--engines-file",
        ])
        .arg(&engines_file)
        .arg("--dump-results")
        .arg(&dump)
        .env("REDIS_PORT", &port)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!output.status.success(), "{stdout}");
    assert!(stderr.contains("is not empty"), "{stderr}");
    assert!(!stdout.contains("Experiment stage:"), "{stdout}");
    assert!(!datasets.join("missing-corpus").exists());
}

/// #343 review: `--dump-results` for an engine/workload without per-query
/// capture is refused at startup — before any download, connection, upload or
/// timed search — not after the first timed search point.
#[test]
fn dump_results_without_capture_is_refused_before_download_or_connection() {
    use serde_json::json;
    for (engine, skip_vector_index) in [("qdrant", false), ("valkey", true)] {
        let root = tempfile::tempdir().unwrap();
        let datasets = root.path().join("datasets");
        std::fs::create_dir(&datasets).unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port().to_string();
        drop(listener);
        std::fs::write(
            datasets.join("datasets.json"),
            json!([{
                "name":"cap-fixture", "type":"tar", "path":"missing-corpus",
                "vector_size":1,"distance":"l2","schema":{"tag":"keyword"},
                "link":format!("http://127.0.0.1:{port}/must-not-download.tar.gz")
            }])
            .to_string(),
        )
        .unwrap();
        let engines_file = root.path().join("engines.json");
        std::fs::write(
            &engines_file,
            json!([{"name":"cap-cfg","engine":engine,"search_params":[{}]}]).to_string(),
        )
        .unwrap();
        let mut cmd = Command::new(common::binary_path());
        cmd.current_dir(root.path())
            .args([
                "--engines",
                "cap-cfg",
                "--datasets",
                "cap-fixture",
                "--host",
                "127.0.0.1",
                "--engines-file",
            ])
            .arg(&engines_file)
            .arg("--dump-results")
            .arg(root.path().join("dumps"))
            .env("QDRANT_PORT", &port)
            .env("REDIS_PORT", &port);
        if skip_vector_index {
            cmd.arg("--skip-vector-index");
        }
        let output = cmd.output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!output.status.success(), "{engine}: {stdout}");
        assert!(
            stderr.contains("does not capture per-query results"),
            "{engine}: {stderr}"
        );
        assert!(!stdout.contains("Experiment stage:"), "{stdout}");
        assert!(!datasets.join("missing-corpus").exists());
    }
}

/// #344 review: `calibration_param` on a DECLARED filter-only config is
/// refused at startup instead of silently ignored.
#[test]
fn calibration_on_declared_filter_only_is_refused_at_startup() {
    use serde_json::json;
    let root = tempfile::tempdir().unwrap();
    let datasets = root.path().join("datasets");
    std::fs::create_dir(&datasets).unwrap();
    std::fs::write(
        datasets.join("datasets.json"),
        json!([{
            "name":"cal-fixture", "type":"tar", "path":"missing-corpus",
            "vector_size":1,"distance":"l2","schema":{"tag":"keyword"},
            "link":"http://127.0.0.1:9/must-not-download.tar.gz"
        }])
        .to_string(),
    )
    .unwrap();
    let engines_file = root.path().join("engines.json");
    std::fs::write(
        &engines_file,
        json!([{"name":"cal-cfg","engine":"redis","search_params":[
            {"workload":"filter-only","calibration_param":"ef","calibration_precision":0.9}
        ]}])
        .to_string(),
    )
    .unwrap();
    let output = Command::new(common::binary_path())
        .current_dir(root.path())
        .args([
            "--engines",
            "cal-cfg",
            "--datasets",
            "cal-fixture",
            "--host",
            "127.0.0.1",
            "--engines-file",
        ])
        .arg(&engines_file)
        .env("REDIS_PORT", "9")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!output.status.success(), "{stdout}");
    assert!(stderr.contains("calibration_param"), "{stderr}");
    assert!(!stdout.contains("Experiment stage:"), "{stdout}");
}

/// #344 review: `--update-search-ratio` with a non-vector search config in the
/// engine config — filter-only on the shared `full` index, or fulltext next to
/// vector — is refused at startup (closed port, undownloadable corpus), not
/// when the mixed phase reaches that config, which used to discard every
/// completed pure-search result. `0:S` (pure search, no mixed phase) is not
/// refused by this check.
#[test]
fn update_search_ratio_with_a_non_vector_search_config_fails_before_connecting() {
    use serde_json::json;
    let root = tempfile::tempdir().unwrap();
    let datasets = root.path().join("datasets");
    std::fs::create_dir(&datasets).unwrap();
    std::fs::write(
        datasets.join("datasets.json"),
        json!([{
            "name":"usr-fixture", "type":"tar", "path":"missing-corpus",
            "vector_size":1,"distance":"l2","schema":{"tag":"keyword","title":"text"},
            "link":"http://127.0.0.1:9/must-not-download.tar.gz"
        }])
        .to_string(),
    )
    .unwrap();
    let engines_file = root.path().join("engines.json");
    std::fs::write(
        &engines_file,
        json!([
            {"name":"usr-full","engine":"redis","search_params":[
                {"parallel":1},{"parallel":1,"workload":"filter-only"}]},
            {"name":"usr-text","engine":"opensearch","search_params":[
                {"parallel":1},{"parallel":1,"workload":"fulltext"}]},
            {"name":"usr-hyb","engine":"redis","search_params":[
                {"parallel":1},{"parallel":1,"top":10,"workload":"hybrid"}]},
            {"name":"usr-hyb-full","engine":"opensearch","search_params":[
                {"parallel":1,"top":10,"workload":"hybrid"},
                {"parallel":1,"workload":"filter-only"}]}
        ])
        .to_string(),
    )
    .unwrap();
    let run = |config: &str, ratio: &str| {
        Command::new(common::binary_path())
            .current_dir(root.path())
            .args([
                "--engines",
                config,
                "--datasets",
                "usr-fixture",
                "--host",
                "127.0.0.1",
                "--update-search-ratio",
                ratio,
                "--engines-file",
            ])
            .arg(&engines_file)
            .env("REDIS_PORT", "9")
            .env("OPENSEARCH_PORT", "9")
            .output()
            .unwrap()
    };
    // #341: hybrid beside vector (`vector` layout) and beside filter-only
    // (`full`) is refused the same way.
    for (config, index, workload) in [
        ("usr-full", 1, "filter-only"),
        ("usr-text", 1, "fulltext"),
        ("usr-hyb", 1, "hybrid"),
        ("usr-hyb-full", 0, "hybrid"),
    ] {
        let output = run(config, "1:10");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!output.status.success(), "{config}: {stdout}");
        assert!(
            stderr.contains(&format!(
                "config '{config}': --update-search-ratio runs the vector workload in its \
                 mixed phases, but search config #{index} is \"{workload}\""
            )),
            "{config}: {stderr}"
        );
        assert!(!stdout.contains("Experiment stage:"), "{config}: {stdout}");
        assert!(!stdout.contains("Running experiment"), "{config}: {stdout}");
    }
    // Pure search only: this check does not fire (the run then fails on the
    // missing corpus / closed port, which is not what is tested here).
    let output = run("usr-full", "0:5");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("--update-search-ratio runs"), "{stderr}");
}

/// #339 review: refusals that happen before anything touches the server (a
/// closed port: the refusal must come first), on both engines.
///
/// - An engine config mixing `fulltext` with a vector config is refused when
///   the dataset's queries filter on a schema `text` field, with ANY condition
///   type (`match.text`, `match.value`, `match.any`): the fulltext config
///   would change the analyzer those filters run through.
/// - A fulltext `fields` entry that is not a `text` field of the dataset
///   schema is refused before configure/upload.
///
/// The same mix filtering only a keyword field is not refused, and neither is
/// a fulltext-only config.
#[test]
fn fulltext_mixed_with_text_filtered_vector_configs_is_refused_before_connecting() {
    use serde_json::json;
    let mix = json!([{}, {"workload": "fulltext"}]);
    let text_only = json!([{"workload": "fulltext"}]);
    let typo = json!([{"workload": "fulltext", "fulltext": {"fields": {"bdy": 1.0}}}]);
    let on = |field: &str, cond: serde_json::Value| json!({"and": [{field: cond}]});
    let refused_mix = "filter on text fields";
    let cases = [
        (
            on("body", json!({"match": {"text": "quick"}})),
            mix.clone(),
            Some(refused_mix),
        ),
        (
            on("body", json!({"match": {"value": "quick"}})),
            mix.clone(),
            Some(refused_mix),
        ),
        (
            on("body", json!({"match": {"any": ["quick"]}})),
            mix.clone(),
            Some(refused_mix),
        ),
        (
            on("tag", json!({"match": {"value": "a"}})),
            mix.clone(),
            None,
        ),
        (
            on("body", json!({"match": {"text": "quick"}})),
            text_only.clone(),
            None,
        ),
        (
            on("tag", json!({"match": {"value": "a"}})),
            typo.clone(),
            Some("fulltext field 'bdy' is not a `text` field"),
        ),
    ];
    // The non-refused cases go on to connect; only Redis fails fast against a
    // closed port (OpenSearch retries), so they run on Redis only — the rules
    // themselves are engine-agnostic and unit-tested in `fulltext`.
    for engine in ["redis", "opensearch"] {
        for (condition, search_params, refusal) in &cases {
            if engine == "opensearch" && refusal.is_none() {
                continue;
            }
            let root = tempfile::tempdir().unwrap();
            let ds = root.path().join("datasets/tm");
            std::fs::create_dir_all(&ds).unwrap();
            std::fs::write(
                ds.join("tests.jsonl"),
                json!({"query": [1.0, 0.0], "conditions": condition, "closest_ids": [0],
                       "query_text": "quick fox"})
                .to_string(),
            )
            .unwrap();
            std::fs::write(
                root.path().join("datasets/datasets.json"),
                json!([{"name": "tm", "type": "tar", "path": "tm", "vector_size": 2,
                        "distance": "l2", "schema": {"body": "text", "tag": "keyword"}}])
                .to_string(),
            )
            .unwrap();
            let engines_file = root.path().join("engines.json");
            std::fs::write(
                &engines_file,
                json!([{"name": "mix", "engine": engine, "search_params": search_params}])
                    .to_string(),
            )
            .unwrap();
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port().to_string();
            drop(listener);
            let out = Command::new(common::binary_path())
                .current_dir(root.path())
                .args([
                    "--engines",
                    "mix",
                    "--datasets",
                    "tm",
                    "--host",
                    "127.0.0.1",
                ])
                .arg("--engines-file")
                .arg(&engines_file)
                .args(["--exit-on-error"])
                .env("REDIS_PORT", &port)
                .env("OPENSEARCH_PORT", &port)
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            let ctx = format!("{engine} {condition} {search_params}: {stderr}");
            match refusal {
                Some(needle) => {
                    assert!(!out.status.success(), "{ctx}");
                    assert!(stderr.contains(needle), "{ctx}");
                    // Refused before configure: nothing touched the server.
                    assert!(!stdout.contains("Experiment stage: Configure"), "{ctx}");
                }
                None => {
                    assert!(!stderr.contains(refused_mix), "{ctx}");
                    assert!(!stderr.contains("is not a `text` field"), "{ctx}");
                }
            }
        }
    }
}

/// #339 follow-up: parity index rules refused at startup, before any
/// download or connection (closed port, missing corpus): parity mixed with
/// english, parity next to a vector config, parity without explicit fields,
/// parity configs searching different field sets.
#[test]
fn fulltext_parity_index_rules_are_refused_at_startup() {
    use serde_json::json;
    let par = |fields: serde_json::Value| json!({"workload": "fulltext", "fulltext": {"analysis": "parity", "fields": fields}});
    let cases = [
        (
            json!([par(json!({"segment": 1})), {"workload": "fulltext"}]),
            "mixes fulltext analysis",
        ),
        (
            json!([{}, par(json!({"segment": 1}))]),
            "must never apply to other workloads",
        ),
        (
            json!([{"workload": "fulltext", "fulltext": {"analysis": "parity"}}]),
            "needs explicit",
        ),
        (
            json!([par(json!({"segment": 1})), par(json!({"title": 1}))]),
            "different field sets",
        ),
        // Redis only: parity on a shared corpus, refused before the download
        // (run with REDIS_KEY_PREFIX set, below).
        (json!([par(json!({"segment": 1}))]), "REDIS_KEY_PREFIX"),
        // Redis only: a shared prefix covering the marker keys (run with
        // REDIS_KEY_PREFIX=vdbb, below), even in english mode.
        (
            json!([{"workload": "fulltext", "fulltext": {"fields": {"segment": 1}}}]),
            "overlaps the fulltext marker keys",
        ),
    ];
    for engine in ["redis", "opensearch"] {
        for (search_params, needle) in &cases {
            let shared_prefix = if needle.contains("REDIS_KEY_PREFIX") {
                Some("shared")
            } else if needle.contains("overlaps the fulltext marker keys") {
                Some("vdbb")
            } else {
                None
            };
            if shared_prefix.is_some() && engine != "redis" {
                continue;
            }
            let root = tempfile::tempdir().unwrap();
            std::fs::create_dir(root.path().join("datasets")).unwrap();
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port().to_string();
            drop(listener);
            std::fs::write(
                root.path().join("datasets/datasets.json"),
                json!([{"name": "pf", "type": "tar", "path": "missing-corpus",
                        "vector_size": 1, "distance": "l2",
                        "schema": {"title": "text", "segment": "text"},
                        "link": format!("http://127.0.0.1:{port}/must-not-download.tgz")}])
                .to_string(),
            )
            .unwrap();
            let engines_file = root.path().join("engines.json");
            std::fs::write(
                &engines_file,
                json!([{"name": "pf", "engine": engine, "search_params": search_params}])
                    .to_string(),
            )
            .unwrap();
            let mut cmd = Command::new(common::binary_path());
            cmd.current_dir(root.path())
                .args(["--engines", "pf", "--datasets", "pf", "--host", "127.0.0.1"])
                .arg("--engines-file")
                .arg(&engines_file)
                .env("REDIS_PORT", &port)
                .env("OPENSEARCH_PORT", &port);
            if let Some(p) = shared_prefix {
                cmd.env("REDIS_KEY_PREFIX", p);
            }
            let out = cmd.output().unwrap();
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(!out.status.success(), "{engine} {search_params}: {stderr}");
            assert!(stderr.contains(needle), "{engine} {needle}: {stderr}");
            assert!(!stderr.contains("must-not-download"), "{stderr}");
        }
    }
}

/// #341: hybrid configuration errors are refused before any engine is built —
/// no server is running here (a closed loopback port), so reaching configure
/// or upload would show up as a connection error, not the expected message.
#[test]
fn hybrid_misconfiguration_fails_before_configure_or_upload() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port().to_string();
    drop(listener);
    let configs = serde_json::json!([
        {"name": "hyb-redis", "engine": "redis",
         "search_params": [{"parallel": 1, "top": 10, "workload": "hybrid",
                            "hybrid_policy": "ADHOC_BF"}]},
        {"name": "hyb-os", "engine": "opensearch",
         "search_params": [{"parallel": 1, "top": 10, "workload": "hybrid"}]}
    ]);
    let proj = common::write_msmarco_hybrid_project("hybrid-test", &configs.to_string());
    let envs = |extra: Option<(&'static str, &'static str)>| {
        let mut e = vec![
            ("REDIS_PORT", port.clone()),
            ("OPENSEARCH_PORT", port.clone()),
        ];
        if let Some((k, v)) = extra {
            e.push((k, v.to_string()));
        }
        e
    };
    let run = |engine: &str, extra| {
        let e = envs(extra);
        let e: Vec<(&str, &str)> = e.iter().map(|(k, v)| (*k, v.as_str())).collect();
        common::run_binary_text(&proj.root, engine, "hybrid-test", "127.0.0.1", &e)
    };

    // 0. --queries N would silently mean N per bucket: refused at startup.
    {
        let e = envs(None);
        let e: Vec<(&str, &str)> = e.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let mut cmd = std::process::Command::new(common::binary_path());
        cmd.args([
            "--engines",
            "hyb-os",
            "--datasets",
            "hybrid-test",
            "--host",
            "127.0.0.1",
            "--queries",
            "5",
        ])
        .current_dir(&proj.root);
        for (k, v) in &e {
            cmd.env(k, v);
        }
        let out = cmd.output().unwrap();
        let (stdout, stderr) = (
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(!out.status.success());
        assert!(stderr.contains("--queries 5"), "{stderr}");
        assert!(!stdout.contains("Running experiment"), "{stdout}");
    }
    // 0a. The CLI open-loop flags are merged into search configs only at
    // search time; Redis passes the duration-bounded engine gate, so they must
    // still be refused for hybrid at startup, not after the upload.
    // (--target-qps alone is refused earlier: it needs --search-duration.)
    for flag in [
        &["--search-duration", "60"][..],
        &["--target-qps", "100", "--search-duration", "60"][..],
    ] {
        let e = envs(None);
        let e: Vec<(&str, &str)> = e.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let mut cmd = std::process::Command::new(common::binary_path());
        cmd.args([
            "--engines",
            "hyb-redis",
            "--datasets",
            "hybrid-test",
            "--host",
            "127.0.0.1",
        ])
        .args(flag)
        .current_dir(&proj.root);
        for (k, v) in &e {
            cmd.env(k, v);
        }
        let out = cmd.output().unwrap();
        let (stdout, stderr) = (
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(!out.status.success(), "{flag:?}");
        assert!(
            stderr.contains("target_qps / duration_seconds are not supported"),
            "{flag:?}: {stderr}"
        );
        assert!(!stdout.contains("Experiment stage:"), "{flag:?}: {stdout}");
    }
    // 0b. A top wider than the narrowest filtered truth (min 10 ids here is
    // fine; 2000 is wider than any) is refused before configure.
    {
        let wide = serde_json::json!([{"name": "hyb-wide", "engine": "opensearch",
            "search_params": [{"parallel": 1, "top": 2000, "workload": "hybrid"}]}]);
        std::fs::write(
            proj.root.join("experiments/configurations/wide.json"),
            wide.to_string(),
        )
        .unwrap();
        let (ok, stdout, stderr) = run("hyb-wide", None);
        assert!(!ok);
        assert!(stderr.contains("top 2000 exceeds"), "{stderr}");
        assert!(!stdout.contains("Experiment stage:"), "{stdout}");
        std::fs::remove_file(proj.root.join("experiments/configurations/wide.json")).unwrap();
    }

    // 0c. Fulltext pins the index's text analyzer, which would change what the
    // hybrid `title` text predicates match: one engine config mixing the two
    // is refused (fulltext's mix rule, fed the hybrid query set) before
    // configure.
    {
        let mix = serde_json::json!([{"name": "hyb-ft-mix", "engine": "opensearch",
        "search_params": [
            {"parallel": 1, "top": 10, "workload": "hybrid"},
            {"parallel": 1, "top": 5, "workload": "fulltext"}
        ]}]);
        let cfg = proj.root.join("experiments/configurations/mix.json");
        std::fs::write(&cfg, mix.to_string()).unwrap();
        let (ok, stdout, stderr) = run("hyb-ft-mix", None);
        assert!(!ok);
        assert!(stderr.contains("filter on text fields"), "{stderr}");
        assert!(!stdout.contains("Experiment stage:"), "{stdout}");
        std::fs::remove_file(cfg).unwrap();
    }

    // 0d. `hybrid_policy` where nothing reads it (a Redis vector config, an
    // OpenSearch hybrid config) is refused while planning, not ignored.
    {
        let knob = serde_json::json!([
            {"name": "knob-redis-vec", "engine": "redis",
             "search_params": [{"parallel": 1, "top": 10, "hybrid_policy": "ADHOC_BF"}]},
            {"name": "knob-os-hyb", "engine": "opensearch",
             "search_params": [{"parallel": 1, "top": 10, "workload": "hybrid",
                                "hybrid_policy": "ADHOC_BF"}]}
        ]);
        let cfg = proj.root.join("experiments/configurations/knob.json");
        std::fs::write(&cfg, knob.to_string()).unwrap();
        for engine in ["knob-redis-vec", "knob-os-hyb"] {
            let (ok, stdout, stderr) = run(engine, None);
            assert!(!ok, "{engine}");
            assert!(stderr.contains("silently ignored"), "{engine}: {stderr}");
            assert!(!stdout.contains("Running experiment"), "{engine}: {stdout}");
        }
        std::fs::remove_file(cfg).unwrap();
    }

    // 1. HYBRID_POLICY conflict: refused while planning, before the loop.
    let (ok, stdout, stderr) = run("hyb-redis", Some(("REDIS_HYBRID_POLICY", "BATCHES")));
    assert!(!ok);
    assert!(stderr.contains("REDIS_HYBRID_POLICY"), "{stderr}");
    assert!(!stdout.contains("Running experiment"), "{stdout}");

    let ds = proj.root.join("datasets/hybrid-test");
    let file = ds.join(vector_db_benchmark::hybrid::HYBRID_QUERIES_FILE);
    let original = std::fs::read_to_string(&file).unwrap();

    // 2. An invalid query set (a row whose bucket contradicts its count).
    let mut lines: Vec<serde_json::Value> = original
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    lines[0]["bucket"] = serde_json::json!("0.1%");
    std::fs::write(
        &file,
        lines
            .iter()
            .map(|l| l.to_string() + "\n")
            .collect::<String>(),
    )
    .unwrap();
    for engine in ["hyb-redis", "hyb-os"] {
        let (ok, stdout, stderr) = run(engine, None);
        assert!(!ok);
        assert!(stderr.contains("declared bucket"), "{engine}: {stderr}");
        assert!(!stdout.contains("Experiment stage:"), "{engine}: {stdout}");
    }

    // 3. No query set at all (e.g. a corpus from a tarball without it).
    std::fs::remove_file(&file).unwrap();
    for engine in ["hyb-redis", "hyb-os"] {
        let (ok, stdout, stderr) = run(engine, None);
        assert!(!ok);
        assert!(
            stderr.contains("--add-hybrid-queries"),
            "{engine}: {stderr}"
        );
        assert!(!stdout.contains("Experiment stage:"), "{engine}: {stdout}");
    }
    std::fs::remove_dir_all(&proj.root).ok();
}
