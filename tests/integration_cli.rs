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

/// #344: a workload the engine has no arm for — or a config mixing index
/// layouts — is refused at startup, before any download or connection, and is
/// never run as the vector workload instead.
#[test]
fn unimplemented_or_mixed_workloads_fail_before_download_or_connection() {
    use serde_json::json;
    let cases = [
        (
            "redis",
            json!([{"workload":"fulltext"}]),
            false,
            "not implemented",
        ),
        (
            "opensearch",
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
        (
            "redis",
            json!([{}, {"workload":"filter-only"}]),
            false,
            "#344",
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
