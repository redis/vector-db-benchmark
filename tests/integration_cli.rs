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
