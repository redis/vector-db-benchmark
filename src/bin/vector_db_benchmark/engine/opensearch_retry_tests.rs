//! Exercise the complete scalar search path against a scripted HTTP peer.
use super::*;
use serde_json::json;
use std::io::{BufRead, Read, Write};
use std::net::TcpListener;

#[test]
fn scalar_search_reports_measured_retries_but_not_priming_retries() {
    let _recorder_lock = crate::effective_config::test_lock();
    for retry_measured in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let good = json!({"timed_out":false,"_shards":{"failed":0},
            "hits":{"total":{"value":1,"relation":"eq"},"hits":[{"_id":"one"}]}});
        let mapping = json!({"retry-test":{"mappings":{"properties":{
            "vector":{"type":"float","index":false,"doc_values":false},
            "color":{"type":"keyword"}}}}});
        let settings = json!({"retry-test":{"settings":{"index":{"knn":"false"}}}});
        let mut script = vec![
            ("/_mapping", 200, mapping),
            ("/_settings", 200, settings),
            // Priming retries must not be reported as measured retries.
            ("/_search", 429, json!({"error":"busy"})),
            ("/_search", 200, good.clone()),
        ];
        if retry_measured {
            script.push(("/_search", 429, json!({"error":"busy"})));
        }
        // Two measured successes, at most one needing a retry.
        script.push(("/_search", 200, good.clone()));
        script.push(("/_search", 200, good));
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + std::time::Duration::from_secs(15);
            for (suffix, status, body) in script {
                let mut socket = loop {
                    match listener.accept() {
                        Ok((socket, _)) => break socket,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "missing mock request {suffix}");
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                        Err(e) => panic!("mock accept: {e}"),
                    }
                };
                socket
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut reader = std::io::BufReader::new(&mut socket);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert!(
                    line.split_whitespace().nth(1).unwrap().ends_with(suffix),
                    "{line}"
                );
                let mut length = 0;
                loop {
                    line.clear();
                    assert!(reader.read_line(&mut line).unwrap() > 0);
                    if line == "\r\n" {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse::<usize>().unwrap();
                        }
                    }
                }
                let mut request_body = vec![0; length];
                reader.read_exact(&mut request_body).unwrap();
                if suffix == "/_search" {
                    let body: serde_json::Value = serde_json::from_slice(&request_body).unwrap();
                    assert!(body["query"]["bool"]["filter"].is_array());
                }
                let body = body.to_string();
                write!(socket, "HTTP/1.1 {status} Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
        });
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tests.jsonl"),
            json!({
                "query":[1.0], "closest_ids":[],
                "conditions":{"and":[{"color":{"match":{"value":"red"}}}]}
            })
            .to_string(),
        )
        .unwrap();
        let dataset = Dataset::new(
            serde_json::from_value(json!({
                "name":"retry-fixture","path":dir.path(),"type":"tar",
                "distance":"l2","vector_size":1,"schema":{"color":"keyword"}
            }))
            .unwrap(),
        );
        let config = serde_json::from_value(
            json!({"name":"retry-test","engine":"opensearch","skip_vector_index":true}),
        )
        .unwrap();
        let mut engine = OpenSearchEngine::new(&config, "http://127.0.0.1").unwrap();
        engine.base_url = url.clone();
        engine.client = Arc::new(create_os_client(&url, 5).unwrap());
        engine.timeout = 5;
        engine.index_name = "retry-test".into();
        // This fixture only exercises query timing; schema is still checked
        // against the peer above, while corpus validation is covered separately.
        engine.scalar_metadata_validated = true;
        let params = serde_json::from_value(json!({"parallel":1,"top":1})).unwrap();
        let results = engine.search(&dataset, &params, 2).unwrap();
        server.join().unwrap();
        assert_eq!(results.requested_queries, 2);
        assert_eq!(results.failed_queries, 0);
        let stats = results.filter_only.as_ref().unwrap();
        assert_eq!(stats.retried_queries, u64::from(retry_measured));
        assert_eq!(stats.matched_documents_sum, 2);
        let encoded = serde_json::to_value(stats).unwrap();
        assert_eq!(encoded["retried_queries"], u64::from(retry_measured));
        if retry_measured {
            // Default first retry sleeps at least 25ms. Removing backoff from
            // scalar latency must fail, even though the query itself is fast.
            assert!(
                results.max_time >= 0.025,
                "backoff missing: {}",
                results.max_time
            );
        }
    }
}
