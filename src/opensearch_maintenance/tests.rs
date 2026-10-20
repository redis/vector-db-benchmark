use super::*;
use opensearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use serde_json::json;
use std::io::{BufRead, Write};
use std::net::TcpListener;

fn done() -> Value {
    json!({"completed":true,"response":{"_shards":{"total":1,"successful":1,"failed":0}}})
}

#[test]
fn completion_requires_all_shards_and_no_task_error() {
    assert_eq!(task_complete(&json!({"completed":false})), Ok(false));
    assert_eq!(task_complete(&done()), Ok(true));
    for body in [
        json!({}),
        json!({"completed":"true"}),
        json!({"completed":true}),
        json!({"completed":false,"error":{"reason":"failed"}}),
        json!({"completed":true,"error":{"reason":"failed"}}),
        json!({"completed":true,"response":{"error":"failed","_shards":{"total":1,"successful":1,"failed":0}}}),
        json!({"completed":true,"response":{"_shards":{"total":1,"successful":0,"failed":1}}}),
        json!({"completed":true,"response":{"_shards":{"total":2,"successful":1,"failed":0}}}),
        json!({"completed":true,"response":{"_shards":{"total":0,"successful":0,"failed":0}}}),
    ] {
        assert!(task_complete(&body).is_err(), "{body}");
    }
}

#[test]
fn resolved_options_preserve_budget_and_cap_requests() {
    for budget in [None, Some(Duration::from_secs(7_200))] {
        for (ordinary, merge, expected) in [(300, 3_600, 300), (300, 120, 120)] {
            let options = MergeOptions::from_resolved(
                Duration::from_secs(ordinary),
                Duration::from_secs(merge),
                budget,
                7,
                Duration::from_millis(123),
            );
            assert_eq!(options.budget, budget);
            assert_eq!(options.http_deadline, Duration::from_secs(expected));
            assert_eq!(options.max_retries, 7);
            assert_eq!(options.retry_base, Duration::from_millis(123));
            assert_eq!(options.poll_interval, Duration::from_secs(2));
        }
    }
}

struct Reply {
    method: &'static str,
    status: u16,
    body: Value,
    delay: Duration,
}
fn submit(status: u16, body: Value) -> Reply {
    Reply {
        method: "POST",
        status,
        body,
        delay: Duration::ZERO,
    }
}
fn poll(status: u16, body: Value) -> Reply {
    Reply {
        method: "GET",
        status,
        body,
        delay: Duration::ZERO,
    }
}
fn options() -> MergeOptions {
    MergeOptions {
        http_deadline: Duration::from_secs(1),
        budget: Some(Duration::from_secs(5)),
        max_retries: 1,
        retry_base: Duration::from_millis(1),
        poll_interval: Duration::from_millis(1),
    }
}
fn run(script: Vec<Reply>, options: MergeOptions) -> Result<(), String> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        for reply in script {
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            Instant::now() < deadline,
                            "missing {} request",
                            reply.method
                        );
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(e) => panic!("accept: {e}"),
                }
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut reader = std::io::BufReader::new(&mut socket);
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let parts: Vec<_> = line.split_whitespace().collect();
            assert_eq!(parts[0], reply.method);
            if reply.method == "POST" {
                let url = reqwest::Url::parse(&format!("http://localhost{}", parts[1])).unwrap();
                assert_eq!(url.path(), "/bench/_forcemerge");
                let query: std::collections::HashMap<_, _> = url.query_pairs().collect();
                assert_eq!(query.get("max_num_segments").map(|s| s.as_ref()), Some("1"));
                assert_eq!(
                    query.get("wait_for_completion").map(|s| s.as_ref()),
                    Some("false")
                );
            } else {
                assert!(parts[1].starts_with("/_tasks/node%3A42?"), "{line}");
                assert!(parts[1].ends_with("wait_for_completion=false"), "{line}");
            }
            loop {
                line.clear();
                assert!(reader.read_line(&mut line).unwrap() > 0);
                if line == "\r\n" {
                    break;
                }
            }
            std::thread::sleep(reply.delay);
            let body = reply.body.to_string();
            // The deadline test intentionally closes its connection early.
            let _ = write!(socket, "HTTP/1.1 {} Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", reply.status, body.len());
        }
    });
    let transport = TransportBuilder::new(SingleNodeConnectionPool::new(url.parse().unwrap()))
        .build()
        .unwrap();
    let client = OpenSearch::new(transport);
    let result = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(force_merge(&client, "bench", &options));
    server.join().unwrap();
    result
}

#[test]
fn pending_tasks_outlive_retry_count_and_gateway_errors_retry_only_get() {
    let mut script = vec![submit(200, json!({"task":"node:42"}))];
    // More healthy polls than the retry allowance must still complete.
    for _ in 0..4 {
        script.push(poll(200, json!({"completed":false})));
    }
    script.extend([
        poll(504, json!({"error":"gateway timeout"})),
        poll(200, json!({"completed":false})),
        poll(503, json!({"error":"unavailable"})),
        poll(200, done()),
    ]);
    run(script, options()).unwrap();
}

#[test]
fn submission_retries_transient_errors_and_requires_task_id() {
    for status in [429, 502, 503, 504] {
        run(
            vec![
                submit(status, json!({"error":"busy"})),
                submit(200, json!({"task":"node:42"})),
                poll(200, done()),
            ],
            options(),
        )
        .unwrap();
    }
    let err = run(
        vec![submit(504, json!({})), submit(504, json!({}))],
        options(),
    )
    .unwrap_err();
    assert!(
        err.starts_with("Force merge submission failed after 1 retries: HTTP 504"),
        "{err}"
    );
    for body in [json!({}), json!({"task":""})] {
        let err = run(vec![submit(200, body)], options()).unwrap_err();
        assert!(
            err.starts_with("Force merge did not return an asynchronous task ID"),
            "{err}"
        );
    }
    for status in [400, 401, 403, 404] {
        let err = run(
            vec![submit(status, json!({"error":"permanent"}))],
            options(),
        )
        .unwrap_err();
        assert!(
            err.starts_with(&format!("Force merge submission HTTP {status}:")),
            "{err}"
        );
    }
}

#[test]
fn poll_errors_fail_closed() {
    for response in [
        poll(403, json!({"error":"forbidden"})),
        poll(404, json!({"error":"missing task"})),
        poll(
            200,
            json!({"completed":true,"error":{"reason":"merge failed"}}),
        ),
        poll(
            200,
            json!({"completed":true,"response":{"_shards":{"total":1,"successful":0,"failed":1}}}),
        ),
    ] {
        let status = response.status;
        let err = run(
            vec![submit(200, json!({"task":"node:42"})), response],
            options(),
        )
        .unwrap_err();
        if status == 403 || status == 404 {
            assert!(
                err.starts_with(&format!("Force merge task node:42 HTTP {status}:")),
                "{err}"
            );
        }
    }
    let err = run(
        vec![
            submit(200, json!({"task":"node:42"})),
            poll(504, json!({})),
            poll(504, json!({})),
        ],
        options(),
    )
    .unwrap_err();
    assert!(err.contains("after 1 retries"), "{err}");
}

#[test]
fn wall_clock_budget_interrupts_http_and_poll_sleep() {
    let mut opts = options();
    opts.budget = Some(Duration::from_millis(500));
    let mut reply = submit(200, json!({"task":"node:42"}));
    reply.delay = Duration::from_millis(750);
    let err = run(vec![reply], opts).unwrap_err();
    assert!(err.contains("wall-clock budget"), "{err}");
    let mut opts = options();
    opts.budget = Some(Duration::from_millis(500));
    opts.poll_interval = Duration::from_secs(1);
    let err = run(
        vec![
            submit(200, json!({"task":"node:42"})),
            poll(200, json!({"completed":false})),
        ],
        opts,
    )
    .unwrap_err();
    assert!(err.contains("wall-clock budget"), "{err}");
}
