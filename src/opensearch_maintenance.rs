//! Asynchronous index preparation for managed OpenSearch gateways.
use std::time::{Duration, Instant};

use opensearch::http::{headers::HeaderMap, request::JsonBody, Method};
use opensearch::indices::IndicesForcemergeParts;
use opensearch::tasks::TasksGetParts;
use opensearch::OpenSearch;
use serde_json::Value;

pub struct MergeOptions {
    pub http_deadline: Duration,
    pub budget: Option<Duration>,
    pub max_retries: u32,
    pub retry_base: Duration,
    pub poll_interval: Duration,
}

/// Submit an asynchronous merge, then poll its task. Transient failures retry
/// within the budget. A lost submission reply may leave an earlier merge running;
/// on this read-only index, a repeated merge queues behind it and finds no extra
/// work once the first finishes. Healthy unfinished tasks consume no retries.
/// The budget includes HTTP bodies and sleeps. Client expiry cannot cancel the
/// server task (force merge is not cancellable).
pub async fn force_merge(
    client: &OpenSearch,
    index: &str,
    options: &MergeOptions,
) -> Result<(), String> {
    let operation = merge_task(client, index, options);
    match options.budget {
        Some(budget) => tokio::time::timeout(budget, operation)
            .await
            .map_err(|_| format!("Force merge exceeded its {budget:?} wall-clock budget; server task may still be running"))?,
        None => operation.await,
    }
}

async fn merge_task(
    client: &OpenSearch,
    index: &str,
    options: &MergeOptions,
) -> Result<(), String> {
    // The pinned Rust client's typed forcemerge builder does not expose
    // wait_for_completion. Use its transport to retain auth, TLS and URL escaping.
    let indices = [index];
    let path = IndicesForcemergeParts::Index(&indices).url();
    let query = [("max_num_segments", "1"), ("wait_for_completion", "false")];
    let mut retries = 0;
    let task = loop {
        let response = client
            .transport()
            .send(
                Method::Post,
                path.as_ref(),
                HeaderMap::new(),
                Some(&query),
                None::<JsonBody<Value>>,
                Some(options.http_deadline),
            )
            .await;
        let failure = match response {
            Ok(response) => {
                let status = response.status_code().as_u16();
                match response.text().await {
                    Ok(text) if (200..300).contains(&status) => {
                        let body: Value = serde_json::from_str(&text)
                            .map_err(|e| format!("Force merge submission JSON: {e}"))?;
                        break body
                            .get("task")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                format!(
                                    "Force merge did not return an asynchronous task ID: {body}"
                                )
                            })?
                            .to_owned();
                    }
                    Ok(text) if matches!(status, 429 | 502 | 503 | 504) => {
                        format!("HTTP {status}: {text}")
                    }
                    Ok(text) => {
                        return Err(format!("Force merge submission HTTP {status}: {text}"))
                    }
                    Err(e) => format!("response body: {e}"),
                }
            }
            Err(e) => e.to_string(),
        };
        retry_wait("Force merge submission", options, &mut retries, &failure).await?;
    };
    println!("Force merge task {task} submitted; polling until all shards complete...");
    let started = Instant::now();
    let mut last_report = Instant::now();
    let mut retries = 0;
    loop {
        let result = client
            .tasks()
            .get(TasksGetParts::TaskId(&task))
            .wait_for_completion(false)
            .request_timeout(options.http_deadline)
            .send()
            .await;
        let failure = match result {
            Ok(response) => {
                let status = response.status_code().as_u16();
                match response.text().await {
                    Ok(text) if (200..300).contains(&status) => {
                        let body: Value = serde_json::from_str(&text)
                            .map_err(|e| format!("Force merge task {task} JSON: {e}"))?;
                        if task_complete(&body)? {
                            println!(
                                "Force merge task {task} completed in {:.1}s",
                                started.elapsed().as_secs_f64()
                            );
                            return Ok(());
                        }
                        // A healthy, unfinished task is not a failed attempt.
                        retries = 0;
                        if last_report.elapsed() >= Duration::from_secs(60) {
                            println!(
                                "Force merge task {task} still running ({:.0}s)",
                                started.elapsed().as_secs_f64()
                            );
                            last_report = Instant::now();
                        }
                        tokio::time::sleep(options.poll_interval).await;
                        continue;
                    }
                    Ok(text) if matches!(status, 429 | 502 | 503 | 504) => {
                        format!("HTTP {status}: {text}")
                    }
                    Ok(text) => {
                        return Err(format!("Force merge task {task} HTTP {status}: {text}"))
                    }
                    Err(e) => format!("response body: {e}"),
                }
            }
            Err(e) => e.to_string(),
        };
        retry_wait(
            &format!("Force merge task {task} polling"),
            options,
            &mut retries,
            &failure,
        )
        .await?;
    }
}

async fn retry_wait(
    stage: &str,
    options: &MergeOptions,
    retries: &mut u32,
    failure: &str,
) -> Result<(), String> {
    if *retries >= options.max_retries {
        return Err(format!("{stage} failed after {retries} retries: {failure}"));
    }
    let delay = options
        .retry_base
        .saturating_mul(1u32 << (*retries).min(16))
        .min(Duration::from_secs(30));
    eprintln!("{stage} retry {}: {failure}", *retries + 1);
    tokio::time::sleep(delay).await;
    *retries += 1;
    Ok(())
}

fn task_complete(body: &Value) -> Result<bool, String> {
    if body.get("error").is_some() {
        return Err(format!("Force merge task failed: {body}"));
    }
    match body.get("completed").and_then(Value::as_bool) {
        Some(false) => Ok(false),
        Some(true) => {
            let shards = &body["response"]["_shards"];
            let total = shards["total"].as_u64();
            if total.is_some_and(|n| n > 0)
                && shards["successful"].as_u64() == total
                && shards["failed"].as_u64() == Some(0)
                && body["response"].get("error").is_none()
            {
                Ok(true)
            } else {
                Err(format!(
                    "Force merge task did not succeed on every shard: {body}"
                ))
            }
        }
        None => Err(format!(
            "Force merge task missing boolean completed: {body}"
        )),
    }
}

#[cfg(test)]
mod tests;
