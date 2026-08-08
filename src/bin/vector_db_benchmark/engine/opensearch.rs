//! OpenSearch engine implementation.
//!
//! Uses the official `opensearch` crate (async, wrapped with tokio block_on).
//! Very similar to Elasticsearch but uses knn_vector type and different query format.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use opensearch::cluster::ClusterHealthParts;
use opensearch::http::request::JsonBody;
use opensearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use opensearch::indices::{
    IndicesCreateParts, IndicesDeleteParts, IndicesForcemergeParts, IndicesPutSettingsParts,
    IndicesRefreshParts,
};
use opensearch::params::WaitForStatus;
use opensearch::{BulkParts, OpenSearch, SearchParts};
use uuid::Uuid;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::readers::metadata::MetadataItem;

#[derive(Clone)]
struct OpenSearchConfig {
    m: i64,
    ef_construction: i64,
    batch_size: usize,
    parallel: usize,
    /// None = inherit the cluster default (previous behaviour).
    number_of_shards: Option<i64>,
}

pub struct OpenSearchEngine {
    name: String,
    index_name: String,
    #[allow(dead_code)]
    timeout: u64,
    config: OpenSearchConfig,
    search_params: Vec<SearchParams>,
    /// Per-attempt deadline for a force merge, and the wall-clock ceiling across
    /// all its attempts. Both resolved in [`Self::new`] rather than inside
    /// `force_merge`, because `force_merge` runs AFTER the whole upload: a
    /// malformed `OPENSEARCH_FORCE_MERGE_BUDGET=2h` parsed there would throw away
    /// a multi-hour ingest at the last step, which is the opposite of what
    /// `parse_env_secs`'s loud rejection is for. `parse_number_of_shards` resolves
    /// in `new()` for the same reason. It also keeps two `env::var` reads out of
    /// the timed index window (see the boundary note on `SearchRetryPolicy`).
    force_merge_deadline: std::time::Duration,
    force_merge_budget: Option<std::time::Duration>,
    /// Base URL for constructing per-thread clients
    base_url: String,
    /// Tokio runtime for async operations
    rt: tokio::runtime::Runtime,
    /// Shared OpenSearch client
    client: Arc<OpenSearch>,
}

impl OpenSearchEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = crate::effective_config::env_parsed("OPENSEARCH_PORT", 9200);

        let index_name = crate::effective_config::env_or("OPENSEARCH_INDEX", "bench");
        let timeout: u64 = crate::effective_config::env_parsed("OPENSEARCH_TIMEOUT", 300);

        // Extract HNSW config from collection_params.method.parameters (OpenSearch format)
        // or fall back to index_options (ES format) or defaults
        let (m, ef_construction) = extract_hnsw_params(engine_config);

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(16) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(500) as usize;

        // See build_index_settings: shard count materially changes OpenSearch
        // vector performance and the Service default has differed from open-source
        // OpenSearch, so it must be settable rather than inherited silently.
        let number_of_shards = parse_number_of_shards(
            engine_config
                .collection_params
                .as_ref()
                .and_then(|c| c.extra.as_ref())
                .and_then(|e| e.get("number_of_shards")),
        )?;

        // Resolved here, not in `force_merge`: a malformed value must be rejected
        // BEFORE the upload it would otherwise discard. See the field docs.
        let force_merge_deadline =
            resolve_force_merge_timeout(timeout, parse_env_secs("OPENSEARCH_FORCE_MERGE_TIMEOUT")?);
        let force_merge_budget = resolve_force_merge_budget(
            force_merge_deadline,
            parse_env_secs("OPENSEARCH_FORCE_MERGE_BUDGET")?,
        );

        let base_url = build_base_url(host, port);

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        let client = create_os_client(&base_url, timeout)?;

        Ok(Self {
            name: engine_config.name.clone(),
            index_name,
            timeout,
            force_merge_deadline,
            force_merge_budget,
            config: OpenSearchConfig {
                m,
                ef_construction,
                batch_size,
                parallel,
                number_of_shards,
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            base_url,
            rt,
            client: Arc::new(client),
        })
    }

    fn create_progress_bar(&self, total: usize) -> ProgressBar {
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec_int}/s)")
                .unwrap()
                .with_key("per_sec_int", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                    write!(w, "{}", HumanCount(state.per_sec() as u64)).unwrap()
                })
                .progress_chars("#>-"),
        );
        pb
    }

    /// Delete the benchmark index, tolerating the transient states a *managed*
    /// cluster puts an index into.
    ///
    /// Amazon OpenSearch Service takes automated snapshots on a schedule that
    /// cannot be disabled, and deleting an index caught in one fails with
    /// HTTP 400 `snapshot_in_progress_exception`. Because every config in a grid
    /// starts by dropping the previous index, a single unlucky snapshot window
    /// otherwise fails not just that config but every config after it — observed
    /// as 1/6 completing and 5/6 dying on "Failed to delete index: status 400".
    ///
    /// 503 is the other transient case (cluster-manager busy). Both are retried;
    /// anything else fails immediately.
    ///
    /// The response body is included in the error. Without it "status 400" is
    /// undiagnosable — the cause has to be guessed from the outside.
    fn delete_index(&self) -> Result<(), String> {
        retry_index_op(
            &self.rt,
            "Delete index",
            index_op_policy(),
            // 404 is success: the index we were asked to drop is already gone.
            |status| status == 200 || status == 404,
            delete_index_retryable,
            || {
                self.rt.block_on(
                    self.client
                        .indices()
                        .delete(IndicesDeleteParts::Index(&[&self.index_name]))
                        .send(),
                )
            },
        )
    }

    fn create_index(&self, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        let dist_lower = distance.to_lowercase();
        let space_type = resolve_index_space_type(&dist_lower, vector_size)?;

        // Build properties with knn_vector type
        let mut properties = serde_json::json!({
            "vector": {
                "type": "knn_vector",
                "dimension": vector_size,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                    "space_type": space_type,
                    "parameters": {
                        "m": self.config.m,
                        "ef_construction": self.config.ef_construction,
                    }
                }
            }
        });

        // Add schema fields from dataset config
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                let props = properties.as_object_mut().unwrap();
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    let os_type = match ft {
                        "int" => "long",
                        "geo" => "geo_point",
                        // "bool"/"datetime" are not valid OS types; map to the real
                        // ones. OS coerces the reader's "true"/"false" string into a
                        // `boolean` field and parses ISO-8601 into a `date` field.
                        // Forwarding them verbatim made index creation reject the
                        // whole mapping.
                        "bool" => "boolean",
                        "datetime" => "date",
                        // A `uuid` is an exact-match opaque string; "uuid" is not a
                        // valid OS type, so (like bool/datetime above) forwarding it
                        // verbatim made index creation reject the whole mapping,
                        // silently breaking every uuid-equality filter. Map it to
                        // `keyword` (exact term match, no analysis).
                        "uuid" => "keyword",
                        other => other,
                    };
                    props.insert(
                        field_name.clone(),
                        serde_json::json!({
                            "type": os_type,
                            "index": true,
                        }),
                    );
                }
            }
        }

        let index_settings = build_index_settings(self.config.number_of_shards);

        let body = serde_json::json!({
            "settings": { "index": index_settings },
            "mappings": {
                "properties": properties,
            }
        });

        // Same transient-state problem as delete_index: a cluster whose manager
        // thread pool is busy answers create-index with
        // `process_cluster_event_timeout_exception` (503) rather than creating it.
        // Retrying rides that out instead of failing the config — and every config
        // after it, since each one starts by creating its index.
        retry_index_op(
            &self.rt,
            "Create index",
            index_op_policy(),
            |status| (200..300).contains(&status),
            create_index_retryable,
            || {
                self.rt.block_on(
                    self.client
                        .indices()
                        .create(IndicesCreateParts::Index(&self.index_name))
                        .body(body.clone())
                        .send(),
                )
            },
        )
    }

    fn upload_parallel(
        &self,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
    ) -> Result<(), String> {
        let pb = self.create_progress_bar(ids.len());
        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.config.batch_size)
            .map(|start| (start, (start + self.config.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let base_url = self.base_url.clone();
        let timeout = self.timeout;
        let index_name = self.index_name.clone();

        std::thread::scope(|s| {
            for _ in 0..self.config.parallel {
                let base_url = base_url.clone();
                let index_name = index_name.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };

                    let client = match create_os_client(&base_url, timeout) {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e);
                            return;
                        }
                    };

                    loop {
                        let idx = batch_idx.fetch_add(1, Ordering::SeqCst);
                        if idx >= total_batches {
                            break;
                        }
                        if error.lock().unwrap().is_some() {
                            break;
                        }

                        let (batch_start, batch_end) = batches[idx];
                        if let Err(e) = upload_bulk_batch(
                            &rt,
                            &client,
                            &index_name,
                            &ids[batch_start..batch_end],
                            &vectors[batch_start..batch_end],
                            &metadata[batch_start..batch_end],
                        ) {
                            *error.lock().unwrap() = Some(e);
                            break;
                        }
                        pb.inc((batch_end - batch_start) as u64);
                    }
                });
            }
        });

        pb.finish_with_message("Upload complete");

        if let Some(e) = error.lock().unwrap().take() {
            return Err(e);
        }
        Ok(())
    }

    /// Refresh the index to make just-uploaded documents searchable. Required
    /// because we set `refresh_interval: -1` (no periodic refresh) during upload.
    fn refresh(&self) -> Result<(), String> {
        retry_index_op(
            &self.rt,
            "Refresh",
            index_op_policy(),
            |status| (200..300).contains(&status),
            index_maintenance_retryable,
            || {
                self.rt.block_on(
                    self.client
                        .indices()
                        .refresh(IndicesRefreshParts::Index(&[&self.index_name]))
                        .send(),
                )
            },
        )
    }

    /// Force-merge to a **single** segment, refresh so queries actually see it,
    /// then wait for the cluster to settle.
    ///
    /// `max_num_segments(1)` mirrors `elasticsearch.rs`, and the parity matters
    /// more than it looks: a segment is one HNSW graph, and a k-NN query searches
    /// every segment of every shard and merges the per-segment result lists. So
    /// segment count moves BOTH recall and latency.
    ///
    /// Latency, because N segments means N graph traversals plus an N-way merge
    /// instead of one. Recall, because each segment holds 1/N of the corpus, and a
    /// *fixed* `ef` explores a far larger *fraction* of a small graph than of a
    /// big one — eighty-five ~14k-doc graphs are close to exhaustively searched at
    /// ef=128, one 1.18M-doc graph is nowhere near. (Note it is NOT that each
    /// segment gets its own `ef` budget: Lucene passes each segment the global
    /// `k`.) Measured on glove-25-angular, 85 searchable segments vs 1: recall
    /// 0.9850 → 0.9405 at ef=128, 0.9977 → 0.9930 at ef=512.
    ///
    /// So leaving OpenSearch on Lucene's dynamic merge policy while Elasticsearch
    /// is pinned to one segment makes the two engines structurally incomparable,
    /// and makes the OpenSearch side irreproducible on its own terms — the merge
    /// policy's answer depends on ingest timing, batch size and thread count, so
    /// two runs of the *same* config can land on different segment counts. (#210)
    ///
    /// Retried on the same transient states as the other index ops. This is the op
    /// that most needs them: it is the longest-running of the four, it is what a
    /// managed front door most often answers with 504, and it runs *after* the
    /// whole ingest (see `upload`). Failing it un-retried discards a multi-hour
    /// upload at the very last step — so the bulk-upload retries above would buy
    /// nothing on the runs they were written for. `elasticsearch.rs` already
    /// retries force-merge for exactly this reason.
    ///
    /// Merging to one segment makes this call substantially longer, which makes
    /// the retry budget and the request timeout load-bearing together — see
    /// [`resolve_force_merge_timeout`] for why the shared `OPENSEARCH_TIMEOUT` is
    /// not the right bound here.
    fn force_merge(&self) -> Result<(), String> {
        println!("Forcing merge into 1 segment...");

        // Named for what it is rather than after the transport setter: a local
        // called `request_timeout` is indistinguishable, to the shipped-config
        // knob guard's token search, from an engine reading
        // `connection_params.request_timeout` — which this does not do.
        let merge_deadline = self.force_merge_deadline;
        retry_index_op(
            &self.rt,
            "Force merge",
            index_op_policy().with_budget(self.force_merge_budget),
            |status| (200..300).contains(&status),
            index_maintenance_retryable,
            || {
                self.rt.block_on(
                    self.client
                        .indices()
                        .forcemerge(IndicesForcemergeParts::Index(&[&self.index_name]))
                        .max_num_segments(1)
                        .request_timeout(merge_deadline)
                        .send(),
                )
            },
        )?;

        // A force merge alone does NOT change what queries search — without this
        // refresh the whole merge is a no-op that costs 20 minutes and 2x disk.
        //
        // `_forcemerge` defaults to `flush=true`, which commits the new segment,
        // but the commit only refreshes Lucene's INTERNAL reader. The searcher
        // that serves queries keeps its handle on the pre-merge segments until an
        // EXTERNAL refresh swaps it, and we set `refresh_interval: -1` during
        // upload, so no periodic refresh will ever do it for us.
        //
        // Measured on glove-25-angular (1,183,514 docs, 1 shard) — immediately
        // after a successful `max_num_segments=1` merge:
        //   `_segments` → 86 segments; the merged 1,183,514-doc segment has
        //   `"search": false`, and all 1,183,514 docs are served by the 85 OLD
        //   segments. After one `POST /<index>/_refresh`: 1 segment, searchable.
        //
        // So the refresh is not tidiness — it is the step that makes
        // `max_num_segments(1)` mean anything to the numbers this tool publishes.
        //
        // Framing note, updated after #248. This used to be described as an
        // OpenSearch-only hazard on the grounds that `elasticsearch.rs` ran with
        // `refresh_interval: "10s"`, so even if its merge had been invisible a
        // periodic refresh would have exposed it within 10 s. That is no longer
        // true: #248 moved Elasticsearch to `refresh_interval: -1` as well, so
        // NEITHER engine has a periodic refresh to fall back on. `elasticsearch.rs`
        // does an explicit `refresh()` before force-merging; this engine does one
        // AFTER instead (see `upload`), which is the whole point of the step below.
        //
        // What is still genuinely engine-specific is the step below. Measured on
        // an identical 150-doc, 3-segment probe against both containers in
        // `tests/docker-compose.test.yml`, after `_forcemerge?max_num_segments=1`
        // and before any refresh: Elasticsearch 9.4.3 reported 1 segment, 1
        // searchable (its force-merge reopened the searcher), while OpenSearch
        // 3.7.0 reported 4 segments, 3 searchable (the old ones still serving).
        // So the post-merge refresh remains required here and is not mirrored in
        // `elasticsearch.rs`. Since #248 removed the periodic-refresh safety net
        // that would previously have masked it, Elasticsearch is now relying on
        // force-merge reopening the searcher on its own — worth a follow-up
        // probe there, but out of scope for #210.
        self.refresh()?;

        self.wait_for_cluster_health()
    }

    /// Wait for the index to report at least yellow before the search phase runs.
    ///
    /// `elasticsearch.rs` does this after its force merge and OpenSearch is
    /// equally entitled to it — arguably more so, since the merge that just ran is
    /// now a single-segment one. Two reasons it is warranted rather than
    /// ceremonial:
    ///
    /// 1. **Comparability.** The health wait is inside the timed index phase on
    ///    both engines, and on both engines the search phase begins only once the
    ///    cluster reports settled. Without it, OpenSearch would start querying
    ///    while the post-merge cluster state is still being applied and pay that
    ///    cost inside the *search* numbers, where Elasticsearch does not.
    /// 2. **Robustness.** Force merge flushes, which on a multi-node or managed
    ///    domain can leave shards initializing or relocating. Querying through
    ///    that produces the flakiest possible first latency samples.
    ///
    /// Deliberate deviation from `elasticsearch.rs`: the check is scoped to **our
    /// index**, not the whole cluster. A managed OpenSearch domain carries system
    /// indices (security, alerting, ISM) whose replicas may be permanently
    /// unassigned on a small domain, which pins cluster-wide health at yellow or
    /// red forever. A cluster-wide wait would then burn the entire budget and fail
    /// a finished ingest over an index the benchmark never touches. Our index has
    /// `number_of_replicas: 0`, so it reaches green on its own as soon as the
    /// primaries are active; asking for yellow keeps the ES semantics while
    /// staying immune to unrelated indices.
    ///
    /// On a missed status both shipped engines answer **HTTP 408** with
    /// `"timed_out": true` (verified against OpenSearch 3.7.0 and Elasticsearch
    /// 9.4.3), so the status range does the work. [`cluster_health_settled`] also
    /// rejects a `"timed_out": true` body carrying a 2xx, which is what
    /// `cluster.health.return_200_for_cluster_health_timeout` produces on the
    /// Elasticsearch versions that support it — defence in depth, not the
    /// behaviour of the containers this repo tests against.
    ///
    /// This runs as the tail of `force_merge`, i.e. after a completed ingest AND a
    /// completed merge, and that shapes the error handling:
    ///
    /// * Transient states retry ([`cluster_health_retryable`]).
    /// * **401/403 warn and continue.** On AWS OpenSearch with fine-grained access
    ///   control, `cluster:monitor/health` is routinely denied even when the index
    ///   actions this tool needs are granted. Failing there would throw away a
    ///   multi-hour ingest over a *monitoring* permission — and the merge and
    ///   refresh that actually determine the measured index state have already
    ///   succeeded, so the wait is belt-and-braces, not correctness. Retrying an
    ///   authorization decision 11 times first would be pure waste — a 403 returns
    ///   immediately, so the cost is the backoff alone (~1.7–3.5 min after jitter),
    ///   but it buys nothing at all.
    ///   This is a NEW failure mode introduced by adding the wait, on exactly the
    ///   managed domains the index-scoping above exists to protect, so it is
    ///   handled here rather than inherited.
    /// * Anything else fails, because it means we genuinely cannot tell whether the
    ///   cluster settled.
    fn wait_for_cluster_health(&self) -> Result<(), String> {
        println!("Waiting for OpenSearch yellow status...");

        let policy = index_op_policy();
        let mut attempt: u32 = 0;
        loop {
            let last_error = match self.rt.block_on(
                self.client
                    .cluster()
                    .health(ClusterHealthParts::Index(&[&self.index_name]))
                    .wait_for_status(WaitForStatus::Yellow)
                    .timeout("30s")
                    .request_timeout(std::time::Duration::from_secs(60))
                    .send(),
            ) {
                Ok(resp) => {
                    let status = resp.status_code().as_u16();
                    match self.rt.block_on(resp.text()) {
                        Ok(text) => match classify_health_response(status, &text) {
                            HealthVerdict::Settled => return Ok(()),
                            HealthVerdict::Unobservable => {
                                println!(
                                    "  ⚠ cluster health not observable (HTTP {}): {}. \
                                     The force merge and refresh already succeeded, so the \
                                     index state is correct; continuing without the settle \
                                     check. Grant cluster:monitor/health to restore it.",
                                    status, text
                                );
                                return Ok(());
                            }
                            HealthVerdict::Failed => {
                                return Err(format!(
                                    "Cluster health wait failed (HTTP {}, {} retries): {}",
                                    status, attempt, text
                                ));
                            }
                            HealthVerdict::Retry => format!("HTTP {}: {}", status, text),
                        },
                        Err(e) => format!("HTTP {}, body unreadable: {}", status, e),
                    }
                }
                Err(e) => format!("transport error: {}", e),
            };

            if attempt >= policy.max_retries {
                return Err(format!(
                    "Cluster health wait failed after {} retries: {}",
                    attempt, last_error
                ));
            }
            backoff_sleep(policy.base_delay_ms, attempt);
            attempt += 1;
        }
    }

    /// Apply search-time settings (e.g., knn.algo_param.ef_search)
    fn setup_search(&self, params: &SearchParams) -> Result<(), String> {
        // Cold-cache graph loading is warmed uniformly by the per-worker prime
        // query in `search` (mirrors redis/vertex), so no engine-specific
        // server-side `_knn/warmup` call is needed here.
        // Accepted flat (our configs) or nested under `search_params`/`config`
        // (upstream's) — see SearchParams::knob.
        let ef_search = params
            .knob("knn.algo_param.ef_search")
            .and_then(|v| v.as_i64());

        if let Some(ef) = ef_search {
            let body = serde_json::json!({
                "index": {
                    "knn.algo_param.ef_search": ef,
                }
            });

            let resp = self
                .rt
                .block_on(
                    self.client
                        .indices()
                        .put_settings(IndicesPutSettingsParts::Index(&[&self.index_name]))
                        .body(body)
                        .send(),
                )
                .map_err(|e| format!("Failed to apply search settings: {}", e))?;

            if !resp.status_code().is_success() {
                let text = self.rt.block_on(resp.text()).unwrap_or_default();
                eprintln!("Warning: failed to set ef_search={}: {}", ef, text);
            }
        }
        Ok(())
    }
}

/// Extract m and ef_construction from OpenSearch collection_params.
/// Supports: collection_params.method.parameters.{m, ef_construction}
/// Falls back to: collection_params.index_options.{m, ef_construction}
fn extract_hnsw_params(engine_config: &EngineConfig) -> (i64, i64) {
    if let Some(cp) = &engine_config.collection_params {
        // Try OpenSearch format: method.parameters
        if let Some(extra) = &cp.extra {
            if let Some(method) = extra.get("method") {
                if let Some(params) = method.get("parameters") {
                    let m = params.get("m").and_then(|v| v.as_i64()).unwrap_or(16);
                    let ef = params
                        .get("ef_construction")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(100);
                    return (m, ef);
                }
            }
        }
        // Try ES format: index_options
        if let Some(io) = &cp.index_options {
            return (io.m.unwrap_or(16), io.ef_construction.unwrap_or(100));
        }
    }
    (16, 100)
}

fn build_base_url(host: &str, port: u16) -> String {
    let user = crate::effective_config::env_or("OPENSEARCH_USER", "admin");
    let password = crate::effective_config::env_or("OPENSEARCH_PASSWORD", "admin");

    let scheme_host = if host.starts_with("http") {
        host.to_string()
    } else {
        format!("https://{}", host)
    };

    if let Some(rest) = scheme_host.strip_prefix("http://") {
        format!("http://{}:{}@{}:{}", user, password, rest, port)
    } else if let Some(rest) = scheme_host.strip_prefix("https://") {
        format!("https://{}:{}@{}:{}", user, password, rest, port)
    } else {
        format!("https://{}:{}@{}:{}", user, password, scheme_host, port)
    }
}

/// Create an OpenSearch client from a base URL.
fn create_os_client(base_url: &str, timeout: u64) -> Result<OpenSearch, String> {
    let url = opensearch::http::Url::parse(base_url)
        .map_err(|e| format!("Invalid base URL '{}': {}", base_url, e))?;
    let pool = SingleNodeConnectionPool::new(url);
    let transport = TransportBuilder::new(pool)
        .timeout(std::time::Duration::from_secs(timeout))
        .disable_proxy()
        .cert_validation(opensearch::cert::CertificateValidation::None)
        .build()
        .map_err(|e| format!("Failed to build transport: {}", e))?;
    Ok(OpenSearch::new(transport))
}

fn id_to_uuid_hex(id: i64) -> String {
    Uuid::from_u128(id as u128).as_simple().to_string()
}

fn uuid_hex_to_int(hex: &str) -> Result<i64, String> {
    let uuid = Uuid::parse_str(hex).map_err(|e| format!("Invalid UUID hex '{}': {}", hex, e))?;
    Ok(uuid.as_u128() as i64)
}

/// Validate index build parameters and resolve the knn `space_type`. Extracted
/// verbatim from `create_index` so the guard order + error strings are unit-
/// testable without a live OpenSearch. `dist_lower` must already be lowercased.
/// dot/ip is rejected first, then the dim cap, then the general mapping.
fn resolve_index_space_type(dist_lower: &str, vector_size: i64) -> Result<&'static str, String> {
    if dist_lower == "dot" || dist_lower == "ip" {
        return Err("OpenSearch does not support DOT product distance".to_string());
    }
    if vector_size > 2048 {
        return Err(format!(
            "OpenSearch does not support vector_size > 2048 (got {})",
            vector_size
        ));
    }
    os_space_type(dist_lower)
}

/// Map a dataset distance name to the OpenSearch knn `space_type`. `dot`/`ip`
/// is unsupported and unknown metrics error. A wrong arm here would silently
/// change ranking, so every arm is unit-tested.
fn os_space_type(distance: &str) -> Result<&'static str, String> {
    match distance.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok("l2"),
        "cosine" | "angular" => Ok("cosinesimil"),
        "dot" | "ip" => Err("OpenSearch does not support DOT product distance".to_string()),
        other => Err(format!(
            "Unsupported distance metric for OpenSearch: {}",
            other
        )),
    }
}

/// Parse conditions into OpenSearch bool query (same DSL as Elasticsearch).
pub(crate) fn parse_os_conditions(conditions: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    build_group(obj)
}

/// Build one boolean GROUP (a `{and:[...], or:[...]}` object) into an OpenSearch
/// `{bool: ...}` query. Recursive: an entry inside `and`/`or` may itself be a
/// nested group, so this and [`build_subfilters`] call each other. A nested group
/// becomes its own `{bool: ...}` object placed inside the parent's `must`/`should`
/// array — OpenSearch's native bool nesting — so `(a AND b) OR (c AND d)` is
/// preserved instead of being silently flattened into one flat clause list.
fn build_group(obj: &serde_json::Map<String, serde_json::Value>) -> Option<serde_json::Value> {
    let and_filters = obj
        .get("and")
        .and_then(|v| v.as_array())
        .map(|entries| build_subfilters(entries))
        .filter(|f| !f.is_empty());
    let or_filters = obj
        .get("or")
        .and_then(|v| v.as_array())
        .map(|entries| build_subfilters(entries))
        .filter(|f| !f.is_empty());

    if and_filters.is_none() && or_filters.is_none() {
        return None;
    }

    let mut bool_query = serde_json::Map::new();
    if let Some(must) = and_filters {
        bool_query.insert("must".to_string(), serde_json::Value::Array(must));
    }
    if let Some(should) = or_filters {
        bool_query.insert("should".to_string(), serde_json::Value::Array(should));
        // Force OR filters to actually restrict results. `minimum_should_match`
        // defaults to 0 as soon as a `must` clause is also present, which would
        // silently drop the OR condition in a mixed AND+OR filter.
        bool_query.insert(
            "minimum_should_match".to_string(),
            serde_json::Value::from(1),
        );
    }

    Some(serde_json::json!({ "bool": bool_query }))
}

/// Build individual subfilters from an array of condition entries. Each entry is
/// either a nested boolean group (`{and:[...]}` / `{or:[...]}`) — recursed via
/// [`build_group`] into a nested `{bool: ...}` — or a leaf `{field: {op: criteria}}`.
fn build_subfilters(entries: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut filters = Vec::new();
    for entry in entries {
        if let Some(entry_obj) = entry.as_object() {
            // Nested group: an entry carrying an `and`/`or` key is a sub-tree,
            // not a field leaf. Recurse and nest it as its own `{bool: ...}`.
            if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
                if let Some(f) = build_group(entry_obj) {
                    filters.push(f);
                }
                continue;
            }
            for (field_name, field_filters) in entry_obj {
                if let Some(filter_obj) = field_filters.as_object() {
                    for (condition_type, criteria) in filter_obj {
                        if let Some(filter) = build_filter(field_name, condition_type, criteria) {
                            filters.push(filter);
                        }
                    }
                }
            }
        }
    }
    filters
}

fn build_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<serde_json::Value> {
    match condition_type {
        "match" => {
            // match_any: field value in a list (keywords or integers). Emit a
            // `terms` query — the exact/case-sensitive OR-of-values semantics of
            // qdrant's Condition::matches(field, Vec). An empty IN-set matches
            // NOTHING, so we emit `terms: []` (a valid match-nothing query)
            // rather than dropping the clause: dropping the sole clause would
            // leave `bool.must:[]`, which OpenSearch treats as match-ALL —
            // silently returning unfiltered results, the inverse of the filter.
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                return Some(serde_json::json!({"terms": {field_name: any}}));
            }
            // Full-text `{"match": {"text": …}}` conditions: emit an OpenSearch
            // `match` query against the analyzed `text` field. `match` tokenizes
            // the query the same way the field is analyzed, so it matches docs
            // CONTAINING the term(s) — aligning with the tokenized semantics
            // redis uses via `@field:($tok)` and the ground truth in
            // `write_fulltext_project` (docs whose body CONTAINS "quick"). We use
            // `match` (not `match_phrase`) because the fixture filters on single
            // tokens; dropping this clause would leave `bool.must:[]`, which
            // OpenSearch treats as match-ALL — silently running the kNN query
            // UNFILTERED while recall is scored against filtered ground truth
            // (#120).
            if let Some(text) = criteria.get("text") {
                return Some(serde_json::json!({"match": {field_name: text}}));
            }
            let value = criteria.get("value")?;
            // Guard non-scalar: an array/object/null under `value` is malformed
            // input — the canonical model uses `match.any` for lists. Drop the
            // clause (return None) instead of forwarding
            // `{"match":{field:[1,2]}}` verbatim, matching qdrant/redis/valkey/
            // vectorsets.
            if !(value.is_string() || value.is_number() || value.is_boolean()) {
                return None;
            }
            Some(serde_json::json!({"match": {field_name: value}}))
        }
        "range" => {
            let criteria_obj = criteria.as_object()?;
            let mut range = serde_json::Map::new();
            for key in &["lt", "gt", "lte", "gte"] {
                if let Some(val) = criteria_obj.get(*key) {
                    if !val.is_null() {
                        range.insert(key.to_string(), val.clone());
                    }
                }
            }
            Some(serde_json::json!({"range": {field_name: range}}))
        }
        "geo" => {
            let lat = criteria.get("lat")?;
            let lon = criteria.get("lon")?;
            let radius = criteria
                .get("radius")
                .and_then(|r| r.as_f64())
                .unwrap_or(1000.0);
            Some(serde_json::json!({
                "geo_distance": {
                    "distance": format!("{}m", radius),
                    field_name: {"lat": lat, "lon": lon},
                }
            }))
        }
        _ => None,
    }
}

/// Upload a batch using the official OpenSearch bulk API.
fn upload_bulk_batch(
    rt: &tokio::runtime::Runtime,
    client: &OpenSearch,
    index_name: &str,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    // Held as plain Values rather than JsonBody so the batch can be rebuilt and
    // resent on a retry: `.body()` consumes the Vec and JsonBody is not Clone.
    let mut lines: Vec<serde_json::Value> = Vec::with_capacity(ids.len() * 2);

    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);

        // Action line
        lines.push(serde_json::json!({"index": {"_id": uuid_hex}}));

        // Document line
        let mut doc = serde_json::Map::new();
        let vec_json: Vec<serde_json::Value> = vectors[i]
            .iter()
            .map(|&f| serde_json::Value::from(f))
            .collect();
        doc.insert("vector".to_string(), serde_json::Value::Array(vec_json));

        if let Some(meta) = &metadata[i] {
            for (k, v) in &meta.fields {
                let val = match v {
                    MetadataValue::String(s) => serde_json::Value::String(s.clone()),
                    MetadataValue::Int(n) => serde_json::Value::from(*n),
                    MetadataValue::Float(f) => serde_json::json!(*f),
                    MetadataValue::Labels(labels) => serde_json::Value::Array(
                        labels
                            .iter()
                            .map(|l| serde_json::Value::String(l.clone()))
                            .collect(),
                    ),
                    MetadataValue::Geo { lon, lat } => {
                        serde_json::json!({ "lon": lon, "lat": lat })
                    }
                };
                doc.insert(k.clone(), val);
            }
        }

        lines.push(serde_json::Value::Object(doc));
    }

    // HTTP 429 (and 503) are back-pressure, not failure: the server is telling us
    // to slow down and come back. Aborting the experiment on the first one makes
    // ingest impossible against a managed service that sheds load — Amazon
    // OpenSearch Service returns 429 readily on a single-node domain, because
    // knn.algo_param.index_thread_qty defaults to 1 and the write path cannot
    // drain as fast as a parallel uploader pushes. Retry with exponential backoff
    // instead, and only give up once the server has had several chances.
    //
    // Rejections also arrive as HTTP 200 with per-item errors (bulk is partial by
    // design), so item-level 429 / es_rejected_execution_exception /
    // circuit_breaking_exception are treated as retryable too.
    let max_retries: u32 = crate::effective_config::env_parsed("OPENSEARCH_BULK_MAX_RETRIES", 8);
    let base_delay_ms: u64 =
        crate::effective_config::env_parsed("OPENSEARCH_BULK_RETRY_BASE_MS", 500);

    let mut attempt: u32 = 0;
    loop {
        let body: Vec<JsonBody<serde_json::Value>> =
            lines.iter().cloned().map(JsonBody::new).collect();

        // A dropped connection or read timeout fails here, BEFORE any HTTP status
        // exists, so status-based retry never sees it. Over a multi-hour ingest a
        // single transient reset would otherwise discard the whole run.
        let resp = match rt.block_on(client.bulk(BulkParts::Index(index_name)).body(body).send()) {
            Ok(r) => r,
            Err(e) => {
                if attempt >= max_retries {
                    return Err(format!(
                        "Bulk upload failed after {} retries: {} \
                         (transport error — the server may be dropping connections \
                         under load; lower upload_params.parallel or batch_size)",
                        max_retries, e
                    ));
                }
                backoff_sleep(base_delay_ms, attempt);
                attempt += 1;
                continue;
            }
        };

        let status = resp.status_code().as_u16();
        let http_retryable = http_status_retryable(status);

        if !http_retryable && !resp.status_code().is_success() {
            let text = rt.block_on(resp.text()).unwrap_or_default();
            return Err(format!("Bulk upload error: HTTP {}: {}", status, text));
        }

        if http_retryable {
            if attempt >= max_retries {
                let text = rt.block_on(resp.text()).unwrap_or_default();
                let hint = bulk_retry_hint(status);
                return Err(format!(
                    "Bulk upload error: HTTP {} still returned after {} retries ({}): {}",
                    status, max_retries, hint, text
                ));
            }
            backoff_sleep(base_delay_ms, attempt);
            attempt += 1;
            continue;
        }

        let resp_body: serde_json::Value = rt
            .block_on(resp.json())
            .map_err(|e| format!("Failed to parse bulk response: {}", e))?;

        if resp_body
            .get("errors")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            let items = resp_body.get("items").and_then(|v| v.as_array());
            let (error_count, retryable_count) =
                bulk_item_error_counts(items.map(Vec::as_slice).unwrap_or(&[]));

            if batch_is_retryable(error_count, retryable_count) && attempt < max_retries {
                backoff_sleep(base_delay_ms, attempt);
                attempt += 1;
                continue;
            }

            return Err(format!(
                "Bulk upload had {} errors out of {} documents ({} retryable, {} retries used)",
                error_count,
                ids.len(),
                retryable_count,
                attempt
            ));
        }

        return Ok(());
    }
}

/// Exponential backoff delay, capped so a long stall cannot sleep for hours. The
/// attempt is clamped *before* the shift, since `1u64 << 64` panics in a debug
/// build and yields a garbage delay in release.
///
/// Split from `backoff_sleep` so the arithmetic is unit-testable without
/// actually sleeping.
fn backoff_delay_ms(base_delay_ms: u64, attempt: u32) -> u64 {
    base_delay_ms
        .saturating_mul(1u64 << attempt.min(6))
        .min(30_000)
}

/// Spread a capped delay over `[d/2, d]` using `rand` as the entropy source.
///
/// A deterministic delay makes every worker retry in lock-step: all `parallel`
/// workers get shed within milliseconds of each other, sleep the identical
/// amount, and fire again simultaneously — so the retry wave reinforces the
/// overload it is meant to relieve. Half the delay is kept fixed so backoff still
/// grows monotonically; the other half is jittered to decorrelate the workers.
fn jittered_delay_ms(capped_delay_ms: u64, rand: u64) -> u64 {
    let half = capped_delay_ms / 2;
    half + rand % (capped_delay_ms - half + 1)
}

/// Per-thread xorshift, seeded from the thread id. Retry jitter only needs to
/// decorrelate workers from each other, not to be cryptographic, and a
/// thread-local generator keeps the sleep path allocation- and lock-free.
fn retry_jitter_rand() -> u64 {
    use std::cell::Cell;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            let mut h = DefaultHasher::new();
            std::thread::current().id().hash(&mut h);
            // A zero seed is a fixed point of xorshift; 1 is an arbitrary escape.
            h.finish() | 1
        });
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        x
    })
}

/// Exponential backoff with jitter, capped so a long stall cannot sleep for
/// hours. Returns how long it actually slept, so a caller on a timed path can
/// subtract the client-side wait from what it reports as server latency.
fn backoff_sleep(base_delay_ms: u64, attempt: u32) -> std::time::Duration {
    let delay = jittered_delay_ms(
        backoff_delay_ms(base_delay_ms, attempt),
        retry_jitter_rand(),
    );
    let delay = std::time::Duration::from_millis(delay);
    std::thread::sleep(delay);
    delay
}

/// A resolved retry budget. Read once at construction rather than per operation:
/// `knn_send` runs inside the measured window, where an `env::var` lookup is
/// avoidable client CPU work (see the boundary note on `knn_send`).
#[derive(Clone, Copy)]
struct RetryPolicy {
    max_retries: u32,
    base_delay_ms: u64,
    /// Optional wall-clock ceiling across *all* attempts. `None` = bounded by
    /// attempt count alone, which is only safe while each attempt is short.
    budget: Option<std::time::Duration>,
}

impl RetryPolicy {
    fn from_env(max_var: &str, base_var: &str, default_max: u32, default_base_ms: u64) -> Self {
        Self {
            max_retries: crate::effective_config::env_parsed(max_var, default_max),
            base_delay_ms: crate::effective_config::env_parsed(base_var, default_base_ms),
            // #246's per-policy budget; kept as it landed.
            budget: None,
        }
    }

    fn with_budget(self, budget: Option<std::time::Duration>) -> Self {
        Self { budget, ..self }
    }
}

/// Run one index-level operation, riding out the transient states a *managed*
/// cluster returns.
///
/// `accept` decides which statuses mean success (delete tolerates 404);
/// `retryable` decides which failures are worth another attempt.
///
/// Transport errors are retried alongside HTTP statuses. A dropped connection or
/// read timeout fails *before* any status exists, so a status-only retry never
/// sees it — and a managed domain resets connections during blue/green node
/// replacement, which is exactly when these operations run. Retrying only
/// statuses here would have shipped the fix and the bug side by side.
fn retry_index_op<S>(
    rt: &tokio::runtime::Runtime,
    what: &str,
    policy: RetryPolicy,
    accept: impl Fn(u16) -> bool,
    retryable: impl Fn(u16, &str) -> bool,
    mut send: S,
) -> Result<(), String>
where
    S: FnMut() -> Result<opensearch::http::response::Response, opensearch::Error>,
{
    let started = std::time::Instant::now();
    let mut attempt: u32 = 0;
    loop {
        let last_error = match send() {
            Ok(resp) => {
                let status = resp.status_code().as_u16();
                if accept(status) {
                    return Ok(());
                }
                // Draining the body serves two purposes: it yields the diagnostic
                // (without it, "status 400" is undiagnosable from the outside) and
                // it lets the connection return to the pool, so a retry against a
                // TLS-terminated managed domain does not pay a fresh handshake.
                match rt.block_on(resp.text()) {
                    Ok(body) => {
                        if !retryable(status, &body) {
                            return Err(format!(
                                "{} failed (HTTP {}, {} retries): {}",
                                what, status, attempt, body
                            ));
                        }
                        format!("HTTP {}: {}", status, body)
                    }
                    // A body we could not read is itself a transport signal, and
                    // the retry predicates key off the body text — classifying it
                    // non-retryable would hard-fail the very case this exists for
                    // (a 400 `snapshot_in_progress_exception` whose body was lost).
                    Err(e) => format!("HTTP {}, body unreadable: {}", status, e),
                }
            }
            Err(e) => format!("transport error: {}", e),
        };

        if attempt >= policy.max_retries {
            return Err(format!(
                "{} failed after {} retries: {}",
                what, attempt, last_error
            ));
        }
        // Attempt count alone stops bounding anything once a single attempt may
        // run for an hour (force merge), so an operation that asked for a
        // wall-clock ceiling gets one. Checked before sleeping: there is no point
        // backing off into a budget that is already spent.
        if let Some(budget) = policy.budget {
            let elapsed = started.elapsed();
            if elapsed >= budget {
                return Err(format!(
                    "{} exceeded its {:.0}s wall-clock budget after {} retries ({:.0}s elapsed): {}",
                    what,
                    budget.as_secs_f64(),
                    attempt,
                    elapsed.as_secs_f64(),
                    last_error
                ));
            }
        }
        backoff_sleep(policy.base_delay_ms, attempt);
        attempt += 1;
    }
}

/// HTTP statuses worth retrying on the bulk and search paths. 429 = write/search
/// queue full. 503 = unavailable. 502/504 = the managed service's front door gave
/// up, which happens both transiently and when a single request is simply too big
/// — retrying distinguishes them, since a size problem survives every attempt.
fn http_status_retryable(status: u16) -> bool {
    matches!(status, 429 | 502 | 503 | 504)
}

/// Which of the two failure modes an exhausted bulk retry most likely hit. The
/// gateway/entity statuses point at request size; anything else is load shedding.
fn bulk_retry_hint(status: u16) -> &'static str {
    if matches!(status, 502 | 504 | 413) {
        "request is likely too large — lower upload_params.batch_size \
         (bulk bytes scale with vector dimension)"
    } else {
        "server is shedding load — lower upload_params.parallel or \
         raise OPENSEARCH_BULK_MAX_RETRIES"
    }
}

/// `(failing items, of which retryable)` for a bulk response's `items` array.
/// Split out of the upload loop so the retry rule is testable without a cluster.
fn bulk_item_error_counts(items: &[serde_json::Value]) -> (usize, usize) {
    let error_count = items
        .iter()
        .filter(|item| item.get("index").and_then(|idx| idx.get("error")).is_some())
        .count();
    let retryable_count = items.iter().filter(|item| item_is_retryable(item)).count();
    (error_count, retryable_count)
}

/// Only retry a partially-failed bulk when every failure is retryable; a genuine
/// mapping or parse error would otherwise be retried pointlessly to exhaustion,
/// and a mixed batch would resend the good documents alongside the doomed one.
fn batch_is_retryable(error_count: usize, retryable_count: usize) -> bool {
    retryable_count > 0 && retryable_count == error_count
}

/// Delete-index responses worth retrying on a managed cluster: 503 is a busy
/// cluster manager, and HTTP 400 `snapshot_in_progress_exception` is an automated
/// snapshot holding the index. A `cluster_block_exception` body is transient under
/// either status. Everything else — a 400 for any other reason especially — fails
/// immediately rather than burning ten backoffs on a permanent error.
fn delete_index_retryable(status: u16, body: &str) -> bool {
    status == 503
        || (status == 400 && body.contains("snapshot_in_progress"))
        || body.contains("cluster_block_exception")
}

/// Create-index responses worth retrying: a cluster whose manager thread pool is
/// busy answers with `process_cluster_event_timeout_exception` (503) instead of
/// creating the index. A 400 `resource_already_exists_exception` is NOT retryable
/// — it is a real conflict that no amount of waiting clears.
fn create_index_retryable(status: u16, body: &str) -> bool {
    status == 503
        || status == 429
        || body.contains("process_cluster_event_timeout_exception")
        || body.contains("cluster_block_exception")
}

/// Read `collection_params.number_of_shards`, rejecting a present-but-unusable
/// value instead of dropping it.
///
/// `collection_params` is a `#[serde(flatten)]` catch-all, so a mistyped or
/// wrongly-typed key parses cleanly and simply never arrives. Silently falling
/// back to `None` would put the run right back into the "inherits whatever the
/// cluster defaults to" state this setting exists to eliminate — and nothing
/// downstream records the shard count, so the operator would have no way to
/// notice. Failing at construction is the only place it can still be caught.
pub(crate) fn parse_number_of_shards(
    raw: Option<&serde_json::Value>,
) -> Result<Option<i64>, String> {
    match raw {
        None => Ok(None),
        Some(v) => match v.as_i64() {
            Some(n) if n >= 1 => Ok(Some(n)),
            _ => Err(format!(
                "collection_params.number_of_shards must be a positive integer, got {}. \
                 It is not applied unless it parses, and an unapplied shard count \
                 silently inherits the cluster default (1 on open-source OpenSearch, \
                 historically 5 on Amazon OpenSearch Service), which makes the run \
                 incomparable.",
                v
            )),
        },
    }
}

/// Refresh / force-merge responses worth retrying: the back-pressure and gateway
/// statuses, plus the two transient cluster-state exceptions.
fn index_maintenance_retryable(status: u16, body: &str) -> bool {
    http_status_retryable(status)
        || body.contains("cluster_block_exception")
        || body.contains("process_cluster_event_timeout_exception")
}

/// Floor for a single force-merge attempt's request timeout, in seconds (1 hour).
///
/// Sized against the work, not against the other requests: merging a large corpus
/// into one segment rewrites every segment once, so it scales with the corpus and
/// routinely exceeds the 300 s client-wide `OPENSEARCH_TIMEOUT`. See
/// `OpenSearchEngine::force_merge_timeout`.
const FORCE_MERGE_TIMEOUT_SECS: u64 = 3_600;

/// What `OPENSEARCH_FORCE_MERGE_TIMEOUT=0` resolves to: a deadline far enough out
/// that it never fires. `0` conventionally means "no limit", and the transport
/// requires *some* duration, so "no limit" is spelled as a century rather than as
/// `Duration::ZERO` — which reqwest would treat as an already-expired deadline,
/// failing every attempt before a byte left the process and discarding a
/// completed ingest.
const FORCE_MERGE_TIMEOUT_UNLIMITED: std::time::Duration =
    std::time::Duration::from_secs(100 * 365 * 24 * 3_600);

/// Read a seconds-valued environment variable, failing loudly on a value that is
/// present but unusable.
///
/// `.parse().ok()` would silently reinstate the default for `3600s`, `1h` or
/// `-1` — the exact "silently ignored setting" failure `parse_number_of_shards`
/// exists to prevent in this same file. An operator who sets a merge timeout and
/// gets the default instead is worse off than one who gets an error.
///
/// Reads through [`crate::effective_config`] so the resolved value reaches the
/// result JSON (#212). It is the STRICT counterpart of `env_parsed`: this one
/// refuses an unusable value rather than defaulting and recording the
/// divergence. #260 wants the rest of this file's knobs moved onto the same
/// footing, at which point this belongs in `effective_config` as a reusable
/// sibling rather than here.
fn parse_env_secs(var: &str) -> Result<Option<u64>, String> {
    match crate::effective_config::env_var(var) {
        Err(_) => Ok(None),
        Ok(raw) => {
            let parsed = raw.trim().parse::<u64>().map_err(|_| {
                format!(
                    "{} must be a whole number of seconds (got {:?}). Use 0 for no limit; \
                     suffixes like \"1h\" or \"3600s\" are not accepted.",
                    var, raw
                )
            })?;
            crate::effective_config::record_effective(var, parsed);
            Ok(Some(parsed))
        }
    }
}

/// How long one force-merge attempt may block before the transport gives up.
///
/// `OPENSEARCH_TIMEOUT` (300 s) is the *client-wide* transport timeout, sized for
/// queries and bulk requests. Applying it to a merge-to-one-segment of a large
/// corpus guarantees a self-inflicted failure loop: every attempt aborts
/// client-side at 300 s while the merge is still running server-side, the retry
/// re-issues it, and all 11 attempts (10 retries) are spent on a merge that was
/// always going to succeed — discarding the ingest the retries exist to protect.
/// So force-merge gets its own, much larger bound.
///
/// Retrying is cheap when it does happen, measured rather than assumed: the
/// client aborted at 3.0 s, the server kept merging for 108 s (`force_merge` pool
/// `active=1` across 36 consecutive samples), and the re-issued request returned
/// in 0.0 s. The pool is `fixed, size=1, queue_size=-1`, so a retry queues behind
/// the running merge rather than being rejected, and then finds nothing left to
/// do. Retries are therefore not N× the merge cost.
///
/// A free function so the rule is testable without an engine, a cluster, or
/// environment mutation; the value is resolved once in `OpenSearchEngine::new`.
fn resolve_force_merge_timeout(
    client_timeout_secs: u64,
    override_secs: Option<u64>,
) -> std::time::Duration {
    match override_secs {
        // 0 means "no client-side deadline", not "expire immediately".
        Some(0) => FORCE_MERGE_TIMEOUT_UNLIMITED,
        // An explicit override wins outright, including downwards — that is the
        // whole point of an explicit override.
        Some(secs) => std::time::Duration::from_secs(secs),
        // Never *below* the client-wide timeout: an operator who raised
        // OPENSEARCH_TIMEOUT for a big corpus meant it for this call too.
        None => std::time::Duration::from_secs(client_timeout_secs.max(FORCE_MERGE_TIMEOUT_SECS)),
    }
}

/// Wall-clock ceiling on `force_merge` *including* its retries.
///
/// Per-attempt bounds stop bounding anything once the per-attempt bound is an
/// hour: `retry_index_op` treats every transport error as retryable, so 11
/// attempts × 3600 s is an ~11 h worst case for a single call. The search path
/// has carried a wall-clock budget for exactly this reason (`SearchRetryPolicy`);
/// raising the index path's per-attempt bound 12× without one would have left
/// force-merge as the only unbounded operation in the engine.
///
/// Default: twice the per-attempt deadline — room for one full-length attempt
/// plus one full-length retry, and no more. `OPENSEARCH_FORCE_MERGE_BUDGET`
/// overrides it, with `0` meaning unlimited (the same convention as
/// `OPENSEARCH_FORCE_MERGE_TIMEOUT`).
///
/// A free function, like [`resolve_force_merge_timeout`], so the knob has
/// coverage that does not need an engine or a cluster.
fn resolve_force_merge_budget(
    per_attempt: std::time::Duration,
    override_secs: Option<u64>,
) -> Option<std::time::Duration> {
    match override_secs {
        Some(0) => None,
        Some(secs) => Some(std::time::Duration::from_secs(secs)),
        None => Some(per_attempt.saturating_mul(2)),
    }
}

/// Whether a `_cluster/health?wait_for_status=…` response means "settled".
///
/// Both engines this repo tests against answer a missed status with **HTTP 408**
/// and `"timed_out": true` (verified on OpenSearch 3.7.0 and Elasticsearch
/// 9.4.3), so the status range does the work. The `timed_out` check is defence in
/// depth for the 2xx-carrying variant that
/// `cluster.health.return_200_for_cluster_health_timeout` produces on the
/// Elasticsearch versions supporting it (the flag is not recognised on OpenSearch
/// 3.7.0): there, a status-only check would read "never went yellow" as success
/// and start the search phase against a cluster still moving shards.
///
/// An absent `timed_out` is treated as settled — inventing a failure from a
/// missing field would hard-fail a finished ingest over a response-shape change.
fn cluster_health_settled(status: u16, body: &serde_json::Value) -> bool {
    (200..300).contains(&status) && body.get("timed_out").and_then(|v| v.as_bool()) != Some(true)
}

/// What `wait_for_cluster_health` should do with one response.
#[derive(Debug, PartialEq, Eq)]
enum HealthVerdict {
    /// The index reached the requested status; proceed.
    Settled,
    /// We are not allowed to look. Warn and proceed anyway — see
    /// [`is_authorization_denied`] and the note on `wait_for_cluster_health`.
    Unobservable,
    /// Transient; try again.
    Retry,
    /// We genuinely cannot tell whether the cluster settled.
    Failed,
}

/// Decide the fate of one health response.
///
/// Extracted from the retry loop so the ORDER of these checks is testable. The
/// predicates below are each individually correct, but the composition is what
/// decides whether a 403 aborts a completed multi-hour ingest or merely warns —
/// and swapping two lines inside the loop would change that while leaving every
/// predicate test green. This is the only path in the wait that returns
/// "proceed" without a settled cluster, so it is the one that most needs pinning.
fn classify_health_response(status: u16, text: &str) -> HealthVerdict {
    let body = serde_json::from_str::<serde_json::Value>(text).unwrap_or(serde_json::Value::Null);
    if cluster_health_settled(status, &body) {
        return HealthVerdict::Settled;
    }
    // A `timed_out` body means "not settled YET", whatever status carried it.
    // Both shipped engines send it with 408, which `cluster_health_retryable`
    // already covers; this also catches the 2xx-carrying variant that
    // `cluster.health.return_200_for_cluster_health_timeout` produces, which
    // would otherwise fall through to Failed and hard-fail a finished ingest on
    // the one cluster configuration that most obviously means "try again".
    if body.get("timed_out").and_then(|v| v.as_bool()) == Some(true) {
        return HealthVerdict::Retry;
    }
    // Before the retry check: a permission denial is not transient, and burning
    // the whole budget on it delays the run by minutes for nothing.
    if is_authorization_denied(status, text) {
        return HealthVerdict::Unobservable;
    }
    if cluster_health_retryable(status, text) {
        return HealthVerdict::Retry;
    }
    HealthVerdict::Failed
}

/// Health responses worth another attempt: whatever the other index-level ops
/// already retry, plus 408 — the timeout answer itself, which only means the
/// cluster had not settled *yet*.
///
/// Delegating to `index_maintenance_retryable` rather than to
/// `http_status_retryable` is deliberate: it brings in the body-carried transient
/// states (`cluster_block_exception`, `process_cluster_event_timeout_exception`),
/// so a 403 disk-watermark block is retried here exactly as it is everywhere else
/// in this file, instead of being mistaken for an authorization denial.
fn cluster_health_retryable(status: u16, body: &str) -> bool {
    status == 408 || index_maintenance_retryable(status, body)
}

/// Whether a response means "you may not ask", as opposed to "it went wrong".
/// Retrying an authorization decision cannot change it.
///
/// The body matters, not just the status: OpenSearch ships `cluster_block_exception`
/// as **HTTP 403** too, and that one is transient — a disk-watermark read-only
/// block clears itself. Classifying it as an authorization denial would skip the
/// settle check and continue the run, where its three sibling predicates in this
/// file (`delete_index_retryable`, `create_index_retryable`,
/// `index_maintenance_retryable`) all inspect the body and retry it. Matching on
/// status alone would have made this the only status-only classifier here, and
/// the only one that treats a transient block as permanent.
fn is_authorization_denied(status: u16, body: &str) -> bool {
    matches!(status, 401 | 403) && !body.contains("cluster_block_exception")
}

/// Retry budget shared by every index-level operation (create, delete, refresh,
/// force-merge). These run once per config rather than per query, so a patient
/// budget costs nothing on a healthy cluster and is what rides out a snapshot
/// window on a managed one.
fn index_op_policy() -> RetryPolicy {
    RetryPolicy::from_env(
        "OPENSEARCH_INDEX_OP_MAX_RETRIES",
        "OPENSEARCH_INDEX_OP_RETRY_BASE_MS",
        10,
        2_000,
    )
}

/// Search-path retry budget. Unlike the index ops this is bounded by wall-clock
/// as well as by attempt count: an attempt can block for `OPENSEARCH_TIMEOUT`
/// (300 s by default) before the transport gives up, so a count-only bound would
/// let a single stalled query occupy half an hour.
#[derive(Clone, Copy)]
struct SearchRetryPolicy {
    max_retries: u32,
    base_delay_ms: u64,
    budget: std::time::Duration,
}

impl SearchRetryPolicy {
    /// Resolved once per run, in `search()`, NOT inside `knn_send` — an
    /// `env::var` lookup is client CPU work and `knn_send` runs inside the
    /// measured window (redis.rs states the same rule: "resolved once in `new()`;
    /// never re-read from the environment inside a timed window").
    fn from_env() -> Self {
        let base = RetryPolicy::from_env(
            "OPENSEARCH_SEARCH_MAX_RETRIES",
            "OPENSEARCH_SEARCH_RETRY_BASE_MS",
            5,
            50,
        );
        Self {
            max_retries: base.max_retries,
            base_delay_ms: base.base_delay_ms,
            budget: std::time::Duration::from_millis(crate::effective_config::env_parsed(
                "OPENSEARCH_SEARCH_RETRY_BUDGET_MS",
                2_000,
            )),
        }
    }
}

/// Index settings for the benchmark index.
///
/// Shard count is a first-order factor for OpenSearch vector search, not a
/// detail: Amazon OpenSearch Service historically defaulted an index to FIVE
/// primary shards while open-source OpenSearch defaults to one, and an internal
/// 2024 benchmark that tested both found 5 clearly better for precision vs
/// qps/latency/index-time — the 1-shard config "deeply impacts OpenSearch
/// indexing speed and precision". Leaving it unset silently inherits whatever the
/// cluster default happens to be for that version, which makes runs incomparable
/// across versions and can hand OpenSearch the worse of two configurations
/// without anyone choosing it.
///
/// Set from collection_params.number_of_shards; `None` preserves the previous
/// behaviour (cluster default) so existing configs are unaffected.
fn build_index_settings(number_of_shards: Option<i64>) -> serde_json::Value {
    let mut settings = serde_json::json!({
        "knn": true,
        // Indexing-throughput tuning: no replicas and no periodic refresh, since
        // the benchmark bulk-loads all data up front and force-merges before
        // searching.
        "number_of_replicas": 0,
        "refresh_interval": -1,
    });
    if let Some(n) = number_of_shards {
        settings["number_of_shards"] = serde_json::Value::from(n);
    }
    settings
}

/// True when a bulk item failed for a reason worth retrying — the server shedding
/// load rather than rejecting the document itself.
fn item_is_retryable(item: &serde_json::Value) -> bool {
    let Some(idx) = item.get("index") else {
        return false;
    };
    if idx.get("status").and_then(|s| s.as_u64()) == Some(429) {
        return true;
    }
    matches!(
        idx.get("error")
            .and_then(|e| e.get("type"))
            .and_then(|t| t.as_str()),
        Some("es_rejected_execution_exception")
            | Some("circuit_breaking_exception")
            | Some("cluster_block_exception")
    )
}

/// OpenSearch KNN search (different format from Elasticsearch).
/// Uses {"query": {"knn": {"vector": {"vector": [...], "k": top}}}} format.
/// Build the OpenSearch kNN search body.
///
/// Efficient (pre-)filtering: the filter is pushed *inside* the kNN clause so the
/// Lucene engine applies it during graph traversal. Wrapping the kNN query in an
/// outer `bool.must` + `filter` instead performs post-filtering, which collapses
/// recall on filtered datasets (see qdrant/vector-db-benchmark#167). Requires the
/// `lucene` engine, which our index mapping uses (see `configure`).
fn build_knn_body(
    query_vector: &[f32],
    top: usize,
    filter: Option<&serde_json::Value>,
) -> serde_json::Value {
    let mut query = serde_json::json!({
        "knn": {
            "vector": {
                "vector": query_vector,
                "k": top,
            }
        }
    });

    if let Some(f) = filter {
        query["knn"]["vector"]["filter"] = f.clone();
    }

    // Response trimming: the benchmark only needs each hit's id, so skip loading
    // `_source` and stored fields and return `_id` via a doc-value field. This
    // trims the response payload for a fairer QPS/latency measurement.
    serde_json::json!({
        "query": query,
        "size": top,
        "_source": false,
        "docvalue_fields": ["_id"],
        "stored_fields": "_none_",
    })
}

/// Extract the document id from a search hit. With response trimming the id is
/// returned as a doc-value under `fields._id[0]`; fall back to the top-level
/// `_id` for untrimmed responses.
fn hit_id(hit: &serde_json::Value) -> Option<&str> {
    hit.get("fields")
        .and_then(|f| f.get("_id"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .or_else(|| hit.get("_id").and_then(|v| v.as_str()))
}

/// What a `knn_send` call cost beyond the query itself.
struct RetryCost {
    /// Wall-clock spent on backoff sleeps and on attempts that were thrown away.
    /// The caller subtracts this from the elapsed time so a retried query reports
    /// the latency of the attempt that actually produced the result.
    overhead: std::time::Duration,
    /// True when the query needed more than one attempt.
    retried: bool,
}

/// Send a pre-serialized KNN search request and return the DECODED response.
/// The consistent timed boundary (see qdrant/pgvector/redis) is: request body
/// pre-serialized to a `RawValue` OUTSIDE the window (the vector-to-JSON ryu
/// formatting is client CPU work); RPC send + receive + decode-to-structured-
/// response INSIDE the window (this fn: send + status check + wire read +
/// `from_str`); id/score extraction OUTSIDE (`extract_knn_hits`). So the JSON
/// decode is billed as latency exactly like qdrant's protobuf decode.
///
/// Retries preserve that boundary rather than widening it. Everything spent on a
/// discarded attempt — the failed round-trip and the backoff sleep after it — is
/// accumulated into `RetryCost::overhead` and subtracted by the caller, so the
/// recorded sample is still one round-trip plus decode. Without that subtraction a
/// query retried four times would publish `server_latency + 750 ms` of client-side
/// `thread::sleep` as OpenSearch's p99, and OpenSearch is the only engine in the
/// suite that retries on the search path — its closest competitor, elasticsearch.rs,
/// runs the identical un-retried code — so an OS-vs-ES tail comparison would stop
/// measuring the same quantity.
fn knn_send(
    rt: &tokio::runtime::Runtime,
    client: &OpenSearch,
    index_name: &str,
    raw_body: &serde_json::value::RawValue,
    policy: SearchRetryPolicy,
) -> Result<(serde_json::Value, RetryCost), String> {
    // Search-path back-pressure matters because the caller does not fail on it: a
    // failed query is logged and DROPPED from the timing and recall vectors, so
    // recall ends up averaged over the queries that survived. Since 429s arrive
    // precisely when the node is loaded, the survivors are the cheaper queries and
    // recall is biased UPWARD. Measured: a glove-100 run shed 17,584 queries across
    // two of six configs. The loss is recorded (`SearchResults::failed_queries`);
    // retrying is what keeps it near zero in the first place.
    //
    // Retries are kept short — a query is on the latency critical path — and are
    // bounded by a total budget, not just a count, because a transport timeout can
    // itself take OPENSEARCH_TIMEOUT (300 s by default). Without the budget, five
    // retries of a stalled query is 30 minutes for one sample.
    let deadline = std::time::Instant::now() + policy.budget;
    let mut overhead = std::time::Duration::ZERO;
    let mut attempt: u32 = 0;

    let resp = loop {
        let attempt_start = std::time::Instant::now();
        let outcome = match rt.block_on(
            client
                .search(SearchParts::Index(&[index_name]))
                .body(raw_body)
                .send(),
        ) {
            Ok(r) => {
                let status = r.status_code().as_u16();
                if r.status_code().is_success() {
                    break r;
                }
                // Drain the body before retrying: it carries the diagnostic, and an
                // undrained response cannot go back to the connection pool, so every
                // retry against a TLS-terminated domain would pay a fresh handshake
                // during exactly the window where the server is already struggling.
                let text = rt.block_on(r.text()).unwrap_or_default();
                if !http_status_retryable(status) {
                    return Err(format!(
                        "KNN search error (HTTP {}, {} retries): {}",
                        status, attempt, text
                    ));
                }
                format!("HTTP {}: {}", status, text)
            }
            Err(e) => format!("transport error: {}", e),
        };

        // The discarded attempt is overhead, not latency.
        overhead += attempt_start.elapsed();

        if attempt >= policy.max_retries || std::time::Instant::now() >= deadline {
            return Err(format!(
                "KNN search failed after {} retries ({:.1?} of retry budget): {}",
                attempt, policy.budget, outcome
            ));
        }
        overhead += backoff_sleep(policy.base_delay_ms, attempt);
        attempt += 1;
    };

    let text = rt
        .block_on(resp.text())
        .map_err(|e| format!("Failed to read search response: {}", e))?;
    let decoded: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("Failed to parse search response: {}", e))?;
    Ok((
        decoded,
        RetryCost {
            overhead,
            retried: attempt > 0,
        },
    ))
}

/// Extract the id/score list from an already-decoded response (done AFTER the
/// timed window — only pulling final ids out of the decoded struct for recall,
/// mirroring pgvector/qdrant).
fn extract_knn_hits(resp_body: &serde_json::Value) -> Result<Vec<(i64, f64)>, String> {
    let hits = resp_body
        .get("hits")
        .and_then(|h| h.get("hits"))
        .and_then(|h| h.as_array())
        .ok_or_else(|| "Missing hits.hits in search response".to_string())?;

    let mut results = Vec::with_capacity(hits.len());
    for hit in hits {
        let id_hex = hit_id(hit).ok_or_else(|| "Missing _id in hit".to_string())?;
        let score = hit.get("_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let id = uuid_hex_to_int(id_hex)?;
        results.push((id, score));
    }

    Ok(results)
}

// ── Engine trait implementation ──────────────────────────────────────────

impl Engine for OpenSearchEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        // Shard count is printed alongside the HNSW knobs so a sweep point is
        // honestly labelled rather than silently measured at whatever the cluster
        // defaults to (same fairness gate as vertex.rs's effective-knob line).
        println!(
            "OpenSearch: HNSW {{ m: {}, ef_construction: {} }}, number_of_shards: {}",
            self.config.m,
            self.config.ef_construction,
            self.config
                .number_of_shards
                .map(|n| n.to_string())
                .unwrap_or_else(|| "cluster-default".to_string()),
        );

        println!("Ensuring index does not exist...");
        self.delete_index()?;

        println!("Creating index '{}'...", self.index_name);
        self.create_index(dataset)?;
        println!("Index '{}' created successfully.", self.index_name);

        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let normalize = dataset.needs_normalization();
        let dataset_path = dataset.get_path()?;
        println!("Reading dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        let (ids, vectors, metadata) = dataset.read_vectors(normalize)?;
        let read_time = read_start.elapsed().as_secs_f64();

        println!(
            "Read {} vectors ({}d) in {:.3}s",
            vectors.len(),
            vectors.first().map(|v| v.len()).unwrap_or(0),
            read_time,
        );

        println!(
            "Starting upload with {} threads, batch size {}...",
            self.config.parallel, self.config.batch_size
        );
        let upload_start = Instant::now();
        self.upload_parallel(&ids, &vectors, &metadata)?;
        let upload_time = upload_start.elapsed().as_secs_f64();

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        // Force merge, which flushes and then refreshes — see `force_merge`; the
        // merge is invisible to queries until that refresh, so it is also what
        // makes the documents searchable here.
        //
        // There is deliberately no refresh BEFORE the merge, so that each engine
        // spends exactly ONE refresh inside its timed index phase:
        //
        //   elasticsearch.rs   refresh -> force_merge(1)
        //   opensearch.rs      force_merge(1) -> refresh
        //
        // Since #248 both engines disable periodic refresh during upload
        // (`refresh_interval: -1`) and both need one explicit refresh to make the
        // result searchable. Only the placement differs, and it has to: an
        // Elasticsearch force merge swaps the external searcher onto the merged
        // segment, an OpenSearch one does not (measured — ES 9.4.3 reports
        // 1 segment / 1 searchable straight after the merge; OpenSearch 3.7.0
        // reports 4 / 3, with the merged segment at `"search": false`). So on
        // OpenSearch the refresh MUST follow the merge.
        //
        // An earlier revision of this comment argued the pre-merge refresh should
        // be kept because #248 gave Elasticsearch one too, making the engines
        // symmetric. That reasoning misses the refresh `force_merge` now performs
        // on its way out: keeping a pre-merge refresh as well would bill
        // OpenSearch TWO refreshes against Elasticsearch's one, re-opening the
        // asymmetry rather than closing it. Measured cost of the redundant
        // refresh on 1.18M docs: ~1 s.
        //
        // Include the merge time in total_time for cross-engine comparability
        // (mirrors mongodb; matches v0's post_upload() timing). Pinning the merge
        // to one segment makes this phase much longer than it used to be (#210).
        let index_start = Instant::now();
        self.force_merge()?;
        let index_time = index_start.elapsed().as_secs_f64();

        let total_time = read_time + upload_time + index_time;
        println!(
            "Index time: {:.3}s, Total time (read+upload+index): {:.3}s",
            index_time, total_time
        );

        Ok(UploadStats {
            upload_time,
            total_time,
            upload_count: vectors.len(),
            parallel: self.config.parallel,
            batch_size: self.config.batch_size,
            memory_usage: None,
        })
    }

    fn search(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        let parallel = params.parallel.unwrap_or(1) as usize;

        // Apply search-time settings (ef_search)
        self.setup_search(params)?;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<QueryFilter<serde_json::Value>> =
            conditions.resolve_all("OpenSearch", parse_os_conditions)?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute per-query `top` and the fully serialized request bodies
        // BEFORE the parallel region so the timed window wraps only the RPC
        // round-trip. `build_knn_body` builds the DOM and `to_raw_value`
        // performs the vector-to-JSON ryu formatting (client CPU work) once here;
        // the timed send only copies the already-formatted bytes. `tops[idx]`
        // reproduces the same k the request embeds, so recall is unchanged.
        let tops: Vec<usize> = (0..num_to_run)
            .map(|idx| {
                explicit_top.unwrap_or_else(|| {
                    let n = neighbors[idx].len();
                    if n > 0 {
                        n
                    } else {
                        10
                    }
                })
            })
            .collect();
        let raw_bodies: Vec<Box<serde_json::value::RawValue>> = (0..num_to_run)
            .map(|idx| {
                let body = build_knn_body(&queries[idx], tops[idx], parsed_filters[idx].as_ref());
                serde_json::value::to_raw_value(&body).expect("serialize KNN search body")
            })
            .collect();

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));
        // Queries that needed more than one attempt. Their backoff is excluded from
        // the latency samples (see `knn_send`), so without this the reported figures
        // would look clean with no trace that the server was shedding load.
        let retried_queries = Arc::new(AtomicUsize::new(0));
        // Resolved here rather than per query: `knn_send` runs inside the measured
        // window, where an env lookup is avoidable client CPU work.
        let search_policy = SearchRetryPolicy::from_env();

        let pb = self.create_progress_bar(num_to_run);
        let base_url = self.base_url.clone();
        let timeout = self.timeout;
        let index_name = self.index_name.clone();

        // Barrier-synchronized start so connection setup AND the cold first query
        // fall OUTSIDE the measured window (mirrors the Redis/Vertex engines).
        // Every worker builds its runtime + client and primes with one discarded
        // query, then blocks on `ready`; the main thread stamps the shared start
        // instant into `start_cell` and releases `go`, so the measurement clock
        // starts only once all workers are warm and poised. A worker that fails to
        // set up MUST still pass both barriers before returning, or the run would
        // deadlock.
        let ready = Arc::new(std::sync::Barrier::new(parallel + 1));
        let go = Arc::new(std::sync::Barrier::new(parallel + 1));
        let start_cell = Arc::new(std::sync::OnceLock::<Instant>::new());

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let base_url = base_url.clone();
                let index_name = index_name.clone();
                let neighbors = &neighbors;
                let tops = &tops;
                let raw_bodies = &raw_bodies;
                let query_idx = Arc::clone(&query_idx);
                let ready = Arc::clone(&ready);
                let go = Arc::clone(&go);
                let retried_queries = Arc::clone(&retried_queries);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();

                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(_) => {
                            // Still cross both barriers so peers aren't stranded.
                            ready.wait();
                            go.wait();
                            return (t, p, r, mr, nd);
                        }
                    };
                    let client = match create_os_client(&base_url, timeout) {
                        Ok(c) => c,
                        Err(_) => {
                            ready.wait();
                            go.wait();
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Prime this connection with ONE discarded query so the cold
                    // first round-trip is not inside the measured window. Best
                    // effort: errors are ignored and its sample is NOT recorded.
                    if num_to_run > 0 {
                        let _ = knn_send(&rt, &client, &index_name, &raw_bodies[0], search_policy);
                    }

                    // Signal "connected + primed", then block until the main thread
                    // stamps the shared measurement start and releases everyone.
                    ready.wait();
                    go.wait();

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }

                        let top = tops[idx];

                        // Timed window: network send + receive + decode of the
                        // response into a structured value. Body is pre-serialized
                        // (out); id/score extraction runs after `elapsed` (out).
                        let query_start = Instant::now();
                        let response =
                            knn_send(&rt, &client, &index_name, &raw_bodies[idx], search_policy);
                        let elapsed = query_start.elapsed();

                        match response.and_then(|(resp_body, cost)| {
                            extract_knn_hits(&resp_body).map(|hits| (hits, cost))
                        }) {
                            Ok((result_ids, cost)) => {
                                // Bill only the attempt that produced the result:
                                // discarded round-trips and backoff sleeps are
                                // client-side cost, not OpenSearch latency.
                                let query_time =
                                    elapsed.saturating_sub(cost.overhead).as_secs_f64();
                                if cost.retried {
                                    retried_queries.fetch_add(1, Ordering::Relaxed);
                                }
                                let ordered_ids: Vec<i64> =
                                    result_ids.iter().map(|(id, _)| *id).collect();
                                let m = crate::metrics::compute_metrics(
                                    &ordered_ids,
                                    &neighbors[idx],
                                    top,
                                );
                                t.push(query_time);
                                p.push(m.precision);
                                r.push(m.recall);
                                mr.push(m.mrr);
                                nd.push(m.ndcg);
                            }
                            Err(e) => {
                                // Not counted here: `compute_search_stats` derives
                                // failed_queries as requested - succeeded, uniformly
                                // for every engine.
                                eprintln!("Search query {} failed: {}", idx, e);
                            }
                        }
                        pb.inc(1);
                    }
                    (t, p, r, mr, nd)
                }));
            }

            // All workers are connected + primed and blocked on `go`. Stamp the
            // shared measurement start and release them simultaneously. The cold
            // setup is already behind the barrier.
            ready.wait();
            let st = Instant::now();
            start_cell.set(st).ok();
            go.wait();

            for h in handles {
                let (t, p, r, mr, nd) = h.join().unwrap();
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrr_vals.extend(mr);
                ndcg_vals.extend(nd);
            }
        });

        pb.finish_and_clear();
        // Measure from the post-barrier start stamp (workers already primed), so
        // total_time excludes connection setup and the cold first query.
        let total_time = start_cell
            .get()
            .map(|st| st.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        // Dropped queries are NOT refused here. `compute_search_stats` derives
        // `failed_queries` as `requested_queries - times.len()` for every engine,
        // experiment.rs prints the ⚠ warning and persists the count to the results
        // JSON, and `--fail-on-dropped-queries` makes it fatal across all engines
        // at once. An engine-local refusal would be redundant with that, would make
        // OpenSearch the only one of 15 engines that reports nothing instead of
        // reporting a partial run with its overload signal intact, and — because a
        // refused rep is skipped by the `--repetitions` loop — would silently
        // select the rep that happened not to hit back-pressure.
        //
        // Queries that succeeded only after a retry are surfaced separately: they
        // are not losses, but they mean the server was shedding, which the clean
        // latency figures no longer show now that the backoff is subtracted out.
        let retried = retried_queries.load(Ordering::Relaxed);
        if retried > 0 {
            eprintln!(
                "⚠  {} of {} search queries succeeded only after a retry ({:.2}%) — the server \
                 was shedding load; backoff is excluded from the reported latency",
                retried,
                num_to_run,
                100.0 * retried as f64 / num_to_run.max(1) as f64
            );
        }

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )
    }

    fn delete(&mut self) -> Result<(), String> {
        self.delete_index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Load-bearing: the hoisted request body is `to_raw_value(&build_knn_body())`.
    /// Its verbatim bytes (what JsonBody writes on the wire) must equal the bytes
    /// the old inline `.body(build_knn_body())` path serialized via `to_vec`.
    #[test]
    fn build_knn_body_raw_value_roundtrips_to_wire_bytes() {
        let vec = vec![0.1f32, -0.2, 0.3];
        let top = 2usize;
        let filter = json!({"term": {"color": "red"}});

        let body = build_knn_body(&vec, top, Some(&filter));
        let to_vec_bytes = serde_json::to_vec(&body).unwrap();
        let raw = serde_json::value::to_raw_value(&body).unwrap();
        assert_eq!(raw.get().as_bytes(), to_vec_bytes.as_slice());

        // Unfiltered variant.
        let body_nf = build_knn_body(&vec, top, None);
        let raw_nf = serde_json::value::to_raw_value(&body_nf).unwrap();
        assert_eq!(
            raw_nf.get().as_bytes(),
            serde_json::to_vec(&body_nf).unwrap().as_slice()
        );
    }

    #[test]
    fn test_match_any_string_list_emits_terms() {
        let c = build_filter("color", "match", &json!({"any": ["red", "blue"]})).unwrap();
        assert_eq!(c, json!({"terms": {"color": ["red", "blue"]}}));
    }

    #[test]
    fn test_match_any_int_list_emits_terms() {
        let c = build_filter("size", "match", &json!({"any": [1, 2, 3]})).unwrap();
        assert_eq!(c, json!({"terms": {"size": [1, 2, 3]}}));
    }

    #[test]
    fn test_match_any_empty_list_matches_nothing() {
        // Empty IN-set must match NOTHING (never invert to match-all): `terms: []`.
        let c = build_filter("color", "match", &json!({"any": []})).unwrap();
        assert_eq!(c, json!({"terms": {"color": []}}));
    }

    #[test]
    fn test_match_exact_value_still_works() {
        let c = build_filter("color", "match", &json!({"value": "red"})).unwrap();
        assert_eq!(c, json!({"match": {"color": "red"}}));
    }

    // #121: a non-scalar `value` (a JSON array) is malformed input — the
    // canonical model uses `match.any` for lists. It must be dropped (None),
    // not forwarded verbatim as `{"match":{"n":[1,2]}}`. Matches qdrant/redis/
    // valkey/vectorsets.
    #[test]
    fn test_match_non_scalar_value_dropped() {
        assert_eq!(build_filter("n", "match", &json!({"value": [1, 2]})), None);
        assert_eq!(
            build_filter("n", "match", &json!({"value": {"x": 1}})),
            None
        );
        assert_eq!(build_filter("n", "match", &json!({"value": null})), None);
        // As the sole clause, the whole filter is dropped (no `bool.must`).
        let conditions = json!({"and": [{"n": {"match": {"value": [1, 2]}}}]});
        assert_eq!(parse_os_conditions(&conditions), None);
    }

    // #120: a full-text `{"match": {"text": …}}` condition must emit an analyzed
    // `match` query — NOT be dropped. Dropping it leaves `bool.must:[]`, which
    // OpenSearch treats as match-ALL, silently running the kNN query UNFILTERED.
    #[test]
    fn test_match_text_emits_match_query() {
        let c = build_filter("body", "match", &json!({"text": "quick"})).unwrap();
        assert_eq!(c, json!({"match": {"body": "quick"}}));
    }

    #[test]
    fn os_text_only_condition_not_dropped() {
        // `{"and":[{"body":{"match":{"text":"quick"}}}]}` — the exact fixture
        // condition from `write_fulltext_project`. Must yield a non-empty
        // `bool.must` containing the `match` clause (was None → dropped → #120).
        let conditions = json!({"and": [{"body": {"match": {"text": "quick"}}}]});
        let parsed =
            parse_os_conditions(&conditions).expect("text-only filter must not be dropped");
        assert_eq!(
            parsed,
            json!({"bool": {"must": [{"match": {"body": "quick"}}]}})
        );
    }

    // Regression for qdrant/vector-db-benchmark#167: the filter must land inside
    // the kNN clause (efficient filtering), not in an outer bool wrapper
    // (post-filtering), otherwise filtered-search recall collapses.
    #[test]
    fn knn_filter_is_pushed_inside_knn_clause() {
        let filter = json!({"bool": {"must": [{"match": {"a": 1}}]}});
        let body = build_knn_body(&[0.1, 0.2, 0.3], 10, Some(&filter));

        // Filter lives at query.knn.vector.filter ...
        assert_eq!(body["query"]["knn"]["vector"]["filter"], filter);
        // ... and there is no post-filtering bool wrapper around the kNN query.
        assert!(
            body["query"].get("bool").is_none(),
            "kNN query must not be wrapped in an outer bool (post-filtering)"
        );
        assert_eq!(body["query"]["knn"]["vector"]["k"], 10);
        assert_eq!(body["size"], 10);
    }

    #[test]
    fn knn_body_without_filter_has_no_filter_key() {
        let body = build_knn_body(&[0.1, 0.2], 5, None);
        assert!(body["query"]["knn"]["vector"].get("filter").is_none());
    }

    #[test]
    fn knn_body_trims_the_response() {
        // Response trimming: no _source, ids via doc-value, no stored fields.
        let body = build_knn_body(&[0.1, 0.2], 5, None);
        assert_eq!(body["_source"], serde_json::json!(false));
        assert_eq!(body["docvalue_fields"], serde_json::json!(["_id"]));
        assert_eq!(body["stored_fields"], serde_json::json!("_none_"));
    }

    #[test]
    fn hit_id_reads_docvalue_then_falls_back() {
        // Trimmed response: id under fields._id[0].
        let trimmed = serde_json::json!({"fields": {"_id": ["deadbeef"]}, "_score": 1.0});
        assert_eq!(hit_id(&trimmed), Some("deadbeef"));
        // Untrimmed response: top-level _id.
        let plain = serde_json::json!({"_id": "cafef00d", "_score": 1.0});
        assert_eq!(hit_id(&plain), Some("cafef00d"));
        // Missing id.
        assert_eq!(hit_id(&serde_json::json!({"_score": 1.0})), None);
    }

    #[test]
    fn or_conditions_require_minimum_should_match() {
        let conditions = json!({"or": [{"a": {"match": {"value": 1}}}]});
        let parsed = parse_os_conditions(&conditions).expect("should parse");
        let bool_query = &parsed["bool"];

        // OR filters must actually restrict results, not just contribute score.
        assert_eq!(bool_query["minimum_should_match"], 1);
        assert!(bool_query["should"].as_array().unwrap().len() == 1);
        // No empty `must` array should be emitted for an OR-only filter.
        assert!(bool_query.get("must").is_none());
    }

    #[test]
    fn and_only_conditions_omit_should() {
        let conditions = json!({"and": [{"a": {"match": {"value": 1}}}]});
        let parsed = parse_os_conditions(&conditions).expect("should parse");
        let bool_query = &parsed["bool"];

        assert!(bool_query["must"].as_array().unwrap().len() == 1);
        assert!(bool_query.get("should").is_none());
        assert!(bool_query.get("minimum_should_match").is_none());
    }

    #[test]
    fn empty_conditions_return_none() {
        assert!(parse_os_conditions(&json!({})).is_none());
        // Present-but-empty sub-arrays should not produce a filter either.
        assert!(parse_os_conditions(&json!({"and": [], "or": []})).is_none());
    }

    // Nested/grouped boolean filter: `(color==red AND size>=50) OR (color==blue
    // AND size<10)`. Each OR arm must become its OWN nested `{bool:{must:[...]}}`
    // inside the outer `should`, NOT be flattened into one flat clause list.
    #[test]
    fn nested_group_nests_native_bool() {
        let conditions = json!({ "or": [
            { "and": [
                { "color": { "match": { "value": "red" } } },
                { "size": { "range": { "gte": 50 } } },
            ] },
            { "and": [
                { "color": { "match": { "value": "blue" } } },
                { "size": { "range": { "lt": 10 } } },
            ] },
        ] });
        let parsed = parse_os_conditions(&conditions).expect("nested filter must parse");
        assert_eq!(
            parsed,
            json!({ "bool": {
                "minimum_should_match": 1,
                "should": [
                    { "bool": { "must": [
                        { "match": { "color": "red" } },
                        { "range": { "size": { "gte": 50 } } },
                    ] } },
                    { "bool": { "must": [
                        { "match": { "color": "blue" } },
                        { "range": { "size": { "lt": 10 } } },
                    ] } },
                ],
            } })
        );
    }

    #[test]
    fn and_or_combined_keeps_both_and_min_should() {
        let conditions = json!({
            "and": [{"a": {"match": {"value": 1}}}],
            "or": [{"b": {"match": {"value": 2}}}],
        });
        let parsed = parse_os_conditions(&conditions).expect("should parse");
        let bool_query = &parsed["bool"];
        assert_eq!(bool_query["must"].as_array().unwrap().len(), 1);
        assert_eq!(bool_query["should"].as_array().unwrap().len(), 1);
        assert_eq!(bool_query["minimum_should_match"], 1);
    }

    // ── Range operators ────────────────────────────────────────────────────

    #[test]
    fn range_lt_lte_gt_gte_map_to_os_range() {
        assert_eq!(
            build_filter("n", "range", &json!({"lt":5})).unwrap(),
            json!({"range":{"n":{"lt":5}}})
        );
        assert_eq!(
            build_filter("n", "range", &json!({"lte":5})).unwrap(),
            json!({"range":{"n":{"lte":5}}})
        );
        assert_eq!(
            build_filter("n", "range", &json!({"gt":5})).unwrap(),
            json!({"range":{"n":{"gt":5}}})
        );
        assert_eq!(
            build_filter("n", "range", &json!({"gte":5})).unwrap(),
            json!({"range":{"n":{"gte":5}}})
        );
    }

    #[test]
    fn range_two_sided_keeps_both_bounds() {
        assert_eq!(
            build_filter("n", "range", &json!({"gte":10,"lt":20})).unwrap(),
            json!({"range":{"n":{"gte":10,"lt":20}}})
        );
    }

    #[test]
    fn range_unknown_op_yields_empty_range_object() {
        assert_eq!(
            build_filter("n", "range", &json!({"foo":5})).unwrap(),
            json!({"range":{"n":{}}})
        );
    }

    #[test]
    fn range_null_bound_is_skipped() {
        assert_eq!(
            build_filter("n", "range", &json!({"gte":serde_json::Value::Null})).unwrap(),
            json!({"range":{"n":{}}})
        );
    }

    // ── Geo filter ─────────────────────────────────────────────────────────

    #[test]
    fn geo_with_radius_emits_geo_distance() {
        assert_eq!(
            build_filter("loc", "geo", &json!({"lat":20.0,"lon":10.0,"radius":500})).unwrap(),
            json!({"geo_distance":{"distance":"500m","loc":{"lat":20.0,"lon":10.0}}})
        );
    }

    #[test]
    fn geo_without_radius_uses_default_1000m() {
        assert_eq!(
            build_filter("loc", "geo", &json!({"lat":20.0,"lon":10.0})).unwrap(),
            json!({"geo_distance":{"distance":"1000m","loc":{"lat":20.0,"lon":10.0}}})
        );
    }

    #[test]
    fn geo_missing_lat_or_lon_is_none() {
        assert!(build_filter("loc", "geo", &json!({"lon":10.0,"radius":5})).is_none());
        assert!(build_filter("loc", "geo", &json!({"lat":20.0,"radius":5})).is_none());
    }

    // ── Distance-metric mapping ────────────────────────────────────────────

    #[test]
    fn os_space_type_covers_all_arms() {
        assert_eq!(os_space_type("l2").unwrap(), "l2");
        assert_eq!(os_space_type("euclidean").unwrap(), "l2");
        assert_eq!(os_space_type("cosine").unwrap(), "cosinesimil");
        assert_eq!(os_space_type("angular").unwrap(), "cosinesimil");
        assert_eq!(os_space_type("COSINE").unwrap(), "cosinesimil");
        assert!(os_space_type("dot").is_err());
        assert!(os_space_type("ip").is_err());
        assert!(os_space_type("nope").is_err());
    }

    // ── Exact-match numeric / bool / non-scalar arms ───────────────────────

    #[test]
    fn exact_match_int_float_bool_pass_through_match() {
        assert_eq!(
            build_filter("n", "match", &json!({"value":5})).unwrap(),
            json!({"match":{"n":5}})
        );
        assert_eq!(
            build_filter("n", "match", &json!({"value":1.5})).unwrap(),
            json!({"match":{"n":1.5}})
        );
        assert_eq!(
            build_filter("flag", "match", &json!({"value":true})).unwrap(),
            json!({"match":{"flag":true}})
        );
    }

    #[test]
    fn exact_match_array_value_is_none() {
        // #121: OS build_filter now guards the scalar exact-match arm; a
        // non-scalar value is dropped (None), matching qdrant/redis/valkey/
        // vectorsets (was previously forwarded verbatim as `{"match":{"n":[1,2]}}`).
        assert_eq!(build_filter("n", "match", &json!({"value":[1,2]})), None);
    }

    // ── uuid_hex_to_int round-trip + invalid input ─────────────────────────
    #[test]
    fn uuid_hex_to_int_round_trips_with_id_to_uuid_hex() {
        for id in [0i64, 1, 255, 12345, 9_999_999] {
            let hex = id_to_uuid_hex(id);
            assert_eq!(uuid_hex_to_int(&hex).unwrap(), id, "round-trip id={}", id);
        }
    }

    #[test]
    fn uuid_hex_to_int_rejects_invalid_hex() {
        let err = uuid_hex_to_int("not-a-uuid").unwrap_err();
        assert!(
            err.starts_with("Invalid UUID hex 'not-a-uuid':"),
            "err={}",
            err
        );
    }

    // ── extract_knn_hits: happy path + missing-field errors ────────────────
    #[test]
    fn extract_knn_hits_reads_id_and_score() {
        // Trimmed hits carry the id under fields._id[0].
        let body = json!({
            "hits": {"hits": [
                {"fields": {"_id": [id_to_uuid_hex(7)]}, "_score": 0.9},
                {"fields": {"_id": [id_to_uuid_hex(3)]}, "_score": 0.5},
            ]}
        });
        assert_eq!(extract_knn_hits(&body).unwrap(), vec![(7, 0.9), (3, 0.5)]);
    }

    #[test]
    fn extract_knn_hits_missing_hits_hits_errors() {
        let body = json!({"hits": {"total": 0}});
        assert_eq!(
            extract_knn_hits(&body).unwrap_err(),
            "Missing hits.hits in search response"
        );
    }

    #[test]
    fn extract_knn_hits_missing_id_errors() {
        let body = json!({"hits": {"hits": [{"_score": 0.9}]}});
        assert_eq!(extract_knn_hits(&body).unwrap_err(), "Missing _id in hit");
    }

    #[test]
    fn extract_knn_hits_missing_score_defaults_zero() {
        let body = json!({"hits": {"hits": [{"_id": id_to_uuid_hex(4)}]}});
        assert_eq!(extract_knn_hits(&body).unwrap(), vec![(4, 0.0)]);
    }

    // ── resolve_index_space_type: distance mapping + rejections ────────────
    #[test]
    fn resolve_index_space_type_maps_and_rejects() {
        assert_eq!(
            resolve_index_space_type("cosine", 128).unwrap(),
            "cosinesimil"
        );
        assert_eq!(resolve_index_space_type("l2", 2048).unwrap(), "l2");
        assert_eq!(
            resolve_index_space_type("dot", 128).unwrap_err(),
            "OpenSearch does not support DOT product distance"
        );
        assert!(resolve_index_space_type("ip", 128).is_err());
        assert_eq!(
            resolve_index_space_type("cosine", 4096).unwrap_err(),
            "OpenSearch does not support vector_size > 2048 (got 4096)"
        );
        assert!(resolve_index_space_type("nope", 128).is_err());
    }

    #[test]
    fn item_level_rejections_are_retryable_but_real_errors_are_not() {
        // Load shedding shows up two ways: an explicit 429 status on the item...
        assert!(item_is_retryable(&json!({"index": {"status": 429}})));
        // ...or the exception type, which is what OpenSearch actually returns
        // when the write queue is full or the circuit breaker trips.
        for t in [
            "es_rejected_execution_exception",
            "circuit_breaking_exception",
            "cluster_block_exception",
        ] {
            assert!(
                item_is_retryable(&json!({"index": {"status": 503, "error": {"type": t}}})),
                "{t} should be retryable"
            );
        }
        // A bad document is NOT retryable — resending it forever cannot help.
        assert!(!item_is_retryable(
            &json!({"index": {"status": 400, "error": {"type": "mapper_parsing_exception"}}})
        ));
        assert!(!item_is_retryable(&json!({"index": {"status": 201}})));
        assert!(!item_is_retryable(&json!({})));
    }

    #[test]
    fn backoff_grows_then_caps() {
        // Calls the real `backoff_delay_ms` — re-deriving the formula here would
        // pass no matter what the production code did.
        //
        // Guards the shift: attempt is clamped so 1u64 << attempt cannot overflow,
        // and the delay is capped so a long stall cannot sleep for hours.
        assert_eq!(backoff_delay_ms(500, 0), 500);
        assert_eq!(backoff_delay_ms(500, 1), 1_000, "must double");
        assert_eq!(backoff_delay_ms(500, 3), 4_000);
        assert_eq!(backoff_delay_ms(500, 6), 30_000);
        assert_eq!(
            backoff_delay_ms(500, 50),
            30_000,
            "must not overflow or exceed the cap"
        );
        // `u32::MAX` is what a caller looping forever eventually reaches; the
        // clamp must survive it rather than panicking on shift overflow.
        assert_eq!(backoff_delay_ms(500, u32::MAX), 30_000);
        // The cap is absolute, not relative to the base: an index-op base of 2s
        // still tops out at 30s, and a base already past the cap is clamped.
        assert_eq!(backoff_delay_ms(2_000, 6), 30_000);
        assert_eq!(backoff_delay_ms(60_000, 0), 30_000);
        // A zero base disables sleeping entirely (used by the tests below).
        assert_eq!(backoff_delay_ms(0, 4), 0);
    }

    #[test]
    fn http_status_retryable_covers_backpressure_and_gateway_only() {
        // Back-pressure and transient gateway failures: retry.
        for s in [429u16, 502, 503, 504] {
            assert!(http_status_retryable(s), "HTTP {s} must be retried");
        }
        // Everything else is the server's final answer. 400/404 are the dangerous
        // ones: retrying a malformed query or a missing index just burns the
        // retry budget and delays the real error.
        for s in [200u16, 201, 400, 401, 403, 404, 409, 413, 500, 501] {
            assert!(!http_status_retryable(s), "HTTP {s} must NOT be retried");
        }
    }

    #[test]
    fn bulk_retry_hint_separates_oversized_request_from_load_shedding() {
        // 502/504/413 mean the request itself was probably too big — telling the
        // user to lower `parallel` there would be actively misleading advice.
        for s in [502u16, 504, 413] {
            assert!(
                bulk_retry_hint(s).contains("batch_size"),
                "HTTP {s} should point at request size"
            );
        }
        for s in [429u16, 503] {
            assert!(
                bulk_retry_hint(s).contains("parallel"),
                "HTTP {s} should point at load shedding"
            );
        }
    }

    #[test]
    fn bulk_item_error_counts_separates_failures_from_retryable_failures() {
        let items = vec![
            json!({"index": {"status": 201}}),
            json!({"index": {"status": 429, "error": {"type": "es_rejected_execution_exception"}}}),
            json!({"index": {"status": 400, "error": {"type": "mapper_parsing_exception"}}}),
        ];
        // Two failed, one of them retryable — the success is counted as neither.
        assert_eq!(bulk_item_error_counts(&items), (2, 1));
        // An empty/absent `items` array must not be read as "everything failed".
        assert_eq!(bulk_item_error_counts(&[]), (0, 0));

        // The two counts read DIFFERENT fields: `error_count` needs an `error`
        // object, `item_is_retryable` also accepts a bare 429 status. OpenSearch
        // always sends `error` on a failed item, so the counts agree in practice
        // — but if it ever did not, `retryable > errors` falls through to the
        // failure path rather than retrying, which is the safe direction.
        let bare_429 = vec![json!({"index": {"status": 429}})];
        assert_eq!(bulk_item_error_counts(&bare_429), (0, 1));
        let (e, r) = bulk_item_error_counts(&bare_429);
        assert!(!batch_is_retryable(e, r));
    }

    #[test]
    fn bulk_batch_retries_only_when_every_failure_is_retryable() {
        // All-retryable: the server is shedding load, resending can work.
        assert!(batch_is_retryable(3, 3));
        // Mixed batch: one poison document would be resent forever alongside the
        // rejected ones, so the batch fails fast instead.
        assert!(!batch_is_retryable(3, 2));
        // No retryable failures at all — a pure mapping/parse error.
        assert!(!batch_is_retryable(3, 0));
        // `errors: true` with nothing recognisable in `items` (a shape we did not
        // anticipate) must fail rather than loop to exhaustion.
        assert!(!batch_is_retryable(0, 0));
    }

    #[test]
    fn bulk_response_shapes_drive_the_retry_decision_end_to_end() {
        // The two shapes that matter, wired through both helpers exactly as the
        // upload loop wires them.
        let shed = vec![
            json!({"index": {"status": 429, "error": {"type": "es_rejected_execution_exception"}}}),
            json!({"index": {"status": 503, "error": {"type": "circuit_breaking_exception"}}}),
        ];
        let (e, r) = bulk_item_error_counts(&shed);
        assert!(batch_is_retryable(e, r), "all-shed batch must retry");

        let mixed = vec![
            json!({"index": {"status": 429, "error": {"type": "es_rejected_execution_exception"}}}),
            json!({"index": {"status": 400, "error": {"type": "mapper_parsing_exception"}}}),
        ];
        let (e, r) = bulk_item_error_counts(&mixed);
        assert!(!batch_is_retryable(e, r), "mixed batch must NOT retry");
    }

    #[test]
    fn delete_index_retries_snapshots_and_cluster_blocks_only() {
        // The observed managed-cluster failure: an automated snapshot holds the
        // index and the delete comes back 400.
        assert!(delete_index_retryable(
            400,
            r#"{"error":{"type":"snapshot_in_progress_exception"}}"#
        ));
        // Busy cluster manager.
        assert!(delete_index_retryable(503, ""));
        // A read-only block set by disk watermarks clears on its own.
        assert!(delete_index_retryable(
            403,
            r#"{"error":{"type":"cluster_block_exception"}}"#
        ));
        // A 400 for any OTHER reason is permanent — retrying it burns ten
        // backoffs (20s+ each by default) before reporting the real error.
        assert!(!delete_index_retryable(
            400,
            r#"{"error":{"type":"invalid_index_name_exception"}}"#
        ));
        assert!(!delete_index_retryable(401, ""));
        assert!(!delete_index_retryable(500, ""));
    }

    #[test]
    fn create_index_retries_cluster_event_timeout_only() {
        // Cluster manager thread pool busy — the case this retry exists for.
        assert!(create_index_retryable(
            503,
            r#"{"error":{"type":"process_cluster_event_timeout_exception"}}"#
        ));
        assert!(create_index_retryable(429, ""));
        assert!(create_index_retryable(
            403,
            r#"{"error":{"type":"cluster_block_exception"}}"#
        ));
        // The index already existing is a real conflict; no amount of waiting
        // clears it, and retrying would mask a name-collision bug.
        assert!(!create_index_retryable(
            400,
            r#"{"error":{"type":"resource_already_exists_exception"}}"#
        ));
        assert!(!create_index_retryable(400, ""));
    }

    #[test]
    fn index_settings_pin_shard_count_only_when_configured() {
        // Unset: previous behaviour, no `number_of_shards` key at all, so the
        // cluster default applies and existing configs are untouched.
        let default = build_index_settings(None);
        assert!(default.get("number_of_shards").is_none());
        assert_eq!(default["knn"], json!(true));
        assert_eq!(default["number_of_replicas"], json!(0));
        assert_eq!(default["refresh_interval"], json!(-1));

        // Set: emitted under the exact key OpenSearch expects inside
        // `settings.index` — a typo here would be silently ignored by the server.
        let pinned = build_index_settings(Some(1));
        assert_eq!(pinned["number_of_shards"], json!(1));
        assert_eq!(build_index_settings(Some(5))["number_of_shards"], json!(5));
        // The throughput tuning must survive the shard override.
        assert_eq!(pinned["number_of_replicas"], json!(0));
        assert_eq!(pinned["refresh_interval"], json!(-1));
    }

    #[test]
    fn number_of_shards_rejects_a_present_but_unusable_value() {
        // Absent is the documented "inherit the cluster default" case.
        assert_eq!(parse_number_of_shards(None), Ok(None));
        assert_eq!(parse_number_of_shards(Some(&json!(1))), Ok(Some(1)));
        assert_eq!(parse_number_of_shards(Some(&json!(5))), Ok(Some(5)));

        // These all parse as valid JSON and would previously have been dropped by
        // `.and_then(|v| v.as_i64())`, silently reinstating the cluster default
        // that setting the key was meant to override.
        for bad in [json!("5"), json!(5.0), json!(true), json!(null), json!(0)] {
            let err = parse_number_of_shards(Some(&bad))
                .expect_err(&format!("{bad} must be rejected, not dropped"));
            assert!(
                err.contains("number_of_shards"),
                "error must name the key: {err}"
            );
        }
    }

    /// Shipped-config guard (#211). Making the shard count settable buys nothing
    /// while no shipped config sets it: `elasticsearch.rs` pins
    /// `ES_NUMBER_OF_SHARDS`, so an unpinned OpenSearch config turns the
    /// published head-to-head into 1-shard ES vs whatever the cluster happens to
    /// default to (1 open-source, historically 5 on Amazon OpenSearch Service).
    ///
    /// `collection_params` is a `#[serde(flatten)]` catch-all, so deleting or
    /// misspelling the key still parses cleanly and simply stops arriving — the
    /// comparison would silently revert with a green build. This walks the real
    /// shipped files through the exact
    /// `collection_params.extra` → `parse_number_of_shards` path
    /// `OpenSearchEngine::new` uses, so that regression fails CI instead.
    ///
    /// It globs *every* shipped configuration file rather than naming two, so an
    /// `engine: "opensearch"` entry added to any third file is covered the day it
    /// lands; the two files this repo ships additionally have their exact pin,
    /// sweep size and HNSW grid asserted.
    ///
    /// Scope: this asserts a pin is PRESENT and carries the expected value. The
    /// complementary generic check — that every knob a shipped config declares is
    /// actually read by its target engine — is engine-agnostic and belongs in
    /// `config.rs` (#216). Repo-wide duplicate config names are #239; the
    /// uniqueness assertion below is scoped to opensearch so this test does not
    /// inherit that failure.
    #[test]
    fn shipped_opensearch_configs_pin_their_shard_count() {
        use std::collections::{HashMap, HashSet};

        // (expected pin, number of sweep points) for the files this repo ships.
        // The single-node pin is read from elasticsearch.rs rather than written
        // as `1`, because "same shard count as ES" is the actual #211 invariant:
        // moving the ES constant must fail here instead of silently unpairing the
        // published ES-vs-OS comparison.
        let known: HashMap<&str, (i64, usize)> = HashMap::from([
            (
                "opensearch-single-node.json",
                (crate::engine::elasticsearch::ES_NUMBER_OF_SHARDS, 7),
            ),
            // Amazon OpenSearch Service's historical (ES-derived) per-index
            // default. Deliberately NOT the ES constant: this file exists to
            // measure the other shard count, not to pair with Elasticsearch.
            ("opensearch-5-shard.json", (5_i64, 7)),
        ]);

        let dir = crate::config::project_root().join("experiments/configurations");
        let paths: Vec<_> = glob::glob(dir.join("*.json").to_str().unwrap())
            .expect("configuration glob is valid")
            .flatten()
            .collect();
        assert!(
            !paths.is_empty(),
            "no configuration files found under {}",
            dir.display()
        );

        let mut visited: HashSet<&str> = HashSet::new();
        let mut grids: HashMap<&str, Vec<(i64, i64)>> = HashMap::new();
        // (config name, file it came from) for every opensearch entry shipped.
        let mut seen: Vec<(String, String)> = Vec::new();

        for path in paths {
            let file = path
                .file_name()
                .and_then(|f| f.to_str())
                .expect("config filename is valid UTF-8")
                .to_string();
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            // A file that is not an array of configs is skipped by the production
            // loader too; the `visited` check below still catches one of OUR files
            // being emptied or corrupted into that state.
            let Ok(raw) = serde_json::from_str::<Vec<serde_json::Value>>(&text) else {
                continue;
            };
            if !raw
                .iter()
                .any(|c| c.get("engine").and_then(|e| e.as_str()) == Some("opensearch"))
            {
                continue;
            }

            let configs = crate::config::read_engine_configs(Some(
                path.to_str().expect("config path is valid UTF-8"),
            ))
            .unwrap_or_else(|e| panic!("{file} ships opensearch configs and must parse: {e}"));
            // read_engine_configs keys by name with last-write-wins, so a name
            // reused inside one file silently *deletes* a sweep point: the run
            // completes, the report is one row short, and nothing says so.
            // (Across files this is #239 — engine-agnostic, fixed in config.rs.)
            assert_eq!(
                configs.len(),
                raw.len(),
                "{file} declares {} configs but only {} survive loading — duplicate \
                 \"name\" values collapse silently (last-write-wins)",
                raw.len(),
                configs.len()
            );

            // Fail CLOSED on a file this test has never heard of. Checking only
            // "pins something positive" would let a new OpenSearch config file
            // ship an arbitrary shard count that no reviewer ever chose, which is
            // the same class of accident as inheriting the cluster default.
            // Registering it here forces the value to be a decision.
            let Some(want) = known.get(file.as_str()).map(|(shards, _)| *shards) else {
                panic!(
                    "{file} declares engine \"opensearch\" configs but is not registered in \
                     this guard. Add it to `known` with the shard count it is meant to \
                     measure (and why), so the value is reviewed rather than inherited."
                );
            };

            for (name, config) in &configs {
                if config.engine.as_deref() != Some("opensearch") {
                    continue;
                }
                // Read it exactly the way `OpenSearchEngine::new` does — off
                // `collection_params.extra` and through `parse_number_of_shards`
                // — so a key that is misspelled, mistyped, or moved one level up
                // fails here for the same reason it would stop reaching the
                // server. `CollectionParams` has no typed field that could absorb
                // `number_of_shards` (only `hnsw_config` / `index_options`), and
                // if one were ever added, `parse_number_of_shards(None)` returns
                // `Ok(None)` and this assertion fails rather than passing blind.
                let raw_shards = config
                    .collection_params
                    .as_ref()
                    .and_then(|c| c.extra.as_ref())
                    .and_then(|e| e.get("number_of_shards"));
                assert_eq!(
                    parse_number_of_shards(raw_shards),
                    Ok(Some(want)),
                    "{file}/{name} must pin collection_params.number_of_shards to \
                     {want}; an unpinned config silently inherits the cluster \
                     default, which is not comparable with elasticsearch.rs's \
                     pinned {}",
                    crate::engine::elasticsearch::ES_NUMBER_OF_SHARDS
                );
                seen.push((name.clone(), file.clone()));
            }

            if let Some((&key, &(_, want_len))) = known.get_key_value(file.as_str()) {
                // Pin the sweep SIZE too: truncating the file leaves every
                // surviving entry correctly pinned, so the loop above stays green
                // while the published sweep quietly loses points.
                assert_eq!(
                    raw.len(),
                    want_len,
                    "{file} must ship {want_len} sweep points, found {}",
                    raw.len()
                );
                let mut grid: Vec<(i64, i64)> = raw
                    .iter()
                    .map(|c| {
                        let p = &c["collection_params"]["method"]["parameters"];
                        let m = p["m"]
                            .as_i64()
                            .unwrap_or_else(|| panic!("{file}: entry missing method.parameters.m"));
                        let ef = p["ef_construction"].as_i64().unwrap_or_else(|| {
                            panic!("{file}: entry missing method.parameters.ef_construction")
                        });
                        (m, ef)
                    })
                    .collect();
                grid.sort_unstable();
                grids.insert(key, grid);
                visited.insert(key);
            }
        }

        let mut missing: Vec<&str> = known
            .keys()
            .copied()
            .filter(|k| !visited.contains(k))
            .collect();
        missing.sort_unstable();
        assert!(
            missing.is_empty(),
            "shipped OpenSearch config file(s) missing, emptied, unparsable, or no \
             longer declaring engine \"opensearch\": {missing:?}"
        );

        // The two files differ in exactly one dimension — shard count. If their
        // HNSW grids drift apart, "same sweep at 1 vs 5 shards" stops being true
        // and the two result sets are no longer comparable to each other either.
        assert_eq!(
            grids["opensearch-single-node.json"], grids["opensearch-5-shard.json"],
            "both shipped OpenSearch files must sweep the same (m, ef_construction) \
             grid; only number_of_shards may differ"
        );

        // Engine configs from every file are globbed into one name-keyed map, so a
        // reused name makes one shard count quietly shadow the other — exactly the
        // failure this test exists to prevent. Scoped to opensearch here; the
        // repo-wide version of this collision (vectorsets-fp32-default) is #239.
        let unique: HashSet<&String> = seen.iter().map(|(name, _)| name).collect();
        assert_eq!(
            unique.len(),
            seen.len(),
            "opensearch config names must be unique across shipped files: {seen:?}"
        );
    }

    #[test]
    fn index_maintenance_retries_gateway_and_transient_cluster_states() {
        // force-merge runs AFTER the whole ingest and is the op a managed front
        // door most often 504s on, so the gateway statuses are the load-bearing
        // cases here.
        for status in [429, 502, 503, 504] {
            assert!(index_maintenance_retryable(status, ""), "{status}");
        }
        assert!(index_maintenance_retryable(
            403,
            r#"{"error":{"type":"cluster_block_exception"}}"#
        ));
        // A real error must not burn the budget: retrying cannot make a missing
        // index appear, and doing so delays the actual diagnosis.
        assert!(!index_maintenance_retryable(
            404,
            r#"{"error":{"type":"index_not_found_exception"}}"#
        ));
        assert!(!index_maintenance_retryable(400, ""));
    }

    // #210: merging to ONE segment makes force-merge far longer than any other
    // request the client sends, so it cannot share the query-sized transport
    // timeout. If it did, every attempt would abort client-side while the merge
    // was still running on the server, and the 10-attempt retry budget added in
    // #208 would be spent failing a merge that was always going to succeed —
    // discarding the ingest those retries exist to protect.
    #[test]
    fn force_merge_timeout_outlives_the_query_sized_client_timeout() {
        use std::time::Duration;

        // Default client timeout (300 s) must NOT be what bounds a merge.
        assert_eq!(
            resolve_force_merge_timeout(300, None),
            Duration::from_secs(3_600)
        );
        // An operator who raised OPENSEARCH_TIMEOUT past the floor meant it here
        // too — the floor must never shorten their bound.
        assert_eq!(
            resolve_force_merge_timeout(7_200, None),
            Duration::from_secs(7_200)
        );
        // An explicit OPENSEARCH_FORCE_MERGE_TIMEOUT wins outright, in both
        // directions; an override that could not lower the bound is not one.
        assert_eq!(
            resolve_force_merge_timeout(300, Some(60)),
            Duration::from_secs(60)
        );
        assert_eq!(
            resolve_force_merge_timeout(7_200, Some(10_800)),
            Duration::from_secs(10_800)
        );
    }

    // `0` conventionally means "no limit". Resolving it to Duration::ZERO would
    // make reqwest reject every attempt before sending a byte (the deadline is
    // already past), so all 11 attempts fail instantly and a COMPLETED ingest is
    // discarded — the loudest possible misreading of "unlimited".
    #[test]
    fn force_merge_timeout_zero_means_unlimited_not_instant_failure() {
        let resolved = resolve_force_merge_timeout(300, Some(0));
        assert!(
            !resolved.is_zero(),
            "0 must not resolve to an already-expired deadline"
        );
        assert!(
            resolved >= std::time::Duration::from_secs(10 * 365 * 24 * 3_600),
            "0 must resolve to a deadline that never fires in practice, got {resolved:?}"
        );
    }

    // `OPENSEARCH_FORCE_MERGE_BUDGET` is a knob this PR introduced, so it needs
    // its own coverage or it is a silent-config-drop waiting to happen: an
    // implementation that parses the variable and then ignores it looks identical
    // from the outside, and the run measures the default while its name and its
    // result JSON claim otherwise.
    #[test]
    fn force_merge_budget_defaults_to_two_attempts_and_honours_its_knob() {
        use std::time::Duration;
        let hour = Duration::from_secs(3_600);

        // Default: room for one full-length attempt plus one full-length retry.
        assert_eq!(
            resolve_force_merge_budget(hour, None),
            Some(Duration::from_secs(7_200))
        );
        // The default TRACKS the per-attempt deadline rather than being a constant,
        // so raising OPENSEARCH_FORCE_MERGE_TIMEOUT does not create a budget that
        // cuts the very first attempt short.
        assert_eq!(
            resolve_force_merge_budget(Duration::from_secs(10_800), None),
            Some(Duration::from_secs(21_600))
        );
        // An explicit value wins, in both directions.
        assert_eq!(
            resolve_force_merge_budget(hour, Some(60)),
            Some(Duration::from_secs(60))
        );
        assert_eq!(
            resolve_force_merge_budget(hour, Some(86_400)),
            Some(Duration::from_secs(86_400))
        );
        // 0 = unlimited, matching OPENSEARCH_FORCE_MERGE_TIMEOUT's convention.
        assert_eq!(resolve_force_merge_budget(hour, Some(0)), None);
        // Saturating, not panicking, at the unlimited-deadline extreme.
        assert!(resolve_force_merge_budget(FORCE_MERGE_TIMEOUT_UNLIMITED, None).is_some());
    }

    // The bounds must be resolved at CONSTRUCTION, not inside force_merge, which
    // runs after the whole upload: a malformed value parsed there rejects the run
    // only once there is a multi-hour ingest to throw away. This covers the
    // resolution path end to end — env in, stored Durations out — including that
    // a bad value fails `new()` rather than the merge.
    #[test]
    fn merge_bounds_are_resolved_when_the_engine_is_constructed() {
        use crate::config::EngineConfig;

        fn engine(env: &[(&str, Option<&str>)]) -> Result<OpenSearchEngine, String> {
            // SAFETY: these variables are touched by no other test.
            for (k, v) in env {
                unsafe {
                    match v {
                        Some(val) => std::env::set_var(k, val),
                        None => std::env::remove_var(k),
                    }
                }
            }
            let cfg: EngineConfig = serde_json::from_value(serde_json::json!({
                "name": "os-bounds-test",
                "engine": "opensearch",
            }))
            .unwrap();
            OpenSearchEngine::new(&cfg, "http://127.0.0.1")
        }

        let clear = [
            ("OPENSEARCH_TIMEOUT", None),
            ("OPENSEARCH_FORCE_MERGE_TIMEOUT", None),
            ("OPENSEARCH_FORCE_MERGE_BUDGET", None),
        ];

        // Defaults: the 300 s client timeout must NOT bound a merge, and the
        // budget tracks the deadline rather than being a constant.
        let e = engine(&clear).unwrap_or_else(|e| panic!("defaults must construct: {e}"));
        assert_eq!(
            e.force_merge_deadline,
            std::time::Duration::from_secs(3_600)
        );
        assert_eq!(
            e.force_merge_budget,
            Some(std::time::Duration::from_secs(7_200))
        );

        // Both knobs reach the stored values — if either were dropped, the run
        // would silently use the default while the operator believed otherwise.
        let e = engine(&[
            ("OPENSEARCH_TIMEOUT", None),
            ("OPENSEARCH_FORCE_MERGE_TIMEOUT", Some("120")),
            ("OPENSEARCH_FORCE_MERGE_BUDGET", Some("300")),
        ])
        .unwrap_or_else(|e| panic!("explicit values must construct: {e}"));
        assert_eq!(e.force_merge_deadline, std::time::Duration::from_secs(120));
        assert_eq!(
            e.force_merge_budget,
            Some(std::time::Duration::from_secs(300))
        );

        // 0 = unlimited on both, not "expire immediately" / "no retries".
        let e = engine(&[
            ("OPENSEARCH_TIMEOUT", None),
            ("OPENSEARCH_FORCE_MERGE_TIMEOUT", Some("0")),
            ("OPENSEARCH_FORCE_MERGE_BUDGET", Some("0")),
        ])
        .unwrap_or_else(|e| panic!("0 must construct: {e}"));
        assert!(!e.force_merge_deadline.is_zero());
        assert_eq!(e.force_merge_budget, None);

        // A malformed value fails HERE, before any upload exists to discard.
        let err = engine(&[
            ("OPENSEARCH_TIMEOUT", None),
            ("OPENSEARCH_FORCE_MERGE_TIMEOUT", None),
            ("OPENSEARCH_FORCE_MERGE_BUDGET", Some("2h")),
        ])
        .err()
        .expect("a malformed budget must fail at construction");
        assert!(
            err.contains("OPENSEARCH_FORCE_MERGE_BUDGET"),
            "the error must name the variable: {err}"
        );

        for (k, _) in clear {
            unsafe { std::env::remove_var(k) };
        }
    }

    // A present-but-unusable value must fail loudly rather than silently
    // reinstating the default — the same rule `parse_number_of_shards` enforces
    // in this file.
    #[test]
    fn env_secs_rejects_present_but_unusable_values() {
        let var = "OPENSEARCH_TEST_SECS_PARSE";
        // SAFETY: this variable is touched by no other test or thread.
        unsafe { std::env::remove_var(var) };
        assert_eq!(
            parse_env_secs(var),
            Ok(None),
            "absent = inherit the default"
        );

        for bad in ["3600s", "1h", "-1", "", "1.5", "abc"] {
            unsafe { std::env::set_var(var, bad) };
            let err = parse_env_secs(var).expect_err(&format!("{bad:?} must be rejected"));
            assert!(err.contains(var), "error must name the variable: {err}");
        }
        unsafe { std::env::set_var(var, " 0 ") };
        assert_eq!(parse_env_secs(var), Ok(Some(0)), "0 is a legal value");
        unsafe { std::env::set_var(var, "1800") };
        assert_eq!(parse_env_secs(var), Ok(Some(1_800)));
        unsafe { std::env::remove_var(var) };
    }

    // #210: `_cluster/health?wait_for_status=yellow` reports a missed status as
    // HTTP 408 on BOTH shipped engines (verified against OpenSearch 3.7.0 and
    // Elasticsearch 9.4.3), and 408 must not be mistaken for success.
    #[test]
    fn cluster_health_timeout_is_not_mistaken_for_success() {
        use serde_json::json;

        assert!(cluster_health_settled(
            200,
            &json!({"status": "green", "timed_out": false})
        ));
        // The real shape both engines return on a missed status.
        assert!(
            !cluster_health_settled(408, &json!({"status": "yellow", "timed_out": true})),
            "HTTP 408 + timed_out:true is the shipped timeout response"
        );
        // Defence in depth for the 2xx-carrying variant produced by
        // `cluster.health.return_200_for_cluster_health_timeout` on the
        // Elasticsearch versions that support it.
        assert!(
            !cluster_health_settled(200, &json!({"status": "red", "timed_out": true})),
            "a timed_out body must lose even when it arrives with a 2xx"
        );
        // Absent field: treat as settled rather than hard-failing a finished
        // ingest over a response-shape change.
        assert!(cluster_health_settled(200, &json!({"status": "green"})));
        assert!(!cluster_health_settled(503, &json!({"timed_out": false})));
    }

    // The health wait is the TAIL of force_merge, so what it does with a failure
    // decides the fate of an already-completed ingest. 408 is "not settled yet"
    // and must retry; an authorization denial must not be retried at all (it
    // cannot change), and is handled by the caller as a warning.
    #[test]
    fn cluster_health_retries_transient_states_but_never_authorization() {
        for status in [408, 429, 502, 503, 504] {
            assert!(cluster_health_retryable(status, ""), "{status} must retry");
        }
        for status in [401, 403] {
            assert!(
                !cluster_health_retryable(status, ""),
                "{status} must not burn the retry budget"
            );
            assert!(is_authorization_denied(status, ""));
        }
        // A genuine error is neither retryable nor an authorization decision, so
        // it surfaces as a hard failure with the body attached.
        assert!(!cluster_health_retryable(400, ""));
        assert!(!is_authorization_denied(404, ""));
        assert!(!is_authorization_denied(503, ""));

        // OpenSearch ships `cluster_block_exception` as HTTP 403 as well, and that
        // one is TRANSIENT — a disk-watermark read-only block clears itself. It
        // must be retried like everywhere else in this file, not read as "you may
        // not ask" and silently skipped.
        let blocked = r#"{"error":{"type":"cluster_block_exception"}}"#;
        assert!(
            !is_authorization_denied(403, blocked),
            "a 403 cluster block is not an authorization decision"
        );
        assert!(
            cluster_health_retryable(403, blocked),
            "a 403 cluster block must retry, as its three sibling predicates do"
        );
        // A real permission denial still short-circuits.
        let denied = r#"{"error":{"type":"security_exception","reason":"no permissions for [cluster:monitor/health]"}}"#;
        assert!(is_authorization_denied(403, denied));
        assert!(!cluster_health_retryable(403, denied));
    }

    // COMPOSITION, not predicates: the order of the checks in the health wait is
    // what decides whether a 403 warns or aborts a COMPLETED multi-hour ingest.
    // Swapping the authorization check with the retry check leaves every
    // predicate test green while flipping "continue the run" into "throw the
    // ingest away", so the ordering is pinned here directly.
    #[test]
    fn health_verdicts_compose_so_a_denial_never_aborts_a_finished_ingest() {
        let ok = r#"{"status":"green","timed_out":false}"#;
        assert_eq!(
            classify_health_response(200, ok),
            HealthVerdict::Settled,
            "the happy path must win before anything else is consulted"
        );

        // The shipped timeout shape on both engines: not settled, but transient.
        let timed_out = r#"{"status":"yellow","timed_out":true}"#;
        assert_eq!(
            classify_health_response(408, timed_out),
            HealthVerdict::Retry
        );

        // A permission denial: warn and proceed. If the retry check were consulted
        // first this would be Retry, burn the budget, and then fail a finished
        // ingest over a monitoring permission.
        let denied = r#"{"error":{"type":"security_exception","reason":"no permissions for [cluster:monitor/health]"}}"#;
        assert_eq!(
            classify_health_response(403, denied),
            HealthVerdict::Unobservable
        );
        assert_eq!(
            classify_health_response(401, denied),
            HealthVerdict::Unobservable
        );

        // A 403 cluster block is transient and must NOT be read as a denial, or
        // the wait would skip a cluster that really had not settled.
        let blocked = r#"{"error":{"type":"cluster_block_exception"}}"#;
        assert_eq!(classify_health_response(403, blocked), HealthVerdict::Retry);

        // Anything else is a genuine failure, surfaced rather than skipped.
        assert_eq!(classify_health_response(400, "{}"), HealthVerdict::Failed);
        assert_eq!(
            classify_health_response(404, r#"{"error":"no such index"}"#),
            HealthVerdict::Failed
        );

        // A 2xx carrying timed_out must not be mistaken for Settled, and is
        // transient rather than fatal.
        assert_eq!(
            classify_health_response(200, r#"{"status":"red","timed_out":true}"#),
            HealthVerdict::Retry
        );
        // An unparseable body must not crash the classifier.
        assert_eq!(
            classify_health_response(200, "not json"),
            HealthVerdict::Settled
        );
    }

    // Raising the per-attempt force-merge bound 12x (300 s -> 3600 s) without a
    // wall-clock ceiling would leave an ~11 h worst case for one force_merge,
    // since retry_index_op treats every transport error as retryable.
    //
    // Reads the REAL default from `index_op_policy()` rather than a hand-built
    // literal. An earlier version of this test constructed a `RetryPolicy { budget:
    // None, .. }` and then asserted the field it had just set — a tautology that
    // stayed green even if every index op were given a 1 s budget.
    #[test]
    fn retry_policy_budget_defaults_off_and_is_opt_in() {
        let shipped = index_op_policy();
        assert_eq!(
            shipped.budget, None,
            "create/delete/refresh keep count-only bounds; only force merge opts in"
        );

        let bounded = shipped.with_budget(Some(std::time::Duration::from_secs(7_200)));
        assert_eq!(bounded.budget, Some(std::time::Duration::from_secs(7_200)));
        // The budget must not disturb the rest of the policy.
        assert_eq!(bounded.max_retries, shipped.max_retries);
        assert_eq!(bounded.base_delay_ms, shipped.base_delay_ms);
        // Explicitly unlimited stays unlimited.
        assert_eq!(shipped.with_budget(None).budget, None);
    }

    // WIRING, not predicate: does `retry_index_op` actually enforce the budget it
    // is handed? Every predicate in this file can be correct while the budget is
    // computed and then dropped on the floor, which restores the ~11 h unbounded
    // force merge. Driving an always-failing `send` proves the loop consults it.
    //
    // No cluster needed: `send` is an `FnMut` closure, so a synthesized transport
    // error is enough, and `base_delay_ms: 1` keeps the whole test in milliseconds.
    #[test]
    fn retry_index_op_stops_at_the_wall_clock_budget_not_just_the_attempt_count() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut calls = 0usize;

        let policy = RetryPolicy {
            // High enough that the budget is what ends a correct run, low enough
            // that a broken one still ENDS. An earlier version used 1_000_000
            // here; with the budget mutated off that does not fail the test, it
            // hangs for hours (1M attempts x capped backoff), which in CI is a
            // timeout rather than a diagnosis. A mutation must fail fast.
            max_retries: 60,
            base_delay_ms: 1,
            budget: Some(std::time::Duration::from_millis(150)),
        };

        let started = std::time::Instant::now();
        let err = retry_index_op(
            &rt,
            "Force merge",
            policy,
            |status| (200..300).contains(&status),
            index_maintenance_retryable,
            || {
                calls += 1;
                // A transport-level failure: the case `retry_index_op` treats as
                // retryable forever, which is exactly why the budget must exist.
                Err(opensearch::Error::from(std::io::Error::other(
                    "connection reset",
                )))
            },
        )
        .expect_err("an always-failing op must not report success");

        assert!(
            err.contains("wall-clock budget"),
            "the error must name the bound that stopped it, got: {err}"
        );
        assert!(err.starts_with("Force merge"), "must name the op: {err}");
        assert!(
            calls > 1,
            "must actually have retried before giving up, got {calls} call(s)"
        );
        // Stopped by time, well short of the 60-retry ceiling.
        assert!(
            calls <= 60,
            "budget did not bound the loop; {calls} attempts"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(30),
            "budget did not bound the wall clock"
        );
    }

    // The count-only policies must NOT acquire a time bound by accident: the same
    // loop serves create/delete/refresh, and a budget leaking into them would cut
    // short the snapshot windows #208 added the patient retries for.
    #[test]
    fn retry_index_op_without_a_budget_is_bounded_only_by_attempts() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut calls = 0usize;
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay_ms: 1,
            budget: None,
        };
        let err = retry_index_op(
            &rt,
            "Refresh",
            policy,
            |status| (200..300).contains(&status),
            index_maintenance_retryable,
            || {
                calls += 1;
                Err(opensearch::Error::from(std::io::Error::other(
                    "connection reset",
                )))
            },
        )
        .expect_err("an always-failing op must not report success");

        assert_eq!(calls, 4, "3 retries = 4 attempts");
        assert!(
            err.contains("after 3 retries") && !err.contains("wall-clock budget"),
            "an unbudgeted op must stop on the attempt count: {err}"
        );
    }

    #[test]
    fn jitter_spreads_the_delay_over_the_upper_half_without_exceeding_it() {
        // Lock-step retries are the failure this exists to prevent: without
        // jitter every worker sheds at the same moment, sleeps the identical
        // delay, and retries simultaneously — reinforcing the overload.
        let capped = 1_000;
        for rand in [0, 1, 7, 12_345, u64::MAX / 3, u64::MAX] {
            let d = jittered_delay_ms(capped, rand);
            assert!(
                (500..=1_000).contains(&d),
                "delay {d} outside [d/2, d] for rand {rand}"
            );
        }
        // Backoff must still grow monotonically in the floor, or a long stall
        // could sleep less than a short one.
        assert_eq!(jittered_delay_ms(0, 12_345), 0);
        assert_eq!(jittered_delay_ms(1, 0), 0);
        assert_eq!(jittered_delay_ms(1, 1), 1);
        // Distinct entropy must actually produce distinct delays.
        let spread: std::collections::HashSet<u64> = (0..64)
            .map(|r| jittered_delay_ms(capped, r * 977))
            .collect();
        assert!(spread.len() > 8, "jitter is not decorrelating: {spread:?}");
    }

    #[test]
    fn jitter_rand_advances_and_never_sticks_at_zero() {
        // A zero state is a fixed point of xorshift: it would return 0 forever and
        // silently reinstate lock-step retries.
        let seen: Vec<u64> = (0..8).map(|_| retry_jitter_rand()).collect();
        assert!(seen.iter().all(|&x| x != 0), "xorshift stuck at zero");
        assert!(
            seen.windows(2).any(|w| w[0] != w[1]),
            "generator is not advancing"
        );
    }
}
