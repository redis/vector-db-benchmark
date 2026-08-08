//! Engine module - Modular vector database engine implementations.
//!
//! Mirrors Python v0/engine/ structure:
//! - `Engine` trait = BaseClient
//! - `Configurator` trait = BaseConfigurator  
//! - `Uploader` trait = BaseUploader
//! - `Searcher` trait = BaseSearcher

mod chroma;
mod dragonfly;
mod elasticsearch;
#[cfg(test)]
mod filter_guard;
pub mod index_naming;
mod kividb;
mod milvus;
mod mongodb_engine;
mod opensearch;
mod pgvector;
mod qdrant;
mod redis;
mod redis_utils;
mod turbopuffer;
mod valkey;
mod vectorsets;
mod vertex;
mod vertex_grpc;
mod weaviate;
mod weaviate_grpc;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use std::time::{Duration, Instant};

pub use chroma::ChromaEngine;
pub use dragonfly::DragonflyEngine;
pub use elasticsearch::ElasticsearchEngine;
pub use kividb::KividbEngine;
pub use milvus::MilvusEngine;
pub use mongodb_engine::MongoDBEngine;
pub use opensearch::OpenSearchEngine;
pub use pgvector::PgVectorEngine;
pub use qdrant::QdrantEngine;
pub use redis::RedisEngine;
pub use turbopuffer::TurbopufferEngine;
pub use valkey::ValkeyEngine;
pub use vectorsets::VectorSetsEngine;
pub use vertex::VertexEngine;
pub use weaviate::WeaviateEngine;

/// Upload statistics
#[derive(Debug, Clone, Default)]
pub struct UploadStats {
    pub upload_time: f64,
    pub total_time: f64,
    pub upload_count: usize,
    pub parallel: usize,
    pub batch_size: usize,
    pub memory_usage: Option<serde_json::Value>,
}

/// Update-to-search ratio for mixed workload benchmarks.
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateSearchRatio {
    pub updates: u64,
    pub searches: u64,
}

/// Search results — matches Python v0 search result JSON fields
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct SearchResults {
    pub total_time: f64,
    pub mean_time: f64,
    /// Mean per-query precision, denominator = results the engine actually
    /// returned: `hits / |deduped results kept|`. Emitted as
    /// `mean_precision_at_returned`, NOT as `mean_precisions` — upstream
    /// `qdrant/vector-db-benchmark` publishes recall@top under that key (#217).
    /// `-1.0` is the filter-only sentinel (no vector search, no quality metric).
    pub mean_precision_at_returned: f64,
    /// Mean per-query recall, denominator = the valid, deduped ground-truth ids
    /// that exist in `expected[:top]` (so a query with 3 true neighbours can
    /// reach 1.0). Equals upstream's `mean_precisions` only when every
    /// ground-truth row is at least `top` wide.
    pub mean_recall: f64,
    /// 10th-percentile per-query recall — the "worst 10%" floor. A healthy mean
    /// with a near-zero p10 means a slice of queries return almost nothing (e.g.
    /// a filter that occasionally matches an empty/degenerate set); the mean
    /// hides that, so this is reported alongside it (VSB/VDBBench methodology).
    pub recall_p10: f64,
    pub mean_mrr: f64,
    pub mean_ndcg: f64,
    pub recalls: Vec<f64>,
    pub mrrs: Vec<f64>,
    pub ndcgs: Vec<f64>,
    pub std_time: f64,
    pub min_time: f64,
    pub max_time: f64,
    pub rps: f64,
    pub p50_time: f64,
    pub p95_time: f64,
    pub p99_time: f64,
    /// Per-query precision-at-returned samples (see `mean_precision_at_returned`).
    pub precisions_at_returned: Vec<f64>,
    pub latencies: Vec<f64>,
    pub top: usize,
    /// Number of *successful* queries folded into the latency/quality stats.
    pub num_queries: usize,
    /// Number of queries requested for this run (num_to_run).
    pub requested_queries: usize,
    /// requested_queries - num_queries: queries that errored/timed out and were
    /// excluded from the latency percentiles. Nonzero means the reported numbers
    /// are over a partial set (e.g. a saturated client shedding timeouts).
    pub failed_queries: usize,
    pub parallel: usize,
    // Client CPU / concurrency-saturation coverage (filled by the runner after
    // the timed window; see proc_cpu). When client_saturated is true the latency
    // and QPS above reflect a client-bound run, not clean server-side numbers.
    pub available_cores: usize,
    pub oversubscribed: bool,
    pub client_cpu_cores_used: Option<f64>,
    pub system_cpu_pct: Option<f64>,
    pub client_saturated: bool,
    pub saturation_reason: String,
    // Open-loop offered-load coverage. None/empty in the original closed-loop mode.
    pub target_qps: Option<f64>,
    pub offered_queries: Option<usize>,
    pub dropped_queries: usize,
    pub late_queries: usize,
    pub schedule_delay_p50_time: Option<f64>,
    pub schedule_delay_p95_time: Option<f64>,
    pub schedule_delay_p99_time: Option<f64>,
    pub end_to_end_p50_time: Option<f64>,
    pub end_to_end_p95_time: Option<f64>,
    pub end_to_end_p99_time: Option<f64>,
    // Mixed benchmark update metrics (None when search-only)
    pub update_count: Option<usize>,
    pub update_rps: Option<f64>,
    pub update_mean_time: Option<f64>,
    pub update_p50_time: Option<f64>,
    pub update_p95_time: Option<f64>,
    pub update_p99_time: Option<f64>,
    pub update_latencies: Option<Vec<f64>>,
    pub update_search_ratio: Option<String>,
}

/// Deterministic arrival schedule for fixed-rate, open-loop search.
#[derive(Debug, Clone, Copy)]
pub struct OpenLoopPlan {
    pub target_qps: f64,
    pub total_requests: usize,
    pub max_lateness: Duration,
    late_threshold: Duration,
}

impl OpenLoopPlan {
    /// Build a plan from search parameters. Returns None for legacy closed-loop runs.
    pub fn from_params(params: &SearchParams) -> Result<Option<Self>, String> {
        let Some(target_qps) = params.target_qps else {
            return Ok(None);
        };
        let duration_seconds = params
            .duration_seconds
            .ok_or("open-loop search requires duration_seconds (use --search-duration)")?;
        let max_lateness_ms = params.max_lateness_ms.unwrap_or(1000.0);
        if !target_qps.is_finite() || target_qps <= 0.0 {
            return Err("target_qps must be finite and greater than zero".to_string());
        }
        // Upper-bound the durations too: `Duration::from_secs_f64` panics on
        // non-finite / out-of-range inputs, so reject absurd CLI values (> 1e9 s)
        // with a clear error instead of aborting.
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 || duration_seconds > 1e9 {
            return Err(
                "duration_seconds must be finite, greater than zero, and <= 1e9 seconds"
                    .to_string(),
            );
        }
        if !max_lateness_ms.is_finite() || !(0.0..=1e12).contains(&max_lateness_ms) {
            return Err("max_lateness_ms must be finite, non-negative, and <= 1e12 ms".to_string());
        }
        let total_f = (target_qps * duration_seconds).round();
        if !total_f.is_finite() || total_f < 1.0 || total_f > usize::MAX as f64 {
            return Err("target_qps * duration_seconds is outside the supported range".to_string());
        }
        // A request is reported as late after two arrival intervals, with a 1 ms
        // floor to avoid treating normal scheduler jitter as overload.
        let late_threshold = Duration::from_secs_f64((2.0 / target_qps).max(0.001));
        Ok(Some(Self {
            target_qps,
            total_requests: total_f as usize,
            max_lateness: Duration::from_secs_f64(max_lateness_ms / 1000.0),
            late_threshold,
        }))
    }

    pub fn scheduled_at(&self, start: Instant, ordinal: usize) -> Instant {
        start + Duration::from_secs_f64(ordinal as f64 / self.target_qps)
    }

    /// Wait until this request's independent arrival time and return dispatch delay.
    pub fn wait_for_slot(&self, start: Instant, ordinal: usize) -> Duration {
        let scheduled = self.scheduled_at(start, ordinal);
        let now = Instant::now();
        if now < scheduled {
            std::thread::sleep(scheduled.duration_since(now));
        }
        Instant::now().saturating_duration_since(scheduled)
    }

    pub fn is_late(&self, delay: Duration) -> bool {
        delay > self.late_threshold
    }
}

/// Return the duration for an unrestricted closed-loop run, if requested.
pub fn closed_loop_duration(params: &SearchParams) -> Result<Option<Duration>, String> {
    if params.target_qps.is_some() {
        return Ok(None);
    }
    let Some(seconds) = params.duration_seconds else {
        return Ok(None);
    };
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 1e9 {
        return Err(
            "closed-loop duration_seconds must be finite, greater than zero, and <= 1e9 seconds"
                .to_string(),
        );
    }
    Ok(Some(Duration::from_secs_f64(seconds)))
}

/// Add open-loop queueing/arrival metrics without changing closed-loop statistics.
pub fn attach_open_loop_metrics(
    results: &mut SearchResults,
    plan: OpenLoopPlan,
    schedule_delays: &[f64],
    end_to_end_latencies: &[f64],
    dropped_queries: usize,
    late_queries: usize,
) {
    let percentiles = |values: &[f64]| -> (f64, f64, f64) {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        (
            percentile_linear(&sorted, 0.50),
            percentile_linear(&sorted, 0.95),
            percentile_linear(&sorted, 0.99),
        )
    };
    let (sd50, sd95, sd99) = percentiles(schedule_delays);
    let (e2e50, e2e95, e2e99) = percentiles(end_to_end_latencies);
    results.target_qps = Some(plan.target_qps);
    results.offered_queries = Some(plan.total_requests);
    results.dropped_queries = dropped_queries;
    results.late_queries = late_queries;
    results.schedule_delay_p50_time = Some(sd50);
    results.schedule_delay_p95_time = Some(sd95);
    results.schedule_delay_p99_time = Some(sd99);
    results.end_to_end_p50_time = Some(e2e50);
    results.end_to_end_p95_time = Some(e2e95);
    results.end_to_end_p99_time = Some(e2e99);
}

/// `numpy.percentile` with linear interpolation — the method v0 uses
/// (`np.percentile(..., 50/95/99)` defaults to linear). `sorted` must be
/// ascending and `q` a fraction in `[0, 1]`. The percentile position is
/// `q * (N - 1)`, interpolating between the two neighbouring samples.
///
/// This replaces nearest-rank indexing (`sorted[floor(N*q)]`), which biased
/// every percentile upward and made `p99 == max` for any `N <= 100` (e.g. with
/// N=100, `floor(0.99*100)=99` always selected the single largest sample).
pub fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = q * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Build `SearchResults` for a search-only run from the per-query samples
/// collected by an engine's parallel harness. Centralizes rps/means/std/
/// percentile computation so every engine reports metrics identically.
///
/// `times`/`precisions`/`recalls`/`mrrs`/`ndcgs` are the per-successful-query
/// samples (see the engines' search loops), `total_time` the wall clock,
/// `top` the k used, `parallel` the client concurrency, and `requested_queries`
/// the number of queries dispatched (num_to_run) so failures can be counted as
/// `requested_queries - times.len()`. RPS stays successes/wall-clock; a nonzero
/// `failed_queries` flags that the stats cover only the successful subset.
#[allow(clippy::too_many_arguments)]
pub fn compute_search_stats(
    times: &[f64],
    precisions: &[f64],
    recalls: &[f64],
    mrrs: &[f64],
    ndcgs: &[f64],
    total_time: f64,
    top: usize,
    parallel: usize,
    requested_queries: usize,
) -> Result<SearchResults, String> {
    if times.is_empty() {
        return Err("No searches completed".to_string());
    }

    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };

    let mean_time = mean(times);
    let std_time =
        (times.iter().map(|t| (t - mean_time).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    let min_time = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max_time = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |q: f64| percentile_linear(&sorted, q);

    // Worst-10% recall floor: catches queries that return almost nothing even
    // when the mean recall looks fine.
    let recall_p10 = if recalls.is_empty() {
        0.0
    } else {
        let mut sorted_recalls = recalls.to_vec();
        sorted_recalls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        percentile_linear(&sorted_recalls, 0.10)
    };

    Ok(SearchResults {
        total_time,
        mean_time,
        mean_precision_at_returned: mean(precisions),
        mean_recall: mean(recalls),
        recall_p10,
        mean_mrr: mean(mrrs),
        mean_ndcg: mean(ndcgs),
        recalls: recalls.to_vec(),
        mrrs: mrrs.to_vec(),
        ndcgs: ndcgs.to_vec(),
        std_time,
        min_time,
        max_time,
        rps: times.len() as f64 / total_time,
        p50_time: pct(0.50),
        p95_time: pct(0.95),
        p99_time: pct(0.99),
        precisions_at_returned: precisions.to_vec(),
        latencies: times.to_vec(),
        top,
        num_queries: times.len(),
        requested_queries,
        failed_queries: requested_queries.saturating_sub(times.len()),
        parallel,
        ..Default::default()
    })
}

/// Build a zero-throughput `SearchResults` for a run that made attempts but
/// completed with NO successful queries (e.g. an overloaded open-loop run that
/// shed every request, or a duration run whose every query errored/timed out).
///
/// Returning this instead of an error preserves the strongest overload signal:
/// rps=0 with the full drop/late/failure accounting attached (open-loop metrics
/// are layered on afterwards by the caller via `attach_open_loop_metrics`).
/// `attempted` is the number of dispatched requests; all of them count as
/// failed. Callers should still return a genuine error when there were zero
/// attempts at all (no queries offered).
pub fn zero_search_results(
    total_time: f64,
    top: usize,
    parallel: usize,
    attempted: usize,
) -> SearchResults {
    SearchResults {
        total_time,
        rps: 0.0,
        top,
        num_queries: 0,
        requested_queries: attempted,
        failed_queries: attempted,
        parallel,
        ..Default::default()
    }
}

/// Engine trait - equivalent to Python BaseClient
///
/// Each engine implementation provides:
/// - configure: Create/setup the index
/// - upload: Upload vectors to the index
/// - search: Run search queries
/// - delete: Clean up resources
pub trait Engine {
    /// Get engine name
    fn name(&self) -> &str;

    /// Configure the index (create if needed)
    fn configure(&mut self, dataset: &Dataset) -> Result<(), String>;

    /// Upload vectors to the index
    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String>;

    /// Run search benchmark
    fn search(
        &mut self,
        dataset: &Dataset,
        search_params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String>;

    /// Delete/cleanup the index
    fn delete(&mut self) -> Result<(), String>;

    /// Get search parameter configurations
    fn search_params(&self) -> &[SearchParams];

    /// Collect memory usage stats after upload (matches Python v0 get_memory_usage)
    /// Whether this engine has a sparse / hybrid (dense+sparse) code path.
    ///
    /// Default `false`: only Qdrant implements one. The runner checks this BEFORE
    /// resolving the dataset path, so pointing a sparse dataset at another engine
    /// is skipped with a clear message instead of downloading hundreds of MB,
    /// building an index at the fallback dimension, and only then failing in the
    /// reader with "Unsupported dataset type: sparse".
    fn supports_sparse(&self) -> bool {
        false
    }

    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        None
    }

    /// Number of corpus rows this config would search, **read back off the live
    /// server** (issue #238).
    ///
    /// This is the reuse precondition for `--skip-upload`: the flag asserts "the
    /// corpus is already loaded", and the runner has to be able to check that
    /// assertion against reality rather than trust it. A missing index/collection
    /// answers `Ok(Some(0))` — indistinguishable from an empty one for this
    /// purpose, and both are equally fatal to the reported recall.
    ///
    /// The count must cover exactly the keyspace/collection **this config**
    /// searches (per-config prefix included), so a sibling config's corpus can
    /// never be mistaken for this one's.
    ///
    /// Default `Ok(None)`: "this engine cannot report one". The runner then says
    /// so and proceeds — an unverifiable engine must not become an unusable one.
    /// `Err` means the probe itself failed (unreachable server, refused command);
    /// the runner treats that as fatal unless `--allow-partial-corpus` is set,
    /// because an unverified reuse is exactly how a partial index gets published
    /// as a full one.
    fn corpus_row_count(&mut self) -> Result<Option<u64>, String> {
        Ok(None)
    }

    /// Snapshot server reproducibility metadata (version, loaded modules +
    /// versions, full INFO/CONFIG). Default `None` (non-Redis engines are
    /// unaffected). Redis-wire engines override this to return
    /// `redis_utils::collect_server_metadata`. Telemetry only — captured
    /// outside any timed window, must never affect measurements.
    fn server_metadata(&mut self) -> Option<serde_json::Value> {
        None
    }

    /// Run mixed benchmark (interleaved search + update).
    /// Default: not supported. Override in engines that support it.
    fn search_mixed(
        &mut self,
        _dataset: &Dataset,
        _search_params: &SearchParams,
        _num_queries: i64,
        _ratio: &UpdateSearchRatio,
    ) -> Result<SearchResults, String> {
        Err(format!(
            "mixed benchmark not supported for engine '{}'",
            self.name()
        ))
    }
}

/// Build a Redis connection URL.
///
/// Priority: `REDIS_URI` env var > `REDIS_USER`/`REDIS_AUTH` env vars + host/port.
pub fn build_redis_url(host: &str) -> String {
    if let Ok(uri) = std::env::var("REDIS_URI") {
        return uri;
    }

    let port: u16 = std::env::var("REDIS_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6379);

    let auth = std::env::var("REDIS_AUTH").ok();
    let user = std::env::var("REDIS_USER").ok();

    let auth_part = match (&user, &auth) {
        (Some(u), Some(p)) => format!("{}:{}@", u, p),
        (None, Some(p)) => format!(":{}@", p),
        _ => String::new(),
    };

    format!("redis://{}{}:{}/", auth_part, host, port)
}

/// Shard count the run will actually be measured at, for the results JSON (#211).
///
/// Shard count materially changes vector indexing speed and precision, but the
/// only trace of it in a result file used to be the config *name* — an auditor
/// could infer it at best, and a renamed or hand-edited config left no trace at
/// all. This resolves it the same way the engine does, so the number lands in
/// `params` next to `parallel`/`batch_size`:
///
/// - `opensearch` reads it from `collection_params.number_of_shards` and is the
///   only engine where it is configurable. Always reported — including the
///   `"cluster-default"` sentinel when nothing pinned it, since "nobody chose
///   this" is exactly what a reader of a published result needs to see.
/// - `elasticsearch` pins [`elasticsearch::ES_NUMBER_OF_SHARDS`] in code and
///   ignores the config, so the constant is reported.
/// - every other engine returns `None` and the key is omitted rather than
///   invented.
///
/// This is deliberately one key, not the full engine-params block: capturing
/// `collection_params` verbatim plus the env-derived knobs (`m`,
/// `ef_construction`, retry budgets, `*_UPLOAD_PARALLEL`, …) for all 15 engines
/// is tracked separately in #212.
pub fn resolved_number_of_shards(
    engine_config: &EngineConfig,
) -> Result<Option<serde_json::Value>, String> {
    match engine_config.engine.as_deref() {
        Some("opensearch") => {
            let raw = engine_config
                .collection_params
                .as_ref()
                .and_then(|c| c.extra.as_ref())
                .and_then(|e| e.get("number_of_shards"));
            Ok(Some(match opensearch::parse_number_of_shards(raw)? {
                Some(n) => serde_json::Value::from(n),
                None => serde_json::Value::from("cluster-default"),
            }))
        }
        Some("elasticsearch") => Ok(Some(serde_json::Value::from(
            elasticsearch::ES_NUMBER_OF_SHARDS,
        ))),
        _ => Ok(None),
    }
}

/// Create an engine based on config
pub fn create_engine(engine_config: &EngineConfig, host: &str) -> Result<Box<dyn Engine>, String> {
    let engine_type = engine_config.engine.as_deref().unwrap_or("unknown");

    match engine_type {
        "redis" => Ok(Box::new(RedisEngine::new(engine_config, host)?)),
        "vectorsets" => Ok(Box::new(VectorSetsEngine::new(engine_config, host)?)),
        "elasticsearch" => Ok(Box::new(ElasticsearchEngine::new(engine_config, host)?)),
        "opensearch" => Ok(Box::new(OpenSearchEngine::new(engine_config, host)?)),
        "qdrant" => Ok(Box::new(QdrantEngine::new(engine_config, host)?)),
        "weaviate" => Ok(Box::new(WeaviateEngine::new(engine_config, host)?)),
        "pgvector" => Ok(Box::new(PgVectorEngine::new(engine_config, host)?)),
        "milvus" => Ok(Box::new(MilvusEngine::new(engine_config, host)?)),
        "mongodb" => Ok(Box::new(MongoDBEngine::new(engine_config, host)?)),
        "valkey" => Ok(Box::new(ValkeyEngine::new(engine_config, host)?)),
        "turbopuffer" => Ok(Box::new(TurbopufferEngine::new(engine_config, host)?)),
        "dragonfly" => Ok(Box::new(DragonflyEngine::new(engine_config, host)?)),
        "kividb" => Ok(Box::new(KividbEngine::new(engine_config, host)?)),
        "vertex" => Ok(Box::new(VertexEngine::new(engine_config, host)?)),
        "chroma" => Ok(Box::new(ChromaEngine::new(engine_config, host)?)),
        other => Err(format!(
            "Unsupported engine type: '{}'. Supported: 'redis', 'vectorsets', 'elasticsearch', 'opensearch', 'qdrant', 'weaviate', 'pgvector', 'milvus', 'mongodb', 'valkey', 'turbopuffer', 'dragonfly', 'kividb', 'vertex', 'chroma'.",
            other
        )),
    }
}

#[cfg(test)]
mod stats_tests {
    use super::{attach_open_loop_metrics, compute_search_stats, OpenLoopPlan};
    use crate::config::SearchParams;

    #[test]
    fn empty_times_errors() {
        assert!(compute_search_stats(&[], &[], &[], &[], &[], 1.0, 10, 1, 0).is_err());
    }

    #[test]
    fn computes_means_rps_and_clamped_percentiles() {
        let times = vec![0.1, 0.2, 0.3, 0.4];
        let ones = vec![1.0, 1.0, 1.0, 1.0];
        let r = compute_search_stats(&times, &ones, &ones, &ones, &ones, 2.0, 10, 4, 5).unwrap();
        assert_eq!(r.num_queries, 4);
        assert_eq!(r.requested_queries, 5);
        assert_eq!(r.failed_queries, 1); // 5 requested, 4 succeeded
        assert!((r.rps - 2.0).abs() < 1e-9); // 4 / 2.0s
        assert!((r.mean_recall - 1.0).abs() < 1e-9);
        assert!((r.mean_time - 0.25).abs() < 1e-9);
        assert!((r.min_time - 0.1).abs() < 1e-9 && (r.max_time - 0.4).abs() < 1e-9);
        // percentile indices stay in-bounds (no panic, no 0.0 fallback)
        assert!(r.p50_time > 0.0 && r.p95_time > 0.0 && r.p99_time > 0.0);
        assert_eq!(r.parallel, 4);
        assert_eq!(r.top, 10);
        assert!(r.update_count.is_none());
    }

    #[test]
    fn single_query_percentiles_dont_panic() {
        let r = compute_search_stats(&[0.5], &[1.0], &[1.0], &[1.0], &[1.0], 1.0, 5, 1, 1).unwrap();
        assert!((r.p99_time - 0.5).abs() < 1e-9);
    }

    #[test]
    fn percentile_linear_matches_numpy() {
        use super::percentile_linear;
        // np.percentile([1..=4], [50,95,99]) with linear interpolation:
        // position = q*(N-1) = q*3.
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile_linear(&v, 0.50) - 2.5).abs() < 1e-9); // 1.5 -> 2.5
        assert!((percentile_linear(&v, 0.95) - 3.85).abs() < 1e-9); // 2.85 -> 3.85
        assert!((percentile_linear(&v, 0.99) - 3.97).abs() < 1e-9); // 2.97 -> 3.97
                                                                    // Degenerate cases.
        assert_eq!(percentile_linear(&[], 0.5), 0.0);
        assert_eq!(percentile_linear(&[7.0], 0.99), 7.0);
    }

    #[test]
    fn p99_is_not_max_for_n100() {
        use super::percentile_linear;
        // The nearest-rank pathology: with N=100, floor(0.99*100)=99 always
        // returned the single max. Linear gives position 0.99*99=98.01, i.e.
        // just above sorted[98], strictly below the max at sorted[99].
        let sorted: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let p99 = percentile_linear(&sorted, 0.99);
        assert!(p99 < 100.0, "p99={} should be below max", p99);
        assert!(p99 > 99.0, "p99={} should be above sorted[98]", p99);
        assert!((p99 - 99.01).abs() < 1e-9, "p99={}", p99);
    }

    #[test]
    fn filter_mixed_stats_use_linear_percentiles() {
        // The filter-only and mixed harnesses now route their latency samples
        // through compute_search_stats (linear interpolation) instead of the
        // old hand-rolled nearest-rank indexing `(len*q) as usize`, which
        // returned the single max as p99 for N<=100. Feeding a known sample set
        // (1..=100) through the shared path must yield the numpy-linear
        // percentiles, proving the biased method is gone for these harnesses.
        let times: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let r = compute_search_stats(&times, &[], &[], &[], &[], 10.0, 0, 4, 100).unwrap();
        // Nearest-rank would have produced p99 == 100 (the max); linear gives 99.01.
        assert!((r.p99_time - 99.01).abs() < 1e-9, "p99={}", r.p99_time);
        assert!((r.p95_time - 95.05).abs() < 1e-9, "p95={}", r.p95_time);
        assert!((r.p50_time - 50.5).abs() < 1e-9, "p50={}", r.p50_time);
        assert!(r.p99_time < 100.0, "p99={} must be below max", r.p99_time);
    }

    #[test]
    fn open_loop_plan_validates_and_counts_arrivals() {
        let params: SearchParams = serde_json::from_value(serde_json::json!({
            "parallel": 32,
            "target_qps": 1500.0,
            "duration_seconds": 2.0,
            "max_lateness_ms": 250.0
        }))
        .unwrap();
        let plan = OpenLoopPlan::from_params(&params).unwrap().unwrap();
        assert_eq!(plan.total_requests, 3000);
        assert_eq!(plan.target_qps, 1500.0);
        assert_eq!(plan.max_lateness.as_millis(), 250);

        let missing_duration: SearchParams = serde_json::from_value(serde_json::json!({
            "target_qps": 100.0
        }))
        .unwrap();
        assert!(OpenLoopPlan::from_params(&missing_duration).is_err());
    }

    #[test]
    fn open_loop_metrics_preserve_service_latency_and_add_queueing() {
        let params: SearchParams = serde_json::from_value(serde_json::json!({
            "target_qps": 10.0,
            "duration_seconds": 1.0
        }))
        .unwrap();
        let plan = OpenLoopPlan::from_params(&params).unwrap().unwrap();
        let ones = vec![1.0, 1.0];
        let mut results =
            compute_search_stats(&[0.010, 0.020], &ones, &ones, &ones, &ones, 1.0, 10, 2, 2)
                .unwrap();
        attach_open_loop_metrics(
            &mut results,
            plan,
            &[0.001, 0.003, 0.5],
            &[0.011, 0.023],
            1,
            1,
        );
        assert_eq!(results.target_qps, Some(10.0));
        assert_eq!(results.offered_queries, Some(10));
        assert_eq!(results.dropped_queries, 1);
        assert_eq!(results.late_queries, 1);
        assert_eq!(results.p50_time, 0.015);
        assert!(results.schedule_delay_p95_time.unwrap() > 0.4);
        assert_eq!(results.end_to_end_p50_time, Some(0.017));
    }

    #[test]
    fn recall_p10_surfaces_worst_decile() {
        // 20% of queries return nothing (recall 0), 80% perfect. The mean looks
        // healthy (0.8) but recall_p10 exposes the zero-recall slice — the whole
        // point of reporting the distribution, not just the mean.
        let recalls = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let times = vec![0.001; 10];
        let r = compute_search_stats(
            &times, &recalls, &recalls, &recalls, &recalls, 1.0, 10, 1, 10,
        )
        .unwrap();
        assert!((r.mean_recall - 0.8).abs() < 1e-9);
        assert!(
            r.recall_p10 <= 0.1,
            "recall_p10 should surface the ~zero worst decile, got {}",
            r.recall_p10
        );
        // All-perfect recall → p10 is also 1.0 (no false alarm).
        let ones = vec![1.0; 10];
        let r2 = compute_search_stats(&times, &ones, &ones, &ones, &ones, 1.0, 10, 1, 10).unwrap();
        assert!((r2.recall_p10 - 1.0).abs() < 1e-9);
    }
}
