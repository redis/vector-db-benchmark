//! Request construction and policy validation for Azure Cosmos DB for NoSQL.

use azure_data_cosmos::models::{
    ContainerProperties, IndexingMode, IndexingPolicy, VectorDataType, VectorDistanceFunction,
    VectorEmbedding, VectorEmbeddingPolicy, VectorIndex, VectorIndexType,
};
use azure_data_cosmos::Query;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use futures::{stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::future::Future;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Semaphore;

const VECTOR_PATH: &str = "/embedding";
const VECTOR_EXCLUSION: &str = "/embedding/*";

/// Resolve a Cosmos index type. Config names use the service's spelling.
pub fn index_type(algorithm: Option<&str>) -> Result<VectorIndexType, String> {
    match algorithm {
        Some("flat") => Ok(VectorIndexType::Flat),
        Some("diskANN") => Ok(VectorIndexType::DiskANN),
        _ => Err("Cosmos DB algorithm must be flat or diskANN".into()),
    }
}

/// Separate containers by both benchmark config and dataset, even when names
/// contain punctuation that Cosmos forbids in resource IDs.
pub fn container_name(config: &str, dataset: &str) -> String {
    let mut hash = Sha256::new();
    hash.update((config.len() as u64).to_le_bytes());
    hash.update(config);
    hash.update(dataset);
    format!("vdb-benchmark-{}", URL_SAFE_NO_PAD.encode(hash.finalize()))
}

/// Optional DiskANN index-build settings. Absent fields leave policy defaults
/// to Azure, including its dimension-dependent quantization size.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiskAnnBuildOptions {
    indexing_search_list_size: Option<u32>,
    quantization_byte_size: Option<u32>,
}

impl DiskAnnBuildOptions {
    /// Validate supplied collection parameters against the data-plane SDK's
    /// limits. The dimension-dependent cap is checked when building the policy.
    pub fn from_config(
        kind: &VectorIndexType,
        indexing_search_list_size: Option<&Value>,
        quantization_byte_size: Option<&Value>,
    ) -> Result<Self, String> {
        if *kind != VectorIndexType::DiskANN
            && (indexing_search_list_size.is_some() || quantization_byte_size.is_some())
        {
            return Err("Cosmos DB collection index-build options require DiskANN".into());
        }

        let parse = |value: &Value, key: &str, min: u32, max: u32| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| (min..=max).contains(value))
                .ok_or_else(|| {
                    format!(
                        "Cosmos DB collection_params.{key} must be an integer from {min} to {max}"
                    )
                })
        };
        Ok(Self {
            indexing_search_list_size: indexing_search_list_size
                .map(|value| parse(value, "indexing_search_list_size", 25, 500))
                .transpose()?,
            quantization_byte_size: quantization_byte_size
                .map(|value| parse(value, "quantization_byte_size", 1, 512))
                .transpose()?,
        })
    }
}

/// Reject a config key the adapter does not read, so a typo cannot run as a
/// default and be reported under the declared name.
pub fn reject_unknown_keys<'a>(
    keys: impl IntoIterator<Item = &'a String>,
    allowed: &[&str],
    message: &str,
) -> Result<(), String> {
    if keys.into_iter().any(|key| !allowed.contains(&key.as_str())) {
        return Err(message.into());
    }
    Ok(())
}

/// A tuning knob that must be a positive 32-bit integer.
pub fn positive_u32(value: &Value, key: &str) -> Result<NonZeroU32, String> {
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .and_then(NonZeroU32::new)
        .ok_or_else(|| format!("Cosmos DB {key} must be a positive 32-bit integer"))
}

/// Manual RU/s for the container. A per-config value wins over the
/// environment setting. `None` means a serverless account.
pub fn manual_throughput(configured: Option<&Value>, env: &str) -> Result<Option<u64>, String> {
    if let Some(value) = configured {
        return value
            .as_u64()
            .filter(|ru| *ru >= 400)
            .map(Some)
            .ok_or_else(|| {
                "Cosmos DB collection_params.ru_per_second must be an integer of at least 400"
                    .into()
            });
    }
    if env == "serverless" {
        return Ok(None);
    }
    env.parse::<u64>()
        .ok()
        .filter(|ru| *ru >= 400)
        .map(Some)
        .ok_or_else(|| "AZURE_COSMOS_RU_PER_SECOND must be at least 400, or serverless".into())
}

/// Concurrent single-item upserts. The adapter sends one item per request.
pub fn upload_parallel(params: Option<&Value>) -> Result<usize, String> {
    const DEFAULT: usize = 8;
    let Some(params) = params else {
        return Ok(DEFAULT);
    };
    let options = params
        .as_object()
        .ok_or("Cosmos DB upload_params must be an object")?;
    reject_unknown_keys(
        options.keys(),
        &["parallel"],
        "Cosmos DB upload_params supports only parallel (one item per request)",
    )?;
    match options.get("parallel") {
        None => Ok(DEFAULT),
        Some(parallel) => parallel
            .as_u64()
            .and_then(|p| usize::try_from(p).ok())
            .filter(|p| *p > 0)
            .ok_or_else(|| "Cosmos DB upload parallel must be positive".into()),
    }
}

/// Cosmos DB metric corresponding to a benchmark corpus's distance.
pub fn vector_distance_function(distance: &str) -> Result<VectorDistanceFunction, String> {
    match distance.to_ascii_lowercase().as_str() {
        "cosine" | "angular" => Ok(VectorDistanceFunction::Cosine),
        "l2" | "euclidean" => Ok(VectorDistanceFunction::Euclidean),
        "dot" | "ip" => Ok(VectorDistanceFunction::DotProduct),
        _ => Err(format!(
            "Cosmos DB does not support dataset distance '{distance}'"
        )),
    }
}

/// Use the dataset metric and dimension in the server-side policy.
pub fn container_policy(
    name: String,
    index_type: VectorIndexType,
    distance: &str,
    dimensions: i64,
    build: DiskAnnBuildOptions,
) -> Result<ContainerProperties, String> {
    let distance = vector_distance_function(distance)?;
    let max_dimensions = if index_type == VectorIndexType::Flat {
        505
    } else {
        4096
    };
    if !(1..=max_dimensions).contains(&dimensions) {
        return Err(format!(
            "Cosmos DB {index_type:?} requires 1..={max_dimensions} dimensions, got {dimensions}"
        ));
    }

    let mut vector_index = VectorIndex::new(VECTOR_PATH, index_type);
    if let Some(size) = build.indexing_search_list_size {
        vector_index = vector_index.with_indexing_search_list_size(size);
    }
    if let Some(size) = build.quantization_byte_size {
        if i64::from(size) > dimensions {
            return Err(format!(
                "Cosmos DB collection_params.quantization_byte_size ({size}) exceeds the vector dimensions ({dimensions})"
            ));
        }
        vector_index = vector_index.with_quantization_byte_size(size);
    }
    let mut indexing = IndexingPolicy::default()
        .with_indexing_mode(IndexingMode::Consistent)
        .with_included_path("/*")
        .with_excluded_path(VECTOR_EXCLUSION)
        .with_vector_index(vector_index);
    indexing.automatic = true;
    let mut properties = ContainerProperties::new(name, "/id".into());
    properties.indexing_policy = Some(indexing);
    properties.vector_embedding_policy = Some(VectorEmbeddingPolicy::default().with_embedding(
        VectorEmbedding::new(
            VECTOR_PATH,
            VectorDataType::Float32,
            dimensions as u32,
            distance,
        ),
    ));
    Ok(properties)
}

/// Check the server's realized vector policy, including before reuse.
pub fn verify_policy(
    actual: &ContainerProperties,
    expected: &ContainerProperties,
) -> Result<(), String> {
    let actual_embedding = actual
        .vector_embedding_policy
        .as_ref()
        .map(|p| &p.embeddings);
    let expected_embedding = expected
        .vector_embedding_policy
        .as_ref()
        .map(|p| &p.embeddings);
    let actual_indexing = actual.indexing_policy.as_ref();
    let expected_index = expected
        .indexing_policy
        .as_ref()
        .and_then(|p| p.vector_indexes.first());
    if actual.partition_key != expected.partition_key
        || actual_embedding != expected_embedding
        || !actual_indexing.is_some_and(|p| {
            p.automatic
                && p.indexing_mode == Some(IndexingMode::Consistent)
                && p.excluded_paths
                    .iter()
                    .any(|path| path.path == VECTOR_EXCLUSION)
                // Azure fills in service-selected DiskANN defaults on readback.
                // Compare the fields we set and record the full realized policy.
                && matches!(p.vector_indexes.as_slice(), [index] if expected_index.is_some_and(|expected| {
                    index.path == expected.path
                        && index.index_type == expected.index_type
                        && expected.indexing_search_list_size.is_none_or(|size| {
                            index.indexing_search_list_size == Some(size)
                        })
                        && expected.quantization_byte_size.is_none_or(|size| {
                            index.quantization_byte_size == Some(size)
                        })
                }))
        })
    {
        return Err(format!(
            "Cosmos DB container '{}' has an incompatible partition key, embedding policy, vector index or index-build settings",
            actual.id
        ));
    }
    Ok(())
}

/// The SDK's buffered cross-partition query window defaults to 1,000.
pub fn validate_top(top: usize) -> Result<(), String> {
    if !(1..=1000).contains(&top) {
        return Err(format!(
            "Cosmos DB top must be between 1 and 1000, got {top}"
        ));
    }
    Ok(())
}

/// Reject an SDK response that exceeds the global TOP K across all its pages.
pub fn validate_hit_count(hits: usize, top: usize) -> Result<(), String> {
    if hits > top {
        return Err(format!(
            "Cosmos DB returned {hits} hits for global TOP {top}"
        ));
    }
    Ok(())
}

/// SDK diagnostics accumulated over the pages of measured queries.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct SearchDiagnostics {
    pub request_charge_ru: f64,
    pub request_attempts: usize,
    pub http_429_attempts: usize,
    pub attempts_without_status: usize,
    pub queries_without_request_diagnostics: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partition_fanout: Option<PartitionFanout>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PartitionFanout {
    pub feed_ranges: usize,
    pub per_query_concurrency: usize,
    pub shared_request_limit: usize,
    pub parity: ParityTally,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PartitionConcurrency {
    Auto,
    Limit(NonZeroU32),
}

impl PartitionConcurrency {
    /// Missing selects the stock SDK path. Explicit fan-out accepts "auto" or a cap.
    pub fn from_config(value: Option<&Value>) -> Result<Option<Self>, String> {
        value
            .map(|value| {
                if value.as_str() == Some("auto") {
                    return Ok(Self::Auto);
                }
                value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .and_then(NonZeroU32::new)
                    .map(Self::Limit)
                    .ok_or_else(|| {
                        "Cosmos DB partition_concurrency must be \"auto\" or a positive 32-bit integer".into()
                    })
            })
            .transpose()
    }

    pub fn resolve(&self, feed_ranges: usize) -> usize {
        match self {
            Self::Auto => feed_ranges,
            Self::Limit(cap) => feed_ranges.min(cap.get() as usize),
        }
    }
}

impl SearchDiagnostics {
    /// Use the exact SDK totals and count HTTP statuses from either individual
    /// attempts or their compacted rollup. Omitted statuses remain unknown.
    pub fn observe_page(
        &mut self,
        request_charge_ru: f64,
        request_attempts: usize,
        attempt_statuses: impl IntoIterator<Item = (u16, usize)>,
    ) {
        self.request_charge_ru += request_charge_ru;
        self.request_attempts += request_attempts;
        let mut classified = 0;
        for (status, count) in attempt_statuses {
            if (100..600).contains(&status) {
                classified += count;
            }
            if status == 429 {
                self.http_429_attempts += count;
            }
        }
        self.attempts_without_status += request_attempts.saturating_sub(classified);
    }

    pub fn add_query(&mut self, query: Self) {
        self.request_charge_ru += query.request_charge_ru;
        self.request_attempts += query.request_attempts;
        self.http_429_attempts += query.http_429_attempts;
        self.attempts_without_status += query.attempts_without_status;
        self.queries_without_request_diagnostics += usize::from(
            query.request_attempts == 0 || query.queries_without_request_diagnostics > 0,
        );
    }
}

/// Fail on corpora too small to qualify for the service's DiskANN threshold.
pub fn validate_corpus_size(kind: &VectorIndexType, count: u64) -> Result<(), String> {
    if *kind == VectorIndexType::DiskANN && count < 1000 {
        return Err(format!(
            "Cosmos DB DiskANN needs at least 1000 vectors to use its index; got {count}. Use cosmosdb-flat for this corpus."
        ));
    }
    Ok(())
}

/// The text of a server-side TOP K query, built once per search. The service
/// uses the policy's distance function and TOP is a validated integer literal.
/// Only the bound vector differs between requests, so a search keeps one text
/// and binds each vector when it sends the request. Pre-building every query
/// would hold each vector a second time as JSON values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnnQueryText(String);

impl KnnQueryText {
    pub fn new(
        top: usize,
        search_list_size_multiplier: Option<NonZeroU32>,
        quantized_vector_list_multiplier: Option<NonZeroU32>,
        project_distance: bool,
    ) -> Result<Self, String> {
        validate_top(top)?;
        let options = match (search_list_size_multiplier, quantized_vector_list_multiplier) {
            (None, None) => String::new(),
            (Some(search), None) => format!(", {{\"searchListSizeMultiplier\": {search}}}"),
            (None, Some(quantized)) => {
                format!(", {{\"quantizedVectorListMultiplier\": {quantized}}}")
            }
            (Some(search), Some(quantized)) => format!(
                ", {{\"searchListSizeMultiplier\": {search}, \"quantizedVectorListMultiplier\": {quantized}}}"
            ),
        };
        Ok(Self(query_text(
            top,
            &format!("false{options}"),
            project_distance,
        )))
    }

    /// Bind one query vector. The caller checks it with
    /// [`validate_query_vector`] before timing, so this does no validation.
    pub fn bind(&self, vector: &[f32]) -> Result<Query, String> {
        bind_vector(self.0.clone(), vector)
    }
}

/// A query vector must match the container's dimensions and be finite. JSON
/// has no encoding for NaN or infinity, so one would be sent as `null`.
pub fn validate_query_vector(vector: &[f32], dimensions: usize) -> Result<(), String> {
    if vector.len() != dimensions {
        return Err("Cosmos DB query vector has the wrong number of dimensions".into());
    }
    if vector.iter().any(|v| !v.is_finite()) {
        return Err("Cosmos DB query vector contains a non-finite value".into());
    }
    Ok(())
}

/// Build and bind one validated TOP K query.
pub fn knn_query(
    vector: &[f32],
    top: usize,
    search_list_size_multiplier: Option<NonZeroU32>,
    quantized_vector_list_multiplier: Option<NonZeroU32>,
    project_distance: bool,
) -> Result<Query, String> {
    validate_query_vector(vector, vector.len())?;
    KnnQueryText::new(
        top,
        search_list_size_multiplier,
        quantized_vector_list_multiplier,
        project_distance,
    )?
    .bind(vector)
}

/// The same TOP K query with the service's brute-force flag set, so it scans
/// every vector in scope. Its request charge is the baseline that the
/// index-use probe compares an index-requested query against.
pub fn scan_query(vector: &[f32], top: usize) -> Result<Query, String> {
    validate_top(top)?;
    validate_query_vector(vector, vector.len())?;
    bind_vector(query_text(top, "true", false), vector)
}

fn query_text(top: usize, distance_arguments: &str, project_distance: bool) -> String {
    let distance = format!("VectorDistance(c.embedding, @vector, {distance_arguments})");
    let projection = if project_distance {
        format!("c.id, {distance} AS distance")
    } else {
        "c.id".into()
    };
    format!("SELECT TOP {top} {projection} FROM c ORDER BY {distance}")
}

fn bind_vector(text: String, vector: &[f32]) -> Result<Query, String> {
    Query::from(text)
        .with_parameter("@vector", vector)
        .map_err(|e| format!("Could not bind Cosmos DB query vector: {e}"))
}

/// An index-requested query must cost less than this share of a forced scan
/// of the same feed range before the range counts as served by DiskANN.
pub const INDEX_PROBE_MAX_RU_RATIO: f64 = 0.5;

/// Request charges of one feed range's forced scan and index-requested query.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct IndexProbe {
    pub feed_range: usize,
    pub scan_ru: f64,
    pub index_ru: f64,
}

impl IndexProbe {
    /// Azure answers an index-requested query with a scan while the index is
    /// building or the range holds too few vectors. A scan costs what the
    /// forced scan costs, so only a clearly cheaper query shows index use.
    pub fn index_in_use(&self) -> bool {
        self.scan_ru > 0.0
            && self.index_ru > 0.0
            && self.index_ru < self.scan_ru * INDEX_PROBE_MAX_RU_RATIO
    }
}

/// How long to wait for every feed range to serve from its DiskANN index.
/// Unset uses 30 minutes. `0` skips the probe and leaves index use unverified.
pub fn index_probe_deadline(value: Option<&str>) -> Result<Option<Duration>, String> {
    const DEFAULT_SECONDS: u64 = 1800;
    let seconds = match value {
        None => DEFAULT_SECONDS,
        Some(value) => value.parse::<u64>().map_err(|_| {
            "AZURE_COSMOS_INDEX_PROBE_SECONDS must be a whole number of seconds, or 0 to skip the probe"
        })?,
    };
    Ok((seconds > 0).then(|| Duration::from_secs(seconds)))
}

/// Pause between index-use polls: a tenth of the time already waited, within
/// 2 to 30 seconds. The recorded index wait is then at most about a tenth
/// too long, and a long build is not polled with a scan every few seconds.
pub fn index_probe_interval(waited: Duration) -> Duration {
    (waited / 10).clamp(Duration::from_secs(2), Duration::from_secs(30))
}

/// The charges of the feed ranges that are not served by the index yet.
pub fn describe_waiting(probes: &[IndexProbe]) -> String {
    const SHOWN: usize = 8;
    let waiting: Vec<_> = probes.iter().filter(|p| !p.index_in_use()).collect();
    let mut text = waiting
        .iter()
        .take(SHOWN)
        .map(|p| {
            format!(
                "range {}: index query {:.2} RU, forced scan {:.2} RU",
                p.feed_range, p.index_ru, p.scan_ru
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    if waiting.len() > SHOWN {
        text.push_str(&format!("; and {} more", waiting.len() - SHOWN));
    }
    text
}

/// Which of the two probe queries to run on a feed range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeQuery {
    ForcedScan,
    IndexRequested,
}

/// Wait until every feed range answers the index-requested query for clearly
/// less than its forced scan. `charges` returns (range index, RU) for the
/// listed ranges. The forced scan runs once per range, and only ranges still
/// waiting are queried again. `waited` and `pause` are the clock.
pub fn await_index_use(
    feed_ranges: usize,
    deadline: Duration,
    mut charges: impl FnMut(ProbeQuery, &[usize]) -> Result<Vec<(usize, f64)>, String>,
    waited: impl Fn() -> Duration,
    mut pause: impl FnMut(Duration),
) -> Result<Vec<IndexProbe>, String> {
    let mut probes: Vec<IndexProbe> = (0..feed_ranges)
        .map(|feed_range| IndexProbe {
            feed_range,
            scan_ru: f64::NAN,
            index_ru: f64::NAN,
        })
        .collect();
    let mut pending: Vec<usize> = (0..feed_ranges).collect();
    let mut query = ProbeQuery::ForcedScan;
    loop {
        let answers = charges(query, &pending)?;
        let mut answered: Vec<usize> = answers.iter().map(|(range, _)| *range).collect();
        answered.sort_unstable();
        if answered != pending {
            return Err(format!(
                "Cosmos DB index-use probe asked feed ranges {pending:?} and got answers for {answered:?}"
            ));
        }
        for (feed_range, ru) in answers {
            match query {
                ProbeQuery::ForcedScan => probes[feed_range].scan_ru = ru,
                ProbeQuery::IndexRequested => probes[feed_range].index_ru = ru,
            }
        }
        if query == ProbeQuery::ForcedScan {
            query = ProbeQuery::IndexRequested;
            continue;
        }
        pending.retain(|&feed_range| !probes[feed_range].index_in_use());
        if pending.is_empty() {
            return Ok(probes);
        }
        let Some(remaining) = deadline
            .checked_sub(waited())
            .filter(|left| !left.is_zero())
        else {
            return Err(format!(
                "Cosmos DB DiskANN index is not serving {} of {feed_ranges} feed ranges after {}s. An index-requested query still costs as much as a forced scan ({}). The index may still be building, or a range may hold too few vectors. Raise AZURE_COSMOS_INDEX_PROBE_SECONDS, or set it to 0 to run unverified.",
                pending.len(),
                deadline.as_secs(),
                describe_waiting(&probes)
            ));
        };
        println!(
            "Waiting for the Cosmos DB DiskANN index on {} of {feed_ranges} feed ranges after {:.0}s ({})",
            pending.len(),
            waited().as_secs_f64(),
            describe_waiting(&probes)
        );
        pause(index_probe_interval(waited()).min(remaining));
    }
}

/// Delay before retrying an upload write that Azure throttled, or `None`
/// once the budget is spent. The SDK retries a 429 a few times on its own.
/// This budget sits on top of that, so a corpus larger than a smoke test
/// still uploads at a throughput below what the writers ask for.
pub fn upload_throttle_backoff(retry: u32) -> Option<Duration> {
    const RETRIES: u32 = 60;
    (retry < RETRIES)
        .then(|| Duration::from_millis(250 << retry.min(5)).min(Duration::from_secs(5)))
}

/// Run `write` until it succeeds, fails for a reason other than throttling,
/// or exhausts [`upload_throttle_backoff`]. Every retry is counted.
pub async fn retry_throttled<T, E, Write, Sleep>(
    mut write: impl FnMut() -> Write,
    throttled: impl Fn(&E) -> bool,
    sleep: impl Fn(Duration) -> Sleep,
    retries: &AtomicUsize,
) -> Result<T, E>
where
    Write: Future<Output = Result<T, E>>,
    Sleep: Future<Output = ()>,
{
    let mut retry = 0;
    loop {
        match write().await {
            Err(error) if throttled(&error) => {
                let Some(delay) = upload_throttle_backoff(retry) else {
                    return Err(error);
                };
                retries.fetch_add(1, Ordering::Relaxed);
                retry += 1;
                sleep(delay).await;
            }
            other => return other,
        }
    }
}

/// Sum a per-range count over disjoint feed ranges, `concurrency` at a time.
/// A split during the count could count a row twice or not at all, so the
/// ranges are read again afterwards and must be unchanged.
pub async fn count_stable_ranges<R, Ranges, Count>(
    read_ranges: impl Fn() -> Ranges,
    concurrency: usize,
    count: impl Fn(R) -> Count,
) -> Result<u64, String>
where
    R: Clone + PartialEq,
    Ranges: Future<Output = Result<Vec<R>, String>>,
    Count: Future<Output = Result<u64, String>>,
{
    let ranges = read_ranges().await?;
    let total = stream::iter(ranges.iter().cloned())
        .map(&count)
        .buffer_unordered(concurrency.max(1))
        .try_fold(0_u64, |total, rows| async move { Ok(total + rows) })
        .await?;
    if read_ranges().await? != ranges {
        return Err("Cosmos DB feed ranges changed during the row count".into());
    }
    Ok(total)
}

/// Query every feed range, at most `per_query` at a time for this query and
/// one `permits` permit per request across all queries, then select the
/// global top K and add up the diagnostics.
pub async fn fan_out_hits<R, Hits>(
    ranges: &[R],
    per_query: usize,
    permits: &Semaphore,
    top: usize,
    metric: &VectorDistanceFunction,
    query: impl Fn(R) -> Hits,
) -> Result<(Vec<QueryHit>, SearchDiagnostics), String>
where
    R: Clone,
    Hits: Future<Output = Result<(Vec<QueryHit>, SearchDiagnostics), String>>,
{
    let query = &query;
    let (hits, diagnostics) = stream::iter(ranges.iter().cloned())
        .map(|range| async move {
            let _permit = permits.acquire().await.map_err(|e| e.to_string())?;
            query(range).await
        })
        .buffer_unordered(per_query.max(1))
        .try_fold(
            (Vec::new(), SearchDiagnostics::default()),
            |(mut hits, mut diagnostics), (range_hits, range_diagnostics)| async move {
                hits.extend(range_hits);
                diagnostics.add_query(range_diagnostics);
                Ok((hits, diagnostics))
            },
        )
        .await?;
    Ok((merge_partition_hits(hits, top, metric)?, diagnostics))
}

#[derive(Deserialize)]
pub struct QueryHit {
    pub id: String,
    pub distance: Option<f64>,
}

/// Select the global top K from the scored candidates of disjoint feed ranges.
pub fn merge_partition_hits(
    hits: Vec<QueryHit>,
    top: usize,
    metric: &VectorDistanceFunction,
) -> Result<Vec<QueryHit>, String> {
    // The SQL vector ORDER BY is best-first for every metric. Its projected
    // value is a similarity for cosine/dot product and a distance for L2.
    let higher_is_better = match metric {
        VectorDistanceFunction::Cosine | VectorDistanceFunction::DotProduct => true,
        VectorDistanceFunction::Euclidean => false,
        _ => return Err("Unsupported Cosmos DB vector score ordering".into()),
    };
    let mut candidates = hits
        .into_iter()
        .map(|hit| {
            let distance = hit
                .distance
                .filter(|distance| distance.is_finite())
                .ok_or("Cosmos DB partition query returned a hit without a finite distance")?;
            Ok((hit, distance))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut seen = HashSet::with_capacity(candidates.len());
    for (hit, _) in &candidates {
        if !seen.insert(hit.id.as_str()) {
            return Err(format!(
                "Cosmos DB partition queries returned duplicate id {}",
                hit.id
            ));
        }
    }
    candidates.sort_unstable_by(|(left, a), (right, b)| {
        let order = a.total_cmp(b);
        (if higher_is_better {
            order.reverse()
        } else {
            order
        })
        .then_with(|| left.id.cmp(&right.id))
    });
    candidates.truncate(top);
    Ok(candidates.into_iter().map(|(hit, _)| hit).collect())
}

/// Fan-out may miss this many times the IDs that a repeated stock request
/// misses. Two stock comparisons are counted per sample, so 2 is the expected
/// ratio when both routes behave alike. 3 leaves headroom for sampling noise.
pub const PARITY_CROSS_MODE_ALLOWANCE: usize = 3;

/// Agreement between the stock SDK query and the feed-range fan-out on
/// sampled queries, measured against the stock query's own repeatability.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ParityTally {
    pub samples: usize,
    /// IDs of one stock response that a repeated stock request did not return.
    pub stock_repeat_missing_ids: usize,
    /// IDs of the two stock responses that the fan-out response did not return.
    pub cross_mode_missing_ids: usize,
}

impl ParityTally {
    /// Record one sampled query: two stock responses and one fan-out response.
    pub fn observe(
        &mut self,
        stock: &[QueryHit],
        stock_repeat: &[QueryHit],
        partitioned: &[QueryHit],
    ) -> Result<(), String> {
        if stock.len() != stock_repeat.len() || stock.len() != partitioned.len() {
            return Err(format!(
                "Cosmos DB routing modes returned different hit counts ({} and {} stock, {} fan-out)",
                stock.len(),
                stock_repeat.len(),
                partitioned.len()
            ));
        }
        let ids = |hits: &[QueryHit]| -> HashSet<String> {
            hits.iter().map(|hit| hit.id.clone()).collect()
        };
        let (stock, stock_repeat, partitioned) = (ids(stock), ids(stock_repeat), ids(partitioned));
        self.samples += 1;
        self.stock_repeat_missing_ids += stock.difference(&stock_repeat).count();
        self.cross_mode_missing_ids +=
            stock.difference(&partitioned).count() + stock_repeat.difference(&partitioned).count();
        Ok(())
    }

    /// Fan-out must agree with the stock query about as well as the stock
    /// query agrees with itself. An exact engine therefore needs exact parity.
    pub fn verdict(&self) -> Result<(), String> {
        let allowed = self.stock_repeat_missing_ids * PARITY_CROSS_MODE_ALLOWANCE;
        if self.samples == 0 || self.cross_mode_missing_ids > allowed {
            return Err(format!(
                "Cosmos DB partition search disagrees with the SDK's global top K: fan-out missed {} stock IDs over {} sampled queries, and repeated stock requests missed {} (allowance {allowed})",
                self.cross_mode_missing_ids, self.samples, self.stock_repeat_missing_ids
            ));
        }
        Ok(())
    }
}

/// Query indices spread evenly over the first `limit` queries, at most `samples`.
pub fn parity_sample_indices(limit: usize, samples: usize) -> Vec<usize> {
    (0..limit)
        .step_by((limit / samples.max(1)).max(1))
        .take(samples)
        .collect()
}

/// Run `sample` for each query index, `concurrency` at a time. It returns the
/// two stock responses and the fan-out response. The tally of all samples
/// must then pass [`ParityTally::verdict`].
pub async fn sample_parity<Sample>(
    indices: Vec<usize>,
    concurrency: usize,
    sample: impl Fn(usize) -> Sample,
) -> Result<ParityTally, String>
where
    Sample: Future<Output = Result<[Vec<QueryHit>; 3], String>>,
{
    let sample = &sample;
    let outcomes: Vec<_> = stream::iter(indices)
        .map(|index| async move {
            sample(index)
                .await
                .map_err(|e| format!("Cosmos DB parity check for query {index}: {e}"))
        })
        .buffer_unordered(concurrency.max(1))
        .try_collect()
        .await?;
    let mut tally = ParityTally::default();
    for [stock, stock_repeat, partitioned] in &outcomes {
        tally.observe(stock, stock_repeat, partitioned)?;
    }
    tally.verdict()?;
    Ok(tally)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, to_value};

    #[test]
    fn distance_and_diskann_policy_are_in_the_wire_request() {
        let defaults = DiskAnnBuildOptions::default();
        let p = container_policy(
            "bench".into(),
            VectorIndexType::DiskANN,
            "dot",
            1024,
            defaults,
        )
        .unwrap();
        let wire = to_value(&p).unwrap();
        assert_eq!(wire["partitionKey"]["paths"], json!(["/id"]));
        assert_eq!(
            wire["vectorEmbeddingPolicy"]["vectorEmbeddings"][0],
            json!({
                "path": "/embedding", "dataType": "float32", "dimensions": 1024,
                "distanceFunction": "dotproduct"
            })
        );
        assert_eq!(
            wire["indexingPolicy"]["vectorIndexes"][0]["type"],
            "diskANN"
        );
        let default_index = &wire["indexingPolicy"]["vectorIndexes"][0];
        assert!(default_index.get("quantizationByteSize").is_none());
        assert!(default_index.get("indexingSearchListSize").is_none());
        let mut realized = wire;
        let vector_index = &mut realized["indexingPolicy"]["vectorIndexes"][0];
        vector_index["quantizerType"] = json!("product");
        vector_index["quantizationByteSize"] = json!(16);
        vector_index["indexingSearchListSize"] = json!(100);
        let actual = serde_json::from_value(realized.clone()).unwrap();
        assert!(verify_policy(&actual, &p).is_ok());
        realized["indexingPolicy"]["vectorIndexes"][0]["type"] = json!("flat");
        let wrong_index = serde_json::from_value(realized).unwrap();
        assert!(verify_policy(&wrong_index, &p).is_err());
        let other = container_policy(
            "bench".into(),
            VectorIndexType::DiskANN,
            "l2",
            1024,
            defaults,
        )
        .unwrap();
        assert!(verify_policy(&p, &other).is_err());
        assert!(verify_policy(&p, &p).is_ok());
        assert!(container_policy(
            "bench".into(),
            VectorIndexType::Flat,
            "cosine",
            506,
            defaults
        )
        .is_err());
    }

    fn tuned_diskann_policy() -> ContainerProperties {
        let kind = VectorIndexType::DiskANN;
        let options =
            DiskAnnBuildOptions::from_config(&kind, Some(&json!(150)), Some(&json!(32))).unwrap();
        container_policy("bench".into(), kind, "cosine", 100, options).unwrap()
    }

    #[test]
    fn diskann_build_settings_are_in_wire_policy() {
        let wire = to_value(tuned_diskann_policy()).unwrap();
        let index = &wire["indexingPolicy"]["vectorIndexes"][0];
        assert_eq!(index["indexingSearchListSize"], 150);
        assert_eq!(index["quantizationByteSize"], 32);
    }

    #[test]
    fn diskann_build_readback_requires_all_explicit_settings() {
        let expected = tuned_diskann_policy();
        let wire = to_value(&expected).unwrap();
        let actual = serde_json::from_value(wire.clone()).unwrap();
        assert!(verify_policy(&actual, &expected).is_ok());
        for key in ["indexingSearchListSize", "quantizationByteSize"] {
            let mut changed = wire.clone();
            let index = &mut changed["indexingPolicy"]["vectorIndexes"][0];
            index[key] = json!(100);
            let actual = serde_json::from_value(changed.clone()).unwrap();
            assert!(
                verify_policy(&actual, &expected).is_err(),
                "mismatched {key}"
            );
            changed["indexingPolicy"]["vectorIndexes"][0]
                .as_object_mut()
                .unwrap()
                .remove(key);
            let actual = serde_json::from_value(changed).unwrap();
            assert!(verify_policy(&actual, &expected).is_err(), "missing {key}");
        }
    }

    #[test]
    fn diskann_build_settings_validate_types_and_bounds() {
        let kind = VectorIndexType::DiskANN;
        for (indexing_search_list, min, max) in [(true, 25, 500), (false, 1, 512)] {
            let parse = |value: &Value| {
                if indexing_search_list {
                    DiskAnnBuildOptions::from_config(&kind, Some(value), None)
                } else {
                    DiskAnnBuildOptions::from_config(&kind, None, Some(value))
                }
            };
            assert!(parse(&json!(min)).is_ok());
            assert!(parse(&json!(max)).is_ok());
            for invalid in [json!(min - 1), json!(max + 1), json!(64.0), json!("64")] {
                assert!(
                    parse(&invalid).is_err(),
                    "accepted invalid setting {invalid}"
                );
            }
        }
        assert!(
            DiskAnnBuildOptions::from_config(&VectorIndexType::Flat, Some(&json!(100)), None)
                .is_err()
        );
        assert!(
            DiskAnnBuildOptions::from_config(&VectorIndexType::Flat, None, Some(&json!(32)))
                .is_err()
        );
        let options = DiskAnnBuildOptions::from_config(&kind, None, Some(&json!(32))).unwrap();
        assert!(container_policy("bench".into(), kind.clone(), "cosine", 32, options).is_ok());
        assert!(container_policy("bench".into(), kind, "cosine", 31, options).is_err());
    }

    #[test]
    fn vector_queries_request_indexed_top_k_and_bind_the_vector() {
        let query = to_value(knn_query(&[0.5, -0.25], 10, None, None, false).unwrap()).unwrap();
        assert_eq!(
            query["query"],
            "SELECT TOP 10 c.id FROM c ORDER BY VectorDistance(c.embedding, @vector, false)"
        );
        assert_eq!(
            query["parameters"],
            json!([{"name": "@vector", "value": [0.5, -0.25]}])
        );
        assert!(knn_query(&[0.5], 0, None, None, false).is_err());
        assert!(knn_query(&[0.5], 1001, None, None, false).is_err());
        let tuned =
            to_value(knn_query(&[0.5], 10, 5.try_into().ok(), 20.try_into().ok(), false).unwrap())
                .unwrap();
        assert_eq!(
            tuned["query"],
            r#"SELECT TOP 10 c.id FROM c ORDER BY VectorDistance(c.embedding, @vector, false, {"searchListSizeMultiplier": 5, "quantizedVectorListMultiplier": 20})"#
        );
        let scored = to_value(knn_query(&[0.5], 10, None, None, true).unwrap()).unwrap();
        assert_eq!(
            scored["query"],
            "SELECT TOP 10 c.id, VectorDistance(c.embedding, @vector, false) AS distance FROM c ORDER BY VectorDistance(c.embedding, @vector, false)"
        );
    }

    #[test]
    fn a_bound_query_text_is_the_query_knn_query_builds() {
        let multiplier = 5.try_into().ok();
        for project_distance in [false, true] {
            let text = KnnQueryText::new(10, multiplier, None, project_distance).unwrap();
            for vector in [[0.5_f32, -0.25], [1.0, 2.0]] {
                assert_eq!(
                    to_value(text.bind(&vector).unwrap()).unwrap(),
                    to_value(knn_query(&vector, 10, multiplier, None, project_distance).unwrap())
                        .unwrap()
                );
            }
        }
        assert!(KnnQueryText::new(0, None, None, false).is_err());
        assert!(KnnQueryText::new(1001, None, None, false).is_err());
        assert!(validate_query_vector(&[0.5, 0.25], 2).is_ok());
        assert!(validate_query_vector(&[0.5], 2).is_err());
        assert!(validate_query_vector(&[0.5, f32::INFINITY], 2).is_err());
        assert!(knn_query(&[f32::NAN], 10, None, None, false).is_err());
    }

    #[test]
    fn partition_concurrency_uses_the_realized_feed_range_count() {
        assert_eq!(PartitionConcurrency::from_config(None).unwrap(), None);
        let auto = PartitionConcurrency::from_config(Some(&json!("auto")))
            .unwrap()
            .unwrap();
        assert_eq!(auto.resolve(5), 5);
        assert_eq!(auto.resolve(12), 12);
        let limited = PartitionConcurrency::from_config(Some(&json!(9)))
            .unwrap()
            .unwrap();
        assert_eq!(limited.resolve(5), 5);
        assert_eq!(limited.resolve(12), 9);
        for invalid in [json!(0), json!(-1), json!(1.5), json!("9"), json!(null)] {
            assert!(PartitionConcurrency::from_config(Some(&invalid)).is_err());
        }
    }

    #[test]
    fn partition_merge_selects_global_top_k_and_rejects_incomplete_hits() {
        let hit = |id: &str, distance| QueryHit {
            id: id.into(),
            distance,
        };
        let candidates = || {
            vec![
                hit("0", Some(0.9)),
                hit("2", Some(0.3)),
                hit("1", Some(0.3)),
                hit("3", Some(0.1)),
            ]
        };
        for (metric, expected) in [
            (VectorDistanceFunction::Euclidean, ["3", "1"]),
            (VectorDistanceFunction::Cosine, ["0", "1"]),
            (VectorDistanceFunction::DotProduct, ["0", "1"]),
        ] {
            let merged = merge_partition_hits(candidates(), 2, &metric).unwrap();
            assert_eq!(
                merged.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
                expected
            );
        }
        let metric = VectorDistanceFunction::Cosine;
        assert!(merge_partition_hits(vec![hit("0", None)], 1, &metric).is_err());
        assert!(
            merge_partition_hits(vec![hit("0", Some(0.1)), hit("0", Some(0.2))], 1, &metric)
                .is_err()
        );
    }

    fn hits(ids: &[&str]) -> Vec<QueryHit> {
        ids.iter()
            .map(|id| QueryHit {
                id: (*id).into(),
                distance: None,
            })
            .collect()
    }

    #[test]
    fn parity_requires_exact_agreement_when_stock_repeats_exactly() {
        let mut exact = ParityTally::default();
        exact
            .observe(&hits(&["1", "2"]), &hits(&["2", "1"]), &hits(&["1", "2"]))
            .unwrap();
        assert_eq!(
            exact,
            ParityTally {
                samples: 1,
                stock_repeat_missing_ids: 0,
                cross_mode_missing_ids: 0
            }
        );
        assert!(exact.verdict().is_ok());

        let mut wrong = ParityTally::default();
        wrong
            .observe(&hits(&["1", "2"]), &hits(&["1", "2"]), &hits(&["1", "3"]))
            .unwrap();
        assert_eq!(wrong.cross_mode_missing_ids, 2);
        assert!(wrong.verdict().is_err());
        assert!(ParityTally::default().verdict().is_err());
    }

    #[test]
    fn parity_allows_fanout_the_variation_stock_shows_against_itself() {
        // Stock misses one ID against itself. Fan-out may miss up to three.
        let mut within = ParityTally::default();
        within
            .observe(
                &hits(&["1", "2", "3"]),
                &hits(&["1", "2", "4"]),
                &hits(&["1", "2", "5"]),
            )
            .unwrap();
        assert_eq!(within.stock_repeat_missing_ids, 1);
        assert_eq!(within.cross_mode_missing_ids, 2);
        assert!(within.verdict().is_ok());

        let mut beyond = ParityTally::default();
        beyond
            .observe(
                &hits(&["1", "2", "3"]),
                &hits(&["1", "2", "4"]),
                &hits(&["1", "8", "9"]),
            )
            .unwrap();
        assert_eq!(beyond.cross_mode_missing_ids, 4);
        assert!(beyond.verdict().is_err());

        // The tally is cumulative: a later exact sample does not reset it.
        beyond
            .observe(&hits(&["1"]), &hits(&["1"]), &hits(&["1"]))
            .unwrap();
        assert_eq!(beyond.samples, 2);
        assert!(beyond.verdict().is_err());
        assert!(ParityTally::default()
            .observe(&hits(&["1", "2"]), &hits(&["1", "2"]), &hits(&["1"]))
            .is_err());
    }

    #[test]
    fn index_probe_needs_a_query_clearly_cheaper_than_a_forced_scan() {
        let probe = |scan_ru, index_ru| IndexProbe {
            feed_range: 0,
            scan_ru,
            index_ru,
        };
        assert!(probe(1000.0, 499.0).index_in_use());
        assert!(!probe(1000.0, 500.0).index_in_use());
        assert!(!probe(1000.0, 1000.0).index_in_use());
        assert!(!probe(0.0, 0.0).index_in_use());
        assert!(!probe(1000.0, 0.0).index_in_use());
        let scan = to_value(scan_query(&[0.5], 10).unwrap()).unwrap();
        assert_eq!(
            scan["query"],
            "SELECT TOP 10 c.id FROM c ORDER BY VectorDistance(c.embedding, @vector, true)"
        );
        assert!(scan_query(&[f32::NAN], 10).is_err());
        assert!(scan_query(&[0.5], 0).is_err());
    }

    #[test]
    fn index_probe_deadline_defaults_and_can_be_skipped() {
        assert_eq!(
            index_probe_deadline(None).unwrap(),
            Some(Duration::from_secs(1800))
        );
        assert_eq!(
            index_probe_deadline(Some("90")).unwrap(),
            Some(Duration::from_secs(90))
        );
        assert_eq!(index_probe_deadline(Some("0")).unwrap(), None);
        for invalid in ["", "-1", "1.5", "soon"] {
            assert!(index_probe_deadline(Some(invalid)).is_err(), "{invalid}");
        }
    }

    #[test]
    fn config_values_are_validated_before_any_request() {
        let keys = [String::from("parallel"), String::from("batch_size")];
        assert!(reject_unknown_keys(&keys[..1], &["parallel"], "no").is_ok());
        assert_eq!(
            reject_unknown_keys(&keys, &["parallel"], "no"),
            Err("no".into())
        );

        assert_eq!(positive_u32(&json!(5), "k").unwrap().get(), 5);
        for invalid in [
            json!(0),
            json!(-1),
            json!(1.5),
            json!("5"),
            json!(1u64 << 32),
        ] {
            assert!(positive_u32(&invalid, "k").is_err(), "{invalid}");
        }

        assert_eq!(manual_throughput(None, "400").unwrap(), Some(400));
        assert_eq!(manual_throughput(None, "serverless").unwrap(), None);
        assert_eq!(
            manual_throughput(Some(&json!(1000)), "serverless").unwrap(),
            Some(1000)
        );
        assert!(manual_throughput(None, "399").is_err());
        assert!(manual_throughput(None, "auto").is_err());
        assert!(manual_throughput(Some(&json!(399)), "400").is_err());
        assert!(manual_throughput(Some(&json!("1000")), "400").is_err());

        assert_eq!(upload_parallel(None).unwrap(), 8);
        assert_eq!(upload_parallel(Some(&json!({}))).unwrap(), 8);
        assert_eq!(upload_parallel(Some(&json!({"parallel": 16}))).unwrap(), 16);
        for invalid in [
            json!({"parallel": 0}),
            json!({"parallel": "4"}),
            json!({"batch_size": 10}),
            json!([4]),
        ] {
            assert!(upload_parallel(Some(&invalid)).is_err(), "{invalid}");
        }
    }

    #[test]
    fn diskann_rejects_the_documented_full_scan_case() {
        assert!(validate_corpus_size(&VectorIndexType::DiskANN, 999).is_err());
        assert!(validate_corpus_size(&VectorIndexType::DiskANN, 1000).is_ok());
        assert!(validate_corpus_size(&VectorIndexType::Flat, 999).is_ok());
    }

    #[test]
    fn hit_count_rejects_concatenated_partition_top_k_lists() {
        assert!(validate_hit_count(2, 3).is_ok());
        assert!(validate_hit_count(3, 3).is_ok());
        assert!(validate_hit_count(6, 3).is_err());
    }

    #[test]
    fn search_diagnostics_sum_pages_retries_and_missing_statuses() {
        let mut query = SearchDiagnostics::default();
        query.observe_page(4.5, 5, [(429, 3), (200, 1)]);
        query.observe_page(2.0, 1, [(200, 1)]);
        let mut run = SearchDiagnostics::default();
        run.add_query(query);
        run.add_query(SearchDiagnostics::default());
        assert_eq!(
            to_value(run).unwrap(),
            json!({
                "request_charge_ru": 6.5,
                "request_attempts": 6,
                "http_429_attempts": 3,
                "attempts_without_status": 1,
                "queries_without_request_diagnostics": 1,
            })
        );
    }

    /// A clock that only `pause` advances, plus a record of every probe call.
    struct ProbeRun {
        clock: std::cell::Cell<Duration>,
        calls: std::cell::RefCell<Vec<(ProbeQuery, Vec<usize>)>>,
        pauses: std::cell::RefCell<Vec<Duration>>,
    }

    /// Each range's index starts serving once the clock reaches `ready_at`.
    fn run_probe(ready_at: &[u64], deadline: u64) -> (Result<Vec<IndexProbe>, String>, ProbeRun) {
        let run = ProbeRun {
            clock: Default::default(),
            calls: Default::default(),
            pauses: Default::default(),
        };
        let result = await_index_use(
            ready_at.len(),
            Duration::from_secs(deadline),
            |query, ranges| {
                run.calls.borrow_mut().push((query, ranges.to_vec()));
                Ok(ranges
                    .iter()
                    .map(|&range| {
                        let ready = run.clock.get() >= Duration::from_secs(ready_at[range]);
                        let served = query == ProbeQuery::IndexRequested && ready;
                        (range, if served { 6.0 } else { 300.0 })
                    })
                    .collect())
            },
            || run.clock.get(),
            |pause| {
                run.pauses.borrow_mut().push(pause);
                run.clock.set(run.clock.get() + pause);
            },
        );
        (result, run)
    }

    #[test]
    fn index_wait_scans_once_and_polls_only_the_ranges_still_waiting() {
        let (probes, run) = run_probe(&[0, 3, 0], 600);
        let probes = probes.unwrap();
        assert!(probes.iter().all(IndexProbe::index_in_use));
        assert_eq!(
            probes[1],
            IndexProbe {
                feed_range: 1,
                scan_ru: 300.0,
                index_ru: 6.0
            }
        );
        let calls = run.calls.borrow();
        assert_eq!(calls[0], (ProbeQuery::ForcedScan, vec![0, 1, 2]));
        assert_eq!(calls[1], (ProbeQuery::IndexRequested, vec![0, 1, 2]));
        // Ranges 0 and 2 passed on the first poll. Only range 1 is asked again.
        assert!(calls[2..]
            .iter()
            .all(|call| *call == (ProbeQuery::IndexRequested, vec![1])));
        assert_eq!(calls.len(), 4);
        assert_eq!(*run.pauses.borrow(), [Duration::from_secs(2); 2]);
    }

    #[test]
    fn index_wait_fails_at_the_deadline_and_names_the_charges() {
        let (result, run) = run_probe(&[0, u64::MAX], 100);
        let error = result.unwrap_err();
        assert!(
            error.contains("not serving 1 of 2 feed ranges after 100s"),
            "{error}"
        );
        assert!(
            error.contains("range 1: index query 300.00 RU, forced scan 300.00 RU"),
            "{error}"
        );
        // Range 0 is served, so it is not listed as waiting.
        assert!(!error.contains("range 0:"), "{error}");
        // It never sleeps past the deadline.
        assert_eq!(
            run.pauses.borrow().iter().sum::<Duration>(),
            Duration::from_secs(100)
        );
        // An answer for the wrong ranges is an error, not a panic or a pass.
        // The clock advances, so a missing check ends at the deadline.
        let clock = std::cell::Cell::new(Duration::ZERO);
        let wrong = await_index_use(
            2,
            Duration::from_secs(1),
            |_, _| Ok(vec![(0, 1.0)]),
            || clock.get(),
            |pause| clock.set(clock.get() + pause),
        );
        assert!(wrong.unwrap_err().contains("got answers for [0]"));
    }

    #[test]
    fn index_poll_interval_grows_with_the_wait() {
        let interval = |seconds| index_probe_interval(Duration::from_secs(seconds));
        assert_eq!(interval(0), Duration::from_secs(2));
        assert_eq!(interval(60), Duration::from_secs(6));
        assert_eq!(interval(3600), Duration::from_secs(30));
        let many: Vec<_> = (0..10)
            .map(|feed_range| IndexProbe {
                feed_range,
                scan_ru: 9.0,
                index_ru: 9.0,
            })
            .collect();
        let text = describe_waiting(&many);
        assert!(text.starts_with("range 0: index query 9.00 RU, forced scan 9.00 RU; range 1"));
        assert!(text.ends_with("; and 2 more"), "{text}");
    }

    #[test]
    fn throttled_writes_are_retried_until_the_budget_is_spent() {
        assert_eq!(upload_throttle_backoff(0), Some(Duration::from_millis(250)));
        assert_eq!(upload_throttle_backoff(3), Some(Duration::from_secs(2)));
        assert_eq!(upload_throttle_backoff(59), Some(Duration::from_secs(5)));
        assert_eq!(upload_throttle_backoff(60), None);

        let run = |outcomes: Vec<Result<u8, u16>>| {
            let retries = AtomicUsize::new(0);
            let slept = std::cell::RefCell::new(Vec::new());
            let mut outcomes = outcomes.into_iter();
            let result = futures::executor::block_on(retry_throttled(
                || {
                    let outcome = outcomes.next().unwrap_or(Err(429));
                    async move { outcome }
                },
                |status: &u16| *status == 429,
                |delay| {
                    slept.borrow_mut().push(delay);
                    async {}
                },
                &retries,
            ));
            (result, retries.into_inner(), slept.into_inner())
        };
        let (result, retries, slept) = run(vec![Err(429), Err(429), Ok(7)]);
        assert_eq!((result, retries), (Ok(7), 2));
        assert_eq!(
            slept,
            [Duration::from_millis(250), Duration::from_millis(500)]
        );
        // Any other failure is returned at once.
        assert_eq!(run(vec![Err(429), Err(503), Ok(7)]).0, Err(503));
        // Endless throttling ends with the 429 after the whole budget.
        let (result, retries, _) = run(vec![]);
        assert_eq!((result, retries), (Err(429), 60));
    }

    /// Pending once, so `buffer_unordered` has to interleave the futures.
    async fn yield_once() {
        let mut yielded = false;
        std::future::poll_fn(|cx| {
            if std::mem::replace(&mut yielded, true) {
                return std::task::Poll::Ready(());
            }
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        })
        .await
    }

    #[test]
    fn row_count_sums_ranges_and_rejects_a_split() {
        let count = |ranges_after: Vec<u64>| {
            let reads = std::cell::Cell::new(0);
            futures::executor::block_on(count_stable_ranges(
                || {
                    reads.set(reads.get() + 1);
                    let ranges = if reads.get() == 1 {
                        vec![10, 20, 30]
                    } else {
                        ranges_after.clone()
                    };
                    async move { Ok(ranges) }
                },
                2,
                |rows: u64| async move {
                    yield_once().await;
                    if rows == 0 {
                        Err("boom".to_string())
                    } else {
                        Ok(rows)
                    }
                },
            ))
        };
        assert_eq!(count(vec![10, 20, 30]), Ok(60));
        assert!(count(vec![10, 20, 15, 15])
            .unwrap_err()
            .contains("feed ranges changed"));
    }

    #[test]
    fn fan_out_merges_the_global_top_k_within_the_request_limit() {
        let scored = |id: &str, distance| QueryHit {
            id: id.into(),
            distance: Some(distance),
        };
        let in_flight = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let fan_out = |per_query, permits| {
            peak.store(0, Ordering::Relaxed);
            let permits = Semaphore::new(permits);
            futures::executor::block_on(fan_out_hits(
                &[0_usize, 1, 2, 3],
                per_query,
                &permits,
                2,
                &VectorDistanceFunction::Cosine,
                |range| {
                    let (in_flight, peak) = (&in_flight, &peak);
                    async move {
                        peak.fetch_max(
                            in_flight.fetch_add(1, Ordering::Relaxed) + 1,
                            Ordering::Relaxed,
                        );
                        yield_once().await;
                        in_flight.fetch_sub(1, Ordering::Relaxed);
                        if range == 3 && per_query == 1 {
                            return Err("range 3 failed".to_string());
                        }
                        let mut diagnostics = SearchDiagnostics::default();
                        diagnostics.observe_page(2.5, 1, [(200, 1)]);
                        let name = format!("r{range}");
                        Ok((
                            vec![
                                scored(&name, range as f64),
                                scored(&format!("{name}b"), -1.0),
                            ],
                            diagnostics,
                        ))
                    }
                },
            ))
        };
        let (hits, diagnostics) = fan_out(4, 4).unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            ["r3", "r2"]
        );
        assert_eq!(diagnostics.request_charge_ru, 10.0);
        assert_eq!(diagnostics.request_attempts, 4);
        assert_eq!(peak.load(Ordering::Relaxed), 4);
        // The shared permits cap in-flight requests below the per-query limit.
        fan_out(4, 2).unwrap();
        assert_eq!(peak.load(Ordering::Relaxed), 2);
        // So does the per-query limit, and a failed range fails the query.
        assert_eq!(fan_out(1, 4).err().unwrap(), "range 3 failed");
        assert_eq!(peak.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn parity_samples_spread_over_the_queries_and_must_pass_the_verdict() {
        assert_eq!(
            parity_sample_indices(100, 8),
            [0, 12, 24, 36, 48, 60, 72, 84]
        );
        assert_eq!(parity_sample_indices(3, 8), [0, 1, 2]);
        assert!(parity_sample_indices(0, 8).is_empty());

        let sample = |fan_out_ids: &'static [&'static str]| {
            futures::executor::block_on(sample_parity(vec![0, 5], 2, |index| async move {
                yield_once().await;
                if index == 5 && fan_out_ids.is_empty() {
                    return Err("timeout".to_string());
                }
                Ok([hits(&["a", "b"]), hits(&["a", "b"]), hits(fan_out_ids)])
            }))
        };
        let tally = sample(&["b", "a"]).unwrap();
        assert_eq!(
            tally,
            ParityTally {
                samples: 2,
                stock_repeat_missing_ids: 0,
                cross_mode_missing_ids: 0
            }
        );
        assert!(sample(&["a", "c"])
            .unwrap_err()
            .contains("fan-out missed 4 stock IDs"));
        assert_eq!(
            sample(&[]).unwrap_err(),
            "Cosmos DB parity check for query 5: timeout"
        );
    }
}
