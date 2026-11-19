//! Request construction and policy validation for Azure Cosmos DB for NoSQL.

use azure_data_cosmos::models::{
    ContainerProperties, IndexingMode, IndexingPolicy, VectorDataType, VectorDistanceFunction,
    VectorEmbedding, VectorEmbeddingPolicy, VectorIndex, VectorIndexType,
};
use azure_data_cosmos::Query;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::time::Duration;

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
}
