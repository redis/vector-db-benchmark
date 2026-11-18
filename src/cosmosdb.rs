//! Request construction and policy validation for Azure Cosmos DB for NoSQL.

use azure_data_cosmos::models::{
    ContainerProperties, IndexingMode, IndexingPolicy, VectorDataType, VectorDistanceFunction,
    VectorEmbedding, VectorEmbeddingPolicy, VectorIndex, VectorIndexType,
};
use azure_data_cosmos::Query;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::Serialize;
use sha2::{Digest, Sha256};

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

/// Use the dataset metric and dimension in the server-side policy.
pub fn container_policy(
    name: String,
    index_type: VectorIndexType,
    distance: &str,
    dimensions: i64,
) -> Result<ContainerProperties, String> {
    let distance = match distance.to_ascii_lowercase().as_str() {
        "cosine" | "angular" => VectorDistanceFunction::Cosine,
        "l2" | "euclidean" => VectorDistanceFunction::Euclidean,
        "dot" | "ip" => VectorDistanceFunction::DotProduct,
        _ => {
            return Err(format!(
                "Cosmos DB does not support dataset distance '{distance}'"
            ))
        }
    };
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

    let mut indexing = IndexingPolicy::default()
        .with_indexing_mode(IndexingMode::Consistent)
        .with_included_path("/*")
        .with_excluded_path(VECTOR_EXCLUSION)
        .with_vector_index(VectorIndex::new(VECTOR_PATH, index_type));
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
                    index.path == expected.path && index.index_type == expected.index_type
                }))
        })
    {
        return Err(format!(
            "Cosmos DB container '{}' has an incompatible partition key, embedding policy or vector index",
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
        self.queries_without_request_diagnostics += usize::from(query.request_attempts == 0);
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

/// Build a server-side TOP K query. The service uses the policy's distance
/// function. TOP is a validated integer literal and the vector is bound.
pub fn knn_query(
    vector: &[f32],
    top: usize,
    search_list_size_multiplier: Option<std::num::NonZeroU32>,
    quantized_vector_list_multiplier: Option<std::num::NonZeroU32>,
) -> Result<Query, String> {
    validate_top(top)?;
    if vector.iter().any(|v| !v.is_finite()) {
        return Err("Cosmos DB query vector contains a non-finite value".into());
    }
    let options = match (search_list_size_multiplier, quantized_vector_list_multiplier) {
        (None, None) => String::new(),
        (Some(search), None) => format!(", {{\"searchListSizeMultiplier\": {search}}}"),
        (None, Some(quantized)) => format!(", {{\"quantizedVectorListMultiplier\": {quantized}}}"),
        (Some(search), Some(quantized)) => format!(
            ", {{\"searchListSizeMultiplier\": {search}, \"quantizedVectorListMultiplier\": {quantized}}}"
        ),
    };
    Query::from(format!(
        "SELECT TOP {top} c.id FROM c ORDER BY VectorDistance(c.embedding, @vector, false{options})"
    ))
    .with_parameter("@vector", vector)
    .map_err(|e| format!("Could not bind Cosmos DB query vector: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, to_value};

    #[test]
    fn distance_and_diskann_policy_are_in_the_wire_request() {
        let p = container_policy("bench".into(), VectorIndexType::DiskANN, "dot", 1024).unwrap();
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
        let other = container_policy("bench".into(), VectorIndexType::DiskANN, "l2", 1024).unwrap();
        assert!(verify_policy(&p, &other).is_err());
        assert!(verify_policy(&p, &p).is_ok());
        assert!(container_policy("bench".into(), VectorIndexType::Flat, "cosine", 506).is_err());
    }

    #[test]
    fn vector_queries_request_indexed_top_k_and_bind_the_vector() {
        let query = to_value(knn_query(&[0.5, -0.25], 10, None, None).unwrap()).unwrap();
        assert_eq!(
            query["query"],
            "SELECT TOP 10 c.id FROM c ORDER BY VectorDistance(c.embedding, @vector, false)"
        );
        assert_eq!(
            query["parameters"],
            json!([{"name": "@vector", "value": [0.5, -0.25]}])
        );
        assert!(knn_query(&[0.5], 0, None, None).is_err());
        assert!(knn_query(&[0.5], 1001, None, None).is_err());
        let tuned = to_value(knn_query(&[0.5], 10, 5.try_into().ok(), 20.try_into().ok()).unwrap())
            .unwrap();
        assert_eq!(
            tuned["query"],
            r#"SELECT TOP 10 c.id FROM c ORDER BY VectorDistance(c.embedding, @vector, false, {"searchListSizeMultiplier": 5, "quantizedVectorListMultiplier": 20})"#
        );
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
