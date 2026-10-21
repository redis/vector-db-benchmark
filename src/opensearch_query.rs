//! OpenSearch query-level effort, independent of the returned K.

/// Reject invalid requested effort before indexing or sending any query.
pub fn validate_ef(ef: Option<i64>) -> Result<(), String> {
    if ef.is_some_and(|v| v <= 0 || v > i32::MAX as i64) {
        return Err("OpenSearch query ef must be a positive 32-bit integer".into());
    }
    Ok(())
}

/// Query-level Lucene method parameters require OpenSearch 2.16 or newer.
pub fn validate_server_version(version: &str) -> Result<(), String> {
    let mut pieces = version.split('.');
    let major = pieces.next().and_then(|v| v.parse::<u64>().ok());
    let minor = pieces.next().and_then(|v| v.parse::<u64>().ok());
    match (major, minor) {
        (Some(major), Some(minor)) if major > 2 || (major == 2 && minor >= 16) => Ok(()),
        _ => Err(format!(
            "OpenSearch query-level ef requires OpenSearch 2.16+; server reports {version:?}"
        )),
    }
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
pub fn build_knn_body(
    query_vector: &[f32],
    top: usize,
    filter: Option<&serde_json::Value>,
    ef: Option<i64>,
) -> Result<serde_json::Value, String> {
    validate_ef(ef)?;
    let mut query = serde_json::json!({
        "knn": {
            "vector": {
                "vector": query_vector,
                "k": top,
            }
        }
    });

    if let Some(ef) = ef {
        query["knn"]["vector"]["method_parameters"] = serde_json::json!({"ef_search": ef});
    }
    if let Some(f) = filter {
        query["knn"]["vector"]["filter"] = f.clone();
    }

    // Response trimming: the benchmark only needs each hit's id, so skip loading
    // `_source` and stored fields and return `_id` via a doc-value field. This
    // trims the response payload for a fairer QPS/latency measurement.
    Ok(serde_json::json!({
        "query": query,
        "size": top,
        "_source": false,
        "docvalue_fields": ["_id"],
        "stored_fields": "_none_",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn query_effort_requires_a_supported_parseable_server_version() {
        for version in ["2.16.0", "2.19.2", "3.7.0", "3.0.0-SNAPSHOT"] {
            assert!(validate_server_version(version).is_ok(), "{version}");
        }
        for version in ["1.3.0", "2.15.9", "2.7.0", "", "3", "unknown", "3.x.0"] {
            assert!(validate_server_version(version).is_err(), "{version}");
        }
    }

    #[test]
    fn effort_is_explicit_and_does_not_change_returned_k() {
        let filter = json!({"range": {"x": {"gte": 1, "lte": 5}}});
        for ef in [None, Some(128), Some(2048)] {
            let body = build_knn_body(&[0.2, 0.4], 25, Some(&filter), ef).unwrap();
            let vector = &body["query"]["knn"]["vector"];
            assert_eq!(body["size"], 25);
            assert_eq!(vector["k"], 25);
            assert_eq!(vector["filter"], filter);
            match ef {
                Some(ef) => assert_eq!(vector["method_parameters"]["ef_search"], ef),
                None => assert!(vector.get("method_parameters").is_none()),
            }
        }
    }

    #[test]
    fn invalid_effort_cannot_silently_become_a_default_search() {
        for ef in [-1, 0, i32::MAX as i64 + 1] {
            assert!(build_knn_body(&[0.1], 25, None, Some(ef)).is_err());
        }
    }
}
