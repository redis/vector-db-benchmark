//! Comparable metadata-only query contract for Redis and OpenSearch.
//! Return up to `limit` IDs in unspecified order, plus an exact total count.
//! Vector bytes may be stored, but must never be indexed or searched.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const DEFAULT_LIMIT: usize = 10;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Stats {
    pub result_limit: usize,
    pub matched_documents_sum: u64,
    pub returned_documents_sum: u64,
}

pub fn opensearch_body(filter: &Value, limit: usize) -> Value {
    json!({
        "query": {"bool": {"filter": [filter]}},
        "size": limit,
        "track_total_hits": true,
        "sort": ["_doc"],
        "_source": false,
        "stored_fields": "_none_",
        "docvalue_fields": ["_id"],
    })
}

/// Disable indexing of the stored vector array, including dynamic mapping.
pub fn disable_vector_index(properties: &mut Value, settings: &mut Value) {
    properties["vector"] = json!({"type": "float", "index": false, "doc_values": false});
    settings["knn"] = json!(false);
}

pub fn validate_page(total: u64, returned: usize, limit: usize) -> Result<(), String> {
    if returned as u64 != total.min(limit as u64) {
        return Err(format!(
            "filter-only incomplete page: total={total}, returned={returned}, limit={limit}"
        ));
    }
    Ok(())
}

pub fn opensearch_counts(body: &Value, limit: usize) -> Result<(u64, usize), String> {
    if body["timed_out"] != false || body["_shards"]["failed"] != 0 {
        return Err("filter-only timed out, failed shards, or missing completion metadata".into());
    }
    if body["terminated_early"] == true || body["hits"]["total"]["relation"] != "eq" {
        return Err("filter-only response does not have an exact total".into());
    }
    let total = body["hits"]["total"]["value"]
        .as_u64()
        .ok_or("missing exact match count")?;
    let hits = body["hits"]["hits"].as_array().ok_or("missing hits")?;
    let mut ids = std::collections::HashSet::new();
    for hit in hits {
        let id = hit["_id"]
            .as_str()
            .or_else(|| hit["fields"]["_id"][0].as_str())
            .ok_or("missing hit ID")?;
        if !ids.insert(id) {
            return Err("duplicate filter-only hit ID".into());
        }
    }
    validate_page(total, hits.len(), limit)?;
    Ok((total, hits.len()))
}

/// RESP3 exposes timeout warnings that RESP2 silently omits. Reject all
/// warnings so partial replies cannot be published as exact match counts.
pub fn redis_counts(value: &redis::Value, limit: usize) -> Result<(u64, usize), String> {
    use redis::Value;
    let Value::Map(fields) = value else {
        return Err("filter-only requires a RESP3 FT.SEARCH map".into());
    };
    fn text(value: &Value) -> Option<&str> {
        match value {
            Value::BulkString(bytes) => std::str::from_utf8(bytes).ok(),
            Value::SimpleString(s) => Some(s),
            _ => None,
        }
    }
    fn field<'a>(fields: &'a [(Value, Value)], name: &str) -> Option<&'a Value> {
        fields
            .iter()
            .find_map(|(k, v)| (text(k) == Some(name)).then_some(v))
    }
    match field(fields, "warning") {
        Some(Value::Array(warnings)) if warnings.is_empty() => {}
        _ => return Err("filter-only Redis warning or missing completion metadata".into()),
    }
    let total = match field(fields, "total_results") {
        Some(Value::Int(n)) if *n >= 0 => *n as u64,
        _ => return Err("missing Redis exact match count".into()),
    };
    let Some(Value::Array(hits)) = field(fields, "results") else {
        return Err("missing Redis results".into());
    };
    let mut ids = std::collections::HashSet::new();
    for hit in hits {
        let Value::Map(fields) = hit else {
            return Err("invalid Redis result".into());
        };
        let id = field(fields, "id")
            .and_then(text)
            .ok_or("missing Redis hit ID")?;
        if !ids.insert(id) {
            return Err("duplicate Redis hit ID".into());
        }
    }
    validate_page(total, hits.len(), limit)?;
    Ok((total, hits.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_is_filter_only_ids_and_exact_count() {
        let f = json!({"range":{"n":{"gt":2,"lte":5}}});
        let b = opensearch_body(&f, 10);
        assert_eq!(b["query"], json!({"bool":{"filter":[f]}}));
        assert_eq!(b["track_total_hits"], true);
        assert_eq!(b["size"], 10);
        assert_eq!(b["_source"], false);
        assert_eq!(b["sort"], json!(["_doc"]));
    }
    #[test]
    fn redis_replies_require_exact_pages_unique_ids_and_no_warnings() {
        use redis::Value::{Array, BulkString, Int, Map};
        let text = |s: &str| BulkString(s.as_bytes().to_vec());
        let reply = |total, ids: &[&str], warnings: Vec<redis::Value>| {
            Map(vec![
                (text("total_results"), Int(total)),
                (text("warning"), Array(warnings)),
                (
                    text("results"),
                    Array(
                        ids.iter()
                            .map(|id| Map(vec![(text("id"), text(id))]))
                            .collect(),
                    ),
                ),
            ])
        };
        assert_eq!(
            redis_counts(&reply(12, &["a", "b"], vec![]), 2).unwrap(),
            (12, 2)
        );
        assert_eq!(redis_counts(&reply(0, &[], vec![]), 2).unwrap(), (0, 0));
        for bad in [
            Array(vec![Int(0)]),
            Map(vec![]),
            reply(-1, &[], vec![]),
            reply(2, &["a"], vec![]),
            reply(2, &["a", "a"], vec![]),
            reply(2, &["a", "b"], vec![text("Timeout limit was reached")]),
        ] {
            assert!(redis_counts(&bad, 2).is_err());
        }
    }
    #[test]
    fn vector_is_stored_but_never_indexed() {
        let mut p = json!({"vector":{"type":"knn_vector"},"tag":{"type":"keyword"}});
        let mut s = json!({"knn":true,"number_of_shards":1});
        disable_vector_index(&mut p, &mut s);
        assert_eq!(
            p["vector"],
            json!({"type":"float","index":false,"doc_values":false})
        );
        assert_eq!(p["tag"]["type"], "keyword");
        assert_eq!(s["knn"], false);
    }
    #[test]
    fn rejects_partial_approximate_or_malformed_responses() {
        let good = json!({"timed_out":false,"_shards":{"failed":0},"hits":{"total":{"value":2,"relation":"eq"},"hits":[{"_id":"a"},{"_id":"b"}]}});
        assert_eq!(opensearch_counts(&good, 10).unwrap(), (2, 2));
        for bad in [
            "timeout",
            "shard",
            "approximate",
            "page",
            "duplicate",
            "id",
            "early",
        ] {
            let mut b = good.clone();
            match bad {
                "timeout" => b["timed_out"] = json!(true),
                "shard" => b["_shards"]["failed"] = json!(1),
                "approximate" => b["hits"]["total"]["relation"] = json!("gte"),
                "page" => b["hits"]["total"]["value"] = json!(3),
                "duplicate" => b["hits"]["hits"][1]["_id"] = json!("a"),
                "id" => b["hits"]["hits"][1] = json!({}),
                _ => b["terminated_early"] = json!(true),
            }
            assert!(opensearch_counts(&b, 10).is_err(), "accepted {bad}");
        }
        let empty = json!({"timed_out":false,"_shards":{"failed":0},"hits":{"total":{"value":0,"relation":"eq"},"hits":[]}});
        assert_eq!(opensearch_counts(&empty, 10).unwrap(), (0, 0));
    }
}
