//! Comparable metadata-only query contract for Redis and OpenSearch.
//! Return up to `limit` IDs in unspecified order, plus an exact total count.
//! Vector bytes may be stored, but must never be indexed or searched.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub mod domain;

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
    })
}

/// Disable indexing of the stored vector array, including dynamic mapping.
pub fn disable_vector_index(properties: &mut Value, settings: &mut Value) {
    properties["vector"] = json!({"type": "float", "index": false, "doc_values": false});
    settings["knn"] = json!(false);
}

/// Pick only explicit predicates, preserving their order and cycling only when
/// a positive query count is requested. Vector ground truth is irrelevant.
pub fn query_indices(
    filtered: impl IntoIterator<Item = bool>,
    requested: i64,
) -> Result<Vec<usize>, String> {
    let selected: Vec<_> = filtered
        .into_iter()
        .enumerate()
        .filter_map(|(i, yes)| yes.then_some(i))
        .collect();
    if selected.is_empty() {
        return Err("No queries with filter conditions for filter-only search".into());
    }
    let count = if requested > 0 {
        requested as usize
    } else {
        selected.len()
    };
    Ok((0..count).map(|i| selected[i % selected.len()]).collect())
}

/// Validate realized schema before timing, including when reusing an index.
pub fn validate_opensearch_schema(
    properties: &Value,
    index: &Value,
    schema: Option<&Value>,
) -> Result<(), String> {
    let vector = &properties["vector"];
    if !matches!(&index["knn"], Value::Bool(false)) && index["knn"] != "false" {
        return Err("filter-only requires index.knn=false".into());
    }
    if vector["type"] != "float" || vector["index"] != false || vector["doc_values"] != false {
        return Err("filter-only requires an unindexed vector field without doc values".into());
    }
    if let Some(fields) = schema.and_then(Value::as_object) {
        for (field, kind) in fields {
            if matches!(kind.as_str(), Some("int" | "float" | "keyword"))
                && properties[field]["index"] == false
            {
                return Err(format!("filter-only requires indexed field {field}"));
            }
            if kind == "int" && properties[field]["type"] != "long" {
                return Err(format!("filter-only requires long mapping for {field}"));
            }
            if kind == "float" && properties[field]["type"] != "double" {
                return Err(format!("filter-only requires double mapping for {field}"));
            }
            if kind == "keyword"
                && (properties[field]["type"] != "keyword"
                    || !properties[field]["normalizer"].is_null())
            {
                return Err(format!(
                    "filter-only requires case-sensitive keyword mapping for {field}"
                ));
            }
        }
    }
    Ok(())
}

/// FT.INFO can arrive as RESP2 pairs or a RESP3 map; attributes also contain
/// standalone flags, so do not deserialize them as key/value maps.
pub fn validate_redis_schema(info: &Value, schema: Option<&Value>) -> Result<(), String> {
    fn field<'a>(v: &'a Value, name: &str) -> Option<&'a Value> {
        if let Some(map) = v.as_object() {
            return map.get(name);
        }
        v.as_array()?
            .windows(2)
            .find_map(|p| (p[0] == name).then_some(&p[1]))
    }
    let attributes = field(info, "attributes")
        .and_then(Value::as_array)
        .ok_or("filter-only missing FT.INFO attributes")?;
    for attr in attributes {
        let kind = field(attr, "type")
            .and_then(Value::as_str)
            .ok_or("filter-only missing FT.INFO attribute type")?;
        if kind.eq_ignore_ascii_case("VECTOR") {
            return Err("filter-only cannot reuse an index containing VECTOR".into());
        }
    }
    if let Some(fields) = schema.and_then(Value::as_object) {
        for (name, kind) in fields {
            if !matches!(kind.as_str(), Some("int" | "float" | "keyword")) {
                continue;
            }
            let attr = attributes
                .iter()
                .find(|a| field(a, "attribute").and_then(Value::as_str) == Some(name))
                .ok_or_else(|| format!("filter-only missing TAG field {name}"))?;
            if kind != "keyword" {
                if field(attr, "type").and_then(Value::as_str) != Some("NUMERIC") {
                    return Err(format!("filter-only requires NUMERIC field {name}"));
                }
                continue;
            }
            if field(attr, "SEPARATOR").and_then(Value::as_str) != Some(";") {
                return Err(format!(
                    "filter-only requires semicolon separator for {name}"
                ));
            }
            let case_sensitive = attr
                .as_array()
                .is_some_and(|a| a.iter().any(|v| v == "CASESENSITIVE"))
                || field(attr, "flags")
                    .and_then(Value::as_array)
                    .is_some_and(|a| a.iter().any(|v| v == "CASESENSITIVE"));
            if field(attr, "type").and_then(Value::as_str) != Some("TAG") || !case_sensitive {
                return Err(format!("filter-only requires CASESENSITIVE TAG for {name}"));
            }
        }
    }
    Ok(())
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

#[cfg(test)]
#[path = "filter_only/review_tests.rs"]
mod review_tests;
