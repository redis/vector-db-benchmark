//! Comparable metadata-only query contract for Redis and OpenSearch.
//! Return up to `limit` IDs in unspecified order, plus an exact total count.
//! Vector bytes may be stored, but must never be indexed or searched.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub mod domain;
pub mod query_set;

pub const DEFAULT_LIMIT: usize = 10;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Stats {
    pub result_limit: usize,
    pub matched_documents_sum: u64,
    pub returned_documents_sum: u64,
    /// Successful measured queries requiring retries; priming is excluded.
    #[serde(default)]
    pub retried_queries: u64,
    /// Answers checked by the per-query count gate against a prepared query
    /// set's `expected_count` (#340); absent when the set has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_query_counts_checked: Option<u64>,
    /// The layout of the index the queries ran on (#344):
    /// `crate::workload::IndexLayout::as_str`. `"full"` means the index also
    /// holds an indexed vector field for the engine config's other workloads;
    /// the filter-only requests themselves were each checked, as sent, to carry
    /// no vector clause before the timed run ([`check_opensearch_body`] on the
    /// request body, [`check_redis_command`] on the FT.SEARCH arguments). Absent on engines that do not report it (the
    /// legacy Valkey/MongoDB harnesses never publish these stats at all).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_layout: Option<String>,
}

/// Refuse a filter-only OpenSearch request body that carries a vector clause
/// anywhere (#344): on a `full` index a stray `knn` clause would be served and
/// measured as a metadata query. Checked for every distinct request before the
/// timed run, on every layout.
pub fn check_opensearch_body(body: &Value) -> Result<(), String> {
    // Query types that score by vector. A schema field with one of these names
    // would be a false positive, which fails loudly rather than silently.
    const VECTOR_CLAUSES: [&str; 3] = ["knn", "neural", "script_score"];
    fn find(v: &Value) -> Option<&str> {
        match v {
            Value::Object(map) => map.iter().find_map(|(k, child)| {
                VECTOR_CLAUSES
                    .iter()
                    .find(|c| k.eq_ignore_ascii_case(c))
                    .copied()
                    .or_else(|| find(child))
            }),
            Value::Array(items) => items.iter().find_map(find),
            _ => None,
        }
    }
    match find(body) {
        Some(clause) => Err(format!(
            "filter-only request carries a vector clause ({clause:?}); filter-only must issue \
             metadata predicates only"
        )),
        None => Ok(()),
    }
}

/// Refuse a filter-only FT.SEARCH command, as sent, that is anything but the
/// metadata shape (#344): `FT.SEARCH <index> <query>` with the query passing
/// [`check_redis_query`], followed only by `LIMIT o n`, `NOCONTENT`,
/// `DIALECT d`, `TIMEOUT t` and `PARAMS n <name value>…`. Checking the
/// finished arguments (not the filter expression) means a vector clause, a
/// vector `PARAMS` blob's `=>[KNN …]` reference or any other option the
/// command builder adds is caught too — on the `full` layout the index would
/// serve it.
pub fn check_redis_command(args: &[&[u8]]) -> Result<(), String> {
    let text = |a: &[u8]| String::from_utf8_lossy(a).into_owned();
    let refuse = |why: String| {
        Err(format!(
            "filter-only FT.SEARCH is not a metadata-only command ({why}); filter-only must \
             issue metadata predicates only"
        ))
    };
    if args.len() < 3 || !args[0].eq_ignore_ascii_case(b"FT.SEARCH") {
        return refuse("expected FT.SEARCH <index> <query>".into());
    }
    check_redis_query(&text(args[2]))?;
    let mut i = 3;
    while i < args.len() {
        let option = text(args[i]).to_ascii_uppercase();
        let operands = match option.as_str() {
            "NOCONTENT" => 0,
            "DIALECT" | "TIMEOUT" => 1,
            "LIMIT" => 2,
            "PARAMS" => {
                let n = args
                    .get(i + 1)
                    .and_then(|a| text(a).parse::<usize>().ok())
                    .ok_or_else(|| "PARAMS without a count".to_string());
                match n {
                    Ok(n) if n % 2 == 0 => n + 1,
                    Ok(n) => return refuse(format!("odd PARAMS count {n}")),
                    Err(e) => return refuse(e),
                }
            }
            _ => return refuse(format!("unexpected argument {option:?}")),
        };
        if i + operands >= args.len() {
            return refuse(format!("{option} is missing operands"));
        }
        i += 1 + operands;
    }
    Ok(())
}

/// Refuse a filter-only RediSearch query string that carries a vector clause
/// (#344). In DIALECT 2 a KNN clause needs `=>[KNN …]` and a vector range
/// needs `VECTOR_RANGE`; filter-only renders every operand through `PARAMS`,
/// so neither token can come from data. [`check_redis_command`] applies it to
/// the query argument of the command actually sent.
pub fn check_redis_query(query: &str) -> Result<(), String> {
    if query.contains("=>") || query.to_ascii_uppercase().contains("VECTOR_RANGE") {
        return Err(format!(
            "filter-only query carries a vector clause ({:?}); filter-only must issue \
             metadata predicates only",
            query.chars().take(120).collect::<String>()
        ));
    }
    Ok(())
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

/// A scalar run renames configs to `<engine>-no-vector`; accepting more than
/// one config for that engine would discard differences in limits or schema.
/// Refuse the selection before any dataset download or engine connection.
pub fn validate_engine_selection<'a>(
    engines: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<(), String> {
    let mut selected = std::collections::HashMap::new();
    for (name, engine) in engines {
        if let Some(previous) = selected.insert(engine, name) {
            return Err(format!(
                "--skip-vector-index requires one config per engine type: {engine:?} selected by {previous:?} and {name:?}; select a single config"
            ));
        }
    }
    Ok(())
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

/// Keep the first search config at each concurrency, ignoring vector-only EF
/// differences. A different result limit changes the scalar workload, so refuse
/// to silently drop it while collapsing redundant vector search configs.
/// Only Redis/OpenSearch have the fixed scalar default of 10. Legacy adapters
/// derive an omitted limit from the dataset, so None must remain distinct from
/// every explicit limit when their default is unknown.
pub fn search_param_indices(
    params: impl IntoIterator<Item = (Option<i64>, Option<i64>)>,
    default_limit: Option<i64>,
) -> Result<Vec<usize>, String> {
    let mut limits = std::collections::HashMap::new();
    let mut selected = Vec::new();
    for (index, (parallel, top)) in params.into_iter().enumerate() {
        let parallel = parallel.unwrap_or(1);
        let top = top.or(default_limit);
        if let Some(previous) = limits.get(&parallel) {
            if *previous != top {
                return Err(format!(
                    "filter-only search configs for parallel={parallel} have conflicting result limits: top={previous:?} vs top={top:?}; use one result limit per parallel level"
                ));
            }
        } else {
            limits.insert(parallel, top);
            selected.push(index);
        }
    }
    Ok(selected)
}

/// [`search_param_indices`] for an engine config that mixes workloads on the
/// shared `full` layout (#344): its filter-only search configs get exactly the
/// treatment they get on `no-vector` — the first per concurrency kept, a
/// conflicting result limit refused — so the same filter-only configs behave
/// the same on both layouts. Every other search config is kept as declared.
/// Input: `(is_filter_only, parallel, top)` per search config, in order.
/// Output: the indices to run, in declared order.
pub fn mixed_search_param_indices(
    params: impl IntoIterator<Item = (bool, Option<i64>, Option<i64>)>,
    default_limit: Option<i64>,
) -> Result<Vec<usize>, String> {
    let params: Vec<_> = params.into_iter().collect();
    let filter_only: Vec<usize> = (0..params.len()).filter(|&i| params[i].0).collect();
    let kept: std::collections::HashSet<usize> = search_param_indices(
        filter_only.iter().map(|&i| (params[i].1, params[i].2)),
        default_limit,
    )?
    .into_iter()
    .map(|k| filter_only[k])
    .collect();
    Ok((0..params.len())
        .filter(|i| !params[*i].0 || kept.contains(i))
        .collect())
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
    validate_opensearch_scalar_fields(properties, schema)
}

/// [`validate_opensearch_schema`] for the shared `full` layout (#344): the
/// index must be a k-NN index with an indexed `knn_vector` field (so it really
/// is the index the engine config's vector workloads run on, not a reused
/// no-vector or foreign index), and every scalar field the filter-only contract
/// uses must be mapped exactly as on the no-vector layout.
pub fn validate_opensearch_full_schema(
    properties: &Value,
    index: &Value,
    schema: Option<&Value>,
) -> Result<(), String> {
    if !matches!(&index["knn"], Value::Bool(true)) && index["knn"] != "true" {
        return Err(
            "filter-only on the full layout requires index.knn=true; the index was not built \
             for the full layout"
                .into(),
        );
    }
    if properties["vector"]["type"] != "knn_vector" {
        return Err(
            "filter-only on the full layout requires an indexed knn_vector field; the index \
             was not built for the full layout"
                .into(),
        );
    }
    validate_opensearch_scalar_fields(properties, schema)
}

fn validate_opensearch_scalar_fields(
    properties: &Value,
    schema: Option<&Value>,
) -> Result<(), String> {
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
            if kind == "keyword"
                && !properties[field]["split_queries_on_whitespace"].is_null()
                && properties[field]["split_queries_on_whitespace"] != false
            {
                return Err(format!(
                    "filter-only requires split_queries_on_whitespace=false for {field}"
                ));
            }
            if kind == "keyword" && !properties[field]["ignore_above"].is_null() {
                return Err(format!(
                    "filter-only requires keyword mapping without ignore_above for {field}"
                ));
            }
        }
    }
    Ok(())
}

/// FT.INFO can arrive as RESP2 pairs or a RESP3 map; attributes also contain
/// standalone flags, so do not deserialize them as key/value maps.
pub fn validate_redis_schema(info: &Value, schema: Option<&Value>) -> Result<(), String> {
    validate_redis_layout(info, schema, false)
}

/// [`validate_redis_schema`] for the shared `full` layout (#344): the index
/// must hold a VECTOR field (it is the index the engine config's vector
/// workloads run on, not a reused no-vector one), and every scalar field the
/// filter-only contract uses must be indexed exactly as on the no-vector
/// layout (`TAG SEPARATOR ; CASESENSITIVE`, `NUMERIC`).
pub fn validate_redis_full_schema(info: &Value, schema: Option<&Value>) -> Result<(), String> {
    validate_redis_layout(info, schema, true)
}

/// FT.INFO key/value entries; attributes also contain standalone flags.
fn ft_info_entries(v: &Value) -> impl Iterator<Item = (&Value, Option<&Value>)> {
    let mut items = v.as_array().into_iter().flatten();
    std::iter::from_fn(move || {
        let key = items.next()?;
        let standalone = matches!(
            key.as_str(),
            Some(
                "CASESENSITIVE"
                    | "SORTABLE"
                    | "UNF"
                    | "NOINDEX"
                    | "NOSTEM"
                    | "WITHSUFFIXTRIE"
                    | "INDEXEMPTY"
                    | "INDEXMISSING"
            )
        );
        Some((key, if standalone { None } else { items.next() }))
    })
}

/// One FT.INFO field, from a RESP3 map or RESP2 pairs.
fn ft_info_field<'a>(v: &'a Value, name: &str) -> Option<&'a Value> {
    if let Some(map) = v.as_object() {
        return map.get(name);
    }
    ft_info_entries(v).find_map(|(key, value)| (key == name).then_some(value).flatten())
}

/// Whether an FT.INFO attribute carries a standalone flag (RESP2 or RESP3).
fn ft_info_has_flag(attr: &Value, flag: &str) -> bool {
    ft_info_entries(attr).any(|(key, value)| key == flag && value.is_none())
        || ft_info_field(attr, "flags")
            .and_then(Value::as_array)
            .is_some_and(|flags| flags.iter().any(|f| f == flag))
}

/// Refuse to label a Redis index `vector` (#344) when it was built for
/// filter-only (the shared `full` or the no-vector layout): a tag field
/// indexed `CASESENSITIVE` matches case-sensitively, so a vector or hybrid
/// result on it would carry the wrong `params.index_layout`. Reached under
/// `--skip-upload`, when a `vector`-plan config reuses another config's index.
pub fn validate_redis_vector_layout(info: &Value, schema: Option<&Value>) -> Result<(), String> {
    // Fail closed: an FT.INFO without attributes is not an index whose layout
    // can be confirmed.
    let attributes = ft_info_field(info, "attributes")
        .and_then(Value::as_array)
        .ok_or(
            "index layout check: FT.INFO has no attributes; cannot confirm the `vector` layout",
        )?;
    for (name, kind) in schema.and_then(Value::as_object).into_iter().flatten() {
        if !matches!(kind.as_str(), Some("keyword" | "uuid" | "bool")) {
            continue;
        }
        let attr = attributes
            .iter()
            .find(|a| ft_info_field(a, "attribute").and_then(Value::as_str) == Some(name));
        if attr.is_some_and(|a| ft_info_has_flag(a, "CASESENSITIVE")) {
            return Err(format!(
                "the index is not a `vector`-layout index: tag field {name} is CASESENSITIVE \
                 (built for the `full` or `no-vector` layout), so its results would be \
                 mislabelled; rebuild it without --skip-upload or run the config that built it"
            ));
        }
    }
    Ok(())
}

/// [`validate_redis_vector_layout`] for OpenSearch: a schema `float` field
/// mapped `double` was built for filter-only (`full` / `no-vector`), where
/// range filters compare binary64 values instead of the vector layout's
/// 32-bit floats.
pub fn validate_opensearch_vector_layout(
    properties: &Value,
    schema: Option<&Value>,
) -> Result<(), String> {
    // Fail closed: a mapping without properties is not an index whose layout
    // can be confirmed (indexing a non-object yields null, which would pass).
    if !properties.is_object() {
        return Err(
            "index layout check: the mapping has no properties; cannot confirm the `vector` layout"
                .into(),
        );
    }
    for (name, kind) in schema.and_then(Value::as_object).into_iter().flatten() {
        if kind == "float" && properties[name]["type"] == "double" {
            return Err(format!(
                "the index is not a `vector`-layout index: float field {name} is mapped double \
                 (built for the `full` or `no-vector` layout), so its results would be \
                 mislabelled; rebuild it without --skip-upload or run the config that built it"
            ));
        }
    }
    Ok(())
}

fn validate_redis_layout(info: &Value, schema: Option<&Value>, full: bool) -> Result<(), String> {
    let field = ft_info_field;
    let attributes = field(info, "attributes")
        .and_then(Value::as_array)
        .ok_or("filter-only missing FT.INFO attributes")?;
    let mut has_vector = false;
    for attr in attributes {
        let kind = field(attr, "type")
            .and_then(Value::as_str)
            .ok_or("filter-only missing FT.INFO attribute type")?;
        if kind.eq_ignore_ascii_case("VECTOR") {
            if !full {
                return Err("filter-only cannot reuse an index containing VECTOR".into());
            }
            has_vector = true;
        }
    }
    if full && !has_vector {
        return Err(
            "filter-only on the full layout requires an index with a VECTOR field; the index \
             was not built for the full layout"
                .into(),
        );
    }
    if let Some(fields) = schema.and_then(Value::as_object) {
        for (name, kind) in fields {
            // Every kind create_index indexes as a TAG (keyword, uuid, bool) gets
            // the same CASESENSITIVE requirement as keyword (#344 review).
            let tag = matches!(kind.as_str(), Some("keyword" | "uuid" | "bool"));
            if !tag && !matches!(kind.as_str(), Some("int" | "float")) {
                continue;
            }
            let attr = attributes
                .iter()
                .find(|a| field(a, "attribute").and_then(Value::as_str) == Some(name))
                .ok_or_else(|| format!("filter-only missing indexed field {name}"))?;
            let has_flag = |flag: &str| ft_info_has_flag(attr, flag);
            if has_flag("NOINDEX") {
                return Err(format!("filter-only requires indexed field {name}"));
            }
            if !tag {
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
            let case_sensitive = has_flag("CASESENSITIVE");
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

/// One validated filter-only page: the exact total, the number of hits, and —
/// only when asked for — the returned ids in response order (`_id` for
/// OpenSearch, the full document key for Redis).
///
/// `ids` is `None` unless the caller passes `keep_ids = true`. The measured
/// worker loops pass `--dump-results`' capture flag here, so a run that does
/// not dump allocates nothing per hit beyond the duplicate check it always had
/// (#343 review): the per-query parse sits after each latency sample but inside
/// the throughput window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    pub total: u64,
    pub returned: usize,
    pub ids: Option<Vec<String>>,
}

pub fn opensearch_counts(body: &Value, limit: usize) -> Result<(u64, usize), String> {
    opensearch_page(body, limit, false).map(|p| (p.total, p.returned))
}

pub fn opensearch_page(body: &Value, limit: usize, keep_ids: bool) -> Result<Page, String> {
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
    let mut seen = std::collections::HashSet::new();
    let mut ids = keep_ids.then(|| Vec::with_capacity(hits.len()));
    for hit in hits {
        let id = hit["_id"]
            .as_str()
            .or_else(|| hit["fields"]["_id"][0].as_str())
            .ok_or("missing hit ID")?;
        if !seen.insert(id) {
            return Err("duplicate filter-only hit ID".into());
        }
        if let Some(ids) = ids.as_mut() {
            ids.push(id.to_string());
        }
    }
    validate_page(total, hits.len(), limit)?;
    Ok(Page {
        total,
        returned: hits.len(),
        ids,
    })
}

/// RESP3 exposes timeout warnings that RESP2 silently omits. Reject all
/// warnings so partial replies cannot be published as exact match counts.
pub fn redis_counts(value: &redis::Value, limit: usize) -> Result<(u64, usize), String> {
    redis_page(value, limit, false).map(|p| (p.total, p.returned))
}

/// See [`Page`] for `keep_ids`.
pub fn redis_page(value: &redis::Value, limit: usize, keep_ids: bool) -> Result<Page, String> {
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
    let mut seen = std::collections::HashSet::new();
    let mut ids = keep_ids.then(|| Vec::with_capacity(hits.len()));
    for hit in hits {
        let Value::Map(fields) = hit else {
            return Err("invalid Redis result".into());
        };
        let id = field(fields, "id")
            .and_then(text)
            .ok_or("missing Redis hit ID")?;
        if !seen.insert(id) {
            return Err("duplicate Redis hit ID".into());
        }
        if let Some(ids) = ids.as_mut() {
            ids.push(id.to_string());
        }
    }
    validate_page(total, hits.len(), limit)?;
    Ok(Page {
        total,
        returned: hits.len(),
        ids,
    })
}

/// Corpus row id from a Redis document key `<key_prefix><row>`. A key outside
/// the prefix or with a non-integer suffix is an error, never a guess: the
/// per-query dump must name the same rows the corpus was uploaded as.
pub fn redis_key_row(key: &str, key_prefix: &str) -> Result<i64, String> {
    key.strip_prefix(key_prefix)
        .and_then(|rest| rest.parse::<i64>().ok())
        .ok_or_else(|| format!("Redis key {key:?} is not <{key_prefix}><row id>"))
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
        // The page keeps the ids, in reply order, only when asked to. The
        // non-capturing path the measured loops take without --dump-results
        // builds no id Strings at all (#343 review).
        let two = reply(12, &["p:2", "p:1"], vec![]);
        assert_eq!(
            redis_page(&two, 2, true).unwrap(),
            Page {
                total: 12,
                returned: 2,
                ids: Some(vec!["p:2".into(), "p:1".into()])
            }
        );
        assert_eq!(
            redis_page(&two, 2, false).unwrap(),
            Page {
                total: 12,
                returned: 2,
                ids: None
            }
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
    fn document_ids_map_back_to_corpus_rows_or_fail() {
        assert_eq!(redis_key_row("cfg:17", "cfg:").unwrap(), 17);
        assert!(redis_key_row("other:17", "cfg:").is_err());
        assert!(redis_key_row("cfg:x", "cfg:").is_err());
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
        // Ids only when asked for (#343 review: no allocation without capture).
        assert_eq!(opensearch_page(&good, 10, false).unwrap().ids, None);
        assert_eq!(
            opensearch_page(&good, 10, true).unwrap().ids,
            Some(vec!["a".to_string(), "b".to_string()])
        );
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

#[cfg(test)]
#[path = "filter_only/full_layout_tests.rs"]
mod full_layout_tests;
