//! Vertex AI Vector Search engine (Google Cloud), pure KNN.
//!
//! Cloud-only, like Turbopuffer — there is no local server. Talks to the
//! Vertex AI REST API (`{region}-aiplatform.googleapis.com`) with a bearer
//! token, using `reqwest::blocking`.
//!
//! Lifecycle (issue: "starting simple by pure KNN"):
//! - `configure`: create a STREAM_UPDATE Index (tree-AH), an IndexEndpoint with
//!   a public endpoint, and deploy the index. Deploying is SLOW (tens of
//!   minutes) — set `VERTEX_DEPLOY_TIMEOUT_SECS` accordingly. To skip the slow
//!   create+deploy, point the engine at an already-deployed index by setting
//!   `VERTEX_INDEX`, `VERTEX_INDEX_ENDPOINT`, and `VERTEX_DEPLOYED_INDEX_ID`.
//! - `upload`: `upsertDatapoints` in batches (streaming index), then poll the
//!   index describe until `indexStats.vectorsCount` catches up to the uploaded
//!   count (STREAM_UPDATE syncs asynchronously, so an immediate search would
//!   race a partially-synced index and under-report recall). Tunable via
//!   `VERTEX_SYNC_TIMEOUT_SECS` (default 900; <= 0 opts out) and
//!   `VERTEX_SYNC_POLL_SECS` (default 10). The wait is timed and printed
//!   separately — it is NOT included in the upload throughput number.
//! - `search`: `findNeighbors` against the public endpoint, one persistent
//!   worker per `parallel`, timing only the RPC + reply parse.
//!
//! No metadata filters, no mixed workload, no quantization — pure vector KNN.
//!
//! Auth: `VERTEX_ACCESS_TOKEN` if set, otherwise `gcloud auth
//! print-access-token`. Tokens are short-lived; the token is re-fetched at the
//! start of each phase (and once before the timed search region).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, OnceLock};
use std::time::{Duration, Instant};

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::vertex_grpc::{
    NumericOp, NumericRestrict, NumericValue, Restrict, VertexFilter, VertexGrpcRequest,
    VertexGrpcWorker,
};
use crate::engine::{
    attach_open_loop_metrics, closed_loop_duration, zero_search_results, Engine, OpenLoopPlan,
    SearchResults, UpdateSearchRatio, UploadStats,
};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use vector_db_benchmark::readers::metadata::{MetadataItem, MetadataValue};

const DEFAULT_REGION: &str = "us-central1";
const DEFAULT_MACHINE_TYPE: &str = "e2-standard-16";
const DEFAULT_DISPLAY_NAME: &str = "vdb_benchmark";
const DEFAULT_APPROX_NEIGHBORS: i64 = 150;
const DEFAULT_LEAF_EMBEDDING_COUNT: i64 = 500;
const DEFAULT_LEAF_SEARCH_PERCENT: i64 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VertexQueryTransport {
    Rest,
    PublicGrpc,
    PrivateGrpc,
}

impl VertexQueryTransport {
    fn from_env() -> Result<Self, String> {
        match std::env::var("VERTEX_QUERY_TRANSPORT")
            .unwrap_or_else(|_| "rest".to_string())
            .to_ascii_lowercase()
            .replace('_', "-")
            .as_str()
        {
            "rest" => Ok(Self::Rest),
            "grpc" | "public-grpc" => Ok(Self::PublicGrpc),
            "private-grpc" | "psc-grpc" => Ok(Self::PrivateGrpc),
            value => Err(format!(
                "unsupported VERTEX_QUERY_TRANSPORT={value}; use rest, public-grpc, or private-grpc"
            )),
        }
    }
}

/// The three env vars that point the engine at an ALREADY-deployed index, in
/// which case `configure` skips create+deploy and `delete` skips teardown (so
/// the caller-owned resources are left in place). Reuse requires all three.
fn reuse_index_ids() -> Option<(String, String, String)> {
    Some((
        std::env::var("VERTEX_INDEX").ok()?,
        std::env::var("VERTEX_INDEX_ENDPOINT").ok()?,
        std::env::var("VERTEX_DEPLOYED_INDEX_ID").ok()?,
    ))
}

/// Access tokens (GCP) live ~60 min; refresh a phase-long token before it can
/// expire mid-flight.
const TOKEN_REFRESH_AFTER: Duration = Duration::from_secs(45 * 60);

/// Map a dataset distance string to a Vertex `distanceMeasureType`.
fn vertex_distance_measure(distance: &str) -> &'static str {
    match distance {
        "cosine" | "angular" => "COSINE_DISTANCE",
        "l2" | "euclidean" => "SQUARED_L2_DISTANCE",
        "dot" | "ip" => "DOT_PRODUCT_DISTANCE",
        _ => "DOT_PRODUCT_DISTANCE",
    }
}

/// Body for `indexes.create` — a STREAM_UPDATE tree-AH index. `shard_size`
/// (e.g. `SHARD_SIZE_SMALL`) constrains which deploy machine types are valid:
/// the default `SHARD_SIZE_MEDIUM` requires `e2-standard-16`+, while
/// `SHARD_SIZE_SMALL` allows smaller machines. Omitted when `None`.
fn build_index_body(
    display_name: &str,
    dimensions: i64,
    distance_measure: &str,
    approx_neighbors: i64,
    leaf_embedding_count: i64,
    leaf_search_percent: i64,
    shard_size: Option<&str>,
) -> serde_json::Value {
    let mut config = serde_json::json!({
        "dimensions": dimensions,
        "approximateNeighborsCount": approx_neighbors,
        "distanceMeasureType": distance_measure,
        "algorithmConfig": {
            "treeAhConfig": {
                "leafNodeEmbeddingCount": leaf_embedding_count.to_string(),
                "leafNodesToSearchPercent": leaf_search_percent,
            }
        }
    });
    if let Some(shard) = shard_size {
        config["shardSize"] = serde_json::json!(shard);
    }
    serde_json::json!({
        "displayName": display_name,
        "indexUpdateMethod": "STREAM_UPDATE",
        "metadata": { "config": config },
    })
}

/// Body for `indexEndpoints.deployIndex`.
fn build_deploy_body(
    deployed_index_id: &str,
    index_name: &str,
    machine_type: &str,
) -> serde_json::Value {
    serde_json::json!({
        "deployedIndex": {
            "id": deployed_index_id,
            "index": index_name,
            "dedicatedResources": {
                "machineSpec": { "machineType": machine_type },
                "minReplicaCount": 1,
                "maxReplicaCount": 1,
            }
        }
    })
}

/// Body for `indexes.upsertDatapoints` — ids are stringified row indices.
/// Map a datapoint's stored metadata to Vertex restrictions for upsert: string
/// and `labels` fields become categorical `restricts`; int/float fields become
/// `numericRestricts` (no operator — a stored value carries none). Geo is not
/// filterable in Vertex and is skipped.
/// Pick the Vertex numeric type for `field`'s value. Vertex compares numeric
/// restrictions by type, so a stored `valueInt` never matches a query
/// `valueDouble` (and vice-versa). The dataset schema is the source of truth:
/// a `float`-declared field is always `Double`, an `int` field always `Int`.
/// Without a schema hint we fall back to an int-first heuristic (whole numbers
/// are `Int`), which keeps integer datasets self-consistent.
fn typed_numeric(
    field: &str,
    i: Option<i64>,
    f: Option<f64>,
    schema: &HashMap<String, String>,
) -> NumericValue {
    match schema.get(field).map(|s| s.as_str()) {
        Some("float") => NumericValue::Double(f.or_else(|| i.map(|n| n as f64)).unwrap_or(0.0)),
        Some("int") => NumericValue::Int(i.or_else(|| f.map(|x| x as i64)).unwrap_or(0)),
        _ => match i {
            Some(n) => NumericValue::Int(n),
            None => NumericValue::Double(f.unwrap_or(0.0)),
        },
    }
}

/// Map a datapoint's stored metadata to Vertex restrictions for upsert: string
/// and `labels` fields become categorical `restricts`; int/float fields become
/// `numericRestricts` (no operator — a stored value carries none), typed by the
/// dataset schema so query restrictions of the same field match. Geo is not
/// filterable in Vertex and is skipped.
fn metadata_to_filter(meta: &MetadataItem, schema: &HashMap<String, String>) -> VertexFilter {
    let mut filter = VertexFilter::default();
    for (key, value) in &meta.fields {
        match value {
            MetadataValue::String(s) => filter.restricts.push(Restrict {
                namespace: key.clone(),
                allow_list: vec![s.clone()],
            }),
            MetadataValue::Labels(labels) => filter.restricts.push(Restrict {
                namespace: key.clone(),
                allow_list: labels.clone(),
            }),
            MetadataValue::Int(n) => filter.numeric_restricts.push(NumericRestrict {
                namespace: key.clone(),
                op: None,
                value: typed_numeric(key, Some(*n), None, schema),
            }),
            MetadataValue::Float(f) => filter.numeric_restricts.push(NumericRestrict {
                namespace: key.clone(),
                op: None,
                value: typed_numeric(key, None, Some(*f), schema),
            }),
            MetadataValue::Geo { .. } => {}
        }
    }
    filter
}

/// Parse the benchmark's query `conditions` into a Vertex query filter. Vertex
/// restrictions AND across namespaces and OR within a namespace's `allowList`,
/// so an `and` of per-field conditions maps cleanly. Anything Vertex cannot
/// express is a hard error (per the engine's "no silent partial filter" policy):
/// cross-field `or`, nested boolean, numeric `match_any` (an IN-list can't be a
/// single numeric restriction), and geo.
fn parse_vertex_filter(
    conditions: &serde_json::Value,
    schema: &HashMap<String, String>,
) -> Result<VertexFilter, String> {
    let mut filter = VertexFilter::default();
    // A missing or explicitly-null condition is "no filter" (unrestricted query),
    // not an error.
    if conditions.is_null() {
        return Ok(filter);
    }
    // Accept `{ "and": [ {field: spec}, ... ] }`, or a bare object of field->spec
    // (treated as an implicit AND). Reject top-level `or`.
    let clauses: Vec<&serde_json::Value> =
        if let Some(and) = conditions.get("and").and_then(|v| v.as_array()) {
            and.iter().collect()
        } else if conditions.get("or").is_some() {
            return Err("Vertex filters cannot express cross-field OR (`or`)".to_string());
        } else if conditions.is_object() {
            vec![conditions]
        } else {
            return Err("unsupported filter conditions shape".to_string());
        };

    for clause in clauses {
        let obj = clause
            .as_object()
            .ok_or("filter clause must be an object")?;
        for (field, spec) in obj {
            if field == "and" || field == "or" {
                return Err("Vertex filters cannot express nested boolean logic".to_string());
            }
            let spec = spec
                .as_object()
                .ok_or_else(|| format!("filter for `{field}` must be an object"))?;
            for (op, criteria) in spec {
                match op.as_str() {
                    "match" => parse_match(field, criteria, &mut filter, schema)?,
                    "range" => parse_range(field, criteria, &mut filter, schema)?,
                    "geo_radius" | "geo_bounding_box" => {
                        return Err(format!("Vertex cannot filter geo field `{field}`"));
                    }
                    other => return Err(format!("unsupported filter operator `{other}`")),
                }
            }
        }
    }
    Ok(filter)
}

fn parse_match(
    field: &str,
    criteria: &serde_json::Value,
    filter: &mut VertexFilter,
    schema: &HashMap<String, String>,
) -> Result<(), String> {
    if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
        // Categorical contains-any → allowList. A numeric IN-list can't be one
        // numeric restriction (multiple numeric restricts AND, not OR), so it is
        // rejected rather than silently mis-applied.
        if any.iter().any(|v| v.is_number()) {
            return Err(format!(
                "Vertex cannot express a numeric `match_any` (IN-list) on `{field}`"
            ));
        }
        let allow_list: Vec<String> = any
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        filter.restricts.push(Restrict {
            namespace: field.to_string(),
            allow_list,
        });
        Ok(())
    } else if let Some(value) = criteria.get("value") {
        if let Some(s) = value.as_str() {
            filter.restricts.push(Restrict {
                namespace: field.to_string(),
                allow_list: vec![s.to_string()],
            });
        } else if value.is_number() {
            filter.numeric_restricts.push(NumericRestrict {
                namespace: field.to_string(),
                op: Some(NumericOp::Equal),
                value: typed_numeric(field, value.as_i64(), value.as_f64(), schema),
            });
        } else {
            return Err(format!("unsupported match value for `{field}`"));
        }
        Ok(())
    } else {
        Err(format!("empty match filter for `{field}`"))
    }
}

fn parse_range(
    field: &str,
    criteria: &serde_json::Value,
    filter: &mut VertexFilter,
    schema: &HashMap<String, String>,
) -> Result<(), String> {
    let obj = criteria
        .as_object()
        .ok_or_else(|| format!("range for `{field}` must be an object"))?;
    for (bound, val) in obj {
        if val.is_null() {
            continue;
        }
        let op = match bound.as_str() {
            "lt" => NumericOp::Less,
            "lte" => NumericOp::LessEqual,
            "gt" => NumericOp::Greater,
            "gte" => NumericOp::GreaterEqual,
            other => return Err(format!("unsupported range bound `{other}` on `{field}`")),
        };
        if !val.is_number() {
            return Err(format!("non-numeric range bound on `{field}`"));
        }
        filter.numeric_restricts.push(NumericRestrict {
            namespace: field.to_string(),
            op: Some(op),
            value: typed_numeric(field, val.as_i64(), val.as_f64(), schema),
        });
    }
    Ok(())
}

/// Extract `field -> declared-type` from the dataset schema (used to type
/// numeric restrictions consistently between upload and query).
fn schema_type_map(dataset: &Dataset) -> HashMap<String, String> {
    let mut m = HashMap::new();
    if let Some(obj) = dataset.config.schema.as_ref().and_then(|s| s.as_object()) {
        for (k, v) in obj {
            if let Some(t) = v.as_str() {
                m.insert(k.clone(), t.to_string());
            }
        }
    }
    m
}

/// Serialize a filter to REST JSON `restricts` / `numericRestricts`. Datapoint
/// restrictions omit `op`; query restrictions include it.
fn filter_to_rest(filter: &VertexFilter) -> (serde_json::Value, serde_json::Value) {
    let restricts: Vec<serde_json::Value> = filter
        .restricts
        .iter()
        .map(|r| serde_json::json!({ "namespace": r.namespace, "allowList": r.allow_list }))
        .collect();
    let numeric: Vec<serde_json::Value> = filter
        .numeric_restricts
        .iter()
        .map(|n| {
            let mut obj = serde_json::Map::new();
            obj.insert("namespace".to_string(), serde_json::json!(n.namespace));
            match n.value {
                NumericValue::Int(i) => {
                    obj.insert("valueInt".to_string(), serde_json::json!(i));
                }
                NumericValue::Double(d) => {
                    obj.insert("valueDouble".to_string(), serde_json::json!(d));
                }
            }
            if let Some(op) = n.op {
                obj.insert("op".to_string(), serde_json::json!(op.as_rest()));
            }
            serde_json::Value::Object(obj)
        })
        .collect();
    (serde_json::json!(restricts), serde_json::json!(numeric))
}

fn build_upsert_body(
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
    schema: &HashMap<String, String>,
) -> serde_json::Value {
    let datapoints: Vec<serde_json::Value> = ids
        .iter()
        .zip(vectors.iter())
        .enumerate()
        .map(|(i, (id, v))| {
            let mut dp = serde_json::json!({
                "datapointId": id.to_string(),
                "featureVector": v,
            });
            if let Some(Some(meta)) = metadata.get(i) {
                let filter = metadata_to_filter(meta, schema);
                if !filter.is_empty() {
                    let (restricts, numeric) = filter_to_rest(&filter);
                    if !filter.restricts.is_empty() {
                        dp["restricts"] = restricts;
                    }
                    if !filter.numeric_restricts.is_empty() {
                        dp["numericRestricts"] = numeric;
                    }
                }
            }
            dp
        })
        .collect();
    serde_json::json!({ "datapoints": datapoints })
}

/// Serialize a filter to Vertex *batch file* `restricts` / `numeric_restricts`
/// JSON. The batch (GCS) schema differs from the streaming upsert schema:
/// batch uses `allow`/`deny` (not `allowList`) and `value_int`/`value_float`
/// (not `valueInt`/`valueDouble`). Datapoint restrictions carry no operator.
///
/// See https://cloud.google.com/vertex-ai/docs/vector-search/setup/format-structure
fn filter_to_batch(filter: &VertexFilter) -> (serde_json::Value, serde_json::Value) {
    let restricts: Vec<serde_json::Value> = filter
        .restricts
        .iter()
        .map(|r| serde_json::json!({ "namespace": r.namespace, "allow": r.allow_list }))
        .collect();
    let numeric: Vec<serde_json::Value> = filter
        .numeric_restricts
        .iter()
        .map(|n| {
            let mut obj = serde_json::Map::new();
            obj.insert("namespace".to_string(), serde_json::json!(n.namespace));
            match n.value {
                NumericValue::Int(i) => {
                    obj.insert("value_int".to_string(), serde_json::json!(i));
                }
                NumericValue::Double(d) => {
                    obj.insert("value_float".to_string(), serde_json::json!(d));
                }
            }
            serde_json::Value::Object(obj)
        })
        .collect();
    (serde_json::json!(restricts), serde_json::json!(numeric))
}

/// Serialize all datapoints to Vertex batch-import JSONL (#187): one JSON
/// object per line in Vertex's *batch file* schema —
/// `{"id","embedding","restricts":[{namespace,allow}],"numeric_restricts":[…]}`.
/// This is what gets staged to GCS and referenced by `contentsDeltaUri`.
///
/// The batch schema deliberately differs from the streaming upsert body
/// (`build_upsert_body`): here ids/vectors are `id`/`embedding` and restrictions
/// use `allow`/`value_int`/`value_float` (see `filter_to_batch`).
fn build_batch_datapoint_jsonl(
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
    schema: &HashMap<String, String>,
) -> String {
    let mut out = String::with_capacity(vectors.len() * 32);
    for (i, v) in vectors.iter().enumerate() {
        let mut obj = serde_json::json!({
            "id": ids[i].to_string(),
            "embedding": v,
        });
        if let Some(Some(meta)) = metadata.get(i) {
            let filter = metadata_to_filter(meta, schema);
            let (restricts, numeric) = filter_to_batch(&filter);
            if !filter.restricts.is_empty() {
                obj["restricts"] = restricts;
            }
            if !filter.numeric_restricts.is_empty() {
                obj["numeric_restricts"] = numeric;
            }
        }
        // serde_json never fails to serialize a Value; fall back to an empty
        // object rather than silently dropping a line on the impossible path.
        out.push_str(&serde_json::to_string(&obj).unwrap_or_else(|_| "{}".to_string()));
        out.push('\n');
    }
    out
}

/// BATCH_UPDATE variant of `build_index_body` (#187). Creates an empty
/// bulk-ingest index; datapoints are added afterwards by UpdateIndex with a
/// `contentsDeltaUri` (see `build_batch_update_body`). `upsertDatapoints`
/// (streaming) does NOT work on a BATCH_UPDATE index and vice-versa.
fn build_batch_index_body(
    display_name: &str,
    dimensions: i64,
    distance_measure: &str,
    approx_neighbors: i64,
    leaf_embedding_count: i64,
    leaf_search_percent: i64,
    shard_size: Option<&str>,
) -> serde_json::Value {
    let mut config = serde_json::json!({
        "dimensions": dimensions,
        "approximateNeighborsCount": approx_neighbors,
        "distanceMeasureType": distance_measure,
        "algorithmConfig": {
            "treeAhConfig": {
                "leafNodeEmbeddingCount": leaf_embedding_count.to_string(),
                "leafNodesToSearchPercent": leaf_search_percent,
            }
        }
    });
    if let Some(shard) = shard_size {
        config["shardSize"] = serde_json::json!(shard);
    }
    serde_json::json!({
        "displayName": display_name,
        "indexUpdateMethod": "BATCH_UPDATE",
        "metadata": { "config": config },
    })
}

/// UpdateIndex body that points a BATCH_UPDATE index at a staged GCS folder and
/// triggers a rebuild. Vertex reads ALL files directly under the folder named
/// here, so callers pass the *folder* URI, not the object URI.
///
/// CRITICAL (confirmed against live Vertex): `contentsDeltaUri` lives inside the
/// Struct-typed `metadata` field, and a GCP field mask does NOT descend into a
/// Struct. Masking `metadata.contentsDeltaUri` is silently accepted but no-ops —
/// the operation returns `done:true` in ~0s and ingests ZERO vectors. The PATCH
/// must therefore mask the WHOLE `metadata` (`updateMask=metadata`) and resend
/// the existing `config` alongside the delta URI (masking `metadata` replaces
/// the entire struct, so omitting `config` would wipe it). `isCompleteOverwrite`
/// rebuilds the index from exactly the staged files.
fn build_batch_update_body(
    config: &serde_json::Value,
    contents_delta_uri: &str,
) -> serde_json::Value {
    serde_json::json!({
        "metadata": {
            "config": config,
            "contentsDeltaUri": contents_delta_uri,
            "isCompleteOverwrite": true,
        },
    })
}

/// Whether GCS batch ingest is worthwhile for this run. Batch trades a
/// GCS round-trip + a full index rebuild for far higher ingest throughput on
/// large corpora, so it only pays off past a threshold; below it, streaming
/// upsert is simpler and lower-latency. Requires a non-empty staging bucket.
fn should_batch_ingest(bucket: Option<&str>, count: usize, threshold: usize) -> bool {
    bucket.map(|b| !b.trim().is_empty()).unwrap_or(false) && count >= threshold
}

/// Resolve the EFFECTIVE `approximateNeighborCount` for a Vertex query (#200).
/// Returns `(value, source)` where source is `"config"` when the sweep set
/// `num_candidates` and `"index-default"` when it fell back to the index's
/// configured `approximateNeighborsCount`. Never returns Vertex's `0` "use
/// index default" sentinel — the value is always explicit. `top` is the floor
/// (Vertex rejects `approximateNeighborCount < neighborCount`).
fn resolve_approx_neighbor_count(
    num_candidates: Option<i64>,
    index_approx_neighbors: i64,
    top: usize,
) -> (i64, &'static str) {
    let floor = top as i64;
    match num_candidates {
        Some(n) => (n.max(floor), "config"),
        None => (index_approx_neighbors.max(floor), "index-default"),
    }
}

/// Body for `indexEndpoints.findNeighbors` (single query). An optional
/// `fraction_leaf_nodes_to_search_override` (0..1) trades recall for latency.
fn build_find_neighbors_body(
    deployed_index_id: &str,
    query: &[f32],
    top: usize,
    fraction_leaf_override: Option<f64>,
    approximate_neighbor_count: Option<i64>,
    filter: Option<&VertexFilter>,
) -> serde_json::Value {
    let mut datapoint = serde_json::json!({ "datapoint": { "featureVector": query } });
    if let Some(f) = filter {
        if !f.is_empty() {
            let (restricts, numeric) = filter_to_rest(f);
            if !f.restricts.is_empty() {
                datapoint["datapoint"]["restricts"] = restricts;
            }
            if !f.numeric_restricts.is_empty() {
                datapoint["datapoint"]["numericRestricts"] = numeric;
            }
        }
    }
    if let Some(frac) = fraction_leaf_override {
        datapoint["fractionLeafNodesToSearchOverride"] = serde_json::json!(frac);
    }
    if let Some(count) = approximate_neighbor_count {
        datapoint["approximateNeighborCount"] = serde_json::json!(count);
    }
    datapoint["neighborCount"] = serde_json::json!(top);
    serde_json::json!({
        "deployedIndexId": deployed_index_id,
        "queries": [datapoint],
        "returnFullDatapoint": false,
    })
}

/// Parse the neighbor ids of the FIRST query from a `findNeighbors` reply.
/// Datapoint ids are stringified integers; unparseable ids are skipped.
fn parse_find_neighbors_response(resp: &serde_json::Value) -> Vec<i64> {
    resp.get("nearestNeighbors")
        .and_then(|nn| nn.as_array())
        .and_then(|arr| arr.first())
        .and_then(|first| first.get("neighbors"))
        .and_then(|n| n.as_array())
        .map(|neighbors| {
            neighbors
                .iter()
                .filter_map(|nb| {
                    nb.get("datapoint")
                        .and_then(|dp| dp.get("datapointId"))
                        .and_then(|id| id.as_str())
                        .and_then(|s| s.parse::<i64>().ok())
                })
                .collect()
        })
        .unwrap_or_default()
}

fn execute_find_neighbors(
    client: &reqwest::blocking::Client,
    url: &str,
    token: &str,
    body: &serde_json::Value,
) -> Result<Vec<i64>, String> {
    let resp = client
        .post(url)
        .bearer_auth(token)
        .json(body)
        .send()
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!(
            "{}: {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }
    let json: serde_json::Value = resp.json().map_err(|e| e.to_string())?;
    Ok(parse_find_neighbors_response(&json))
}

enum VertexWorkerRequest {
    Rest(serde_json::Value),
    Grpc(VertexGrpcRequest),
}

enum VertexWorker {
    Rest {
        client: reqwest::blocking::Client,
        url: String,
        token: String,
    },
    Grpc(VertexGrpcWorker),
}

struct VertexWorkerConfig<'a> {
    transport: VertexQueryTransport,
    public_domain: &'a str,
    private_address: Option<&'a str>,
    token: &'a str,
    index_endpoint: &'a str,
    deployed_index_id: &'a str,
}

impl VertexWorker {
    fn new(config: VertexWorkerConfig<'_>) -> Result<Self, String> {
        match config.transport {
            VertexQueryTransport::Rest => {
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(60))
                    .build()
                    .map_err(|e| format!("Vertex REST worker client build failed: {e}"))?;
                Ok(Self::Rest {
                    client,
                    url: format!(
                        "https://{}/v1/{}:findNeighbors",
                        config.public_domain, config.index_endpoint
                    ),
                    token: config.token.to_string(),
                })
            }
            VertexQueryTransport::PublicGrpc => Ok(Self::Grpc(VertexGrpcWorker::public(
                config.public_domain,
                config.token,
                config.index_endpoint,
                config.deployed_index_id,
            )?)),
            VertexQueryTransport::PrivateGrpc => Ok(Self::Grpc(VertexGrpcWorker::private(
                config
                    .private_address
                    .ok_or("VERTEX_GRPC_ADDRESS is required for private-grpc transport")?,
                config.deployed_index_id,
            )?)),
        }
    }

    fn request(
        &self,
        deployed_index_id: &str,
        vector: &[f32],
        top: usize,
        fraction_leaf_override: Option<f64>,
        approximate_neighbor_count: Option<i64>,
        filter: Option<&VertexFilter>,
    ) -> VertexWorkerRequest {
        match self {
            Self::Rest { .. } => VertexWorkerRequest::Rest(build_find_neighbors_body(
                deployed_index_id,
                vector,
                top,
                fraction_leaf_override,
                approximate_neighbor_count,
                filter,
            )),
            Self::Grpc(worker) => VertexWorkerRequest::Grpc(worker.request(
                vector,
                top,
                fraction_leaf_override,
                approximate_neighbor_count,
                filter,
            )),
        }
    }

    fn execute(&mut self, request: VertexWorkerRequest) -> Result<Vec<i64>, String> {
        match (self, request) {
            (Self::Rest { client, url, token }, VertexWorkerRequest::Rest(body)) => {
                execute_find_neighbors(client, url, token, &body)
            }
            (Self::Grpc(worker), VertexWorkerRequest::Grpc(message)) => worker.execute(message),
            _ => Err("Vertex worker/request transport mismatch".to_string()),
        }
    }
}

/// Inspect a long-running-operation reply: `Ok(Some(response))` when done,
/// `Ok(None)` while still running, `Err` when the operation reported an error.
fn parse_lro(resp: &serde_json::Value) -> Result<Option<serde_json::Value>, String> {
    if let Some(err) = resp.get("error") {
        if !err.is_null() {
            return Err(format!("operation failed: {}", err));
        }
    }
    if resp.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
        // A done op with no `response` (e.g. delete) still counts as complete.
        return Ok(Some(
            resp.get("response")
                .cloned()
                .unwrap_or(serde_json::json!({})),
        ));
    }
    Ok(None)
}

/// Pull `indexStats.vectorsCount` out of an index describe reply.
///
/// GCP's REST convention encodes int64 fields as JSON *strings* (e.g.
/// `"vectorsCount": "10000"`), but we also tolerate a bare JSON number in case
/// the API ever changes or a mock feeds us one — the value is the same integer
/// either way. Returns `None` when the field is absent or unparseable, letting
/// the caller treat "no reading yet" as "keep waiting".
fn parse_vectors_count(idx: &serde_json::Value) -> Option<u64> {
    let v = idx.get("indexStats")?.get("vectorsCount")?;
    match v {
        serde_json::Value::String(s) => s.trim().parse::<u64>().ok(),
        serde_json::Value::Number(n) => n.as_u64(),
        _ => None,
    }
}

pub struct VertexEngine {
    name: String,
    project: String,
    region: String,
    machine_type: String,
    display_name: String,
    batch_size: usize,
    parallel: usize,
    search_params: Vec<SearchParams>,
    distance_measure: String,
    deploy_timeout: Duration,
    approx_neighbors: i64,
    leaf_embedding_count: i64,
    leaf_search_percent: i64,
    shard_size: Option<String>,
    query_transport: VertexQueryTransport,
    grpc_address: Option<String>,
    // Batch (bulk) ingest via GCS contentsDeltaUri (#187). When a staging
    // bucket is set the index is created BATCH_UPDATE and upload stages all
    // datapoints to GCS instead of streaming them via upsertDatapoints.
    gcs_staging_bucket: Option<String>,
    batch_threshold: usize,
    // Populated during configure().
    index_name: String,
    index_endpoint_name: String,
    deployed_index_id: String,
    public_endpoint_domain: String,
    client: reqwest::blocking::Client,
}

impl VertexEngine {
    pub fn new(engine_config: &EngineConfig, _host: &str) -> Result<Self, String> {
        let project = std::env::var("VERTEX_PROJECT").map_err(|_| {
            "VERTEX_PROJECT environment variable is required for the vertex engine".to_string()
        })?;
        let region = std::env::var("VERTEX_REGION").unwrap_or_else(|_| DEFAULT_REGION.to_string());
        let machine_type = std::env::var("VERTEX_MACHINE_TYPE")
            .unwrap_or_else(|_| DEFAULT_MACHINE_TYPE.to_string());
        let display_name = std::env::var("VERTEX_INDEX_DISPLAY_NAME")
            .unwrap_or_else(|_| DEFAULT_DISPLAY_NAME.to_string());

        let deploy_timeout = Duration::from_secs(
            std::env::var("VERTEX_DEPLOY_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3600),
        );

        let env_i64 = |k: &str, default: i64| -> i64 {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(4) as usize;
        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1000) as usize;

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

        Ok(Self {
            name: engine_config.name.clone(),
            project,
            region,
            machine_type,
            display_name,
            batch_size,
            parallel,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            distance_measure: String::new(),
            deploy_timeout,
            approx_neighbors: env_i64("VERTEX_APPROX_NEIGHBORS", DEFAULT_APPROX_NEIGHBORS),
            leaf_embedding_count: env_i64(
                "VERTEX_LEAF_EMBEDDING_COUNT",
                DEFAULT_LEAF_EMBEDDING_COUNT,
            ),
            leaf_search_percent: env_i64("VERTEX_LEAF_SEARCH_PERCENT", DEFAULT_LEAF_SEARCH_PERCENT),
            shard_size: std::env::var("VERTEX_SHARD_SIZE")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            query_transport: VertexQueryTransport::from_env()?,
            grpc_address: std::env::var("VERTEX_GRPC_ADDRESS")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            gcs_staging_bucket: std::env::var("VERTEX_GCS_STAGING_BUCKET")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            batch_threshold: std::env::var("VERTEX_BATCH_THRESHOLD")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(100_000),
            index_name: String::new(),
            index_endpoint_name: String::new(),
            deployed_index_id: String::new(),
            public_endpoint_domain: String::new(),
            client,
        })
    }

    fn base_url(&self) -> String {
        format!("https://{}-aiplatform.googleapis.com/v1", self.region)
    }

    /// True when a GCS staging bucket is configured, i.e. the operator has
    /// opted into BATCH_UPDATE bulk ingest (#187). The index-update method is a
    /// fixed property of the index, so configure() and upload() must agree on
    /// this: a BATCH_UPDATE index rejects streaming `upsertDatapoints`.
    fn batch_mode(&self) -> bool {
        self.gcs_staging_bucket
            .as_deref()
            .map(|b| !b.trim().is_empty())
            .unwrap_or(false)
    }

    fn parent(&self) -> String {
        format!("projects/{}/locations/{}", self.project, self.region)
    }

    /// Resolve the EFFECTIVE search knobs and log them, never sending Vertex the
    /// "unset → use index default" sentinel silently (#200).
    ///
    /// Vertex treats a `0` `approximate_neighbor_count` (and `0.0`
    /// `fraction_leaf_nodes_to_search_override`) as "use the deployed index
    /// default". So a sweep point that forgets to set `num_candidates` would run
    /// at the index default (~150) while still being LABELLED with whatever the
    /// config claims — fewer candidates = less work = flattering QPS/latency at
    /// lower recall, which breaks recall-matching against the Redis `ef` sweep
    /// (fairness gate #4 of the #148 review).
    ///
    /// Fix: honor the config's `num_candidates` when set (clamped to `top`, which
    /// Vertex requires as the floor for `approximateNeighborCount`); when unset,
    /// fall back to the index's OWN configured `approximateNeighborsCount`
    /// (`self.approx_neighbors`) **explicitly** — the value the index was built
    /// with, sent as a real number rather than a hidden `0` — and log it. The
    /// effective leaf-fraction is logged too (index default when unset). Returns
    /// `(approximate_neighbor_count, fraction_leaf_override)` for `request()`.
    fn resolve_search_knobs(&self, params: &SearchParams, top: usize) -> (i64, Option<f64>) {
        let (count, count_src) =
            resolve_approx_neighbor_count(params.num_candidates, self.approx_neighbors, top);
        let fraction_leaf_override: Option<f64> = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.extra.as_ref())
            .and_then(|e| e.get("fraction_leaf_nodes_to_search_override"))
            .and_then(|v| v.as_f64());
        match fraction_leaf_override {
            Some(f) => println!(
                "\tVertex effective search knobs: approximateNeighborCount={count} ({count_src}), fraction_leaf_nodes_to_search={f:.4} (config)"
            ),
            None => println!(
                "\tVertex effective search knobs: approximateNeighborCount={count} ({count_src}), fraction_leaf_nodes_to_search=index-default (leafNodesToSearchPercent={}%)",
                self.leaf_search_percent
            ),
        }
        (count, fraction_leaf_override)
    }

    /// A fresh bearer token: `VERTEX_ACCESS_TOKEN` if set, else `gcloud auth
    /// print-access-token`. Re-fetched per phase so a long deploy doesn't run on
    /// an expired token.
    fn access_token(&self) -> Result<String, String> {
        if let Ok(t) = std::env::var("VERTEX_ACCESS_TOKEN") {
            let t = t.trim().to_string();
            if !t.is_empty() {
                return Ok(t);
            }
        }
        let out = std::process::Command::new("gcloud")
            .args(["auth", "print-access-token"])
            .output()
            .map_err(|e| {
                format!("VERTEX_ACCESS_TOKEN unset and running `gcloud auth print-access-token` failed: {}", e)
            })?;
        if !out.status.success() {
            return Err(format!(
                "`gcloud auth print-access-token` failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    /// POST JSON to `url`, returning the parsed reply. Non-2xx is an error with
    /// the response body (Vertex returns a helpful `error.message`).
    fn post_json(
        &self,
        url: &str,
        token: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let resp = self
            .client
            .post(url)
            .bearer_auth(token)
            .json(body)
            .send()
            .map_err(|e| format!("POST {} failed: {}", url, e))?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(format!("POST {} -> {}: {}", url, status, text));
        }
        serde_json::from_str(&text).map_err(|e| format!("invalid JSON from {}: {}", url, e))
    }

    fn get_json(&self, url: &str, token: &str) -> Result<serde_json::Value, String> {
        let resp = self
            .client
            .get(url)
            .bearer_auth(token)
            .send()
            .map_err(|e| format!("GET {} failed: {}", url, e))?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(format!("GET {} -> {}: {}", url, status, text));
        }
        serde_json::from_str(&text).map_err(|e| format!("invalid JSON from {}: {}", url, e))
    }

    /// Poll a long-running operation until done, returning its `response`.
    fn poll_operation(
        &self,
        operation_name: &str,
        timeout: Duration,
        label: &str,
    ) -> Result<serde_json::Value, String> {
        let url = format!("{}/{}", self.base_url(), operation_name);
        let start = Instant::now();
        let mut logged = false;
        loop {
            // Re-fetch the token each poll so a multi-minute wait can't expire it.
            let token = self.access_token()?;
            let resp = self.get_json(&url, &token)?;
            match parse_lro(&resp)? {
                Some(response) => return Ok(response),
                None => {
                    if start.elapsed() > timeout {
                        return Err(format!(
                            "{} did not finish within {}s",
                            label,
                            timeout.as_secs()
                        ));
                    }
                    if !logged {
                        println!(
                            "\tWaiting for {} (polling, timeout {}s)...",
                            label,
                            timeout.as_secs()
                        );
                        logged = true;
                    }
                    std::thread::sleep(Duration::from_secs(15));
                }
            }
        }
    }

    /// Block until the STREAM_UPDATE index has finished ingesting the vectors we
    /// just upserted, or a timeout elapses.
    ///
    /// WHY: Vertex indexes STREAM_UPDATE datapoints ASYNCHRONOUSLY. `upsertDatapoints`
    /// returns as soon as the write is accepted, but the tree-AH index keeps
    /// syncing in the background for a while afterwards. Searching immediately
    /// therefore hits a partially-synced index and reports low/zero recall
    /// (issue #151 item 7). There is no "sync done" signal in the API, so we
    /// poll the index describe and watch `indexStats.vectorsCount` climb toward
    /// the number of vectors we uploaded.
    ///
    /// This is telemetry/correctness, NOT part of the upload throughput number —
    /// the caller measures and prints the wait separately so `upload_time` stays
    /// pure upsert wall-clock.
    ///
    /// CAVEAT: `vectorsCount` is the TOTAL vector count in the index, not just
    /// this run's contribution. In reuse-index mode (`VERTEX_INDEX` et al.) the
    /// index may already hold vectors from a prior run, so `>= expected_count`
    /// can be satisfied instantly (over-count) or, if a prior run left fewer,
    /// approached from below as expected. We keep the rule deliberately simple:
    /// wait until `vectorsCount >= expected_count` OR the timeout fires.
    ///
    /// Soft by design: on timeout or repeated describe failures we print a
    /// WARNING and return `Ok(())` so the benchmark still proceeds — a genuinely
    /// unsynced index will surface as poor recall in the results, which is more
    /// honest than aborting the run.
    fn wait_for_index_sync(&self, expected_count: usize) -> Result<(), String> {
        // Env-gated knobs. Malformed values fall back to the defaults rather
        // than aborting the whole benchmark over a typo.
        let timeout_secs = std::env::var("VERTEX_SYNC_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(900);
        // A non-positive timeout is an explicit opt-out of the wait entirely.
        if timeout_secs <= 0 {
            println!("Vertex index sync wait disabled (VERTEX_SYNC_TIMEOUT_SECS <= 0)");
            return Ok(());
        }
        let timeout = Duration::from_secs(timeout_secs as u64);
        let poll_secs = std::env::var("VERTEX_SYNC_POLL_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|s| *s > 0)
            .unwrap_or(10);

        let expected = expected_count as u64;
        println!(
            "Waiting for Vertex STREAM_UPDATE index to sync {} vectors (timeout {}s, poll {}s)...",
            expected,
            timeout.as_secs(),
            poll_secs
        );

        let describe_url = format!("{}/{}", self.base_url(), self.index_name);
        let start = Instant::now();
        let mut consecutive_failures = 0usize;
        // A handful of transient describe failures are normal on a busy index;
        // bail (softly) only if they persist, so we never wait out the full
        // timeout hammering a broken endpoint.
        const MAX_CONSECUTIVE_FAILURES: usize = 5;

        loop {
            // Re-mint the token each iteration: sync waits can outlast the ~60-min
            // GCP token lifetime, and describe on an expired token would 401.
            let token = match self.access_token() {
                Ok(t) => t,
                Err(e) => {
                    consecutive_failures += 1;
                    eprintln!(
                        "Vertex index sync: token refresh failed ({e}) [{consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}]"
                    );
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                        eprintln!(
                            "WARNING: giving up on Vertex index sync wait after {consecutive_failures} consecutive token failures; proceeding (recall may be low if the index is not yet synced)"
                        );
                        return Ok(());
                    }
                    std::thread::sleep(Duration::from_secs(poll_secs));
                    continue;
                }
            };

            match self.get_json(&describe_url, &token) {
                Ok(idx) => {
                    consecutive_failures = 0;
                    let count = parse_vectors_count(&idx).unwrap_or(0);
                    println!(
                        "Vertex index sync: {}/{} vectors (elapsed {:.0}s)",
                        count,
                        expected,
                        start.elapsed().as_secs_f64()
                    );
                    if count >= expected {
                        println!(
                            "Vertex index sync complete: {}/{} vectors after {:.0}s",
                            count,
                            expected,
                            start.elapsed().as_secs_f64()
                        );
                        return Ok(());
                    }
                }
                Err(e) => {
                    consecutive_failures += 1;
                    eprintln!(
                        "Vertex index sync: describe failed ({e}) [{consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}]"
                    );
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                        eprintln!(
                            "WARNING: giving up on Vertex index sync wait after {consecutive_failures} consecutive describe failures; proceeding (recall may be low if the index is not yet synced)"
                        );
                        return Ok(());
                    }
                }
            }

            if start.elapsed() > timeout {
                eprintln!(
                    "WARNING: Vertex index did not report {} synced vectors within {}s; proceeding anyway (recall may be low if the index is still syncing)",
                    expected,
                    timeout.as_secs()
                );
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(poll_secs));
        }
    }

    /// Bulk-ingest via GCS `contentsDeltaUri` (#187). Serializes every datapoint
    /// to a single JSONL object under `gs://<bucket>/<folder>/`, then issues
    /// UpdateIndex (masking the whole `metadata`, see `build_batch_update_body`)
    /// pointing at that folder and blocks on the returned LRO while Vertex
    /// rebuilds the index.
    ///
    /// LIVE-VALIDATED (project redislabs-cto, 2026-07-21): a 1000×8d batch of
    /// this exact JSONL + request shape ingested to `vectorsCount=1000` in ~1min
    /// on a fresh BATCH_UPDATE index. The batch-file schema (`id`/`embedding`,
    /// `restricts[].allow`, `numeric_restricts[].value_int`/`value_float`) parses
    /// cleanly and the folder-URI layout is correct. The one correction the live
    /// run forced is captured in `build_batch_update_body` (mask whole `metadata`,
    /// not `metadata.contentsDeltaUri`). Query-time restriction matching over a
    /// deployed endpoint is not asserted here (endpoint deploy is billable), but
    /// the restrictions ingested without a parse error.
    fn batch_upload(
        &self,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
        schema: &HashMap<String, String>,
    ) -> Result<(), String> {
        let bucket = self
            .gcs_staging_bucket
            .as_deref()
            .ok_or("batch_upload called without VERTEX_GCS_STAGING_BUCKET")?;

        let jsonl = build_batch_datapoint_jsonl(ids, vectors, metadata, schema);

        // Vertex reads ALL files directly under the folder named in
        // contentsDeltaUri, so stage the single object one level below the
        // folder we point the index at. The folder is namespaced by index
        // display name so concurrent runs don't clobber each other's staging.
        let folder = format!("vdbb-batch/{}", self.display_name);
        let object = format!("{}/datapoints.json", folder);
        let contents_delta_uri = format!("gs://{}/{}", bucket, folder);

        // 1. Stage the JSONL to GCS (media upload). reqwest URL-encodes the
        //    object name in the query string, which GCS requires for names
        //    containing slashes.
        println!(
            "Staging {} datapoints to gs://{}/{} ({:.1} MiB)...",
            ids.len(),
            bucket,
            object,
            jsonl.len() as f64 / (1024.0 * 1024.0)
        );
        // Stage the JSONL to GCS via `gsutil cp` (resumable + parallel-composite)
        // rather than a single in-memory `uploadType=media` POST. Simple media
        // upload buffers the whole body and runs under the client timeout, so a
        // multi-GiB batch (e.g. 1M x 2048-d ~= 41 GiB) always fails with
        // "error sending request". gsutil chunks and resumes.
        let local_path = format!("/tmp/vdbb-batch-{}.json", self.display_name);
        std::fs::write(&local_path, jsonl.as_bytes())
            .map_err(|e| format!("failed writing local staging file {local_path}: {e}"))?;
        let gcs_dest = format!("gs://{}/{}", bucket, object);
        let gs_status = std::process::Command::new("gsutil")
            .args(["-q", "cp", local_path.as_str(), gcs_dest.as_str()])
            .status()
            .map_err(|e| format!("failed to spawn gsutil for staging: {e}"))?;
        let _ = std::fs::remove_file(&local_path);
        if !gs_status.success() {
            return Err(format!(
                "gsutil cp of staging JSONL to {gcs_dest} failed: {gs_status}"
            ));
        }

        // 2. Point the index at the staged folder and trigger the rebuild.
        //    contentsDeltaUri sits inside the Struct-typed `metadata` field, so
        //    the PATCH must mask the WHOLE `metadata` and resend the existing
        //    `config` (see build_batch_update_body). Fetch the current config
        //    from the index describe so we preserve it exactly (this also works
        //    in reuse-index mode, where we never built the config locally).
        let token = self.access_token()?;
        let describe =
            self.get_json(&format!("{}/{}", self.base_url(), self.index_name), &token)?;
        let config = describe
            .get("metadata")
            .and_then(|m| m.get("config"))
            .cloned()
            .ok_or("index describe missing metadata.config; cannot preserve it on batch update")?;

        println!("Updating index with contentsDeltaUri={contents_delta_uri}...");
        let update_url = format!(
            "{}/{}?updateMask=metadata",
            self.base_url(),
            self.index_name
        );
        let token = self.access_token()?;
        let body = build_batch_update_body(&config, &contents_delta_uri);
        let resp = self
            .client
            .patch(&update_url)
            .bearer_auth(&token)
            .json(&body)
            .send()
            .map_err(|e| format!("UpdateIndex (batch ingest) failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(format!(
                "UpdateIndex (batch ingest) returned {status}: {text}"
            ));
        }
        let op: serde_json::Value = resp
            .json()
            .map_err(|e| format!("UpdateIndex response was not JSON: {e}"))?;
        let op_name = op
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or("UpdateIndex (batch ingest) returned no operation name")?
            .to_string();

        // 3. Block on the rebuild LRO (reuses the deploy-timeout budget — a
        //    full batch rebuild can take a long time on large corpora).
        self.poll_operation(&op_name, self.deploy_timeout, "batch index ingest")?;
        Ok(())
    }
}

impl Engine for VertexEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        self.distance_measure = vertex_distance_measure(dataset.distance()).to_string();
        let token = self.access_token()?;

        // Reuse path: skip the slow create+deploy when the caller points at an
        // already-deployed index.
        if let Some((index, endpoint, deployed)) = reuse_index_ids() {
            self.index_name = index;
            self.index_endpoint_name = endpoint;
            self.deployed_index_id = deployed;
            println!("Reusing deployed Vertex index {}", self.index_name);
        } else {
            // 1. Create the index.
            println!(
                "Creating Vertex index (dim={}, {}, {})...",
                dataset.vector_size(),
                self.distance_measure,
                if self.batch_mode() {
                    "BATCH_UPDATE"
                } else {
                    "STREAM_UPDATE"
                }
            );
            // BATCH_UPDATE when a GCS staging bucket is set (#187); the index
            // starts empty and upload() ingests via contentsDeltaUri. Otherwise
            // the default STREAM_UPDATE index, populated by upsertDatapoints.
            let index_body = if self.batch_mode() {
                build_batch_index_body(
                    &self.display_name,
                    dataset.vector_size(),
                    &self.distance_measure,
                    self.approx_neighbors,
                    self.leaf_embedding_count,
                    self.leaf_search_percent,
                    self.shard_size.as_deref(),
                )
            } else {
                build_index_body(
                    &self.display_name,
                    dataset.vector_size(),
                    &self.distance_measure,
                    self.approx_neighbors,
                    self.leaf_embedding_count,
                    self.leaf_search_percent,
                    self.shard_size.as_deref(),
                )
            };
            let op = self.post_json(
                &format!("{}/{}/indexes", self.base_url(), self.parent()),
                &token,
                &index_body,
            )?;
            let op_name = op
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("index create returned no operation name")?
                .to_string();
            let created = self.poll_operation(&op_name, self.deploy_timeout, "index creation")?;
            self.index_name = created
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("index create response missing name")?
                .to_string();

            // 2. Create the public index endpoint.
            println!("Creating Vertex index endpoint...");
            let ep_body = serde_json::json!({
                "displayName": self.display_name,
                "publicEndpointEnabled": true,
            });
            let op = self.post_json(
                &format!("{}/{}/indexEndpoints", self.base_url(), self.parent()),
                &token,
                &ep_body,
            )?;
            let op_name = op
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("endpoint create returned no operation name")?
                .to_string();
            let created =
                self.poll_operation(&op_name, self.deploy_timeout, "endpoint creation")?;
            self.index_endpoint_name = created
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("endpoint create response missing name")?
                .to_string();

            // 3. Deploy the index (SLOW).
            self.deployed_index_id = format!("{}_deployed", self.display_name);
            println!(
                "Deploying index to endpoint (id={}, machine={})...",
                self.deployed_index_id, self.machine_type
            );
            let deploy_body = build_deploy_body(
                &self.deployed_index_id,
                &self.index_name,
                &self.machine_type,
            );
            let op = self.post_json(
                &format!(
                    "{}/{}:deployIndex",
                    self.base_url(),
                    self.index_endpoint_name
                ),
                &token,
                &deploy_body,
            )?;
            let op_name = op
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("deployIndex returned no operation name")?
                .to_string();
            self.poll_operation(&op_name, self.deploy_timeout, "index deployment")?;
        }

        // Resolve the public endpoint domain used by REST/public gRPC. Private
        // gRPC connects directly to the PSC/VPC match address instead.
        let token = self.access_token()?;
        let ep = self.get_json(
            &format!("{}/{}", self.base_url(), self.index_endpoint_name),
            &token,
        )?;
        if self.query_transport == VertexQueryTransport::PrivateGrpc {
            let address = self
                .grpc_address
                .as_deref()
                .ok_or("VERTEX_GRPC_ADDRESS is required for private-grpc transport")?;
            println!("Vertex private gRPC endpoint ready at {address}");
        } else {
            self.public_endpoint_domain = ep
                .get("publicEndpointDomainName")
                .and_then(|d| d.as_str())
                .ok_or(
                    "index endpoint has no publicEndpointDomainName (is a public endpoint enabled?)",
                )?
                .to_string();
            println!(
                "Vertex {:?} endpoint ready at {}",
                self.query_transport, self.public_endpoint_domain
            );
        }
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
            read_time
        );

        let schema = schema_type_map(dataset);

        // Batch (bulk) ingest path (#187): stage all datapoints to GCS and let
        // Vertex rebuild the index from them, instead of streaming millions of
        // upsertDatapoints RPCs through the per-project write quota. Only taken
        // when a staging bucket is configured (the index was then created
        // BATCH_UPDATE, which rejects upsertDatapoints).
        if self.batch_mode() {
            if !should_batch_ingest(
                self.gcs_staging_bucket.as_deref(),
                ids.len(),
                self.batch_threshold,
            ) {
                println!(
                    "NOTE: VERTEX_GCS_STAGING_BUCKET is set but {} vectors is below the batch threshold ({}); \
                     batching anyway because the index is BATCH_UPDATE (streaming upsert is unavailable). \
                     Unset the bucket for streaming ingest.",
                    ids.len(),
                    self.batch_threshold
                );
            }
            let upload_start = Instant::now();
            self.batch_upload(&ids, &vectors, &metadata, &schema)?;
            let upload_time = upload_start.elapsed().as_secs_f64();
            println!(
                "Batch upload time: {:.3}s ({:.0} records/sec)",
                upload_time,
                vectors.len() as f64 / upload_time.max(f64::EPSILON)
            );
            return Ok(UploadStats {
                upload_time,
                total_time: read_time + upload_time,
                upload_count: vectors.len(),
                // Batch ingest is a single index-rebuild operation, not a
                // parallel-worker stream, so parallel/batch_size describe the
                // one staged file rather than per-request fan-out.
                parallel: 1,
                batch_size: vectors.len(),
                memory_usage: None,
            });
        }

        let url = format!("{}/{}:upsertDatapoints", self.base_url(), self.index_name);
        let batch_size = self.batch_size.max(1);
        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(batch_size)
            .map(|s| (s, (s + batch_size).min(ids.len())))
            .collect();
        // Honest parallelism: never spin up more workers than batches.
        let workers = self.parallel.max(1).min(batches.len().max(1));

        let pb = self.create_progress_bar(ids.len());
        let upload_start = Instant::now();

        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let engine = &*self;
        let ids = &ids;
        let vectors = &vectors;
        let metadata = &metadata;
        let schema = &schema;
        let batches = &batches;
        let url = url.as_str();

        std::thread::scope(|s| {
            for _ in 0..workers {
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;
                s.spawn(move || {
                    let client = match reqwest::blocking::Client::builder()
                        .timeout(Duration::from_secs(300))
                        .build()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };
                    // Each worker holds its own token and refreshes it before the
                    // ~60-min GCP token lifetime elapses, so a long streaming
                    // upload can't die on a 401 mid-flight.
                    let (mut token, mut token_at) = match engine.access_token() {
                        Ok(t) => (t, Instant::now()),
                        Err(e) => {
                            *error.lock().unwrap() = Some(e);
                            return;
                        }
                    };
                    loop {
                        let idx = batch_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= batches.len() || error.lock().unwrap().is_some() {
                            break;
                        }
                        if token_at.elapsed() > TOKEN_REFRESH_AFTER {
                            match engine.access_token() {
                                Ok(t) => {
                                    token = t;
                                    token_at = Instant::now();
                                }
                                Err(e) => {
                                    *error.lock().unwrap() = Some(e);
                                    break;
                                }
                            }
                        }
                        let (start, end) = batches[idx];
                        let body = build_upsert_body(
                            &ids[start..end],
                            &vectors[start..end],
                            &metadata[start..end],
                            schema,
                        );
                        let mut quota_retries = 0usize;
                        let mut auth_retries = 0usize;
                        loop {
                            match client.post(url).bearer_auth(&token).json(&body).send() {
                                Ok(r) if r.status().is_success() => break,
                                // A transient 401 (e.g. a token that expired or was
                                // invalidated early) must NOT abort the whole upload
                                // and leave a partially-populated index (#151):
                                // re-mint the token and retry the (idempotent) batch.
                                Ok(r)
                                    if r.status() == reqwest::StatusCode::UNAUTHORIZED
                                        && auth_retries < 5 =>
                                {
                                    auth_retries += 1;
                                    match engine.access_token() {
                                        Ok(t) => {
                                            token = t;
                                            token_at = Instant::now();
                                        }
                                        Err(e) => {
                                            *error.lock().unwrap() = Some(e);
                                            break;
                                        }
                                    }
                                    eprintln!(
                                        "Vertex upsert got 401; refreshed token, retrying batch {} (attempt {}/5)",
                                        idx, auth_retries
                                    );
                                }
                                Ok(r)
                                    if r.status() == reqwest::StatusCode::TOO_MANY_REQUESTS
                                        && quota_retries < 30 =>
                                {
                                    quota_retries += 1;
                                    let retry_after = r
                                        .headers()
                                        .get(reqwest::header::RETRY_AFTER)
                                        .and_then(|value| value.to_str().ok())
                                        .and_then(|value| value.parse::<u64>().ok())
                                        .unwrap_or(60);
                                    eprintln!(
                                        "Vertex stream-update quota reached; retrying batch {} in {}s (attempt {}/30)",
                                        idx, retry_after, quota_retries
                                    );
                                    std::thread::sleep(Duration::from_secs(retry_after));
                                    if token_at.elapsed() > TOKEN_REFRESH_AFTER {
                                        match engine.access_token() {
                                            Ok(t) => {
                                                token = t;
                                                token_at = Instant::now();
                                            }
                                            Err(e) => {
                                                *error.lock().unwrap() = Some(e);
                                                break;
                                            }
                                        }
                                    }
                                }
                                Ok(r) => {
                                    *error.lock().unwrap() = Some(format!(
                                        "{}: {}",
                                        r.status(),
                                        r.text().unwrap_or_default()
                                    ));
                                    break;
                                }
                                Err(e) => {
                                    *error.lock().unwrap() = Some(e.to_string());
                                    break;
                                }
                            }
                        }
                        if error.lock().unwrap().is_some() {
                            break;
                        }
                        pb.inc((end - start) as u64);
                    }
                });
            }
        });
        pb.finish_and_clear();

        if let Some(e) = error.lock().unwrap().take() {
            return Err(format!("upsertDatapoints failed: {}", e));
        }

        let upload_time = upload_start.elapsed().as_secs_f64();
        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        // The upsert RPCs have all returned, but a STREAM_UPDATE index syncs the
        // datapoints ASYNCHRONOUSLY (#151 item 7). Searching now would race the
        // index and report artificially low recall, so block until the index
        // reports it has ingested our vectors (or the timeout fires). This wait
        // is measured and reported SEPARATELY — it is deliberately NOT folded
        // into upload_time, which stays pure upsert wall-clock. The helper
        // returns Ok even on timeout/soft failure, so `?` only propagates true
        // hard errors (of which there are none today).
        let sync_start = Instant::now();
        self.wait_for_index_sync(vectors.len())?;
        let sync_time = sync_start.elapsed().as_secs_f64();
        println!(
            "Index sync wait: {:.3}s (not counted in upload_time)",
            sync_time
        );

        Ok(UploadStats {
            upload_time,
            total_time: read_time + upload_time,
            upload_count: vectors.len(),
            parallel: workers,
            batch_size: self.batch_size,
            memory_usage: None,
        })
    }

    fn search(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        self.distance_measure = vertex_distance_measure(dataset.distance()).to_string();

        if self.index_name.is_empty()
            || self.index_endpoint_name.is_empty()
            || self.deployed_index_id.is_empty()
        {
            let (index, endpoint, deployed) = reuse_index_ids().ok_or(
                "Vertex search with --skip-upload requires VERTEX_INDEX, VERTEX_INDEX_ENDPOINT, and VERTEX_DEPLOYED_INDEX_ID",
            )?;
            self.index_name = index;
            self.index_endpoint_name = endpoint;
            self.deployed_index_id = deployed;
        }

        if self.public_endpoint_domain.is_empty()
            && self.query_transport != VertexQueryTransport::PrivateGrpc
        {
            let token = self.access_token()?;
            let ep = self.get_json(
                &format!("{}/{}", self.base_url(), self.index_endpoint_name),
                &token,
            )?;
            self.public_endpoint_domain = ep
                .get("publicEndpointDomainName")
                .and_then(|d| d.as_str())
                .ok_or("index endpoint has no publicEndpointDomainName")?
                .to_string();
        }

        let parallel = params.parallel.unwrap_or(1).max(1) as usize;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;
        if queries.is_empty() {
            return Err("dataset contains no search queries".to_string());
        }

        // Parse per-query filter conditions up front (outside the timed window).
        // A condition Vertex cannot express is a hard error rather than a silent
        // partial filter (which would inflate recall against filtered ground
        // truth).
        let schema = schema_type_map(dataset);
        let parsed_filters: Vec<Option<VertexFilter>> = conditions
            .iter()
            .map(|c| {
                c.as_ref()
                    .map(|v| parse_vertex_filter(v, &schema))
                    .transpose()
            })
            .collect::<Result<_, _>>()?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let open_loop = OpenLoopPlan::from_params(params)?;
        let closed_loop_duration = closed_loop_duration(params)?;
        let num_to_run = if closed_loop_duration.is_some() {
            usize::MAX
        } else {
            open_loop.map(|p| p.total_requests).unwrap_or_else(|| {
                if num_queries > 0 {
                    (num_queries as usize).min(queries.len())
                } else {
                    queries.len()
                }
            })
        };
        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));

        if neighbors.len() < queries.len() {
            return Err(format!(
                "dataset misaligned: {} neighbor lists for {} dataset queries",
                neighbors.len(),
                queries.len()
            ));
        }

        // Resolve the effective num_candidates + leaf-fraction, never sending the
        // silent "use index default" sentinel (#200). approximate_neighbor_count
        // is always Some now, so request() never falls back to 0.
        let (approx_count, fraction_leaf_override) = self.resolve_search_knobs(params, top);
        let approximate_neighbor_count = Some(approx_count);

        if let Some(duration) = closed_loop_duration {
            println!(
                "\tRunning unrestricted closed-loop for {:.1}s (top={}, parallel={})...",
                duration.as_secs_f64(),
                top,
                parallel
            );
        } else {
            println!(
                "\tRunning {} queries (top={}, parallel={})...",
                HumanCount(num_to_run as u64),
                top,
                parallel
            );
        }

        // One access token for the whole timed region (avoids a gcloud shell-out
        // in the hot loop). Each worker builds its own blocking client so no
        // connection pool is shared across threads.
        let token = self.access_token()?;
        let deployed_index_id = self.deployed_index_id.as_str();
        let query_transport = self.query_transport;
        let public_endpoint_domain = self.public_endpoint_domain.as_str();
        let private_address = self.grpc_address.as_deref();
        let index_endpoint_name = self.index_endpoint_name.as_str();

        let workers = parallel.min(num_to_run.max(1));
        let query_idx = Arc::new(AtomicUsize::new(0));
        let pb = self.create_progress_bar(if closed_loop_duration.is_some() {
            0
        } else {
            num_to_run
        });
        let closed_loop_start = Instant::now();
        let open_loop_start = Arc::new(OnceLock::<Instant>::new());
        let worker_ready = Arc::new(Barrier::new(workers + 1));

        let queries = &queries;
        let neighbors = &neighbors;
        let parsed_filters = &parsed_filters;
        let token = token.as_str();

        let sample_capacity = if closed_loop_duration.is_some() {
            queries.len()
        } else {
            num_to_run
        };
        let mut latencies: Vec<f64> = Vec::with_capacity(sample_capacity);
        let mut precisions: Vec<f64> = Vec::with_capacity(sample_capacity);
        let mut recalls: Vec<f64> = Vec::with_capacity(sample_capacity);
        let mut mrrs: Vec<f64> = Vec::with_capacity(sample_capacity);
        let mut ndcgs: Vec<f64> = Vec::with_capacity(sample_capacity);
        let mut schedule_delays: Vec<f64> = Vec::new();
        let mut end_to_end_latencies: Vec<f64> = Vec::new();
        let mut dropped_queries = 0usize;
        let mut late_queries = 0usize;

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(workers);
            for worker_id in 0..workers {
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;
                let open_loop_start = Arc::clone(&open_loop_start);
                let worker_ready = Arc::clone(&worker_ready);
                handles.push(s.spawn(move || {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();
                    let mut sd = Vec::new();
                    let mut e2e = Vec::new();
                    let mut dropped = 0usize;
                    let mut late = 0usize;
                    let mut pb_pending: u64 = 0;

                    let mut client = match VertexWorker::new(VertexWorkerConfig {
                        transport: query_transport,
                        public_domain: public_endpoint_domain,
                        private_address,
                        token,
                        index_endpoint: index_endpoint_name,
                        deployed_index_id,
                    }) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("Vertex worker client build failed: {}", e);
                            // Always release the barrier (even on failure) so the
                            // main thread isn't left blocked — the barrier and
                            // connection-prime below are now unconditional, not
                            // gated on timed_mode.
                            worker_ready.wait();
                            return (t, p, r, mr, nd, sd, e2e, dropped, late);
                        }
                    };

                    // Prime the connection with one discarded request, then wait at
                    // the barrier, in EVERY mode (not just timed_mode). This keeps
                    // the per-worker gRPC/TLS handshake + cold first RPC OUT of the
                    // measured window: the main thread stamps the measurement start
                    // only after all workers have connected. Previously the default
                    // closed-loop path skipped this, so its rps denominator included
                    // connection setup and understated QPS (#151-audit).
                    {
                        let prime_pos = worker_id % queries.len();
                        let prime_request = client.request(
                            deployed_index_id,
                            &queries[prime_pos],
                            top,
                            fraction_leaf_override,
                            approximate_neighbor_count,
                            parsed_filters[prime_pos].as_ref(),
                        );
                        if let Err(e) = client.execute(prime_request) {
                            eprintln!("Vertex worker connection prime failed: {}", e);
                        }
                        worker_ready.wait();
                    }

                    // Measurement start is always set by the main thread after the
                    // barrier; spin until it is visible so no measured request
                    // predates the timer.
                    let schedule_start = loop {
                        if let Some(start) = open_loop_start.get() {
                            break *start;
                        }
                        std::thread::yield_now();
                    };

                    loop {
                        if closed_loop_duration
                            .map(|duration| Instant::now() >= schedule_start + duration)
                            .unwrap_or(false)
                        {
                            break;
                        }
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }
                        let query_pos = idx % queries.len();
                        let request = client.request(
                            deployed_index_id,
                            &queries[query_pos],
                            top,
                            fraction_leaf_override,
                            approximate_neighbor_count,
                            parsed_filters[query_pos].as_ref(),
                        );

                        let scheduled_at = open_loop.map(|plan| {
                            let delay = plan.wait_for_slot(schedule_start, idx);
                            let will_drop = delay > plan.max_lateness;
                            // Count schedule-delay + lateness for DISPATCHED
                            // requests only. A dropped (never-run) request must not
                            // push its delay into schedule_delays nor increment
                            // `late`, or late_queries >= dropped_queries always and
                            // the schedule-delay percentiles are inflated by
                            // requests that never dispatched. It is accounted
                            // solely as `dropped` below.
                            if !will_drop {
                                sd.push(delay.as_secs_f64());
                                if plan.is_late(delay) {
                                    late += 1;
                                }
                            }
                            (plan.scheduled_at(schedule_start, idx), will_drop)
                        });
                        if scheduled_at.map(|(_, drop)| drop).unwrap_or(false) {
                            dropped += 1;
                            pb_pending += 1;
                            if pb_pending >= 256 {
                                pb.inc(pb_pending);
                                pb_pending = 0;
                            }
                            continue;
                        }

                        // Timed window: RPC round-trip + reply parse only. The
                        // request body is built above (client-side work), matching
                        // the other engines' boundary.
                        let start = Instant::now();
                        let outcome = client.execute(request);
                        let elapsed = start.elapsed().as_secs_f64();
                        let completion = Instant::now();

                        match outcome {
                            Ok(result_ids) => {
                                let m = crate::metrics::compute_metrics(
                                    &result_ids,
                                    &neighbors[query_pos],
                                    top,
                                );
                                t.push(elapsed);
                                p.push(m.precision);
                                r.push(m.recall);
                                mr.push(m.mrr);
                                nd.push(m.ndcg);
                                if let Some((scheduled, _)) = scheduled_at {
                                    e2e.push(
                                        completion
                                            .saturating_duration_since(scheduled)
                                            .as_secs_f64(),
                                    );
                                }
                            }
                            Err(e) => eprintln!("Search query {} failed: {}", idx, e),
                        }
                        pb_pending += 1;
                        if pb_pending >= 256 {
                            pb.inc(pb_pending);
                            pb_pending = 0;
                        }
                    }
                    if pb_pending > 0 {
                        pb.inc(pb_pending);
                    }
                    (t, p, r, mr, nd, sd, e2e, dropped, late)
                }));
            }
            {
                // Wait for every worker to finish connecting (barrier), then stamp
                // the measurement start — unconditional now, so the default
                // closed-loop rps denominator excludes connection setup, matching
                // the open-loop/duration path.
                worker_ready.wait();
                let measurement_start = if open_loop.is_some() {
                    Instant::now() + Duration::from_millis(100)
                } else {
                    Instant::now()
                };
                let _ = open_loop_start.set(measurement_start);
            }
            for h in handles {
                let (t, p, r, mr, nd, sd, e2e, dropped, late) = h.join().unwrap();
                latencies.extend(t);
                precisions.extend(p);
                recalls.extend(r);
                mrrs.extend(mr);
                ndcgs.extend(nd);
                schedule_delays.extend(sd);
                end_to_end_latencies.extend(e2e);
                dropped_queries += dropped;
                late_queries += late;
            }
        });

        pb.finish_and_clear();
        let total_start = open_loop_start.get().copied().unwrap_or(closed_loop_start);
        let total_time = total_start.elapsed().as_secs_f64();

        let succeeded = latencies.len();
        let attempted_queries = query_idx
            .load(Ordering::Relaxed)
            .min(num_to_run)
            .saturating_sub(dropped_queries);
        if succeeded == 0 {
            // Genuinely zero attempts is an error; an overload-shed / all-failed
            // run that DID attempt work is a real data point — report rps=0 with
            // the drop accounting instead of erroring.
            if attempted_queries == 0 && dropped_queries == 0 {
                return Err("No searches completed (all queries failed)".to_string());
            }
            let mut results = zero_search_results(total_time, top, parallel, attempted_queries);
            if let Some(plan) = open_loop {
                attach_open_loop_metrics(
                    &mut results,
                    plan,
                    &schedule_delays,
                    &end_to_end_latencies,
                    dropped_queries,
                    late_queries,
                );
            }
            return Ok(results);
        }
        let failed = attempted_queries.saturating_sub(succeeded);
        if failed > 0 {
            eprintln!(
                "WARNING: {} of {} attempted queries failed",
                failed, attempted_queries
            );
        }

        let mut results = crate::engine::compute_search_stats(
            &latencies,
            &precisions,
            &recalls,
            &mrrs,
            &ndcgs,
            total_time,
            top,
            parallel,
            attempted_queries,
        )?;
        if let Some(plan) = open_loop {
            attach_open_loop_metrics(
                &mut results,
                plan,
                &schedule_delays,
                &end_to_end_latencies,
                dropped_queries,
                late_queries,
            );
        }
        Ok(results)
    }

    fn search_mixed(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
        ratio: &UpdateSearchRatio,
    ) -> Result<SearchResults, String> {
        self.distance_measure = vertex_distance_measure(dataset.distance()).to_string();

        // Resolve the deployed index (reuse env on --skip-upload) + public domain.
        if self.index_name.is_empty()
            || self.index_endpoint_name.is_empty()
            || self.deployed_index_id.is_empty()
        {
            let (index, endpoint, deployed) = reuse_index_ids().ok_or(
                "Vertex mixed benchmark needs a deployed index (VERTEX_INDEX, VERTEX_INDEX_ENDPOINT, VERTEX_DEPLOYED_INDEX_ID)",
            )?;
            self.index_name = index;
            self.index_endpoint_name = endpoint;
            self.deployed_index_id = deployed;
        }
        if self.public_endpoint_domain.is_empty()
            && self.query_transport != VertexQueryTransport::PrivateGrpc
        {
            let token = self.access_token()?;
            let ep = self.get_json(
                &format!("{}/{}", self.base_url(), self.index_endpoint_name),
                &token,
            )?;
            self.public_endpoint_domain = ep
                .get("publicEndpointDomainName")
                .and_then(|d| d.as_str())
                .ok_or("index endpoint has no publicEndpointDomainName")?
                .to_string();
        }

        let parallel = params.parallel.unwrap_or(1).max(1) as usize;
        let schema = schema_type_map(dataset);

        let (queries, neighbors, conditions) = dataset.read_queries()?;
        if queries.is_empty() {
            return Err("dataset contains no search queries".to_string());
        }
        let parsed_filters: Vec<Option<VertexFilter>> = conditions
            .iter()
            .map(|c| {
                c.as_ref()
                    .map(|v| parse_vertex_filter(v, &schema))
                    .transpose()
            })
            .collect::<Result<_, _>>()?;

        // Vectors + metadata for the update half (deterministic shuffled order,
        // so re-upserts hit the same datapoints run-to-run).
        let normalize = dataset.needs_normalization();
        let (upd_ids, upd_vectors, upd_metadata) = dataset.read_vectors(normalize)?;
        if upd_ids.is_empty() {
            return Err("dataset has no vectors for the update half".to_string());
        }
        let mut update_seq: Vec<usize> = (0..upd_ids.len()).collect();
        update_seq.shuffle(&mut rand::rngs::StdRng::seed_from_u64(42));

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };
        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        if neighbors.len() < queries.len() {
            return Err(format!(
                "dataset misaligned: {} neighbor lists for {} queries",
                neighbors.len(),
                queries.len()
            ));
        }
        // Effective knobs, explicit (never the silent index-default sentinel) — #200.
        let (approx_count, fraction_leaf_override) = self.resolve_search_knobs(params, top);
        let approximate_neighbor_count = Some(approx_count);

        let ratio_searches = ratio.searches.max(1) as usize;
        let ratio_updates = ratio.updates as usize;
        let update_seq_len = update_seq.len();

        println!(
            "\tRunning mixed {}:{} (updates:searches) — {} searches, top={}, parallel={}...",
            ratio.updates, ratio.searches, num_to_run, top, parallel
        );

        // One access token for the whole timed region (as in search()). gRPC
        // query workers embed it at construction; the REST update client re-uses
        // it. Mixed runs are bounded by `num_to_run` searches.
        let token = self.access_token()?;
        let deployed_index_id = self.deployed_index_id.as_str();
        let query_transport = self.query_transport;
        let public_endpoint_domain = self.public_endpoint_domain.as_str();
        let private_address = self.grpc_address.as_deref();
        let index_endpoint_name = self.index_endpoint_name.as_str();
        let upsert_url = format!("{}/{}:upsertDatapoints", self.base_url(), self.index_name);

        let workers = parallel.min(num_to_run.max(1));
        let search_idx = Arc::new(AtomicUsize::new(0));
        let update_idx = Arc::new(AtomicUsize::new(0));
        let pb = self.create_progress_bar(num_to_run);
        // Fallback start; the measured window actually begins at `measured_start`,
        // stamped after every worker has connected (below), so the rps denominator
        // excludes per-worker gRPC/TLS connection setup (#151-audit).
        let start_time = Instant::now();
        let worker_ready = Arc::new(Barrier::new(workers + 1));
        let measured_start = Arc::new(OnceLock::<Instant>::new());

        let queries = &queries;
        let neighbors = &neighbors;
        let parsed_filters = &parsed_filters;
        let upd_ids = &upd_ids;
        let upd_vectors = &upd_vectors;
        let upd_metadata = &upd_metadata;
        let update_seq = &update_seq;
        let schema = &schema;
        let token = token.as_str();
        let upsert_url = upsert_url.as_str();

        let mut latencies: Vec<f64> = Vec::new();
        let mut precisions: Vec<f64> = Vec::new();
        let mut recalls: Vec<f64> = Vec::new();
        let mut mrrs: Vec<f64> = Vec::new();
        let mut ndcgs: Vec<f64> = Vec::new();
        let mut update_times: Vec<f64> = Vec::new();

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let search_idx = Arc::clone(&search_idx);
                let update_idx = Arc::clone(&update_idx);
                let worker_ready = Arc::clone(&worker_ready);
                let measured_start = Arc::clone(&measured_start);
                let pb = &pb;
                handles.push(s.spawn(move || {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();
                    let mut ut = Vec::new();
                    let mut pb_pending: u64 = 0;

                    let mut client = match VertexWorker::new(VertexWorkerConfig {
                        transport: query_transport,
                        public_domain: public_endpoint_domain,
                        private_address,
                        token,
                        index_endpoint: index_endpoint_name,
                        deployed_index_id,
                    }) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("Vertex mixed worker build failed: {e}");
                            worker_ready.wait();
                            return (t, p, r, mr, nd, ut);
                        }
                    };
                    let update_client = match reqwest::blocking::Client::builder()
                        .timeout(Duration::from_secs(60))
                        .build()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("Vertex mixed update client build failed: {e}");
                            worker_ready.wait();
                            return (t, p, r, mr, nd, ut);
                        }
                    };

                    // Prime the query connection with one discarded search, then
                    // wait at the barrier, so the measured window excludes the
                    // per-worker gRPC/TLS handshake + cold first RPC — both search
                    // rps and update_rps use total_time as denominator. Mirrors the
                    // closed-loop search() path.
                    if !queries.is_empty() {
                        let prime = client.request(
                            deployed_index_id,
                            &queries[0],
                            top,
                            fraction_leaf_override,
                            approximate_neighbor_count,
                            parsed_filters[0].as_ref(),
                        );
                        if let Err(e) = client.execute(prime) {
                            eprintln!("Vertex mixed worker connection prime failed: {e}");
                        }
                    }
                    worker_ready.wait();
                    // Don't issue a measured request before the timer starts.
                    loop {
                        if measured_start.get().is_some() {
                            break;
                        }
                        std::thread::yield_now();
                    }

                    'outer: loop {
                        // Search phase: S searches.
                        for _ in 0..ratio_searches {
                            let idx = search_idx.fetch_add(1, Ordering::Relaxed);
                            if idx >= num_to_run {
                                break 'outer;
                            }
                            let qpos = idx % queries.len();
                            let request = client.request(
                                deployed_index_id,
                                &queries[qpos],
                                top,
                                fraction_leaf_override,
                                approximate_neighbor_count,
                                parsed_filters[qpos].as_ref(),
                            );
                            let start = Instant::now();
                            let outcome = client.execute(request);
                            let elapsed = start.elapsed().as_secs_f64();
                            match outcome {
                                Ok(ids) => {
                                    let m = crate::metrics::compute_metrics(
                                        &ids,
                                        &neighbors[qpos],
                                        top,
                                    );
                                    t.push(elapsed);
                                    p.push(m.precision);
                                    r.push(m.recall);
                                    mr.push(m.mrr);
                                    nd.push(m.ndcg);
                                }
                                Err(e) => eprintln!("Mixed search {} failed: {}", idx, e),
                            }
                            pb_pending += 1;
                            if pb_pending >= 256 {
                                pb.inc(pb_pending);
                                pb_pending = 0;
                            }
                        }
                        // Update phase: U single-datapoint upserts (with restricts).
                        for _ in 0..ratio_updates {
                            let uidx = update_idx.fetch_add(1, Ordering::Relaxed);
                            let dpos = update_seq[uidx % update_seq_len];
                            let body = build_upsert_body(
                                &upd_ids[dpos..dpos + 1],
                                &upd_vectors[dpos..dpos + 1],
                                &upd_metadata[dpos..dpos + 1],
                                schema,
                            );
                            let ustart = Instant::now();
                            match update_client
                                .post(upsert_url)
                                .bearer_auth(token)
                                .json(&body)
                                .send()
                            {
                                Ok(rr) if rr.status().is_success() => {
                                    ut.push(ustart.elapsed().as_secs_f64())
                                }
                                Ok(rr) => eprintln!(
                                    "Mixed update {} failed: {}: {}",
                                    uidx,
                                    rr.status(),
                                    rr.text().unwrap_or_default()
                                ),
                                Err(e) => eprintln!("Mixed update {} failed: {}", uidx, e),
                            }
                        }
                    }
                    if pb_pending > 0 {
                        pb.inc(pb_pending);
                    }
                    (t, p, r, mr, nd, ut)
                }));
            }
            // All workers have connected + primed; stamp the measured start now so
            // total_time excludes connection setup.
            worker_ready.wait();
            let _ = measured_start.set(Instant::now());
            for h in handles {
                let (t, p, r, mr, nd, ut) = h.join().unwrap();
                latencies.extend(t);
                precisions.extend(p);
                recalls.extend(r);
                mrrs.extend(mr);
                ndcgs.extend(nd);
                update_times.extend(ut);
            }
        });
        pb.finish_and_clear();
        let total_time = measured_start
            .get()
            .copied()
            .unwrap_or(start_time)
            .elapsed()
            .as_secs_f64();

        if latencies.is_empty() {
            return Err("No searches completed (all mixed queries failed)".to_string());
        }

        let (update_count, update_rps, update_mean, u50, u95, u99) = if !update_times.is_empty() {
            let rps = update_times.len() as f64 / total_time;
            let mean = update_times.iter().sum::<f64>() / update_times.len() as f64;
            let mut sorted = update_times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (
                Some(update_times.len()),
                Some(rps),
                Some(mean),
                Some(crate::engine::percentile_linear(&sorted, 0.50)),
                Some(crate::engine::percentile_linear(&sorted, 0.95)),
                Some(crate::engine::percentile_linear(&sorted, 0.99)),
            )
        } else {
            (None, None, None, None, None, None)
        };

        let mut results = crate::engine::compute_search_stats(
            &latencies,
            &precisions,
            &recalls,
            &mrrs,
            &ndcgs,
            total_time,
            top,
            parallel,
            num_to_run,
        )?;
        results.update_count = update_count;
        results.update_rps = update_rps;
        results.update_mean_time = update_mean;
        results.update_p50_time = u50;
        results.update_p95_time = u95;
        results.update_p99_time = u99;
        results.update_latencies = Some(update_times);
        results.update_search_ratio = Some(format!("{}:{}", ratio.updates, ratio.searches));
        Ok(results)
    }

    fn delete(&mut self) -> Result<(), String> {
        // Only tear down resources this run created. Use the SAME three-var
        // condition as configure()'s reuse path, so a run that set only
        // VERTEX_INDEX (and therefore created a fresh endpoint+deployment) still
        // cleans those up instead of leaking billable resources.
        if reuse_index_ids().is_some() {
            return Ok(());
        }
        let token = match self.access_token() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Vertex delete: cannot get token: {}", e);
                return Ok(());
            }
        };
        if !self.index_endpoint_name.is_empty() && !self.deployed_index_id.is_empty() {
            let body = serde_json::json!({ "deployedIndexId": self.deployed_index_id });
            let url = format!(
                "{}/{}:undeployIndex",
                self.base_url(),
                self.index_endpoint_name
            );
            if let Ok(op) = self.post_json(&url, &token, &body) {
                if let Some(name) = op.get("name").and_then(|n| n.as_str()) {
                    let _ = self.poll_operation(name, self.deploy_timeout, "undeploy");
                }
            }
        }
        if !self.index_endpoint_name.is_empty() {
            let _ = self
                .client
                .delete(format!("{}/{}", self.base_url(), self.index_endpoint_name))
                .bearer_auth(&token)
                .send();
        }
        if !self.index_name.is_empty() {
            let _ = self
                .client
                .delete(format!("{}/{}", self.base_url(), self.index_name))
                .bearer_auth(&token)
                .send();
        }
        println!("Vertex resources deleted");
        Ok(())
    }

    /// Post-upload footprint telemetry. Vertex is managed, so there is no
    /// client-visible memory figure; instead report the index's stats (vector
    /// count, shard count) from a describe. Best-effort, outside any timed
    /// window; `None` before the index exists or if the describe fails.
    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        if self.index_name.is_empty() {
            return None;
        }
        let token = self.access_token().ok()?;
        let idx = self
            .get_json(&format!("{}/{}", self.base_url(), self.index_name), &token)
            .ok()?;
        let stats = idx
            .get("indexStats")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        Some(serde_json::json!({ "index_stats": stats }))
    }

    /// Reproducibility metadata: the deployment configuration (region, machine
    /// type, shard size, distance measure, tree-AH params) and, once the index
    /// exists, its resource names + a live describe (config + stats + deployment
    /// state). Called before upload (resources still empty → static config only)
    /// and after search. Telemetry only — captured outside every timed window.
    fn server_metadata(&mut self) -> Option<serde_json::Value> {
        let mut meta = serde_json::json!({
            "engine": "vertex_ai_vector_search",
            "project": self.project,
            "region": self.region,
            "machine_type": self.machine_type,
            "shard_size": self.shard_size,
            "distance_measure": self.distance_measure,
            "index_algorithm": "tree_ah",
            "approximate_neighbors_count": self.approx_neighbors,
            "leaf_node_embedding_count": self.leaf_embedding_count,
            "leaf_nodes_to_search_percent": self.leaf_search_percent,
            "display_name": self.display_name,
            "index": self.index_name,
            "index_endpoint": self.index_endpoint_name,
            "deployed_index_id": self.deployed_index_id,
            "public_endpoint_domain": self.public_endpoint_domain,
        });
        // Enrich with a live index describe once the index exists (config +
        // indexStats + deployedIndexes). Best-effort.
        if !self.index_name.is_empty() {
            if let Ok(token) = self.access_token() {
                if let Ok(idx) =
                    self.get_json(&format!("{}/{}", self.base_url(), self.index_name), &token)
                {
                    if let Some(obj) = meta.as_object_mut() {
                        obj.insert("index_resource".to_string(), idx);
                    }
                }
            }
        }
        Some(meta)
    }
}

impl VertexEngine {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Vertex AI is cloud-only (needs a GCP project, a bearer token, and a
    // ~tens-of-minutes index deploy), so there is no live integration test.
    // These pins cover the pure request-body builders and reply parsers.

    #[test]
    fn distance_measure_mapping() {
        assert_eq!(vertex_distance_measure("cosine"), "COSINE_DISTANCE");
        assert_eq!(vertex_distance_measure("angular"), "COSINE_DISTANCE");
        assert_eq!(vertex_distance_measure("l2"), "SQUARED_L2_DISTANCE");
        assert_eq!(vertex_distance_measure("euclidean"), "SQUARED_L2_DISTANCE");
        assert_eq!(vertex_distance_measure("dot"), "DOT_PRODUCT_DISTANCE");
        assert_eq!(vertex_distance_measure("ip"), "DOT_PRODUCT_DISTANCE");
        assert_eq!(vertex_distance_measure("mystery"), "DOT_PRODUCT_DISTANCE");
    }

    #[test]
    fn vectors_count_parses_string_and_number() {
        // GCP's REST convention: int64 fields come back as JSON strings.
        let s = json!({ "indexStats": { "vectorsCount": "10000" } });
        assert_eq!(parse_vectors_count(&s), Some(10_000));
        // Tolerate a bare JSON number too (defensive / mockable).
        let n = json!({ "indexStats": { "vectorsCount": 42 } });
        assert_eq!(parse_vectors_count(&n), Some(42));
    }

    #[test]
    fn vectors_count_missing_or_malformed_is_none() {
        // No indexStats at all (fresh index before first stats refresh).
        assert_eq!(parse_vectors_count(&json!({})), None);
        // indexStats present but no vectorsCount field.
        assert_eq!(parse_vectors_count(&json!({ "indexStats": {} })), None);
        // Unparseable string.
        assert_eq!(
            parse_vectors_count(&json!({ "indexStats": { "vectorsCount": "not-a-number" } })),
            None
        );
        // Wrong JSON type entirely.
        assert_eq!(
            parse_vectors_count(&json!({ "indexStats": { "vectorsCount": true } })),
            None
        );
    }

    #[test]
    fn index_body_is_stream_update_tree_ah() {
        let b = build_index_body("bench", 768, "COSINE_DISTANCE", 150, 500, 7, None);
        assert_eq!(b["indexUpdateMethod"], "STREAM_UPDATE");
        let cfg = &b["metadata"]["config"];
        assert_eq!(cfg["dimensions"], 768);
        assert_eq!(cfg["distanceMeasureType"], "COSINE_DISTANCE");
        assert_eq!(cfg["approximateNeighborsCount"], 150);
        // leafNodeEmbeddingCount is a STRING per the API.
        assert_eq!(
            cfg["algorithmConfig"]["treeAhConfig"]["leafNodeEmbeddingCount"],
            "500"
        );
        assert_eq!(
            cfg["algorithmConfig"]["treeAhConfig"]["leafNodesToSearchPercent"],
            7
        );
        // shardSize omitted when None, set when Some.
        assert!(cfg.get("shardSize").is_none());
        let b2 = build_index_body(
            "bench",
            8,
            "COSINE_DISTANCE",
            150,
            500,
            7,
            Some("SHARD_SIZE_SMALL"),
        );
        assert_eq!(b2["metadata"]["config"]["shardSize"], "SHARD_SIZE_SMALL");
    }

    #[test]
    fn upsert_body_stringifies_ids_and_keeps_vectors() {
        let b = build_upsert_body(
            &[0, 42],
            &[vec![1.0, 2.0], vec![3.0, 4.0]],
            &[None, None],
            &HashMap::new(),
        );
        let dps = b["datapoints"].as_array().unwrap();
        assert_eq!(dps.len(), 2);
        assert_eq!(dps[0]["datapointId"], "0");
        assert_eq!(dps[1]["datapointId"], "42");
        assert_eq!(dps[1]["featureVector"], json!([3.0, 4.0]));
    }

    #[test]
    fn find_neighbors_body_sets_count_and_optional_override() {
        let b = build_find_neighbors_body("dep", &[1.0, 2.0], 10, None, None, None);
        assert_eq!(b["deployedIndexId"], "dep");
        assert_eq!(b["returnFullDatapoint"], false);
        let q = &b["queries"][0];
        assert_eq!(q["neighborCount"], 10);
        assert_eq!(q["datapoint"]["featureVector"], json!([1.0, 2.0]));
        assert!(q.get("fractionLeafNodesToSearchOverride").is_none());

        let b2 = build_find_neighbors_body("dep", &[1.0], 5, Some(0.2), Some(500), None);
        assert_eq!(b2["queries"][0]["fractionLeafNodesToSearchOverride"], 0.2);
    }

    // ── Metadata filters ──────────────────────────────────────────────────

    #[test]
    fn parse_filter_keyword_match_any_becomes_allowlist() {
        let f = parse_vertex_filter(
            &json!({"and": [{"color": {"match": {"any": ["red", "blue"]}}}]}),
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(f.restricts.len(), 1);
        assert_eq!(f.restricts[0].namespace, "color");
        assert_eq!(f.restricts[0].allow_list, vec!["red", "blue"]);
        assert!(f.numeric_restricts.is_empty());
    }

    #[test]
    fn parse_filter_numeric_range_becomes_two_ops() {
        let f = parse_vertex_filter(
            &json!({"and": [{"size": {"range": {"gte": 3, "lte": 7}}}]}),
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(f.numeric_restricts.len(), 2);
        let ops: Vec<_> = f
            .numeric_restricts
            .iter()
            .map(|n| (n.op, n.value))
            .collect();
        assert!(ops.contains(&(Some(NumericOp::GreaterEqual), NumericValue::Int(3))));
        assert!(ops.contains(&(Some(NumericOp::LessEqual), NumericValue::Int(7))));
    }

    #[test]
    fn parse_filter_exact_value_typed_by_json() {
        let sf = parse_vertex_filter(
            &json!({"and": [{"c": {"match": {"value": "x"}}}]}),
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(sf.restricts[0].allow_list, vec!["x"]);
        let nf = parse_vertex_filter(
            &json!({"and": [{"n": {"match": {"value": 5}}}]}),
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(nf.numeric_restricts[0].op, Some(NumericOp::Equal));
        assert_eq!(nf.numeric_restricts[0].value, NumericValue::Int(5));
    }

    #[test]
    fn parse_filter_rejects_unexpressible_shapes() {
        // Cross-field OR, nested boolean, numeric IN-list, and geo are hard errors.
        assert!(parse_vertex_filter(
            &json!({"or": [{"a": {"match": {"value": "x"}}}]}),
            &HashMap::new()
        )
        .is_err());
        assert!(parse_vertex_filter(
            &json!({"and": [{"n": {"match": {"any": [1, 2]}}}]}),
            &HashMap::new()
        )
        .is_err());
        assert!(parse_vertex_filter(
            &json!({"and": [{"loc": {"geo_radius": {"lat": 1.0, "lon": 2.0, "radius": 5.0}}}]}),
            &HashMap::new(),
        )
        .is_err());
    }

    #[test]
    fn schema_forces_numeric_type_consistency() {
        // A `float`-declared field must serialize as Double on BOTH the stored
        // side (even a whole-number value) and the query side (even an integer
        // bound), so Vertex's type-strict numeric compare matches.
        let schema = HashMap::from([("price".to_string(), "float".to_string())]);
        let stored = metadata_to_filter(
            &MetadataItem {
                fields: vec![("price".into(), MetadataValue::Int(3))],
            },
            &schema,
        );
        assert_eq!(stored.numeric_restricts[0].value, NumericValue::Double(3.0));
        let q = parse_vertex_filter(&json!({"and": [{"price": {"range": {"gte": 5}}}]}), &schema)
            .unwrap();
        assert_eq!(q.numeric_restricts[0].value, NumericValue::Double(5.0));
    }

    #[test]
    fn null_conditions_are_no_filter() {
        let f = parse_vertex_filter(&serde_json::Value::Null, &HashMap::new()).unwrap();
        assert!(f.is_empty());
    }

    #[test]
    fn metadata_maps_to_restricts_by_variant() {
        let meta = MetadataItem {
            fields: vec![
                ("color".into(), MetadataValue::String("red".into())),
                (
                    "labels".into(),
                    MetadataValue::Labels(vec!["a".into(), "b".into()]),
                ),
                ("size".into(), MetadataValue::Int(7)),
                ("price".into(), MetadataValue::Float(3.5)),
            ],
        };
        let f = metadata_to_filter(&meta, &HashMap::new());
        assert_eq!(f.restricts.len(), 2); // color + labels
        assert_eq!(f.numeric_restricts.len(), 2); // size + price
                                                  // A stored datapoint value carries no operator.
        assert!(f.numeric_restricts.iter().all(|n| n.op.is_none()));
    }

    #[test]
    fn upsert_body_carries_restricts_from_metadata() {
        let meta = vec![Some(MetadataItem {
            fields: vec![
                ("color".into(), MetadataValue::String("red".into())),
                ("size".into(), MetadataValue::Int(7)),
            ],
        })];
        let b = build_upsert_body(&[0], &[vec![1.0]], &meta, &HashMap::new());
        let dp = &b["datapoints"][0];
        assert_eq!(dp["restricts"][0]["namespace"], "color");
        assert_eq!(dp["restricts"][0]["allowList"][0], "red");
        assert_eq!(dp["numericRestricts"][0]["namespace"], "size");
        assert_eq!(dp["numericRestricts"][0]["valueInt"], 7);
        // Stored value has no op.
        assert!(dp["numericRestricts"][0].get("op").is_none());
    }

    // ---- #187 batch (GCS contentsDeltaUri) ingest -------------------------

    #[test]
    fn batch_index_body_is_batch_update_tree_ah() {
        let b = build_batch_index_body("bench", 768, "COSINE_DISTANCE", 150, 500, 7, None);
        // The ONLY structural difference from the stream body is the method:
        // both are tree-AH with an identical config block.
        assert_eq!(b["indexUpdateMethod"], "BATCH_UPDATE");
        let cfg = &b["metadata"]["config"];
        assert_eq!(cfg["dimensions"], 768);
        assert_eq!(cfg["distanceMeasureType"], "COSINE_DISTANCE");
        assert_eq!(
            cfg["algorithmConfig"]["treeAhConfig"]["leafNodeEmbeddingCount"],
            "500"
        );
        // A freshly-created BATCH_UPDATE index is EMPTY — the delta URI is
        // supplied later by UpdateIndex, never at create time.
        assert!(b["metadata"].get("contentsDeltaUri").is_none());
        // shardSize honored like the stream body.
        let b2 = build_batch_index_body(
            "bench",
            8,
            "COSINE_DISTANCE",
            150,
            500,
            7,
            Some("SHARD_SIZE_SMALL"),
        );
        assert_eq!(b2["metadata"]["config"]["shardSize"], "SHARD_SIZE_SMALL");
    }

    #[test]
    fn batch_update_body_masks_whole_metadata_and_preserves_config() {
        // A field mask does NOT descend into the Struct-typed `metadata`, so the
        // update body must carry the delta URI AND the preserved config together
        // under `metadata` (paired with updateMask=metadata). Masking only
        // metadata.contentsDeltaUri no-ops on live Vertex (ingests nothing).
        let config = json!({
            "dimensions": 8,
            "distanceMeasureType": "COSINE_DISTANCE",
            "algorithmConfig": { "treeAhConfig": { "leafNodeEmbeddingCount": "100" } }
        });
        let b = build_batch_update_body(&config, "gs://my-bucket/vdbb-batch/bench");
        assert_eq!(
            b["metadata"]["contentsDeltaUri"],
            "gs://my-bucket/vdbb-batch/bench"
        );
        // config is resent verbatim so masking `metadata` doesn't wipe it.
        assert_eq!(b["metadata"]["config"], config);
        // isCompleteOverwrite forces a clean rebuild from exactly the staged files.
        assert_eq!(b["metadata"]["isCompleteOverwrite"], true);
        assert!(b.get("displayName").is_none());
    }

    #[test]
    fn batch_jsonl_one_line_per_datapoint_in_batch_schema() {
        let meta = vec![
            Some(MetadataItem {
                fields: vec![
                    ("color".into(), MetadataValue::String("red".into())),
                    ("size".into(), MetadataValue::Int(7)),
                    ("price".into(), MetadataValue::Float(3.5)),
                ],
            }),
            None,
        ];
        let jsonl = build_batch_datapoint_jsonl(
            &[0, 42],
            &[vec![1.0, 2.0], vec![3.0, 4.0]],
            &meta,
            &HashMap::new(),
        );
        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 2);

        let d0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        // Batch schema uses id/embedding (NOT datapointId/featureVector).
        assert_eq!(d0["id"], "0");
        assert_eq!(d0["embedding"], json!([1.0, 2.0]));
        // Categorical uses `allow` (NOT allowList).
        assert_eq!(d0["restricts"][0]["namespace"], "color");
        assert_eq!(d0["restricts"][0]["allow"][0], "red");
        // Numeric uses value_int / value_float (NOT valueInt/valueDouble) and
        // carries no operator on a stored datapoint.
        let nr = d0["numeric_restricts"].as_array().unwrap();
        assert_eq!(nr.len(), 2);
        assert!(nr
            .iter()
            .any(|n| n["namespace"] == "size" && n["value_int"] == 7));
        assert!(nr
            .iter()
            .any(|n| n["namespace"] == "price" && n["value_float"] == 3.5));
        assert!(nr.iter().all(|n| n.get("op").is_none()));

        // A datapoint without metadata omits restriction keys entirely.
        let d1: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(d1["id"], "42");
        assert!(d1.get("restricts").is_none());
        assert!(d1.get("numeric_restricts").is_none());
    }

    #[test]
    fn batch_jsonl_types_numerics_from_schema() {
        // A float-declared field must serialize as value_float even when the
        // stored value is a whole number, so query restrictions of the same
        // field match (Vertex compares numeric restrictions by type).
        let mut schema = HashMap::new();
        schema.insert("year".to_string(), "float".to_string());
        let meta = vec![Some(MetadataItem {
            fields: vec![("year".into(), MetadataValue::Int(2020))],
        })];
        let jsonl = build_batch_datapoint_jsonl(&[0], &[vec![1.0]], &meta, &schema);
        let d: serde_json::Value = serde_json::from_str(jsonl.lines().next().unwrap()).unwrap();
        let nr = &d["numeric_restricts"][0];
        assert_eq!(nr["namespace"], "year");
        assert!(nr.get("value_float").is_some());
        assert!(nr.get("value_int").is_none());
    }

    #[test]
    fn approx_neighbor_count_never_silently_defaults() {
        // Config sets num_candidates → honored, source "config".
        assert_eq!(
            resolve_approx_neighbor_count(Some(1000), 150, 10),
            (1000, "config")
        );
        // Config UNSET → explicit fall back to the index's configured value, NOT
        // the silent 0 sentinel that would make Vertex pick its own default (#200).
        assert_eq!(
            resolve_approx_neighbor_count(None, 150, 10),
            (150, "index-default")
        );
        // Vertex requires approximateNeighborCount >= neighborCount (= top): a
        // config value below top is clamped up to the floor...
        assert_eq!(
            resolve_approx_neighbor_count(Some(5), 150, 100),
            (100, "config")
        );
        // ...and so is a small index default.
        assert_eq!(
            resolve_approx_neighbor_count(None, 50, 100),
            (100, "index-default")
        );
        // The result is NEVER 0 (the "use index default" sentinel) for any input.
        for nc in [None, Some(0), Some(1)] {
            let (v, _) = resolve_approx_neighbor_count(nc, 150, 10);
            assert!(v >= 10, "must be >= top floor, never the 0 sentinel: {v}");
        }
    }

    #[test]
    fn should_batch_ingest_gates_on_bucket_and_threshold() {
        // No bucket → never batch, regardless of size.
        assert!(!should_batch_ingest(None, 10_000_000, 100_000));
        assert!(!should_batch_ingest(Some(""), 10_000_000, 100_000));
        assert!(!should_batch_ingest(Some("   "), 10_000_000, 100_000));
        // Bucket set but below threshold → no.
        assert!(!should_batch_ingest(Some("b"), 99_999, 100_000));
        // Bucket set and at/above threshold → yes.
        assert!(should_batch_ingest(Some("b"), 100_000, 100_000));
        assert!(should_batch_ingest(Some("b"), 5_000_000, 100_000));
    }

    #[test]
    fn find_neighbors_body_carries_query_filter() {
        let filter = VertexFilter {
            restricts: vec![Restrict {
                namespace: "color".into(),
                allow_list: vec!["red".into(), "blue".into()],
            }],
            numeric_restricts: vec![NumericRestrict {
                namespace: "size".into(),
                op: Some(NumericOp::GreaterEqual),
                value: NumericValue::Int(3),
            }],
        };
        let b = build_find_neighbors_body("dep", &[1.0], 10, None, None, Some(&filter));
        let dp = &b["queries"][0]["datapoint"];
        assert_eq!(dp["restricts"][0]["allowList"][1], "blue");
        assert_eq!(dp["numericRestricts"][0]["op"], "GREATER_EQUAL");
        assert_eq!(dp["numericRestricts"][0]["valueInt"], 3);
    }

    #[test]
    fn parse_neighbors_extracts_first_query_ids_in_order() {
        let resp = json!({
            "nearestNeighbors": [{
                "id": "q0",
                "neighbors": [
                    {"datapoint": {"datapointId": "7"}, "distance": 0.1},
                    {"datapoint": {"datapointId": "3"}, "distance": 0.2},
                    {"datapoint": {"datapointId": "not-an-int"}, "distance": 0.3}
                ]
            }]
        });
        assert_eq!(parse_find_neighbors_response(&resp), vec![7, 3]);
    }

    #[test]
    fn parse_neighbors_empty_on_missing_fields() {
        assert_eq!(parse_find_neighbors_response(&json!({})), Vec::<i64>::new());
        assert_eq!(
            parse_find_neighbors_response(&json!({"nearestNeighbors": []})),
            Vec::<i64>::new()
        );
    }

    #[test]
    fn lro_states() {
        // Pending.
        assert!(parse_lro(&json!({"name": "op", "done": false}))
            .unwrap()
            .is_none());
        // Done with a response.
        let done = parse_lro(&json!({"done": true, "response": {"name": "idx"}}))
            .unwrap()
            .unwrap();
        assert_eq!(done["name"], "idx");
        // Done with no response (e.g. delete) -> empty object, still complete.
        assert!(parse_lro(&json!({"done": true})).unwrap().is_some());
        // Error.
        assert!(parse_lro(&json!({"error": {"code": 3, "message": "bad"}})).is_err());
    }

    #[test]
    fn deploy_body_shape() {
        let b = build_deploy_body(
            "dep_id",
            "projects/p/locations/r/indexes/1",
            "e2-standard-16",
        );
        assert_eq!(b["deployedIndex"]["id"], "dep_id");
        assert_eq!(
            b["deployedIndex"]["index"],
            "projects/p/locations/r/indexes/1"
        );
        assert_eq!(
            b["deployedIndex"]["dedicatedResources"]["machineSpec"]["machineType"],
            "e2-standard-16"
        );
        assert_eq!(
            b["deployedIndex"]["dedicatedResources"]["minReplicaCount"],
            1
        );
    }
}
