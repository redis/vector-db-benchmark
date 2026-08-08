//! Milvus engine implementation.
//!
//! Uses the Milvus RESTful API (v2) via reqwest::blocking.
//! Supports HNSW index with configurable M/efConstruction,
//! multiple distance metrics (L2, IP, COSINE), and schema-based collections.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

use super::geo;
use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::parsers::datetime_to_epoch_secs;
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::readers::metadata::{is_multivalued_keyword_field, MetadataItem};
use vector_db_benchmark::start_gate::WorkerPool;

const DEFAULT_COLLECTION: &str = "Benchmark";

/// How one dataset schema field is materialised in Milvus: the column's
/// `dataType`, and the SCALAR INDEX type that column must get.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MilvusFieldKind {
    data_type: &'static str,
    /// Milvus `indexType` for the scalar index on this column, or `None` for a
    /// column that must deliberately stay UNINDEXED. `geo` is the only such
    /// case — see the `"geo"` arm of [`milvus_field_kind`] for why.
    index_type: Option<&'static str>,
}

/// Map a dataset schema field to its Milvus column type AND its scalar index
/// type, or `None` if the field is not materialised as a column at all.
///
/// This is the single source of truth shared by `create_collection` (which
/// builds the columns) and `scalar_index_plan` (which indexes them). Keeping
/// one mapping is the point: issue #218 was exactly a column that existed with
/// no index, so every filtered query degenerated into a brute-force scalar scan
/// behind the ANN search, and no recall check could ever notice (an unindexed
/// scan returns the SAME rows — only slower).
///
/// ## Why these index types (verified live against `milvusdb/milvus:v2.6.19`,
/// the image pinned in `tests/docker-compose.test.yml`)
///
/// Milvus 2.6 offers INVERTED, BITMAP, STL_SORT, Trie and NGRAM for scalar
/// fields. Probing every (column type × index type) pair against v2.6.19 gives:
///
/// | column          | INVERTED | BITMAP | STL_SORT | Trie |
/// |-----------------|----------|--------|----------|------|
/// | VarChar         | ok       | ok     | ok       | ok   |
/// | Int64           | ok       | ok     | ok       | ✗    |
/// | Double          | ok       | ✗      | ok       | ✗    |
/// | Bool            | ok       | ok     | ✗        | ✗    |
/// | Array(VarChar)  | ok       | ok     | ✗        | ✗    |
///
/// (✗ = server rejects with error 1100: BITMAP is "only supported on bool, int,
/// string and array field"; STL_SORT "only on numeric, varchar or timestamptz";
/// Trie "only supported on varchar field".)
///
/// The choice per column is not "whatever is permitted" — it is what MEASURED
/// fastest on this version, and it happens to coincide exactly with what Milvus
/// picks for itself:
///
/// * **INVERTED for VarChar / Int64 / Double / Array** — the only type accepted
///   on all four, and it serves BOTH predicate shapes this benchmark emits
///   (point: `==`, `in [...]`, `array_contains*`, `TEXT_MATCH`; and range:
///   `<`/`>=` over `int`, `float` and the Int64-epoch `datetime` column).
///   BITMAP is the only real alternative on VarChar, and it does NOT beat
///   INVERTED even where it should be strongest: on a 10-distinct-value keyword
///   column (`random-100-match-kw-small-vocab-filters`) the measured BITMAP /
///   INVERTED ratio is ~0.95–1.03, i.e. a coin flip. So INVERTED leaves nothing
///   on the table, while avoiding BITMAP's degradation on the high-cardinality
///   end (`prod_name` in `h-and-m-2048-angular-filters` has tens of thousands of
///   distinct values).
/// * **BITMAP for Bool** — this arm is NECESSARY, not merely permitted.
///   Measured on a Bool column, INVERTED is worth nothing at all: INVERTED /
///   no-index = 0.987–1.017, indistinguishable from having no index. BITMAP
///   delivers a 20–24% improvement. Simplifying to "INVERTED everywhere" would
///   therefore have left `synthetic-filter-32`'s `flag` filter with no benefit
///   whatsoever. STL_SORT is rejected on Bool by the server.
/// * **This mapping is what Milvus's own AUTOINDEX resolves to**, measured by
///   comparing an AUTOINDEX column against explicitly-typed ones: on VarChar
///   `auto / INVERTED = 0.937` (AUTOINDEX behaves as INVERTED); on Bool
///   `auto / BITMAP = 1.009` while `auto / INVERTED = 0.774` (AUTOINDEX behaves
///   as BITMAP, definitively not INVERTED). We are therefore not hand-tuning
///   Milvus past its own defaults — we give each column exactly the index Milvus
///   would choose for itself, which is the fairest possible setting for a
///   competitor in a vendor benchmark.
/// * **Not STL_SORT** for the numerics: it is a sorted-array structure aimed at
///   ranges only, and it cannot cover the Array/Bool columns, so it would mean
///   two structures where INVERTED covers both shapes — and it is not what
///   AUTOINDEX picks.
/// * **Not Trie**: VARCHAR-only and prefix-only. This benchmark never emits a
///   prefix predicate (`like "abc%"`), so Trie would not serve `==` / `in`.
/// * **`text` columns get an INVERTED index on top of `enable_match`.** A
///   `text` field is already created with `enable_analyzer` + `enable_match`
///   (see `create_collection`), which builds Milvus's tokenised match index for
///   `TEXT_MATCH`. The explicit INVERTED index here is therefore a SECOND
///   structure on the same column. It is kept deliberately: `text` columns are
///   also filtered with plain `==` / `in [...]` (nothing stops a dataset from
///   doing so), which the match index does not serve, and skipping it would
///   reintroduce exactly the "column with no scalar index" hole for one field
///   type. The cost is extra index build time and memory on `text` columns
///   only — `h-and-m-2048-angular-filters`'s `detail_desc` is the sole shipped
///   instance.
/// * **Not a bare `create_index` with no `indexType`** (upstream's approach,
///   i.e. AUTOINDEX): the REST `indexes/describe` reply then reports
///   `indexType: ""`, which makes the read-back assertion in
///   `tests/integration_milvus.rs` unable to prove WHICH index exists. Naming
///   the type keeps the choice auditable from the server's own reply.
///
/// `geo` is a native `Geometry` column with an `RTREE` index (Milvus >= 2.6.4,
/// issue #223). Routing both the column pass and the index pass through this one
/// function is what guarantees we never index a column that does not exist, or
/// leave one that does unindexed.
fn milvus_field_kind(field_name: &str, schema_type: &str) -> Option<MilvusFieldKind> {
    // A multi-valued keyword field (`labels`) is an Array of VarChar (#88).
    if (schema_type == "keyword" || schema_type == "text")
        && is_multivalued_keyword_field(field_name)
    {
        return Some(MilvusFieldKind {
            data_type: "Array",
            index_type: Some("INVERTED"),
        });
    }
    let (data_type, index_type) = match schema_type {
        "int" => ("Int64", Some("INVERTED")),
        // A `uuid` is an exact-match opaque string → a plain VarChar (no
        // analyzer), same as keyword.
        "keyword" | "text" | "uuid" => ("VarChar", Some("INVERTED")),
        "float" => ("Double", Some("INVERTED")),
        // Milvus has a native Bool type; 2 distinct values → BITMAP.
        "bool" => ("Bool", Some("BITMAP")),
        // No native date type, so datetimes are stored as Int64 epoch seconds
        // (upload + the range filter both convert via datetime_to_epoch_secs)
        // and take the same INVERTED range index as `int`.
        "datetime" => ("Int64", Some("INVERTED")),
        // Native geospatial column (issue #223). `Geometry` stores OGC WKT and
        // is queried with `ST_DWITHIN(field, 'POINT(lon lat)', metres)`, which
        // for a POINT column against a POINT query is an exact great-circle
        // haversine test on R = 6 371 000 m in the server's own C++
        // (`internal/core/src/common/Geometry.h::dwithin`) — the same earth
        // radius `engine::geo::EARTH_RADIUS_M` uses. Added in Milvus 2.6.4; the
        // pinned test image is v2.6.19.
        //
        // DELIBERATELY UNINDEXED, the one exception to the #218
        // column-implies-index rule. `RTREE` is the only index type Milvus
        // offers for a Geometry column, and it is not merely a *coarse* filter —
        // it prunes with a box that is SMALLER than the cap it is supposed to
        // bound, so true hits are discarded before the exact refine step ever
        // sees them. `GISFunctionFilterExpr.cpp::create_bounding_box_for_dwithin`
        // builds it with
        //
        // ```cpp
        // const double metersPerDegreeLat = 111320.0;
        // double lonOffset = distance_meters / (metersPerDegreeLat * cos(latRad));
        // ```
        //
        // 111 320 is larger than the true 111 194.93 m/degree, so the box is
        // ~0.11 % short in every direction; the flat `theta/cos(phi)` longitude
        // half-width also underestimates at high latitude, and there is no
        // antimeridian wrap at all. Measured on the shipped
        // `random-geo-radius-100-angular-filters` (first 500 queries against the
        // 1M payload set, ground truth from `tests.jsonl`): **806 of 12 500
        // ground-truth neighbours become unreachable (6.4 %)**, 145 of 500
        // queries (29 %) lose at least one, and top-25 recall is capped at
        // ~0.935 before HNSW loses anything — silently. Worst cases land exactly
        // where the geometry predicts: `lat 81.1`, `lat -86.3`, `lon 178.32`.
        //
        // Verified first-hand on v2.6.19, identical rows in two collections
        // differing only by the RTREE: centre (81, 10) with r = 200 km returned
        // 14 of 20 in-cap documents WITH the index (the six at 0.9990-0.9995 *
        // radius due north and south were pruned) and 20 of 20 without it;
        // centre (0, 179.9) returned 4 of 8 with it (every document across +180
        // pruned) and 8 of 8 without. `tests/common/mod.rs::write_geo_edge_project`
        // is that experiment as a fixture.
        //
        // Without the index Milvus scans the column and `ST_DWITHIN` alone is
        // exact. A slower scan is the right trade for a tool whose output is a
        // recall number.
        "geo" => ("Geometry", None),
        _ => return None,
    };
    Some(MilvusFieldKind {
        data_type,
        index_type,
    })
}

/// One stored point as OGC WKT for a `Geometry` column (issue #223).
///
/// WKT is `POINT(x y)`, i.e. **longitude first**. The value this replaced was
/// `"{lat},{lon}"` — the wrong format AND the wrong axis order — and it had no
/// column to land in, because the `geo` schema type materialised nothing.
///
/// Extracted from the insert loop purely so it can be pinned: a lat/lon swap
/// applied to BOTH the storage and the query side is self-consistent and selects
/// the identical documents, so no recall fixture can see it. The query side is
/// pinned by `geo_emits_st_dwithin_with_lon_first_and_metres`; this is the other
/// half. (The edge fixture is a partial backstop — a swapped 179.9 is not a
/// valid latitude — but a pin is cheaper and exact.)
fn geo_wkt_point(lon: f64, lat: f64) -> String {
    format!(
        "POINT({} {})",
        geo::plain_decimal(lon),
        geo::plain_decimal(lat)
    )
}

/// Every `(column, scalar index type)` pair that must exist for this dataset —
/// i.e. every schema field `create_collection` materialised as a column. `id`
/// (the primary key) and `vector` are excluded: the PK is indexed implicitly and
/// `vector` gets the ANN index. Mirrors upstream
/// `engine/clients/milvus/upload.py`, which loops every non-`id`/`vector` field
/// of the collection schema and calls `create_index` on it.
fn scalar_index_plan(dataset: &Dataset) -> Vec<(String, &'static str)> {
    let mut plan = Vec::new();
    if let Some(obj) = dataset.config.schema.as_ref().and_then(|s| s.as_object()) {
        for (field_name, field_type) in obj {
            if field_name == "id" || field_name == "vector" {
                continue;
            }
            // `index_type: None` = materialised but deliberately unindexed
            // (geo — see `milvus_field_kind`), so it is not in the plan and the
            // read-back must not expect an index for it.
            if let Some(index_type) =
                milvus_field_kind(field_name, field_type.as_str().unwrap_or(""))
                    .and_then(|kind| kind.index_type)
            {
                plan.push((field_name.clone(), index_type));
            }
        }
    }
    plan
}

/// True when a failed `indexes/create` response means "this field already has an
/// index" — mirrors upstream tolerating pymilvus error code 1 ("index already
/// exist"). v2.6.19 returns code 0 for a byte-identical re-create, and 1100 with
/// one of these messages when an index already covers the field.
///
/// This says only "an index exists", NOT "the RIGHT index exists": the same 1100
/// reply comes back when the existing index is of a DIFFERENT type. Callers must
/// therefore read the existing index's type back and confirm it before treating
/// this as success (see `create_scalar_indexes`) — otherwise a stale wrong-type
/// index would be silently accepted, which is the same class of silent-wrong as
/// #218 itself. Unreachable today because `configure()` drops the collection
/// first, but a future `--skip-upload` path would hit it.
fn is_already_indexed_error(message: &str) -> bool {
    let m = message.to_lowercase();
    m.contains("already exist")
        || m.contains("at most one distinct index is allowed per field")
        || m.contains("creating multiple indexes on same field is not supported")
}

/// One index's build progress, as reported by `indexes/describe`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexProgress {
    field_name: String,
    index_type: String,
    state: String,
    indexed_rows: i64,
    pending_rows: i64,
    total_rows: i64,
}

impl IndexProgress {
    /// An index is only actually USABLE when it is built AND covers every row.
    ///
    /// `state == "Finished"` alone is worthless: before the collection is
    /// flushed, every index on v2.6.19 reports `Finished` with
    /// `indexedRows = totalRows = 0` — "finished indexing nothing" — and stays
    /// that way indefinitely. Queries then fall back to brute force over the
    /// growing segments regardless of the index. `indexedRows == totalRows`
    /// with `totalRows > 0` is the condition that actually distinguishes a
    /// usable index from that trap, and from a still-building one
    /// (`indexedRows < totalRows`).
    ///
    /// `pendingRows` is deliberately NOT part of the test. Measured on v2.6.19:
    /// once a 1M-row collection is fully indexed it reports
    /// `Finished, indexedRows = totalRows = 1000000, pendingRows = 847872` —
    /// the pending counter tracks segments queued for RE-indexing by background
    /// compaction, which runs indefinitely and does not mean the data is
    /// unindexed. Requiring `pendingRows == 0` therefore hangs until the
    /// timeout on exactly the large uploads that matter most.
    fn is_built(&self, expect_rows: bool) -> bool {
        self.state == "Finished"
            && self.indexed_rows == self.total_rows
            && (!expect_rows || self.total_rows > 0)
    }
}

fn parse_index_progress(row: &serde_json::Value) -> IndexProgress {
    let s = |k: &str| {
        row.get(k)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    };
    let n = |k: &str| row.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
    IndexProgress {
        field_name: s("fieldName"),
        index_type: s("indexType"),
        state: s("indexState"),
        indexed_rows: n("indexedRows"),
        pending_rows: n("pendingRows"),
        total_rows: n("totalRows"),
    }
}

pub struct MilvusEngine {
    name: String,
    collection_name: String,
    timeout: u64,
    batch_size: usize,
    parallel: usize,
    base_url: String,
    search_params: Vec<SearchParams>,
    /// M and efConstruction from upload_params.index_params
    index_m: i64,
    index_ef_construction: i64,
    /// Distance metric type (L2, IP, COSINE)
    metric_type: String,
    /// Index type (HNSW, IVF_FLAT, etc.)
    index_type: String,
}

impl MilvusEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = crate::effective_config::env_parsed("MILVUS_PORT", 19530);

        let collection_name =
            crate::effective_config::env_or("MILVUS_COLLECTION_NAME", DEFAULT_COLLECTION);

        let timeout: u64 = crate::effective_config::env_parsed("MILVUS_TIMEOUT", 300);

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
            .unwrap_or(1024) as usize;

        // Extract index params from upload_params
        let index_params = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("index_params"));

        let index_m = index_params
            .and_then(|p| p.get("M"))
            .and_then(|v| v.as_i64())
            .unwrap_or(16);

        let index_ef_construction = index_params
            .and_then(|p| p.get("efConstruction"))
            .and_then(|v| v.as_i64())
            .unwrap_or(128);

        let index_type = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("index_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("HNSW")
            .to_string();

        // Build base URL - Milvus REST API uses port 19530 (same as gRPC in newer versions)
        // or a dedicated REST port (9091 in some setups)
        let base_url = if host.starts_with("http") {
            host.to_string()
        } else {
            format!("http://{}:{}", host, port)
        };

        Ok(Self {
            name: engine_config.name.clone(),
            collection_name,
            timeout,
            batch_size,
            parallel,
            base_url,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            index_m,
            index_ef_construction,
            metric_type: String::new(), // Set during configure
            index_type,
        })
    }

    fn create_client(&self) -> Result<reqwest::blocking::Client, String> {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))
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

    fn drop_collection(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let body = serde_json::json!({
            "collectionName": self.collection_name,
        });

        let url = format!("{}/v2/vectordb/collections/drop", self.base_url);
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| e.to_string())?;

        // Ignore errors (collection might not exist)
        let _ = resp;
        Ok(())
    }

    fn create_collection(
        &self,
        client: &reqwest::blocking::Client,
        dataset: &Dataset,
    ) -> Result<(), String> {
        let vector_size = dataset.vector_size();

        // Build schema
        let mut fields = vec![
            serde_json::json!({
                "fieldName": "id",
                "dataType": "Int64",
                "isPrimary": true,
            }),
            serde_json::json!({
                "fieldName": "vector",
                "dataType": "FloatVector",
                "elementTypeParams": {
                    "dim": vector_size.to_string(),
                }
            }),
        ];

        // Add schema fields from dataset config
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    // Single source of truth for "does this schema field become a
                    // column, and of what Milvus type" — shared with the scalar
                    // index pass so the two can never disagree (a column with no
                    // index means a brute-force scan; see issue #218).
                    let Some(kind) = milvus_field_kind(field_name, ft) else {
                        continue;
                    };
                    // A multi-valued keyword field (`labels`) is declared as an
                    // Array of VarChar so `array_contains_any` can match a single
                    // element; a scalar VarChar could only test whole-string
                    // equality against the joined value (issue #88).
                    let field = if kind.data_type == "Array" {
                        serde_json::json!({
                            "fieldName": field_name,
                            "dataType": "Array",
                            "elementDataType": "VarChar",
                            "elementTypeParams": {"max_length": "500", "max_capacity": "128"},
                        })
                    } else {
                        let milvus_type = kind.data_type;
                        let mut field = serde_json::json!({
                            "fieldName": field_name,
                            "dataType": milvus_type,
                        });
                        if milvus_type == "VarChar" {
                            let mut params = serde_json::json!({"max_length": "500"});
                            if ft == "text" {
                                // Enable the analyzer + match inverted index so
                                // TEXT_MATCH full-text filtering works on this field.
                                let p = params.as_object_mut().unwrap();
                                p.insert("enable_analyzer".to_string(), serde_json::json!(true));
                                p.insert("enable_match".to_string(), serde_json::json!(true));
                            }
                            field
                                .as_object_mut()
                                .unwrap()
                                .insert("elementTypeParams".to_string(), params);
                        }
                        field
                    };
                    fields.push(field);
                }
            }
        }

        let body = serde_json::json!({
            "collectionName": self.collection_name,
            "schema": {
                "fields": fields,
                "enableDynamicField": false,
            }
        });

        let url = format!("{}/v2/vectordb/collections/create", self.base_url);
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| format!("Failed to create collection: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Failed to create collection: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ));
        }

        let resp_body: serde_json::Value = resp.json().unwrap_or_default();
        let code = resp_body.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
        if code != 0 {
            let msg = resp_body
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Err(format!("Failed to create collection: {}", msg));
        }

        Ok(())
    }

    fn create_index(
        &self,
        client: &reqwest::blocking::Client,
        dataset: &Dataset,
    ) -> Result<(), String> {
        let body = serde_json::json!({
            "collectionName": self.collection_name,
            "indexParams": [{
                "fieldName": "vector",
                "indexName": "vector_index",
                "metricType": self.metric_type,
                "indexType": self.index_type,
                "params": {
                    "M": self.index_m,
                    "efConstruction": self.index_ef_construction,
                }
            }]
        });

        let url = format!("{}/v2/vectordb/indexes/create", self.base_url);
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| format!("Failed to create index: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Failed to create index: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ));
        }

        self.create_scalar_indexes(client, dataset)
    }

    /// Create a scalar (payload) index on EVERY column built from the dataset
    /// schema. Without this, Milvus has nothing to look up the filter with and
    /// resolves every filtered query by scanning the scalar column behind the
    /// ANN search — the same rows, just far slower, which is why recall-based
    /// tests never caught it (issue #218).
    ///
    /// One request per field (rather than one batched request) so that an
    /// "already indexed" field is tolerated individually instead of failing the
    /// whole batch — the same granularity as upstream's per-field loop.
    fn create_scalar_indexes(
        &self,
        client: &reqwest::blocking::Client,
        dataset: &Dataset,
    ) -> Result<(), String> {
        let plan = scalar_index_plan(dataset);
        if plan.is_empty() {
            return Ok(());
        }

        let url = format!("{}/v2/vectordb/indexes/create", self.base_url);
        let mut created = Vec::with_capacity(plan.len());
        for (field_name, index_type) in &plan {
            let body = serde_json::json!({
                "collectionName": self.collection_name,
                "indexParams": [{
                    "fieldName": field_name,
                    "indexName": format!("{}_index", field_name),
                    "indexType": index_type,
                }]
            });
            let resp = client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .map_err(|e| format!("Failed to create scalar index on {}: {}", field_name, e))?;

            if !resp.status().is_success() {
                return Err(format!(
                    "Failed to create scalar index on {}: {} {}",
                    field_name,
                    resp.status(),
                    resp.text().unwrap_or_default()
                ));
            }

            // The REST layer answers 200 with an application-level `code`, so the
            // HTTP status alone proves nothing — an unchecked `code != 0` here is
            // precisely how a missing index stays invisible.
            let resp_body: serde_json::Value = resp.json().unwrap_or_default();
            let code = resp_body.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
            if code != 0 {
                let msg = resp_body
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error");
                if is_already_indexed_error(msg) {
                    // "An index exists" is NOT "the RIGHT index exists" — the
                    // same reply comes back for an index of a different type.
                    // Confirm the existing one from the server before accepting,
                    // or a stale wrong-type index passes silently (#218's class).
                    let existing = self
                        .describe_indexes(client)?
                        .into_iter()
                        .find(|p| p.field_name == *field_name);
                    match existing {
                        Some(p) if p.index_type == *index_type => {
                            println!(
                                "  scalar index on '{}' already exists ({}), skipping",
                                field_name, p.index_type
                            );
                            continue;
                        }
                        Some(p) => {
                            return Err(format!(
                                "Field '{}' already carries a {} index but this dataset needs \
                                 {}; drop the collection and re-upload",
                                field_name, p.index_type, index_type
                            ));
                        }
                        None => {
                            return Err(format!(
                                "Milvus reported an existing index on '{}' ({}) but none is \
                                 present in the index catalogue",
                                field_name, msg
                            ));
                        }
                    }
                }
                return Err(format!(
                    "Failed to create scalar index on {}: {}",
                    field_name, msg
                ));
            }
            created.push(format!("{}({})", field_name, index_type));
        }

        println!(
            "Created {}/{} scalar index(es): {}",
            created.len(),
            plan.len(),
            created.join(", ")
        );
        Ok(())
    }

    /// Seal the growing segments so an index can actually cover the data.
    ///
    /// A Milvus index only ever covers SEALED segments. Freshly inserted rows
    /// live in growing segments, which are brute-force scanned no matter what
    /// indexes exist. Without this call every index — the HNSW vector index
    /// included — reports `state: "Finished"` with `indexedRows: 0,
    /// totalRows: 0` and stays that way indefinitely, so the whole corpus is
    /// scanned and creating the scalar indexes at all is pointless.
    ///
    /// Verified on v2.6.19 with this engine's exact ingest order (100k rows,
    /// insert -> create indexes -> load): at the moment `load` reports
    /// `LoadStateLoaded`, and still 60s later, both `vector` and the scalar
    /// column read back as `Finished, indexedRows=0, totalRows=0`; immediately
    /// after an explicit flush they read `indexedRows=100000,
    /// totalRows=100000`. Upstream does the same thing — `upload.py` calls
    /// `collection.flush()` on the line directly above its `create_index` loop.
    fn flush_collection(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let body = serde_json::json!({ "collectionName": self.collection_name });
        let url = format!("{}/v2/vectordb/collections/flush", self.base_url);
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| format!("Failed to flush collection: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Failed to flush collection: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ));
        }
        let resp_body: serde_json::Value = resp.json().unwrap_or_default();
        let code = resp_body.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
        if code != 0 {
            let msg = resp_body
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Err(format!("Failed to flush collection: {}", msg));
        }
        Ok(())
    }

    /// Every index on the collection, with its build progress.
    fn describe_indexes(
        &self,
        client: &reqwest::blocking::Client,
    ) -> Result<Vec<IndexProgress>, String> {
        let list_url = format!("{}/v2/vectordb/indexes/list", self.base_url);
        let resp = client
            .post(&list_url)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({ "collectionName": self.collection_name }))
            .send()
            .map_err(|e| format!("Failed to list indexes: {}", e))?;
        let body: serde_json::Value = resp.json().unwrap_or_default();
        let names: Vec<String> = body
            .get("data")
            .and_then(|d| d.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let describe_url = format!("{}/v2/vectordb/indexes/describe", self.base_url);
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let resp = client
                .post(&describe_url)
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "collectionName": self.collection_name,
                    "indexName": name,
                }))
                .send()
                .map_err(|e| format!("Failed to describe index {}: {}", name, e))?;
            let body: serde_json::Value = resp.json().unwrap_or_default();
            if let Some(rows) = body.get("data").and_then(|d| d.as_array()) {
                out.extend(rows.iter().map(parse_index_progress));
            }
        }
        Ok(out)
    }

    /// Block until every index covers every row, so the search phase measures an
    /// INDEXED collection.
    ///
    /// `load_collection` is not sufficient: it returns `LoadStateLoaded` while
    /// indexes cover nothing (see `flush_collection`). Upstream waits the same
    /// way — `wait_for_index_building_complete` on every index before `load()`.
    /// Timing out is a hard error: silently searching a half-indexed collection
    /// is exactly the failure mode this PR exists to remove.
    fn wait_for_indexes_built(
        &self,
        client: &reqwest::blocking::Client,
        uploaded_rows: usize,
    ) -> Result<(), String> {
        let expect_rows = uploaded_rows > 0;
        let timeout_secs: u64 =
            crate::effective_config::env_parsed("MILVUS_INDEX_BUILD_TIMEOUT", 1800);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
        let mut backoff = std::time::Duration::from_millis(500);
        let max_backoff = std::time::Duration::from_secs(10);

        loop {
            let progress = self.describe_indexes(client)?;
            if !progress.is_empty() && progress.iter().all(|p| p.is_built(expect_rows)) {
                let summary: Vec<String> = progress
                    .iter()
                    .map(|p| {
                        format!(
                            "{}({}, {} rows)",
                            p.field_name, p.index_type, p.indexed_rows
                        )
                    })
                    .collect();
                println!("All indexes built: {}", summary.join(", "));
                return Ok(());
            }
            if std::time::Instant::now() > deadline {
                let summary: Vec<String> = progress
                    .iter()
                    .map(|p| {
                        format!(
                            "{}({}, state={}, indexed={}/{}, pending={})",
                            p.field_name,
                            p.index_type,
                            p.state,
                            p.indexed_rows,
                            p.total_rows,
                            p.pending_rows
                        )
                    })
                    .collect();
                return Err(format!(
                    "Timed out after {}s waiting for indexes to cover all rows: {}",
                    timeout_secs,
                    summary.join(", ")
                ));
            }
            std::thread::sleep(backoff);
            backoff = (backoff * 2).min(max_backoff);
        }
    }

    fn load_collection(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let body = serde_json::json!({
            "collectionName": self.collection_name,
        });

        let url = format!("{}/v2/vectordb/collections/load", self.base_url);
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| format!("Failed to load collection: {}", e))?;

        if !resp.status().is_success() {
            eprintln!(
                "Warning: load collection returned: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            );
        }

        // Wait for loading to complete (exponential backoff: 1s, 2s, 4s, ... capped at 16s)
        println!("Waiting for collection to be loaded...");
        let mut backoff_secs = 1u64;
        let max_backoff = 16u64;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
        loop {
            std::thread::sleep(std::time::Duration::from_secs(backoff_secs));
            let body = serde_json::json!({
                "collectionName": self.collection_name,
            });
            let url = format!("{}/v2/vectordb/collections/get_load_state", self.base_url);
            if let Ok(resp) = client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
            {
                if let Ok(body) = resp.json::<serde_json::Value>() {
                    if let Some(data) = body.get("data") {
                        if let Some(state) = data.get("loadState").and_then(|s| s.as_str()) {
                            if state == "LoadStateLoaded" {
                                println!("Collection loaded.");
                                return Ok(());
                            }
                        }
                    }
                }
            }
            if std::time::Instant::now() > deadline {
                return Err("Timed out waiting for collection to load".to_string());
            }
            backoff_secs = (backoff_secs * 2).min(max_backoff);
        }
    }

    fn upload_parallel(
        &self,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
        schema_types: &HashMap<String, String>,
    ) -> Result<(), String> {
        let pb = self.create_progress_bar(ids.len());
        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.batch_size)
            .map(|start| (start, (start + self.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        std::thread::scope(|s| {
            for _ in 0..self.parallel {
                let base_url = self.base_url.clone();
                let collection_name = self.collection_name.clone();
                let timeout = self.timeout;
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let client = match reqwest::blocking::Client::builder()
                        .timeout(std::time::Duration::from_secs(timeout))
                        .build()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
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
                        if let Err(e) = insert_batch(
                            &client,
                            &base_url,
                            &collection_name,
                            &ids[batch_start..batch_end],
                            &vectors[batch_start..batch_end],
                            &metadata[batch_start..batch_end],
                            schema_types,
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
}

/// Insert a batch of vectors using Milvus REST API.
/// Extract `field -> declared-type` from the dataset schema, so uploads keep a
/// numeric-valued keyword field as a string (the column is declared VarChar).
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

fn insert_batch(
    client: &reqwest::blocking::Client,
    base_url: &str,
    collection_name: &str,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
    schema_types: &HashMap<String, String>,
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    let mut data = Vec::with_capacity(ids.len());
    for i in 0..ids.len() {
        let mut row = serde_json::json!({
            "id": ids[i],
            "vector": vectors[i],
        });

        if let Some(meta) = &metadata[i] {
            let row_obj = row.as_object_mut().unwrap();
            for (k, v) in &meta.fields {
                // A numeric value under a keyword/text-declared field must stay a
                // string, or the strict VarChar column rejects the whole batch.
                let v = v.coerce_for_schema(schema_types.get(k).map(|s| s.as_str()));
                let val = match v.as_ref() {
                    MetadataValue::String(s) => match schema_types.get(k).map(|t| t.as_str()) {
                        // Native Bool column needs a JSON bool, not the reader's
                        // "true"/"false" string.
                        Some("bool") => match s.as_str() {
                            "true" => serde_json::Value::Bool(true),
                            "false" => serde_json::Value::Bool(false),
                            _ => serde_json::Value::String(s.clone()),
                        },
                        // datetime -> Int64 epoch seconds (same conversion the range
                        // filter uses, so stored and queried values compare exactly).
                        Some("datetime") => datetime_to_epoch_secs(s)
                            .map(|e| serde_json::Value::from(e as i64))
                            .unwrap_or_else(|| serde_json::Value::String(s.clone())),
                        _ => serde_json::Value::String(s.clone()),
                    },
                    MetadataValue::Int(n) => serde_json::Value::from(*n),
                    MetadataValue::Float(f) => serde_json::json!(*f),
                    MetadataValue::Labels(labels) => {
                        // Multi-valued keyword field: store as a native Array so
                        // `array_contains_any` can match individual elements (#88).
                        serde_json::Value::Array(
                            labels
                                .iter()
                                .map(|l| serde_json::Value::String(l.clone()))
                                .collect(),
                        )
                    }
                    MetadataValue::Geo { lon, lat } => {
                        serde_json::Value::String(geo_wkt_point(*lon, *lat))
                    }
                };
                row_obj.insert(k.clone(), val);
            }
        }

        data.push(row);
    }

    let body = serde_json::json!({
        "collectionName": collection_name,
        "data": data,
    });

    let url = format!("{}/v2/vectordb/entities/insert", base_url);
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .map_err(|e| format!("Insert failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Insert error: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }

    let resp_body: serde_json::Value = resp.json().unwrap_or_default();
    let code = resp_body.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
    if code != 0 {
        let msg = resp_body
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(format!("Insert failed: {}", msg));
    }

    Ok(())
}

/// Search Milvus via REST API.
#[allow(clippy::too_many_arguments)]
/// Build and serialize one search request body to JSON bytes. Done OUTSIDE the
/// per-query timed window: serializing the query vector to JSON decimal text
/// (ryu formatting over every dimension) is client CPU work, not server latency.
/// Pre-serializing means the timed send only copies the finished bytes onto the
/// socket, matching the reference engines (pgvector/qdrant). The bytes are
/// identical to what `.json(&body)` would have sent inline.
fn build_search_body(
    collection_name: &str,
    query_vector: &[f32],
    top: usize,
    metric_type: &str,
    ef: Option<i64>,
    filter: Option<&str>,
) -> Vec<u8> {
    let mut body = serde_json::json!({
        "collectionName": collection_name,
        "data": [query_vector],
        "annsField": "vector",
        "limit": top,
        "outputFields": ["id"],
    });

    let mut search_params = serde_json::json!({
        "metric_type": metric_type,
    });
    if let Some(ef_val) = ef {
        search_params
            .as_object_mut()
            .unwrap()
            .insert("params".to_string(), serde_json::json!({"ef": ef_val}));
    }
    body.as_object_mut()
        .unwrap()
        .insert("searchParams".to_string(), search_params);

    if let Some(f) = filter {
        body.as_object_mut().unwrap().insert(
            "filter".to_string(),
            serde_json::Value::String(f.to_string()),
        );
    }

    serde_json::to_vec(&body).expect("serialize search body")
}

/// Send a pre-serialized search request and return the DECODED response. The
/// consistent timed boundary (see qdrant/pgvector/redis) is: request body
/// pre-serialized OUTSIDE the window; RPC send + receive + decode-to-structured-
/// response INSIDE the window (this fn: post + HTTP-status check + wire read +
/// `from_str`); app-level code check + id/score extraction OUTSIDE
/// (`extract_search_hits`). So the JSON decode is billed as latency exactly like
/// qdrant's protobuf decode.
fn send_search(
    client: &reqwest::blocking::Client,
    base_url: &str,
    body: &[u8],
) -> Result<serde_json::Value, String> {
    let url = format!("{}/v2/vectordb/entities/search", base_url);
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body.to_vec())
        .send()
        .map_err(|e| format!("Search failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Search error: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }

    let text = resp
        .text()
        .map_err(|e| format!("Failed to read search response: {}", e))?;
    serde_json::from_str(&text).map_err(|e| format!("Failed to parse search response: {}", e))
}

/// Extract the id/score list from an already-decoded response (done AFTER the
/// timed window — the app-level `code` check and id extraction pull the final
/// ids out of the decoded struct for recall, mirroring pgvector/qdrant).
fn extract_search_hits(resp_body: &serde_json::Value) -> Result<Vec<(i64, f64)>, String> {
    let code = resp_body.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
    if code != 0 {
        let msg = resp_body
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(format!("Search failed: {}", msg));
    }

    let results = resp_body
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| "Missing data array in search response".to_string())?;

    let mut hits = Vec::with_capacity(results.len());
    for result in results {
        let id = result.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
        let distance = result
            .get("distance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        hits.push((id, distance));
    }

    Ok(hits)
}

/// Parse conditions into Milvus filter expression.
/// Map a dataset distance name to the Milvus `metric_type`. Cosine deliberately
/// maps to `IP` (Milvus scores cosine via inner-product over normalized vectors).
/// Unknown metrics error. A wrong arm here would silently change ranking, so
/// every arm is unit-tested.
fn map_milvus_metric_type(distance: &str) -> Result<&'static str, String> {
    match distance.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok("L2"),
        "dot" | "ip" => Ok("IP"),
        "cosine" | "angular" => Ok("IP"), // Milvus uses IP for cosine (normalized vectors)
        other => Err(format!("Unsupported distance metric for Milvus: {}", other)),
    }
}

/// Milvus uses string-based filter expressions like "field == value && field > 10"
pub(crate) fn parse_milvus_conditions(conditions: &serde_json::Value) -> Option<String> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let mut clauses = Vec::new();

    if let Some(and_items) = obj.get("and").and_then(|v| v.as_array()) {
        let and_filters: Vec<String> = and_items
            .iter()
            .filter_map(build_milvus_entry_filter)
            .collect();
        if !and_filters.is_empty() {
            clauses.push(format!("({})", and_filters.join(" && ")));
        }
    }

    if let Some(or_items) = obj.get("or").and_then(|v| v.as_array()) {
        let or_filters: Vec<String> = or_items
            .iter()
            .filter_map(build_milvus_entry_filter)
            .collect();
        if !or_filters.is_empty() {
            clauses.push(format!("({})", or_filters.join(" || ")));
        }
    }

    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" && "))
    }
}

fn build_milvus_entry_filter(entry: &serde_json::Value) -> Option<String> {
    let entry_obj = entry.as_object()?;

    // A nested group (an entry that is itself a `{and:[...]}` / `{or:[...]}`
    // tree) is built as its own PARENTHESISED boolean-expr sub-string via the
    // top-level parser, then combined by the parent's && / ||. Without this the
    // group falls through to the leaf path below, where `field_name` is "and"/
    // "or" and `field_filters` is an array (not an object), so `as_object()`
    // fails, nothing is emitted, and the whole clause silently collapses to
    // "no filter" — returning every row instead of the nested union.
    if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
        return parse_milvus_conditions(entry).map(|f| format!("({})", f));
    }

    let mut filters = Vec::new();

    for (field_name, field_filters) in entry_obj {
        if let Some(filter_obj) = field_filters.as_object() {
            for (cond_type, criteria) in filter_obj {
                if let Some(f) = build_milvus_filter(field_name, cond_type, criteria) {
                    filters.push(f);
                }
            }
        }
    }

    if filters.is_empty() {
        None
    } else {
        Some(filters.join(" && "))
    }
}

/// Quote a string value for a Milvus filter expression.
///
/// Milvus expressions use double-quoted string literals. Backslashes and
/// double-quotes inside the value must be escaped (backslash first, so we don't
/// double-escape the backslashes we just introduced), otherwise a value such as
/// `15"laptop` would terminate the literal early and produce a malformed /
/// injectable expression that errors the whole search. Returns the value WITH
/// its surrounding double-quotes so callers can inline it directly.
fn quote_milvus_string(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn build_milvus_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<String> {
    match condition_type {
        "match" => {
            // A multi-valued keyword field (`labels`) is stored as an Array of
            // VarChar (see create_collection / insert_batch), so it uses Milvus's
            // array membership functions — `array_contains_any` for `match_any`
            // and `array_contains` for exact-match — which test per-element
            // membership. A scalar field uses `in [...]` / `==`. (Issue #88.)
            let multivalued = is_multivalued_keyword_field(field_name);
            // match_any: OR-of-values, mirroring qdrant's Condition::matches.
            // Strings are quoted/escaped, numbers inlined; bool/null/nested items
            // are skipped so an invalid expression is never produced. An empty
            // (or all-skipped) IN-set matches NOTHING — we still emit a valid
            // match-nothing expression rather than dropping the sole clause,
            // which would leave no filter and return every row (the inverse).
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                let items: Vec<String> = any
                    .iter()
                    .filter_map(|v| {
                        if let Some(s) = v.as_str() {
                            Some(quote_milvus_string(s))
                        } else if v.is_number() {
                            Some(v.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();
                if multivalued {
                    return Some(format!(
                        "array_contains_any({}, [{}])",
                        field_name,
                        items.join(", ")
                    ));
                }
                return Some(format!("{} in [{}]", field_name, items.join(", ")));
            }
            // Full-text: `{match:{text}}` -> Milvus TEXT_MATCH over an
            // analyzer-enabled VarChar field (schema creation sets
            // enable_analyzer/enable_match for `text` fields). Matches rows whose
            // analyzed field CONTAINS the token; dropping the clause would run the
            // search UNFILTERED while recall is scored against filtered truth.
            if let Some(text) = criteria.get("text").and_then(|v| v.as_str()) {
                return Some(format!(
                    "TEXT_MATCH({}, {})",
                    field_name,
                    quote_milvus_string(text)
                ));
            }
            let value = criteria.get("value")?;
            if let Some(s) = value.as_str() {
                if multivalued {
                    Some(format!(
                        "array_contains({}, {})",
                        field_name,
                        quote_milvus_string(s)
                    ))
                } else {
                    Some(format!("{} == {}", field_name, quote_milvus_string(s)))
                }
            } else if !multivalued && (value.is_number() || value.is_boolean()) {
                Some(format!("{} == {}", field_name, value))
            } else {
                // Non-scalar exact-match value (a JSON array/object/null), or a
                // numeric/bool value against the string-array `labels` field, is
                // malformed input — the canonical model uses `match.any` for
                // lists. Drop the clause (return None) instead of forwarding a
                // bad expression, matching qdrant/redis/valkey/vectorsets.
                None
            }
        }
        "range" => {
            let criteria_obj = criteria.as_object()?;
            let mut clauses = Vec::new();
            // Render a range bound as a Milvus scalar literal. A numeric bound is
            // emitted verbatim; a string bound is an ISO-8601 datetime over the
            // Int64-epoch column, converted with datetime_to_epoch_secs (the same
            // conversion upload uses). A bound we can't render is dropped.
            let bound_literal = |v: &serde_json::Value| -> Option<String> {
                if v.is_number() {
                    Some(v.to_string())
                } else if let Some(s) = v.as_str() {
                    datetime_to_epoch_secs(s).map(|e| (e as i64).to_string())
                } else {
                    None
                }
            };
            for (key, sql_op) in [("lt", "<"), ("gt", ">"), ("lte", "<="), ("gte", ">=")] {
                if let Some(bound) = criteria_obj.get(key) {
                    if !bound.is_null() {
                        if let Some(lit) = bound_literal(bound) {
                            clauses.push(format!("{} {} {}", field_name, sql_op, lit));
                        }
                    }
                }
            }
            if clauses.is_empty() {
                None
            } else {
                Some(format!("({})", clauses.join(" && ")))
            }
        }
        // Geo-radius (issue #223). `ST_DWITHIN(field, 'POINT(lon lat)', metres)`
        // is Milvus' native geodesic radius predicate: for a POINT column against
        // a POINT query the server evaluates a haversine great-circle distance on
        // R = 6 371 000 m and compares `<= radius` (v2.6.19
        // `internal/core/src/common/Geometry.h::dwithin`). Available since 2.6.4;
        // before this arm existed the clause was dropped and, when it was the
        // only clause, the query ran with NO filter against geo-filtered ground
        // truth.
        //
        // WKT is `POINT(x y)` = `POINT(lon lat)`. Literals go through
        // `geo::plain_decimal`, the shortest FIXED-POINT form that round-trips,
        // so the centre the server parses is bit-identical to the dataset's.
        //
        // A missing or non-finite component is a DROP (→ the #219 hard error),
        // never a default radius.
        "geo" => {
            let (lon, lat, radius) = (
                criteria.get("lon").and_then(|v| v.as_f64())?,
                criteria.get("lat").and_then(|v| v.as_f64())?,
                criteria.get("radius").and_then(|v| v.as_f64())?,
            );
            if !lon.is_finite() || !lat.is_finite() || !radius.is_finite() || radius < 0.0 {
                return None;
            }
            // `{:?}` would switch to exponent form below 1e-4, and a WKT
            // literal like `POINT(-52.5 -1.9e-5)` is not portable across WKT
            // parsers. See `geo::plain_decimal`.
            Some(format!(
                "ST_DWITHIN({field_name}, 'POINT({} {})', {})",
                geo::plain_decimal(lon),
                geo::plain_decimal(lat),
                geo::plain_decimal(radius)
            ))
        }
        _ => None,
    }
}

impl Engine for MilvusEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let dist_lower = distance.to_lowercase();

        // Map distance metric
        self.metric_type = map_milvus_metric_type(&dist_lower)?.to_string();

        let client = self.create_client()?;

        println!("Dropping existing collection...");
        self.drop_collection(&client)?;

        println!("Creating collection '{}'...", self.collection_name);
        self.create_collection(&client, dataset)?;

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
            self.parallel, self.batch_size
        );
        let upload_start = Instant::now();
        let schema_types = schema_type_map(dataset);
        self.upload_parallel(&ids, &vectors, &metadata, &schema_types)?;
        let upload_time = upload_start.elapsed().as_secs_f64();

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        // Flush -> create indexes -> wait for them to cover every row -> load.
        // This whole block is part of the ingest cost and is included in
        // total_time for cross-engine comparability (mirrors mongodb; matches
        // v0's post_upload() timing), and it mirrors upstream's post_upload
        // order exactly (flush, create_index, wait_for_index_building_complete,
        // load). Every step is load-bearing: without the flush the indexes cover
        // zero rows, and without the wait the search phase starts against a
        // collection Milvus reports as "loaded" while it is still brute-forcing.
        let client = self.create_client()?;
        let index_start = Instant::now();

        println!("Flushing collection (seal growing segments so indexes cover them)...");
        self.flush_collection(&client)?;

        println!(
            "Creating {} index (M={}, efConstruction={}, metric={})...",
            self.index_type, self.index_m, self.index_ef_construction, self.metric_type
        );
        self.create_index(&client, dataset)?;

        println!("Waiting for all indexes to cover every row...");
        self.wait_for_indexes_built(&client, vectors.len())?;

        // Load collection into memory
        self.load_collection(&client)?;
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
            parallel: self.parallel,
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
        // Re-derive the metric type here as well as in configure(): on the
        // `--skip-upload` path configure() never runs (#238) and this field would
        // still be the `new()` default of "" — an empty metric_type in the search
        // body. Pure function of the dataset, so recomputing is idempotent.
        self.metric_type = map_milvus_metric_type(&dataset.distance().to_lowercase())?.to_string();

        let parallel = params.parallel.unwrap_or(1) as usize;

        // Extract ef from search params. The typed `ef` field first, then the
        // `params: { ef }` shape resolved through knob() so the nested
        // (upstream) placement is honoured too, not just the flat one.
        let ef = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .or_else(|| {
                params
                    .knob("params")
                    .and_then(|p| p.get("ef"))
                    .and_then(|v| v.as_i64())
            });

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<QueryFilter<String>> =
            conditions.resolve_all("Milvus", parse_milvus_conditions)?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute per-query `top` and the fully serialized request bodies
        // BEFORE the parallel region so the timed window wraps only the RPC
        // round-trip (see build_search_body). `tops[idx]` reproduces the same k
        // the request embeds, so recall is computed against an identical result
        // set.
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
        let bodies: Vec<Vec<u8>> = (0..num_to_run)
            .map(|idx| {
                build_search_body(
                    &self.collection_name,
                    &queries[idx],
                    tops[idx],
                    &self.metric_type,
                    ef,
                    parsed_filters[idx].as_deref(),
                )
            })
            .collect();

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);

        // Gate-synchronized start so connection setup AND the cold first query
        // fall OUTSIDE the measured window. Every worker connects + primes, then
        // parks at the gate; `WorkerPool::start` stamps the shared start instant and
        // releases everyone, so the measurement clock starts only once all workers
        // are warm and poised. The gate is count-agnostic: a worker that fails to
        // set up, panics, or is never started by the OS settles its ticket and turns
        // the run into a hard error instead of a hang (#214).

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        let measured_start = std::thread::scope(|s| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(s, "milvus-search", parallel);
            for _ in 0..parallel {
                let base_url = self.base_url.clone();
                let timeout = self.timeout;
                let neighbors = &neighbors;
                let tops = &tops;
                let bodies = &bodies;
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                pool.spawn(move |ticket| {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();

                    let client = match reqwest::blocking::Client::builder()
                        .timeout(std::time::Duration::from_secs(timeout))
                        .build()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!("milvus-search worker setup failed: {e}"));
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Prime this client with ONE discarded query (body 0) so the
                    // cold first round-trip is not inside the measured window. Best
                    // effort: errors are ignored and its sample is NOT recorded.
                    if !bodies.is_empty() {
                        let _ = send_search(&client, &base_url, &bodies[0]);
                    }

                    // Signal "connected + primed", then block until the coordinator
                    // stamps the shared measurement start and releases everyone.
                    let Some(_start_time) = ticket.arrive_and_wait() else {
                        return (t, p, r, mr, nd);
                    };

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }

                        let top = tops[idx];

                        // Timed window: network send + receive + decode of the
                        // response into a structured value. Body is pre-serialized
                        // (out); code check + id extraction run after `elapsed` (out).
                        let query_start = Instant::now();
                        let response = send_search(&client, &base_url, &bodies[idx]);
                        let query_time = query_start.elapsed().as_secs_f64();

                        match response.and_then(|resp_body| extract_search_hits(&resp_body)) {
                            Ok(result_ids) => {
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
                                eprintln!("Search query {} failed: {}", idx, e);
                            }
                        }
                        pb.inc(1);
                    }
                    (t, p, r, mr, nd)
                })?;
            }

            // Every worker is connected + primed and parked at the gate.
            // Stamp the shared measurement start and release them together.
            let (per_worker, measured_start) = pool.start()?;

            for (t, p, r, mr, nd) in per_worker {
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrr_vals.extend(mr);
                ndcg_vals.extend(nd);
            }
            Ok(measured_start)
        })?;

        pb.finish_and_clear();
        // total_time excludes connection setup and the cold first query: it is
        // measured from the gate release stamp.
        let total_time = measured_start.elapsed().as_secs_f64();

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )
    }

    fn delete(&mut self) -> Result<(), String> {
        let client = self.create_client()?;
        self.drop_collection(&client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Load-bearing: the hoisted request body bytes must equal what the old inline
    /// `.json(&body)` path put on the wire (reqwest's `.json` uses `to_vec`).
    #[test]
    fn build_search_body_bytes_match_json_serialization() {
        let vec = vec![0.1f32, -0.2, 0.3];
        let top = 2usize;
        let filter = "color == \"red\"";

        // serde_json Map is a BTreeMap (no preserve_order feature), so an
        // equivalent literal serializes byte-identically regardless of insert order.
        let expected = json!({
            "collectionName": "bench",
            "data": [vec],
            "annsField": "vector",
            "limit": top,
            "outputFields": ["id"],
            "searchParams": {"metric_type": "COSINE", "params": {"ef": 64}},
            "filter": filter,
        });
        let expected_bytes = serde_json::to_vec(&expected).unwrap();

        let body = build_search_body("bench", &vec, top, "COSINE", Some(64), Some(filter));
        assert_eq!(body, expected_bytes);

        // Unfiltered + no-ef variant.
        let expected_nf = json!({
            "collectionName": "bench",
            "data": [vec],
            "annsField": "vector",
            "limit": top,
            "outputFields": ["id"],
            "searchParams": {"metric_type": "COSINE"},
        });
        let body_nf = build_search_body("bench", &vec, top, "COSINE", None, None);
        assert_eq!(body_nf, serde_json::to_vec(&expected_nf).unwrap());
    }

    #[test]
    fn match_any_string_list_emits_in() {
        let e = json!({"and": [{"color": {"match": {"any": ["red", "blue"]}}}]});
        let expr = parse_milvus_conditions(&e).unwrap();
        assert!(
            expr.contains(r#"color in ["red", "blue"]"#),
            "expr={}",
            expr
        );
    }

    #[test]
    fn match_any_int_list_emits_in() {
        let e = json!({"and": [{"size": {"match": {"any": [1, 2, 3]}}}]});
        let expr = parse_milvus_conditions(&e).unwrap();
        assert!(expr.contains("size in [1, 2, 3]"), "expr={}", expr);
    }

    #[test]
    fn match_any_empty_list_matches_nothing() {
        // Empty IN-set must match NOTHING (never invert to returning all rows):
        // `field in []`, not a dropped clause.
        let e = json!({"and": [{"color": {"match": {"any": []}}}]});
        let expr = parse_milvus_conditions(&e).unwrap();
        assert!(expr.contains("color in []"), "expr={}", expr);
    }

    #[test]
    fn match_exact_value_still_works() {
        assert_eq!(
            build_milvus_filter("color", "match", &json!({"value": "red"})).unwrap(),
            r#"color == "red""#
        );
    }

    // #88: the multi-valued keyword field `labels` is an Array(VarChar), so it
    // uses array membership functions instead of scalar `in`/`==`.
    #[test]
    fn labels_match_any_uses_array_contains_any() {
        let expr =
            build_milvus_filter("labels", "match", &json!({"any": ["red", "blue"]})).unwrap();
        assert_eq!(expr, r#"array_contains_any(labels, ["red", "blue"])"#);
    }

    #[test]
    fn labels_exact_value_uses_array_contains() {
        let expr = build_milvus_filter("labels", "match", &json!({"value": "red"})).unwrap();
        assert_eq!(expr, r#"array_contains(labels, "red")"#);
    }

    #[test]
    fn labels_numeric_value_dropped() {
        // A numeric exact-match against the string-array `labels` is malformed.
        assert!(build_milvus_filter("labels", "match", &json!({"value": 5})).is_none());
    }

    // #121: a non-scalar `value` (a JSON array/object/null) is malformed input —
    // the canonical model uses `match.any` for lists. It must be dropped (None),
    // not forwarded verbatim as `n == [1,2]`. Matches qdrant/redis/valkey/
    // vectorsets. (Scalar kinds int/float/bool covered by exact_match_int_float_bool.)
    #[test]
    fn match_non_scalar_value_dropped() {
        assert!(build_milvus_filter("n", "match", &json!({"value": [1, 2]})).is_none());
        assert!(build_milvus_filter("n", "match", &json!({"value": {"x": 1}})).is_none());
        assert!(build_milvus_filter("n", "match", &json!({"value": null})).is_none());
        // As the sole clause, the whole filter is dropped.
        let e = json!({"and": [{"n": {"match": {"value": [1, 2]}}}]});
        assert!(parse_milvus_conditions(&e).is_none());
    }

    #[test]
    fn match_exact_value_escapes_quotes_and_backslashes() {
        // Exact-value string branch must escape through the shared helper:
        // backslash first (so introduced backslashes aren't doubled), then quote.
        assert_eq!(
            build_milvus_filter("color", "match", &json!({"value": r#"a"b\c"#})).unwrap(),
            r#"color == "a\"b\\c""#
        );
    }

    #[test]
    fn match_any_escapes_quotes_and_backslashes() {
        // A keyword value containing BOTH a double-quote and a backslash must be
        // escaped identically to the exact-value branch (shared helper). Input
        // `a"b\c` -> literal `"a\"b\\c"`.
        let e = json!({"and": [{"color": {"match": {"any": [r#"a"b\c"#]}}}]});
        let expr = parse_milvus_conditions(&e).unwrap();
        assert_eq!(expr, r#"(color in ["a\"b\\c"])"#, "expr={}", expr);
    }

    // ── OR-branch of the condition parser ──────────────────────────────────

    #[test]
    fn or_only_emits_double_pipe_group() {
        let cond = json!({"or":[
            {"a":{"match":{"value":"x"}}},
            {"b":{"match":{"value":"y"}}},
        ]});
        assert_eq!(
            parse_milvus_conditions(&cond).unwrap(),
            r#"(a == "x" || b == "y")"#
        );
    }

    #[test]
    fn and_plus_or_keeps_both_groups() {
        let cond = json!({
            "and":[{"a":{"match":{"value":"x"}}}],
            "or":[{"b":{"match":{"value":"y"}}}],
        });
        assert_eq!(
            parse_milvus_conditions(&cond).unwrap(),
            r#"(a == "x") && (b == "y")"#
        );
    }

    // ── Range operators ────────────────────────────────────────────────────

    fn range_expr(criteria: serde_json::Value) -> Option<String> {
        build_milvus_filter("age", "range", &criteria)
    }

    #[test]
    fn range_lt_lte_gt_gte() {
        assert_eq!(range_expr(json!({"lt":5})).unwrap(), "(age < 5)");
        assert_eq!(range_expr(json!({"lte":5})).unwrap(), "(age <= 5)");
        assert_eq!(range_expr(json!({"gt":5})).unwrap(), "(age > 5)");
        assert_eq!(range_expr(json!({"gte":5})).unwrap(), "(age >= 5)");
    }

    #[test]
    fn range_two_sided_gte_lt() {
        // Fixed order lt, gt, lte, gte joined by &&.
        assert_eq!(
            range_expr(json!({"gte":10,"lt":20})).unwrap(),
            "(age < 20 && age >= 10)"
        );
    }

    #[test]
    fn range_unknown_op_is_none() {
        assert!(range_expr(json!({"foo":5})).is_none());
    }

    #[test]
    fn range_null_bound_is_none() {
        assert!(range_expr(json!({"gte":serde_json::Value::Null})).is_none());
    }

    // ── Geo filter (native ST_DWITHIN, issue #223) ─────────────────────────

    /// The exact string that goes on the wire. WKT is `POINT(x y)` — LONGITUDE
    /// first — so an axis swap here is a silently displaced query centre, and
    /// the fixture in `tests/integration_milvus.rs` is what proves the server
    /// agrees.
    #[test]
    fn geo_emits_st_dwithin_with_lon_first_and_metres() {
        assert_eq!(
            build_milvus_filter("loc", "geo", &json!({"lat":20.0,"lon":10.0,"radius":500})),
            Some("ST_DWITHIN(loc, 'POINT(10.0 20.0)', 500.0)".to_string())
        );
    }

    /// The INSERT side of the axis order. A lat/lon swap applied to both halves
    /// is self-consistent — it selects the identical documents — so this cannot
    /// be caught by any recall fixture; it needs a pin.
    #[test]
    fn stored_wkt_is_point_lon_lat_not_lat_lon() {
        // Deliberately asymmetric and outside latitude range if swapped.
        assert_eq!(geo_wkt_point(179.9, 40.5), "POINT(179.9 40.5)");
        assert_eq!(geo_wkt_point(-74.0, 40.0), "POINT(-74.0 40.0)");
        // Storage and query must agree on the order, which is the only thing
        // that makes the pair meaningful.
        let stored = geo_wkt_point(10.0, 20.0);
        let queried =
            build_milvus_filter("loc", "geo", &json!({"lat":20.0,"lon":10.0,"radius":500}))
                .unwrap();
        assert!(
            queried.contains(&stored),
            "stored {stored} vs query {queried}"
        );
        // And no exponent form on the insert side either (see geo::plain_decimal).
        let tiny = geo_wkt_point(-52.545, -0.000019426574824796435);
        assert!(!tiny.contains('e') && !tiny.contains('E'), "{tiny}");
    }

    /// An incomplete criteria object is a DROP (→ the #219 hard error), not a
    /// default radius: Qdrant/Elasticsearch/Weaviate substitute 1000 m, which
    /// would invent a filter nobody asked for.
    #[test]
    fn geo_missing_component_is_none() {
        for bad in [
            json!({"lat":20.0,"lon":10.0}),
            json!({"lon":10.0,"radius":500}),
            json!({"lat":20.0,"radius":500}),
            json!({"lat":20.0,"lon":10.0,"radius":-1}),
        ] {
            assert!(build_milvus_filter("loc", "geo", &bad).is_none(), "{bad}");
        }
    }

    /// Two geo leaves must both survive the `&&`/`||` joins — the partial-drop
    /// class `filter_guard::no_shipped_multi_leaf_filter_loses_a_leaf` covers
    /// generically, pinned here as an exact string.
    #[test]
    fn two_geo_leaves_both_reach_the_expression() {
        let expr = parse_milvus_conditions(&json!({"and":[
            {"a":{"geo":{"lon":116.0,"lat":-52.0,"radius":326341.0}}},
            {"b":{"geo":{"lon":12.0,"lat":40.0,"radius":100000.0}}}
        ]}))
        .unwrap();
        assert_eq!(
            expr,
            "(ST_DWITHIN(a, 'POINT(116.0 -52.0)', 326341.0) && \
             ST_DWITHIN(b, 'POINT(12.0 40.0)', 100000.0))"
        );
    }

    // ── Distance-metric mapping ────────────────────────────────────────────

    #[test]
    fn metric_type_mapping_covers_all_arms() {
        assert_eq!(map_milvus_metric_type("l2").unwrap(), "L2");
        assert_eq!(map_milvus_metric_type("euclidean").unwrap(), "L2");
        assert_eq!(map_milvus_metric_type("dot").unwrap(), "IP");
        assert_eq!(map_milvus_metric_type("ip").unwrap(), "IP");
        // Cosine deliberately maps to IP (inner-product over normalized vectors).
        assert_eq!(map_milvus_metric_type("cosine").unwrap(), "IP");
        assert_eq!(map_milvus_metric_type("angular").unwrap(), "IP");
        assert!(map_milvus_metric_type("nope").is_err());
    }

    // ── Exact-match numeric / bool / non-scalar arms ───────────────────────

    #[test]
    fn exact_match_int_float_bool() {
        assert_eq!(
            build_milvus_filter("n", "match", &json!({"value":5})).unwrap(),
            "n == 5"
        );
        assert_eq!(
            build_milvus_filter("n", "match", &json!({"value":1.5})).unwrap(),
            "n == 1.5"
        );
        assert_eq!(
            build_milvus_filter("flag", "match", &json!({"value":true})).unwrap(),
            "flag == true"
        );
    }

    // ── #218: scalar (payload) index coverage ──────────────────────────────
    //
    // Before the fix, `create_index` posted a single `indexParams` entry for
    // `vector` and nothing else, so every filtered query was resolved by a
    // brute-force scan of the unindexed scalar column. No recall assertion can
    // see that (an unindexed scan returns the SAME rows), so these tests pin
    // the structural property instead.

    fn dataset_with_schema(schema: serde_json::Value) -> Dataset {
        Dataset::new(crate::config::DatasetConfig {
            name: "t".into(),
            dataset_type: Some("tar".into()),
            path: json!("t/"),
            distance: Some("l2".into()),
            vector_size: Some(8),
            vector_count: Some(1),
            link: None,
            schema: Some(schema),
            description: None,
        })
    }

    /// THE invariant behind #218: a schema field that becomes a COLUMN must
    /// also get an INDEX. Both passes read `milvus_field_kind`, so this asserts
    /// the two can never diverge again for any schema type we support.
    #[test]
    fn every_materialised_column_is_also_indexed() {
        let schema = json!({
            "kw": "keyword", "txt": "text", "uid": "uuid", "n": "int",
            "f": "float", "flag": "bool", "ts": "datetime", "labels": "keyword",
            "loc": "geo", "weird": "not-a-type",
        });
        let plan = scalar_index_plan(&dataset_with_schema(schema.clone()));
        let indexed: std::collections::BTreeSet<&str> =
            plan.iter().map(|(f, _)| f.as_str()).collect();

        for (field, ty) in schema.as_object().unwrap() {
            let kind = milvus_field_kind(field, ty.as_str().unwrap());
            // The invariant moved from `has_column <=> indexed` to
            // `wants_index <=> indexed`. That is INCOMPARABLE to the old rule,
            // not stronger: `wants_index -> has_column` strictly, so neither
            // implies the other, and the state `(column: Some, index: None)`
            // that was previously unrepresentable is now accepted here. What
            // stops that becoming a hole is not this assertion but
            // `only_geo_is_a_deliberately_unindexed_column` below, which pins
            // the exception SET to exactly {geo} — a newly added type that
            // forgot its index would otherwise pass silently.
            let wants_index = kind.and_then(|k| k.index_type).is_some();
            assert_eq!(
                wants_index,
                indexed.contains(field.as_str()),
                "field '{}' ({}) wants_index={} but indexed={} — a column whose \
                 mapping names an index must get it, and one whose mapping says \
                 None must not (#218, #223)",
                field,
                ty,
                wants_index,
                indexed.contains(field.as_str())
            );
        }
        // `geo` is materialised (issue #223) but deliberately NOT indexed, so
        // it must be absent from the plan; an unknown type gets neither.
        assert!(!indexed.contains("loc"));
        assert!(!indexed.contains("weird"));
    }

    /// The exception set is asserted, not just documented.
    ///
    /// `index_type: Option` made `(column: Some, index_type: None)` a
    /// representable state, so "the exception has to be spelled out with its
    /// reason" became a comment convention rather than a check: a new schema
    /// type that forgot its index would be accepted by
    /// `every_materialised_column_is_also_indexed` without comment. This pins
    /// the set to exactly {geo} over every type the engine claims to support, so
    /// adding a second unindexed column is a deliberate edit here.
    #[test]
    fn only_geo_is_a_deliberately_unindexed_column() {
        // Every schema type this engine maps, plus the multi-valued spelling.
        let types = [
            "int", "float", "keyword", "text", "uuid", "bool", "datetime", "geo",
        ];
        let unindexed: Vec<&str> = types
            .iter()
            .filter(|ty| milvus_field_kind("f", ty).is_some_and(|k| k.index_type.is_none()))
            .copied()
            .collect();
        assert_eq!(
            unindexed,
            ["geo"],
            "exactly one materialised column may be deliberately unindexed (#223); \
             adding another needs its own reason and a live measurement"
        );
        // `labels` (Array of VarChar) must not sneak in as a second one.
        assert!(milvus_field_kind("labels", "keyword")
            .unwrap()
            .index_type
            .is_some());
    }

    /// The per-type index choice, verified live against milvusdb/milvus:v2.6.19
    /// (see `milvus_field_kind`'s doc comment for the full acceptance matrix).
    #[test]
    fn scalar_index_types_per_field_type() {
        let expect = |field: &str, ty: &str| milvus_field_kind(field, ty).unwrap();
        // Point + range predicates over one structure; also what AUTOINDEX picks.
        assert_eq!(expect("kw", "keyword").index_type, Some("INVERTED"));
        assert_eq!(expect("b", "text").index_type, Some("INVERTED"));
        assert_eq!(expect("u", "uuid").index_type, Some("INVERTED"));
        assert_eq!(expect("n", "int").index_type, Some("INVERTED"));
        assert_eq!(expect("f", "float").index_type, Some("INVERTED"));
        // datetime is an Int64 epoch column, filtered by range → same as int.
        assert_eq!(expect("ts", "datetime").index_type, Some("INVERTED"));
        assert_eq!(expect("ts", "datetime").data_type, "Int64");
        // Bool has exactly 2 distinct values → BITMAP (STL_SORT is rejected on
        // Bool by the server).
        assert_eq!(expect("flag", "bool").index_type, Some("BITMAP"));
        assert_eq!(expect("flag", "bool").data_type, "Bool");
        // Multi-valued `labels` is an Array(VarChar) (#88); only INVERTED and
        // BITMAP are accepted there, and cardinality is unbounded → INVERTED.
        assert_eq!(expect("labels", "keyword").data_type, "Array");
        assert_eq!(expect("labels", "keyword").index_type, Some("INVERTED"));
        // A scalar keyword named anything else stays a VarChar.
        assert_eq!(expect("color", "keyword").data_type, "VarChar");
        // Native geospatial column: OGC WKT in a `Geometry` field, and
        // DELIBERATELY UNINDEXED — Milvus' only Geometry index (RTREE) prunes
        // with a box smaller than the cap and silently drops true hits (#223,
        // see `milvus_field_kind`). This is the sole column-without-index in the
        // engine, and it is asserted so re-adding the index is a deliberate act.
        assert_eq!(expect("loc", "geo").data_type, "Geometry");
        assert_eq!(expect("loc", "geo").index_type, None);
        // Every OTHER materialised type must still carry an index.
        for ty in [
            "int", "float", "keyword", "text", "uuid", "bool", "datetime",
        ] {
            assert!(expect("f", ty).index_type.is_some(), "{ty}");
        }
        // An unknown type is still not materialised.
        assert!(milvus_field_kind("loc", "nope").is_none());
    }

    /// `id`/`vector` are excluded (PK is implicitly indexed, `vector` gets the
    /// ANN index) — mirrors upstream's `field.name not in ("id", "vector")`.
    #[test]
    fn scalar_index_plan_skips_id_and_vector() {
        let plan = scalar_index_plan(&dataset_with_schema(
            json!({"id": "int", "vector": "int", "a": "keyword"}),
        ));
        assert_eq!(plan, vec![("a".to_string(), "INVERTED")]);
    }

    #[test]
    fn scalar_index_plan_empty_for_unfiltered_dataset() {
        assert!(scalar_index_plan(&dataset_with_schema(json!({}))).is_empty());
    }

    /// The zero-row trap that made the #218 fix inert until the flush landed:
    /// before a collection is flushed, EVERY index (the HNSW one included)
    /// reports `Finished` with `indexedRows = totalRows = 0` and stays that way
    /// indefinitely, while queries brute-force the growing segments. Verified
    /// live on v2.6.19. So "built" must mean "covers every row", not "state is
    /// Finished".
    #[test]
    fn index_is_built_requires_row_coverage_not_just_finished() {
        let p = |state: &str, indexed: i64, pending: i64, total: i64| IndexProgress {
            field_name: "a".into(),
            index_type: "INVERTED".into(),
            state: state.into(),
            indexed_rows: indexed,
            pending_rows: pending,
            total_rows: total,
        };
        // The exact reply a pre-flush collection gives: Finished over nothing.
        assert!(!p("Finished", 0, 0, 0).is_built(true));
        // Still building.
        assert!(!p("InProgress", 0, 100, 100).is_built(true));
        // Built but only partially covering.
        assert!(!p("Finished", 60, 40, 100).is_built(true));
        // Genuinely usable.
        assert!(p("Finished", 100, 0, 100).is_built(true));
        // Fully indexed WHILE background compaction re-queues segments: the
        // verbatim v2.6.19 reply for a settled 1M-row collection. Must count as
        // built, or every large upload hangs until the timeout.
        assert!(p("Finished", 1_000_000, 847_872, 1_000_000).is_built(true));
        // An empty upload legitimately has no rows to cover, so the row
        // requirement is waived rather than deadlocking the wait loop.
        assert!(p("Finished", 0, 0, 0).is_built(false));
    }

    #[test]
    fn parse_index_progress_reads_row_counts() {
        let row = json!({
            "fieldName": "color", "indexType": "INVERTED", "indexState": "Finished",
            "indexedRows": 1000, "pendingRows": 0, "totalRows": 1000,
        });
        let p = parse_index_progress(&row);
        assert_eq!(p.field_name, "color");
        assert_eq!(p.index_type, "INVERTED");
        assert_eq!(p.indexed_rows, 1000);
        assert_eq!(p.total_rows, 1000);
        assert!(p.is_built(true));
    }

    /// Only a genuine "the field is already indexed" reply is tolerated. Both
    /// strings are the verbatim v2.6.19 messages for re-creating an index on a
    /// covered field; anything else must fail loudly rather than leave a column
    /// unindexed.
    #[test]
    fn already_indexed_error_detection() {
        assert!(is_already_indexed_error(
            "at most one distinct index is allowed per field: invalid parameter"
        ));
        assert!(is_already_indexed_error(
            "CreateIndex failed: creating multiple indexes on same field is not supported: invalid parameter"
        ));
        assert!(is_already_indexed_error("index already exist"));
        // Real failures must NOT be swallowed.
        assert!(!is_already_indexed_error(
            "bitmap index are only supported on bool, int, string and array field: invalid parameter"
        ));
        assert!(!is_already_indexed_error("collection not found"));
    }

    #[test]
    fn exact_match_array_value_is_none() {
        // #121: the scalar exact-match arm now guards non-scalars; a JSON array
        // value is dropped (None), matching qdrant/redis/valkey/vectorsets (was
        // previously Display-formatted verbatim as `n == [1,2]`).
        assert!(build_milvus_filter("n", "match", &json!({"value":[1,2]})).is_none());
    }
}
