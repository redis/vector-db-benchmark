//! Cross-engine coverage guard for the #219 filter-drop rule.
//!
//! `query_filter.rs` makes "the dataset declared a filter and the builder
//! produced nothing" a hard error. That is only safe if the builders can in fact
//! express the filters our SHIPPED datasets emit — otherwise the guard turns a
//! silently-wrong run into a broken run.
//!
//! This module runs every shipped condition shape through every engine's filter
//! builder, at the same choke point production uses
//! ([`crate::query_filter::resolve`]), and pins the verdict. Adding an engine or
//! narrowing a builder flips a verdict here and fails the build.
//!
//! ## Where the shapes come from
//!
//! * `Match{Str,Int}`/`OrMatch`: read straight out of the shipped `tests.jsonl`
//!   of `h-and-m-2048-angular-filters` (10 000/10 000 rows are
//!   `{"and":[{kw:{"match":{"value":str}}}]}`) and
//!   `random-100-match-kw-small-vocab-filters` (6 667 `and`, 3 333 two-leaf
//!   `or`). The int/float/geo tarballs are the same generator with a different
//!   payload type, and the canonical spelling of all three operators is fixed by
//!   `v0/engine/base_client/parser.py` (`FilterType` = `match` | `range` |
//!   `geo`, geo criteria `{lon, lat, radius}` — `v0/tests/engine/clients/qdrant/
//!   test_qdrant_parser.py`).
//! * `MatchAny*`, `MatchBool`, `RangeDatetime`, `RangeIntLt`: emitted verbatim by
//!   `src/bin/generate_dataset.rs` for the locally generated
//!   `synthetic-filter-32` and `synthetic-selectivity-32`.
//!
//! The `schema` on each shape is the one `datasets.json` declares for the
//! dataset that emits it; Vertex needs it to pick numeric vs string restricts.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::query_filter::{resolve, QueryFilter};

/// A condition shape that a shipped dataset actually puts on the wire.
struct ShippedShape {
    /// Identifier used in assertion messages.
    id: &'static str,
    /// Datasets from `datasets/datasets.json` that emit this shape.
    datasets: &'static str,
    conditions: Value,
    /// The emitting dataset's declared schema (field -> type).
    schema: &'static [(&'static str, &'static str)],
}

fn shipped_shapes() -> Vec<ShippedShape> {
    vec![
        ShippedShape {
            id: "match_str",
            datasets: "h-and-m-*-filters, arxiv-*-filters, random-match-keyword-*, \
                       random-100-match-kw-small-vocab-filters, random-768-100-tenants",
            conditions: json!({"and": [{"a": {"match": {"value": "giNHA"}}}]}),
            schema: &[("a", "keyword")],
        },
        ShippedShape {
            id: "or_match_str",
            datasets: "random-100-match-kw-small-vocab-filters (3 333 of 10 000 queries)",
            conditions: json!({"or": [
                {"a": {"match": {"value": "pPZIB"}}},
                {"a": {"match": {"value": "qOGCZ"}}}
            ]}),
            schema: &[("a", "keyword")],
        },
        ShippedShape {
            id: "match_int",
            datasets: "random-match-int-100-angular-filters, random-match-int-2048-angular-filters",
            conditions: json!({"and": [{"a": {"match": {"value": 80}}}]}),
            schema: &[("a", "int")],
        },
        ShippedShape {
            id: "range_float",
            datasets: "random-range-100-angular-filters, random-range-2048-angular-filters, \
                       laion-small-clip",
            conditions: json!({"and": [{"a": {"range": {"gt": 0.1, "lt": 0.9}}}]}),
            schema: &[("a", "float")],
        },
        ShippedShape {
            id: "geo_radius",
            datasets: "random-geo-radius-100-angular-filters, \
                       random-geo-radius-2048-angular-filters",
            conditions: json!({"and": [{"a": {"geo": {
                "lon": 116.0, "lat": -52.0, "radius": 326_341.0
            }}}]}),
            schema: &[("a", "geo")],
        },
        ShippedShape {
            id: "match_any_keyword",
            datasets: "synthetic-filter-32",
            conditions: json!({"and": [{"color": {"match": {"any": ["red", "blue"]}}}]}),
            schema: &[("color", "keyword")],
        },
        ShippedShape {
            id: "match_any_int",
            datasets: "synthetic-filter-32",
            conditions: json!({"and": [{"size": {"match": {"any": [1, 2, 3]}}}]}),
            schema: &[("size", "int")],
        },
        ShippedShape {
            id: "match_bool",
            datasets: "synthetic-filter-32",
            conditions: json!({"and": [{"flag": {"match": {"value": true}}}]}),
            schema: &[("flag", "bool")],
        },
        ShippedShape {
            id: "range_datetime",
            datasets: "synthetic-filter-32",
            conditions: json!({"and": [{"ts": {"range": {
                "gte": "2021-04-11T00:00:00+00:00", "lt": "2021-10-28T00:00:00+00:00"
            }}}]}),
            schema: &[("ts", "datetime")],
        },
        ShippedShape {
            id: "range_int_lt",
            datasets: "synthetic-selectivity-32",
            conditions: json!({"and": [{"rank": {"range": {"lt": 20}}}]}),
            schema: &[("rank", "int")],
        },
    ]
}

/// Resolve one shape through one engine, at the production choke point.
///
/// `Ok(true)` = a filter goes on the wire. `Ok(false)` is unreachable for a
/// declared filter by construction (that is what `resolve` guarantees); `Err` is
/// the guard firing.
type EngineResolver = fn(&Value, &HashMap<String, String>) -> Result<bool, String>;

/// One entry per engine that reads `conditions`. `resolve` is the same function
/// the engine's `search()` calls into.
fn engines() -> Vec<(&'static str, EngineResolver)> {
    vec![
        ("redis", |c, _| {
            through("Redis", c, |v| {
                Ok(super::redis::parse_conditions(v).map(|_| ()))
            })
        }),
        // Dragonfly reuses redis.rs's RediSearch builder verbatim.
        ("dragonfly", |c, _| {
            through("Dragonfly", c, |v| {
                Ok(super::redis::parse_conditions(v).map(|_| ()))
            })
        }),
        ("valkey", |c, _| {
            through("Valkey", c, |v| {
                Ok(super::valkey::parse_conditions(v).map(|_| ()))
            })
        }),
        ("vectorsets", |c, _| {
            through("VectorSets", c, |v| {
                Ok(super::vectorsets::build_filter_expression(v))
            })
        }),
        ("elasticsearch", |c, _| {
            through("Elasticsearch", c, |v| {
                Ok(super::elasticsearch::parse_es_conditions(v))
            })
        }),
        ("opensearch", |c, _| {
            through("OpenSearch", c, |v| {
                Ok(super::opensearch::parse_os_conditions(v))
            })
        }),
        ("weaviate", |c, _| {
            through("Weaviate", c, |v| {
                Ok(super::weaviate::parse_weaviate_conditions(v))
            })
        }),
        ("milvus", |c, _| {
            through("Milvus", c, |v| {
                Ok(super::milvus::parse_milvus_conditions(v))
            })
        }),
        ("mongodb", |c, _| {
            through("MongoDB", c, |v| {
                Ok(super::mongodb_engine::parse_mongo_conditions(v).map(|_| ()))
            })
        }),
        ("pgvector", |c, _| {
            through("pgvector", c, |v| {
                Ok(super::pgvector::parse_pg_conditions(v, 3).map(|_| ()))
            })
        }),
        ("turbopuffer", |c, _| {
            through("Turbopuffer", c, |v| {
                Ok(super::turbopuffer::parse_turbopuffer_filter(v))
            })
        }),
        ("qdrant", |c, _| {
            through("Qdrant", c, |v| {
                super::qdrant::parse_qdrant_conditions(v).map(|f| f.map(|_| ()))
            })
        }),
        // Chroma splits the tree over `where` + `where_document`; either half may
        // be empty, so the pair is what must be non-empty — same rule the engine
        // applies in `search()`.
        ("chroma", |c, _| {
            through("Chroma", c, |v| {
                Ok(
                    match (
                        super::chroma::build_chroma_where(v),
                        super::chroma::build_chroma_where_document(v),
                    ) {
                        (None, None) => None,
                        _ => Some(()),
                    },
                )
            })
        }),
        // KiviDB's builder has no "produced nothing" state: it returns the
        // prefilter or an error. A declared filter rendering the match-all
        // prefilter IS the drop, so it counts as unexpressible here.
        ("kividb", |c, _| {
            through("KiviDB", c, |v| {
                let rendered = super::kividb::kividb_filter::parse_conditions(v)?;
                Ok((rendered != super::kividb::kividb_filter::MATCH_ALL).then_some(()))
            })
        }),
        ("vertex", |c, schema| {
            through("Vertex AI", c, |v| {
                let filter = super::vertex::parse_vertex_filter(v, schema)?;
                Ok((!filter.is_empty()).then_some(()))
            })
        }),
    ]
}

fn through<T>(
    engine: &str,
    conditions: &Value,
    parse: impl FnOnce(&Value) -> Result<Option<T>, String>,
) -> Result<bool, String> {
    resolve(engine, 0, Some(conditions), parse).map(|f: QueryFilter<T>| f.is_filtered())
}

/// Engine/shape pairs where the builder legitimately cannot express the shape,
/// so the #219 guard fires and the run stops instead of reporting a recall for a
/// filter that was never applied.
///
/// Every entry is a real filter gap tracked by its own issue — this PR does not
/// close them, it stops them being silent. Anything NOT listed here must be
/// expressible, and anything listed here that starts working must be removed
/// (the test asserts both directions).
const KNOWN_GAPS: &[(&str, &str, &str)] = &[
    // ── #223: geo dropped by 5 engines on the 4 shipped geo datasets ────────
    // Before this PR each of these returned `None` and the query ran with NO
    // filter at all, scored against geo-filtered ground truth. The guard does
    // not fix the gap; it stops the wrong number. Redis, Valkey, Dragonfly,
    // Elasticsearch, OpenSearch, Weaviate, pgvector and Qdrant do express geo.
    (
        "vectorsets",
        "geo_radius",
        "#223 — was silently dropped, now refused",
    ),
    (
        "milvus",
        "geo_radius",
        "#223 — was silently dropped, now refused",
    ),
    (
        "turbopuffer",
        "geo_radius",
        "#223 — was silently dropped, now refused",
    ),
    (
        "chroma",
        "geo_radius",
        "#223 — was silently dropped, now refused",
    ),
    (
        "mongodb",
        "geo_radius",
        "#223 — was silently dropped, now refused",
    ),
    // Already an explicit `Err` on master — behaviour unchanged by this PR.
    (
        "kividb",
        "geo_radius",
        "#223 — KiviDB has no GEO field type (already Err)",
    ),
    (
        "vertex",
        "geo_radius",
        "#223 — Vertex restrictions have no geo namespace (already Err)",
    ),
    // ── Vertex AI: pre-existing hard errors, unchanged by this PR ───────────
    // `parse_vertex_filter` has returned `Err` for these since it was written
    // (it is the model this PR generalises), so Vertex already refused these
    // datasets rather than running them unfiltered. Listed so the table is
    // complete and so implementing any of them has to update this file.
    (
        "vertex",
        "or_match_str",
        "pre-existing Err: cross-field OR unimplemented (same-field OR could \
         map to a restrict allowList)",
    ),
    (
        "vertex",
        "match_any_int",
        "pre-existing Err: a numeric IN-list is not a single numeric restrict",
    ),
    (
        "vertex",
        "match_bool",
        "pre-existing Err: no bool restrict type",
    ),
    (
        "vertex",
        "range_datetime",
        "pre-existing Err: numeric restricts only, no datetime bound (see #232)",
    ),
];

fn is_known_gap(engine: &str, shape: &str) -> Option<&'static str> {
    KNOWN_GAPS
        .iter()
        .find(|(e, s, _)| *e == engine && *s == shape)
        .map(|(_, _, why)| *why)
}

/// The evidence table: every shipped condition shape against every engine.
#[test]
fn every_shipped_condition_shape_is_expressible_or_a_tracked_gap() {
    let mut rows: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for shape in shipped_shapes() {
        let schema: HashMap<String, String> = shape
            .schema
            .iter()
            .map(|(f, t)| (f.to_string(), t.to_string()))
            .collect();
        for (engine, resolver) in engines() {
            let verdict = resolver(&shape.conditions, &schema);
            let gap = is_known_gap(engine, shape.id);
            match (&verdict, gap) {
                (Ok(true), None) => rows.push(format!("{:<14} {:<18} FILTERED", engine, shape.id)),
                (Ok(true), Some(why)) => failures.push(format!(
                    "{engine}/{} is listed as a known gap ({why}) but now resolves — remove it \
                     from KNOWN_GAPS",
                    shape.id
                )),
                (Ok(false), _) => failures.push(format!(
                    "{engine}/{} resolved to NO filter for a declared condition; \
                     query_filter::resolve must never do this",
                    shape.id
                )),
                (Err(e), Some(why)) => rows.push(format!(
                    "{:<14} {:<18} REJECTED ({why}) [{}]",
                    engine,
                    shape.id,
                    e.lines().next().unwrap_or("")
                )),
                (Err(e), None) => failures.push(format!(
                    "{engine} cannot express shipped shape `{}` (from {}), and it is not a \
                     tracked gap. The #219 guard would fail this dataset. Either implement the \
                     condition or add it to KNOWN_GAPS with its issue.\n    {e}",
                    shape.id, shape.datasets
                )),
            }
        }
    }

    println!("{}", rows.join("\n"));
    assert!(failures.is_empty(), "\n{}", failures.join("\n\n"));
}

/// The other half of the guard: an unfiltered shipped dataset must not error.
///
/// `random_keywords_1m_vocab_10_no_filters` and the other `*_no_filters`
/// tarballs spell every row `"conditions": null`, which `compound_reader.rs`
/// hands over as `Some(Value::Null)`. An `is_some()`-keyed guard would fail all
/// 10 000 queries on all 15 engines.
#[test]
fn the_shipped_no_filters_datasets_stay_unfiltered_on_every_engine() {
    let schema = HashMap::new();
    for (engine, resolver) in engines() {
        for absent in [Value::Null, json!({})] {
            let verdict = resolver(&absent, &schema);
            // `resolve` short-circuits before the builder, so the builder is
            // never consulted and the query is simply unfiltered.
            assert_eq!(
                resolve::<()>(engine, 0, Some(&absent), |_| unreachable!(
                    "{engine}: the builder must not be consulted for absent conditions"
                ))
                .map(|f| f.is_filtered()),
                Ok(false),
                "{engine} treated {absent} as a declared filter"
            );
            let _ = verdict;
        }
        assert_eq!(
            resolve::<()>(engine, 0, None, |_| unreachable!()).map(|f| f.is_filtered()),
            Ok(false)
        );
    }
}
