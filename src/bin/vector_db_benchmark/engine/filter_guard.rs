//! Cross-engine coverage guard for the #219 filter-drop rule.
//!
//! `query_filter.rs` makes "the dataset declared a filter and the builder
//! produced nothing" a hard error. That is only safe if the builders can in fact
//! express the filters our SHIPPED datasets emit — otherwise the guard turns a
//! silently-wrong run into a broken run.
//!
//! This module runs every shipped condition shape through every engine's filter
//! builder, at the same choke point production uses (`query_filter::resolve`),
//! and pins the verdict. Adding an engine or narrowing a builder flips a verdict
//! here and fails the build.
//!
//! ## What it can and cannot catch
//!
//! It catches a builder that produces **nothing** for a shape, and — for the
//! multi-leaf shapes — a builder that silently keeps a *subset* of the leaves,
//! by comparing the rendered filter against the render of every proper
//! sub-multiset of the same leaves ([`no_shipped_multi_leaf_filter_loses_a_leaf`]).
//! It cannot catch a builder that renders a match-all filter for a real
//! condition, and it only knows about the shapes listed here.
//!
//! ## Where the shapes come from
//!
//! A census of the shipped `tests.jsonl` files, not a sample. Locally verified
//! on the two tarballs present in this checkout:
//!
//! * `h-and-m-2048-angular-filters`: 10 000/10 000 rows are a single-leaf
//!   `{"and":[{kw:{"match":{"value":str}}}]}`;
//! * `random-100-match-kw-small-vocab-filters`: 3 334 single-leaf `and`,
//!   3 333 **two-leaf cross-field** `and`, 3 333 two-leaf same-field `or`.
//!
//! Nine of the sixteen filter-bearing shipped datasets emit that two-leaf `and`
//! — 29 997 queries in total — which is why the multi-leaf shapes below are not
//! optional. The int/float/geo tarballs are the same generator over a different
//! payload type, and the operator vocabulary is fixed at `match` | `range` |
//! `geo` by `v0/engine/base_client/parser.py` (geo criteria `{lon, lat, radius}`
//! — `v0/tests/engine/clients/qdrant/test_qdrant_parser.py`).
//!
//! `match_any_*`, `match_bool`, `range_datetime` and `range_int_lt` are emitted
//! verbatim by `src/bin/generate_dataset.rs` for the locally generated
//! `synthetic-filter-32` and `synthetic-selectivity-32`.
//!
//! The `schema` on each shape is the one `datasets.json` declares for the
//! dataset that emits it; Vertex needs it to pick numeric vs string restricts.

use std::collections::HashMap;

use serde_json::{json, Value};

use vector_db_benchmark::query_filter::{resolve, QueryFilter};

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

impl ShippedShape {
    /// Top-level connective and leaves, for the sub-multiset expansion. `None`
    /// for a single-leaf shape, which cannot lose a leaf and stay a filter.
    fn split(&self) -> Option<(&'static str, Vec<Value>)> {
        for conn in ["and", "or"] {
            if let Some(items) = self.conditions.get(conn).and_then(|v| v.as_array()) {
                if items.len() > 1 {
                    return Some((conn, items.clone()));
                }
            }
        }
        None
    }
}

const KW: &[(&str, &str)] = &[("a", "keyword"), ("b", "keyword")];
const INT: &[(&str, &str)] = &[("a", "int"), ("b", "int")];
const FLOAT: &[(&str, &str)] = &[("a", "float"), ("b", "float")];
const GEO: &[(&str, &str)] = &[("a", "geo"), ("b", "geo")];

fn shipped_shapes() -> Vec<ShippedShape> {
    let geo_a = json!({"a": {"geo": {"lon": 116.0, "lat": -52.0, "radius": 326_341.0}}});
    let geo_b = json!({"b": {"geo": {"lon": 12.0, "lat": 40.0, "radius": 100_000.0}}});
    vec![
        // ── keyword family: h-and-m, arxiv, random-match-keyword-*, tenants ──
        ShippedShape {
            id: "and_match_str",
            datasets: "h-and-m-*-filters, arxiv-*-filters, random-match-keyword-*, \
                       random-100-match-kw-small-vocab-filters (3 334/10 000), \
                       random-768-100-tenants",
            conditions: json!({"and": [{"a": {"match": {"value": "giNHA"}}}]}),
            schema: KW,
        },
        ShippedShape {
            id: "and_match_str_x2",
            datasets: "random-match-keyword-* and 8 sibling tarballs (3 333/10 000 each)",
            conditions: json!({"and": [
                {"a": {"match": {"value": "qOGCZ"}}},
                {"b": {"match": {"value": "pPZIB"}}}
            ]}),
            schema: KW,
        },
        ShippedShape {
            id: "or_match_str_x2",
            datasets: "random-100-match-kw-small-vocab-filters (3 333/10 000)",
            conditions: json!({"or": [
                {"a": {"match": {"value": "pPZIB"}}},
                {"a": {"match": {"value": "qOGCZ"}}}
            ]}),
            schema: KW,
        },
        // ── int family: random-match-int-* ──────────────────────────────────
        ShippedShape {
            id: "and_match_int",
            datasets: "random-match-int-100-angular-filters, random-match-int-2048-angular-filters",
            conditions: json!({"and": [{"a": {"match": {"value": 80}}}]}),
            schema: INT,
        },
        ShippedShape {
            id: "and_match_int_x2",
            datasets: "random-match-int-* (the two-leaf third)",
            conditions: json!({"and": [
                {"a": {"match": {"value": 80}}},
                {"b": {"match": {"value": 2}}}
            ]}),
            schema: INT,
        },
        ShippedShape {
            id: "or_match_int_x2",
            datasets: "random-match-int-* (the or third)",
            conditions: json!({"or": [
                {"a": {"match": {"value": 80}}},
                {"a": {"match": {"value": 2}}}
            ]}),
            schema: INT,
        },
        // ── float/range family: random-range-*, laion-small-clip ────────────
        ShippedShape {
            id: "and_range_float",
            datasets: "random-range-100-angular-filters, random-range-2048-angular-filters, \
                       laion-small-clip",
            conditions: json!({"and": [{"a": {"range": {"gt": 0.1, "lt": 0.9}}}]}),
            schema: FLOAT,
        },
        ShippedShape {
            id: "and_range_float_x2",
            datasets: "random-range-* (the two-leaf third)",
            conditions: json!({"and": [
                {"a": {"range": {"gt": 0.1, "lt": 0.9}}},
                {"b": {"range": {"gte": 0.2, "lte": 0.8}}}
            ]}),
            schema: FLOAT,
        },
        ShippedShape {
            id: "range_gt_only",
            datasets: "random-range-*, laion-small-clip (one-sided bounds)",
            conditions: json!({"and": [{"a": {"range": {"gt": 0.5}}}]}),
            schema: FLOAT,
        },
        // ── geo family: random-geo-radius-* ─────────────────────────────────
        ShippedShape {
            id: "and_geo",
            datasets: "random-geo-radius-100-angular-filters, \
                       random-geo-radius-2048-angular-filters",
            conditions: json!({"and": [geo_a.clone()]}),
            schema: GEO,
        },
        ShippedShape {
            id: "and_geo_x2",
            datasets: "random-geo-radius-* (the two-leaf third)",
            conditions: json!({"and": [geo_a.clone(), geo_b]}),
            schema: GEO,
        },
        ShippedShape {
            id: "or_geo_x2",
            datasets: "random-geo-radius-* (the or third, same field)",
            conditions: json!({"or": [
                geo_a,
                {"a": {"geo": {"lon": 120.0, "lat": -50.0, "radius": 50_000.0}}}
            ]}),
            schema: GEO,
        },
        // ── locally generated sets ──────────────────────────────────────────
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

/// Resolve one shape through one engine, at the production choke point, and
/// return a **rendering** of the filter that would go on the wire.
///
/// The rendering must be deterministic and must change when a leaf changes,
/// because [`no_shipped_multi_leaf_filter_loses_a_leaf`] compares renderings.
/// `Ok(None)` is unreachable for a declared filter by construction (that is what
/// `resolve` guarantees); `Err` is the guard firing.
type EngineResolver = fn(&Value, &HashMap<String, String>) -> Result<Option<String>, String>;

/// Deterministic rendering of a `(query, params)` RediSearch filter: the params
/// live in a `HashMap`, whose `Debug` order is not stable, so they are sorted.
fn render_redisearch<V: std::fmt::Debug>(f: &(String, HashMap<String, V>)) -> String {
    let mut params: Vec<String> = f.1.iter().map(|(k, v)| format!("{k}={v:?}")).collect();
    params.sort();
    format!("{} [{}]", f.0, params.join(","))
}

/// One entry per engine that reads `conditions`. Each closure calls the same
/// builder the engine's `search()` calls.
fn engines() -> Vec<(&'static str, EngineResolver)> {
    vec![
        ("redis", |c, _| {
            through("Redis", c, |v| {
                Ok(super::redis::parse_conditions(v).map(|f| render_redisearch(&f)))
            })
        }),
        // Dragonfly reuses redis.rs's RediSearch builder verbatim.
        ("dragonfly", |c, _| {
            through("Dragonfly", c, |v| {
                Ok(super::redis::parse_conditions(v).map(|f| render_redisearch(&f)))
            })
        }),
        ("valkey", |c, _| {
            through("Valkey", c, |v| {
                Ok(super::valkey::parse_conditions(v).map(|f| render_redisearch(&f)))
            })
        }),
        ("vectorsets", |c, _| {
            through("VectorSets", c, |v| {
                Ok(super::vectorsets::build_filter_expression(v))
            })
        }),
        ("elasticsearch", |c, _| {
            through("Elasticsearch", c, |v| {
                Ok(super::elasticsearch::parse_es_conditions(v).map(|f| f.to_string()))
            })
        }),
        ("opensearch", |c, _| {
            through("OpenSearch", c, |v| {
                Ok(super::opensearch::parse_os_conditions(v).map(|f| f.to_string()))
            })
        }),
        ("weaviate", |c, _| {
            through("Weaviate", c, |v| {
                Ok(super::weaviate::parse_weaviate_conditions(v).map(|f| f.to_string()))
            })
        }),
        ("milvus", |c, _| {
            through("Milvus", c, |v| {
                Ok(super::milvus::parse_milvus_conditions(v))
            })
        }),
        ("mongodb", |c, _| {
            through("MongoDB", c, |v| {
                Ok(super::mongodb_engine::parse_mongo_conditions(v).map(|d| format!("{d:?}")))
            })
        }),
        ("pgvector", |c, _| {
            through("pgvector", c, |v| {
                Ok(super::pgvector::parse_pg_conditions(v, 3)
                    .map(|(sql, vals)| format!("{sql} {vals:?}")))
            })
        }),
        ("turbopuffer", |c, _| {
            through("Turbopuffer", c, |v| {
                Ok(super::turbopuffer::parse_turbopuffer_filter(v).map(|f| f.to_string()))
            })
        }),
        ("qdrant", |c, _| {
            through("Qdrant", c, |v| {
                super::qdrant::parse_qdrant_conditions(v).map(|f| f.map(|f| format!("{f:?}")))
            })
        }),
        // Chroma splits the tree over `where` + `where_document`; either half may
        // legitimately be empty, so the PAIR is what must be non-empty — the same
        // rule the engine applies in `search()`.
        ("chroma", |c, _| {
            through("Chroma", c, |v| {
                Ok(
                    match (
                        super::chroma::build_chroma_where(v),
                        super::chroma::build_chroma_where_document(v),
                    ) {
                        (None, None) => None,
                        (w, d) => Some(format!("where={w:?} where_document={d:?}")),
                    },
                )
            })
        }),
        // KiviDB's builder has no "produced nothing" state: it returns the
        // prefilter or an error, and renders match-all for the shapes that would
        // otherwise have vanished. The search path maps that render back to
        // "produced nothing" before resolving, and so does this.
        ("kividb", |c, _| {
            through("KiviDB", c, |v| {
                let rendered = super::kividb::kividb_filter::parse_conditions(v)?;
                Ok((rendered != super::kividb::kividb_filter::MATCH_ALL).then_some(rendered))
            })
        }),
        ("vertex", |c, schema| {
            through("Vertex AI", c, |v| {
                let filter = super::vertex::parse_vertex_filter(v, schema)?;
                Ok((!filter.is_empty()).then(|| format!("{filter:?}")))
            })
        }),
    ]
}

fn through(
    engine: &str,
    conditions: &Value,
    parse: impl FnOnce(&Value) -> Result<Option<String>, String>,
) -> Result<Option<String>, String> {
    resolve(engine, 0, Some(conditions), parse).map(QueryFilter::into_inner)
}

/// Engine/shape pairs where the builder legitimately cannot express the shape,
/// so the #219 guard fires and the run stops instead of reporting a recall for a
/// filter that was never applied.
///
/// Every entry is a real filter gap tracked by its own issue — this PR does not
/// close them, it stops them being silent. Anything NOT listed here must be
/// expressible, anything listed here that starts working must be removed, and
/// every entry must name an engine and a shape that actually exist — all three
/// are asserted, so an exemption cannot outlive the thing it exempts.
const KNOWN_GAPS: &[(&str, &str, &str)] = &[
    // ── #223: geo dropped by 5 engines on the 2 shipped geo-FILTER datasets ──
    // (four datasets are geo-*named*; the two `-no-filters` twins are
    // `"conditions": null` throughout and stay green).
    // Before this PR each of these returned `None` and the query ran with NO
    // filter at all, scored against geo-filtered ground truth. The guard does
    // not fix the gap; it stops the wrong number. Redis, Valkey, Dragonfly,
    // Elasticsearch, OpenSearch, Weaviate, pgvector and Qdrant express geo.
    (
        "vectorsets",
        "and_geo",
        "#223 — was silently dropped, now refused",
    ),
    ("vectorsets", "and_geo_x2", "#223"),
    ("vectorsets", "or_geo_x2", "#223"),
    (
        "milvus",
        "and_geo",
        "#223 — was silently dropped, now refused",
    ),
    ("milvus", "and_geo_x2", "#223"),
    ("milvus", "or_geo_x2", "#223"),
    (
        "turbopuffer",
        "and_geo",
        "#223 — was silently dropped, now refused",
    ),
    ("turbopuffer", "and_geo_x2", "#223"),
    ("turbopuffer", "or_geo_x2", "#223"),
    (
        "chroma",
        "and_geo",
        "#223 — was silently dropped, now refused",
    ),
    ("chroma", "and_geo_x2", "#223"),
    ("chroma", "or_geo_x2", "#223"),
    (
        "mongodb",
        "and_geo",
        "#223 — was silently dropped, now refused",
    ),
    ("mongodb", "and_geo_x2", "#223"),
    ("mongodb", "or_geo_x2", "#223"),
    // Already an explicit `Err` on master — behaviour unchanged by this PR.
    (
        "kividb",
        "and_geo",
        "#223 — KiviDB has no GEO field type (already Err)",
    ),
    ("kividb", "and_geo_x2", "#223 (already Err)"),
    ("kividb", "or_geo_x2", "#223 (already Err)"),
    (
        "vertex",
        "and_geo",
        "#223 — no geo restrict namespace (already Err)",
    ),
    ("vertex", "and_geo_x2", "#223 (already Err)"),
    ("vertex", "or_geo_x2", "#223 (already Err)"),
    // ── Vertex AI: pre-existing hard errors, unchanged by this PR ───────────
    // `parse_vertex_filter` has returned `Err` for these since it was written
    // (it is the model this PR generalises), so Vertex already refused these
    // datasets rather than running them unfiltered.
    (
        "vertex",
        "or_match_str_x2",
        "pre-existing Err: top-level `or` unimplemented (same-field OR could map to an allowList)",
    ),
    (
        "vertex",
        "or_match_int_x2",
        "pre-existing Err: top-level `or` is refused before the leaves are read",
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
    let shapes = shipped_shapes();
    let engines = engines();
    let mut rows: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut filtered = 0usize;
    let mut rejected = 0usize;

    for shape in &shapes {
        let schema: HashMap<String, String> = shape
            .schema
            .iter()
            .map(|(f, t)| (f.to_string(), t.to_string()))
            .collect();
        for (engine, resolver) in &engines {
            let verdict = resolver(&shape.conditions, &schema);
            let gap = is_known_gap(engine, shape.id);
            match (&verdict, gap) {
                (Ok(Some(render)), None) => {
                    filtered += 1;
                    rows.push(format!("{engine:<14} {:<20} FILTERED  {render}", shape.id));
                }
                (Ok(Some(_)), Some(why)) => failures.push(format!(
                    "{engine}/{} is listed as a known gap ({why}) but now resolves — remove it \
                     from KNOWN_GAPS",
                    shape.id
                )),
                (Ok(None), _) => failures.push(format!(
                    "{engine}/{} resolved to NO filter for a declared condition; \
                     query_filter::resolve must never do this",
                    shape.id
                )),
                (Err(e), Some(why)) => {
                    rejected += 1;
                    rows.push(format!(
                        "{engine:<14} {:<20} REJECTED ({why}) [{}]",
                        shape.id,
                        e.lines().next().unwrap_or("")
                    ));
                }
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

    // The table must not be able to shrink to nothing and stay green: deleting a
    // shape row, deleting an engine column, or returning an empty `engines()`
    // were all invisible before these four assertions.
    assert_eq!(
        shapes.len(),
        17,
        "shipped shapes changed; update the count and the module docs deliberately"
    );
    assert_eq!(
        engines.len(),
        15,
        "every engine that reads `conditions` must have a column here"
    );
    assert_eq!(
        filtered + rejected,
        shapes.len() * engines.len(),
        "every cell must have a verdict"
    );
    assert_eq!(
        (filtered, rejected),
        (229, 26),
        "the expressible/refused split changed — say why in the PR body"
    );
}

/// Every exemption must name an engine and a shape that exist. Otherwise a
/// renamed shape silently retires a gap, which is how a stale exemption starts
/// hiding a live bug.
#[test]
fn every_known_gap_maps_to_a_live_cell() {
    let shapes = shipped_shapes();
    let shape_ids: Vec<&str> = shapes.iter().map(|s| s.id).collect();
    let engine_names: Vec<&str> = engines().iter().map(|(n, _)| *n).collect();
    assert_eq!(
        KNOWN_GAPS.len(),
        26,
        "KNOWN_GAPS size changed — was that deliberate?"
    );
    for (engine, shape, why) in KNOWN_GAPS {
        assert!(
            engine_names.contains(engine),
            "KNOWN_GAPS names engine `{engine}`, which has no column ({why})"
        );
        assert!(
            shape_ids.contains(shape),
            "KNOWN_GAPS names shape `{shape}`, which is not in the table ({why})"
        );
    }
}

/// A builder that keeps the leaves it understands and skips the rest renders the
/// same filter for a multi-leaf condition as for the sub-condition it actually
/// applied. For every shipped multi-leaf shape, assert the full render differs
/// from the render of every proper non-empty sub-multiset of its leaves.
///
/// This is the check that would catch the residual partial-drop bug on shipped
/// data. It passes today because every shipped multi-leaf condition is
/// type-homogeneous, so an engine either expresses all of its leaves or none of
/// them. A heterogeneous multi-leaf condition — `and(keyword, geo)` — is exactly
/// what this cannot save you from today, and it is the successor issue.
#[test]
fn no_shipped_multi_leaf_filter_loses_a_leaf() {
    let engines = engines();
    let mut checked = 0usize;
    for shape in shipped_shapes() {
        let Some((conn, leaves)) = shape.split() else {
            continue;
        };
        let schema: HashMap<String, String> = shape
            .schema
            .iter()
            .map(|(f, t)| (f.to_string(), t.to_string()))
            .collect();
        for (engine, resolver) in &engines {
            let Ok(Some(full)) = resolver(&shape.conditions, &schema) else {
                continue; // refused outright — covered by the matrix test
            };
            for skip in 0..leaves.len() {
                let subset: Vec<Value> = leaves
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != skip)
                    .map(|(_, l)| l.clone())
                    .collect();
                let sub_conditions = json!({ conn: subset });
                let Ok(Some(sub)) = resolver(&sub_conditions, &schema) else {
                    continue;
                };
                assert_ne!(
                    full, sub,
                    "{engine} renders shape `{}` identically to the same condition with leaf \
                     {skip} removed — the leaf vanished, and the query would be scored against \
                     ground truth that did apply it (issue #219)",
                    shape.id
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked >= 100,
        "expected the multi-leaf shapes to produce many comparisons, got {checked}"
    );
}

/// The other half of the guard: an unfiltered shipped dataset must not error.
///
/// `random_keywords_1m_vocab_10_no_filters` and the other `*_no_filters`
/// tarballs spell every row `"conditions": null`, which `compound_reader.rs`
/// hands over as `Some(Value::Null)`. An `is_some()`-keyed guard would fail all
/// 10 000 queries on all 15 engines.
///
/// Every engine's verdict is asserted on. An earlier version of this test
/// computed the verdict and then discarded it, so making all 15 resolvers `Err`
/// left it green.
#[test]
fn the_shipped_no_filters_datasets_stay_unfiltered_on_every_engine() {
    let schema = HashMap::new();
    let mut asserted = 0usize;
    for (engine, resolver) in engines() {
        for absent in [Value::Null, json!({})] {
            assert_eq!(
                resolver(&absent, &schema),
                Ok(None),
                "{engine} must treat {absent} as no filter, not as a declared one"
            );
            asserted += 1;
        }
        // `resolve` short-circuits before the builder, so the builder is never
        // consulted for an absent condition.
        assert_eq!(
            resolve::<()>(engine, 0, None, |_| unreachable!(
                "{engine}: the builder must not be consulted for absent conditions"
            ))
            .map(|f| f.is_filtered()),
            Ok(false)
        );
        asserted += 1;
    }
    assert_eq!(asserted, 45, "all 15 engines must be exercised");
}
