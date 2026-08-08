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
//!
//! It does **not** catch widening that still mentions every leaf, because the
//! comparison is against sub-multisets only. Two live examples, both verified to
//! survive this module: rendering a top-level `and` as `OR`, and dropping only
//! the upper bound of a two-sided range. Both are silent-wrong-result bugs and
//! both pass here. Nor can it catch a builder that renders a match-all filter
//! for a real condition. Catching either needs a canonical *semantic* form to
//! compare against, not a render; until then they sit in the same disclaimed set
//! as the escapes listed in `query_filter.rs`.
//!
//! It also only knows about the shapes listed here — which is why
//! [`the_table_matches_the_shipped_condition_fixture`] pins them against a
//! checked-in file rather than a comment.
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
//! **Provenance caveat.** Only 4 of the 57 registry datasets are on disk in a
//! typical checkout (`datasets/.gitignore` is `*/*`, so every filter-bearing
//! `tests.jsonl` is a download). Those four are censused for real by
//! [`any_locally_present_shipped_dataset_emits_only_known_shapes`]; for the
//! other 53 the claim rests on reading the two generators that produce them —
//! the upstream ann-filtered-benchmark generator, whose operator vocabulary
//! `v0/engine/base_client/parser.py` fixes at `match` | `range` | `geo`, and
//! `src/bin/generate_dataset.rs`. Both were read and confirmed structurally.
//! That is a structural argument, not a full census, and it is the weakest link
//! in this module.
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
    erase_placeholder_ordinals(&format!("{} [{}]", f.0, params.join(",")))
}

/// Strip the ordinals out of parameter placeholders (`$b_1` -> `$b_N`,
/// `$3` -> `$N`) before a render is compared.
///
/// Without this, [`no_shipped_multi_leaf_filter_loses_a_leaf`] is **void on the
/// engines that number their parameters**. Redis, Valkey, Dragonfly and
/// pgvector all thread a shared monotonic counter through the leaf builders, so
/// a builder that silently drops the FIRST leaf renumbers the survivor:
///
/// ```text
/// full = "(@b:{$b_1}) [b_1=Str(\"pPZIB\")]"     // leaf 0 vanished
/// sub  = "(@b:{$b_0}) [b_0=Str(\"pPZIB\")]"     // leaf 0 removed from the input
/// ```
///
/// Semantically identical, textually different — `assert_ne!` passes and the
/// drop goes unseen. That was every one of Redis's `skip 0` comparisons.
///
/// This is reachable by shipped code, not only by a mutant: `build_range_filter`
/// and `build_geo_filter` in `redis.rs` both `*counter += 1` before they can
/// `return None`, so `and(unparseable_range, keyword)` renumbers the survivor.
///
/// **Over-erasure is safe only because both sides of the comparison are
/// erased.** `$b_1` and `$b_11` collide, as do `v_1` and `v_2`; since the
/// comparison is `assert_ne!`, a collision can only make it fire when it should
/// not — a false alarm, never a false pass. That reasoning holds for the
/// sub-multiset comparison and **stops holding** the moment anything compares
/// two renders of the *same* arity, which is exactly what the widening check
/// (`and` emitted as `or`, a two-sided range emitted with one bound) listed as
/// future work in the module docs would need. Do not reuse this function there
/// without re-deriving its safety.
///
/// Digits are eaten only after `$` or `_`, so literal values survive: `Int(11)`
/// stays `Int(11)`. [`the_ordinal_eraser_erases_ordinals_and_nothing_else`]
/// pins that directly — the function is otherwise only exercised transitively,
/// and both the identity function and "strip every digit" left all tests green.
fn erase_placeholder_ordinals(render: &str) -> String {
    let bytes: Vec<char> = render.chars().collect();
    let mut out = String::with_capacity(render.len());
    let mut i = 0;
    while i < bytes.len() {
        // `$3` / `$12` — a positional placeholder (pgvector).
        if bytes[i] == '$' && bytes.get(i + 1).is_some_and(|c| c.is_ascii_digit()) {
            out.push_str("$N");
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            continue;
        }
        // `_0` / `_17` — the counter suffix on a named param (RediSearch).
        if bytes[i] == '_' && bytes.get(i + 1).is_some_and(|c| c.is_ascii_digit()) {
            out.push_str("_N");
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
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
        // MongoDB picks its query stage — and therefore its filter GRAMMAR —
        // from the dataset schema: a geo-carrying dataset uses `$search` +
        // the `vectorSearch` operator (the only MongoDB vector path with a geo
        // pre-filter), everything else uses `$vectorSearch`. The resolver
        // reproduces exactly that rule, so this column tests what production
        // sends rather than a third code path.
        ("mongodb", |c, schema| {
            let declared = serde_json::json!(schema);
            let dialect_is_search = super::mongodb_engine::schema_declares_geo(Some(&declared));
            through("MongoDB", c, |v| {
                if dialect_is_search {
                    super::mongodb_engine::parse_mongo_search_conditions(v)
                        .map(|d| d.map(|d| format!("{d:?}")))
                } else {
                    Ok(super::mongodb_engine::parse_mongo_conditions(v).map(|d| format!("{d:?}")))
                }
            })
        }),
        ("pgvector", |c, _| {
            through("pgvector", c, |v| {
                Ok(super::pgvector::parse_pg_conditions(v, 3)
                    .map(|(sql, vals)| erase_placeholder_ordinals(&format!("{sql} {vals:?}"))))
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
    // ── #223: geo ───────────────────────────────────────────────────────────
    // (four datasets are geo-*named*; the two `-no-filters` twins are
    // `"conditions": null` throughout and stay green).
    //
    // VectorSets, Milvus and MongoDB used to sit here and now express geo:
    // VectorSets by comparing the stored unit vector against a cosine threshold
    // (`engine::geo`), Milvus with the native `ST_DWITHIN` on a `Geometry`
    // column, MongoDB with `geoWithin`/`circle` inside a `$search` stage. Redis,
    // Valkey, Dragonfly, Elasticsearch, OpenSearch, Weaviate, pgvector and
    // Qdrant already did.
    //
    // Chroma and Turbopuffer remain, and the entries below are the evidence
    // rather than a TODO. Both filter DSLs are a CLOSED enum of
    // `field OP literal` comparisons — Chroma:
    // `$eq $ne $gt $gte $lt $lte $in $nin $and $or`; Turbopuffer:
    // `Eq NotEq In NotIn Lt Lte Gt Gte Any* Contains* Glob* Regex
    // ContainsAllTokens ContainsAnyToken ContainsTokenSequence Fuzzy And Or Not`
    // — with no geo primitive, no attribute-vs-attribute comparison, and no
    // arithmetic anywhere in the filter grammar (Turbopuffer's arithmetic lives
    // in `rank_by`, which its docs state "never affect[s] matching"). That
    // closes every exact encoding: a spherical cap is not an axis-aligned box in
    // any query-INDEPENDENT coordinate system, so no conjunction of range
    // comparisons can carve it out, and the linear `x*qx + y*qy + z*qz >= c`
    // form the other two use needs cross-field arithmetic neither has. A
    // bounding box would admit up to √2·r away and is a widening, i.e. exactly
    // the silently-wrong recall #219 exists to stop.
    (
        "turbopuffer",
        "and_geo",
        "#223 — no geo operator and no arithmetic in the filter grammar",
    ),
    ("turbopuffer", "and_geo_x2", "#223"),
    ("turbopuffer", "or_geo_x2", "#223"),
    (
        "chroma",
        "and_geo",
        "#223 — `where` is a closed enum of field-OP-literal comparisons",
    ),
    ("chroma", "and_geo_x2", "#223"),
    ("chroma", "or_geo_x2", "#223"),
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
        (238, 17),
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
        17,
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
///
/// Scope, precisely: it catches renders that LOSE a leaf. It does not catch
/// renders that keep every leaf but WIDEN — `and` emitted as `or`, or a
/// two-sided range emitted with only its lower bound. Those still mention every
/// leaf, so no sub-multiset comparison can see them.
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

// ─────────────────────────────────────────────────────────────────────────────
// Source-level nets for the escapes the type system cannot close
// ─────────────────────────────────────────────────────────────────────────────

/// Strip `//` line comments and `/* */` block comments, respecting string and
/// char literals.
///
/// Both source scans below run on the stripped text. Without this they are
/// decorative in one direction and actively harmful in the other: a comment
/// containing `.resolve_all(` satisfies the pairing scan while the real call is
/// gutted, and — worse — a comment reading "NEVER call
/// `QueryConditions::unfiltered` here" *trips* the other scan, so the guard
/// would forbid documenting the very bug it removes.
fn strip_comments(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let (mut i, mut in_str, mut in_char, mut raw_hashes) = (0usize, false, false, 0usize);
    while i < b.len() {
        if in_str {
            // Raw strings (r#"..."#) have no escapes; normal strings do.
            if raw_hashes > 0 {
                if b[i] == '"' && b[i + 1..].iter().take(raw_hashes).all(|c| *c == '#') {
                    in_str = false;
                    out.push(b[i]);
                    for _ in 0..raw_hashes {
                        i += 1;
                        out.push('#');
                    }
                    raw_hashes = 0;
                    i += 1;
                    continue;
                }
            } else if b[i] == '\\' {
                out.push(b[i]);
                if i + 1 < b.len() {
                    out.push(b[i + 1]);
                }
                i += 2;
                continue;
            } else if b[i] == '"' {
                in_str = false;
            }
            out.push(b[i]);
            i += 1;
            continue;
        }
        if in_char {
            if b[i] == '\\' {
                out.push(b[i]);
                if i + 1 < b.len() {
                    out.push(b[i + 1]);
                }
                i += 2;
                continue;
            }
            if b[i] == '\'' {
                in_char = false;
            }
            out.push(b[i]);
            i += 1;
            continue;
        }
        if b[i] == '/' && b.get(i + 1) == Some(&'/') {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if b[i] == '/' && b.get(i + 1) == Some(&'*') {
            let mut depth = 1;
            i += 2;
            while i < b.len() && depth > 0 {
                if b[i] == '/' && b.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if b[i] == '*' && b.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        if b[i] == 'r' && matches!(b.get(i + 1), Some('"') | Some('#')) {
            let mut j = i + 1;
            let mut hashes = 0;
            while b.get(j) == Some(&'#') {
                hashes += 1;
                j += 1;
            }
            if b.get(j) == Some(&'"') {
                in_str = true;
                raw_hashes = hashes;
                out.extend(&b[i..=j]);
                i = j + 1;
                continue;
            }
        }
        if b[i] == '"' {
            in_str = true;
        } else if b[i] == '\'' {
            // Lifetimes (`'a`) are not char literals; a char literal closes.
            let is_lifetime = b.get(i + 1).is_some_and(|c| c.is_alphabetic() || *c == '_')
                && b.get(i + 2) != Some(&'\'');
            if !is_lifetime {
                in_char = true;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// Every `.rs` file under `engine/`, recursively, with comments stripped.
///
/// Recursive because a non-recursive `read_dir` cannot see an engine that lives
/// in a subdirectory (`engine/zz_sub/mod.rs`) and would silently exempt it.
fn engine_sources() -> Vec<(String, String)> {
    fn walk(dir: &std::path::Path, root: &std::path::Path, out: &mut Vec<(String, String)>) {
        for entry in std::fs::read_dir(dir).expect("engine dir").flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, root, out);
                continue;
            }
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string();
            // This file names every pattern it forbids.
            if rel == "filter_guard.rs" {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("read engine source");
            out.push((rel, strip_comments(&src)));
        }
    }
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/vector_db_benchmark/engine");
    let mut out = Vec::new();
    walk(&root, &root, &mut out);
    out.sort();
    out
}

/// Count `needle` occurrences allowing arbitrary whitespace where `\s*` markers
/// appear, so `QueryConditions :: unfiltered` reads the same as
/// `QueryConditions::unfiltered`.
fn count_loose(haystack: &str, parts: &[&str]) -> usize {
    let compact: String = haystack.chars().filter(|c| !c.is_whitespace()).collect();
    compact.matches(&parts.concat()).count()
}

/// `#[must_use]` on `QueryConditions` does **not** catch an engine that ignores
/// the conditions: it fires only when the whole tuple is a bare expression
/// statement, and every engine destructures. `let (q, n, _conditions) = …` is
/// silent, passes clippy, and already ships at `ground_truth.rs:68` (correctly,
/// since ground truth has no filter to apply).
///
/// This is the net that actually covers it: inside `engine/`, every
/// `dataset.read_queries()` must be paired with a resolver call in the same
/// file. An engine that reads the conditions and throws them away fails here.
#[test]
fn every_engine_read_of_query_conditions_is_paired_with_a_resolver() {
    let mut reads_total = 0usize;
    let mut resolves_total = 0usize;
    for (name, src) in engine_sources() {
        // `.read_queries()` on ANY receiver: an engine that names it `ds`
        // rather than `dataset` must not be exempt.
        let reads = count_loose(&src, &[".read_queries()"]);
        let resolves =
            count_loose(&src, &[".resolve_all("]) + count_loose(&src, &[".try_resolve_all("]);
        assert_eq!(
            reads, resolves,
            "{name} calls .read_queries() {reads}x but resolves conditions {resolves}x — \
             an unresolved read runs every query UNFILTERED (issue #219)"
        );
        reads_total += reads;
        resolves_total += resolves;
    }
    assert_eq!(
        (reads_total, resolves_total),
        (23, 23),
        "the read/resolve census changed; confirm the new site is guarded"
    );
}

/// `QueryConditions::unfiltered(n)` has to be public because the binary's
/// `dataset.rs` builds one for the HDF5/JSONL formats, which have nowhere to put
/// conditions. It cannot fabricate a filter, but it CAN erase every real one —
/// swapping it in for a resolved value turns a filtered dataset into an
/// unfiltered run with no error. Nothing in the type system stops an engine
/// calling it; this does.
///
/// Scope of the three constructor names checked:
///
/// * `unfiltered` — the live one, and the only reason this test exists;
/// * `new` — **already unreachable**. It is `pub(crate)` of the *library* crate
///   and every engine is in the binary, so the crate boundary closes it, not
///   this scan. Listed so that relaxing its visibility fails here.
/// * `default` — `Default` is not derived on `QueryConditions`, precisely
///   because `..Default::default()` would be a quiet second spelling of
///   `unfiltered(0)`. Listed so that re-deriving it fails here.
///
/// Matching is whitespace-insensitive and follows `use ... as Alias` and
/// `type Alias = QueryConditions` renames, and runs on comment-stripped source
/// so that *documenting* the hazard is not itself a failure.
#[test]
fn no_engine_erases_its_conditions_with_the_unfiltered_constructor() {
    for (name, src) in engine_sources() {
        let mut names = vec!["QueryConditions".to_string()];
        // `use ...::QueryConditions as QC;`
        for (idx, _) in src.match_indices("QueryConditions as ") {
            let alias: String = src[idx + "QueryConditions as ".len()..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !alias.is_empty() {
                names.push(alias);
            }
        }
        // `type QCAlias = QueryConditions;`
        for (idx, _) in src.match_indices("= QueryConditions") {
            let head: String = src[..idx].chars().filter(|c| !c.is_whitespace()).collect();
            if let Some(t) = head.rfind("type") {
                let alias: String = head[t + "type".len()..]
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !alias.is_empty() {
                    names.push(alias);
                }
            }
        }
        let compact: String = src.chars().filter(|c| !c.is_whitespace()).collect();
        for ty in &names {
            for ctor in ["unfiltered", "new", "default"] {
                assert!(
                    !compact.contains(&format!("{ty}::{ctor}(")),
                    "{name} calls {ty}::{ctor}() — only dataset.rs and the readers may construct \
                     QueryConditions, and erasing one runs every query UNFILTERED (issue #219, \
                     documented escape 3)"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The shape table, checked against data rather than against a comment
// ─────────────────────────────────────────────────────────────────────────────

/// Structural signature of a condition: connective, and per leaf the operator
/// plus the JSON types of its criteria — field names and literal values erased.
/// Two conditions with the same signature exercise the same builder paths.
fn shape_signature(cond: &Value) -> String {
    fn type_of(v: &Value) -> &'static str {
        match v {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(n) if n.is_i64() || n.is_u64() => "int",
            Value::Number(_) => "float",
            Value::String(_) => "str",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }
    fn leaf(v: &Value, out: &mut String) {
        let Some(obj) = v.as_object() else {
            out.push_str("<non-object>");
            return;
        };
        let mut parts: Vec<String> = Vec::new();
        for (key, spec) in obj {
            if key == "and" || key == "or" {
                let mut inner = String::new();
                group(v, &mut inner);
                parts.push(inner);
                continue;
            }
            let mut ops: Vec<String> = Vec::new();
            match spec.as_object() {
                Some(spec_obj) => {
                    for (op, criteria) in spec_obj {
                        let mut crit: Vec<String> = match criteria.as_object() {
                            Some(c) => c
                                .iter()
                                .map(|(k, v)| format!("{k}:{}", type_of(v)))
                                .collect(),
                            None => vec![type_of(criteria).to_string()],
                        };
                        crit.sort();
                        ops.push(format!("{op}({})", crit.join(",")));
                    }
                }
                None => ops.push(format!("<{}>", type_of(spec))),
            }
            ops.sort();
            parts.push(ops.join("+"));
        }
        parts.sort();
        out.push_str(&parts.join("&"));
    }
    fn group(v: &Value, out: &mut String) {
        for conn in ["and", "or"] {
            if let Some(items) = v.get(conn).and_then(|x| x.as_array()) {
                let mut rendered: Vec<String> = items
                    .iter()
                    .map(|i| {
                        let mut s = String::new();
                        leaf(i, &mut s);
                        s
                    })
                    .collect();
                rendered.sort();
                out.push_str(&format!("{conn}[{}]", rendered.join(",")));
                return;
            }
        }
        leaf(v, out);
    }
    let mut out = String::new();
    group(cond, &mut out);
    out
}

/// Read the checked-in fixture through the REAL reader, and require the table to
/// match it exactly.
///
/// **What this is and is not.** Both the table and the fixture are transcribed
/// by hand from the shipped tarballs, so this does not make the shapes
/// independently verified — it makes them *two synced copies of the same
/// claim*, and a drift on either side fails. That is a real step up from a
/// comment (it catches a table-only edit, a fixture-only edit, an added shape
/// with no fixture row, and a deleted row) but it is not a census. The only
/// independent check is
/// [`any_locally_present_shipped_dataset_emits_only_known_shapes`], which
/// prints SKIPPED in CI because every filter-bearing `tests.jsonl` is a
/// gitignored download (`datasets/.gitignore` is `*/*`).
///
/// The fixture's absent-condition rows do travel the real `compound_reader`
/// path. `"conditions": null` and `"conditions": {}` are **different values**
/// coming out of the reader — `Some(Value::Null)` vs `Some(Value::Object{})`,
/// neither of them `None` — and both must resolve unfiltered; the library-side
/// test `readers::tests::fixture_null_and_empty_conditions_are_distinct_values`
/// pins the distinction, which cannot be seen from here because
/// `QueryConditions` does not expose it.
#[test]
fn the_table_matches_the_shipped_condition_fixture() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/shipped_conditions");
    let (queries, _neighbors, conditions) =
        vector_db_benchmark::readers::read_compound_queries(dir.to_str().unwrap(), false)
            .expect("read the condition fixture");
    assert_eq!(queries.len(), 21, "fixture row count");
    assert_eq!(
        conditions.declared_count(),
        17,
        "17 declared shapes + 3 `\"conditions\": null` rows + 1 `{{}}` row"
    );

    // Recovering the raw conditions needs the echo escape documented in
    // `query_filter.rs`. Using it here is deliberate: a test may look, an engine
    // may not, and the absent rows still come back unfiltered.
    let echoed = conditions
        .resolve_all("fixture", |c| Some(c.clone()))
        .expect("the fixture must not contain a droppable condition");
    let from_file: Vec<&Value> = echoed.iter().filter_map(|f| f.as_ref()).collect();
    assert_eq!(from_file.len(), 17);
    assert_eq!(
        echoed.iter().filter(|f| !f.is_filtered()).count(),
        4,
        "the three `null` rows and the `{{}}` row must read back as unfiltered, not as \
         declared filters"
    );

    let table = shipped_shapes();
    let mut in_table: Vec<String> = table.iter().map(|s| s.conditions.to_string()).collect();
    let mut in_file: Vec<String> = from_file.iter().map(|c| c.to_string()).collect();
    in_table.sort();
    in_file.sort();
    assert_eq!(
        in_table, in_file,
        "the guard's shape table and the checked-in shipped-condition fixture have drifted"
    );
}

/// Slice the `conditions` value out of a `tests.jsonl` line without parsing the
/// row (the query vector dominates it — up to 2048 floats).
fn conditions_slice(line: &str) -> Option<&str> {
    let start = line.find("\"conditions\":")? + "\"conditions\":".len();
    let rest = line[start..].trim_start();
    let offset = line.len() - rest.len();
    if rest.starts_with("null") {
        return Some("null");
    }
    let bytes = rest.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let (mut depth, mut in_str, mut escaped) = (0i32, false, false);
    for (i, b) in bytes.iter().enumerate() {
        match (in_str, escaped, b) {
            (true, true, _) => escaped = false,
            (true, false, b'\\') => escaped = true,
            (true, false, b'"') => in_str = false,
            (false, _, b'"') => in_str = true,
            (false, _, b'{') => depth += 1,
            (false, _, b'}') => {
                depth -= 1;
                if depth == 0 {
                    return Some(&line[offset..offset + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Opportunistic: when a real shipped tarball happens to be present (they are
/// gitignored downloads, so CI will not have them), every distinct condition
/// SIGNATURE in its `tests.jsonl` must already be covered by the table.
///
/// This is the check that would notice the table describing a dataset that does
/// not exist. It prints and returns when no dataset is on disk — say so when
/// quoting it, rather than counting it as CI coverage.
#[test]
fn any_locally_present_shipped_dataset_emits_only_known_shapes() {
    let roots = [
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets"),
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("v0/datasets"),
    ];
    let known: Vec<String> = shipped_shapes()
        .iter()
        .map(|s| shape_signature(&s.conditions))
        .collect();

    let mut found: Vec<(String, String, String)> = Vec::new(); // (dataset, signature, example)
    for root in roots {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        // A compound dataset's `tests.jsonl` sits either directly under its
        // registry directory (the locally generated sets) or one level deeper
        // (the tarballs, which unpack into a named subdirectory).
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        for entry in entries.flatten() {
            candidates.push(entry.path());
            if let Ok(inner) = std::fs::read_dir(entry.path()) {
                candidates.extend(inner.flatten().map(|d| d.path()));
            }
        }
        {
            for sub in candidates {
                let tests = sub.join("tests.jsonl");
                if !tests.exists() {
                    continue;
                }
                let Ok(file) = std::fs::File::open(&tests) else {
                    continue;
                };
                let label = sub
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                // Read line by line and slice out ONLY the `conditions` value:
                // a shipped `tests.jsonl` row carries a 2048-float query vector,
                // so parsing whole rows turns this into a three-minute test.
                use std::io::BufRead;
                for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
                    let Some(slice) = conditions_slice(&line) else {
                        continue;
                    };
                    let Ok(cond) = serde_json::from_str::<Value>(slice) else {
                        continue;
                    };
                    if cond.is_null() {
                        continue;
                    }
                    let sig = shape_signature(&cond);
                    if !found.iter().any(|(d, s, _)| d == &label && s == &sig) {
                        found.push((label.clone(), sig, cond.to_string()));
                    }
                }
            }
        }
    }

    if found.is_empty() {
        println!(
            "SKIPPED: no shipped tests.jsonl on disk (they are gitignored downloads). The \
             table is pinned by the checked-in fixture instead."
        );
        return;
    }
    let mut unknown: Vec<String> = Vec::new();
    for (dataset, sig, example) in &found {
        println!("{dataset:34} {sig}");
        if !known.contains(sig) {
            unknown.push(format!("{dataset}: {sig}\n    e.g. {example}"));
        }
    }
    assert!(
        unknown.is_empty(),
        "shipped datasets on disk emit condition shapes the guard table does not cover:\n{}",
        unknown.join("\n")
    );
}

/// Direct test of [`erase_placeholder_ordinals`]. Without it the function is
/// only exercised transitively, and replacing its body with the identity
/// function — or with "strip all digits" — leaves every other test green.
#[test]
fn the_ordinal_eraser_erases_ordinals_and_nothing_else() {
    // RediSearch named params, in the query and in the sorted param list.
    assert_eq!(
        erase_placeholder_ordinals("(@b:{$b_1}) [b_1=Str(\"pPZIB\")]"),
        "(@b:{$b_N}) [b_N=Str(\"pPZIB\")]"
    );
    // pgvector positional placeholders, including multi-digit.
    assert_eq!(
        erase_placeholder_ordinals("a = $3 AND b = $12"),
        "a = $N AND b = $N"
    );
    // Literal values must survive: digits are eaten only after `$` or `_`.
    assert_eq!(
        erase_placeholder_ordinals("[a_0=Int(11), b_1=Float(0.5)]"),
        "[a_N=Int(11), b_N=Float(0.5)]"
    );
    // An identity implementation fails the first case; "strip all digits" fails
    // this one.
    assert_eq!(erase_placeholder_ordinals("Int(11)"), "Int(11)");
    // A leading-leaf drop is what this exists to expose: the two renders below
    // differ only in the ordinal, and must compare equal after erasure.
    assert_eq!(
        erase_placeholder_ordinals("(@b:{$b_1})"),
        erase_placeholder_ordinals("(@b:{$b_0})")
    );
}
