//! Filter-only on the shared `full` index layout (#344): the realized-schema
//! checks for each layout, and the per-request "no vector clause" checks.
use super::*;

fn schema() -> Value {
    json!({"n":"int","price":"float","tag":"keyword","body":"text"})
}

/// What OpenSearch reports for a `full` index: an indexed knn_vector, `double`
/// floats and plain keywords, plus a text field the contract ignores.
fn os_full_properties() -> Value {
    json!({
        "vector":{"type":"knn_vector","dimension":4},
        "n":{"type":"long"}, "price":{"type":"double"},
        "tag":{"type":"keyword"}, "body":{"type":"text"}
    })
}

#[test]
fn opensearch_full_schema_needs_a_knn_index_and_the_filter_only_scalar_mapping() {
    let schema = schema();
    for knn in [json!(true), json!("true")] {
        validate_opensearch_full_schema(&os_full_properties(), &json!({"knn":knn}), Some(&schema))
            .unwrap();
    }
    // A no-vector index (or one without the knn setting) is not the full layout.
    for knn in [json!(false), json!("false"), Value::Null] {
        let e = validate_opensearch_full_schema(
            &os_full_properties(),
            &json!({"knn":knn}),
            Some(&schema),
        )
        .unwrap_err();
        assert!(e.contains("index.knn=true"), "{e}");
    }
    let mut props = os_full_properties();
    props["vector"] = json!({"type":"float","index":false,"doc_values":false});
    let e =
        validate_opensearch_full_schema(&props, &json!({"knn":true}), Some(&schema)).unwrap_err();
    assert!(e.contains("knn_vector"), "{e}");
    // The vector layout maps `float` as `float`: refused, since filter-only
    // range semantics need binary64.
    let mut props = os_full_properties();
    props["price"]["type"] = json!("float");
    let e =
        validate_opensearch_full_schema(&props, &json!({"knn":true}), Some(&schema)).unwrap_err();
    assert!(e.contains("double mapping for price"), "{e}");
    let mut props = os_full_properties();
    props["tag"]["normalizer"] = json!("lowercase");
    assert!(validate_opensearch_full_schema(&props, &json!({"knn":true}), Some(&schema)).is_err());
    // And the no-vector validator still refuses the full index.
    assert!(
        validate_opensearch_schema(&os_full_properties(), &json!({"knn":true}), Some(&schema))
            .is_err()
    );
}

fn redis_full_attrs(case_sensitive: bool) -> Value {
    let mut tag = json!({"attribute":"tag","type":"TAG","SEPARATOR":";","flags":[]});
    if case_sensitive {
        tag["flags"] = json!(["CASESENSITIVE"]);
    }
    json!({"attributes":[
        {"attribute":"vector","type":"VECTOR","algorithm":"HNSW"},
        tag,
        {"attribute":"n","type":"NUMERIC"},
        {"attribute":"price","type":"NUMERIC"},
        {"attribute":"body","type":"TEXT"}
    ]})
}

#[test]
fn redis_full_schema_needs_a_vector_field_and_case_sensitive_tags() {
    let schema = schema();
    validate_redis_full_schema(&redis_full_attrs(true), Some(&schema)).unwrap();
    // The vector layout's case-insensitive TAG would answer filter-only
    // differently from OpenSearch's keyword: refused.
    let e = validate_redis_full_schema(&redis_full_attrs(false), Some(&schema)).unwrap_err();
    assert!(e.contains("CASESENSITIVE"), "{e}");
    // A reused no-vector index is not the full layout.
    let mut no_vector = redis_full_attrs(true);
    no_vector["attributes"].as_array_mut().unwrap().remove(0);
    let e = validate_redis_full_schema(&no_vector, Some(&schema)).unwrap_err();
    assert!(e.contains("VECTOR"), "{e}");
    // The no-vector validator keeps refusing any VECTOR field.
    let e = validate_redis_schema(&redis_full_attrs(true), Some(&schema)).unwrap_err();
    assert!(e.contains("containing VECTOR"), "{e}");
    validate_redis_schema(&no_vector, Some(&schema)).unwrap();
}

#[test]
fn opensearch_filter_only_bodies_must_not_carry_a_vector_clause() {
    let filter = json!({"term":{"tag":"red"}});
    check_opensearch_body(&opensearch_body(&filter, 10)).unwrap();
    for body in [
        json!({"query":{"knn":{"vector":{"vector":[1.0],"k":10}}}}),
        json!({"query":{"bool":{"filter":[filter.clone()],"must":[{"knn":{"vector":{}}}]}}}),
        json!({"query":{"KNN":{}}}),
        json!({"query":{"script_score":{"query":{"match_all":{}}}}}),
        json!({"query":{"neural":{"vector":{}}}}),
    ] {
        let e = check_opensearch_body(&body).unwrap_err();
        assert!(e.contains("vector clause"), "{body}: {e}");
    }
}

#[test]
fn redis_filter_only_queries_must_not_carry_a_vector_clause() {
    for q in ["@tag:{$p0}", "(@n:[$p0 $p1]) | (@tag:{$p2})", "*"] {
        check_redis_query(q).unwrap();
    }
    for q in [
        "(@tag:{$p0})=>[KNN 10 @vector $BLOB]",
        "*=>[KNN $K @vector $BLOB AS score]",
        "@vector:[VECTOR_RANGE $r $BLOB]",
        "@vector:[vector_range 0.5 $BLOB]",
    ] {
        let e = check_redis_query(q).unwrap_err();
        assert!(e.contains("vector clause"), "{q}: {e}");
    }
}

/// #344 review: `uuid` and `bool` fields are TAGs too (create_index marks
/// them CASESENSITIVE on `full` / `no-vector`), so the realized-layout check
/// requires the flag on them exactly as on `keyword`.
#[test]
fn redis_full_schema_checks_every_tag_kind() {
    for kind in ["uuid", "bool"] {
        let schema = json!({ "flag": kind });
        let attrs = |flags: Value| {
            json!({"attributes":[
                {"attribute":"vector","type":"VECTOR","algorithm":"HNSW"},
                {"attribute":"flag","type":"TAG","SEPARATOR":";","flags":flags}
            ]})
        };
        validate_redis_full_schema(&attrs(json!(["CASESENSITIVE"])), Some(&schema)).unwrap();
        let e = validate_redis_full_schema(&attrs(json!([])), Some(&schema)).unwrap_err();
        assert!(e.contains("CASESENSITIVE TAG for flag"), "{kind}: {e}");
        let missing = json!({"attributes":[
            {"attribute":"vector","type":"VECTOR","algorithm":"HNSW"}
        ]});
        assert!(
            validate_redis_full_schema(&missing, Some(&schema)).is_err(),
            "{kind}"
        );
    }
}

#[test]
fn a_vector_plan_refuses_an_index_built_for_filter_only() {
    let schema = schema();
    // Redis: case-insensitive tags are the vector layout; CASESENSITIVE is not.
    validate_redis_vector_layout(&redis_full_attrs(false), Some(&schema)).unwrap();
    let e = validate_redis_vector_layout(&redis_full_attrs(true), Some(&schema)).unwrap_err();
    assert!(e.contains("tag field tag is CASESENSITIVE"), "{e}");
    // RESP2 pairs carry the flag standalone.
    let resp2 = json!([
        "attributes",
        [[
            "identifier",
            "tag",
            "attribute",
            "tag",
            "type",
            "TAG",
            "SEPARATOR",
            ";",
            "CASESENSITIVE"
        ]]
    ]);
    assert!(validate_redis_vector_layout(&resp2, Some(&schema)).is_err());
    // No schema, nothing to tell apart.
    validate_redis_vector_layout(&redis_full_attrs(true), None).unwrap();
    // OpenSearch: `float` is the vector layout; `double` is filter-only's.
    let mut props = os_full_properties();
    let e = validate_opensearch_vector_layout(&props, Some(&schema)).unwrap_err();
    assert!(e.contains("float field price is mapped double"), "{e}");
    props["price"]["type"] = json!("float");
    validate_opensearch_vector_layout(&props, Some(&schema)).unwrap();
    // Fail closed: nothing to read the layout from is not a pass.
    for info in [
        json!({}),
        json!({"attributes": null}),
        json!(["index_name", "x"]),
    ] {
        let e = validate_redis_vector_layout(&info, Some(&schema)).unwrap_err();
        assert!(e.contains("FT.INFO has no attributes"), "{info}: {e}");
    }
    for props in [Value::Null, json!([]), json!("mapping")] {
        let e = validate_opensearch_vector_layout(&props, Some(&schema)).unwrap_err();
        assert!(e.contains("mapping has no properties"), "{props}: {e}");
    }
}

#[test]
fn redis_filter_only_commands_are_checked_as_sent() {
    let ok: [&[&[u8]]; 3] = [
        &[
            b"FT.SEARCH",
            b"idx",
            b"@tag:{$p0}",
            b"LIMIT",
            b"0",
            b"10",
            b"NOCONTENT",
            b"DIALECT",
            b"2",
            b"TIMEOUT",
            b"90000",
            b"PARAMS",
            b"2",
            b"p0",
            b"KNN",
        ],
        &[b"FT.SEARCH", b"idx", b"*"],
        &[
            b"ft.search",
            b"idx",
            b"@n:[$a $b]",
            b"PARAMS",
            b"4",
            b"a",
            b"1",
            b"b",
            b"=>",
        ],
    ];
    for args in ok {
        check_redis_command(args).unwrap();
    }
    let bad: [(&[&[u8]], &str); 7] = [
        (
            &[
                b"FT.SEARCH",
                b"idx",
                b"(@tag:{$p0})=>[KNN 10 @vector $BLOB]",
            ],
            "vector clause",
        ),
        (
            &[b"FT.SEARCH", b"idx", b"*", b"SORTBY", b"__vector_score"],
            "unexpected argument",
        ),
        (&[b"FT.AGGREGATE", b"idx", b"*"], "expected FT.SEARCH"),
        (&[b"FT.SEARCH", b"idx"], "expected FT.SEARCH"),
        (
            &[
                b"FT.SEARCH",
                b"idx",
                b"*",
                b"PARAMS",
                b"3",
                b"a",
                b"1",
                b"b",
            ],
            "odd PARAMS",
        ),
        (
            &[b"FT.SEARCH", b"idx", b"*", b"PARAMS", b"2", b"BLOB"],
            "missing operands",
        ),
        (
            &[b"FT.SEARCH", b"idx", b"*", b"LIMIT", b"0"],
            "missing operands",
        ),
    ];
    for (args, why) in bad {
        let e = check_redis_command(args).unwrap_err();
        assert!(e.contains(why), "{args:?}: {e}");
    }
}

/// #344 review: filter-only configs on `full` follow the `no-vector` rule
/// (first per concurrency kept, conflicting limit refused); vector / fulltext
/// configs at the same concurrency are never collapsed by it.
#[test]
fn filter_only_configs_on_full_follow_the_no_vector_rule() {
    // (is_filter_only, parallel, top)
    let mixed = [
        (false, Some(1), Some(10)),
        (true, Some(1), Some(5)),
        (false, Some(1), Some(10)),
        (true, Some(1), Some(5)),
        (true, Some(4), None),
        (true, Some(4), Some(10)),
    ];
    assert_eq!(
        mixed_search_param_indices(mixed, Some(10)).unwrap(),
        vec![0, 1, 2, 4]
    );
    // The same filter-only configs alone select the same filter-only subset.
    let alone: Vec<usize> =
        search_param_indices(mixed.iter().filter(|p| p.0).map(|p| (p.1, p.2)), Some(10)).unwrap();
    assert_eq!(alone, vec![0, 2]);
    let e = mixed_search_param_indices(
        [
            (false, Some(2), Some(10)),
            (true, Some(2), Some(3)),
            (true, Some(2), Some(7)),
        ],
        Some(10),
    )
    .unwrap_err();
    assert!(e.contains("conflicting result limits"), "{e}");
    // A different limit on a NON-filter-only config is not a conflict.
    assert_eq!(
        mixed_search_param_indices(
            [(false, Some(2), Some(100)), (true, Some(2), Some(3))],
            Some(10)
        )
        .unwrap(),
        vec![0, 1]
    );
}

#[test]
fn stats_layout_is_serialized_only_when_reported() {
    let mut stats = Stats::default();
    assert!(serde_json::to_value(&stats)
        .unwrap()
        .get("index_layout")
        .is_none());
    stats.index_layout = Some("full".into());
    assert_eq!(
        serde_json::to_value(&stats).unwrap()["index_layout"],
        "full"
    );
    // Old result files (no key) still parse.
    let old: Stats = serde_json::from_value(json!({
        "result_limit": 3, "matched_documents_sum": 1, "returned_documents_sum": 1
    }))
    .unwrap();
    assert_eq!(old.index_layout, None);
}
