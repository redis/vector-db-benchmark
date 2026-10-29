//! Contract edge cases independent of live servers. These fixtures deliberately
//! include malformed replies so completion guards cannot become permissive.
use super::*;
use redis::Value as Resp;

#[test]
fn selects_only_explicit_filters_and_cycles_without_ground_truth() {
    for requested in [-1, 0, 1, 7] {
        assert!(query_indices([], requested).is_err());
        assert!(query_indices([false, false], requested).is_err());
    }
    for (requested, expected) in [
        (-5, vec![1, 3]),
        (-1, vec![1, 3]),
        (0, vec![1, 3]),
        (1, vec![1]),
        (2, vec![1, 3]),
        (5, vec![1, 3, 1, 3, 1]),
    ] {
        assert_eq!(
            query_indices([false, true, false, true], requested).unwrap(),
            expected
        );
    }
    assert_eq!(query_indices([true], 3).unwrap(), vec![0, 0, 0]);
}

fn os_properties() -> Value {
    json!({
        "vector":{"type":"float","index":false,"doc_values":false},
        "n":{"type":"long"}, "price":{"type":"double"},
        "tag":{"type":"keyword"}, "ignored":{"type":"geo_point"}
    })
}

fn scalar_schema() -> Value {
    json!({"n":"int","price":"float","tag":"keyword","ignored":"geo"})
}

#[test]
fn opensearch_schema_accepts_realized_defaults_and_both_boolean_encodings() {
    let schema = scalar_schema();
    for knn in [json!(false), json!("false")] {
        let index = json!({"knn":knn});
        for schema in [None, Some(&schema), Some(&Value::Null), Some(&json!([]))] {
            assert!(validate_opensearch_schema(&os_properties(), &index, schema).is_ok());
        }
    }
    let mut props = os_properties();
    for field in ["n", "price", "tag"] {
        props[field]["index"] = json!(true);
    }
    props["tag"]["normalizer"] = Value::Null;
    assert!(validate_opensearch_schema(&props, &json!({"knn":false}), Some(&schema)).is_ok());
}

#[test]
fn opensearch_schema_rejects_every_vector_and_scalar_contract_violation() {
    let schema = scalar_schema();
    for bad in [
        Value::Null,
        json!(true),
        json!("true"),
        json!(0),
        json!("FALSE"),
    ] {
        assert!(
            validate_opensearch_schema(&os_properties(), &json!({"knn":bad}), Some(&schema))
                .is_err()
        );
    }
    assert!(validate_opensearch_schema(&os_properties(), &json!({}), Some(&schema)).is_err());
    for (pointer, bad) in [
        ("/vector", Value::Null),
        ("/vector/type", json!("knn_vector")),
        ("/vector/type", Value::Null),
        ("/vector/index", json!(true)),
        ("/vector/index", Value::Null),
        ("/vector/doc_values", json!(true)),
        ("/vector/doc_values", Value::Null),
        ("/n/type", json!("integer")),
        ("/n", Value::Null),
        ("/price/type", json!("float")),
        ("/price", Value::Null),
        ("/tag/type", json!("text")),
        ("/tag", Value::Null),
    ] {
        let mut props = os_properties();
        *props.pointer_mut(pointer).unwrap() = bad;
        assert!(
            validate_opensearch_schema(&props, &json!({"knn":false}), Some(&schema)).is_err(),
            "accepted {pointer}"
        );
    }
    for field in ["n", "price", "tag"] {
        let mut props = os_properties();
        props[field]["index"] = json!(false);
        assert!(
            validate_opensearch_schema(&props, &json!({"knn":false}), Some(&schema))
                .unwrap_err()
                .contains("indexed field")
        );
    }
    for normalizer in [json!("lowercase"), json!(""), json!(false), json!({})] {
        let mut props = os_properties();
        props["tag"]["normalizer"] = normalizer;
        assert!(
            validate_opensearch_schema(&props, &json!({"knn":false}), Some(&schema))
                .unwrap_err()
                .contains("case-sensitive")
        );
    }
}

fn redis_attrs(resp3: bool) -> Value {
    if resp3 {
        json!([
            {"attribute":"tag","type":"TAG","SEPARATOR":";","flags":["CASESENSITIVE"]},
            {"attribute":"n","type":"NUMERIC"},
            {"attribute":"price","type":"NUMERIC"}
        ])
    } else {
        json!([
            [
                "identifier",
                "tag",
                "attribute",
                "tag",
                "type",
                "TAG",
                "SEPARATOR",
                ";",
                "CASESENSITIVE"
            ],
            ["attribute", "n", "type", "NUMERIC"],
            ["attribute", "price", "type", "NUMERIC"]
        ])
    }
}

#[test]
fn redis_schema_accepts_resp2_resp3_mixed_forms_and_non_scalar_schema() {
    let schema = scalar_schema();
    for resp3_attrs in [false, true] {
        let attrs = redis_attrs(resp3_attrs);
        for info in [
            json!({"attributes":attrs}),
            json!(["index_name", "idx", "attributes", attrs]),
        ] {
            for schema in [None, Some(&schema), Some(&Value::Null), Some(&json!([]))] {
                assert!(validate_redis_schema(&info, schema).is_ok(), "{info}");
            }
        }
    }
    // CASESENSITIVE may also be exposed under a flags array in RESP2 pairs.
    let info = json!({"attributes":[["attribute","tag","type","TAG","SEPARATOR",";","flags",["CASESENSITIVE"]]]});
    assert!(validate_redis_schema(&info, Some(&json!({"tag":"keyword"}))).is_ok());
    assert!(validate_redis_schema(
        &json!({"attributes":[]}),
        Some(&json!({"geo":"geo","other":null}))
    )
    .is_ok());
}

#[test]
fn redis_schema_rejects_missing_metadata_and_vector_attributes_without_schema() {
    for info in [
        Value::Null,
        json!({}),
        json!([]),
        json!(["attributes"]),
        json!({"attributes":{}}),
        json!({"attributes":null}),
    ] {
        assert!(validate_redis_schema(&info, None)
            .unwrap_err()
            .contains("attributes"));
    }
    for attr in [
        Value::Null,
        json!({}),
        json!([]),
        json!(["type"]),
        json!({"type":123}),
    ] {
        assert!(validate_redis_schema(&json!({"attributes":[attr]}), None)
            .unwrap_err()
            .contains("attribute type"));
    }
    for kind in ["VECTOR", "vector", "VeCtOr"] {
        assert!(
            validate_redis_schema(&json!({"attributes":[{"type":kind}]}), None)
                .unwrap_err()
                .contains("containing VECTOR")
        );
    }
}

#[test]
fn redis_schema_requires_present_numeric_and_exact_keyword_attributes() {
    for kind in ["int", "float", "keyword"] {
        assert!(
            validate_redis_schema(&json!({"attributes":[]}), Some(&json!({"field":kind})))
                .unwrap_err()
                .contains("missing")
        );
    }
    for kind in ["int", "float"] {
        for actual in ["TAG", "TEXT", "numeric"] {
            let info = json!({"attributes":[{"attribute":"field","type":actual}]});
            assert!(validate_redis_schema(&info, Some(&json!({"field":kind})))
                .unwrap_err()
                .contains("NUMERIC"));
        }
    }
    for separator in [Value::Null, json!(","), json!(""), json!(0)] {
        let info = json!({"attributes":[{"attribute":"tag","type":"TAG","SEPARATOR":separator,"flags":["CASESENSITIVE"]}]});
        assert!(
            validate_redis_schema(&info, Some(&json!({"tag":"keyword"})))
                .unwrap_err()
                .contains("separator")
        );
    }
    for flags in [
        Value::Null,
        json!([]),
        json!(["SORTABLE"]),
        json!("CASESENSITIVE"),
        json!(["casesensitive"]),
    ] {
        let info =
            json!({"attributes":[{"attribute":"tag","type":"TAG","SEPARATOR":";","flags":flags}]});
        assert!(
            validate_redis_schema(&info, Some(&json!({"tag":"keyword"})))
                .unwrap_err()
                .contains("CASESENSITIVE")
        );
    }
    for actual in ["TEXT", "tag"] {
        let info = json!({"attributes":[{"attribute":"tag","type":actual,"SEPARATOR":";","flags":["CASESENSITIVE"]}]});
        assert!(
            validate_redis_schema(&info, Some(&json!({"tag":"keyword"})))
                .unwrap_err()
                .contains("CASESENSITIVE")
        );
    }
    let resp2 = json!([
        "attributes",
        [["attribute", "tag", "type", "TAG", "SEPARATOR", ";"]]
    ]);
    assert!(validate_redis_schema(&resp2, Some(&json!({"tag":"keyword"}))).is_err());
}

#[test]
fn page_validation_handles_zero_and_exact_limit_boundaries() {
    for (total, returned, limit) in [
        (0, 0, 0),
        (0, 0, 10),
        (4, 0, 0),
        (2, 2, 10),
        (2, 2, 2),
        (10, 2, 2),
        (u64::MAX, 2, 2),
    ] {
        assert!(validate_page(total, returned, limit).is_ok());
    }
    for (total, returned, limit) in [(0, 1, 10), (1, 0, 10), (3, 2, 10), (10, 3, 2), (4, 1, 0)] {
        assert!(validate_page(total, returned, limit).is_err());
    }
}

fn os_reply(total: u64, hits: Value) -> Value {
    json!({"timed_out":false,"_shards":{"failed":0},"hits":{"total":{"value":total,"relation":"eq"},"hits":hits}})
}

#[test]
fn opensearch_counts_accepts_fallback_ids_and_zero_sized_pages() {
    let body = os_reply(10, json!([{"_id":"a"},{"fields":{"_id":["b"]}}]));
    assert_eq!(opensearch_counts(&body, 2).unwrap(), (10, 2));
    assert_eq!(
        opensearch_counts(&os_reply(7, json!([])), 0).unwrap(),
        (7, 0)
    );
    assert_eq!(
        opensearch_counts(&os_reply(0, json!([])), 10).unwrap(),
        (0, 0)
    );
    let body = os_reply(1, json!([{"_id":123,"fields":{"_id":["fallback"]}}]));
    assert_eq!(opensearch_counts(&body, 10).unwrap(), (1, 1));
}

#[test]
fn opensearch_counts_rejects_missing_or_wrong_type_completion_metadata() {
    for (pointer, bad) in [
        ("/timed_out", Value::Null),
        ("/timed_out", json!("false")),
        ("/_shards", Value::Null),
        ("/_shards/failed", Value::Null),
        ("/_shards/failed", json!("0")),
        ("/_shards/failed", json!(-1)),
        ("/hits/total/relation", Value::Null),
        ("/hits/total/relation", json!(0)),
        ("/hits/total/value", Value::Null),
        ("/hits/total/value", json!(-1)),
        ("/hits/total/value", json!(1.5)),
        ("/hits/total/value", json!("1")),
        ("/hits/hits", Value::Null),
        ("/hits/hits", json!({})),
    ] {
        let mut body = os_reply(1, json!([{"_id":"a"}]));
        *body.pointer_mut(pointer).unwrap() = bad;
        assert!(
            opensearch_counts(&body, 10).is_err(),
            "accepted {pointer}: {body}"
        );
    }
    for key in ["timed_out", "_shards", "hits"] {
        let mut body = os_reply(1, json!([{"_id":"a"}]));
        body.as_object_mut().unwrap().remove(key);
        assert!(opensearch_counts(&body, 10).is_err());
    }
    for hit in [
        Value::Null,
        json!({}),
        json!({"_id":1}),
        json!({"fields":{"_id":[]}}),
        json!({"fields":{"_id":[null]}}),
    ] {
        assert!(opensearch_counts(&os_reply(1, json!([hit])), 10)
            .unwrap_err()
            .contains("ID"));
    }
    let duplicate = os_reply(2, json!([{"_id":"a"},{"fields":{"_id":["a"]}}]));
    assert!(opensearch_counts(&duplicate, 10)
        .unwrap_err()
        .contains("duplicate"));
}

fn bulk(s: &str) -> Resp {
    Resp::BulkString(s.as_bytes().to_vec())
}
fn resp_reply(total: Resp, hits: Resp) -> Resp {
    Resp::Map(vec![
        (bulk("warning"), Resp::Array(vec![])),
        (bulk("total_results"), total),
        (bulk("results"), hits),
    ])
}
fn resp_hit(id: Resp) -> Resp {
    Resp::Map(vec![(bulk("id"), id)])
}

#[test]
fn redis_counts_accepts_simple_strings_and_ignores_nontext_extension_keys() {
    let body = Resp::Map(vec![
        (Resp::Int(1), Resp::Nil),
        (Resp::BulkString(vec![0xff]), Resp::Nil),
        (Resp::SimpleString("warning".into()), Resp::Array(vec![])),
        (Resp::SimpleString("total_results".into()), Resp::Int(1)),
        (
            Resp::SimpleString("results".into()),
            Resp::Array(vec![Resp::Map(vec![(
                Resp::SimpleString("id".into()),
                Resp::SimpleString("document".into()),
            )])]),
        ),
    ]);
    assert_eq!(redis_counts(&body, 10).unwrap(), (1, 1));
    for (total, limit) in [(0, 0), (0, 10), (9, 0)] {
        assert_eq!(
            redis_counts(&resp_reply(Resp::Int(total), Resp::Array(vec![])), limit).unwrap(),
            (total as u64, 0)
        );
    }
}

#[test]
fn redis_counts_rejects_missing_fields_bad_types_and_malformed_ids() {
    for nonmap in [Resp::Nil, Resp::Int(0), bulk("x"), Resp::Array(vec![])] {
        assert!(redis_counts(&nonmap, 10).is_err());
    }
    let good_fields = match resp_reply(Resp::Int(1), Resp::Array(vec![resp_hit(bulk("a"))])) {
        Resp::Map(fields) => fields,
        _ => unreachable!(),
    };
    for omitted in 0..good_fields.len() {
        let mut fields = good_fields.clone();
        fields.remove(omitted);
        assert!(redis_counts(&Resp::Map(fields), 10).is_err());
    }
    for warning in [Resp::Nil, bulk(""), Resp::Array(vec![bulk("Timeout")])] {
        let mut fields = good_fields.clone();
        fields[0].1 = warning;
        assert!(redis_counts(&Resp::Map(fields), 10)
            .unwrap_err()
            .contains("warning"));
    }
    for total in [Resp::Nil, Resp::Int(-1), bulk("1"), Resp::Double(1.0)] {
        assert!(redis_counts(&resp_reply(total, Resp::Array(vec![])), 10)
            .unwrap_err()
            .contains("count"));
    }
    for hits in [Resp::Nil, Resp::Map(vec![]), bulk("[]")] {
        assert!(redis_counts(&resp_reply(Resp::Int(0), hits), 10)
            .unwrap_err()
            .contains("results"));
    }
    for hit in [Resp::Nil, Resp::Int(1), Resp::Array(vec![])] {
        assert!(
            redis_counts(&resp_reply(Resp::Int(1), Resp::Array(vec![hit])), 10)
                .unwrap_err()
                .contains("invalid Redis result")
        );
    }
    for id in [
        Resp::Nil,
        Resp::Int(7),
        Resp::Array(vec![]),
        Resp::BulkString(vec![0xff]),
    ] {
        assert!(redis_counts(
            &resp_reply(Resp::Int(1), Resp::Array(vec![resp_hit(id)])),
            10
        )
        .unwrap_err()
        .contains("ID"));
    }
    assert!(redis_counts(
        &resp_reply(Resp::Int(1), Resp::Array(vec![Resp::Map(vec![])])),
        10
    )
    .unwrap_err()
    .contains("ID"));
    let duplicate = resp_reply(
        Resp::Int(2),
        Resp::Array(vec![
            resp_hit(bulk("a")),
            resp_hit(Resp::SimpleString("a".into())),
        ]),
    );
    assert!(redis_counts(&duplicate, 10)
        .unwrap_err()
        .contains("duplicate"));
}

#[test]
fn scalar_stats_roundtrip_keeps_exact_integer_counts() {
    let stats = Stats {
        result_limit: 0,
        matched_documents_sum: u64::MAX,
        returned_documents_sum: 0,
        retried_queries: 7,
        per_query_counts_checked: Some(u64::MAX),
        index_layout: None,
    };
    let encoded = serde_json::to_value(&stats).unwrap();
    let decoded: Stats = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded.result_limit, 0);
    assert_eq!(decoded.matched_documents_sum, u64::MAX);
    assert_eq!(decoded.returned_documents_sum, 0);
    assert_eq!(decoded.retried_queries, 7);
    assert_eq!(decoded.per_query_counts_checked, Some(u64::MAX));
    // Absent (no prepared counts) stays absent in the artifact, and an old
    // artifact without the key still reads.
    let legacy = serde_json::to_value(Stats::default()).unwrap();
    assert!(legacy.get("per_query_counts_checked").is_none(), "{legacy}");
    let decoded: Stats = serde_json::from_value(legacy).unwrap();
    assert_eq!(decoded.per_query_counts_checked, None);
    assert_eq!(Stats::default().retried_queries, 0);
    assert_eq!(Stats::default().matched_documents_sum, 0);
}

#[test]
fn redis_schema_does_not_confuse_values_with_field_names_or_flags() {
    for name in ["type", "attribute", "CASESENSITIVE", "SEPARATOR"] {
        let attr = json!([
            "identifier",
            name,
            "attribute",
            name,
            "type",
            "TAG",
            "SEPARATOR",
            ";",
            "SORTABLE",
            "UNF",
            "CASESENSITIVE"
        ]);
        let schema = json!({name:"keyword"});
        validate_redis_schema(
            &json!(["index_name", "attributes", "attributes", [attr]]),
            Some(&schema),
        )
        .unwrap();
    }
    let attr = json!([
        "identifier",
        "CASESENSITIVE",
        "attribute",
        "CASESENSITIVE",
        "type",
        "TAG",
        "SEPARATOR",
        ";",
        "SORTABLE",
        "UNF"
    ]);
    assert!(validate_redis_schema(
        &json!({"attributes":[attr]}),
        Some(&json!({"CASESENSITIVE":"keyword"}))
    )
    .is_err());
}

#[test]
fn redis_schema_rejects_unindexed_scalar_attributes() {
    for (name, kind, field_type) in [
        ("tag", "keyword", "TAG"),
        ("n", "int", "NUMERIC"),
        ("f", "float", "NUMERIC"),
    ] {
        for attr in [
            json!([
                "attribute",
                name,
                "type",
                field_type,
                "SEPARATOR",
                ";",
                "CASESENSITIVE",
                "NOINDEX"
            ]),
            json!({"attribute":name,"type":field_type,"SEPARATOR":";","flags":["CASESENSITIVE","NOINDEX"]}),
        ] {
            assert!(validate_redis_schema(
                &json!({"attributes":[attr]}),
                Some(&json!({name:kind}))
            )
            .unwrap_err()
            .contains("indexed field"));
        }
    }
}

#[test]
fn scalar_search_params_keep_first_config_per_parallel_with_effective_defaults() {
    assert!(search_param_indices([], Some(10)).unwrap().is_empty());
    assert_eq!(
        search_param_indices(
            [
                (None, None),
                (Some(1), Some(10)),
                (Some(100), Some(3)),
                (Some(100), Some(3)),
                (Some(2), None),
                (Some(2), Some(10)),
            ],
            Some(10)
        )
        .unwrap(),
        vec![0, 2, 4]
    );
    // Parallel levels may deliberately use different limits. Only a config
    // that would be silently discarded at the same level is conflicting.
    assert_eq!(
        search_param_indices([(Some(100), Some(3)), (None, Some(20))], Some(10)).unwrap(),
        vec![0, 1]
    );
}

#[test]
fn scalar_search_params_reject_different_effective_limits_at_same_parallel() {
    for params in [
        vec![(None, None), (Some(1), Some(3))],
        vec![(Some(4), Some(3)), (Some(4), None)],
        vec![(Some(4), Some(3)), (Some(2), Some(3)), (Some(4), Some(4))],
    ] {
        let error = search_param_indices(params, Some(10)).unwrap_err();
        assert!(error.contains("conflicting result limits"), "{error}");
        assert!(error.contains("parallel="), "{error}");
        assert!(error.contains("top="), "{error}");
    }
}

#[test]
fn legacy_search_params_do_not_invent_a_ten_result_default() {
    // Valkey/Mongo can derive 100 from ground truth; None is not Some(10).
    for params in [
        [(None, None), (Some(1), Some(10))],
        [(None, Some(10)), (Some(1), None)],
    ] {
        assert!(search_param_indices(params, None).is_err());
        assert_eq!(search_param_indices(params, Some(10)).unwrap(), vec![0]);
    }
    assert_eq!(
        search_param_indices([(None, None), (Some(1), None), (Some(100), None)], None).unwrap(),
        vec![0, 2]
    );
    assert_eq!(
        search_param_indices([(None, Some(10)), (Some(1), Some(10))], None).unwrap(),
        vec![0]
    );
}

#[test]
fn opensearch_keyword_reuse_rejects_length_truncation() {
    let schema = scalar_schema();
    let mut properties = os_properties();
    for limit in [json!(0), json!(1), json!(256), json!(2147483647)] {
        properties["tag"]["ignore_above"] = limit;
        let error = validate_opensearch_schema(&properties, &json!({"knn":false}), Some(&schema))
            .unwrap_err();
        assert!(error.contains("ignore_above"), "{error}");
    }
    properties["tag"]["ignore_above"] = Value::Null;
    validate_opensearch_schema(&properties, &json!({"knn":false}), Some(&schema)).unwrap();
}

#[test]
fn missing_numeric_schema_field_diagnostic_does_not_claim_tag() {
    let error = validate_redis_schema(&json!({"attributes":[]}), Some(&json!({"price":"float"})))
        .unwrap_err();
    assert_eq!(error, "filter-only missing indexed field price");
}

#[test]
fn scalar_engine_selection_refuses_every_duplicate_type_before_collapsing_names() {
    validate_engine_selection([]).unwrap();
    validate_engine_selection([("r", "redis"), ("os", "opensearch"), ("v", "valkey")]).unwrap();
    for engine in ["redis", "opensearch", "valkey", "mongodb"] {
        let error =
            validate_engine_selection([("first", engine), ("other", "qdrant"), ("second", engine)])
                .unwrap_err();
        for detail in ["one config per engine type", "first", "second", engine] {
            assert!(error.contains(detail), "{error}");
        }
    }
}

#[test]
fn opensearch_keyword_reuse_rejects_query_token_splitting() {
    let schema = scalar_schema();
    let mut properties = os_properties();
    for bad in [json!(true), json!("true"), json!(0)] {
        properties["tag"]["split_queries_on_whitespace"] = bad;
        assert!(
            validate_opensearch_schema(&properties, &json!({"knn":false}), Some(&schema))
                .unwrap_err()
                .contains("split_queries_on_whitespace")
        );
    }
    for good in [json!(false), Value::Null] {
        properties["tag"]["split_queries_on_whitespace"] = good;
        validate_opensearch_schema(&properties, &json!({"knn":false}), Some(&schema)).unwrap();
    }
}
