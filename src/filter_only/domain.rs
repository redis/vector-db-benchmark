//! The shared domain for scalar Redis/OpenSearch comparisons.
//!
//! Redis TAG splits on `;` and trims edge whitespace, unlike an OpenSearch
//! keyword. Reject those tokens (and empty tokens) instead of comparing different
//! predicates. Redis NUMERIC uses binary64: integer data and operands must lie
//! within +/-2^53. Numeric query operands must be JSON numbers, and numeric
//! `match.any` currently supports integer lists only, matching the Redis parser.
//! These checks apply only to declared `keyword`, `int`, and `float` fields;
//! they do not impose a new contract on text, geo, datetime, or untyped fields.

use crate::readers::metadata::{MetadataItem, MetadataValue};
use serde_json::Value;

const MAX_EXACT_INTEGER: i64 = 1_i64 << 53;

fn error(field: &str, reason: &str) -> String {
    // Names are user-controlled too. Never include full operands or documents.
    let name: String = field.chars().take(64).collect();
    format!("filter-only field {name:?}: {reason}")
}

fn keyword(field: &str, token: &str) -> Result<(), String> {
    if token.is_empty() || token.trim() != token || token.contains(';') {
        return Err(error(
            field,
            "keyword tokens must be nonempty, have no edge whitespace, and contain no ';'",
        ));
    }
    Ok(())
}

fn integer(field: &str, value: i64) -> Result<(), String> {
    if !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&value) {
        return Err(error(
            field,
            "integer exceeds the exact binary64 range +/-2^53",
        ));
    }
    Ok(())
}

fn finite(field: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() {
        return Err(error(field, "numeric values must be finite"));
    }
    // OpenSearch double terms distinguish -0.0 from +0.0; Redis NUMERIC
    // compares them equally. Refuse the ambiguous value in the shared domain.
    if value == 0.0 && value.is_sign_negative() {
        return Err(error(
            field,
            "negative zero is outside the common numeric domain",
        ));
    }
    Ok(())
}

fn kind<'a>(schema: Option<&'a Value>, field: &str) -> Option<&'a str> {
    schema?.get(field)?.as_str()
}

/// Check the metadata that will be uploaded, before starting a measured run.
/// Missing values remain missing; this guard does not require dense metadata.
pub fn validate_metadata(
    metadata: &[Option<MetadataItem>],
    schema: Option<&Value>,
) -> Result<(), String> {
    for item in metadata.iter().flatten() {
        for (field, value) in &item.fields {
            match kind(schema, field) {
                Some("keyword") => match value {
                    MetadataValue::String(s) => keyword(field, s)?,
                    MetadataValue::Labels(labels) => {
                        for token in labels {
                            keyword(field, token)?;
                        }
                    }
                    // Both upload paths coerce numeric keyword data to strings.
                    MetadataValue::Int(_) => {}
                    MetadataValue::Float(n) => finite(field, *n)?,
                    MetadataValue::Geo { .. } => {
                        return Err(error(field, "keyword metadata must be scalar or labels"));
                    }
                },
                Some("int") => match value {
                    MetadataValue::Int(n) => integer(field, *n)?,
                    _ => return Err(error(field, "integer metadata must be a JSON integer")),
                },
                Some("float") => match value {
                    MetadataValue::Int(n) => integer(field, *n)?,
                    MetadataValue::Float(n) => finite(field, *n)?,
                    _ => return Err(error(field, "numeric metadata must be a JSON number")),
                },
                _ => {}
            }
        }
    }
    Ok(())
}

fn number(field: &str, value: &Value, integer_only: bool) -> Result<(), String> {
    if let Some(n) = value.as_i64() {
        return integer(field, n);
    }
    // Do not round a large unsigned integer through as_f64() before checking it.
    if value.as_u64().is_some() {
        return Err(error(
            field,
            "integer exceeds the exact binary64 range +/-2^53",
        ));
    }
    if !integer_only {
        if let Some(n) = value.as_f64() {
            return finite(field, n);
        }
    }
    Err(error(
        field,
        if integer_only {
            "this numeric operand must be a JSON integer"
        } else {
            "numeric operands must be JSON numbers"
        },
    ))
}

fn operand(field: &str, value: &Value, field_kind: &str, integer_only: bool) -> Result<(), String> {
    if field_kind == "keyword" {
        let token = value
            .as_str()
            .ok_or_else(|| error(field, "keyword query operands must be strings"))?;
        keyword(field, token)
    } else {
        number(field, value, integer_only)
    }
}

fn leaf(field: &str, filters: &Value, field_kind: &str) -> Result<(), String> {
    let filters = filters
        .as_object()
        .ok_or_else(|| error(field, "expected an object of scalar predicates"))?;
    for (operation, criteria) in filters {
        match operation.as_str() {
            "match" => {
                // Match the adapters' precedence: any, then text, then value.
                if let Some(any) = criteria.get("any") {
                    let values = any
                        .as_array()
                        .ok_or_else(|| error(field, "match.any must be an array"))?;
                    if field_kind != "keyword" && values.is_empty() {
                        return Err(error(field, "numeric match.any requires an integer list"));
                    }
                    for value in values {
                        operand(field, value, field_kind, true)?;
                    }
                } else if criteria.get("text").is_some() {
                    return Err(error(
                        field,
                        "text matching is outside the numeric/tag contract",
                    ));
                } else {
                    let value = criteria
                        .get("value")
                        .ok_or_else(|| error(field, "match requires value or any"))?;
                    operand(field, value, field_kind, field_kind == "int")?;
                }
            }
            "range" => {
                if field_kind == "keyword" {
                    return Err(error(
                        field,
                        "keyword ranges are outside the numeric/tag contract",
                    ));
                }
                let bounds = criteria
                    .as_object()
                    .ok_or_else(|| error(field, "range must be an object"))?;
                if bounds.is_empty() {
                    return Err(error(field, "range requires a numeric bound"));
                }
                for (bound, value) in bounds {
                    if !matches!(bound.as_str(), "gt" | "gte" | "lt" | "lte") {
                        return Err(error(field, "range supports only gt/gte/lt/lte"));
                    }
                    number(field, value, false)?;
                    // Integer ranges may use fractional bounds, but must not
                    // escape the common exact-integer domain via float syntax.
                    if field_kind == "int"
                        && value
                            .as_f64()
                            .is_some_and(|n| n.abs() > MAX_EXACT_INTEGER as f64)
                    {
                        return Err(error(field, "integer bound exceeds +/-2^53"));
                    }
                }
            }
            _ => return Err(error(field, "unsupported numeric/tag predicate operation")),
        }
    }
    if filters.is_empty() {
        return Err(error(field, "expected a scalar predicate"));
    }
    Ok(())
}

/// Inspect boolean groups and field leaves, never arbitrary JSON inside an
/// operand. Non-scalar schema fields remain the existing parser's responsibility.
pub fn validate_predicate(condition: &Value, schema: Option<&Value>) -> Result<(), String> {
    let Some(object) = condition.as_object() else {
        return Ok(());
    };
    if object.contains_key("and") || object.contains_key("or") {
        for name in ["and", "or"] {
            if let Some(entries) = object.get(name).and_then(Value::as_array) {
                for entry in entries {
                    validate_predicate(entry, schema)?;
                }
            }
        }
    } else {
        for (field, filters) in object {
            if let Some(field_kind @ ("keyword" | "int" | "float")) = kind(schema, field) {
                leaf(field, filters, field_kind)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> Value {
        json!({"tag":"keyword", "labels":"keyword", "n":"int", "f":"float", "body":"text"})
    }

    fn metadata(field: &str, value: MetadataValue) -> Vec<Option<MetadataItem>> {
        vec![
            None,
            Some(MetadataItem {
                fields: vec![(field.into(), value)],
            }),
        ]
    }

    #[test]
    fn rejects_negative_zero_but_accepts_positive_zero() {
        for n in [-0.0_f64, 0.0] {
            let expected = !n.is_sign_negative();
            assert_eq!(
                validate_metadata(&metadata("f", MetadataValue::Float(n)), Some(&schema())).is_ok(),
                expected
            );
            for condition in [
                json!({"f":{"match":{"value":n}}}),
                json!({"f":{"range":{"gte":n}}}),
            ] {
                assert_eq!(
                    validate_predicate(&condition, Some(&schema())).is_ok(),
                    expected
                );
            }
        }
    }

    #[test]
    fn accepts_safe_metadata_and_missing_values() {
        for (field, value) in [
            ("tag", MetadataValue::String("Mixed Case café".into())),
            ("tag", MetadataValue::Int(i64::MAX)),
            ("tag", MetadataValue::Float(1.5)),
            (
                "labels",
                MetadataValue::Labels(vec!["a".into(), "B".into()]),
            ),
            ("labels", MetadataValue::Labels(vec![])),
            ("n", MetadataValue::Int(MAX_EXACT_INTEGER)),
            ("n", MetadataValue::Int(-MAX_EXACT_INTEGER)),
            ("f", MetadataValue::Float(0.1)),
            ("f", MetadataValue::Float(1e30)),
            ("f", MetadataValue::Int(42)),
        ] {
            validate_metadata(&metadata(field, value), Some(&schema())).unwrap();
        }
        validate_metadata(&[None], Some(&schema())).unwrap();
    }

    #[test]
    fn rejects_ambiguous_keyword_tokens_in_strings_and_labels() {
        for token in ["", "a;b", " a", "a ", "\ta", "a\n", "\u{2003}a"] {
            for value in [
                MetadataValue::String(token.into()),
                MetadataValue::Labels(vec!["safe".into(), token.into()]),
            ] {
                assert!(validate_metadata(&metadata("tag", value), Some(&schema())).is_err());
            }
            assert!(validate_predicate(
                &json!({"and":[{"tag":{"match":{"value":token}}}]}),
                Some(&schema())
            )
            .is_err());
        }
    }

    #[test]
    fn rejects_lossy_integer_and_malformed_numeric_metadata() {
        for field in ["n", "f"] {
            for value in [
                MetadataValue::Int(MAX_EXACT_INTEGER + 1),
                MetadataValue::Int(-MAX_EXACT_INTEGER - 1),
                MetadataValue::Int(i64::MIN),
                MetadataValue::String("12".into()),
                MetadataValue::Labels(vec!["12".into()]),
                MetadataValue::Geo { lon: 1.0, lat: 2.0 },
            ] {
                assert!(validate_metadata(&metadata(field, value), Some(&schema())).is_err());
            }
        }
        assert!(
            validate_metadata(&metadata("n", MetadataValue::Float(1.5)), Some(&schema())).is_err()
        );
        for field in ["f", "tag"] {
            for n in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert!(validate_metadata(
                    &metadata(field, MetadataValue::Float(n)),
                    Some(&schema())
                )
                .is_err());
            }
        }
    }

    #[test]
    fn accepts_nested_groups_numeric_bounds_and_empty_membership() {
        let condition = json!({"and":[
            {"or":[{"tag":{"match":{"any":["a","B"]}}}, {"n":{"match":{"any":[1,2]}}}]},
            {"n":{"range":{"gt":-MAX_EXACT_INTEGER,"lte":MAX_EXACT_INTEGER}}},
            {"n":{"range":{"gt":0.5,"lt":10.5}}},
            {"f":{"range":{"gte":0.1,"lte":0.2}}},
            {"f":{"match":{"value":0.1}}},
            {"f":{"match":{"any":[1,2]}}},
            {"tag":{"match":{"any":[]}}}
        ]});
        validate_predicate(&condition, Some(&schema())).unwrap();
    }

    #[test]
    fn checks_every_numeric_operand_without_unsigned_rounding() {
        for value in [
            json!(MAX_EXACT_INTEGER + 1),
            json!(-MAX_EXACT_INTEGER - 1),
            json!(u64::MAX),
            json!(null),
            json!("12"),
            json!(true),
            json!([1]),
        ] {
            for field in ["n", "f"] {
                for predicate in [
                    json!({"match":{"value":value}}),
                    json!({"match":{"any":[0,value]}}),
                    json!({"range":{"gte":0,"lt":value}}),
                ] {
                    assert!(validate_predicate(
                        &json!({"and":[{(field):predicate}]}),
                        Some(&schema())
                    )
                    .is_err());
                }
            }
        }
        assert!(validate_predicate(&json!({"n":{"range":{"gt":1e30}}}), Some(&schema())).is_err());
        assert!(
            validate_predicate(&json!({"n":{"match":{"value":1.5}}}), Some(&schema())).is_err()
        );
        assert!(
            validate_predicate(&json!({"f":{"match":{"any":[0.1,0.2]}}}), Some(&schema())).is_err()
        );
        assert!(validate_predicate(&json!({"n":{"match":{"any":[]}}}), Some(&schema())).is_err());
    }

    #[test]
    fn rejects_dropped_or_differently_interpreted_predicates() {
        for predicate in [
            json!({}),
            json!(null),
            json!({"match":{}}),
            json!({"match":{"any":"a"}}),
            json!({"match":{"text":"a"}}),
            json!({"match":{"value":42}}),
            json!({"match":{"any":["a",42]}}),
            json!({"range":{"gt":"a"}}),
            json!({"geo":{}}),
        ] {
            assert!(validate_predicate(&json!({"tag":predicate}), Some(&schema())).is_err());
        }
        for range in [
            json!({}),
            json!({"wat":1}),
            json!({"gte":1,"wat":2}),
            json!([]),
        ] {
            assert!(validate_predicate(&json!({"n":{"range":range}}), Some(&schema())).is_err());
        }
    }

    #[test]
    fn ignores_unrelated_fields_and_respects_parser_precedence() {
        let condition = json!({"and":[{"body":{"match":{"text":" text; "}}},{"unknown":{"match":{"value":" x; "}}}]});
        validate_predicate(&condition, Some(&schema())).unwrap();
        validate_metadata(
            &metadata("body", MetadataValue::String(" x; ".into())),
            Some(&schema()),
        )
        .unwrap();
        validate_metadata(&metadata("tag", MetadataValue::String(" x; ".into())), None).unwrap();
        validate_predicate(
            &json!({"tag":{"match":{"any":["safe"],"value":" bad; ","text":"ignored"}}}),
            Some(&schema()),
        )
        .unwrap();
        validate_predicate(
            &json!({"and":[], "tag":{"match":{"value":"ignored;"}}}),
            Some(&schema()),
        )
        .unwrap();
        validate_predicate(&Value::Null, Some(&schema())).unwrap();
    }

    #[test]
    fn diagnostics_are_bounded_and_do_not_dump_operands() {
        let field = "x".repeat(10_000);
        let secret = "sensitive;".repeat(10_000);
        let error = keyword(&field, &secret).unwrap_err();
        assert!(error.len() < 256);
        assert!(!error.contains("sensitive"));
    }
}
