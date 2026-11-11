//! Reproducible reduced TEXT/TAG index layout. Payloads remain unchanged.
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScalarIndexing {
    #[default]
    All,
    TextTagOnly,
}

impl ScalarIndexing {
    pub fn is_reduced(self) -> bool {
        self == Self::TextTagOnly
    }

    pub fn project(self, schema: &Value) -> Value {
        if !self.is_reduced() {
            return schema.clone();
        }
        Value::Object(
            schema
                .as_object()
                .into_iter()
                .flatten()
                .filter(|(_, kind)| {
                    matches!(kind.as_str(), Some("text" | "keyword" | "uuid" | "bool"))
                })
                .map(|(name, kind)| (name.clone(), kind.clone()))
                .collect(),
        )
    }
}

pub fn validate_lifecycle(engine: &str, keep_data: bool) -> Result<(), String> {
    match engine {
        "redis" if !keep_data => Err("text-tag-only Redis requires retained data: use --keep-data without --reset-between-configs (or --skip-upload)".into()),
        "redis" | "opensearch" => Ok(()),
        _ => Err("text-tag-only is supported only by Redis and OpenSearch".into()),
    }
}

/// Ensure a claimed reduced layout did not retain a full scalar index.
pub fn validate_redis(info: &Value) -> Result<(), String> {
    use crate::filter_only::{ft_info_field, ft_info_has_flag};
    let attrs = ft_info_field(info, "attributes")
        .and_then(Value::as_array)
        .ok_or("text-tag-only requires FT.INFO attributes")?;
    for attr in attrs {
        if !matches!(
            ft_info_field(attr, "type").and_then(Value::as_str),
            Some("TEXT" | "TAG")
        ) || ft_info_has_flag(attr, "SORTABLE")
            || ft_info_has_flag(attr, "UNF")
        {
            return Err(
                "text-tag-only Redis index contains unsupported fields or sorting storage".into(),
            );
        }
    }
    Ok(())
}

pub fn disable_excluded(properties: &mut Value, schema: &Value) {
    let projected = ScalarIndexing::TextTagOnly.project(schema);
    for (name, _) in schema.as_object().into_iter().flatten() {
        if projected.get(name).is_none() {
            properties[name]["index"] = Value::Bool(false);
            properties[name]["doc_values"] = Value::Bool(false);
        }
    }
}

pub fn validate_opensearch(properties: &Value, schema: &Value) -> Result<(), String> {
    // Check every realized field, not just the registry: stray numeric fields
    // and multifields would otherwise silently retain a larger index layout.
    fn check_field(field: &Value) -> Result<(), String> {
        if !matches!(field["type"].as_str(), Some("text" | "keyword" | "boolean"))
            && (field["index"] != false || field["doc_values"] != false)
        {
            return Err("text-tag-only OpenSearch has an indexed non-text/tag property".into());
        }
        for child in field
            .get("fields")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            check_field(child.1)?;
        }
        if field.get("properties").is_some() {
            return Err("text-tag-only OpenSearch does not support nested properties".into());
        }
        Ok(())
    }
    for field in properties
        .as_object()
        .ok_or("missing mapping properties")?
        .values()
    {
        check_field(field)?;
    }
    let projected = ScalarIndexing::TextTagOnly.project(schema);
    for (name, _) in schema.as_object().into_iter().flatten() {
        if projected.get(name).is_none() {
            let field = &properties[name];
            if field["index"] != false
                || field["doc_values"] != false
                || field
                    .get("fields")
                    .is_some_and(|v| v.as_object().is_none_or(|o| !o.is_empty()))
            {
                return Err(format!("text-tag-only OpenSearch field {name} must disable indexing/doc values and have no multifields"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn projection_rejects_numeric_ast_even_with_tag_family() {
        let schema = json!({"title":"text","category":"keyword","n":"int"});
        let reduced = ScalarIndexing::TextTagOnly.project(&schema);
        assert!(
            crate::compound_text::Expr::compile(&json!({"n":{"range":{"gte":1}}}), &reduced)
                .is_err()
        );
        assert!(crate::compound_text::Expr::compile(
            &json!({"category":{"match":{"value":"X"}}}),
            &reduced
        )
        .is_ok());
        assert_eq!(ScalarIndexing::All.project(&schema), schema);
        for invalid in [json!(null), json!(true), json!("text-only")] {
            assert!(serde_json::from_value::<ScalarIndexing>(invalid).is_err());
        }
    }
    #[test]
    fn old_and_partially_disabled_indexes_are_rejected() {
        let schema = json!({"n":"int"});
        let mut properties = json!({"n":{"type":"long"}});
        assert!(validate_opensearch(&properties, &schema).is_err());
        disable_excluded(&mut properties, &schema);
        assert!(validate_opensearch(&properties, &schema).is_ok());
        let mut extra = properties.clone();
        extra["extra_numeric"] = json!({"type":"long"});
        assert!(validate_opensearch(&extra, &schema).is_err());
        properties["n"]["fields"] = json!({"raw":{"type":"keyword"}});
        assert!(validate_opensearch(&properties, &schema).is_err());
        for attr in [
            json!(["identifier", "n", "type", "NUMERIC"]),
            json!(["identifier", "t", "type", "TAG", "SORTABLE", "UNF"]),
            json!({"identifier":"t","type":"TEXT","flags":["SORTABLE"]}),
        ] {
            assert!(validate_redis(&json!({"attributes":[attr]})).is_err());
        }
        assert!(
            validate_redis(&json!({"attributes":[["identifier","t","type","TEXT","NOSTEM"]]}))
                .is_ok()
        );
    }
    /// The msmarco-disk-v1 `url-eq` / `docid-eq` / `docid-batch` families need no new index
    /// flag: the reduced layout already keeps every keyword field, and the
    /// pre-search validators already require each one as a CASESENSITIVE
    /// TAG (Redis) / case-sensitive keyword (OpenSearch). A reduced index
    /// missing `url` is refused before anything is timed.
    #[test]
    fn reduced_layout_indexes_and_requires_url_and_docid() {
        let schema = json!({"docid":"keyword","url":"keyword","title":"text","segment":"text",
            "category":"keyword","doc_len":"int"});
        let projected = ScalarIndexing::TextTagOnly.project(&schema);
        for f in ["docid", "url", "category"] {
            assert_eq!(projected[f], "keyword", "{f}");
        }
        let tag = |name: &str| {
            json!([
                "identifier",
                name,
                "attribute",
                name,
                "type",
                "TAG",
                "SEPARATOR",
                ";",
                "CASESENSITIVE"
            ])
        };
        let text = |name: &str| json!(["identifier", name, "attribute", name, "type", "TEXT"]);
        let full = json!({"attributes":[tag("docid"), tag("url"), tag("category"),
            text("title"), text("segment")]});
        crate::filter_only::validate_redis_schema(&full, Some(&projected)).unwrap();
        validate_redis(&full).unwrap();
        let no_url = json!({"attributes":[tag("docid"), tag("category"), text("title"),
            text("segment")]});
        assert!(
            crate::filter_only::validate_redis_schema(&no_url, Some(&projected))
                .unwrap_err()
                .contains("url")
        );
        let mut props = json!({"vector":{"type":"float","index":false,"doc_values":false},
            "docid":{"type":"keyword"},"url":{"type":"keyword"},"category":{"type":"keyword"},
            "title":{"type":"text"},"segment":{"type":"text"}});
        let index = json!({"knn":false});
        crate::filter_only::validate_opensearch_schema(&props, &index, Some(&projected)).unwrap();
        props["url"]["index"] = json!(false);
        assert!(
            crate::filter_only::validate_opensearch_schema(&props, &index, Some(&projected))
                .unwrap_err()
                .contains("url")
        );
    }
    #[test]
    fn teardown_and_other_engines_refused() {
        assert!(validate_lifecycle("redis", false).is_err());
        assert!(validate_lifecycle("redis", true).is_ok());
        assert!(validate_lifecycle("opensearch", false).is_ok());
        assert!(validate_lifecycle("qdrant", true).is_err());
    }
}
