//! Explicit Boolean text/scalar matching (#389). Scalar leaves reuse the
//! preparation oracle's typed grammar. This is a separate, unboosted matching
//! suite; scores/order are engine-specific and are not relevance truth.
use crate::msmarco::derived::{Leaf, Operand, Predicate};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::BufRead;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// Exactly one of `suite` and `queries` is required.
    pub suite: Option<String>,
    pub queries: Option<Vec<Query>>,
    /// Select one family for separately attributable throughput/latency.
    pub family: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    pub name: String,
    pub family: String,
    pub conditions: Value,
    #[serde(default)]
    pub expected_count: Option<u64>,
    #[serde(default)]
    pub minimum_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Not(Box<Expr>),
    Leaf(Predicate),
}

impl Expr {
    pub fn compile(value: &Value, schema: &Value) -> Result<Self, String> {
        Self::node(value, schema, 0)
    }
    fn node(value: &Value, schema: &Value, depth: usize) -> Result<Self, String> {
        if depth > 32 {
            return Err("compound query nesting exceeds 32".into());
        }
        let o = value
            .as_object()
            .filter(|o| o.len() == 1)
            .ok_or("compound condition must have exactly one operator/field")?;
        let (key, body) = o.iter().next().unwrap();
        match key.as_str() {
            "and" | "or" => {
                let a = body
                    .as_array()
                    .filter(|a| !a.is_empty())
                    .ok_or("empty Boolean group")?;
                let children = a
                    .iter()
                    .map(|v| Self::node(v, schema, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(if key == "and" {
                    Self::And(children)
                } else {
                    Self::Or(children)
                })
            }
            "not" => Ok(Self::Not(Box::new(Self::node(body, schema, depth + 1)?))),
            field => {
                if field.is_empty()
                    || !field
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'_')
                    || field.as_bytes()[0].is_ascii_digit()
                {
                    return Err("compound field must be a plain identifier".into());
                }
                let Predicate::And(mut p) = Predicate::compile(&json!({"and":[value]}))? else {
                    unreachable!()
                };
                let p = p.remove(0);
                let Predicate::Leaf { leaf, .. } = &p else {
                    unreachable!()
                };
                let kind = schema[field]
                    .as_str()
                    .ok_or_else(|| format!("unknown compound field {field}"))?;
                let valid = match leaf {
                    Leaf::Text(_) => kind == "text",
                    Leaf::Match(ops) => ops.iter().all(|v| {
                        matches!(
                            (kind, v),
                            ("keyword", Operand::Keyword(_)) | ("int", Operand::Int(_))
                        )
                    }),
                    Leaf::Range { .. } => kind == "int",
                };
                if !valid {
                    return Err(format!(
                        "compound predicate type disagrees with schema for {field}"
                    ));
                }
                if let Leaf::Match(ops) = leaf {
                    for op in ops {
                        if let Operand::Keyword(s) = op {
                            if s.is_empty()
                                || s.contains(';')
                                || s.trim() != s
                                || s.chars().any(char::is_control)
                            {
                                return Err("compound tags must be nonempty single tags without separator/control/edge whitespace".into());
                            }
                        }
                    }
                }
                Ok(Self::Leaf(p))
            }
        }
    }
    pub fn has_text(&self) -> bool {
        match self {
            Self::And(c) | Self::Or(c) => c.iter().any(Self::has_text),
            Self::Not(p) => p.has_text(),
            Self::Leaf(p) => crate::hybrid::has_text(p),
        }
    }
    pub fn text_fields(&self, out: &mut HashSet<String>) {
        match self {
            Self::And(c) | Self::Or(c) => {
                for p in c {
                    p.text_fields(out);
                }
            }
            Self::Not(p) => p.text_fields(out),
            Self::Leaf(Predicate::Leaf {
                field,
                leaf: Leaf::Text(_),
            }) => {
                out.insert(field.clone());
            }
            Self::Leaf(_) => {}
        }
    }
    /// Fixture/scalar oracle. Text uses the conservative shared tokenizer
    /// classifier, not an invented general English analyzer.
    pub fn matches(&self, row: &serde_json::Map<String, Value>) -> Result<bool, String> {
        match self {
            Self::And(c) => {
                for p in c {
                    if !p.matches(row)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Self::Or(c) => {
                for p in c {
                    if p.matches(row)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Self::Not(p) => Ok(!p.matches(row)?),
            Self::Leaf(p) => p.matches(row),
        }
    }
    pub fn redis(&self) -> String {
        fn esc(s: &str) -> String {
            s.chars()
                .flat_map(|c| {
                    if c.is_ascii_alphanumeric() {
                        vec![c]
                    } else {
                        vec!['\\', c]
                    }
                })
                .collect()
        }
        match self {
            Self::And(c) => format!(
                "({})",
                c.iter().map(Self::redis).collect::<Vec<_>>().join(" ")
            ),
            Self::Or(c) => format!(
                "({})",
                c.iter().map(Self::redis).collect::<Vec<_>>().join(" | ")
            ),
            Self::Not(p) => format!("(-{})", p.redis()),
            Self::Leaf(Predicate::Leaf { field, leaf }) => match leaf {
                Leaf::Text(t) => format!("(@{field}:({t}))"),
                Leaf::Match(ops) => format!(
                    "({})",
                    ops.iter()
                        .map(|v| match v {
                            Operand::Keyword(s) => format!("@{field}:{{{}}}", esc(s)),
                            Operand::Int(n) => format!("@{field}:[{n} {n}]"),
                        })
                        .collect::<Vec<_>>()
                        .join(" | ")
                ),
                Leaf::Range { gt, gte, lt, lte } => {
                    let lo = gt
                        .map(|n| format!("({n}"))
                        .or_else(|| gte.map(|n| n.to_string()))
                        .unwrap_or("-inf".into());
                    let hi = lt
                        .map(|n| format!("({n}"))
                        .or_else(|| lte.map(|n| n.to_string()))
                        .unwrap_or("+inf".into());
                    format!("(@{field}:[{lo} {hi}])")
                }
            },
            Self::Leaf(_) => unreachable!("compiler creates leaf predicates only"),
        }
    }
    pub fn opensearch(&self) -> Value {
        match self {
            Self::And(c) => {
                json!({"bool":{"must":c.iter().map(Self::opensearch).collect::<Vec<_>>()}})
            }
            Self::Or(c) => {
                json!({"bool":{"should":c.iter().map(Self::opensearch).collect::<Vec<_>>(),"minimum_should_match":1}})
            }
            Self::Not(p) => json!({"bool":{"must_not":[p.opensearch()]}}),
            Self::Leaf(Predicate::Leaf { field, leaf }) => match leaf {
                Leaf::Text(t) => json!({"match":{field:{"query":t,"operator":"and"}}}),
                Leaf::Match(ops) => {
                    json!({"terms":{field:ops.iter().map(|v|match v {Operand::Keyword(s)=>json!(s),Operand::Int(n)=>json!(n)}).collect::<Vec<_>>()}})
                }
                Leaf::Range { gt, gte, lt, lte } => {
                    let mut b = serde_json::Map::new();
                    for (k, v) in [("gt", gt), ("gte", gte), ("lt", lt), ("lte", lte)] {
                        if let Some(n) = v {
                            b.insert(k.into(), json!(n));
                        }
                    }
                    json!({"range":{field:b}})
                }
            },
            Self::Leaf(_) => unreachable!("compiler creates leaf predicates only"),
        }
    }
}

impl Spec {
    pub fn validate(&self) -> Result<(), String> {
        match (&self.suite, &self.queries) {
            (Some(s), None) if s == "msmarco-v1" => {}
            (None, Some(q)) if !q.is_empty() => {}
            _ => {
                return Err(
                    "compound requires suite: msmarco-v1 OR nonempty explicit queries".into(),
                )
            }
        }
        if self.family.as_ref().is_some_and(|s| s.is_empty()) {
            return Err("empty compound family".into());
        }
        Ok(())
    }
    pub fn load(&self, dir: &Path, schema: &Value) -> Result<Vec<Query>, String> {
        self.validate()?;
        let mut queries = match &self.queries {
            Some(q) => q.clone(),
            None => msmarco_queries(dir)?,
        };
        if let Some(f) = &self.family {
            queries.retain(|q| &q.family == f);
        }
        if queries.is_empty() {
            return Err("compound family selects no queries".into());
        }
        let mut names = HashSet::new();
        for q in &queries {
            if q.name.is_empty() || q.family.is_empty() || !names.insert(&q.name) {
                return Err("compound names must be unique and name/family nonempty".into());
            }
            let expr = Expr::compile(&q.conditions, schema)?;
            if !expr.has_text() && q.expected_count.is_none() {
                return Err(format!(
                    "scalar compound query {} needs an exact expected_count",
                    q.name
                ));
            }
        }
        Ok(queries)
    }
}

/// Bounded scan for an actual document with terms in both text fields. This
/// chooses positive examples, not a claim of representative term frequencies.
/// Scalar families retain the prepared corpus-wide exact counts unchanged.
fn msmarco_queries(dir: &Path) -> Result<Vec<Query>, String> {
    let file = std::fs::File::open(dir.join("payloads.jsonl")).map_err(|e| e.to_string())?;
    // Index only the bounded sample. Each selected exclusion must have a
    // witness that matches the segment term and contains the excluded title
    // term, while the seed itself remains a positive parity match.
    let mut rows = Vec::new();
    let mut segment_rows = std::collections::BTreeMap::<String, Vec<usize>>::new();
    for line in std::io::BufReader::new(file).lines().take(4096) {
        let row: Value =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let terms = |field: &str| -> Vec<String> {
            row[field]
                .as_str()
                .map(|text| {
                    crate::fulltext::query_terms(text)
                        .into_iter()
                        .filter(|term| {
                            crate::msmarco::text_match::validate_term(term).is_ok()
                                && crate::msmarco::text_match::classify(text, term)
                                    == crate::msmarco::text_match::TextMatch::Yes
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        let title = terms("title");
        let segment = terms("segment");
        for term in &segment {
            segment_rows
                .entry(term.clone())
                .or_default()
                .push(rows.len());
        }
        rows.push((row, title, segment));
    }
    let mut queries = Vec::new();
    for (row, title_terms, segment_terms) in &rows {
        if queries.len() >= 32 * 5 {
            break;
        }
        let (Some(title), Some(segment)) = (title_terms.first(), segment_terms.first()) else {
            continue;
        };
        let mut candidates = std::collections::BTreeMap::<&String, usize>::new();
        for &other in &segment_rows[segment] {
            for term in &rows[other].1 {
                if crate::msmarco::text_match::classify(row["title"].as_str().unwrap(), term)
                    == crate::msmarco::text_match::TextMatch::No
                {
                    *candidates.entry(term).or_default() += 1;
                }
            }
        }
        // Prefer the most frequent co-occurring title term; lexical tie-break
        // keeps this stable across executions and engines. No witness: skip
        // the seed instead of advertising a NOT that does no work.
        let Some((exclusion, _removed)) = candidates
            .into_iter()
            .max_by(|(ta, ca), (tb, cb)| ca.cmp(cb).then_with(|| tb.cmp(ta)))
        else {
            continue;
        };
        let i = queries.len() / 5;
        let text = |f: &str, t: &str| json!({f:{"match":{"text":t}}});
        let cat = json!({"category":{"match":{"value":row["category"]}}});
        let len = row["doc_len"]
            .as_u64()
            .ok_or("selected passage has no doc_len")?;
        let range = json!({"doc_len":{"range":{"gte":len.saturating_sub(16),"lte":len+16}}});
        for (family, conditions) in [
            ("segment", text("segment", segment)),
            (
                "title-and-segment",
                json!({"and":[text("title",title),text("segment",segment)]}),
            ),
            ("tag-and-text", json!({"and":[cat,text("segment",segment)]})),
            (
                "text-tag-range",
                json!({"and":[text("segment",segment),cat,range]}),
            ),
            (
                "text-exclusion",
                json!({"and":[text("segment",segment),{"not":text("title",exclusion)}]}),
            ),
        ] {
            queries.push(Query {
                name: format!("msmarco-compound-{family}-{i:03}"),
                family: family.into(),
                conditions,
                expected_count: None,
                minimum_count: 1,
            });
        }
    }
    if queries.is_empty() {
        return Err("no nontrivial text exclusion in first 4096 passages".into());
    }
    let manifest = crate::msmarco::preparation::read_manifest(dir)?;
    let dataset = manifest["dataset"]
        .as_str()
        .ok_or("preparation lacks dataset identity")?;
    let count = manifest["passages"]
        .as_u64()
        .ok_or("preparation lacks corpus row count")?;
    let set =
        crate::filter_only::query_set::FilterQuerySet::read_prepared(dir, dataset, Some(count))?
            .ok_or("prepared scalar queries missing")?;
    // This generator retains logical conditions, not raw engine syntax. The
    // typed engine translators compile them again against the declared schema.
    let resolved = set.try_resolve_all("compound-text generator", |v| {
        Predicate::compile(v)?;
        Ok(Some(v.clone()))
    })?;
    for (meta, conditions) in set
        .prepared()
        .ok_or("scalar query metadata missing")?
        .iter()
        .zip(resolved)
    {
        queries.push(Query {
            name: meta.name.clone(),
            family: meta.shape.clone(),
            conditions: conditions
                .into_inner()
                .ok_or("scalar condition was dropped")?,
            expected_count: Some(meta.expected_count),
            minimum_count: 0,
        });
    }
    Ok(queries)
}

pub fn check_count(query: &Query, count: u64) -> Result<(), String> {
    if count < query.minimum_count {
        return Err(format!(
            "compound {} returned {count}, below minimum {}",
            query.name, query.minimum_count
        ));
    }
    if let Some(expected) = query.expected_count {
        if expected != count {
            return Err(format!(
                "compound {} exact count: expected {expected}, got {count}",
                query.name
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn queries_with_minimum(q: &Query) -> Query {
        let mut q = q.clone();
        q.minimum_count = 1;
        q
    }
    fn schema() -> Value {
        json!({"title":"text","segment":"text","category":"keyword","doc_len":"int"})
    }
    #[test]
    fn boolean_precedence_and_missing_fields() {
        let v = json!({"and":[{"or":[{"title":{"match":{"text":"alpha"}}},{"category":{"match":{"value":"X"}}}]},{"not":{"doc_len":{"range":{"gte":5,"lt":10}}}}]});
        let e = Expr::compile(&v, &schema()).unwrap();
        for (row, want) in [
            (json!({"title":"ALPHA","doc_len":4}), true),
            (json!({"category":"X","doc_len":5}), false),
            (json!({"category":"x","doc_len":4}), false),
            (json!({"category":"X"}), true),
        ] {
            assert_eq!(e.matches(row.as_object().unwrap()).unwrap(), want);
        }
        assert_eq!(
            e.redis(),
            "(((@title:(alpha)) | (@category:{X})) (-(@doc_len:[5 (10])))"
        );
        assert_eq!(
            e.opensearch()["bool"]["must"][0]["bool"]["minimum_should_match"],
            1
        );
        assert_eq!(
            e.opensearch()["bool"]["must"][1]["bool"]["must_not"][0]["range"]["doc_len"]["lt"],
            10.0
        );
    }
    #[test]
    fn refuses_ambiguous_types_and_raw_language() {
        for v in [
            json!({"and":[]}),
            json!({"title":{"match":{"text":"x | y"}}}),
            json!({"title":{"match":{"value":"alpha"}}}),
            json!({"doc_len":{"match":{"value":"5"}}}),
            json!({"bad":{"match":{"value":5}}}),
            json!({"category":{"match":{"value":"a;b"}}}),
            json!({"not":[],"or":[]}),
        ] {
            assert!(Expr::compile(&v, &schema()).is_err(), "{v}");
        }
    }
    #[test]
    fn tag_escaping_and_numeric_bounds_are_explicit() {
        let e = Expr::compile(
            &json!({"category":{"match":{"value":"A|B {C}"}}}),
            &schema(),
        )
        .unwrap();
        assert_eq!(e.redis(), "(@category:{A\\|B\\ \\{C\\}})");
        assert_eq!(e.opensearch(), json!({"terms":{"category":["A|B {C}"]}}));
        let e = Expr::compile(&json!({"doc_len":{"range":{"gt":3,"lte":5}}}), &schema()).unwrap();
        assert!(!e
            .matches(json!({"doc_len":3}).as_object().unwrap())
            .unwrap());
        assert!(e
            .matches(json!({"doc_len":5}).as_object().unwrap())
            .unwrap());
    }
    #[test]
    fn suite_is_deterministic_and_positive_examples_come_from_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let seed = json!({"title":"apples orchard","segment":"green trees","category":"cat00","doc_len":100});
        let other =
            json!({"title":"apple","segment":"green fruit","category":"cat01","doc_len":200});
        std::fs::write(
            dir.path().join("payloads.jsonl"),
            format!("{seed}\n{other}\n"),
        )
        .unwrap();
        std::fs::write(dir.path().join("filter_queries.jsonl"),json!({"query_id":"scalar","shape":"tag_eq","conditions":{"and":[{"category":{"match":{"value":"cat00"}}}]},"expected_count":1,"corpus_size":2}).to_string()).unwrap();
        std::fs::write(
            dir.path().join("PREPARED.json"),
            json!({"dataset":"fixture","passages":2}).to_string(),
        )
        .unwrap();
        let spec = Spec {
            suite: Some("msmarco-v1".into()),
            queries: None,
            family: None,
        };
        let q = spec.load(dir.path(), &schema()).unwrap();
        assert_eq!(q, spec.load(dir.path(), &schema()).unwrap());
        assert_eq!(q.len(), 11);
        for query in &q[..5] {
            assert!(
                Expr::compile(&query.conditions, &schema())
                    .unwrap()
                    .matches(seed.as_object().unwrap())
                    .unwrap(),
                "{}",
                query.name
            );
        }
        assert_eq!(q[10].expected_count, Some(1));
        let base = Expr::compile(&q[0].conditions, &schema()).unwrap();
        let excluded = Expr::compile(&q[4].conditions, &schema()).unwrap();
        assert!(base.matches(other.as_object().unwrap()).unwrap());
        assert!(!excluded.matches(other.as_object().unwrap()).unwrap());
        // The real witness must be co-occurring, not merely a corpus term.
        let isolated =
            json!({"title":"apple","segment":"unrelated fruit","category":"cat01","doc_len":200});
        std::fs::write(
            dir.path().join("payloads.jsonl"),
            format!("{seed}\n{isolated}\n"),
        )
        .unwrap();
        assert!(spec
            .load(dir.path(), &schema())
            .unwrap_err()
            .contains("no nontrivial"));
        std::fs::write(
            dir.path().join("payloads.jsonl"),
            format!("{seed}\n{other}\n"),
        )
        .unwrap();
        for analysis in ["parity", "english"] {
            let full=crate::fulltext::parse_spec(Some(&json!({"analysis":analysis,"fields":{"title":1.0,"segment":1.0},"compound":{"suite":"msmarco-v1"}}))).unwrap();
            let set = crate::fulltext::load_query_set(dir.path(), &full, Some(&schema())).unwrap();
            let q = set.queries[4].compound.as_ref().unwrap();
            assert_eq!(q.minimum_count, if analysis == "english" { 0 } else { 1 });
            assert_eq!(
                q.conditions["and"][1]["not"]["title"]["match"]["text"],
                "apple"
            );
            assert!(set.preparation_sha256.is_some());
            // Explicit assertions are authored truth and must never be relaxed.
            let explicit=crate::fulltext::parse_spec(Some(&json!({"analysis":analysis,"fields":{"title":1.0,"segment":1.0},"compound":{"queries":[queries_with_minimum(q)]}}))).unwrap();
            assert_eq!(
                crate::fulltext::load_query_set(dir.path(), &explicit, Some(&schema()))
                    .unwrap()
                    .queries[0]
                    .compound
                    .as_ref()
                    .unwrap()
                    .minimum_count,
                1
            );
        }
        let mut bad = spec;
        bad.family = Some("unknown".into());
        assert!(bad.load(dir.path(), &schema()).is_err());
    }
    #[test]
    fn exact_count_guard_detects_inverted_or_omitted_clause() {
        let q = Query {
            name: "guard".into(),
            family: "scalar".into(),
            conditions: json!({}),
            expected_count: Some(2),
            minimum_count: 0,
        };
        assert!(check_count(&q, 2).is_ok());
        assert!(check_count(&q, 3).is_err());
        assert!(check_count(&q, 0).is_err());
        let mut text = q;
        text.expected_count = None;
        text.minimum_count = 1;
        assert!(check_count(&text, 0).is_err());
        assert!(check_count(&text, 1).is_ok());
    }
}
