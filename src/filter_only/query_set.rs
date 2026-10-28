//! The filter-only query set, the index layout it implies, and the per-query
//! count gate (#340, query side).
//!
//! # Where the queries come from
//!
//! A compound dataset directory may ship a prepared query set,
//! [`FILTER_QUERIES_FILE`]: one row per query with its `query_id`, `shape`,
//! `conditions` and the EXACT `expected_count` computed at prepare time
//! (`msmarco::derived::DerivedCorpus::filter_query_rows`). When it does, the
//! filter-only workload runs exactly those queries. Otherwise it falls back to
//! the predicates in `tests.jsonl` (the synthetic `*-filters` datasets), which
//! carry no shape and no expected count.
//!
//! An MS MARCO directory that declares the derived fields but predates them
//! (no file, payloads without the fields) never reaches this module: the
//! loader refuses it first (`msmarco::derived::check_prepared_dir`, called by
//! `Dataset::get_path`), naming `prepare-msmarco --add-derived-fields`.
//!
//! # What gets indexed
//!
//! [`index_schema`]. Without a prepared set (H&M, the synthetic datasets) the
//! layout is exactly the declared schema, as before #340: every declared field
//! is indexed and every declared scalar field validated. With a prepared set,
//! the index holds only the `keyword`/`int`/`float` fields its predicates
//! reference (TAG/NUMERIC on Redis, keyword/long/double on OpenSearch, which
//! also maps with `dynamic: false`). Every other field — MS MARCO's `text`
//! fields, `docid`, and `url`, whose values legitimately contain `;` — stays
//! stored in the document, unindexed, so the shared-domain metadata checks (no
//! `;` or edge whitespace in a tag) apply to what is indexed and queried while
//! every guard on an indexed field stays exactly as strict.
//!
//! # The per-query count gate
//!
//! [`FilterQuerySet::check_counts`]: every answered query's exact total must
//! equal its prepared `expected_count`. The sum across queries is not enough —
//! two compensating errors keep the sum and are two wrong answers.

use crate::msmarco::derived::Predicate;
use crate::query_filter::QueryConditions;
use serde_json::{Map, Value};
use std::collections::{BTreeSet, HashSet};
use std::io::BufRead;
use std::path::Path;

pub use crate::msmarco::derived::FILTER_QUERIES_FILE;

/// Field kinds a filter-only predicate may reference, and therefore the only
/// kinds the filter-only workload indexes.
pub const INDEXED_KINDS: [&str; 3] = ["keyword", "int", "float"];

/// One prepared query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedQuery {
    /// e.g. `msmarco-filter-003`.
    pub name: String,
    /// e.g. `numeric_point`; becomes the per-query record's `shape`.
    pub shape: String,
    pub expected_count: u64,
}

/// The queries a filter-only run executes, by query index.
#[derive(Debug, Clone)]
pub struct FilterQuerySet {
    conditions: QueryConditions,
    /// `Some` for a prepared set: one entry per query, same order as
    /// `conditions`.
    prepared: Option<Vec<PreparedQuery>>,
    /// Fields referenced by a prepared set's predicates.
    referenced: BTreeSet<String>,
}

impl FilterQuerySet {
    /// The legacy source: predicates from `tests.jsonl`, no shapes, no
    /// expected counts.
    pub fn from_conditions(conditions: QueryConditions) -> Self {
        Self {
            conditions,
            prepared: None,
            referenced: BTreeSet::new(),
        }
    }

    /// Parse a prepared set. Every row must hold exactly `query_id`, `shape`,
    /// `conditions`, `expected_count` and `corpus_size`; `corpus_size` must
    /// equal the corpus being benchmarked when that is known (a query set for
    /// another variant has other expected counts). Conditions must be in the
    /// exact-match grammar `msmarco::derived::Predicate` evaluates — the
    /// grammar the expected counts were computed under.
    pub fn parse(
        reader: impl BufRead,
        dataset: &str,
        corpus_size: Option<u64>,
    ) -> Result<Self, String> {
        let err = |line: usize, msg: String| format!("{FILTER_QUERIES_FILE} line {line}: {msg}");
        let mut lines: Vec<String> = Vec::new();
        for line in reader.lines() {
            lines.push(line.map_err(|e| format!("{FILTER_QUERIES_FILE}: {e}"))?);
        }
        // One trailing newline is a file ending, not a row.
        if lines.last().is_some_and(|l| l.trim().is_empty()) {
            lines.pop();
        }
        if lines.is_empty() {
            return Err(format!("{FILTER_QUERIES_FILE} holds no queries"));
        }
        let mut conditions = Vec::with_capacity(lines.len());
        let mut prepared = Vec::with_capacity(lines.len());
        let mut referenced = BTreeSet::new();
        let mut names = HashSet::new();
        for (i, line) in lines.iter().enumerate() {
            let n = i + 1;
            let row: Map<String, Value> = serde_json::from_str(line)
                .map_err(|e| err(n, format!("not a JSON object: {e}")))?;
            const KEYS: [&str; 5] = [
                "query_id",
                "shape",
                "conditions",
                "expected_count",
                "corpus_size",
            ];
            if let Some(k) = row.keys().find(|k| !KEYS.contains(&k.as_str())) {
                return Err(err(n, format!("unexpected key {k:?}")));
            }
            let text = |k: &str| {
                row.get(k)
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .ok_or_else(|| err(n, format!("{k} must be a nonempty string")))
            };
            let count = |k: &str| {
                row.get(k)
                    .and_then(Value::as_u64)
                    .ok_or_else(|| err(n, format!("{k} must be a non-negative integer")))
            };
            let name = text("query_id")?;
            let shape = text("shape")?;
            let expected_count = count("expected_count")?;
            let size = count("corpus_size")?;
            if let Some(corpus) = corpus_size {
                if size != corpus {
                    return Err(err(
                        n,
                        format!(
                            "query {name} was prepared for a {size}-row corpus but this corpus \
                             holds {corpus} rows; its expected counts are for another corpus"
                        ),
                    ));
                }
            }
            if expected_count > size {
                return Err(err(
                    n,
                    format!("expected_count {expected_count} exceeds corpus_size {size}"),
                ));
            }
            if !names.insert(name.clone()) {
                return Err(err(n, format!("duplicate query_id {name:?}")));
            }
            let condition = row
                .get("conditions")
                .ok_or_else(|| err(n, "missing conditions".into()))?;
            let predicate = Predicate::compile(condition).map_err(|e| err(n, e))?;
            collect_fields(&predicate, &mut referenced);
            conditions.push(Some(condition.clone()));
            prepared.push(PreparedQuery {
                name,
                shape,
                expected_count,
            });
        }
        Ok(Self {
            conditions: QueryConditions::new(conditions).from_dataset(dataset),
            prepared: Some(prepared),
            referenced,
        })
    }

    /// The prepared set in `dir`, if the dataset has one (`Ok(None)` when
    /// `dir` holds no [`FILTER_QUERIES_FILE`]).
    pub fn read_prepared(
        dir: &Path,
        dataset: &str,
        corpus_size: Option<u64>,
    ) -> Result<Option<Self>, String> {
        let path = dir.join(FILTER_QUERIES_FILE);
        if !path.is_file() {
            return Ok(None);
        }
        let file = std::fs::File::open(&path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        Self::parse(std::io::BufReader::new(file), dataset, corpus_size)
            .map(Some)
            .map_err(|e| format!("{}: {e}", dir.display()))
    }

    /// The prepared set if there is one, else the legacy `tests.jsonl`
    /// predicates (read lazily, since that read loads the query vectors too).
    pub fn load(
        dir: &Path,
        dataset: &str,
        corpus_size: Option<u64>,
        legacy: impl FnOnce() -> Result<QueryConditions, String>,
    ) -> Result<Self, String> {
        match Self::read_prepared(dir, dataset, corpus_size)? {
            Some(set) => Ok(set),
            None => Ok(Self::from_conditions(legacy()?)),
        }
    }

    /// [`QueryConditions::try_resolve_all`] over this set's conditions — the
    /// ONLY way out for them: there is no accessor, so an engine (in the
    /// binary) cannot read unresolved conditions from a set it obtained through
    /// `Dataset::filter_query_file`, and the `filter_guard` read/resolve census
    /// sees every read (#219).
    pub fn try_resolve_all<T>(
        &self,
        engine: &str,
        parse: impl Fn(&Value) -> Result<Option<T>, String>,
    ) -> Result<Vec<crate::query_filter::QueryFilter<T>>, String> {
        self.conditions.try_resolve_all(engine, parse)
    }

    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Whether the set carries shapes and expected counts.
    pub fn is_prepared(&self) -> bool {
        self.prepared.is_some()
    }

    /// The query's shape, for the per-query record. `None` for a legacy set.
    pub fn shape(&self, query: usize) -> Option<String> {
        self.prepared
            .as_ref()
            .and_then(|p| p.get(query))
            .map(|q| q.shape.clone())
    }

    pub fn prepared(&self) -> Option<&[PreparedQuery]> {
        self.prepared.as_deref()
    }

    /// The per-query count gate. `observed` holds `(query index, exact total)`
    /// for every successfully answered query (a query cycled by `--queries N`
    /// appears once per answer, and each answer is checked); `answered` is the
    /// number of successful answers the run counted for its statistics, so an
    /// engine that answers without recording totals cannot pass vacuously.
    ///
    /// `Ok(None)` for a legacy set (nothing to check against); `Ok(Some(n))`
    /// with the number of answers checked; `Err` naming the disagreeing
    /// queries otherwise.
    pub fn check_counts(
        &self,
        observed: &[(usize, u64)],
        answered: usize,
    ) -> Result<Option<u64>, String> {
        let Some(prepared) = &self.prepared else {
            return Ok(None);
        };
        if observed.len() != answered {
            return Err(format!(
                "per-query count gate: {} totals recorded for {answered} answered queries; \
                 every answer must be checked",
                observed.len()
            ));
        }
        let mut wrong = Vec::new();
        let mut wrong_answers = 0usize;
        let mut seen = HashSet::new();
        for &(query, total) in observed {
            let q = prepared.get(query).ok_or_else(|| {
                format!(
                    "per-query count gate: answer for query index {query}, but the prepared set \
                     holds {} queries",
                    prepared.len()
                )
            })?;
            if total != q.expected_count {
                wrong_answers += 1;
                if seen.insert(query) && wrong.len() < 5 {
                    wrong.push(format!(
                        "{} ({}): engine total {total}, expected {}",
                        q.name, q.shape, q.expected_count
                    ));
                }
            }
        }
        if wrong_answers > 0 {
            return Err(format!(
                "per-query count gate: {wrong_answers} of {} answers disagree with the prepared \
                 expected_count: {}",
                observed.len(),
                wrong.join("; ")
            ));
        }
        Ok(Some(observed.len() as u64))
    }
}

fn collect_fields(p: &Predicate, out: &mut BTreeSet<String>) {
    match p {
        Predicate::And(c) | Predicate::Or(c) => c.iter().for_each(|p| collect_fields(p, out)),
        Predicate::Leaf { field, .. } => {
            out.insert(field.clone());
        }
    }
}

/// The schema the filter-only workload indexes, and validates metadata and
/// predicates against (see the module docs). Without a prepared set: the
/// declared schema, unchanged. With one: the declared `keyword`/`int`/`float`
/// fields its predicates reference; other fields are stored, not indexed, and
/// no predicate may reference them.
pub fn index_schema(schema: Option<&Value>, prepared: Option<&FilterQuerySet>) -> Option<Value> {
    let Some(set) = prepared.filter(|s| s.is_prepared()) else {
        return schema.cloned();
    };
    let fields = schema?.as_object()?;
    let keep = |name: &str| set.referenced.contains(name);
    Some(Value::Object(
        fields
            .iter()
            .filter(|(name, kind)| {
                kind.as_str().is_some_and(|k| INDEXED_KINDS.contains(&k)) && keep(name)
            })
            .map(|(n, k)| (n.clone(), k.clone()))
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(id: &str, shape: &str, cond: Value, expected: u64, size: u64) -> String {
        json!({"query_id": id, "shape": shape, "conditions": cond,
               "expected_count": expected, "corpus_size": size})
        .to_string()
    }

    fn tag(v: &str) -> Value {
        json!({"and": [{"category": {"match": {"value": v}}}]})
    }

    fn two_queries() -> String {
        [
            row("q0", "tag_eq", tag("cat00"), 7, 100),
            row(
                "q1",
                "range_or_range",
                json!({"or": [{"tier": {"range": {"lte": 1}}}, {"doc_len": {"range": {"gte": 9}}}]}),
                3,
                100,
            ),
        ]
        .join("\n")
            + "\n"
    }

    fn parse(text: &str, size: Option<u64>) -> Result<FilterQuerySet, String> {
        FilterQuerySet::parse(text.as_bytes(), "ds", size)
    }

    #[test]
    fn parses_shapes_counts_and_referenced_fields() {
        let s = parse(&two_queries(), Some(100)).unwrap();
        assert_eq!(s.len(), 2);
        assert!(s.is_prepared());
        assert_eq!(s.conditions.declared_count(), 2);
        assert_eq!(s.shape(0).as_deref(), Some("tag_eq"));
        assert_eq!(s.shape(1).as_deref(), Some("range_or_range"));
        assert_eq!(s.shape(2), None);
        assert_eq!(s.prepared().unwrap()[1].expected_count, 3);
        assert_eq!(
            s.referenced.iter().map(String::as_str).collect::<Vec<_>>(),
            ["category", "doc_len", "tier"]
        );
        // Unknown corpus size: accepted (the caller could not measure it).
        parse(&two_queries(), None).unwrap();
    }

    #[test]
    fn malformed_or_foreign_rows_are_refused() {
        let ok = |c: Value| row("q", "s", c, 1, 100);
        for (text, needle) in [
            (String::new(), "no queries"),
            ("\n".into(), "no queries"),
            (two_queries() + "\n" + &ok(tag("cat01")), "not a JSON object"),
            (two_queries().replace("100", "99"), "99-row corpus"),
            (row("q0", "s", tag("a"), 101, 100), "exceeds corpus_size"),
            (format!("{}\n{}", ok(tag("a")), ok(tag("b"))), "duplicate"),
            (ok(json!({"category": {"match": {"value": "a"}}})), "top level"),
            (ok(json!({"and": [{"t": {"match": {"text": "a"}}}]})), "full-text"),
            (
                json!({"query_id":"q","shape":"s","conditions":tag("a"),"expected_count":1,"corpus_size":100,"x":1}).to_string(),
                "unexpected key",
            ),
            (
                json!({"query_id":"","shape":"s","conditions":tag("a"),"expected_count":1,"corpus_size":100}).to_string(),
                "query_id",
            ),
            (
                json!({"query_id":"q","shape":"s","conditions":tag("a"),"expected_count":-1,"corpus_size":100}).to_string(),
                "expected_count",
            ),
            (
                json!({"query_id":"q","shape":"s","expected_count":1,"corpus_size":100}).to_string(),
                "missing conditions",
            ),
        ] {
            let e = parse(&text, Some(100)).unwrap_err();
            assert!(e.contains(needle), "{needle:?} not in {e:?} for {text:?}");
        }
    }

    #[test]
    fn the_gate_checks_every_answer_not_the_sum() {
        let s = parse(&two_queries(), Some(100)).unwrap();
        assert_eq!(
            s.check_counts(&[(0, 7), (1, 3), (0, 7)], 3).unwrap(),
            Some(3)
        );
        assert_eq!(s.check_counts(&[], 0).unwrap(), Some(0));
        // Answers whose totals were never recorded cannot pass vacuously.
        let e = s.check_counts(&[], 5).unwrap_err();
        assert!(e.contains("0 totals recorded for 5"), "{e}");
        assert!(s.check_counts(&[(0, 7)], 2).is_err());
        // Sum preserved (7 + 3 == 8 + 2), both answers wrong.
        let e = s.check_counts(&[(0, 8), (1, 2)], 2).unwrap_err();
        assert!(e.contains("2 of 2 answers"), "{e}");
        assert!(e.contains("q0 (tag_eq): engine total 8, expected 7"), "{e}");
        assert!(e.contains("q1 (range_or_range)"), "{e}");
        // One wrong answer among right ones still fails.
        assert!(s.check_counts(&[(0, 7), (0, 6)], 2).is_err());
        // An answer for a query the set does not hold is a harness bug.
        assert!(s
            .check_counts(&[(2, 0)], 1)
            .unwrap_err()
            .contains("index 2"));
        // Legacy sets have nothing to check.
        let legacy = FilterQuerySet::from_conditions(QueryConditions::new(vec![Some(tag("a"))]));
        assert_eq!(legacy.check_counts(&[(0, 99)], 7).unwrap(), None);
        assert_eq!(legacy.shape(0), None);
        assert!(!legacy.is_prepared());
    }

    fn msmarco_schema() -> Value {
        json!({"docid":"keyword","url":"keyword","title":"text","start_char":"int",
               "category":"keyword","doc_len":"int","tier":"int","rand_key":"int",
               "loc":"geo","when":"datetime","price":"float"})
    }

    #[test]
    fn index_schema_drops_unqueryable_kinds_and_narrows_to_referenced_fields() {
        let schema = msmarco_schema();
        // No prepared set: the declared schema unchanged (text/geo/datetime
        // included), exactly the pre-#340 layout.
        assert_eq!(index_schema(Some(&schema), None).unwrap(), schema);
        let legacy = FilterQuerySet::from_conditions(QueryConditions::new(vec![]));
        assert_eq!(index_schema(Some(&schema), Some(&legacy)).unwrap(), schema);
        // A prepared set referencing every field still never indexes the
        // unqueryable kinds.
        let mut every = String::new();
        for (i, f) in schema.as_object().unwrap().keys().enumerate() {
            every += &row(
                &format!("q{i}"),
                "s",
                json!({"and": [{f: {"match": {"value": "x"}}}]}),
                0,
                100,
            );
            every += "\n";
        }
        let all = index_schema(Some(&schema), Some(&parse(&every, None).unwrap())).unwrap();
        let keys = |v: &Value| {
            let mut k = v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
            k.sort();
            k
        };
        assert_eq!(
            keys(&all),
            [
                "category",
                "doc_len",
                "docid",
                "price",
                "rand_key",
                "start_char",
                "tier",
                "url"
            ]
        );
        // Prepared set: only what its predicates reference.
        let s = parse(&two_queries(), None).unwrap();
        assert_eq!(
            index_schema(Some(&schema), Some(&s)).unwrap(),
            json!({"category":"keyword","doc_len":"int","tier":"int"})
        );
        // A referenced field the schema does not declare as scalar is NOT
        // added: predicate validation then refuses the query.
        let text_only = json!({"category":"text","tier":"int","doc_len":"int"});
        assert_eq!(
            index_schema(Some(&text_only), Some(&s)).unwrap(),
            json!({"tier":"int","doc_len":"int"})
        );
        assert_eq!(index_schema(None, Some(&s)), None);
    }

    #[test]
    fn the_prepared_file_wins_and_its_corpus_size_is_checked() {
        let dir = tempfile::tempdir().unwrap();
        // No file: tests.jsonl's predicates, read lazily.
        assert!(FilterQuerySet::read_prepared(dir.path(), "syn", None)
            .unwrap()
            .is_none());
        let s = FilterQuerySet::load(dir.path(), "syn", None, || {
            Ok(QueryConditions::new(vec![Some(tag("a")), None]))
        })
        .unwrap();
        assert!(!s.is_prepared());
        assert_eq!(s.len(), 2);
        // With the file present, it is used and tests.jsonl is never read.
        std::fs::write(dir.path().join(FILTER_QUERIES_FILE), two_queries()).unwrap();
        let s = FilterQuerySet::load(dir.path(), "msm", Some(100), || {
            panic!("a prepared set must not read tests.jsonl")
        })
        .unwrap();
        assert!(s.is_prepared());
        let e = FilterQuerySet::read_prepared(dir.path(), "msm", Some(5)).unwrap_err();
        assert!(e.contains("5 rows"), "{e}");
    }
}
