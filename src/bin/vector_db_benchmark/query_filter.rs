//! The one place a query's `conditions` JSON becomes the filter that is actually
//! sent to an engine (issue #219).
//!
//! # The bug this module exists to make unrepresentable
//!
//! Every engine used to write some spelling of
//!
//! ```ignore
//! let parsed: Vec<Option<F>> = conditions.iter().map(|c| c.as_ref().and_then(parse)).collect();
//! ```
//!
//! and then, per query, `if let Some(f) = &parsed[i] { request.filter(f) }` —
//! with no `else`. `and_then` collapses two states that are not the same thing:
//!
//! * the query declared **no** filter, so its ground truth is unfiltered too and
//!   omitting the filter is correct; and
//! * the query declared a filter that the builder **dropped** — an operator the
//!   engine does not implement, an unparseable bound, a value of the wrong JSON
//!   type — so omitting the filter runs a **fully unfiltered search whose recall
//!   is then scored against filtered ground truth**.
//!
//! The second case produces a plausible number, writes it to the results JSON
//! under a config name that claims a filter was applied, and prints nothing.
//!
//! # The fix
//!
//! [`QueryConditions`] is the only thing [`crate::dataset::Dataset::read_queries`]
//! hands back, and it has **no accessor for the raw per-query JSON**. The only
//! way to get from it to something you can attach to a request is one of the
//! `resolve_*` methods here, and each of them makes the drop an `Err`. A new
//! engine cannot reproduce the old shape by copying a neighbour, because the
//! type it would need to copy from no longer exists.
//!
//! The three states are kept apart at the type level, in the return type
//! `Result<Vec<QueryFilter<T>>, String>`:
//!
//! | state | value |
//! |---|---|
//! | no filter declared | `Ok(..)`, entry is [`QueryFilter::unfiltered`] |
//! | filter declared and expressible | `Ok(..)`, entry is [`QueryFilter::filtered`] |
//! | filter declared and dropped | `Err(..)` — the run stops |
//!
//! [`QueryFilter`]'s inner `Option` is private and there is no public
//! constructor, so an engine cannot mint "unfiltered" out of a parse that
//! dropped something; only the `resolve_*` functions below can, and they only do
//! it for genuinely absent conditions.
//!
//! # What counts as "no filter declared"
//!
//! A per-query conditions value is treated as absent when it is `None`, JSON
//! `null`, or a literally empty object `{}`.
//!
//! The `null` case is load-bearing, not defensive. `readers/compound_reader.rs`
//! builds the vector with `row.get("conditions").cloned()`, so a `tests.jsonl`
//! row that spells `"conditions": null` — which is **every one of the 10 000
//! rows** of the shipped `random_keywords_1m_vocab_10_no_filters` dataset, and
//! of the other `*_no_filters` tarballs — arrives here as `Some(Value::Null)`,
//! not `None`. A guard keyed on `is_some()` would fail every query of a dataset
//! whose queries are intentionally unfiltered.
//!
//! Anything else — including `{"and": []}` or a tree whose every leaf dropped —
//! is a declared filter, and a builder returning nothing for it is an error.

use serde_json::Value;

/// A query's filter, after resolution.
///
/// Deliberately not a bare `Option<T>`: the inner field is private and there is
/// no public constructor, so the only route to the "no filter" state is
/// [`QueryConditions::resolve_all`] & friends, which reach it only for a
/// genuinely absent conditions value. Downstream `if let Some(f) = qf.as_ref()`
/// is therefore sound — the `None` arm can no longer be a dropped filter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryFilter<T>(Option<T>);

impl<T> QueryFilter<T> {
    /// The query declared no filter. Private on purpose — see the type docs.
    fn unfiltered() -> Self {
        Self(None)
    }

    /// The query declared a filter and the engine can express it.
    fn filtered(filter: T) -> Self {
        Self(Some(filter))
    }

    /// The filter to attach to the request, or `None` when — and only when —
    /// the query genuinely declared no filter.
    pub fn as_ref(&self) -> Option<&T> {
        self.0.as_ref()
    }

    /// Consume into the underlying option, with the same guarantee as
    /// [`Self::as_ref`].
    ///
    /// Test-only: production code keeps the `QueryFilter` so the two states stay
    /// distinguishable all the way to the request builder.
    #[cfg(test)]
    pub fn into_inner(self) -> Option<T> {
        self.0
    }

    /// Whether this query carries a filter.
    pub fn is_filtered(&self) -> bool {
        self.0.is_some()
    }
}

impl<T: std::ops::Deref> QueryFilter<T> {
    /// `Option::as_deref` for the wrapped filter (`String` → `&str`), with the
    /// same guarantee as [`QueryFilter::as_ref`].
    pub fn as_deref(&self) -> Option<&T::Target> {
        self.0.as_deref()
    }
}

/// Per-query filter conditions as read from the dataset.
///
/// Opaque by design: there is no way to read the raw JSON out of it, only to
/// resolve it through one of the guarded constructors below.
#[derive(Debug, Clone, Default)]
pub struct QueryConditions {
    per_query: Vec<Option<Value>>,
}

impl QueryConditions {
    /// Wrap the raw per-query conditions read from a dataset file.
    ///
    /// Crate-visible so only `dataset.rs` can build one; engines receive it and
    /// must resolve it.
    pub(crate) fn new(per_query: Vec<Option<Value>>) -> Self {
        Self { per_query }
    }

    /// Resolve every query's conditions with a builder that returns `Option<T>`.
    ///
    /// `Err` when a query declared a filter the builder produced nothing for:
    /// running it would search unfiltered against filtered ground truth.
    pub fn resolve_all<T>(
        &self,
        engine: &str,
        parse: impl Fn(&Value) -> Option<T>,
    ) -> Result<Vec<QueryFilter<T>>, String> {
        self.try_resolve_all(engine, |c| Ok(parse(c)))
    }

    /// Same as [`Self::resolve_all`] for a builder that can itself report why a
    /// condition is unexpressible. Its `Ok(None)` is treated exactly like
    /// `resolve_all`'s `None`: a drop, and therefore an error, unless the
    /// conditions value was absent to begin with.
    pub fn try_resolve_all<T>(
        &self,
        engine: &str,
        parse: impl Fn(&Value) -> Result<Option<T>, String>,
    ) -> Result<Vec<QueryFilter<T>>, String> {
        self.per_query
            .iter()
            .enumerate()
            .map(|(idx, cond)| resolve(engine, idx, cond.as_ref(), &parse))
            .collect()
    }

    /// Resolve with a builder that has **no** "produced nothing" state at all —
    /// it returns the filter or says why it cannot (`vertex.rs`, `kividb.rs`).
    ///
    /// `unfiltered` supplies the value used for queries that declared no filter
    /// (an empty restriction set, a `*` prefilter, …).
    pub fn resolve_all_total<T>(
        &self,
        unfiltered: impl Fn() -> T,
        parse: impl Fn(&Value) -> Result<T, String>,
    ) -> Result<Vec<T>, String> {
        self.per_query
            .iter()
            .map(|cond| match declared(cond.as_ref()) {
                Some(cond) => parse(cond),
                None => Ok(unfiltered()),
            })
            .collect()
    }
}

/// Resolve one query's conditions. The rule, in one place:
///
/// * conditions absent / `null` / `{}` → the query declared no filter;
/// * builder produced a filter → use it;
/// * builder produced nothing for a declared filter → `Err`.
pub fn resolve<T>(
    engine: &str,
    idx: usize,
    conditions: Option<&Value>,
    parse: impl FnOnce(&Value) -> Result<Option<T>, String>,
) -> Result<QueryFilter<T>, String> {
    let Some(cond) = declared(conditions) else {
        return Ok(QueryFilter::unfiltered());
    };
    match parse(cond)? {
        Some(filter) => Ok(QueryFilter::filtered(filter)),
        None => Err(dropped(
            engine,
            idx,
            cond,
            "it produced no condition at all",
        )),
    }
}

/// The declared-filter test: `None` for an absent, null or empty-object value.
///
/// See the module docs on why `null` must be folded in with absent.
fn declared(cond: Option<&Value>) -> Option<&Value> {
    let cond = cond?;
    if cond.is_null() {
        return None;
    }
    if cond.as_object().is_some_and(|o| o.is_empty()) {
        return None;
    }
    Some(cond)
}

/// The error every dropped filter reports, in one voice across engines.
pub fn dropped(engine: &str, idx: usize, conditions: &Value, why: &str) -> String {
    format!(
        "{engine} cannot express the filter on query {idx}: {why}. Conditions: {conditions}. \
         Running the query without it would search UNFILTERED while its ground truth is \
         filtered, so the recall reported would be for a filter that was never applied \
         (issue #219). Fix the engine's filter builder or drop this dataset from the run."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A builder that expresses `{"and":[{"a":{"match":{"value":..}}}]}` and
    /// nothing else — a stand-in for a real engine's `parse_conditions`.
    fn toy(conditions: &Value) -> Option<String> {
        let leaf = conditions.get("and")?.as_array()?.first()?;
        let (field, spec) = leaf.as_object()?.iter().next()?;
        let value = spec.get("match")?.get("value")?.as_str()?;
        Some(format!("@{field}:{{{value}}}"))
    }

    fn conds(rows: Vec<Option<Value>>) -> QueryConditions {
        QueryConditions::new(rows)
    }

    #[test]
    fn absent_conditions_are_unfiltered_not_an_error() {
        let resolved = conds(vec![None, None]).resolve_all("Toy", toy).unwrap();
        assert!(resolved.iter().all(|f| !f.is_filtered()));
    }

    /// The shape of every row of the shipped `*_no_filters` tarballs: the
    /// `conditions` key IS present, spelled `null`. `compound_reader` delivers
    /// that as `Some(Value::Null)`, so an `is_some()`-keyed guard would fail all
    /// 10 000 queries of a legitimately unfiltered dataset.
    #[test]
    fn explicit_json_null_is_unfiltered_not_an_error() {
        let resolved = conds(vec![Some(Value::Null); 3])
            .resolve_all("Toy", toy)
            .unwrap();
        assert_eq!(resolved.len(), 3);
        assert!(resolved.iter().all(|f| !f.is_filtered()));
    }

    #[test]
    fn empty_object_is_unfiltered_not_an_error() {
        let resolved = conds(vec![Some(json!({}))])
            .resolve_all("Toy", toy)
            .unwrap();
        assert!(!resolved[0].is_filtered());
    }

    #[test]
    fn an_expressible_filter_resolves() {
        let resolved = conds(vec![Some(json!({"and":[{"a":{"match":{"value":"red"}}}]}))])
            .resolve_all("Toy", toy)
            .unwrap();
        assert_eq!(resolved[0].as_ref().map(String::as_str), Some("@a:{red}"));
    }

    /// The bug, as a test: a declared filter the builder cannot express must
    /// stop the run rather than become an unfiltered query.
    #[test]
    fn a_declared_filter_that_parses_to_nothing_is_an_error() {
        for shape in [
            json!({"and":[{"a":{"geo_radius":{"lat":1.0,"lon":2.0,"radius":3.0}}}]}),
            json!({"and":[]}),
            json!({"or":[{"and":[]}]}),
            json!({"a":{"match":{"value":"red"}}}),
            json!({"and":[{"a":{"match":{"value":7}}}]}),
        ] {
            let err = conds(vec![Some(shape.clone())])
                .resolve_all("Toy", toy)
                .unwrap_err();
            assert!(err.contains("UNFILTERED"), "{shape}: {err}");
            assert!(err.contains("#219"), "{shape}: {err}");
        }
    }

    #[test]
    fn the_failing_query_index_is_named() {
        let err = conds(vec![
            Some(json!({"and":[{"a":{"match":{"value":"red"}}}]})),
            Some(Value::Null),
            Some(json!({"and":[]})),
        ])
        .resolve_all("Toy", toy)
        .unwrap_err();
        assert!(err.contains("query 2"), "{err}");
    }

    #[test]
    fn a_builders_own_error_propagates_unchanged() {
        let err = conds(vec![Some(json!({"and":[]}))])
            .try_resolve_all::<()>("Toy", |_| Err("nope".to_string()))
            .unwrap_err();
        assert_eq!(err, "nope");
    }

    #[test]
    fn try_resolve_all_ok_none_on_a_declared_filter_is_still_an_error() {
        let err = conds(vec![Some(json!({"and":[{"a":{"match":{"value":"x"}}}]}))])
            .try_resolve_all::<()>("Toy", |_| Ok(None))
            .unwrap_err();
        assert!(err.contains("UNFILTERED"), "{err}");
    }

    #[test]
    fn resolve_all_total_uses_the_unfiltered_value_only_for_absent_conditions() {
        let out = conds(vec![Some(Value::Null), Some(json!({"a":1}))])
            .resolve_all_total(|| "*".to_string(), |c| Ok(c.to_string()))
            .unwrap();
        assert_eq!(out, vec!["*".to_string(), "{\"a\":1}".to_string()]);
    }

    /// The four spellings of "absent" all resolve; only the declared one is a
    /// filter. One query per spelling, in one call, so ordering is checked too.
    #[test]
    fn only_a_declared_filter_becomes_a_filter() {
        let resolved = conds(vec![
            None,
            Some(Value::Null),
            Some(json!({})),
            Some(json!({"and":[{"a":{"match":{"value":"x"}}}]})),
        ])
        .resolve_all("Toy", toy)
        .unwrap();
        assert_eq!(
            resolved.iter().map(|f| f.is_filtered()).collect::<Vec<_>>(),
            vec![false, false, false, true]
        );
    }
}
