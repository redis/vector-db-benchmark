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
//! [`QueryConditions`] is the only thing `Dataset::read_queries` hands back, and
//! it exposes **no field, iterator or getter for the raw per-query JSON** — the
//! only way through it is one of the `resolve_*` methods here, and each makes "a
//! declared filter produced nothing" an `Err`. (A *declared* condition can still
//! be echoed back out through the resolver itself; see escape 2 below. An absent
//! one cannot be, at all.) Restoring the old expression at a call site does not
//! compile: `Vec<Option<Value>>` is no longer what an engine receives, and
//! [`QueryFilter`]'s inner `Option` is a private field with no public
//! constructor, so an engine cannot mint "unfiltered" out of a parse that
//! dropped something.
//!
//! The three states are kept apart at the type level, in the return type
//! `Result<Vec<QueryFilter<T>>, String>`:
//!
//! | state | value |
//! |---|---|
//! | no filter declared | `Ok(..)`, entry is unfiltered |
//! | filter declared and expressible | `Ok(..)`, entry carries the filter |
//! | filter declared and dropped | `Err(..)` — the run stops |
//!
//! # What this does NOT guarantee — read before trusting it
//!
//! The guarantee is **"no silent `None`"**, not "no silent drop". Three escapes
//! are outside what the type system can see, and all three are live in-tree:
//!
//! 1. **A builder that keeps the leaves it understands and skips the rest.**
//!    12 of the 15 builders are written that way (`build_subfilters` in
//!    `redis.rs`, `vectorsets.rs`, ...), so a two-leaf `and` that loses one leaf
//!    still returns a real filter, passes every check here, and constrains less
//!    than the ground truth does. Nine shipped datasets emit 3 333 two-leaf
//!    `and` conditions each -- 29 997 queries -- so this is reachable, not
//!    theoretical. Measured on the branch that introduced this module:
//!    VectorSets scored **0.3000** on an `and(keyword, geo)` corpus where Redis,
//!    which expresses both leaves, scored **1.0000** (both on the corpus
//!    described in the successor issue; a reviewer measured **0.2250** for
//!    VectorSets on a differently-generated corpus of the same shape, with no
//!    paired Redis figure). Closing this is the per-builder `Option -> Result`
//!    refactor tracked in its own issue; this module only removes the
//!    "everything vanished" case. A builder that WIDENS without losing a leaf —
//!    `and` emitted as `or`, a two-sided range emitted with one bound — is in
//!    the same disclaimed set, and `engine/filter_guard.rs` cannot see it
//!    either.
//! 2. **A builder that returns a match-all `T`, or echoes its input.**
//!    `resolve_all("X", |c| Some(c.clone()))` type-checks and rebuilds the #219
//!    expression in one line -- shorter than the correct spelling. Nothing here
//!    can tell a match-all filter from a real one; `engine/filter_guard.rs`
//!    checks the builders themselves for that.
//! 3. **[`QueryConditions::unfiltered`] erases conditions.** It cannot fabricate
//!    a filter, but swapping it in for a real `QueryConditions` turns a filtered
//!    dataset into an unfiltered run with no error. It has to be public because
//!    the binary's `dataset.rs` builds one for the HDF5/JSONL formats. The net
//!    is a source-level test —
//!    `filter_guard::no_engine_erases_its_conditions_with_the_unfiltered_constructor`
//!    — not the type system. (A derived `Default` would have been a second,
//!    quieter spelling of the same thing, reachable through `..Default::
//!    default()`; it is not derived. `new` is a third, but it is `pub(crate)`
//!    of the library crate and so unreachable from any engine — the crate
//!    boundary closes that one, not the scan.)
//!
//! ## The ceiling on the source-level nets
//!
//! `engine/filter_guard.rs` adds two source scans (every `read_queries()` in an
//! engine file is paired with a resolver; no engine constructs a
//! `QueryConditions`). Neither can see a **helper hop**: a function outside
//! `engine/` that returns a blanked `QueryConditions`, called from a perfectly
//! ordinary-looking `.resolve_all("Redis", parse_conditions)?` site, rebuilds
//! #219 in compiling, fully silent form with both scans green. No per-file
//! scan can see that, and this one does not claim to.
//!
//! What forces the hop in the first place is [`QueryFilter`]'s private
//! constructors and `QueryConditions::new` being `pub(crate)` of the *library*
//! crate — an engine, which lives in the binary, cannot call it at all. The
//! types are the wall; the scans are a second line that catches the careless
//! case, not the determined one.
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
/// is therefore sound — the `None` arm can no longer be a *vanished* filter. It
/// can still be a filter the builder rendered as match-all; see the module docs
/// on what this type does not guarantee.
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
    /// Prefer keeping the `QueryFilter` as far as the request builder; unwrap
    /// only where the engine has an explicit value for "no filter" (KiviDB's
    /// match-all prefilter).
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
///
/// `#[must_use]` covers only the narrow case of the value being discarded as a
/// bare expression statement. It does **not** catch
/// `let (q, n, _conditions) = read_queries()?;` — that form is silent, passes
/// `clippy -D warnings`, and legitimately ships at `ground_truth.rs:68`, which
/// needs the neighbours and has no filter to apply. The check that actually
/// covers an engine ignoring its conditions is the source-level pairing test
/// `filter_guard::every_engine_read_of_query_conditions_is_paired_with_a_resolver`.
#[must_use = "a dataset's filter conditions must be resolved, not discarded: ignoring them \
              runs every query UNFILTERED against filtered ground truth (issue #219)"]
/// `Default` is deliberately NOT derived. A derived `Default` is a public
/// erasing constructor reachable by `..Default::default()` struct-update
/// syntax: it yields length 0, `declared_count() == 0`, and an unfiltered run
/// with no error. Nothing needs it, so it does not exist.
#[derive(Debug, Clone)]
pub struct QueryConditions {
    /// Dataset these came from, so a dropped filter can name it.
    dataset: String,
    per_query: Vec<Option<Value>>,
}

impl QueryConditions {
    /// Wrap the raw per-query conditions read from a dataset file.
    ///
    /// `pub(crate)` **of the library crate**: the only code that can turn raw
    /// JSON into a `QueryConditions` is `readers/`. The binary crate — every
    /// engine — cannot reach it, and no reader hands out the raw vector.
    pub(crate) fn new(per_query: Vec<Option<Value>>) -> Self {
        Self {
            dataset: String::new(),
            per_query,
        }
    }

    /// `n` queries that declare no filter — the HDF5/JSONL formats, which have
    /// nowhere to put conditions.
    ///
    /// Public because the binary's `dataset.rs` builds these. Unlike
    /// [`Self::new`] it cannot *fabricate* a filter — but it can **erase** every
    /// real one, so it is escape 3 in the module docs, and
    /// `filter_guard::no_engine_erases_its_conditions_with_the_unfiltered_constructor`
    /// keeps engines away from it.
    pub fn unfiltered(n: usize) -> Self {
        Self {
            dataset: String::new(),
            per_query: vec![None; n],
        }
    }

    /// Label these conditions with the dataset they came from, so a dropped
    /// filter names the dataset as well as the engine and the query index.
    pub fn from_dataset(mut self, dataset: &str) -> Self {
        self.dataset = dataset.to_string();
        self
    }

    /// How many queries declare a filter. Absent / `null` / `{}` do not count.
    pub fn declared_count(&self) -> usize {
        self.per_query
            .iter()
            .filter(|c| declared(c.as_ref()).is_some())
            .count()
    }

    /// Number of queries.
    pub fn len(&self) -> usize {
        self.per_query.len()
    }

    pub fn is_empty(&self) -> bool {
        self.per_query.is_empty()
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
            .map(|(idx, cond)| resolve_in(engine, &self.dataset, idx, cond.as_ref(), &parse))
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
    resolve_in(engine, "", idx, conditions, parse)
}

/// [`resolve`] with the dataset name for the error message.
pub fn resolve_in<T>(
    engine: &str,
    dataset: &str,
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
            dataset,
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
pub fn dropped(engine: &str, dataset: &str, idx: usize, conditions: &Value, why: &str) -> String {
    let on = if dataset.is_empty() {
        String::new()
    } else {
        format!(" of dataset `{dataset}`")
    };
    format!(
        "{engine} cannot express the filter on query {idx}{on}: {why}. Conditions: \
         {conditions}. Running the query without it would search UNFILTERED while its ground \
         truth is filtered, so the recall reported would be for a filter that was never \
         applied (issue #219). Fix the engine's filter builder, or drop this dataset from \
         the run."
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

    /// The shape KiviDB uses: a builder whose "produced nothing" state is a
    /// match-all SENTINEL rather than `None`. There used to be a
    /// `resolve_all_total` here that returned a bare `Vec<T>` and never compared
    /// the builder's output against the unfiltered value, so `{"and": []}` was
    /// laundered into a match-all prefilter and the run continued. The sentinel
    /// must be mapped back to `None` at the call site, and then rejected.
    #[test]
    fn a_match_all_sentinel_is_a_drop_not_an_unfiltered_query() {
        let sentinel = "*";
        // The real builder renders `*` where `toy` returns `None`; the caller
        // must map that sentinel back to "produced nothing".
        let render = |c: &Value| -> Result<Option<String>, String> {
            let rendered = toy(c).unwrap_or_else(|| sentinel.to_string());
            Ok((rendered != sentinel).then_some(rendered))
        };
        // absent -> unfiltered, and the caller may substitute the sentinel
        let out = conds(vec![Some(Value::Null)])
            .try_resolve_all("Toy", render)
            .unwrap();
        assert!(!out[0].is_filtered());
        // declared but rendering the sentinel -> Err, never a match-all run
        let err = conds(vec![Some(json!({"and":[]}))])
            .try_resolve_all("Toy", render)
            .unwrap_err();
        assert!(err.contains("UNFILTERED"), "{err}");
    }

    /// `resolve_all_total` also doubled as a raw-JSON accessor
    /// (`resolve_all_total(|| Null, |c| Ok(c.clone()))` handed back
    /// `Vec<Value>` for every query, absent ones included). It is gone.
    ///
    /// The nearest surviving spelling is `resolve_all("X", |c| Some(c.clone()))`
    /// — documented in the module header as escape (2). This test pins exactly
    /// how far it gets: it can echo a DECLARED condition (nothing here can tell
    /// an echo from a real filter), but it still cannot produce raw JSON for an
    /// absent one, and it still cannot turn a declared filter into an unfiltered
    /// query.
    #[test]
    fn the_echo_escape_still_cannot_unfilter_a_declared_query() {
        let out = conds(vec![Some(Value::Null), Some(json!({"and":[]}))])
            .resolve_all("Toy", |v| Some(v.clone()))
            .unwrap();
        assert!(!out[0].is_filtered(), "an absent condition yields no JSON");
        assert_eq!(out[1].as_ref(), Some(&json!({"and":[]})));
    }

    #[test]
    fn query_conditions_reports_its_shape_without_exposing_it() {
        let c = conds(vec![Some(Value::Null), Some(json!({"and":[]}))]);
        assert_eq!(c.len(), 2);
        assert_eq!(c.declared_count(), 1);
    }

    #[test]
    fn the_dataset_name_is_named_in_the_error() {
        let err = QueryConditions::new(vec![Some(json!({"and":[]}))])
            .from_dataset("random-geo-radius-100-angular-filters")
            .resolve_all("Toy", toy)
            .unwrap_err();
        assert!(
            err.contains("random-geo-radius-100-angular-filters"),
            "{err}"
        );
    }

    #[test]
    fn unfiltered_conditions_declare_nothing() {
        let c = QueryConditions::unfiltered(4);
        assert_eq!(c.len(), 4);
        assert_eq!(c.declared_count(), 0);
        assert!(c
            .resolve_all("Toy", toy)
            .unwrap()
            .iter()
            .all(|f| !f.is_filtered()));
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
