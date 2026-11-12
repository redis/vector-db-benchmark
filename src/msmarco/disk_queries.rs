//! `msmarco-disk-v1`: compound text/tag query families designed to stress
//! storage rather than CPU (cloud-benchmarks-private
//! `PLAN-disk-bound-text-tag-2026-11-08.md`).
//!
//! The `msmarco-v1` suite runs 32 queries per family on common terms and
//! low-cardinality tags, so its working set is a few posting lists that stay
//! in page cache on every engine. These families invert every one of those
//! properties, from corpus statistics only and identically for every engine:
//!
//! | family | query | matches |
//! |---|---|---|
//! | `rare-term` | one `segment` term, document frequency [`RARE_DF`] | the df |
//! | `two-rare-terms` | AND of two co-occurring `segment` terms, each df [`PAIR_DF`] | ≥1 |
//! | `rare-term-and-category` | a rare `segment` term AND the `category` of one of its passages | ≥1 |
//! | `url-eq` | `url` exact match | the passages of that URL |
//! | `docid-eq` | `docid` exact match | 1 |
//! | `deep-fetch` | one `segment` term, df [`DEEP_DF`], run with `top: 100` and `fetch` | the df |
//! | `docid-batch` | `docid` exact match on any of [`DOCID_BATCH_K`] random docids, run with `top: 100` and `fetch` | [`DOCID_BATCH_K`] |
//! | `docid-batch-500` | any of [`DOCID_BATCH_500_K`] random docids, `top: 500`, `fetch` | 500 |
//! | `docid-batch-1000` | any of [`DOCID_BATCH_1000_K`] random docids, `top: 1000`, `fetch` | 1000 |
//! | `url-batch` | `url` any of [`URL_BATCH_N`] random URLs, `top: 100`, `fetch` | the passages of those URLs |
//! | `rare-terms-or` | OR of [`RARE_OR_N`] `segment` terms, each df [`RARE_OR_DF`], `top: 100`, `fetch` | the DISTINCT passages holding any term |
//! | `rare-terms-or-and-category` | that OR AND the `category` of one of its passages | ≥1 |
//!
//! # Exact by construction
//!
//! Every term is a [`text_match`] term (one lowercase ASCII word, not a
//! stopword) with ZERO ambiguous passages in the whole corpus under the
//! [`text_match::classify`] rule, on the segment text as `analysis: parity`
//! uploads it (whitespace normalized, no stemming on either engine). A term
//! contained in any over-long ASCII run of the corpus is excluded too, and so
//! is any term [`text_match::redis_escape_terms`] names for any passage: a
//! term next to, or joined by, a RediSearch backslash escape or control byte
//! (`\ term` is the Redis token ` term`, `plant\n` the token `plantn`), whose
//! Redis count the classifier cannot vouch for. So the
//! number of passages each engine matches is the document frequency counted
//! here, and every query carries that EXACT `expected_count`; the runner checks
//! it on every reply (the total under `total_hits: exact`, the page length
//! under `none`). Tag values are restricted to strings both engines index as
//! one exact tag (printable ASCII, no `;` — the Redis TAG separator).
//!
//! # Deterministic, uniform, never repeated (unless cycled on purpose)
//!
//! Each family's items are ranked by a seeded SHA-256 key of the item and the
//! first `per_family + RESERVED` are kept. The timed lines are in key order,
//! so any contiguous slice of them is a uniform sample of that family's
//! selection, less the [`PROBE_RESERVE`] items taken for the probe reserve.
//! The rare-term and rare-term-and-category term sets are disjoint;
//! deep-fetch is drawn from its own band independently (the overlap is
//! recorded in the manifest).
//!
//! `docid-batch` reaches many documents spread uniformly over the whole
//! corpus, where every other family reaches a few (or the frequent ones):
//! each query is a disjunction of [`DOCID_BATCH_K`] distinct docids, so a
//! `top: 100` fetch reads [`DOCID_BATCH_K`] random documents. Its lines are
//! epochs of seeded random partitions of the docid-eq eligible pool
//! ([`write_docid_batch`]); within one epoch no docid repeats (docids do
//! repeat across epochs, and the prime and reserve docids come back in later
//! epochs' timed lines).
//!
//! # High-pressure families (v6)
//!
//! On Redis Flex the search index stays in RAM and only fetching documents
//! reads flash, so a family loads storage only if it matches many documents
//! spread over the whole corpus AND fetches a large page. The v6 families do
//! both, identically on both engines, and refuse `total_hits: "none"`
//! ([`Family::check_params`]). Each is an epoch batch like `docid-batch`
//! ([`Batches`], [`write_docid_batch`]): epoch `e` ranks the family's pool
//! by `rank_key(seed, "<family>/epoch-<e>", item)` and cuts it into batches,
//! so no item repeats within an epoch. Counts are exact: a docid batch is K;
//! a `url-batch` line is the sum of its distinct URLs' counts (one URL per
//! passage, `url-eq` eligibility); a `rare-terms-or` line is the size of the
//! union of its terms' postings, each passage ONCE ([`union_rows`]); the
//! category variant counts the union's passages in the chosen category. The
//! tag-union batches hold `per_family × 100 / items` timed lines
//! ([`tag_batch_timed_lines`]) so their files stay near `docid-batch`'s
//! size. Parameters are constants checked at compile time against the
//! OpenSearch defaults ([`OPENSEARCH_MAX_RESULT_WINDOW`],
//! [`OPENSEARCH_MAX_TERMS_COUNT`], [`OPENSEARCH_MAX_CLAUSE_COUNT`]) and
//! recorded in the manifest (`high_pressure`, `families.<f>.batch`).
//!
//! Line 0 of each family file is the connection-priming query (the first
//! item in key order) and lines 1..=[`PROBE_RESERVE`] are reserved for the
//! Redis `WITHOUTCOUNT` probe: neither ever appears in a timed window. The
//! reserve holds the [`PROBE_RESERVE`] other items with the highest expected
//! count (ties in key order), so every reserve count is at least every timed
//! count ([`probe_layout`]): whenever a timed query has more matches than the
//! page, every probe query does too, and the probe always has truncated pages
//! to compare. A uniform reserve would usually hold none for the families
//! whose count is typically 1, and the probe would refuse the run. A timed
//! window is `compound.offset` + `--queries` lines of the rest; the loader
//! refuses a window that runs past the end of the file instead of cycling.
//! A point that must run for a fixed time instead opts in to cycling its
//! window in order with `compound.duration_s` ([`crate::duration_mode`]);
//! the window loaded is the same, once, and the result records the repeats.
//!
//! # Files
//!
//! `prepare-msmarco --add-disk-queries` writes `<dataset>/msmarco-disk-v1/`:
//! one `<family>.jsonl` per family (one JSON array per line, see
//! [`Family::conditions`]) and `manifest.json` (seed, df bands, pool sizes,
//! counts, expected-count statistics, per-file SHA-256, and the identity of the
//! corpus it was computed from). `PREPARED.json` is not modified, so the
//! preparation identity the existing results carry is unchanged.

use crate::compound_text::Query;
use crate::msmarco::text_match::{self, TextMatch};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

/// The `compound.suite` name.
pub const SUITE: &str = "msmarco-disk-v1";
/// Directory (inside the prepared dataset) holding the query files.
pub const DIR: &str = "msmarco-disk-v1";
pub const MANIFEST: &str = "manifest.json";
/// Version of the selection and layout rules below. Bump it when any rule
/// changes. v2: the probe reserve holds the highest-count items
/// ([`probe_layout`]); v1 files are refused. v3: a safe `url` / `docid`
/// that is also a Redis tag piece of an unsafe value ([`redis_tag_pieces`])
/// is never selected; v2 files are refused. v4: a term
/// [`text_match::redis_escape_terms`] names for any passage (RediSearch
/// backslash escapes and control bytes) is never selected, in any text
/// family; v3 files are refused. v5: adds the `docid-batch` family; the
/// six v4 family files are byte-identical (same rules, same keys); v4 files
/// are refused (they lack `docid-batch.jsonl`). v6: adds the high-pressure
/// families `docid-batch-500`, `docid-batch-1000`, `url-batch`,
/// `rare-terms-or` and `rare-terms-or-and-category`; the seven v5 family
/// files are byte-identical; v5 files are refused (they lack the new files).
pub const GENERATOR: &str = "vdb-msmarco-disk/v6";
/// Domain of the seeded rank keys ([`rank_key`]). It stays at its first value
/// while only the file layout changes, so the same items are selected.
pub const KEY_DOMAIN: &str = "vdb-msmarco-disk/v1";
/// The only text field the families query.
pub const TEXT_FIELD: &str = "segment";
/// Document-frequency band of `rare-term` and `rare-term-and-category` terms.
pub const RARE_DF: (u64, u64) = (1, 100);
/// Document-frequency band of each `two-rare-terms` term.
pub const PAIR_DF: (u64, u64) = (100, 1000);
/// Document-frequency band of `deep-fetch` terms: the part of [`RARE_DF`]
/// above the default 10-hit page, so a `top: 100` fetch reads 11–100
/// documents per query. Uniform over the long tail, most `rare-term` terms
/// match one or two passages (on the 10M corpus 62% of eligible terms have
/// df 1), which would make a top-100 fetch no deeper than a top-10 one; and a
/// narrower band at the top (50–100) holds only ~15K eligible terms at 10M,
/// below [`MIN_PER_FAMILY`].
pub const DEEP_DF: (u64, u64) = (11, 100);
/// Line 0: the priming query. Lines 1..=PROBE_RESERVE: the probe reserve.
pub const PROBE_RESERVE: usize = 64;
/// Lines before the timed pool.
pub const RESERVED: usize = 1 + PROBE_RESERVE;
/// Every family must offer at least this many timed queries.
pub const MIN_PER_FAMILY: usize = 100_000;
/// Timed queries generated per family unless `--disk-queries-per-family` says
/// otherwise (capped by the family's eligible pool, except `docid-batch`,
/// whose epochs reach any count).
pub const DEFAULT_PER_FAMILY: usize = 1_000_000;
pub const DEFAULT_SEED: u64 = 1;
/// The fixed result size of `deep-fetch`.
pub const DEEP_FETCH_TOP: usize = 100;
/// Docids per `docid-batch` query, its exact count and its fixed `top`.
/// 100 is `deep-fetch`'s page: `deep-fetch` returns its term's df (11–100,
/// [`DEEP_DF`]) documents from one rare term's posting list, `docid-batch`
/// always fills the page with 100 documents from uniformly random places in
/// the corpus.
pub const DOCID_BATCH_K: usize = 100;
/// Docids per `docid-batch-500` query, its exact count and its fixed `top`.
pub const DOCID_BATCH_500_K: usize = 500;
/// Docids per `docid-batch-1000` query, its exact count and its fixed `top`.
pub const DOCID_BATCH_1000_K: usize = 1000;
/// Distinct URLs per `url-batch` query. The 10M corpus is a uniform 8.8%
/// hash sample of the 113.5M MS MARCO v2.1 segments, so most URLs keep one
/// or two of their document's segments (a long document keeps more): at a
/// mean of 1–3 passages per URL, 300 URLs match about 300–1,000 passages.
/// The manifest records the pool's mean (`batch.pool_mean_count`) so the
/// constant can be checked against the corpus.
pub const URL_BATCH_N: usize = 300;
/// `segment` terms per `rare-terms-or` / `rare-terms-or-and-category` query.
pub const RARE_OR_N: usize = 40;
/// Document-frequency band of each `rare-terms-or` term. On a long-tailed
/// (Zipf) vocabulary the mean df over 11–100 is about 27, so the union of
/// [`RARE_OR_N`] = 40 terms is about 1,100 passages (at least 440 less
/// overlaps, at most 4,000); the band is [`DEEP_DF`]'s, which holds more
/// than [`MIN_PER_FAMILY`] eligible terms on the 10M corpus.
pub const RARE_OR_DF: (u64, u64) = (11, 100);
/// The fixed page (`top`, with `fetch`) of `url-batch`, `rare-terms-or` and
/// `rare-terms-or-and-category`: their match sets are larger, so each query
/// fetches a full 100-document page.
pub const HIGH_PRESSURE_TOP: usize = 100;
/// The tag values per line whose line budget the tag-union batches share:
/// `docid-batch-K` holds `per_family × 100 / K` timed lines and `url-batch`
/// `per_family × 100 / URL_BATCH_N` (at least the minimum), so each file
/// carries about as many tag values as `docid-batch` (3–4 GB of docids at
/// 10M) instead of K/100 times as many.
pub const DOCID_BATCH_LINE_BUDGET_K: usize = DOCID_BATCH_K;
/// OpenSearch `index.max_result_window` default: the largest `from + size`.
pub const OPENSEARCH_MAX_RESULT_WINDOW: usize = 10_000;
/// OpenSearch `index.max_terms_count` default: the most values of a `terms`.
pub const OPENSEARCH_MAX_TERMS_COUNT: usize = 65_536;
/// OpenSearch `indices.query.bool.max_clause_count` default (1,024). Lucene
/// counts the leaf queries of the whole rewritten query, not per `bool`: a
/// `rare-terms-or` query has one `match` (one term query) per term, the
/// category variant one more for the category filter. A `terms` query is
/// one term-set query and does not count per value.
pub const OPENSEARCH_MAX_CLAUSE_COUNT: usize = 1_024;
const _: () = {
    let tag_batches = [
        DOCID_BATCH_K,
        DOCID_BATCH_500_K,
        DOCID_BATCH_1000_K,
        URL_BATCH_N,
    ];
    let mut i = 0;
    while i < tag_batches.len() {
        assert!(tag_batches[i] <= OPENSEARCH_MAX_TERMS_COUNT);
        assert!(tag_batches[i] <= OPENSEARCH_MAX_RESULT_WINDOW);
        i += 1;
    }
    assert!(HIGH_PRESSURE_TOP <= OPENSEARCH_MAX_RESULT_WINDOW);
    // Every term of rare-terms-or-and-category plus its category filter.
    assert!(RARE_OR_N < OPENSEARCH_MAX_CLAUSE_COUNT);
};
/// The fields every v6 high-pressure family fetches, on both engines, in
/// this order: the same stored bytes leave each engine per hit.
pub const HIGH_PRESSURE_FETCH: [&str; 3] = ["title", "segment", "url"];
/// Longest `url` used as a tag value.
pub const MAX_TAG_LEN: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    RareTerm,
    TwoRareTerms,
    RareTermAndCategory,
    UrlEq,
    DocidEq,
    DeepFetch,
    DocidBatch,
    DocidBatch500,
    DocidBatch1000,
    UrlBatch,
    RareTermsOr,
    RareTermsOrAndCategory,
}

impl Family {
    pub const ALL: [Family; 12] = [
        Family::RareTerm,
        Family::TwoRareTerms,
        Family::RareTermAndCategory,
        Family::UrlEq,
        Family::DocidEq,
        Family::DeepFetch,
        Family::DocidBatch,
        Family::DocidBatch500,
        Family::DocidBatch1000,
        Family::UrlBatch,
        Family::RareTermsOr,
        Family::RareTermsOrAndCategory,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Family::RareTerm => "rare-term",
            Family::TwoRareTerms => "two-rare-terms",
            Family::RareTermAndCategory => "rare-term-and-category",
            Family::UrlEq => "url-eq",
            Family::DocidEq => "docid-eq",
            Family::DeepFetch => "deep-fetch",
            Family::DocidBatch => "docid-batch",
            Family::DocidBatch500 => "docid-batch-500",
            Family::DocidBatch1000 => "docid-batch-1000",
            Family::UrlBatch => "url-batch",
            Family::RareTermsOr => "rare-terms-or",
            Family::RareTermsOrAndCategory => "rare-terms-or-and-category",
        }
    }

    pub fn parse(name: &str) -> Result<Family, String> {
        Family::ALL
            .into_iter()
            .find(|f| f.name() == name)
            .ok_or_else(|| {
                format!(
                    "unknown {SUITE} family {name:?} (families: {})",
                    Family::ALL.map(Family::name).join(", ")
                )
            })
    }

    pub fn file_name(self) -> String {
        format!("{}.jsonl", self.name())
    }

    /// The document-frequency band of each text term, if the family has one.
    pub fn df_band(self) -> Option<(u64, u64)> {
        match self {
            Family::RareTerm | Family::RareTermAndCategory => Some(RARE_DF),
            Family::TwoRareTerms => Some(PAIR_DF),
            Family::DeepFetch => Some(DEEP_DF),
            Family::RareTermsOr | Family::RareTermsOrAndCategory => Some(RARE_OR_DF),
            Family::UrlEq
            | Family::DocidEq
            | Family::DocidBatch
            | Family::DocidBatch500
            | Family::DocidBatch1000
            | Family::UrlBatch => None,
        }
    }

    /// Fields the family's conditions use (text, then tags).
    pub fn fields(self) -> &'static [&'static str] {
        match self {
            Family::RareTerm | Family::TwoRareTerms | Family::DeepFetch | Family::RareTermsOr => {
                &[TEXT_FIELD]
            }
            Family::RareTermAndCategory | Family::RareTermsOrAndCategory => {
                &[TEXT_FIELD, "category"]
            }
            Family::UrlEq | Family::UrlBatch => &["url"],
            Family::DocidEq
            | Family::DocidBatch
            | Family::DocidBatch500
            | Family::DocidBatch1000 => &["docid"],
        }
    }

    /// Items per line of a batch family: the docids of a `docid-batch*`
    /// line, the URLs of `url-batch`, the terms of `rare-terms-or*`.
    pub fn batch_size(self) -> Option<usize> {
        match self {
            Family::DocidBatch => Some(DOCID_BATCH_K),
            Family::DocidBatch500 => Some(DOCID_BATCH_500_K),
            Family::DocidBatch1000 => Some(DOCID_BATCH_1000_K),
            Family::UrlBatch => Some(URL_BATCH_N),
            Family::RareTermsOr | Family::RareTermsOrAndCategory => Some(RARE_OR_N),
            _ => None,
        }
    }

    /// A docid batch: every line is K docids with count K.
    pub fn is_docid_batch(self) -> bool {
        matches!(
            self,
            Family::DocidBatch | Family::DocidBatch500 | Family::DocidBatch1000
        )
    }

    /// The `top` a fetch family is defined with: `deep-fetch`, the docid
    /// batches (their K), `url-batch` and `rare-terms-or*`
    /// ([`HIGH_PRESSURE_TOP`]) must run with exactly this page and
    /// `fulltext.fetch`.
    pub fn fixed_top(self) -> Option<usize> {
        match self {
            Family::DeepFetch => Some(DEEP_FETCH_TOP),
            Family::DocidBatch | Family::DocidBatch500 | Family::DocidBatch1000 => {
                self.batch_size()
            }
            Family::UrlBatch | Family::RareTermsOr | Family::RareTermsOrAndCategory => {
                Some(HIGH_PRESSURE_TOP)
            }
            _ => None,
        }
    }

    /// Whether the family refuses `total_hits: "none"`: the docid batches
    /// and every v6 high-pressure family.
    pub fn requires_exact_total(self) -> bool {
        self.batch_size().is_some()
    }

    /// The exact `fulltext.fetch` field list a v6 high-pressure family
    /// requires ([`HIGH_PRESSURE_FETCH`]): fetch parity is the point of these
    /// families, so another list (fewer, more or other fields) is refused.
    /// `None` for the v5 families, which only require `fetch` to be set.
    pub fn fixed_fetch(self) -> Option<&'static [&'static str]> {
        (self.batch_size().is_some() && self != Family::DocidBatch)
            .then_some(&HIGH_PRESSURE_FETCH[..])
    }

    /// Refuse a fetch field list other than [`Family::fixed_fetch`].
    pub fn check_fetch_fields(self, fetch: Option<&[String]>) -> Result<(), String> {
        let Some(want) = self.fixed_fetch() else {
            return Ok(());
        };
        if fetch.is_some_and(|f| f.iter().map(String::as_str).eq(want.iter().copied())) {
            return Ok(());
        }
        Err(format!(
            "{SUITE} family {} fetches exactly {want:?} on every engine (fetch parity); got \
             {fetch:?}",
            self.name()
        ))
    }

    /// Search-config requirements the family's name promises. `deep-fetch`
    /// is defined as the top-100 fetch, `docid-batch` as the fetch of all of
    /// its [`DOCID_BATCH_K`] matches (`docid-batch-500` / `-1000` of their
    /// 500 / 1,000), `url-batch` and `rare-terms-or*` as the top-100 fetch of
    /// a larger match set; any other result size or the ids-only variant
    /// would be a different measurement under the same name.
    ///
    /// The batch families also refuse `total_hits: "none"`. A docid batch's
    /// page is the whole match set, so counting costs nothing, while
    /// RediSearch `WITHOUTCOUNT` returns some of these 100-tag unions out of
    /// score order (seen on Redis 8.8), which the per-reply rank check
    /// rightly refuses; and with no truncated page the `WITHOUTCOUNT` probe
    /// has nothing to compare. `url-batch` and `rare-terms-or*` are unions
    /// too (Redis Flex rejects `WITHOUTCOUNT` outright), so every
    /// high-pressure family is measured with exact totals on both engines.
    pub fn check_params(self, top: usize, fetch: bool, total_none: bool) -> Result<(), String> {
        if let Some(want) = self.fixed_top() {
            if top != want || !fetch {
                return Err(format!(
                    "{SUITE} family {} is defined as `top: {want}` with `fulltext.fetch`; got \
                     top {top}{}",
                    self.name(),
                    if fetch { "" } else { " without fetch" }
                ));
            }
        }
        if self.requires_exact_total() && total_none {
            return Err(if self.is_docid_batch() {
                format!(
                    "{SUITE} family {} needs `total_hits: \"exact\"` (the default): every page \
                     is the whole {}-document match set, so the count is free, and RediSearch \
                     WITHOUTCOUNT does not keep these tag unions in score order",
                    self.name(),
                    self.batch_size().unwrap_or(0)
                )
            } else {
                format!(
                    "{SUITE} family {} needs `total_hits: \"exact\"` (the default): it is a \
                     union whose exact count is checked on every reply, and RediSearch \
                     WITHOUTCOUNT does not keep unions in score order (Redis Flex rejects it)",
                    self.name()
                )
            });
        }
        Ok(())
    }

    /// Logical conditions (the compound AST) and exact count of one line.
    pub fn conditions(self, line: &Value) -> Result<(Value, u64), String> {
        let a = line
            .as_array()
            .ok_or_else(|| format!("{SUITE} {}: line is not an array", self.name()))?;
        let s = |i: usize| -> Result<&str, String> {
            a.get(i)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{SUITE} {}: field {i} is not a string", self.name()))
        };
        let arity = match self {
            Family::RareTerm
            | Family::DeepFetch
            | Family::UrlEq
            | Family::DocidEq
            | Family::DocidBatch
            | Family::DocidBatch500
            | Family::DocidBatch1000
            | Family::UrlBatch
            | Family::RareTermsOr => 2,
            Family::TwoRareTerms | Family::RareTermAndCategory | Family::RareTermsOrAndCategory => {
                3
            }
        };
        if a.len() != arity {
            return Err(format!(
                "{SUITE} {}: expected {arity} fields, got {}",
                self.name(),
                a.len()
            ));
        }
        let count = a[arity - 1]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or_else(|| format!("{SUITE} {}: count must be a positive integer", self.name()))?;
        let text = |t: &str| json!({TEXT_FIELD: {"match": {"text": t}}});
        let tag = |f: &str, v: &str| json!({f: {"match": {"value": v}}});
        if self.is_docid_batch() {
            return Ok((docid_batch_conditions(self, &a[0], count)?, count));
        }
        let conditions = match self {
            Family::RareTerm | Family::DeepFetch => text(s(0)?),
            Family::TwoRareTerms => json!({"and": [text(s(0)?), text(s(1)?)]}),
            Family::RareTermAndCategory => json!({"and": [text(s(0)?), tag("category", s(1)?)]}),
            Family::UrlEq => tag("url", s(0)?),
            Family::DocidEq => tag("docid", s(0)?),
            Family::UrlBatch => {
                let urls = batch_items(self, &a[0], is_safe_tag, "a safe url tag")?;
                json!({"url": {"match": {"any": urls}}})
            }
            Family::RareTermsOr | Family::RareTermsOrAndCategory => {
                let terms = batch_items(
                    self,
                    &a[0],
                    |t| text_match::validate_term(t).is_ok(),
                    "a text_match term",
                )?;
                let any = json!({"or": terms.iter().map(|t| text(t)).collect::<Vec<_>>()});
                if self == Family::RareTermsOr {
                    any
                } else {
                    let cat = s(1)?;
                    if !is_safe_tag(cat) {
                        return Err(format!(
                            "{SUITE} {}: category {cat:?} is not a safe tag",
                            self.name()
                        ));
                    }
                    json!({"and": [any, tag("category", cat)]})
                }
            }
            Family::DocidBatch | Family::DocidBatch500 | Family::DocidBatch1000 => {
                unreachable!("handled above")
            }
        };
        Ok((conditions, count))
    }
}

/// The items of a batch line's field 0: exactly [`Family::batch_size`]
/// distinct strings, each passing `valid`.
fn batch_items<'a>(
    family: Family,
    items: &'a Value,
    valid: impl Fn(&str) -> bool,
    what: &str,
) -> Result<Vec<&'a str>, String> {
    let name = family.name();
    let k = family.batch_size().expect("a batch family");
    let a = items
        .as_array()
        .ok_or_else(|| format!("{SUITE} {name}: field 0 is not an array"))?;
    if a.len() != k {
        return Err(format!(
            "{SUITE} {name}: a line holds exactly {k} items; got {}",
            a.len()
        ));
    }
    let mut seen = HashSet::with_capacity(a.len());
    let mut out = Vec::with_capacity(a.len());
    for v in a {
        let s = v
            .as_str()
            .filter(|s| valid(s))
            .ok_or_else(|| format!("{SUITE} {name}: {v} is not {what}"))?;
        if !seen.insert(s) {
            return Err(format!("{SUITE} {name}: {s:?} repeats in one line"));
        }
        out.push(s);
    }
    Ok(out)
}

/// The conditions of a docid-batch line `[[d1, …, dK], K]`: `docid` equal
/// to any of exactly K ([`Family::batch_size`]) distinct safe tags, `K`
/// being the count. Each docid matches exactly one passage, so the count is
/// `K`.
fn docid_batch_conditions(family: Family, docids: &Value, count: u64) -> Result<Value, String> {
    let name = family.name();
    let k = family.batch_size().expect("a docid batch");
    let d = docids
        .as_array()
        .ok_or_else(|| format!("{SUITE} {name}: field 0 is not an array of docids"))?;
    if d.len() != k || count != k as u64 {
        return Err(format!(
            "{SUITE} {name}: a line holds exactly {k} docids with count {k}; got {} docids, \
             count {count}",
            d.len()
        ));
    }
    batch_items(family, docids, is_safe_tag, "a safe docid tag")?;
    Ok(json!({"docid": {"match": {"any": d}}}))
}

/// A tag value both engines index and query as exactly one tag: printable
/// ASCII without space (so no edge whitespace, nothing a client or engine
/// normalizes) and without `;` (the Redis TAG separator).
pub fn is_safe_tag(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TAG_LEN
        && value
            .bytes()
            .all(|b| (0x21..=0x7e).contains(&b) && b != b';')
}

/// The tags Redis indexes for one stored value of a `TAG SEPARATOR ;` field:
/// the value split on `;`, each piece trimmed of the whitespace RediSearch
/// trims (C `isspace`: space, `\t`, `\n`, `\v`, `\f`, `\r`), empty pieces
/// dropped. OpenSearch indexes the whole value as one keyword, so a safe
/// value that is also a piece of a *different* (unsafe) value matches more
/// passages on Redis than on OpenSearch: `http://e.com/p;jsessionid=1` and
/// `http://e.com/p ` both carry the tag `http://e.com/p`. Such values are
/// not selected ([`generate`]).
pub fn redis_tag_pieces(value: &str) -> impl Iterator<Item = &str> {
    value
        .split(';')
        .map(|p| p.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r')))
        .filter(|p| !p.is_empty())
}

fn sha256_hex(bytes: impl AsRef<[u8]>) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((h.finalize().iter().map(|b| format!("{b:02x}")).collect(), n))
}

/// The preparation identity every compound result already carries:
/// SHA-256 of `PREPARED.json`.
pub fn preparation_sha256(dir: &Path) -> Result<String, String> {
    let path = dir.join("PREPARED.json");
    std::fs::read(&path)
        .map(sha256_hex)
        .map_err(|e| format!("{SUITE}: {}: {e}", path.display()))
}

/// The seeded rank key of one item of one family.
pub fn rank_key(seed: u64, family: &str, item: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(KEY_DOMAIN.as_bytes());
    h.update([0]);
    h.update(seed.to_be_bytes());
    h.update(family.as_bytes());
    h.update([0]);
    h.update(item.as_bytes());
    let d = h.finalize();
    u64::from_be_bytes(d[..8].try_into().unwrap())
}

/// The rank key of a term pair from its two terms' keys (`a` sorts first).
/// Plain integer arithmetic (SplitMix64 finalizer), so it is the same on
/// every platform.
pub fn pair_key(a: u64, b: u64) -> u64 {
    let mut z = a.rotate_left(32) ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The `cap` smallest distinct keys seen, each with how often it was seen.
///
/// The count of a key that is in the final set is exact: such a key was
/// inserted at its first occurrence (the set was not full, or the key was
/// below the maximum then, which only decreases) and was never evicted
/// (eviction is permanent: an evicted key is above every later maximum).
#[derive(Debug)]
pub struct Smallest<K: Ord + Clone> {
    cap: usize,
    map: BTreeMap<K, u64>,
}

impl<K: Ord + Clone> Smallest<K> {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            map: BTreeMap::new(),
        }
    }
    pub fn add(&mut self, key: K) {
        if self.map.len() >= self.cap {
            match self.map.last_key_value() {
                Some((max, _)) if key > *max => return,
                _ => {}
            }
        }
        if let Some(n) = self.map.get_mut(&key) {
            *n += 1;
            return;
        }
        if self.cap == 0 {
            return;
        }
        if self.map.len() >= self.cap {
            self.map.pop_last();
        }
        self.map.insert(key, 1);
    }
    pub fn into_sorted(self) -> Vec<(K, u64)> {
        self.map.into_iter().collect()
    }
}

#[derive(Deserialize)]
struct Row<'a> {
    #[serde(borrow)]
    docid: Cow<'a, str>,
    #[serde(borrow)]
    url: Cow<'a, str>,
    #[serde(borrow)]
    category: Cow<'a, str>,
    #[serde(borrow)]
    segment: Cow<'a, str>,
}

/// What one passage contributes, computed in parallel.
struct Scanned {
    docid: String,
    url: String,
    category: String,
    /// (term, Yes?) for every term not `No` in this passage.
    terms: Vec<(String, bool)>,
    long_runs: Vec<String>,
    /// [`text_match::redis_escape_terms`] of the passage.
    escape_terms: Vec<String>,
}

fn scan_line(line: &str, row: usize) -> Result<Scanned, String> {
    let r: Row = serde_json::from_str(line)
        .map_err(|e| format!("{SUITE}: payloads.jsonl row {row}: {e}"))?;
    let text = crate::fulltext::normalize_whitespace(&r.segment);
    let mut verdicts = HashMap::new();
    let mut long_runs = Vec::new();
    text_match::doc_verdicts(&text, &mut verdicts, &mut long_runs);
    let mut escape_terms = Vec::new();
    text_match::redis_escape_terms(&text, &mut escape_terms);
    Ok(Scanned {
        docid: r.docid.into_owned(),
        url: r.url.into_owned(),
        category: r.category.into_owned(),
        terms: verdicts
            .into_iter()
            .map(|(t, v)| (t, v == TextMatch::Yes))
            .collect(),
        long_runs,
        escape_terms,
    })
}

/// Stream `payloads.jsonl` in parallel batches, in row order.
fn for_each_row(
    path: &Path,
    mut visit: impl FnMut(usize, Scanned) -> Result<(), String>,
) -> Result<(u64, String), String> {
    const BATCH: usize = 1 << 16;
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 22, file);
    let mut hasher = Sha256::new();
    let mut bytes = 0u64;
    let mut row = 0usize;
    loop {
        let mut lines = Vec::with_capacity(BATCH);
        while lines.len() < BATCH {
            let mut line = String::new();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if n == 0 {
                break;
            }
            hasher.update(line.as_bytes());
            bytes += n as u64;
            lines.push(line);
        }
        if lines.is_empty() {
            break;
        }
        let scanned: Vec<Result<Scanned, String>> = lines
            .par_iter()
            .enumerate()
            .map(|(i, l)| scan_line(l.trim_end_matches(['\n', '\r']), row + i))
            .collect();
        for s in scanned {
            visit(row, s?)?;
            row += 1;
        }
    }
    let _ = row;
    Ok((
        bytes,
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    ))
}

/// Terms contained in an over-long ASCII run anywhere in the corpus: the
/// classifier calls them ambiguous in that passage, and a per-passage table
/// cannot see it (see [`text_match::doc_verdicts`]).
fn long_run_poison(runs: &[String], vocabulary: &HashMap<String, (u64, u64)>) -> HashSet<String> {
    let mut out = HashSet::new();
    for run in runs {
        for seg in run.split(|c: char| !c.is_ascii_lowercase()) {
            let b = seg.as_bytes();
            for i in 0..b.len() {
                for len in text_match::MIN_TERM_LEN..=text_match::MAX_TERM_LEN {
                    if i + len > b.len() {
                        break;
                    }
                    let t = &seg[i..i + len];
                    if vocabulary.contains_key(t) {
                        out.insert(t.to_string());
                    }
                }
            }
        }
    }
    out
}

fn stats(counts: &[u64]) -> Value {
    if counts.is_empty() {
        return Value::Null;
    }
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let q = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
    let mut buckets: BTreeMap<String, u64> = BTreeMap::new();
    for &c in counts {
        let b = match c {
            1 => "1",
            2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            11..=25 => "11-25",
            26..=50 => "26-50",
            51..=100 => "51-100",
            101..=1000 => "101-1000",
            _ => ">1000",
        };
        *buckets.entry(b.to_string()).or_default() += 1;
    }
    json!({
        "min": sorted[0],
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "max": sorted[sorted.len() - 1],
        "mean": counts.iter().sum::<u64>() as f64 / counts.len() as f64,
        "histogram": buckets,
    })
}

/// Arrange one family's selected items (in seeded key order) into the file
/// layout: line 0 the first item (the prime); lines 1..=[`PROBE_RESERVE`] the
/// [`PROBE_RESERVE`] remaining items with the highest expected count (ties in
/// key order), kept in key order; then every other item in key order (the
/// timed pool). Every reserve line's count is therefore at least every timed
/// line's count, so for any `top` below some timed query's count every probe
/// query is cut by the page too.
pub fn probe_layout(family: Family, items: Vec<Value>) -> Result<Vec<Value>, String> {
    if items.len() < RESERVED {
        return Err(format!(
            "{SUITE}: family {} has {} items, fewer than its {RESERVED} reserved lines",
            family.name(),
            items.len()
        ));
    }
    let counts = items
        .iter()
        .map(|l| family.conditions(l).map(|(_, n)| n))
        .collect::<Result<Vec<u64>, String>>()?;
    let order = probe_order(&counts);
    let mut items: Vec<Option<Value>> = items.into_iter().map(Some).collect();
    Ok(order
        .into_iter()
        .map(|i| items[i].take().expect("each line once"))
        .collect())
}

/// The line order [`probe_layout`] gives items with these counts (at least
/// [`RESERVED`] of them): 0, then the reserve in index order, then the rest
/// in index order.
fn probe_order(counts: &[u64]) -> Vec<usize> {
    let mut rest: Vec<usize> = (1..counts.len()).collect();
    rest.sort_unstable_by_key(|&i| (std::cmp::Reverse(counts[i]), i));
    let mut in_reserve = vec![false; counts.len()];
    for &i in &rest[..PROBE_RESERVE] {
        in_reserve[i] = true;
    }
    let reserve = (1..counts.len()).filter(|&i| in_reserve[i]);
    let timed = (1..counts.len()).filter(|&i| !in_reserve[i]);
    std::iter::once(0).chain(reserve).chain(timed).collect()
}

/// The passages (row ids) matched by at least one of `lists`, ascending,
/// each ONCE: two terms of a `rare-terms-or` line can occur in the same
/// passage, which the engines count once.
pub fn union_rows(lists: &[&[u32]]) -> Vec<u32> {
    let mut rows: Vec<u32> = Vec::with_capacity(lists.iter().map(|l| l.len()).sum());
    for l in lists {
        rows.extend_from_slice(l);
    }
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// The pool items ranked by their epoch key, `rank_key(seed,
/// "<family>/epoch-<e>", item)`, ascending: a seeded uniformly random order,
/// cut into consecutive batches by the caller.
fn epoch_order<'a>(
    ids: &[u32],
    item: impl Fn(u32) -> &'a str + Sync,
    seed: u64,
    family: Family,
    epoch: usize,
) -> Vec<(u64, u32)> {
    let domain = format!("{}/epoch-{epoch}", family.name());
    let mut keyed: Vec<(u64, u32)> = ids
        .par_iter()
        .map(|&i| (rank_key(seed, &domain, item(i)), i))
        .collect();
    keyed.par_sort_unstable();
    keyed
}

/// The RediSearch query-string length of a tag union `(@<field>:{v1 | … |
/// vK})` as [`crate::compound_text::Expr::redis`] renders it (every
/// non-alphanumeric byte of a value escaped).
fn redis_tag_union_len<'a>(field: &str, values: impl Iterator<Item = &'a str>) -> usize {
    let mut n = 0usize;
    let mut k = 0usize;
    for v in values {
        n += v.len() + v.bytes().filter(|b| !b.is_ascii_alphanumeric()).count();
        k += 1;
    }
    "(@".len() + field.len() + ":{".len() + n + 3 * k.saturating_sub(1) + "})".len()
}

/// The `docid-batch*` pool: every docid-eq eligible docid (a safe tag, not a
/// Redis tag piece of an unsafe docid) that is unique in the corpus, in row
/// order. A docid that occurs on two passages would match both, so every
/// copy of it is excluded and the count of a batch stays exactly K.
pub struct DocidPool {
    arena: String,
    ends: Vec<usize>,
    /// Indices into `ends` of the unique docids, ascending.
    ids: Vec<u32>,
    /// Distinct docids excluded because they occur more than once.
    duplicates: usize,
}

impl DocidPool {
    fn new(arena: String, ends: Vec<usize>) -> Result<Self, String> {
        let n = u32::try_from(ends.len()).map_err(|_| "too many docids for docid-batch")?;
        let mut pool = Self {
            arena,
            ends,
            ids: Vec::new(),
            duplicates: 0,
        };
        let mut order: Vec<u32> = (0..n).collect();
        order.par_sort_unstable_by(|&a, &b| pool.get(a).cmp(pool.get(b)).then(a.cmp(&b)));
        let mut dup = vec![false; n as usize];
        let mut duplicates = 0;
        for run in order.chunk_by(|&a, &b| pool.get(a) == pool.get(b)) {
            if run.len() > 1 {
                duplicates += 1;
                for &i in run {
                    dup[i as usize] = true;
                }
            }
        }
        pool.duplicates = duplicates;
        pool.ids = (0..n).filter(|&i| !dup[i as usize]).collect();
        Ok(pool)
    }

    fn get(&self, i: u32) -> &str {
        let i = i as usize;
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        &self.arena[start..self.ends[i]]
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Timed lines of a tag-union batch of `k` values (`docid-batch*`,
/// `url-batch`): `per_family` for `docid-batch` (k =
/// [`DOCID_BATCH_LINE_BUDGET_K`]), proportionally fewer for larger batches
/// so each file holds about as many tag values, never below `min`.
pub fn tag_batch_timed_lines(k: usize, per_family: usize, min: usize) -> usize {
    (per_family.saturating_mul(DOCID_BATCH_LINE_BUDGET_K) / k).max(min)
}

/// Write `lines` lines `[[d1, …, dK], K]` of the docid batch `family` to
/// `w`, pushing each line's count onto `counts`, and return the manifest's
/// `batch` record.
///
/// Epoch `e` ranks every pool docid by [`rank_key`] under the family name
/// `<family>/epoch-<e>` and cuts that order into consecutive batches of
/// K (the last `len % K` docids of the epoch are unused):
/// a seeded uniformly random partition, so each batch is a uniformly random
/// K-subset of the pool and no docid repeats within an epoch. Epochs follow
/// each other until `lines` lines are written; within a line the docids are
/// in key order. Every count is K, so [`probe_layout`]
/// (ties in key order) is the identity here: line 0 is the prime, lines
/// 1..=[`PROBE_RESERVE`] the reserve, the rest the timed pool, all in key
/// order.
fn write_docid_batch(
    w: &mut impl Write,
    family: Family,
    pool: &DocidPool,
    seed: u64,
    lines: usize,
    counts: &mut Vec<u64>,
) -> Result<Value, String> {
    let k = family.batch_size().expect("a docid batch");
    let per_epoch = pool.len() / k;
    if per_epoch == 0 {
        return Err(format!("fewer than {k} eligible docids"));
    }
    let mut epoch = 0usize;
    let mut redis_bytes_max = 0usize;
    while counts.len() < lines {
        let keyed = epoch_order(&pool.ids, |i| pool.get(i), seed, family, epoch);
        for chunk in keyed.chunks_exact(k) {
            if counts.len() == lines {
                break;
            }
            let ids: Vec<&str> = chunk.iter().map(|&(_, i)| pool.get(i)).collect();
            if family != Family::DocidBatch {
                redis_bytes_max =
                    redis_bytes_max.max(redis_tag_union_len("docid", ids.iter().copied()));
            }
            let line = json!([ids, k]);
            let (_, n) = family.conditions(&line)?;
            counts.push(n);
            writeln!(w, "{line}").map_err(|e| e.to_string())?;
        }
        epoch += 1;
    }
    let mut record = json!({
        "k": k,
        "epochs": epoch,
        "batches_per_epoch": per_epoch,
        "docids_unused_per_epoch": pool.len() % k,
        "docids_excluded_duplicate": pool.duplicates,
        "rule": format!("the pool is every docid-eq eligible docid that occurs on exactly one passage; epoch e ranks the pool by SHA-256(key domain \"vdb-msmarco-disk/v1\", seed, \"{}/epoch-<e>\", docid) and cuts it into consecutive batches of {k} (the last len % {k} unused); epochs follow each other; a line is one batch in key order with count {k}, every docid matching exactly one passage", family.name()),
    });
    if family != Family::DocidBatch {
        record["timed_lines_rule"] = json!(format!(
            "per_family x {DOCID_BATCH_LINE_BUDGET_K} / {k} timed lines (at least the minimum): \
             about as many docids per file as docid-batch"
        ));
        record["redis_query_bytes_max"] = json!(redis_bytes_max);
        record["opensearch_terms_count"] = json!(k);
        record["opensearch_max_terms_count"] = json!(OPENSEARCH_MAX_TERMS_COUNT);
        record["opensearch_max_result_window"] = json!(OPENSEARCH_MAX_RESULT_WINDOW);
    }
    Ok(record)
}

/// A materialized batch family (`url-batch`, `rare-terms-or*`): every line
/// is `n` pool item ids (a flat array, `n` per line) with its exact count.
struct Batches {
    n: usize,
    items: Vec<u32>,
    epochs: usize,
    per_epoch: usize,
    unused_per_epoch: usize,
}

impl Batches {
    /// `lines` batches of `n` of the pool items `0..pool_len`: epochs of
    /// seeded random partitions of the pool (as [`write_docid_batch`]), each
    /// batch in key order.
    fn new<'a>(
        family: Family,
        pool_len: usize,
        item: impl Fn(u32) -> &'a str + Sync,
        seed: u64,
        lines: usize,
    ) -> Result<Self, String> {
        let n = family.batch_size().expect("a batch family");
        let per_epoch = pool_len / n;
        if per_epoch == 0 {
            return Err(format!(
                "{SUITE}: family {} has {pool_len} eligible items, fewer than one batch of {n}",
                family.name()
            ));
        }
        let ids: Vec<u32> = (0..u32::try_from(pool_len).map_err(|_| "pool too large")?).collect();
        let mut items = Vec::with_capacity(lines * n);
        let mut epoch = 0usize;
        while items.len() < lines * n {
            let keyed = epoch_order(&ids, &item, seed, family, epoch);
            for chunk in keyed.chunks_exact(n) {
                if items.len() == lines * n {
                    break;
                }
                items.extend(chunk.iter().map(|&(_, i)| i));
            }
            epoch += 1;
        }
        Ok(Self {
            n,
            items,
            epochs: epoch,
            per_epoch,
            unused_per_epoch: pool_len % n,
        })
    }

    fn lines(&self) -> std::slice::ChunksExact<'_, u32> {
        self.items.chunks_exact(self.n)
    }

    fn record(&self, pool_len: usize, rule: String) -> Value {
        json!({
            "n": self.n,
            "epochs": self.epochs,
            "batches_per_epoch": self.per_epoch,
            "items_unused_per_epoch": self.unused_per_epoch,
            "pool_items": pool_len,
            "rule": rule,
        })
    }
}

/// Write the lines of a materialized batch family in the probe layout
/// ([`probe_order`] of their counts), pushing each written line's count onto
/// `counts`.
fn write_batches(
    w: &mut impl Write,
    family: Family,
    batches: &Batches,
    line: impl Fn(usize, &[u32]) -> Value,
    line_counts: &[u64],
    counts: &mut Vec<u64>,
) -> Result<(), String> {
    let all: Vec<&[u32]> = batches.lines().collect();
    if all.len() < RESERVED {
        return Err(format!(
            "{SUITE}: family {} has {} lines, fewer than its {RESERVED} reserved lines",
            family.name(),
            all.len()
        ));
    }
    for i in probe_order(line_counts) {
        let l = line(i, all[i]);
        // Validates the line's shape only (arity, distinct valid items, a
        // positive count): the count itself is the one computed by the
        // caller, whose exactness the pass-1/pass-2 df check and the fixture
        // oracles establish.
        let (_, n) = family.conditions(&l)?;
        counts.push(n);
        writeln!(w, "{l}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub struct GenerateOptions {
    /// Timed queries per family (capped by each family's pool, except
    /// `docid-batch`).
    pub per_family: usize,
    pub seed: u64,
    /// The floor every family must reach: [`MIN_PER_FAMILY`], except in unit
    /// fixtures (the CLI has no way to lower it).
    pub min_per_family: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            per_family: DEFAULT_PER_FAMILY,
            seed: DEFAULT_SEED,
            min_per_family: MIN_PER_FAMILY,
        }
    }
}

/// Generate every family from the prepared corpus in `dir` and write the
/// family files and the manifest into `out` (an empty staging directory).
/// Two passes over `payloads.jsonl`. Returns the manifest.
pub fn generate(dir: &Path, opts: &GenerateOptions, out: &Path) -> Result<Value, String> {
    let min = opts.min_per_family;
    if opts.per_family < min || min == 0 {
        return Err(format!(
            "{SUITE}: --disk-queries-per-family {} is below the minimum {min}",
            opts.per_family
        ));
    }
    let prepared = crate::msmarco::preparation::read_manifest(dir)?;
    let passages = prepared["passages"]
        .as_u64()
        .ok_or("PREPARED.json lacks the corpus row count")?;
    let preparation_sha256 = preparation_sha256(dir)?;
    let payloads = dir.join("payloads.jsonl");
    let cap = opts.per_family + RESERVED;
    let seed = opts.seed;

    // Pass 1: document frequencies, and the safe Redis tag pieces of every
    // unsafe url / docid (a safe value equal to one is not selectable).
    let mut vocabulary: HashMap<String, (u64, u64)> = HashMap::new();
    let mut long_runs: Vec<String> = Vec::new();
    let mut escape_poison = HashSet::<String>::new();
    let mut escape_passages = 0u64;
    let mut url_pieces = HashSet::<String>::new();
    let mut docid_pieces = HashSet::<String>::new();
    let mut urls_unsafe = 0u64;
    let mut docids_unsafe = 0u64;
    let mut rows = 0u64;
    let (payload_bytes, payload_sha256) = for_each_row(&payloads, |_, s| {
        rows += 1;
        for (t, yes) in s.terms {
            let e = vocabulary.entry(t).or_insert((0, 0));
            if yes {
                e.0 += 1
            } else {
                e.1 += 1
            }
        }
        long_runs.extend(s.long_runs);
        if !s.escape_terms.is_empty() {
            escape_passages += 1;
            escape_poison.extend(s.escape_terms);
        }
        for (value, pieces, unsafe_rows) in [
            (&s.docid, &mut docid_pieces, &mut docids_unsafe),
            (&s.url, &mut url_pieces, &mut urls_unsafe),
        ] {
            if !is_safe_tag(value) {
                *unsafe_rows += 1;
                for p in redis_tag_pieces(value).filter(|p| is_safe_tag(p)) {
                    if !pieces.contains(p) {
                        pieces.insert(p.to_string());
                    }
                }
            }
        }
        Ok(())
    })?;
    if rows != passages {
        return Err(format!(
            "{SUITE}: payloads.jsonl has {rows} rows but PREPARED.json says {passages}"
        ));
    }
    long_runs.sort();
    long_runs.dedup();
    let poison = long_run_poison(&long_runs, &vocabulary);
    let ambiguous_terms = vocabulary.values().filter(|(_, a)| *a > 0).count();
    // Eligible but for the Redis escape rule: the pool v3 drew from.
    let eligible_v3 = |t: &str, (yes, amb): (u64, u64), band: (u64, u64)| {
        amb == 0
            && yes >= band.0
            && yes <= band.1
            && !poison.contains(t)
            && text_match::validate_term(t).is_ok()
    };
    let eligible = |t: &str, c: (u64, u64), band: (u64, u64)| {
        eligible_v3(t, c, band) && !escape_poison.contains(t)
    };
    // Terms each band loses to the escape rule (before any family's own
    // exclusions), and over every df.
    let escape_excluded = |band: (u64, u64)| {
        vocabulary
            .iter()
            .filter(|(t, c)| eligible_v3(t, **c, band) && escape_poison.contains(t.as_str()))
            .count()
    };
    let pool = |family: Family, exclude: &HashSet<String>| -> Vec<(u64, String, u64)> {
        let band = family.df_band().unwrap();
        let mut p: Vec<(u64, String, u64)> = vocabulary
            .iter()
            .filter(|(t, c)| eligible(t, **c, band) && !exclude.contains(t.as_str()))
            .map(|(t, c)| (rank_key(seed, family.name(), t), t.clone(), c.0))
            .collect();
        p.sort_unstable();
        p
    };
    // Eligible-term document-frequency profile (zero ambiguity, not in a long
    // run), recorded so the bands can be judged against the corpus.
    let mut df_profile: BTreeMap<String, u64> = BTreeMap::new();
    for (t, c) in &vocabulary {
        if eligible(t, *c, (1, u64::MAX)) {
            let b = match c.0 {
                1 => "0001",
                2 => "0002",
                3..=5 => "0003-0005",
                6..=10 => "0006-0010",
                11..=25 => "0011-0025",
                26..=49 => "0026-0049",
                50..=100 => "0050-0100",
                101..=1000 => "0101-1000",
                1001..=10000 => "1001-10000",
                _ => ">10000",
            };
            *df_profile.entry(b.to_string()).or_default() += 1;
        }
    }
    let mut pools: BTreeMap<Family, usize> = BTreeMap::new();
    let take = |family: Family, p: &[(u64, String, u64)]| -> Result<usize, String> {
        if p.len() < min + RESERVED {
            return Err(format!(
                "{SUITE}: family {} has {} eligible items; at least {} are needed \
                 ({min} timed + {RESERVED} reserved)",
                family.name(),
                p.len(),
                min + RESERVED
            ));
        }
        Ok(p.len().min(cap))
    };

    let mut lines: BTreeMap<Family, Vec<Value>> = BTreeMap::new();
    let mut chosen_terms: BTreeMap<Family, HashSet<String>> = BTreeMap::new();
    for family in [Family::RareTerm, Family::DeepFetch] {
        let p = pool(family, &HashSet::new());
        let n = take(family, &p)?;
        pools.insert(family, p.len());
        let chosen = &p[..n];
        chosen_terms.insert(family, chosen.iter().map(|(_, t, _)| t.clone()).collect());
        lines.insert(
            family,
            chosen.iter().map(|(_, t, df)| json!([t, df])).collect(),
        );
    }
    let rac_pool = pool(
        Family::RareTermAndCategory,
        &chosen_terms[&Family::RareTerm],
    );
    let rac_n = take(Family::RareTermAndCategory, &rac_pool)?;
    pools.insert(Family::RareTermAndCategory, rac_pool.len());
    chosen_terms.insert(
        Family::RareTermAndCategory,
        rac_pool[..rac_n]
            .iter()
            .map(|(_, t, _)| t.clone())
            .collect(),
    );
    let overlap = |a: Family, b: Family| chosen_terms[&a].intersection(&chosen_terms[&b]).count();
    let term_overlap = json!({
        "rare-term & rare-term-and-category": overlap(Family::RareTerm, Family::RareTermAndCategory),
        "deep-fetch & rare-term": overlap(Family::DeepFetch, Family::RareTerm),
        "deep-fetch & rare-term-and-category": overlap(Family::DeepFetch, Family::RareTermAndCategory),
    });
    let rac_terms: Vec<(String, u64)> = rac_pool[..rac_n]
        .iter()
        .map(|(_, t, df)| (t.clone(), *df))
        .collect();
    let rac_index: HashMap<&str, usize> = rac_terms
        .iter()
        .enumerate()
        .map(|(i, (t, _))| (t.as_str(), i))
        .collect();
    let pair_pool = pool(Family::TwoRareTerms, &HashSet::new());
    // rare-terms-or (and its category variant): every eligible term of
    // RARE_OR_DF, with its postings collected in pass 2.
    let rare_or_pool = pool(Family::RareTermsOr, &HashSet::new());
    let rare_or_index: HashMap<&str, u32> = rare_or_pool
        .iter()
        .enumerate()
        .map(|(i, (_, t, _))| (t.as_str(), i as u32))
        .collect();
    // Pair-pool term ids in lexicographic order, so an id-sorted pair is a
    // term-sorted pair.
    let mut pair_terms: Vec<(&str, u64)> =
        pair_pool.iter().map(|(k, t, _)| (t.as_str(), *k)).collect();
    pair_terms.sort_unstable();
    let pair_index: HashMap<&str, u32> = pair_terms
        .iter()
        .enumerate()
        .map(|(i, (t, _))| (*t, i as u32))
        .collect();

    // Pass 2: postings of the rare-term-and-category terms, pairs, and the
    // URL and docid selection.
    let mut docids = Smallest::<(u64, String)>::new(cap);
    let mut urls = Smallest::<(u64, String)>::new(cap);
    let mut url_keys = HashSet::<u64>::new();
    let mut docid_rows = 0u64;
    // Every docid-eq eligible docid, in row order, for docid-batch: one
    // arena string and end offsets (the 10M corpus holds ~10M of them).
    let mut batch_arena = String::new();
    let mut batch_ends: Vec<usize> = Vec::new();
    let mut urls_colliding = HashSet::<String>::new();
    let mut docids_colliding = HashSet::<String>::new();
    let mut url_rows_colliding = 0u64;
    let mut docid_rows_colliding = 0u64;
    let mut rac_cats: Vec<Vec<String>> = vec![Vec::new(); rac_terms.len()];
    let mut rac_best: Vec<Option<(u64, String)>> = vec![None; rac_terms.len()];
    let mut pairs = Smallest::<(u64, u32, u32)>::new(cap);
    let mut ids: Vec<u32> = Vec::new();
    // Row ids of each rare-terms-or pool term's passages, and per row (only
    // rows holding one) its interned category and seeded docid key.
    let mut rare_or_postings: Vec<Vec<u32>> = vec![Vec::new(); rare_or_pool.len()];
    let mut row_cat: Vec<u32> = vec![u32::MAX; passages as usize];
    let mut row_key: Vec<u64> = vec![0; passages as usize];
    let mut categories: Vec<String> = Vec::new();
    let mut category_ids: HashMap<String, u32> = HashMap::new();
    // Passages per url-batch eligible URL (url-eq's eligibility).
    let mut url_counts: HashMap<String, u64> = HashMap::new();
    let (bytes2, sha2) = for_each_row(&payloads, |row, s| {
        ids.clear();
        let row32 = u32::try_from(row).map_err(|_| "too many passages")?;
        let mut rare_or_row = false;
        for (t, yes) in &s.terms {
            if !yes {
                continue;
            }
            if let Some(&i) = rare_or_index.get(t.as_str()) {
                rare_or_postings[i as usize].push(row32);
                rare_or_row = true;
            }
            if let Some(&i) = rac_index.get(t.as_str()) {
                rac_cats[i].push(s.category.clone());
                let k = rank_key(seed, "rare-term-and-category/passage", &s.docid);
                if rac_best[i].as_ref().is_none_or(|(b, _)| k < *b) {
                    rac_best[i] = Some((k, s.category.clone()));
                }
            }
            if let Some(&i) = pair_index.get(t.as_str()) {
                ids.push(i);
            }
        }
        ids.sort_unstable();
        for (x, &a) in ids.iter().enumerate() {
            for &b in &ids[x + 1..] {
                let k = pair_key(pair_terms[a as usize].1, pair_terms[b as usize].1);
                pairs.add((k, a, b));
            }
        }
        if rare_or_row {
            let c = match category_ids.get(&s.category) {
                Some(&c) => c,
                None => {
                    let c = categories.len() as u32;
                    categories.push(s.category.clone());
                    category_ids.insert(s.category.clone(), c);
                    c
                }
            };
            row_cat[row] = c;
            row_key[row] = rank_key(seed, "rare-terms-or-and-category/passage", &s.docid);
        }
        if is_safe_tag(&s.docid) {
            if docid_pieces.contains(&s.docid) {
                docid_rows_colliding += 1;
                docids_colliding.insert(s.docid);
            } else {
                docid_rows += 1;
                batch_arena.push_str(&s.docid);
                batch_ends.push(batch_arena.len());
                docids.add((rank_key(seed, "docid-eq", &s.docid), s.docid));
            }
        }
        if is_safe_tag(&s.url) {
            if url_pieces.contains(&s.url) {
                url_rows_colliding += 1;
                urls_colliding.insert(s.url);
            } else {
                *url_counts.entry(s.url.clone()).or_default() += 1;
                let k = rank_key(seed, "url-eq", &s.url);
                url_keys.insert(k);
                urls.add((k, s.url));
            }
        }
        Ok(())
    })?;
    if (bytes2, &sha2) != (payload_bytes, &payload_sha256) {
        return Err(format!(
            "{SUITE}: payloads.jsonl changed between the two passes"
        ));
    }
    let mut rac_lines = Vec::with_capacity(rac_terms.len());
    for (i, (t, df)) in rac_terms.iter().enumerate() {
        if rac_cats[i].len() as u64 != *df {
            return Err(format!(
                "{SUITE}: term {t:?} matched {} passages in pass 2 but {df} in pass 1",
                rac_cats[i].len()
            ));
        }
        let (_, cat) = rac_best[i].as_ref().ok_or("rare term without a passage")?;
        if !is_safe_tag(cat) {
            return Err(format!("{SUITE}: category {cat:?} is not a safe tag value"));
        }
        let n = rac_cats[i].iter().filter(|c| *c == cat).count();
        rac_lines.push(json!([t, cat, n]));
    }
    lines.insert(Family::RareTermAndCategory, rac_lines);
    let pairs = pairs.into_sorted();
    if pairs.len() < min + RESERVED {
        return Err(format!(
            "{SUITE}: only {} co-occurring term pairs in the {PAIR_DF:?} band",
            pairs.len()
        ));
    }
    pools.insert(Family::TwoRareTerms, pairs.len());
    lines.insert(
        Family::TwoRareTerms,
        pairs
            .iter()
            .map(|((_, a, b), n)| json!([pair_terms[*a as usize].0, pair_terms[*b as usize].0, n]))
            .collect(),
    );
    for (family, sel, distinct) in [
        (Family::UrlEq, urls, url_keys.len()),
        (Family::DocidEq, docids, docid_rows as usize),
    ] {
        let sel = sel.into_sorted();
        if sel.len() < min + RESERVED {
            return Err(format!(
                "{SUITE}: family {} has {} eligible values; at least {} are needed",
                family.name(),
                sel.len(),
                min + RESERVED
            ));
        }
        pools.insert(family, distinct);
        lines.insert(
            family,
            sel.into_iter().map(|((_, v), n)| json!([v, n])).collect(),
        );
    }

    let batch_pool = DocidPool::new(batch_arena, batch_ends)?;
    if batch_pool.len() < DOCID_BATCH_K.max(min + RESERVED) {
        return Err(format!(
            "{SUITE}: family {} has {} eligible docids; at least {} are needed",
            Family::DocidBatch.name(),
            batch_pool.len(),
            DOCID_BATCH_K.max(min + RESERVED)
        ));
    }
    for family in [
        Family::DocidBatch,
        Family::DocidBatch500,
        Family::DocidBatch1000,
    ] {
        pools.insert(family, batch_pool.len());
    }

    // url-batch: every url-eq eligible URL, by value (a deterministic pool
    // order; each epoch re-ranks it by its seeded key). URLs are distinct in a
    // line and every passage has one URL, so a line's count is the sum of
    // its URLs' counts: the url-eq eligibility makes each of those exact on
    // both engines (no other value's Redis tag piece equals a selected URL).
    let mut url_pool: Vec<(String, u64)> = url_counts.into_iter().collect();
    url_pool.sort_unstable();
    let floor = |family: Family, have: usize| -> Result<(), String> {
        let need = family
            .batch_size()
            .expect("a batch family")
            .max(min + RESERVED);
        if have < need {
            return Err(format!(
                "{SUITE}: family {} has {have} eligible items; at least {need} are needed",
                family.name()
            ));
        }
        Ok(())
    };
    floor(Family::UrlBatch, url_pool.len())?;
    pools.insert(Family::UrlBatch, url_pool.len());
    let url_batches = Batches::new(
        Family::UrlBatch,
        url_pool.len(),
        |i| url_pool[i as usize].0.as_str(),
        seed,
        tag_batch_timed_lines(URL_BATCH_N, opts.per_family, min) + RESERVED,
    )?;
    let url_line_counts: Vec<u64> = url_batches
        .lines()
        .map(|l| l.iter().map(|&i| url_pool[i as usize].1).sum())
        .collect();

    // rare-terms-or: the exact union of its terms' postings; every term has
    // zero ambiguous passages, so each engine matches exactly the passages
    // classified Yes for some term of the line, each passage once.
    for (i, (_, t, df)) in rare_or_pool.iter().enumerate() {
        if rare_or_postings[i].len() as u64 != *df {
            return Err(format!(
                "{SUITE}: term {t:?} matched {} passages in pass 2 but {df} in pass 1",
                rare_or_postings[i].len()
            ));
        }
    }
    for family in [Family::RareTermsOr, Family::RareTermsOrAndCategory] {
        floor(family, rare_or_pool.len())?;
        pools.insert(family, rare_or_pool.len());
    }
    let postings = |line: &[u32]| -> Vec<&[u32]> {
        line.iter()
            .map(|&i| rare_or_postings[i as usize].as_slice())
            .collect()
    };
    let df_sum = |line: &[u32]| -> u64 { line.iter().map(|&i| rare_or_pool[i as usize].2).sum() };
    let rare_or_batches = Batches::new(
        Family::RareTermsOr,
        rare_or_pool.len(),
        |i| rare_or_pool[i as usize].1.as_str(),
        seed,
        cap,
    )?;
    let rare_or_lines: Vec<&[u32]> = rare_or_batches.lines().collect();
    let rare_or_counts: Vec<u64> = rare_or_lines
        .par_iter()
        .map(|l| union_rows(&postings(l)).len() as u64)
        .collect();
    // rare-terms-or-and-category: the union AND the category of its passage
    // with the smallest seeded docid key (so the count is at least 1).
    let rac_or_batches = Batches::new(
        Family::RareTermsOrAndCategory,
        rare_or_pool.len(),
        |i| rare_or_pool[i as usize].1.as_str(),
        seed,
        cap,
    )?;
    let rac_or_lines: Vec<&[u32]> = rac_or_batches.lines().collect();
    let rac_or: Vec<(u32, u64)> = rac_or_lines
        .par_iter()
        .map(|l| {
            let rows = union_rows(&postings(l));
            let best = rows
                .iter()
                .min_by_key(|&&r| (row_key[r as usize], r))
                .expect("every term has a passage");
            let c = row_cat[*best as usize];
            let n = rows.iter().filter(|&&r| row_cat[r as usize] == c).count() as u64;
            (c, n)
        })
        .collect();
    for c in &categories {
        if !is_safe_tag(c) && rac_or.iter().any(|(i, _)| categories[*i as usize] == *c) {
            return Err(format!("{SUITE}: category {c:?} is not a safe tag value"));
        }
    }
    let rac_or_counts: Vec<u64> = rac_or.iter().map(|(_, n)| *n).collect();
    let overlap_lines = |lines: &[&[u32]], counts: &[u64]| {
        lines
            .iter()
            .zip(counts)
            .filter(|(l, n)| df_sum(l) > **n)
            .count()
    };
    let rare_or_overlap = overlap_lines(&rare_or_lines, &rare_or_counts);
    let mean = |v: &[u64]| v.iter().sum::<u64>() as f64 / v.len().max(1) as f64;
    let url_pool_mean = mean(&url_pool.iter().map(|(_, n)| *n).collect::<Vec<_>>());
    let rare_or_pool_mean = mean(
        &rare_or_pool
            .iter()
            .map(|(_, _, df)| *df)
            .collect::<Vec<_>>(),
    );
    let url_redis_max = url_batches
        .lines()
        .map(|l| redis_tag_union_len("url", l.iter().map(|&i| url_pool[i as usize].0.as_str())))
        .max()
        .unwrap_or(0);
    let or_rule = |family: Family, extra: &str| {
        format!(
            "the pool is every eligible segment term of df {RARE_OR_DF:?} (zero ambiguous \
             passages, not in any over-long run, not named by the Redis escape rule: the \
             deep-fetch eligibility); epoch e ranks the pool by SHA-256(key domain \
             \"vdb-msmarco-disk/v1\", seed, \"{}/epoch-<e>\", term) and cuts it into \
             consecutive batches of {RARE_OR_N} (the last len % {RARE_OR_N} unused); a line is \
             one batch in key order; its count is the number of DISTINCT passages holding any \
             of its terms{extra}",
            family.name()
        )
    };

    // Write the files.
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut families = serde_json::Map::new();
    for family in Family::ALL {
        let path = out.join(family.file_name());
        let mut w = std::io::BufWriter::new(
            std::fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?,
        );
        let mut counts = Vec::new();
        let mut batch = Value::Null;
        let err = |e: String| format!("{}: {e}", path.display());
        if family.is_docid_batch() {
            let k = family.batch_size().expect("a docid batch");
            let lines = tag_batch_timed_lines(k, opts.per_family, min) + RESERVED;
            batch = write_docid_batch(&mut w, family, &batch_pool, seed, lines, &mut counts)
                .map_err(err)?;
        } else if family == Family::UrlBatch {
            write_batches(
                &mut w,
                family,
                &url_batches,
                |i, l| {
                    let urls: Vec<&str> =
                        l.iter().map(|&u| url_pool[u as usize].0.as_str()).collect();
                    json!([urls, url_line_counts[i]])
                },
                &url_line_counts,
                &mut counts,
            )
            .map_err(err)?;
            batch = url_batches.record(
                url_pool.len(),
                format!(
                    "the pool is every url-eq eligible URL; epoch e ranks the pool by \
                     SHA-256(key domain \"vdb-msmarco-disk/v1\", seed, \"url-batch/epoch-<e>\", url) \
                     and cuts it into consecutive batches of {URL_BATCH_N} (the last len % \
                     {URL_BATCH_N} unused); a line is one batch in key order; its count is the \
                     sum of its URLs' passage counts (distinct URLs, one URL per passage)"
                ),
            );
            batch["pool_mean_count"] = json!(url_pool_mean);
            batch["timed_lines_rule"] = json!(format!(
                "per_family x {DOCID_BATCH_LINE_BUDGET_K} / {URL_BATCH_N} timed lines (at least \
                 the minimum): about as many tag values per file as docid-batch"
            ));
            batch["redis_query_bytes_max"] = json!(url_redis_max);
            batch["opensearch_terms_count"] = json!(URL_BATCH_N);
            batch["opensearch_max_terms_count"] = json!(OPENSEARCH_MAX_TERMS_COUNT);
        } else if family == Family::RareTermsOr {
            write_batches(
                &mut w,
                family,
                &rare_or_batches,
                |i, l| {
                    let terms: Vec<&str> = l
                        .iter()
                        .map(|&t| rare_or_pool[t as usize].1.as_str())
                        .collect();
                    json!([terms, rare_or_counts[i]])
                },
                &rare_or_counts,
                &mut counts,
            )
            .map_err(err)?;
            batch = rare_or_batches.record(rare_or_pool.len(), or_rule(family, ""));
            batch["df_band"] = json!(RARE_OR_DF);
            batch["pool_mean_df"] = json!(rare_or_pool_mean);
            batch["lines_with_overlapping_terms"] = json!(rare_or_overlap);
            batch["opensearch_bool_should_clauses"] = json!(RARE_OR_N);
            batch["opensearch_max_clause_count"] = json!(OPENSEARCH_MAX_CLAUSE_COUNT);
        } else if family == Family::RareTermsOrAndCategory {
            write_batches(
                &mut w,
                family,
                &rac_or_batches,
                |i, l| {
                    let terms: Vec<&str> = l
                        .iter()
                        .map(|&t| rare_or_pool[t as usize].1.as_str())
                        .collect();
                    json!([terms, categories[rac_or[i].0 as usize], rac_or[i].1])
                },
                &rac_or_counts,
                &mut counts,
            )
            .map_err(err)?;
            batch = rac_or_batches.record(
                rare_or_pool.len(),
                or_rule(
                    family,
                    " whose category is the category of the union's passage with the smallest \
                     SHA-256(key domain, seed, \"rare-terms-or-and-category/passage\", docid)",
                ),
            );
            batch["df_band"] = json!(RARE_OR_DF);
            batch["categories"] = json!(categories.len());
            batch["opensearch_bool_should_clauses"] = json!(RARE_OR_N);
            batch["opensearch_max_clause_count"] = json!(OPENSEARCH_MAX_CLAUSE_COUNT);
        } else {
            let l = probe_layout(family, lines.remove(&family).expect("every family"))?;
            counts.reserve(l.len());
            for line in &l {
                let (_, n) = family.conditions(line)?;
                counts.push(n);
                writeln!(w, "{line}").map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
        w.flush().map_err(|e| format!("{}: {e}", path.display()))?;
        drop(w);
        let (sha, _) = sha256_file(&path)?;
        let mut entry = json!({
                "file": family.file_name(),
                "sha256": sha,
                "lines": counts.len(),
                "timed_queries": counts.len() - RESERVED,
                "df_band": family.df_band().map(|(a, b)| json!([a, b])),
                "fields": family.fields(),
                "eligible_pool": pools[&family],
                "terms_excluded_redis_escape": family.df_band().map(escape_excluded),
                "expected_count": stats(&counts),
                "expected_count_timed": stats(&counts[RESERVED..]),
                "expected_count_probe_reserve": stats(&counts[1..RESERVED]),
        });
        if !batch.is_null() {
            entry["batch"] = batch;
        }
        families.insert(family.name().into(), entry);
    }
    let manifest = json!({
        "suite": SUITE,
        "generator": GENERATOR,
        "seed": seed,
        "per_family_requested": opts.per_family,
        "per_family_minimum": min,
        "preparation_sha256": preparation_sha256,
        "dataset": prepared["dataset"],
        "payloads": {"rows": rows, "bytes": payload_bytes, "sha256": payload_sha256},
        "text_field": TEXT_FIELD,
        "text_match_rules": text_match::RULE_VERSION,
        "redis_escape_rules": text_match::REDIS_ESCAPE_RULE,
        "analysis": "parity: segment text with every whitespace other than U+0020 replaced by a space (as uploaded), no stemming",
        "df_bands": {"rare": RARE_DF, "pair": PAIR_DF, "deep": DEEP_DF, "rare_or": RARE_OR_DF},
        "high_pressure": {
            "docid_batch_500_k": DOCID_BATCH_500_K,
            "docid_batch_1000_k": DOCID_BATCH_1000_K,
            "docid_batch_line_budget_k": DOCID_BATCH_LINE_BUDGET_K,
            "url_batch_n": URL_BATCH_N,
            "rare_or_n": RARE_OR_N,
            "rare_or_df": RARE_OR_DF,
            "top": HIGH_PRESSURE_TOP,
            "total_hits": "exact (none is refused for every batch family)",
            "opensearch_limits": {"max_result_window": OPENSEARCH_MAX_RESULT_WINDOW,
                                  "max_terms_count": OPENSEARCH_MAX_TERMS_COUNT,
                                  "max_clause_count": OPENSEARCH_MAX_CLAUSE_COUNT},
        },
        "layout": {"prime_line": 0, "probe_reserve_lines": [1, PROBE_RESERVE], "timed_from_line": RESERVED,
                   "rule": "line 0: the first selected item in key order (the prime); lines 1..=64: the 64 other selected items with the highest expected count, ties in key order, written in key order (the WITHOUTCOUNT probe reserve; every reserve count >= every timed count); then the rest in key order (the timed pool)"},
        "vocabulary": {
            "eligible_terms_by_df": df_profile,
            "terms": vocabulary.len(),
            "terms_with_ambiguous_passages": ambiguous_terms,
            "long_runs": long_runs.len(),
            "terms_in_long_runs": poison.len(),
            "passages_with_redis_escape": escape_passages,
            "terms_excluded_redis_escape": escape_excluded((1, u64::MAX)),
            "redis_escape_rule": "a term is not selected if, in any passage, it is a lowercase ASCII-alphanumeric run (raw or ASCII-folded) of a RediSearch toksep piece (cut at unescaped ' \\t,./(){}[]:;~!@#$%^&*-=+|'`\"<>?') that holds a backslash or a control byte, or of that piece as RediSearch normalizes it (escape backslashes and unescaped blanks removed, control bytes removed, lowercased), or of any text after a NUL: RediSearch escapes the byte after a backslash into the token (\\ term is the token ' term', plant\\n the token 'plantn'), which the text_match classifier does not model",
        },
        "tags": {"urls_not_safe": urls_unsafe, "docids_not_safe": docids_unsafe,
                 "safe": "printable ASCII 0x21-0x7E, no ';', at most 2048 bytes",
                 "urls_excluded_redis_piece_collision": urls_colliding.len(),
                 "url_rows_excluded_redis_piece_collision": url_rows_colliding,
                 "docids_excluded_redis_piece_collision": docids_colliding.len(),
                 "docid_rows_excluded_redis_piece_collision": docid_rows_colliding,
                 "redis_piece_rule": "a safe value equal to a Redis tag piece (split on ';', C-isspace trimmed, non-empty) of an unsafe value is not selected: Redis TAG SEPARATOR ; would match the unsafe value's passages too, an OpenSearch keyword would not"},
        "term_overlap": term_overlap,
        "selection": "per family: eligible items ranked by SHA-256(key domain \"vdb-msmarco-disk/v1\", seed, family, item); the smallest kept in key order. rare-term-and-category terms are disjoint from rare-term terms; deep-fetch is drawn from its own band independently (overlaps recorded in term_overlap). Terms: zero ambiguous passages, not in any over-long run, not named by the Redis escape rule in any passage. rare-term-and-category: the category of the term's passage with the smallest seeded docid key. two-rare-terms: co-occurring pairs (one passage has both) ranked by a SplitMix64 mix of the two term keys. docid-batch, docid-batch-500, docid-batch-1000, url-batch, rare-terms-or, rare-terms-or-and-category: see families.<family>.batch.rule (rare-terms-or* draw from the deep-fetch eligible terms, re-ranked per epoch under their own family name).",
        "families": families,
    });
    let text = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    std::fs::write(out.join(MANIFEST), text + "\n")
        .map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(manifest)
}

/// One timed window of one family, loaded and verified.
#[derive(Debug, Clone)]
pub struct Window {
    pub family: Family,
    /// `[prime, probe reserve…, timed…]`, each with `query_id` = file line.
    pub queries: Vec<(usize, Query)>,
    /// Index into `queries` of the first timed query.
    pub timed_from: usize,
    pub offset: u64,
    pub manifest: Value,
    pub manifest_sha256: String,
}

impl Window {
    /// Record of the window for the result metadata.
    pub fn record(&self) -> Value {
        let timed = &self.queries[self.timed_from..];
        json!({
            "suite": SUITE,
            "family": self.family.name(),
            "generator": self.manifest["generator"],
            "seed": self.manifest["seed"],
            "manifest_sha256": self.manifest_sha256,
            "family_file_sha256": self.manifest["families"][self.family.name()]["sha256"],
            "df_band": self.family.df_band().map(|(a, b)| json!([a, b])),
            "eligible_pool": self.manifest["families"][self.family.name()]["eligible_pool"],
            "timed_pool": self.manifest["families"][self.family.name()]["timed_queries"],
            "offset": self.offset,
            "timed_queries": timed.len(),
            "first_line": timed.first().map(|q| q.0),
            "last_line": timed.last().map(|q| q.0),
            "prime_line": 0,
            "probe_reserve_lines": [1, PROBE_RESERVE],
            "payloads": self.manifest["payloads"],
        })
    }
}

fn read_manifest(dir: &Path) -> Result<(Value, String), String> {
    let path = dir.join(DIR).join(MANIFEST);
    let bytes = std::fs::read(&path).map_err(|e| {
        format!(
            "{SUITE}: {} is missing ({e}); write it with `prepare-msmarco --dataset <name> \
             --add-disk-queries`",
            path.display()
        )
    })?;
    let manifest: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((manifest, sha256_hex(&bytes)))
}

/// Check the manifest belongs to this corpus and this generator.
pub fn check_manifest(dir: &Path, manifest: &Value) -> Result<(), String> {
    if manifest["suite"] != SUITE || manifest["generator"] != GENERATOR {
        return Err(format!(
            "{SUITE}: {} was written by generator {} (suite {}); this build needs {GENERATOR} \
             — regenerate with --add-disk-queries",
            dir.join(DIR).join(MANIFEST).display(),
            manifest["generator"],
            manifest["suite"]
        ));
    }
    let prep = preparation_sha256(dir)?;
    if manifest["preparation_sha256"] != prep.as_str() {
        return Err(format!(
            "{SUITE}: the query files were computed from preparation {} but PREPARED.json is \
             {prep}; regenerate with --add-disk-queries",
            manifest["preparation_sha256"]
        ));
    }
    let payloads = dir.join("payloads.jsonl");
    let size = std::fs::metadata(&payloads)
        .map_err(|e| format!("{}: {e}", payloads.display()))?
        .len();
    if manifest["payloads"]["bytes"].as_u64() != Some(size) {
        return Err(format!(
            "{SUITE}: payloads.jsonl is {size} bytes but the query files were computed from {} \
             bytes; regenerate with --add-disk-queries",
            manifest["payloads"]["bytes"]
        ));
    }
    Ok(())
}

/// Load the window `[offset, offset + count)` of `family`'s timed pool, plus
/// the priming query and the probe reserve. `count` `None` = the rest of the
/// pool. A window past the end of the pool is an error: the loader never
/// cycles (duration mode cycles the loaded window at dispatch,
/// [`crate::duration_mode`]).
pub fn load_window(
    dir: &Path,
    family: Family,
    offset: u64,
    count: Option<usize>,
) -> Result<Window, String> {
    let (manifest, manifest_sha256) = read_manifest(dir)?;
    check_manifest(dir, &manifest)?;
    let entry = &manifest["families"][family.name()];
    let path = dir.join(DIR).join(family.file_name());
    let (sha, _) = sha256_file(&path)?;
    if entry["sha256"] != sha.as_str() {
        return Err(format!(
            "{SUITE}: {} does not match its manifest SHA-256; regenerate with \
             --add-disk-queries",
            path.display()
        ));
    }
    let lines = entry["lines"]
        .as_u64()
        .ok_or("disk manifest lacks a line count")? as usize;
    let pool = lines
        .checked_sub(RESERVED)
        .ok_or("disk family file is shorter than its reserved lines")?;
    let offset_us = usize::try_from(offset).map_err(|_| "offset out of range")?;
    let count = match count {
        Some(n) => n,
        None => pool.saturating_sub(offset_us),
    };
    if count == 0 || offset_us.saturating_add(count) > pool {
        return Err(format!(
            "{SUITE} family {}: the window needs {count} queries from offset {offset}, but the \
             timed pool holds {pool}; queries are never repeated within a run — use a smaller \
             --queries or another offset, or regenerate with a larger \
             --disk-queries-per-family (or keep a window that fits and run it for a time, \
             cycling it, with `compound.duration_s`)",
            family.name()
        ));
    }
    let first = RESERVED + offset_us;
    let last = first + count;
    let file = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut queries = Vec::with_capacity(RESERVED + count);
    let mut seen = 0usize;
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        seen += 1;
        if line_no >= last {
            continue;
        }
        if line_no >= RESERVED && line_no < first {
            continue;
        }
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        let v: Value =
            serde_json::from_str(&line).map_err(|e| format!("{}: {e}", path.display()))?;
        let (conditions, n) = family.conditions(&v)?;
        queries.push((
            line_no,
            Query {
                name: format!("{}-{line_no:07}", family.name()),
                family: family.name().into(),
                conditions,
                expected_count: Some(n),
                minimum_count: 1,
            },
        ));
    }
    if seen != lines {
        return Err(format!(
            "{SUITE}: {} has {seen} lines but its manifest says {lines}",
            path.display()
        ));
    }
    Ok(Window {
        family,
        queries,
        timed_from: RESERVED,
        offset,
        manifest,
        manifest_sha256,
    })
}

/// A small MS MARCO-shaped corpus for tests (unit and integration): `n`
/// passages with `docid`, `url`, `title`, `segment`, `category` and
/// `doc_len`, unit vectors of `dim`, a `tests.jsonl` and a `PREPARED.json`.
/// The vocabulary is built so every family has eligible items at small
/// minimums, and so every exclusion rule has something to exclude: unique
/// terms (df 1), `mid*` terms (df n/150, the deep band at n = 3000), `pair*`
/// terms (df ≈ n/10), a term ambiguous in some passages (`edge`), a term
/// that only reaches df through newline normalization (`newl`), a term
/// inside an over-long run, a folding collision (`cafe`), terms a RediSearch
/// backslash escape changes (`malingered` after `\ `, `plantn` from
/// `plant\n`, both in the deep band) and unsafe URLs.
/// Every value passes the upload validator (`filter_only::domain`) even with
/// `url` and `docid` declared `keyword` and no prepared filter-only query set
/// (as in the integration tests), so both engines can ingest it.
pub fn write_fixture_corpus(dir: &Path, n: usize, dim: usize) -> Result<(), String> {
    write_fixture(dir, n, dim, false)
}

/// [`write_fixture_corpus`] plus unsafe URLs and docids whose Redis tag
/// pieces equal a safe value of another passage (`<url>;jsessionid=<i>`,
/// `<url> `, `<docid> `), so the generator's tag-piece exclusion has
/// something to exclude. With `url`/`docid` in the filter-only index schema
/// the upload validator refuses such values (no `;`, no edge whitespace), so
/// this corpus is for generator tests only. A real MS MARCO corpus is not
/// protected that way: its `filter_queries.jsonl` narrows the validated schema
/// to the derived fields, so its `;` / edge-whitespace URLs upload, and the
/// exclusion is what keeps `url-eq` / `docid-eq` counts exact on Redis.
pub fn write_fixture_corpus_with_tag_piece_collisions(
    dir: &Path,
    n: usize,
    dim: usize,
) -> Result<(), String> {
    write_fixture(dir, n, dim, true)
}

fn write_fixture(dir: &Path, n: usize, dim: usize, collisions: bool) -> Result<(), String> {
    use rand::{Rng, SeedableRng};
    fn letters(mut i: usize) -> String {
        let mut s = String::new();
        loop {
            s.push((b'a' + (i % 26) as u8) as char);
            i /= 26;
            if i == 0 {
                return s;
            }
        }
    }
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xD15C);
    let mut payloads = String::new();
    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        let a = i % 20;
        let b = (a + 1 + (i / 20) % 19) % 20;
        let mut words = vec![
            format!("uniq{}", letters(i)),
            format!("mid{}", letters(i % 150)),
            format!("pair{}", letters(a)),
            format!("pair{}", letters(b)),
        ];
        match i % 50 {
            0 => words.push("edge.Case".into()),
            1 => words.push("edge".into()),
            _ => {}
        }
        if i % 30 == 5 {
            words.push("\nnewl".into());
        }
        if i == 7 {
            words.push(format!("{}uniq{}", "a".repeat(260), letters(8)));
        }
        match i {
            9 => words.push("caf\u{e9}".into()),
            10 => words.push("cafe".into()),
            _ => {}
        }
        // RediSearch escapes (both seen live on MS MARCO): `\ ` joins the
        // escaped space onto `malingered`, so Redis has one passage fewer
        // than the classifier; `plant\n` is the Redis token `plantn`, one
        // passage more.
        match i % 100 {
            21 => words.push("malingered".into()),
            23 => words.push("plantn".into()),
            _ => {}
        }
        match i {
            22 => words.push("(audio) \\ malingered;".into()),
            24 => words.push("plant\\n".into()),
            _ => {}
        }
        let url = match i % 100 {
            3 => format!("http://bad site/{i}"),
            4 => format!("http://x/caf\u{e9}/{i}"),
            // Redis tags `https://example.org/<k>` (k % 100 == 15 / 16, a
            // safe url of other passages) for these two unsafe values.
            5 if collisions => format!("https://example.org/{};jsessionid={i}", i % 400 + 10),
            6 if collisions => format!("https://example.org/{} ", i % 400 + 10),
            _ => format!("https://example.org/{}", i % 400),
        };
        // Redis trims this unsafe docid to the docid of the next passage.
        let docid = if collisions && i % 1000 == 7 {
            format!("fix_doc_{}#0_{} ", i + 1, i + 1)
        } else {
            format!("fix_doc_{i}#0_{i}")
        };
        payloads.push_str(
            &json!({
                "docid": docid,
                "url": url,
                "title": format!("title {}", letters(i % 10)),
                "segment": words.join(" ").replace(" \n", "\n"),
                "category": format!("cat{:02}", i % 7),
                "doc_len": i,
            })
            .to_string(),
        );
        payloads.push('\n');
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        super::normalize_in_place(&mut v);
        vectors.push(v);
    }
    std::fs::write(dir.join("payloads.jsonl"), payloads).map_err(|e| e.to_string())?;
    crate::readers::write_npy_vectors(dir.join("vectors.npy").to_str().ok_or("path")?, &vectors)?;
    let tests: String = (0..4)
        .map(|q| {
            format!(
                "{}\n",
                json!({"query": vectors[q], "closest_ids": [q], "query_text": "fixture"})
            )
        })
        .collect();
    std::fs::write(dir.join("tests.jsonl"), tests).map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("PREPARED.json"),
        json!({"dataset": "disk-fixture", "passages": n}).to_string(),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(n: usize) -> (tempfile::TempDir, std::path::PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("corpus");
        write_fixture_corpus_with_tag_piece_collisions(&dir, n, 4).unwrap();
        (tmp, dir)
    }

    /// The default fixture is what the integration tests upload: it must pass
    /// the upload validator with `url`/`docid` as keywords (the schema of
    /// `tests/common` `write_disk_project`); the collision fixture must not.
    #[test]
    fn default_fixture_passes_the_upload_validator_collision_fixture_does_not() {
        let schema = serde_json::json!({"docid":"keyword","url":"keyword","title":"text",
            "segment":"text","category":"keyword","doc_len":"int"});
        let check = |dir: &Path| {
            crate::readers::visit_compound_metadata(dir, |item| {
                crate::filter_only::domain::validate_metadata(
                    std::slice::from_ref(&item),
                    Some(&schema),
                )
            })
        };
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("corpus");
        write_fixture_corpus(&dir, 3000, 4).unwrap();
        check(&dir).unwrap();
        let (_tmp, dir) = fixture(3000);
        assert!(check(&dir).is_err());
    }

    fn opts(per_family: usize, seed: u64) -> GenerateOptions {
        GenerateOptions {
            per_family,
            seed,
            min_per_family: 5,
        }
    }

    fn rows(dir: &Path) -> Vec<Value> {
        std::fs::read_to_string(dir.join("payloads.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn verdict(row: &Value, term: &str) -> TextMatch {
        text_match::classify(
            &crate::fulltext::normalize_whitespace(row["segment"].as_str().unwrap()),
            term,
        )
    }

    /// Brute-force oracle, independent of the generator's single-pass table:
    /// `classify` per (passage, term) on the normalized text.
    fn df(rows: &[Value], term: &str) -> (u64, u64) {
        rows.iter()
            .fold((0, 0), |(y, a), r| match verdict(r, term) {
                TextMatch::Yes => (y + 1, a),
                TextMatch::Ambiguous => (y, a + 1),
                TextMatch::No => (y, a),
            })
    }

    fn lines(dir: &Path, f: Family) -> Vec<Value> {
        std::fs::read_to_string(dir.join(f.file_name()))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn generated_counts_match_a_brute_force_oracle() {
        let (_tmp, dir) = fixture(3000);
        let out = dir.join(DIR);
        let m = generate(&dir, &opts(200, 7), &out).unwrap();
        let corpus = rows(&dir);
        // As `analysis: parity` uploads it.
        let uploaded: Vec<Value> = corpus
            .iter()
            .map(|r| {
                let mut r = r.clone();
                r["segment"] = json!(crate::fulltext::normalize_whitespace(
                    r["segment"].as_str().unwrap()
                ));
                r
            })
            .collect();
        // RediSearch's tokens of each uploaded passage (reference
        // transcription of its tokenizer, escapes included): every text
        // family's count must hold on Redis too, not only on the classifier.
        let redis: Vec<HashSet<String>> = uploaded
            .iter()
            .map(|r| text_match::redis_tokens(r["segment"].as_str().unwrap()))
            .collect();
        let redis_count = |terms: &[&str], category: Option<&str>| {
            uploaded
                .iter()
                .zip(&redis)
                .filter(|(r, toks)| {
                    terms.iter().all(|t| toks.contains(*t))
                        && category.is_none_or(|c| r["category"] == c)
                })
                .count() as u64
        };
        for family in Family::ALL {
            let l = lines(&out, family);
            assert_eq!(m["families"][family.name()]["lines"], l.len(), "{family:?}");
            for line in &l {
                let a = line.as_array().unwrap();
                let s = |i: usize| a[i].as_str().unwrap();
                let (_, count) = family.conditions(line).unwrap();
                let on_redis = match family {
                    Family::RareTerm | Family::DeepFetch => redis_count(&[s(0)], None),
                    Family::TwoRareTerms => redis_count(&[s(0), s(1)], None),
                    Family::RareTermAndCategory => redis_count(&[s(0)], Some(s(1))),
                    Family::UrlEq | Family::DocidEq => count,
                    // Redis `TAG SEPARATOR ;`: a passage matches when any
                    // of its tag pieces is one of the batch's docids.
                    Family::DocidBatch
                    | Family::DocidBatch500
                    | Family::DocidBatch1000
                    | Family::UrlBatch => {
                        let field = family.fields()[0];
                        let batch: HashSet<&str> = a[0]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_str().unwrap())
                            .collect();
                        corpus
                            .iter()
                            .filter(|r| {
                                redis_tag_pieces(r[field].as_str().unwrap())
                                    .any(|p| batch.contains(p))
                            })
                            .count() as u64
                    }
                    // A passage matches when its Redis tokens hold ANY term
                    // (each passage once).
                    Family::RareTermsOr | Family::RareTermsOrAndCategory => {
                        let terms: Vec<&str> = a[0]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_str().unwrap())
                            .collect();
                        let cat = (family == Family::RareTermsOrAndCategory).then(|| s(1));
                        uploaded
                            .iter()
                            .zip(&redis)
                            .filter(|(r, toks)| {
                                terms.iter().any(|t| toks.contains(*t))
                                    && cat.is_none_or(|c| r["category"] == c)
                            })
                            .count() as u64
                    }
                };
                assert_eq!(
                    on_redis, count,
                    "{family:?} {line}: RediSearch tokenizes differently"
                );
            }
            let mut distinct = HashSet::new();
            for (i, line) in l.iter().enumerate() {
                assert!(distinct.insert(line.to_string()), "duplicate {line}");
                // The v6 families' exact counts are proven over every line
                // above (Redis tokens / tag pieces) and in
                // `high_pressure_families_have_exact_union_counts`; the slow
                // classifier-per-row Expr oracle samples their lines. The v5
                // families (docid-batch included) keep every line.
                if family.fixed_fetch().is_some() && i >= 24 && i % 16 != 0 {
                    continue;
                }
                let (conditions, count) = family.conditions(line).unwrap();
                let expr = crate::compound_text::Expr::compile(
                    &conditions,
                    &json!({"segment":"text","category":"keyword","url":"keyword","docid":"keyword"}),
                )
                .unwrap();
                let mut n = 0;
                for r in &uploaded {
                    // The oracle refuses an ambiguous passage outright.
                    if expr.matches(r.as_object().unwrap()).unwrap() {
                        n += 1;
                    }
                }
                assert_eq!(n, count, "{family:?} {line}");
                if let Some((lo, hi)) = family.df_band() {
                    let a = line.as_array().unwrap();
                    let terms: Vec<&str> = match a[0].as_array() {
                        Some(batch) => batch.iter().map(|v| v.as_str().unwrap()).collect(),
                        None => a.iter().filter_map(Value::as_str).collect(),
                    };
                    for t in terms {
                        if t.starts_with("cat") {
                            continue;
                        }
                        let (yes, amb) = df(&corpus, t);
                        assert_eq!(amb, 0, "{t}");
                        assert!((lo..=hi).contains(&yes), "{family:?} {t} df {yes}");
                    }
                }
            }
        }
        // Pool sizes equal the brute-force eligible sets: a rule that excluded
        // too much (newline normalization lost) or too little (an ambiguous,
        // folded or long-run term kept) changes them.
        let vocab: HashSet<String> = corpus
            .iter()
            .flat_map(|r| {
                r["segment"]
                    .as_str()
                    .unwrap()
                    .split(|c: char| !c.is_ascii_alphabetic())
                    .map(str::to_ascii_lowercase)
                    .collect::<Vec<_>>()
            })
            .filter(|t| text_match::validate_term(t).is_ok())
            .collect();
        let eligible = |lo: u64, hi: u64| {
            vocab
                .iter()
                .filter(|t| {
                    let (yes, amb) = df(&corpus, t);
                    amb == 0 && (lo..=hi).contains(&yes) && redis_count(&[t], None) == yes
                })
                .count()
        };
        assert_eq!(
            m["families"]["rare-term"]["eligible_pool"],
            eligible(1, 100)
        );
        assert_eq!(
            m["families"]["deep-fetch"]["eligible_pool"],
            eligible(11, 100)
        );
        for (term, why) in [
            ("edge", "ambiguous passage"),
            ("cafe", "folding"),
            ("uniqi", "inside an over-long run"),
            ("malingered", "a Redis escape (`\\ malingered`)"),
            ("plantn", "a Redis escape (`plant\\n`)"),
        ] {
            assert!(
                !Family::ALL
                    .iter()
                    .any(|f| lines(&out, *f).iter().any(|l| l[0] == term)),
                "{term} kept despite {why}"
            );
        }
        assert_eq!(df(&corpus, "newl").1, 0, "newline normalization is applied");
        // The classifier alone would have kept both escape terms (v3), with
        // a count Redis does not return.
        assert_eq!(df(&corpus, "malingered"), (31, 0));
        assert_eq!(redis_count(&["malingered"], None), 30);
        assert_eq!(df(&corpus, "plantn"), (30, 0));
        assert_eq!(redis_count(&["plantn"], None), 31);
        assert_eq!(m["vocabulary"]["passages_with_redis_escape"], 2);
        assert_eq!(m["vocabulary"]["terms_excluded_redis_escape"], 2);
        assert_eq!(m["families"]["rare-term"]["terms_excluded_redis_escape"], 2);
        assert_eq!(
            m["families"]["deep-fetch"]["terms_excluded_redis_escape"],
            2
        );
        assert_eq!(
            m["families"]["two-rare-terms"]["terms_excluded_redis_escape"],
            0
        );
        assert_eq!(
            m["families"]["url-eq"]["terms_excluded_redis_escape"],
            Value::Null
        );
        // Single-term families: the prime is the first key, then the reserve
        // and the timed pool are each in seeded key order (uniform slices).
        for family in [
            Family::RareTerm,
            Family::DeepFetch,
            Family::DocidEq,
            Family::UrlEq,
        ] {
            let keys: Vec<u64> = lines(&out, family)
                .iter()
                .map(|l| rank_key(7, family.name(), l[0].as_str().unwrap()))
                .collect();
            assert!(keys[1..].iter().all(|k| *k > keys[0]), "{family:?}");
            for part in [&keys[1..RESERVED], &keys[RESERVED..]] {
                assert!(part.windows(2).all(|w| w[0] < w[1]), "{family:?}");
            }
        }
        // The probe reserve holds the highest counts: every reserve line
        // matches at least as many passages as every timed line, so whenever
        // a timed query is cut by the page every probe query is too.
        for family in Family::ALL {
            let counts: Vec<u64> = lines(&out, family)
                .iter()
                .map(|l| family.conditions(l).unwrap().1)
                .collect();
            let reserve_min = counts[1..RESERVED].iter().min().unwrap();
            let timed_max = counts[RESERVED..].iter().max().unwrap();
            assert!(reserve_min >= timed_max, "{family:?}");
            assert_eq!(
                m["families"][family.name()]["expected_count_probe_reserve"]["min"],
                *reserve_min,
                "{family:?}"
            );
        }
        // Unsafe URLs never become queries.
        for l in lines(&out, Family::UrlEq) {
            assert!(is_safe_tag(l[0].as_str().unwrap()), "{l}");
        }
        assert_eq!(m["tags"]["urls_not_safe"], 120);
        assert_eq!(m["tags"]["docids_not_safe"], 3);
        // The exact counts hold on Redis too (`TAG SEPARATOR ;`: a passage
        // matches when any of its tag pieces equals the value), not only on
        // an OpenSearch keyword: a safe value that is a piece of an unsafe
        // one (`<url>;jsessionid=…`, `<url> `, `<docid> `) is never selected.
        for (family, field) in [(Family::UrlEq, "url"), (Family::DocidEq, "docid")] {
            for l in lines(&out, family) {
                let v = l[0].as_str().unwrap();
                let redis = corpus
                    .iter()
                    .filter(|r| redis_tag_pieces(r[field].as_str().unwrap()).any(|p| p == v))
                    .count() as u64;
                assert_eq!(redis, l[1].as_u64().unwrap(), "{family:?} {l}");
            }
        }
        let colliding_urls: HashSet<String> = [15, 115, 215, 315, 16, 116, 216, 316]
            .iter()
            .map(|k| format!("https://example.org/{k}"))
            .collect();
        assert!(lines(&out, Family::UrlEq)
            .iter()
            .all(|l| !colliding_urls.contains(l[0].as_str().unwrap())));
        assert_eq!(m["tags"]["urls_excluded_redis_piece_collision"], 8);
        assert_eq!(m["tags"]["url_rows_excluded_redis_piece_collision"], 60);
        assert_eq!(m["tags"]["docids_excluded_redis_piece_collision"], 3);
        assert_eq!(m["tags"]["docid_rows_excluded_redis_piece_collision"], 3);
        assert_eq!(m["families"]["docid-eq"]["eligible_pool"], 3000 - 3 - 3);
        assert_eq!(m["families"]["docid-batch"]["eligible_pool"], 3000 - 3 - 3);
        // rare-term-and-category: the category of the term's passage with the
        // smallest seeded docid key (the documented rule, not just any one).
        for l in lines(&out, Family::RareTermAndCategory) {
            let t = l[0].as_str().unwrap();
            let best = corpus
                .iter()
                .filter(|r| verdict(r, t) == TextMatch::Yes)
                .min_by_key(|r| {
                    rank_key(
                        7,
                        "rare-term-and-category/passage",
                        r["docid"].as_str().unwrap(),
                    )
                })
                .unwrap();
            assert_eq!(l[1], best["category"], "{l}");
        }
        // rare-term-and-category terms are disjoint from rare-term terms.
        let rare: HashSet<Value> = lines(&out, Family::RareTerm)
            .iter()
            .map(|l| l[0].clone())
            .collect();
        assert!(lines(&out, Family::RareTermAndCategory)
            .iter()
            .all(|l| !rare.contains(&l[0])));
        // Deterministic for a seed, and the seed matters.
        let again = dir.join("again");
        let m2 = generate(&dir, &opts(200, 7), &again).unwrap();
        assert_eq!(m, m2);
        for family in Family::ALL {
            assert_eq!(
                std::fs::read(out.join(family.file_name())).unwrap(),
                std::fs::read(again.join(family.file_name())).unwrap()
            );
        }
        let other = dir.join("other");
        generate(&dir, &opts(200, 8), &other).unwrap();
        assert_ne!(
            std::fs::read(out.join("rare-term.jsonl")).unwrap(),
            std::fs::read(other.join("rare-term.jsonl")).unwrap()
        );
    }

    #[test]
    fn probe_layout_reserves_the_highest_counts_and_keeps_key_order() {
        // Key order: counts mostly 1 with a sparse tail, as url-eq and
        // rare-term-and-category have; the uniform v1 reserve (lines 1..=64)
        // held no count above 1 here.
        let n = RESERVED + 200;
        let items: Vec<Value> = (0..n)
            .map(|i| {
                let c = if i > RESERVED && i % 50 == 0 {
                    30 + i as u64
                } else {
                    1
                };
                json!([format!("u{i:04}"), c])
            })
            .collect();
        let laid = probe_layout(Family::UrlEq, items.clone()).unwrap();
        assert_eq!(laid.len(), n);
        assert_eq!(laid[0], items[0]);
        let count = |l: &Value| l[1].as_u64().unwrap();
        let timed_max = laid[RESERVED..].iter().map(count).max().unwrap();
        let reserve: Vec<&Value> = laid[1..RESERVED].iter().collect();
        assert!(reserve.iter().all(|l| count(l) >= timed_max));
        // Every truncating item is in the reserve; ties fill the rest of it
        // by key, and the reserve is written in key order.
        let over: Vec<&Value> = items[1..].iter().filter(|l| count(l) > 1).collect();
        assert_eq!(over.len(), 4);
        let mut want: Vec<&Value> = items[1..=PROBE_RESERVE - over.len()].iter().collect();
        want.extend(over);
        assert_eq!(reserve, want);
        // The timed pool keeps key order and holds every other item once.
        let names = |v: &[Value]| {
            v.iter()
                .map(|l| l[0].as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        let timed = names(&laid[RESERVED..]);
        let mut sorted = timed.clone();
        sorted.sort();
        assert_eq!(timed, sorted);
        let mut all = names(&laid);
        all.sort();
        assert_eq!(all, names(&items));
        assert!(probe_layout(Family::UrlEq, items[..RESERVED - 1].to_vec()).is_err());
    }

    #[test]
    fn a_family_below_its_floor_is_an_error_not_a_smaller_set() {
        let (_tmp, dir) = fixture(3000);
        let mut o = opts(200, 1);
        o.min_per_family = 100_000;
        o.per_family = 100_000;
        let e = generate(&dir, &o, &dir.join("x")).unwrap_err();
        assert!(e.contains("eligible"), "{e}");
        assert!(generate(&dir, &opts(4, 1), &dir.join("y")).is_err());
    }

    #[test]
    fn windows_never_cycle_and_check_corpus_identity() {
        let (_tmp, dir) = fixture(3000);
        generate(&dir, &opts(100, 1), &dir.join(DIR)).unwrap();
        let pool = lines(&dir.join(DIR), Family::RareTerm).len() - RESERVED;
        let w = load_window(&dir, Family::RareTerm, 10, Some(20)).unwrap();
        let ids: Vec<usize> = w.queries.iter().map(|q| q.0).collect();
        let want: Vec<usize> = (0..RESERVED).chain(RESERVED + 10..RESERVED + 30).collect();
        assert_eq!(ids, want);
        assert_eq!(w.timed_from, RESERVED);
        assert_eq!(w.record()["first_line"], RESERVED + 10);
        assert_eq!(w.record()["last_line"], RESERVED + 29);
        let rest = load_window(&dir, Family::RareTerm, 5, None).unwrap();
        assert_eq!(rest.queries.len() - RESERVED, pool - 5);
        for (offset, count) in [(0, pool + 1), (pool as u64 - 3, 4), (pool as u64, 1)] {
            let e = load_window(&dir, Family::RareTerm, offset, Some(count)).unwrap_err();
            assert!(e.contains("never repeated"), "{e}");
        }
        assert!(load_window(&dir, Family::RareTerm, pool as u64, None).is_err());
        // Corpus identity: PREPARED.json, payload size, the family file.
        let prepared = std::fs::read(dir.join("PREPARED.json")).unwrap();
        std::fs::write(
            dir.join("PREPARED.json"),
            b"{\"dataset\":\"x\",\"passages\":3000}",
        )
        .unwrap();
        assert!(load_window(&dir, Family::RareTerm, 0, Some(1))
            .unwrap_err()
            .contains("preparation"));
        std::fs::write(dir.join("PREPARED.json"), &prepared).unwrap();
        let payloads = std::fs::read(dir.join("payloads.jsonl")).unwrap();
        std::fs::write(dir.join("payloads.jsonl"), [&payloads[..], b"\n"].concat()).unwrap();
        assert!(load_window(&dir, Family::RareTerm, 0, Some(1))
            .unwrap_err()
            .contains("bytes"));
        std::fs::write(dir.join("payloads.jsonl"), &payloads).unwrap();
        let f = dir.join(DIR).join(Family::UrlEq.file_name());
        let body = std::fs::read_to_string(&f).unwrap();
        std::fs::write(&f, body.replacen("example", "exbmple", 1)).unwrap();
        assert!(load_window(&dir, Family::UrlEq, 0, Some(1))
            .unwrap_err()
            .contains("SHA-256"));
        std::fs::write(&f, body).unwrap();
        assert!(load_window(&dir, Family::UrlEq, 0, Some(1)).is_ok());
        // A v5 manifest (no high-pressure families) or a v4 one is refused
        // for every family, as v5 refused v4: regenerate.
        let mp = dir.join(DIR).join(MANIFEST);
        let manifest = std::fs::read_to_string(&mp).unwrap();
        for old in ["vdb-msmarco-disk/v5", "vdb-msmarco-disk/v4"] {
            let stale = manifest.replace(GENERATOR, old);
            assert_ne!(stale, manifest);
            std::fs::write(&mp, stale).unwrap();
            for family in [Family::UrlEq, Family::DocidBatch, Family::RareTermsOr] {
                let e = load_window(&dir, family, 0, Some(1)).unwrap_err();
                assert!(e.contains(old) && e.contains(GENERATOR), "{e}");
            }
        }
        std::fs::write(&mp, manifest).unwrap();
        std::fs::remove_dir_all(dir.join(DIR)).unwrap();
        assert!(load_window(&dir, Family::UrlEq, 0, Some(1))
            .unwrap_err()
            .contains("--add-disk-queries"));
    }

    #[test]
    fn smallest_keeps_exact_counts_of_survivors() {
        // Brute force over a fixed stream: the kept set is the `cap`
        // smallest distinct keys, and each kept count is its full count.
        let stream = [9u64, 3, 7, 3, 1, 9, 8, 1, 2, 3, 7, 0, 9, 2, 2];
        for cap in 0..8 {
            let mut s = Smallest::new(cap);
            for k in stream {
                s.add(k);
            }
            let mut distinct: Vec<u64> = stream.to_vec();
            distinct.sort();
            distinct.dedup();
            let want: Vec<(u64, u64)> = distinct
                .into_iter()
                .take(cap)
                .map(|k| (k, stream.iter().filter(|x| **x == k).count() as u64))
                .collect();
            assert_eq!(s.into_sorted(), want, "cap {cap}");
        }
    }

    #[test]
    fn redis_tag_pieces_split_and_trim_like_redisearch() {
        fn p(v: &str) -> Vec<&str> {
            redis_tag_pieces(v).collect()
        }
        assert_eq!(p("http://e.com/p"), ["http://e.com/p"]);
        assert_eq!(
            p("http://e.com/p;jsessionid=1"),
            ["http://e.com/p", "jsessionid=1"]
        );
        assert_eq!(p("http://e.com/p "), ["http://e.com/p"]);
        assert_eq!(p("\t a \x0b;; b\r\n;  "), ["a", "b"]);
        assert!(p(" ; ").is_empty());
    }

    #[test]
    fn tag_safety() {
        assert!(is_safe_tag("http://0-60.reviews/0-60-times/"));
        assert!(is_safe_tag("msmarco_v2.1_doc_00_0#2_3101"));
        for bad in [
            "",
            "a;b",
            "a b",
            " a",
            "caf\u{e9}",
            "a\tb",
            &"x".repeat(2049),
        ] {
            assert!(!is_safe_tag(bad), "{bad:?}");
        }
    }

    #[test]
    fn family_lines_round_trip_and_reject_malformed() {
        let (c, n) = Family::TwoRareTerms
            .conditions(&json!(["alpha", "beta", 3]))
            .unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            c,
            json!({"and":[{"segment":{"match":{"text":"alpha"}}},{"segment":{"match":{"text":"beta"}}}]})
        );
        assert_eq!(
            Family::UrlEq.conditions(&json!(["u", 1])).unwrap().0,
            json!({"url":{"match":{"value":"u"}}})
        );
        for bad in [
            json!(["alpha", 0]),
            json!(["alpha"]),
            json!(["alpha", "beta", 1]),
            json!([1, 1]),
            json!({"t": "alpha"}),
        ] {
            assert!(Family::RareTerm.conditions(&bad).is_err(), "{bad}");
        }
        for f in Family::ALL {
            assert_eq!(Family::parse(f.name()).unwrap(), f);
        }
        assert!(Family::parse("segment").is_err());
        assert!(Family::DeepFetch.check_params(100, true, false).is_ok());
        assert!(Family::DeepFetch.check_params(10, true, false).is_err());
        assert!(Family::DeepFetch.check_params(100, false, false).is_err());
        assert!(Family::RareTerm.check_params(10, false, false).is_ok());
    }

    /// docid-batch: every line is [`DOCID_BATCH_K`] distinct eligible docids
    /// with count K (each matches one passage, on Redis tag pieces too); the
    /// lines are epochs of seeded random partitions of the pool: disjoint
    /// within an epoch, each epoch covering the pool less `len % K` docids,
    /// in key order; and the layout is the probe layout.
    #[test]
    fn docid_batch_is_epochs_of_random_partitions_with_exact_count_k() {
        let (_tmp, dir) = fixture(3000);
        let out = dir.join(DIR);
        let m = generate(&dir, &opts(200, 7), &out).unwrap();
        let corpus = rows(&dir);
        let mut rows_per_docid: HashMap<&str, u64> = HashMap::new();
        for r in &corpus {
            for p in redis_tag_pieces(r["docid"].as_str().unwrap()) {
                *rows_per_docid.entry(p).or_default() += 1;
            }
        }
        let eligible: HashSet<&str> = corpus
            .iter()
            .map(|r| r["docid"].as_str().unwrap())
            .filter(|d| is_safe_tag(d) && rows_per_docid[d] == 1)
            .collect();
        assert_eq!(eligible.len(), 3000 - 3 - 3);
        let l = lines(&out, Family::DocidBatch);
        assert_eq!(l.len(), 200 + RESERVED);
        let k = DOCID_BATCH_K;
        let per_epoch = eligible.len() / k;
        let f = &m["families"]["docid-batch"];
        assert_eq!(f["timed_queries"], 200);
        assert_eq!(f["fields"], json!(["docid"]));
        assert_eq!(f["batch"]["k"], k);
        assert_eq!(f["batch"]["batches_per_epoch"], per_epoch);
        assert_eq!(f["batch"]["epochs"], l.len().div_ceil(per_epoch));
        assert_eq!(f["batch"]["docids_unused_per_epoch"], eligible.len() % k);
        assert_eq!(f["batch"]["docids_excluded_duplicate"], 0);
        for key in [
            "expected_count",
            "expected_count_timed",
            "expected_count_probe_reserve",
        ] {
            assert_eq!(f[key]["min"], k, "{key}");
            assert_eq!(f[key]["max"], k, "{key}");
        }
        for (e, epoch) in l.chunks(per_epoch).enumerate() {
            let domain = format!("docid-batch/epoch-{e}");
            let mut seen = HashSet::new();
            let mut last = None;
            for line in epoch {
                let (_, count) = Family::DocidBatch.conditions(line).unwrap();
                assert_eq!(count, k as u64, "{line}");
                let ids: Vec<&str> = line[0]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect();
                assert_eq!(ids.len(), k);
                for d in &ids {
                    assert!(eligible.contains(d), "{d}");
                    assert!(seen.insert(*d), "{d} repeats within epoch {e}");
                    let key = rank_key(7, &domain, d);
                    assert!(last.is_none_or(|p| p < key), "epoch {e}: not in key order");
                    last = Some(key);
                }
            }
            if epoch.len() == per_epoch {
                // A full epoch: the pool less exactly the `len % K` docids
                // with the highest keys.
                assert_eq!(seen.len(), per_epoch * k);
                let unused: Vec<&str> = eligible
                    .iter()
                    .copied()
                    .filter(|d| !seen.contains(d))
                    .collect();
                assert_eq!(unused.len(), eligible.len() % k);
                assert!(unused
                    .iter()
                    .all(|d| rank_key(7, &domain, d) > last.unwrap()));
            }
        }
        // Equal counts: the probe layout keeps key order.
        assert_eq!(probe_layout(Family::DocidBatch, l.clone()).unwrap(), l);
        // Seeded: another seed, other batches.
        let other = dir.join("other");
        generate(&dir, &opts(200, 8), &other).unwrap();
        assert_ne!(lines(&other, Family::DocidBatch), l);
    }

    /// Every batch family is epochs of seeded random partitions of its pool
    /// under its OWN key domain `<family>/epoch-<e>`, reconstructed here
    /// independently: epoch e ranks the pool by `rank_key(seed,
    /// "<family>/epoch-<e>", item)` and cuts it into consecutive batches of
    /// the family's batch size (no item twice in an epoch). The file holds
    /// exactly the first `lines` such batches, each in key order (the docid
    /// batches in that order; the others in the probe layout, so compared as
    /// a multiset).
    #[test]
    fn every_batch_family_is_epochs_of_its_own_partitions() {
        let (_tmp, dir) = fixture(3000);
        let out = dir.join(DIR);
        let m = generate(&dir, &opts(200, 7), &out).unwrap();
        for family in Family::ALL.into_iter().filter(|f| f.batch_size().is_some()) {
            let n = family.batch_size().unwrap();
            let file: Vec<Vec<String>> = lines(&out, family)
                .iter()
                .map(|l| {
                    l[0].as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_string())
                        .collect()
                })
                .collect();
            // The pool: every item the file uses (each epoch uses all but
            // len % n of them, and the fixture runs many epochs).
            let pool: Vec<String> = file
                .iter()
                .flatten()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            assert_eq!(
                m["families"][family.name()]["eligible_pool"],
                pool.len(),
                "{family:?}"
            );
            let mut want: Vec<Vec<String>> = Vec::new();
            let mut epoch = 0;
            while want.len() < file.len() {
                let domain = format!("{}/epoch-{epoch}", family.name());
                let mut keyed: Vec<(u64, &String)> =
                    pool.iter().map(|i| (rank_key(7, &domain, i), i)).collect();
                keyed.sort();
                for chunk in keyed.chunks_exact(n) {
                    if want.len() == file.len() {
                        break;
                    }
                    want.push(chunk.iter().map(|(_, i)| (*i).clone()).collect());
                }
                epoch += 1;
            }
            assert_eq!(
                m["families"][family.name()]["batch"]["epochs"],
                epoch,
                "{family:?}"
            );
            if family.is_docid_batch() {
                assert_eq!(file, want, "{family:?}");
            } else {
                let (mut a, mut b) = (file.clone(), want.clone());
                a.sort();
                b.sort();
                assert_eq!(a, b, "{family:?}");
            }
        }
    }

    /// The five v6 family files, pinned like the v5 ones: a change to their
    /// selection, keys, layout or counts changes these.
    #[test]
    fn v6_family_files_are_pinned() {
        let pins = [
            (
                Family::DocidBatch500,
                "29ab80ca450b5740fd878d62b2d87ef438fa95bca2db139b101fd4e266236153",
                "6a1ee92249700672754c0843eea27d22cbbe0de15959f78fae11fdbe00abd271",
            ),
            (
                Family::DocidBatch1000,
                "e6b6271f22f07705f92b2adc9b59559a2a1ab8c2e8ce8d8350538e144f502586",
                "d4f54d82925292784c9973302a3e314540599b10a958043f6636703e7456b5cf",
            ),
            (
                Family::UrlBatch,
                "dad68b6810a04a02116bd2245569bd648bc0a1fcd85a79c290269334ae1f2f7b",
                "005173cf7d10bec0ac6827a91d18b8f58db84d6938f4840300c13e00fa1ae525",
            ),
            (
                Family::RareTermsOr,
                "d993da94437b4ed477f98c8a123132e5a466fa5584958b9e28d26a0e092f0096",
                "3808e780c76eb2067f13d8798dabb6a02c26eae3a7aa7a3aa6056d52fbdc3cd1",
            ),
            (
                Family::RareTermsOrAndCategory,
                "96a45cf92c30bb4ab743672399eb7dbf6babb8daad7c9bdfd1f36bc6f2a1c00e",
                "e0259cc87134f60fdefdeaf71f9eea43ff9ad84bea932bbb1f6110d75c6f6ebc",
            ),
        ];
        let (_tmp, collisions) = fixture(3000);
        let a = generate(&collisions, &opts(200, 7), &collisions.join(DIR)).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let plain = tmp.path().join("corpus");
        write_fixture_corpus(&plain, 3000, 4).unwrap();
        let b = generate(&plain, &opts(100, 1), &plain.join(DIR)).unwrap();
        for (family, sha_a, sha_b) in pins {
            assert_eq!(a["families"][family.name()]["sha256"], sha_a, "{family:?}");
            assert_eq!(b["families"][family.name()]["sha256"], sha_b, "{family:?}");
        }
    }

    /// The seven v5 families are byte-identical under v6: the SHA-256 of
    /// each fixture file as generator v5 (901e8c3) wrote it, for both unit
    /// fixtures (the integration targets generate with `per_family: 200`
    /// since v6, so their files are not these). The six v4 families kept
    /// their v4 (fff214c) hashes under v5.
    #[test]
    fn v5_family_files_are_unchanged() {
        let pins = [
            (
                Family::RareTerm,
                "6156a2046a6297d32fb65c83abd481bad57730ac14798c08f72103f885d78505",
                "8399b8af54186772b0780ae72356cebebdabe4f4e7bc2581c5e003bebe3b5f9b",
            ),
            (
                Family::TwoRareTerms,
                "54672ae5e81ce87925daa7237fd98e7d8f586d965e002389f5f93f737bf8ee11",
                "69c8428b0ca02a53a8a661ac789287548f8f78e4237e189265b01503ad9ba305",
            ),
            (
                Family::RareTermAndCategory,
                "3a1e779efc93e8db3bf9402b78ba6221c141ecbf9300d762d4367a7851c00453",
                "cef3724594aed3a93098170c9da307dc43c750e6edd89669d1ae69f9621f4e2e",
            ),
            (
                Family::UrlEq,
                "884b388524e29cf56709f1a58e18b05b99eb15f0ebbd8cb2ea17df54224ef654",
                "e2b230e083bcea67a703845a2883b5dbd740f3da520defafdd5c203c78bb8297",
            ),
            (
                Family::DocidEq,
                "c6fb88d3bbc909d5d2ee8358fb8a7ff05b77597e556a04f1364f15e769d0486c",
                "df99be807f6351771f4c983b06142a31f866b1d2da3b714002bc0bf418ff0127",
            ),
            (
                Family::DeepFetch,
                "1248dff4383978eec0eb325e93cc025a877f8fa0ed1b7013aa651c48c5b64678",
                "1716b1b778ec06d1ea3221ad56dca152a83594147e4a7a0b16e09d9d2494202d",
            ),
            (
                Family::DocidBatch,
                "6a2db1f45e5b4fda8139c34552f163f107d9b4189d4217c7c119ce6dbd2f1360",
                "d1bb81f6e6c46922f5c55a9713794c31861172b90789e4bf8fe91cb2483b9288",
            ),
        ];
        // Every v5 family is pinned; the rest are the v6 additions.
        assert_eq!(pins.len() + 5, Family::ALL.len());
        let (_tmp, collisions) = fixture(3000);
        let a = generate(&collisions, &opts(200, 7), &collisions.join(DIR)).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let plain = tmp.path().join("corpus");
        write_fixture_corpus(&plain, 3000, 4).unwrap();
        let b = generate(&plain, &opts(100, 1), &plain.join(DIR)).unwrap();
        for (family, sha_a, sha_b) in pins {
            assert_eq!(a["families"][family.name()]["sha256"], sha_a, "{family:?}");
            assert_eq!(b["families"][family.name()]["sha256"], sha_b, "{family:?}");
        }
    }

    #[test]
    fn docid_pool_excludes_every_copy_of_a_repeated_docid() {
        let docids = ["b", "a", "c", "a", "d", "c", "a"];
        let mut arena = String::new();
        let mut ends = Vec::new();
        for d in docids {
            arena.push_str(d);
            ends.push(arena.len());
        }
        let pool = DocidPool::new(arena, ends).unwrap();
        let kept: Vec<&str> = pool.ids.iter().map(|&i| pool.get(i)).collect();
        assert_eq!(kept, ["b", "d"]);
        assert_eq!(pool.duplicates, 2);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn docid_batch_lines_round_trip_and_reject_malformed() {
        let k = DOCID_BATCH_K;
        let ids: Vec<String> = (0..k)
            .map(|i| format!("msmarco_v2.1_doc_00_{i}#0_1"))
            .collect();
        let (c, n) = Family::DocidBatch.conditions(&json!([ids, k])).unwrap();
        assert_eq!(n, k as u64);
        assert_eq!(c, json!({"docid":{"match":{"any":ids}}}));
        let expr = crate::compound_text::Expr::compile(&c, &json!({"docid":"keyword"})).unwrap();
        let redis = expr.redis();
        assert!(
            redis.starts_with("(@docid:{msmarco\\_v2\\.1\\_doc\\_00\\_0\\#0\\_1 | "),
            "{redis}"
        );
        assert_eq!(redis.matches(" | ").count(), k - 1);
        assert_eq!(
            expr.opensearch(),
            json!({"bool":{"filter":[{"terms":{"docid":ids}}]}})
        );
        let mut dup = ids.clone();
        dup[5] = dup[4].clone();
        let mut unsafe_ = ids.clone();
        unsafe_[0] = "a;b".into();
        for bad in [
            json!([ids, k - 1]),
            json!([ids[1..], k - 1]),
            json!([ids[1..], k]),
            json!([dup, k]),
            json!([unsafe_, k]),
            json!([ids, k, 1]),
            json!([ids[0], 1]),
            json!([ids, 0]),
        ] {
            assert!(Family::DocidBatch.conditions(&bad).is_err(), "{bad}");
        }
        assert!(Family::DocidBatch.check_params(k, true, false).is_ok());
        assert!(Family::DocidBatch.check_params(10, true, false).is_err());
        assert!(Family::DocidBatch.check_params(k, false, false).is_err());
        let e = Family::DocidBatch.check_params(k, true, true).unwrap_err();
        assert!(e.contains("exact"), "{e}");
        assert!(Family::DeepFetch.check_params(100, true, true).is_ok());
        assert!(Family::DocidEq.check_params(10, false, true).is_ok());
        assert_eq!(Family::DocidBatch.fixed_top(), Some(k));
        assert_eq!(Family::DocidEq.fixed_top(), None);
    }

    /// The exact union counts each passage once, whatever the overlap
    /// between the lists (and agrees with a set on random lists).
    #[test]
    fn union_rows_counts_each_passage_once() {
        let a: &[u32] = &[1, 4, 7, 9];
        let b: &[u32] = &[2, 4, 9, 11];
        let c: &[u32] = &[9];
        assert_eq!(union_rows(&[a, b, c]), [1, 2, 4, 7, 9, 11]);
        // Overlapping lists: 4 + 4 + 1 postings, 6 passages.
        assert_eq!(union_rows(&[a, b, c]).len(), 6);
        // The same list twice, a list inside another, disjoint lists.
        assert_eq!(union_rows(&[a, a]).len(), 4);
        assert_eq!(union_rows(&[a, &[4, 9]]).len(), 4);
        assert_eq!(union_rows(&[&[1, 2], &[3]]).len(), 3);
        assert!(union_rows(&[]).is_empty());
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(41);
        for _ in 0..200 {
            let lists: Vec<Vec<u32>> = (0..rng.gen_range(1..8))
                .map(|_| {
                    let mut l: Vec<u32> = (0..rng.gen_range(0..30))
                        .map(|_| rng.gen_range(0..60))
                        .collect();
                    l.sort_unstable();
                    l.dedup();
                    l
                })
                .collect();
            let refs: Vec<&[u32]> = lists.iter().map(Vec::as_slice).collect();
            let set: std::collections::BTreeSet<u32> = lists.iter().flatten().copied().collect();
            assert_eq!(union_rows(&refs), set.into_iter().collect::<Vec<_>>());
        }
    }

    /// The v6 high-pressure families on the fixture: every count is the exact
    /// number of DISTINCT passages a brute-force scan of the corpus matches
    /// (the per-family oracle of `generated_counts_match_a_brute_force_oracle`
    /// checks the Redis tokenization and tag pieces too); `rare-terms-or`
    /// lines do overlap (so a union that counted a passage per term would
    /// fail here); the docid batches and their Redis query sizes; the
    /// manifest records every parameter.
    #[test]
    fn high_pressure_families_have_exact_union_counts() {
        let (_tmp, dir) = fixture(3000);
        let out = dir.join(DIR);
        let m = generate(&dir, &opts(200, 7), &out).unwrap();
        let corpus = rows(&dir);
        let schema =
            json!({"segment":"text","category":"keyword","url":"keyword","docid":"keyword"});
        let hp = &m["high_pressure"];
        assert_eq!(hp["docid_batch_500_k"], DOCID_BATCH_500_K);
        assert_eq!(hp["docid_batch_1000_k"], DOCID_BATCH_1000_K);
        assert_eq!(hp["url_batch_n"], URL_BATCH_N);
        assert_eq!(hp["rare_or_n"], RARE_OR_N);
        assert_eq!(hp["rare_or_df"], json!(RARE_OR_DF));
        assert_eq!(hp["top"], HIGH_PRESSURE_TOP);
        assert_eq!(m["df_bands"]["rare_or"], json!(RARE_OR_DF));
        // Rule texts are single-spaced prose (no source indentation).
        for family in Family::ALL {
            if let Some(rule) = m["families"][family.name()]["batch"]["rule"].as_str() {
                assert!(!rule.contains("  "), "{family:?}: {rule}");
            }
        }
        // Brute-force postings (classifier per passage) of every term used.
        let mut postings: HashMap<String, HashSet<usize>> = HashMap::new();
        for f in [Family::RareTermsOr, Family::RareTermsOrAndCategory] {
            for line in lines(&out, f) {
                for t in line[0].as_array().unwrap() {
                    let t = t.as_str().unwrap();
                    if !postings.contains_key(t) {
                        let rows = (0..corpus.len())
                            .filter(|&r| verdict(&corpus[r], t) == TextMatch::Yes)
                            .collect();
                        postings.insert(t.to_string(), rows);
                    }
                }
            }
        }
        let df_of = |t: &str| postings[t].len() as u64;
        let union_of = |terms: &[&str]| -> HashSet<usize> {
            terms
                .iter()
                .flat_map(|t| postings[*t].iter().copied())
                .collect()
        };

        // Docid batches: K distinct eligible docids, count K, the line
        // budget, and the Redis query length the manifest records.
        for family in [Family::DocidBatch500, Family::DocidBatch1000] {
            let k = family.batch_size().unwrap();
            let l = lines(&out, family);
            assert_eq!(l.len(), tag_batch_timed_lines(k, 200, 5) + RESERVED);
            assert_eq!(l.len(), 200 * 100 / k + RESERVED, "{family:?}");
            let f = &m["families"][family.name()];
            assert_eq!(f["batch"]["k"], k);
            assert_eq!(f["batch"]["batches_per_epoch"], (3000 - 6) / k);
            assert_eq!(f["expected_count"]["min"], k);
            assert_eq!(f["expected_count"]["max"], k);
            let mut redis_max = 0;
            for line in &l {
                let (c, n) = family.conditions(line).unwrap();
                assert_eq!(n, k as u64);
                let expr = crate::compound_text::Expr::compile(&c, &schema).unwrap();
                redis_max = redis_max.max(expr.redis().len());
                assert_eq!(
                    expr.opensearch()["bool"]["filter"][0]["terms"]["docid"]
                        .as_array()
                        .unwrap()
                        .len(),
                    k
                );
            }
            assert_eq!(f["batch"]["redis_query_bytes_max"], redis_max, "{family:?}");
            assert!(family.check_params(k, true, false).is_ok());
            assert!(family.check_params(100, true, false).is_err());
            assert!(family.check_params(k, true, true).is_err());
        }

        // url-batch: N distinct eligible URLs; the count is the number of
        // passages carrying one of them (each passage has one URL).
        let url_eq_eligible: HashSet<String> = {
            let mut pieces = HashSet::new();
            for r in &corpus {
                let u = r["url"].as_str().unwrap();
                if !is_safe_tag(u) {
                    pieces.extend(redis_tag_pieces(u).map(str::to_string));
                }
            }
            corpus
                .iter()
                .map(|r| r["url"].as_str().unwrap())
                .filter(|u| is_safe_tag(u) && !pieces.contains(*u))
                .map(str::to_string)
                .collect()
        };
        let f = &m["families"]["url-batch"];
        assert_eq!(f["eligible_pool"], url_eq_eligible.len());
        assert_eq!(f["batch"]["n"], URL_BATCH_N);
        assert_eq!(
            lines(&out, Family::UrlBatch).len(),
            200 * 100 / URL_BATCH_N + RESERVED
        );
        assert_eq!(
            f["batch"]["batches_per_epoch"],
            url_eq_eligible.len() / URL_BATCH_N
        );
        let mut redis_max = 0;
        for line in lines(&out, Family::UrlBatch) {
            let urls: HashSet<&str> = line[0]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(urls.len(), URL_BATCH_N);
            assert!(urls.iter().all(|u| url_eq_eligible.contains(*u)));
            let want = corpus
                .iter()
                .filter(|r| urls.contains(r["url"].as_str().unwrap()))
                .count() as u64;
            let (c, n) = Family::UrlBatch.conditions(&line).unwrap();
            assert_eq!(n, want, "{line}");
            let expr = crate::compound_text::Expr::compile(&c, &schema).unwrap();
            assert!(expr.redis().starts_with("(@url:{https\\:\\/\\/"));
            redis_max = redis_max.max(expr.redis().len());
        }
        assert_eq!(f["batch"]["redis_query_bytes_max"], redis_max);

        // rare-terms-or: the distinct passages holding any term.
        let mut overlapping = 0;
        let f = &m["families"]["rare-terms-or"];
        assert_eq!(f["df_band"], json!(RARE_OR_DF));
        // The deep-fetch eligibility: the same band and rules.
        assert_eq!(
            f["eligible_pool"],
            m["families"]["deep-fetch"]["eligible_pool"]
        );
        for line in lines(&out, Family::RareTermsOr) {
            let terms: Vec<&str> = line[0]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(terms.len(), RARE_OR_N);
            let want = union_of(&terms).len() as u64;
            let df_sum: u64 = terms.iter().map(|t| df_of(t)).sum();
            let (_, n) = Family::RareTermsOr.conditions(&line).unwrap();
            assert_eq!(n, want, "{line}");
            assert!(n <= df_sum);
            if n < df_sum {
                overlapping += 1;
            }
        }
        assert!(
            overlapping > 0,
            "the fixture must exercise overlapping terms"
        );
        assert_eq!(f["batch"]["lines_with_overlapping_terms"], overlapping);

        // rare-terms-or-and-category: the union AND the category of its
        // passage with the smallest seeded docid key.
        for line in lines(&out, Family::RareTermsOrAndCategory) {
            let terms: Vec<&str> = line[0]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            let union: Vec<&Value> = union_of(&terms).into_iter().map(|r| &corpus[r]).collect();
            let best = union
                .iter()
                .min_by_key(|r| {
                    rank_key(
                        7,
                        "rare-terms-or-and-category/passage",
                        r["docid"].as_str().unwrap(),
                    )
                })
                .unwrap();
            assert_eq!(line[1], best["category"], "{line}");
            let want = union.iter().filter(|r| r["category"] == line[1]).count() as u64;
            assert_eq!(
                Family::RareTermsOrAndCategory.conditions(&line).unwrap().1,
                want
            );
        }
    }

    #[test]
    fn high_pressure_lines_round_trip_and_reject_malformed() {
        let terms: Vec<String> = (0..RARE_OR_N)
            .map(|i| format!("term{}", (b'a' + i as u8 % 26) as char).repeat(1 + i / 26))
            .collect();
        let (c, n) = Family::RareTermsOr
            .conditions(&json!([terms, 900]))
            .unwrap();
        assert_eq!(n, 900);
        let schema = json!({"segment":"text","category":"keyword","url":"keyword"});
        let expr = crate::compound_text::Expr::compile(&c, &schema).unwrap();
        // Redis: one field-scoped union; OpenSearch: a bool.should of one
        // match per term, at least one required.
        assert_eq!(expr.redis(), format!("(@segment:({}))", terms.join(" | ")));
        let os = expr.opensearch();
        assert_eq!(os["bool"]["minimum_should_match"], 1);
        let should = os["bool"]["should"].as_array().unwrap();
        assert_eq!(should.len(), RARE_OR_N);
        assert_eq!(
            should[0],
            json!({"match":{"segment":{"query":terms[0],"operator":"and"}}})
        );
        let (c, n) = Family::RareTermsOrAndCategory
            .conditions(&json!([terms, "cat03", 120]))
            .unwrap();
        assert_eq!(n, 120);
        let expr = crate::compound_text::Expr::compile(&c, &schema).unwrap();
        assert_eq!(
            expr.redis(),
            format!("((@segment:({})) (@category:{{cat03}}))", terms.join(" | "))
        );
        let os = expr.opensearch();
        assert_eq!(
            os["bool"]["must"][0]["bool"]["should"]
                .as_array()
                .unwrap()
                .len(),
            RARE_OR_N
        );
        assert_eq!(
            os["bool"]["must"][1],
            json!({"bool":{"filter":[{"terms":{"category":["cat03"]}}]}})
        );
        let urls: Vec<String> = (0..URL_BATCH_N)
            .map(|i| format!("https://e.org/{i}"))
            .collect();
        let (c, _) = Family::UrlBatch.conditions(&json!([urls, 500])).unwrap();
        assert_eq!(c, json!({"url":{"match":{"any":urls}}}));
        let mut dup_terms = terms.clone();
        dup_terms[1] = dup_terms[0].clone();
        let mut bad_term = terms.clone();
        bad_term[0] = "the".into();
        for (family, bad) in [
            (Family::RareTermsOr, json!([terms[1..], 5])),
            (Family::RareTermsOr, json!([dup_terms, 5])),
            (Family::RareTermsOr, json!([bad_term, 5])),
            (Family::RareTermsOr, json!([terms, 0])),
            (Family::RareTermsOr, json!([terms, "cat03", 5])),
            (Family::RareTermsOrAndCategory, json!([terms, 5])),
            (Family::RareTermsOrAndCategory, json!([terms, "a;b", 5])),
            (Family::UrlBatch, json!([urls[1..], 5])),
            (Family::UrlBatch, json!([terms, 5])),
            (Family::UrlBatch, json!([urls[0], 5])),
        ] {
            assert!(family.conditions(&bad).is_err(), "{family:?} {bad}");
        }
        for family in [
            Family::UrlBatch,
            Family::RareTermsOr,
            Family::RareTermsOrAndCategory,
        ] {
            assert_eq!(family.fixed_top(), Some(HIGH_PRESSURE_TOP));
            assert!(family.check_params(100, true, false).is_ok());
            assert!(family.check_params(10, true, false).is_err());
            assert!(family.check_params(100, false, false).is_err());
            let e = family.check_params(100, true, true).unwrap_err();
            assert!(e.contains("exact"), "{e}");
        }
        // Fetch parity: the v6 families fetch exactly title, segment, url.
        let f = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        for family in Family::ALL {
            let pinned = family.fixed_fetch().is_some();
            assert_eq!(
                pinned,
                family.batch_size().is_some() && family != Family::DocidBatch,
                "{family:?}"
            );
            assert!(family
                .check_fetch_fields(Some(&f(&["title", "segment", "url"])))
                .is_ok());
            for bad in [
                Some(f(&["title", "segment"])),
                Some(f(&["segment", "title", "url"])),
                Some(f(&["title", "segment", "url", "docid"])),
                Some(f(&["docid"])),
                None,
            ] {
                assert_eq!(
                    family.check_fetch_fields(bad.as_deref()).is_err(),
                    pinned,
                    "{family:?} {bad:?}"
                );
            }
        }
        // The documented parameters, as literals (a retuned constant must
        // update the docs and the manifest consumers).
        assert_eq!(RARE_OR_DF, (11, 100));
        assert_eq!((RARE_OR_N, URL_BATCH_N, HIGH_PRESSURE_TOP), (40, 300, 100));
        assert_eq!((DOCID_BATCH_500_K, DOCID_BATCH_1000_K), (500, 1000));
        assert_eq!(Family::DocidBatch500.fixed_top(), Some(500));
        assert_eq!(Family::DocidBatch1000.fixed_top(), Some(1000));
        let ids: Vec<String> = (0..DOCID_BATCH_500_K).map(|i| format!("d{i}")).collect();
        assert!(Family::DocidBatch500
            .conditions(&json!([ids, DOCID_BATCH_500_K]))
            .is_ok());
        assert!(Family::DocidBatch500
            .conditions(&json!([ids, 100]))
            .is_err());
        assert!(Family::DocidBatch1000
            .conditions(&json!([ids, DOCID_BATCH_500_K]))
            .is_err());
        assert_eq!(tag_batch_timed_lines(100, 1_000_000, 100_000), 1_000_000);
        assert_eq!(tag_batch_timed_lines(500, 1_000_000, 100_000), 200_000);
        assert_eq!(tag_batch_timed_lines(1000, 1_000_000, 100_000), 100_000);
        assert_eq!(tag_batch_timed_lines(1000, 100_000, 100_000), 100_000);
    }

    const RARE_FRIZZES: u64 = 0x6859_78ae_b850_f4a3;

    #[test]
    fn pair_key_is_order_sensitive_and_stable() {
        assert_ne!(pair_key(1, 2), pair_key(2, 1));
        // Pinned: a platform or refactor that changes the mix changes every
        // generated two-rare-terms file.
        assert_eq!(pair_key(1, 2), 0x37ad_5fdd_5756_bd3d);
        // Pinned against an independent implementation (Python hashlib):
        // sha256(b"vdb-msmarco-disk/v1\0" + (1).to_bytes(8, "big") +
        // b"rare-term\0frizzes")[:8], big-endian.
        assert_eq!(rank_key(1, "rare-term", "frizzes"), RARE_FRIZZES);
    }
}
