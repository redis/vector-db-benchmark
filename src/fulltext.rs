//! Ranked full-text (BM25) workload: Redis `FT.SEARCH` vs OpenSearch
//! `multi_match` (#339).
//!
//! Everything both engines must agree on lives here, so the two arms in the
//! binary are I/O only: the query terms each engine receives, the pinned
//! analyzer settings, the per-field boosts, the request each engine is sent,
//! the reply parsers, the checks that the LIVE index was built the way this
//! module says, and the metadata the result JSON carries.
//!
//! # Analyzer parity (decided for #339, recorded in every result)
//!
//! | | Redis (RediSearch) | OpenSearch |
//! |---|---|---|
//! | tokenizer | RediSearch default: splits on [`REDISEARCH_SEPARATORS`] + space/tab | `standard` (Unicode UAX#29) |
//! | possessives | none (`'` is a separator) | `stemmer` / `possessive_english` |
//! | case | lowercased | `lowercase` |
//! | stopwords | [`STOPWORDS`], passed explicitly as `STOPWORDS 33 …` | `stop` with [`STOPWORDS`] listed explicitly |
//! | stemmer | Snowball English (= Porter2), the RediSearch default | `stemmer` / `porter2` (Snowball English) |
//! | scorer | `BM25STD` (explicit `SCORER`) | Lucene BM25 (k1=1.2, b=0.75) |
//!
//! [`STOPWORDS`] is RediSearch's `DEFAULT_STOPWORDS` (`src/stopwords.h`) and
//! Lucene's `EnglishAnalyzer.ENGLISH_STOP_WORDS_SET`; the two lists were
//! compared and are the same 33 words. Both engines are still given the list
//! explicitly, so the server reports it back (`FT.INFO stopwords_list`,
//! `_settings`) and the run refuses an index built any other way — a server
//! whose built-in default drifted cannot silently change what is measured.
//!
//! The built-in OpenSearch `english` analyzer is NOT used because its stemmer
//! is the original Porter algorithm (`english` -> `PorterStemFilter`), while
//! RediSearch stems with Snowball English (Porter2): `generously` is `gener`
//! under one and `generous` under the other. The custom analyzer is the
//! `english` analyzer with `porter2` swapped in (and its empty
//! `keyword_marker` step dropped).
//!
//! Differences that cannot be pinned away are listed in
//! [`KNOWN_DIFFERENCES`] and travel with every result.
//!
//! # Query construction
//!
//! The SAME term list is sent to both engines, derived once by
//! [`query_terms`]: drop possessive `'s`, split on RediSearch's separators and all whitespace,
//! lowercase, drop [`STOPWORDS`], drop duplicates, keep first-seen order.
//! Splitting on RediSearch's own separator set is what makes a term safe to
//! embed in a RediSearch query without escaping (every query-syntax character
//! is a separator) and matches how RediSearch tokenized the documents. A query
//! with no terms left (all stopwords / punctuation) is skipped on both engines
//! and listed in the result.
//!
//! Terms are OR-ed (`|` / `operator: or`) — a ranked-retrieval benchmark, not
//! a boolean filter. Field boosts are applied at QUERY time on both engines so
//! that one index serves every search config:
//!
//! - Redis: `(@f1:(t1|t2)) => {$weight: b1;} | (@f2:(t1|t2)) => {$weight: b2;}`
//! - OpenSearch: `multi_match` `type: most_fields`, `fields: ["f1^b1", "f2^b2"]`
//!
//! Both sum, over the fields a document matches in, the boost times that
//! field clause's BM25 score — `most_fields` is the OpenSearch type with that
//! shape (`best_fields`, the default, takes the max instead).

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::BufRead;
use std::path::Path;

use crate::workload::Workload;

/// The English stopword list both engines are pinned to. Identical to
/// RediSearch `DEFAULT_STOPWORDS` (src/stopwords.h) and Lucene
/// `EnglishAnalyzer.ENGLISH_STOP_WORDS_SET` (OpenSearch `_english_`).
pub const STOPWORDS: [&str; 33] = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// RediSearch's default token separators (`ToksepMap_g`, src/toksep.h), plus
/// `\` (its escape character, which must never reach a query unescaped).
/// Space and tab are separators there too; [`query_terms`] additionally
/// splits on every Unicode whitespace character.
pub const REDISEARCH_SEPARATORS: &str = ",./(){}[]:;~!@#$%^&*-=+|'`\"<>?\\";

/// Name of the custom OpenSearch analyzer (index settings + text mappings).
pub const OPENSEARCH_ANALYZER: &str = "vdbb_english_porter2";
const OS_POSSESSIVE: &str = "vdbb_english_possessive";
const OS_STOP: &str = "vdbb_english_stop";
const OS_STEMMER: &str = "vdbb_english_porter2_stemmer";

/// The Redis scorer, passed explicitly on every query.
pub const REDIS_SCORER: &str = "BM25STD";

/// RediSearch's default minimum stem length (`search-min-stem-len` /
/// `MINSTEMLEN`): tokens shorter than this are not stemmed.
pub const REDIS_DEFAULT_MIN_STEM_LEN: i64 = 4;

/// Read a numeric setting from a `CONFIG GET` / `FT.CONFIG GET` reply
/// converted to JSON: a map `{name: value}`, a flat `[name, value]` or a nested
/// `[[name, value]]`. `None` when the reply does not carry `name`.
pub fn config_get_value(reply: &Value, name: &str) -> Option<i64> {
    let parse = |v: &Value| match v {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    };
    match reply {
        Value::Object(o) => o
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .and_then(|(_, v)| parse(v)),
        Value::Array(a) => {
            if a.len() == 2 && a[0].as_str().is_some_and(|k| k.eq_ignore_ascii_case(name)) {
                return parse(&a[1]);
            }
            a.iter().find_map(|e| config_get_value(e, name))
        }
        _ => None,
    }
}

/// The server's minimum stem length must be the default: it changes which
/// tokens RediSearch stems at index and query time, and FT.INFO does not
/// report it. `read` is what the server reported (`None`: the server would not
/// say — e.g. CONFIG disabled on a managed service — which is recorded as
/// unverified rather than refused). Returns the `redis_min_stem_len` record.
pub fn check_redis_min_stem_len(read: Option<i64>) -> Result<Value, String> {
    match read {
        Some(REDIS_DEFAULT_MIN_STEM_LEN) => {
            Ok(json!({"value": REDIS_DEFAULT_MIN_STEM_LEN, "verified": true}))
        }
        Some(n) => Err(format!(
            "fulltext: the Redis server's search-min-stem-len is {n}, not the default \
             {REDIS_DEFAULT_MIN_STEM_LEN} — tokens shorter than it are not stemmed, so the \
             pinned english analysis would not be what runs; reset it \
             (CONFIG SET search-min-stem-len {REDIS_DEFAULT_MIN_STEM_LEN})"
        )),
        None => Ok(json!({"value": null, "verified": false,
            "note": "neither CONFIG GET search-min-stem-len nor FT.CONFIG GET MINSTEMLEN answered"})),
    }
}

/// `multi_match` type sent to OpenSearch. See the module docs.
pub const OPENSEARCH_MULTI_MATCH_TYPE: &str = "most_fields";

/// Differences between the two engines' text pipelines that the pinned
/// settings do NOT remove. Published in every fulltext result.
pub const KD_WHITESPACE_TOKENS: &str =
    "tokenizer: RediSearch splits only on its ASCII separator set plus space and tab — NOT on \
     newline, carriage return or other Unicode whitespace (U+00A0, U+2028, …), so lines joined \
     by '\\n' (e.g. MS MARCO `headings`) index as one token at each join; OpenSearch's standard \
     tokenizer (UAX#29) splits on all whitespace. Parity normalizes this away at upload";
pub const KD_UNICODE_PUNCTUATION: &str =
    "tokenizer: RediSearch does not split on non-ASCII punctuation (e.g. – U+2013, — U+2014, \
     … U+2026, “ ” U+201C/U+201D, • U+2022, · U+00B7), which stays inside the token \
     ('word—word' is one token); OpenSearch's standard tokenizer (UAX#29) splits on it (it \
     keeps ’ U+2019 between letters). Not normalized in either mode — 49% of the 100K MS MARCO \
     `segment` values contain such punctuation";
pub const KD_PUNCTUATION: &str =
    "tokenizer: RediSearch splits on '.', '-', '\\'' inside words ('u.s.a' -> u, s, a; \
     'e-mail' -> e, mail; '3.5' -> 3, 5; \"dog's\" -> dog, s; \"don't\" -> don, t); OpenSearch \
     keeps 'u.s.a', '3.5' and \"don't\" as single tokens and removes possessive 's. The query \
     side drops possessive 's before splitting (so no bare `s` term is sent), but other \
     apostrophes still split into the same terms on both engines";
pub const KD_STEM_TWICE: &str =
    "stemming: RediSearch indexes the original token AND its stem and expands each query term \
     to both, so a term whose stem differs scores twice; Lucene indexes only the stem";
pub const KD_WHOLE_DOCUMENT: &str =
    "scoring: RediSearch BM25STD uses one term frequency and one document length for the whole \
     document (all TEXT fields), restricted per field only by the field mask; Lucene BM25 keeps \
     per-field term frequencies and per-field length norms";
pub const KD_SHARD_STATS: &str =
    "scoring: Lucene IDF and length norms are per shard and per segment statistics; RediSearch \
     statistics are per index";
pub const KD_TIES: &str =
    "ties: documents with equal scores may be returned in different orders by the two engines";
pub const KD_MIN_STEM_LEN: &str =
    "stemming: RediSearch stems only tokens of at least `search-min-stem-len` characters \
     (default 4, verified on the server and recorded as `redis_min_stem_len`); Lucene porter2 \
     has no length floor";

/// Every difference that the pinned settings do not remove (english mode).
pub const KNOWN_DIFFERENCES: [&str; 8] = [
    KD_WHITESPACE_TOKENS,
    KD_UNICODE_PUNCTUATION,
    KD_PUNCTUATION,
    KD_STEM_TWICE,
    KD_WHOLE_DOCUMENT,
    KD_SHARD_STATS,
    KD_TIES,
    KD_MIN_STEM_LEN,
];

/// The known differences parity removes: whitespace is normalized at upload,
/// and nothing is stemmed (so neither stem double counting nor the minimum
/// stem length applies).
pub const PARITY_REMOVES: [&str; 3] = [KD_WHITESPACE_TOKENS, KD_STEM_TWICE, KD_MIN_STEM_LEN];

/// The `fulltext` block of a search config.
///
/// ```json
/// {"workload": "fulltext", "top": 10, "fulltext": {"fields": {"title": 5.0, "segment": 1.0}}}
/// ```
///
/// `fields` omitted = every `text` field of the dataset schema at boost 1.0.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FulltextSpec {
    /// Explicit reduced index contract for Flex: only TEXT/TAG, no sorting storage.
    #[serde(default)]
    pub scalar_indexing: crate::text_tag_profile::ScalarIndexing,
    /// Explicit schema-only layout, preserving text fields.
    #[serde(default)]
    pub no_vector_index: bool,
    #[serde(default)]
    pub compound: Option<crate::compound_text::Spec>,
    #[serde(default)]
    pub fields: Option<BTreeMap<String, f64>>,
    /// OpenSearch only: accept an index with more than one shard (per-shard
    /// BM25 statistics). Off by default; the shard count is always recorded.
    #[serde(default)]
    pub allow_multi_shard: bool,
    /// Text analysis mode (see [`Analysis`]). Default `english`.
    #[serde(default)]
    pub analysis: Analysis,
    /// Whether each engine is asked for an exact total match count (see
    /// [`TotalHits`]). Absent = the legacy per-engine default, recorded in the
    /// result as asymmetric for the ranked query ([`total_hits_plan`]).
    #[serde(default)]
    pub total_hits: Option<TotalHits>,
    /// Fetch these stored fields of every hit (Redis `RETURN n f…` instead of
    /// `NOCONTENT`; OpenSearch `_source.includes` instead of `_source: false`).
    /// Absent = ids and scores only, as before. Each must be a `text` or
    /// `keyword` schema field (never the vector); every hit must come back with
    /// exactly these fields, checked on every reply on both engines.
    #[serde(default)]
    pub fetch: Option<Vec<String>>,
}

/// How both engines analyze text for the fulltext workload.
///
/// - `english` (default, the headline configuration): each engine's default
///   English pipeline, pinned — Porter2 stemming on both, the shared stopword
///   list, every schema `text` field indexed, the corpus uploaded as is.
/// - `parity` (the agreement check against OpenSearch as source of truth):
///   removes the three differences that dominate the Redis/OpenSearch gap —
///   no stemming (Redis `NOSTEM` + `VERBATIM` queries; OpenSearch analyzer
///   without the porter2 filter, same tokenizer / possessive / lowercase /
///   stop), only the searched fields indexed as text (so RediSearch's
///   whole-document statistics cover exactly those fields), and `\n` / `\r`
///   in those fields replaced by a space at upload on both engines (RediSearch
///   does not split tokens on line breaks). The normalization changes the
///   uploaded corpus, so it is recorded in the index (Redis marker key,
///   OpenSearch `_meta`) and verified on reuse, and parity never shares an
///   engine config with any other workload.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Analysis {
    #[default]
    English,
    Parity,
}

impl Analysis {
    pub fn as_str(self) -> &'static str {
        match self {
            Analysis::English => "english",
            Analysis::Parity => "parity",
        }
    }
}

/// The total-hits contract of a fulltext search config (`fulltext.total_hits`):
/// what each engine is asked to count besides the top-`k` page. Counting is
/// work, and the two engines must be asked to do the same work.
///
/// | mode | OpenSearch | Redis (RediSearch) |
/// |---|---|---|
/// | `exact` | `track_total_hits: true` (`hits.total.relation` must be `eq`) | `FT.SEARCH` default: `total_results` is the exact count |
/// | `none` | `track_total_hits: false` (`hits.total` must be absent) | `WITHOUTCOUNT`: `total_results` is `min(count, LIMIT)` per shard, not a count |
///
/// `none` frees OpenSearch to skip non-competitive blocks (block-max WAND).
/// On Redis, `WITHOUTCOUNT` turns on the query optimizer. From RediSearch 8.4
/// the optimizer recognises `SCORER BM25STD`, so for a query with a text term
/// and no `SORTBY` it still visits and scores every match
/// (`src/query_optimizer.c`, `Q_OPT_NONE`) and keeps the score-ordered top
/// `k`: the ranking is unchanged and only the reported total is clamped.
/// RediSearch 8.0/8.2 parse `WITHOUTCOUNT`, but their optimizer does not know
/// the `BM25STD` scorer name, so it drops the score sorter and returns the
/// first `k` matches in document-id order. The ranking is therefore never
/// assumed: before the timed window, [`probe_withoutcount`] compares counted
/// and `WITHOUTCOUNT` pages on the live server and refuses `none` unless they
/// are identical, and a `none` run fails if any timed reply is out of score
/// order ([`check_rank_violations`]). `none` is refused for the compound
/// suite, whose per-query count gate needs the exact total.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TotalHits {
    Exact,
    None,
}

impl TotalHits {
    pub fn as_str(self) -> &'static str {
        match self {
            TotalHits::Exact => "exact",
            TotalHits::None => "none",
        }
    }
}

/// The total-hits mode one engine runs a fulltext search config with, and
/// whether the Redis/OpenSearch pair does the same counting work. Resolved
/// once per search config by [`total_hits_plan`]. The request builders and
/// the result metadata ([`record_total_hits`]) both take it, so what is
/// recorded is what was sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TotalHitsPlan {
    pub mode: TotalHits,
    /// The config stated `fulltext.total_hits`.
    pub declared: bool,
    /// Both engines run this search config with the same mode.
    pub symmetric: bool,
}

/// The total-hits mode `engine` sends for `spec`.
///
/// - Stated: that mode on every engine (symmetric), or an error when it
///   cannot be honoured. There is never a silent fallback.
/// - Not stated, compound suite: `exact` on both engines, as before (the
///   count gate needs it), so symmetric.
/// - Not stated, ranked BM25: the legacy defaults, kept so a shipped config
///   measures what it always measured: Redis `exact`, OpenSearch `none`. The
///   pair is asymmetric, and [`record_total_hits`] says so in the result.
pub fn total_hits_plan(engine: &str, spec: &FulltextSpec) -> Result<TotalHitsPlan, String> {
    if !matches!(engine, "redis" | "opensearch") {
        return Err(format!(
            "fulltext: engine '{engine}' has no total-hits contract"
        ));
    }
    check_total_hits(spec)?;
    Ok(match (spec.total_hits, spec.compound.is_some()) {
        (Some(mode), _) => TotalHitsPlan {
            mode,
            declared: true,
            symmetric: true,
        },
        (None, true) => TotalHitsPlan {
            mode: TotalHits::Exact,
            declared: false,
            symmetric: true,
        },
        (None, false) => TotalHitsPlan {
            mode: if engine == "opensearch" {
                TotalHits::None
            } else {
                TotalHits::Exact
            },
            declared: false,
            symmetric: false,
        },
    })
}

/// Whether `spec` runs the `msmarco-disk-v1` compound suite.
pub fn is_disk_suite(spec: &FulltextSpec) -> bool {
    spec.compound.as_ref().and_then(|c| c.suite.as_deref())
        == Some(crate::msmarco::disk_queries::SUITE)
}

/// `total_hits: "none"` cannot be honoured by the compound suite: every
/// compound query's total is checked against its expected count. The
/// `msmarco-disk-v1` suite is the exception: every one of its queries has an
/// exact expected count, so under `none` the runner checks the page length
/// against it instead (`min(expected, top)`, [`check_query_page`]).
fn check_total_hits(spec: &FulltextSpec) -> Result<(), String> {
    if spec.compound.is_some() && spec.total_hits == Some(TotalHits::None) && !is_disk_suite(spec) {
        return Err(format!(
            "`{SPEC_KEY}.total_hits: \"none\"` cannot be used with `{SPEC_KEY}.compound`: \
             the compound suite checks every query's exact total against its expected \
             count; use \"exact\""
        ));
    }
    Ok(())
}

/// What an engine config's fulltext search configs require of the index:
/// resolved once from all of them ([`index_profile`]), because the index is
/// shared by every search config of the engine config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexProfile {
    pub scalar_indexing: crate::text_tag_profile::ScalarIndexing,
    pub analysis: Analysis,
    /// Parity: exactly the searched fields (the only fields indexed as text,
    /// and the only ones normalized). English: empty — every schema `text`
    /// field is indexed and nothing is normalized.
    pub text_fields: BTreeSet<String>,
}

impl IndexProfile {
    pub fn english() -> Self {
        IndexProfile {
            scalar_indexing: crate::text_tag_profile::ScalarIndexing::All,
            analysis: Analysis::English,
            text_fields: BTreeSet::new(),
        }
    }

    /// Whether schema `text` field `field` is indexed as text.
    pub fn indexes_text_field(&self, field: &str) -> bool {
        match self.analysis {
            Analysis::English => true,
            Analysis::Parity => self.text_fields.contains(field),
        }
    }

    pub fn stems(&self) -> bool {
        self.analysis == Analysis::English
    }

    pub fn normalizes_whitespace(&self) -> bool {
        self.analysis == Analysis::Parity
    }

    /// The record written into the index (Redis marker key, OpenSearch
    /// `_meta.vdbb_fulltext`) and required to match on reuse.
    pub fn marker(&self) -> Value {
        let mut marker = json!({
            "analysis": self.analysis.as_str(),
            "normalize_whitespace": self.normalizes_whitespace(),
            "text_fields": self.text_fields.iter().collect::<Vec<_>>(),
        });
        if self.scalar_indexing.is_reduced() {
            marker["scalar_indexing"] = json!(self.scalar_indexing);
        }
        marker
    }

    /// Check an index's recorded marker against this profile. `Null` means the
    /// index carries none — accepted only in english mode (indexes built before
    /// markers existed, or by a vector-only config, are english).
    pub fn check_marker(&self, recorded: &Value) -> Result<(), String> {
        let ok = match (self.analysis, recorded) {
            (Analysis::English, Value::Null) => true,
            (_, r) => *r == self.marker(),
        };
        if ok {
            Ok(())
        } else {
            Err(format!(
                "fulltext: the index records analysis {recorded} but this run needs {} — an \
                 index uploaded {} whitespace normalization (or with a different text field set) \
                 cannot be reused; rebuild it without --skip-upload",
                self.marker(),
                if self.normalizes_whitespace() {
                    "without"
                } else {
                    "with"
                }
            ))
        }
    }
}

/// Resolve the [`IndexProfile`] an engine config's search configs need.
/// `configs` are every search config's `(workload, fulltext block)`.
/// `None` when no search config is fulltext. Errors (all at startup):
/// - fulltext configs disagreeing on `analysis` (they need different indexes);
/// - parity without explicit `fields`, or parity configs with different field
///   sets (the indexed text fields are exactly the searched ones);
/// - parity sharing the engine config with any non-fulltext workload: the
///   normalization and the reduced text index must never apply to vector,
///   hybrid or filter-only configs.
pub fn index_profile<'a>(
    configs: impl IntoIterator<Item = (Workload, Option<&'a Value>)>,
) -> Result<Option<IndexProfile>, String> {
    let mut scalar_indexing = None;
    let mut analysis: Option<Analysis> = None;
    let mut parity_fields: Option<BTreeSet<String>> = None;
    let mut other_workload: Option<Workload> = None;
    for (i, (w, block)) in configs.into_iter().enumerate() {
        // `hybrid-rrf` searches the same text fields through the same block
        // (its BM25 arm), so it takes part in the profile like `fulltext`.
        if !w.uses_text_index() {
            other_workload.get_or_insert(w);
            continue;
        }
        let spec = parse_spec(block)?;
        if scalar_indexing.is_some_and(|old| old != spec.scalar_indexing) {
            return Err("fulltext configs disagree on scalar_indexing".into());
        }
        scalar_indexing = Some(spec.scalar_indexing);
        match analysis {
            Some(a) if a != spec.analysis => {
                return Err(format!(
                    "engine config mixes fulltext analysis \"{}\" and \"{}\" (search config \
                     #{i}); they need different indexes — split them into separate engine configs",
                    a.as_str(),
                    spec.analysis.as_str()
                ))
            }
            _ => analysis = Some(spec.analysis),
        }
        if spec.analysis == Analysis::Parity {
            let fields: BTreeSet<String> = spec
                .fields
                .as_ref()
                .ok_or_else(|| {
                    format!(
                        "fulltext search config #{i}: `analysis: parity` needs explicit \
                         `fields` — only the searched fields are indexed as text"
                    )
                })?
                .keys()
                .cloned()
                .collect();
            match &parity_fields {
                Some(prev) if *prev != fields => {
                    return Err(format!(
                        "parity fulltext search configs search different field sets ({prev:?} \
                         vs {fields:?}); in parity mode the indexed text fields are exactly the \
                         searched ones, so they need different indexes — split them"
                    ))
                }
                _ => parity_fields = Some(fields),
            }
        }
    }
    let Some(analysis) = analysis else {
        return Ok(None);
    };
    if analysis == Analysis::Parity {
        if let Some(w) = other_workload {
            return Err(format!(
                "`analysis: parity` changes the uploaded corpus (whitespace normalized) and the \
                 text index, which must never apply to other workloads; this engine config \
                 also declares \"{w}\" — split it"
            ));
        }
    }
    Ok(Some(IndexProfile {
        scalar_indexing: scalar_indexing.unwrap_or_default(),
        analysis,
        text_fields: parity_fields.unwrap_or_default(),
    }))
}

/// Prefix of every Redis marker key ([`redis_marker_key`]).
pub const REDIS_MARKER_NAMESPACE: &str = "vdbb:fulltext-profile:";

/// `REDIS_KEY_PREFIX` as the Redis engine resolves it (trimmed, `:`-terminated,
/// unset when blank).
pub fn resolve_shared_prefix(raw: Option<&str>) -> Option<String> {
    let s = raw?.trim();
    if s.is_empty() {
        None
    } else if s.ends_with(':') {
        Some(s.to_string())
    } else {
        Some(format!("{s}:"))
    }
}

/// Startup (and configure) check for a Redis shared corpus (`REDIS_KEY_PREFIX`,
/// reused by every config of a sweep), for an engine config with a fulltext
/// profile:
/// - parity normalizes the uploaded text, which the shared corpus would impose
///   on every other config;
/// - a shared prefix that covers the marker keys (e.g. `vdbb`) would make the
///   corpus-completeness count (`SCAN <prefix>*`) count markers as documents.
pub fn check_redis_shared_prefix(
    engine_type: &str,
    profile: Option<&IndexProfile>,
    shared_prefix: Option<&str>,
) -> Result<(), String> {
    let (Some(prefix), Some(profile)) = (shared_prefix, profile) else {
        return Ok(());
    };
    if engine_type != "redis" {
        return Ok(());
    }
    if profile.normalizes_whitespace() || profile.scalar_indexing.is_reduced() {
        return Err(
            "fulltext `analysis: parity` normalizes the uploaded text, which a shared corpus \
             (REDIS_KEY_PREFIX) would impose on every other config — unset REDIS_KEY_PREFIX"
                .into(),
        );
    }
    if REDIS_MARKER_NAMESPACE.starts_with(prefix) || prefix.starts_with(REDIS_MARKER_NAMESPACE) {
        return Err(format!(
            "REDIS_KEY_PREFIX '{prefix}' overlaps the fulltext marker keys \
             ({REDIS_MARKER_NAMESPACE}…), which the shared-corpus completeness count would \
             then count as documents — choose another prefix"
        ));
    }
    Ok(())
}

/// Redis commands that keep an index's marker in sync when `FT.CREATE`
/// (re)builds it. Without a fulltext profile: none — a vector-only config never
/// touches a key outside its own prefix (an ACL user limited to that prefix
/// must keep working). With one: clear any stale marker, write the profile's.
pub fn redis_marker_create_commands(
    index_name: &str,
    profile: Option<&IndexProfile>,
) -> Vec<Vec<String>> {
    let Some(p) = profile else {
        return Vec::new();
    };
    let key = redis_marker_key(index_name);
    vec![
        vec!["DEL".into(), key.clone()],
        vec!["SET".into(), key, p.marker().to_string()],
    ]
}

/// What a failed marker command (write or clear at create time, read before
/// the search) means for `profile`'s mode.
///
/// - English: the marker is bookkeeping, not part of what is measured, so a
///   failure (typically NOPERM: an ACL user limited to its key prefix cannot
///   reach `vdbb:fulltext-profile:*`) is non-fatal — `Ok(note)` to print, and
///   the result records `index_marker: null`, as for a pre-marker index.
/// - Parity: the marker is what proves the uploaded corpus was normalized, so
///   the run stops, naming the access it needs.
pub fn redis_marker_failure(
    profile: &IndexProfile,
    operation: &str,
    error: &str,
) -> Result<String, String> {
    match profile.analysis {
        Analysis::English => Ok(format!(
            "\tfulltext: could not {operation} the index marker ({error}); english mode \
             continues and records index_marker: null"
        )),
        Analysis::Parity => Err(format!(
            "fulltext parity: could not {operation} the index marker ({error}). Parity records \
             the upload normalization in the key {REDIS_MARKER_NAMESPACE}<index> and verifies it on \
             reuse, so the Redis user needs read/write access to the key pattern \
             '{REDIS_MARKER_NAMESPACE}*' (an ACL limited to the config's key prefix denies it)"
        )),
    }
}

/// Redis commands that remove an index's marker at teardown: only for an
/// engine config that has a fulltext profile (see
/// [`redis_marker_create_commands`]).
pub fn redis_marker_teardown_commands(
    index_name: &str,
    profile: Option<&IndexProfile>,
) -> Vec<Vec<String>> {
    match profile {
        Some(_) => vec![vec!["DEL".into(), redis_marker_key(index_name)]],
        None => Vec::new(),
    }
}

/// Whether `c` is whitespace that [`normalize_whitespace`] rewrites: every
/// Unicode `White_Space` character except U+0020 itself. RediSearch splits
/// tokens only on space and tab (and its ASCII punctuation set), OpenSearch's
/// standard tokenizer on all of them; in the 100K MS MARCO corpus `segment`
/// carries `\n`, `\r`, U+00A0, U+2028, `\x0B`, `\x0C` and U+0085.
pub fn is_normalized_whitespace(c: char) -> bool {
    c.is_whitespace() && c != ' '
}

/// Replace every Unicode whitespace character other than U+0020 (newlines,
/// tab, vertical tab, form feed, NEL, no-break space, line/paragraph
/// separators, …) by a space (parity mode, at upload).
pub fn normalize_whitespace(text: &str) -> String {
    text.chars()
        .map(|c| if is_normalized_whitespace(c) { ' ' } else { c })
        .collect()
}

/// Normalize the string values of `fields` in place; returns how many values
/// changed (for the upload log). A value of an indexed text field that is not
/// a string (a JSON array becomes `Labels`, which Redis joins with `;` and
/// OpenSearch indexes as several values) is refused: parity could not claim
/// `normalize_whitespace` for it, and the two engines would not index it alike.
pub fn normalize_metadata(
    items: &mut [Option<crate::readers::metadata::MetadataItem>],
    fields: &BTreeSet<String>,
) -> Result<usize, String> {
    use crate::readers::metadata::MetadataValue;
    let mut changed = 0;
    for (row, item) in items.iter_mut().enumerate() {
        let Some(item) = item else { continue };
        for (k, v) in item.fields.iter_mut() {
            if !fields.contains(k) {
                continue;
            }
            match v {
                MetadataValue::String(s) => {
                    if s.chars().any(is_normalized_whitespace) {
                        *s = normalize_whitespace(s);
                        changed += 1;
                    }
                }
                other => {
                    return Err(format!(
                        "fulltext parity: corpus row {row} field '{k}' is not a string \
                         ({other:?}); parity normalizes and indexes plain text fields only"
                    ))
                }
            }
        }
    }
    Ok(changed)
}

/// Redis key holding an index's [`IndexProfile::marker`] (outside every
/// config's key prefix, so it is neither indexed nor counted as corpus).
pub fn redis_marker_key(index_name: &str) -> String {
    format!("{REDIS_MARKER_NAMESPACE}{index_name}")
}

/// Key of the profile marker inside OpenSearch `mappings._meta`.
pub const OPENSEARCH_META_KEY: &str = "vdbb_fulltext";

/// Key of the block inside a search config.
pub const SPEC_KEY: &str = "fulltext";

/// Startup check, before anything is downloaded or connected: a `fulltext`
/// block is only legal on a `fulltext` search config, must parse, and may only
/// carry a key the engine reads (`allow_multi_shard` is OpenSearch-only — on
/// any other engine it would be accepted and do nothing).
pub fn validate_declared(
    engine_type: &str,
    workload: Workload,
    block: Option<&Value>,
) -> Result<(), String> {
    match (workload, block) {
        (Workload::Fulltext | Workload::HybridRrf, b) => {
            let spec = parse_spec(b)?;
            if workload == Workload::HybridRrf {
                crate::hybrid_rrf::check_text_spec(&spec)?;
            }
            if spec.allow_multi_shard && engine_type != "opensearch" {
                return Err(format!(
                    "`{SPEC_KEY}.allow_multi_shard` only applies to OpenSearch; engine \
                     '{engine_type}' has no shards to allow — remove it"
                ));
            }
            Ok(())
        }
        (w, Some(_)) => Err(format!(
            "a `{SPEC_KEY}` block is only valid on a fulltext or hybrid-rrf search config, not \
             workload \"{w}\""
        )),
        (_, None) => Ok(()),
    }
}

/// The fulltext harness schedules a declared number of requests. Refuse time
/// and offered-rate knobs it does not implement rather than mislabel a run.
pub fn check_execution_params(
    target_qps: Option<f64>,
    duration_seconds: Option<f64>,
) -> Result<(), String> {
    if target_qps.is_some_and(|v| v != 0.0) || duration_seconds.is_some_and(|v| v != 0.0) {
        return Err(format!(
            "fulltext supports query-count scheduling only; use --queries, not \
             target-qps/search-duration (an {} window runs for a time with \
             `compound.duration_s`)",
            crate::msmarco::disk_queries::SUITE
        ));
    }
    Ok(())
}

/// Every field a query's `conditions` filter on, whatever the condition type
/// (`match.text`, `match.value`, `match.any`, `range`, `geo`, …), through any
/// nesting of `and` / `or` groups. An entry carrying `and` / `or` is a group;
/// any other key of an entry is a field.
pub fn filtered_fields(conditions: &Value) -> BTreeSet<String> {
    fn walk(v: &Value, out: &mut BTreeSet<String>) {
        match v {
            Value::Array(a) => a.iter().for_each(|e| walk(e, out)),
            Value::Object(o) if o.contains_key("and") || o.contains_key("or") => {
                for key in ["and", "or"] {
                    if let Some(g) = o.get(key) {
                        walk(g, out);
                    }
                }
            }
            Value::Object(o) => out.extend(o.keys().cloned()),
            _ => {}
        }
    }
    let mut out = BTreeSet::new();
    walk(conditions, &mut out);
    out
}

/// Queries of `<dir>/tests.jsonl` whose `conditions` filter on any of
/// `text_fields` (any condition type — on OpenSearch `match.value` is an
/// analyzed `match` and `match.any` a `terms` query, so the analyzer changes
/// what either matches on a text field). A dataset without `tests.jsonl`
/// (HDF5 / JSONL formats) has no conditions.
pub fn count_text_field_queries(dir: &Path, text_fields: &[String]) -> Result<usize, String> {
    let path = dir.join("tests.jsonl");
    if !path.is_file() || text_fields.is_empty() {
        return Ok(0);
    }
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("fulltext: cannot open {}: {e}", path.display()))?;
    let mut n = 0;
    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("fulltext: reading {}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .map_err(|e| format!("fulltext: {} line {}: {e}", path.display(), i + 1))?;
        if row
            .get("conditions")
            .is_some_and(|c| filtered_fields(c).iter().any(|f| text_fields.contains(f)))
        {
            n += 1;
        }
    }
    Ok(n)
}

/// Pre-connect check (#339 review): every fulltext search config's `fields`
/// resolve against the dataset schema, so a typo is refused before configure
/// and upload rather than after a corpus upload. `blocks` are the `fulltext`
/// blocks of the engine config's fulltext search configs, each paired with its
/// index among all of the engine config's search configs, so the error names
/// the same `search config #i` as the startup check.
pub fn check_fields_before_connecting<'a>(
    blocks: impl Iterator<Item = (usize, Option<&'a Value>)>,
    schema: Option<&Value>,
    dataset: &str,
) -> Result<(), String> {
    for (i, block) in blocks {
        let spec = parse_spec(block)?;
        if let Some(compound) = &spec.compound {
            if compound.queries.is_some() {
                compound.load(
                    Path::new("."),
                    &spec
                        .scalar_indexing
                        .project(schema.ok_or("compound needs schema")?),
                )?;
            }
        }
        resolve_fields(&spec, schema)
            .map_err(|e| format!("search config #{i} (fulltext) on dataset '{dataset}': {e}"))?;
        resolve_fetch(&spec, schema)
            .map_err(|e| format!("search config #{i} (fulltext) on dataset '{dataset}': {e}"))?;
    }
    Ok(())
}

/// `msmarco-disk-v1` windows already claimed in one invocation, so no two
/// search configs run the same queries against one engine: within one engine
/// config ([`check_compound_before_connecting`]) and across the engine configs
/// of the invocation ([`check_disk_windows_across_configs`]).
#[derive(Debug, Default)]
pub struct DiskWindows {
    claimed: Vec<ClaimedWindow>,
}

#[derive(Debug)]
struct ClaimedWindow {
    /// Engine type and dataset: the cache a window warms. `--host` is one per
    /// invocation, so it is the same for every claim.
    scope: String,
    /// The engine config that claimed it (`None`: this engine config).
    owner: Option<String>,
    search_config: usize,
    family: &'static str,
    first: usize,
    last: usize,
}

/// Validate/generate every compound request before engine creation/upload.
/// `blocks` are each fulltext search config's block and result size `top`;
/// `num_queries` is `--queries`. `msmarco-disk-v1` windows are loaded as the
/// run will load them, and two windows of one family in the same engine
/// config must not overlap (each would repeat the other's queries).
pub fn check_compound_before_connecting<'a>(
    dir: &Path,
    blocks: impl Iterator<Item = (Option<&'a Value>, usize)>,
    schema: Option<&Value>,
    num_queries: i64,
) -> Result<(), String> {
    claim_compound_windows(
        dir,
        blocks,
        schema,
        num_queries,
        &mut DiskWindows::default(),
        "",
        None,
    )
}

/// [`check_compound_before_connecting`], recording each `msmarco-disk-v1`
/// window in `claimed` under `scope` (engine type + dataset). A window of a
/// family overlapping one already claimed in the same scope is refused,
/// whether the earlier claim came from this engine config or (`owner` naming
/// this one) from an earlier engine config of the invocation. Different
/// scopes (a Redis and an OpenSearch config, two datasets) may share a window.
pub fn claim_compound_windows<'a>(
    dir: &Path,
    blocks: impl Iterator<Item = (Option<&'a Value>, usize)>,
    schema: Option<&Value>,
    num_queries: i64,
    claimed: &mut DiskWindows,
    scope: &str,
    owner: Option<&str>,
) -> Result<(), String> {
    for (i, (block, top)) in blocks.enumerate() {
        let spec = parse_spec(block)?;
        if spec.compound.is_none() {
            continue;
        }
        let set = load_query_set_for(dir, &spec, schema, top, num_queries)?;
        let Some(w) = &set.window else {
            continue;
        };
        let first = set.queries[w.timed_from].query_id;
        let last = set.queries[set.queries.len() - 1].query_id;
        let family = w.family.name();
        if let Some(c) = claimed
            .claimed
            .iter()
            .find(|c| c.scope == scope && c.family == family && first <= c.last && c.first <= last)
        {
            let name = |o: Option<&str>, n: usize| match o {
                Some(o) => format!("engine config '{o}' search config #{n}"),
                None => format!("search config #{n}"),
            };
            let on = if scope.is_empty() {
                String::new()
            } else {
                format!(" on {scope}")
            };
            return Err(format!(
                "fulltext {} and {} both run {} family {family}{on} over overlapping lines \
                 ({}..={} and {first}..={last}); give each its own `compound.offset` range: \
                 a disk-bound window never repeats a query, and the second would be served \
                 from the cache the first warmed",
                name(c.owner.as_deref(), c.search_config),
                name(owner, i),
                crate::msmarco::disk_queries::SUITE,
                c.first,
                c.last,
            ));
        }
        claimed.claimed.push(ClaimedWindow {
            scope: scope.to_string(),
            owner: owner.map(str::to_string),
            search_config: i,
            family,
            first,
            last,
        });
    }
    Ok(())
}

/// One engine config's fulltext search blocks, for
/// [`check_disk_windows_across_configs`].
pub struct DiskConfig<'a> {
    pub name: &'a str,
    pub engine_type: &'a str,
    /// Each fulltext search config's block and result size `top`.
    pub blocks: Vec<(Option<&'a Value>, usize)>,
}

/// A dataset a [`DiskConfig`] runs on: name, directory, schema.
pub type DiskDataset = (String, std::path::PathBuf, Option<Value>);

/// Refuse, before any engine config of the invocation connects, two engine
/// configs of one engine type that run overlapping `msmarco-disk-v1` windows
/// of one family on one dataset: the second would serve the first's queries
/// from the cache the first warmed (`--host` is one per invocation). Fewer
/// than two configs with a disk-suite block (every existing config has none)
/// returns before `datasets` is called. Only disk-suite blocks are claimed;
/// each config's own check still validates all of its compound blocks.
pub fn check_disk_windows_across_configs(
    configs: &[DiskConfig<'_>],
    datasets: impl FnOnce() -> Result<Vec<DiskDataset>, String>,
    num_queries: i64,
) -> Result<(), String> {
    let is_disk = |b: Option<&Value>| parse_spec(b).is_ok_and(|s| is_disk_suite(&s));
    let disk: Vec<&DiskConfig<'_>> = configs
        .iter()
        .filter(|c| c.blocks.iter().any(|&(b, _)| is_disk(b)))
        .collect();
    if disk.len() < 2 {
        return Ok(());
    }
    let datasets = datasets()?;
    let mut claimed = DiskWindows::default();
    for c in disk {
        // Non-disk blocks become `None` (no compound), keeping each block's
        // search-config index for the message.
        let blocks: Vec<(Option<&Value>, usize)> = c
            .blocks
            .iter()
            .map(|&(b, top)| (b.filter(|b| is_disk(Some(b))), top))
            .collect();
        for (dataset, dir, schema) in &datasets {
            // A pair whose windows do not load (a dataset without disk query
            // files, a window past the end) is refused by its own per-config
            // check when it runs, as before; it never runs a window here.
            if check_compound_before_connecting(
                dir,
                blocks.iter().copied(),
                schema.as_ref(),
                num_queries,
            )
            .is_err()
            {
                continue;
            }
            claim_compound_windows(
                dir,
                blocks.iter().copied(),
                schema.as_ref(),
                num_queries,
                &mut claimed,
                &format!("{} on dataset '{dataset}'", c.engine_type),
                Some(c.name),
            )?;
        }
    }
    Ok(())
}

/// Refuse `--repetitions` above 1 for a disk-bound suite: every repetition
/// would run the same window again, from cache.
pub fn check_disk_repetitions<'a>(
    blocks: impl Iterator<Item = Option<&'a Value>>,
    repetitions: usize,
) -> Result<(), String> {
    for block in blocks {
        if repetitions > 1 && is_disk_suite(&parse_spec(block)?) {
            return Err(format!(
                "--repetitions {repetitions} would rerun the same {} window; run each \
                 repetition as its own invocation with its own `compound.offset`",
                crate::msmarco::disk_queries::SUITE
            ));
        }
    }
    Ok(())
}

/// Whether an engine config's workloads need [`check_mixed_text_filters`]: a
/// fulltext config sharing the index with a vector or hybrid config.
pub fn mixes_with_filtered_workloads(workloads: &[Workload]) -> bool {
    workloads.iter().any(|w| w.uses_text_index())
        && workloads
            .iter()
            .any(|w| matches!(w, Workload::Vector | Workload::Hybrid))
}

/// Refuse an engine config that mixes `fulltext` with `vector` / `hybrid`
/// search configs on a dataset whose queries filter on a schema `text` field
/// (any condition type, see [`count_text_field_queries`]).
///
/// A fulltext search config builds the index's text fields with the pinned
/// analyzer. On OpenSearch that replaces `standard` (no stemming, no
/// stopwords), so the vector configs' filters on those fields would
/// silently match more documents than a vector-only config of the same name —
/// and recall would still be scored against the same filtered ground truth.
/// On Redis the pinned settings equal the defaults, but one rule for both
/// engines keeps the comparison symmetric. Split the configs instead.
pub fn check_mixed_text_filters(
    workloads: &[Workload],
    text_match_queries: usize,
    dataset: &str,
) -> Result<(), String> {
    if mixes_with_filtered_workloads(workloads) && text_match_queries > 0 {
        return Err(format!(
            "engine config mixes a fulltext search config with vector/hybrid search configs, \
             and {text_match_queries} queries of dataset '{dataset}' filter on text fields \
             (any condition on a schema `text` field): the fulltext config pins the text \
             analyzer for the whole \
             index, which would change what those filters match — split the fulltext search \
             configs into their own engine config"
        ));
    }
    Ok(())
}

/// Parse a search config's `fulltext` block (absent = defaults).
pub fn parse_spec(block: Option<&Value>) -> Result<FulltextSpec, String> {
    let spec: FulltextSpec = match block {
        None => FulltextSpec::default(),
        Some(v) => serde_json::from_value(v.clone())
            .map_err(|e| format!("invalid `{SPEC_KEY}` block: {e}"))?,
    };
    if spec.scalar_indexing.is_reduced()
        && (spec.compound.is_none() || !spec.no_vector_index || spec.analysis != Analysis::Parity)
    {
        return Err(
            "text-tag-only requires parity compound fulltext and no_vector_index=true".into(),
        );
    }
    if let Some(fields) = &spec.fields {
        if fields.is_empty() {
            return Err(format!("`{SPEC_KEY}.fields` must name at least one field"));
        }
        for (name, boost) in fields {
            if !boost.is_finite() || *boost <= 0.0 {
                return Err(format!(
                    "`{SPEC_KEY}.fields.{name}` boost must be a finite number > 0, got {boost}"
                ));
            }
        }
    }
    check_total_hits(&spec)?;
    if let Some(fetch) = &spec.fetch {
        check_fetch_names(fetch)?;
    }
    if is_disk_suite(&spec) && spec.analysis != Analysis::Parity {
        return Err(format!(
            "compound suite {} needs `analysis: parity`: its exact counts are document \
             frequencies of unstemmed, whitespace-normalized terms",
            crate::msmarco::disk_queries::SUITE
        ));
    }
    if let Some(compound) = &spec.compound {
        compound.validate()?;
        if spec
            .fields
            .as_ref()
            .is_none_or(|f| f.values().any(|b| *b != 1.0))
        {
            return Err("compound matching requires explicit text fields with unit boosts".into());
        }
    }
    Ok(spec)
}

/// Names-only check of `fulltext.fetch` (the schema check is
/// [`resolve_fetch`]): nonempty, unique, plain identifiers, never `vector`.
fn check_fetch_names(fetch: &[String]) -> Result<(), String> {
    if fetch.is_empty() {
        return Err(format!(
            "`{SPEC_KEY}.fetch` must name at least one field (omit it to fetch ids only)"
        ));
    }
    let mut seen = HashSet::new();
    for f in fetch {
        if !is_plain_field_name(f) || f == "vector" {
            return Err(format!(
                "`{SPEC_KEY}.fetch` field {f:?} must be a plain identifier and never the vector"
            ));
        }
        if !seen.insert(f) {
            return Err(format!("`{SPEC_KEY}.fetch` names {f:?} twice"));
        }
    }
    Ok(())
}

/// The fields to fetch, checked against the dataset schema: each a `text` or
/// `keyword` field (string-valued, stored by both engines). `None` = ids only.
pub fn resolve_fetch(
    spec: &FulltextSpec,
    schema: Option<&Value>,
) -> Result<Option<Vec<String>>, String> {
    let Some(fetch) = &spec.fetch else {
        return Ok(None);
    };
    check_fetch_names(fetch)?;
    for f in fetch {
        match schema.and_then(|s| s.get(f)).and_then(Value::as_str) {
            Some("text" | "keyword") => {}
            other => {
                return Err(format!(
                    "`{SPEC_KEY}.fetch` field {f:?} must be a `text` or `keyword` field of the \
                     dataset schema (got {other:?})"
                ))
            }
        }
    }
    Ok(Some(fetch.clone()))
}

/// One searched field and its query-time boost.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldBoost {
    pub field: String,
    pub boost: f64,
}

/// The `text` fields of a dataset schema, sorted by name (a dependency may
/// enable serde_json's `preserve_order`, so map order is not relied on).
pub fn schema_text_fields(schema: Option<&Value>) -> Vec<String> {
    let mut fields: Vec<String> = schema
        .and_then(Value::as_object)
        .map(|o| {
            o.iter()
                .filter(|(_, t)| t.as_str() == Some("text"))
                .map(|(k, _)| k.clone())
                .collect()
        })
        .unwrap_or_default();
    fields.sort();
    fields
}

/// Resolve the searched fields against the dataset schema. Every configured
/// field must be a `text` field of the schema — a typo would otherwise search
/// a field that does not exist, which both engines answer with silence.
/// Whether `name` can be written as `@name` in a RediSearch query without
/// escaping: ASCII letters, digits and `_`, not starting with a digit.
pub fn is_plain_field_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub fn resolve_fields(
    spec: &FulltextSpec,
    schema: Option<&Value>,
) -> Result<Vec<FieldBoost>, String> {
    let fields = resolve_schema_fields(spec, schema)?;
    // `redis_query` writes `@field:` unescaped; a name with `-`, `.` or other
    // query syntax would make every query a syntax error. Refused here, which
    // the runner calls before connecting.
    if let Some(bad) = fields.iter().find(|f| !is_plain_field_name(&f.field)) {
        return Err(format!(
            "fulltext field '{}' is not a plain identifier (ASCII letters, digits, `_`, not \
             starting with a digit) — it cannot be written unescaped in a RediSearch query",
            bad.field
        ));
    }
    Ok(fields)
}

fn resolve_schema_fields(
    spec: &FulltextSpec,
    schema: Option<&Value>,
) -> Result<Vec<FieldBoost>, String> {
    let text_fields = schema_text_fields(schema);
    if text_fields.is_empty() {
        return Err("fulltext workload needs a dataset whose schema declares `text` fields".into());
    }
    match &spec.fields {
        None => Ok(text_fields
            .into_iter()
            .map(|field| FieldBoost { field, boost: 1.0 })
            .collect()),
        Some(fields) => fields
            .iter()
            .map(|(field, boost)| {
                if text_fields.contains(field) {
                    Ok(FieldBoost {
                        field: field.clone(),
                        boost: *boost,
                    })
                } else {
                    Err(format!(
                        "fulltext field '{field}' is not a `text` field of the dataset schema \
                         (text fields: {})",
                        text_fields.join(", ")
                    ))
                }
            })
            .collect(),
    }
}

fn is_separator(c: char) -> bool {
    c.is_whitespace() || REDISEARCH_SEPARATORS.contains(c)
}

/// Drop a possessive `'s` / `’s` (followed by the end or a non-alphanumeric
/// character) before splitting. Splitting `dog's` on `'` would otherwise send
/// the term `s` to both engines: on Redis it matches every possessive in the
/// corpus (RediSearch indexes `'s` as a separate `s` token), on OpenSearch
/// almost nothing (`possessive_english` removed them at index time). Stripping
/// it on the query side, which both engines share, removes that asymmetry
/// from the query; it mirrors what `possessive_english` does.
pub fn strip_possessives(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let is_possessive = (c == '\'' || c == '\u{2019}')
            && i > 0
            && chars[i - 1].is_alphanumeric()
            && chars.get(i + 1).is_some_and(|n| *n == 's' || *n == 'S')
            && chars.get(i + 2).is_none_or(|n| !n.is_alphanumeric());
        if is_possessive {
            i += 2;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// The query terms both engines receive (see the module docs).
pub fn query_terms(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    strip_possessives(text)
        .split(is_separator)
        .map(str::to_lowercase)
        .filter(|t| t.chars().any(char::is_alphanumeric))
        .filter(|t| !STOPWORDS.contains(&t.as_str()))
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

/// One runnable query.
#[derive(Debug, Clone, PartialEq)]
pub struct TextQuery {
    pub compound: Option<crate::compound_text::Query>,
    pub expression: Option<crate::compound_text::Expr>,
    /// Line of `tests.jsonl` (the dataset query index).
    pub query_id: usize,
    pub terms: Vec<String>,
}

/// The query set derived from `tests.jsonl`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuerySet {
    pub queries: Vec<TextQuery>,
    /// SHA-256 of the sealed PREPARED.json; absent for unprepared tiny fixtures.
    pub preparation_sha256: Option<String>,
    /// Query ids with no term left after [`query_terms`]; skipped on both engines.
    pub skipped: Vec<usize>,
    /// Lines in `tests.jsonl`.
    pub total: usize,
    /// `msmarco-disk-v1`: the loaded window (see [`DiskWindow`]). `None` for
    /// every other query set, which keeps the legacy schedule.
    pub window: Option<DiskWindow>,
}

/// A `msmarco-disk-v1` window inside a [`QuerySet`]: `queries[0]` is the
/// connection-priming query, `queries[1..timed_from]` the `WITHOUTCOUNT`
/// probe reserve, and `queries[timed_from..]` the timed window, each
/// dispatched exactly once. `query_id` is the line of the family file.
#[derive(Debug, Clone, PartialEq)]
pub struct DiskWindow {
    pub family: crate::msmarco::disk_queries::Family,
    pub timed_from: usize,
    /// [`crate::msmarco::disk_queries::Window::record`].
    pub record: Value,
    /// `compound.duration_s`: cycle the timed window for this many seconds
    /// ([`crate::duration_mode`]). `None`: each timed query once.
    pub duration_s: Option<f64>,
}

/// Which queries the Redis `WITHOUTCOUNT` probe may send.
#[derive(Debug, Clone, PartialEq)]
pub enum ProbePlan {
    /// Probe with these indices of [`QuerySet::queries`], in order.
    Run(Vec<usize>),
    /// No probe is needed; the reason is recorded.
    NotNeeded(&'static str),
}

/// Recorded when a disk window needs no `WITHOUTCOUNT` probe.
pub const PROBE_NOT_NEEDED: &str = "not needed: every timed query's exact match count is at \
     most the page size, so each page is the whole match set; the page length is checked \
     against the exact count on every reply";

impl QuerySet {
    /// The query each worker sends once to prime its connection, outside the
    /// timed window: query 0 (for a disk window, its reserved priming line).
    pub fn prime_index(&self) -> usize {
        0
    }

    /// Which entries of `queries` the timed window dispatches, in order. The
    /// legacy sets keep [`schedule`] (all once, or `num_queries` cycling); a
    /// disk window dispatches each of its timed queries exactly once — it was
    /// loaded with exactly `num_queries` of them, and anything else is refused
    /// — unless it runs in duration mode ([`Self::deadline`]), where the
    /// runner cycles this order until the deadline
    /// ([`crate::duration_mode::Dispatcher`]).
    pub fn dispatch_order(&self, num_queries: i64) -> Result<Vec<usize>, String> {
        let Some(w) = &self.window else {
            return Ok(schedule(self.queries.len(), num_queries));
        };
        let order: Vec<usize> = (w.timed_from..self.queries.len()).collect();
        if order.is_empty() || (num_queries > 0 && num_queries as usize != order.len()) {
            return Err(format!(
                "{}: the loaded window holds {} timed queries but --queries is {num_queries}; \
                 a disk-bound window never repeats a query",
                crate::msmarco::disk_queries::SUITE,
                order.len()
            ));
        }
        Ok(order)
    }

    /// How long a duration-mode disk window runs (`compound.duration_s`,
    /// validated by [`parse_spec`]); `None` for a single-pass window and every
    /// other query set.
    pub fn deadline(&self) -> Option<std::time::Duration> {
        self.window
            .as_ref()
            .and_then(|w| w.duration_s)
            .map(std::time::Duration::from_secs_f64)
    }

    /// The duration-mode record ([`crate::duration_mode::record`]) of a run of
    /// this set, stored at `compound.duration` of the result metadata. Does
    /// nothing for a set without a [`Self::deadline`].
    pub fn record_duration(
        &self,
        meta: &mut Value,
        dispatched: usize,
        completed_at: &[f64],
        latencies: &[f64],
        span_s: f64,
    ) {
        let Some(w) = &self.window else { return };
        let Some(d) = w.duration_s else { return };
        meta["compound"]["duration"] = crate::duration_mode::record(
            d,
            &w.record,
            self.queries.len() - w.timed_from,
            dispatched,
            completed_at,
            latencies,
            span_s,
        );
    }

    /// The queries [`probe_withoutcount`] may send: every query of a legacy
    /// set, the probe reserve of a disk window (never a timed query, which
    /// the probe would warm), or no probe when no timed query can be cut by
    /// the page.
    pub fn withoutcount_probe_plan(&self, top: usize) -> ProbePlan {
        let Some(w) = &self.window else {
            return ProbePlan::Run((0..self.queries.len()).collect());
        };
        let over = self.queries[w.timed_from..].iter().any(|q| {
            q.compound
                .as_ref()
                .is_none_or(|c| c.expected_count.is_none_or(|e| e > top as u64))
        });
        if over {
            ProbePlan::Run((1..w.timed_from).collect())
        } else {
            ProbePlan::NotNeeded(PROBE_NOT_NEEDED)
        }
    }

    /// For a disk window under `total_hits: "none"`, refuse — before
    /// connecting, and on both engines so the pair stays symmetric — a window
    /// whose `WITHOUTCOUNT` probe could not succeed: some timed query has more
    /// matches than `top` (so [`Self::withoutcount_probe_plan`] runs the
    /// probe) but no reserve line does (so [`probe_withoutcount`] would find
    /// no truncated page to compare). The generator puts the highest counts
    /// in the reserve ([`crate::msmarco::disk_queries::probe_layout`]), so
    /// this holds for any file it wrote; legacy sets have no reserve.
    pub fn check_probe_reserve(&self, top: usize) -> Result<(), String> {
        let Some(w) = &self.window else {
            return Ok(());
        };
        let count = |q: &TextQuery| q.compound.as_ref().and_then(|c| c.expected_count);
        let timed_max = self.queries[w.timed_from..]
            .iter()
            .map(count)
            .max()
            .flatten();
        let Some(timed_max) = timed_max.filter(|m| *m > top as u64) else {
            return Ok(());
        };
        if self.queries[1..w.timed_from]
            .iter()
            .any(|q| count(q).is_some_and(|c| c > top as u64))
        {
            return Ok(());
        }
        Err(format!(
            "{} family {} (seed {}): a timed query matches {timed_max} documents, more than \
             `top` {top}, but no probe-reserve line (1..{}) matches more than {top}, so the \
             Redis WITHOUTCOUNT probe could not compare a truncated page; `total_hits: \
             \"none\"` is refused on both engines for this window — regenerate the family \
             files with --add-disk-queries (generator {}), or use \"exact\"",
            crate::msmarco::disk_queries::SUITE,
            w.family.name(),
            w.record["seed"],
            w.timed_from - 1,
            crate::msmarco::disk_queries::GENERATOR
        ))
    }
}

/// Resolve the explicitly named compound suite independently of ranked BM25.
/// Not for `msmarco-disk-v1`, which is loaded per window
/// ([`load_query_set_for`]).
pub fn load_query_set(
    dir: &Path,
    spec: &FulltextSpec,
    schema: Option<&Value>,
) -> Result<QuerySet, String> {
    if is_disk_suite(spec) {
        return Err(format!(
            "{} needs the search config's top and --queries: use load_query_set_for",
            crate::msmarco::disk_queries::SUITE
        ));
    }
    load_query_set_for(dir, spec, schema, 0, -1)
}

/// [`load_query_set`] for one search config: `top` and `num_queries`
/// (`--queries`) size an `msmarco-disk-v1` window and check its family's
/// parameters; every other query set ignores them.
pub fn load_query_set_for(
    dir: &Path,
    spec: &FulltextSpec,
    schema: Option<&Value>,
    top: usize,
    num_queries: i64,
) -> Result<QuerySet, String> {
    let Some(compound) = &spec.compound else {
        return build_query_set(&read_query_texts(dir)?);
    };
    let schema = schema.ok_or("compound suite needs a schema")?;
    let projected = spec.scalar_indexing.project(schema);
    let schema = &projected;
    let fields = resolve_fields(spec, Some(schema))?;
    let check_text = |expression: &crate::compound_text::Expr| -> Result<(), String> {
        let mut used = HashSet::new();
        expression.text_fields(&mut used);
        if used.iter().any(|f| !fields.iter().any(|b| &b.field == f)) {
            return Err(
                "compound query references text outside the declared indexed fields".into(),
            );
        }
        Ok(())
    };
    if let Some(family) = crate::compound_text::disk_family(compound)? {
        family.check_params(
            top,
            spec.fetch.is_some(),
            spec.total_hits == Some(TotalHits::None),
        )?;
        let count = (num_queries > 0).then_some(num_queries as usize);
        let w = crate::msmarco::disk_queries::load_window(
            dir,
            family,
            compound.offset.unwrap_or(0),
            count,
        )?;
        let record = w.record();
        let timed_from = w.timed_from;
        let queries = w
            .queries
            .into_iter()
            .map(|(line, q)| {
                let expression = crate::compound_text::Expr::compile(&q.conditions, schema)?;
                check_text(&expression)?;
                Ok(TextQuery {
                    query_id: line,
                    terms: vec![],
                    compound: Some(q),
                    expression: Some(expression),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let set = QuerySet {
            total: queries.len(),
            queries,
            skipped: vec![],
            preparation_sha256: Some(crate::msmarco::disk_queries::preparation_sha256(dir)?),
            window: Some(DiskWindow {
                family,
                timed_from,
                record,
                duration_s: compound.duration_s,
            }),
        };
        if spec.total_hits == Some(TotalHits::None) {
            set.check_probe_reserve(top)?;
        }
        return Ok(set);
    }
    let queries = compound
        .load(dir, schema)?
        .into_iter()
        .enumerate()
        .map(|(query_id, mut q)| {
            // The literal classifier proves an exclusion only without stemming.
            // English may stem the excluded title term into an existing token.
            if compound.suite.as_deref() == Some("msmarco-v1")
                && spec.analysis == Analysis::English
                && q.family == "text-exclusion"
            {
                q.minimum_count = 0;
            }
            let expression = crate::compound_text::Expr::compile(&q.conditions, schema)?;
            check_text(&expression)?;
            Ok(TextQuery {
                query_id,
                terms: vec![],
                compound: Some(q),
                expression: Some(expression),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let preparation_sha256 = match std::fs::read(dir.join("PREPARED.json")) {
        Ok(bytes) => {
            use sha2::{Digest, Sha256};
            Some(
                Sha256::digest(bytes)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>(),
            )
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && compound.suite.is_none() => None,
        Err(e) => return Err(format!("compound preparation identity: {e}")),
    };
    Ok(QuerySet {
        total: queries.len(),
        queries,
        skipped: vec![],
        preparation_sha256,
        window: None,
    })
}

pub fn query_redis(fields: &[FieldBoost], q: &TextQuery) -> String {
    q.expression
        .as_ref()
        .map(|e| e.redis())
        .unwrap_or_else(|| redis_query(fields, &q.terms))
}
/// The OpenSearch request for `q` with the legacy totals: exact for a
/// compound query, none for the ranked one.
pub fn query_opensearch(fields: &[FieldBoost], q: &TextQuery, top: usize) -> Value {
    let legacy = if q.expression.is_some() {
        TotalHits::Exact
    } else {
        TotalHits::None
    };
    query_opensearch_with(fields, q, top, legacy)
}

/// The OpenSearch request for `q`, with `track_total_hits` set by `total`
/// (resolved by [`total_hits_plan`]).
pub fn query_opensearch_with(
    fields: &[FieldBoost],
    q: &TextQuery,
    top: usize,
    total: TotalHits,
) -> Value {
    query_opensearch_fetch(fields, q, top, total, None)
}

/// `_source` of an OpenSearch request: `false` (ids only) or the fetched
/// fields.
pub fn opensearch_source(fetch: Option<&[String]>) -> Value {
    match fetch {
        None => json!(false),
        Some(f) => json!({"includes": f}),
    }
}

/// [`query_opensearch_with`] fetching `fetch` (`None` = ids only, the same
/// request as before).
pub fn query_opensearch_fetch(
    fields: &[FieldBoost],
    q: &TextQuery,
    top: usize,
    total: TotalHits,
    fetch: Option<&[String]>,
) -> Value {
    let mut body = match &q.expression {
        Some(e) => json!({"size":top,"_source":false,
            "track_total_hits":opensearch_track_total_hits(total),"query":e.opensearch()}),
        None => opensearch_body_with(fields, &q.terms, top, total),
    };
    body["_source"] = opensearch_source(fetch);
    body
}

/// `track_total_hits` for `total`.
pub fn opensearch_track_total_hits(total: TotalHits) -> bool {
    total == TotalHits::Exact
}
pub fn check_query_count(q: &TextQuery, total: Option<u64>) -> Result<(), String> {
    if let Some(c) = &q.compound {
        crate::compound_text::check_count(
            c,
            total.ok_or("compound query requires an exact total")?,
        )?;
    }
    Ok(())
}

/// [`check_query_count`] for a validated page. With a total, the total is
/// checked. Without one (`total_hits: none`), a compound query with an exact
/// expected count must return exactly `min(expected, top)` hits — the page
/// parsers already refuse an incomplete or duplicated page — and one without
/// an expected count is refused, as before.
pub fn check_query_page(
    q: &TextQuery,
    total: Option<u64>,
    returned: usize,
    top: usize,
) -> Result<(), String> {
    let Some(c) = &q.compound else {
        return Ok(());
    };
    match (total, c.expected_count) {
        (Some(_), _) => check_query_count(q, total),
        (None, Some(e)) if returned as u64 == e.min(top as u64) => Ok(()),
        (None, Some(e)) => Err(format!(
            "compound {} returned {returned} hits without a total; its exact count {e} with \
             top {top} requires {}",
            c.name,
            e.min(top as u64)
        )),
        (None, None) => Err("compound query requires an exact total".into()),
    }
}

/// Build the query set from per-line query texts. No runnable query is an
/// error: there would be nothing to measure.
pub fn build_query_set(texts: &[String]) -> Result<QuerySet, String> {
    let mut queries = Vec::new();
    let mut skipped = Vec::new();
    for (query_id, text) in texts.iter().enumerate() {
        let terms = query_terms(text);
        if terms.is_empty() {
            skipped.push(query_id);
        } else {
            queries.push(TextQuery {
                query_id,
                terms,
                compound: None,
                expression: None,
            });
        }
    }
    if queries.is_empty() {
        return Err(format!(
            "fulltext: none of the {} query texts has a searchable term",
            texts.len()
        ));
    }
    Ok(QuerySet {
        queries,
        skipped,
        total: texts.len(),
        preparation_sha256: None,
        window: None,
    })
}

/// `query_text` of every line of `<dir>/tests.jsonl`, in line order. A line
/// without a non-empty string `query_text` is an error, never skipped: the
/// fulltext workload must not run on a dataset that cannot supply it.
pub fn read_query_texts(dir: &Path) -> Result<Vec<String>, String> {
    let path = dir.join("tests.jsonl");
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("fulltext: cannot open {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("fulltext: reading {}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .map_err(|e| format!("fulltext: {} line {}: {e}", path.display(), i + 1))?;
        match row.get("query_text").and_then(Value::as_str) {
            Some(t) if !t.trim().is_empty() => out.push(t.to_string()),
            _ => {
                return Err(format!(
                    "fulltext: {} line {} has no non-empty `query_text`",
                    path.display(),
                    i + 1
                ))
            }
        }
    }
    if out.is_empty() {
        return Err(format!("fulltext: {} has no queries", path.display()));
    }
    Ok(out)
}

/// Which entries of `queries` the timed window dispatches, in order:
/// all of them once, or `num_queries` of them cycling (`--queries N`).
pub fn schedule(num_eligible: usize, num_queries: i64) -> Vec<usize> {
    let n = if num_queries > 0 {
        num_queries as usize
    } else {
        num_eligible
    };
    (0..n).map(|i| i % num_eligible.max(1)).collect()
}

fn fmt_boost(b: f64) -> String {
    format!("{b:?}")
}

/// The RediSearch query string (DIALECT 2).
pub fn redis_query(fields: &[FieldBoost], terms: &[String]) -> String {
    let union = terms.join("|");
    fields
        .iter()
        .map(|f| {
            format!(
                "(@{}:({union})) => {{$weight: {};}}",
                f.field,
                fmt_boost(f.boost)
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Every `FT.SEARCH` argument after the command name.
pub fn redis_args(index: &str, query: &str, top: usize, timeout_ms: i64) -> Vec<String> {
    redis_args_for(index, query, top, timeout_ms, Analysis::English)
}

/// [`redis_args`] for `analysis`: parity adds `VERBATIM` (no query-time stem
/// expansion). Counts the total (the legacy Redis default).
pub fn redis_args_for(
    index: &str,
    query: &str,
    top: usize,
    timeout_ms: i64,
    analysis: Analysis,
) -> Vec<String> {
    redis_args_with(index, query, top, timeout_ms, analysis, TotalHits::Exact)
}

/// The `FT.SEARCH` flag that turns counting off. It keeps the BM25 ranking
/// only on RediSearch 8.4+ ([`TotalHits`]); [`probe_withoutcount`] checks that
/// on the live server before a `none` run uses it.
pub const REDIS_WITHOUTCOUNT: &str = "WITHOUTCOUNT";

/// [`redis_args_for`] with the total-hits mode resolved by
/// [`total_hits_plan`]: `none` appends [`REDIS_WITHOUTCOUNT`].
pub fn redis_args_with(
    index: &str,
    query: &str,
    top: usize,
    timeout_ms: i64,
    analysis: Analysis,
    total: TotalHits,
) -> Vec<String> {
    redis_args_fetch(index, query, top, timeout_ms, analysis, total, None)
}

/// [`redis_args_with`] fetching `fetch`: `RETURN <n> <fields…>` takes the
/// place of `NOCONTENT`. `None` is the same command as before.
pub fn redis_args_fetch(
    index: &str,
    query: &str,
    top: usize,
    timeout_ms: i64,
    analysis: Analysis,
    total: TotalHits,
    fetch: Option<&[String]>,
) -> Vec<String> {
    let mut args = redis_args_base(index, query, top, timeout_ms);
    if analysis == Analysis::Parity {
        args.insert(3, "VERBATIM".to_string());
    }
    if total == TotalHits::None {
        args.push(REDIS_WITHOUTCOUNT.to_string());
    }
    if let Some(fetch) = fetch {
        assert_eq!(args[2], "NOCONTENT", "redis_args_base layout changed");
        let mut ret = vec!["RETURN".to_string(), fetch.len().to_string()];
        ret.extend(fetch.iter().cloned());
        args.splice(2..3, ret);
    }
    args
}

fn redis_args_base(index: &str, query: &str, top: usize, timeout_ms: i64) -> Vec<String> {
    [
        index,
        query,
        "NOCONTENT",
        "WITHSCORES",
        "SCORER",
        REDIS_SCORER,
        "LIMIT",
        "0",
        &top.to_string(),
        "DIALECT",
        "2",
        "TIMEOUT",
        &timeout_ms.to_string(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The OpenSearch search body, without a total (the legacy OpenSearch
/// default).
pub fn opensearch_body(fields: &[FieldBoost], terms: &[String], top: usize) -> Value {
    opensearch_body_with(fields, terms, top, TotalHits::None)
}

/// [`opensearch_body`] with the total-hits mode resolved by
/// [`total_hits_plan`].
pub fn opensearch_body_with(
    fields: &[FieldBoost],
    terms: &[String],
    top: usize,
    total: TotalHits,
) -> Value {
    json!({
        "size": top,
        "_source": false,
        "track_total_hits": opensearch_track_total_hits(total),
        "query": {"multi_match": {
            "query": terms.join(" "),
            "type": OPENSEARCH_MULTI_MATCH_TYPE,
            "operator": "or",
            "fields": fields
                .iter()
                .map(|f| format!("{}^{}", f.field, fmt_boost(f.boost)))
                .collect::<Vec<_>>(),
        }},
    })
}

/// Name of the parity-mode OpenSearch analyzer (no stemmer).
pub const OPENSEARCH_PARITY_ANALYZER: &str = "vdbb_english_nostem";

/// The analyzer an index in `analysis` mode uses.
pub fn opensearch_analyzer_name(analysis: Analysis) -> &'static str {
    match analysis {
        Analysis::English => OPENSEARCH_ANALYZER,
        Analysis::Parity => OPENSEARCH_PARITY_ANALYZER,
    }
}

/// `settings.index.analysis` for `analysis` mode. Parity is the english
/// pipeline minus the porter2 filter.
pub fn opensearch_analysis_for(analysis: Analysis) -> Value {
    match analysis {
        Analysis::English => opensearch_analysis(),
        Analysis::Parity => json!({
            "filter": {
                OS_POSSESSIVE: {"type": "stemmer", "language": "possessive_english"},
                OS_STOP: {"type": "stop", "stopwords": STOPWORDS.as_slice()},
            },
            "analyzer": {
                OPENSEARCH_PARITY_ANALYZER: {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [OS_POSSESSIVE, "lowercase", OS_STOP],
                }
            }
        }),
    }
}

/// Mapping of a schema `text` field in an index built for `profile`: the
/// mode's analyzer when indexed, else stored but not indexed (parity).
pub fn opensearch_text_mapping_for(profile: &IndexProfile, field: &str) -> Value {
    if profile.indexes_text_field(field) {
        json!({"type": "text", "index": true,
               "analyzer": opensearch_analyzer_name(profile.analysis)})
    } else {
        json!({"type": "text", "index": false})
    }
}

/// `settings.index.analysis` for an OpenSearch index serving this workload.
pub fn opensearch_analysis() -> Value {
    json!({
        "filter": {
            OS_POSSESSIVE: {"type": "stemmer", "language": "possessive_english"},
            OS_STOP: {"type": "stop", "stopwords": STOPWORDS.as_slice()},
            OS_STEMMER: {"type": "stemmer", "language": "porter2"},
        },
        "analyzer": {
            OPENSEARCH_ANALYZER: {
                "type": "custom",
                "tokenizer": "standard",
                "filter": [OS_POSSESSIVE, "lowercase", OS_STOP, OS_STEMMER],
            }
        }
    })
}

/// Mapping of a `text` field in an OpenSearch index serving this workload.
pub fn opensearch_text_mapping() -> Value {
    json!({"type": "text", "index": true, "analyzer": OPENSEARCH_ANALYZER})
}

/// `FT.CREATE … STOPWORDS <n> <words…>` arguments, placed before `SCHEMA`.
pub fn redis_stopword_args() -> Vec<String> {
    std::iter::once("STOPWORDS".to_string())
        .chain(std::iter::once(STOPWORDS.len().to_string()))
        .chain(STOPWORDS.iter().map(|s| s.to_string()))
        .collect()
}

fn same_word_set<'a>(got: impl Iterator<Item = &'a str>) -> bool {
    let mut got: Vec<&str> = got.collect();
    got.sort_unstable();
    let mut want = STOPWORDS.to_vec();
    want.sort_unstable();
    got == want
}

/// Check a live Redis index (`FT.INFO` over RESP3, converted to JSON) serves
/// the fulltext workload as pinned: every searched field is an indexed,
/// stemmed TEXT field with index-time weight 1 (boosts are query time; an
/// index weight would multiply them), and the index stopword list is
/// [`STOPWORDS`]. Any mismatch is a hard error before the timed window.
pub fn validate_redis_index(info: &Value, fields: &[FieldBoost]) -> Result<(), String> {
    validate_redis_index_for(info, fields, &IndexProfile::english(), &Value::Null)
}

/// [`validate_redis_index`] for an [`IndexProfile`]. Parity additionally
/// requires every searched field NOSTEM (english refuses it), no other TEXT
/// field in the index, and `marker` (the value of [`redis_marker_key`],
/// parsed; `Null` when absent) equal to the profile's.
pub fn validate_redis_index_for(
    info: &Value,
    fields: &[FieldBoost],
    profile: &IndexProfile,
    marker: &Value,
) -> Result<(), String> {
    profile.check_marker(marker)?;
    let attributes = info["attributes"]
        .as_array()
        .ok_or("fulltext: FT.INFO has no attributes (RESP3 map expected)")?;
    for f in fields {
        let attr = attributes
            .iter()
            .find(|a| a["attribute"].as_str() == Some(f.field.as_str()))
            .ok_or_else(|| {
                format!(
                    "fulltext: field '{}' is not in the Redis index — refusing to search a \
                     field the index does not have",
                    f.field
                )
            })?;
        if attr["type"].as_str() != Some("TEXT") {
            return Err(format!(
                "fulltext: Redis field '{}' is {} — fulltext needs TEXT",
                f.field, attr["type"]
            ));
        }
        let weight = match &attr["WEIGHT"] {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.parse::<f64>().ok(),
            _ => None,
        };
        if weight != Some(1.0) {
            return Err(format!(
                "fulltext: Redis field '{}' has index WEIGHT {} — boosts are applied at query \
                 time, so the index weight must be 1",
                f.field, attr["WEIGHT"]
            ));
        }
        let flags: Vec<&str> = attr["flags"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        for bad in ["NOINDEX", "PHONETIC"] {
            if flags.contains(&bad) || !attr[bad].is_null() {
                return Err(format!(
                    "fulltext: Redis field '{}' is {bad} — the pinned analyzer indexes and \
                     matches terms exactly",
                    f.field
                ));
            }
        }
        let nostem = flags.contains(&"NOSTEM") || !attr["NOSTEM"].is_null();
        if nostem == profile.stems() {
            return Err(format!(
                "fulltext: Redis field '{}' is {} — analysis \"{}\" {}",
                f.field,
                if nostem { "NOSTEM" } else { "stemmed" },
                profile.analysis.as_str(),
                if profile.stems() {
                    "stems (Snowball English)"
                } else {
                    "needs NOSTEM"
                }
            ));
        }
    }
    if profile.analysis == Analysis::Parity {
        // Only the searched fields may be TEXT: BM25STD's document length and
        // term frequency span every TEXT field of the index.
        for a in attributes {
            let name = a["attribute"].as_str().unwrap_or_default();
            if a["type"].as_str() == Some("TEXT") && !profile.text_fields.contains(name) {
                return Err(format!(
                    "fulltext: parity index has TEXT field '{name}' that is not searched — \
                     only {:?} may be indexed as text",
                    profile.text_fields
                ));
            }
        }
    }
    // Index-level settings `engine_settings("redis")` claims are pinned. Each
    // changes stemming or BM25STD while the result would still say Porter2/BM25.
    let def = &info["index_definition"];
    if !def.is_object() {
        return Err("fulltext: FT.INFO has no index_definition map (RESP3 expected)".into());
    }
    match &def["default_language"] {
        Value::Null => {}
        Value::String(l) if l.eq_ignore_ascii_case("english") => {}
        other => {
            return Err(format!(
                "fulltext: Redis index LANGUAGE is {other} — the pinned stemmer is Snowball \
                 English, the default"
            ))
        }
    }
    for key in ["language_field", "score_field", "payload_field"] {
        if !def[key].is_null() {
            return Err(format!(
                "fulltext: Redis index sets {key} ({}) — per-document language or score \
                 would change stemming or scale BM25STD",
                def[key]
            ));
        }
    }
    let default_score = match &def["default_score"] {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    };
    if default_score != Some(1.0) {
        return Err(format!(
            "fulltext: Redis index default_score is {} — BM25STD is multiplied by it, it must be 1",
            def["default_score"]
        ));
    }
    let options: Vec<&str> = match &info["index_options"] {
        Value::Null => Vec::new(),
        Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
        other => {
            return Err(format!(
                "fulltext: unreadable FT.INFO index_options {other}"
            ))
        }
    };
    for bad in ["NOFREQS", "NOOFFSETS", "NOFIELDS"] {
        if options.contains(&bad) {
            return Err(format!(
                "fulltext: Redis index was built with {bad} — the pinned index keeps term \
                 frequencies (BM25STD), field flags (@field clauses) and term offsets"
            ));
        }
    }
    let stop = info["stopwords_list"].as_array().ok_or(
        "fulltext: the Redis index reports no stopwords_list — it was not built with the \
         pinned STOPWORDS (was it created by a run without a fulltext search config?)",
    )?;
    if !same_word_set(stop.iter().filter_map(Value::as_str)) || stop.len() != STOPWORDS.len() {
        return Err(format!(
            "fulltext: Redis index stopwords_list {stop:?} is not the pinned list {STOPWORDS:?}"
        ));
    }
    Ok(())
}

/// Check a live OpenSearch index serves the fulltext workload as pinned:
/// every searched field is a `text` field analyzed by [`OPENSEARCH_ANALYZER`],
/// and that analyzer and its filters are exactly [`opensearch_analysis`].
/// `properties` is `mappings.properties`, `index` is `settings.index` (as
/// `GET _settings` returns it — scalars may come back as strings).
///
/// Also refused: a `char_filter` on the analyzer, a non-BM25 similarity (per
/// field or `index.similarity`), and more than one shard unless
/// `allow_multi_shard` (Lucene IDF and length norms are per shard, so a
/// multi-shard index is compared against Redis on per-shard statistics).
/// Returns the index's shard count, for the result metadata.
pub fn validate_opensearch_index(
    properties: &Value,
    index: &Value,
    fields: &[FieldBoost],
    allow_multi_shard: bool,
) -> Result<u64, String> {
    validate_opensearch_index_for(
        properties,
        index,
        fields,
        allow_multi_shard,
        &IndexProfile::english(),
        &Value::Null,
        &[],
    )
}

/// [`validate_opensearch_index`] for an [`IndexProfile`]: the mode's analyzer
/// and filters, `meta` (`mappings._meta.vdbb_fulltext`, `Null` when absent)
/// equal to the profile's marker, and in parity no other `text` field indexed.
pub fn validate_opensearch_index_for(
    properties: &Value,
    index: &Value,
    fields: &[FieldBoost],
    allow_multi_shard: bool,
    profile: &IndexProfile,
    meta: &Value,
    schema_text_fields: &[String],
) -> Result<u64, String> {
    profile.check_marker(meta)?;
    let analyzer_name = opensearch_analyzer_name(profile.analysis);
    if profile.analysis == Analysis::Parity {
        // The schema's text fields the tool maps: only the searched ones may
        // be indexed. A payload key outside the schema that OpenSearch maps
        // dynamically cannot change the searched fields' scores (Lucene keeps
        // per-field statistics), so it is not this check's concern.
        for name in schema_text_fields {
            let m = &properties[name.as_str()];
            let indexed = !(m["index"] == false || m["index"] == "false");
            if m["type"] == "text" && indexed && !profile.text_fields.contains(name) {
                return Err(format!(
                    "fulltext: parity index has indexed text field '{name}' that is not \
                     searched — only {:?} may be indexed as text",
                    profile.text_fields
                ));
            }
        }
    }
    for f in fields {
        let m = &properties[&f.field];
        if m.is_null() {
            return Err(format!(
                "fulltext: field '{}' is not in the OpenSearch mapping — refusing to search a \
                 field the index does not have",
                f.field
            ));
        }
        if m["type"] != "text" {
            return Err(format!(
                "fulltext: OpenSearch field '{}' is {} — fulltext needs text",
                f.field, m["type"]
            ));
        }
        if m["analyzer"] != analyzer_name
            || !(m["search_analyzer"].is_null() || m["search_analyzer"] == analyzer_name)
        {
            return Err(format!(
                "fulltext: OpenSearch field '{}' is analyzed by {} / search {} — expected the \
                 pinned {analyzer_name}",
                f.field, m["analyzer"], m["search_analyzer"]
            ));
        }
        if m["index"] == false || m["index"] == "false" {
            return Err(format!(
                "fulltext: OpenSearch field '{}' is not indexed",
                f.field
            ));
        }
        if !(m["similarity"].is_null() || m["similarity"] == "BM25") {
            return Err(format!(
                "fulltext: OpenSearch field '{}' uses similarity {} — the pinned scorer is \
                 Lucene BM25",
                f.field, m["similarity"]
            ));
        }
    }
    // `index.similarity` may only restate the default BM25, without parameters.
    if let Some(sims) = index["similarity"].as_object() {
        let default_bm25 = |v: &Value| {
            v.as_object()
                .is_some_and(|o| o.len() == 1 && o.get("type") == Some(&json!("BM25")))
        };
        if sims.len() != 1 || !sims.get("default").is_some_and(default_bm25) {
            return Err(format!(
                "fulltext: OpenSearch index.similarity is {} — the pinned scorer is the default \
                 Lucene BM25 (k1=1.2, b=0.75)",
                index["similarity"]
            ));
        }
    } else if !index["similarity"].is_null() {
        return Err(format!(
            "fulltext: unreadable OpenSearch index.similarity {}",
            index["similarity"]
        ));
    }
    let shards = match &index["number_of_shards"] {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse::<u64>().ok(),
        _ => None,
    }
    .ok_or_else(|| {
        format!(
            "fulltext: unreadable OpenSearch number_of_shards {}",
            index["number_of_shards"]
        )
    })?;
    if shards > 1 && !allow_multi_shard {
        return Err(format!(
            "fulltext: the OpenSearch index has {shards} shards — BM25 statistics are per \
             shard, so set collection_params.number_of_shards to 1, or opt in with \
             `fulltext.allow_multi_shard: true` (the shard count is recorded either way)"
        ));
    }
    let want = opensearch_analysis_for(profile.analysis);
    let got = &index["analysis"];
    let analyzer = &got["analyzer"][analyzer_name];
    let a_want = &want["analyzer"][analyzer_name];
    let no_char_filter = match &analyzer["char_filter"] {
        Value::Null => true,
        Value::Array(a) => a.is_empty(),
        _ => false,
    };
    if analyzer["tokenizer"] != a_want["tokenizer"]
        || analyzer["filter"] != a_want["filter"]
        || !no_char_filter
    {
        return Err(format!(
            "fulltext: OpenSearch analyzer {analyzer_name} is {analyzer} — expected {a_want}"
        ));
    }
    let checked: &[&str] = if profile.stems() {
        &[OS_POSSESSIVE, OS_STEMMER]
    } else {
        &[OS_POSSESSIVE]
    };
    for &name in checked {
        let (g, w) = (&got["filter"][name], &want["filter"][name]);
        if g["type"] != w["type"] || g["language"] != w["language"] {
            return Err(format!(
                "fulltext: OpenSearch filter {name} is {g} — expected {w}"
            ));
        }
    }
    let stop = &got["filter"][OS_STOP];
    let words = stop["stopwords"].as_array();
    if stop["type"] != "stop"
        || !words.is_some_and(|w| {
            w.len() == STOPWORDS.len() && same_word_set(w.iter().filter_map(Value::as_str))
        })
    {
        return Err(format!(
            "fulltext: OpenSearch stop filter {OS_STOP} is {stop} — expected the pinned list"
        ));
    }
    // A stop filter that ignores case would also drop e.g. "IT"; ours runs
    // after `lowercase`, so the flag must not be set to anything but false.
    if !(stop["ignore_case"].is_null() || stop["ignore_case"] == false) {
        return Err(format!(
            "fulltext: OpenSearch stop filter {OS_STOP} sets ignore_case"
        ));
    }
    Ok(shards)
}

/// The error [`RankCheck`] returns for a page that is not score-ordered.
pub const RANK_VIOLATION: &str = "fulltext: reply is not ranked by descending score";

/// The error [`RankCheck`] returns for a NaN or infinite score.
const NON_FINITE_SCORE: &str = "fulltext: non-finite score in a ranked reply";

/// `true` for a query error that says the reply was not a ranked page (out
/// of score order, or a non-finite score).
pub fn is_rank_violation(error: &str) -> bool {
    error.contains(RANK_VIOLATION) || error.contains(NON_FINITE_SCORE)
}

/// Under `total_hits: "none"`, a Redis reply that is not score-ordered means
/// `WITHOUTCOUNT` changed the ranking on this server, so the run is refused
/// instead of reporting the other queries' QPS and latency as a BM25 top-k
/// result. Other modes keep the existing behaviour (the failed query is
/// dropped and reported).
pub fn check_rank_violations(total_hits: TotalHits, violations: usize) -> Result<(), String> {
    if total_hits == TotalHits::None && violations > 0 {
        return Err(format!(
            "fulltext: {violations} Redis replies sent with {REDIS_WITHOUTCOUNT} \
             (`total_hits: \"none\"`) were not ranked by descending score: this server \
             does not keep the BM25 ranking under {REDIS_WITHOUTCOUNT}; use \"exact\""
        ));
    }
    Ok(())
}

/// Streaming rank check: every score finite and none above its predecessor.
/// Holds no buffer, so the non-capturing path validates without allocating.
#[derive(Default)]
struct RankCheck {
    prev: Option<f64>,
}

impl RankCheck {
    fn push(&mut self, score: f64) -> Result<(), String> {
        if !score.is_finite() {
            return Err(NON_FINITE_SCORE.into());
        }
        if let Some(prev) = self.prev {
            if score > prev {
                return Err(format!("{RANK_VIOLATION} ({score} after {prev})"));
            }
        }
        self.prev = Some(score);
        Ok(())
    }
}

/// A ranked page: `(engine id, score)` in rank order.
pub type RankedHits = Vec<(String, f64)>;

/// A validated ranked reply. `hits` is `Some` only when asked to keep them
/// (per-query capture); validation never needs them, so the timed loop of a
/// non-capturing run builds no id strings.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedPage {
    pub total: Option<u64>,
    pub returned: usize,
    pub hits: Option<RankedHits>,
    /// Bytes of fetched field values in the page (0 without `fetch`).
    pub fetched_bytes: u64,
}

/// Check one hit's fetched fields: with `fetch`, exactly those fields, each a
/// string, and the value bytes are returned; without, none at all (`fields`
/// absent, or an empty map). `fields` yields `(name, value)` with the value
/// `None` when it is not a string.
fn check_fetched(
    fetch: Option<&[String]>,
    fields: Option<Vec<(&str, Option<usize>)>>,
    engine: &str,
) -> Result<u64, String> {
    let fields = fields.unwrap_or_default();
    let Some(fetch) = fetch else {
        if fields.is_empty() {
            return Ok(0);
        }
        return Err(format!(
            "fulltext: {engine} returned document fields for an ids-only request"
        ));
    };
    let mut bytes = 0u64;
    let mut seen = HashSet::new();
    for (name, len) in &fields {
        if !fetch.iter().any(|f| f == name) || !seen.insert(*name) {
            return Err(format!(
                "fulltext: {engine} returned field {name:?}, which fetch {fetch:?} did not ask for"
            ));
        }
        bytes += len.ok_or_else(|| {
            format!("fulltext: {engine} returned a non-string value for fetched field {name:?}")
        })? as u64;
    }
    if seen.len() != fetch.len() {
        return Err(format!(
            "fulltext: {engine} hit is missing fetched fields: got {:?}, asked for {fetch:?}",
            fields.iter().map(|f| f.0).collect::<Vec<_>>()
        ));
    }
    Ok(bytes)
}

/// Parse a RESP3 `FT.SEARCH … NOCONTENT WITHSCORES` reply. Any warning
/// (timeout partials included) is an error, as on the filter-only path; the
/// page must be complete (`min(total, top)` hits), duplicate-free and ranked.
pub fn redis_ranked_page(
    value: &redis::Value,
    top: usize,
    keep_hits: bool,
) -> Result<RankedPage, String> {
    redis_ranked_page_with(value, top, keep_hits, TotalHits::Exact)
}

/// [`redis_ranked_page`] for a request sent with `total`. Under `none`
/// (`WITHOUTCOUNT`) `total_results` is the page size clamped per shard, not a
/// count: the page is still checked against it, but [`RankedPage::total`] is
/// `None` so the clamped value is never published as a total.
pub fn redis_ranked_page_with(
    value: &redis::Value,
    top: usize,
    keep_hits: bool,
    total_hits: TotalHits,
) -> Result<RankedPage, String> {
    redis_ranked_page_fetch(value, top, keep_hits, total_hits, None)
}

/// [`redis_ranked_page_with`] for a request sent with `fetch`
/// ([`redis_args_fetch`]): every hit's `extra_attributes` must hold exactly
/// those fields as strings; without `fetch`, it must be absent or empty.
pub fn redis_ranked_page_fetch(
    value: &redis::Value,
    top: usize,
    keep_hits: bool,
    total_hits: TotalHits,
    fetch: Option<&[String]>,
) -> Result<RankedPage, String> {
    use redis::Value as V;
    fn text(v: &V) -> Option<&str> {
        match v {
            V::BulkString(b) => std::str::from_utf8(b).ok(),
            V::SimpleString(s) => Some(s),
            _ => None,
        }
    }
    fn field<'a>(fields: &'a [(V, V)], name: &str) -> Option<&'a V> {
        fields
            .iter()
            .find_map(|(k, v)| (text(k) == Some(name)).then_some(v))
    }
    let V::Map(fields) = value else {
        return Err("fulltext requires a RESP3 FT.SEARCH map".into());
    };
    match field(fields, "warning") {
        Some(V::Array(w)) if w.is_empty() => {}
        _ => return Err("fulltext: Redis warning or missing completion metadata".into()),
    }
    let total = match field(fields, "total_results") {
        Some(V::Int(n)) if *n >= 0 => *n as usize,
        _ => return Err("fulltext: missing Redis total_results".into()),
    };
    let Some(V::Array(results)) = field(fields, "results") else {
        return Err("fulltext: missing Redis results".into());
    };
    if results.len() != total.min(top) {
        return Err(format!(
            "fulltext: incomplete Redis page: total={total}, returned={}, limit={top}",
            results.len()
        ));
    }
    let mut seen = HashSet::new();
    let mut rank = RankCheck::default();
    let mut hits = keep_hits.then(|| Vec::with_capacity(results.len()));
    let mut fetched_bytes = 0u64;
    for r in results {
        let V::Map(r) = r else {
            return Err("fulltext: invalid Redis result entry".into());
        };
        let attrs = match field(r, "extra_attributes") {
            None => None,
            Some(V::Map(m)) => Some(
                m.iter()
                    .map(|(k, v)| {
                        let len = match v {
                            V::BulkString(b) => Some(b.len()),
                            V::SimpleString(s) => Some(s.len()),
                            V::VerbatimString { text, .. } => Some(text.len()),
                            _ => None,
                        };
                        text(k)
                            .map(|k| (k, len))
                            .ok_or("fulltext: Redis field name is not a string")
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Some(_) => return Err("fulltext: Redis extra_attributes is not a map".into()),
        };
        fetched_bytes += check_fetched(fetch, attrs, "Redis")?;
        let id = field(r, "id")
            .and_then(text)
            .ok_or("fulltext: Redis hit without id")?;
        let score = match field(r, "score") {
            Some(V::Double(d)) => *d,
            Some(v) => text(v)
                .and_then(|s| s.parse::<f64>().ok())
                .ok_or("fulltext: unparseable Redis score")?,
            None => return Err("fulltext: Redis hit without score (WITHSCORES missing?)".into()),
        };
        if !seen.insert(id) {
            return Err(format!("fulltext: duplicate Redis hit {id}"));
        }
        rank.push(score)?;
        if let Some(h) = hits.as_mut() {
            h.push((id.to_string(), score));
        }
    }
    Ok(RankedPage {
        total: (total_hits == TotalHits::Exact).then_some(total as u64),
        returned: results.len(),
        hits,
        fetched_bytes,
    })
}

/// How many queries with more matches than the page [`probe_withoutcount`]
/// compares before it stops.
pub const WITHOUTCOUNT_PROBE_QUERIES: usize = 8;

/// What [`probe_withoutcount`] compared: `compared` queries with at least one
/// match, of which `over_limit` had more matches than the page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WithoutCountProbe {
    pub compared: usize,
    pub over_limit: usize,
}

/// Check, before the timed window, that `WITHOUTCOUNT` does not change the
/// ranked page on this server. `send(i, mode)` runs query `i` of the query
/// set built with `mode` ([`redis_args_with`]) and returns the raw reply.
///
/// Queries are taken in order; each with at least one match is sent counted
/// and with `WITHOUTCOUNT`, and the two pages must hold the same `(id, score)`
/// hits in the same order (ids may differ only inside the tied-score group cut
/// by the page boundary). It stops after [`WITHOUTCOUNT_PROBE_QUERIES`] queries
/// with more matches than `top`, and fails if it found none: a page that is
/// not truncated cannot show that the server still selects the top `k`.
/// RediSearch 8.0/8.2 fail it (their optimizer does not recognise
/// `SCORER BM25STD` and returns the first `k` matches in document-id order).
pub fn probe_withoutcount<F>(
    queries: usize,
    top: usize,
    mut send: F,
) -> Result<WithoutCountProbe, String>
where
    F: FnMut(usize, TotalHits) -> Result<redis::Value, String>,
{
    let refuse = |why: String| {
        format!(
            "fulltext: `total_hits: \"none\"` refused before the timed window: {why}. \
             {REDIS_WITHOUTCOUNT} must return the same BM25-ranked page as a counted \
             FT.SEARCH (RediSearch before 8.4 does not recognise SCORER {REDIS_SCORER} \
             under {REDIS_WITHOUTCOUNT} and returns the first matches in document-id \
             order); use \"exact\" on this server"
        )
    };
    let mut probe = WithoutCountProbe {
        compared: 0,
        over_limit: 0,
    };
    for i in 0..queries {
        if probe.over_limit >= WITHOUTCOUNT_PROBE_QUERIES {
            break;
        }
        let counted = send(i, TotalHits::Exact)
            .and_then(|v| redis_ranked_page_with(&v, top, true, TotalHits::Exact))
            .map_err(|e| refuse(format!("query #{i} (counted): {e}")))?;
        let total = counted
            .total
            .ok_or("fulltext: counted page without a total")?;
        if total == 0 {
            continue;
        }
        let uncounted = send(i, TotalHits::None)
            .and_then(|v| redis_ranked_page_with(&v, top, true, TotalHits::None))
            .map_err(|e| refuse(format!("query #{i} ({REDIS_WITHOUTCOUNT}): {e}")))?;
        let truncated = total > top as u64;
        same_ranked_page(
            counted.hits.as_deref().unwrap_or_default(),
            uncounted.hits.as_deref().unwrap_or_default(),
            truncated,
        )
        .map_err(|e| refuse(format!("query #{i}: {e}")))?;
        probe.compared += 1;
        if truncated {
            probe.over_limit += 1;
        }
    }
    if probe.over_limit == 0 {
        // A property of the probed queries, not of the server: say so rather
        // than blame the RediSearch version.
        return Err(format!(
            "fulltext: `total_hits: \"none\"` refused before the timed window: none of the \
             {} probed queries with a match has more than {top}, so the probe cannot show that \
             {REDIS_WITHOUTCOUNT} still selects the top {top}; use \"exact\" for this query set",
            probe.compared
        ));
    }
    Ok(probe)
}

/// `counted` and `uncounted` hold the same hits: equal scores rank by rank,
/// and the same ids within each tied-score group. When the page is cut by
/// the limit (`truncated`), the last group may hold other ids of that score.
fn same_ranked_page(
    counted: &[(String, f64)],
    uncounted: &[(String, f64)],
    truncated: bool,
) -> Result<(), String> {
    let scores = |p: &[(String, f64)]| p.iter().map(|h| h.1).collect::<Vec<_>>();
    if counted.len() != uncounted.len() || scores(counted) != scores(uncounted) {
        return Err(format!(
            "the {REDIS_WITHOUTCOUNT} page {uncounted:?} differs from the counted page \
             {counted:?}"
        ));
    }
    let mut start = 0;
    while start < counted.len() {
        let score = counted[start].1;
        let end = start + counted[start..].iter().take_while(|h| h.1 == score).count();
        let ids = |p: &[(String, f64)]| -> HashSet<String> {
            p[start..end].iter().map(|h| h.0.clone()).collect()
        };
        if !(truncated && end == counted.len()) && ids(counted) != ids(uncounted) {
            return Err(format!(
                "the {REDIS_WITHOUTCOUNT} page {uncounted:?} holds other documents than \
                 the counted page {counted:?}"
            ));
        }
        start = end;
    }
    Ok(())
}

/// Record that a disk window ran `total_hits: none` without the
/// [`probe_withoutcount`] probe, and why ([`QuerySet::withoutcount_probe_plan`]).
pub fn record_withoutcount_probe_skipped(meta: &mut Value, reason: &str) {
    meta["total_hits"]["withoutcount_probe"] = json!({"skipped": reason});
}

/// Record what a fetch run returned, and the request/command that asked for
/// it. `fetch` `None` records `null` (ids only) for a disk window, so the
/// pair always states it; other ids-only results are left as before.
pub fn record_fetch(
    meta: &mut Value,
    engine: &str,
    fetch: Option<&[String]>,
    hits: u64,
    bytes: u64,
) -> Result<(), String> {
    let Some(fields) = fetch else {
        if meta["compound"]["suite"] == crate::msmarco::disk_queries::SUITE {
            meta["fetch"] = Value::Null;
        }
        return Ok(());
    };
    meta["fetch"] = json!({
        "fields": fields,
        "hits": hits,
        "bytes": bytes,
        "checked": "every hit of every reply returned exactly these fields, each a string",
    });
    match engine {
        "opensearch" => meta["engine_settings"]["request"]["_source"] = opensearch_source(fetch),
        "redis" => {
            let command = meta["engine_settings"]["command"]
                .as_str()
                .ok_or("fulltext: Redis engine_settings has no command")?
                .replacen(
                    "NOCONTENT",
                    &format!("RETURN {} {}", fields.len(), fields.join(" ")),
                    1,
                );
            meta["engine_settings"]["command"] = json!(command);
        }
        other => return Err(format!("fulltext: engine '{other}' has no fetch")),
    }
    Ok(())
}

/// Record the [`probe_withoutcount`] outcome under `total_hits`.
pub fn record_withoutcount_probe(meta: &mut Value, probe: &WithoutCountProbe) {
    meta["total_hits"]["withoutcount_probe"] = json!({
        "compared_queries": probe.compared,
        "queries_over_limit": probe.over_limit,
        "page": "identical (id, score) to the counted FT.SEARCH, checked before the timed window",
    });
}

/// Parse an OpenSearch search response: no timeout, no failed shard, at most
/// `top` hits, each with an `_id` and a `_score`, duplicate-free and ranked.
pub fn opensearch_ranked_page(
    body: &Value,
    top: usize,
    keep_hits: bool,
) -> Result<RankedPage, String> {
    opensearch_page_checked(body, top, keep_hits, None, None)
}

/// [`opensearch_ranked_page`] for a request sent with `total`. The reply must
/// show the server honoured it: `exact` needs `hits.total.relation == "eq"`,
/// `none` needs `hits.total` absent.
pub fn opensearch_ranked_page_with(
    body: &Value,
    top: usize,
    keep_hits: bool,
    total: TotalHits,
) -> Result<RankedPage, String> {
    opensearch_page_checked(body, top, keep_hits, Some(total), None)
}

/// [`opensearch_ranked_page_with`] for a request sent with `fetch`
/// ([`query_opensearch_fetch`]): every hit's `_source` must hold exactly those
/// fields as strings; without `fetch`, `_source` must be absent.
pub fn opensearch_ranked_page_fetch(
    body: &Value,
    top: usize,
    keep_hits: bool,
    total: TotalHits,
    fetch: Option<&[String]>,
) -> Result<RankedPage, String> {
    opensearch_page_checked(body, top, keep_hits, Some(total), fetch)
}

fn opensearch_page_checked(
    body: &Value,
    top: usize,
    keep_hits: bool,
    expect: Option<TotalHits>,
    fetch: Option<&[String]>,
) -> Result<RankedPage, String> {
    match expect {
        Some(TotalHits::Exact) if body["hits"]["total"]["relation"] != "eq" => {
            return Err(format!(
                "fulltext: exact total requested (track_total_hits: true) but the reply's \
                 hits.total is {}",
                body["hits"]["total"]
            ))
        }
        Some(TotalHits::None) if body["hits"].get("total").is_some() => {
            return Err(format!(
                "fulltext: no total requested (track_total_hits: false) but the reply carries \
                 hits.total {}",
                body["hits"]["total"]
            ))
        }
        _ => {}
    }
    if body["timed_out"] != false || body["_shards"]["failed"] != 0 {
        return Err("fulltext: timed out, failed shards, or missing completion metadata".into());
    }
    let results = body["hits"]["hits"]
        .as_array()
        .ok_or("fulltext: missing hits.hits")?;
    if results.len() > top {
        return Err(format!(
            "fulltext: {} hits for a limit of {top}",
            results.len()
        ));
    }
    let mut seen = HashSet::new();
    let mut rank = RankCheck::default();
    let mut hits = keep_hits.then(|| Vec::with_capacity(results.len()));
    let mut fetched_bytes = 0u64;
    for h in results {
        let source = match h.get("_source") {
            None => None,
            Some(Value::Object(o)) => Some(
                o.iter()
                    .map(|(k, v)| (k.as_str(), v.as_str().map(str::len)))
                    .collect::<Vec<_>>(),
            ),
            Some(_) => return Err("fulltext: OpenSearch _source is not an object".into()),
        };
        // An ids-only request must not carry `_source` at all (an empty
        // object would mean the request asked for a source).
        if fetch.is_none() && h.get("_source").is_some() {
            return Err("fulltext: OpenSearch returned _source for an ids-only request".into());
        }
        fetched_bytes += check_fetched(fetch, source, "OpenSearch")?;
        let id = h["_id"]
            .as_str()
            .or_else(|| h["fields"]["_id"][0].as_str())
            .ok_or("fulltext: hit without _id")?;
        let score = h["_score"].as_f64().ok_or("fulltext: hit without _score")?;
        if !seen.insert(id) {
            return Err(format!("fulltext: duplicate OpenSearch hit {id}"));
        }
        rank.push(score)?;
        if let Some(out) = hits.as_mut() {
            out.push((id.to_string(), score));
        }
    }
    let total = (body["hits"]["total"]["relation"] == "eq")
        .then(|| body["hits"]["total"]["value"].as_u64())
        .flatten();
    if total.is_some_and(|n| results.len() != (n as usize).min(top)) {
        return Err("fulltext: incomplete exact-total OpenSearch page".into());
    }
    Ok(RankedPage {
        total,
        returned: results.len(),
        hits,
        fetched_bytes,
    })
}

/// Engines with a fulltext arm, and how each was pinned.
fn engine_settings(engine: &str) -> Result<Value, String> {
    match engine {
        "redis" => Ok(json!({
            "index": {
                "field_type": "TEXT, index WEIGHT 1 (built SORTABLE; SORTABLE is not verified — it changes neither matching nor BM25STD)",
                "tokenizer": "RediSearch default (separators: space, tab and the ASCII set in `query_terms.separators`)",
                "stemmer": "RediSearch default: Snowball English (Porter2); LANGUAGE and NOSTEM not set",
                "stopwords": "explicit FT.CREATE STOPWORDS 33 <pinned list> (identical to the RediSearch default)",
                "verified_on_live_index": "FT.INFO before the timed window: each searched field TEXT, WEIGHT 1, \
                    no NOSTEM/NOINDEX/PHONETIC; default_language absent or english; no language_field, \
                    score_field or payload_field; default_score 1; index_options without NOFREQS, \
                    NOOFFSETS, NOFIELDS; stopwords_list = the pinned list",
            },
            "query": "(@field:(t1|t2|…)) => {$weight: boost;} joined with |",
            "command": "FT.SEARCH <idx> <query> NOCONTENT WITHSCORES SCORER BM25STD LIMIT 0 <top> DIALECT 2 TIMEOUT <ms>",
            "scorer": REDIS_SCORER,
            "protocol": "RESP3, replies with any warning rejected",
        })),
        "opensearch" => Ok(json!({
            "index": {
                "field_type": "text",
                "analyzer": OPENSEARCH_ANALYZER,
                "analysis": opensearch_analysis(),
                "note": "the built-in `english` analyzer with porter2 (Snowball English) instead of porter, and its empty keyword_marker dropped",
                "verified_on_live_index": "_mapping + _settings before the timed window: each searched field \
                    text, indexed, analyzer (and search analyzer) vdbb_english_porter2, similarity absent or BM25; \
                    analyzer tokenizer/filters as pinned and no char_filter; filters as pinned; index.similarity \
                    absent or a parameterless default BM25; number_of_shards 1 unless allow_multi_shard",
            },
            "query": {"multi_match": {"type": OPENSEARCH_MULTI_MATCH_TYPE, "operator": "or"}},
            "request": {"_source": false, "track_total_hits": false},
            "scorer": "Lucene BM25 (k1=1.2, b=0.75)",
        })),
        other => Err(format!(
            "fulltext: no recorded settings for engine '{other}'"
        )),
    }
}

/// The `fulltext` block of a result JSON: everything that defines what was
/// measured, so two results can be checked for comparability.
pub fn result_metadata(
    engine: &str,
    fields: &[FieldBoost],
    set: &QuerySet,
    top: usize,
) -> Result<Value, String> {
    result_metadata_for(
        engine,
        fields,
        set,
        top,
        &IndexProfile::english(),
        &Value::Null,
    )
}

/// The differences that remain in `analysis` mode. Parity removes the newline
/// tokenization one (normalized at upload) and the stemming one (no stemming
/// on either engine); whole-document statistics remain but span exactly the
/// searched fields.
pub fn known_differences(analysis: Analysis) -> Vec<&'static str> {
    match analysis {
        Analysis::English => KNOWN_DIFFERENCES.to_vec(),
        Analysis::Parity => KNOWN_DIFFERENCES
            .into_iter()
            .filter(|d| !PARITY_REMOVES.contains(d))
            .collect(),
    }
}

/// [`result_metadata`] for an [`IndexProfile`]: records the analysis mode,
/// whether the uploaded text was normalized and which text fields are indexed.
///
/// `recorded_marker` is the marker the live index actually carries (Redis
/// marker key / OpenSearch `_meta`, `Null` when absent — an index built before
/// markers existed, reused in english mode). It is published as
/// `index_marker`; the profile's marker as `index_marker_expected`.
pub fn result_metadata_for(
    engine: &str,
    fields: &[FieldBoost],
    set: &QuerySet,
    top: usize,
    profile: &IndexProfile,
    recorded_marker: &Value,
) -> Result<Value, String> {
    let mut m = result_metadata_english(engine, fields, set, top)?;
    if let Some(w) = &set.window {
        // A disk window can hold a million queries: the result records the
        // window and the SHA-256 of exactly the timed queries served, not the
        // queries themselves (they are lines of the recorded family file).
        use sha2::{Digest, Sha256};
        let hex = |b: &[u8]| -> String {
            Sha256::digest(b)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        };
        let timed: Vec<_> = set.queries[w.timed_from..]
            .iter()
            .map(|q| (q.query_id, q.compound.as_ref()))
            .collect();
        let served = hex(&serde_json::to_vec(&timed).map_err(|e| e.to_string())?);
        let scoring_contract = "native-engine-scoring-v1";
        let mut identity = json!({"scoring_contract":scoring_contract,"window":w.record,
            "served_sha256":served});
        // Duration mode measures a different schedule (the window cycled for
        // a time), so it is part of the identity; absent, the identity is
        // exactly the single-pass one.
        if let Some(d) = w.duration_s {
            identity["duration_s"] = json!(d);
        }
        if profile.scalar_indexing.is_reduced() {
            identity["scalar_indexing"] = json!(profile.scalar_indexing);
        }
        let sha = hex(&serde_json::to_vec(&identity).map_err(|e| e.to_string())?);
        m["compound"] = json!({"version":2,"suite":crate::msmarco::disk_queries::SUITE,
            "family":w.family.name(),"scoring_contract":scoring_contract,"window":w.record,
            "served_sha256":served,"sha256":sha,"preparation_sha256":set.preparation_sha256,
            "exact_counts":true,"scalar_indexing":profile.scalar_indexing,
            "ordering":"engine score descending; ties unspecified; cross-engine ranks are not relevance truth"});
        if let Some(d) = w.duration_s {
            m["compound"]["duration_s"] = json!(d);
        }
        m["query_source"] = json!(format!(
            "{} family file; query_id is its line (compound.window)",
            crate::msmarco::disk_queries::SUITE
        ));
        m["queries_total"] = json!(set.queries.len() - w.timed_from);
        m["queries_eligible"] = json!(set.queries.len() - w.timed_from);
        m["operator"] = json!("explicit Boolean AST");
        m["boosts_applied"] = json!("unit boosts; no field reweighting");
        m["query_terms"] = json!({"method":"one validated field-scoped ASCII term per text leaf; exact Boolean AST retained"});
        m["engine_settings"]["query"] =
            json!("explicit Boolean AST; engine translation of the family file's lines");
        if engine == "opensearch" {
            m["engine_settings"]["scalar_context"] =
                json!("bool.filter; no scalar score contribution");
        } else if engine == "redis" {
            m["engine_settings"]["scalar_context"] =
                json!("native BM25STD; TAG/nontext iterator contributions may be nonzero");
        }
    } else if set.queries.iter().any(|q| q.compound.is_some()) {
        use sha2::{Digest, Sha256};
        let queries: Vec<_> = set
            .queries
            .iter()
            .filter_map(|q| q.compound.as_ref())
            .collect();
        let scoring_contract = "native-engine-scoring-v1";
        let mut identity = json!({"scoring_contract":scoring_contract,"queries":queries});
        if profile.scalar_indexing.is_reduced() {
            identity["scalar_indexing"] = json!(profile.scalar_indexing);
        }
        let bytes = serde_json::to_vec(&identity).map_err(|e| e.to_string())?;
        m["compound"] = json!({"version":2,"scoring_contract":scoring_contract,"queries":queries,"sha256":Sha256::digest(bytes).iter().map(|b|format!("{b:02x}")).collect::<String>(),
            "preparation_sha256":set.preparation_sha256,"exact_counts":true,"ordering":"engine score descending; ties unspecified; cross-engine ranks are not relevance truth"});
        m["compound"]["scalar_indexing"] = json!(profile.scalar_indexing);
        m["query_source"] = json!("explicit compound suite; query_id indexes compound.queries");
        m["operator"] = json!("explicit Boolean AST");
        m["boosts_applied"] = json!("unit boosts; no field reweighting");
        m["query_terms"] = json!({"method":"one validated field-scoped ASCII term per text leaf; exact Boolean AST retained"});
        m["engine_settings"]["query"] =
            json!("explicit Boolean AST; engine translation recorded by compound.queries");
        if engine == "opensearch" {
            m["engine_settings"]["request"]["track_total_hits"] = json!(true);
            m["engine_settings"]["scalar_context"] =
                json!("bool.filter; no scalar score contribution");
        } else if engine == "redis" {
            m["engine_settings"]["scalar_context"] =
                json!("native BM25STD; TAG/nontext iterator contributions may be nonzero");
        }
    }

    m["analysis"] = json!(profile.analysis.as_str());
    m["normalize_whitespace"] = json!(profile.normalizes_whitespace());
    m["index_marker"] = recorded_marker.clone();
    m["index_marker_expected"] = profile.marker();
    m["known_differences"] = json!(known_differences(profile.analysis));
    if set.queries.iter().any(|q| q.compound.is_some()) {
        m["known_differences"].as_array_mut().unwrap().push(json!(
            "Compound scoring is engine-native: Redis BM25STD can score TAG/nontext iterators; OpenSearch scalar bool.filter clauses contribute zero. Matching/counts are the contract; cross-engine scores/ranks are diagnostic."
        ));
    }
    if profile.analysis == Analysis::Parity {
        m["stemmer"] = json!("none on either engine (parity)");
        m["indexed_text_fields"] = json!(profile.text_fields);
        m["normalization"] = json!(
            "every Unicode whitespace character other than U+0020 (\\n, \\r, \\t, \\x0B, \\x0C, \
             U+0085, U+00A0, U+2028, U+2029, …) replaced by a space in the indexed text fields \
             at upload, on both engines; recorded in the index and verified on reuse"
        );
        let settings = &mut m["engine_settings"];
        match engine {
            "redis" => {
                settings["index"]["stemmer"] =
                    json!("none: NOSTEM on the indexed text fields (verified via FT.INFO)");
                settings["index"]["field_type"] =
                    json!("TEXT NOSTEM, index WEIGHT 1, only the searched fields (verified)");
                settings["command"] = json!(
                    "FT.SEARCH <idx> <query> NOCONTENT VERBATIM WITHSCORES SCORER BM25STD \
                     LIMIT 0 <top> DIALECT 2 TIMEOUT <ms>"
                );
            }
            _ => {
                settings["index"]["analyzer"] = json!(OPENSEARCH_PARITY_ANALYZER);
                settings["index"]["analysis"] = opensearch_analysis_for(Analysis::Parity);
                settings["index"]["note"] = json!(
                    "the english pipeline without the porter2 filter; text fields that are not \
                     searched are mapped with index: false"
                );
            }
        }
    }
    Ok(m)
}

/// The warning a ranked fulltext result carries when its config does not
/// state `fulltext.total_hits` (printed at run time, and in `warnings`).
pub const ASYMMETRIC_TOTAL_HITS_WARNING: &str =
    "total_hits_symmetric: false: `fulltext.total_hits` is not stated, so the legacy \
     per-engine default ran (Redis counts every match exactly; OpenSearch sends \
     track_total_hits: false and may skip non-competitive documents). The Redis and \
     OpenSearch results of this config do different work; state \"exact\" or \"none\" \
     to compare them.";

/// Record in the result metadata the total-hits mode `plan` that `engine`'s
/// requests were built with: `total_hits`, `total_hits_symmetric`, and the
/// `engine_settings` request/command that carry it. A plan the config did not
/// state that leaves the pair asymmetric also adds a `warnings` entry.
pub fn record_total_hits(
    meta: &mut Value,
    engine: &str,
    plan: &TotalHitsPlan,
) -> Result<(), String> {
    let (sent, server_total) = match (engine, plan.mode) {
        ("opensearch", TotalHits::Exact) => (
            "track_total_hits: true",
            "exact: hits.total with relation eq (checked on every reply)",
        ),
        ("opensearch", TotalHits::None) => (
            "track_total_hits: false",
            "not computed: hits.total absent (checked on every reply)",
        ),
        ("redis", TotalHits::Exact) => (
            "FT.SEARCH without WITHOUTCOUNT (counted)",
            "exact: total_results",
        ),
        ("redis", TotalHits::None) => (
            "FT.SEARCH … WITHOUTCOUNT",
            "not computed: total_results is min(count, LIMIT) per shard and is not published",
        ),
        (other, _) => {
            return Err(format!(
                "fulltext: engine '{other}' has no total-hits contract"
            ))
        }
    };
    meta["total_hits"] = json!({
        "mode": plan.mode.as_str(),
        "declared": plan.declared,
        "sent": sent,
        "server_total": server_total,
    });
    if meta["compound"]["suite"] == crate::msmarco::disk_queries::SUITE {
        meta["compound"]["count_check"] = json!(match plan.mode {
            TotalHits::Exact => "every reply's exact total equals the query's expected count",
            TotalHits::None =>
                "every reply's page length equals min(expected count, top); no total is computed",
        });
    }
    meta["total_hits_symmetric"] = json!(plan.symmetric);
    match engine {
        "opensearch" => {
            meta["engine_settings"]["request"]["track_total_hits"] =
                json!(opensearch_track_total_hits(plan.mode));
        }
        _ => {
            if plan.mode == TotalHits::None {
                let command = meta["engine_settings"]["command"]
                    .as_str()
                    .ok_or("fulltext: Redis engine_settings has no command")?
                    .to_string();
                meta["engine_settings"]["command"] =
                    json!(format!("{command} {REDIS_WITHOUTCOUNT}"));
            }
        }
    }
    if !plan.symmetric {
        let warning = ASYMMETRIC_TOTAL_HITS_WARNING;
        match meta["warnings"].as_array_mut() {
            Some(w) => w.push(json!(warning)),
            None => meta["warnings"] = json!([warning]),
        }
    }
    Ok(())
}

fn result_metadata_english(
    engine: &str,
    fields: &[FieldBoost],
    set: &QuerySet,
    top: usize,
) -> Result<Value, String> {
    Ok(json!({
        "query_source": "tests.jsonl `query_text`",
        "queries_total": set.total,
        "queries_eligible": set.queries.len(),
        "queries_skipped_no_terms": set.skipped,
        "result_limit": top,
        "fields": fields
            .iter()
            .map(|f| (f.field.clone(), json!(f.boost)))
            .collect::<serde_json::Map<_, _>>(),
        "boosts_applied": "query time, on both engines",
        "operator": "or",
        "query_terms": {
            "method": "client side, identical for both engines: drop possessive 's, split on separators and all \
                       whitespace, lowercase, drop pinned stopwords, drop duplicates, keep order",
            "separators": REDISEARCH_SEPARATORS,
        },
        "stopwords": {
            "list": STOPWORDS.as_slice(),
            "provenance": "RediSearch DEFAULT_STOPWORDS (src/stopwords.h) and Lucene \
                           EnglishAnalyzer.ENGLISH_STOP_WORDS_SET are the same 33 words; both \
                           engines are given the list explicitly",
        },
        "stemmer": "Snowball English (Porter2) on both engines",
        "engine": engine,
        "engine_settings": engine_settings(engine)?,
        "known_differences": KNOWN_DIFFERENCES,
    }))
}

/// Opt-in text/scalar layout. Existing ranked profiles retain their layout.
/// No-vector declarations must agree across every fulltext search point and
/// can never coexist with KNN. Compound tags use the exact scalar contract.
pub fn apply_layout<'a>(
    plan: &mut crate::workload::WorkloadPlan,
    blocks: impl Iterator<Item = Option<&'a Value>>,
) -> Result<(), String> {
    use crate::workload::{IndexLayout, Workload};
    let specs = blocks.map(parse_spec).collect::<Result<Vec<_>, _>>()?;
    let no_vector = specs.iter().any(|s| s.no_vector_index);
    let compound = specs.iter().any(|s| s.compound.is_some());
    if no_vector {
        if specs.iter().any(|s| !s.no_vector_index)
            || plan.workloads.iter().any(|w| w.has_ground_truth())
        {
            return Err(
                "no_vector_index must agree across fulltext configs and cannot serve KNN".into(),
            );
        }
        plan.layout = IndexLayout::NoVector;
    } else if compound {
        if plan
            .workloads
            .iter()
            .any(|w| matches!(w, Workload::Vector | Workload::Hybrid | Workload::HybridRrf))
        {
            return Err("compound matching requires a dedicated text/scalar engine config".into());
        }
        plan.layout = IndexLayout::Full;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn compound_layout_preserves_legacy_ranked_no_vector_refusal() {
        use crate::workload::{self, Workload};
        for engine in ["redis", "opensearch"] {
            assert!(workload::plan(engine, &[Some(Workload::Fulltext)], true)
                .unwrap_err()
                .contains("--skip-vector-index"));
            let mut plan = workload::plan(engine, &[Some(Workload::Fulltext)], false).unwrap();
            let explicit = json!({"no_vector_index":true});
            apply_layout(&mut plan, [Some(&explicit)].into_iter()).unwrap();
            let mut ranked = workload::plan(engine, &[Some(Workload::Fulltext)], false).unwrap();
            apply_layout(&mut ranked, [None].into_iter()).unwrap();
        }
    }

    #[test]
    fn compound_opensearch_request_always_tracks_exact_totals() {
        let query = TextQuery {
            query_id: 0,
            terms: vec![],
            expression: Some(
                crate::compound_text::Expr::compile(
                    &json!({"title":{"match":{"text":"apple"}}}),
                    &json!({"title":"text"}),
                )
                .unwrap(),
            ),
            compound: None,
        };
        assert_eq!(
            query_opensearch(&[], &query, 10)["track_total_hits"],
            json!(true)
        );
    }

    #[test]
    fn text_tag_profile_is_explicit_and_cannot_mix_layouts() {
        let block = json!({"analysis":"parity","fields":{"title":1.0},"no_vector_index":true,
            "scalar_indexing":"text-tag-only","compound":{"queries":[{"name":"q","family":"tag_eq",
            "conditions":{"title":{"match":{"text":"apple"}}},"minimum_count":1}]}});
        let mut all = block.clone();
        all.as_object_mut().unwrap().remove("scalar_indexing");
        assert!(index_profile([
            (Workload::Fulltext, Some(&block)),
            (Workload::Fulltext, Some(&all))
        ])
        .is_err());
        let profile = index_profile([(Workload::Fulltext, Some(&block))])
            .unwrap()
            .unwrap();
        assert!(profile
            .check_marker(
                &index_profile([(Workload::Fulltext, Some(&all))])
                    .unwrap()
                    .unwrap()
                    .marker()
            )
            .is_err());
        for (key, value) in [
            ("scalar_indexing", json!(false)),
            ("no_vector_index", json!(false)),
            ("analysis", json!("english")),
        ] {
            let mut bad = block.clone();
            bad[key] = value;
            assert!(parse_spec(Some(&bad)).is_err());
        }
        let mut numeric = block;
        numeric["compound"]["queries"][0]["conditions"] = json!({"n":{"range":{"gte":1}}});
        assert!(check_fields_before_connecting(
            [(0, Some(&numeric))].into_iter(),
            Some(&json!({"title":"text","n":"int"})),
            "fixture"
        )
        .is_err());
    }

    #[test]
    fn compound_fingerprints_bind_queries_counts_and_preparation_bytes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("PREPARED.json"), "{}").unwrap();
        let schema = json!({"title":"text"});
        let block = json!({"analysis":"parity","fields":{"title":1.0},"compound":{"queries":[{"name":"q","family":"f","conditions":{"title":{"match":{"text":"apple"}}},"minimum_count":1}]}});
        let metadata = |block: &Value| {
            let spec = parse_spec(Some(block)).unwrap();
            let set = load_query_set(dir.path(), &spec, Some(&schema)).unwrap();
            let fields = resolve_fields(&spec, Some(&schema)).unwrap();
            let profile = IndexProfile {
                scalar_indexing: spec.scalar_indexing,
                analysis: Analysis::Parity,
                text_fields: ["title".into()].into(),
            };
            result_metadata_for("redis", &fields, &set, 10, &profile, &profile.marker()).unwrap()
        };
        let original = metadata(&block);
        assert_eq!(original, metadata(&block));
        let mut reduced = block.clone();
        reduced["scalar_indexing"] = json!("text-tag-only");
        reduced["no_vector_index"] = json!(true);
        let reduced_meta = metadata(&reduced);
        assert_ne!(
            original["compound"]["sha256"],
            reduced_meta["compound"]["sha256"]
        );
        assert_ne!(original["index_marker"], reduced_meta["index_marker"]);
        assert_eq!(
            original["compound"]["queries"],
            reduced_meta["compound"]["queries"]
        );
        assert_eq!(
            original["compound"]["preparation_sha256"],
            reduced_meta["compound"]["preparation_sha256"]
        );
        assert!(original["known_differences"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v
                .as_str()
                .unwrap()
                .contains("Compound scoring is engine-native")));
        let legacy_bytes = serde_json::to_vec(&original["compound"]["queries"]).unwrap();
        use sha2::{Digest, Sha256};
        let legacy_hash = Sha256::digest(legacy_bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_ne!(original["compound"]["sha256"], json!(legacy_hash));
        assert_eq!(
            original["compound"]["scoring_contract"],
            "native-engine-scoring-v1"
        );
        for (field, value) in [
            ("minimum_count", json!(2)),
            ("expected_count", json!(2)),
            ("conditions", json!({"title":{"match":{"text":"orange"}}})),
        ] {
            let mut changed = block.clone();
            changed["compound"]["queries"][0][field] = value;
            assert_ne!(
                original["compound"]["sha256"],
                metadata(&changed)["compound"]["sha256"]
            );
        }
        std::fs::write(dir.path().join("PREPARED.json"), "{}\n").unwrap();
        let changed = metadata(&block);
        assert_eq!(
            original["compound"]["sha256"],
            changed["compound"]["sha256"]
        );
        assert_ne!(
            original["compound"]["preparation_sha256"],
            changed["compound"]["preparation_sha256"]
        );
    }

    #[test]
    fn compound_cannot_silently_ignore_duration_or_offered_rate() {
        assert!(check_execution_params(None, None).is_ok());
        assert!(check_execution_params(Some(100.0), None).is_err());
        assert!(check_execution_params(None, Some(60.0)).is_err());
    }
    #[test]
    fn compound_layout_is_explicit_and_refuses_knn_or_conflicting_search_points() {
        use crate::workload::{self, IndexLayout, Workload};
        let block = json!({"fields":{"title":1.0,"segment":1.0},"compound":{"suite":"msmarco-v1"},"no_vector_index":true});
        let mut plan = workload::plan("redis", &[Some(Workload::Fulltext)], false).unwrap();
        apply_layout(&mut plan, [Some(&block)].into_iter()).unwrap();
        assert_eq!(plan.layout, IndexLayout::NoVector);
        let mut full = block.clone();
        full["no_vector_index"] = json!(false);
        apply_layout(&mut plan, [Some(&full)].into_iter()).unwrap();
        assert_eq!(plan.layout, IndexLayout::Full);
        assert!(apply_layout(&mut plan, [Some(&block), Some(&full)].into_iter()).is_err());
        let mut plan = workload::plan(
            "opensearch",
            &[Some(Workload::Fulltext), Some(Workload::Vector)],
            false,
        )
        .unwrap();
        assert!(apply_layout(&mut plan, [Some(&block)].into_iter()).is_err());
        assert!(apply_layout(&mut plan, [Some(&full)].into_iter()).is_err());
    }
    #[test]
    fn compound_preflight_rejects_wrong_field_type_and_unindexed_text() {
        let schema = json!({"title":"text","segment":"text","category":"keyword"});
        let block = json!({"fields":{"title":1.0},"compound":{"queries":[{"name":"q","family":"f","conditions":{"category":{"match":{"text":"apple"}}}}]}});
        assert!(check_fields_before_connecting(
            [(0, Some(&block))].into_iter(),
            Some(&schema),
            "fixture"
        )
        .is_err());
        let mut wrong = block.clone();
        wrong["compound"]["queries"][0]["conditions"] =
            json!({"segment":{"match":{"text":"apple"}}});
        let spec = parse_spec(Some(&wrong)).unwrap();
        assert!(load_query_set(Path::new("."), &spec, Some(&schema)).is_err());
    }

    use super::*;

    /// The list both engines are pinned to, written out from the two sources
    /// it was checked against. If either is edited, the engines would drop
    /// different words; the test names what changed.
    #[test]
    fn stopword_list_is_the_redisearch_and_lucene_default() {
        // RediSearch src/stopwords.h DEFAULT_STOPWORDS, in its own order.
        let redisearch = [
            "a", "is", "the", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
            "into", "it", "no", "not", "of", "on", "or", "such", "that", "their", "then", "there",
            "these", "they", "this", "to", "was", "will", "with",
        ];
        // Lucene EnglishAnalyzer.ENGLISH_STOP_WORDS_SET, in its own order.
        let lucene = [
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
            "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
            "there", "these", "they", "this", "to", "was", "will", "with",
        ];
        assert_eq!(STOPWORDS.len(), 33);
        assert_eq!(STOPWORDS, lucene);
        assert!(same_word_set(redisearch.iter().copied()));
        assert_eq!(
            STOPWORDS.iter().collect::<HashSet<_>>().len(),
            33,
            "duplicates"
        );
        // The FT.CREATE arguments carry exactly that list and its length.
        let args = redis_stopword_args();
        assert_eq!(args[0], "STOPWORDS");
        assert_eq!(args[1], "33");
        assert_eq!(&args[2..], STOPWORDS.map(String::from).as_slice());
        // And so does the OpenSearch stop filter.
        assert_eq!(
            opensearch_analysis()["filter"][OS_STOP]["stopwords"],
            json!(STOPWORDS.as_slice())
        );
    }

    #[test]
    fn opensearch_analyzer_is_english_with_porter2() {
        let a = opensearch_analysis();
        assert_eq!(
            a["analyzer"][OPENSEARCH_ANALYZER],
            json!({"type":"custom","tokenizer":"standard",
                   "filter":[OS_POSSESSIVE,"lowercase",OS_STOP,OS_STEMMER]})
        );
        assert_eq!(a["filter"][OS_STEMMER]["language"], "porter2");
        assert_eq!(a["filter"][OS_POSSESSIVE]["language"], "possessive_english");
    }

    #[test]
    fn query_terms_split_like_redisearch_and_drop_stopwords() {
        assert_eq!(
            query_terms("What is the U.S. e-mail rate, (2023)? Rate!"),
            vec!["what", "u", "s", "e", "mail", "rate", "2023"]
        );
        // Possessive 's is dropped on the query side (both engines), so no
        // bare `s` term reaches Redis; other apostrophes still split.
        assert_eq!(query_terms("dog's bone"), vec!["dog", "bone"]);
        assert_eq!(query_terms("DOG\u{2019}S bone's"), vec!["dog", "bone"]);
        assert_eq!(
            query_terms("don't sing 'solo' cats' s"),
            vec!["don", "t", "sing", "solo", "cats", "s"]
        );
        assert_eq!(query_terms("it's"), Vec::<String>::new());
        assert_eq!(query_terms("'s"), vec!["s"]);
        assert!(query_terms("the a an IS ... --").is_empty());
        // Newlines and other whitespace separate on the query side.
        assert_eq!(
            query_terms("alpha\nbeta\u{a0}gamma"),
            vec!["alpha", "beta", "gamma"]
        );
        // `_` is not a separator, but a term must hold a letter or digit.
        assert_eq!(query_terms("foo_bar _ __"), vec!["foo_bar"]);
    }

    /// Every query-syntax character is a separator, so no term can carry one
    /// into the RediSearch query.
    #[test]
    fn terms_never_carry_redisearch_syntax() {
        let nasty = "@title:{x} | -y ~z *w% $p (a) [1 2] \"q\" 'r' \\s =>{$weight:9}; <> ! ^ & # ?";
        for t in query_terms(nasty) {
            assert!(
                !t.chars()
                    .any(|c| REDISEARCH_SEPARATORS.contains(c) || c.is_whitespace()),
                "{t:?}"
            );
        }
        for c in "@|-~*%$()[]{}\"':;=<>!^&#?\\".chars() {
            assert!(REDISEARCH_SEPARATORS.contains(c), "{c}");
        }
    }

    fn fb(f: &str, b: f64) -> FieldBoost {
        FieldBoost {
            field: f.into(),
            boost: b,
        }
    }

    #[test]
    fn redis_query_weights_every_field_clause() {
        let q = redis_query(
            &[fb("segment", 1.0), fb("title", 5.0)],
            &["apple".into(), "pie".into()],
        );
        assert_eq!(
            q,
            "(@segment:(apple|pie)) => {$weight: 1.0;} | (@title:(apple|pie)) => {$weight: 5.0;}"
        );
        let args = redis_args("idx", &q, 7, 500);
        assert_eq!(
            args,
            [
                "idx",
                &q,
                "NOCONTENT",
                "WITHSCORES",
                "SCORER",
                "BM25STD",
                "LIMIT",
                "0",
                "7",
                "DIALECT",
                "2",
                "TIMEOUT",
                "500"
            ]
        );
    }

    #[test]
    fn opensearch_body_is_multi_match_most_fields_with_boosts() {
        let b = opensearch_body(
            &[fb("headings", 3.0), fb("title", 5.5)],
            &["apple".into(), "pie".into()],
            9,
        );
        assert_eq!(
            b,
            json!({"size":9,"_source":false,"track_total_hits":false,
                   "query":{"multi_match":{"query":"apple pie","type":"most_fields",
                   "operator":"or","fields":["headings^3.0","title^5.5"]}}})
        );
    }

    fn schema() -> Value {
        json!({"docid":"keyword","title":"text","headings":"text","segment":"text","n":"int"})
    }

    #[test]
    fn fields_resolve_against_schema_text_fields() {
        let all = resolve_fields(&FulltextSpec::default(), Some(&schema())).unwrap();
        assert_eq!(
            all,
            vec![fb("headings", 1.0), fb("segment", 1.0), fb("title", 1.0)]
        );
        let spec = parse_spec(Some(&json!({"fields":{"title":5.0,"segment":1}}))).unwrap();
        assert_eq!(
            resolve_fields(&spec, Some(&schema())).unwrap(),
            vec![fb("segment", 1.0), fb("title", 5.0)]
        );
        for bad in ["docid", "n", "missing"] {
            let spec = parse_spec(Some(&json!({"fields":{bad:1.0}}))).unwrap();
            let e = resolve_fields(&spec, Some(&schema())).unwrap_err();
            assert!(e.contains(bad), "{e}");
        }
        assert!(resolve_fields(&FulltextSpec::default(), Some(&json!({"n":"int"}))).is_err());
        assert!(resolve_fields(&FulltextSpec::default(), None).is_err());
    }

    #[test]
    fn spec_rejects_unknown_keys_bad_boosts_and_misplacement() {
        assert!(parse_spec(Some(&json!({"feilds":{"title":1.0}}))).is_err());
        assert!(parse_spec(Some(&json!({"fields":{}}))).is_err());
        for b in [json!(0.0), json!(-1.0), json!("5")] {
            assert!(
                parse_spec(Some(&json!({"fields":{"title":b}}))).is_err(),
                "{b}"
            );
        }
        assert!(validate_declared("redis", Workload::Fulltext, None).is_ok());
        assert!(validate_declared("redis", Workload::Vector, None).is_ok());
        for w in [Workload::Vector, Workload::Hybrid, Workload::FilterOnly] {
            assert!(
                validate_declared("redis", w, Some(&json!({}))).is_err(),
                "{w}"
            );
        }
        assert!(validate_declared("redis", Workload::Fulltext, Some(&json!({"x":1}))).is_err());
    }

    #[test]
    fn allow_multi_shard_is_opensearch_only() {
        let b = json!({"allow_multi_shard": true});
        assert!(validate_declared("opensearch", Workload::Fulltext, Some(&b)).is_ok());
        let e = validate_declared("redis", Workload::Fulltext, Some(&b)).unwrap_err();
        assert!(e.contains("only applies to OpenSearch"), "{e}");
        let off = json!({"allow_multi_shard": false});
        assert!(validate_declared("redis", Workload::Fulltext, Some(&off)).is_ok());
    }

    #[test]
    fn filtered_fields_cover_every_condition_type_at_any_depth() {
        let f = |v: Value| filtered_fields(&v).into_iter().collect::<Vec<_>>();
        assert_eq!(
            f(json!({"and":[{"body":{"match":{"text":"quick"}}}]})),
            ["body"]
        );
        assert_eq!(
            f(json!({"and":[{"body":{"match":{"value":"x"}}}]})),
            ["body"]
        );
        assert_eq!(
            f(json!({"and":[{"body":{"match":{"any":["x"]}}}]})),
            ["body"]
        );
        assert_eq!(
            f(
                json!({"or":[{"n":{"range":{"gt":1}}},{"and":[{"b":{"match":{"text":"x"}}},
                       {"or":[{"c":{"match":{"value":1}}}]}]}]})
            ),
            ["b", "c", "n"]
        );
        assert!(f(json!(null)).is_empty());
        assert!(f(json!({})).is_empty());
        let text = vec!["body".to_string(), "title".to_string()];
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(count_text_field_queries(dir.path(), &text).unwrap(), 0);
        std::fs::write(
            dir.path().join("tests.jsonl"),
            "{\"conditions\":{\"and\":[{\"body\":{\"match\":{\"text\":\"q\"}}}]}}\n\
             {\"conditions\":{\"and\":[{\"title\":{\"match\":{\"value\":\"r\"}}}]}}\n\
             {\"conditions\":{\"or\":[{\"body\":{\"match\":{\"any\":[\"s\"]}}}]}}\n\
             {\"conditions\":{\"and\":[{\"tag\":{\"match\":{\"value\":\"t\"}}}]}}\n\
             {\"query\":[1]}\n",
        )
        .unwrap();
        // text, value and any on text fields count; a keyword filter does not.
        assert_eq!(count_text_field_queries(dir.path(), &text).unwrap(), 3);
        assert_eq!(count_text_field_queries(dir.path(), &[]).unwrap(), 0);
    }

    #[test]
    fn fulltext_may_not_share_an_index_with_text_filtered_vector_configs() {
        use Workload::*;
        for others in [vec![Vector], vec![Hybrid], vec![Vector, Hybrid]] {
            let mut w = others.clone();
            w.push(Fulltext);
            assert!(mixes_with_filtered_workloads(&w));
            let e = check_mixed_text_filters(&w, 3, "ds").unwrap_err();
            assert!(e.contains("3 queries of dataset 'ds'"), "{e}");
            // No text filter in the dataset: the analyzer change is invisible.
            assert!(check_mixed_text_filters(&w, 0, "ds").is_ok());
        }
        // Fulltext alone, or no fulltext at all, is never refused.
        for w in [vec![Fulltext, Fulltext], vec![Vector], vec![FilterOnly]] {
            assert!(!mixes_with_filtered_workloads(&w));
            assert!(check_mixed_text_filters(&w, 3, "ds").is_ok());
        }
    }

    #[test]
    fn field_names_must_be_plain_identifiers() {
        for ok in ["title", "_x", "seg_2", "A1"] {
            assert!(is_plain_field_name(ok), "{ok}");
        }
        for bad in [
            "",
            "2nd",
            "page-title",
            "a.b",
            "a b",
            "ti@tle",
            "tïtle",
            "a|b",
        ] {
            assert!(!is_plain_field_name(bad), "{bad}");
        }
        let schema = json!({"page-title": "text", "body": "text"});
        // Named explicitly, or picked up as a default field: refused either way.
        let spec = parse_spec(Some(&json!({"fields": {"page-title": 1.0}}))).unwrap();
        let e = resolve_fields(&spec, Some(&schema)).unwrap_err();
        assert!(e.contains("'page-title' is not a plain identifier"), "{e}");
        assert!(resolve_fields(&FulltextSpec::default(), Some(&schema)).is_err());
        let spec = parse_spec(Some(&json!({"fields": {"body": 1.0}}))).unwrap();
        assert!(resolve_fields(&spec, Some(&schema)).is_ok());
    }

    #[test]
    fn pre_connect_field_errors_name_the_search_config_by_its_original_index() {
        // In a mixed engine config the fulltext configs are a filtered subset;
        // the error must name the same `search config #i` as the startup check.
        let schema = json!({"title": "text", "segment": "text"});
        let good = json!({"fields": {"title": 1.0}});
        let typo = json!({"fields": {"titel": 1.0}});
        let blocks = [(1, Some(&good)), (3, Some(&typo))];
        let e =
            check_fields_before_connecting(blocks.into_iter(), Some(&schema), "ds").unwrap_err();
        assert!(
            e.starts_with("search config #3 (fulltext) on dataset 'ds'"),
            "{e}"
        );
    }

    #[test]
    fn redis_min_stem_len_must_be_the_default_or_recorded_unverified() {
        assert_eq!(
            check_redis_min_stem_len(Some(4)).unwrap(),
            json!({"value": 4, "verified": true})
        );
        for n in [0, 2, 3, 5] {
            let e = check_redis_min_stem_len(Some(n)).unwrap_err();
            assert!(e.contains(&format!("search-min-stem-len is {n}")), "{e}");
        }
        let u = check_redis_min_stem_len(None).unwrap();
        assert_eq!(u["verified"], false);
        assert!(u["value"].is_null());
        // Reply shapes: RESP3 map, RESP2 flat pair, FT.CONFIG nested pairs.
        assert_eq!(
            config_get_value(&json!({"search-min-stem-len": "4"}), "search-min-stem-len"),
            Some(4)
        );
        assert_eq!(
            config_get_value(&json!(["search-min-stem-len", "3"]), "search-min-stem-len"),
            Some(3)
        );
        assert_eq!(
            config_get_value(&json!([["MINSTEMLEN", "5"]]), "MINSTEMLEN"),
            Some(5)
        );
        assert_eq!(config_get_value(&json!([]), "MINSTEMLEN"), None);
        assert_eq!(config_get_value(&json!({}), "search-min-stem-len"), None);
        assert_eq!(
            config_get_value(&json!({"other": "4"}), "search-min-stem-len"),
            None
        );
    }

    fn parity_block(fields: Value) -> Value {
        json!({"analysis": "parity", "fields": fields})
    }

    #[test]
    fn index_profile_resolves_one_mode_per_engine_config() {
        use Workload::*;
        let ft = |b: &Value| (Fulltext, Some(b.clone()));
        let eng = json!({"fields": {"title": 1.0}});
        let par = parity_block(json!({"segment": 1.0, "headings": 2.0}));
        let profile = |cfgs: Vec<(Workload, Option<Value>)>| {
            index_profile(cfgs.iter().map(|(w, b)| (*w, b.as_ref())))
        };
        // No fulltext config: no profile.
        assert_eq!(profile(vec![(Vector, None)]).unwrap(), None);
        // English (default), alone or with vector configs.
        let p = profile(vec![(Vector, None), ft(&eng), (Fulltext, None)])
            .unwrap()
            .unwrap();
        assert_eq!(p, IndexProfile::english());
        assert!(p.stems() && !p.normalizes_whitespace() && p.indexes_text_field("anything"));
        // Parity: exactly the searched fields, same set across configs (boosts may differ).
        let par2 = parity_block(json!({"segment": 5.0, "headings": 1.0}));
        let p = profile(vec![ft(&par), ft(&par2)]).unwrap().unwrap();
        assert_eq!(p.analysis, Analysis::Parity);
        assert!(!p.stems() && p.normalizes_whitespace());
        assert!(p.indexes_text_field("segment") && !p.indexes_text_field("title"));
        assert_eq!(
            p.marker(),
            json!({"analysis":"parity","normalize_whitespace":true,"text_fields":["headings","segment"]})
        );
        // Refusals.
        let e = profile(vec![ft(&eng), ft(&par)]).unwrap_err();
        assert!(e.contains("mixes fulltext analysis"), "{e}");
        let e = profile(vec![ft(&parity_block(json!({"segment": 1.0}))), ft(&par)]).unwrap_err();
        assert!(e.contains("different field sets"), "{e}");
        let e = profile(vec![ft(&json!({"analysis": "parity"}))]).unwrap_err();
        assert!(e.contains("needs explicit"), "{e}");
        for other in [Vector, Hybrid, FilterOnly] {
            let e = profile(vec![(other, None), ft(&par)]).unwrap_err();
            assert!(
                e.contains("must never apply to other workloads"),
                "{other}: {e}"
            );
        }
        assert!(parse_spec(Some(&json!({"analysis": "porter"}))).is_err());
    }

    #[test]
    fn newline_normalization_touches_only_the_indexed_text_fields() {
        use crate::readers::metadata::{MetadataItem, MetadataValue};
        assert_eq!(normalize_whitespace("a\r\nb\nc\rd"), "a  b c d");
        // Every Unicode whitespace other than U+0020 seen in MS MARCO, and more.
        assert_eq!(
            normalize_whitespace("a\tb\x0Bc\x0Cd\u{85}e\u{A0}f\u{2028}g\u{2029}h\u{3000}i"),
            "a b c d e f g h i"
        );
        assert_eq!(normalize_whitespace("plain text"), "plain text");
        assert_eq!(normalize_whitespace("plain"), "plain");
        let item = |h: &str, t: &str| {
            Some(MetadataItem {
                fields: vec![
                    ("headings".into(), MetadataValue::String(h.into())),
                    ("title".into(), MetadataValue::String(t.into())),
                    ("n".into(), MetadataValue::Int(1)),
                ],
            })
        };
        let mut items = vec![item("a\nb", "x\ny"), None, item("c", "z")];
        let fields: BTreeSet<String> = ["headings".to_string()].into();
        assert_eq!(normalize_metadata(&mut items, &fields), Ok(1));
        let text = |i: usize, k: usize| match &items[i].as_ref().unwrap().fields[k].1 {
            MetadataValue::String(s) => s.clone(),
            _ => unreachable!(),
        };
        assert_eq!(text(0, 0), "a b");
        assert_eq!(
            text(0, 1),
            "x\ny",
            "a field that is not indexed is left as uploaded"
        );
        assert_eq!(text(2, 0), "c");
        // A non-string value of an indexed text field is refused (an array
        // payload becomes Labels); outside the indexed fields it is ignored.
        let labels = |field: &str| {
            Some(MetadataItem {
                fields: vec![(
                    field.into(),
                    MetadataValue::Labels(vec!["a\nb".into(), "c".into()]),
                )],
            })
        };
        let e = normalize_metadata(&mut [None, labels("headings")], &fields).unwrap_err();
        assert!(e.contains("row 1 field 'headings' is not a string"), "{e}");
        assert_eq!(normalize_metadata(&mut [labels("tags")], &fields), Ok(0));
    }

    #[test]
    fn parity_refuses_a_redis_shared_corpus() {
        let par = index_profile([(
            Workload::Fulltext,
            Some(&parity_block(json!({"segment": 1}))),
        )])
        .unwrap()
        .unwrap();
        let eng = IndexProfile::english();
        let shared = Some("bench:");
        assert!(check_redis_shared_prefix("redis", Some(&par), shared)
            .unwrap_err()
            .contains("REDIS_KEY_PREFIX"));
        assert!(check_redis_shared_prefix("redis", Some(&par), None).is_ok());
        assert!(check_redis_shared_prefix("redis", Some(&eng), shared).is_ok());
        assert!(check_redis_shared_prefix("redis", None, shared).is_ok());
        assert!(check_redis_shared_prefix("opensearch", Some(&par), shared).is_ok());
        // A shared prefix covering the marker keys would count them as corpus.
        for p in [
            "vdbb:",
            "vdbb:fulltext-profile:",
            "vdbb:fulltext-profile:idx:",
        ] {
            let e = check_redis_shared_prefix("redis", Some(&eng), Some(p)).unwrap_err();
            assert!(e.contains("overlaps the fulltext marker keys"), "{p}: {e}");
        }
        for p in ["v:", "vdbbx:", "vdb:", "bench:"] {
            assert!(
                check_redis_shared_prefix("redis", Some(&eng), Some(p)).is_ok(),
                "{p}"
            );
        }
        // Vector-only configs write no marker, so any prefix is fine for them.
        assert!(check_redis_shared_prefix("redis", None, Some("vdbb:")).is_ok());
        assert_eq!(resolve_shared_prefix(Some(" vdbb ")), Some("vdbb:".into()));
        assert_eq!(resolve_shared_prefix(Some("a:")), Some("a:".into()));
        assert_eq!(resolve_shared_prefix(Some("  ")), None);
        assert_eq!(resolve_shared_prefix(None), None);
    }

    /// #358 review 5: a marker failure is a note in english mode and a
    /// refusal naming the needed key pattern in parity mode.
    #[test]
    fn marker_failures_are_fatal_only_in_parity() {
        let eng = IndexProfile::english();
        let note = redis_marker_failure(&eng, "write", "NOPERM").unwrap();
        assert!(
            note.contains("index_marker: null") && note.contains("NOPERM"),
            "{note}"
        );
        let par = index_profile([(
            Workload::Fulltext,
            Some(&parity_block(json!({"segment": 1}))),
        )])
        .unwrap()
        .unwrap();
        let e = redis_marker_failure(&par, "read", "NOPERM").unwrap_err();
        assert!(
            e.contains("'vdbb:fulltext-profile:*'") && e.contains("NOPERM"),
            "{e}"
        );
    }

    /// #358 review 3: a vector-only config never touches a key outside its
    /// own prefix; a fulltext one clears a stale marker and writes its own.
    #[test]
    fn marker_commands_only_for_fulltext_configs() {
        let eng = IndexProfile::english();
        assert!(redis_marker_create_commands("idx:v", None).is_empty());
        assert!(redis_marker_teardown_commands("idx:v", None).is_empty());
        let key = "vdbb:fulltext-profile:idx:f".to_string();
        assert_eq!(
            redis_marker_create_commands("idx:f", Some(&eng)),
            vec![
                vec!["DEL".to_string(), key.clone()],
                vec!["SET".to_string(), key.clone(), eng.marker().to_string()],
            ]
        );
        assert_eq!(
            redis_marker_teardown_commands("idx:f", Some(&eng)),
            vec![vec!["DEL".to_string(), key]]
        );
    }

    #[test]
    fn markers_must_match_the_mode() {
        let eng = IndexProfile::english();
        let par = index_profile([(
            Workload::Fulltext,
            Some(&parity_block(json!({"segment": 1}))),
        )])
        .unwrap()
        .unwrap();
        // English accepts no marker (older or vector-built indexes) or its own.
        assert!(eng.check_marker(&Value::Null).is_ok());
        assert!(eng.check_marker(&eng.marker()).is_ok());
        assert!(eng.check_marker(&par.marker()).is_err());
        // Parity needs its own marker: an unnormalized upload is refused.
        assert!(par.check_marker(&par.marker()).is_ok());
        let e = par.check_marker(&Value::Null).unwrap_err();
        assert!(e.contains("without whitespace normalization"), "{e}");
        assert!(par.check_marker(&eng.marker()).is_err());
        let mut other = par.marker();
        other["text_fields"] = json!(["headings", "segment"]);
        assert!(par.check_marker(&other).is_err());
    }

    #[test]
    fn parity_requests_and_analyzers() {
        let a = redis_args_for("idx", "q", 5, 100, Analysis::Parity);
        assert_eq!(a[..4], ["idx", "q", "NOCONTENT", "VERBATIM"]);
        assert!(!redis_args_for("idx", "q", 5, 100, Analysis::English)
            .contains(&"VERBATIM".to_string()));
        assert_eq!(
            redis_args_for("idx", "q", 5, 100, Analysis::English),
            redis_args("idx", "q", 5, 100)
        );
        let an = opensearch_analysis_for(Analysis::Parity);
        assert_eq!(
            an["analyzer"][OPENSEARCH_PARITY_ANALYZER]["filter"],
            json!([OS_POSSESSIVE, "lowercase", OS_STOP])
        );
        assert!(an["filter"][OS_STEMMER].is_null());
        assert_eq!(
            an["filter"][OS_STOP],
            opensearch_analysis()["filter"][OS_STOP]
        );
        assert_eq!(
            opensearch_analysis_for(Analysis::English),
            opensearch_analysis()
        );
        let par = index_profile([(
            Workload::Fulltext,
            Some(&parity_block(json!({"segment": 1}))),
        )])
        .unwrap()
        .unwrap();
        assert_eq!(
            opensearch_text_mapping_for(&par, "segment")["analyzer"],
            OPENSEARCH_PARITY_ANALYZER
        );
        assert_eq!(
            opensearch_text_mapping_for(&par, "title"),
            json!({"type":"text","index":false})
        );
        assert_eq!(
            opensearch_text_mapping_for(&IndexProfile::english(), "title"),
            opensearch_text_mapping()
        );
    }

    #[test]
    fn parity_redis_index_is_nostem_searched_fields_only_and_marked() {
        let par = index_profile([(
            Workload::Fulltext,
            Some(&parity_block(json!({"segment": 1, "title": 1}))),
        )])
        .unwrap()
        .unwrap();
        let fields = [fb("segment", 1.0), fb("title", 1.0)];
        let nostem = |n: &str| json!({"identifier":n,"attribute":n,"type":"TEXT","WEIGHT":"1","flags":["SORTABLE","NOSTEM"]});
        let info = |attrs: Value| redis_info(Some(STOPWORDS.to_vec()), attrs);
        let good = json!([nostem("segment"), nostem("title")]);
        assert!(
            validate_redis_index_for(&info(good.clone()), &fields, &par, &par.marker()).is_ok()
        );
        // Stemmed field (an english-built index) refused in parity...
        let stemmed = json!([text_attr("segment"), nostem("title")]);
        let e = validate_redis_index_for(&info(stemmed), &fields, &par, &par.marker()).unwrap_err();
        assert!(e.contains("needs NOSTEM"), "{e}");
        // ...and a NOSTEM field refused in english.
        let e = validate_redis_index_for(
            &info(good.clone()),
            &fields,
            &IndexProfile::english(),
            &Value::Null,
        )
        .unwrap_err();
        assert!(e.contains("NOSTEM"), "{e}");
        // An extra TEXT field (not searched) refused in parity.
        let extra = json!([nostem("segment"), nostem("title"), text_attr("headings")]);
        let e = validate_redis_index_for(&info(extra), &fields, &par, &par.marker()).unwrap_err();
        assert!(e.contains("'headings'"), "{e}");
        // Marker: absent or english = unnormalized upload, refused in parity.
        assert!(
            validate_redis_index_for(&info(good.clone()), &fields, &par, &Value::Null).is_err()
        );
        assert!(validate_redis_index_for(
            &info(good),
            &fields,
            &par,
            &IndexProfile::english().marker()
        )
        .is_err());
    }

    #[test]
    fn parity_opensearch_index_is_nostem_searched_fields_only_and_marked() {
        let par = index_profile([(
            Workload::Fulltext,
            Some(&parity_block(json!({"segment": 1}))),
        )])
        .unwrap()
        .unwrap();
        let fields = [fb("segment", 1.0)];
        let schema_text = vec!["segment".to_string(), "title".to_string()];
        let props = json!({
            "segment": opensearch_text_mapping_for(&par, "segment"),
            "title": opensearch_text_mapping_for(&par, "title"),
            "docid": {"type": "keyword"},
        });
        let index =
            json!({"number_of_shards":"1","analysis": opensearch_analysis_for(Analysis::Parity)});
        let check = |p: &Value, i: &Value, m: &Value| {
            validate_opensearch_index_for(p, i, &fields, false, &par, m, &schema_text)
        };
        // A payload key outside the schema, dynamically mapped as text, does not
        // count (per-field statistics) — the 100K corpus's derived fields hit this.
        let mut p = props.clone();
        p["category"] = json!({"type": "text", "fields": {"keyword": {"type": "keyword"}}});
        assert_eq!(check(&p, &index, &par.marker()), Ok(1));
        assert_eq!(check(&props, &index, &par.marker()), Ok(1));
        // Marker missing (unnormalized upload) or english.
        assert!(check(&props, &index, &Value::Null).is_err());
        assert!(check(&props, &index, &IndexProfile::english().marker()).is_err());
        // English analysis (porter2) refused in parity.
        let mut p = props.clone();
        p["segment"] = opensearch_text_mapping();
        assert!(check(&p, &index, &par.marker()).is_err());
        let mut i = index.clone();
        i["analysis"]["analyzer"][OPENSEARCH_PARITY_ANALYZER]["filter"] =
            json!([OS_POSSESSIVE, "lowercase", OS_STOP, OS_STEMMER]);
        assert!(check(&props, &i, &par.marker()).is_err());
        // A non-searched text field that is indexed.
        let mut p = props.clone();
        p["title"] = json!({"type":"text","index":true,"analyzer":OPENSEARCH_PARITY_ANALYZER});
        assert!(check(&p, &index, &par.marker())
            .unwrap_err()
            .contains("'title'"));
        // And the parity index reused in english mode.
        assert!(validate_opensearch_index_for(
            &props,
            &index,
            &fields,
            false,
            &IndexProfile::english(),
            &par.marker(),
            &schema_text
        )
        .is_err());
    }

    #[test]
    fn parity_metadata_records_the_mode_and_drops_removed_differences() {
        let set = build_query_set(&["apple".into()]).unwrap();
        let par = index_profile([(
            Workload::Fulltext,
            Some(&parity_block(json!({"segment": 1}))),
        )])
        .unwrap()
        .unwrap();
        for engine in ["redis", "opensearch"] {
            let m =
                result_metadata_for(engine, &[fb("segment", 1.0)], &set, 10, &par, &par.marker())
                    .unwrap();
            assert_eq!(m["analysis"], "parity");
            assert_eq!(m["normalize_whitespace"], true);
            assert_eq!(m["indexed_text_fields"], json!(["segment"]));
            assert_eq!(m["index_marker"], par.marker());
            // By identity, whatever the order of KNOWN_DIFFERENCES: parity
            // keeps every entry except exactly the ones it removes.
            let kd: Vec<&str> = m["known_differences"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_str().unwrap())
                .collect();
            for removed in PARITY_REMOVES {
                assert!(!kd.contains(&removed), "{removed}");
            }
            // Named, not via PARITY_REMOVES: what parity actually removes
            // (whitespace normalized, nothing stemmed) must not be listed.
            for removed in [KD_WHITESPACE_TOKENS, KD_STEM_TWICE, KD_MIN_STEM_LEN] {
                assert!(!kd.contains(&removed), "{removed}");
            }
            let kept: Vec<&str> = KNOWN_DIFFERENCES
                .into_iter()
                .filter(|d| !PARITY_REMOVES.contains(d))
                .collect();
            assert_eq!(kd, kept);
            assert_eq!(kd.len(), KNOWN_DIFFERENCES.len() - PARITY_REMOVES.len());
            // Parity normalizes whitespace only: the punctuation difference stays.
            for d in [
                KD_UNICODE_PUNCTUATION,
                KD_PUNCTUATION,
                KD_WHOLE_DOCUMENT,
                KD_SHARD_STATS,
                KD_TIES,
            ] {
                assert!(kd.contains(&d), "{d}");
            }
            // English reuse of an index built before markers existed: the
            // result records what the index carries (nothing), not the
            // marker this run would have written.
            let e = result_metadata_for(
                engine,
                &[fb("segment", 1.0)],
                &set,
                10,
                &IndexProfile::english(),
                &Value::Null,
            )
            .unwrap();
            assert!(e["index_marker"].is_null(), "{}", e["index_marker"]);
            assert_eq!(e["index_marker_expected"], IndexProfile::english().marker());
            assert_eq!(m["index_marker"], par.marker());
            assert_eq!(m["index_marker_expected"], par.marker());
            assert_eq!(e["analysis"], "english");
            assert_eq!(e["normalize_whitespace"], false);
            assert_eq!(
                e["known_differences"].as_array().unwrap().len(),
                KNOWN_DIFFERENCES.len()
            );
        }
        let r = result_metadata_for(
            "redis",
            &[fb("segment", 1.0)],
            &set,
            10,
            &par,
            &par.marker(),
        )
        .unwrap();
        assert!(r["engine_settings"]["command"]
            .as_str()
            .unwrap()
            .contains("VERBATIM"));
        let o = result_metadata_for(
            "opensearch",
            &[fb("segment", 1.0)],
            &set,
            10,
            &par,
            &par.marker(),
        )
        .unwrap();
        assert_eq!(
            o["engine_settings"]["index"]["analyzer"],
            OPENSEARCH_PARITY_ANALYZER
        );
    }

    /// The shipped MS MARCO fulltext configs resolve to the mode their name
    /// says: `*-english` to the headline english profile, `*-parity` to parity
    /// over exactly the searched fields, so the two run side by side.
    #[test]
    fn shipped_msmarco_fulltext_configs_resolve_to_their_mode() {
        for file in [
            "redis-msmarco-fulltext.json",
            "opensearch-msmarco-fulltext.json",
            "redis-msmarco-hybrid-rrf.json",
            "opensearch-msmarco-hybrid-rrf.json",
        ] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("experiments/configurations")
                .join(file);
            let configs: Vec<Value> =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let mut modes = Vec::new();
            for c in &configs {
                let name = c["name"].as_str().unwrap();
                let sps = c["search_params"].as_array().unwrap();
                let profile = index_profile(sps.iter().map(|sp| {
                    let w: Workload = serde_json::from_value(sp["workload"].clone()).unwrap();
                    (w, sp.get(SPEC_KEY))
                }))
                .unwrap()
                .unwrap_or_else(|| panic!("{file} [{name}] has no fulltext config"));
                let want = if name.ends_with("-parity") {
                    Analysis::Parity
                } else {
                    assert!(name.ends_with("-english"), "{name}");
                    Analysis::English
                };
                assert_eq!(profile.analysis, want, "{file} [{name}]");
                if want == Analysis::Parity {
                    assert_eq!(profile.text_fields, BTreeSet::from(["segment".to_string()]));
                }
                modes.push(want);
            }
            assert_eq!(modes, [Analysis::English, Analysis::Parity], "{file}");
        }
    }

    #[test]
    fn query_set_skips_termless_queries_and_refuses_an_empty_set() {
        let texts: Vec<String> = ["apple pie", "the of a", "Pie!"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let s = build_query_set(&texts).unwrap();
        assert_eq!(s.total, 3);
        assert_eq!(s.skipped, vec![1]);
        assert_eq!(
            s.queries.iter().map(|q| q.query_id).collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert!(build_query_set(&["the".to_string()]).is_err());
    }

    #[test]
    fn schedule_runs_each_once_or_cycles() {
        assert_eq!(schedule(3, -1), vec![0, 1, 2]);
        assert_eq!(schedule(3, 0), vec![0, 1, 2]);
        assert_eq!(schedule(2, 5), vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn query_texts_are_required_on_every_line() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tests.jsonl");
        std::fs::write(&p, "{\"query_text\":\"a b\"}\n\n{\"query_text\":\"c\"}\n").unwrap();
        assert_eq!(read_query_texts(dir.path()).unwrap(), vec!["a b", "c"]);
        std::fs::write(&p, "{\"query_text\":\"a b\"}\n{\"query\":[1]}\n").unwrap();
        assert!(read_query_texts(dir.path()).unwrap_err().contains("line 2"));
        std::fs::write(&p, "{\"query_text\":\"  \"}\n").unwrap();
        assert!(read_query_texts(dir.path()).is_err());
        std::fs::write(&p, "").unwrap();
        assert!(read_query_texts(dir.path()).is_err());
        assert!(read_query_texts(&dir.path().join("nope")).is_err());
    }

    fn redis_info(stop: Option<Vec<&str>>, attrs: Value) -> Value {
        // The RESP3 shape of a default-built index (redis 8.8).
        let mut v = json!({
            "attributes": attrs,
            "index_options": [],
            "index_definition": {"key_type": "HASH", "prefixes": ["p:"], "default_score": 1.0},
        });
        if let Some(s) = stop {
            v["stopwords_list"] = json!(s);
        }
        v
    }

    fn text_attr(name: &str) -> Value {
        json!({"identifier":name,"attribute":name,"type":"TEXT","WEIGHT":"1","flags":["SORTABLE"]})
    }

    #[test]
    fn redis_index_must_have_every_field_as_pinned() {
        let fields = [fb("title", 5.0), fb("segment", 1.0)];
        let good = || {
            json!([
                text_attr("title"),
                text_attr("segment"),
                text_attr("headings")
            ])
        };
        let stop = || Some(STOPWORDS.to_vec());
        assert!(validate_redis_index(&redis_info(stop(), good()), &fields).is_ok());
        // Missing field -> hard error.
        let e = validate_redis_index(&redis_info(stop(), json!([text_attr("title")])), &fields)
            .unwrap_err();
        assert!(e.contains("'segment' is not in the Redis index"), "{e}");
        // Not TEXT.
        let mut a = good();
        a[1]["type"] = json!("TAG");
        assert!(validate_redis_index(&redis_info(stop(), a), &fields).is_err());
        // Index-time weight.
        let mut a = good();
        a[0]["WEIGHT"] = json!(5);
        assert!(validate_redis_index(&redis_info(stop(), a), &fields).is_err());
        // NOSTEM.
        let mut a = good();
        a[0]["flags"] = json!(["NOSTEM"]);
        assert!(validate_redis_index(&redis_info(stop(), a), &fields).is_err());
        // Default (unreported) or different stopwords.
        assert!(validate_redis_index(&redis_info(None, good()), &fields).is_err());
        let mut short = STOPWORDS.to_vec();
        short.pop();
        assert!(validate_redis_index(&redis_info(Some(short.clone()), good()), &fields).is_err());
        short.push("hello");
        assert!(validate_redis_index(&redis_info(Some(short), good()), &fields).is_err());
        // Numeric WEIGHT 1 also accepted.
        let mut a = good();
        a[0]["WEIGHT"] = json!(1);
        assert!(validate_redis_index(&redis_info(stop(), a), &fields).is_ok());
        // PHONETIC field.
        let mut a = good();
        a[1]["PHONETIC"] = json!("dm:en");
        assert!(validate_redis_index(&redis_info(stop(), a), &fields).is_err());
    }

    /// #339 review: every index-level claim of `engine_settings("redis")` is
    /// verified on the live index, not only recorded.
    #[test]
    fn redis_index_language_options_and_scores_are_checked() {
        let fields = [fb("title", 1.0)];
        let info = || redis_info(Some(STOPWORDS.to_vec()), json!([text_attr("title")]));
        assert!(validate_redis_index(&info(), &fields).is_ok());
        // English stated explicitly is the default.
        let mut i = info();
        i["index_definition"]["default_language"] = json!("english");
        assert!(validate_redis_index(&i, &fields).is_ok());
        // String default_score (RESP2-converted) is read too.
        let mut i = info();
        i["index_definition"]["default_score"] = json!("1");
        assert!(validate_redis_index(&i, &fields).is_ok());
        for (key, value, needle) in [
            ("default_language", json!("german"), "LANGUAGE"),
            ("language_field", json!("lang"), "language_field"),
            ("score_field", json!("s"), "score_field"),
            ("payload_field", json!("pl"), "payload_field"),
            ("default_score", json!(0.5), "default_score"),
        ] {
            let mut i = info();
            i["index_definition"][key] = value;
            let e = validate_redis_index(&i, &fields).unwrap_err();
            assert!(e.contains(needle), "{key}: {e}");
        }
        let mut i = info();
        i["index_definition"] = Value::Null;
        assert!(validate_redis_index(&i, &fields).is_err());
        let mut i = info();
        i["index_definition"]
            .as_object_mut()
            .unwrap()
            .remove("default_score");
        assert!(validate_redis_index(&i, &fields).is_err());
        for opt in ["NOFREQS", "NOOFFSETS", "NOFIELDS"] {
            let mut i = info();
            i["index_options"] = json!(["NOHL", opt]);
            let e = validate_redis_index(&i, &fields).unwrap_err();
            assert!(e.contains(opt), "{e}");
        }
        // NOHL alone does not change matching or scoring.
        let mut i = info();
        i["index_options"] = json!(["NOHL"]);
        assert!(validate_redis_index(&i, &fields).is_ok());
        let mut i = info();
        i["index_options"] = json!("NOFREQS");
        assert!(validate_redis_index(&i, &fields).is_err());
    }

    /// The shape `GET _settings` returns: analysis as written, scalars strings.
    fn os_index() -> Value {
        json!({"number_of_shards":"1","analysis": opensearch_analysis()})
    }

    fn os_props() -> Value {
        json!({"title": opensearch_text_mapping(), "segment": opensearch_text_mapping(),
               "docid": {"type":"keyword"}})
    }

    #[test]
    fn opensearch_index_must_have_every_field_as_pinned() {
        let fields = [fb("title", 5.0), fb("segment", 1.0)];
        assert!(validate_opensearch_index(&os_props(), &os_index(), &fields, false).is_ok());
        // Missing field -> hard error.
        let mut p = os_props();
        p.as_object_mut().unwrap().remove("segment");
        let e = validate_opensearch_index(&p, &os_index(), &fields, false).unwrap_err();
        assert!(
            e.contains("'segment' is not in the OpenSearch mapping"),
            "{e}"
        );
        // Default (standard) analyzer.
        let mut p = os_props();
        p["title"] = json!({"type":"text"});
        assert!(validate_opensearch_index(&p, &os_index(), &fields, false).is_err());
        // Different search analyzer.
        let mut p = os_props();
        p["title"]["search_analyzer"] = json!("standard");
        assert!(validate_opensearch_index(&p, &os_index(), &fields, false).is_err());
        // keyword instead of text.
        let mut p = os_props();
        p["title"]["type"] = json!("keyword");
        assert!(validate_opensearch_index(&p, &os_index(), &fields, false).is_err());
        // Porter (english) instead of porter2.
        let mut i = os_index();
        i["analysis"]["filter"][OS_STEMMER]["language"] = json!("english");
        assert!(validate_opensearch_index(&os_props(), &i, &fields, false).is_err());
        // Filter order changed (stemming before stop).
        let mut i = os_index();
        i["analysis"]["analyzer"][OPENSEARCH_ANALYZER]["filter"] =
            json!([OS_POSSESSIVE, "lowercase", OS_STEMMER, OS_STOP]);
        assert!(validate_opensearch_index(&os_props(), &i, &fields, false).is_err());
        // `_english_` by name, or a different list.
        let mut i = os_index();
        i["analysis"]["filter"][OS_STOP]["stopwords"] = json!("_english_");
        assert!(validate_opensearch_index(&os_props(), &i, &fields, false).is_err());
        let mut i = os_index();
        i["analysis"]["filter"][OS_STOP]["stopwords"][0] = json!("hello");
        assert!(validate_opensearch_index(&os_props(), &i, &fields, false).is_err());
        let mut i = os_index();
        i["analysis"]["filter"][OS_STOP]["ignore_case"] = json!("true");
        assert!(validate_opensearch_index(&os_props(), &i, &fields, false).is_err());
        // No analysis at all (index built for a vector-only config).
        let i = json!({"number_of_shards":"1"});
        assert!(validate_opensearch_index(&os_props(), &i, &fields, false).is_err());
    }

    /// #339 review: char filters, non-BM25 similarities and multi-shard
    /// statistics are refused; the shard count is returned for the metadata.
    #[test]
    fn opensearch_index_char_filter_similarity_and_shards_are_checked() {
        let fields = [fb("title", 5.0), fb("segment", 1.0)];
        let ok = |p: &Value, i: &Value| validate_opensearch_index(p, i, &fields, false);
        assert_eq!(ok(&os_props(), &os_index()), Ok(1));
        // A char filter (e.g. mapping '\n' to a space).
        let mut i = os_index();
        i["analysis"]["analyzer"][OPENSEARCH_ANALYZER]["char_filter"] = json!(["nl"]);
        assert!(ok(&os_props(), &i).is_err());
        let mut i = os_index();
        i["analysis"]["analyzer"][OPENSEARCH_ANALYZER]["char_filter"] = json!([]);
        assert!(ok(&os_props(), &i).is_ok());
        // Per-field similarity.
        let mut p = os_props();
        p["segment"]["similarity"] = json!("boolean");
        assert!(ok(&p, &os_index()).unwrap_err().contains("similarity"));
        let mut p = os_props();
        p["segment"]["similarity"] = json!("BM25");
        assert!(ok(&p, &os_index()).is_ok());
        // Index-level similarity: only a parameterless default BM25.
        let mut i = os_index();
        i["similarity"] = json!({"default": {"type": "BM25"}});
        assert!(ok(&os_props(), &i).is_ok());
        for sim in [
            json!({"default": {"type": "boolean"}}),
            json!({"default": {"type": "BM25", "k1": "2.0"}}),
            json!({"default": {"type": "BM25"}, "custom": {"type": "DFR"}}),
            json!({"custom": {"type": "BM25"}}),
            json!("BM25"),
        ] {
            let mut i = os_index();
            i["similarity"] = sim.clone();
            assert!(ok(&os_props(), &i).is_err(), "{sim}");
        }
        // Shards: more than one refused unless opted in; unreadable refused.
        let mut i = os_index();
        i["number_of_shards"] = json!("3");
        assert!(ok(&os_props(), &i).unwrap_err().contains("3 shards"));
        assert_eq!(
            validate_opensearch_index(&os_props(), &i, &fields, true),
            Ok(3)
        );
        let mut i = os_index();
        i.as_object_mut().unwrap().remove("number_of_shards");
        assert!(ok(&os_props(), &i).is_err());
        // The opt-in is a spec key, off by default.
        assert!(!parse_spec(None).unwrap().allow_multi_shard);
        assert!(
            parse_spec(Some(&json!({"allow_multi_shard": true})))
                .unwrap()
                .allow_multi_shard
        );
    }

    /// Both modes must agree on validity and count; only `keep` returns hits.
    fn rp(v: &redis::Value, top: usize) -> Result<RankedHits, String> {
        both(
            redis_ranked_page(v, top, true),
            redis_ranked_page(v, top, false),
        )
    }

    fn op(v: &Value, top: usize) -> Result<RankedHits, String> {
        both(
            opensearch_ranked_page(v, top, true),
            opensearch_ranked_page(v, top, false),
        )
    }

    fn both(
        kept: Result<RankedPage, String>,
        bare: Result<RankedPage, String>,
    ) -> Result<RankedHits, String> {
        match (kept, bare) {
            (Ok(k), Ok(b)) => {
                assert!(b.hits.is_none(), "non-capturing mode built hits");
                assert_eq!(k.returned, b.returned);
                let hits = k.hits.expect("capturing mode keeps hits");
                assert_eq!(hits.len(), k.returned);
                Ok(hits)
            }
            (Err(k), Err(b)) => {
                assert_eq!(k, b);
                Err(k)
            }
            (k, b) => panic!("modes disagree: {k:?} vs {b:?}"),
        }
    }

    fn reply(total: i64, hits: &[(&str, f64)], warnings: Vec<redis::Value>) -> redis::Value {
        use redis::Value::{Array, BulkString, Double, Int, Map};
        let t = |s: &str| BulkString(s.as_bytes().to_vec());
        Map(vec![
            (t("total_results"), Int(total)),
            (t("warning"), Array(warnings)),
            (
                t("results"),
                Array(
                    hits.iter()
                        .map(|(id, s)| Map(vec![(t("id"), t(id)), (t("score"), Double(*s))]))
                        .collect(),
                ),
            ),
        ])
    }

    #[test]
    fn redis_reply_must_be_complete_ranked_and_warning_free() {
        let ok = rp(&reply(5, &[("k:1", 3.0), ("k:2", 1.5)], vec![]), 2).unwrap();
        assert_eq!(ok, vec![("k:1".into(), 3.0), ("k:2".into(), 1.5)]);
        // Fewer results than min(total, top).
        assert!(rp(&reply(5, &[("k:1", 3.0)], vec![]), 2).is_err());
        // Total smaller than top is a complete page.
        assert!(rp(&reply(1, &[("k:1", 3.0)], vec![]), 10).is_ok());
        // Warning (e.g. timeout partials).
        let w = vec![redis::Value::BulkString(
            b"Timeout limit was reached".to_vec(),
        )];
        assert!(rp(&reply(1, &[("k:1", 3.0)], w), 10).is_err());
        // Not ranked.
        assert!(rp(&reply(2, &[("k:1", 1.0), ("k:2", 2.0)], vec![]), 2).is_err());
        // Duplicate.
        assert!(rp(&reply(2, &[("k:1", 2.0), ("k:1", 1.0)], vec![]), 2).is_err());
        // RESP2 shape.
        assert!(rp(&redis::Value::Array(vec![]), 2).is_err());
        // No score.
        use redis::Value::{Array, BulkString, Int, Map};
        let t = |s: &str| BulkString(s.as_bytes().to_vec());
        let noscore = Map(vec![
            (t("total_results"), Int(1)),
            (t("warning"), Array(vec![])),
            (t("results"), Array(vec![Map(vec![(t("id"), t("k:1"))])])),
        ]);
        assert!(rp(&noscore, 2).is_err());
    }

    #[test]
    fn compound_exact_total_requires_a_complete_page() {
        let body = json!({"timed_out":false,"_shards":{"failed":0},"hits":{"total":{"value":2,"relation":"eq"},"hits":[{"_id":"a","_score":1.0}]}});
        assert!(opensearch_ranked_page(&body, 10, true)
            .unwrap_err()
            .contains("incomplete exact-total"));
        let mut complete = body.clone();
        complete["hits"]["total"]["value"] = json!(1);
        assert_eq!(
            opensearch_ranked_page(&complete, 10, true).unwrap().total,
            Some(1)
        );
        let mut capped = body;
        capped["hits"]["total"]["relation"] = json!("gte");
        capped["hits"]["total"]["value"] = json!(1);
        let page = opensearch_ranked_page(&capped, 10, true).unwrap();
        let query = TextQuery {
            query_id: 0,
            terms: vec![],
            expression: None,
            compound: Some(crate::compound_text::Query {
                name: "q".into(),
                family: "f".into(),
                conditions: json!({}),
                expected_count: Some(1),
                minimum_count: 0,
            }),
        };
        assert!(check_query_count(&query, page.total)
            .unwrap_err()
            .contains("exact total"));
    }

    #[test]
    fn opensearch_reply_must_be_complete_and_ranked() {
        let body =
            |hits: Value| json!({"timed_out":false,"_shards":{"failed":0},"hits":{"hits":hits}});
        let ok = op(
            &body(json!([{"_id":"a","_score":2.0},{"_id":"b","_score":2.0}])),
            2,
        )
        .unwrap();
        assert_eq!(ok, vec![("a".into(), 2.0), ("b".into(), 2.0)]);
        assert!(op(
            &body(json!([{"_id":"a","_score":1.0},{"_id":"b","_score":2.0}])),
            2
        )
        .is_err());
        assert!(op(
            &body(json!([{"_id":"a","_score":1.0},{"_id":"a","_score":1.0}])),
            2
        )
        .is_err());
        assert!(op(&body(json!([{"_id":"a"}])), 2).is_err());
        assert!(op(
            &body(json!([{"_id":"a","_score":1.0},{"_id":"b","_score":0.5}])),
            1
        )
        .is_err());
        let mut t = body(json!([]));
        t["timed_out"] = json!(true);
        assert!(op(&t, 2).is_err());
        let mut f = body(json!([]));
        f["_shards"]["failed"] = json!(1);
        assert!(op(&f, 2).is_err());
    }

    #[test]
    fn metadata_records_analyzers_boosts_and_skips() {
        let set = build_query_set(&["apple".into(), "the".into()]).unwrap();
        let fields = [fb("title", 5.0)];
        for engine in ["redis", "opensearch"] {
            let m = result_metadata(engine, &fields, &set, 10).unwrap();
            assert_eq!(m["fields"], json!({"title": 5.0}));
            assert_eq!(m["queries_skipped_no_terms"], json!([1]));
            assert_eq!(m["stopwords"]["list"], json!(STOPWORDS.as_slice()));
            assert_eq!(m["engine"], engine);
            assert_eq!(
                m["known_differences"].as_array().unwrap().len(),
                KNOWN_DIFFERENCES.len()
            );
        }
        assert_eq!(
            result_metadata("opensearch", &fields, &set, 10).unwrap()["engine_settings"]["index"]
                ["analysis"],
            opensearch_analysis()
        );
        assert_eq!(
            result_metadata("redis", &fields, &set, 10).unwrap()["engine_settings"]["scorer"],
            "BM25STD"
        );
        assert!(result_metadata("qdrant", &fields, &set, 10).is_err());
    }

    // ── total-hits contract ───────────────────────────────────────────────

    fn compound_block(total_hits: Option<&str>) -> Value {
        let mut b = json!({"fields":{"title":1.0},"compound":{"suite":"msmarco-v1"}});
        if let Some(t) = total_hits {
            b["total_hits"] = json!(t);
        }
        b
    }

    fn plan(engine: &str, block: &Value) -> Result<TotalHitsPlan, String> {
        total_hits_plan(engine, &parse_spec(Some(block))?)
    }

    #[test]
    fn total_hits_plan_is_symmetric_when_stated_and_legacy_otherwise() {
        let p = |mode, declared, symmetric| TotalHitsPlan {
            mode,
            declared,
            symmetric,
        };
        // Stated: the same mode on both engines.
        for (value, mode) in [("exact", TotalHits::Exact), ("none", TotalHits::None)] {
            let block = json!({"fields":{"segment":1.0},"total_hits":value});
            for engine in ["redis", "opensearch"] {
                assert_eq!(plan(engine, &block), Ok(p(mode, true, true)), "{engine}");
            }
        }
        // Not stated, ranked: legacy per-engine defaults, marked asymmetric.
        let ranked = json!({"fields":{"segment":1.0}});
        assert_eq!(
            plan("redis", &ranked),
            Ok(p(TotalHits::Exact, false, false))
        );
        assert_eq!(
            plan("opensearch", &ranked),
            Ok(p(TotalHits::None, false, false))
        );
        // Not stated, compound: exact on both, as before.
        for engine in ["redis", "opensearch"] {
            assert_eq!(
                plan(engine, &compound_block(None)),
                Ok(p(TotalHits::Exact, false, true))
            );
            assert_eq!(
                plan(engine, &compound_block(Some("exact"))),
                Ok(p(TotalHits::Exact, true, true))
            );
        }
        // An engine without a fulltext arm has no contract.
        assert!(plan("qdrant", &ranked).is_err());
    }

    #[test]
    fn total_hits_refuses_what_cannot_be_honoured() {
        // The compound count gate needs exact totals: refused at parse time,
        // i.e. at startup, before anything connects.
        let e = parse_spec(Some(&compound_block(Some("none")))).unwrap_err();
        assert!(e.contains("cannot be used with `fulltext.compound`"), "{e}");
        let mut spec = parse_spec(Some(&compound_block(None))).unwrap();
        spec.total_hits = Some(TotalHits::None);
        assert!(total_hits_plan("redis", &spec).is_err());
        // Unknown values are refused, not defaulted.
        for bad in [json!("all"), json!(true), json!("EXACT")] {
            assert!(
                parse_spec(Some(&json!({ "total_hits": bad }))).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn redis_command_adds_withoutcount_only_for_none() {
        for analysis in [Analysis::English, Analysis::Parity] {
            let exact = redis_args_with("idx", "q", 7, 500, analysis, TotalHits::Exact);
            // `exact` is byte-for-byte the legacy command.
            assert_eq!(exact, redis_args_for("idx", "q", 7, 500, analysis));
            assert!(!exact.iter().any(|a| a == REDIS_WITHOUTCOUNT));
            let none = redis_args_with("idx", "q", 7, 500, analysis, TotalHits::None);
            assert_eq!(none.last().map(String::as_str), Some("WITHOUTCOUNT"));
            assert_eq!(none.iter().filter(|a| *a == "WITHOUTCOUNT").count(), 1);
            assert_eq!(&none[..none.len() - 1], &exact[..]);
        }
    }

    #[test]
    fn opensearch_request_tracks_total_hits_by_mode() {
        let fields = [fb("segment", 1.0)];
        let terms = vec!["apple".to_string()];
        assert_eq!(
            opensearch_body_with(&fields, &terms, 5, TotalHits::Exact)["track_total_hits"],
            json!(true)
        );
        let none = opensearch_body_with(&fields, &terms, 5, TotalHits::None);
        assert_eq!(none["track_total_hits"], json!(false));
        // The legacy builder is `none`, unchanged.
        assert_eq!(none, opensearch_body(&fields, &terms, 5));
        let ranked = TextQuery {
            query_id: 0,
            terms: terms.clone(),
            expression: None,
            compound: None,
        };
        for mode in [TotalHits::Exact, TotalHits::None] {
            assert_eq!(
                query_opensearch_with(&fields, &ranked, 5, mode),
                opensearch_body_with(&fields, &terms, 5, mode)
            );
        }
        let compound = TextQuery {
            expression: Some(
                crate::compound_text::Expr::compile(
                    &json!({"title":{"match":{"text":"apple"}}}),
                    &json!({"title":"text"}),
                )
                .unwrap(),
            ),
            ..ranked
        };
        assert_eq!(
            query_opensearch_with(&fields, &compound, 5, TotalHits::Exact)["track_total_hits"],
            json!(true)
        );
    }

    #[test]
    fn replies_must_show_the_requested_total_contract() {
        // Redis `none`: total_results is clamped, never published as a total,
        // but the page is still checked against it.
        let r = reply(2, &[("k:1", 3.0), ("k:2", 1.5)], vec![]);
        let total = |v: &redis::Value, m| redis_ranked_page_with(v, 2, false, m).map(|p| p.total);
        assert_eq!(total(&r, TotalHits::Exact), Ok(Some(2)));
        assert_eq!(total(&r, TotalHits::None), Ok(None));
        let short = reply(2, &[("k:1", 3.0)], vec![]);
        assert!(total(&short, TotalHits::None).is_err());
        // OpenSearch: the reply must match what was asked for.
        let hits = json!([{"_id":"a","_score":1.0}]);
        let exact = json!({"timed_out":false,"_shards":{"failed":0},
            "hits":{"total":{"value":1,"relation":"eq"},"hits":hits}});
        let none = json!({"timed_out":false,"_shards":{"failed":0},"hits":{"hits":hits}});
        let mut gte = exact.clone();
        gte["hits"]["total"]["relation"] = json!("gte");
        let page = |b: &Value, m| opensearch_ranked_page_with(b, 10, false, m);
        assert_eq!(page(&exact, TotalHits::Exact).unwrap().total, Some(1));
        assert_eq!(page(&none, TotalHits::None).unwrap().total, None);
        assert!(page(&none, TotalHits::Exact)
            .unwrap_err()
            .contains("exact total requested"));
        assert!(page(&gte, TotalHits::Exact).is_err());
        assert!(page(&exact, TotalHits::None)
            .unwrap_err()
            .contains("no total requested"));
    }

    /// A fake server for [`probe_withoutcount`]: each query's matches as
    /// `(id, score)` in document-id order. Counted replies are BM25-ranked;
    /// `WITHOUTCOUNT` replies are ranked (`keeps_ranking`, RediSearch 8.4+)
    /// or the first `top` matches in document-id order (8.0/8.2).
    fn probe_server(
        matches: &[Vec<(&'static str, f64)>],
        top: usize,
        keeps_ranking: bool,
    ) -> Result<WithoutCountProbe, String> {
        probe_withoutcount(matches.len(), top, |i, mode| {
            let mut page = matches[i].clone();
            if mode == TotalHits::Exact || keeps_ranking {
                page.sort_by(|a, b| b.1.total_cmp(&a.1));
            }
            page.truncate(top);
            let total = match mode {
                TotalHits::Exact => matches[i].len(),
                TotalHits::None => page.len(),
            };
            Ok(reply(total as i64, &page, vec![]))
        })
    }

    #[test]
    fn withoutcount_probe_refuses_a_server_that_drops_the_ranking() {
        let matches = vec![
            vec![("k:1", 0.5), ("k:2", 2.0), ("k:3", 1.0)],
            vec![("k:4", 3.0), ("k:5", 1.0)],
            vec![],
        ];
        // 8.4+: same pages; two queries have more matches than the page.
        assert_eq!(
            probe_server(&matches, 1, true),
            Ok(WithoutCountProbe {
                compared: 2,
                over_limit: 2
            })
        );
        // 8.0/8.2: first match by doc id. Query 0's page is (k:1, 0.5), not
        // (k:2, 2.0): it is score-ordered (one hit) but not the top-1.
        let e = probe_server(&matches, 1, false).unwrap_err();
        assert!(e.contains("query #0") && e.contains("differs"), "{e}");
        // With a two-hit page the doc-id order is also out of score order.
        let e = probe_server(&matches, 2, false).unwrap_err();
        assert!(e.contains(RANK_VIOLATION), "{e}");
        // Query 1's doc-id order happens to be ranked; only query 0 shows it.
        let e = probe_server(&matches[1..], 1, false);
        assert!(e.is_ok(), "{e:?}");
        // No query with more matches than the page: nothing can be shown.
        let e = probe_server(&matches, 3, true).unwrap_err();
        assert!(e.contains("has more than 3, so the probe cannot"), "{e}");
        assert!(!e.contains("RediSearch before 8.4"), "{e}");
    }

    #[test]
    fn withoutcount_probe_stops_after_enough_truncated_queries() {
        let matches = vec![vec![("k:1", 2.0), ("k:2", 1.0)]; WITHOUTCOUNT_PROBE_QUERIES + 5];
        let mut sent = 0;
        let probe = probe_withoutcount(matches.len(), 1, |i, mode| {
            sent += 1;
            let mut page = matches[i].clone();
            page.truncate(1);
            let total = if mode == TotalHits::Exact { 2 } else { 1 };
            Ok(reply(total, &page, vec![]))
        })
        .unwrap();
        assert_eq!(probe.over_limit, WITHOUTCOUNT_PROBE_QUERIES);
        assert_eq!(sent, 2 * WITHOUTCOUNT_PROBE_QUERIES);
    }

    #[test]
    fn withoutcount_page_comparison_allows_only_boundary_ties() {
        let p = |h: &[(&str, f64)]| {
            h.iter()
                .map(|(i, s)| (i.to_string(), *s))
                .collect::<Vec<_>>()
        };
        let counted = p(&[("a", 3.0), ("b", 2.0), ("c", 2.0), ("d", 1.0)]);
        // Reordered inside a tie: same page.
        let swapped = p(&[("a", 3.0), ("c", 2.0), ("b", 2.0), ("d", 1.0)]);
        assert!(same_ranked_page(&counted, &swapped, false).is_ok());
        // Another document of the boundary score: only when truncated.
        let boundary = p(&[("a", 3.0), ("b", 2.0), ("c", 2.0), ("e", 1.0)]);
        assert!(same_ranked_page(&counted, &boundary, true).is_ok());
        assert!(same_ranked_page(&counted, &boundary, false).is_err());
        // Another document inside the page is never the same page.
        let inner = p(&[("a", 3.0), ("b", 2.0), ("e", 2.0), ("d", 1.0)]);
        assert!(same_ranked_page(&counted, &inner, true).is_err());
        // Different scores or length.
        let scored = p(&[("a", 3.0), ("b", 2.0), ("c", 2.0), ("d", 0.5)]);
        assert!(same_ranked_page(&counted, &scored, true).is_err());
        assert!(same_ranked_page(&counted, &counted[..3], true).is_err());
    }

    #[test]
    fn none_fails_closed_on_any_out_of_order_reply() {
        let e = rp(&reply(2, &[("k:1", 1.0), ("k:2", 2.0)], vec![]), 2).unwrap_err();
        assert!(is_rank_violation(&e), "{e}");
        let e = rp(&reply(1, &[("k:1", f64::NAN)], vec![]), 2).unwrap_err();
        assert!(is_rank_violation(&e), "{e}");
        assert!(!is_rank_violation("fulltext: incomplete Redis page"));
        assert!(check_rank_violations(TotalHits::None, 1)
            .unwrap_err()
            .contains("WITHOUTCOUNT"));
        assert!(check_rank_violations(TotalHits::None, 0).is_ok());
        assert!(check_rank_violations(TotalHits::Exact, 3).is_ok());
    }

    #[test]
    fn result_records_the_total_hits_mode_sent() {
        let fields = [fb("segment", 1.0)];
        let set = build_query_set(&["apple".to_string()]).unwrap();
        let meta = |engine: &str, block: &Value| {
            let mut m = result_metadata(engine, &fields, &set, 10).unwrap();
            record_total_hits(&mut m, engine, &plan(engine, block).unwrap()).unwrap();
            m
        };
        let none = json!({"fields":{"segment":1.0},"total_hits":"none"});
        let exact = json!({"fields":{"segment":1.0},"total_hits":"exact"});
        let legacy = json!({"fields":{"segment":1.0}});

        let os = meta("opensearch", &exact);
        assert_eq!(os["engine_settings"]["request"]["track_total_hits"], true);
        assert_eq!(os["total_hits"]["mode"], "exact");
        assert_eq!(os["total_hits"]["declared"], true);
        assert_eq!(os["total_hits"]["sent"], "track_total_hits: true");
        assert_eq!(os["total_hits_symmetric"], true);
        assert!(os.get("warnings").is_none());
        let os = meta("opensearch", &none);
        assert_eq!(os["engine_settings"]["request"]["track_total_hits"], false);
        assert_eq!(os["total_hits"]["mode"], "none");

        let redis = meta("redis", &none);
        assert!(redis["engine_settings"]["command"]
            .as_str()
            .unwrap()
            .ends_with(" WITHOUTCOUNT"));
        assert_eq!(redis["total_hits"]["mode"], "none");
        assert_eq!(redis["total_hits_symmetric"], true);
        let redis = meta("redis", &exact);
        assert!(!redis["engine_settings"]["command"]
            .as_str()
            .unwrap()
            .contains("WITHOUTCOUNT"));

        // Not stated: each engine records its legacy mode, the pair is marked
        // asymmetric, and the result carries the warning.
        for (engine, mode) in [("redis", "exact"), ("opensearch", "none")] {
            let m = meta(engine, &legacy);
            assert_eq!(m["total_hits"]["mode"], mode, "{engine}");
            assert_eq!(m["total_hits"]["declared"], false);
            assert_eq!(m["total_hits_symmetric"], false);
            assert_eq!(m["warnings"], json!([ASYMMETRIC_TOTAL_HITS_WARNING]));
        }
        assert!(
            record_total_hits(&mut json!({}), "qdrant", &plan("redis", &legacy).unwrap()).is_err()
        );
    }

    /// The configs of the Redis vs OpenSearch comparison state the mode, so
    /// their pairs are symmetric: ranked BM25 `none` (OpenSearch's measured
    /// behaviour kept; Redis gets WITHOUTCOUNT), compound `exact` (its count
    /// gate needs it, and it is the behaviour it always had).
    #[test]
    fn shipped_msmarco_text_configs_state_a_symmetric_total_hits_mode() {
        for (file, want) in [
            ("redis-msmarco-fulltext.json", TotalHits::None),
            ("opensearch-msmarco-fulltext.json", TotalHits::None),
            ("msmarco-compound.json", TotalHits::Exact),
        ] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("experiments/configurations")
                .join(file);
            let configs: Vec<Value> =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let mut n = 0;
            for c in &configs {
                let engine = c["engine"].as_str().unwrap();
                for sp in c["search_params"].as_array().unwrap() {
                    let got = plan(engine, &sp[SPEC_KEY]).unwrap();
                    assert_eq!(
                        (got.mode, got.declared, got.symmetric),
                        (want, true, true),
                        "{file} [{}]",
                        c["name"]
                    );
                    n += 1;
                }
            }
            assert!(n > 0, "{file}");
        }
    }

    // ---- msmarco-disk-v1 and fetch -------------------------------------

    fn disk_schema() -> Value {
        json!({"docid":"keyword","url":"keyword","title":"text","segment":"text",
               "category":"keyword","doc_len":"int"})
    }

    fn disk_block(family: &str, extra: Value) -> Value {
        let mut b = json!({"analysis":"parity","fields":{"title":1.0,"segment":1.0},
            "no_vector_index":true,"scalar_indexing":"text-tag-only",
            "compound":{"suite":"msmarco-disk-v1","family":family}});
        for (k, v) in extra.as_object().unwrap() {
            if k == "offset" || k == "duration_s" {
                b["compound"][k] = v.clone();
            } else {
                b[k] = v.clone();
            }
        }
        b
    }

    fn disk_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
        use crate::msmarco::disk_queries as dq;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("corpus");
        dq::write_fixture_corpus(&dir, 3000, 4).unwrap();
        dq::generate(
            &dir,
            &dq::GenerateOptions {
                per_family: 100,
                seed: 1,
                min_per_family: 5,
            },
            &dir.join(dq::DIR),
        )
        .unwrap();
        (tmp, dir)
    }

    #[test]
    fn disk_suite_and_fetch_spec_rules() {
        assert!(parse_spec(Some(&disk_block("rare-term", json!({})))).is_ok());
        for (block, why) in [
            (
                disk_block(
                    "rare-term",
                    json!({"analysis":"english",
                "scalar_indexing":"all"}),
                ),
                "parity",
            ),
            (disk_block("bogus", json!({})), "unknown"),
            (
                json!({"analysis":"parity","fields":{"segment":1.0},
                "compound":{"suite":"msmarco-disk-v1"}}),
                "family",
            ),
            (
                json!({"analysis":"parity","fields":{"segment":1.0},
                "compound":{"suite":"msmarco-v1","offset":3}}),
                "offset",
            ),
            (
                json!({"analysis":"parity","fields":{"segment":1.0},"total_hits":"none",
                "compound":{"suite":"msmarco-v1"}}),
                "none",
            ),
            (disk_block("rare-term", json!({"fetch":[]})), "at least one"),
            (
                disk_block("rare-term", json!({"fetch":["title","title"]})),
                "twice",
            ),
            (
                disk_block("rare-term", json!({"fetch":["vector"]})),
                "vector",
            ),
            (
                disk_block("rare-term", json!({"fetch":["a-b"]})),
                "identifier",
            ),
        ] {
            let e = parse_spec(Some(&block)).unwrap_err();
            assert!(e.contains(why), "{block}: {e}");
        }
        // The disk suite may run without totals: every query has an exact count.
        let none = parse_spec(Some(&disk_block("docid-eq", json!({"total_hits":"none"})))).unwrap();
        assert_eq!(
            total_hits_plan("redis", &none).unwrap().mode,
            TotalHits::None
        );
        let schema = disk_schema();
        let fetch = |f: Value| {
            resolve_fetch(
                &parse_spec(Some(&disk_block("rare-term", json!({"fetch": f})))).unwrap(),
                Some(&schema),
            )
        };
        assert_eq!(
            fetch(json!(["title", "segment", "url"])).unwrap().unwrap(),
            ["title", "segment", "url"]
        );
        assert!(fetch(json!(["doc_len"])).unwrap_err().contains("text"));
        assert!(fetch(json!(["missing"])).is_err());
        assert_eq!(
            resolve_fetch(&FulltextSpec::default(), Some(&schema)).unwrap(),
            None
        );
        let rrf = parse_spec(Some(&json!({"fetch":["title"]}))).unwrap();
        assert!(crate::hybrid_rrf::check_text_spec(&rrf)
            .unwrap_err()
            .contains("fetch"));
    }

    #[test]
    fn fetch_requests_replace_ids_only_and_leave_it_unchanged_otherwise() {
        let f: Vec<String> = ["title", "segment", "url"].map(String::from).to_vec();
        for analysis in [Analysis::English, Analysis::Parity] {
            for mode in [TotalHits::Exact, TotalHits::None] {
                assert_eq!(
                    redis_args_fetch("i", "q", 10, 5, analysis, mode, None),
                    redis_args_with("i", "q", 10, 5, analysis, mode)
                );
            }
        }
        assert_eq!(
            redis_args_fetch("i", "q", 10, 5, Analysis::Parity, TotalHits::None, Some(&f)),
            [
                "i",
                "q",
                "RETURN",
                "3",
                "title",
                "segment",
                "url",
                "VERBATIM",
                "WITHSCORES",
                "SCORER",
                "BM25STD",
                "LIMIT",
                "0",
                "10",
                "DIALECT",
                "2",
                "TIMEOUT",
                "5",
                "WITHOUTCOUNT"
            ]
        );
        assert!(!redis_args_fetch(
            "i",
            "q",
            10,
            5,
            Analysis::English,
            TotalHits::Exact,
            Some(&f)
        )
        .contains(&"NOCONTENT".to_string()));
        let q = TextQuery {
            query_id: 0,
            terms: vec!["x".into()],
            expression: None,
            compound: None,
        };
        let fields = [FieldBoost {
            field: "title".into(),
            boost: 1.0,
        }];
        for mode in [TotalHits::Exact, TotalHits::None] {
            assert_eq!(
                query_opensearch_fetch(&fields, &q, 10, mode, None),
                query_opensearch_with(&fields, &q, 10, mode)
            );
            assert_eq!(
                query_opensearch_fetch(&fields, &q, 10, mode, Some(&f))["_source"],
                json!({"includes":["title","segment","url"]})
            );
        }
    }

    #[test]
    fn fetched_fields_are_checked_on_every_hit() {
        use redis::Value as V;
        let f: Vec<String> = ["title", "url"].map(String::from).to_vec();
        let s = |x: &str| V::BulkString(x.as_bytes().to_vec());
        let hit = |id: &str, attrs: Option<Vec<(&str, V)>>| {
            let mut m = vec![(s("id"), s(id)), (s("score"), V::Double(1.0))];
            if let Some(a) = attrs {
                m.push((
                    s("extra_attributes"),
                    V::Map(a.into_iter().map(|(k, v)| (s(k), v)).collect()),
                ));
            }
            V::Map(m)
        };
        let reply = |hits: Vec<V>| {
            V::Map(vec![
                (s("warning"), V::Array(vec![])),
                (s("total_results"), V::Int(hits.len() as i64)),
                (s("results"), V::Array(hits)),
            ])
        };
        let ok = reply(vec![hit(
            "k:1",
            Some(vec![("title", s("ab")), ("url", s("http://c"))]),
        )]);
        let page = redis_ranked_page_fetch(&ok, 10, false, TotalHits::Exact, Some(&f)).unwrap();
        assert_eq!(page.fetched_bytes, 10);
        for (bad, why) in [
            (
                reply(vec![hit("k:1", Some(vec![("title", s("ab"))]))]),
                "missing",
            ),
            (
                reply(vec![hit(
                    "k:1",
                    Some(vec![("title", s("a")), ("url", s("u")), ("vector", s("v"))]),
                )]),
                "did not ask",
            ),
            (
                reply(vec![hit(
                    "k:1",
                    Some(vec![("title", V::Int(1)), ("url", s("u"))]),
                )]),
                "non-string",
            ),
            (reply(vec![hit("k:1", None)]), "missing"),
        ] {
            let e =
                redis_ranked_page_fetch(&bad, 10, false, TotalHits::Exact, Some(&f)).unwrap_err();
            assert!(e.contains(why), "{e}");
        }
        // Ids only: no document fields at all.
        assert!(redis_ranked_page_with(&ok, 10, false, TotalHits::Exact)
            .unwrap_err()
            .contains("ids-only"));
        let ids = reply(vec![hit("k:1", None)]);
        assert_eq!(
            redis_ranked_page_with(&ids, 10, false, TotalHits::Exact)
                .unwrap()
                .fetched_bytes,
            0
        );
        let os = |source: Option<Value>| {
            let mut h = json!({"_id":"1","_score":1.0});
            if let Some(src) = source {
                h["_source"] = src;
            }
            json!({"timed_out":false,"_shards":{"failed":0},
                "hits":{"total":{"value":1,"relation":"eq"},"hits":[h]}})
        };
        let good = os(Some(json!({"title":"ab","url":"http://c"})));
        assert_eq!(
            opensearch_ranked_page_fetch(&good, 10, false, TotalHits::Exact, Some(&f))
                .unwrap()
                .fetched_bytes,
            10
        );
        for bad in [
            os(Some(json!({"title":"ab"}))),
            os(Some(json!({"title":"ab","url":"u","segment":"s"}))),
            os(Some(json!({"title":"ab","url":3}))),
            os(None),
        ] {
            assert!(
                opensearch_ranked_page_fetch(&bad, 10, false, TotalHits::Exact, Some(&f)).is_err()
            );
        }
        assert!(
            opensearch_ranked_page_with(&good, 10, false, TotalHits::Exact)
                .unwrap_err()
                .contains("ids-only")
        );
        assert!(opensearch_ranked_page_with(&os(None), 10, false, TotalHits::Exact).is_ok());
        // Even an empty `_source` means the request asked for a source.
        assert!(
            opensearch_ranked_page_with(&os(Some(json!({}))), 10, false, TotalHits::Exact)
                .unwrap_err()
                .contains("ids-only")
        );
    }

    #[test]
    fn page_count_check_without_a_total_uses_the_exact_count() {
        let q = |expected: Option<u64>| TextQuery {
            query_id: 0,
            terms: vec![],
            expression: None,
            compound: Some(crate::compound_text::Query {
                name: "q".into(),
                family: "rare-term".into(),
                conditions: json!({}),
                expected_count: expected,
                minimum_count: 1,
            }),
        };
        assert!(check_query_page(&q(Some(3)), None, 3, 10).is_ok());
        assert!(check_query_page(&q(Some(3)), None, 2, 10).is_err());
        assert!(check_query_page(&q(Some(30)), None, 10, 10).is_ok());
        assert!(check_query_page(&q(Some(30)), None, 9, 10).is_err());
        assert!(check_query_page(&q(None), None, 3, 10).is_err());
        assert!(check_query_page(&q(Some(3)), Some(3), 3, 10).is_ok());
        assert!(check_query_page(&q(Some(3)), Some(4), 3, 10).is_err());
        // Ranked BM25 has no count contract.
        let ranked = TextQuery {
            query_id: 0,
            terms: vec!["x".into()],
            expression: None,
            compound: None,
        };
        assert!(check_query_page(&ranked, None, 0, 10).is_ok());
    }

    #[test]
    fn disk_windows_dispatch_once_probe_off_the_window_and_record_identity() {
        use crate::msmarco::disk_queries as dq;
        let (_tmp, dir) = disk_fixture();
        let schema = disk_schema();
        let load = |family: &str, extra: Value, top: usize, n: i64| {
            load_query_set_for(
                &dir,
                &parse_spec(Some(&disk_block(family, extra))).unwrap(),
                Some(&schema),
                top,
                n,
            )
        };
        let set = load("rare-term", json!({"offset":3}), 10, 7).unwrap();
        let w = set.window.as_ref().unwrap();
        assert_eq!(set.queries.len(), dq::RESERVED + 7);
        assert_eq!(set.prime_index(), 0);
        assert_eq!(set.queries[0].query_id, 0);
        let order = set.dispatch_order(7).unwrap();
        assert_eq!(order, (dq::RESERVED..dq::RESERVED + 7).collect::<Vec<_>>());
        let ids: Vec<usize> = order.iter().map(|&i| set.queries[i].query_id).collect();
        assert_eq!(
            ids,
            (dq::RESERVED + 3..dq::RESERVED + 10).collect::<Vec<_>>()
        );
        // Never more, never cycling.
        assert!(set.dispatch_order(8).is_err());
        assert!(set.dispatch_order(-1).is_ok());
        // The probe never touches a timed query. (The whole pool: the reserve
        // took the highest counts, so a short window may need no probe.)
        let whole = load("two-rare-terms", json!({}), 1, -1).unwrap();
        match whole.withoutcount_probe_plan(1) {
            ProbePlan::Run(p) => assert!(p.iter().all(|&i| i > 0 && i < w.timed_from)),
            other => panic!("{other:?}"),
        }
        let docids = load("docid-eq", json!({}), 10, 5).unwrap();
        assert_eq!(
            docids.withoutcount_probe_plan(10),
            ProbePlan::NotNeeded(PROBE_NOT_NEEDED)
        );
        // Legacy query sets keep the legacy schedule.
        let legacy = build_query_set(&["a b".to_string(), "c".to_string()]).unwrap();
        assert_eq!(legacy.dispatch_order(5).unwrap(), vec![0, 1, 0, 1, 0]);
        // Result identity: no query list, one family, the window and a
        // served-queries hash that changes with the window.
        let profile = index_profile([(
            Workload::Fulltext,
            Some(&disk_block("rare-term", json!({}))),
        )])
        .unwrap()
        .unwrap();
        let fields = resolve_fields(
            &parse_spec(Some(&disk_block("rare-term", json!({})))).unwrap(),
            Some(&schema),
        )
        .unwrap();
        let meta = |set: &QuerySet| {
            result_metadata_for("redis", &fields, set, 10, &profile, &profile.marker()).unwrap()
        };
        let m = meta(&set);
        assert!(m["compound"].get("queries").is_none());
        assert_eq!(m["compound"]["family"], "rare-term");
        assert_eq!(m["compound"]["window"]["offset"], 3);
        assert_eq!(m["compound"]["window"]["first_line"], dq::RESERVED + 3);
        assert_eq!(
            m["compound"]["preparation_sha256"],
            json!(dq::preparation_sha256(&dir).unwrap())
        );
        let id = crate::results_dump::CompoundIdentity::from_metadata(Some(&m))
            .unwrap()
            .unwrap();
        let shifted = meta(&load("rare-term", json!({"offset":4}), 10, 7).unwrap());
        assert_ne!(shifted["compound"]["sha256"], m["compound"]["sha256"]);
        assert_ne!(
            shifted["compound"]["served_sha256"],
            m["compound"]["served_sha256"]
        );
        assert_eq!(
            meta(&load("rare-term", json!({"offset":3}), 10, 7).unwrap())["compound"]["sha256"],
            json!(id.query_sha256)
        );
        // deep-fetch is the top-100 fetch, nothing else.
        assert!(load("deep-fetch", json!({}), 10, 5)
            .unwrap_err()
            .contains("deep-fetch"));
        assert!(load("deep-fetch", json!({}), 100, 5)
            .unwrap_err()
            .contains("fetch"));
        assert!(load(
            "deep-fetch",
            json!({"fetch":["title","segment","url"]}),
            100,
            5
        )
        .is_ok());
        // docid-batch is the fetch of its whole batch, nothing else; its
        // page is the whole match set, so `none` needs no probe.
        let fetch = json!({"fetch":["title","segment","url"]});
        assert!(load("docid-batch", fetch.clone(), 10, 5)
            .unwrap_err()
            .contains("docid-batch"));
        assert!(load("docid-batch", json!({}), 100, 5)
            .unwrap_err()
            .contains("fetch"));
        let none = json!({"fetch":["title","segment","url"],"total_hits":"none"});
        let e = load("docid-batch", none, 100, 5).unwrap_err();
        assert!(e.contains("exact") && e.contains("WITHOUTCOUNT"), "{e}");
        let batch = load("docid-batch", fetch, 100, 5).unwrap();
        assert!(
            batch
                .queries
                .iter()
                .all(|q| q.compound.as_ref().unwrap().expected_count
                    == Some(dq::DOCID_BATCH_K as u64))
        );
        assert_eq!(
            batch.withoutcount_probe_plan(100),
            ProbePlan::NotNeeded(PROBE_NOT_NEEDED)
        );
        let q = &batch.queries[batch.window.as_ref().unwrap().timed_from];
        let redis = query_redis(&[], q);
        assert!(
            redis.starts_with("(@docid:{") && redis.matches(" | ").count() == 99,
            "{redis}"
        );
        let body = query_opensearch_fetch(
            &[],
            q,
            100,
            TotalHits::Exact,
            Some(&[
                "title".to_string(),
                "segment".to_string(),
                "url".to_string(),
            ]),
        );
        assert_eq!(body["size"], 100);
        assert_eq!(
            body["query"]["bool"]["filter"][0]["terms"]["docid"]
                .as_array()
                .unwrap()
                .len(),
            100
        );
        // The legacy loader refuses the per-window suite.
        assert!(load_query_set(
            &dir,
            &parse_spec(Some(&disk_block("rare-term", json!({})))).unwrap(),
            Some(&schema)
        )
        .is_err());
    }

    #[test]
    fn disk_none_refuses_before_connecting_when_the_reserve_cannot_prove_the_probe() {
        use crate::msmarco::disk_queries as dq;
        let (_tmp, dir) = disk_fixture();
        let schema = disk_schema();
        // The generator's reserve covers every top below the timed maximum,
        // for every family, so `none` loads and the probe has truncated
        // pages to compare.
        // (docid-batch refuses `none`; its page is its whole match set.)
        for family in dq::Family::ALL
            .into_iter()
            .filter(|f| *f != dq::Family::DocidBatch)
        {
            let fixed = family.fixed_top();
            let mut extra = json!({"total_hits":"none"});
            if fixed.is_some() {
                extra["fetch"] = json!(["title", "segment", "url"]);
            }
            let spec = parse_spec(Some(&disk_block(family.name(), extra))).unwrap();
            let tops: Vec<usize> = match fixed {
                Some(top) => vec![top],
                None => vec![1, 2, 5, 10],
            };
            for top in tops {
                let set = load_query_set_for(&dir, &spec, Some(&schema), top, 50).unwrap();
                let w = set.window.as_ref().unwrap();
                let count = |q: &TextQuery| q.compound.as_ref().unwrap().expected_count.unwrap();
                if set.queries[w.timed_from..]
                    .iter()
                    .any(|q| count(q) > top as u64)
                {
                    assert!(
                        set.queries[1..w.timed_from]
                            .iter()
                            .any(|q| count(q) > top as u64),
                        "{family:?} top {top}"
                    );
                }
            }
        }
        // A window whose timed queries are cut by the page but whose reserve
        // is not (a uniform reserve over a typically-1 family): refused with
        // the family, top and seed, instead of after deployment by the probe.
        let spec = parse_spec(Some(&disk_block("url-eq", json!({"total_hits":"none"})))).unwrap();
        let mut set = load_query_set_for(&dir, &spec, Some(&schema), 1, 50).unwrap();
        let timed_from = set.window.as_ref().unwrap().timed_from;
        for (i, q) in set.queries.iter_mut().enumerate() {
            let c = q.compound.as_mut().unwrap();
            c.expected_count = Some(if i == timed_from + 3 { 37 } else { 1 });
        }
        let e = set.check_probe_reserve(10).unwrap_err();
        assert!(
            e.contains("url-eq") && e.contains("seed 1") && e.contains("`top` 10"),
            "{e}"
        );
        assert!(!e.contains("RediSearch before 8.4"), "{e}");
        // A reserve line over `top` is enough; and no timed query over
        // `top` needs no probe at all.
        set.queries[2].compound.as_mut().unwrap().expected_count = Some(11);
        assert!(set.check_probe_reserve(10).is_ok());
        set.queries[2].compound.as_mut().unwrap().expected_count = Some(1);
        assert!(set.check_probe_reserve(37).is_ok());
        // Legacy sets have no reserve to check.
        let legacy = build_query_set(&["a b".to_string()]).unwrap();
        assert!(legacy.check_probe_reserve(1).is_ok());
    }

    /// The single-pass identity of a fixed window of the deterministic disk
    /// fixture, pinned: it is the value master computed before duration mode
    /// existed, so a run without `duration_s` keeps its identity.
    const SINGLE_PASS_IDENTITY: &str =
        "5a7ddc4c5feff1bd8554c8f2c08ffe8546cd73e35a5a6e7717e2cf53aa5f6df7";

    #[test]
    fn disk_duration_mode_cycles_the_window_records_it_and_keeps_single_pass_identity() {
        use crate::duration_mode::Dispatcher;
        use crate::msmarco::disk_queries as dq;
        let (_tmp, dir) = disk_fixture();
        let schema = disk_schema();
        let load = |extra: Value| {
            load_query_set_for(
                &dir,
                &parse_spec(Some(&disk_block("rare-term", extra))).unwrap(),
                Some(&schema),
                10,
                7,
            )
            .unwrap()
        };
        // Spec rules: disk suite only, a finite positive duration.
        for bad in [json!(0), json!(-1), json!(1e12)] {
            let e =
                parse_spec(Some(&disk_block("rare-term", json!({"duration_s": bad})))).unwrap_err();
            assert!(e.contains("duration_s"), "{e}");
        }
        let e = parse_spec(Some(&json!({"analysis":"parity","fields":{"segment":1.0},
            "compound":{"suite":"msmarco-v1","duration_s":600}})))
        .unwrap_err();
        assert!(e.contains("duration_s"), "{e}");
        assert!(check_execution_params(None, Some(600.0))
            .unwrap_err()
            .contains("compound.duration_s"));

        let once = load(json!({"offset":3}));
        let timed = load(json!({"offset":3,"duration_s":600}));
        assert_eq!(once.deadline(), None);
        assert_eq!(timed.deadline(), Some(std::time::Duration::from_secs(600)));
        // The same window, loaded once: duration mode loads nothing more.
        assert_eq!(timed.queries, once.queries);
        let order = timed.dispatch_order(7).unwrap();
        assert_eq!(order, once.dispatch_order(7).unwrap());
        // Dispatch k sends timed line first_line + (k mod 7).
        let d = Dispatcher::new(&order, timed.deadline());
        let start = std::time::Instant::now();
        let lines: Vec<usize> = (0..17)
            .map(|_| timed.queries[d.next(start).unwrap().1].query_id)
            .collect();
        let want: Vec<usize> = (0..17).map(|k| dq::RESERVED + 3 + k % 7).collect();
        assert_eq!(lines, want);

        let profile = index_profile([(
            Workload::Fulltext,
            Some(&disk_block("rare-term", json!({}))),
        )])
        .unwrap()
        .unwrap();
        let fields = resolve_fields(
            &parse_spec(Some(&disk_block("rare-term", json!({})))).unwrap(),
            Some(&schema),
        )
        .unwrap();
        let meta = |set: &QuerySet| {
            result_metadata_for("redis", &fields, set, 10, &profile, &profile.marker()).unwrap()
        };
        let (mut m_once, mut m_timed) = (meta(&once), meta(&timed));
        assert_eq!(m_once["compound"]["sha256"], SINGLE_PASS_IDENTITY);
        assert!(m_once["compound"].get("duration_s").is_none());
        assert_eq!(m_timed["compound"]["duration_s"], 600.0);
        assert_ne!(m_timed["compound"]["sha256"], m_once["compound"]["sha256"]);
        assert_eq!(
            m_timed["compound"]["served_sha256"],
            m_once["compound"]["served_sha256"]
        );
        // The record: 17 dispatched over a window of 7.
        once.record_duration(&mut m_once, 7, &[0.1; 7], &[0.01; 7], 1.0);
        assert!(m_once["compound"].get("duration").is_none());
        timed.record_duration(&mut m_timed, 17, &[1.0; 17], &[0.01; 17], 600.5);
        let r = &m_timed["compound"]["duration"];
        assert_eq!(r["mode"], "cycle-window");
        assert_eq!(r["window"]["queries"], 7);
        assert_eq!(r["window"]["offset"], 3);
        assert_eq!(r["window"]["first_line"], dq::RESERVED + 3);
        assert_eq!(r["full_passes"], 2);
        assert_eq!(r["partial_position"], 3);
        assert_eq!(r["repeated"], true);
        assert_eq!(r["dispatched"], 17);
        assert_eq!(r["intervals"][0]["buckets"][0]["completed"], 17);
    }

    #[test]
    fn disk_windows_of_one_config_never_overlap_and_never_repeat() {
        let (_tmp, dir) = disk_fixture();
        let schema = disk_schema();
        let a = disk_block("rare-term", json!({"offset":0}));
        let b = disk_block("rare-term", json!({"offset":10}));
        let c = disk_block("url-eq", json!({"offset":0}));
        let check = |blocks: Vec<(&Value, usize)>, n: i64| {
            check_compound_before_connecting(
                &dir,
                blocks.into_iter().map(|(v, t)| (Some(v), t)),
                Some(&schema),
                n,
            )
        };
        assert!(check(vec![(&a, 10), (&b, 10), (&c, 10)], 10).is_ok());
        let e = check(vec![(&a, 10), (&b, 10)], 11).unwrap_err();
        assert!(e.contains("overlapping"), "{e}");
        // `--queries -1` runs the rest of the pool from each offset.
        assert!(check(vec![(&a, 10), (&b, 10)], -1).is_err());
        assert!(check(vec![(&a, 10)], 100_000)
            .unwrap_err()
            .contains("never repeated"));
        assert!(check_disk_repetitions([Some(&a)].into_iter(), 3)
            .unwrap_err()
            .contains("--repetitions"));
        assert!(check_disk_repetitions([Some(&a)].into_iter(), 1).is_ok());
        let legacy = json!({"analysis":"parity","fields":{"segment":1.0},
            "compound":{"suite":"msmarco-v1"}});
        assert!(check_disk_repetitions([Some(&legacy)].into_iter(), 3).is_ok());
    }

    #[test]
    fn disk_windows_never_overlap_across_engine_configs_of_one_engine() {
        let (_tmp, dir) = disk_fixture();
        let schema = disk_schema();
        let a = disk_block("rare-term", json!({"offset":0}));
        let b = disk_block("rare-term", json!({"offset":10}));
        let other = disk_block("url-eq", json!({"offset":0}));
        let legacy = json!({"analysis":"parity","fields":{"segment":1.0}});
        let cfg =
            |name: &'static str, engine: &'static str, blocks: Vec<&'static Value>| DiskConfig {
                name,
                engine_type: engine,
                blocks: blocks.into_iter().map(|v| (Some(v), 10)).collect(),
            };
        let leak = |v: Value| -> &'static Value { Box::leak(Box::new(v)) };
        let (a, b, other, legacy) = (leak(a), leak(b), leak(other), leak(legacy));
        let ds = || {
            Ok(vec![(
                "msmarco-disk".to_string(),
                dir.clone(),
                Some(schema.clone()),
            )])
        };
        let check =
            |configs: &[DiskConfig<'_>], n: i64| check_disk_windows_across_configs(configs, ds, n);
        // The finding: two Redis configs (e.g. differing only in `parallel`)
        // on the same window — each config alone passes its own check.
        let same = [
            cfg("flex-p8", "redis", vec![a]),
            cfg("flex-p16", "redis", vec![a]),
        ];
        for c in &same {
            assert!(check_compound_before_connecting(
                &dir,
                c.blocks.iter().copied(),
                Some(&schema),
                10
            )
            .is_ok());
        }
        let e = check(&same, 10).unwrap_err();
        assert!(
            e.contains("'flex-p8' search config #0")
                && e.contains("'flex-p16' search config #0")
                && e.contains("redis on dataset 'msmarco-disk'")
                && e.contains("overlapping"),
            "{e}"
        );
        // Overlap through `--queries`, and the index of a later search config.
        let e = check(
            &[
                cfg("p8", "redis", vec![b]),
                cfg("p16", "redis", vec![legacy, a]),
            ],
            11,
        )
        .unwrap_err();
        assert!(e.contains("'p16' search config #1"), "{e}");
        // Disjoint windows, another family, or another engine type share
        // nothing; Redis vs OpenSearch on one window is the comparison itself.
        assert!(check(
            &[cfg("p8", "redis", vec![a]), cfg("p16", "redis", vec![b])],
            10
        )
        .is_ok());
        assert!(check(
            &[
                cfg("p8", "redis", vec![a]),
                cfg("p16", "redis", vec![other])
            ],
            10
        )
        .is_ok());
        assert!(check(
            &[
                cfg("flex", "redis", vec![a]),
                cfg("os", "opensearch", vec![a])
            ],
            10
        )
        .is_ok());
        // Configs without a disk-suite block never load a dataset.
        assert!(check_disk_windows_across_configs(
            &[
                cfg("x", "redis", vec![legacy]),
                cfg("y", "redis", vec![legacy])
            ],
            || Err("datasets must not be resolved".into()),
            10
        )
        .is_ok());
        // A window that does not load is left to its own per-config check.
        assert!(check(&same, 100_000).is_ok());
    }

    #[test]
    fn fetch_and_count_check_are_recorded_for_both_engines() {
        let f: Vec<String> = ["title", "segment", "url"].map(String::from).to_vec();
        let mut redis = json!({"compound":{"suite":"msmarco-disk-v1"},"engine_settings":{
            "command":"FT.SEARCH <idx> <query> NOCONTENT VERBATIM WITHSCORES"}});
        record_fetch(&mut redis, "redis", Some(&f), 4, 99).unwrap();
        assert_eq!(
            redis["engine_settings"]["command"],
            "FT.SEARCH <idx> <query> RETURN 3 title segment url VERBATIM WITHSCORES"
        );
        assert_eq!(redis["fetch"]["bytes"], 99);
        let mut os = json!({"compound":{"suite":"msmarco-disk-v1"},"engine_settings":{"request":{"_source":false}}});
        record_fetch(&mut os, "opensearch", Some(&f), 4, 99).unwrap();
        assert_eq!(
            os["engine_settings"]["request"]["_source"]["includes"],
            json!(f)
        );
        let mut ids = json!({"compound":{"suite":"msmarco-disk-v1"}});
        record_fetch(&mut ids, "opensearch", None, 0, 0).unwrap();
        assert!(ids.get("fetch").is_some_and(Value::is_null));
        let mut legacy = json!({});
        record_fetch(&mut legacy, "redis", None, 0, 0).unwrap();
        assert!(legacy.get("fetch").is_none());
        for (mode, want) in [
            (TotalHits::Exact, "exact total"),
            (TotalHits::None, "page length"),
        ] {
            let mut m = json!({"compound":{"suite":"msmarco-disk-v1"},
                "engine_settings":{"command":"FT.SEARCH x"}});
            let plan = TotalHitsPlan {
                mode,
                declared: true,
                symmetric: true,
            };
            record_total_hits(&mut m, "redis", &plan).unwrap();
            assert!(m["compound"]["count_check"]
                .as_str()
                .unwrap()
                .contains(want));
        }
    }
}
