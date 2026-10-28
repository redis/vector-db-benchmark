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
pub const KD_NEWLINE_TOKENS: &str =
    "tokenizer: RediSearch splits only on its ASCII separator set plus space and tab — NOT on \
     newline or carriage return, so lines joined by '\\n' (e.g. MS MARCO `headings`) index as \
     one token at each join; OpenSearch's standard tokenizer (UAX#29) splits on all whitespace \
     and Unicode punctuation";
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
pub const KNOWN_DIFFERENCES: [&str; 7] = [
    KD_NEWLINE_TOKENS,
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
pub const PARITY_REMOVES: [&str; 3] = [KD_NEWLINE_TOKENS, KD_STEM_TWICE, KD_MIN_STEM_LEN];

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
    #[serde(default)]
    pub fields: Option<BTreeMap<String, f64>>,
    /// OpenSearch only: accept an index with more than one shard (per-shard
    /// BM25 statistics). Off by default; the shard count is always recorded.
    #[serde(default)]
    pub allow_multi_shard: bool,
    /// Text analysis mode (see [`Analysis`]). Default `english`.
    #[serde(default)]
    pub analysis: Analysis,
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

/// What an engine config's fulltext search configs require of the index:
/// resolved once from all of them ([`index_profile`]), because the index is
/// shared by every search config of the engine config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexProfile {
    pub analysis: Analysis,
    /// Parity: exactly the searched fields (the only fields indexed as text,
    /// and the only ones normalized). English: empty — every schema `text`
    /// field is indexed and nothing is normalized.
    pub text_fields: BTreeSet<String>,
}

impl IndexProfile {
    pub fn english() -> Self {
        IndexProfile {
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
        json!({
            "analysis": self.analysis.as_str(),
            "normalize_whitespace": self.normalizes_whitespace(),
            "text_fields": self.text_fields.iter().collect::<Vec<_>>(),
        })
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
    let mut analysis: Option<Analysis> = None;
    let mut parity_fields: Option<BTreeSet<String>> = None;
    let mut other_workload: Option<Workload> = None;
    for (i, (w, block)) in configs.into_iter().enumerate() {
        if w != Workload::Fulltext {
            other_workload.get_or_insert(w);
            continue;
        }
        let spec = parse_spec(block)?;
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
    if profile.normalizes_whitespace() {
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
        (Workload::Fulltext, b) => {
            let spec = parse_spec(b)?;
            if spec.allow_multi_shard && engine_type != "opensearch" {
                return Err(format!(
                    "`{SPEC_KEY}.allow_multi_shard` only applies to OpenSearch; engine \
                     '{engine_type}' has no shards to allow — remove it"
                ));
            }
            Ok(())
        }
        (w, Some(_)) => Err(format!(
            "a `{SPEC_KEY}` block is only valid on a fulltext search config, not workload \"{w}\""
        )),
        (_, None) => Ok(()),
    }
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
        resolve_fields(&parse_spec(block)?, schema)
            .map_err(|e| format!("search config #{i} (fulltext) on dataset '{dataset}': {e}"))?;
    }
    Ok(())
}

/// Whether an engine config's workloads need [`check_mixed_text_filters`]: a
/// fulltext config sharing the index with a vector or hybrid config.
pub fn mixes_with_filtered_workloads(workloads: &[Workload]) -> bool {
    workloads.contains(&Workload::Fulltext)
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
    Ok(spec)
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
    /// Line of `tests.jsonl` (the dataset query index).
    pub query_id: usize,
    pub terms: Vec<String>,
}

/// The query set derived from `tests.jsonl`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuerySet {
    pub queries: Vec<TextQuery>,
    /// Query ids with no term left after [`query_terms`]; skipped on both engines.
    pub skipped: Vec<usize>,
    /// Lines in `tests.jsonl`.
    pub total: usize,
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
            queries.push(TextQuery { query_id, terms });
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
/// expansion).
pub fn redis_args_for(
    index: &str,
    query: &str,
    top: usize,
    timeout_ms: i64,
    analysis: Analysis,
) -> Vec<String> {
    let mut args = redis_args_base(index, query, top, timeout_ms);
    if analysis == Analysis::Parity {
        args.insert(3, "VERBATIM".to_string());
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

/// The OpenSearch search body.
pub fn opensearch_body(fields: &[FieldBoost], terms: &[String], top: usize) -> Value {
    json!({
        "size": top,
        "_source": false,
        "track_total_hits": false,
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

/// Streaming rank check: every score finite and none above its predecessor.
/// Holds no buffer, so the non-capturing path validates without allocating.
#[derive(Default)]
struct RankCheck {
    prev: Option<f64>,
}

impl RankCheck {
    fn push(&mut self, score: f64) -> Result<(), String> {
        if !score.is_finite() {
            return Err("fulltext: non-finite score in a ranked reply".into());
        }
        if let Some(prev) = self.prev {
            if score > prev {
                return Err(format!(
                    "fulltext: reply is not ranked by descending score ({score} after {prev})"
                ));
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
    pub returned: usize,
    pub hits: Option<RankedHits>,
}

/// Parse a RESP3 `FT.SEARCH … NOCONTENT WITHSCORES` reply. Any warning
/// (timeout partials included) is an error, as on the filter-only path; the
/// page must be complete (`min(total, top)` hits), duplicate-free and ranked.
pub fn redis_ranked_page(
    value: &redis::Value,
    top: usize,
    keep_hits: bool,
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
    for r in results {
        let V::Map(r) = r else {
            return Err("fulltext: invalid Redis result entry".into());
        };
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
        returned: results.len(),
        hits,
    })
}

/// Parse an OpenSearch search response: no timeout, no failed shard, at most
/// `top` hits, each with an `_id` and a `_score`, duplicate-free and ranked.
pub fn opensearch_ranked_page(
    body: &Value,
    top: usize,
    keep_hits: bool,
) -> Result<RankedPage, String> {
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
    for h in results {
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
    Ok(RankedPage {
        returned: results.len(),
        hits,
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
    m["analysis"] = json!(profile.analysis.as_str());
    m["normalize_whitespace"] = json!(profile.normalizes_whitespace());
    m["index_marker"] = recorded_marker.clone();
    m["index_marker_expected"] = profile.marker();
    m["known_differences"] = json!(known_differences(profile.analysis));
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

#[cfg(test)]
mod tests {
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
            let kept: Vec<&str> = KNOWN_DIFFERENCES
                .into_iter()
                .filter(|d| !PARITY_REMOVES.contains(d))
                .collect();
            assert_eq!(kd, kept);
            assert_eq!(kd.len(), KNOWN_DIFFERENCES.len() - PARITY_REMOVES.len());
            for d in [KD_PUNCTUATION, KD_WHOLE_DOCUMENT, KD_SHARD_STATS, KD_TIES] {
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
            assert_eq!(m["known_differences"].as_array().unwrap().len(), 7);
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
}
