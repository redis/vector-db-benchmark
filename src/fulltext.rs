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
pub const KNOWN_DIFFERENCES: [&str; 7] = [
    "tokenizer: RediSearch splits only on its ASCII separator set plus space and tab — NOT on \
     newline or carriage return, so lines joined by '\\n' (e.g. MS MARCO `headings`) index as \
     one token at each join; OpenSearch's standard tokenizer (UAX#29) splits on all whitespace \
     and Unicode punctuation",
    "tokenizer: RediSearch splits on '.', '-', '\\'' inside words ('u.s.a' -> u, s, a; \
     'e-mail' -> e, mail; '3.5' -> 3, 5; \"dog's\" -> dog, s; \"don't\" -> don, t); OpenSearch \
     keeps 'u.s.a', '3.5' and \"don't\" as single tokens and removes possessive 's. The query \
     side drops possessive 's before splitting (so no bare `s` term is sent), but other \
     apostrophes still split into the same terms on both engines",
    "stemming: RediSearch indexes the original token AND its stem and expands each query term \
     to both, so a term whose stem differs scores twice; Lucene indexes only the stem",
    "scoring: RediSearch BM25STD uses one term frequency and one document length for the whole \
     document (all TEXT fields), restricted per field only by the field mask; Lucene BM25 keeps \
     per-field term frequencies and per-field length norms",
    "scoring: Lucene IDF and length norms are per shard and per segment statistics; RediSearch \
     statistics are per index",
    "ties: documents with equal scores may be returned in different orders by the two engines",
    "stemming: RediSearch stems only tokens of at least `search-min-stem-len` characters \
     (default 4, verified on the server and recorded as `redis_min_stem_len`); Lucene porter2 \
     has no length floor",
];

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
}

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
        for bad in ["NOSTEM", "NOINDEX", "PHONETIC"] {
            if flags.contains(&bad) || !attr[bad].is_null() {
                return Err(format!(
                    "fulltext: Redis field '{}' is {bad} — the pinned analyzer stems, indexes \
                     and matches terms exactly",
                    f.field
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
        if m["analyzer"] != OPENSEARCH_ANALYZER
            || !(m["search_analyzer"].is_null() || m["search_analyzer"] == OPENSEARCH_ANALYZER)
        {
            return Err(format!(
                "fulltext: OpenSearch field '{}' is analyzed by {} / search {} — expected the \
                 pinned {OPENSEARCH_ANALYZER}",
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
    let want = opensearch_analysis();
    let got = &index["analysis"];
    let analyzer = &got["analyzer"][OPENSEARCH_ANALYZER];
    let a_want = &want["analyzer"][OPENSEARCH_ANALYZER];
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
            "fulltext: OpenSearch analyzer {OPENSEARCH_ANALYZER} is {analyzer} — expected {a_want}"
        ));
    }
    for name in [OS_POSSESSIVE, OS_STEMMER] {
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
