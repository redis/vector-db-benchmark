//! What the run *actually* used, recorded as it is resolved (#212).
//!
//! # Why this exists
//!
//! Before this module a result file carried only the configuration *name*
//! (`params.experiment`), the dataset, and the search metrics. It carried
//! neither `collection_params` (so not `m`, not `ef_construction`) nor any of
//! the ~90 environment knobs the engines read. Two runs of the same committed
//! configuration could therefore differ by `OPENSEARCH_SEARCH_RETRY_BUDGET_MS`
//! or `DRAGONFLY_UPLOAD_PARALLEL`, report materially different `p99_time` and
//! `rps`, and produce byte-identical `params` blocks. An artifact could not be
//! attributed to the settings that produced it, which is a replicability hole
//! in a tool whose output is used for competitive comparison.
//!
//! # The rule this module enforces
//!
//! **Record what was used, not what was declared.** Writing the declared
//! configuration into the artifact would reproduce the very failure this repo
//! keeps hitting — a file asserting something the run did not do. So:
//!
//! * A knob is recorded *at the point it is resolved*, by the code that
//!   resolves it. [`env_parsed`] both parses and records, so the recorded
//!   number is by construction the number the caller received.
//! * A knob that was never read does not appear. The `env` map lists the
//!   variables this run *looked at*, so a variable that is set in the shell but
//!   irrelevant to the engine cannot masquerade as having mattered.
//! * A knob that was read and unset appears with a `null` value and its
//!   defaulted result in `effective` — "the default applied" is itself a fact
//!   worth recording.
//!
//! # What this module does NOT tell you — read this before trusting `env`
//!
//! There are three tiers of environment read, and only two of them are strong.
//!
//! 1. [`env_parsed`] / [`env_or`] / [`env_flag`] resolve *and* record, so
//!    `effective[NAME]` is by construction the value the caller received. When a
//!    declared value lost — unparseable, unrecognised — both sides land in
//!    `overridden`, never just the winner. **These are the strong ones.**
//! 2. [`env_var`] is a recording shim over [`std::env::var`]. It records the raw
//!    text in `env` and *nothing else*. For a pass-through knob (a credential, a
//!    URI, an index name) the raw text **is** the value used and that is
//!    sufficient. For a knob whose text is then transformed — compared
//!    case-insensitively, matched against `"1"`/`"true"`, parsed with a fallback
//!    to another config source — it is **not**: `VALKEY_PROTOCOL=" resp3 "`
//!    records as `" resp3 "` and selects RESP2, and the artifact cannot show
//!    that. Every remaining such site is enumerated, with a reason, in
//!    `KNOWN_UNRECORDED` (see the guard in this module's tests) and asserted in
//!    both directions, so migrating one to tier 1 *forces* deleting its entry.
//! 3. A raw `std::env::var` would be invisible to all of the above. The guard
//!    below fails the build for any that appears outside this module.
//!
//! So: a variable present in `env` was definitely consulted; a variable whose
//! name is *absent* from `effective` was consulted but its resolution is not
//! recorded, and `overridden: []` does not mean "nothing was overridden".
//!
//! # Scope
//!
//! The recorder is per *experiment* (one engine configuration against one
//! dataset), not per process: [`reset`] runs before each engine is built so a
//! multi-config sweep cannot bleed config A's knobs into config B's artifact.
//!
//! Because the block is scoped to *what had been resolved when the file was
//! written*, the upload and search artifacts of one run legitimately differ: the
//! upload file predates the search phase. Each block carries its own `phase` so
//! that asymmetry reads as sequencing rather than as "this run never consulted
//! that variable".

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Bumped when the shape of the `engine_params` block changes. Independent of
/// `metrics_schema_version`, which versions the *metric definitions*; this one
/// versions the provenance block. Both are emitted.
pub const ENGINE_PARAMS_SCHEMA_VERSION: u32 = 1;

/// Placeholder written in place of a credential's value.
///
/// The variable name and the fact that it was set are provenance (a run against
/// an authenticated cluster is not the same run as one against an open one);
/// the value is not, and these files get published.
const REDACTED: &str = "<redacted:set>";

/// Substrings that make an environment variable's *value* a credential.
///
/// Matched case-insensitively against the variable name. Deliberately not
/// `KEY` on its own — `REDIS_KEY_PREFIX` and `MILVUS_COLLECTION_NAME` are
/// benign and their values are needed to tell two runs apart.
const SECRET_MARKERS: &[&str] = &[
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "APIKEY",
    "ACCESS_TOKEN",
    "SECRET",
    "_AUTH",
    "CREDENTIAL",
];

/// True when `name`'s value must never reach an artifact.
pub fn is_secret(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    SECRET_MARKERS.iter().any(|m| upper.contains(m))
}

/// Strip `user:password@` from a URL-shaped value.
///
/// `REDIS_URI`, `QDRANT_URL` and friends are not credential variables by name,
/// but a connection string routinely carries one inline. Returning the URL with
/// the userinfo blanked keeps the host/port — which is provenance worth having —
/// without publishing the secret.
fn strip_userinfo(value: &str) -> String {
    // Only touch things that look like `scheme://…`; a bare host has no userinfo.
    let Some(sep) = value.find("://") else {
        return value.to_string();
    };
    let (scheme, rest) = value.split_at(sep + 3);
    // The authority ends at the first '/', '?' or '#'.
    let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(auth_end);
    match authority.rfind('@') {
        Some(at) => format!("{scheme}{REDACTED}@{}{tail}", &authority[at + 1..]),
        None => value.to_string(),
    }
}

/// Redact a raw environment value for publication.
fn redact(name: &str, raw: &str) -> Value {
    if is_secret(name) {
        Value::from(REDACTED)
    } else {
        Value::from(strip_userinfo(raw))
    }
}

/// The knobs one experiment resolved.
///
/// Pure data with pure methods — no environment access, no globals — so the
/// tests can drive every branch deterministically. The process-wide instance is
/// a thin wrapper below.
#[derive(Debug, Default, Clone)]
pub struct Recorder {
    /// `collection_params` / `upload_params` exactly as they appear in the
    /// configuration FILE — the raw JSON, not a serde round-trip of the typed
    /// struct, which injects `null`s for undeclared fields, normalises key
    /// casing, and drops any key the struct has no field for. Present so a
    /// reader can *compare* declaration against outcome; never the thing the
    /// artifact claims was used.
    declared: Option<Value>,
    /// How the tool was invoked: the flags that change what a run measures but
    /// live neither in the config file nor in the environment.
    invocation: Option<Value>,
    /// Which phase's state this block describes.
    phase: Option<String>,
    /// Environment variables this run read, mapped to the raw text seen
    /// (`null` when the variable was unset, i.e. a default applied).
    env: BTreeMap<String, Value>,
    /// Resolved values, keyed by knob name. This is the authoritative "what ran".
    effective: BTreeMap<String, Value>,
    /// Declared-vs-used divergences, both sides retained.
    overridden: Vec<Value>,
    /// Declared configuration keys known not to be consumed. NOT exhaustive —
    /// see the note emitted alongside it.
    ignored: BTreeSet<String>,
}

impl Recorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stash the declared blocks, as raw configuration-file JSON, for
    /// side-by-side comparison.
    pub fn set_declared(
        &mut self,
        collection_params: Option<&Value>,
        upload_params: Option<&Value>,
    ) {
        self.declared = Some(json!({
            "collection_params": collection_params,
            "upload_params": upload_params,
        }));
    }

    pub fn set_invocation(&mut self, invocation: Value) {
        self.invocation = Some(invocation);
    }

    pub fn set_phase(&mut self, phase: &str) {
        self.phase = Some(phase.to_string());
    }

    /// Note that `name` was read from the environment, and what was there.
    ///
    /// First observation wins: a variable read twice in a run yields the same
    /// value both times (the process environment is not mutated mid-run), and
    /// keeping the first keeps the map stable.
    pub fn observe_env(&mut self, name: &str, raw: Option<&str>) {
        self.env
            .entry(name.to_string())
            .or_insert_with(|| match raw {
                Some(v) => redact(name, v),
                None => Value::Null,
            });
    }

    /// Record the value a knob resolved to. Last write wins: a knob resolved
    /// twice (e.g. a per-phase re-resolution) should report the latest.
    ///
    /// **Redaction happens HERE, not in the callers.** It used to live in
    /// `env_or` only, which meant a knob resolved through `env_parsed::<String>`
    /// or a bare `record_effective` published its plaintext while the `env` map
    /// two keys away said `<redacted:set>` — the artifact claiming to have
    /// redacted a value it had already printed. One choke point removes that
    /// from every present and future caller; [`Recorder::snapshot`] then asserts
    /// it held.
    pub fn record_effective(&mut self, key: &str, value: Value) {
        let value = match value.as_str() {
            Some(text) => redact(key, text),
            // A non-string (port, timeout, bool) cannot carry a credential.
            None => value,
        };
        self.effective.insert(key.to_string(), value);
    }

    /// Record that `declared` lost to `effective`, and why.
    pub fn note_override(&mut self, key: &str, declared: Value, effective: Value, reason: &str) {
        let scrub = |v: Value| match v.as_str() {
            Some(text) => redact(key, text),
            None => v,
        };
        let entry = json!({
            "key": key,
            "declared": scrub(declared),
            "effective": scrub(effective),
            "reason": reason,
        });
        if !self.overridden.contains(&entry) {
            self.overridden.push(entry);
        }
    }

    /// Record a declared configuration key that is known not to be read.
    pub fn note_ignored(&mut self, key: &str) {
        self.ignored.insert(key.to_string());
    }

    /// Every string value this snapshot would publish, with the key it sits
    /// under. The single place that enumerates the artifact's string leaves, so
    /// the redaction invariant below cannot miss a map somebody adds later.
    fn published_strings(&self) -> Vec<(&str, &str)> {
        fn from_map(m: &BTreeMap<String, Value>) -> impl Iterator<Item = (&str, &str)> {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.as_str(), s)))
        }
        from_map(&self.env)
            .chain(from_map(&self.effective))
            .collect()
    }

    /// The `engine_params` block for an artifact.
    ///
    /// # Panics
    ///
    /// If a credential-named key would be published in the clear. These files
    /// get pasted into issues; aborting the run is the correct response to
    /// discovering we are about to leak, and the invariant is checked here —
    /// once, over every map — rather than trusted to each writer.
    pub fn snapshot(&self) -> Value {
        for (key, value) in self.published_strings() {
            assert!(
                !(is_secret(key) && value != REDACTED),
                "refusing to publish `{key}` in the clear: every credential-named \
                 key must go through `redact` (see Recorder::record_effective)"
            );
        }
        json!({
            "schema_version": ENGINE_PARAMS_SCHEMA_VERSION,
            "phase": self.phase,
            "invocation": self.invocation,
            "declared": self.declared,
            "effective": self.effective,
            "env": self.env,
            "overridden": self.overridden,
            "ignored_declared_keys": {
                // Naming the limit inside the artifact, because `[]` under a
                // bare `ignored_declared_keys` reads as "every declared key was
                // honoured" and that is not what this can establish.
                "exhaustive": false,
                "covers": "hnsw_config keys serde could not type, plus the \
                           CI-asserted KNOWN_UNREAD inventory for this engine",
                "keys": self.ignored.iter().collect::<Vec<_>>(),
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Process-wide instance
// ---------------------------------------------------------------------------

fn recorder() -> MutexGuard<'static, Recorder> {
    static R: OnceLock<Mutex<Recorder>> = OnceLock::new();
    // A poisoned lock must not abort a benchmark: provenance is telemetry, and
    // losing it is strictly better than losing the run that produced it.
    let m = R.get_or_init(|| Mutex::new(Recorder::new()));
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Start a fresh recording. Prefer [`begin_experiment`], which cannot be
/// half-called; this exists for tests that drive the recorder directly.
pub fn reset() {
    *recorder() = Recorder::new();
}

/// Begin recording one experiment: clear the previous configuration's state,
/// then stash this one's declared configuration and invocation.
///
/// **One entry point on purpose.** These three steps used to be three separate
/// calls at the top of the experiment loop, and a mutation campaign showed that
/// deleting the `reset()` line left the whole suite green while a sweep's later
/// configurations silently inherited the earlier ones' knobs — the exact
/// sweep-bleed this module's docs warn about. Resetting is now inseparable from
/// declaring, so forgetting it is not expressible.
pub fn begin_experiment(engine_config: &crate::config::EngineConfig, invocation: Value) {
    reset();
    set_declared(engine_config);
    set_invocation(invocation);
}

/// Record how the tool was invoked.
///
/// `--host` and `--skip-upload` in particular change what a run measures while
/// living in neither the config file nor the environment: two runs against two
/// different servers, or one that built the index and one that searched somebody
/// else's, were otherwise byte-identical here.
pub fn set_invocation(invocation: Value) {
    recorder().set_invocation(invocation);
}

/// Tag the block with the phase whose resolved state it describes.
pub fn set_phase(phase: &str) {
    recorder().set_phase(phase);
}

/// Stash the declared configuration blocks for this experiment.
///
/// Takes the **raw configuration-file JSON** for this entry, not the typed
/// [`crate::config::EngineConfig`]. Serialising the struct instead would inject
/// `null`s for fields nobody declared, rewrite `{"m","ef_construct"}` as
/// `{"M","EF_CONSTRUCTION"}` through the serde aliases, and silently drop any
/// key the struct has no field for (`index_options.type`,
/// `index_options.confidence_interval`) — a "declared" record that is not what
/// was declared.
pub fn set_declared(engine_config: &crate::config::EngineConfig) {
    let raw = engine_config.raw.as_ref();
    let field = |name: &str| raw.and_then(|r| r.get(name)).cloned();
    let collection = field("collection_params");
    let upload = field("upload_params");
    recorder().set_declared(collection.as_ref(), upload.as_ref());

    // `hnsw_config` keys that serde's catch-all absorbed are, by definition,
    // keys no engine reads. Recording them keeps a typo'd knob from passing for
    // a configured one in the artifact the same way it already warns on stderr.
    if let Some(hnsw) = engine_config
        .collection_params
        .as_ref()
        .and_then(|c| c.hnsw_config.as_ref())
    {
        for key in hnsw.unsupported_keys() {
            note_ignored(&format!("collection_params.hnsw_config.{key}"));
        }
    }

    // The repo already maintains a CI-asserted inventory of knobs shipped
    // configs declare and their engine does not read (`KNOWN_UNREAD`, asserted
    // still-unread so a fix must delete its entry). It was `#[cfg(test)]`-only,
    // so the guard said "documented debt, #245" while the artifact for the very
    // same config said nothing. Same source of truth, both places.
    if let Some(engine) = engine_config.engine.as_deref() {
        for (path, why) in crate::config::known_unread_for(engine) {
            if raw.map(|r| json_path_exists(r, path)).unwrap_or(false) {
                note_ignored(path);
                note_override(
                    path,
                    raw.and_then(|r| json_path_get(r, path))
                        .unwrap_or(Value::Null),
                    Value::Null,
                    why,
                );
            }
        }
    }
}

/// Walk a dotted path (`"connection_params.request_timeout"`) into a JSON value.
fn json_path_get(root: &Value, path: &str) -> Option<Value> {
    path.split('.')
        .try_fold(root, |cur, seg| cur.get(seg))
        .cloned()
}

fn json_path_exists(root: &Value, path: &str) -> bool {
    json_path_get(root, path).is_some()
}

/// Recording drop-in for [`std::env::var`].
///
/// Behaviourally identical — same return type, same errors — so a call site
/// migrates by changing the path and nothing else. Every engine's environment
/// read goes through here, which is what makes the `env` map a record of what
/// the run *looked at* rather than a dump of the ambient environment.
///
/// Cost: one mutex acquisition on top of the libc environment scan and `String`
/// allocation [`std::env::var`] already performs. No call site sits inside a
/// per-query timed window — the engines resolve their knobs in `new()` or at
/// phase start precisely to keep `env::var` off the query path. Two sites do sit
/// inside a *measured region*: `opensearch::upload_bulk_batch` re-resolves its
/// retry budget once per bulk batch inside the loop `upload()` times as
/// `upload_time`, and valkey's auth reads run once per search worker inside the
/// spawned connection closures. Both are per-batch / per-worker rather than
/// per-operation, so the added lock is orders below the network round trip it
/// precedes — but "enters no timed region" would be false, and in a benchmarking
/// tool that claim has to be true rather than asserted.
pub fn env_var(name: &str) -> Result<String, std::env::VarError> {
    let got = std::env::var(name);
    recorder().observe_env(name, got.as_deref().ok());
    got
}

/// Resolve `name` to `T`, falling back to `default`, recording the value the
/// caller actually receives.
///
/// Replaces the ubiquitous
/// `env::var(N).ok().and_then(|v| v.parse().ok()).unwrap_or(D)` idiom, with two
/// deliberate differences:
///
/// 1. **It records.** The value written to the artifact is the value returned,
///    so the two cannot drift. When the variable was set to text that does not
///    parse, the run silently used `default` — that divergence goes to
///    `overridden` with both sides rather than being papered over by recording
///    either one alone.
/// 2. **It trims before parsing**, so `" 5 "` yields 5 where the old idiom fell
///    back to the default. This is the only behaviour change in the module: it
///    makes the ~90 knobs uniform (several sites already trimmed), matches the
///    repo's newer `parse_env_secs`, and is a plain safety win on the 14 `*_PORT`
///    knobs. The untrimmed text still lands verbatim in `env`, so nothing is
///    hidden. Note `str::trim` uses the Unicode `White_Space` property, so a
///    NO-BREAK SPACE pasted out of documentation (`"\u{a0}5"`) is also accepted.
///
/// This does **not** make an unusable value fatal — that is
/// [#260](https://github.com/redis/vector-db-benchmark/issues/260), which wants
/// a present-but-unparseable knob to fail at construction rather than default.
/// The two compose: #260 swaps this call for a fallible sibling, and because the
/// divergence is already recorded here, the intermediate state is "silently
/// defaulted" -> "defaulted and said so" -> "refused". Nothing #260 needs is
/// undone by this migration; it changes the same one line per site.
pub fn env_parsed<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr + Clone + Into<Value>,
{
    let raw = std::env::var(name).ok();
    let mut rec = recorder();
    rec.observe_env(name, raw.as_deref());

    let value = match raw.as_deref() {
        Some(text) => match text.trim().parse::<T>() {
            Ok(v) => v,
            Err(_) => {
                rec.note_override(
                    name,
                    Value::from(text),
                    default.clone().into(),
                    "environment value did not parse; the default was used instead",
                );
                default
            }
        },
        None => default,
    };
    rec.record_effective(name, value.clone().into());
    value
}

/// Resolve `name` to a string, falling back to `default`, recording the result.
///
/// The string counterpart of [`env_parsed`], for the
/// `env::var(N).unwrap_or_else(|_| D.to_string())` idiom.
pub fn env_or(name: &str, default: &str) -> String {
    let raw = std::env::var(name).ok();
    let mut rec = recorder();
    rec.observe_env(name, raw.as_deref());
    let value = raw.unwrap_or_else(|| default.to_string());
    // Redaction is applied by `record_effective` itself; passing the plaintext
    // is correct and is what every other caller does.
    rec.record_effective(name, Value::from(value.clone()));
    value
}

/// Resolve `name` as an opt-in boolean flag, recording the boolean the caller
/// received rather than the text it came from.
///
/// `accept` names the values that mean "on"; anything else means "off". This is
/// the tier-1 form of the `env_var(N).map(|v| v == "1" || v == "true")` and
/// `env_var(N).map(|v| v.eq_ignore_ascii_case("resp3"))` idioms, where the raw
/// text alone is actively misleading: `VALKEY_PROTOCOL=" resp3 "` selects RESP2
/// while `env` shows `" resp3 "`, and a reader grepping for the variable sees
/// what looks like a successful opt-in. A set-but-unrecognised value therefore
/// lands in `overridden` — somebody tried to turn this on and did not.
pub fn env_flag(name: &str, accept: &[&str]) -> bool {
    let raw = std::env::var(name).ok();
    let mut rec = recorder();
    rec.observe_env(name, raw.as_deref());

    let value = match raw.as_deref() {
        Some(text) => {
            let on = accept.iter().any(|a| text.eq_ignore_ascii_case(a));
            if !on {
                rec.note_override(
                    name,
                    Value::from(text),
                    Value::from(false),
                    "environment value is not one of the accepted flag values; \
                     the flag stayed off",
                );
            }
            on
        }
        None => false,
    };
    rec.record_effective(name, Value::from(value));
    value
}

/// Resolve `name` as a presence-only switch: **set to anything** means on.
///
/// Records the boolean the caller received. The distinction matters more here
/// than anywhere else in the module, because the value is discarded outright —
/// `WEAVIATE_USE_GRAPHQL=false` selects GraphQL. Recording only the raw text
/// would put `"false"` in an artifact for a run that took the GraphQL path.
pub fn env_present(name: &str) -> bool {
    let raw = std::env::var(name).ok();
    let mut rec = recorder();
    rec.observe_env(name, raw.as_deref());
    let value = raw.is_some();
    rec.record_effective(name, Value::from(value));
    value
}

/// Resolve `name` as an optional non-empty string, recording whether it ended up
/// applying. Records `null` when unset or blank — the value the caller got, not
/// the whitespace it came from.
pub fn env_opt(name: &str) -> Option<String> {
    let raw = std::env::var(name).ok();
    let mut rec = recorder();
    rec.observe_env(name, raw.as_deref());
    let value = raw.filter(|s| !s.trim().is_empty());
    rec.record_effective(name, value.clone().map(Value::from).unwrap_or(Value::Null));
    value
}

/// Record a resolved knob that did not come straight from one environment
/// variable — a value chosen between the environment, the configuration file
/// and a built-in default, or read back off the server.
pub fn record_effective(key: &str, value: impl Into<Value>) {
    recorder().record_effective(key, value.into());
}

/// Record that a declared value was not the one used.
pub fn note_override(
    key: &str,
    declared: impl Into<Value>,
    effective: impl Into<Value>,
    reason: &str,
) {
    recorder().note_override(key, declared.into(), effective.into(), reason);
}

/// Record a declared configuration key that nothing consumed.
pub fn note_ignored(key: &str) {
    recorder().note_ignored(key);
}

/// The `engine_params` block to embed in an artifact.
pub fn snapshot() -> Value {
    recorder().snapshot()
}

/// Serialises the tests that mutate the process environment and drive the
/// process-wide recorder. `cargo test` runs a binary's tests on many threads,
/// and both of those are global.
#[cfg(test)]
pub fn test_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Source-scanning guards that keep this module the ONLY way to read the
/// environment, and keep the weak tier honest about its own membership.
///
/// Why source scanning rather than a runtime check: the failure being guarded is
/// a knob that is *never recorded*, which by definition leaves no runtime trace.
/// A raw `std::env::var` added the ordinary way compiles, passes every test, and
/// silently drops its variable — and, worse, an early return past a recorded read
/// removes a variable that WAS consulted, so the artifact affirmatively reports
/// that no such knob was consulted on a run that changed behaviour.
#[cfg(test)]
mod recorder_coverage_guard {
    use std::path::{Path, PathBuf};

    /// Files allowed to call `std::env::*` directly.
    ///
    /// `effective_config.rs` is the recorder itself. `config.rs` (the library
    /// crate) carries `RedisConfig::from_env`, a `pub` recorder-bypassing
    /// constructor whose only callers are `src/redisearch/` and
    /// `src/vectorsets/` — directories that are NOT in `lib.rs`'s module list
    /// and therefore are not compiled at all. Exempted rather than deleted here
    /// to keep this change to provenance; removing the dead constructor and the
    /// two dead directories is tracked separately.
    const EXEMPT: &[&str] = &[
        "src/bin/vector_db_benchmark/effective_config.rs",
        "src/config.rs",
    ];

    /// Directories on disk that no crate root declares, so nothing in them is
    /// compiled and nothing in them can read the environment at runtime.
    const UNCOMPILED: &[&str] = &["src/redisearch/", "src/vectorsets/"];

    /// Environment variables still read through the plain [`super::env_var`]
    /// shim, which records the raw text and nothing else.
    ///
    /// Asserted in BOTH directions, exactly like `config::KNOWN_UNREAD`: every
    /// `env_var` site must be listed here, and every entry here must still be an
    /// `env_var` site. Migrating one to `env_parsed`/`env_flag`/`env_opt`
    /// therefore FORCES deleting its row, so the list cannot rot into a stale
    /// excuse — and adding a new weak read forces writing down why.
    ///
    /// Every row below is a pass-through: the raw text IS the value used, so
    /// `env` alone is a complete record of it. A knob whose text gets
    /// transformed does not belong here — it belongs in a recording helper.
    const KNOWN_UNRECORDED: &[(&str, &str)] = &[
        (
            "DRAGONFLY_AUTH",
            "credential, used verbatim; redacted in `env`",
        ),
        ("DRAGONFLY_USER", "username, used verbatim"),
        (
            "DRAGONFLY_UPLOAD_BATCH_SIZE",
            "env > upload_params > default; no single helper sees the winner, so \
             the resolved value is recorded as `upload_batch_size`",
        ),
        (
            "DRAGONFLY_UPLOAD_PARALLEL",
            "as DRAGONFLY_UPLOAD_BATCH_SIZE; resolved value recorded as `upload_parallel`",
        ),
        (
            "ELASTIC_API_KEY",
            "credential, used verbatim; redacted in `env`",
        ),
        (
            "KIVIDB_AUTH",
            "credential, used verbatim; redacted in `env`",
        ),
        ("KIVIDB_USER", "username, used verbatim"),
        (
            "KIVIDB_UPLOAD_BATCH_SIZE",
            "as DRAGONFLY_UPLOAD_BATCH_SIZE; resolved value recorded as `upload_batch_size`",
        ),
        (
            "KIVIDB_UPLOAD_PARALLEL",
            "as DRAGONFLY_UPLOAD_PARALLEL; resolved value recorded as `upload_parallel`",
        ),
        (
            "MONGODB_PASSWORD",
            "credential, used verbatim; redacted in `env`",
        ),
        ("MONGODB_USER", "username, used verbatim"),
        (
            "QDRANT_API_KEY",
            "credential, used verbatim; redacted in `env`",
        ),
        (
            "QDRANT_REST_URL",
            "URL, used verbatim; userinfo stripped in `env`",
        ),
        (
            "QDRANT_URL",
            "URL, used verbatim; userinfo stripped in `env`",
        ),
        (
            "REDIS_KEY_PREFIX",
            "trimmed and `:`-suffixed before use, so the resolved namespace is \
             recorded as `shared_corpus_key_prefix`",
        ),
        (
            "REDIS_SVS_COMPRESSION",
            "config takes precedence, then trim + empty-filter; resolved value \
             recorded as `svs_compression`",
        ),
        (
            "REDIS_SVS_REDUCE",
            "config takes precedence, then trim + positivity filter; resolved \
             value recorded as `svs_reduce`",
        ),
        ("REDIS_AUTH", "credential, used verbatim; redacted in `env`"),
        (
            "REDIS_URI",
            "URL, used verbatim; userinfo stripped in `env`",
        ),
        ("REDIS_USER", "username, used verbatim"),
        (
            "TURBOPUFFER_API_KEY",
            "credential, used verbatim; redacted in `env`",
        ),
        (
            "VALKEY_AUTH",
            "credential, used verbatim; redacted in `env`",
        ),
        ("VALKEY_USER", "username, used verbatim"),
        (
            "VERTEX_ACCESS_TOKEN",
            "credential, used verbatim; redacted in `env`",
        ),
        ("VERTEX_DEPLOYED_INDEX_ID", "resource id, used verbatim"),
        ("VERTEX_INDEX", "resource id, used verbatim"),
        ("VERTEX_INDEX_ENDPOINT", "resource id, used verbatim"),
        ("VERTEX_PROJECT", "GCP project, used verbatim"),
        (
            "WEAVIATE_API_KEY",
            "credential, used verbatim; redacted in `env`",
        ),
    ];

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    fn rust_sources() -> Vec<(String, String)> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        let root = repo_root();
        let mut files = Vec::new();
        walk(&root.join("src"), &mut files);
        files
            .into_iter()
            .filter_map(|p| {
                let rel = p.strip_prefix(&root).ok()?.to_string_lossy().to_string();
                Some((rel, std::fs::read_to_string(&p).ok()?))
            })
            .collect()
    }

    /// Drop `#[cfg(test)]` modules so a test helper mutating the environment is
    /// not mistaken for a production read. Brace-counts from the module's
    /// opening `{`, which is sufficient for this codebase's formatting (rustfmt
    /// keeps braces balanced and string literals here contain none).
    fn strip_test_modules(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        let mut rest = src;
        while let Some(idx) = rest.find("#[cfg(test)]") {
            out.push_str(&rest[..idx]);
            let after = &rest[idx..];
            let Some(open) = after.find('{') else {
                break;
            };
            let mut depth = 0usize;
            let mut end = None;
            for (i, c) in after[open..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(open + i + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            match end {
                Some(e) => rest = &after[e..],
                None => break,
            }
        }
        out.push_str(rest);
        out
    }

    /// GUARD 1 — nothing outside the recorder reads the environment.
    ///
    /// Catches a raw `std::env::var` added the ordinary way, which compiles
    /// clean, passes the suite, and drops its knob from every artifact.
    #[test]
    fn no_production_code_reads_the_environment_outside_the_recorder() {
        let mut offenders = Vec::new();
        for (path, src) in rust_sources() {
            if EXEMPT.contains(&path.as_str()) || UNCOMPILED.iter().any(|d| path.starts_with(d)) {
                continue;
            }
            let production = strip_test_modules(&src);
            for probe in ["env::var(", "env::var_os(", "env::vars("] {
                if production.contains(probe) {
                    offenders.push(format!("{path}: {probe}"));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "these read the environment without recording it, so the knob will be \
             missing from every result file — route them through \
             `effective_config` (env_parsed / env_or / env_flag / env_opt / \
             env_var):\n  {}",
            offenders.join("\n  ")
        );
    }

    /// GUARD 3 — the experiment loop actually begins a recording.
    ///
    /// A source scan, because `experiment::run` cannot be unit-tested: it builds
    /// engines and talks to servers. Deleting its
    /// `effective_config::begin_experiment(...)` line therefore leaves every
    /// unit test green while every artifact loses its provenance — and a sweep
    /// silently carries configuration A's knobs into configuration B's file.
    /// Crude, but it fails on the one edit that would otherwise be invisible
    /// until someone reads a published result.
    #[test]
    fn the_experiment_loop_begins_a_recording_before_building_the_engine() {
        let src =
            std::fs::read_to_string(repo_root().join("src/bin/vector_db_benchmark/experiment.rs"))
                .expect("experiment.rs");
        let production = strip_test_modules(&src);
        let begin = production
            .find("effective_config::begin_experiment(")
            .expect(
                "experiment.rs no longer calls `effective_config::begin_experiment` — every \
                 result file this run writes will have empty provenance, and a sweep will \
                 carry each config's knobs into the next one's artifact",
            );
        let create = production
            .find("create_engine(&engine_config")
            .expect("experiment.rs no longer calls create_engine as expected");
        assert!(
            begin < create,
            "the recording must begin BEFORE the engine is built — engines resolve most of \
             their environment knobs in `new()`, so a later reset would erase them"
        );
    }

    /// GUARD 2 — the weak tier's membership is written down, both ways.
    #[test]
    fn known_unrecorded_matches_the_env_var_call_sites_exactly() {
        let mut actual: Vec<String> = Vec::new();
        for (path, src) in rust_sources() {
            if path.ends_with("effective_config.rs") {
                continue;
            }
            let production = strip_test_modules(&src);
            let mut rest = production.as_str();
            while let Some(i) = rest.find("effective_config::env_var(") {
                rest = &rest[i + "effective_config::env_var(".len()..];
                // Literal `"NAME"` argument; the two dynamic-name sites build
                // their name with `format!` and are covered by their base var.
                if let Some(stripped) = rest.strip_prefix('"') {
                    if let Some(end) = stripped.find('"') {
                        actual.push(stripped[..end].to_string());
                    }
                }
            }
        }
        actual.sort();
        actual.dedup();

        let mut listed: Vec<String> = KNOWN_UNRECORDED
            .iter()
            .map(|(v, _)| v.to_string())
            .collect();
        listed.sort();
        listed.dedup();

        let missing: Vec<_> = actual.iter().filter(|v| !listed.contains(v)).collect();
        assert!(
            missing.is_empty(),
            "new `env_var` call sites are not listed in KNOWN_UNRECORDED. Either \
             use a recording helper, or add a row saying why the raw text is the \
             whole story: {missing:?}"
        );
        let stale: Vec<_> = listed.iter().filter(|v| !actual.contains(v)).collect();
        assert!(
            stale.is_empty(),
            "KNOWN_UNRECORDED lists variables that are no longer read through \
             `env_var` — delete their rows so the list cannot become a stale \
             excuse: {stale:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Set a variable for the duration of a test and restore it afterwards.
    struct EnvGuard(&'static str, Option<String>);
    impl EnvGuard {
        fn set(name: &'static str, value: Option<&str>) -> Self {
            let prev = std::env::var(name).ok();
            match value {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
            Self(name, prev)
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.1 {
                Some(v) => std::env::set_var(self.0, v),
                None => std::env::remove_var(self.0),
            }
        }
    }

    #[test]
    fn env_parsed_records_the_number_the_caller_received() {
        let _l = test_lock();
        let _g = EnvGuard::set("VDBB_TEST_BUDGET_MS", Some("250"));
        reset();
        let got: u64 = env_parsed("VDBB_TEST_BUDGET_MS", 2_000);
        assert_eq!(got, 250);
        let s = snapshot();
        // The artifact must agree with the caller, not with the default.
        assert_eq!(s["effective"]["VDBB_TEST_BUDGET_MS"], 250);
        assert_eq!(s["env"]["VDBB_TEST_BUDGET_MS"], "250");
        assert!(s["overridden"].as_array().unwrap().is_empty());
    }

    #[test]
    fn env_parsed_records_the_default_when_the_variable_is_unset() {
        let _l = test_lock();
        let _g = EnvGuard::set("VDBB_TEST_BUDGET_MS", None);
        reset();
        let got: u64 = env_parsed("VDBB_TEST_BUDGET_MS", 2_000);
        assert_eq!(got, 2_000);
        let s = snapshot();
        assert_eq!(s["effective"]["VDBB_TEST_BUDGET_MS"], 2_000);
        assert_eq!(s["env"]["VDBB_TEST_BUDGET_MS"], Value::Null);
    }

    /// The dishonest-artifact case. A variable set to text that does not parse
    /// leaves the run on its default: recording only the raw text would claim a
    /// budget the run never used, and recording only the default would hide that
    /// somebody tried to set one. Both sides, or the file lies either way.
    #[test]
    fn unparseable_env_value_records_the_default_and_the_text_it_discarded() {
        let _l = test_lock();
        let _g = EnvGuard::set("VDBB_TEST_BUDGET_MS", Some("500ms"));
        reset();
        let got: u64 = env_parsed("VDBB_TEST_BUDGET_MS", 2_000);
        assert_eq!(got, 2_000, "the run used the default");

        let s = snapshot();
        assert_eq!(
            s["effective"]["VDBB_TEST_BUDGET_MS"], 2_000,
            "effective must be what the run used"
        );
        let o = &s["overridden"][0];
        assert_eq!(o["key"], "VDBB_TEST_BUDGET_MS");
        assert_eq!(o["declared"], "500ms");
        assert_eq!(o["effective"], 2_000);
        assert!(o["reason"].as_str().unwrap().contains("did not parse"));
    }

    #[test]
    fn env_or_records_the_resolved_string_either_way() {
        let _l = test_lock();
        {
            let _g = EnvGuard::set("VDBB_TEST_INDEX", Some("tuned"));
            reset();
            assert_eq!(env_or("VDBB_TEST_INDEX", "bench"), "tuned");
            assert_eq!(snapshot()["effective"]["VDBB_TEST_INDEX"], "tuned");
        }
        let _g = EnvGuard::set("VDBB_TEST_INDEX", None);
        reset();
        assert_eq!(env_or("VDBB_TEST_INDEX", "bench"), "bench");
        assert_eq!(snapshot()["effective"]["VDBB_TEST_INDEX"], "bench");
    }

    /// `env_or` resolves credential variables too (`ELASTIC_PASSWORD` has a
    /// built-in default). Its `effective` entry must be redacted as well — the
    /// `env` map alone is not the only place a value can escape.
    #[test]
    fn env_or_redacts_credentials_in_the_effective_map_too() {
        let _l = test_lock();
        let _g = EnvGuard::set("VDBB_TEST_PASSWORD", Some("hunter2"));
        reset();
        assert_eq!(env_or("VDBB_TEST_PASSWORD", "passwd"), "hunter2");
        let dumped = serde_json::to_string(&snapshot()).unwrap();
        assert!(!dumped.contains("hunter2"), "{dumped}");
    }

    /// A sweep runs many configurations in one process. Without a reset per
    /// experiment, configuration B's artifact would inherit configuration A's
    /// knobs and assert settings that had nothing to do with it.
    #[test]
    fn reset_drops_the_previous_experiments_knobs() {
        let _l = test_lock();
        reset();
        record_effective("m", 64);
        assert_eq!(snapshot()["effective"]["m"], 64);
        reset();
        assert!(snapshot()["effective"].as_object().unwrap().is_empty());
        assert!(snapshot()["declared"].is_null());
    }

    #[test]
    fn unset_variable_is_recorded_as_read_and_null() {
        let mut r = Recorder::new();
        r.observe_env("OPENSEARCH_TIMEOUT", None);
        let s = r.snapshot();
        // Present-with-null is the point: "this run read the variable and found
        // nothing, so the default applied" is different from "never read".
        assert!(s["env"]
            .as_object()
            .unwrap()
            .contains_key("OPENSEARCH_TIMEOUT"));
        assert_eq!(s["env"]["OPENSEARCH_TIMEOUT"], Value::Null);
    }

    #[test]
    fn variable_never_read_is_absent() {
        let r = Recorder::new();
        let s = r.snapshot();
        assert!(s["env"].as_object().unwrap().is_empty());
    }

    #[test]
    fn first_observation_wins() {
        let mut r = Recorder::new();
        r.observe_env("X_TIMEOUT", Some("1"));
        r.observe_env("X_TIMEOUT", Some("2"));
        assert_eq!(r.snapshot()["env"]["X_TIMEOUT"], "1");
    }

    #[test]
    fn effective_last_write_wins() {
        let mut r = Recorder::new();
        r.record_effective("ef", json!(64));
        r.record_effective("ef", json!(128));
        assert_eq!(r.snapshot()["effective"]["ef"], 128);
    }

    #[test]
    fn credentials_are_recorded_as_set_but_not_disclosed() {
        let mut r = Recorder::new();
        for (name, raw) in [
            ("ELASTIC_PASSWORD", "hunter2"),
            ("QDRANT_API_KEY", "sk-abc"),
            ("REDIS_AUTH", "s3cret"),
            ("VERTEX_ACCESS_TOKEN", "ya29.xyz"),
            ("MONGODB_PASSWORD", "pw"),
        ] {
            r.observe_env(name, Some(raw));
        }
        let s = r.snapshot();
        let serialized = serde_json::to_string(&s).unwrap();
        for leak in ["hunter2", "sk-abc", "s3cret", "ya29.xyz"] {
            assert!(
                !serialized.contains(leak),
                "{leak} leaked into the artifact"
            );
        }
        // …but the fact that authentication was configured survives.
        assert_eq!(s["env"]["ELASTIC_PASSWORD"], REDACTED);
    }

    #[test]
    fn benign_names_containing_key_are_not_redacted() {
        let mut r = Recorder::new();
        r.observe_env("REDIS_KEY_PREFIX", Some("bench:"));
        r.observe_env("MILVUS_COLLECTION_NAME", Some("benchmark"));
        let s = r.snapshot();
        assert_eq!(s["env"]["REDIS_KEY_PREFIX"], "bench:");
        assert_eq!(s["env"]["MILVUS_COLLECTION_NAME"], "benchmark");
    }

    #[test]
    fn connection_string_userinfo_is_stripped_but_host_kept() {
        let mut r = Recorder::new();
        r.observe_env("REDIS_URI", Some("redis://admin:hunter2@10.0.0.5:6379/0"));
        r.observe_env("QDRANT_URL", Some("http://qdrant.internal:6334"));
        let s = r.snapshot();
        let uri = s["env"]["REDIS_URI"].as_str().unwrap();
        assert!(!uri.contains("hunter2"), "password leaked: {uri}");
        assert!(uri.contains("10.0.0.5:6379"), "host lost: {uri}");
        // A URL with no userinfo is untouched.
        assert_eq!(s["env"]["QDRANT_URL"], "http://qdrant.internal:6334");
    }

    #[test]
    fn strip_userinfo_leaves_non_urls_alone() {
        assert_eq!(strip_userinfo("localhost:6379"), "localhost:6379");
        assert_eq!(strip_userinfo(""), "");
        assert_eq!(strip_userinfo("http://h/p@th"), "http://h/p@th");
    }

    #[test]
    fn override_keeps_both_sides() {
        let mut r = Recorder::new();
        r.note_override(
            "collection_params.number_of_shards",
            json!(3),
            json!(1),
            "pinned",
        );
        let o = &r.snapshot()["overridden"][0];
        assert_eq!(o["declared"], 3);
        assert_eq!(o["effective"], 1);
        assert_eq!(o["reason"], "pinned");
    }

    #[test]
    fn overrides_deduplicate() {
        let mut r = Recorder::new();
        for _ in 0..3 {
            r.note_override("k", json!(1), json!(2), "why");
        }
        assert_eq!(r.snapshot()["overridden"].as_array().unwrap().len(), 1);
    }

    /// A LITERAL, not the constant against itself — the previous form could not
    /// fail for any value, including one that silently changed the artifact
    /// contract. Bumping the version is a deliberate act that updates this line.
    #[test]
    fn snapshot_carries_schema_version_one() {
        assert_eq!(Recorder::new().snapshot()["schema_version"], 1);
    }

    /// The redaction invariant, at the choke point.
    ///
    /// `redact` used to live in `env_or` alone, so resolving a credential
    /// through `env_parsed::<String>` or a bare `record_effective` published the
    /// plaintext in `effective` while `env`, two keys away, showed
    /// `<redacted:set>` — an artifact claiming to have redacted a value it had
    /// already printed, with every test green. Both paths are exercised here,
    /// and `snapshot` asserts the invariant over every map regardless.
    #[test]
    fn credentials_cannot_reach_effective_by_any_route() {
        let mut r = Recorder::new();
        // The `env_parsed::<String>` route.
        r.record_effective("ELASTIC_PASSWORD", json!("es-hunter2-secret"));
        // The bare `record_effective` route.
        r.record_effective("QDRANT_API_KEY", json!("sk-live-abc"));
        r.record_effective("VERTEX_ACCESS_TOKEN", json!("ya29.leak"));
        // And an override entry, which carries two more string slots.
        r.note_override("REDIS_AUTH", json!("old-pw"), json!("new-pw"), "rotated");

        let dumped = serde_json::to_string(&r.snapshot()).unwrap();
        for leak in [
            "es-hunter2-secret",
            "sk-live-abc",
            "ya29.leak",
            "old-pw",
            "new-pw",
        ] {
            assert!(!dumped.contains(leak), "{leak} leaked: {dumped}");
        }
        assert_eq!(r.snapshot()["effective"]["ELASTIC_PASSWORD"], REDACTED);
    }

    /// The invariant is enforced by `snapshot`, not merely by the writers, so a
    /// future map or a route that skips `record_effective` still cannot publish.
    #[test]
    #[should_panic(expected = "refusing to publish")]
    fn snapshot_refuses_to_publish_a_credential_in_the_clear() {
        let mut r = Recorder::new();
        // Bypass `record_effective` the way a careless future edit would.
        r.effective
            .insert("MONGODB_PASSWORD".into(), json!("plaintext"));
        r.snapshot();
    }

    /// Non-string values cannot carry a credential and must survive intact —
    /// redacting a port would be its own kind of wrong record.
    #[test]
    fn numeric_and_boolean_effective_values_are_untouched() {
        let mut r = Recorder::new();
        r.record_effective("REDIS_PORT", json!(7777));
        r.record_effective("WEAVIATE_USE_GRAPHQL", json!(true));
        let s = r.snapshot();
        assert_eq!(s["effective"]["REDIS_PORT"], 7777);
        assert_eq!(s["effective"]["WEAVIATE_USE_GRAPHQL"], true);
    }

    /// `env_var` is the shim on the remaining pass-through sites. Nothing tested
    /// it, so it could stop recording entirely with the suite green — and it
    /// covers `REDIS_AUTH`, `ELASTIC_API_KEY` and every connection URI.
    #[test]
    fn env_var_shim_records_and_still_returns_what_std_would() {
        let _l = test_lock();
        {
            let _g = EnvGuard::set("VDBB_TEST_SHIM", Some("value"));
            reset();
            assert_eq!(env_var("VDBB_TEST_SHIM").unwrap(), "value");
            assert_eq!(snapshot()["env"]["VDBB_TEST_SHIM"], "value");
        }
        let _g = EnvGuard::set("VDBB_TEST_SHIM", None);
        reset();
        assert!(env_var("VDBB_TEST_SHIM").is_err());
        assert_eq!(snapshot()["env"]["VDBB_TEST_SHIM"], Value::Null);
    }

    /// `env_flag` is the fix for the tier-2 hazard: the raw text alone cannot
    /// show that an opt-in did not take.
    #[test]
    fn env_flag_records_the_boolean_used_and_flags_an_unrecognised_value() {
        let _l = test_lock();
        {
            let _g = EnvGuard::set("VDBB_TEST_PROTOCOL", Some("RESP3"));
            reset();
            assert!(env_flag("VDBB_TEST_PROTOCOL", &["resp3"]));
            assert_eq!(snapshot()["effective"]["VDBB_TEST_PROTOCOL"], true);
        }
        // The reviewer's live reproduction: two space characters silently select
        // the other wire protocol.
        let _g = EnvGuard::set("VDBB_TEST_PROTOCOL", Some(" resp3 "));
        reset();
        assert!(!env_flag("VDBB_TEST_PROTOCOL", &["resp3"]));
        let s = snapshot();
        assert_eq!(s["effective"]["VDBB_TEST_PROTOCOL"], false);
        assert_eq!(s["env"]["VDBB_TEST_PROTOCOL"], " resp3 ");
        assert_eq!(s["overridden"][0]["declared"], " resp3 ");
        assert_eq!(s["overridden"][0]["effective"], false);
    }

    /// `env_present` covers the extreme case: the value is discarded, so
    /// `WEAVIATE_USE_GRAPHQL=false` still selects GraphQL and the artifact must
    /// say which transport ran, not which text was set.
    #[test]
    fn env_present_records_the_switch_not_the_text() {
        let _l = test_lock();
        let _g = EnvGuard::set("VDBB_TEST_PRESENT", Some("false"));
        reset();
        assert!(env_present("VDBB_TEST_PRESENT"));
        let s = snapshot();
        assert_eq!(s["effective"]["VDBB_TEST_PRESENT"], true);
        assert_eq!(s["env"]["VDBB_TEST_PRESENT"], "false");
    }

    #[test]
    fn env_opt_records_null_for_a_blank_setting() {
        let _l = test_lock();
        let _g = EnvGuard::set("VDBB_TEST_OPT", Some("   "));
        reset();
        assert_eq!(env_opt("VDBB_TEST_OPT"), None);
        assert_eq!(snapshot()["effective"]["VDBB_TEST_OPT"], Value::Null);
    }

    /// The trim rider — the one behaviour change in the module, previously
    /// untested.
    #[test]
    fn env_parsed_trims_before_parsing_and_keeps_the_raw_text() {
        let _l = test_lock();
        let _g = EnvGuard::set("VDBB_TEST_PORT", Some(" 6380 "));
        reset();
        let port: u16 = env_parsed("VDBB_TEST_PORT", 6379);
        assert_eq!(port, 6380, "the old idiom fell back to the default here");
        let s = snapshot();
        assert_eq!(s["effective"]["VDBB_TEST_PORT"], 6380);
        // Nothing is hidden: the untrimmed text survives verbatim.
        assert_eq!(s["env"]["VDBB_TEST_PORT"], " 6380 ");
        assert!(s["overridden"].as_array().unwrap().is_empty());
    }

    #[test]
    fn phase_and_invocation_reach_the_snapshot() {
        let mut r = Recorder::new();
        r.set_phase("upload");
        r.set_invocation(json!({"host": "db-7", "skip_upload": true}));
        let s = r.snapshot();
        assert_eq!(s["phase"], "upload");
        assert_eq!(s["invocation"]["host"], "db-7");
        assert_eq!(s["invocation"]["skip_upload"], true);
    }

    /// `[]` under a bare `ignored_declared_keys` reads as "every declared key
    /// was honoured", which this cannot establish. The limit ships inside the
    /// artifact.
    #[test]
    fn ignored_keys_declare_their_own_incompleteness() {
        let s = Recorder::new().snapshot();
        assert_eq!(s["ignored_declared_keys"]["exhaustive"], false);
        assert!(s["ignored_declared_keys"]["covers"].is_string());
        assert_eq!(s["ignored_declared_keys"]["keys"], json!([]));
    }

    /// Drives the layer that actually writes the file — `set_declared` taking an
    /// [`crate::config::EngineConfig`] — not `Recorder::set_declared`, where
    /// "verbatim" is trivially true because the caller hands over the JSON.
    ///
    /// The three losses this pins were all real: serde injects `null` for every
    /// undeclared `Option`, the aliases rewrite `{"m","ef_construct"}` as
    /// `{"M","EF_CONSTRUCTION"}`, and `IndexOptions` has no catch-all so
    /// `type` / `confidence_interval` — both genuine Elasticsearch keys —
    /// vanished from the record of what was declared.
    #[test]
    fn declared_block_is_the_configuration_file_text_not_a_serde_round_trip() {
        let _l = test_lock();
        let raw = json!({
            "name": "es-verbatim",
            "engine": "elasticsearch",
            "collection_params": {
                "hnsw_config": {"m": 64, "ef_construct": 512},
                "index_options": {
                    "m": 24,
                    "ef_construction": 200,
                    "type": "int8_hnsw",
                    "confidence_interval": 0.9
                }
            },
            "upload_params": {"parallel": 8}
        });
        let mut cfg: crate::config::EngineConfig = serde_json::from_value(raw.clone()).unwrap();
        cfg.raw = Some(raw);

        reset();
        set_declared(&cfg);
        let d = &snapshot()["declared"];

        // Casing preserved as written, no alias normalisation.
        let hnsw = &d["collection_params"]["hnsw_config"];
        assert_eq!(hnsw["m"], 64);
        assert_eq!(hnsw["ef_construct"], 512);
        assert!(hnsw.get("M").is_none(), "alias normalised the key: {hnsw}");
        // No injected nulls for fields nobody declared.
        assert!(
            hnsw.get("on_disk").is_none(),
            "serde injected a null: {hnsw}"
        );
        assert_eq!(hnsw.as_object().unwrap().len(), 2);
        // Keys the typed struct has no field for survive.
        let io = &d["collection_params"]["index_options"];
        assert_eq!(io["type"], "int8_hnsw");
        assert_eq!(io["confidence_interval"], 0.9);
        assert_eq!(d["upload_params"]["parallel"], 8);
    }

    /// The other half of the same claim: `declared` never leaks into
    /// `effective`. A block that quietly used the declaration as the outcome is
    /// the failure this whole module exists to prevent.
    #[test]
    fn declared_never_becomes_effective() {
        let mut r = Recorder::new();
        r.set_declared(Some(&json!({"hnsw_config": {"M": 64}})), None);
        r.record_effective("m", json!(16));
        let s = r.snapshot();
        assert_eq!(s["declared"]["collection_params"]["hnsw_config"]["M"], 64);
        assert_eq!(s["effective"]["m"], 16);
    }

    /// The flagship demonstration, in **both** directions — neither was tested.
    ///
    /// Elasticsearch pins `number_of_shards` in code and ignores the config, so
    /// a config declaring 3 must produce an artifact recording declared=3 AND
    /// effective=1. A config that declares nothing must NOT invent an override:
    /// a spurious "we overrode you" is its own false claim.
    #[test]
    fn elasticsearch_pinned_shard_count_is_recorded_as_an_override() {
        let _l = test_lock();
        let with_declaration = json!({
            "name": "es-3-shards",
            "engine": "elasticsearch",
            "collection_params": {"number_of_shards": 3},
        });
        let mut cfg: crate::config::EngineConfig =
            serde_json::from_value(with_declaration.clone()).unwrap();
        cfg.raw = Some(with_declaration);

        reset();
        set_declared(&cfg);
        let shards = crate::engine::resolved_number_of_shards(&cfg).unwrap();
        assert_eq!(shards, Some(Value::from(1)));

        let s = snapshot();
        let o = s["overridden"]
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["key"] == "collection_params.number_of_shards")
            .expect("the pinned shard count must be recorded as an override");
        assert_eq!(o["declared"], 3);
        assert_eq!(o["effective"], 1);
        assert!(o["reason"].as_str().unwrap().contains("pins"));
    }

    #[test]
    fn elasticsearch_without_a_declared_shard_count_invents_no_override() {
        let _l = test_lock();
        let silent = json!({"name": "es-default", "engine": "elasticsearch"});
        let mut cfg: crate::config::EngineConfig = serde_json::from_value(silent.clone()).unwrap();
        cfg.raw = Some(silent);

        reset();
        set_declared(&cfg);
        assert_eq!(
            crate::engine::resolved_number_of_shards(&cfg).unwrap(),
            Some(Value::from(1))
        );
        assert!(
            snapshot()["overridden"].as_array().unwrap().is_empty(),
            "nothing was declared, so nothing was overridden"
        );
    }

    /// Wiring the CI-asserted `KNOWN_UNREAD` inventory: the guard used to say
    /// "declares a knob it never reads, documented debt (#245)" while the
    /// artifact for the same config said nothing at all.
    #[test]
    fn known_unread_debt_reaches_the_artifact() {
        let _l = test_lock();
        let raw = json!({
            "name": "es-debt",
            "engine": "elasticsearch",
            "connection_params": {"request_timeout": 10000},
            "collection_params": {"index_options": {"m": 16, "ef_construction": 100}}
        });
        let mut cfg: crate::config::EngineConfig = serde_json::from_value(raw.clone()).unwrap();
        cfg.raw = Some(raw);

        reset();
        set_declared(&cfg);
        let s = snapshot();
        let keys = s["ignored_declared_keys"]["keys"].as_array().unwrap();
        assert!(
            keys.iter()
                .any(|k| k == "connection_params.request_timeout"),
            "{keys:?}"
        );
        let o = &s["overridden"][0];
        assert_eq!(o["key"], "connection_params.request_timeout");
        assert_eq!(o["declared"], 10000);
        assert!(o["reason"].as_str().unwrap().contains("#245"));
    }

    #[test]
    fn ignored_keys_are_sorted_and_deduplicated() {
        let mut r = Recorder::new();
        r.note_ignored("b");
        r.note_ignored("a");
        r.note_ignored("b");
        assert_eq!(
            r.snapshot()["ignored_declared_keys"]["keys"],
            json!(["a", "b"])
        );
    }
}
