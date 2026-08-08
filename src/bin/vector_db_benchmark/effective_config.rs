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

/// Placeholder for a credential the run used but which came from a built-in
/// DEFAULT, not from the environment.
///
/// `<redacted:set>` next to `env: null` in the same block asserted "set" for a
/// variable the block had just said was unset — the self-contradiction class
/// this module exists to remove.
const REDACTED_DEFAULT: &str = "<redacted:default>";

/// Substrings that make a KEY's value a credential.
///
/// Matched case-insensitively against the key, and against every *ancestor* key
/// on the path to a value (see [`scrub_value`]), so these cover configuration
/// file keys like `api_key` and `auth_token` as well as environment variable
/// names. Bias is deliberately towards over-matching: a redacted knob costs
/// provenance, a published one costs a rotation.
///
/// Deliberately not bare `KEY` — `REDIS_KEY_PREFIX` and `MILVUS_COLLECTION_NAME`
/// are benign and their values are needed to tell two runs apart.
const SECRET_MARKERS: &[&str] = &[
    "PASSWORD",
    "PASSWD",
    // Bare `PASS` also catches `REDIS_PASS` and `pass`.
    "PASS",
    // ODBC/JDBC spell it `Pwd=`; `password`/`passwd`/`PGPASSWORD` all matched
    // and this one did not.
    "PWD",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "SIGNING_KEY",
    // Bare `TOKEN`/`AUTH` catch `ACCESS_TOKEN`, `AUTH_TOKEN`, `AUTHORIZATION`,
    // `AWS_SESSION_TOKEN`, `REDIS_AUTH`. Note `"AUTH_TOKEN".contains("_AUTH")`
    // is FALSE, which is how the leading-underscore forms missed them.
    "TOKEN",
    "AUTH",
    "SECRET",
    "CREDENTIAL",
    "SESSION",
];

/// True when `name`'s value must never reach an artifact.
pub fn is_secret(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    SECRET_MARKERS.iter().any(|m| upper.contains(m))
}

/// Strip `user:password@` from a URL-shaped value.
///
/// A connection string routinely carries a credential inline under a key that
/// is not credential-named (`REDIS_URI`, `--host`, a config file's `endpoint`).
/// Blanking the userinfo keeps the host/port — provenance worth having —
/// without publishing the secret.
///
/// Two passes, because a redactor must fail safe:
///
/// 1. RFC 3986: the authority ends at the first `/`, `?` or `#`; an `@` inside
///    it delimits userinfo.
/// 2. If that finds nothing but an `@` appears later, the value is not a
///    well-formed URL — and computing the authority boundary *first* is exactly
///    how `redis://admin:hunt/er2@10.0.0.5:6379/0` used to be returned
///    untouched, password and all. When the text before the last `@` contains a
///    `:` it looks like `user:password`, so it is redacted anyway. A path that
///    merely contains an `@` (`http://h/p@th`) has no colon and is left alone.
fn strip_userinfo(value: &str) -> String {
    // Scheme-less authority: `--host 'default:pw@127.0.0.1'` is exactly how the
    // Redis-wire engines take a password, and `mongodb_engine` short-circuits on
    // a `mongodb`-prefixed host where `user:pass@` is the convention. No `://`
    // is required for the value to be a credential.
    let (scheme, rest) = match value.find("://") {
        Some(sep) => value.split_at(sep + 3),
        None => ("", value),
    };

    let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(auth_end);
    if let Some(at) = authority.rfind('@') {
        return format!("{scheme}{REDACTED}@{}{tail}", &authority[at + 1..]);
    }

    // Malformed-but-dangerous fallback.
    match rest.rfind('@') {
        Some(at) if rest[..at].contains(':') => {
            format!("{scheme}{REDACTED}@{}", &rest[at + 1..])
        }
        _ => value.to_string(),
    }
}

/// Blank the value of any `key=value` credential embedded in an opaque string.
///
/// Name-based redaction cannot see these: the secret sits inside the value of a
/// key that is not itself credential-named. Three families, all live shapes:
///
/// * libpq DSNs — `host=db user=bench password=hunter2 sslmode=require`
/// * URL query strings — `mongodb://h/db?w=majority&password=hunter2`
/// * ODBC/JDBC — `Server=h;Database=d;Uid=u;Pwd=hunter2;`
///
/// The query-string family is why this is not a whitespace splitter. It used to
/// be, so it read the key of `mongodb://h/db?w=majority&password=X` as
/// `mongodb://h/db?w` — not credential-named — and copied the password through.
/// `?authSource=admin&password=X` *looked* fixed only because the first key
/// contains `AUTH` and blanked the whole tail; any benign leading option
/// (`w=`, `retryWrites=`, `ssl=`, `db=`) restored the leak.
///
/// Separators are whitespace, `&`, `;` and `?`. Spaces are permitted around the
/// `=` (PostgreSQL documents them as optional). Values may be quoted with `'`
/// or `"`, honouring both backslash escapes and SQL's doubled-quote escape.
/// Non-secret pairs are reproduced byte for byte, spacing and quoting included.
fn strip_dsn_secrets(value: &str) -> String {
    if !value.contains('=') {
        return value.to_string();
    }
    let c: Vec<char> = value.chars().collect();
    let is_sep = |ch: char| ch.is_whitespace() || ch == '&' || ch == ';' || ch == '?';
    let take = |from: usize, to: usize| -> String { c[from..to].iter().collect() };

    let mut out = String::with_capacity(value.len());
    let mut i = 0usize;
    while i < c.len() {
        if is_sep(c[i]) {
            out.push(c[i]);
            i += 1;
            continue;
        }
        // Key: up to `=` or a separator.
        let key_start = i;
        while i < c.len() && c[i] != '=' && !is_sep(c[i]) {
            i += 1;
        }
        let key = take(key_start, i);

        // Permit spaces between the key and the `=`.
        let mut probe = i;
        while probe < c.len() && (c[probe] == ' ' || c[probe] == '\t') {
            probe += 1;
        }
        if probe >= c.len() || c[probe] != '=' {
            out.push_str(&key);
            continue; // not a pair; the separator loop handles what follows
        }
        probe += 1; // consume '='
                    // …and after it.
        while probe < c.len() && (c[probe] == ' ' || c[probe] == '\t') {
            probe += 1;
        }
        // Everything from the key through the `=` and its padding, verbatim.
        let prefix = take(key_start, probe);
        i = probe;

        // Value: a quoted run, or a bare run up to the next separator.
        let val_start = i;
        if i < c.len() && (c[i] == '\'' || c[i] == '"') {
            let quote = c[i];
            i += 1;
            while i < c.len() {
                if c[i] == '\\' && i + 1 < c.len() {
                    i += 2;
                    continue;
                }
                if c[i] == quote {
                    // SQL doubles a quote to escape it: 'ab''cd' is one value.
                    if i + 1 < c.len() && c[i + 1] == quote {
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                i += 1;
            }
        } else {
            while i < c.len() && !is_sep(c[i]) {
                i += 1;
            }
        }
        let raw = take(val_start, i);
        let bare = raw.trim_matches(['\'', '"']);
        if is_secret(key.trim()) && !bare.trim().is_empty() {
            out.push_str(&prefix);
            out.push_str(REDACTED);
        } else {
            out.push_str(&prefix);
            out.push_str(&raw);
        }
    }
    out
}

/// Fields that a sibling `key` must NOT make secret: they carry the knob's name
/// and the human explanation, not its value. Redacting them turns an
/// `overridden` entry into four identical placeholders that record nothing.
const ENTRY_KEY_EXEMPT: &[&str] = &["key", "reason"];

/// Scrub one JSON value for publication.
///
/// `secret_ancestor` is true when any key on the path to this value is
/// credential-named, which is what makes the rule total: a secret nested inside
/// an object (`record_effective("ELASTIC_PASSWORD", json!({"value": s}))`) is
/// caught just like a bare string, and so is every leaf of a raw configuration
/// file block copied in wholesale.
fn scrub_value(value: &Value, secret_ancestor: bool) -> Value {
    match value {
        // `null` under a secret key means "read and unset" — a fact, not a leak.
        Value::Null => Value::Null,
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| {
                    // An `overridden` entry names its knob in a sibling `key`
                    // field rather than in its own JSON key, so `declared` /
                    // `effective` there would otherwise look benign.
                    //
                    // `key` and `reason` are exempt from that seeding: they hold
                    // the knob's NAME and a prose explanation, never its value.
                    // Without the exemption an override on a credential knob
                    // rendered as four `<redacted:set>`s and recorded nothing at
                    // all — the entry destroying the very thing it exists to
                    // produce. Same for any `{"key":…, "value":…}` pair, which
                    // lost the name alongside the secret.
                    let seeded = !ENTRY_KEY_EXEMPT.contains(&k.as_str())
                        && map
                            .get("key")
                            .and_then(Value::as_str)
                            .is_some_and(is_secret);
                    (
                        k.clone(),
                        scrub_value(v, secret_ancestor || is_secret(k) || seeded),
                    )
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|v| scrub_value(v, secret_ancestor))
                .collect(),
        ),
        Value::String(s) if secret_ancestor && s == REDACTED_DEFAULT => {
            Value::from(REDACTED_DEFAULT)
        }
        // NOTE: this changes a JSON *type* — `session_pool_size: 8` becomes the
        // string `"<redacted:set>"`. A scan of all 31 shipped configs and all
        // 104 recorded knob names found zero false positives today, so this is
        // a future-drift risk rather than a live defect; the direction of the
        // bias (over-redact) is deliberate. Tracked with the marker widening.
        _ if secret_ancestor => Value::from(REDACTED),
        Value::String(s) => Value::from(strip_dsn_secrets(&strip_userinfo(s))),
        other => other.clone(),
    }
}

/// Redact a raw environment value for publication.
fn redact(name: &str, raw: &str) -> Value {
    scrub_value(&Value::from(raw), is_secret(name))
}

/// Assert that no credential-named key anywhere in `value` carries anything but
/// [`REDACTED`]. The mirror image of [`scrub_value`], deliberately written as a
/// separate walk: if the two ever disagree the assertion is the one that fires,
/// and it fires in release builds too (`assert!`, under no `cfg`).
fn assert_no_cleartext_secrets(value: &Value, secret_ancestor: bool, path: &str) {
    match value {
        Value::Null => {}
        Value::Object(map) => {
            let entry_key = map
                .get("key")
                .and_then(Value::as_str)
                .is_some_and(is_secret);
            for (k, v) in map {
                // Must mirror `scrub_value`'s exemption exactly, or the two
                // disagree and the assertion panics on a correctly-scrubbed
                // document (it did: it REQUIRED `overridden[0].key` to be
                // blanked, so the two had to be fixed together).
                let seeded = entry_key && !ENTRY_KEY_EXEMPT.contains(&k.as_str());
                assert_no_cleartext_secrets(
                    v,
                    secret_ancestor || is_secret(k) || seeded,
                    &format!("{path}.{k}"),
                );
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                assert_no_cleartext_secrets(v, secret_ancestor, &format!("{path}[{i}]"));
            }
        }
        leaf => assert!(
            !secret_ancestor || matches!(leaf.as_str(), Some(REDACTED) | Some(REDACTED_DEFAULT)),
            "refusing to publish `{path}` in the clear: a credential-named key \
             reached the artifact carrying {leaf}. Every value under such a key \
             must go through `scrub_value`."
        ),
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
        // `scrub_value`, not a string-only check: a credential nested inside an
        // object under a secret-named key is still a credential.
        self.effective
            .insert(key.to_string(), scrub_value(&value, is_secret(key)));
    }

    /// Insert a value that is ALREADY a safe placeholder, bypassing the scrub
    /// that would rewrite `<redacted:default>` into `<redacted:set>`.
    fn record_effective_raw(&mut self, key: &str, value: Value) {
        self.effective.insert(key.to_string(), value);
    }

    /// Record that `declared` lost to `effective`, and why.
    pub fn note_override(&mut self, key: &str, declared: Value, effective: Value, reason: &str) {
        let secret = is_secret(key);
        let entry = json!({
            "key": key,
            "declared": scrub_value(&declared, secret),
            "effective": scrub_value(&effective, secret),
            "reason": reason,
        });
        if !self.overridden.contains(&entry) {
            self.overridden.push(entry);
        }
    }

    /// Clear every field, so the next experiment starts from nothing.
    ///
    /// `declared`, `invocation` and `phase` are overwritten each experiment and
    /// self-heal; `env`, `effective`, `overridden` and `ignored` only ACCUMULATE
    /// and are cleared here and nowhere else. That asymmetry is why dropping the
    /// clear was caught by only one test, and flakily: it is the single point of
    /// failure for four of the seven fields.
    pub fn begin(&mut self) {
        *self = Recorder::new();
    }

    /// Record a declared configuration key that is known not to be read.
    pub fn note_ignored(&mut self, key: &str) {
        self.ignored.insert(key.to_string());
    }

    /// The `engine_params` block for an artifact.
    ///
    /// Everything the block will publish is passed through [`scrub_value`] as
    /// one tree, then re-walked to assert the invariant held.
    ///
    /// The previous version enumerated two flat maps (`env`, `effective`) while
    /// `snapshot` emitted five value-bearing members. The three it never
    /// inspected — `invocation`, `declared`, `overridden` — were exactly the
    /// three this block added, and two of them leaked live: `--host
    /// 'user:pw@127.0.0.1'` and a configuration file's `api_key` were published
    /// verbatim, three keys from a `<redacted:set>`. Scrubbing the assembled
    /// document instead of named maps is what makes "cannot miss a map somebody
    /// adds later" true rather than aspirational.
    ///
    /// # Panics
    ///
    /// If a credential-named key would be published in the clear. These files
    /// get pasted into issues; aborting the run is the correct response to
    /// discovering we are about to leak.
    pub fn snapshot(&self) -> Value {
        let doc = self.assemble();
        let scrubbed = scrub_value(&doc, false);
        assert_no_cleartext_secrets(&scrubbed, false, "engine_params");
        scrubbed
    }

    fn assemble(&self) -> Value {
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
                // Key NAMES only, never values — the one snapshot member the
                // redaction assertion has nothing to check.
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
    recorder().begin();
}

/// Begin recording one experiment: clear the previous configuration's state,
/// then stash this one's declared configuration and invocation.
///
/// **One entry point on purpose.** These three steps used to be three separate
/// calls at the top of the experiment loop, and a mutation campaign showed that
/// deleting the `reset()` line left the whole suite green while a sweep's later
/// configurations silently inherited the earlier ones' knobs.
///
/// To be precise about what this does and does not buy: deleting the `reset()`
/// call BELOW still compiles. It is caught by a test
/// (`begin_experiment_clears_the_previous_experiments_accumulation`), not by the
/// type system. What the [`Recording`] token makes inexpressible is the three
/// CALL-SITE mutations — commenting the call out, calling it conditionally, and
/// hoisting it out of the per-experiment loop.
pub fn begin_experiment(
    engine_config: &crate::config::EngineConfig,
    invocation: Value,
) -> Recording {
    reset();
    set_declared(engine_config);
    set_invocation(invocation);
    Recording(())
}

/// Proof that a recording was started for the experiment about to run.
///
/// `run_single_experiment` **consumes** one of these. That is the whole point,
/// and it is load-bearing: a source-scanning guard could be satisfied by a
/// commented-out call, and was — `// TODO(#212): re-enable begin_experiment(…)`
/// passed 776/776 while every artifact shipped `declared: null`. So did
/// `if engine_idx == 0 { begin_experiment(…) }`, which is worse than the bug
/// #212 fixed: the second config then publishes the FIRST config's `declared`
/// block and `effective` knobs, so the artifact affirmatively asserts a
/// configuration that never ran where master merely said nothing.
///
/// Because the token is neither `Copy` nor `Clone` and is taken by value:
///
/// * commenting the call out  -> the binding does not exist, does not compile;
/// * calling it conditionally -> the binding is not in scope, does not compile;
/// * hoisting it out of the loop so it runs once per *config* instead of once
///   per (config, dataset) -> moved on the first iteration, does not compile.
///
/// None of those three is expressible now. That is what the guard could not do.
///
/// It proves a recording BEGAN, not that it began for THIS (config, dataset)
/// pair: handing `run_single_experiment` a token minted for a different
/// experiment compiles and passes. Making it prove identity would mean carrying
/// the config name in the token and checking it at the far end; the ordering
/// half of that is covered by guard 3 instead.
#[must_use = "run_single_experiment consumes this; dropping it means the \
              experiment ran without its provenance being started"]
pub struct Recording(());

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
    let was_set = raw.is_some();
    let value = raw.unwrap_or_else(|| default.to_string());
    // Redaction is applied by `record_effective` itself; passing the plaintext
    // is correct and is what every other caller does. The one thing it cannot
    // know is whether the value came from the environment or from the built-in
    // default, which is the difference between `<redacted:set>` and
    // `<redacted:default>`.
    if is_secret(name) && !was_set {
        rec.record_effective_raw(name, Value::from(REDACTED_DEFAULT));
    } else {
        rec.record_effective(name, Value::from(value.clone()));
    }
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
    const EXEMPT: &[&str] = &["src/bin/vector_db_benchmark/effective_config.rs"];

    /// Individual reads that are allowed to stay raw, as (file, snippet, why).
    /// Scoped to the exact call rather than exempting a whole file, and asserted
    /// to still be present so a removed one cannot leave a stale excuse.
    const EXEMPT_CALLS: &[(&str, &str, &str)] = &[
        (
            "src/config.rs",
            "env::var(\"REDIS_PORT\")",
            "RedisConfig::from_env is a pub constructor whose only callers are \
             src/redisearch/ and src/vectorsets/, neither of which lib.rs \
             declares — dead code, tracked for deletion in #275",
        ),
        (
            "src/config.rs",
            "env::var(\"REDIS_AUTH\")",
            "as REDIS_PORT above (#275)",
        ),
        (
            "src/config.rs",
            "env::var(\"REDIS_USER\")",
            "as REDIS_PORT above (#275)",
        ),
        (
            "src/config.rs",
            "env::var(\"REDIS_CLUSTER\")",
            "as REDIS_PORT above (#275); also documented in v0/DOCKER_README.md \
             and read by nothing live",
        ),
        (
            "src/bin/vector_db_benchmark/config.rs",
            "env::current_dir()",
            "project_root() resolves the configurations and results directories \
             from the cwd. Both RESOLVED paths are recorded per run in \
             `invocation.configurations_dir` / `invocation.results_dir` — the \
             facts that matter, since the first decides which configs the sweep \
             globs. Asserted by \
             `the_cited_compensating_record_for_current_dir_exists`",
        ),
        (
            "src/bin/vector_db_benchmark/download.rs",
            "env::temp_dir()",
            "staging directory for a dataset download; affects where bytes land \
             on disk, not what is measured",
        ),
    ];

    /// Directories on disk that no crate root declares, so nothing in them is
    /// compiled and nothing in them can read the environment at runtime.
    const UNCOMPILED: &[&str] = &["src/redisearch/", "src/vectorsets/"];

    /// Environment variables still read through the plain [`super::env_var`]
    /// shim, which records the raw text and nothing else.
    ///
    /// Asserted in BOTH directions, exactly like `config::KNOWN_UNREAD`: every
    /// LITERAL-named `env_var` site must be listed here, and every entry here
    /// must still be an `env_var` site.
    ///
    /// Exactly one site takes a non-literal name — `opensearch::parse_env_secs`,
    /// called with `OPENSEARCH_FORCE_MERGE_TIMEOUT` and
    /// `..._FORCE_MERGE_BUDGET`. It is NOT listed, and does not need to be: it
    /// records the resolved value with `record_effective`, so those two knobs
    /// are tier 1, not tier 2. `NON_LITERAL_ENV_VAR_SITES` below pins that count
    /// so a second dynamic site cannot appear unnoticed. (An earlier version of
    /// this comment claimed two `format!`-built sites covered by a base
    /// variable; there is one, it uses no `format!`, and it has no base
    /// variable.) Migrating one to `env_parsed`/`env_flag`/`env_opt`
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

    /// `env_var` call sites whose variable name is not a literal, as
    /// (file, function, why it is not in `KNOWN_UNRECORDED`).
    const NON_LITERAL_ENV_VAR_SITES: &[(&str, &str, &str)] = &[(
        "src/bin/vector_db_benchmark/engine/opensearch.rs",
        "parse_env_secs",
        "records the parsed value with `record_effective`, so its two knobs are \
         tier 1; the raw-text-only contract of KNOWN_UNRECORDED does not apply",
    )];

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
    /// not mistaken for a production read.
    ///
    /// Indentation-based, NOT brace-counting. Brace counting was wrong twice
    /// over: a `#[cfg(test)]` item with no brace of its own (a `const`) sent it
    /// hunting for the next `{` — the following function's body — deleting the
    /// production code in between; and its stated invariant ("string literals
    /// here contain none") is false in at least five files (`start_gate.rs`,
    /// `npy_reader.rs`, bin `config.rs`, `redis.rs`, `valkey.rs` all contain
    /// `'{'` or `"{{"` literals), so the count desynchronised, the loop broke,
    /// and whole test modules were scanned as production.
    ///
    /// rustfmt guarantees a module's closing `}` sits at the module's own
    /// indentation, which is a far cheaper invariant than balanced braces and
    /// does not care what is inside string literals.
    fn strip_test_modules(src: &str) -> String {
        // Comments first: a doc comment *mentioning* `#[cfg(test)] const` is
        // prose, not an item, and matching it truncated the scan.
        let src = strip_line_comments(src);
        let lines: Vec<&str> = src.lines().collect();
        let mut out: Vec<&str> = Vec::with_capacity(lines.len());
        let mut i = 0usize;
        while i < lines.len() {
            let trimmed = lines[i].trim_start();
            if trimmed.starts_with("#[cfg(test)]") {
                let indent = lines[i].len() - trimmed.len();
                let mut j = i + 1;
                while j < lines.len() && lines[j].trim().is_empty() {
                    j += 1;
                }
                let head = lines.get(j).map(|l| l.trim_start()).unwrap_or("");
                let is_mod = head.starts_with("mod ") || head.starts_with("pub mod ");
                // A BODILESS declaration (`#[cfg(test)] mod filter_guard;`) has
                // no block to strip. Treating it as one sent the scan hunting
                // for the next `}` at this indent and deleted the 48 production
                // lines in between — the `#[cfg(test)] const` bug in a different
                // item shape. `engine/mod.rs:12` is exactly this.
                if is_mod && !head.trim_end().ends_with(';') {
                    let closer = format!("{}}}", " ".repeat(indent));
                    let mut k = j + 1;
                    // `trim_end`: `strip_line_comments` turns
                    // `} // end of the config tests` into `"} "`, which never
                    // equalled the closer, so `k` ran to EOF and silently
                    // disabled the guard for the rest of the file.
                    while k < lines.len() && lines[k].trim_end() != closer {
                        k += 1;
                    }
                    if k >= lines.len() {
                        // No closer found: emit the lines rather than deleting
                        // to EOF. Over-scanning shows up as a false positive
                        // somebody can see; under-scanning is invisible.
                        out.push(lines[i]);
                        i += 1;
                        continue;
                    }
                    i = k + 1;
                    continue;
                }
                // Any other `#[cfg(test)]` item — a `const`, a `use`, a
                // bodiless `mod X;` — stays in the scanned text. A false
                // positive there is visible and fixable; a deletion is neither.
            }
            out.push(lines[i]);
            i += 1;
        }
        out.join("\n")
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
            for probe in [
                "env::var(",
                "env::var_os(",
                "env::vars(",
                "env::vars_os(",
                "env::temp_dir(",
                "env::current_dir(",
            ] {
                let mut from = 0usize;
                while let Some(i) = production[from..].find(probe) {
                    let at = from + i;
                    from = at + probe.len();
                    // A scoped exemption must match the exact call text.
                    let tail = &production[at..(at + 64).min(production.len())];
                    if EXEMPT_CALLS
                        .iter()
                        .any(|(f, snippet, _)| *f == path && tail.starts_with(snippet))
                    {
                        continue;
                    }
                    offenders.push(format!("{path}: {probe}"));
                }
            }
            // Aliasing defeats a textual probe outright.
            for alias in ["use std::env::var", "use std::env::vars"] {
                if production.contains(alias) {
                    offenders.push(format!("{path}: {alias} (aliased import)"));
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

    /// Remove `//` line comments so a commented-out call cannot satisfy a
    /// source scan. `// TODO(#212): re-enable begin_experiment(…)` passed the
    /// unstripped version while every artifact shipped `declared: null`.
    fn strip_line_comments(src: &str) -> String {
        src.lines()
            .map(|l| match l.find("//") {
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// GUARD 3 — a cheap backstop; the real protection is the type.
    ///
    /// `effective_config::Recording` is move-only and consumed by
    /// `run_single_experiment`, so commenting the call out, calling it
    /// conditionally, or hoisting it out of the per-experiment loop are all
    /// compile errors (verified). This scan only adds the ORDERING check the
    /// type cannot express: the recording must begin before the engine is built,
    /// because engines resolve most of their environment knobs in `new()`.
    #[test]
    fn the_experiment_loop_begins_a_recording_before_building_the_engine() {
        let src =
            std::fs::read_to_string(repo_root().join("src/bin/vector_db_benchmark/experiment.rs"))
                .expect("experiment.rs");
        let production = strip_line_comments(&strip_test_modules(&src));
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

    /// GUARD 1b — the scoped exemptions are real.
    ///
    /// `EXEMPT_CALLS` was documented as "asserted to still be present so a
    /// removed one cannot leave a stale excuse" and was not: it was only a
    /// skip-list, so unlike `KNOWN_UNRECORDED` it could rot silently. Now it
    /// matches its own docstring.
    #[test]
    fn every_scoped_exemption_still_names_a_real_call_site() {
        let sources = rust_sources();
        let mut stale = Vec::new();
        for (file, snippet, _why) in EXEMPT_CALLS {
            // Stripped production text, not raw source: an exempted call that
            // migrated into a `#[cfg(test)]` module would otherwise keep its
            // excuse alive while excusing nothing.
            let found = sources
                .iter()
                .any(|(path, src)| path == file && strip_test_modules(src).contains(snippet));
            if !found {
                stale.push(format!("{file}: {snippet}"));
            }
        }
        assert!(
            stale.is_empty(),
            "EXEMPT_CALLS excuses reads that no longer exist — delete their rows \
             so the list cannot become a stale excuse:\n  {}",
            stale.join("\n  ")
        );
    }

    /// A scoped exemption may cite a compensating record. If it does, that
    /// record has to exist.
    ///
    /// `every_scoped_exemption_still_names_a_real_call_site` only checks that
    /// the excused CALL is still there — so the `env::current_dir()` waiver
    /// could name `invocation.configurations_dir`, a field that was never
    /// built, and stay green. That is exactly the stale excuse these inventories
    /// exist to prevent.
    #[test]
    fn the_cited_compensating_record_for_current_dir_exists() {
        let src =
            std::fs::read_to_string(repo_root().join("src/bin/vector_db_benchmark/experiment.rs"))
                .expect("experiment.rs");
        let production = strip_line_comments(&strip_test_modules(&src));
        for field in ["configurations_dir", "results_dir"] {
            assert!(
                production.contains(&format!("\"{field}\"")),
                "the `env::current_dir()` exemption cites `invocation.{field}`, \
                 which `invocation_provenance` does not emit"
            );
        }
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
            // Match a bare `env_var(` too: an `use crate::effective_config::env_var;`
            // would otherwise hide a site from this guard entirely.
            let mut rest = production.as_str();
            while let Some(i) = rest.find("env_var(") {
                rest = &rest[i + "env_var(".len()..];
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
        // A non-literal `env_var(some_var)` is invisible to the name matching
        // above, so it would evade the bidirectional assertion entirely. Pin the
        // count: a second dynamic site has to be justified in
        // NON_LITERAL_ENV_VAR_SITES before it can appear.
        let mut dynamic = Vec::new();
        for (path, src) in rust_sources() {
            if path.ends_with("effective_config.rs") {
                continue;
            }
            let production = strip_test_modules(&src);
            let mut rest = production.as_str();
            while let Some(i) = rest.find("env_var(") {
                rest = &rest[i + "env_var(".len()..];
                if !rest.starts_with('"') && !rest.starts_with('&') {
                    dynamic.push(path.clone());
                }
            }
        }
        dynamic.sort();
        dynamic.dedup();
        let documented: Vec<String> = NON_LITERAL_ENV_VAR_SITES
            .iter()
            .map(|(f, _, _)| f.to_string())
            .collect();
        assert_eq!(
            dynamic, documented,
            "the set of `env_var` sites taking a NON-LITERAL name changed. Such a \
             site cannot be name-matched, so it must be justified in \
             NON_LITERAL_ENV_VAR_SITES (or given a literal name)"
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

    /// A route that bypasses `record_effective` entirely is repaired by
    /// `snapshot`, not merely detected by it: the scrub runs over the assembled
    /// document, so the write path is no longer trusted at all.
    #[test]
    fn snapshot_scrubs_a_credential_that_bypassed_the_write_path() {
        let mut r = Recorder::new();
        // Exactly what a careless future edit would do.
        r.effective
            .insert("MONGODB_PASSWORD".into(), json!("plaintext"));
        assert_eq!(r.snapshot()["effective"]["MONGODB_PASSWORD"], REDACTED);
    }

    /// The two live leaks the review found, at the layer that produced them.
    ///
    /// `snapshot()` emitted five value-bearing members while the assertion
    /// walked two flat maps. The three it never inspected — `invocation`,
    /// `declared`, `overridden` — were exactly the three this block added.
    #[test]
    fn every_member_of_the_snapshot_is_scrubbed_not_just_env_and_effective() {
        let mut r = Recorder::new();
        // BLOCKER 1: `--host 'user:pw@node'`, scheme-less, run succeeds.
        r.set_invocation(json!({"host": "default:HOST-CANARY@127.0.0.1"}));
        // BLOCKER 2: the raw config file copied in wholesale.
        r.set_declared(
            Some(&json!({
                "api_key": "DECLARED-CANARY-bbb",
                "endpoint": "https://svc:DECLARED-CANARY-ccc@vendor.cloud/v1",
                "nested": {"deep": {"auth_token": "DECLARED-CANARY-eee"}}
            })),
            Some(&json!({"parallel": 8, "auth_token": "DECLARED-CANARY-ddd"})),
        );
        // Latent: an override whose secret-ness is named in a SIBLING field.
        r.note_override(
            "REDIS_AUTH",
            json!({"was": "OVERRIDE-CANARY-fff"}),
            json!("OVERRIDE-CANARY-ggg"),
            "rotated",
        );

        let dumped = serde_json::to_string(&r.snapshot()).unwrap();
        for canary in [
            "HOST-CANARY",
            "DECLARED-CANARY-bbb",
            "DECLARED-CANARY-ccc",
            "DECLARED-CANARY-ddd",
            "DECLARED-CANARY-eee",
            "OVERRIDE-CANARY-fff",
            "OVERRIDE-CANARY-ggg",
        ] {
            assert!(!dumped.contains(canary), "{canary} leaked: {dumped}");
        }
        // Provenance that is NOT a secret survives.
        let s = r.snapshot();
        assert!(s["invocation"]["host"]
            .as_str()
            .unwrap()
            .contains("127.0.0.1"));
        assert_eq!(s["declared"]["upload_params"]["parallel"], 8);
        assert!(s["declared"]["collection_params"]["endpoint"]
            .as_str()
            .unwrap()
            .contains("vendor.cloud"));
    }

    /// The assertion is a real backstop, not a restatement of the scrubber: it
    /// walks the assembled document independently and fires in release builds.
    #[test]
    #[should_panic(expected = "refusing to publish `engine_params.declared")]
    fn the_assertion_catches_a_secret_the_scrubber_would_have_missed() {
        let mut doc = json!({"declared": {"api_key": "leaked"}});
        // Simulate a future member that skipped `scrub_value`.
        doc["declared"]["api_key"] = json!("leaked");
        assert_no_cleartext_secrets(&doc, false, "engine_params");
    }

    /// Name-based redaction cannot see a secret INSIDE the value of a
    /// non-credential key. libpq DSNs are the live shape.
    #[test]
    fn dsn_embedded_password_is_blanked() {
        let mut r = Recorder::new();
        r.record_effective(
            "PGVECTOR_DSN",
            json!("host=db user=bench password=DSN-CANARY sslmode=require"),
        );
        let out = r.snapshot()["effective"]["PGVECTOR_DSN"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(!out.contains("DSN-CANARY"), "{out}");
        // The parts that are provenance survive.
        assert!(out.contains("host=db"), "{out}");
        assert!(out.contains("sslmode=require"), "{out}");
    }

    /// A quoted libpq value may contain spaces. Splitting on whitespace first —
    /// which is what this function did when it was added — redacted up to the
    /// first space and published the rest of the password verbatim.
    #[test]
    fn dsn_quoted_password_containing_spaces_is_fully_blanked() {
        for dsn in [
            "host=db password='LEAKONE LEAKTWO LEAKTHREE' sslmode=require",
            "host=db password=\"LEAKONE LEAKTWO LEAKTHREE\" sslmode=require",
            "password='LEAKONE LEAKTWO LEAKTHREE'",
            "host=db password='LEAKONE LEAKTWO LEAKTHREE'",
        ] {
            let out = strip_dsn_secrets(dsn);
            for canary in ["LEAKONE", "LEAKTWO", "LEAKTHREE"] {
                assert!(!out.contains(canary), "{dsn} -> {out}");
            }
        }
        // The shapes that already worked must keep working.
        let out = strip_dsn_secrets("host=db  PASSWORD=bare\tapi_key=k sslmode=require");
        assert!(!out.contains("bare") && !out.contains("=k"), "{out}");
        assert!(out.contains("host=db"), "{out}");
        assert!(out.contains("sslmode=require"), "{out}");
        // A non-secret key keeps its quoted value intact.
        assert_eq!(
            strip_dsn_secrets("options='-c statement_timeout=5s'"),
            "options='-c statement_timeout=5s'"
        );
    }

    /// **B1 regression.** A password in a URL QUERY STRING, which is the other
    /// way `--host` carries one.
    ///
    /// `strip_userinfo` declines (no `@`) and the old whitespace splitter read
    /// the key as `mongodb://h.example/db?w` — not credential-named — so the
    /// password was copied through into `invocation.host` in all three file
    /// kinds.
    ///
    /// The `authSource` cases are load-bearing: they passed BEFORE the fix, by
    /// accident, because the first key contains `AUTH` and that blanked the
    /// whole tail. Every case below therefore also appears with a benign
    /// leading option, so the accidental masking cannot make a regression look
    /// green.
    #[test]
    fn query_string_password_is_blanked_whatever_precedes_it() {
        for value in [
            "mongodb://h.example/db?w=majority&password=CANARYQ",
            "mongodb://h.example/db?retryWrites=true&password=CANARYQ",
            "redis://h:6379/0?db=1&password=CANARYQ",
            "h.example:27017/?ssl=true&password=CANARYQ",
            "mongodb://h.example/db?authSource=admin&password=CANARYQ",
            "mongodb+srv://h/db?w=1&authToken=CANARYQ&ssl=true",
            "jdbc:postgresql://h:5432/db?user=u&password=CANARYQ",
            "user=u&password=CANARYQ&ssl=true",
            "Server=h;Database=d;Uid=u;Pwd=CANARYQ;",
            "Server=h;Password=CANARYQ;Encrypt=yes",
        ] {
            let out = strip_dsn_secrets(value);
            assert!(!out.contains("CANARYQ"), "{value} -> {out}");
        }

        // Through the real snapshot path, on the flag the leak was found on.
        let mut r = Recorder::new();
        r.set_invocation(json!({
            "host": "mongodb://h.example/db?w=majority&password=CANARYLIVE"
        }));
        let s = r.snapshot();
        assert!(
            !serde_json::to_string(&s).unwrap().contains("CANARYLIVE"),
            "{}",
            s["invocation"]["host"]
        );
        // Host, path and the benign option survive.
        let host = s["invocation"]["host"].as_str().unwrap();
        assert!(host.contains("h.example/db"), "{host}");
        assert!(host.contains("w=majority"), "{host}");
    }

    /// The nine shapes a 44-case corpus scan found still leaking after the
    /// first DSN fix. `spaces-around-eq` matters most: PostgreSQL documents
    /// spaces around `=` as optional, and libpq is the family this function
    /// exists for.
    #[test]
    fn structured_secret_shapes_that_previously_leaked() {
        for (label, value) in [
            ("spaces-around-eq", "password = CANARYX host=db"),
            ("space-before-eq", "password =CANARYX"),
            ("space-after-eq", "password= CANARYX"),
            ("sql-doubled-quote", "password='ab''CANARYX'"),
            ("pwd", "pwd=CANARYX"),
            ("pwd-upper", "PWD=CANARYX"),
            (
                "jdbc",
                "jdbc:postgresql://h:5432/db?user=u&password=CANARYX",
            ),
            ("amp-separated", "user=u&password=CANARYX&ssl=true"),
            ("semicolon-odbc", "Server=h;Database=d;Uid=u;Pwd=CANARYX;"),
            (
                "semicolon-odbc-pass",
                "Server=h;Password=CANARYX;Encrypt=yes",
            ),
        ] {
            let out = strip_dsn_secrets(value);
            assert!(!out.contains("CANARYX"), "[{label}] {value} -> {out}");
        }
    }

    /// Non-secret pairs must come through byte for byte — spacing, quoting and
    /// separators included. Over-redaction here destroys provenance.
    #[test]
    fn non_secret_pairs_are_reproduced_verbatim() {
        for value in [
            "options='-c statement_timeout=5s'",
            "host=db sslmode=require connect_timeout=10",
            "mongodb://h.example/db?w=majority&retryWrites=true",
            "Server=h;Database=d;Encrypt=yes",
            "http://h/a=b",
            "host = db",
            "no_pairs_here",
            "",
        ] {
            assert_eq!(strip_dsn_secrets(value), value, "mangled: {value}");
        }
        // Mixed: the secret goes, everything around it stays.
        assert_eq!(
            strip_dsn_secrets("host=db password='a b c' sslmode=require"),
            format!("host=db password={REDACTED} sslmode=require")
        );
    }

    /// An override ON a credential knob must still record the knob's NAME and
    /// the REASON — the entry exists to say what was overridden and why.
    ///
    /// Seeding secret-ness from the sibling `key` field blanked all four fields,
    /// so the entry recorded nothing at all. The assertion had the same bug and
    /// REQUIRED the blank, so the two had to be fixed together.
    #[test]
    fn an_override_on_a_credential_knob_keeps_its_name_and_reason() {
        let mut r = Recorder::new();
        r.note_override(
            "REDIS_AUTH",
            json!("OLD-CANARY"),
            json!("NEW-CANARY"),
            "rotated between phases",
        );
        let s = r.snapshot();
        let o = &s["overridden"][0];
        assert_eq!(o["key"], "REDIS_AUTH", "the knob name was destroyed");
        assert_eq!(
            o["reason"], "rotated between phases",
            "the explanation was destroyed"
        );
        // …while the values themselves are gone.
        assert_eq!(o["declared"], REDACTED);
        assert_eq!(o["effective"], REDACTED);
        let dumped = serde_json::to_string(&s).unwrap();
        assert!(!dumped.contains("OLD-CANARY") && !dumped.contains("NEW-CANARY"));
    }

    /// The same shape more generally: a `{"key":…, "value":…}` pair must keep
    /// its name. A header list rendered every credential entry as two identical
    /// placeholders, losing which header it was.
    #[test]
    fn a_key_value_pair_keeps_its_name_when_the_value_is_secret() {
        let mut r = Recorder::new();
        r.record_effective(
            "request_headers",
            json!([
                {"key": "Authorization", "value": "Bearer HEADER-CANARY"},
                {"key": "Content-Type", "value": "application/json"},
            ]),
        );
        let s = r.snapshot();
        let headers = &s["effective"]["request_headers"];
        assert_eq!(headers[0]["key"], "Authorization", "header name destroyed");
        assert_eq!(headers[0]["value"], REDACTED);
        assert_eq!(headers[1]["key"], "Content-Type");
        assert_eq!(headers[1]["value"], "application/json");
        assert!(!serde_json::to_string(&s).unwrap().contains("HEADER-CANARY"));
    }

    /// `strip_userinfo` computed the authority boundary BEFORE looking for `@`,
    /// so an unencoded `/` in a password returned the URL untouched.
    #[test]
    fn userinfo_is_stripped_even_when_the_password_breaks_url_syntax() {
        for (input, must_not_contain) in [
            ("redis://admin:hunt/er2@10.0.0.5:6379/0", "hunt/er2"),
            ("redis://admin:hu?nt@10.0.0.5:6379/0", "hu?nt"),
            ("redis://admin:h#nt@10.0.0.5:6379/0", "h#nt"),
            ("default:plainpw@127.0.0.1", "plainpw"),
            ("mongodb+srv://u:mongopw@cluster.example.net", "mongopw"),
        ] {
            let out = strip_userinfo(input);
            assert!(!out.contains(must_not_contain), "{input} -> {out}");
        }
        // A path containing `@` but no `user:pass` shape is left alone.
        assert_eq!(strip_userinfo("http://h/p@th"), "http://h/p@th");
        assert_eq!(strip_userinfo("localhost:6379"), "localhost:6379");
        assert_eq!(strip_userinfo("127.0.0.1"), "127.0.0.1");
    }

    /// The marker list missed whole families. `"AUTH_TOKEN".contains("_AUTH")`
    /// is false, which is how `auth_token` slipped through.
    #[test]
    fn secret_markers_cover_the_families_that_slipped_through() {
        for name in [
            "auth_token",
            "AUTH_TOKEN",
            "AUTHORIZATION",
            "AWS_SESSION_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "SERVICE_PRIVATE_KEY",
            "REDIS_PASS",
            "X_SIGNING_KEY",
            "api_key",
            "ELASTIC_PASSWORD",
            "REDIS_AUTH",
            "QDRANT_API_KEY",
            "VERTEX_ACCESS_TOKEN",
            "MONGODB_PASSWORD",
            "OPENSEARCH_PASSWORD",
            "PGVECTOR_PASSWORD",
        ] {
            assert!(is_secret(name), "{name} is not treated as a secret");
        }
        // Benign keys whose values are load-bearing provenance stay in the clear.
        for name in [
            "REDIS_KEY_PREFIX",
            "MILVUS_COLLECTION_NAME",
            "host",
            "parallel",
            "hnsw_config",
            "number_of_shards",
            "ELASTIC_INDEX",
        ] {
            assert!(!is_secret(name), "{name} was redacted unnecessarily");
        }
    }

    /// Deliberate, pinned decision: usernames are recorded in the clear.
    ///
    /// A username is provenance (which role the run authenticated as) and is not
    /// itself a credential. It is half of one, and PII in some deployments, so
    /// the call is written down and pinned rather than left to the marker list's
    /// shape. Flip this test to flip the policy.
    #[test]
    fn usernames_are_deliberately_recorded_in_the_clear() {
        let mut r = Recorder::new();
        r.observe_env("REDIS_USER", Some("benchrunner"));
        assert_eq!(r.snapshot()["env"]["REDIS_USER"], "benchrunner");
    }

    /// Drives `begin_experiment` — the function the mutation actually edits —
    /// and seeds the state it must clear ITSELF.
    ///
    /// The previous guard was `a_sweep_does_not_inherit_the_previous_configs_knobs`,
    /// which relied on residue other tests happened to leave in the global
    /// recorder: measured at 7 red / 4 green over 11 default-threaded runs and
    /// **3/3 GREEN run alone**. `begin_clears_every_accumulating_field` could
    /// never catch it at all, because it exercises `Recorder::begin`, which the
    /// mutation does not touch. This one seeds every accumulating field
    /// explicitly, so deleting `reset()` from `begin_experiment` fails it
    /// deterministically, alone or in a suite.
    #[test]
    fn begin_experiment_clears_the_previous_experiments_accumulation() {
        let _l = test_lock();
        let raw = json!({"name": "cfg-b", "engine": "redis"});
        let mut cfg: crate::config::EngineConfig = serde_json::from_value(raw.clone()).unwrap();
        cfg.raw = Some(raw);

        // Stand in for configuration A: seed all four accumulating fields here,
        // rather than hoping another test left them behind.
        reset();
        let _g = EnvGuard::set("VDBB_TEST_BLEED", Some("11"));
        assert_eq!(env_parsed::<u16>("VDBB_TEST_BLEED", 0), 11);
        record_effective("config_a_only", json!("leftover"));
        note_override("config_a_knob", json!(1), json!(2), "config A");
        note_ignored("config_a_ignored");
        let before = snapshot();
        assert_eq!(before["env"]["VDBB_TEST_BLEED"], "11");
        assert_eq!(before["effective"]["config_a_only"], "leftover");

        // Configuration B starts.
        let _recording = begin_experiment(&cfg, json!({"host": "b"}));

        let s = snapshot();
        assert!(
            s["env"].as_object().unwrap().is_empty(),
            "config A's env observations survived into config B: {}",
            s["env"]
        );
        assert!(
            s["effective"].as_object().unwrap().is_empty(),
            "config A's resolved knobs survived into config B: {}",
            s["effective"]
        );
        assert!(
            s["overridden"].as_array().unwrap().is_empty(),
            "config A's overrides survived into config B"
        );
        assert_eq!(
            s["ignored_declared_keys"]["keys"],
            json!([]),
            "config A's ignored keys survived into config B"
        );
        // …and config B's own declaration is in place.
        assert_eq!(s["invocation"]["host"], "b");
    }

    /// `begin` clears the four ACCUMULATING fields. Pure — no process globals,
    /// no environment, no scheduling. The global-state version of this passed
    /// 9 runs in 11 and 3/3 under `--test-threads=1` with `reset()` deleted,
    /// because residual state from a previously-scheduled test pre-seeded
    /// `env` and `observe_env` is first-write-wins.
    #[test]
    fn begin_clears_every_accumulating_field() {
        let mut r = Recorder::new();
        r.observe_env("A", Some("1"));
        r.record_effective("b", json!(2));
        r.note_override("c", json!(3), json!(4), "why");
        r.note_ignored("d");
        r.set_declared(Some(&json!({"x": 1})), None);
        r.set_invocation(json!({"host": "h"}));
        r.set_phase("search");

        r.begin();

        let s = r.snapshot();
        assert!(s["env"].as_object().unwrap().is_empty(), "env survived");
        assert!(
            s["effective"].as_object().unwrap().is_empty(),
            "effective survived"
        );
        assert!(
            s["overridden"].as_array().unwrap().is_empty(),
            "overridden survived"
        );
        assert_eq!(
            s["ignored_declared_keys"]["keys"],
            json!([]),
            "ignored survived"
        );
        assert!(s["declared"].is_null());
        assert!(s["invocation"].is_null());
        assert!(s["phase"].is_null());
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
        drop(_g);
        // `str::trim` uses the Unicode White_Space property, so a NO-BREAK SPACE
        // pasted out of documentation is accepted too. Documented AND pinned.
        let _g2 = EnvGuard::set("VDBB_TEST_PORT", Some("\u{a0}6381"));
        reset();
        assert_eq!(env_parsed::<u16>("VDBB_TEST_PORT", 6379), 6381);
        let _g = EnvGuard::set("VDBB_TEST_PORT", Some(" 6380 "));
        reset();
        let port: u16 = env_parsed("VDBB_TEST_PORT", 6379);
        assert_eq!(port, 6380);
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
