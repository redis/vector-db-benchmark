//! Per-config index-name and key-prefix derivation for the Redis-wire engines
//! (Redis / Valkey / Dragonfly / KiviDB / VectorSets).
//!
//! Issue #151-4: an M×EF_CONSTRUCTION sweep runs several configs of the same
//! engine against one server. When every config used the literal index name
//! `idx` with `PREFIX 1 ""`, the configs shared one index and one keyspace, so
//! an "upload all, then `--skip-upload` search each" flow silently overwrote
//! each config's graph with the next — collapsing recall and memory to a single
//! (last-writer-wins) point.
//!
//! The fix makes each config address a *disjoint* index + keyspace derived
//! purely from `engine_config.name`, so N configs coexist on one server.
//!
//! VectorSets (#236) joined late. It has no FT index and no doc keyspace — its
//! whole corpus is ONE Redis key that VADD/VSIM/VINFO/DEL address — so it uses
//! [`derive_index_name`] for that key and has no use for [`derive_key_prefix`].
//! The failure mode was identical and worse: `configure()` issued a literal
//! `DEL idx`, so starting config B deleted config A's entire corpus outright.

/// Map any char outside `[A-Za-z0-9_-]` to `_`. Guarantees: (a) the only `:` in
/// a derived name/prefix is our own separator; (b) no SCAN glob metacharacters
/// (`* ? [ ] \`) can originate from a config name. Both properties are
/// load-bearing: the prefix-scoped SCAN+UNLINK teardown treats `<prefix>*` as a
/// glob, and the doc-key → id recovery splits on the last `:`.
pub fn sanitize_token(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Per-config index name: `"<base>:<sanitized-config-name>"`.
///
/// `base` is the env override (`base_env`) if set, else `default_base` (`"idx"`).
/// The config suffix is ALWAYS appended so a pinned base cannot re-collapse a
/// sweep into one shared index.
///
/// Exact-pin escape hatch: if `<base_env>_EXACT=1` is set, the base is used
/// verbatim with NO config suffix (point a single config at an out-of-band
/// index). Combining exact mode with >1 config for the engine is caught by the
/// startup collision guard in `experiment::run`.
pub fn derive_index_name(base_env: &str, default_base: &str, engine_name: &str) -> String {
    let base = crate::effective_config::env_or(base_env, default_base);
    if index_name_exact(base_env) {
        return base;
    }
    format!("{base}:{}", sanitize_token(engine_name))
}

/// Whether the `<base_env>_EXACT` escape hatch is enabled (value `1`/`true`).
pub fn index_name_exact(base_env: &str) -> bool {
    // `env_flag`, not `env_var`: the raw text alone cannot show that
    // `..._EXACT="yes"` left the escape hatch CLOSED, and a reader grepping the
    // artifact for the variable would read "yes" as an opt-in that happened.
    crate::effective_config::env_flag(&format!("{base_env}_EXACT"), &["1", "true"])
}

/// Per-config key prefix: `"<sanitized-config-name>:"`. The trailing `:` is the
/// only `:` in a doc key, so `doc_key_to_id` recovers the id as the tail after
/// the last `:`. Debug-asserted non-empty (keyspace-hygiene invariant: an empty
/// prefix would make the scoped teardown a keyspace-wide `*` wipe).
pub fn derive_key_prefix(engine_name: &str) -> String {
    let t = sanitize_token(engine_name);
    debug_assert!(
        !t.is_empty(),
        "config name must sanitize to a non-empty token"
    );
    format!("{t}:")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_maps_glob_and_colon_metachars() {
        // Colons, globs and backslashes all collapse to '_'; alnum/-/_ survive.
        assert_eq!(sanitize_token("redis-m-16_ef-64"), "redis-m-16_ef-64");
        assert_eq!(sanitize_token("a:b:c"), "a_b_c");
        assert_eq!(sanitize_token("a*b?c[d]e\\f"), "a_b_c_d_e_f");
        assert_eq!(sanitize_token("space here"), "space_here");
    }

    #[test]
    fn derive_index_name_appends_sanitized_config() {
        // No env set → default base "idx" + ':' + sanitized name.
        assert_eq!(
            derive_index_name("NONEXISTENT_ENV_151_4", "idx", "redis-m-8"),
            "idx:redis-m-8"
        );
    }

    /// #236: two VectorSets configs must derive two DIFFERENT keys, because that
    /// key is the whole corpus — a shared one means `configure()`'s `DEL` wipes
    /// the sibling. Pinned to the literal expected strings so a future change to
    /// the separator or the default base has to be made deliberately.
    #[test]
    fn vectorsets_configs_derive_distinct_keys() {
        let a = derive_index_name("VECTORSETS_INDEX_NAME_UNSET_236", "idx", "vectorsets-fp32");
        let b = derive_index_name("VECTORSETS_INDEX_NAME_UNSET_236", "idx", "vectorsets-q8");
        assert_eq!(a, "idx:vectorsets-fp32");
        assert_eq!(b, "idx:vectorsets-q8");
        assert_ne!(a, b);
        // And neither may be the bare legacy key that every config used to share.
        assert_ne!(a, "idx");
        assert_ne!(b, "idx");
    }

    /// The `_EXACT` escape hatch drops the config suffix entirely. Two configs
    /// then DO collide by design — which is why `experiment::run`'s startup guard
    /// rejects exact mode with >1 config for the engine. Uses a private env name
    /// so no other test observes it.
    #[test]
    fn exact_pin_drops_the_config_suffix() {
        let base_env = "VECTORSETS_INDEX_NAME_EXACTTEST_236";
        std::env::set_var(base_env, "myvset");
        std::env::set_var(format!("{base_env}_EXACT"), "1");
        assert!(index_name_exact(base_env));
        assert_eq!(
            derive_index_name(base_env, "idx", "vectorsets-fp32"),
            "myvset"
        );
        // Without the pin the base is still honoured, but the suffix returns.
        std::env::remove_var(format!("{base_env}_EXACT"));
        assert!(!index_name_exact(base_env));
        assert_eq!(
            derive_index_name(base_env, "idx", "vectorsets-fp32"),
            "myvset:vectorsets-fp32"
        );
        std::env::remove_var(base_env);
    }

    #[test]
    fn derive_key_prefix_has_single_trailing_colon() {
        let p = derive_key_prefix("redis-m-8");
        assert_eq!(p, "redis-m-8:");
        assert_eq!(p.matches(':').count(), 1);
    }
}
