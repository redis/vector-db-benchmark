//! Per-config index-name and key-prefix derivation for the Redis-wire engines
//! (Redis / Valkey / Dragonfly).
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

    #[test]
    fn derive_key_prefix_has_single_trailing_colon() {
        let p = derive_key_prefix("redis-m-8");
        assert_eq!(p, "redis-m-8:");
        assert_eq!(p.matches(':').count(), 1);
    }
}
