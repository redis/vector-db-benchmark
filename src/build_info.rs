//! Which source these binaries were built from (#366).
//!
//! `build.rs` resolves the commit with [`resolve`] and passes the answer in
//! through `VDB_*` compile-time variables; this module only exposes it. The
//! resolution rules, and why "unknown" is preferred to a guess, are on
//! [`resolve`].

pub mod resolve;

/// The crate version (`Cargo.toml`).
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Full commit SHA, or the literal `unknown`.
pub const GIT_SHA: &str = env!("VDB_GIT_SHA");

/// `clean`, `dirty` or `unknown` — see [`resolve::tree_state`].
pub const GIT_TREE: &str = env!("VDB_GIT_TREE");

/// `git`, `env` or `unknown` — see [`resolve::Origin`].
pub const GIT_SHA_SOURCE: &str = env!("VDB_GIT_SHA_SOURCE");

/// What every binary prints after its name for `--version`, e.g.
/// `0.1.0 (git 0123…cdef)`. Built by [`resolve::version_line`].
pub const LONG_VERSION: &str = env!("VDB_LONG_VERSION");

/// Target triple the binaries were compiled for, e.g.
/// `x86_64-unknown-linux-gnu`.
pub const BUILD_TARGET: &str = env!("VDB_BUILD_TARGET");

/// The non-default cargo features compiled in. `release-static` (what the
/// release archives are built with) swaps the client stack: vendored OpenSSL
/// and bundled HDF5 instead of the system libraries, and TLS cost sits in the
/// timed path of the managed engines. Add a feature here when one is added to
/// `Cargo.toml`.
pub fn features() -> Vec<&'static str> {
    let mut f = Vec::new();
    if cfg!(feature = "release-static") {
        f.push("release-static");
    }
    f
}

/// The `params.build` block of every result file: which build produced the
/// numbers, so a result can be traced to a release archive, and told apart
/// from a source build of the same commit (`features`, `target`).
pub fn provenance() -> serde_json::Value {
    serde_json::json!({
        "crate_version": CRATE_VERSION,
        "git_sha": if GIT_SHA == "unknown" { serde_json::Value::Null } else { GIT_SHA.into() },
        "git_tree": GIT_TREE,
        "git_sha_source": GIT_SHA_SOURCE,
        "target": BUILD_TARGET,
        "features": features(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four compile-time values must agree with each other: a build that
    /// knows its SHA prints it in `--version`, and one that does not says so.
    #[test]
    fn compiled_in_values_are_consistent() {
        assert!(LONG_VERSION.starts_with(&format!("{CRATE_VERSION} (git ")));
        assert!(
            matches!(GIT_TREE, "clean" | "dirty" | "unknown"),
            "{GIT_TREE}"
        );
        match GIT_SHA_SOURCE {
            "git" | "env" => {
                assert!(resolve::is_full_sha(GIT_SHA), "{GIT_SHA}");
                assert!(LONG_VERSION.contains(GIT_SHA), "{LONG_VERSION}");
                assert_eq!(LONG_VERSION.contains("dirty tree"), GIT_TREE == "dirty");
            }
            "unknown" => {
                assert_eq!(GIT_SHA, "unknown");
                assert!(
                    LONG_VERSION.ends_with("(git commit unknown)"),
                    "{LONG_VERSION}"
                );
            }
            other => panic!("unexpected VDB_GIT_SHA_SOURCE {other:?}"),
        }
    }

    #[test]
    fn provenance_block_mirrors_the_constants() {
        let p = provenance();
        assert_eq!(p["crate_version"], CRATE_VERSION);
        assert_eq!(p["git_tree"], GIT_TREE);
        assert_eq!(p["git_sha_source"], GIT_SHA_SOURCE);
        if GIT_SHA == "unknown" {
            assert!(p["git_sha"].is_null(), "unknown must be null, not a string");
        } else {
            assert_eq!(p["git_sha"], GIT_SHA);
        }
        // The release archives are built with `release-static` (and their
        // unit tests run with it in release-binaries.yml); `make agent-check`
        // builds without it. Both directions are checked between the two.
        let expect: Vec<&str> = if cfg!(feature = "release-static") {
            vec!["release-static"]
        } else {
            vec![]
        };
        assert_eq!(p["features"], serde_json::json!(expect));
        assert_eq!(p["target"], BUILD_TARGET);
        let mut keys: Vec<_> = p.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "crate_version",
                "features",
                "git_sha",
                "git_sha_source",
                "git_tree",
                "target"
            ]
        );
    }

    /// The triple is the one this test binary runs as: its arch and OS agree
    /// with the compiled `std::env::consts`, checked independently of how
    /// `build.rs` obtained it.
    #[test]
    fn build_target_names_the_platform_the_code_was_compiled_for() {
        let arch = std::env::consts::ARCH;
        assert!(
            BUILD_TARGET.starts_with(&format!("{arch}-")),
            "{BUILD_TARGET} vs arch {arch}"
        );
        let os = match std::env::consts::OS {
            "macos" => "darwin",
            o => o,
        };
        assert!(BUILD_TARGET.contains(os), "{BUILD_TARGET} vs os {os}");
    }
}
