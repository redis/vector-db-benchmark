//! Embeds the git commit the binaries are built from, and how they were
//! built (target, features, profile, rustc), for `--version` and every
//! result file's `params.build` (#366).
//!
//! I/O only: runs `git` and `rustc -V`, hands the answers to
//! `src/build_info/resolve.rs` (unit-tested as `build_info::resolve`) and
//! emits its decision as `VDB_*` compile-time variables. With no git, or no `.git` of the crate's own (the
//! Docker build context), the build still succeeds and reports the commit as
//! unknown; `VDB_BUILD_GIT_SHA` can assert it instead.

#[allow(dead_code)]
#[path = "src/build_info/resolve.rs"]
mod resolve;

use std::path::{Path, PathBuf};
use std::process::Command;

/// What the binaries are compiled from. The dirty flag covers exactly these
/// (an edited README or test is not a different binary), and they are what
/// cargo watches below, so the flag can never be stale against them.
/// `scripts/release/lib.sh` has the same list (COMPILED_INPUTS); INV-A4 in
/// tests/harness_invariants.rs keeps the two equal.
const COMPILED_INPUTS: [&str; 4] = ["build.rs", "src", "Cargo.toml", "Cargo.lock"];

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        // `git status` otherwise refreshes `.git/index` under an optional
        // lock. A build script must not write to the repository: at 33caa04,
        // which also watched the index, that recompiled the crate on EVERY
        // `cargo build` in a bind-mounted checkout.
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let manifest_canon = manifest_dir.canonicalize().unwrap_or(manifest_dir.clone());

    println!("cargo:rerun-if-env-changed=VDB_BUILD_GIT_SHA");
    println!("cargo:rerun-if-env-changed=RUSTC");
    // A directory here is scanned recursively by cargo.
    for p in COMPILED_INPUTS {
        println!("cargo:rerun-if-changed={p}");
    }

    // Untracked files count under the compiled inputs: a new module that is
    // compiled in but not committed makes the binary differ from the commit.
    let mut status = vec!["status", "--porcelain", "--untracked-files=all", "--"];
    status.extend(COMPILED_INPUTS);
    let probe = resolve::GitProbe {
        toplevel: git(&manifest_dir, &["rev-parse", "--show-toplevel"])
            .map(|s| PathBuf::from(s.trim()))
            .map(|p| p.canonicalize().unwrap_or(p)),
        head: git(&manifest_dir, &["rev-parse", "HEAD"]),
        dirty: git(&manifest_dir, &status).map(|s| !s.trim().is_empty()),
    };

    // Re-run when HEAD moves (checkout, commit, reset). The dirty flag needs
    // nothing more: it covers only COMPILED_INPUTS, which are watched above.
    // NOT `.git/index`: any `git status` (an IDE polling, a prompt) rewrites
    // it, which would recompile the lib and every binary with no change.
    //
    // `logs/HEAD` (HEAD's reflog) is appended on every move of HEAD. Where
    // reflogs are off (`core.logAllRefUpdates=false`) and the branch ref is
    // packed, a commit instead CREATES the loose ref file, and for a branch
    // like `feat/x` after `git pack-refs --all --prune` even its directory
    // (`refs/heads/feat`) is gone. So the nearest EXISTING ancestor of the
    // ref's directory is watched (cargo scans a directory recursively): the
    // file appearing under it re-runs this script instead of keeping the old
    // SHA.
    if probe.toplevel.as_deref() == Some(manifest_canon.as_path()) {
        let git_path = |p: &str| {
            git(&manifest_dir, &["rev-parse", "--git-path", p]).map(|s| manifest_dir.join(s.trim()))
        };
        let mut watch: Vec<PathBuf> = ["HEAD", "logs/HEAD", "packed-refs"]
            .iter()
            .filter_map(|p| git_path(p))
            .filter(|p| p.exists())
            .collect();
        if let Some(r) = git(&manifest_dir, &["symbolic-ref", "-q", "HEAD"]) {
            if let Some(ref_file) = git_path(r.trim()) {
                if ref_file.exists() {
                    watch.push(ref_file.clone());
                }
                if let Some(dir) = ref_file.parent() {
                    watch.extend(resolve::nearest_existing_ancestor(dir, |p| p.exists()));
                }
            }
        }
        for path in watch {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    let env_override = std::env::var("VDB_BUILD_GIT_SHA").ok();
    let src = resolve::resolve(env_override.as_deref(), &manifest_canon, &probe)
        .unwrap_or_else(|e| panic!("{e}"));

    // Cargo always sets TARGET for build scripts; it is the triple being
    // compiled for (not the host), which is what a result file should name.
    println!(
        "cargo:rustc-env=VDB_BUILD_TARGET={}",
        std::env::var("TARGET").unwrap()
    );
    // The features cargo enabled for this crate, from its CARGO_FEATURE_*
    // variables: derived, so a feature added to Cargo.toml is recorded
    // without anyone remembering to list it.
    println!(
        "cargo:rustc-env=VDB_FEATURES={}",
        resolve::features_from_env(std::env::vars()).join(",")
    );
    // `release` or `debug` (the base profile cargo builds with).
    println!(
        "cargo:rustc-env=VDB_BUILD_PROFILE={}",
        std::env::var("PROFILE").unwrap_or_else(|_| "unknown".into())
    );
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = Command::new(rustc)
        .arg("-V")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=VDB_RUSTC_VERSION={rustc_version}");
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    println!(
        "cargo:rustc-env=VDB_GIT_SHA={}",
        src.sha.as_deref().unwrap_or("unknown")
    );
    println!("cargo:rustc-env=VDB_GIT_TREE={}", resolve::tree_state(&src));
    println!("cargo:rustc-env=VDB_GIT_SHA_SOURCE={}", src.origin.as_str());
    println!(
        "cargo:rustc-env=VDB_LONG_VERSION={}",
        resolve::version_line(&version, &src)
    );
}
