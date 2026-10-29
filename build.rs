//! Embeds the git commit the binaries are built from (#366).
//!
//! I/O only: runs `git`, hands the answers to `src/build_info/resolve.rs`
//! (unit-tested as `build_info::resolve`) and emits its decision as `VDB_*`
//! compile-time variables. With no git, or no `.git` of the crate's own (the
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
const COMPILED_INPUTS: [&str; 4] = ["build.rs", "src", "Cargo.toml", "Cargo.lock"];

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        // `git status` otherwise refreshes `.git/index` under an optional
        // lock. Rewriting the index from a build script makes the build its
        // own input: observed recompiling the crate on EVERY `cargo build`
        // in a bind-mounted checkout, where the stat cache never matches.
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

    // Re-run when HEAD moves (checkout, commit, reset) or the index changes.
    // `logs/HEAD` (HEAD's reflog) is appended on every move of HEAD, including
    // a commit to a branch whose ref was packed and so had no loose file to
    // watch until that commit wrote one.
    if probe.toplevel.as_deref() == Some(manifest_canon.as_path()) {
        let mut paths = vec![
            "HEAD".to_string(),
            "logs/HEAD".to_string(),
            "index".to_string(),
            "packed-refs".to_string(),
        ];
        if let Some(r) = git(&manifest_dir, &["symbolic-ref", "-q", "HEAD"]) {
            paths.push(r.trim().to_string());
        }
        for p in paths {
            if let Some(path) = git(&manifest_dir, &["rev-parse", "--git-path", &p]) {
                let path = manifest_dir.join(path.trim());
                if path.exists() {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
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
