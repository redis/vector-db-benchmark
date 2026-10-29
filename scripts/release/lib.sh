# shellcheck shell=bash disable=SC2034  # arrays are used by the scripts that source this
# Shared helpers for the release-archive scripts (#366). Sourced, not run.
#
# verify.sh sources this inside a bare `ubuntu:22.04` container, which has no
# cargo, python or binutils: defining the functions that need them is fine,
# calling them there is not (only package.sh does, on the build host).

# The binaries #366 names. Every archive must carry them; package.sh packages
# every [[bin]] cargo knows about, so this is the floor, not the list.
REQUIRED_BINS=(vector-db-benchmark prepare-msmarco generate-dataset bench_npy bench_jsonl)

# The binaries that parse flags with clap, and so answer --help / --version.
CLAP_BINS=(vector-db-benchmark prepare-msmarco generate-dataset)

# What the binaries are compiled from: build.rs's COMPILED_INPUTS, which scope
# the dirty flag a binary reports (INV-A4 keeps the two lists equal).
COMPILED_INPUTS=(build.rs src Cargo.toml Cargo.lock)

# `uncommitted_changes`: print what makes this checkout differ from HEAD for a
# release archive, nothing when it is clean. Two scopes, both refused:
#  * the compiled inputs with build.rs's exact flags (untracked files count),
#    so a packaged binary can never report "dirty tree";
#  * every other TRACKED file, because the archive also ships README, LICENSE,
#    datasets/ and experiments/ from the tree.
uncommitted_changes() {
  git status --porcelain --untracked-files=all -- "${COMPILED_INPUTS[@]}"
  git status --porcelain --untracked-files=no
}

# GNU tar, as `$TAR`. The scripts use --transform / --mtime / --owner, which
# macOS's bsdtar rejects; on macOS install it with `brew install gnu-tar`.
require_gnu_tar() {
  local t v
  for t in gtar tar; do
    command -v "$t" >/dev/null 2>&1 || continue
    v="$("$t" --version 2>/dev/null || true)"
    if [[ "$v" == *"GNU tar"* ]]; then TAR="$t"; return 0; fi
  done
  echo "GNU tar is required (bsdtar lacks --transform/--mtime); on macOS: brew install gnu-tar" >&2
  return 1
}

# `in_list <needle> <items...>`: exact membership without a pipe. Never write
# `producer | grep -q` under `set -o pipefail`: grep exits at the first match,
# the producer takes SIGPIPE, and the pipeline reports failure (141), which an
# `if` then reads as "not found".
in_list() {
  local needle="$1" item
  shift
  for item in "$@"; do [[ "$item" == "$needle" ]] && return 0; done
  return 1
}

# Rust target triple -> the <platform> part of the archive name.
platform_for_target() {
  case "$1" in
    x86_64-unknown-linux-gnu) echo linux-x86_64 ;;
    aarch64-unknown-linux-gnu) echo linux-aarch64 ;;
    aarch64-apple-darwin) echo macos-aarch64 ;;
    *) echo "unsupported release target: $1" >&2; return 1 ;;
  esac
}

# A tag becomes part of a file name and a URL: no separators, no spaces.
check_tag() {
  if [[ ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
    echo "invalid tag '$1': expected letters, digits and . _ + - only" >&2
    return 1
  fi
}

# `tag_matches_version <tag> <crate version>`: the tag names that version,
# optionally with a `v` prefix and/or a pre-release/build suffix after `-`/`+`.
# (A RELEASE tag is further restricted by check_release_tag: no `+`.)
# `0.10.0` does NOT match crate version `0.1.0`, and `0.1.0` does not match
# `0.1.0.1`.
tag_matches_version() {
  local tag="${1#v}" version="$2"
  [[ "$tag" == "$version" || "$tag" == "$version"-* || "$tag" == "$version"+* ]]
}

# `check_release_tag <tag> <crate version>`: may this tag be PUBLISHED with
# archives? Run by the workflow's plan job, before anything is built, and by
# `package.sh --release`. Refuses:
#  * a tag that does not name the crate version — the release would go public
#    as "latest" with no archives once every build had failed on it;
#  * a `+` (semver build metadata): GitHub may rename an uploaded asset whose
#    name holds one, which would leave SHA256SUMS naming files the release
#    does not serve.
check_release_tag() {
  local tag="$1" version="$2"
  check_tag "$tag" || return 1
  if [[ "$tag" == *+* ]]; then
    echo "release tag '$tag' contains '+': GitHub may rename assets named after it; use '-' (e.g. ${tag//+/-})" >&2
    return 1
  fi
  if ! tag_matches_version "$tag" "$version"; then
    echo "release tag '$tag' does not name the crate version '$version' (Cargo.toml [package].version)." >&2
    echo "  Bump the version in Cargo.toml, merge, and tag that commit X.Y.Z, vX.Y.Z or X.Y.Z-rc.N." >&2
    return 1
  fi
}

crate_version() {
  # The first `version = "…"` in Cargo.toml is [package].version.
  sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n1
}

# Every binary target of the crate, one per line, from cargo itself.
package_bins() {
  cargo metadata --no-deps --format-version 1 --locked |
    python3 -c 'import json, sys
for p in json.load(sys.stdin)["packages"]:
    for t in p["targets"]:
        if "bin" in t["kind"]:
            print(t["name"])' | LC_ALL=C sort
}

# Newest glibc symbol version the Linux archives may need: Amazon Linux 2023's
# (2.34), the default AWS client image, which is older than Ubuntu 22.04's 2.35.
MAX_GLIBC=2.34
# Oldest macOS the macOS archive promises (MACOSX_DEPLOYMENT_TARGET in the workflow).
MAX_MACOS_MINOS=11.0

# `check_portable_linkage <binary> <target>` — fail unless the binary needs
# nothing a fresh host lacks. Linux: only glibc's own libraries and libgcc_s
# (no libhdf5, libssl, libcrypto, libz …), and no glibc symbol newer than
# $MAX_GLIBC. macOS: only system libraries and frameworks (nothing from
# Homebrew). Runs on the build host, which has binutils / otool.
check_portable_linkage() {
  local bin="$1" target="$2" lib newest
  case "$target" in
    *-linux-gnu)
      while read -r lib; do
        case "$lib" in
          libc.so.6|libm.so.6|libgcc_s.so.1|libpthread.so.0|libdl.so.2|librt.so.1|ld-linux-*.so.*) ;;
          *) echo "$bin: links $lib, which a fresh host may not have (release-static not applied?)" >&2
             return 1 ;;
        esac
      done < <(readelf -d "$bin" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
      # `|| true`: grep finds nothing in a binary with no GLIBC_ symbols, which
      # under pipefail would abort here silently instead of reporting "none".
      newest="$(objdump -T "$bin" | { grep -o 'GLIBC_[0-9][0-9.]*' || true; } | sed 's/GLIBC_//' | sort -uV | tail -n1)"
      if [[ -n "$newest" && "$(printf '%s\n%s\n' "$newest" "$MAX_GLIBC" | sort -V | tail -n1)" != "$MAX_GLIBC" ]]; then
        echo "$bin: needs GLIBC_$newest, newer than the promised $MAX_GLIBC (Amazon Linux 2023)" >&2
        return 1
      fi
      echo "linkage ok: $(basename "$bin") (newest glibc symbol ${newest:-none})"
      ;;
    *-apple-darwin)
      while read -r lib; do
        case "$lib" in
          /usr/lib/*|/System/Library/*) ;;
          *) echo "$bin: links $lib, which is not part of macOS" >&2; return 1 ;;
        esac
      done < <(otool -L "$bin" | tail -n +2 | awk '{print $1}')
      # The oldest macOS the binary loads on (LC_BUILD_VERSION minos, which the
      # linker sets from MACOSX_DEPLOYMENT_TARGET) must not exceed the one
      # promised. This catches a wrong deployment target for the link; it can
      # NOT see a C object compiled for a newer macOS inside the binary (the
      # linker only warns "built for newer macOS version"), which is why the
      # workflow exports MACOSX_DEPLOYMENT_TARGET for the cmake/cc builds too.
      # Captured first: `otool | awk '{...; exit}'` would SIGPIPE otool.
      local loads
      loads="$(otool -l "$bin")" || { echo "$bin: otool -l failed" >&2; return 1; }
      newest="$(awk '/LC_BUILD_VERSION/{f=1} f && $1 == "minos" {print $2; exit}' <<<"$loads")"
      [[ -n "$newest" ]] || { echo "$bin: no LC_BUILD_VERSION minos" >&2; return 1; }
      if [[ "$(printf '%s\n%s\n' "$newest" "$MAX_MACOS_MINOS" | sort -V | tail -n1)" != "$MAX_MACOS_MINOS" ]]; then
        echo "$bin: minos $newest is newer than the promised macOS $MAX_MACOS_MINOS" >&2
        return 1
      fi
      echo "linkage ok: $(basename "$bin") (minos $newest)"
      ;;
    *) echo "check_portable_linkage: unsupported target $target" >&2; return 1 ;;
  esac
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}
