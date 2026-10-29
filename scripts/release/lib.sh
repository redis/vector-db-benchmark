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
# `0.10.0` does NOT match crate version `0.1.0`, and `0.1.0` does not match
# `0.1.0.1`.
tag_matches_version() {
  local tag="${1#v}" version="$2"
  [[ "$tag" == "$version" || "$tag" == "$version"-* || "$tag" == "$version"+* ]]
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

# Oldest glibc the Linux archives promise to run on: Ubuntu 22.04's.
MAX_GLIBC=2.35
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
      newest="$(objdump -T "$bin" | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -n1)"
      if [[ -n "$newest" && "$(printf '%s\n%s\n' "$newest" "$MAX_GLIBC" | sort -V | tail -n1)" != "$MAX_GLIBC" ]]; then
        echo "$bin: needs GLIBC_$newest, newer than the promised $MAX_GLIBC (Ubuntu 22.04)" >&2
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
      # The oldest macOS the binary loads on (LC_BUILD_VERSION minos) must not
      # exceed the one promised; C code built by cmake/cc defaults to the SDK's.
      newest="$(otool -l "$bin" | awk '/LC_BUILD_VERSION/{f=1} f && $1 == "minos" {print $2; exit}')"
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
