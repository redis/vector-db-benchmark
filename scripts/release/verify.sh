#!/usr/bin/env bash
# Extract a release archive into an empty directory and run what it ships (#366).
#
#   scripts/release/verify.sh <archive.tar.gz> --expect-sha <40-hex sha> [--expect-tag <tag>]
#
# Needs only bash, coreutils, tar and gzip: the release workflow runs it inside
# a bare `ubuntu:22.04` container (no toolchain, no libhdf5, no libssl-dev, no
# ca-certificates) to check the acceptance criterion of #366 — "extract and run
# on a fresh Ubuntu 22.04 host".
#
# Checks, in order:
#  * the archive is named vector-db-benchmark-<tag>-<platform>.tar.gz and is FLAT
#    (the binaries are at its root, so `tar xz && ./vector-db-benchmark` works);
#  * every required binary is present and executable, and BUILD-INFO.txt lists
#    exactly the binaries the archive holds;
#  * each clap binary's `--version` is exactly `<name> <crate version> (git <sha>)`
#    for the expected commit — not "dirty tree", not "commit unknown" — and its
#    `--help` exits 0;
#  * `vector-db-benchmark --describe build` (the `params.build` block of every
#    result file) names that commit, a clean tree, `release-static`, and, on
#    THIS host, the same OpenSSL and HDF5 as BUILD-INFO.txt recorded at build
#    time: a binary that picked up the host's libraries would differ;
#  * every binary actually runs a real workload from the archive's own files:
#    `--describe datasets` reads the bundled datasets.json, generate-dataset
#    writes an .npy corpus that bench_npy reads back, bench_jsonl reads the
#    bundled random-100 corpus, and bench_hdf5 initialises the linked HDF5.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"

archive="" expect_sha="" expect_tag=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --expect-sha) expect_sha="$2"; shift 2 ;;
    --expect-tag) expect_tag="$2"; shift 2 ;;
    -*) echo "verify.sh: unknown flag: $1" >&2; exit 2 ;;
    *) archive="$1"; shift ;;
  esac
done
[[ -n "$archive" && -n "$expect_sha" ]] || {
  echo "usage: verify.sh <archive.tar.gz> --expect-sha <sha> [--expect-tag <tag>]" >&2; exit 2; }
[[ "$expect_sha" =~ ^[0-9a-f]{40}$ ]] || { echo "verify.sh: --expect-sha must be a full 40-hex SHA" >&2; exit 2; }

fail() { echo "verify.sh: FAIL: $*" >&2; exit 1; }

base="$(basename "$archive")"
[[ "$base" =~ ^vector-db-benchmark-(.+)-(linux-x86_64|linux-aarch64|macos-aarch64)\.tar\.gz$ ]] ||
  fail "unexpected archive name '$base'"
tag="${BASH_REMATCH[1]}"
[[ -z "$expect_tag" || "$tag" == "$expect_tag" ]] || fail "archive is for tag '$tag', expected '$expect_tag'"

archive="$(cd "$(dirname "$archive")" && pwd)/$base"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

# Inspect every member BEFORE extracting anything: an absolute path or a `..`
# component could write outside this directory, and a link member could point
# a later member (or a binary we are about to run) anywhere. The archive is
# plain files and directories only.
members="$(tar tzf "$archive")" || fail "cannot list $base"
while IFS= read -r p; do
  case "/$p/" in
    //*|*/../*) fail "archive member escapes the extraction directory: $p" ;;
  esac
done <<<"$members"
# `tar tv` prints the type as the first character of the mode column.
# Captured first, then matched: `tar tv | grep -q` under pipefail would turn a
# match early in a long listing into SIGPIPE, which an `if` reads as "no link".
listing="$(tar tvzf "$archive")" || fail "cannot list $base"
odd="$(grep -v '^[-d]' <<<"$listing" || true)"
[[ -z "$odd" ]] || fail "archive contains a link or special member: $(head -n3 <<<"$odd")"

# The way the README tells people to install it.
gzip -dc "$archive" | tar x

info="BUILD-INFO.txt"
[[ -f "$info" ]] || fail "no $info at the archive root"
field() { sed -n "s/^$1: //p" "$info" | head -n1; }
[[ "$(field git_sha)" == "$expect_sha" ]] || fail "$info git_sha '$(field git_sha)' != expected '$expect_sha'"
[[ "$(field tag)" == "$tag" ]] || fail "$info tag '$(field tag)' != archive tag '$tag'"
version="$(field crate_version)"
[[ -n "$version" ]] || fail "$info has no crate_version"

read -r -a listed <<<"$(field binaries)"
[[ ${#listed[@]} -gt 0 ]] || fail "$info lists no binaries"
for b in "${REQUIRED_BINS[@]}"; do
  in_list "$b" "${listed[@]}" || fail "required binary '$b' not in the archive"
done
# Exactly the listed binaries are executables at the root — nothing missing,
# nothing extra that BUILD-INFO does not account for.
actual="$(find . -maxdepth 1 -type f -perm -u+x | sed 's|^\./||' | LC_ALL=C sort | tr '\n' ' ')"
expected="$(printf '%s\n' "${listed[@]}" | LC_ALL=C sort | tr '\n' ' ')"
[[ "$actual" == "$expected" ]] || fail "executables at the archive root [$actual] != BUILD-INFO binaries [$expected]"

# What the dynamic loader actually resolves on THIS host: nothing missing, and
# none of the libraries the archive is supposed to carry inside it.
if [[ "$(uname -s)" == Linux ]] && command -v ldd >/dev/null 2>&1; then
  for b in "${listed[@]}"; do
    # A fully static executable has nothing to resolve; ldd says so and
    # exits non-zero.
    deps="$(ldd "./$b" 2>&1)" || [[ "$deps" == *"not a dynamic executable"* ]] || fail "ldd ./$b failed: $deps"
    if grep -E 'not found|libssl|libcrypto|libhdf5|libz\.so' <<<"$deps"; then
      fail "$b resolves a library the archive should not need (above)"
    fi
  done
  echo "ok: ldd resolves only system libraries for ${#listed[@]} binaries"
fi

for f in datasets/datasets.json datasets/random-100/vectors.jsonl experiments/configurations LICENSE README.md; do
  [[ -e "$f" ]] || fail "archive is missing $f"
done

for b in "${CLAP_BINS[@]}"; do
  got="$("./$b" --version)" || fail "$b --version exited non-zero"
  want="$b $version (git $expect_sha)"
  [[ "$got" == "$want" ]] || fail "$b --version printed '$got', expected '$want'"
  "./$b" --help >/dev/null || fail "$b --help exited non-zero"
  echo "ok: $got"
done

./vector-db-benchmark --describe build > build.json || fail "--describe build exited non-zero"
has() { grep -qF -- "$1" build.json; }
has "\"git_sha\": \"$expect_sha\"" || fail "--describe build does not name $expect_sha: $(cat build.json)"
has '"git_tree": "clean"' || fail "--describe build does not report a clean tree: $(cat build.json)"
has '"release-static"' || fail "--describe build does not list the release-static feature: $(cat build.json)"
for lib in openssl hdf5; do
  recorded="$(field "$lib")"
  [[ -n "$recorded" ]] || fail "$info records no $lib version"
  if [[ "$recorded" == null ]]; then want="\"$lib\": null"; else want="\"$lib\": \"$recorded\""; fi
  has "$want" || fail "$lib loaded here differs from the build's ($recorded): $(grep -F "\"$lib\"" build.json)"
done
echo "ok: --describe build: $expect_sha, release-static, openssl $(field openssl), hdf5 $(field hdf5)"

./vector-db-benchmark --describe datasets > describe.out || fail "--describe datasets exited non-zero"
grep -q 'random-100' describe.out || fail "--describe datasets did not list the bundled random-100 dataset"
echo "ok: vector-db-benchmark --describe datasets read the bundled datasets.json"

./generate-dataset --out-dir gen --only hybrid > /dev/null || fail "generate-dataset failed"
./bench_npy gen/synthetic-hybrid-16/vectors.npy 1 > bench_npy.out || fail "bench_npy failed on a generated corpus"
grep -q 'Throughput' bench_npy.out || fail "bench_npy printed no result"
echo "ok: generate-dataset + bench_npy"

./bench_jsonl datasets/random-100/vectors.jsonl 1 > bench_jsonl.out || fail "bench_jsonl failed on random-100"
grep -q 'Throughput' bench_jsonl.out || fail "bench_jsonl printed no result"
echo "ok: bench_jsonl"

if in_list bench_hdf5 "${listed[@]}"; then
  # No HDF5 file ships in the archive, so this checks the statically linked
  # library loads and initialises, and that the error is HDF5's own.
  # (The full read path, DEFLATE included, is a unit test the release build
  # job runs under `--features release-static`.)
  set +e
  ./bench_hdf5 does-not-exist.hdf5 1 > bench_hdf5.out 2>&1
  rc=$?
  set -e
  grep -q 'Failed to open HDF5 file' bench_hdf5.out ||
    fail "bench_hdf5 did not reach HDF5 (exit $rc): $(cat bench_hdf5.out)"
  echo "ok: bench_hdf5 initialised HDF5"
fi

echo "verify.sh: $base OK ($(uname -sm))"
