#!/usr/bin/env bash
# Self-test of the release shell logic (#366): `make release-selftest`.
#
# Runs in the release workflow's PR check, so every branch of the scripts that
# only execute on a real release — publish.sh's refuse / clobber / resume
# decisions above all — is exercised on every PR that touches them. publish.sh
# talks to a FAKE `gh` here that keeps a release's assets in a directory and
# behaves like the real one where it matters (upload refuses an existing name
# without --clobber; the listing carries `sha256:` digests).
#
# Needs bash, coreutils, GNU tar (gtar on macOS: `brew install gnu-tar`),
# gzip and git. No network, no cargo (a fake one stands in).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"
require_gnu_tar || exit 1

t="$(mktemp -d)"
trap 'rm -rf "$t"' EXIT
pass=0 failed=0
ok() { pass=$((pass + 1)); echo "  ok   $1"; }
bad() { failed=$((failed + 1)); echo "  FAIL $1" >&2; }
expect() { # expect <name> <0|1> <cmd...>   0 = must succeed, 1 = must fail
  local name="$1" want="$2" rc=0
  shift 2
  "$@" >"$t/out" 2>&1 || rc=$?
  if [[ "$want" == 0 && "$rc" == 0 ]] || [[ "$want" == 1 && "$rc" != 0 ]]; then ok "$name"
  else bad "$name (exit $rc)"; sed 's/^/       /' "$t/out" >&2; fi
}

echo "tag_matches_version / check_tag"
expect "0.1.0 ~ 0.1.0" 0 tag_matches_version 0.1.0 0.1.0
expect "v0.1.0 ~ 0.1.0" 0 tag_matches_version v0.1.0 0.1.0
expect "0.1.0-rc.1 ~ 0.1.0" 0 tag_matches_version 0.1.0-rc.1 0.1.0
expect "0.1.0+build ~ 0.1.0" 0 tag_matches_version 0.1.0+build.7 0.1.0
expect "0.10.0 !~ 0.1.0" 1 tag_matches_version 0.10.0 0.1.0
expect "0.1.0.1 !~ 0.1.0" 1 tag_matches_version 0.1.0.1 0.1.0
expect "0.1.01 !~ 0.1.0" 1 tag_matches_version 0.1.01 0.1.0
expect "vv0.1.0 !~ 0.1.0" 1 tag_matches_version vv0.1.0 0.1.0
expect "check_tag 0.2.0-rc.1" 0 check_tag 0.2.0-rc.1
expect "check_tag rejects a/b" 1 check_tag a/b
expect "check_tag rejects ../x" 1 check_tag ../x
expect "check_tag rejects empty" 1 check_tag ""
expect "platform_for_target rejects riscv" 1 platform_for_target riscv64gc-unknown-linux-gnu

echo "check_release_tag (the plan job's gate, before any build)"
expect "release 0.2.0 for 0.2.0" 0 check_release_tag 0.2.0 0.2.0
expect "release v0.2.0 for 0.2.0" 0 check_release_tag v0.2.0 0.2.0
expect "release 0.2.0-rc.1 for 0.2.0" 0 check_release_tag 0.2.0-rc.1 0.2.0
expect "release 0.2.0 for crate 0.1.0 refused" 1 check_release_tag 0.2.0 0.1.0
if grep -q 'does not name the crate version' "$t/out"; then ok "version refusal says why"; else bad "version refusal message: $(cat "$t/out")"; fi
expect "release 0.2.0+build.1 refused (asset rename)" 1 check_release_tag 0.2.0+build.1 0.2.0
if grep -q "contains '+'" "$t/out"; then ok "'+' refusal says why"; else bad "'+' refusal message: $(cat "$t/out")"; fi
expect "release tag a/b refused" 1 check_release_tag a/b 0.2.0

echo "checksums.sh"
d="$t/cs"
mkdir -p "$d"
printf 'a' >"$d/vector-db-benchmark-T-linux-x86_64.tar.gz"
printf 'b' >"$d/vector-db-benchmark-T-linux-aarch64.tar.gz"
expect "write" 0 bash "$here/checksums.sh" write "$d"
expect "check a fresh set" 0 bash "$here/checksums.sh" check "$d"
printf 'x' >>"$d/vector-db-benchmark-T-linux-x86_64.tar.gz"
expect "tampered archive" 1 bash "$here/checksums.sh" check "$d"
bash "$here/checksums.sh" write "$d" >/dev/null
printf 'c' >"$d/vector-db-benchmark-T-macos-aarch64.tar.gz"
expect "archive not listed (different set)" 1 bash "$here/checksums.sh" check "$d"
rm "$d/vector-db-benchmark-T-macos-aarch64.tar.gz" "$d/vector-db-benchmark-T-linux-aarch64.tar.gz"
expect "listed archive missing (different set)" 1 bash "$here/checksums.sh" check "$d"
bash "$here/checksums.sh" write "$d" >/dev/null
{ cat "$d/SHA256SUMS"; head -n1 "$d/SHA256SUMS"; } >"$t/sums" && mv "$t/sums" "$d/SHA256SUMS"
expect "duplicate line" 1 bash "$here/checksums.sh" check "$d"

echo "verify.sh refuses unsafe archives before extracting"
mk_evil() { # mk_evil <archive> <member name>  — a tar whose member is named literally
  local src="$t/evilsrc"
  rm -rf "$src"; mkdir -p "$src"; printf 'x' >"$src/f"
  "$TAR" -C "$src" --transform "s|^f\$|$2|" -czf "$1" f
}
ev="$t/ev"; mkdir -p "$ev/tmp"
mk_evil "$ev/vector-db-benchmark-T-linux-x86_64.tar.gz" "../escaped"
# verify.sh extracts into a mktemp dir under $TMPDIR, so `../escaped` would
# land in $ev/tmp.
expect "member ../escaped" 1 env TMPDIR="$ev/tmp" bash "$here/verify.sh" "$ev/vector-db-benchmark-T-linux-x86_64.tar.gz" --expect-sha 0123456789abcdef0123456789abcdef01234567
if grep -q 'escapes' "$t/out"; then ok "refused for the traversal"; else bad "traversal refusal reason: $(cat "$t/out")"; fi
if [[ ! -e "$ev/tmp/escaped" ]]; then ok "nothing written outside the extraction dir"; else bad "a member escaped"; fi
mk_evil "$ev/vector-db-benchmark-T-linux-x86_64.tar.gz" "a/../../escaped2"
expect "member a/../../escaped2" 1 bash "$here/verify.sh" "$ev/vector-db-benchmark-T-linux-x86_64.tar.gz" --expect-sha 0123456789abcdef0123456789abcdef01234567
lsrc="$t/lsrc"; mkdir -p "$lsrc"; ln -s /etc/passwd "$lsrc/vector-db-benchmark"
"$TAR" -C "$lsrc" -czf "$ev/vector-db-benchmark-T-linux-x86_64.tar.gz" vector-db-benchmark
expect "symlink member" 1 bash "$here/verify.sh" "$ev/vector-db-benchmark-T-linux-x86_64.tar.gz" --expect-sha 0123456789abcdef0123456789abcdef01234567
if grep -q 'link' "$t/out"; then ok "refused for the link, not something else"; else bad "symlink refusal reason: $(cat "$t/out")"; fi

# A link FIRST in a listing far larger than a pipe buffer: a `tar tv | grep -q`
# check would stop reading at the match, tar would take SIGPIPE, and under
# pipefail the `if` would read the failure as "no link" and extract anyway.
bsrc="$t/bsrc"; mkdir -p "$bsrc"
ln -s /etc/passwd "$bsrc/aaa-link"
for i in $(seq 1 3000); do : >"$bsrc/f$i"; done
(cd "$bsrc" && "$TAR" -czf "$ev/vector-db-benchmark-T-linux-x86_64.tar.gz" aaa-link f*)
expect "link first in a long listing" 1 bash "$here/verify.sh" "$ev/vector-db-benchmark-T-linux-x86_64.tar.gz" --expect-sha 0123456789abcdef0123456789abcdef01234567
if grep -q 'link' "$t/out"; then ok "long listing: refused for the link"; else bad "long-listing link refusal reason: $(tail -n2 "$t/out")"; fi

echo "verify.sh against stand-in archives"
# A well-formed archive whose "binaries" are scripts answering exactly what
# verify.sh asks, so every refusal below is checked against a positive control
# that passes, and fails only for the one thing each case breaks.
fake_sha=0123456789abcdef0123456789abcdef01234567
other_sha=fedcba9876543210fedcba9876543210fedcba98
mk_fake() { # mk_fake <out dir> <tag> <BUILD-INFO sha> <--version sha> <describe sha> <features>
  local out="$1" tag="$2" info_sha="$3" ver_sha="$4" desc_sha="$5" feats="$6" st="$t/fake-stage" b
  rm -rf "$st"; mkdir -p "$st/datasets/random-100" "$st/experiments/configurations" "$out"
  for b in "${REQUIRED_BINS[@]}"; do
    cat >"$st/$b" <<FAKEBIN
#!/usr/bin/env bash
case "\${1:-}" in
  --version) echo "$b 0.1.0 (git $ver_sha)" ;;
  --help) echo usage ;;
  --describe)
    if [[ "\$2" == build ]]; then
      printf '{\n  "git_sha": "%s",\n  "git_tree": "clean",\n  "features": [\n    "%s"\n  ],\n  "openssl": "OpenSSL 3.6.3",\n  "hdf5": "1.10.7"\n}\n' "$desc_sha" "$feats"
    else echo random-100; fi ;;
  --out-dir) mkdir -p "\$2/synthetic-hybrid-16" && : >"\$2/synthetic-hybrid-16/vectors.npy" ;;
  *) if [[ "$b" == bench_hdf5 ]]; then echo "Failed to open HDF5 file"; else echo "Throughput: 1 vec/s"; fi ;;
esac
FAKEBIN
    chmod +x "$st/$b"
  done
  echo '[{"name":"random-100"}]' >"$st/datasets/datasets.json"
  echo '{}' >"$st/datasets/random-100/vectors.jsonl"
  : >"$st/LICENSE"; : >"$st/README.md"
  printf 'archive: x\ntag: %s\ncrate_version: 0.1.0\ngit_sha: %s\nfeatures: release-static\nopenssl: OpenSSL 3.6.3\nhdf5: 1.10.7\nbinaries: %s\n' \
    "$tag" "$info_sha" "${REQUIRED_BINS[*]}" >"$st/BUILD-INFO.txt"
  "$TAR" -C "$st" -czf "$out/vector-db-benchmark-$tag-linux-x86_64.tar.gz" .
}
vf="$t/vf"
verify() { bash "$here/verify.sh" "$vf/vector-db-benchmark-$1-linux-x86_64.tar.gz" --expect-sha "$fake_sha" --expect-tag "${2:-$1}"; }
mk_fake "$vf" T "$fake_sha" "$fake_sha" "$fake_sha" release-static
expect "positive control: a well-formed archive passes" 0 verify T
expect "wrong --expect-tag refused" 1 verify T U
if grep -q "archive is for tag 'T', expected 'U'" "$t/out"; then ok "  ... for the tag"; else bad "tag refusal reason: $(tail -n1 "$t/out")"; fi
mk_fake "$vf" T "$other_sha" "$fake_sha" "$fake_sha" release-static
expect "BUILD-INFO naming another commit refused" 1 verify T
if grep -q 'BUILD-INFO.txt git_sha' "$t/out"; then ok "  ... for BUILD-INFO's sha"; else bad "sha refusal reason: $(tail -n1 "$t/out")"; fi
mk_fake "$vf" T "$fake_sha" "$other_sha" "$fake_sha" release-static
expect "--version naming another commit refused" 1 verify T
if grep -q -- '--version printed' "$t/out"; then ok "  ... for --version"; else bad "--version refusal reason: $(tail -n1 "$t/out")"; fi
mk_fake "$vf" T "$fake_sha" "$fake_sha" "$fake_sha" none
expect "--describe build without release-static refused" 1 verify T
if grep -q 'release-static feature' "$t/out"; then ok "  ... for the features"; else bad "features refusal reason: $(tail -n1 "$t/out")"; fi

echo "package.sh gates, before cargo runs"
# A fake cargo that records being reached: every case below must refuse first,
# and the positive control must get as far as cargo.
fakecargo="$t/fakecargo"; mkdir -p "$fakecargo"
printf '#!/usr/bin/env bash\necho "FAKE CARGO REACHED $*"\nexit 1\n' >"$fakecargo/cargo"; chmod +x "$fakecargo/cargo"
pkg() { (cd "$1" && PATH="$fakecargo:$PATH" bash scripts/release/package.sh --target x86_64-unknown-linux-gnu --out "$t/pkg-out" "${@:2}"); }
# The fake cargo also exits 1, so a refusal only counts if it names the dirty
# tree and cargo was never reached.
refused_dirty() {
  if grep -q 'uncommitted changes' "$t/out" && ! grep -q 'FAKE CARGO' "$t/out"; then ok "  ... as uncommitted changes, before cargo"
  else bad "dirty gate: $(cat "$t/out")"; fi
}
# A throwaway repository holding the release scripts, a Cargo.toml and a source
# file, so the dirty-tree cases cannot touch this checkout.
pr="$t/pkgrepo"; mkdir -p "$pr/scripts/release" "$pr/src"
cp "$here"/*.sh "$pr/scripts/release/"
printf '[package]\nname = "x"\nversion = "0.1.0"\n' >"$pr/Cargo.toml"
echo 'fn main() {}' >"$pr/src/main.rs"; echo readme >"$pr/README.md"
git -C "$pr" init -q
git -C "$pr" add -A
git -C "$pr" -c user.email=selftest@example.invalid -c user.name=selftest commit -qm init
expect "--release with a tag that is not the crate version refused" 1 pkg "$pr" --tag 9.9.9 --release
if grep -q 'does not name the crate version' "$t/out" && ! grep -q 'FAKE CARGO' "$t/out"; then ok "  ... before cargo"; else bad "tag gate: $(cat "$t/out")"; fi
expect "--release with a '+' tag refused" 1 pkg "$pr" --tag 0.1.0+b --release
if ! grep -q 'FAKE CARGO' "$t/out"; then ok "  ... before cargo"; else bad "'+' tag reached cargo"; fi
echo '// edit' >>"$pr/src/main.rs"
expect "modified compiled source refused" 1 pkg "$pr" --tag T
refused_dirty
git -C "$pr" checkout -q -- src/main.rs
echo 'pub fn x() {}' >"$pr/src/new.rs"
expect "untracked file under src/ refused (it would be compiled in)" 1 pkg "$pr" --tag T
refused_dirty
rm "$pr/src/new.rs"
echo edit >>"$pr/README.md"
expect "modified shipped README refused" 1 pkg "$pr" --tag T
refused_dirty
git -C "$pr" checkout -q -- README.md
echo notes >"$pr/notes.txt"
expect "positive control: a clean tree reaches cargo (untracked notes.txt is not an input)" 1 pkg "$pr" --tag 0.1.0 --release
if grep -q 'FAKE CARGO REACHED build --release --locked --features release-static' "$t/out"; then ok "  ... and builds release-static, --locked"; else bad "clean tree did not reach cargo: $(cat "$t/out")"; fi

echo "check_portable_linkage"
if [[ "$(uname -s)" == Linux ]] && command -v readelf >/dev/null && command -v curl >/dev/null; then
  # curl links libcurl, libssl …: exactly what a release binary must not need.
  expect "refuses a host binary linking system libraries ($(command -v curl))" 1 check_portable_linkage "$(command -v curl)" x86_64-unknown-linux-gnu
  if grep -q 'which a fresh host may not have' "$t/out"; then ok "  ... naming the library"; else bad "linkage refusal: $(cat "$t/out")"; fi
else
  echo "  skip (needs Linux, binutils and curl)"
fi

echo "in_list"
expect "in_list finds an exact member" 0 in_list b a b c
expect "in_list is exact, not a prefix" 1 in_list b ab bc
expect "in_list on an empty list" 1 in_list a

echo "publish.sh against a fake gh"
fake="$t/fakebin"; mkdir -p "$fake"
cat >"$fake/gh" <<'FAKE'
#!/usr/bin/env bash
# Minimal stand-in for gh: the release's assets live in $FAKE_REL.
set -euo pipefail
echo "gh $*" >>"$FAKE_LOG"
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
case "$1 $2" in
  "api repos/"*)
    [[ -z "${FAKE_FAIL_API:-}" ]] || { echo "HTTP 502: Bad Gateway" >&2; exit 1; }
    for f in "$FAKE_REL"/*; do
      [[ -e "$f" ]] || continue
      n="$(basename "$f")"
      if [[ -n "${FAKE_INCOMPLETE:-}" && "$n" == "$FAKE_INCOMPLETE" ]]; then printf '%s\tstarter\t\n' "$n"
      elif [[ -n "${FAKE_NO_DIGEST:-}" && "$n" == "$FAKE_NO_DIGEST" ]]; then printf '%s\tuploaded\t\n' "$n"
      else printf '%s\tuploaded\tsha256:%s\n' "$n" "$(sha "$f")"; fi
    done ;;
  "release upload")
    shift 3; clobber=0; files=()
    while [[ $# -gt 0 ]]; do
      case "$1" in --clobber) clobber=1; shift ;; -R) shift 2 ;; *) files+=("$1"); shift ;; esac
    done
    for f in "${files[@]}"; do
      if [[ -e "$FAKE_REL/$(basename "$f")" && $clobber == 0 ]]; then
        echo "asset under the same name already exists: $(basename "$f")" >&2; exit 1
      fi
    done
    for f in "${files[@]}"; do cp "$f" "$FAKE_REL/"; done ;;
  "release delete-asset")
    rm "$FAKE_REL/$4" ;;
  *) echo "fake gh: unsupported: $*" >&2; exit 2 ;;
esac
FAKE
chmod +x "$fake/gh"

set_dist() { # a fresh "build": two archives with content $1, plus SHA256SUMS
  rm -rf "$t/dist"; mkdir -p "$t/dist"
  printf 'x86-%s' "$1" >"$t/dist/vector-db-benchmark-T-linux-x86_64.tar.gz"
  printf 'arm-%s' "$1" >"$t/dist/vector-db-benchmark-T-linux-aarch64.tar.gz"
  bash "$here/checksums.sh" write "$t/dist" >/dev/null
}
reset_rel() { rm -rf "$t/rel"; mkdir -p "$t/rel"; : >"$t/log"; }
same_as_dist() { diff -r "$t/rel" "$t/dist" >/dev/null; }
publish() { GH="$fake/gh" REPO=o/r FAKE_REL="$t/rel" FAKE_LOG="$t/log" bash "$here/publish.sh" T "$t/dist"; }
last_upload_is_sums() { grep 'release upload' "$t/log" | tail -n1 | grep -q 'SHA256SUMS'; }

set_dist one; reset_rel
expect "empty release: uploads" 0 publish
if same_as_dist; then ok "empty release: serves exactly the set"; else bad "empty release: content differs"; fi
if last_upload_is_sums; then ok "empty release: SHA256SUMS uploaded last"; else bad "SHA256SUMS not last"; fi
if ! grep -q -- '--clobber' "$t/log"; then ok "empty release: no --clobber"; else bad "--clobber used without CLOBBER"; fi

: >"$t/log"
expect "identical re-run: succeeds" 0 publish
if ! grep -q 'release upload' "$t/log"; then ok "identical re-run: uploads nothing"; else bad "identical re-run uploaded"; fi

rm "$t/rel/vector-db-benchmark-T-linux-aarch64.tar.gz" "$t/rel/SHA256SUMS"; : >"$t/log"
expect "partial upload resumed" 0 publish
if same_as_dist; then ok "partial upload: completed"; else bad "partial upload: content differs"; fi
if ! grep 'release upload' "$t/log" | grep -q 'linux-x86_64'; then ok "partial upload: present archive not re-sent"; else bad "re-sent a present archive"; fi

set_dist two; : >"$t/log"
expect "different build, no clobber: refused" 1 publish
if grep -q 'Refusing to mix builds' "$t/out"; then ok "refusal names the reason"; else bad "refusal message"; fi
if ! grep -q 'release upload\|delete-asset' "$t/log"; then ok "refused run changed nothing"; else bad "refused run touched the release"; fi

: >"$t/log"
expect "different build, CLOBBER=true: replaces" 0 env CLOBBER=true GH="$fake/gh" REPO=o/r FAKE_REL="$t/rel" FAKE_LOG="$t/log" bash "$here/publish.sh" T "$t/dist"
if same_as_dist; then ok "clobber: serves exactly the new set"; else bad "clobber: content differs"; fi
first_mut="$(grep -m1 -E 'release (upload|delete-asset)' "$t/log")"
if [[ "$first_mut" == *"delete-asset T SHA256SUMS"* ]]; then ok "clobber: old SHA256SUMS deleted before any archive changes"; else bad "clobber order: $first_mut"; fi
if last_upload_is_sums; then ok "clobber: new SHA256SUMS uploaded last"; else bad "clobber: SHA256SUMS not last"; fi

printf 'old' >"$t/rel/vector-db-benchmark-T-macos-aarch64.tar.gz"; : >"$t/log"
expect "stale archive of the tag, no clobber: refused" 1 publish
: >"$t/log"
expect "stale archive, CLOBBER=true" 0 env CLOBBER=true GH="$fake/gh" REPO=o/r FAKE_REL="$t/rel" FAKE_LOG="$t/log" bash "$here/publish.sh" T "$t/dist"
if same_as_dist; then ok "clobber: stale archive deleted"; else bad "stale archive survived"; fi

printf 'notes' >"$t/rel/release-notes.txt"; : >"$t/log"
expect "unrelated asset left alone" 0 publish
if [[ -e "$t/rel/release-notes.txt" ]]; then ok "unrelated asset kept"; else bad "deleted an unrelated asset"; fi
rm "$t/rel/release-notes.txt"

: >"$t/log"
expect "unknown digest counts as different" 1 env FAKE_NO_DIGEST=SHA256SUMS GH="$fake/gh" REPO=o/r FAKE_REL="$t/rel" FAKE_LOG="$t/log" bash "$here/publish.sh" T "$t/dist"
if grep -q 'no digest reported' "$t/out"; then ok "unknown digest: says so, not 'different digest'"; else bad "unknown-digest wording: $(cat "$t/out")"; fi

# A runner that died mid-upload leaves a half-written asset (state "starter",
# no digest). It is not served, so a plain re-run replaces it without clobber.
printf 'half' >"$t/rel/vector-db-benchmark-T-linux-aarch64.tar.gz"; rm -f "$t/rel/SHA256SUMS"; : >"$t/log"
expect "unfinished upload replaced without CLOBBER" 0 env FAKE_INCOMPLETE=vector-db-benchmark-T-linux-aarch64.tar.gz GH="$fake/gh" REPO=o/r FAKE_REL="$t/rel" FAKE_LOG="$t/log" bash "$here/publish.sh" T "$t/dist"
if same_as_dist; then ok "unfinished upload: release completed with this build"; else bad "unfinished upload: content differs"; fi
if ! grep -q -- '--clobber' "$t/log"; then ok "unfinished upload: no --clobber"; else bad "unfinished upload used --clobber"; fi
printf 'half' >"$t/rel/vector-db-benchmark-T-macos-aarch64.tar.gz"; : >"$t/log"
expect "unfinished upload of an archive not in this set is removed too" 0 env FAKE_INCOMPLETE=vector-db-benchmark-T-macos-aarch64.tar.gz GH="$fake/gh" REPO=o/r FAKE_REL="$t/rel" FAKE_LOG="$t/log" bash "$here/publish.sh" T "$t/dist"
if same_as_dist; then ok "  ... and the release serves exactly this set"; else bad "unfinished stale archive survived"; fi

reset_rel
expect "API failure aborts" 1 env FAKE_FAIL_API=1 GH="$fake/gh" REPO=o/r FAKE_REL="$t/rel" FAKE_LOG="$t/log" bash "$here/publish.sh" T "$t/dist"
if ! grep -q 'release upload' "$t/log"; then ok "API failure: nothing uploaded"; else bad "uploaded after an API failure"; fi
printf 'existing' >"$t/rel/vector-db-benchmark-T-linux-x86_64.tar.gz"; : >"$t/log"
expect "API failure with CLOBBER=true still aborts" 1 env CLOBBER=true FAKE_FAIL_API=1 GH="$fake/gh" REPO=o/r FAKE_REL="$t/rel" FAKE_LOG="$t/log" bash "$here/publish.sh" T "$t/dist"
if [[ "$(cat "$t/rel/vector-db-benchmark-T-linux-x86_64.tar.gz")" == existing ]]; then ok "API failure: existing asset untouched"; else bad "overwrote after an API failure"; fi

printf 'x' >>"$t/dist/vector-db-benchmark-T-linux-x86_64.tar.gz"; reset_rel
expect "set failing checksums.sh check is not published" 1 publish
if ! grep -q 'release upload' "$t/log"; then ok "bad set: nothing uploaded"; else bad "uploaded a set that fails its own SHA256SUMS"; fi

echo
echo "release selftest: $pass passed, $failed failed"
[[ "$failed" == 0 ]]
