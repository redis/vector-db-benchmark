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
# Needs bash, coreutils, tar, gzip. No network, no cargo.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"

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
  tar -C "$src" --transform "s|^f\$|$2|" -czf "$1" f
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
tar -C "$lsrc" -czf "$ev/vector-db-benchmark-T-linux-x86_64.tar.gz" vector-db-benchmark
expect "symlink member" 1 bash "$here/verify.sh" "$ev/vector-db-benchmark-T-linux-x86_64.tar.gz" --expect-sha 0123456789abcdef0123456789abcdef01234567
if grep -q 'link' "$t/out"; then ok "refused for the link, not something else"; else bad "symlink refusal reason: $(cat "$t/out")"; fi

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
      if [[ -n "${FAKE_NO_DIGEST:-}" && "$n" == "$FAKE_NO_DIGEST" ]]; then printf '%s\t\n' "$n"
      else printf '%s\tsha256:%s\n' "$n" "$(sha "$f")"; fi
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
