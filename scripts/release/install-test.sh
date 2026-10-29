#!/usr/bin/env bash
# Drive scripts/install-release.sh against a local stand-in for GitHub releases (#366).
#
#   scripts/release/install-test.sh <dir with the archives + SHA256SUMS> <tag> <sha>
#
# The README one-liner runs the installer with NO tag: it follows
# /releases/latest to find one. That path, and every refusal, is exercised here
# against a local HTTP server (python3's http.server) that answers like
# github.com does:
#
#   * latest resolves, archive + SHA256SUMS present  -> installs, --version names <sha>
#   * an explicit tag                                 -> installs
#   * no release at all (/latest lands on /releases)  -> "no published release"
#   * the latest release has no archives (a bare 0.1.0) -> "has no prebuilt archive"
#   * SHA256SUMS does not list this platform          -> "does not list"
#   * the archive does not match SHA256SUMS           -> "checksum mismatch"
#   * an archive member outside the destination       -> refused before extracting
#   * a link member                                    -> refused before extracting
#   * a non-empty destination                         -> refused
#   * glibc older than 2.34 (Linux)                   -> refused before downloading
#
# Runs in the release workflow's PR check on a Linux runner.
set -euo pipefail

src="${1:-}" tag="${2:-}" sha="${3:-}"
[[ -d "$src" && -n "$tag" && "$sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "usage: install-test.sh <archives dir> <tag> <sha>" >&2; exit 2; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
installer="$here/../install-release.sh"
src="$(cd "$src" && pwd)"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) plat=linux-x86_64 ;;
  Linux-aarch64) plat=linux-aarch64 ;;
  Darwin-arm64) plat=macos-aarch64 ;;
  *) echo "install-test.sh: unsupported host $(uname -sm)" >&2; exit 2 ;;
esac
name="vector-db-benchmark-$tag-$plat.tar.gz"
[[ -f "$src/$name" && -f "$src/SHA256SUMS" ]] || { echo "install-test.sh: $src lacks $name or SHA256SUMS" >&2; exit 2; }

w="$(mktemp -d)"
srv="$w/srv"
pid=""
cleanup() {
  if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  rm -rf "$w"
}
trap cleanup EXIT

# One directory per scenario, laid out like github.com/<repo>/releases:
# <scenario>/download/<tag>/<asset>, plus a `latest` file naming where
# /<scenario>/latest redirects to.
release() { # release <scenario> <latest target> [files...]
  local s="$1" target="$2"; shift 2
  mkdir -p "$srv/$s/download/$tag"
  printf '%s' "$target" >"$srv/$s/latest"
  for f in "$@"; do cp "$src/$f" "$srv/$s/download/$tag/"; done
}
release ok "tag/$tag" "$name" SHA256SUMS
release none "releases"
release old "tag/0.0.1"
release nolist "tag/$tag" "$name"
grep -vF "  $name" "$src/SHA256SUMS" >"$srv/nolist/download/$tag/SHA256SUMS" || true
release corrupt "tag/$tag" "$name" SHA256SUMS
printf 'x' >>"$srv/corrupt/download/$tag/$name"
release evil "tag/$tag"
python3 -I - "$srv/evil/download/$tag/$name" <<'PY'
import io, sys, tarfile
with tarfile.open(sys.argv[1], "w:gz") as t:
    data = b"x"
    info = tarfile.TarInfo("../escaped"); info.size = len(data)
    t.addfile(info, io.BytesIO(data))
PY
sums_for() { (cd "$1" && { if command -v sha256sum >/dev/null; then sha256sum "$name"; else shasum -a 256 "$name"; fi; } >SHA256SUMS); }
sums_for "$srv/evil/download/$tag"
# A link member named like the binary: extracting it would make
# ./vector-db-benchmark point anywhere on the host.
release link "tag/$tag"
python3 -I - "$srv/link/download/$tag/$name" <<'PY'
import sys, tarfile
with tarfile.open(sys.argv[1], "w:gz") as t:
    info = tarfile.TarInfo("vector-db-benchmark"); info.type = tarfile.SYMTYPE; info.linkname = "/bin/sh"
    t.addfile(info)
PY
sums_for "$srv/link/download/$tag"

# The stand-in server: /<scenario>/latest redirects as GitHub does, every
# other path is served from $srv (a missing asset is a 404). Listens on a
# free port it reports through $w/port.
python3 -I - "$srv" "$w/port" <<'PY' &
import functools, http.server, os, sys
root, port_file = sys.argv[1], sys.argv[2]
class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        parts = self.path.strip("/").split("/")
        if len(parts) == 2 and parts[1] == "latest":
            with open(os.path.join(root, parts[0], "latest")) as f:
                target = f.read()
            self.send_response(302)
            self.send_header("Location", f"/{parts[0]}/{target}")
            self.end_headers()
            return
        if len(parts) >= 2 and parts[1] in ("tag", "releases"):
            self.send_response(200); self.end_headers(); self.wfile.write(b"release page"); return
        super().do_GET()
s = http.server.HTTPServer(("127.0.0.1", 0), functools.partial(H, directory=root))
with open(port_file + ".tmp", "w") as f:
    f.write(str(s.server_address[1]))
os.rename(port_file + ".tmp", port_file)
s.serve_forever()
PY
pid=$!
for _ in $(seq 1 50); do [[ -s "$w/port" ]] && break; sleep 0.1; done
[[ -s "$w/port" ]] || { echo "install-test.sh: stub server did not start" >&2; exit 1; }
url="http://127.0.0.1:$(cat "$w/port")"

pass=0 failed=0
ok() { pass=$((pass + 1)); echo "  ok   $1"; }
bad() { failed=$((failed + 1)); echo "  FAIL $1" >&2; sed 's/^/       /' "$w/out" >&2; }
# install <name> <scenario> <dest> [tag]  -> exit status, output in $w/out
install() {
  local rc=0
  VDB_RELEASE_BASE_URL="$url/$2" bash "$installer" "${4:-}" "$3" >"$w/out" 2>&1 || rc=$?
  return "$rc"
}
refused() { # refused <name> <scenario> <message> [tag]
  local d="$w/dest-$2"
  if install "$1" "$2" "$d" "${4:-}"; then bad "$1: installed, expected a refusal"
  elif ! grep -qF -- "$3" "$w/out"; then bad "$1: refused, but not with '$3'"
  elif [[ -e "$d/vector-db-benchmark" ]]; then bad "$1: refused, but left a binary in $d"
  else ok "$1"; fi
}

if install "no tag: resolves /latest and installs" ok "$w/latest"; then
  v="$("$w/latest/vector-db-benchmark" --version)"
  if [[ "$v" == *"(git $sha)" ]]; then ok "no tag: resolves /latest and installs ($v)"; else bad "no tag: --version '$v'"; fi
else bad "no tag: resolves /latest and installs"; fi
if install "explicit tag" ok "$w/explicit" "$tag" && [[ -x "$w/explicit/vector-db-benchmark" ]]; then ok "explicit tag installs"
else bad "explicit tag installs"; fi

refused "no release at all" none "has no published release"
refused "latest release has no archives" old "has no prebuilt archive"
refused "SHA256SUMS does not list the archive" nolist "does not list $name"
refused "archive does not match SHA256SUMS" corrupt "checksum mismatch" "$tag"
refused "archive member outside the destination" evil "outside the destination" "$tag"
if [[ ! -e "$w/escaped" ]]; then ok "evil archive: nothing written outside"; else bad "evil archive escaped"; fi
refused "archive with a link member" link "link or special member" "$tag"

# An old glibc is refused before anything is downloaded (a fake getconf
# reports glibc 2.31, Ubuntu 20.04's). Linux only: the check is.
if [[ "$(uname -s)" == Linux ]]; then
  mkdir -p "$w/oldlibc"; printf '#!/bin/sh\necho "glibc 2.31"\n' >"$w/oldlibc/getconf"; chmod +x "$w/oldlibc/getconf"
  rc=0; PATH="$w/oldlibc:$PATH" VDB_RELEASE_BASE_URL="$url/ok" bash "$installer" "$tag" "$w/oldglibc" >"$w/out" 2>&1 || rc=$?
  if [[ "$rc" != 0 ]] && grep -qF "has glibc 2.31" "$w/out" && ! grep -qF Downloading "$w/out"; then ok "glibc 2.31 refused before downloading"
  else bad "glibc 2.31 (exit $rc)"; fi
fi

mkdir -p "$w/full"; : >"$w/full/keep"
if install "non-empty destination" ok "$w/full" "$tag"; then bad "non-empty destination: installed over it"
elif grep -qF "is not empty" "$w/out" && [[ ! -e "$w/full/vector-db-benchmark" ]]; then ok "non-empty destination refused"
else bad "non-empty destination: refused for another reason"; fi

echo
echo "install-test: $pass passed, $failed failed"
[[ "$failed" == 0 ]]
