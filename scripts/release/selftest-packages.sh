#!/usr/bin/env bash
# Self-test of the package publish scripts: `make release-selftest-packages`.
#
# publish-apt.sh and publish-yum.sh only run for real during a publish, into
# repositories shared with every other Redis product, so every decision they
# make is exercised here first, on every PR, against FAKES:
#   aws     `s3 sync` / `s3 cp` against a directory standing in for the bucket
#   deb-s3  records its arguments, keeps a per-codename manifest, honours
#           --fail-if-exists, REFUSES a bare `--sign` (a string option in
#           deb-s3: bare, it would swallow the first .deb as the key id), and
#           clearsigns the codename's InRelease with the --sign key
# with tiny .deb / .rpm fixtures and a throwaway GPG key. Covered: grouping by
# ~codename and refusing a .deb without one; dist tag -> rpm/<distro>/<arch>;
# refusing unmapped tags; signing mandatory, --dry-run without a key or tools;
# re-run idempotency (same build skipped, different build refused, --overwrite);
# RPMs uploaded before repodata; another product's RPM in the same directory
# kept and indexed; another product's RPM uploaded DURING our publish (caught
# by the re-list, or failing the post-upload index check with a re-run hint);
# lib.sh's path-style S3 URL for a dotted bucket; which bucket each environment
# may publish to (production is our own vdb-packages, never staging or
# packages.redis.io); the APT Origin; reading the fingerprints of a <base>/gpg.
#
# Needs: bash, gpg, dpkg-deb, rpm, rpmbuild, createrepo_c, gzip. No network.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"
for tool in gpg dpkg-deb rpm rpmbuild createrepo_c gzip; do
  command -v "$tool" >/dev/null || { echo "selftest-packages.sh: $tool not on PATH" >&2; exit 2; }
done

t="$(mktemp -d)"
trap 'rm -rf "$t"' EXIT
pass=0 failed=0
ok() { pass=$((pass + 1)); echo "  ok   $1"; }
bad() { failed=$((failed + 1)); echo "  FAIL $1" >&2; }
expect() { # expect <name> <0|1> <cmd...>   0 = must succeed, 1 = must fail
  local name="$1" want="$2" rc=0
  shift 2
  "$@" > "$t/last.out" 2>&1 || rc=$?
  if [[ ( "$want" == 0 && "$rc" == 0 ) || ( "$want" != 0 && "$rc" != 0 ) ]]; then ok "$name"; else
    bad "$name (exit $rc, wanted $([[ "$want" == 0 ]] && echo success || echo failure))"; sed 's/^/       | /' "$t/last.out" >&2; fi
}
check() { # check <name> <condition...>
  local name="$1"; shift
  if "$@"; then ok "$name"; else bad "$name"; fi
}

# ---- fakes -------------------------------------------------------------------
export FAKE_S3="$t/s3" FAKE_LOG="$t/fake.log"
mkdir -p "$t/bin" "$FAKE_S3"
cat > "$t/bin/aws" <<'FAKE'
#!/usr/bin/env bash
# fake aws: s3 sync|cp|ls with s3://bucket/key mapped to $FAKE_S3/bucket/key.
# FAKE_LS_FAIL (a regex on the s3:// argument) makes that `ls` fail as an
# AccessDenied would (stderr, exit 254).
# A concurrent publish by another product is simulated with FAKE_RACE_MATCH (a
# regex on the arguments) + FAKE_RACE_SRC (a file) + FAKE_RACE_DST (s3://...):
# the first call matching moves the file into the bucket BEFORE it runs.
set -euo pipefail
echo "aws $*" >> "$FAKE_LOG"
map() { local p="$1"; [[ "$p" == s3://* ]] && p="$FAKE_S3/${p#s3://}"; echo "$p"; }
if [[ -n "${FAKE_RACE_MATCH:-}" && "$*" =~ $FAKE_RACE_MATCH && -f "${FAKE_RACE_SRC:-}" ]]; then
  mkdir -p "$(dirname "$(map "$FAKE_RACE_DST")")"; mv "$FAKE_RACE_SRC" "$(map "$FAKE_RACE_DST")"
fi
[[ "$1" == s3 ]] || { echo "fake aws: only s3" >&2; exit 1; }
if [[ "$2" == ls && -n "${FAKE_LS_FAIL:-}" && "$3" =~ $FAKE_LS_FAIL ]]; then
  echo "An error occurred (AccessDenied) when calling the ListObjectsV2 operation: Access Denied" >&2; exit 254
fi
if [[ "$2" == ls ]]; then
  # Direct children, as `aws s3 ls s3://b/dir/` prints them; exit 1, no
  # output, for a prefix with nothing under it.
  d="$(map "$3")"; out=""
  if [[ -f "$d" ]]; then
    out="$(printf '2026-10-29 07:00:00 %10d %s' "$(wc -c < "$d")" "$(basename "$d")")"
  elif [[ -d "$d" ]]; then
    out="$(cd "$d" && for e in *; do
      [[ -e "$e" ]] || continue
      if [[ -d "$e" ]]; then printf '                           PRE %s/\n' "$e"
      else printf '2026-10-29 07:00:00 %10d %s\n' "$(wc -c < "$e")" "$e"; fi; done)"
  fi
  [[ -n "$out" ]] || exit 1
  echo "$out"; exit 0
fi
cmd="$2"; src="$3"; dst="$4"; shift 4
delete=0 exclude=""
while [[ $# -gt 0 ]]; do
  case "$1" in --delete) delete=1 ;; --exclude) exclude="$2"; shift ;; esac
  shift
done
src="$(map "$src")"; dst="$(map "$dst")"
case "$cmd" in
  sync)
    mkdir -p "$dst"
    if [[ -d "$src" ]]; then
      (cd "$src" && find . -type f -print0) | while IFS= read -r -d '' f; do
        # shellcheck disable=SC2053  # the exclude is a glob, as in aws
        [[ -n "$exclude" && "${f#./}" == $exclude ]] && continue
        mkdir -p "$dst/$(dirname "$f")"; cp "$src/$f" "$dst/$f"; done
    fi
    if [[ "$delete" == 1 ]]; then
      (cd "$dst" && find . -type f -print0) | while IFS= read -r -d '' f; do
        # shellcheck disable=SC2053
        [[ -n "$exclude" && "${f#./}" == $exclude ]] && continue
        [[ -f "$src/$f" ]] || rm -f "$dst/$f"; done
    fi ;;
  cp) mkdir -p "$(dirname "$dst")"; cp "$src" "$dst" ;;
  *) echo "fake aws: unknown s3 command $cmd" >&2; exit 1 ;;
esac
FAKE
cat > "$t/bin/deb-s3" <<'FAKE'
#!/usr/bin/env bash
# fake deb-s3 upload: manifest per codename under $FAKE_S3/<bucket>/<prefix>/dists/<codename>/
set -euo pipefail
echo "deb-s3 $*" >> "$FAKE_LOG"
[[ "$1" == upload ]] || { echo "fake deb-s3: only upload" >&2; exit 1; }
shift
bucket="" prefix="" codename="" fail_if_exists=0 sign="" debs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket=*) bucket="${1#*=}" ;;
    --prefix=*) prefix="${1#*=}" ;;
    --codename=*) codename="${1#*=}" ;;
    --fail-if-exists) fail_if_exists=1 ;;
    --sign=*) sign="${1#*=}" ;;
    --sign) echo "fake deb-s3: bare --sign would take '$2' as the key id" >&2; exit 97 ;;
    --*) ;;   # other options: --suite=, --component=, --origin=, --visibility=, --preserve-versions, --lock, --s3-region=, --gpg-options=
    *) debs+=("$1") ;;
  esac
  shift
done
[[ -n "$bucket" && -n "$prefix" && -n "$codename" && -n "$sign" && ${#debs[@]} -gt 0 ]] ||
  { echo "fake deb-s3: missing bucket/prefix/codename/sign/packages" >&2; exit 98; }
dir="$FAKE_S3/$bucket/$prefix/dists/$codename"
mkdir -p "$dir"
for d in "${debs[@]}"; do
  [[ -f "$d" && "$d" == *.deb ]] || { echo "fake deb-s3: not a .deb: $d" >&2; exit 99; }
  entry="$(dpkg-deb -W --showformat='${Package} ${Version} ${Architecture}' "$d")"
  if [[ "$fail_if_exists" == 1 ]] && grep -qxF "$entry" "$dir/manifest" 2>/dev/null; then
    echo "fake deb-s3: package already exists: $entry" >&2; exit 1
  fi
  echo "$entry" >> "$dir/manifest"
done
echo "signed-with=$sign" > "$dir/Release"
# Like deb-s3: re-sign the codename's InRelease with the --sign key.
gpg --batch --yes --no-tty --local-user "$sign" --clearsign -o "$dir/InRelease" "$dir/Release"
FAKE
chmod +x "$t/bin/aws" "$t/bin/deb-s3"
export PATH="$t/bin:$PATH"

# ---- throwaway signing key -----------------------------------------------------
export GNUPGHOME="$t/gnupg"
mkdir -m 0700 "$GNUPGHOME"
gpg --batch --quiet --passphrase '' --quick-gen-key 'VDB selftest <selftest@example.invalid>' rsa2048 sign never 2>/dev/null
fpr="$(gpg --list-secret-keys --with-colons | awk -F: '/^fpr:/{print $10; exit}')"
export HOME="$t/home"; mkdir -p "$HOME"   # ~/.rpmmacros goes here, not the user's

# ---- fixtures ------------------------------------------------------------------
mkdeb() { # mkdeb <version> <arch> <out dir>
  local v="$1" a="$2" out="$3" d; d="$(mktemp -d -p "$t")"
  mkdir -p "$d/DEBIAN" "$d/usr/share/doc/vector-db-benchmark"
  printf 'Package: vector-db-benchmark\nVersion: %s\nArchitecture: %s\nMaintainer: t <t@example.invalid>\nDescription: selftest fixture\n' "$v" "$a" > "$d/DEBIAN/control"
  echo "$v $a" > "$d/usr/share/doc/vector-db-benchmark/fixture"
  mkdir -p "$out"; dpkg-deb -b "$d" "$out/vector-db-benchmark_${v}_${a}.deb" >/dev/null
}
mkrpm() { # mkrpm <version> <release-with-dist> <arch> <out dir> [payload]
  local v="$1" r="$2" a="$3" out="$4" payload="${5:-fixture}" top; top="$(mktemp -d -p "$t")"
  mkdir -p "$top"/{BUILD,RPMS,SOURCES,SPECS,BUILDROOT}
  cat > "$top/SPECS/f.spec" <<SPEC
Name: vector-db-benchmark
Version: $v
Release: $r
Summary: selftest fixture
License: ASL 2.0
AutoReqProv: no
%global debug_package %{nil}
%description
fixture
%install
mkdir -p %{buildroot}/usr/share/doc/vector-db-benchmark
echo "$payload" > %{buildroot}/usr/share/doc/vector-db-benchmark/fixture
%files
/usr/share/doc/vector-db-benchmark/fixture
SPEC
  rpmbuild -bb --quiet --target "$a" --define "_topdir $top" --define "use_source_date_epoch_as_buildtime 1" --define "_build_name_fmt %%{NAME}-%%{VERSION}-%%{RELEASE}.%%{ARCH}.rpm" "$top/SPECS/f.spec" >/dev/null 2>&1
  mkdir -p "$out"; find "$top/RPMS" -name '*.rpm' -exec cp {} "$out/" \;
}

# Every fixture RPM gets the SAME build time: two different builds of one
# NEVRA within one second are what NEVRA + BUILDTIME alone cannot tell apart.
export SOURCE_DATE_EPOCH=1700000000

echo "publish-apt.sh"
mkdeb 9.9.9~jammy amd64 "$t/debs/a"; mkdeb 9.9.9~jammy arm64 "$t/debs/b"; mkdeb 9.9.9~noble amd64 "$t/debs/c"
: > "$FAKE_LOG"
expect "publishes two codenames from one tree" 0 bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs"
check "one deb-s3 call per codename" test "$(grep -c '^deb-s3 ' "$FAKE_LOG")" -eq 2
check "jammy call carries both jammy debs and no noble one" bash -c "l=\$(grep -- '--codename=jammy' '$FAKE_LOG'); [[ \"\$l\" == *_amd64.deb* && \"\$l\" == *_arm64.deb* && \"\$l\" != *noble* ]]"
check "--sign=<fingerprint> of the one secret key" grep -q -- "--sign=$fpr" "$FAKE_LOG"
check "--suite, --origin, --fail-if-exists, --lock passed" bash -c "l=\$(grep -- '--codename=noble' '$FAKE_LOG'); [[ \"\$l\" == *--suite=noble* && \"\$l\" == *--origin=vector-db-benchmark* && \"\$l\" == *--fail-if-exists* && \"\$l\" == *--lock* ]]"
check "the default Origin is lib.sh's APT_ORIGIN" test "$APT_ORIGIN" = vector-db-benchmark
check "test-package.sh's local repository carries the same Origin" grep -qF -- "-o \"APT::FTPArchive::Release::Origin=$APT_ORIGIN\"" "$here/test-package.sh"
check "fake deb-s3 recorded the manifest" grep -qx 'vector-db-benchmark 9.9.9~noble amd64' "$FAKE_S3/bkt/deb/dists/noble/manifest"
expect "re-run of the same version is refused (--fail-if-exists)" 1 bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs"
expect "... unless --overwrite" 0 bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs" --overwrite
check "--overwrite drops --fail-if-exists" bash -c "! tail -n 2 '$FAKE_LOG' | grep -q -- --fail-if-exists"
mkdeb 9.9.9 amd64 "$t/debs-bad"
expect "an --origin with a space is refused" 1 bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs" --origin "Redis Inc" --dry-run
expect "a .deb without ~codename is refused" 1 bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs-bad"
expect "--dry-run runs deb-s3 not at all" 0 bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs" --dry-run
check "dry run printed the commands" grep -q '^dry-run: deb-s3' "$t/last.out"
# Other keyrings, passed through env (not a subshell: the pass/fail counters
# must survive). nokeys: empty. twokeys: two secret keys.
mkdir -p "$t/nokeys" "$t/twokeys"; chmod 0700 "$t/nokeys" "$t/twokeys"
GNUPGHOME="$t/twokeys" gpg --batch --quiet --passphrase '' --quick-gen-key 'k1 <k1@example.invalid>' rsa2048 sign never 2>/dev/null
GNUPGHOME="$t/twokeys" gpg --batch --quiet --passphrase '' --quick-gen-key 'k2 <k2@example.invalid>' rsa2048 sign never 2>/dev/null
expect "no secret key: refused (never publish unsigned)" 1 env GNUPGHOME="$t/nokeys" bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs" --overwrite
expect "no secret key: dry run still allowed" 0 env GNUPGHOME="$t/nokeys" bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs" --dry-run
expect "two secret keys: refused (ambiguous signer)" 1 env GNUPGHOME="$t/twokeys" bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs" --overwrite
check "an existing InRelease signed by our key was checked" grep -q '^aws s3 cp s3://bkt/deb/dists/jammy/InRelease ' "$FAKE_LOG"
# The codename's InRelease is shared and signed with the repository's key:
# publishing with ANOTHER key would re-sign every product's index with it.
mkdir -m 0700 "$t/otherkey"
GNUPGHOME="$t/otherkey" gpg --batch --quiet --passphrase '' --quick-gen-key 'other <other@example.invalid>' rsa2048 sign never 2>/dev/null
GNUPGHOME="$t/otherkey" gpg --export --armor | gpg --batch --quiet --import 2>/dev/null  # its PUBLIC half is trusted-readable: the signature is valid, just not ours
mkdir -p "$FAKE_S3/foreign/deb/dists/jammy"
echo "Origin: someone-else" > "$FAKE_S3/foreign/deb/dists/jammy/Release"
GNUPGHOME="$t/otherkey" gpg --batch --yes --clearsign -o "$FAKE_S3/foreign/deb/dists/jammy/InRelease" "$FAKE_S3/foreign/deb/dists/jammy/Release" 2>/dev/null
: > "$FAKE_LOG"
expect "a codename signed by another key is refused" 1 bash "$here/publish-apt.sh" --bucket foreign --region us-east-2 --in "$t/debs"
check "... before deb-s3 wrote anything" bash -c "! grep -q '^deb-s3 ' '$FAKE_LOG'"
check "... naming both keys" bash -c "grep -q 'not by $fpr' '$t/last.out' && grep -q 'signed by [0-9A-F]\{16,\}' '$t/last.out'"
mkdir -m 0700 "$t/lockedkey"
GNUPGHOME="$t/lockedkey" gpg --batch --quiet --pinentry-mode loopback --passphrase 'not-in-ci' --quick-gen-key 'locked <locked@example.invalid>' rsa2048 sign never 2>/dev/null
: > "$FAKE_LOG"
expect "a key that cannot sign non-interactively is refused" 1 env GNUPGHOME="$t/lockedkey" bash "$here/publish-apt.sh" --bucket fresh --region us-east-2 --in "$t/debs"
check "... before deb-s3 wrote anything" bash -c "! grep -q '^deb-s3 ' '$FAKE_LOG'"
: > "$FAKE_LOG"
expect "an InRelease listing error is refused, not taken for a new codename" 1 env FAKE_LS_FAIL='dists/jammy/InRelease$' bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs"
check "... before deb-s3 wrote anything" bash -c "! grep -q '^deb-s3 ' '$FAKE_LOG' && grep -q 'AccessDenied' '$t/last.out'"
mkdir -p "$FAKE_S3/gpgonly/deb/dists/jammy"; echo sig > "$FAKE_S3/gpgonly/deb/dists/jammy/Release.gpg"
: > "$FAKE_LOG"
expect "a codename with Release.gpg but no InRelease is refused" 1 bash "$here/publish-apt.sh" --bucket gpgonly --region us-east-2 --in "$t/debs"
check "... before deb-s3 wrote anything" bash -c "! grep -q '^deb-s3 ' '$FAKE_LOG'"
# A key whose primary only certifies and whose SIGNING SUBKEY makes the
# signatures: VALIDSIG names the subkey first and the primary last.
mkdir -m 0700 "$t/subkey"
GNUPGHOME="$t/subkey" gpg --batch --quiet --passphrase '' --quick-gen-key 'sub <sub@example.invalid>' rsa2048 cert never 2>/dev/null
sub_fpr="$(GNUPGHOME="$t/subkey" gpg --list-secret-keys --with-colons | awk -F: '/^fpr:/{print $10; exit}')"
GNUPGHOME="$t/subkey" gpg --batch --quiet --passphrase '' --quick-add-key "$sub_fpr" rsa2048 sign never 2>/dev/null
expect "a signing-subkey key publishes a new codename" 0 env GNUPGHOME="$t/subkey" bash "$here/publish-apt.sh" --bucket subkey --region us-east-2 --in "$t/debs"
expect "... and its own InRelease passes the signer check on the next run" 0 env GNUPGHOME="$t/subkey" bash "$here/publish-apt.sh" --bucket subkey --region us-east-2 --in "$t/debs" --overwrite
check "... which really was a subkey signature" bash -c "GNUPGHOME='$t/subkey' gpg --batch --status-fd 1 --verify '$FAKE_S3/subkey/deb/dists/jammy/InRelease' 2>/dev/null | awk '\$2==\"VALIDSIG\" {exit !(\$3 != \"$sub_fpr\" && \$NF == \"$sub_fpr\")}'"

echo "publish-yum.sh"
mkrpm 9.9.9 1.el9 x86_64 "$t/rpms/a"; mkrpm 9.9.9 1.amzn2023 x86_64 "$t/rpms/b"; mkrpm 9.9.9 1.el9_4 aarch64 "$t/rpms/c"
# another product already in the shared el9/x86_64 directory
mkrpm 1.0 1.el9 x86_64 "$t/other"; mkdir -p "$FAKE_S3/bkt/rpm/el9/x86_64"
other="$(basename "$(find "$t/other" -name '*.rpm')")"
mv "$t/other/$other" "$FAKE_S3/bkt/rpm/el9/x86_64/redis-other-$other"
: > "$FAKE_LOG"
expect "publishes three dist/arch directories" 0 bash "$here/publish-yum.sh" --bucket bkt --in "$t/rpms"
for d in el9/x86_64 amzn2023/x86_64 el9/aarch64; do
  check "$d has the RPM and repodata/repomd.xml" bash -c "ls '$FAKE_S3/bkt/rpm/$d/'vector-db-benchmark-9.9.9-*.rpm >/dev/null && test -f '$FAKE_S3/bkt/rpm/$d/repodata/repomd.xml'"
done
check "el9_4 dist tag landed in el9/" test -f "$FAKE_S3/bkt/rpm/el9/aarch64/vector-db-benchmark-9.9.9-1.el9_4.aarch64.rpm"
check "RPM uploaded before its repodata" bash -c "grep -n 'aws s3 cp .*el9/x86_64/vector-db-benchmark' '$FAKE_LOG' | head -n1 | cut -d: -f1 | xargs -I{} test {} -lt \"\$(grep -n 'aws s3 sync .*el9/x86_64/repodata' '$FAKE_LOG' | head -n1 | cut -d: -f1)\""
check "the other product's RPM is still there" test -f "$FAKE_S3/bkt/rpm/el9/x86_64/redis-other-$other"
primary="$(find "$FAKE_S3/bkt/rpm/el9/x86_64/repodata" -name '*primary.xml.gz' | head -n1)"
check "... and indexed next to ours" bash -c "gzip -dc '$primary' | grep -q 'redis-other-' && gzip -dc '$primary' | grep -q 'vector-db-benchmark-9.9.9-1.el9.x86_64.rpm'"
check "published RPMs are signed with the selftest key" bash -c "rpm --dbpath '$t/vdb' --initdb && gpg --export --armor '$fpr' > '$t/pub.asc' && rpm --dbpath '$t/vdb' --import '$t/pub.asc' && rpm --dbpath '$t/vdb' -K '$FAKE_S3/bkt/rpm/el9/x86_64/vector-db-benchmark-9.9.9-1.el9.x86_64.rpm' | grep -q 'OK'"
: > "$FAKE_LOG"
expect "re-run with the same build succeeds" 0 bash "$here/publish-yum.sh" --bucket bkt --in "$t/rpms"
check "... and uploads no RPM again" bash -c "! grep -qE '^aws s3 cp [^ ]+ s3://[^ ]+[.]rpm ' '$FAKE_LOG'"
rm -rf "$t/rpms2"; mkrpm 9.9.9 1.el9 x86_64 "$t/rpms2" different-payload
check "... which has the published build's NEVRA and build time" test "$(rpm -qp --qf '%{BUILDTIME}' "$t/rpms2/vector-db-benchmark-9.9.9-1.el9.x86_64.rpm")" = "$(rpm -qp --qf '%{BUILDTIME}' "$FAKE_S3/bkt/rpm/el9/x86_64/vector-db-benchmark-9.9.9-1.el9.x86_64.rpm")"
expect "a DIFFERENT build under a published name is refused" 1 bash "$here/publish-yum.sh" --bucket bkt --in "$t/rpms2"
expect "... unless --overwrite" 0 bash "$here/publish-yum.sh" --bucket bkt --in "$t/rpms2" --overwrite
check "overwrite uploaded it" grep -q 'aws s3 cp .*el9/x86_64/vector-db-benchmark-9.9.9-1.el9.x86_64.rpm' "$FAKE_LOG"
mkrpm 9.9.9 1.fc40 x86_64 "$t/rpms-bad"
expect "an unmapped dist tag (.fc40) is refused" 1 bash "$here/publish-yum.sh" --bucket bkt --in "$t/rpms-bad"
# No fake aws on PATH either: the dry run must not need it.
expect "--dry-run needs neither aws nor a key" 0 env PATH="/usr/bin:/bin" GNUPGHOME="$t/nokeys" bash "$here/publish-yum.sh" --bucket bkt --in "$t/rpms" --dry-run
check "dry run printed the aws commands" grep -q '^dry-run: aws s3 cp' "$t/last.out"
expect "no secret key: refused" 1 env GNUPGHOME="$t/nokeys" bash "$here/publish-yum.sh" --bucket bkt --in "$t/rpms" --overwrite
check "the post-upload index check ran" grep -q '^aws s3 ls s3://bkt/rpm/el9/x86_64/' "$FAKE_LOG"
check "repomd.xml is uploaded after the rest of repodata/" bash -c "s=\$(grep -n 'aws s3 sync .*el9/x86_64/repodata/ s3://' '$FAKE_LOG' | tail -n1 | cut -d: -f1); c=\$(grep -n 'aws s3 cp .*el9/x86_64/repodata/repomd.xml s3://' '$FAKE_LOG' | tail -n1 | cut -d: -f1); [[ -n \$s && -n \$c && \$s -lt \$c ]] && ! grep -q 'repodata/ s3://.*--exclude repomd.xml.*--delete\|repodata/ s3://.*--delete' '$FAKE_LOG'"

# Another product publishing into the same directory while we do. No lock
# covers rpm/, so publish-yum.sh re-lists before uploading repodata/ and checks
# afterwards that every RPM in the directory is in the index it serves.
indexed() { # indexed <bucket> <dir> <file name...>: all listed in the served primary.xml.gz
  local b="$1" d="$2" p xml f; shift 2
  p="$(sed -n 's/.*href="\(repodata\/[^"]*primary\.xml\.gz\)".*/\1/p' "$FAKE_S3/$b/rpm/$d/repodata/repomd.xml" | head -n1)"
  xml="$(gzip -dc "$FAKE_S3/$b/rpm/$d/$p")" || return 1
  for f in "$@"; do [[ "$xml" == *"location href=\"$f\""* ]] || return 1; done
}
race_rpm() { # race_rpm <version> -> path of another product's RPM, ready to drop in
  rm -rf "$t/race"; mkrpm "$1" 1.el9 x86_64 "$t/race" "race-$1"
  local f; f="$(find "$t/race" -name '*.rpm')"; mv "$f" "$t/race/redis-race-$1.el9.x86_64.rpm"
  echo "$t/race/redis-race-$1.el9.x86_64.rpm"
}
mkrpm 9.9.9 1.el9 x86_64 "$t/rpms-race"
# (a) it lands while our RPM uploads, after our sync: the re-list catches it.
src="$(race_rpm 1.1)"
expect "an RPM uploaded after our sync, before the re-list, is published" 0 \
  env FAKE_RACE_MATCH='^s3 cp .*/rpm/el9/x86_64/vector-db-benchmark' FAKE_RACE_SRC="$src" \
      FAKE_RACE_DST="s3://race/rpm/el9/x86_64/redis-race-1.1.el9.x86_64.rpm" \
      bash "$here/publish-yum.sh" --bucket race --in "$t/rpms-race"
check "... the re-list said so" grep -q 'since the sync, re-indexing: redis-race-1.1.el9.x86_64.rpm' "$t/last.out"
check "... and the index lists it next to ours" indexed race el9/x86_64 redis-race-1.1.el9.x86_64.rpm vector-db-benchmark-9.9.9-1.el9.x86_64.rpm
# (b) it lands during our repodata/ upload, after the re-list: the publish
# fails, naming it and telling the operator to re-run.
src="$(race_rpm 1.2)"
expect "an RPM uploaded during our repodata upload fails the publish" 1 \
  env FAKE_RACE_MATCH='^s3 sync .*/el9/x86_64/repodata/ s3://' FAKE_RACE_SRC="$src" \
      FAKE_RACE_DST="s3://race/rpm/el9/x86_64/redis-race-1.2.el9.x86_64.rpm" \
      bash "$here/publish-yum.sh" --bucket race --in "$t/rpms-race"
check "... naming the RPM the index lost" grep -q '^  redis-race-1.2.el9.x86_64.rpm$' "$t/last.out"
check "... and saying to re-run the publish" grep -q 'Re-run this publish' "$t/last.out"
expect "the re-run it asks for succeeds" 0 bash "$here/publish-yum.sh" --bucket race --in "$t/rpms-race"
check "... and indexes every RPM in the directory" indexed race el9/x86_64 redis-race-1.1.el9.x86_64.rpm redis-race-1.2.el9.x86_64.rpm vector-db-benchmark-9.9.9-1.el9.x86_64.rpm
# (c) another product's NEW hashed metadata lands during our repodata/ upload:
# its repomd.xml (perhaps already cached by the CDN) names it, so our upload
# must never delete it.
echo "their primary" > "$t/their-primary.xml.gz"
expect "a publish racing another product's metadata upload" 0 \
  env FAKE_RACE_MATCH='^s3 sync .*/el9/x86_64/repodata/ s3://' FAKE_RACE_SRC="$t/their-primary.xml.gz" \
      FAKE_RACE_DST="s3://race/rpm/el9/x86_64/repodata/0123abcd-primary.xml.gz" \
      bash "$here/publish-yum.sh" --bucket race --in "$t/rpms-race"
check "... leaves the other product's hashed metadata in place" test -f "$FAKE_S3/race/rpm/el9/x86_64/repodata/0123abcd-primary.xml.gz"

# A directory whose metadata another product SIGNS: regenerating repomd.xml
# would leave repomd.xml.asc stale, so the publish refuses before writing.
mkdir -p "$FAKE_S3/signedmd/rpm/el9/x86_64/repodata"
echo '<repomd/>' > "$FAKE_S3/signedmd/rpm/el9/x86_64/repodata/repomd.xml"
echo 'sig' > "$FAKE_S3/signedmd/rpm/el9/x86_64/repodata/repomd.xml.asc"
: > "$FAKE_LOG"
# Only el9/x86_64 is signed; it sorts LAST of the three target directories, so
# the refusal must come from the preflight, before amzn2023/ and el9/aarch64.
expect "a directory with signed metadata (repomd.xml.asc) is refused" 1 bash "$here/publish-yum.sh" --bucket signedmd --in "$t/rpms"
check "... naming the signature" grep -q 'repomd.xml.asc exists' "$t/last.out"
check "... before anything was uploaded to ANY directory" bash -c "! grep -qE '^aws s3 (cp|sync) [^ ]+ s3://' '$FAKE_LOG' && ! test -e '$FAKE_S3/signedmd/rpm/amzn2023'"
: > "$FAKE_LOG"
expect "a repodata/ listing error is refused" 1 env FAKE_LS_FAIL='/repodata/$' bash "$here/publish-yum.sh" --bucket bkt --in "$t/rpms"
check "... before anything was uploaded" bash -c "! grep -qE '^aws s3 (cp|sync) [^ ]+ s3://' '$FAKE_LOG'"

echo "lib.sh s3_origin_url"
# Path-style only: the virtual-hosted host <bucket>.s3.<region>.amazonaws.com
# is not covered by S3's one-label wildcard certificate for a dotted bucket.
check "production's base URL (the README's)" test "$(s3_origin_url "$PRODUCTION_BUCKET" "$PRODUCTION_REGION")" = "https://s3.us-east-2.amazonaws.com/vdb-packages"
check "dotted bucket -> path-style URL" test "$(s3_origin_url example.dotted.bucket us-east-2)" = "https://s3.us-east-2.amazonaws.com/example.dotted.bucket"
check "... with a prefix" test "$(s3_origin_url example.dotted.bucket us-east-2 rehearsal/x/)" = "https://s3.us-east-2.amazonaws.com/example.dotted.bucket/rehearsal/x"
check "us-east-1 uses its regional host too" test "$(s3_origin_url vdb-package-staging us-east-1)" = "https://s3.us-east-1.amazonaws.com/vdb-package-staging"
for pair in example.dotted.bucket:us-east-2 vdb-package-staging:us-east-1 a.b.c:eu-west-1; do
  u="$(s3_origin_url "${pair%%:*}" "${pair##*:}")"; host="${u#https://}"; host="${host%%/*}"
  check "${pair%%:*}: host $host is the regional endpoint, no bucket label in it" \
    test "$host" = "s3.${pair##*:}.amazonaws.com"
done
expect "an empty region is refused" 1 s3_origin_url example.dotted.bucket ""
expect "a bucket with a slash is refused" 1 s3_origin_url "vdb-packages/deb" us-east-2

echo "lib.sh check_publish_target"
# Production is our own bucket only; staging only a staging bucket. Nothing
# reaches the shared Redis repository, whatever an inherited secret says.
check "production is vdb-packages in us-east-2" test "$PRODUCTION_BUCKET:$PRODUCTION_REGION" = vdb-packages:us-east-2
expect "build -> vdb-packages, us-east-2, root" 0 check_publish_target build vdb-packages us-east-2 ""
expect "build -> packages.redis.io is refused" 1 check_publish_target build packages.redis.io us-east-2 ""
check "... naming it a shared Redis repository" grep -q 'shared Redis repository' "$t/last.out"
expect "build -> the Redis-wide staging bucket is refused" 1 check_publish_target build redis-test-package-repository us-east-2 ""
expect "build -> our staging bucket is refused" 1 check_publish_target build vdb-package-staging us-east-2 ""
check "... naming the staging bucket" grep -q 'points at the staging bucket' "$t/last.out"
expect "build -> any other bucket is refused" 1 check_publish_target build vdb-packages-2 us-east-2 ""
expect "build in another region is refused" 1 check_publish_target build vdb-packages us-east-1 ""
expect "build under a prefix is refused" 1 check_publish_target build vdb-packages us-east-2 rehearsal
expect "staging -> vdb-package-staging" 0 check_publish_target staging vdb-package-staging us-east-2 ""
expect "staging -> the shared staging bucket under a prefix" 0 check_publish_target staging redis-test-package-repository us-east-1 vdb/rehearsal
expect "staging -> the shared staging bucket at its root is refused" 1 check_publish_target staging redis-test-package-repository us-east-1 ""
expect "staging -> the production bucket is refused" 1 check_publish_target staging vdb-packages us-east-2 ""
expect "staging -> packages.redis.io is refused" 1 check_publish_target staging packages.redis.io us-east-2 ""
expect "an unknown environment is refused" 1 check_publish_target none vdb-packages us-east-2 ""

echo "lib.sh key_file_fprs"
gpg --export --armor "$fpr" > "$t/one.asc"
check "an armored public key -> its primary fingerprint" test "$(key_file_fprs "$t/one.asc")" = "$fpr"
GNUPGHOME="$t/subkey" gpg --export > "$t/sub.gpg"
check "... a binary key with a signing subkey -> its PRIMARY only" test "$(key_file_fprs "$t/sub.gpg")" = "$sub_fpr"
{ cat "$t/one.asc"; GNUPGHOME="$t/otherkey" gpg --export --armor; } > "$t/two.asc"
check "a file with two keys -> two fingerprints" test "$(key_file_fprs "$t/two.asc" | wc -l | tr -d ' ')" = 2
echo "not a key" > "$t/nokey.asc"
check "a file with no key -> nothing" test -z "$(key_file_fprs "$t/nokey.asc")"

echo "lib.sh check_served_key / s3_head_missing (the <base>/gpg guard)"
expect "<base>/gpg holding exactly our key" 0 check_served_key "$t/one.asc" "$fpr"
expect "... another key is refused" 1 check_served_key "$t/sub.gpg" "$fpr"
check "... naming the key it holds" grep -q "'$sub_fpr'" "$t/last.out"
expect "... ours plus another is refused" 1 check_served_key "$t/two.asc" "$fpr"
expect "... no key at all is refused" 1 check_served_key "$t/nokey.asc" "$fpr"
expect "... an empty pin never matches an empty file" 1 check_served_key "$t/nokey.asc" ""
echo 'An error occurred (404) when calling the HeadObject operation: Not Found' > "$t/head404"
echo 'An error occurred (403) when calling the HeadObject operation: Forbidden' > "$t/head403"
: > "$t/head-empty"
expect "head-object 404 -> missing (a first publish)" 0 s3_head_missing "$t/head404"
expect "head-object 403 -> NOT missing (stop, never assume new)" 1 s3_head_missing "$t/head403"
expect "no error text -> NOT missing" 1 s3_head_missing "$t/head-empty"

echo "README Getting Started"
# The install steps users copy must name the repository the workflow publishes
# to, and every codename it builds by default (and no other).
readme="$(cat "$here/../../README.md")"
prod_base="$(s3_origin_url "$PRODUCTION_BUCKET" "$PRODUCTION_REGION")"
for want in "curl -fsSL $prod_base/gpg | sudo gpg --dearmor" \
            "] $prod_base/deb \$(lsb_release -cs) main\"" \
            "baseurl=$prod_base/rpm/el\$releasever/\$basearch" \
            "baseurl=$prod_base/rpm/amzn2023/\$basearch" \
            "sudo curl -fsSL $prod_base/gpg -o /etc/pki/rpm-gpg/"; do
  check "README has: $want" grep -qF -- "$want" <<<"$readme"
done
lacks() { ! grep -qE "$1" <<<"$2"; }
check "README sends nobody to packages.redis.io for our packages" lacks 'packages\.redis\.io/(gpg|deb|rpm)' "$readme"
default_debs="$(sed -n "s/^  DEFAULT_DEB_DISTS: '\(.*\)'$/\1/p" "$here/../../.github/workflows/release-packages.yml" | tr -d '[]" ' | tr ',' ' ')"
check "DEFAULT_DEB_DISTS read from the workflow" test -n "$default_debs"
apt_section="$(sed -n '/^### Installing on Debian and Ubuntu/,/^### /p' <<<"$readme")"
for c in $default_debs; do
  check "README lists codename $c" grep -qF "($c)" <<<"$apt_section"
done
check "README lists no codename the workflow does not build" test \
  "$(grep -oE '\([a-z]+\)' <<<"$apt_section" | tr -d '()' | sort -u | paste -sd' ' -)" \
  = "$(tr ' ' '\n' <<<"$default_debs" | sort -u | paste -sd' ' -)"

echo "lib.sh check_production_tag"
# Packages carry the crate version without a pre-release suffix: an rc tag
# would publish as the final version. Only X.Y.Z / vX.Y.Z reach production.
for tag in 0.2.0 v0.2.0 10.20.30; do
  expect "$tag may publish to production" 0 check_production_tag "$tag"
done
for tag in 0.2.0-rc.1 v0.2.0-rc.1 0.2.0+build.1 0.2 v0.2.0.1 0.2.0rc1 "" " 0.2.0" 00.2.0 0.02.0; do
  expect "'$tag' may not" 1 check_production_tag "$tag"
done

echo "lib.sh production_gate"
check "variable on: build" test "$(production_gate release true)" = build
check "variable on, public dispatch: build" test "$(production_gate workflow_dispatch true)" = build
check "variable off, release: none" test "$(production_gate release '')" = none
check "variable 'yes' is not 'true': none" test "$(production_gate release yes)" = none
expect "variable off, public dispatch: refused" 1 production_gate workflow_dispatch ''

echo "lib.sh check_signing_fpr"
F=0123456789ABCDEF0123456789ABCDEF01234567   # stands in for our production key
expect "build with the pinned key" 0 check_signing_fpr "$F" "$F" build
expect "pin with spaces and lower case" 0 check_signing_fpr "0123 4567 89ab cdef 0123  4567 89ab cdef 0123 4567" "$F" build
expect "build without a pin: refused" 1 check_signing_fpr "" "$F" build
expect "build with another key: refused" 1 check_signing_fpr "$F" FD183D88E9814786987424DAE26208135FFAAD25 build
expect "staging without a pin" 0 check_signing_fpr "" FD183D88E9814786987424DAE26208135FFAAD25 staging
expect "staging with another key: refused" 1 check_signing_fpr "$F" FD183D88E9814786987424DAE26208135FFAAD25 staging
expect "a short key id is not a pin" 1 check_signing_fpr 01234567 "$F" build

echo "lib.sh gpg_status_signed_by"
st_primary="[GNUPG:] VALIDSIG $F 2026-10-29 1 0 4 0 1 10 01 $F"
st_sub="[GNUPG:] VALIDSIG 48263292AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA 2026-10-29 1 0 4 0 1 10 01 $F"
expect "signed by the primary key" 0 gpg_status_signed_by "$F" <<<"$st_primary"
expect "signed by its subkey (primary in the last field)" 0 gpg_status_signed_by "$F" <<<"$st_sub"
expect "GOODSIG alone is not enough" 1 gpg_status_signed_by "$F" <<<"[GNUPG:] GOODSIG 89ABCDEF01234567 vdb"
expect "another key" 1 gpg_status_signed_by FD183D88E9814786987424DAE26208135FFAAD25 <<<"$st_sub"
expect "a fingerprint prefix does not match" 1 gpg_status_signed_by "${F:0:39}" <<<"$st_primary"

echo "lib.sh deb_index_entry"
# A shared Packages index: another product with the SAME version string, our
# package on the other architecture, and ours.
cat > "$t/Packages" <<'IDX'
Package: redis-tools
Version: 0.1.0~jammy
Architecture: amd64
Filename: pool/r/redis-tools_0.1.0~jammy_amd64.deb
SHA256: 1111111111111111111111111111111111111111111111111111111111111111

Package: vector-db-benchmark
Version: 0.1.0~jammy
Architecture: arm64
Filename: pool/v/vector-db-benchmark_0.1.0~jammy_arm64.deb
SHA256: 2222222222222222222222222222222222222222222222222222222222222222

Package: vector-db-benchmark
Version: 0.1.0~jammy
Architecture: amd64
Filename: pool/v/vector-db-benchmark_0.1.0~jammy_amd64.deb
SHA256: 3333333333333333333333333333333333333333333333333333333333333333
IDX
check "our stanza, not another package's with the same version" test "$(deb_index_entry "$t/Packages" vector-db-benchmark 0.1.0~jammy amd64)" = "3333333333333333333333333333333333333333333333333333333333333333 pool/v/vector-db-benchmark_0.1.0~jammy_amd64.deb"
check "... and per architecture" test "$(deb_index_entry "$t/Packages" vector-db-benchmark 0.1.0~jammy arm64)" = "2222222222222222222222222222222222222222222222222222222222222222 pool/v/vector-db-benchmark_0.1.0~jammy_arm64.deb"
check "nothing when only another package has the version" test -z "$(deb_index_entry "$t/Packages" vector-db-benchmark 0.1.0~jammy riscv64)"
check "nothing for a version we did not publish" test -z "$(deb_index_entry "$t/Packages" vector-db-benchmark 0.2.0~jammy amd64)"

echo "lib.sh rpm_release_to_distro"
for pair in 1.el8:el8 1.el9:el9 1.el9_4:el9 1.el10:el10 2.el10_1:el10 1.amzn2023:amzn2023; do
  check "${pair%%:*} -> ${pair##*:}" test "$(rpm_release_to_distro "${pair%%:*}")" = "${pair##*:}"
done
for rel in 1.fc40 1.el8.rocky 1 1.amzn2; do
  expect "$rel is refused" 1 rpm_release_to_distro "$rel"
done

echo
echo "package selftest: $pass passed, $failed failed"
[[ "$failed" -eq 0 ]]
