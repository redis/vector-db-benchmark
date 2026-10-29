#!/usr/bin/env bash
# Self-test of the package publish scripts: `make release-selftest-packages`.
#
# publish-apt.sh and publish-yum.sh only run for real during a publish, into
# repositories shared with every other Redis product, so every decision they
# make is exercised here first, on every PR, against FAKES:
#   aws     `s3 sync` / `s3 cp` against a directory standing in for the bucket
#   deb-s3  records its arguments, keeps a per-codename manifest, honours
#           --fail-if-exists, and REFUSES a bare `--sign` (a string option in
#           deb-s3: bare, it would swallow the first .deb as the key id)
# with tiny .deb / .rpm fixtures and a throwaway GPG key. Covered: grouping by
# ~codename and refusing a .deb without one; dist tag -> rpm/<distro>/<arch>;
# refusing unmapped tags; signing mandatory, --dry-run without a key or tools;
# re-run idempotency (same build skipped, different build refused, --overwrite);
# RPMs uploaded before repodata; another product's RPM in the same directory
# kept and indexed; another product's RPM uploaded DURING our publish (caught
# by the re-list, or failing the post-upload index check with a re-run hint);
# lib.sh's path-style S3 URL for a dotted bucket.
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
if [[ "$2" == ls ]]; then
  # Direct children, as `aws s3 ls s3://b/dir/` prints them; exit 1, no
  # output, for a prefix with nothing under it.
  d="$(map "$3")"; out=""
  if [[ -d "$d" ]]; then
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
check "--suite, --origin, --fail-if-exists, --lock passed" bash -c "l=\$(grep -- '--codename=noble' '$FAKE_LOG'); [[ \"\$l\" == *--suite=noble* && \"\$l\" == *--origin=packages.redis.io* && \"\$l\" == *--fail-if-exists* && \"\$l\" == *--lock* ]]"
check "fake deb-s3 recorded the manifest" grep -qx 'vector-db-benchmark 9.9.9~noble amd64' "$FAKE_S3/bkt/deb/dists/noble/manifest"
expect "re-run of the same version is refused (--fail-if-exists)" 1 bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs"
expect "... unless --overwrite" 0 bash "$here/publish-apt.sh" --bucket bkt --region us-east-2 --in "$t/debs" --overwrite
check "--overwrite drops --fail-if-exists" bash -c "! tail -n 2 '$FAKE_LOG' | grep -q -- --fail-if-exists"
mkdeb 9.9.9 amd64 "$t/debs-bad"
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

echo "lib.sh s3_origin_url"
# Path-style only: production's bucket (packages.redis.io) has dots, and the
# virtual-hosted host <bucket>.s3.<region>.amazonaws.com is not covered by
# S3's one-label wildcard certificate *.s3.<region>.amazonaws.com.
check "dotted bucket -> path-style URL" test "$(s3_origin_url packages.redis.io us-east-2)" = "https://s3.us-east-2.amazonaws.com/packages.redis.io"
check "... with a prefix" test "$(s3_origin_url packages.redis.io us-east-2 rehearsal/x/)" = "https://s3.us-east-2.amazonaws.com/packages.redis.io/rehearsal/x"
check "us-east-1 uses its regional host too" test "$(s3_origin_url vdb-package-staging us-east-1)" = "https://s3.us-east-1.amazonaws.com/vdb-package-staging"
for pair in packages.redis.io:us-east-2 vdb-package-staging:us-east-1 a.b.c:eu-west-1; do
  u="$(s3_origin_url "${pair%%:*}" "${pair##*:}")"; host="${u#https://}"; host="${host%%/*}"
  check "${pair%%:*}: host $host is the regional endpoint, no bucket label in it" \
    test "$host" = "s3.${pair##*:}.amazonaws.com"
done
expect "an empty region is refused" 1 s3_origin_url packages.redis.io ""
expect "a bucket with a slash is refused" 1 s3_origin_url "packages.redis.io/deb" us-east-2

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
