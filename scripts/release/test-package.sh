#!/usr/bin/env bash
# Install a built vector-db-benchmark .deb or .rpm the way a user would, and run it.
#
#   scripts/release/test-package.sh <package> --expect-sha <40-hex> [--expect-version <crate version>] [--repo-id <el8|el9|el10|amzn2023>]
#
# Needs only the distro's package tools: the release workflow runs it inside a
# BARE container of every codename / distro the package is published for, on
# both architectures (it refuses a host with a Rust toolchain, since the point
# is that the package works without one). It adds and removes the package and
# a local repository definition, so run it on a throwaway container, not a
# workstation. Checks, in order:
#  * the package installs from the file with its own dependencies resolved by
#    the package manager (nothing pre-installed by hand, so a missing
#    Depends/Requires shows);
#  * each CLI's `--version` is exactly `<name> <version> (git <sha>)` and its
#    `--help` exits 0;
#  * the installed data files work: `--describe datasets` reads the packaged
#    datasets.json from a copy of /usr/share/vector-db-benchmark, bench_jsonl
#    reads the packaged random-100 corpus, bench_hdf5 reaches the linked HDF5;
#  * the package also installs from a LOCAL repository laid out as
#    packages.redis.io serves it — deb: dists/<codename>/main/binary-<arch>/
#    Packages + Release (apt-ftparchive); rpm: createrepo_c under the README's
#    rpm/el$releasever/$basearch or rpm/amzn2023/$basearch baseurl — and the
#    installed package is confirmed to have come from that repository. This
#    catches index/metadata and Depends/Requires bugs a direct file install
#    does not.
set -euo pipefail

pkg="" expect_sha="" expect_version="" repo_id=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --expect-sha) expect_sha="$2"; shift 2 ;;
    --expect-version) expect_version="$2"; shift 2 ;;
    --repo-id) repo_id="$2"; shift 2 ;;
    -*) echo "test-package.sh: unknown flag: $1" >&2; exit 2 ;;
    *) [[ -z "$pkg" ]] || { echo "test-package.sh: one package at a time ('$pkg' and '$1' given)" >&2; exit 2; }
       pkg="$1"; shift ;;
  esac
done
[[ -n "$pkg" && -n "$expect_sha" ]] || {
  echo "usage: test-package.sh <package.deb|.rpm> --expect-sha <sha> [--expect-version <v>] [--repo-id <id>]" >&2; exit 2; }
[[ "$expect_sha" =~ ^[0-9a-f]{40}$ ]] || { echo "test-package.sh: --expect-sha must be a full 40-hex SHA" >&2; exit 2; }
[[ -f "$pkg" ]] || { echo "test-package.sh: no such file: $pkg" >&2; exit 2; }
pkg="$(cd "$(dirname "$pkg")" && pwd)/$(basename "$pkg")"

fail() { echo "test-package.sh: FAIL: $*" >&2; exit 1; }
# shellcheck disable=SC1091
. /etc/os-release
echo "host: ${PRETTY_NAME:-?} $(uname -m)"

case "$pkg" in
  *.deb) kind=deb ;;
  *.rpm) kind=rpm ;;
  *) fail "not a .deb or .rpm: $pkg" ;;
esac

repo_root=/tmp/vdb-local-repo
cleanup() {
  rm -rf "$repo_root" /etc/apt/sources.list.d/vdb-local-test.list /etc/yum.repos.d/vdb-local-test.repo
}
trap cleanup EXIT

pm_install_file() {
  if [[ "$kind" == deb ]]; then
    apt-get install -y "$1"
  elif command -v dnf >/dev/null; then
    dnf install -y "$1"
  else
    yum install -y "$1"
  fi
}
pm_remove() {
  if [[ "$kind" == deb ]]; then
    apt-get remove -y vector-db-benchmark
  elif command -v dnf >/dev/null; then
    dnf remove -y vector-db-benchmark
  else
    yum remove -y vector-db-benchmark
  fi
}

if [[ "$kind" == deb ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  echo "--- package metadata"; dpkg-deb -f "$pkg" Package Version Architecture Depends
  pkg_arch="$(dpkg-deb -f "$pkg" Architecture)"
  [[ "$pkg_arch" == "$(dpkg --print-architecture)" ]] || fail "package is $pkg_arch, host is $(dpkg --print-architecture)"
else
  echo "--- package metadata"; rpm -qp --queryformat 'Name: %{NAME}\nVersion: %{VERSION}-%{RELEASE}\nArch: %{ARCH}\n' "$pkg"
  echo "Requires:"; rpm -qpR "$pkg" | sed 's/^/  /'
fi

for t in cargo rustc; do
  command -v "$t" >/dev/null && fail "not a bare host: $t is installed"
done

echo "--- install from file"
pm_install_file "$pkg"

run_checks() {
  local version="$1"
  hash -r   # bash caches the paths it ran before the remove/reinstall
  for b in vector-db-benchmark prepare-msmarco generate-dataset; do
    command -v "$b" >/dev/null || fail "$b not on PATH after install"
    got="$("$b" --version)" || fail "$b --version exited non-zero"
    want="$b $version (git $expect_sha)"
    [[ "$got" == "$want" ]] || fail "$b --version printed '$got', expected '$want'"
    "$b" --help >/dev/null || fail "$b --help exited non-zero"
    echo "ok: $got"
  done
  local share=/usr/share/vector-db-benchmark libdir
  [[ -f "$share/datasets/datasets.json" ]] || fail "$share/datasets/datasets.json not installed"
  [[ -f "$share/datasets/random-100/vectors.jsonl" ]] || fail "random-100 corpus not installed"
  [[ -d "$share/experiments/configurations" ]] || fail "experiments/configurations not installed"
  for libdir in /usr/lib/vector-db-benchmark /usr/libexec/vector-db-benchmark; do
    [[ -x "$libdir/bench_jsonl" ]] && break
  done
  [[ -x "$libdir/bench_jsonl" ]] || fail "bench_jsonl not installed under /usr/lib or /usr/libexec"
  for b in bench_npy bench_jsonl bench_hdf5; do
    [[ -x "$libdir/$b" ]] || fail "$libdir/$b missing"
  done
  # The CLI finds datasets/datasets.json from the working directory, and writes
  # results next to it: work from a copy, as the README tells users to.
  local work; work="$(mktemp -d)"
  cp -R "$share/." "$work/"
  ( cd "$work" || fail "cannot cd to $work"
    vector-db-benchmark --describe datasets > describe.out || fail "--describe datasets exited non-zero"
    grep -q 'random-100' describe.out || fail "--describe datasets did not list the packaged random-100 dataset"
    echo "ok: vector-db-benchmark --describe datasets read the packaged datasets.json"
    "$libdir/bench_jsonl" datasets/random-100/vectors.jsonl 1 > bench_jsonl.out || fail "bench_jsonl failed on the packaged random-100"
    grep -q 'Throughput' bench_jsonl.out || fail "bench_jsonl printed no result"
    echo "ok: bench_jsonl read the packaged corpus"
    set +e; "$libdir/bench_hdf5" does-not-exist.hdf5 1 > bench_hdf5.out 2>&1; set -e
    grep -q 'Failed to open HDF5 file' bench_hdf5.out || fail "bench_hdf5 did not reach the linked HDF5: $(cat bench_hdf5.out)"
    echo "ok: bench_hdf5 initialised the statically linked HDF5" )
  rm -rf "$work"
}

if [[ "$kind" == deb ]]; then
  installed_version="$(dpkg-query -W -f '${Version}' vector-db-benchmark)"
  crate_version="${installed_version%%~*}"
else
  crate_version="$(rpm -q --queryformat '%{VERSION}' vector-db-benchmark)"
fi
[[ -z "$expect_version" || "$crate_version" == "$expect_version" ]] ||
  fail "installed version '$crate_version' != expected '$expect_version'"
run_checks "$crate_version"

echo "--- reinstall from a local repository laid out like packages.redis.io"
pm_remove
hash -r
for f in /usr/bin/vector-db-benchmark /usr/bin/prepare-msmarco /usr/bin/generate-dataset /usr/share/vector-db-benchmark/datasets/datasets.json; do
  [[ -e "$f" ]] && fail "$f still present after remove"
done
rm -rf "$repo_root"
if [[ "$kind" == deb ]]; then
  apt-get install -y --no-install-recommends apt-utils >/dev/null
  codename="${VERSION_CODENAME:?no VERSION_CODENAME in /etc/os-release}"
  [[ "$installed_version" == *"~$codename" ]] || fail "package version '$installed_version' is not for this host's codename '$codename'"
  arch="$(dpkg --print-architecture)"
  # The deb-s3 layout: pool/ plus dists/<codename>/main/binary-<arch>/Packages
  # and a dists/<codename>/Release, served with the codename as the suite.
  mkdir -p "$repo_root/pool/$codename" "$repo_root/dists/$codename/main/binary-$arch"
  cp "$pkg" "$repo_root/pool/$codename/"
  ( cd "$repo_root" &&
    apt-ftparchive packages pool > "dists/$codename/main/binary-$arch/Packages" &&
    gzip -k -f "dists/$codename/main/binary-$arch/Packages" &&
    apt-ftparchive -o "APT::FTPArchive::Release::Origin=packages.redis.io" \
                   -o "APT::FTPArchive::Release::Suite=$codename" \
                   -o "APT::FTPArchive::Release::Codename=$codename" \
                   -o "APT::FTPArchive::Release::Components=main" \
                   -o "APT::FTPArchive::Release::Architectures=$arch" \
                   release "dists/$codename" > "dists/$codename/Release" )
  echo "deb [trusted=yes] file:$repo_root $codename main" > /etc/apt/sources.list.d/vdb-local-test.list
  apt-get update
  policy="$(apt-cache policy vector-db-benchmark)"
  echo "$policy"
  grep -qF "file:$repo_root" <<<"$policy" || fail "apt does not see vector-db-benchmark in the local repository"
  apt-get install -y vector-db-benchmark
  dpkg-query -W -f 'installed from repo: ${Package} ${Version}\n' vector-db-benchmark
else
  if command -v dnf >/dev/null; then dnf install -y createrepo_c; else yum install -y createrepo_c || yum install -y createrepo; fi
  arch="$(uname -m)"
  # Same baseurl pattern as the README: rpm/el$releasever/$basearch, or the
  # fixed rpm/amzn2023/$basearch on Amazon Linux, whose $releasever is a point
  # release (2023.x.y) rather than an el<n> value.
  if [[ -z "$repo_id" ]]; then
    if [[ "${ID:-}" == amzn ]]; then repo_id=amzn2023; else repo_id="el${VERSION_ID%%.*}"; fi
  fi
  mkdir -p "$repo_root/$repo_id/$arch"
  cp "$pkg" "$repo_root/$repo_id/$arch/"
  createrepo_c "$repo_root/$repo_id/$arch"
  if [[ "$repo_id" == amzn2023 ]]; then
    baseurl="file://$repo_root/amzn2023/\$basearch"
  else
    baseurl="file://$repo_root/el\$releasever/\$basearch"
  fi
  cat > /etc/yum.repos.d/vdb-local-test.repo <<REPO
[vdb-local-test]
name=vector-db-benchmark local test
baseurl=$baseurl
enabled=1
gpgcheck=0
REPO
  if command -v dnf >/dev/null; then
    dnf install -y vector-db-benchmark
    from="$(dnf repoquery --installed --qf '%{from_repo}' vector-db-benchmark 2>/dev/null | tail -n1)"
    [[ -z "$from" || "$from" == vdb-local-test || "$from" == "@System" ]] || fail "installed from '$from', not the local repository"
  else
    yum install -y vector-db-benchmark
  fi
  rpm -q vector-db-benchmark | sed 's/^/installed from repo: /'
fi
run_checks "$crate_version"
echo "test-package.sh: $(basename "$pkg") OK on ${PRETTY_NAME:-?} $(uname -m)"
