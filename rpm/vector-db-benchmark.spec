# RPM spec for vector-db-benchmark. Built by scripts/release/build-rpm.sh, which
# sets the version from Cargo.toml and passes the git commit:
#   rpmbuild -bb --define "vdb_version 0.2.0" --define "vdb_git_sha <40-hex>" rpm/vector-db-benchmark.spec
#
# The binaries are built with --features release-static: HDF5, zlib and OpenSSL
# are compiled in, so the package depends on nothing but glibc and works the
# same on el8/el9/el10 and Amazon Linux 2023 (x86_64 and aarch64).
#
# cargo/rustc are not BuildRequires: the distros' own are too old for this
# crate, so build-rpm.sh installs a current stable toolchain with rustup.

%global vdb_version %{?vdb_version}%{!?vdb_version:0.0.0}
# No -debuginfo / -debugsource: release profile, Rust binaries, nothing useful
# to extract, and it keeps the shared rpm/ repo small.
%global debug_package %{nil}
# Rust binaries are not PIE-relinkable by brp; the static HDF5 is already PIC.
%undefine _hardened_build
# Don't let brp-strip-static-archive / python byte-compile wander around.
%global __brp_python_bytecompile %{nil}

Name:           vector-db-benchmark
Version:        %{vdb_version}
Release:        1%{?dist}
Summary:        Framework for benchmarking vector search engines

License:        ASL 2.0
URL:            https://github.com/redis/vector-db-benchmark
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  cmake >= 3.12
BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  make
BuildRequires:  perl
BuildRequires:  pkgconfig
BuildRequires:  git
# OpenSSL's Configure (vendored by --features release-static) needs these
# perl modules, which el9+/AL2023 split out of the perl package.
%if 0%{?rhel} >= 9 || 0%{?amzn} >= 2023
BuildRequires:  perl-IPC-Cmd
BuildRequires:  perl-FindBin
%endif
ExclusiveArch:  x86_64 aarch64

%description
vector-db-benchmark runs upload and search workloads against vector databases
(Redis Query Engine and Vector Sets, Valkey, Dragonfly, KiviDB, Qdrant,
Elasticsearch, OpenSearch, pgvector, MongoDB, Turbopuffer, and more), with
filtered, hybrid, sparse and full-text variants, and writes per-run JSON
results with recall, throughput and latency percentiles.

The package installs the CLI (vector-db-benchmark), the dataset tools
(prepare-msmarco, generate-dataset) and the reader micro-benchmarks
(bench_npy, bench_jsonl, bench_hdf5 under %{_libexecdir}/%{name}). The
dataset catalogue and the engine configurations the CLI reads from its
working directory are installed under %{_datadir}/%{name}.

HDF5, zlib and OpenSSL are linked statically; the binaries need only glibc.

%prep
%setup -q

%build
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
command -v cargo >/dev/null || { echo "cargo not found on PATH; run scripts/release/build-rpm.sh (installs rustup)" >&2; exit 1; }
cargo --version && rustc --version
# hdf5-sys links the system libhdf5 when HDF5_DIR is set; the package must get
# the bundled static one.
unset HDF5_DIR HDF5_VERSION
# The source tarball has no .git: the commit comes from the spec define.
%if 0%{?vdb_git_sha:1}
export VDB_BUILD_GIT_SHA="%{vdb_git_sha}"
%endif
cargo build --release --locked --features release-static --bins

%install
install -d %{buildroot}%{_bindir} %{buildroot}%{_libexecdir}/%{name} \
           %{buildroot}%{_datadir}/%{name}/datasets %{buildroot}%{_datadir}/%{name}/experiments
for b in vector-db-benchmark prepare-msmarco generate-dataset; do
  install -m 0755 target/release/$b %{buildroot}%{_bindir}/$b
done
for b in bench_npy bench_jsonl bench_hdf5; do
  install -m 0755 target/release/$b %{buildroot}%{_libexecdir}/%{name}/$b
done
install -m 0644 datasets/datasets.json %{buildroot}%{_datadir}/%{name}/datasets/
cp -R datasets/random-100 %{buildroot}%{_datadir}/%{name}/datasets/
cp -R experiments/configurations %{buildroot}%{_datadir}/%{name}/experiments/
find %{buildroot}%{_datadir}/%{name} -type f -exec chmod 0644 {} + -o -type d -exec chmod 0755 {} +

%check
# A binary that cannot report its version is not worth packaging.
target/release/vector-db-benchmark --version

%files
%license LICENSE
%doc README.md
%{_bindir}/vector-db-benchmark
%{_bindir}/prepare-msmarco
%{_bindir}/generate-dataset
%{_libexecdir}/%{name}/
%{_datadir}/%{name}/

%changelog
* Thu Oct 29 2026 Redis Team <oss@redis.com> - 0.0.0-1
- See https://github.com/redis/vector-db-benchmark/releases for per-release notes.
