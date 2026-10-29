#!/usr/bin/env bash
# Publish built RPMs into an S3-backed YUM repository laid out like packages.redis.io.
#
#   scripts/release/publish-yum.sh --bucket <bucket> --in <dir> [--prefix rpm] [--overwrite] [--dry-run]
#
# <dir> is searched recursively for *.rpm. Each RPM goes to
# <prefix>/<distro>/<arch>/ where <distro> comes from its own dist tag
# (Release `1.el9` -> el9, `1.amzn2023` -> amzn2023; lib.sh rpm_release_to_distro)
# and <arch> from its header, i.e. the README's baseurl pattern
# rpm/el$releasever/$basearch and rpm/amzn2023/$basearch. The bucket's rpm/
# prefix is SHARED with other Redis products (prod: redis, redis-stack; staging:
# redis-rpm's rockylinux*/ dirs), so this script:
#  * touches only the <distro>/<arch>/ directories it has an RPM for,
#  * downloads each such directory in full before regenerating its metadata,
#    so createrepo_c --update indexes everything already there (and keeps any
#    extra metadata other products placed: --keep-all-metadata),
#  * never deletes anything: RPMs are uploaded first, then the new hashed
#    metadata files, then repomd.xml LAST, so no index ever names a file that
#    is not there yet. Old metadata generations are left in repodata/ and
#    accumulate, in S3 and in each publish's download (with --update
#    --keep-all-metadata, createrepo_c keeps every older generation despite
#    `--retain-old-md 1`; and nothing deletes remotely: a `sync --delete`
#    could only ever remove files ANOTHER product uploaded after our sync,
#    which its cached repomd.xml may still name). Pruning old generations, if
#    ever needed, is by age, never by a delete-sync.
#  * refuses, before uploading anything to ANY directory, when a target
#    directory's metadata is SIGNED (repodata/repomd.xml.asc or any
#    repomd.xml.* beside it):
#    regenerating repomd.xml would leave that detached signature stale and
#    break `repo_gpgcheck=1` for every product there. None is signed today
#    (2026-10-29); if one becomes so, this must learn to re-sign repomd.xml
#    first.
# There is no cross-product lock on rpm/ (deb-s3's --lock covers only deb/),
# so the sync -> createrepo_c -> upload sequence is a read-modify-write that
# another product's publish can race. Two guards, per directory:
#  * just before uploading repodata/, the directory is listed again; RPMs that
#    appeared since the sync are downloaded and the index rebuilt once, which
#    shrinks the window to the repodata upload itself;
#  * after the upload, every *.rpm the directory holds must be listed in the
#    primary.xml.gz it now serves. One that is not fails the run, naming it:
#    re-run the publish, which re-syncs the directory and rebuilds the index.
# RESIDUAL RISK, one direction only: a product whose tooling has no such
# re-list can still upload its repodata/ AFTER our check, built from a sync
# taken before our RPM landed, and drop our RPM from the index while this run
# stays green. Only a lock shared by every product would close that; the
# workflow's verify step (run after both publishes) catches it if it happens
# within the job.
# Objects are uploaded public-read (the CDN serves per-object ACLs).
#
# Re-runs are idempotent: an RPM already in the repo with the same NEVRA,
# build time and header+payload digest is skipped (a partial publish — one directory uploaded, the next
# failed on a transient S3 error — completes on re-run); a DIFFERENT build
# under the same file name is refused. --overwrite replaces it anyway
# (staging rehearsals only).
#
# SIGNING IS MANDATORY (except --dry-run): the keyring must hold exactly one
# secret key (the workflow imports APT_SIGNING_KEY); every RPM is signed with
# it and the signature verified against a private rpm database before upload.
# --dry-run maps, copies and generates metadata locally, prints the aws
# commands instead of running them, and needs no network, credentials or key.
set -euo pipefail

bucket="" indir="" prefix=rpm overwrite=0 dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket) bucket="$2"; shift 2 ;;
    --in) indir="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "publish-yum.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$bucket" && -d "$indir" ]] || {
  echo "usage: publish-yum.sh --bucket <bucket> --in <dir> [--prefix rpm] [--overwrite] [--dry-run]" >&2; exit 2; }
for t in createrepo_c rpm; do
  command -v "$t" >/dev/null || { echo "publish-yum.sh: $t not on PATH" >&2; exit 1; }
done
[[ "$dry_run" == 1 ]] || command -v aws >/dev/null || { echo "publish-yum.sh: aws not on PATH" >&2; exit 1; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"

run() { if [[ "$dry_run" == 1 ]]; then printf 'dry-run:'; printf ' %q' "$@"; echo; else "$@"; fi; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

rpms=()
while IFS= read -r -d '' f; do rpms+=("$f"); done < <(find "$indir" -type f -name '*.rpm' -print0 | LC_ALL=C sort -z)
[[ ${#rpms[@]} -gt 0 ]] || { echo "publish-yum.sh: no .rpm under $indir" >&2; exit 1; }

secret_keys="$(gpg --list-secret-keys --with-colons 2>/dev/null || true)"
n_keys="$(grep -c '^sec:' <<<"$secret_keys" || true)"
fpr="$(awk -F: '/^fpr:/{print $10; exit}' <<<"$secret_keys")"
if [[ "$n_keys" -eq 1 && -n "$fpr" && "$dry_run" != 1 ]]; then
  echo "publish-yum.sh: signing RPMs with $fpr"
  # rpm's %__gpg defaults to a gpg2 binary name some distros (Ubuntu) do not
  # ship: point it at the gpg on PATH, and name the key by fingerprint, on the
  # command line (a stale ~/.rpmmacros %_gpg_name cannot pick another key).
  for f in "${rpms[@]}"; do rpm --define "__gpg $(command -v gpg)" --define "_gpg_name $fpr" --addsign "$f"; done
  # Verify against a private rpm database: no root, no host keyring.
  mkdir -p "$work/rpmdb"
  rpm --dbpath "$work/rpmdb" --initdb
  gpg --export --armor "$fpr" > "$work/signing-pub.asc"
  rpm --dbpath "$work/rpmdb" --import "$work/signing-pub.asc"
  for f in "${rpms[@]}"; do
    out="$(rpm --dbpath "$work/rpmdb" -Kv "$f" 2>&1 || true)"
    if grep -qwE 'BAD|NOTTRUSTED|NOKEY' <<<"$out"; then echo "publish-yum.sh: $f: signature does not verify:" >&2; echo "$out" >&2; exit 1; fi
    if ! grep -qE 'Header V[34] (RSA|DSA|EdDSA)/.*: OK' <<<"$out"; then echo "publish-yum.sh: $f: not signed:" >&2; echo "$out" >&2; exit 1; fi
  done
elif [[ "$dry_run" == 1 ]]; then
  echo "publish-yum.sh: dry run, RPMs left unsigned"
else
  echo "publish-yum.sh: the GPG keyring must hold exactly ONE secret key (found $n_keys); refusing to publish unsigned RPMs" >&2
  exit 1
fi

declare -A dirs=()
for f in "${rpms[@]}"; do
  distro="$(rpm_release_to_distro "$(rpm -qp --queryformat '%{RELEASE}' "$f")")"
  arch="$(rpm -qp --queryformat '%{ARCH}' "$f")"
  [[ "$arch" =~ ^(x86_64|aarch64)$ ]] || { echo "publish-yum.sh: $f: refusing unexpected arch '$arch'" >&2; exit 1; }
  mkdir -p "$work/$distro/$arch"
  cp "$f" "$work/$distro/$arch/"
  dirs["$distro/$arch"]=1
done

# `remote_rpms <s3://.../dir>`: the *.rpm object names directly under it, one
# per line, sorted. `aws s3 ls` exits 1 with no output for an empty prefix;
# any other failure is an error.
remote_rpms() {
  local out rc=0
  out="$(aws s3 ls "$1/" 2>"$work/ls.err")" || rc=$?
  if [[ "$rc" -ne 0 ]] && ! [[ "$rc" -eq 1 && -z "$out" && ! -s "$work/ls.err" ]]; then
    echo "publish-yum.sh: aws s3 ls $1/ failed (exit $rc):" >&2; cat "$work/ls.err" >&2; return 1
  fi
  awk '$1 != "PRE" && $NF ~ /\.rpm$/ {print $NF}' <<<"$out" | LC_ALL=C sort
}

# `served_index <s3://.../dir> <scratch dir>`: the file names the primary.xml.gz
# named by the directory's repodata/repomd.xml lists, one per line, sorted.
served_index() {
  local remote="$1" scratch="$2" primary
  rm -rf "$scratch"; mkdir -p "$scratch"
  aws s3 cp "$remote/repodata/repomd.xml" "$scratch/repomd.xml" --only-show-errors
  primary="$(sed -n 's/.*href="\(repodata\/[^"]*primary\.xml\.gz\)".*/\1/p' "$scratch/repomd.xml" | head -n1)"
  [[ -n "$primary" ]] || { echo "publish-yum.sh: $remote/repodata/repomd.xml names no primary.xml.gz" >&2; return 1; }
  aws s3 cp "$remote/$primary" "$scratch/primary.xml.gz" --only-show-errors
  gzip -dc "$scratch/primary.xml.gz" | sed -n 's/.*<location href="\([^"]*\)".*/\1/p' | sed 's|.*/||' | LC_ALL=C sort -u
}

# A build's identity: NEVRA, build time (whole seconds) and SIGMD5, the digest
# of header + payload, which signing leaves alone (the published copy is
# signed, ours is signed again on a re-run), but which tells apart two builds
# of one NEVRA made within the same second.
nevra_of() { rpm -qp --queryformat '%{NAME}-%{EPOCHNUM}:%{VERSION}-%{RELEASE}.%{ARCH} %{BUILDTIME} %{SIGMD5}' "$1" 2>/dev/null || echo unreadable; }

signed_md_refusal() { # signed_md_refusal <dir> <signature file name>
  echo "publish-yum.sh: $prefix/$1/repodata/$2 exists: this directory's metadata is signed, and regenerating" >&2
  echo "  repomd.xml would leave that signature stale (repo_gpgcheck=1 users of every product there would fail)." >&2
  echo "  Refusing. publish-yum.sh must learn to re-sign repomd.xml before it can publish there." >&2
}
if [[ "$dry_run" != 1 ]]; then
  # Every target directory first, so a refusal leaves the YUM side untouched
  # (the workflow's APT step runs before this script).
  for d in $(printf '%s\n' "${!dirs[@]}" | LC_ALL=C sort); do
    rc=0; listing="$(aws s3 ls "s3://$bucket/$prefix/$d/repodata/" 2>"$work/ls.err")" || rc=$?
    if [[ "$rc" -ne 0 ]] && ! [[ "$rc" -eq 1 && -z "$listing" && ! -s "$work/ls.err" ]]; then
      echo "publish-yum.sh: aws s3 ls s3://$bucket/$prefix/$d/repodata/ failed (exit $rc):" >&2; cat "$work/ls.err" >&2; exit 1
    fi
    sig="$(awk '$1 != "PRE" && $NF ~ /^repomd\.xml\./ {print $NF; exit}' <<<"$listing")"
    if [[ -n "$sig" ]]; then signed_md_refusal "$d" "$sig"; echo "  No RPM or metadata was uploaded by this run." >&2; exit 1; fi
  done
fi

for d in $(printf '%s\n' "${!dirs[@]}" | LC_ALL=C sort); do
  remote="s3://$bucket/$prefix/$d"
  local_dir="$work/.repo/$d"
  mkdir -p "$local_dir"
  echo "--- $d"
  # Everything already there, so the regenerated metadata covers it. A sync
  # error is a real error: without the existing files, --update would publish
  # metadata that forgets the other products' RPMs in this directory.
  run aws s3 sync "$remote/" "$local_dir/" --only-show-errors
  # Again on what the sync brought down (it may have appeared since the
  # preflight); directories before this one may already be published.
  for sig in "$local_dir"/repodata/repomd.xml.*; do
    [[ -e "$sig" ]] || continue
    signed_md_refusal "$d" "$(basename "$sig")"
    echo "  Nothing was written to $prefix/$d; directories before it in this run may already be published." >&2
    exit 1
  done
  upload=()
  for f in "$work/$d"/*.rpm; do
    b="$(basename "$f")"
    if [[ -f "$local_dir/$b" ]]; then
      have="$(nevra_of "$local_dir/$b")"; want="$(nevra_of "$f")"
      if [[ "$have" == "$want" && "$overwrite" != 1 ]]; then
        echo "already published, skipping upload: $prefix/$d/$b ($have)"
        continue
      fi
      if [[ "$overwrite" != 1 ]]; then
        echo "publish-yum.sh: $prefix/$d/$b already exists in s3://$bucket as a DIFFERENT build ($have vs $want); refusing to replace a published RPM (--overwrite for staging rehearsals)" >&2
        exit 1
      fi
    fi
    cp "$f" "$local_dir/$b"
    upload+=("$b")
  done
  createrepo_c --update --keep-all-metadata --retain-old-md 1 "$local_dir" >/dev/null
  # RPMs first, metadata last, so no index ever points at a missing file.
  for b in ${upload[@]+"${upload[@]}"}; do
    run aws s3 cp "$local_dir/$b" "$remote/$b" --acl public-read --only-show-errors
  done
  if [[ "$dry_run" != 1 ]]; then
    # Narrow the race: RPMs another product uploaded since the sync above get
    # indexed too, instead of being dropped by our repodata/ upload.
    now="$(remote_rpms "$remote")" || exit 1
    late=()
    while IFS= read -r b; do
      if [[ -n "$b" && ! -f "$local_dir/$b" ]]; then late+=("$b"); fi
    done <<<"$now"
    if [[ ${#late[@]} -gt 0 ]]; then
      echo "appeared in $prefix/$d since the sync, re-indexing: ${late[*]}"
      aws s3 sync "$remote/" "$local_dir/" --exclude 'repodata/*' --only-show-errors
      for b in "${late[@]}"; do
        [[ -f "$local_dir/$b" ]] || { echo "publish-yum.sh: $remote/$b listed but not downloaded" >&2; exit 1; }
      done
      createrepo_c --update --keep-all-metadata --retain-old-md 1 "$local_dir" >/dev/null
    fi
  fi
  # New hashed files first, repomd.xml (which names them) last. No --delete
  # (see the header).
  run aws s3 sync "$local_dir/repodata/" "$remote/repodata/" --exclude repomd.xml --acl public-read --only-show-errors
  run aws s3 cp "$local_dir/repodata/repomd.xml" "$remote/repodata/repomd.xml" --acl public-read --only-show-errors
  if [[ "$dry_run" != 1 ]]; then
    # Every RPM in the directory, ours or another product's, must be in the
    # index it now serves: a publish that raced ours between the re-list and
    # the upload would otherwise lose its package from the index silently.
    in_dir="$(remote_rpms "$remote")" || exit 1
    listed="$(served_index "$remote" "$work/check")" || exit 1
    missing="$(LC_ALL=C comm -23 <(printf '%s\n' "$in_dir") <(printf '%s\n' "$listed") | sed '/^$/d')"
    if [[ -n "$missing" ]]; then
      echo "publish-yum.sh: $remote/repodata/ does not index these RPMs in the directory:" >&2
      while IFS= read -r m; do echo "  $m" >&2; done <<<"$missing"
      echo "  Another publish most likely wrote to $prefix/$d at the same time. Re-run this publish (Re-run failed jobs):" >&2
      echo "  it re-syncs the directory and rebuilds the index, which then lists them. Ours already uploaded are skipped." >&2
      exit 1
    fi
    echo "index check: $(grep -c . <<<"$in_dir") RPMs in $prefix/$d, all listed in the served primary.xml.gz"
  fi
  echo "published: $(ls "$work/$d") -> $remote/ (uploaded: ${upload[*]:-none, all already there})"
done
echo "publish-yum.sh: done (s3://$bucket/$prefix, dirs: $(printf '%s ' "${!dirs[@]}"))"
