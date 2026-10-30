#!/usr/bin/env bash
# One-time AWS setup of the production package repository (release-packages.yml's
# `build` environment): bucket vdb-packages in us-east-2, in the AWS account of
# the staging bucket (726902207197), plus the publisher role the workflow assumes.
#
#   scripts/release/setup-production-repo.sh                # DRY RUN: reads state, prints every change
#   scripts/release/setup-production-repo.sh --apply        # makes the changes
#   options: --account <id> (default 726902207197)  --role <name> (default vdb-packages-publisher)
#            --repo <owner/name> (default redis/vector-db-benchmark)  --reviewer <github login>
#
# Credentials come from the usual AWS CLI chain; set AWS_PROFILE (e.g. `bench`)
# and `aws sso login` first. The caller's account must be --account, or nothing
# is done. Idempotent: every step is a create-if-missing or a put that sets the
# whole state, so a re-run converges and changes nothing that is already right.
#
# What it sets up, mirroring the staging bucket (PR #370):
#  * the bucket, versioned, tagged team/owner (the repository's ownership tags);
#  * public READS through object ACLs, as staging: ObjectOwnership
#    BucketOwnerPreferred (ACLs enabled: publish-apt.sh / publish-yum.sh and
#    deb-s3 upload with --acl public-read), the bucket's own ACL private, and
#    the public-access block allowing object ACLs but no public policy;
#  * public WRITES blocked by a bucket policy of Deny statements only: no
#    write from outside the account, no public-read object outside deb/, rpm/
#    and gpg, no public bucket ACL;
#  * the GitHub OIDC provider (if the account lacks it) and the role
#    vdb-packages-publisher, assumable ONLY by this repository's `build`
#    environment (both the immutable and the plain-name subject, as staging's
#    role), with an inline policy limited to this bucket: list it; get, put and
#    put-ACL objects under deb/, rpm/ and gpg; delete only deb-s3's lockfile.
# It does NOT create the signing key, the GitHub environment, its secrets or
# the variables: it prints those commands at the end.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"

apply=0 account=726902207197 role=vdb-packages-publisher repo=redis/vector-db-benchmark reviewer=fcostaoliveira
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) apply=1; shift ;;
    --account) account="$2"; shift 2 ;;
    --role) role="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --reviewer) reviewer="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "setup-production-repo.sh: unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done
[[ "$account" =~ ^[0-9]{12}$ ]] || { echo "setup-production-repo.sh: --account must be a 12-digit AWS account id" >&2; exit 2; }
[[ "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo "setup-production-repo.sh: --repo must be owner/name" >&2; exit 2; }
command -v aws >/dev/null || { echo "setup-production-repo.sh: aws CLI not on PATH" >&2; exit 1; }

bucket="$PRODUCTION_BUCKET" region="$PRODUCTION_REGION"
base="$(s3_origin_url "$bucket" "$region")"
arn="arn:aws:s3:::$bucket"
oidc_host=token.actions.githubusercontent.com
role_arn="arn:aws:iam::$account:role/$role"
# shellcheck disable=SC2054  # each element is one Key=...,Value=... shorthand
tags_kv=(Key=team,Value=performance_analysis_optimization Key=owner,Value=filipe_oliveira)

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [[ "$apply" == 1 ]]; then mode=APPLY; else mode="DRY RUN (nothing is changed; --apply to make the changes)"; fi
echo "== setup-production-repo.sh: $mode"
echo "   bucket s3://$bucket ($region), served at $base"
echo "   role   $role_arn, for $repo environment 'build'"

# `run <cmd...>`: a CHANGE. Printed in a dry run, executed with --apply.
run() {
  if [[ "$apply" == 1 ]]; then
    echo "+ $*"; "$@"
  else
    printf 'would run:'; printf ' %q' "$@"; echo
  fi
}
step() { echo; echo "-- $*"; }

# ---- who we are --------------------------------------------------------------
online=1
if ! caller="$(aws sts get-caller-identity --query Account --output text 2>"$work/sts.err")"; then
  if [[ "$apply" == 1 ]]; then
    echo "setup-production-repo.sh: no AWS credentials ($(grep -m1 . "$work/sts.err" || true)); set AWS_PROFILE and run 'aws sso login'" >&2
    exit 1
  fi
  online=0
  echo "WARNING: no AWS credentials ($(grep -m1 . "$work/sts.err" || true)): the current state is unknown, so every step is listed as if nothing existed yet."
elif [[ "$caller" != "$account" ]]; then
  echo "setup-production-repo.sh: these credentials are for account $caller, not $account (the staging bucket's); refusing" >&2
  exit 1
else
  echo "   caller account $caller"
fi

# ---- the bucket ----------------------------------------------------------------
step "bucket s3://$bucket"
exists=0
if [[ "$online" == 1 ]]; then
  rc=0; aws s3api head-bucket --bucket "$bucket" --expected-bucket-owner "$account" >/dev/null 2>"$work/head.err" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    exists=1
    loc="$(aws s3api get-bucket-location --bucket "$bucket" --expected-bucket-owner "$account" --query LocationConstraint --output text)"
    [[ "$loc" == "$region" ]] || { echo "setup-production-repo.sh: s3://$bucket exists in '$loc', not $region" >&2; exit 1; }
    echo "exists in $region, owned by $account"
  elif grep -q '404\|Not Found' "$work/head.err"; then
    echo "does not exist yet"
  else
    echo "setup-production-repo.sh: s3://$bucket exists but is not this account's (or cannot be read):" >&2
    cat "$work/head.err" >&2
    exit 1
  fi
fi
if [[ "$exists" == 0 ]]; then
  run aws s3api create-bucket --bucket "$bucket" --region "$region" \
    --create-bucket-configuration "LocationConstraint=$region" --object-ownership BucketOwnerPreferred
fi
s3() { run aws s3api "$@" --bucket "$bucket" --region "$region" --expected-bucket-owner "$account"; }
s3 put-bucket-versioning --versioning-configuration Status=Enabled
s3 put-bucket-tagging --tagging "TagSet=[{Key=team,Value=performance_analysis_optimization},{Key=owner,Value=filipe_oliveira}]"
# ACLs enabled, as staging: the publish scripts upload with --acl public-read.
s3 put-bucket-ownership-controls --ownership-controls 'Rules=[{ObjectOwnership=BucketOwnerPreferred}]'
# Object ACLs may make objects public (that is how deb/, rpm/ and gpg are
# served); no bucket POLICY may ever grant public access.
s3 put-public-access-block --public-access-block-configuration \
  BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=true,RestrictPublicBuckets=true
s3 put-bucket-acl --acl private

cat > "$work/bucket-policy.json" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyWritesFromOutsideTheAccount",
      "Effect": "Deny",
      "Principal": "*",
      "Action": ["s3:PutObject", "s3:PutObjectAcl", "s3:PutObjectVersionAcl", "s3:DeleteObject", "s3:DeleteObjectVersion", "s3:PutBucketAcl"],
      "Resource": ["$arn", "$arn/*"],
      "Condition": {"StringNotEquals": {"aws:PrincipalAccount": "$account"}}
    },
    {
      "Sid": "DenyPublicObjectsOutsideTheRepository",
      "Effect": "Deny",
      "Principal": "*",
      "Action": ["s3:PutObject", "s3:PutObjectAcl"],
      "NotResource": ["$arn/deb/*", "$arn/rpm/*", "$arn/gpg"],
      "Condition": {"StringEquals": {"s3:x-amz-acl": ["public-read", "public-read-write", "authenticated-read"]}}
    },
    {
      "Sid": "DenyPublicWriteObjectAcls",
      "Effect": "Deny",
      "Principal": "*",
      "Action": ["s3:PutObject", "s3:PutObjectAcl"],
      "Resource": "$arn/*",
      "Condition": {"StringEquals": {"s3:x-amz-acl": ["public-read-write", "authenticated-read"]}}
    },
    {
      "Sid": "DenyPublicBucketAcl",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutBucketAcl",
      "Resource": "$arn",
      "Condition": {"StringEquals": {"s3:x-amz-acl": ["public-read", "public-read-write", "authenticated-read"]}}
    }
  ]
}
JSON
echo "bucket policy (Deny statements only, so BlockPublicPolicy accepts it):"
sed 's/^/    /' "$work/bucket-policy.json"
s3 put-bucket-policy --policy "file://$work/bucket-policy.json"

# ---- GitHub OIDC provider ---------------------------------------------------------
step "IAM OIDC provider $oidc_host"
have_oidc=0
if [[ "$online" == 1 ]]; then
  providers="$(aws iam list-open-id-connect-providers --query 'OpenIDConnectProviderList[].Arn' --output text)"
  for p in $providers; do [[ "$p" == */"$oidc_host" ]] && have_oidc=1; done
fi
if [[ "$have_oidc" == 1 ]]; then
  echo "exists (staging's role uses it too)"
else
  # GitHub's documented thumbprints; IAM no longer checks them for this host,
  # but older CLIs require the argument.
  run aws iam create-open-id-connect-provider --url "https://$oidc_host" --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 1c58a3a8518e8759bf075b76b750d4f2df264fcd \
    --tags "${tags_kv[@]}"
fi

# ---- the publisher role --------------------------------------------------------------
step "IAM role $role"
# The repository issues IMMUTABLE OIDC subjects (repo:<owner>@<id>/<name>@<id>:...;
# PR #370), so the trust lists that form, read from GitHub when gh is
# available, and the plain-name form, exactly as staging's role does.
subjects="\"repo:$repo:environment:build\""
if command -v gh >/dev/null && ids="$(env -u GH_TOKEN gh api "repos/$repo" --jq '"\(.owner.login)@\(.owner.id)/\(.name)@\(.id)"' 2>/dev/null)" && [[ -n "$ids" ]]; then
  subjects="\"repo:$ids:environment:build\", $subjects"
elif [[ "$repo" == redis/vector-db-benchmark ]]; then
  subjects="\"repo:redis@1529926/vector-db-benchmark@662205136:environment:build\", $subjects"
else
  echo "WARNING: could not read $repo's ids with gh; the trust lists only the plain-name subject"
fi
cat > "$work/trust.json" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Federated": "arn:aws:iam::$account:oidc-provider/$oidc_host"},
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "$oidc_host:aud": "sts.amazonaws.com",
          "$oidc_host:sub": [$subjects]
        }
      }
    }
  ]
}
JSON
echo "trust policy:"
sed 's/^/    /' "$work/trust.json"
have_role=0
if [[ "$online" == 1 ]] && aws iam get-role --role-name "$role" >/dev/null 2>&1; then have_role=1; fi
if [[ "$have_role" == 1 ]]; then
  run aws iam update-assume-role-policy --role-name "$role" --policy-document "file://$work/trust.json"
else
  run aws iam create-role --role-name "$role" --assume-role-policy-document "file://$work/trust.json" \
    --description "Publishes vector-db-benchmark packages to s3://$bucket (GitHub $repo, environment build)"
fi
run aws iam tag-role --role-name "$role" --tags "${tags_kv[@]}"

cat > "$work/publish-policy.json" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListTheRepositoryBucket",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "$arn"
    },
    {
      "Sid": "ReadAndPublishRepositoryObjects",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:PutObjectAcl"],
      "Resource": ["$arn/deb/*", "$arn/rpm/*", "$arn/gpg"]
    },
    {
      "Sid": "ReleaseTheDebS3Lock",
      "Effect": "Allow",
      "Action": "s3:DeleteObject",
      "Resource": "$arn/deb/dists/*/lockfile"
    }
  ]
}
JSON
echo "inline policy vdb-packages-publish (s3:PutObjectAcl: every object is uploaded public-read;"
echo "s3:DeleteObject: only deb-s3 --lock's lockfile; nothing else is ever deleted):"
sed 's/^/    /' "$work/publish-policy.json"
run aws iam put-role-policy --role-name "$role" --policy-name vdb-packages-publish --policy-document "file://$work/publish-policy.json"

# ---- what is left: key, environment, secrets, variables, publish -------------------------
gh_repo="-R $repo"
cat <<EOF

== Next steps (not done by this script). In order, from a trusted machine:

# 1. Our own signing key: RSA 4096 (what staging's key is; deb-s3 signs through
#    gpg and rpm --addsign through gpg, and every target's rpm, el8 included,
#    verifies RSA). No passphrase: the publish job signs non-interactively.
export GNUPGHOME="\$(mktemp -d)"; chmod 700 "\$GNUPGHOME"
gpg --batch --pinentry-mode loopback --passphrase '' \\
    --quick-gen-key 'vector-db-benchmark package signing key' rsa4096 sign 5y
FPR="\$(gpg --list-secret-keys --with-colons | awk -F: '/^fpr:/{print \$10; exit}')"; echo "\$FPR"
gpg --armor --export-secret-keys "\$FPR" > vdb-packages-signing.secret.asc
cp "\$GNUPGHOME/openpgp-revocs.d/\$FPR.rev" vdb-packages-signing.rev
#    Store vdb-packages-signing.secret.asc and .rev OFFLINE (team vault) now.
#    Before 5 years: extend with 'gpg --quick-set-expire \$FPR 5y', update
#    APT_SIGNING_KEY; the next publish re-uploads <base>/gpg with the new expiry.

# 2. The 'build' environment: required reviewer(s), deployments only from master
#    and release tags, plus a ruleset so only admins create/move/delete release tags.
REVIEWER_ID="\$(gh api users/$reviewer --jq .id)"
gh api -X PUT repos/$repo/environments/build --input - <<JSON
{"reviewers": [{"type": "User", "id": \$REVIEWER_ID}], "prevent_self_review": false,
 "deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
JSON
gh api -X POST repos/$repo/environments/build/deployment-branch-policies -f name=master -f type=branch
gh api -X POST repos/$repo/environments/build/deployment-branch-policies -f 'name=[0-9]*.[0-9]*.[0-9]*' -f type=tag
gh api -X POST repos/$repo/environments/build/deployment-branch-policies -f 'name=v[0-9]*.[0-9]*.[0-9]*' -f type=tag
gh api -X POST repos/$repo/rulesets --input - <<'JSON'
{"name": "release tags", "target": "tag", "enforcement": "active",
 "conditions": {"ref_name": {"include": ["refs/tags/[0-9]*.[0-9]*.[0-9]*", "refs/tags/v[0-9]*.[0-9]*.[0-9]*"], "exclude": []}},
 "rules": [{"type": "creation"}, {"type": "update"}, {"type": "deletion"}],
 "bypass_actors": [{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}]}
JSON

# 3. The environment's secrets (the publish job reads exactly these names).
gh secret set APT_S3_BUCKET --env build $gh_repo --body $bucket
gh secret set APT_S3_REGION --env build $gh_repo --body $region
gh secret set APT_S3_IAM_ARN --env build $gh_repo --body $role_arn
gh secret set APT_SIGNING_KEY --env build $gh_repo < vdb-packages-signing.secret.asc

# 4. The environment variable pinning the key, then switch production on
#    (PACKAGES_PRODUCTION is a REPOSITORY variable: the plan job reads it
#    before any environment is chosen).
gh variable set APT_SIGNING_FPR --env build $gh_repo --body "\$FPR"
rm -rf "\$GNUPGHOME" vdb-packages-signing.secret.asc   # once stored offline
gh variable set PACKAGES_PRODUCTION $gh_repo --body true

# 5. Publish the release that is already out, and approve the 'build' deployment.
gh workflow run release-packages.yml $gh_repo --ref master -f release_type=public -f tag=0.2.0
#    Then check $base/gpg serves \$FPR and install per the README's "Getting Started".
EOF
if [[ "$apply" != 1 ]]; then echo; echo "(dry run: nothing above was changed; re-run with --apply)"; fi
