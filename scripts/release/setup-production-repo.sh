#!/usr/bin/env bash
# One-time (and re-runnable) setup of the production package repository, the
# side release-packages.yml's `build` environment publishes to: bucket
# vdb-packages in us-east-2, in the AWS account of the staging bucket
# (726902207197), the GitHub `build` environment's protection, and the
# publisher role the workflow assumes.
#
#   scripts/release/setup-production-repo.sh            # DRY RUN: reads the live state, prints what would change
#   scripts/release/setup-production-repo.sh --apply    # makes those changes
#   options: --account <id> (default 726902207197)   --role <name> (default vdb-packages-publisher)
#            --repo <owner/name> (default redis/vector-db-benchmark)   --reviewer <github login>
#            --admin-arn <IAM ARN pattern> (default: this account's SSO AdministratorAccess role)
#
# Needs: aws (set AWS_PROFILE, e.g. `bench`, and `aws sso login`), gh logged in
# with admin on the repository, python3. Idempotent: each setting is read and
# compared first, and only what differs is changed (a dry run against the live
# state prints exactly that), so a re-run converges on existing resources.
#
# What it sets up, in this order:
#  1. the GitHub environment `build`, protected (required reviewer, deployments
#     only from master and the release tags) and the "release tags" ruleset.
#     Created or fixed only when missing; existing branch policies and the
#     ruleset are never re-posted. This comes BEFORE the role: GitHub creates
#     an UNprotected environment for any job that names one, so a role trusting
#     `environment:build` must not exist while `build` is unprotected. With
#     --apply the role step refuses unless `build` is protected.
#  2. the bucket, versioned, tagged team/owner, noncurrent versions expired
#     after a year; public READS through object ACLs, as staging
#     (ObjectOwnership BucketOwnerPreferred: the publish scripts upload with
#     --acl public-read; bucket ACL private; the public-access block allows
#     object ACLs but no public policy); and a bucket policy of Deny statements:
#       * object writes and deletes by anyone but the publisher role and the
#         admin role (the account is shared: instance roles and CI users hold
#         s3:* on *);
#       * bucket policy/ACL/ownership/public-access-block/versioning/lifecycle
#         changes and bucket deletion by anyone but the admin role;
#       * any ACL outside deb/, rpm/ and gpg; public-read-write/authenticated
#         canned ACLs anywhere; public bucket ACLs; any x-amz-grant-* header.
#     Lock-out safety: --apply refuses unless the caller IS the admin role
#     (--admin-arn), and the account root can always delete a bucket policy.
#  3. the GitHub OIDC provider (if the account lacks it) and the role, trusting
#     ONLY the subject GitHub issues for `build` (the repository's immutable
#     `repo:<owner>@<id>/<name>@<id>` form, read from its OIDC customization;
#     the plain-name form only if that cannot be read), with an inline policy
#     limited to this bucket: list it; get, put and put-ACL objects under deb/,
#     rpm/ and gpg; delete only deb-s3's lockfile.
# It does NOT create the signing key, the secrets or the variables: it prints
# those steps (scripts/release/setup-signing-key.sh) at the end, gated on this
# code being on master.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"

apply=0 account=726902207197 role=vdb-packages-publisher repo=redis/vector-db-benchmark reviewer=fcostaoliveira admin_arn=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) apply=1; shift ;;
    --account) account="$2"; shift 2 ;;
    --role) role="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --reviewer) reviewer="$2"; shift 2 ;;
    --admin-arn) admin_arn="$2"; shift 2 ;;
    -h|--help) sed -n '2,49p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "setup-production-repo.sh: unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done
[[ "$account" =~ ^[0-9]{12}$ ]] || { echo "setup-production-repo.sh: --account must be a 12-digit AWS account id" >&2; exit 2; }
[[ "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo "setup-production-repo.sh: --repo must be owner/name" >&2; exit 2; }
admin_arn="${admin_arn:-arn:aws:iam::$account:role/aws-reserved/sso.amazonaws.com/*AWSReservedSSO_AdministratorAccess_*}"
for t in aws python3; do command -v "$t" >/dev/null || { echo "setup-production-repo.sh: $t not on PATH" >&2; exit 1; }; done

bucket="$PRODUCTION_BUCKET" region="$PRODUCTION_REGION"
base="$(s3_origin_url "$bucket" "$region")"
arn="arn:aws:s3:::$bucket"
oidc_host=token.actions.githubusercontent.com
role_arn="arn:aws:iam::$account:role/$role"
# shellcheck disable=SC2054  # each element is one Key=...,Value=... shorthand
tags_kv=(Key=team,Value=performance_analysis_optimization Key=owner,Value=filipe_oliveira)
release_tag_patterns=('[0-9]*.[0-9]*.[0-9]*' 'v[0-9]*.[0-9]*.[0-9]*')

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [[ "$apply" == 1 ]]; then mode=APPLY; else mode="DRY RUN (nothing is changed; --apply to make the changes)"; fi
echo "== setup-production-repo.sh: $mode"
echo "   bucket s3://$bucket ($region), served at $base"
echo "   role   $role_arn, for $repo environment 'build'"
echo "   admin  $admin_arn (exempt from the bucket policy's write denies)"

changes=0
# `run <cmd...>`: a CHANGE. Printed in a dry run, executed with --apply.
run() {
  changes=$((changes + 1))
  if [[ "$apply" == 1 ]]; then
    echo "+ $*"; "$@"
  else
    printf 'would run:'; printf ' %q' "$@"; echo
  fi
}
step() { echo; echo "-- $*"; }
ok() { echo "ok: $*"; }
die() { echo "setup-production-repo.sh: $*" >&2; exit 1; }
ghc() { env -u GH_TOKEN gh "$@"; }
# `json_same <current file> <desired file>`: true when equal as JSON (key order
# ignored); otherwise prints a unified diff of the two, pretty-printed.
json_same() {
  python3 - "$1" "$2" <<'PY'
import difflib, json, sys
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
a, b = load(sys.argv[1]), load(sys.argv[2])
if a == b:
    sys.exit(0)
fmt = lambda x: json.dumps(x, indent=2, sort_keys=True).splitlines()
for line in difflib.unified_diff(fmt(a), fmt(b), "live", "desired", lineterm=""):
    print("    " + line)
sys.exit(1)
PY
}

# ---- who we are --------------------------------------------------------------
online=1
if ! caller="$(aws sts get-caller-identity --query Arn --output text 2>"$work/sts.err")"; then
  [[ "$apply" != 1 ]] || die "no AWS credentials ($(grep -m1 . "$work/sts.err" || true)); set AWS_PROFILE and run 'aws sso login'"
  online=0
  echo "WARNING: no AWS credentials ($(grep -m1 . "$work/sts.err" || true)): the AWS state is unknown, so every AWS step is listed as if nothing existed yet."
else
  [[ "$caller" == "arn:aws:sts::$account:"* || "$caller" == "arn:aws:iam::$account:"* ]] ||
    die "these credentials ($caller) are not for account $account (the staging bucket's); refusing"
  # The bucket policy denies writes and policy changes to every principal but
  # the publisher and $admin_arn; aws:PrincipalArn of an assumed-role session
  # is the ROLE's ARN, path included, so resolve it before comparing.
  principal="$caller"
  if [[ "$caller" == *":assumed-role/"* ]]; then
    rname="${caller#*:assumed-role/}"; rname="${rname%%/*}"
    principal="$(aws iam get-role --role-name "$rname" --query Role.Arn --output text 2>/dev/null)" || principal=""
  fi
  echo "   caller $caller"
  echo "   principal ${principal:-<unresolved>}"
  # shellcheck disable=SC2053  # the right side is a glob, as ArnLike's * is
  if [[ -n "$principal" && "$principal" == $admin_arn ]]; then
    ok "the caller is the exempt admin role: a re-run can always change the bucket again"
  elif [[ "$apply" == 1 ]]; then
    die "the caller's principal (${principal:-unresolved}) does not match --admin-arn $admin_arn; the bucket policy this applies would lock it out of re-running. Run as that role (or pass the right --admin-arn)."
  else
    echo "WARNING: the caller does not match --admin-arn; --apply would refuse."
  fi
fi

# ---- 1. GitHub environment `build` (before the role) ------------------------------
step "GitHub environment 'build' on $repo"
have_gh=0
if command -v gh >/dev/null && ghc auth status >/dev/null 2>&1; then have_gh=1; fi
env_protected=0
if [[ "$have_gh" == 0 ]]; then
  [[ "$apply" != 1 ]] || die "gh is not available or not logged in; it is needed to protect 'build' before the role is created"
  echo "WARNING: gh not available: listing the environment steps as if nothing existed."
  env_json='{}'
else
  env_json="$(ghc api "repos/$repo/environments/build" 2>/dev/null)" || env_json='{}'
fi
env_state() { # prints: missing | unprotected | protected
  python3 -c '
import json, sys
e = json.loads(sys.argv[1])
if not e.get("name"):
    print("missing"); sys.exit()
rev = [r for r in e.get("protection_rules", []) if r.get("type") == "required_reviewers" and r.get("reviewers")]
dbp = e.get("deployment_branch_policy") or {}
print("protected" if rev and dbp.get("custom_branch_policies") else "unprotected")' "$1"
}
state="$(env_state "$env_json")"
if [[ "$state" == protected ]]; then
  ok "'build' has required reviewers and custom deployment branch policies (left as they are)"
else
  echo "'build' is $state: set one required reviewer ($reviewer) and custom deployment branch policies"
  reviewer_id=0
  if [[ "$have_gh" == 1 ]]; then reviewer_id="$(ghc api "users/$reviewer" --jq .id)"; fi
  printf '{"reviewers": [{"type": "User", "id": %s}], "prevent_self_review": false, "deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}\n' "$reviewer_id" > "$work/env.json"
  run env -u GH_TOKEN gh api -X PUT "repos/$repo/environments/build" --input "$work/env.json"
fi
policies=""
if [[ "$have_gh" == 1 && "$state" != missing ]]; then
  policies="$(ghc api "repos/$repo/environments/build/deployment-branch-policies" --jq '.branch_policies[] | "\(.type) \(.name)"' 2>/dev/null)" || policies=""
fi
for want in "branch master" "tag ${release_tag_patterns[0]}" "tag ${release_tag_patterns[1]}"; do
  if grep -qxF "$want" <<<"$policies"; then
    ok "deployment policy: $want"
  else
    run env -u GH_TOKEN gh api -X POST "repos/$repo/environments/build/deployment-branch-policies" -f "name=${want#* }" -f "type=${want%% *}"
  fi
done
rulesets=""
if [[ "$have_gh" == 1 ]]; then rulesets="$(ghc api "repos/$repo/rulesets" --jq '.[] | select(.target == "tag") | "\(.id) \(.name)"' 2>/dev/null)" || rulesets=""; fi
if grep -q ' release tags$' <<<"$rulesets"; then
  ok "tag ruleset: $(grep ' release tags$' <<<"$rulesets" | head -n1) (only admins create, move or delete release tags)"
else
  cat > "$work/ruleset.json" <<JSON
{"name": "release tags", "target": "tag", "enforcement": "active",
 "conditions": {"ref_name": {"include": ["refs/tags/${release_tag_patterns[0]}", "refs/tags/${release_tag_patterns[1]}"], "exclude": []}},
 "rules": [{"type": "creation"}, {"type": "update"}, {"type": "deletion"}],
 "bypass_actors": [{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}]}
JSON
  run env -u GH_TOKEN gh api -X POST "repos/$repo/rulesets" --input "$work/ruleset.json"
fi
if [[ "$apply" == 1 ]]; then
  [[ "$(env_state "$(ghc api "repos/$repo/environments/build")")" == protected ]] ||
    die "'build' is still not protected; refusing to touch the role"
  env_protected=1
elif [[ "$state" == protected ]]; then
  env_protected=1
fi

# ---- 2. the bucket ----------------------------------------------------------------
step "bucket s3://$bucket"
exists=0
if [[ "$online" == 1 ]]; then
  rc=0; aws s3api head-bucket --bucket "$bucket" --expected-bucket-owner "$account" >/dev/null 2>"$work/head.err" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    exists=1
    loc="$(aws s3api get-bucket-location --bucket "$bucket" --expected-bucket-owner "$account" --query LocationConstraint --output text)"
    [[ "$loc" == "$region" ]] || die "s3://$bucket exists in '$loc', not $region"
    ok "exists in $region, owned by $account"
  elif grep -q '404\|Not Found' "$work/head.err"; then
    echo "does not exist yet"
  else
    cat "$work/head.err" >&2
    die "s3://$bucket exists but is not this account's (or cannot be read)"
  fi
fi
if [[ "$exists" == 0 ]]; then
  run aws s3api create-bucket --bucket "$bucket" --region "$region" \
    --create-bucket-configuration "LocationConstraint=$region" --object-ownership BucketOwnerPreferred
fi
s3() { run aws s3api "$@" --bucket "$bucket" --region "$region" --expected-bucket-owner "$account"; }
# `live <s3api get-... args>`: the live value (JSON), or nothing when absent/unknown.
live() {
  [[ "$exists" == 1 ]] || return 0
  aws s3api "$@" --bucket "$bucket" --region "$region" --expected-bucket-owner "$account" --output json 2>/dev/null || true
}
# `converge <label> <live json> <desired json> <put command...>`
converge() {
  local label="$1" have="$2" want="$3"; shift 3
  printf '%s' "$have" > "$work/have.json"; printf '%s' "$want" > "$work/want.json"
  if json_same "$work/have.json" "$work/want.json" > "$work/diff.txt"; then ok "$label"; else
    echo "$label differs:"; cat "$work/diff.txt"; "$@"; fi
}
converge "versioning Enabled" "$(live get-bucket-versioning)" '{"Status": "Enabled"}' \
  s3 put-bucket-versioning --versioning-configuration Status=Enabled
converge "tags team/owner" "$(live get-bucket-tagging)" \
  '{"TagSet": [{"Key": "owner", "Value": "filipe_oliveira"}, {"Key": "team", "Value": "performance_analysis_optimization"}]}' \
  s3 put-bucket-tagging --tagging "TagSet=[{Key=owner,Value=filipe_oliveira},{Key=team,Value=performance_analysis_optimization}]"
# ACLs enabled, as staging: the publish scripts upload with --acl public-read.
converge "ObjectOwnership BucketOwnerPreferred" "$(live get-bucket-ownership-controls)" \
  '{"OwnershipControls": {"Rules": [{"ObjectOwnership": "BucketOwnerPreferred"}]}}' \
  s3 put-bucket-ownership-controls --ownership-controls 'Rules=[{ObjectOwnership=BucketOwnerPreferred}]'
# Object ACLs may make objects public (that is how deb/, rpm/ and gpg are
# served); no bucket POLICY may ever grant public access.
converge "public-access block (object ACLs allowed, public policies blocked)" "$(live get-public-access-block)" \
  '{"PublicAccessBlockConfiguration": {"BlockPublicAcls": false, "IgnorePublicAcls": false, "BlockPublicPolicy": true, "RestrictPublicBuckets": true}}' \
  s3 put-public-access-block --public-access-block-configuration \
    BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=true,RestrictPublicBuckets=true
acl_grants="$(live get-bucket-acl | python3 -c 'import json,sys
d = sys.stdin.read().strip()
print(json.dumps([g["Permission"] + " " + (g["Grantee"].get("URI") or g["Grantee"].get("Type")) for g in json.loads(d)["Grants"]] if d else None))')"
converge "bucket ACL private (owner FULL_CONTROL only)" "$acl_grants" '["FULL_CONTROL CanonicalUser"]' \
  s3 put-bucket-acl --acl private
# Every publish rewrites the indexes; old versions are kept a year (forensics,
# rollback), then expire. Incomplete multipart uploads are cleaned up.
cat > "$work/lifecycle.json" <<'JSON'
{"Rules": [{"ID": "expire-noncurrent-versions", "Status": "Enabled", "Filter": {"Prefix": ""},
            "NoncurrentVersionExpiration": {"NoncurrentDays": 365},
            "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}}]}
JSON
converge "lifecycle (noncurrent versions expire after 365 days)" \
  "$(live get-bucket-lifecycle-configuration | python3 -c 'import json,sys
d = sys.stdin.read().strip()
if d:
    d = json.loads(d); d.pop("TransitionDefaultMinimumObjectSize", None); print(json.dumps(d))')" \
  "$(cat "$work/lifecycle.json")" \
  s3 put-bucket-lifecycle-configuration --lifecycle-configuration "file://$work/lifecycle.json"

grant_deny() { # one statement per grant header: Null keys within one block are ANDed
  printf '    {"Sid": "DenyAclGrantHeader%s", "Effect": "Deny", "Principal": "*",\n' "$1"
  printf '     "Action": ["s3:PutObject", "s3:PutObjectAcl", "s3:PutObjectVersionAcl", "s3:PutBucketAcl"],\n'
  printf '     "Resource": ["%s", "%s/*"], "Condition": {"Null": {"s3:x-amz-grant-%s": "false"}}}' "$arn" "$arn" "$2"
}
cat > "$work/bucket-policy.json" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyObjectWritesExceptPublisherAndAdmin",
      "Effect": "Deny",
      "Principal": "*",
      "Action": ["s3:PutObject", "s3:PutObjectAcl", "s3:PutObjectVersionAcl", "s3:DeleteObject", "s3:DeleteObjectVersion"],
      "Resource": "$arn/*",
      "Condition": {"ArnNotLike": {"aws:PrincipalArn": ["$role_arn", "$admin_arn"]}}
    },
    {
      "Sid": "DenyBucketChangesExceptAdmin",
      "Effect": "Deny",
      "Principal": "*",
      "Action": ["s3:PutBucketPolicy", "s3:DeleteBucketPolicy", "s3:PutBucketAcl", "s3:PutBucketOwnershipControls",
                 "s3:PutBucketPublicAccessBlock", "s3:PutBucketVersioning", "s3:PutLifecycleConfiguration", "s3:DeleteBucket"],
      "Resource": "$arn",
      "Condition": {"ArnNotLike": {"aws:PrincipalArn": "$admin_arn"}}
    },
    {
      "Sid": "DenyObjectAclsOutsideTheRepository",
      "Effect": "Deny",
      "Principal": "*",
      "Action": ["s3:PutObjectAcl", "s3:PutObjectVersionAcl"],
      "NotResource": ["$arn/deb/*", "$arn/rpm/*", "$arn/gpg"]
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
    },
$(grant_deny Read read),
$(grant_deny Write write),
$(grant_deny ReadAcp read-acp),
$(grant_deny WriteAcp write-acp),
$(grant_deny FullControl full-control)
  ]
}
JSON
python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$work/bucket-policy.json" || die "internal error: the bucket policy is not valid JSON"
# Last of the bucket steps: from here on only the admin role may change the bucket.
converge "bucket policy" "$(live get-bucket-policy | python3 -c 'import sys,json
d = sys.stdin.read().strip()
if d: print(json.loads(d)["Policy"])')" "$(cat "$work/bucket-policy.json")" \
  s3 put-bucket-policy --policy "file://$work/bucket-policy.json"

# ---- 3. GitHub OIDC provider and the publisher role ------------------------------------
step "IAM OIDC provider $oidc_host"
have_oidc=0
if [[ "$online" == 1 ]]; then
  for p in $(aws iam list-open-id-connect-providers --query 'OpenIDConnectProviderList[].Arn' --output text); do
    [[ "$p" == */"$oidc_host" ]] && have_oidc=1
  done
fi
if [[ "$have_oidc" == 1 ]]; then ok "exists"; else
  # GitHub's documented thumbprints; IAM no longer checks them for this host,
  # but older CLIs require the argument.
  run aws iam create-open-id-connect-provider --url "https://$oidc_host" --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 1c58a3a8518e8759bf075b76b750d4f2df264fcd \
    --tags "${tags_kv[@]}"
fi

step "IAM role $role"
# Trust exactly the subject GitHub issues for a job in `build`: the
# repository's OIDC customization says which form (immutable ids or names).
subject=""
if [[ "$have_gh" == 1 ]] && sub_json="$(ghc api "repos/$repo/actions/oidc/customization/sub" 2>/dev/null)"; then
  subject="$(python3 -c '
import json, sys
d = json.loads(sys.argv[1])
if d.get("use_default") is False:
    sys.exit("custom sub claim keys")
print((d.get("sub_claim_prefix") or "repo:" + sys.argv[2]) + ":environment:build")' "$sub_json" "$repo")" ||
    die "$repo uses a custom OIDC sub claim; this script only knows the default subject format"
  ok "subject from the repository's OIDC customization: $subject"
else
  subject="repo:$repo:environment:build"
  echo "WARNING: could not read $repo's OIDC customization; trusting the plain-name subject $subject (if the repository issues immutable subjects, the role cannot be assumed until a re-run with gh)."
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
          "$oidc_host:sub": "$subject"
        }
      }
    }
  ]
}
JSON
cat > "$work/publish-policy.json" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "ListTheRepositoryBucket", "Effect": "Allow", "Action": "s3:ListBucket", "Resource": "$arn"},
    {"Sid": "ReadAndPublishRepositoryObjects", "Effect": "Allow",
     "Action": ["s3:GetObject", "s3:PutObject", "s3:PutObjectAcl"],
     "Resource": ["$arn/deb/*", "$arn/rpm/*", "$arn/gpg"]},
    {"Sid": "ReleaseTheDebS3Lock", "Effect": "Allow", "Action": "s3:DeleteObject", "Resource": "$arn/deb/dists/*/lockfile"}
  ]
}
JSON
if [[ "$apply" == 1 && "$env_protected" != 1 ]]; then die "'build' is not protected; refusing to create or update the role"; fi
have_role=0
if [[ "$online" == 1 ]] && aws iam get-role --role-name "$role" --output json > "$work/role.json" 2>/dev/null; then have_role=1; fi
if [[ "$have_role" == 1 ]]; then
  converge "trust policy (environment build only)" \
    "$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["Role"]["AssumeRolePolicyDocument"]))' "$work/role.json")" \
    "$(cat "$work/trust.json")" \
    run aws iam update-assume-role-policy --role-name "$role" --policy-document "file://$work/trust.json"
  converge "role tags team/owner" \
    "$(aws iam list-role-tags --role-name "$role" --output json | python3 -c 'import json,sys; print(json.dumps(sorted((t["Key"], t["Value"]) for t in json.load(sys.stdin)["Tags"] if t["Key"] in ("team", "owner"))))')" \
    '[["owner", "filipe_oliveira"], ["team", "performance_analysis_optimization"]]' \
    run aws iam tag-role --role-name "$role" --tags "${tags_kv[@]}"
else
  [[ "$env_protected" == 1 ]] || echo "(the role is created only after 'build' is protected, above)"
  echo "trust policy:"; sed 's/^/    /' "$work/trust.json"
  run aws iam create-role --role-name "$role" --assume-role-policy-document "file://$work/trust.json" \
    --description "Publishes vector-db-benchmark packages to s3://$bucket (GitHub $repo, environment build)" \
    --tags "${tags_kv[@]}"
fi
have_policy=""
if [[ "$have_role" == 1 ]]; then
  have_policy="$(aws iam get-role-policy --role-name "$role" --policy-name vdb-packages-publish \
    --query PolicyDocument --output json 2>/dev/null)" || have_policy=""
fi
converge "inline policy vdb-packages-publish (put-ACL: objects are public-read; delete: deb-s3's lockfile only)" \
  "$have_policy" "$(cat "$work/publish-policy.json")" \
  run aws iam put-role-policy --role-name "$role" --policy-name vdb-packages-publish --policy-document "file://$work/publish-policy.json"

echo
if [[ "$changes" -eq 0 ]]; then echo "== Nothing to change: the live state is already what this script sets up."
elif [[ "$apply" == 1 ]]; then echo "== Applied $changes change(s)."
else echo "== $changes change(s) would be made (re-run with --apply)."; fi

# ---- what is left: key + secrets, variables, publish ----------------------------------
# PACKAGES_PRODUCTION and a public dispatch need THIS code on master: master's
# older workflow would publish with packages.redis.io's Origin and no <base>/gpg.
master_ready=0
if [[ "$have_gh" == 1 ]] && ghc api "repos/$repo/contents/scripts/release/lib.sh?ref=master" \
     -H "Accept: application/vnd.github.raw+json" > "$work/master-lib.sh" 2>/dev/null &&
   grep -qx 'APT_ORIGIN=vector-db-benchmark' "$work/master-lib.sh" && grep -q '^check_publish_target()' "$work/master-lib.sh"; then
  master_ready=1
fi
cat <<EOF

== Next steps (not done by this script), in order:

# 1. The signing key and the 'build' secrets + APT_SIGNING_FPR, from a machine
#    with gpg. Writes the key material to ~/vdb-packages-signing (mode 700),
#    never into the checkout:
scripts/release/setup-signing-key.sh
EOF
if [[ "$master_ready" == 1 ]]; then
  cat <<EOF

# 2. Switch production on (a REPOSITORY variable: the plan job reads it before
#    any environment is chosen). master has check_publish_target and the
#    vector-db-benchmark Origin (checked just now).
gh variable set PACKAGES_PRODUCTION -R $repo --body true

# 3. Publish the release that is already out; approve the 'build' deployment.
gh workflow run release-packages.yml -R $repo --ref master -f release_type=public -f tag=0.2.0
#    Then install from $base per the README's "Getting Started" on bare
#    el8, el9, el10, amzn2023, jammy and trixie containers.
EOF
else
  cat <<EOF

# 2-3. STOP: master does not have this code yet (no check_publish_target /
#      APT_ORIGIN=vector-db-benchmark in master's scripts/release/lib.sh, or gh
#      could not read it). Merge #386 first. Setting PACKAGES_PRODUCTION=true or
#      dispatching release_type=public before that runs master's OLD workflow,
#      which publishes with packages.redis.io's Origin and no <base>/gpg.
#      Re-run this script after the merge to print those two commands.
EOF
fi
if [[ "$apply" != 1 ]]; then echo; echo "(dry run: nothing above was changed; re-run with --apply)"; fi
