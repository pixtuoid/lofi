#!/usr/bin/env bash
# Judges TREE's committed head against its merge-base with origin/main, the
# pair `judge.yml` renders, under OUT, which stays outside the repo and keeps
# each commit's render for the next run: a factor gets a verdict before it
# gets a PR. One at a time: a judge takes `judge_local.py`'s SHARDS, and two
# at once queue until they time out.
#
#   ab.sh TREE OUT
set -euo pipefail

tree=$1 out=$2
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -n "$(git -C "$tree" status --porcelain)" ]; then
    echo "ab.sh: $tree has uncommitted changes" >&2
    exit 1
fi
cand_sha=$(git -C "$tree" rev-parse HEAD)
base_sha=$(git -C "$tree" merge-base origin/main HEAD)
judge_dir="$tree/.claude/skills/procedural-lofi/judge"
modal_version=$(sed -nE 's/^modal==([^ ]+).*/\1/p' "$judge_dir/modal-requirements.txt")

base=$("$here/render.sh" "$tree" "$base_sha" "$out")
cand=$("$here/render.sh" "$tree" "$cand_sha" "$out")
work="$(cd "$out" && pwd)/ab-${cand_sha:0:12}"
mkdir -p "$work/base" "$work/cand"
# the validator refuses a symlinked side; hard links cost no disk
ln -f "$base"/* "$work/base/"
ln -f "$cand"/* "$work/cand/"

status=0
(cd "$judge_dir" && uvx --from "modal==$modal_version" modal run modal_judge.py::judge \
    --batch "$work" --base "$base_sha" --cand "$cand_sha" --out "$work/out") \
    >"$work/judge.log" 2>&1 </dev/null || status=$?

echo "== ${cand_sha:0:12} exit $status"
grep -E '^(group|day |night |rain|twilight|all )|verdict:|invalid|takes: base' "$work/judge.log" |
    grep -vE '^(day|night|rain|twilight)_' | sed -E 's/^FAD calibration.*takes: /  /' | cut -c1-120 || true
# `modal run` exited 1 when the workspace's spend limit refused it (observed),
# as the judge does for a REGRESSION, so a run with no verdict line never scored
if ! grep -q '^verdict: ' "$work/judge.log"; then
    echo "ab.sh: no verdict; see $work/judge.log" >&2
    tail -3 "$work/judge.log" >&2
    exit $((status == 0 ? 1 : status))
fi
# a REGRESSION is this experiment's result; `judge_local.py`'s NOT_A_SCORE, an
# infra failure among them, is not
exit $((status == 1 ? 0 : status))
