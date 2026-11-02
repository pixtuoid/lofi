#!/usr/bin/env bash
# Renders the judge batch at TREE's committed head and judges it against
# BASE_DIR's renders of its base, under OUT, which stays outside the repo: a
# factor gets a verdict before it gets a PR. One at a time: a judge takes
# `judge_local.py`'s SHARDS, and two at once queue until they time out.
#
#   ab.sh TREE BASE_DIR OUT
set -euo pipefail

tree=$1 base_dir=$2 out=$3

if [ -n "$(git -C "$tree" status --porcelain)" ]; then
    echo "ab.sh: $tree has uncommitted changes" >&2
    exit 1
fi
cand_sha=$(git -C "$tree" rev-parse HEAD)
work="$out/${cand_sha:0:12}"
mkdir -p "$work/base"
work=$(cd "$work" && pwd)
judge_dir="$tree/.claude/skills/procedural-lofi/judge"
modal_version=$(sed -nE 's/^modal==([^ ]+).*/\1/p' "$judge_dir/modal-requirements.txt")
base_sha=$(cat "$base_dir/COMMIT")

(cd "$tree" && cargo build --release --example lofi_audition)
"$tree/target/release/examples/lofi_audition" --batch judge --out "$work/cand" >"$work/render.log" 2>&1

rendered=$(find "$work/cand" -name '*.wav' | wc -l | tr -d ' ')
expected=$(find "$base_dir" -name '*.wav' | wc -l | tr -d ' ')
if [ "$rendered" != "$expected" ]; then
    echo "ab.sh: rendered $rendered of $expected takes" >&2
    tail -1 "$work/render.log" >&2
    exit 1
fi

printf '%s\n' "$cand_sha" >"$work/cand/COMMIT"
# the validator refuses a symlinked side; hard links cost no disk
for file in "$base_dir"/*; do
    ln -f "$file" "$work/base/"
done

status=0
(cd "$judge_dir" && uvx --from "modal==$modal_version" modal run modal_judge.py::judge \
    --batch "$work" --base "$base_sha" --cand "$cand_sha" --out "$work/out") \
    >"$work/judge.log" 2>&1 </dev/null || status=$?

echo "== ${cand_sha:0:12} exit $status"
grep -E '^(group|day |night |rain|twilight|all )|verdict:|invalid|takes: base' "$work/judge.log" |
    grep -vE '^(day|night|rain|twilight)_' | sed -E 's/^FAD calibration.*takes: /  /' | cut -c1-120
# a REGRESSION is this experiment's result; a judge that never scored isn't one
exit $((status == 1 ? 0 : status))
