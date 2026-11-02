#!/usr/bin/env bash
# Renders the judge batch from TREE and judges it against BASE_DIR's renders,
# one at a time: a judge takes SHARDS of the Starter plan's 10 GPUs, so two at
# once queue until they time out.
#
#   ab.sh NAME TREE BASE_DIR [VAR=1 ...]
#
# Bash, not zsh: the VAR=1 list has to reach `env` as separate words.
set -euo pipefail

name=$1 tree=$2 base_dir=$3
shift 3

work="${LOFI_AB_DIR:?set LOFI_AB_DIR to a directory outside the repo}/$name"
judge_dir="$tree/.claude/skills/procedural-lofi/judge"
modal_version=$(sed -nE 's/^modal==([^ ]+).*/\1/p' "$judge_dir/modal-requirements.txt")
base_sha=$(cat "$base_dir/COMMIT")
cand_sha=$(printf '%s' "$name $*" | shasum | cut -c1-40)

mkdir -p "$work/base"
(cd "$tree" && cargo build --release --example lofi_audition)
env "$@" "$tree/target/release/examples/lofi_audition" --batch judge --out "$work/cand" >"$work/render.log" 2>&1

rendered=$(find "$work/cand" -name '*.wav' | wc -l | tr -d ' ')
expected=$(find "$base_dir" -name '*.wav' | wc -l | tr -d ' ')
if [ "$rendered" != "$expected" ]; then
    echo "$name: rendered $rendered of $expected takes" >&2
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

echo "== $name ($*) exit $status"
grep -E '^(group|day |night |rain|twilight|all )|verdict:|invalid|takes: base' "$work/judge.log" |
    grep -vE '^(day|night|rain|twilight)_' | sed -E 's/^FAD calibration.*takes: /  /' | cut -c1-120
# a REGRESSION is this experiment's result, not its failure
exit 0
