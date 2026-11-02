#!/usr/bin/env bash
# Prints the scorecard for TREE's committed head: renders the judge batch,
# splits the working rooms' takes into stems on Modal, measures them as the
# references were measured, and reads them against targets.json.
#
#   score.sh TREE OUT
#
# OUT stays outside the repo: it holds renders, in a directory of each commit's
# own, so no card reads another's takes. A Volume's stems are keyed by the
# take's name, which the head's sha makes fresh, so a dirty tree is refused
# rather than scored under a commit that isn't what played.
set -euo pipefail

tree=$1 out=$2
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
modal_version=$(sed -nE 's/^modal==([^ ]+).*/\1/p' "$here/../judge/modal-requirements.txt")

if [ -n "$(git -C "$tree" status --porcelain)" ]; then
    echo "score.sh: $tree has uncommitted changes" >&2
    exit 1
fi
all=$("$here/render.sh" "$tree" HEAD "$out")
out=$(dirname "$all")
name=$(basename "$out")
prefix="take-${name}_"
mkdir -p "$out/in"

# the rooms where every lane plays, as every lane does in the tracks' stems
for room in day_busy day_moderate night_busy night_moderate; do
    for take in "$out/all/gen_${room}_"*.wav; do
        ln -f "$take" "$out/in/${name}_$(basename "$take")"
    done
done

modal_run() {
    uvx --with numpy --with scipy --from "modal==$modal_version" modal run "$@" </dev/null
}

cd "$here"
modal_run stems.py::split_takes --takes "$out/in"
modal_run analyze.py::run --prefix "$prefix" --out "$out"
modal_run basstone.py::run --prefix "$prefix" --out "$out"
modal_run stemfad.py::run --prefix "$prefix" --out "$out"
"$HOME/.cache/lofi-judge/bin/python" -I timbre.py "$out/timbre.json" "$out/in/"*.wav
python3 -I scorecard.py score "$out" "$prefix"
