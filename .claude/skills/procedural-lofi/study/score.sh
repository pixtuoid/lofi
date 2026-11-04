#!/usr/bin/env bash
# Prints TREE's committed head's scorecards, one per style its rooms play:
# renders the judge batch, splits the working rooms' takes into stems on Modal,
# measures them as their style's references were measured (`targets.sh`), and
# reads them against that style's targets.json entry.
#
#   score.sh TREE OUT
#
# OUT stays outside the repo: it holds renders, in a directory of each commit's
# own, so no card reads another's takes. A Volume's stems are keyed by the
# take's name, which the head's sha makes fresh, so a dirty tree is refused
# rather than scored under a commit that isn't what played.
set -euo pipefail

# each mood's style: the playlist its record is fitted to (#93)
styles=(day:hiphop night:sleep)

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
for pair in "${styles[@]}"; do
    for room in "${pair%%:*}_busy" "${pair%%:*}_moderate"; do
        for take in "$out/all/gen_${room}_"*.wav; do
            ln -f "$take" "$out/in/${name}_$(basename "$take")"
        done
    done
done

modal_run() {
    uvx --with numpy --with scipy --from "modal==$modal_version" modal run "$@" </dev/null
}

cd "$here"
modal_run stems.py::split_takes --takes "$out/in"
for pair in "${styles[@]}"; do
    mood=${pair%%:*} style=${pair#*:}
    takes="${prefix}gen_${mood}_"
    dir="$out/$style"
    mkdir -p "$dir"
    modal_run analyze.py::run --prefix "$takes" --out "$dir"
    modal_run basstone.py::run --prefix "$takes" --out "$dir"
    modal_run stemfad.py::run --corpus "$style" --prefix "$takes" --out "$dir"
    "$HOME/.cache/lofi-judge/bin/python" -I timbre.py "$dir/timbre.json" "$out/in/${name}_gen_${mood}_"*.wav
    echo "== $style: the $mood rooms"
    python3 -I scorecard.py score "$dir" "$takes" "$style"
done
