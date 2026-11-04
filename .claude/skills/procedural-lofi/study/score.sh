#!/usr/bin/env bash
# Prints TREE's committed head's scorecards, one per style: renders the judge
# batch, splits each style's working rooms' takes into stems on Modal,
# measures them as the style's references were measured (`targets.sh`), and
# reads them against the style's targets.json entry.
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

# each style's rooms where every lane plays, as every lane does in the
# tracks' stems: a style's own group, gen_STYLE_MIX_STYLE_SEED.wav
styles=()
for take in "$out/all/gen_"*.wav; do
    IFS=_ read -r _ group mix style _ <<<"$(basename "$take")"
    [ "$group" = "$style" ] || continue
    case $mix in busy | moderate) ln -f "$take" "$out/in/${name}_$(basename "$take")" ;; esac
    case " ${styles[*]-} " in *" $style "*) ;; *) styles+=("$style") ;; esac
done

modal_run() {
    uvx --with numpy --with scipy --from "modal==$modal_version" modal run "$@" </dev/null
}

cd "$here"
modal_run stems.py::split_takes --takes "$out/in"
for style in "${styles[@]}"; do
    takes="${prefix}gen_${style}_"
    dir="$out/$style"
    mkdir -p "$dir"
    modal_run analyze.py::run --prefix "$takes" --out "$dir"
    modal_run basstone.py::run --prefix "$takes" --out "$dir"
    modal_run stemfad.py::run --corpus "$style" --prefix "$takes" --out "$dir"
    "$HOME/.cache/lofi-judge/bin/python" -I timbre.py "$dir/timbre.json" "$out/in/${name}_gen_${style}_"*.wav
    echo "== $style"
    python3 -I scorecard.py score "$dir" "$takes" "$style"
done
