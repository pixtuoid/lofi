#!/usr/bin/env bash
# Measures STYLE's private corpus as score.sh measures a commit's takes, and
# writes STYLE's quartiles into targets.json; the per-track measures stay in
# OUT, so the targets regenerate from the corpus alone.
#
#   targets.sh STYLE REFS OUT
#
# REFS is the corpus, its audio/ and manifest.json, never committed; OUT stays
# outside the repo. A reference's stems are keyed STYLE-NNN on the Volume.
set -euo pipefail

style=$1 refs=$(cd "$2" && pwd) out=$3
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
modal_version=$(sed -nE 's/^modal==([^ ]+).*/\1/p' "$here/../judge/modal-requirements.txt")
mkdir -p "$out"
out=$(cd "$out" && pwd)

modal_run() {
    uvx --with numpy --with scipy --from "modal==$modal_version" modal run "$@" </dev/null
}

cd "$here"
modal_run ../judge/modal_judge.py::upload_refs --refs-dir "$refs" --corpus "$style"
modal_run stems.py::split --corpus "$style"
modal_run analyze.py::run --prefix "$style-" --out "$out"
modal_run basstone.py::run --prefix "$style-" --out "$out"
modal_run stemfad.py::run --corpus "$style" --out "$out"
"$HOME/.cache/lofi-judge/bin/python" -I timbre.py "$out/timbre.json" "$refs/audio/"*
python3 -I scorecard.py targets "$style" "$out"
