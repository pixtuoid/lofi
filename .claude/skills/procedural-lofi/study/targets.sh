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
# the manifest's tracks, not whatever else sits in audio/
files=()
while IFS= read -r file; do
    files+=("$refs/audio/$file")
done < <(python3 -I -c 'import json, sys; print("\n".join(t["file"] for t in json.load(open(sys.argv[1]))["tracks"]))' "$refs/manifest.json")
"$HOME/.cache/lofi-judge/bin/python" -I timbre.py "$out/timbre.json" "${files[@]}"
python3 -I scorecard.py targets "$style" "$out"
