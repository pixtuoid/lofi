#!/usr/bin/env bash
# Renders the judge batch at COMMIT of TREE's repository, once, and prints the
# directory: OUT/g<sha>/all, which no other commit's takes share. It builds
# the commit's own files, as `git archive` writes them, so what renders is the
# commit whatever TREE holds; a render that stopped short is never read as one.
#
#   render.sh TREE COMMIT OUT
set -euo pipefail

tree=$1 out=$3
sha=$(git -C "$tree" rev-parse --verify "$2^{commit}")
mkdir -p "$out"
out=$(cd "$out" && pwd)
all="$out/g${sha:0:12}/all"

if [ ! -f "$all/COMMIT" ]; then
    mkdir -p "$out/g${sha:0:12}"
    src=$(mktemp -d "$out/src.XXXXXX")
    trap 'rm -rf "$src"' EXIT
    git -C "$tree" archive "$sha" | tar -x -C "$src"
    # one target for every commit's build: only the crate itself rebuilds
    (cd "$src" && CARGO_TARGET_DIR="$out/target" cargo build --locked --release --example lofi_audition)
    rm -rf "$all.partial"
    "$out/target/release/examples/lofi_audition" --batch judge --out "$all.partial" \
        >"$out/g${sha:0:12}/render.log" 2>&1
    printf '%s\n' "$sha" >"$all.partial/COMMIT"
    mv "$all.partial" "$all"
fi
printf '%s\n' "$all"
