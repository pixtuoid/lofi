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
dir="$(cd "$out" && pwd)/g${sha:0:12}"
all="$dir/all"

if [ ! -f "$all/COMMIT" ]; then
    mkdir -p "$dir"
    # each run's own, so a run alongside for the same commit touches neither
    src=$(mktemp -d "$dir/src.XXXXXX")
    stage=$(mktemp -d "$dir/stage.XXXXXX")
    trap 'rm -rf "$src" "$stage"' EXIT
    git -C "$tree" archive "$sha" | tar -x -C "$src"
    # a target of its own: `git archive` stamps the commit's time, and a target
    # another commit built later than that read as fresh, its binary rendered
    (cd "$src" && cargo build --locked --release --example lofi_audition)
    "$src/target/release/examples/lofi_audition" --batch judge --out "$stage/all" \
        >"$stage/render.log" 2>&1
    printf '%s\n' "$sha" >"$stage/all/COMMIT"
    mv -f "$stage/render.log" "$dir/render.log"
    # a run alongside may have published first; an `all` with no COMMIT never
    # finished, and `mv` would nest the render inside it
    if [ ! -f "$all/COMMIT" ]; then
        rm -rf "$all"
        mv "$stage/all" "$all"
    fi
fi
printf '%s\n' "$all"
