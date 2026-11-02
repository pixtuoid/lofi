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
mkdir -p "$dir"

# one run per commit at a time, so a second reads the first's render: mkdir
# is atomic, and a lock a SIGKILL left stays until removed by hand
# (https://mywiki.wooledge.org/BashFAQ/045)
until mkdir "$dir/lock" 2>/dev/null; do
    [ -n "${waiting:-}" ] || echo "render.sh: waiting on $dir/lock" >&2
    waiting=1
    sleep 1
done
src="" stage=""
trap 'rm -rf ${src:+"$src"} ${stage:+"$stage"}; rmdir "$dir/lock"' EXIT

if [ ! -f "$all/COMMIT" ]; then
    src=$(mktemp -d "$dir/src.XXXXXX")
    stage=$(mktemp -d "$dir/stage.XXXXXX")
    git -C "$tree" archive "$sha" | tar -x -C "$src"
    # a target of its own: `git archive` stamps each file with the commit's time
    # (git-archive(1)), and cargo rebuilds a source only if it is newer than the
    # last build (https://github.com/rust-lang/cargo/blob/master/src/compiler/fingerprint/mod.rs,
    # "Mtime comparison"), so a shared target rendered another commit's binary
    (cd "$src" && cargo build --locked --release --example lofi_audition)
    if ! "$src/target/release/examples/lofi_audition" --batch judge --out "$stage/all" \
        >"$dir/render.log" 2>&1; then
        echo "render.sh: the render at ${sha:0:12} failed; see $dir/render.log" >&2
        tail -3 "$dir/render.log" >&2
        exit 1
    fi
    printf '%s\n' "$sha" >"$stage/all/COMMIT"
    # an `all` with no COMMIT never finished, and `mv` would nest the render in it
    rm -rf "$all"
    mv "$stage/all" "$all"
fi
printf '%s\n' "$all"
