#!/usr/bin/env python3
"""Blind A/B of a sound change. Pairs the takes a batch's base/ and cand/
share by name (rendered as SKILL.md's GENERATOR loop step 5 says), plays
each pair that differs in a random order with the candidate on a random side,
and records every verdict with its unblinding. A byte-identical pair is left
out: no ear tells its sides apart.
Verdicts save after each pair, and a rerun with the same --out resumes.

Usage: python3 ab_listen.py DIR --out VERDICT.json
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "judge"))
from validate_batch import NAME, Invalid, scored, validate

VERDICTS = ("cand", "base", "same")


def ask(prompt, choices):
    while True:
        answer = input(prompt).strip().lower()
        if answer in choices:
            return answer


def play(afplay, path):
    if afplay:
        print(f"  {path.stem}")
        subprocess.run([afplay, str(path)], check=True)
    else:
        print(f"  {path}")


def load(out, dirs):
    if not out.exists():
        return []
    record = json.loads(out.read_text())
    if {k: record[k] for k in dirs} != dirs:
        sys.exit(f"{out} records another pair of batches; pick a new --out")
    return record["pairs"]


def save(out, dirs, pairs):
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({**dirs, "pairs": pairs}, indent=2) + "\n")
    os.replace(tmp, out)


def tally(pairs):
    counts = Counter()
    for p in pairs:
        m = NAME.fullmatch(p["name"])
        counts[(m["mood"], m["mix"]), p["verdict"]] += 1
    print(f"\n{'mood':<6} {'mix':<9}" + "".join(f" {v:>5}" for v in VERDICTS))
    for group in sorted({g for g, _ in counts}):
        print(f"{group[0]:<6} {group[1]:<9}" + "".join(f" {counts[group, v]:>5}" for v in VERDICTS))


def main():
    ap = argparse.ArgumentParser(description="Blind A/B of a sound change.")
    ap.add_argument("dir", type=Path, help="the batch: base/ and cand/ of same-named takes")
    ap.add_argument("--out", type=Path, required=True, help="the verdict file (JSON)")
    args = ap.parse_args()

    dirs = {"dir": str(args.dir.resolve())}
    pairs = load(args.out, dirs)
    judged = {p["name"] for p in pairs}
    try:
        takes = validate(args.dir)
    except Invalid as e:
        sys.exit(f"invalid batch: {e}")
    if takes.new or takes.gone:
        print(f"unpaired, left out: {', '.join(takes.new + takes.gone)}")
    if takes.identical:
        print(f"byte-identical, left out: {len(takes.identical)} pairs")
    names = sorted(set(scored(args.dir, takes)) - judged)
    if not names and not pairs:
        sys.exit("no pair differs")
    rng = random.SystemRandom()
    rng.shuffle(names)
    afplay = shutil.which("afplay")
    if not afplay:
        print("no afplay here: play each listed file yourself, then answer")
    # each pair plays from copies named A and B, since a path names its batch
    blind = Path(tempfile.mkdtemp(prefix="ab_listen_"))
    try:
        for i, name in enumerate(names, 1):
            cand_side = rng.choice("ab")
            for side in "ab":
                src = args.dir / ("cand" if side == cand_side else "base") / name
                shutil.copyfile(src, blind / f"{side.upper()}.wav")
            print(f"\npair {i}/{len(names)}")
            while True:
                for side in "AB":
                    play(afplay, blind / f"{side}.wav")
                pick = ask("a / b / s(ame) / r(eplay)? ", ("a", "b", "s", "r"))
                if pick != "r":
                    break
            note = input("note (enter for none): ").strip()
            verdict = "same" if pick == "s" else "cand" if pick == cand_side else "base"
            pairs.append(
                {"name": name, "cand_side": cand_side, "pick": pick, "verdict": verdict, "note": note}
            )
            save(args.out, dirs, pairs)
    except (KeyboardInterrupt, EOFError):
        print(f"\nstopped; {len(pairs)} verdicts kept in {args.out}, rerun to resume")
    finally:
        shutil.rmtree(blind)
    tally(pairs)


if __name__ == "__main__":
    main()
