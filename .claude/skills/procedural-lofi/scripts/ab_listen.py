#!/usr/bin/env python3
"""Blind A/B of a sound change. Pairs the renders a baseline batch and a
candidate batch share by name (rendered as SKILL.md's GENERATOR loop step 5
says), plays each pair in a random order with the candidate on a random
side, and records every verdict with its unblinding.
Verdicts save after each pair, and a rerun with the same --out resumes.

Usage: python3 ab_listen.py BASE_DIR CAND_DIR --out VERDICT.json
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

# lofi_audition's unshuffled names: gen_<mood>_<tier or solo lane>_<seed>.wav
NAME = re.compile(r"gen_(?P<mood>[a-z]+)_(?P<mix>[a-z]+)_\d+\.wav")
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
    ap.add_argument("base", type=Path, help="the baseline batch's directory")
    ap.add_argument("cand", type=Path, help="the candidate batch's directory")
    ap.add_argument("--out", type=Path, required=True, help="the verdict file (JSON)")
    args = ap.parse_args()

    dirs = {"base": str(args.base.resolve()), "cand": str(args.cand.resolve())}
    pairs = load(args.out, dirs)
    judged = {p["name"] for p in pairs}
    base = {p.name for p in args.base.glob("*.wav")}
    cand = {p.name for p in args.cand.glob("*.wav")}
    if base ^ cand:
        print(f"unpaired, left out: {', '.join(sorted(base ^ cand))}")
    # a --shuffle run draws its own take_XX order, so equal names there are
    # different takes
    opaque = sorted(n for n in base & cand if not NAME.fullmatch(n))
    if opaque:
        sys.exit(f"not unshuffled lofi_audition names, so a pair may hold two takes: {', '.join(opaque)}")
    names = sorted((base & cand) - judged)
    if not names and not pairs:
        sys.exit("no .wav name is in both directories")
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
                src = (args.cand if side == cand_side else args.base) / name
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
