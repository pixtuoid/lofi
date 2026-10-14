#!/usr/bin/env python3
"""Checks an A/B batch before the judge reads it, in the standard library
alone: the batch is the PR's render, so nothing in it runs or is trusted.

Usage: python3 -I validate_batch.py DIR --base SHA --cand SHA
  DIR holds base/ and cand/. Valid when every take is an `ab_listen.NAME`
  name on both sides, a PCM wav whose data the header accounts for byte for
  byte, beside a COMMIT naming SHA, and there is at least one take.
"""

import argparse
import re
import stat
import struct
import sys
import wave
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ab_listen import NAME

COMMIT = re.compile(r"[0-9a-f]{40}\n?")


class Invalid(Exception):
    pass


def plain(path):
    """Whether `path` is a regular file and no link."""
    return stat.S_ISREG(path.lstat().st_mode)


def check_wav(path):
    """Raises unless `path` is a PCM wav whose chunks fill the file exactly and
    whose data chunk holds a whole number of frames."""
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise Invalid(f"{path.parent.name}/{path.name} is not a RIFF WAVE file")
    if struct.unpack_from("<I", raw, 4)[0] != len(raw) - 8:
        raise Invalid(f"{path.parent.name}/{path.name}'s RIFF size isn't its length")
    data, at = None, 12
    while at < len(raw):
        if at + 8 > len(raw):
            raise Invalid(f"{path.parent.name}/{path.name} ends inside a chunk header")
        tag, size = raw[at : at + 4], struct.unpack_from("<I", raw, at + 4)[0]
        # RIFF pads an odd chunk to a whole word
        end = at + 8 + size + size % 2
        if end > len(raw):
            raise Invalid(f"{path.parent.name}/{path.name}'s {tag!r} chunk runs past the file")
        if tag == b"data":
            data = size
        at = end
    try:
        with wave.open(str(path)) as w:
            frame = w.getnchannels() * w.getsampwidth()
            if data is None or data % frame or w.getnframes() * frame != data:
                raise Invalid(f"{path.parent.name}/{path.name}'s data isn't whole frames")
    except (wave.Error, EOFError) as e:
        raise Invalid(f"{path.parent.name}/{path.name} is no PCM wav: {e}") from None


def validate(batch, base_sha, cand_sha):
    """The batch's take count; raises `Invalid` on anything else."""
    names = {}
    for side, sha in (("base", base_sha), ("cand", cand_sha)):
        d = batch / side
        if d.is_symlink() or not d.is_dir():
            raise Invalid(f"{side}/ is not a directory")
        entries = {p.name: p for p in d.iterdir()}
        for name, p in entries.items():
            if not plain(p):
                raise Invalid(f"{side}/{name} is not a plain file")
        commit = entries.pop("COMMIT", None)
        if commit is None or not COMMIT.fullmatch(commit.read_text()) or commit.read_text().strip() != sha:
            raise Invalid(f"{side}/COMMIT doesn't name {sha}")
        for name in entries:
            if not NAME.fullmatch(name):
                raise Invalid(f"{side}/{name} is not a lofi_audition take")
        names[side] = set(entries)
    if names["base"] != names["cand"]:
        raise Invalid(f"takes on one side only: {', '.join(sorted(names['base'] ^ names['cand']))}")
    if not names["base"]:
        raise Invalid("no take in the batch")
    for side in ("base", "cand"):
        for name in sorted(names[side]):
            check_wav(batch / side / name)
    return len(names["base"])


def main():
    ap = argparse.ArgumentParser(description="Check an A/B batch before the judge reads it.")
    ap.add_argument("dir", type=Path)
    ap.add_argument("--base", required=True, help="the commit base/ must have rendered at")
    ap.add_argument("--cand", required=True, help="the commit cand/ must have rendered at")
    args = ap.parse_args()
    try:
        print(f"{validate(args.dir, args.base, args.cand)} pairs, each a PCM wav on both sides")
    except Invalid as e:
        sys.exit(f"invalid batch: {e}")


if __name__ == "__main__":
    main()
