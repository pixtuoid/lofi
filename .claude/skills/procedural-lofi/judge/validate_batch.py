#!/usr/bin/env python3
"""Checks an A/B batch before the judge reads it, and pairs its takes, in the
standard library alone: the batch is the PR's render, so nothing in it runs
or is trusted. Every tool that pairs a batch's takes pairs them here.

Usage: python3 -I validate_batch.py DIR --base SHA --cand SHA [--stage UP]
  DIR holds base/ and cand/. Valid when every take is a `NAME`, a PCM wav
  whose data the header accounts for byte for byte, beside a COMMIT naming
  SHA, and at least one take is on both sides. Prints the takes as JSON.
  --stage moves the pairs that differ into UP, with each side's COMMIT and an
  IDENTICAL list of the rest, the batch the judge is sent.
"""

import argparse
import json
import re
import shutil
import stat
import struct
import sys
import wave
from collections import namedtuple
from pathlib import Path

sys.dont_write_bytecode = True

# lofi_audition's unshuffled names, gen_<mood>_<tier, sweep start or solo
# lane>_<seed>.wav: a --shuffle run draws its own take_XX order, so equal
# names there would pair two takes
NAME = re.compile(r"gen_(?P<mood>[a-z]+)_(?P<mix>[a-z]+)_\d+\.wav")

COMMIT = re.compile(r"[0-9a-f]{40}\n?")

# the pairs a staged batch left behind as byte-identical, one name a line
IDENTICAL = "IDENTICAL"

# the fmt chunk's format code for PCM, "0x0001 WAVE_FORMAT_PCM"
# (https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html):
# checked here, since `wave` reads IEEE float too from 3.15 ("Support for
# reading and writing WAVE_FORMAT_IEEE_FLOAT files was added",
# https://docs.python.org/3.15/library/wave.html)
WAVE_FORMAT_PCM = 1


# a valid batch's take names, each sorted: on both sides, then on cand's or
# base's alone, as a matrix that grew or shrank leaves them; then the pairs
# whose sides differ, which a judge scores, and the byte-identical ones, which
# can only tie, those an IDENTICAL list names among them
Takes = namedtuple("Takes", "paired new gone differ identical")


class Invalid(Exception):
    pass


def plain(path):
    """Whether `path` is a regular file and no link."""
    return stat.S_ISREG(path.lstat().st_mode)


def check_wav(path):
    """Raises unless `path` is a PCM wav whose chunks fill the file exactly:
    one fmt chunk, then one data chunk of whole frames, so `wave` can read no
    other data than the walk measured."""
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise Invalid(f"{path.parent.name}/{path.name} is not a RIFF WAVE file")
    if struct.unpack_from("<I", raw, 4)[0] != len(raw) - 8:
        raise Invalid(f"{path.parent.name}/{path.name}'s RIFF size isn't its length")
    fmt, data, at = None, None, 12
    while at < len(raw):
        if at + 8 > len(raw):
            raise Invalid(f"{path.parent.name}/{path.name} ends inside a chunk header")
        tag, size = raw[at : at + 4], struct.unpack_from("<I", raw, at + 4)[0]
        # a "Padding byte if n is odd" follows an odd chunk
        # (https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html)
        end = at + 8 + size + size % 2
        if end > len(raw):
            raise Invalid(f"{path.parent.name}/{path.name}'s {tag!r} chunk runs past the file")
        if tag == b"fmt ":
            if fmt is not None:
                raise Invalid(f"{path.parent.name}/{path.name} has more than one 'fmt ' chunk")
            if data is not None:
                raise Invalid(f"{path.parent.name}/{path.name}'s 'fmt ' chunk follows its 'data'")
            fmt = struct.unpack_from("<H", raw, at + 8)[0] if size >= 2 else None
        if tag == b"data":
            if data is not None:
                raise Invalid(f"{path.parent.name}/{path.name} has more than one 'data' chunk")
            data = size
        at = end
    if fmt != WAVE_FORMAT_PCM:
        raise Invalid(f"{path.parent.name}/{path.name} is no PCM wav: format code {fmt}")
    try:
        with wave.open(str(path)) as w:
            frame = w.getnchannels() * w.getsampwidth()
            if data % frame:
                raise Invalid(f"{path.parent.name}/{path.name}'s data isn't whole frames")
    except (wave.Error, EOFError) as e:
        raise Invalid(f"{path.parent.name}/{path.name} is no PCM wav: {e}") from None


def listed(batch):
    """The pairs `batch`'s IDENTICAL file names, if it has one."""
    path = batch / IDENTICAL
    if not path.exists():
        return set()
    if not plain(path):
        raise Invalid(f"{IDENTICAL} is not a plain file")
    names = path.read_text().split()
    for name in names:
        if not NAME.fullmatch(name):
            raise Invalid(f"{IDENTICAL} names {name}, not a lofi_audition take")
    return set(names)


def validate(batch, base_sha=None, cand_sha=None):
    """The batch's `Takes`; raises `Invalid` on anything else. A `None` sha
    leaves that side's COMMIT unread: a batch rendered here and listened to
    here names no commit."""
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
        if sha is not None and (
            commit is None or not COMMIT.fullmatch(commit.read_text()) or commit.read_text().strip() != sha
        ):
            raise Invalid(f"{side}/COMMIT doesn't name {sha}")
        for name in entries:
            if not NAME.fullmatch(name):
                raise Invalid(f"{side}/{name} is not a lofi_audition take")
        names[side] = set(entries)
    base, cand = names["base"], names["cand"]
    left = listed(batch)
    if left & (base | cand):
        raise Invalid(f"{IDENTICAL} names a take still in the batch: {' '.join(sorted(left & (base | cand)))}")
    if not base & cand and not left:
        raise Invalid("no take on both sides")
    for side in ("base", "cand"):
        for name in sorted(names[side]):
            check_wav(batch / side / name)
    paired = sorted(base & cand)
    same = {n for n in paired if (batch / "base" / n).read_bytes() == (batch / "cand" / n).read_bytes()}
    return Takes(
        sorted(set(paired) | left),
        sorted(cand - base),
        sorted(base - cand),
        [n for n in paired if n not in same],
        sorted(same | left),
    )


def scored(batch, takes, every=False):
    """The pairs of `batch`'s `takes` a judge scores and a listener hears:
    the ones whose sides differ, since a byte-identical pair can only tie, or
    with `every` each pair `batch` still holds, not the ones its IDENTICAL
    list names."""
    if not every:
        return takes.differ
    return [n for n in takes.paired if (batch / "base" / n).exists()]


def stage(batch, takes, up):
    """Moves `takes`' pairs that differ into `up`, beside each side's COMMIT
    and an IDENTICAL list of the rest: what the judge is sent, since a
    byte-identical pair can only tie."""
    for side in ("base", "cand"):
        (up / side).mkdir(parents=True)
        shutil.copyfile(batch / side / "COMMIT", up / side / "COMMIT")
        for name in takes.differ:
            shutil.move(batch / side / name, up / side / name)
    (up / IDENTICAL).write_text("".join(f"{name}\n" for name in takes.identical))


def main():
    ap = argparse.ArgumentParser(description="Check an A/B batch before the judge reads it.")
    ap.add_argument("dir", type=Path)
    ap.add_argument("--base", required=True, help="the commit base/ must have rendered at")
    ap.add_argument("--cand", required=True, help="the commit cand/ must have rendered at")
    ap.add_argument("--stage", type=Path, help="where to move the pairs that differ")
    args = ap.parse_args()
    try:
        takes = validate(args.dir, args.base, args.cand)
    except Invalid as e:
        sys.exit(f"invalid batch: {e}")
    if args.stage:
        stage(args.dir, takes, args.stage)
    print(json.dumps(takes._asdict()))


if __name__ == "__main__":
    main()
