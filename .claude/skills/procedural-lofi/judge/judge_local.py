"""What `modal_judge.py` decides, in the standard library alone so
`test_judge_local.py` runs it without the Modal client."""

import hashlib
import json
import platform
import re
import sys
from importlib.metadata import distributions
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
# `scripts/` sits beside `judge/` here and in the Modal image alike
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
from validate_batch import Invalid, scored, validate
from verdict import FAILING, RANKS

# a run's outcome as its exit code: a score that fails is 1, and anything
# that isn't a score at all is 2
EXIT_CODES = {rank: int(rank in FAILING) for rank in RANKS}
NOT_A_SCORE = 2

# the T4s a batch takes at once, under the Starter plan's 10 GPUs
# (https://modal.com/pricing), so a rehearsal or a second PR still gets one
# (opinion)
SHARDS = 8


def exit_code(outcome):
    return EXIT_CODES.get(outcome, NOT_A_SCORE)


def select(root, base_sha, cand_sha, force_score):
    """The batch's `Takes` and the pairs to send: the ones whose sides
    differ, since a byte-identical pair can only tie, or with `force_score`
    every one the batch holds. The batch is validated first, against the
    commits the caller names: a batch with no pair, or another commit's, ends
    the run with `NOT_A_SCORE` and its reason on stderr, refused before any
    score rather than read as unchanged or as a verdict."""
    try:
        takes = validate(root, base_sha, cand_sha)
    except Invalid as e:
        print(f"invalid batch: {e}", file=sys.stderr)
        raise SystemExit(NOT_A_SCORE) from None
    return takes, scored(root, takes, force_score)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def batch_key(root, paired, sent, code):
    """The key a verdict on `root`'s pairs `paired` is kept under, before its
    corpora join it (`verdict_key`): the pairs, the ones `sent` and both
    sides' bytes of each, and every file of the judge under `code`, the
    directories its image holds, bytecode aside, so an edit to the judge
    scores anew. No COMMIT is in it: a push that renders the same bytes at a
    new commit reads the verdict they already have."""
    takes = {}
    for side in ("base", "cand"):
        for name in sorted(sent):
            with open(root / side / name, "rb") as f:
                takes[f"{side}/{name}"] = hashlib.file_digest(f, "sha256").hexdigest()
    judge = {}
    for d in code:
        for f in sorted(d.rglob("*")):
            if f.is_file() and "__pycache__" not in f.relative_to(d).parts:
                judge[str(f.relative_to(d.parent))] = sha256(f.read_bytes())
    return sha256(json.dumps({"paired": sorted(paired), "takes": takes, "judge": judge}, sort_keys=True).encode())


def environment():
    """The Python and every package the judge runs on, which compute its
    numbers as much as its models do: a lock bump, or an image rebuilt on
    another Python, keys new caches and new verdicts."""
    return [platform.python_version(), *sorted(f"{d.metadata['Name'].lower()}=={d.version}" for d in distributions())]


def verdict_key(key, manifests, installed):
    """`batch_key`'s `key` joined by each style's corpus `manifests` holds
    as bytes and the `environment` `installed`, so another selection of a
    corpus, or another environment, scores anew."""
    digests = {style: sha256(manifest) for style, manifest in manifests.items()}
    return sha256(json.dumps({"batch": key, "manifests": digests, "environment": installed}, sort_keys=True).encode())


def dealt(pairs):
    """`pairs` dealt to `SHARDS` containers, or one each if fewer. A pair
    stays whole, so no difference between two hosts enters its deltas."""
    return [pairs[i::SHARDS] for i in range(min(SHARDS, len(pairs)))]


def merged(shards, changed):
    """The seat `shards` share and their rows by take: one environment the
    judges ran in, and each corpus a shard read. A shard takes its own takes'
    level in each corpus, and a row is comparable to another only under one
    environment and one reading of its corpus, so anything but each pair in
    `changed` once, so read, ends the run rather than reach a verdict."""
    rows = [row for shard in shards for row in shard["rows"].items()]
    if sorted(name for name, _ in rows) != sorted(changed):
        sys.exit("the shards' rows aren't each pair sent, once")
    seats = [shard["seat"] for shard in shards]
    if not seats:
        return {}, {}
    if any(seat["judges"] != seats[0]["judges"] for seat in seats):
        sys.exit("the shards ran in different environments")
    corpora = {}
    for seat in seats:
        for style, corpus in seat["corpora"].items():
            if corpora.setdefault(style, corpus) != corpus:
                sys.exit(f"the shards read the {style} corpus differently")
    return {"judges": seats[0]["judges"], "corpora": dict(sorted(corpora.items()))}, dict(rows)


def redact(text, corpora):
    """`text` with every reference's file name, and its stem, replaced by its
    style and manifest index, `corpora` each style's manifest tracks: in one
    pass, so no label is read again, and longest first, so a name holding
    another's whole goes first."""
    labels = {}
    for style, tracks in corpora.items():
        for i, t in enumerate(tracks):
            for name in (t["file"], Path(t["file"]).stem):
                labels.setdefault(name, f"{style}#{i}")
    if not labels:
        return text
    names = re.compile("|".join(map(re.escape, sorted(labels, key=len, reverse=True))))
    return names.sub(lambda m: labels[m[0]], text)


def published(record, corpora):
    """`ab_judge.py`'s judge.json as `score` returns it: the outcome, the
    table it printed with every reference name of `corpora` masked, and the
    rest without the corpora's paths and names."""
    table = redact(record.pop("table"), corpora)
    record.pop("dir", None)
    for corpus in record.get("corpora", {}).values():
        corpus.pop("refs", None)
        corpus["fad_calibration"] = sorted(corpus["fad_calibration"].values())
    return {"outcome": record["overall"], "table": table, "record": record}



def replaced(corpus, held, local):
    """Why `upload_refs` refuses `corpus`'s manifest `local` over the one the
    refs Volume `held`, or None: its stems, keyed by manifest index, and the
    judge's caches, keyed by the manifest's sha256, would read the old one."""
    if held in (None, local):
        return None
    return (
        f"the refs Volume holds another {corpus} corpus: a new selection replaces it, so remove /{corpus}"
        f" there and /stems/{corpus}-* on the study Volume first, then provision if verdict.STYLES names it"
    )
