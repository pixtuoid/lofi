"""What `modal_judge.py` decides, in the standard library alone so
`test_judge_local.py` runs it without the Modal client."""

import sys
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


def redact(text, tracks):
    """`text` with every reference's file name, and its stem, replaced by its
    manifest index: longest first, so a name holding another's whole goes
    first."""
    for i, t in sorted(enumerate(tracks), key=lambda it: -len(it[1]["file"])):
        text = text.replace(t["file"], f"ref#{i}").replace(Path(t["file"]).stem, f"ref#{i}")
    return text


def published(record, tracks):
    """`ab_judge.py`'s judge.json as `score` returns it: the outcome, the
    table it printed with every reference name masked, and the rest without
    the corpus's paths and names."""
    table = redact(record.pop("table"), tracks)
    record.pop("dir", None)
    record.get("judges", {}).get("fad", {}).pop("refs", None)
    if "fad_calibration" in record:
        record["fad_calibration"] = sorted(record["fad_calibration"].values())
    return {"outcome": record["overall"], "table": table, "record": record}
