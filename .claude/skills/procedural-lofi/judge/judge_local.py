"""What `modal_judge.py` decides, in the standard library alone so
`test_judge_local.py` runs it without the Modal client."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_batch import validate

# a run's outcome as its exit code: a score that fails is 1, and anything
# that isn't a score at all is 2
EXIT_CODES = {"PASS": 0, "UNCHANGED": 0, "REGRESSION": 1, "INSUFFICIENT": 1}
NOT_A_SCORE = 2


def exit_code(outcome):
    return EXIT_CODES.get(outcome, NOT_A_SCORE)


def select(root, force_score):
    """The batch's pair count and the takes to send: the ones whose sides
    differ, since a byte-identical pair can only tie, or with `force_score`
    every one. The batch is validated first, against the commits its own
    COMMIT files name, so an empty, one-sided or COMMIT-less batch is
    refused rather than read as unchanged."""
    shas = [(root / side / "COMMIT").read_text().strip() if (root / side / "COMMIT").is_file() else "" for side in ("base", "cand")]
    pairs = validate(root, *shas)
    names = sorted(p.name for p in (root / "base").glob("*.wav"))
    differ = [n for n in names if force_score or (root / "base" / n).read_bytes() != (root / "cand" / n).read_bytes()]
    return pairs, differ


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
