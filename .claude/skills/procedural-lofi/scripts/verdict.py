"""The gate `ab_judge.py`'s scores pass through, in the standard library alone
so `test_verdict.py` runs it without the judges' venv."""

from itertools import count
from math import comb

# the verdict's significance (opinion)
ALPHA = 0.05


def tail_p(losses, untied):
    """The exact one-sided sign test's p: the chance of at least `losses`
    successes in `untied` fair coin flips, as `scipy.stats.binomtest(losses,
    untied, 0.5, alternative="greater")` computes it
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)."""
    return sum(comb(untied, k) for k in range(losses, untied + 1)) / 2**untied if untied else 1.0


# the fewest untied pairs a sign test can reject at ALPHA with: every one a
# loss, its most extreme outcome
MIN_UNTIED = next(n for n in count(1) if tail_p(n, n) < ALPHA)


def sign_test(gains):
    """Whether cand does worse than base more often than chance allows,
    `gains` each pair's improvement from base to cand, ties dropped."""
    wins = sum(g > 0 for g in gains)
    losses = sum(g < 0 for g in gains)
    p = tail_p(losses, wins + losses)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(gains) - wins - losses,
        "p": p,
        "significant": p < ALPHA,
        "decidable": wins + losses >= MIN_UNTIED,
    }


def group_verdict(tests):
    """REGRESSION if any test rejects, INSUFFICIENT if one never could, else
    PASS."""
    if any(t["significant"] for t in tests.values()):
        return "REGRESSION"
    if not all(t["decidable"] for t in tests.values()):
        return "INSUFFICIENT"
    return "PASS"


def batch_verdict(groups):
    """The batch's verdict over its groups': REGRESSION if any is, since a
    detected regression stands however few pairs another group had, then
    INSUFFICIENT if any is, else PASS."""
    for worst in ("REGRESSION", "INSUFFICIENT"):
        if worst in groups:
            return worst
    return "PASS"


def exit_message(overall):
    """Why the batch fails, so the script exits non-zero, or None on PASS."""
    return {
        "REGRESSION": "cand scores worse than base",
        "INSUFFICIENT": f"a group has fewer than {MIN_UNTIED} untied pairs in some test, too few to ever reject at {ALPHA}",
    }.get(overall)
