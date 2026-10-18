"""The gate `ab_judge.py`'s scores pass through, in the standard library alone
so `test_verdict.py` runs it without the judges' venv."""

from itertools import count
from math import comb

# the verdict's significance over every test it gates on (opinion)
ALPHA = 0.05

# each pair's gain on these, cand over base: a judge's headline score, or
# nearness to the corpus by FAD
JUDGES = ("audiobox_ce", "muq_eval_mi", "fad")

# the groups the judge batch holds, a take's first name: `examples/common/batch.rs`'s
# `MOODS`, then its `JUDGE_SWEEPS`
GROUPS = ("day", "night", "twilight", "rain")

# from worst: a group's verdict, and the batch's
RANKS = ("REGRESSION", "INSUFFICIENT", "PASS", "UNCHANGED")


def tail_p(losses, untied):
    """The exact one-sided sign test's p: the chance of at least `losses`
    successes in `untied` fair coin flips, as `scipy.stats.binomtest(losses,
    untied, 0.5, alternative="greater")` computes it
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)."""
    return sum(comb(untied, k) for k in range(losses, untied + 1)) / 2**untied if untied else 1.0


def min_untied(alpha):
    """The fewest untied pairs a sign test can reject at `alpha` with: every
    one a loss, its most extreme outcome."""
    return next(n for n in count(1) if tail_p(n, n) < alpha)


# the fewest when every judge tests every group, the most tests a batch gates
# on: the take count the judge batch builds each group to
MIN_UNTIED = min_untied(ALPHA / (len(JUDGES) * len(GROUPS)))


def sign_test(gains, alpha):
    """Whether cand does worse than base more often than chance allows at
    `alpha`, `gains` each pair's improvement from base to cand, ties dropped."""
    wins = sum(g > 0 for g in gains)
    losses = sum(g < 0 for g in gains)
    p = tail_p(losses, wins + losses)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(gains) - wins - losses,
        "p": p,
        "significant": p < alpha,
        "decidable": wins + losses >= min_untied(alpha),
    }


def grouped(pairs):
    """Each group's pairs, then every pair as "all": the changed ones, and how
    many were byte-identical, which no judge scores."""
    groups = {mood: [p for p in pairs if p["mood"] == mood] for mood in sorted({p["mood"] for p in pairs})}
    groups["all"] = pairs
    return {
        group: {
            "changed": [p for p in members if not p["identical"]],
            "identical": sum(p["identical"] for p in members),
        }
        for group, members in groups.items()
    }


def judged(pairs):
    """The verdict on `pairs`, each a dict of its group as `mood`, whether it
    is `identical`, and otherwise its `gains` on each of `JUDGES`.

    Each group with a changed pair gets a sign test a judge, at ALPHA split
    Bonferroni over every such test; a group with none is UNCHANGED. The
    pooled "all" group is tested alike but gates nothing, since it repeats
    the groups' pairs."""
    groups = grouped(pairs)
    moods = [g for g in groups if g != "all"]
    tests = len(JUDGES) * sum(bool(groups[m]["changed"]) for m in moods)
    alpha = ALPHA / tests if tests else ALPHA
    for group in groups.values():
        if not group["changed"]:
            group["verdict"] = "UNCHANGED"
            continue
        group["tests"] = {j: sign_test([p["gains"][j] for p in group["changed"]], alpha) for j in JUDGES}
        group["verdict"] = (
            "REGRESSION"
            if any(t["significant"] for t in group["tests"].values())
            else "INSUFFICIENT"
            if not all(t["decidable"] for t in group["tests"].values())
            else "PASS"
        )
    gating = {groups[m]["verdict"] for m in moods}
    return {
        "alpha": alpha,
        "min_untied": min_untied(alpha),
        "groups": groups,
        "overall": next((r for r in RANKS if r in gating), "UNCHANGED"),
    }


def exit_message(result):
    """Why the batch fails, so the script exits non-zero, or None when it
    passes or changed nothing."""
    return {
        "REGRESSION": "cand scores worse than base",
        "INSUFFICIENT": f"a group has fewer than {result['min_untied']} untied pairs in some test,"
        f" too few to ever reject at {result['alpha']:.4g}",
    }.get(result["overall"])
