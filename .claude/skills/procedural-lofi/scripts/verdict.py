"""The gate `ab_judge.py`'s scores pass through, in the standard library alone
so `test_verdict.py` runs it without the judges' venv."""

from itertools import count
from math import comb
from statistics import median, stdev

# the verdict's significance over every test it gates on (opinion)
ALPHA = 0.05

# the panel: learned judges of human preference, each pair's gain on them
# cand over base, which ITU-R BS.1534-3 pools as it pools listeners: "the
# median of normalized scores of all listeners that remain after
# post-screening" (§9), `ab_judge.screened` the post-screening
PANEL = ("audiobox_ce", "tunejury")
# the genre guard: nearness to the corpus by per-song FAD, which no
# listener's preference trained
GUARD = "fad"
JUDGES = PANEL + (GUARD,)
# what each group gates on
GATES = ("panel", "guard")

# the guard's margin, in the held-out references' standard deviation: Cohen's
# "small" effect, `0.2 <= d < 0.5 - Small` (`interpret_cohens_d`'s "cohen1988"
# rules, https://www.rdocumentation.org/packages/effectsize/versions/0.7.0/topics/interpret_cohens_d),
# so a move the corpus's own spread dwarfs leaves no genre (opinion)
SMALL_EFFECT = 0.2

# the groups the judge batch holds, a take's first name: `examples/common/batch.rs`'s
# `MOODS`, then its `JUDGE_SWEEPS`
GROUPS = ("day", "night", "twilight", "rain")

# from worst: a group's verdict, and the batch's
RANKS = ("REGRESSION", "INSUFFICIENT", "PASS", "UNCHANGED")
# the verdicts that fail a batch
FAILING = RANKS[:2]


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


# the fewest when every group is gated, the most tests a batch gates on: the
# take count the judge batch builds each group to
MIN_UNTIED = min_untied(ALPHA / (len(GATES) * len(GROUPS)))


def sign_test(gains, alpha):
    """Whether cand does worse than base more often than chance allows at
    `alpha`, `gains` each pair's improvement from base to cand, ties dropped;
    and whether, the same way, it does better."""
    wins = sum(g > 0 for g in gains)
    losses = sum(g < 0 for g in gains)
    p = tail_p(losses, wins + losses)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(gains) - wins - losses,
        "p": p,
        "significant": p < alpha,
        "favours_cand": tail_p(wins, wins + losses) < alpha,
        "decidable": wins + losses >= min_untied(alpha),
    }


def pooled(gains, scales):
    """The panel's gain on a pair: the median of its judges' gains, each
    judge of `scales` normalized by its spread over the held-out references.
    Of three judges it takes the majority's side, and of two their mean."""
    return median(gains[j] / scale for j, scale in scales.items())


def margin(held):
    """How far a take may move from the corpus before the guard counts it:
    `SMALL_EFFECT` of the spread the held-out references' own FADs show."""
    return SMALL_EFFECT * stdev(held)


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


def judged(pairs, held, scales):
    """The verdict on `pairs`, each a dict of its group as `mood`, whether it
    is `identical`, and otherwise its `gains` on each of `JUDGES`; `held` the
    held-out references' FADs, and `scales` each panel judge's spread.

    Each group with a changed pair gets two sign tests, at ALPHA split
    Bonferroni over every such test: the panel's `pooled` gains, and the
    guard's FAD gains past `margin`, a test of the median against it as the
    sign test takes one. Either one significant is a REGRESSION. A group with none is
    UNCHANGED. The pooled "all" group is tested alike but gates nothing,
    since it repeats the groups' pairs. Each judge's own test is kept, and a
    gating group where one significantly favours base yet the group's
    verdict isn't REGRESSION is a `dissent`, which the verdict names but
    doesn't fail on."""
    groups = grouped(pairs)
    moods = [g for g in groups if g != "all"]
    tests = len(GATES) * sum(bool(groups[m]["changed"]) for m in moods)
    alpha = ALPHA / tests if tests else ALPHA
    # a batch nothing changed in has no held-out FADs, since nothing loaded the corpus
    within = margin(held) if tests else None
    for group in groups.values():
        if not group["changed"]:
            group["verdict"] = "UNCHANGED"
            continue
        changed = group["changed"]
        group["judges"] = {j: sign_test([p["gains"][j] for p in changed], alpha) for j in JUDGES}
        group["tests"] = {
            "panel": sign_test([pooled(p["gains"], scales) for p in changed], alpha),
            "guard": sign_test([p["gains"][GUARD] + within for p in changed], alpha),
        }
        group["verdict"] = (
            "REGRESSION"
            if any(t["significant"] for t in group["tests"].values())
            else "INSUFFICIENT"
            if not all(t["decidable"] for t in group["tests"].values())
            else "PASS"
        )
    gating = {groups[m]["verdict"] for m in moods}
    dissent = [
        (m, j)
        for m in moods
        if groups[m]["verdict"] != "REGRESSION"
        for j, t in groups[m].get("judges", {}).items()
        if t["significant"]
    ]
    return {
        "alpha": alpha,
        "min_untied": min_untied(alpha),
        "margin": within,
        "groups": groups,
        "dissent": dissent,
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
