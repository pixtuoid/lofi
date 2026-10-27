#!/usr/bin/env python3
"""The remote judge's local logic's tests: `python3 -I test_judge_local.py`, with no venv."""

import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_local import NOT_A_SCORE, exit_code, published, redact, run_steps, select
from test_validate_batch import write_wav
from validate_batch import Takes, stage, validate

# the takes `Selection` writes on both sides
PAIRS = ["gen_day_empty_000.wav", "gen_night_busy_001.wav"]
BASE, CAND = "a" * 40, "b" * 40
TAKES = Takes(PAIRS, [], [], PAIRS[1:], PAIRS[:1])


class Selection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for side, sha in (("base", BASE), ("cand", CAND)):
            (self.dir / side).mkdir()
            (self.dir / side / "COMMIT").write_text(sha + "\n")
            write_wav(self.dir / side / "gen_day_empty_000.wav")
            write_wav(self.dir / side / "gen_night_busy_001.wav", frames=8 if side == "base" else 9)

    def tearDown(self):
        self.tmp.cleanup()

    def assertRefused(self, base_sha, cand_sha, reason):
        """`select` ends the run as no score at all, never as a verdict, and
        says why on stderr."""
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()) as err:
            select(self.dir, base_sha, cand_sha, force_score=False)
        self.assertEqual(caught.exception.code, NOT_A_SCORE)
        self.assertIn(f"invalid batch: {reason}", err.getvalue())

    def test_only_the_pairs_that_differ_are_sent(self):
        self.assertEqual(select(self.dir, BASE, CAND, force_score=False), (TAKES, PAIRS[1:]))

    def test_force_score_sends_every_pair(self):
        self.assertEqual(select(self.dir, BASE, CAND, force_score=True), (TAKES, PAIRS))

    def test_a_staged_batch_sends_only_the_pairs_it_holds(self):
        up = self.dir / "up"
        stage(self.dir, validate(self.dir, BASE, CAND), up)
        for force_score in (False, True):
            self.assertEqual(select(up, BASE, CAND, force_score), (TAKES, PAIRS[1:]))

    def test_a_batch_of_other_commits_is_refused(self):
        self.assertRefused(CAND, BASE, "base/COMMIT doesn't name")

    def test_a_batch_with_no_pair_is_refused(self):
        for side in ("base", "cand"):
            for p in (self.dir / side).glob("*.wav"):
                p.unlink()
        self.assertRefused(BASE, CAND, "no take on both sides")

    def test_a_take_on_one_side_only_is_left_out_and_the_pairs_still_sent(self):
        (self.dir / "cand" / "gen_day_empty_000.wav").unlink()
        write_wav(self.dir / "cand" / "gen_dusk_moderate_000.wav")
        takes = Takes(PAIRS[1:], ["gen_dusk_moderate_000.wav"], ["gen_day_empty_000.wav"], PAIRS[1:], [])
        self.assertEqual(select(self.dir, BASE, CAND, force_score=True), (takes, PAIRS[1:]))

    def test_a_batch_whose_takes_are_each_on_one_side_is_refused(self):
        (self.dir / "cand" / "gen_day_empty_000.wav").unlink()
        (self.dir / "base" / "gen_night_busy_001.wav").unlink()
        self.assertRefused(BASE, CAND, "no take on both sides")

    def test_a_missing_commit_is_refused(self):
        (self.dir / "cand" / "COMMIT").unlink()
        self.assertRefused(BASE, CAND, "cand/COMMIT doesn't name")


class Outcomes(unittest.TestCase):
    def test_only_a_score_that_fails_exits_one_and_anything_else_two(self):
        self.assertEqual([exit_code(o) for o in ("PASS", "UNCHANGED")], [0, 0])
        self.assertEqual([exit_code(o) for o in ("REGRESSION", "INSUFFICIENT")], [1, 1])
        self.assertEqual(exit_code("infra failure"), 2)


class Redaction(unittest.TestCase):
    TRACKS = [{"file": "Some-Artist-Some-Title.mp3"}, {"file": "Some-Artist-Some-Title-Extended.mp3"}]

    def test_full_names_and_stems_become_manifest_indices(self):
        text = (
            "/refs/audio/Some-Artist-Some-Title-Extended.mp3 failed;"
            " Some-Artist-Some-Title decoded; Some-Artist-Some-Title.mp3 too"
        )
        self.assertEqual(redact(text, self.TRACKS), "/refs/audio/ref#1 failed; ref#0 decoded; ref#0 too")


class Publishing(unittest.TestCase):
    TRACKS = Redaction.TRACKS

    def test_the_record_publishes_its_own_table_masked_and_no_corpus_name(self):
        record = {
            "dir": "/batch",
            "judges": {"fad": {"refs": "/refs", "lufs": -20.0}},
            "fad_calibration": {"Some-Artist-Some-Title.mp3": 0.4, "Some-Artist-Some-Title-Extended.mp3": 0.2},
            "table": "take dCE\nday_empty_000 +0.01\nnearest Some-Artist-Some-Title",
            "overall": "PASS",
        }
        got = published(record, self.TRACKS)
        self.assertEqual(got["outcome"], "PASS")
        self.assertEqual(got["table"], "take dCE\nday_empty_000 +0.01\nnearest ref#0")
        self.assertEqual(got["record"], {"judges": {"fad": {"lufs": -20.0}}, "fad_calibration": [0.2, 0.4], "overall": "PASS"})


class RustCopies(unittest.TestCase):
    def test_the_warm_up_take_is_at_lofis_rate(self):
        root = Path(__file__).resolve().parents[4]
        rust = re.search(r"pub const SAMPLE_RATE: u32 = ([\d_]+);", (root / "src" / "dsp.rs").read_text())
        py = re.search(r"^WARMUP_RATE = (\d+)$", (root / ".claude/skills/procedural-lofi/judge/modal_judge.py").read_text(), re.M)
        self.assertIsNotNone(rust, "no SAMPLE_RATE in src/dsp.rs")
        self.assertIsNotNone(py, "no WARMUP_RATE in modal_judge.py")
        self.assertEqual(int(py[1]), int(rust[1].replace("_", "")))



class Steps(unittest.TestCase):
    def run_steps(self, *codes, timeout_s=10):
        """`run_steps` over a Python step per code in `codes`: the step it
        names and what it printed."""
        steps = [(f"step {i}", [sys.executable, "-I", "-c", code]) for i, code in enumerate(codes)]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            failed = run_steps(steps, None, timeout_s)
        return failed, out.getvalue()

    def test_steps_that_all_pass_name_none_and_print_their_time_but_not_their_stderr(self):
        failed, out = self.run_steps("print('one')", "import sys; print('two'); print('noise', file=sys.stderr)")
        self.assertIsNone(failed)
        self.assertIn("one", out)
        self.assertIn("two", out)
        self.assertNotIn("noise", out)
        self.assertRegex(out, r"the step 0 took \d+ s")
        self.assertRegex(out, r"the step 1 took \d+ s")

    def test_the_first_step_to_fail_is_named_with_its_stderr_and_ends_the_run(self):
        failed, out = self.run_steps(
            "print('one')", "import sys; print('broke', file=sys.stderr); sys.exit(1)", "print('never')"
        )
        self.assertEqual(failed, "step 1")
        self.assertIn("broke", out)
        self.assertNotIn("never", out)

    def test_a_step_past_the_limit_is_named_with_what_it_printed(self):
        failed, out = self.run_steps("import time; print('loading', flush=True); time.sleep(60)", timeout_s=1)
        self.assertEqual(failed, "step 0")
        self.assertIn("loading", out)
        self.assertIn("ran past 1 s", out)


if __name__ == "__main__":
    unittest.main()
