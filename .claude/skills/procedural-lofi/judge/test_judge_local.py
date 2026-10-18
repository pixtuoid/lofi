#!/usr/bin/env python3
"""The remote judge's local logic's tests: `python3 -I test_judge_local.py`, with no venv."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_local import exit_code, published, redact, select
from test_validate_batch import write_wav
from validate_batch import Invalid, Takes

# the takes `Selection` writes on both sides
PAIRS = Takes(["gen_day_empty_000.wav", "gen_night_busy_001.wav"], [], [])


class Selection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for side, sha in (("base", "a" * 40), ("cand", "b" * 40)):
            (self.dir / side).mkdir()
            (self.dir / side / "COMMIT").write_text(sha + "\n")
            write_wav(self.dir / side / "gen_day_empty_000.wav")
            write_wav(self.dir / side / "gen_night_busy_001.wav", frames=8 if side == "base" else 9)

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_the_pairs_that_differ_are_sent(self):
        self.assertEqual(select(self.dir, force_score=False), (PAIRS, ["gen_night_busy_001.wav"]))

    def test_force_score_sends_every_pair(self):
        self.assertEqual(select(self.dir, force_score=True), (PAIRS, ["gen_day_empty_000.wav", "gen_night_busy_001.wav"]))

    def test_a_batch_with_no_pair_is_refused(self):
        for side in ("base", "cand"):
            for p in (self.dir / side).glob("*.wav"):
                p.unlink()
        with self.assertRaises(Invalid):
            select(self.dir, force_score=False)

    def test_a_take_on_one_side_only_is_left_out_and_the_pairs_still_sent(self):
        (self.dir / "cand" / "gen_day_empty_000.wav").unlink()
        write_wav(self.dir / "cand" / "gen_dusk_moderate_000.wav")
        self.assertEqual(select(self.dir, force_score=True), (Takes(["gen_night_busy_001.wav"], ["gen_dusk_moderate_000.wav"], ["gen_day_empty_000.wav"]), ["gen_night_busy_001.wav"]))

    def test_a_batch_whose_takes_are_each_on_one_side_is_refused(self):
        (self.dir / "cand" / "gen_day_empty_000.wav").unlink()
        (self.dir / "base" / "gen_night_busy_001.wav").unlink()
        with self.assertRaises(Invalid):
            select(self.dir, force_score=False)

    def test_a_missing_commit_is_refused(self):
        (self.dir / "cand" / "COMMIT").unlink()
        with self.assertRaises(Invalid):
            select(self.dir, force_score=False)


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


if __name__ == "__main__":
    unittest.main()
