#!/usr/bin/env python3
"""The batch validator's tests: `python3 -I test_validate_batch.py`, with no venv."""

import os
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_batch import Invalid, validate

BASE, CAND = "a" * 40, "b" * 40


def riff(*chunks):
    """A RIFF WAVE file of `chunks`, `(tag, declared size, payload)`, its RIFF
    size consistent with its length."""
    body = b"WAVE" + b"".join(tag + struct.pack("<I", size) + payload for tag, size, payload in chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


# 16-bit stereo PCM at 44.1 kHz: 4-byte frames
FMT = (b"fmt ", 16, struct.pack("<HHIIHH", 1, 2, 44100, 44100 * 4, 4, 16))


def write_wav(path, frames=8):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\0" * (frames * 4))


class Batches(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for side, sha in (("base", BASE), ("cand", CAND)):
            (self.dir / side).mkdir()
            (self.dir / side / "COMMIT").write_text(sha + "\n")
            for name in ("gen_day_empty_000.wav", "gen_night_busy_003.wav"):
                write_wav(self.dir / side / name)

    def tearDown(self):
        self.tmp.cleanup()

    def assertInvalid(self, why):
        """The batch is refused with a message holding `why`."""
        with self.assertRaises(Invalid) as caught:
            validate(self.dir, BASE, CAND)
        self.assertIn(why, str(caught.exception))

    def test_a_batch_of_named_pairs_and_its_commits_is_valid(self):
        self.assertEqual(validate(self.dir, BASE, CAND), 2)

    def test_any_subset_of_takes_is_valid(self):
        for side in ("base", "cand"):
            (self.dir / side / "gen_night_busy_003.wav").unlink()
        self.assertEqual(validate(self.dir, BASE, CAND), 1)

    def test_no_pair_is_invalid(self):
        for side in ("base", "cand"):
            for p in (self.dir / side).glob("*.wav"):
                p.unlink()
        self.assertInvalid("no take")

    def test_a_take_on_one_side_only_is_invalid(self):
        (self.dir / "cand" / "gen_night_busy_003.wav").unlink()
        self.assertInvalid("one side")

    def test_a_name_lofi_audition_never_writes_is_invalid(self):
        for side in ("base", "cand"):
            write_wav(self.dir / side / "take_00.wav")
        self.assertInvalid("take_00.wav")

    def test_a_link_or_a_directory_in_a_side_is_invalid(self):
        os.symlink(self.dir / "base" / "COMMIT", self.dir / "cand" / "gen_day_moderate_001.wav")
        self.assertInvalid("not a plain file")
        (self.dir / "cand" / "gen_day_moderate_001.wav").unlink()
        (self.dir / "cand" / "sub").mkdir()
        self.assertInvalid("not a plain file")

    def test_a_side_that_is_a_link_is_invalid(self):
        (self.dir / "cand").rename(self.dir / "elsewhere")
        os.symlink(self.dir / "elsewhere", self.dir / "cand")
        self.assertInvalid("not a directory")

    def test_a_commit_other_than_the_one_given_is_invalid(self):
        (self.dir / "cand" / "COMMIT").write_text(BASE + "\n")
        self.assertInvalid("COMMIT")

    def test_a_commit_that_is_not_a_sha_is_invalid(self):
        (self.dir / "base" / "COMMIT").write_text("main\n")
        self.assertInvalid("COMMIT")

    def test_a_missing_commit_is_invalid(self):
        (self.dir / "base" / "COMMIT").unlink()
        self.assertInvalid("COMMIT")

    def test_a_take_that_is_not_a_wav_is_invalid(self):
        (self.dir / "base" / "gen_day_empty_000.wav").write_bytes(b"ID3 not a wav")
        self.assertInvalid("gen_day_empty_000.wav is not a RIFF WAVE file")

    def test_a_take_whose_data_outruns_its_header_is_invalid(self):
        path = self.dir / "base" / "gen_day_empty_000.wav"
        path.write_bytes(path.read_bytes() + b"\0" * 64)
        self.assertInvalid("gen_day_empty_000.wav's RIFF size isn't its length")

    def test_a_take_cut_short_of_its_header_is_invalid(self):
        path = self.dir / "cand" / "gen_day_empty_000.wav"
        path.write_bytes(path.read_bytes()[:-4])
        self.assertInvalid("gen_day_empty_000.wav's RIFF size isn't its length")

    def test_a_compressed_take_is_invalid(self):
        path = self.dir / "cand" / "gen_day_empty_000.wav"
        raw = bytearray(path.read_bytes())
        # the fmt chunk's format tag, after "RIFF", the size, "WAVE", "fmt " and its size
        struct.pack_into("<H", raw, 20, 3)
        path.write_bytes(raw)
        self.assertInvalid("is no PCM wav: format code 3")

    def test_a_second_data_chunk_is_invalid(self):
        take = riff(FMT, (b"data", 16, b"\0" * 16), (b"data", 8, b"\0" * 8))
        (self.dir / "cand" / "gen_day_empty_000.wav").write_bytes(take)
        self.assertInvalid("has more than one 'data' chunk")

    def test_a_second_fmt_chunk_is_invalid(self):
        take = riff(FMT, FMT, (b"data", 8, b"\0" * 8))
        (self.dir / "cand" / "gen_day_empty_000.wav").write_bytes(take)
        self.assertInvalid("has more than one 'fmt ' chunk")

    def test_a_fmt_chunk_after_the_data_is_invalid(self):
        take = riff((b"data", 8, b"\0" * 8), FMT)
        (self.dir / "cand" / "gen_day_empty_000.wav").write_bytes(take)
        self.assertInvalid("'fmt ' chunk follows its 'data'")

    def test_a_data_chunk_that_runs_past_the_file_is_invalid(self):
        take = riff(FMT, (b"data", 64, b"\0" * 32))
        (self.dir / "cand" / "gen_day_empty_000.wav").write_bytes(take)
        self.assertInvalid("'data' chunk runs past the file")

    def test_data_that_isnt_whole_frames_is_invalid(self):
        take = riff(FMT, (b"data", 6, b"\0" * 6))
        (self.dir / "cand" / "gen_day_empty_000.wav").write_bytes(take)
        self.assertInvalid("data isn't whole frames")

    def test_a_stub_after_the_last_chunk_is_invalid(self):
        take = riff(FMT, (b"data", 8, b"\0" * 8))
        take = take[:4] + struct.pack("<I", len(take) + 3 - 8) + take[8:] + b"LIS"
        (self.dir / "cand" / "gen_day_empty_000.wav").write_bytes(take)
        self.assertInvalid("ends inside a chunk header")


if __name__ == "__main__":
    unittest.main()
