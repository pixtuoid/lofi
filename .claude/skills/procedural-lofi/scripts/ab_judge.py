#!/usr/bin/env python3
"""Judges an A/B batch of a sound change by measurement. Pairs the takes a
baseline batch and a candidate batch share by name (rendered as SKILL.md's
GENERATOR loop step 5 says, which also sets up the venv this runs in),
scores both sides with two learned judges and their distance from real lofi,
describes each pair, and writes the verdict.

Usage: $J/bin/python ab_judge.py DIR --refs REFS
  DIR holds base/ and cand/; writes DIR/judge.json and prints a table.
  REFS is the reference corpus: audio/ and its manifest.json of sha256s.

The verdict is the controller's proposal (opinion): per mood and over every
take, an exact one-sided sign test, ties dropped, that cand scores LOWER than
base on each judge's headline score, or sits FARTHER from the corpus by
per-song FAD. Any test significant at ALPHA is REGRESSION; a group with too
few untied pairs for a test to ever reach ALPHA is INSUFFICIENT; anything else
is PASS. The batch takes its worst group's verdict, a detected regression
first, and exits non-zero unless it PASSes. The owner's blind listen, when
given, overrides this verdict. Every third-octave band whose difference
passes TO_EXPLAIN_DB is listed TO-EXPLAIN: a PR names its cause, but it fails
nothing.

A score is a guard against regression from a base batch to a candidate,
never an objective to maximize: lofi's character (hiss, crackle, roll-off) is
held by the descriptors and LOFI-BIBLE.md. Each measure's blind spots:

Audiobox Aesthetics (https://arxiv.org/abs/2502.05139):
- It hears 16 kHz mono ("we re-sample all input waveform to 16kHz and
  single-channel", §3; `sample_rate = 16000` at
  https://github.com/facebookresearch/audiobox-aesthetics/blob/2618e9d451b456e9328b39495b5e6234678aa550/src/audiobox_aesthetics/infer.py#L85),
  so nothing above 8 kHz and no stereo.
- "we did not use any synthetic data to train" it (§4.2.2), and on
  PAM-music its CE correlates r = 0.661 with human CE (Table 4).
- Its PQ axis scores "clarity & fidelity, dynamics, frequencies and
  spatialization" (§2.1), a polish lofi trades away on purpose (opinion).

MuQ-Eval (https://arxiv.org/abs/2603.22677):
- A v1 preprint with no journal reference (its abs page).
- Trained on MusicEval alone: "Cross-dataset generalization ... is
  untested" (§VI-F).
- "insensitive to musical-structural distortions (pitch shift ..., tempo
  stretch ...)" (§VI-F).
- Its concordance on MILD Gaussian noise is 0.39 and on mild MP3 0.52, at
  chance 0.50 (Table V, measured on its A3b variant, not the A1 scored
  here), so it can miss a subtle change; the descriptors and Audiobox cover
  that.

Per-song FAD over fadtk (https://github.com/microsoft/fadtk, MIT):
- FAD measures "distributional distance but weakly correlate[s] with
  aesthetic perception" (§1 of https://arxiv.org/abs/2601.07237).
- The corpus is 88 of 100 tracks from one netlabel, Dusted Wax Kingdom (its
  manifest), so it guards against drifting away from real lofi and is never
  a target to chase.
- FAD is biased by its sample size (§3.3 of
  https://arxiv.org/abs/2311.01616); both takes of a pair are one score's
  length, so the sign test compares equal frame counts.

The weights' licences: MuQ-310M CC-BY-NC 4.0
(https://github.com/tencent-ailab/MuQ/blob/28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a/README.md#license),
MuQ-Eval A1 MIT, Audiobox Aesthetics CC-BY 4.0 and LAION-CLAP CC0 1.0 (each
Hugging Face card at the revision pinned below).
"""

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from itertools import count
from pathlib import Path

import fadtk.model_loader
import numpy as np
import pyloudnorm
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from audiobox_aesthetics.infer import initialize_predictor
from fadtk.fad import calc_embd_statistics, calc_frechet_distance
from huggingface_hub import hf_hub_download, snapshot_download
from omegaconf import OmegaConf
from scipy import signal
from scipy.stats import binomtest

# no __pycache__ beside it, since the next batch renders from a clean tree
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_listen import NAME

# the clone of https://github.com/dgtql/MuQ-Eval that SKILL.md's setup puts
# in the venv: its model code is no package
MUQ_EVAL = Path(sys.prefix) / "MuQ-Eval"
sys.path.insert(0, str(MUQ_EVAL))
from src.model import MusicQualityModel

# each checkpoint at the revision this judge was ratified on
AUDIOBOX = ("facebook/audiobox-aesthetics", "9b1dd8e5df9af7216e836a98974fe3b82c56ded6")
MUQ_EVAL_A1 = ("zhudi2825/MuQ-Eval-A1", "b082e238827df51d99ac66bdd1b6f0d2406bc90e")
MUQ_310M = ("OpenMuQ/MuQ-large-msd-iter", "0562a57814f6f8bbd9fdea0a25921a2fce1a841a")
LAION_CLAP = (
    "lukewys/laion_clap",
    "b3708341862f581175dba5c356a4ebf74a9b6651",
    "music_audioset_epoch_15_esc_90.14.pt",
)

# fadtk's LAION-CLAP music embedding ("L-CLAP mus"): "CLAP and L-CLAP
# outperform other embeddings for AQ and MQ MOS prediction" (§4.4 of
# https://arxiv.org/abs/2311.01616)
FAD_EMBEDDING = "music"
# reference tracks left out of the statistics, so their own per-song FAD
# shows what real lofi scores (the controller's proposal)
HOLDOUT = 12
# fadtk's own resampler to a model's rate, `load_audio` in its fad.py
RESAMPLE = {
    "lowpass_filter_width": 64,
    "rolloff": 0.9475937167399596,
    "resampling_method": "sinc_interp_kaiser",
    "beta": 14.769656459379492,
}

# the floor every take gets before anything scores it, in dBFS RMS: over both
# engines' numerical floors, which a learned judge scores and a listener
# can't hear (inaudible is opinion)
NOISE_FLOOR_DBFS = -70.0

# Audiobox's own window, `make_inference_batch(wavs, 10, 10, ...)` in its
# infer.py: both judges score a take as consecutive windows this long
WINDOW_S = 10
# windows per forward pass, which bounds its device memory (opinion)
MUQ_BATCH = 6

# the verdict's significance and the TO-EXPLAIN threshold (opinion)
ALPHA = 0.05
TO_EXPLAIN_DB = 3.0
# the fewest untied pairs a sign test can reject at ALPHA with: every one a
# loss, its most extreme outcome
MIN_UNTIED = next(n for n in count(1) if binomtest(n, n, 0.5, alternative="greater").pvalue < ALPHA)

# third octaves on base-2 centres, 1000·2^(k/3) Hz: 20 Hz to 16 kHz
BANDS_HZ = [1000 * 2 ** (k / 3) for k in range(-17, 13)]
# so the 20 Hz band, under 5 Hz wide, holds several bins
WELCH_NPERSEG = 1 << 15
POWER_EPS = 1e-30

# the hat band's flux, frame to frame (the controller's analysis, opinion)
HAT_HZ = (6000, 12000)
FLUX_NFFT = 4096
FLUX_HOP = 1024

# a click: the 4 kHz high band standing CLICK_MADS median absolute
# deviations out of its own floor, events closer than CLICK_GAP_S one click
# (the controller's analysis, opinion)
CLICK_HZ = 4000
CLICK_ORDER = 4
CLICK_MADS = 12
CLICK_GAP_S = 0.005


def channels(x):
    """`(left, right)` of a frames × channels take; a mono take is both."""
    return (x, x) if x.ndim == 1 else (x[:, 0], x[:, 1])


def mono(x):
    left, right = channels(x)
    return (left + right) / 2


def floored(x, side, name):
    """`x` over `NOISE_FLOOR_DBFS` of white noise, the same in each channel
    and its own for each side and take, seeded by their names."""
    seed = int.from_bytes(hashlib.sha256(f"{side}/{name}".encode()).digest()[:8], "little")
    noise = np.random.default_rng(seed).standard_normal(len(x)).astype(np.float32)
    noise *= np.float32(10 ** (NOISE_FLOOR_DBFS / 20))
    return x + (noise if x.ndim == 1 else noise[:, None])


def mono_lufs(x, sr):
    """The integrated loudness of the mono the embeddings hear."""
    return pyloudnorm.Meter(sr).integrated_loudness(mono(x).astype(np.float64))


def describe(x, sr):
    left, right = channels(x)
    mid = (left + right) / 2
    f, psd = signal.welch(mid, sr, nperseg=WELCH_NPERSEG)
    df = f[1] - f[0]
    bands = [
        10 * np.log10(psd[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))].sum() * df + POWER_EPS)
        for c in BANDS_HZ
    ]
    rms = np.sqrt(np.mean(mid**2))
    freqs, _, z = signal.stft(mid, sr, nperseg=FLUX_NFFT, noverlap=FLUX_NFFT - FLUX_HOP, boundary=None)
    hat = np.abs(z[(freqs >= HAT_HZ[0]) & (freqs < HAT_HZ[1])])
    flux = np.diff(hat, axis=1).clip(min=0).sum(axis=0)
    return {
        "bands_db": bands,
        "centroid_hz": float((f * psd).sum() / psd.sum()),
        "crest_db": float(20 * np.log10(np.abs(mid).max() / rms)),
        "lr_corr": float(np.corrcoef(left, right)[0, 1]),
        "clicks_per_min": clicks_per_min(mid, sr),
        "hat_flux_p95_over_median": float(np.percentile(flux, 95) / np.median(flux)),
    }


def clicks_per_min(mid, sr):
    sos = signal.butter(CLICK_ORDER, CLICK_HZ, "highpass", fs=sr, output="sos")
    high = signal.sosfilt(sos, mid)
    mad = np.median(np.abs(high - np.median(high)))
    over = np.flatnonzero(np.abs(high) > CLICK_MADS * mad)
    starts = over[np.insert(np.diff(over) > CLICK_GAP_S * sr, 0, True)] if over.size else over
    return starts.size / (mid.size / sr / 60)


class MuqEval:
    """MuQ-Eval's A1 checkpoint, its MI head on a 1–5 scale."""

    def __init__(self, device):
        repo, rev = MUQ_EVAL_A1
        config = hf_hub_download(repo, "config.yaml", revision=rev)
        base = hf_hub_download(repo, "base.yaml", revision=rev)
        # merged as MuQ-Eval's own `train.py:load_config` merges an
        # experiment over the config its `defaults` names
        cfg = OmegaConf.merge(OmegaConf.load(base), OmegaConf.load(config))
        del cfg["defaults"]
        if cfg.model.encoder_id != MUQ_310M[0]:
            sys.exit(f"MuQ-Eval's A1 config names encoder {cfg.model.encoder_id}, not {MUQ_310M[0]}")
        # a local snapshot, so the encoder loads at its pinned revision
        cfg.model.encoder_id = snapshot_download(
            MUQ_310M[0], revision=MUQ_310M[1], allow_patterns=["config.json", "model.safetensors"]
        )
        self.model = MusicQualityModel(cfg)
        weights = hf_hub_download(repo, "model_state_dict.pt", revision=rev)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(device).eval()
        self.sr = cfg.data.sample_rate
        if cfg.data.clip_samples != WINDOW_S * self.sr:
            sys.exit(f"MuQ-Eval's clip is {cfg.data.clip_samples} samples, not {WINDOW_S} s")
        self.device = device

    def score(self, x, sr):
        """The take's windows' MI, averaged as Audiobox averages its own:
        the last window zero-padded and weighted by the share it fills."""
        heard = torch.from_numpy(np.ascontiguousarray(mono(x), dtype=np.float32))
        # as MuQ-Eval's `AudioProcessor.process` resamples
        wav = torchaudio.functional.resample(heard, sr, self.sr)
        clip = WINDOW_S * self.sr
        windows = [wav[i : i + clip] for i in range(0, wav.shape[-1], clip)]
        weights = torch.tensor([w.shape[-1] / clip for w in windows])
        padded = torch.stack([F.pad(w, (0, clip - w.shape[-1])) for w in windows])
        with torch.inference_mode():
            mi = torch.cat(
                [
                    self.model(padded[i : i + MUQ_BATCH].to(self.device))["MI"].float().cpu()
                    for i in range(0, len(windows), MUQ_BATCH)
                ]
            )
        return float((mi * weights).sum() / weights.sum())


class Fad:
    """Per-song FAD from a take to the reference corpus, over fadtk's
    LAION-CLAP embedding: the Fréchet distance between the corpus's frame
    statistics and the take's, as fadtk's `score_individual` takes it."""

    def __init__(self, refs, takes_lufs):
        manifest = (refs / "manifest.json").read_bytes()
        tracks = json.loads(manifest)["tracks"]
        for t in tracks:
            digest = hashlib.sha256((refs / "audio" / t["file"]).read_bytes()).hexdigest()
            if digest != t["sha256"]:
                sys.exit(f"{t['file']} doesn't match its manifest sha256; refusing the corpus")
        # fadtk loads whatever sits at its checkpoint path and downloads a
        # missing one from the repo's head
        # (https://github.com/microsoft/fadtk/blob/6f815282007288a7165d037a95dacd6d783b6f32/fadtk/model_loader.py#L303-L310),
        # so only a link to the pinned one may sit there
        repo, rev, file = LAION_CLAP
        pinned = Path(hf_hub_download(repo, file, revision=rev))
        seeded = Path(fadtk.model_loader.__file__).parent / ".model-checkpoints" / file
        if not (seeded.is_symlink() and seeded.resolve() == pinned.resolve()):
            seeded.unlink(missing_ok=True)
            seeded.parent.mkdir(parents=True, exist_ok=True)
            seeded.symlink_to(pinned)
        self.model = fadtk.model_loader.CLAPLaionModel(FAD_EMBEDDING)
        self.model.load_model()
        manifest_sha = hashlib.sha256(manifest).hexdigest()
        caches = Path(sys.prefix) / "fad-cache"
        caches.mkdir(parents=True, exist_ok=True)
        # the corpus alone sets it, and measuring it decodes every track
        quietest_file = caches / f"{manifest_sha}-quietest.json"
        if quietest_file.exists():
            quietest = json.loads(quietest_file.read_text())["lufs"]
        else:
            quietest = min(mono_lufs(*self.read(refs, t)) for t in tracks)
            quietest_file.write_text(json.dumps({"lufs": quietest}) + "\n")
        # one level for every input, the takes' unless a reference is
        # quieter, so no reference is ever raised and clipped
        self.lufs = min(takes_lufs, quietest)
        # keyed by the level it holds, which any batch louder than the
        # quietest reference shares
        cache = caches / f"{manifest_sha}-{self.model.name}-{self.lufs:.2f}.npz"
        if cache.exists():
            with np.load(cache) as stats:
                self.mu, self.cov = stats["mu"], stats["cov"]
                self.calibration = dict(zip(stats["held"].tolist(), stats["held_fad"].tolist()))
            return
        # the lowest sha256s, a choice no file order moves
        held = {t["file"] for t in sorted(tracks, key=lambda t: t["sha256"])[:HOLDOUT]}
        frames = np.concatenate([self.embed(*self.read(refs, t)) for t in tracks if t["file"] not in held])
        self.mu, self.cov = calc_embd_statistics(frames)
        self.calibration = {f: self.score(*self.read(refs, {"file": f})) for f in sorted(held)}
        np.savez(
            cache,
            lufs=self.lufs,
            mu=self.mu,
            cov=self.cov,
            held=np.array(list(self.calibration)),
            held_fad=np.array(list(self.calibration.values())),
        )

    @staticmethod
    def read(refs, track):
        return sf.read(refs / "audio" / track["file"], dtype="float32")

    def embed(self, x, sr):
        """`x`'s frames of the embedding, at the one level every input shares."""
        heard = mono(x).astype(np.float64)
        heard *= 10 ** ((self.lufs - mono_lufs(x, sr)) / 20)
        wav = torch.from_numpy(heard.astype(np.float32))[None]
        wav = torchaudio.transforms.Resample(sr, self.model.sr, **RESAMPLE)(wav)[0]
        return self.model.get_embedding(wav.numpy())

    def score(self, x, sr):
        mu, cov = calc_embd_statistics(self.embed(x, sr))
        return float(calc_frechet_distance(self.mu, self.cov, mu, cov))


def sign_test(gains):
    """Whether cand does worse than base more often than chance allows,
    `gains` each pair's improvement from base to cand."""
    wins = sum(g > 0 for g in gains)
    losses = sum(g < 0 for g in gains)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html:
    # losses as successes out of the untied pairs, H1 a success rate over 1/2
    p = binomtest(losses, wins + losses, 0.5, alternative="greater").pvalue if wins + losses else 1.0
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(gains) - wins - losses,
        "p": float(p),
        "significant": bool(p < ALPHA),
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


def to_explain(pairs):
    """Every band some pair moves past TO_EXPLAIN_DB, with how many pairs do
    and the band's median and range over all of them."""
    deltas = np.array([p["bands_delta_db"] for p in pairs])
    rows = []
    for i, hz in enumerate(BANDS_HZ):
        over = int((np.abs(deltas[:, i]) > TO_EXPLAIN_DB).sum())
        if over:
            col = deltas[:, i]
            rows.append(
                {
                    "band_hz": round(hz),
                    "pairs_over": over,
                    "median_db": float(np.median(col)),
                    "min_db": float(col.min()),
                    "max_db": float(col.max()),
                }
            )
    return rows


def read_commit(side):
    commit = side / "COMMIT"
    return commit.read_text().strip() if commit.exists() else None


def main():
    ap = argparse.ArgumentParser(description="Judge an A/B batch by measurement.")
    ap.add_argument("dir", type=Path, help="the batch: base/ and cand/ of same-named takes")
    ap.add_argument("--refs", type=Path, required=True, help="the reference corpus: audio/ and manifest.json")
    args = ap.parse_args()
    base_dir, cand_dir = args.dir / "base", args.dir / "cand"

    base = {p.name for p in base_dir.glob("*.wav")}
    cand = {p.name for p in cand_dir.glob("*.wav")}
    if base ^ cand:
        print(f"unpaired, left out: {', '.join(sorted(base ^ cand))}")
    # a --shuffle run draws its own take_XX order, so equal names there are
    # different takes
    opaque = sorted(n for n in base & cand if not NAME.fullmatch(n))
    if opaque:
        sys.exit(f"not unshuffled lofi_audition names, so a pair may hold two takes: {', '.join(opaque)}")
    names = sorted(base & cand)
    if not names:
        sys.exit("no .wav name is in both directories")

    # the device Audiobox's own `setup_model` picks
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    audiobox = initialize_predictor(hf_hub_download(AUDIOBOX[0], "checkpoint.pt", revision=AUDIOBOX[1]))
    muq = MuqEval(device)
    sides = (("base", base_dir), ("cand", cand_dir))
    takes_lufs = float(np.median([mono_lufs(*sf.read(d / n, dtype="float32")) for n in names for _, d in sides]))
    fad = Fad(args.refs, takes_lufs)

    pairs = []
    for name in names:
        row = {"take": name.removesuffix(".wav"), "mood": NAME.fullmatch(name)["mood"]}
        takes = {}
        for side, d in sides:
            x, sr = sf.read(d / name, dtype="float32")
            # the descriptors measure the take as rendered, the judges hear it floored
            row[side] = describe(x, sr)
            x = floored(x, side, name)
            takes[side] = (x, sr)
            row[side]["muq_mi"] = muq.score(x, sr)
            row[side]["fad"] = fad.score(x, sr)
        # Audiobox reads a path with `torchaudio.load`
        # (https://github.com/facebookresearch/audiobox-aesthetics/blob/2618e9d451b456e9328b39495b5e6234678aa550/src/audiobox_aesthetics/infer.py#L32-L45),
        # which "relies on TorchCodec" since TorchAudio 2.9
        # (https://docs.pytorch.org/audio/2.11.0/generated/torchaudio.load.html),
        # so it gets tensors
        scores = audiobox.forward(
            [{"path": torch.from_numpy(np.atleast_2d(x.T).copy()), "sample_rate": sr} for x, sr in takes.values()]
        )
        for side, s in zip(("base", "cand"), scores):
            row[side]["audiobox"] = s
        row["bands_delta_db"] = [c - b for b, c in zip(row["base"]["bands_db"], row["cand"]["bands_db"])]
        pairs.append(row)
        print(f"scored {row['take']}", flush=True)

    groups = {mood: [p for p in pairs if p["mood"] == mood] for mood in sorted({p["mood"] for p in pairs})}
    groups["all"] = pairs
    verdict = {}
    for group, members in groups.items():
        ce = [p["cand"]["audiobox"]["CE"] - p["base"]["audiobox"]["CE"] for p in members]
        mi = [p["cand"]["muq_mi"] - p["base"]["muq_mi"] for p in members]
        fd = [p["cand"]["fad"] - p["base"]["fad"] for p in members]
        tests = {
            "audiobox_ce": {**sign_test(ce), "mean_delta": float(np.mean(ce))},
            "muq_eval_mi": {**sign_test(mi), "mean_delta": float(np.mean(mi))},
            # nearer the corpus is better, so a gain is a fall in FAD
            "fad": {**sign_test([-d for d in fd]), "mean_delta": float(np.mean(fd))},
        }
        verdict[group] = {
            "pairs": len(members),
            **tests,
            "audiobox_mean_delta_pq": float(
                np.mean([p["cand"]["audiobox"]["PQ"] - p["base"]["audiobox"]["PQ"] for p in members])
            ),
            "verdict": group_verdict(tests),
            "to_explain": to_explain(members),
        }
    overall = batch_verdict([v["verdict"] for v in verdict.values()])

    record = {
        "dir": str(args.dir.resolve()),
        "commits": {"base": read_commit(base_dir), "cand": read_commit(cand_dir)},
        "judges": {
            "audiobox": {"package": version("audiobox_aesthetics"), "checkpoint": AUDIOBOX},
            "muq_eval": {"code": str(MUQ_EVAL), "checkpoint": MUQ_EVAL_A1, "encoder": MUQ_310M},
            "fad": {
                "package": version("fadtk"),
                "embedding": fad.model.name,
                "checkpoint": LAION_CLAP,
                "refs": str(args.refs.resolve()),
                "lufs": fad.lufs,
            },
            "torch": version("torch"),
            "device": str(device),
        },
        "noise_floor_dbfs": NOISE_FLOOR_DBFS,
        "fad_calibration": fad.calibration,
        "alpha": ALPHA,
        "min_untied": MIN_UNTIED,
        "bands_hz": BANDS_HZ,
        "pairs": pairs,
        "verdict": verdict,
        "overall": overall,
    }
    (args.dir / "judge.json").write_text(json.dumps(record, indent=1) + "\n")

    print(f"\n{'take':<24} {'dCE':>6} {'dPQ':>6} {'dMI':>6} {'dFAD':>6} {'max|dB|':>8} {'@Hz':>6}"
          f" {'centroid b/c':>13} {'crest b/c':>11} {'LR b/c':>11} {'clk/min b/c':>11} {'hat b/c':>11}")
    for p in pairs:
        b, c = p["base"], p["cand"]
        worst = int(np.argmax(np.abs(p["bands_delta_db"])))
        print(
            f"{p['take'].removeprefix('gen_'):<24}"
            f" {c['audiobox']['CE'] - b['audiobox']['CE']:+6.2f}"
            f" {c['audiobox']['PQ'] - b['audiobox']['PQ']:+6.2f}"
            f" {c['muq_mi'] - b['muq_mi']:+6.2f}"
            f" {c['fad'] - b['fad']:+6.2f}"
            f" {p['bands_delta_db'][worst]:+8.1f} {round(BANDS_HZ[worst]):>6}"
            f" {b['centroid_hz']:6.0f}/{c['centroid_hz']:<6.0f}"
            f" {b['crest_db']:5.1f}/{c['crest_db']:<5.1f}"
            f" {b['lr_corr']:5.2f}/{c['lr_corr']:<5.2f}"
            f" {b['clicks_per_min']:5.0f}/{c['clicks_per_min']:<5.0f}"
            f" {b['hat_flux_p95_over_median']:5.2f}/{c['hat_flux_p95_over_median']:<5.2f}"
        )
    print(f"\n{'group':<6} {'pairs':>5}  {'Audiobox CE w/l/t':>17} {'p':>7} {'mean dCE':>8} {'mean dPQ':>8}"
          f"  {'MuQ-Eval MI w/l/t':>17} {'p':>7} {'mean dMI':>8}  {'FAD w/l/t':>9} {'p':>7} {'mean dFAD':>9}  verdict")
    for group, v in verdict.items():
        ce, mi, fd = v["audiobox_ce"], v["muq_eval_mi"], v["fad"]
        print(
            f"{group:<6} {v['pairs']:>5}  {ce['wins']:>7}/{ce['losses']}/{ce['ties']:<5} {ce['p']:7.4f}"
            f" {ce['mean_delta']:+8.3f} {v['audiobox_mean_delta_pq']:+8.3f}"
            f"  {mi['wins']:>7}/{mi['losses']}/{mi['ties']:<5} {mi['p']:7.4f} {mi['mean_delta']:+8.3f}"
            f"  {fd['wins']:>3}/{fd['losses']}/{fd['ties']:<3} {fd['p']:7.4f} {fd['mean_delta']:+9.3f}  {v['verdict']}"
        )
    held = np.array(list(fad.calibration.values()))
    print(
        f"\nFAD calibration, {len(held)} held-out references: median {np.median(held):.3f}"
        f" ({held.min():.3f}..{held.max():.3f}); takes: base median"
        f" {np.median([p['base']['fad'] for p in pairs]):.3f}, cand median {np.median([p['cand']['fad'] for p in pairs]):.3f}"
    )
    for group, v in verdict.items():
        for t in v["to_explain"]:
            print(
                f"TO-EXPLAIN {group}: {t['band_hz']} Hz, {t['pairs_over']} of {v['pairs']} pairs past"
                f" ±{TO_EXPLAIN_DB:g} dB, median {t['median_db']:+.1f} ({t['min_db']:+.1f}..{t['max_db']:+.1f})"
            )
    print(f"\nverdict: {overall}  ({args.dir / 'judge.json'})")
    if overall == "REGRESSION":
        sys.exit("cand scores worse than base")
    if overall == "INSUFFICIENT":
        sys.exit(f"a group has fewer than {MIN_UNTIED} untied pairs in some test, too few to ever reject at {ALPHA}")


if __name__ == "__main__":
    main()
