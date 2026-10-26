#!/usr/bin/env python3
"""Judges an A/B batch of a sound change by measurement. Pairs the takes a
baseline batch and a candidate batch share by name (rendered as SKILL.md's
GENERATOR loop step 5 says, which also sets up the venv this runs in),
scores both sides with two learned judges and their distance from real lofi,
describes each pair, and writes the verdict.

Usage: $J/bin/python ab_judge.py DIR --refs REFS [--fad-cache CACHE] [--score-identical]
  DIR holds base/ and cand/, each with the COMMIT it rendered at; writes
  DIR/judge.json and prints its table.
  REFS is the reference corpus: audio/ and its manifest.json of sha256s.
  CACHE holds the corpus's FAD statistics, built on a miss.
  --score-identical scores byte-identical pairs too: the proof that they tie.

The verdict is the controller's proposal (opinion), `verdict.judged`'s, and
this exits non-zero on REGRESSION or INSUFFICIENT. The owner's blind listen,
when given, overrides it. Every third-octave band whose difference passes
TO_EXPLAIN_DB is listed TO-EXPLAIN: a PR names its cause, but it fails
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
  https://arxiv.org/abs/2311.01616), so both takes of a pair are scored on
  equally many frames.

The weights' licences: MuQ-310M CC-BY-NC 4.0
(https://github.com/tencent-ailab/MuQ/blob/28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a/README.md#license),
MuQ-Eval A1 MIT, Audiobox Aesthetics CC-BY 4.0 and LAION-CLAP CC0 1.0 (each
Hugging Face card at the revision pinned below).
"""

import argparse
import hashlib
import json
import sys
import tempfile
from importlib.metadata import version
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

# no __pycache__ beside it, since the next batch renders from a clean tree
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
# `judge/` sits beside `scripts/` here and in the Modal image alike
sys.path[:0] = [str(HERE), str(HERE.parent / "judge")]
from validate_batch import NAME, Invalid, validate
from verdict import JUDGES, exit_message, judged

# the clone of https://github.com/dgtql/MuQ-Eval that SKILL.md's setup and
# the Modal image put in the venv, at the commit this judge was ratified on:
# its model code is no package
MUQ_EVAL = Path(sys.prefix) / "MuQ-Eval"
MUQ_EVAL_COMMIT = "60a88f8ac0909ca1fd1a78af3660f1fc376977a1"
# a detached checkout's HEAD holds its commit: "HEAD can also record a
# specific commit directly" (https://git-scm.com/docs/gitrepository-layout)
head = MUQ_EVAL / ".git" / "HEAD"
if not head.is_file() or head.read_text().strip() != MUQ_EVAL_COMMIT:
    sys.exit(f"{MUQ_EVAL} isn't a MuQ-Eval checkout at {MUQ_EVAL_COMMIT}")
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

# the floor every take gets before anything scores it: white noise over both
# engines' numerical floors, which a learned judge scores and a listener
# can't hear (inaudible is opinion). It was calibrated at this RMS in dBFS
# under takes at this integrated loudness, lofi_audition's `TARGET_LUFS` then,
# and keeps that distance under each take's own loudness: the engines'
# numerical floors scale with the music, and a pure gain change must move a
# take and its floor together, or FAD's loudness normalization hears the
# floor move
FLOOR_CALIBRATED_DBFS = -70.0
FLOOR_CALIBRATED_LUFS = -18.0
FLOOR_RE_TAKE_DB = FLOOR_CALIBRATED_DBFS - FLOOR_CALIBRATED_LUFS

# Audiobox's own window, `make_inference_batch(wavs, 10, 10, ...)` in its
# infer.py: both judges score a take as consecutive windows this long
WINDOW_S = 10
# windows per forward pass, which bounds its device memory (opinion)
MUQ_BATCH = 6

# the TO-EXPLAIN threshold (opinion)
TO_EXPLAIN_DB = 3.0

# third octaves on base-2 centres, 1000·2^(k/3) Hz: 20 Hz to 16 kHz
BANDS_HZ = [1000 * 2 ** (k / 3) for k in range(-17, 13)]
# so the 20 Hz band, under 5 Hz wide, holds several bins
WELCH_NPERSEG = 1 << 15
POWER_EPS = 1e-30

# the hat band's flux, frame to frame (the controller's analysis, opinion)
HAT_HZ = (6000, 12000)
FLUX_NFFT = 4096
FLUX_HOP = 1024
# the flux's peak, as this percentile of its frames over their median
HAT_FLUX_PERCENTILE = 95
HAT_FLUX_KEY = f"hat_flux_p{HAT_FLUX_PERCENTILE}_over_median"

# a click: the CLICK_HZ high band standing CLICK_MADS median absolute
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


def floored(x, name, lufs):
    """`x`, whose integrated loudness is `lufs`, over white noise
    `FLOOR_RE_TAKE_DB` under it, the same in each channel and seeded by the
    take's name: both sides of a pair share the draw, so two identical takes
    still tie and the floor adds no variance to the pair's difference."""
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    noise = np.random.default_rng(seed).standard_normal(len(x)).astype(np.float32)
    noise *= np.float32(10 ** ((lufs + FLOOR_RE_TAKE_DB) / 20))
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
        HAT_FLUX_KEY: float(np.percentile(flux, HAT_FLUX_PERCENTILE) / np.median(flux)),
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


def quietest_file(caches, manifest_sha):
    # the corpus alone sets it, and measuring it decodes every track
    return caches / f"{manifest_sha}-quietest.json"


def stats_file(caches, manifest_sha, lufs):
    # keyed by everything the statistics are built from; the level is one any
    # batch louder than the quietest reference shares
    built = json.dumps([LAION_CLAP, FAD_EMBEDDING, HOLDOUT, RESAMPLE])
    return caches / f"{manifest_sha}-{hashlib.sha256(built.encode()).hexdigest()[:16]}-{lufs:.2f}.npz"


def cached_quietest(caches, manifest_sha, takes_lufs):
    """The quietest reference's loudness if `caches` holds it, else None.
    Exits first when `caches` lacks a file this run builds and can't take
    it, as the remote judge's read-only cache can't: with the corpus's own
    statistics missing, provisioning builds them; a batch quieter than the
    quietest reference needs its own level, which only a local run builds.
    Tried rather than asked, since Modal promises a read-only mount only as
    "read only from within containers" (modal 1.6.1 `volume.py`:598)."""
    cached = quietest_file(caches, manifest_sha)
    quietest = json.loads(cached.read_text())["lufs"] if cached.exists() else None
    if quietest is not None and stats_file(caches, manifest_sha, min(takes_lufs, quietest)).exists():
        return quietest
    try:
        caches.mkdir(parents=True, exist_ok=True)
        tempfile.TemporaryFile(dir=caches).close()
    except OSError:
        if quietest is not None and takes_lufs < quietest:
            sys.exit(
                "no FAD statistics at this batch's level, and the cache is read-only: the batch"
                " is quieter than the corpus's quietest reference, so judge it locally"
            )
        sys.exit("the FAD cache lacks this corpus's statistics and is read-only: run provision")
    return quietest


class Fad:
    """Per-song FAD from a take to the reference corpus, over fadtk's
    LAION-CLAP embedding: the Fréchet distance between the corpus's frame
    statistics and the take's, as fadtk's `score_individual` takes it."""

    def __init__(self, refs, takes_lufs, caches):
        manifest = (refs / "manifest.json").read_bytes()
        manifest_sha = hashlib.sha256(manifest).hexdigest()
        quietest = cached_quietest(caches, manifest_sha, takes_lufs)
        tracks = json.loads(manifest)["tracks"]
        # by index: the corpus's names are private, and the CI's log is not
        for i, t in enumerate(tracks):
            path = refs / "audio" / t["file"]
            if not path.is_file():
                sys.exit(f"reference {i} of the manifest is missing; refusing the corpus")
            if hashlib.sha256(path.read_bytes()).hexdigest() != t["sha256"]:
                sys.exit(f"reference {i} doesn't match its manifest sha256; refusing the corpus")
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
        if quietest is None:
            quietest = min(mono_lufs(*self.read(refs, t)) for t in tracks)
            quietest_file(caches, manifest_sha).write_text(json.dumps({"lufs": quietest}) + "\n")
        # one level for every input, the takes' unless a reference is
        # quieter, so no reference is ever raised and clipped
        self.lufs = min(takes_lufs, quietest)
        cache = stats_file(caches, manifest_sha, self.lufs)
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

    def distance(self, frames):
        mu, cov = calc_embd_statistics(frames)
        return float(calc_frechet_distance(self.mu, self.cov, mu, cov))

    def score(self, x, sr):
        return self.distance(self.embed(x, sr))

    def score_pair(self, base, cand):
        """Both takes' FAD on equally many embedding frames, the longer's
        thinned evenly across its length."""
        frames = [self.embed(*take) for take in (base, cand)]
        n = min(len(f) for f in frames)
        return [self.distance(f[np.linspace(0, len(f) - 1, n).round().astype(int)]) for f in frames]


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
    """The commit `side` rendered at, as SKILL.md's step 5 records it."""
    commit = side / "COMMIT"
    if not commit.exists():
        sys.exit(f"{commit} is missing, and a verdict names the commits it compared")
    return commit.read_text().strip()


def main():
    ap = argparse.ArgumentParser(description="Judge an A/B batch by measurement.")
    ap.add_argument("dir", type=Path, help="the batch: base/ and cand/ of same-named takes")
    ap.add_argument("--refs", type=Path, required=True, help="the reference corpus: audio/ and manifest.json")
    ap.add_argument(
        "--fad-cache",
        type=Path,
        default=Path(sys.prefix) / "fad-cache",
        help="the corpus's FAD statistics, built here on a miss (default: the venv's)",
    )
    ap.add_argument("--score-identical", action="store_true", help="score byte-identical pairs too")
    args = ap.parse_args()
    base_dir, cand_dir = args.dir / "base", args.dir / "cand"

    try:
        takes = validate(args.dir)
    except Invalid as e:
        sys.exit(f"invalid batch: {e}")
    if takes.new or takes.gone:
        print(f"unpaired, left out: {', '.join(takes.new + takes.gone)}")
    commits = {"base": read_commit(base_dir), "cand": read_commit(cand_dir)}
    sides = (("base", base_dir), ("cand", cand_dir))
    # a pair the change left byte for byte as it was can only tie, so no judge
    # hears it; --score-identical scores every pair the batch still holds
    changed = [n for n in takes.paired if (base_dir / n).exists()] if args.score_identical else takes.differ
    pairs = [
        {"take": n.removesuffix(".wav"), "mood": NAME.fullmatch(n)["mood"], "identical": n not in changed}
        for n in takes.paired
    ]
    record = {"dir": str(args.dir.resolve()), "commits": commits, "floor_re_take_db": FLOOR_RE_TAKE_DB}

    if changed:
        # the device Audiobox's own `setup_model` picks
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if device.type == "cuda":
            # cuDNN "can be a source of nondeterminism", and the algorithm it
            # picks "may be nondeterministic, unless ... cudnn.deterministic =
            # True" (https://docs.pytorch.org/docs/2.14/notes/randomness.html),
            # which would let two identical takes' scores differ
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        audiobox = initialize_predictor(hf_hub_download(AUDIOBOX[0], "checkpoint.pt", revision=AUDIOBOX[1]))
        muq = MuqEval(device)
        takes_lufs = float(
            np.median([mono_lufs(*sf.read(d / n, dtype="float32")) for n in changed for _, d in sides])
        )
        fad = Fad(args.refs, takes_lufs, args.fad_cache)
        record["judges"] = {
            "audiobox": {"package": version("audiobox_aesthetics"), "checkpoint": AUDIOBOX},
            "muq_eval": {"code": MUQ_EVAL_COMMIT, "checkpoint": MUQ_EVAL_A1, "encoder": MUQ_310M},
            "fad": {
                "package": version("fadtk"),
                "embedding": fad.model.name,
                "checkpoint": LAION_CLAP,
                "refs": str(args.refs.resolve()),
                "lufs": fad.lufs,
            },
            "torch": version("torch"),
            "device": str(device),
        }
        record["fad_calibration"] = fad.calibration
        for row in pairs:
            if row["identical"]:
                continue
            name = f"{row['take']}.wav"
            takes = {}
            for side, d in sides:
                x, sr = sf.read(d / name, dtype="float32")
                # the descriptors measure the take as rendered, the judges hear it floored
                row[side] = describe(x, sr)
                x = floored(x, name, mono_lufs(x, sr))
                takes[side] = (x, sr)
                row[side]["muq_mi"] = muq.score(x, sr)
            row["base"]["fad"], row["cand"]["fad"] = fad.score_pair(takes["base"], takes["cand"])
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
            b, c = row["base"], row["cand"]
            row["deltas"] = dict(
                zip(JUDGES, (c["audiobox"]["CE"] - b["audiobox"]["CE"], c["muq_mi"] - b["muq_mi"], c["fad"] - b["fad"]))
            )
            # nearer the corpus is better, so a gain is a fall in FAD
            row["gains"] = {j: -d if j == "fad" else d for j, d in row["deltas"].items()}
            print(f"scored {row['take']}", flush=True)

    result = judged(pairs)
    verdict = {}
    for group, g in result["groups"].items():
        members = g["changed"]
        v = {"pairs": len(members), "identical": g["identical"], "gates": group != "all", "verdict": g["verdict"]}
        if members:
            for j, test in g["tests"].items():
                v[j] = {**test, "mean_delta": float(np.mean([p["deltas"][j] for p in members]))}
            v["audiobox_mean_delta_pq"] = float(
                np.mean([p["cand"]["audiobox"]["PQ"] - p["base"]["audiobox"]["PQ"] for p in members])
            )
            v["to_explain"] = to_explain(members)
        verdict[group] = v
    record.update(
        alpha=result["alpha"],
        min_untied=result["min_untied"],
        bands_hz=BANDS_HZ,
        pairs=pairs,
        verdict=verdict,
        split=result["split"],
        overall=result["overall"],
    )
    # built once, so judge.json's table is the one printed
    table = []
    table.append(f"\n{'take':<24} {'dCE':>6} {'dPQ':>6} {'dMI':>6} {'dFAD':>6} {'max|dB|':>8} {'@Hz':>6}"
          f" {'centroid b/c':>13} {'crest b/c':>11} {'LR b/c':>11} {'clk/min b/c':>11} {'hat b/c':>11}")
    for p in pairs:
        if p["identical"]:
            table.append(f"{p['take'].removeprefix('gen_'):<24} identical")
            continue
        b, c = p["base"], p["cand"]
        worst = int(np.argmax(np.abs(p["bands_delta_db"])))
        table.append(
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
            f" {b[HAT_FLUX_KEY]:5.2f}/{c[HAT_FLUX_KEY]:<5.2f}"
        )
    table.append(f"\n{'group':<6} {'pairs':>5} {'same':>4}  {'Audiobox CE w/l/t':>17} {'p':>7} {'mean dCE':>8} {'mean dPQ':>8}"
          f"  {'MuQ-Eval MI w/l/t':>17} {'p':>7} {'mean dMI':>8}  {'FAD w/l/t':>9} {'p':>7} {'mean dFAD':>9}  verdict")
    for group, v in verdict.items():
        label = v["verdict"] if v["gates"] else f"({v['verdict']}, not gating)"
        if not v["pairs"]:
            table.append(f"{group:<6} {0:>5} {v['identical']:>4}  {label}")
            continue
        ce, mi, fd = v["audiobox_ce"], v["muq_eval_mi"], v["fad"]
        table.append(
            f"{group:<6} {v['pairs']:>5} {v['identical']:>4}  {ce['wins']:>7}/{ce['losses']}/{ce['ties']:<5} {ce['p']:7.4f}"
            f" {ce['mean_delta']:+8.3f} {v['audiobox_mean_delta_pq']:+8.3f}"
            f"  {mi['wins']:>7}/{mi['losses']}/{mi['ties']:<5} {mi['p']:7.4f} {mi['mean_delta']:+8.3f}"
            f"  {fd['wins']:>3}/{fd['losses']}/{fd['ties']:<3} {fd['p']:7.4f} {fd['mean_delta']:+9.3f}  {label}"
        )
    if result["split"]:
        table.append(
            f"SPLIT {', '.join(result['split'])}: one judge significantly favours cand and another base,"
            " so no vote settles it; the owner's blind listen does (ab_listen.py)"
        )
    if changed:
        table.append(f"alpha {result['alpha']:.4g} a test, so one needs {result['min_untied']} untied pairs to ever reject")
        held = np.array(list(fad.calibration.values()))
        table.append(
            f"\nFAD calibration, {len(held)} held-out references: median {np.median(held):.3f}"
            f" ({held.min():.3f}..{held.max():.3f}); takes: base median"
            f" {np.median([p['base']['fad'] for p in pairs if not p['identical']]):.3f},"
            f" cand median {np.median([p['cand']['fad'] for p in pairs if not p['identical']]):.3f}"
        )
    for group, v in verdict.items():
        for t in v.get("to_explain", []):
            table.append(
                f"TO-EXPLAIN {group}: {t['band_hz']} Hz, {t['pairs_over']} of {v['pairs']} pairs past"
                f" ±{TO_EXPLAIN_DB:g} dB, median {t['median_db']:+.1f} ({t['min_db']:+.1f}..{t['max_db']:+.1f})"
            )
    record["table"] = "\n".join(table)
    (args.dir / "judge.json").write_text(json.dumps(record, indent=1) + "\n")
    print(record["table"])
    print(f"\nverdict: {result['overall']}  ({args.dir / 'judge.json'})")
    if failure := exit_message(result):
        sys.exit(failure)


if __name__ == "__main__":
    main()
