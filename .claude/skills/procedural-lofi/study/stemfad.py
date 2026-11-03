"""Which stem sits furthest from a style's Lofi Girl playlist: per-song FAD
over ab_judge's CLAP embedding, one stem at a time, each take's stem against
the style's references'.

  modal run stemfad.py::run --corpus STYLE --out DIR [--prefix take-NAME_]

Held-out references say what a real track scores, stem by stem, so a take's
distance reads against that and not against a number on its own. What comes
home is distances, the held-out references' and the takes PREFIX picks; the
stems stay on the Volume.
"""

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judge") if modal.is_local() else "/root/judge")
from judge_local import SHARDS
from modal_judge import CPU, MEMORY_MIB, MODELS, REMOTE_SCRIPTS, image, models

STUDY = "/study"
STEMS = ("drums", "bass", "other")
# the middle of a track, where every stem plays, for a fraction of the whole
# track's embedding cost
MIDDLE_S = 60
LEVEL_LUFS = -23.0
HOLDOUT_EVERY = 9
TIMEOUT_S = 1800
COV_FLOOR = 1e-6

app = modal.App("lofi-study-stemfad", image=image)
study = modal.Volume.from_name("lofi-study")


@app.function(
    gpu="T4", cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_S, max_containers=SHARDS,
    volumes={MODELS: models.with_mount_options(read_only=True), STUDY: study.with_mount_options(read_only=True)},
)
def embed(keys):
    import os

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, REMOTE_SCRIPTS)
    import ab_judge

    model = ab_judge.clap()
    out = {}
    for key in keys:
        frames = {}
        for name in STEMS:
            x, sr = sf.read(Path(STUDY) / "stems" / key / f"{name}.flac", dtype="float32")
            mid = x.shape[0] // 2
            half = MIDDLE_S * sr // 2
            x = x[max(mid - half, 0) : mid + half]
            lufs = ab_judge.mono_lufs(x, sr)
            if not np.isfinite(lufs):
                continue
            heard = ab_judge.mono(x).astype(np.float64) * 10 ** ((LEVEL_LUFS - lufs) / 20)
            wav = torch.from_numpy(heard.astype(np.float32))[None]
            wav = torchaudio.transforms.Resample(sr, model.sr, **ab_judge.RESAMPLE)(wav)[0]
            frames[name] = model.get_embedding(wav.numpy()).astype(np.float32).tolist()
        out[key] = frames
    return out


@app.local_entrypoint()
def run(corpus: str, out: str, prefix: str = ""):
    import numpy as np
    from scipy import linalg

    refs_prefix = f"{corpus}-"
    keys = sorted(e.path.rsplit("/", 1)[1] for e in study.listdir("/stems"))
    keys = [k for k in keys if k.startswith(refs_prefix) or (prefix and k.startswith(prefix))]
    batches = [keys[i::SHARDS] for i in range(SHARDS)]
    frames = {}
    for part in embed.map(batches):
        frames.update(part)
    refs = sorted(k for k in frames if k.startswith(refs_prefix))
    if not refs:
        raise SystemExit(f"stemfad.py: no stems start with {refs_prefix!r}")
    held = refs[::HOLDOUT_EVERY]
    train = [k for k in refs if k not in held]

    def stats(rows):
        return rows.mean(axis=0), np.cov(rows, rowvar=False)

    def distance(a, b):
        (mu1, cov1), (mu2, cov2) = a, b
        mean = linalg.sqrtm((cov1 + np.eye(cov1.shape[0]) * COV_FLOOR) @ (cov2 + np.eye(cov2.shape[0]) * COV_FLOOR))
        return float(((mu1 - mu2) ** 2).sum() + np.trace(cov1 + cov2 - 2 * mean.real))

    by_track = {}
    for name in STEMS:
        target = stats(np.concatenate([np.array(frames[k][name]) for k in train if name in frames[k]]))
        rows = {k: distance(target, stats(np.array(frames[k][name]))) for k in frames if k not in train and name in frames[k]}
        for key, value in rows.items():
            by_track.setdefault(key, {})[name] = value
        groups = {"held": [v for k, v in rows.items() if k in held],
                  "takes": [v for k, v in rows.items() if prefix and k.startswith(prefix)]}
        print(name, {g: (round(float(np.median(v)), 3), len(v)) for g, v in groups.items() if v})
    Path(out, "stemfad.json").write_text(json.dumps(by_track, indent=0) + "\n")
