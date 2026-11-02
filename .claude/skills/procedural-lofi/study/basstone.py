"""What the bass stem is made of, track by track: where its power sits, how its
notes hold, and how much of it is overtones you can hear on a small speaker.

  modal run basstone.py::run --prefix take-NAME_ --out DIR

What comes home is numbers; the stems stay on the Volume.
"""

import json
from pathlib import Path

import modal

STUDY = "/study"
# as `stemfad.py`'s, so a track's rows read the same minute
MIDDLE_S = 60
FRAME = 2048
HOP = 512
CONCURRENT = 8
CPU = 2.0
MEMORY_MIB = 4096
TIMEOUT_S = 900
LOG_FLOOR = 1e-12
VOICED_DB = -30
PITCH_LO_HZ, PITCH_HI_HZ = 30, 125
OVERTONE_LO_HZ, OVERTONE_HI_HZ = 125, 2000

app = modal.App("lofi-study-basstone")
image = modal.Image.debian_slim(python_version="3.11").apt_install("libsndfile1").pip_install("numpy<2", "scipy", "soundfile==0.12.1")
study = modal.Volume.from_name("lofi-study")
BANDS = ((20, 45), (45, 62), (62, 90), (90, 125), (125, 250), (250, 500), (500, 2000))


@app.function(image=image, cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_S, max_containers=CONCURRENT,
              volumes={STUDY: study.with_mount_options(read_only=True)})
def one(key):
    import numpy as np
    import soundfile as sf
    from scipy import signal

    x, sr = sf.read(Path(STUDY) / "stems" / key / "bass.flac", dtype="float32")
    x = x.mean(axis=1) if x.ndim > 1 else x
    mid, half = len(x) // 2, MIDDLE_S * sr // 2
    x = x[max(mid - half, 0) : mid + half]
    f, t, s = signal.stft(x, sr, nperseg=FRAME * 2, noverlap=FRAME * 2 - HOP)
    power = np.abs(s) ** 2
    total = power.sum() or 1.0
    shares = {f"{lo}-{hi}": float(100 * power[(f >= lo) & (f < hi)].sum() / total) for lo, hi in BANDS}
    centroid = float((f[None].T * power).sum() / total)
    env = np.sqrt(np.maximum(power.sum(axis=0), LOG_FLOOR))
    db = 20 * np.log10(env / env.max())
    voiced = db > VOICED_DB
    pitch_band = (f >= PITCH_LO_HZ) & (f < PITCH_HI_HZ)
    fundamental = power[pitch_band].sum()
    overtones = power[(f >= OVERTONE_LO_HZ) & (f < OVERTONE_HI_HZ)].sum()
    return key, {
        "shares": shares, "centroid_hz": centroid, "occupancy": float(voiced.mean()),
        "overtone_db": float(10 * np.log10((overtones + LOG_FLOOR) / (fundamental + LOG_FLOOR))),
    }


@app.local_entrypoint()
def run(prefix: str, out: str):
    keys = sorted(e.path.rsplit("/", 1)[1] for e in study.listdir("/stems"))
    keys = [k for k in keys if k.startswith(prefix) or not k.startswith("take-")]
    rows = dict(one.map(keys))
    Path(out, "basstone.json").write_text(json.dumps(rows, indent=0) + "\n")
    print(len(rows), "read")
