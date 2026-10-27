"""`ab_judge.py` on a Modal T4, over the reference corpus and the models two
Volumes cache, for a batch this machine uploads.

Usage, with modal-requirements.txt's client and a configured token:
  modal volume put lofi-judge-refs REFS/manifest.json /manifest.json
  modal run modal_judge.py::provision     fetch the corpus, then build the caches
  modal run modal_judge.py::check         judge the warm-up pair in this lock's
                                          image, offline, caching nothing
  modal run modal_judge.py::upload_missing --refs-dir REFS
                                          what archive.org didn't serve, from REFS
  modal run modal_judge.py::judge --batch DIR --base SHA --cand SHA [--out OUT] [--force-score]
                                          judge DIR's base/ and cand/, rendered
                                          at the commits their COMMIT files name

REFS is the private corpus: its manifest's names and its audio never leave
Modal in anything this prints or returns, which names a reference by its
manifest index.
"""

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

# where the image keeps ab_judge.py and the scripts it imports, and this
# directory beside them, where ab_judge finds validate_batch
REMOTE_SCRIPTS, REMOTE_JUDGE = "/root/scripts", "/root/judge"
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts" if modal.is_local() else Path(REMOTE_SCRIPTS)
JUDGE = HERE if modal.is_local() else Path(REMOTE_JUDGE)
MODELS, REFS, BATCH = "/models", "/refs", "/batch"
sys.path.insert(0, str(JUDGE))
from judge_local import exit_code, published, redact, select
from validate_batch import IDENTICAL

# the uv the image and SKILL.md's install run
UV_VERSION = "0.13.6"
# the lock, baked in where uv reads it a second time as the build constraints:
# encodec, wget and progressbar publish only an sdist
# (https://pypi.org/pypi/encodec/0.1.1/json and wget's and progressbar's), and
# uv's `--require-build-hashes`, a preview feature, "only applies to build
# dependencies" (https://docs.astral.sh/uv/pip/compile/,
# /uv/concepts/preview/): under `--require-hashes` alone, uv 0.13.0 built
# them with the newest setuptools, unpinned
LOCK = HERE / "judge-requirements.txt"
REMOTE_LOCK = "/root/judge-requirements.txt"

# what archive.org's bots guide asks of automated requests: "a descriptive
# User-Agent header that identifies: The tool or bot name, Version number",
# and to "Honor 429 Too Many Requests responses and Retry-After headers"
# (https://archive.org/developers/bots.html); this one names no person
USER_AGENT = "lofi-fetch-refs/1"
SPACING_S = 2
TRIES = 5
FETCH_TIMEOUT_S = 120

# over what the cache build and the full judge batches used, which the
# commit that set them records; the timeout is twice the slowest batch
CPU = 2.0
MEMORY_MIB = 12288
JUDGE_TIMEOUT_S = 4500

# the warm-up pair `build` judges: a take at lofi's own rate (`SAMPLE_RATE`
# in src/dsp.rs), the one a batch's takes arrive at
WARMUP_RATE = 44100
# a few of ab_judge.py's `WINDOW_S` windows, so every judge scores whole ones
# (opinion)
WARMUP_S = 30
# the noise's RMS: the pair sits louder than any reference, so `Fad` builds
# its statistics at the quietest reference's level, which every real batch's
# takes are over
WARMUP_LEVEL = 0.25
# the cand side's gain, so the pair differs: an identical one is never scored
WARMUP_CAND_GAIN = 0.5


image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(LOCK, REMOTE_LOCK, copy=True)
    .uv_pip_install(
        requirements=[str(LOCK)],
        extra_options=f"--require-hashes --build-constraints {REMOTE_LOCK} --require-build-hashes",
        uv_version=UV_VERSION,
    )
    .env({"HF_HOME": MODELS})
    .add_local_dir(SCRIPTS, REMOTE_SCRIPTS, ignore=["**/__pycache__"])
    .add_local_dir(JUDGE, REMOTE_JUDGE, ignore=["**/__pycache__"])
)
app = modal.App("lofi-judge", image=image)
models = modal.Volume.from_name("lofi-judge-models", create_if_missing=True)
refs = modal.Volume.from_name("lofi-judge-refs", create_if_missing=True)


def tracks():
    return json.loads((Path(REFS) / "manifest.json").read_text())["tracks"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def took(started):
    """What the calling function's children have taken since `started`: wall
    time, their CPU and their peak RSS."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"wall_s": time.time() - started, "cpu_s": usage.ru_utime + usage.ru_stime, "peak_rss_mib": usage.ru_maxrss / 1024}


def fetch(url):
    """`url`'s bytes, or None once TRIES requests have failed."""
    for attempt in range(TRIES):
        time.sleep(SPACING_S)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            after = e.headers.get("Retry-After", "")
            time.sleep(int(after) if after.isdigit() else SPACING_S * 2**attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(SPACING_S * 2**attempt)
    return None


@app.function(volumes={REFS: refs}, timeout=JUDGE_TIMEOUT_S)
def fetch_refs():
    """Fetches each reference missing or unverified in the refs Volume, one at
    a time, and keeps it only when its sha256 matches the manifest. With none
    left missing, builds the caches."""
    audio = Path(REFS) / "audio"
    audio.mkdir(exist_ok=True)
    served, kept, missing = 0, 0, []
    for i, t in enumerate(tracks()):
        path = audio / t["file"]
        if path.is_file() and sha256(path.read_bytes()) == t["sha256"]:
            kept += 1
            continue
        data = fetch(t["file_url"])
        if data is None or sha256(data) != t["sha256"]:
            missing.append(i)
            continue
        path.write_bytes(data)
        served += 1
    refs.commit()
    report = {"served": served, "kept": kept, "missing": missing}
    if not missing:
        report["caches"] = build.remote()
    return report


def warm_up(cache, env=None):
    """Judges a warm-up pair as a batch would be judged, over `cache`, which
    builds what it lacks, and says whether a verdict came out; prints the
    log, names masked, when none did."""
    import numpy as np
    import soundfile as sf

    warm = Path("/tmp/warm")
    rng = np.random.default_rng(0)
    take = WARMUP_LEVEL * rng.standard_normal((WARMUP_S * WARMUP_RATE, 2))
    for side, gain in (("base", 1.0), ("cand", WARMUP_CAND_GAIN)):
        (warm / side).mkdir(parents=True)
        sf.write(warm / side / "gen_day_empty_000.wav", take * gain, WARMUP_RATE, subtype="PCM_16")
        (warm / side / "COMMIT").write_text("0" * 40 + "\n")
    run = subprocess.run(
        [sys.executable, f"{REMOTE_SCRIPTS}/ab_judge.py", str(warm), "--refs", REFS, "--corpus-cache", cache],
        capture_output=True,
        text=True,
        env=env,
    )
    built = (warm / "judge.json").is_file()
    if not built:
        print(redact(run.stdout + run.stderr, tracks()))
    return built


@app.function(gpu="T4", cpu=CPU, memory=MEMORY_MIB, volumes={MODELS: models, REFS: refs}, timeout=JUDGE_TIMEOUT_S)
def build():
    """Caches every model, the corpus's FAD statistics and the panel's
    screening by judging the warm-up pair, online, then measures what it
    took."""
    started = time.time()
    built = warm_up(f"{REFS}/corpus-cache")
    models.commit()
    refs.commit()
    return {"built": built, **took(started)}


# offline, its network blocked and its Volumes read-only, so a lock not yet
# merged writes no cache a judge reads and sends the corpus nowhere
@app.function(gpu="T4", cpu=CPU, memory=MEMORY_MIB, timeout=JUDGE_TIMEOUT_S, block_network=True, restrict_modal_access=True)
def rehearse():
    """The judge as this image would run it: `ab_judge.py`'s tests, then the
    warm-up pair over a cache this container builds and drops."""
    started = time.time()
    offline = {**os.environ, "HF_HUB_OFFLINE": "1"}
    tests = subprocess.run(
        [sys.executable, "-I", f"{REMOTE_SCRIPTS}/test_ab_judge.py"], capture_output=True, text=True, env=offline
    )
    if tests.returncode:
        print(redact(tests.stdout + tests.stderr, tracks()))
        return {"built": False, **took(started)}
    return {"built": warm_up("/tmp/corpus-cache", offline), **took(started)}


@app.function(gpu="T4", cpu=CPU, memory=MEMORY_MIB, timeout=JUDGE_TIMEOUT_S)
def score(score_identical=False):
    """Judges the batch the caller mounts at BATCH, offline, over the
    read-only caches, and returns what's safe to publish: the table, the
    record without the corpus's names, and what the run took."""
    started = time.time()
    known = tracks()
    offline = {**os.environ, "HF_HUB_OFFLINE": "1"}
    # the tests that need this venv, which runs nowhere else: a broken floor
    # or corpus cache still prints a verdict
    tests = subprocess.run(
        [sys.executable, "-I", f"{REMOTE_SCRIPTS}/test_ab_judge.py"], capture_output=True, text=True, env=offline
    )
    if tests.returncode:
        print(redact(tests.stdout + tests.stderr, known))
        return {"outcome": "infra failure", "message": "ab_judge.py's tests failed; their log, names masked, is Modal's", **took(started)}
    run = subprocess.run(
        [sys.executable, f"{REMOTE_SCRIPTS}/ab_judge.py", BATCH, "--refs", REFS, "--corpus-cache", f"{REFS}/corpus-cache"]
        + (["--score-identical"] if score_identical else []),
        capture_output=True,
        text=True,
        env=offline,
    )
    written = Path(BATCH) / "judge.json"
    if not written.is_file():
        print(redact(run.stdout + run.stderr, known))
        return {"outcome": "infra failure", "message": "the judge stopped before a verdict; its log, names masked, is Modal's", **took(started)}
    return {**published(json.loads(written.read_text()), known), **took(started)}


@app.local_entrypoint()
def provision():
    """Fetches the corpus into the refs Volume and, once it is whole, builds
    the caches; prints what each step did."""
    report = fetch_refs.remote()
    print(f"archive.org served {report['served']}; {report['kept']} were in the Volume;"
          f" missing: {report['missing'] or 'none'}")
    if "caches" in report:
        print(f"caches: {report['caches']}")
    if not report.get("caches", {}).get("built"):
        raise SystemExit(1)


@app.local_entrypoint()
def check():
    """Rehearses the judge in an image built from this checkout's lock, over
    the cached models and corpus, read-only; exits non-zero when no verdict
    came out."""
    result = rehearse.with_options(
        volumes={MODELS: models.with_mount_options(read_only=True), REFS: refs.with_mount_options(read_only=True)}
    ).remote()
    print(f"rehearsed: {'a verdict' if result['built'] else 'no verdict'}; wall {result['wall_s']:.0f} s,"
          f" judge CPU {result['cpu_s']:.0f} s, peak RSS {result['peak_rss_mib']:.0f} MiB")
    if not result["built"]:
        raise SystemExit(1)


@app.local_entrypoint()
def judge(batch: str, base: str, cand: str, out: str = "", force_score: bool = False):
    """Judges `batch`'s base/ and cand/, rendered at commits `base` and
    `cand`, through a Volume that lives as long as the run, prints the table,
    and writes the result to `out`. Only the pairs `judge_local.select` picks
    go up, with the IDENTICAL list the batch keeps, so a batch with none
    makes no remote call; `force_score` sends and scores every pair, the
    proof that identical ones tie."""
    root = Path(batch)
    takes, differ = select(root, base, cand, force_score)
    if differ:
        with modal.Volume.ephemeral() as vol:
            with vol.batch_upload() as upload:
                for side in ("base", "cand"):
                    upload.put_file(str(root / side / "COMMIT"), f"/{side}/COMMIT")
                    for n in differ:
                        upload.put_file(str(root / side / n), f"/{side}/{n}")
                if (root / IDENTICAL).is_file():
                    upload.put_file(str(root / IDENTICAL), f"/{IDENTICAL}")
            result = score.with_options(
                volumes={
                    BATCH: vol,
                    MODELS: models.with_mount_options(read_only=True),
                    REFS: refs.with_mount_options(read_only=True),
                }
            ).remote(score_identical=force_score)
        print(result.get("table", result.get("message")))
        print(f"\nverdict: {result['outcome']}")
        print(f"\nwall {result['wall_s']:.0f} s, judge CPU {result['cpu_s']:.0f} s, peak RSS {result['peak_rss_mib']:.0f} MiB")
    else:
        result = {"outcome": "UNCHANGED"}
        print(f"verdict: UNCHANGED, all {len(takes.paired)} pairs byte-identical; nothing was sent or scored")
    result["identical_not_sent"] = len(takes.paired) - len(differ)
    for side in ("new", "gone"):
        result[side] = getattr(takes, side)
        if result[side]:
            print(f"{side} at cand, unscored: {' '.join(result[side])}")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "verdict.json").write_text(json.dumps(result, indent=1) + "\n")
    code = exit_code(result["outcome"])
    if code:
        raise SystemExit(code)


@app.local_entrypoint()
def upload_missing(refs_dir: str):
    """Uploads from `refs_dir` each reference the refs Volume lacks, once its
    sha256 matches the manifest."""
    local = Path(refs_dir)
    manifest = json.loads((local / "manifest.json").read_text())["tracks"]
    present = {Path(e.path).name for e in refs.listdir("/audio")}
    missing = [i for i, t in enumerate(manifest) if t["file"] not in present]
    with refs.batch_upload() as upload:
        for i in missing:
            path = local / "audio" / manifest[i]["file"]
            if sha256(path.read_bytes()) != manifest[i]["sha256"]:
                raise SystemExit(f"reference {i} doesn't match its manifest sha256 here either")
            upload.put_file(str(path), f"/audio/{manifest[i]['file']}")
    print(f"uploaded references {missing or 'none'}")
