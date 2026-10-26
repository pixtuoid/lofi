//! The listening loop's numbers: what a sound change moves in a pixtuoid
//! tier's mix and in each of its lanes, and where the mix's spectrum sits in
//! the reference corpus. They steer a change; `ab_judge.py` judges it.
//! `--engine stream`, the default, measures a `Performance`'s song, its lanes
//! as the stems its mix sums; `--engine whole` measures the whole-lane
//! engine's (`batch::Engine`) at the stage where the stream's mix meets its
//! ceiling, so the two mixes compare.
//!
//! Usage:
//!   cargo run --release --example lofi_measure -- \
//!     [--engine whole|stream] [--mood day|night] [--seeds N] [--start S] [--tier empty|moderate|busy] \
//!     [--sweep intensity|twilight|rain] [--songs 1] [--batch judge]
//!   A take plays one song, so a run of more is refused.
//!   cargo run --release --example lofi_measure -- [--corpus DIR]
//!   reads every track in DIR/audio as a take is read, through ffmpeg's
//!   decoder, and prints the `CORPUS` and `MEASURED_ON` they give.

#[path = "../common/batch.rs"]
mod batch;
mod corpus;

use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

use pixtuoid_lofi::Frame;
use pixtuoid_lofi::audition::{self, Lane, Stems};
use pixtuoid_lofi::dsp::SAMPLE_RATE;

use batch::{Batch, Engine, WholeSong, lane_gains, usage_error, valid_flags};
use corpus::{BAND_EDGES_HZ, CORPUS, LTAS_BANDS_PER_OCTAVE, LTAS_SPAN_HZ, percentile, spread};
use pixtuoid_lofi::audition::{channels, fold, gated_lufs};

const SR: f64 = SAMPLE_RATE as f64;

/// `LOFI-BIBLE.md`'s MEASURED tempo, printed as a guard rail beside the take's.
const BPM_RAIL: [f64; 2] = [68.0, 90.0];

/// The seam ratio's reference: this quantile of the lane's own second
/// differences (opinion).
const SEAM_QUANTILE: f64 = 0.999;

/// What a run reads: a batch's takes, or the reference corpus's tracks.
enum Mode {
    Takes(Batch),
    Corpus(PathBuf),
}

/// The flags lofi_measure takes beyond the batch's.
const OWN_FLAGS: [&str; 1] = ["--corpus"];

/// What `args` ask to read.
fn mode(mut args: impl Iterator<Item = String>) -> io::Result<Mode> {
    let args: Vec<String> = args.by_ref().collect();
    match args.as_slice() {
        [flag, dir] if flag == "--corpus" => Ok(Mode::Corpus(PathBuf::from(dir))),
        _ if args.iter().any(|a| a == "--corpus") => Err(usage_error(
            "--corpus reads the corpus alone, so it takes one directory and no other flag".into(),
        )),
        _ => parse(args.into_iter()).map(Mode::Takes),
    }
}

/// The batch `args` select.
fn parse(mut args: impl Iterator<Item = String>) -> io::Result<Batch> {
    let mut batch = Batch::default();
    while let Some(a) = args.next() {
        if !batch.parse_flag(&a, &mut args)? {
            return Err(usage_error(format!(
                "unknown argument {a:?}; valid: {}",
                valid_flags(&OWN_FLAGS)
            )));
        }
    }
    // its rows, the whole-lane engine's included, read a take's one song
    if let Some(run) = batch.runs().iter().find(|run| run.songs() > 1) {
        return Err(usage_error(format!(
            "lofi_measure reads a take's one song, so it takes no run of {} songs",
            run.songs()
        )));
    }
    Ok(batch)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lofi_measure: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> io::Result<()> {
    let batch = match mode(std::env::args().skip(1))? {
        Mode::Takes(batch) => batch,
        Mode::Corpus(dir) => return read_corpus(&dir),
    };
    let bands = bands();
    let cells = |cell: &dyn Fn(f64, f64) -> String| -> Vec<String> {
        bands.iter().map(|&(lo, hi)| cell(lo, hi)).collect()
    };
    let pct = |share: f64| format!("{:.2}%", 100.0 * share);
    for batch in batch.runs() {
        let mut mix_lufs = Vec::new();
        for seed in batch.seeds() {
            let score = batch.score(seed);
            // the whole-lane chain renders each lane's loop once, then plays it again
            let (rows, mix, lufs, took, rendered) = match batch.engine {
                Engine::WholeLane => {
                    let started = Instant::now();
                    let song = batch.whole(seed, lane_gains).remove(0);
                    let took = started.elapsed();
                    let mix = song.mix();
                    // a mono mix, heard from both speakers
                    let lufs = gated_lufs(&[&mix, &mix], SAMPLE_RATE);
                    // one loop of each lane, which the song plays again
                    let rendered = audition::samples_at(score.loop_secs(), SAMPLE_RATE);
                    (whole_rows(&song), mix, lufs, took, rendered)
                }
                Engine::Stream => {
                    let started = Instant::now();
                    let frames = batch.perform(seed);
                    let took = started.elapsed();
                    let [left, right] = channels(&frames);
                    let lufs = gated_lufs(&[&left, &right], SAMPLE_RATE);
                    let mix = fold(&frames);
                    // rendered again, so the render's time is the shipped one's
                    let mut stems = Stems::default();
                    batch.perform_with(seed, |p, out| stems.render(p, out));
                    let rendered = mix.len();
                    (stream_rows(&stems), mix, lufs, took, rendered)
                }
            };
            let loop_len = audition::samples_at(score.loop_secs(), SAMPLE_RATE);
            println!(
                "{} seed {seed}: {}; lead {}, loop {:.2} s; song {:.1} s: {}",
                batch.mood_name(),
                rail(f64::from(score.bpm()), BPM_RAIL, " bpm"),
                score.lead_voice_name(),
                loop_len as f64 / SR,
                audition::song_len(seed, 0, batch.controls()) as f64 / SR,
                audition::song_form(seed, 0, batch.controls())
            );
            let header = cells(&|lo, hi| band_label((lo, hi)));
            print_row("lane", "LUFS", &header, "centroid", "seam", "ms");
            println!("  stems in shares of the mix's power, which they sum to");
            let transform = Transform::of(&mix);
            let spectrum = Spectrum::cross(&transform, &transform);
            for row in &rows {
                let stem_transform = Transform::of(&row.stem);
                let own = Spectrum::cross(&stem_transform, &stem_transform);
                let shares = shares_of(&Spectrum::cross(&stem_transform, &transform), &spectrum);
                print_row(
                    row.name,
                    // a mono stem, heard from both speakers
                    &format!("{:.1}", gated_lufs(&[&row.stem, &row.stem], SAMPLE_RATE)),
                    &shares.into_iter().map(pct).collect::<Vec<_>>(),
                    &format!("{:.0}", own.centroid()),
                    &row.seam.map_or("-".into(), |seam| format!("{seam:.2}")),
                    "-",
                );
            }
            mix_lufs.push(lufs);
            print_row(
                "mix",
                &format!("{lufs:.1}"),
                &cells(&|lo, hi| pct(spectrum.share(lo, hi))),
                &format!("{:.0}", spectrum.centroid()),
                "-",
                &format!("{:.0}", took.as_secs_f64() * 1e3),
            );
            let spreads = bands.iter().zip(CORPUS.shares_pct);
            print_row(
                "corpus",
                "",
                &(spreads
                    .map(|(&(lo, hi), spread)| percentile(100.0 * spectrum.share(lo, hi), spread)))
                .collect::<Vec<_>>(),
                &percentile(spectrum.centroid(), CORPUS.centroid_hz),
                "",
                "",
            );
            let slope = spectrum.ltas_slope();
            println!(
                "  {} mix through {}: LTAS density slope {}-{} Hz {slope:.1} dB/oct ({} of the corpus); autocorrelation at the loop {:.4}; rendered at {:.4} of real time",
                batch.mix_name(),
                batch.engine.name(),
                LTAS_SPAN_HZ[0],
                LTAS_SPAN_HZ[1],
                percentile(slope, CORPUS.ltas_db_per_oct),
                loop_autocorrelation(&mix, loop_len),
                took.as_secs_f64() / (rendered as f64 / SR)
            );
        }
        // one seed's sample sd divides by zero
        if mix_lufs.len() > 1 {
            println!(
                "{} {} mix through {}, seeds {:?}: gated loudness sd {:.4} LU",
                batch.mood_name(),
                batch.mix_name(),
                batch.engine.name(),
                batch.seeds(),
                audition::lufs_sd(&mix_lufs)
            );
        }
    }
    Ok(())
}

/// Reads every track in `dir`'s `audio/` as a take's mix is read, and prints
/// the [`CORPUS`] and `MEASURED_ON` they give, for `corpus.rs`.
fn read_corpus(dir: &Path) -> io::Result<()> {
    let mut files: Vec<PathBuf> = (std::fs::read_dir(dir.join("audio"))?)
        .map(|entry| entry.map(|e| e.path()))
        .collect::<io::Result<_>>()?;
    files.sort();
    let mut shares = vec![Vec::new(); bands().len()];
    let (mut centroids, mut slopes) = (Vec::new(), Vec::new());
    for file in &files {
        let spectrum = Spectrum::of(&fold(&decode(file)?));
        for (share, (lo, hi)) in shares.iter_mut().zip(bands()) {
            share.push(100.0 * spectrum.share(lo, hi));
        }
        centroids.push(spectrum.centroid());
        slopes.push(spectrum.ltas_slope());
        eprintln!("read {}", file.display());
    }
    let row = |values: Vec<f64>, digits: usize| {
        let cells: Vec<String> = spread(values)
            .iter()
            .map(|v| format!("{v:.digits$}"))
            .collect();
        format!("[{}]", cells.join(", "))
    };
    println!("// {} tracks", files.len());
    println!("pub(crate) const CORPUS: Corpus = Corpus {{\n    shares_pct: [");
    for share in shares {
        println!("        {},", row(share, 2));
    }
    println!("    ],\n    centroid_hz: {},", row(centroids, 1));
    println!("    ltas_db_per_oct: {},\n}};", row(slopes, 2));
    println!(
        "pub(crate) const MEASURED_ON: Axes = Axes {{\n    band_edges_hz: {BAND_EDGES_HZ:?},\n    ltas_span_hz: {LTAS_SPAN_HZ:?},\n    ltas_bands_per_octave: {LTAS_BANDS_PER_OCTAVE:?},\n    sample_rate: {SAMPLE_RATE},\n}};"
    );
    Ok(())
}

/// `path` decoded by ffmpeg to stereo f32 at [`SAMPLE_RATE`], piped rather
/// than written, so no audio lands on disk.
fn decode(path: &Path) -> io::Result<Vec<Frame>> {
    let out = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args([
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-",
        ])
        .output()?;
    if !out.status.success() {
        return Err(io::Error::other(format!(
            "ffmpeg can't decode {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let samples: Vec<f32> = (out.stdout.as_chunks::<4>().0.iter())
        .map(|&bytes| f32::from_le_bytes(bytes))
        .collect();
    Ok(samples.as_chunks::<2>().0.to_vec())
}

/// One stem a take's mix sums, by name, as its row reads it: with the seam
/// of the bed it loops, where the whole-lane engine rendered one.
struct Row {
    name: &'static str,
    stem: Vec<f32>,
    seam: Option<f32>,
}

/// A whole-lane song's rows: each lane's stem and its bed's seam.
fn whole_rows(song: &WholeSong) -> Vec<Row> {
    (Lane::NAMES.into_iter().zip(song.stems()).zip(&song.beds))
        .map(|((name, stem), bed)| Row {
            name,
            stem,
            seam: Some(seam_ratio(bed)),
        })
        .collect()
}

/// A stream take's rows: its lanes' stems, then the plate's.
fn stream_rows(stems: &Stems) -> Vec<Row> {
    let lanes = stems.lanes().iter().map(Vec::as_slice);
    (Lane::NAMES.into_iter().zip(lanes))
        .chain([("plate", stems.plate())])
        .map(|(name, stem)| Row {
            name,
            stem: stem.to_vec(),
            seam: None,
        })
        .collect()
}

/// One row of the table: a cell per band between its LUFS and centroid.
fn print_row(name: &str, lufs: &str, bands: &[String], centroid: &str, seam: &str, ms: &str) {
    let bands: String = bands.iter().map(|cell| format!(" {cell:>7}")).collect();
    println!("  {name:<8} {lufs:>6}{bands} {centroid:>8} {seam:>6} {ms:>6}");
}

/// A stem's part of `mix`'s power in each band, its `cross` with the mix, as
/// a fraction of the mix's whole power.
fn shares_of(cross: &Spectrum, mix: &Spectrum) -> Vec<f64> {
    let total = mix.power(0.0, f64::INFINITY);
    bands()
        .into_iter()
        .map(|(lo, hi)| cross.power(lo, hi) / total)
        .collect()
}

/// The share bands, `[lo, hi)` Hz, from DC to Nyquist across [`BAND_EDGES_HZ`].
fn bands() -> Vec<(f64, f64)> {
    std::iter::once(0.0)
        .chain(BAND_EDGES_HZ)
        .zip(
            BAND_EDGES_HZ
                .into_iter()
                .chain(std::iter::once(f64::INFINITY)),
        )
        .collect()
}

fn band_label((lo, hi): (f64, f64)) -> String {
    let hz = |f: f64| {
        if f < 1000.0 {
            format!("{f}")
        } else {
            format!("{}k", f / 1000.0)
        }
    };
    if lo == 0.0 {
        format!("<{}", hz(hi))
    } else if hi.is_infinite() {
        format!(">{}", hz(lo))
    } else {
        format!("{}-{}", hz(lo), hz(hi))
    }
}

/// `value` beside its guard rail and which side of it the value sits on.
fn rail(value: f64, [lo, hi]: [f64; 2], unit: &str) -> String {
    let side = if value < lo {
        "under"
    } else if value > hi {
        "over"
    } else {
        "inside"
    };
    format!("{value:.1}{unit} ({side} the MEASURED {lo}-{hi}{unit})")
}

/// The wrap's second difference over the lane's own [`SEAM_QUANTILE`] one: at
/// most 1, the seam reads as interior signal.
fn seam_ratio(buf: &[f32]) -> f32 {
    let n = buf.len();
    let d2 = |i: usize| (buf[(i + 1) % n] - 2.0 * buf[i] + buf[(i + n - 1) % n]).abs();
    let wrap = d2(0).max(d2(n - 1));
    let mut all: Vec<f32> = (0..n).map(d2).collect();
    let k = ((n as f64 * SEAM_QUANTILE) as usize).min(n - 1);
    let (_, reference, _) = all.select_nth_unstable_by(k, f32::total_cmp);
    wrap / reference.max(f32::MIN_POSITIVE)
}

/// How alike the soak is to itself a loop later, 1 when every loop repeats.
fn loop_autocorrelation(mix: &[f32], loop_len: usize) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&a, &b) in mix.iter().zip(&mix[loop_len..]) {
        let (a, b) = (f64::from(a), f64::from(b));
        ab += a * b;
        aa += a * a;
        bb += b * b;
    }
    ab / (aa * bb).sqrt()
}

/// A buffer's DFT, zero-padded to a power of two.
struct Transform {
    re: Vec<f64>,
    im: Vec<f64>,
}

impl Transform {
    fn of(buf: &[f32]) -> Self {
        let (re, im) = audition::transform(buf);
        Self { re, im }
    }
}

/// A power spectrum: bins `0..=n/2` of an `n`-point [`Transform`].
struct Spectrum {
    power: Vec<f64>,
    hz_per_bin: f64,
}

impl Spectrum {
    fn of(buf: &[f32]) -> Self {
        let transform = Transform::of(buf);
        Self::cross(&transform, &transform)
    }

    /// `a`'s cross-power with `b`, Re(A·B̄) per bin: `a`'s power when `b` is
    /// `a`, and over parts that sum to `b`, summing to `b`'s power where the
    /// parts' own powers would miss the terms between them.
    ///
    /// # Panics
    ///
    /// If `a` and `b` are of different lengths, whose bins differ.
    fn cross(a: &Transform, b: &Transform) -> Self {
        let n = a.re.len();
        assert_eq!(n, b.re.len(), "transforms of different lengths");
        let power = (0..=n / 2)
            .map(|k| a.re[k] * b.re[k] + a.im[k] * b.im[k])
            .collect();
        Self {
            power,
            hz_per_bin: SR / n as f64,
        }
    }

    /// `(Hz, power)` per bin, DC left out: it carries no pitch.
    fn bins(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        (1..self.power.len()).map(|k| (k as f64 * self.hz_per_bin, self.power[k]))
    }

    /// The power in `[lo, hi)` Hz.
    fn power(&self, lo: f64, hi: f64) -> f64 {
        (self.bins())
            .filter(|&(f, _)| f >= lo && f < hi)
            .map(|(_, p)| p)
            .sum()
    }

    /// The fraction of the power in `[lo, hi)` Hz.
    fn share(&self, lo: f64, hi: f64) -> f64 {
        self.power(lo, hi) / self.power(0.0, f64::INFINITY)
    }

    fn centroid(&self) -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        for (f, p) in self.bins() {
            num += f * p;
            den += p;
        }
        num / den
    }

    /// The least-squares slope, in dB per octave, of power DENSITY (per bin,
    /// not per band, which would add 10·log10(2) dB/oct) over the fractional
    /// octave bands across [`LTAS_SPAN_HZ`].
    fn ltas_slope(&self) -> f64 {
        let [lo, hi] = LTAS_SPAN_HZ;
        let width = 2f64.powf(1.0 / LTAS_BANDS_PER_OCTAVE);
        let count = (LTAS_BANDS_PER_OCTAVE * (hi / lo).log2()).round() as i32;
        let points: Vec<(f64, f64)> = (0..count)
            .map(|b| {
                let f0 = lo * width.powi(b);
                let f1 = f0 * width;
                let (sum, bins) = self
                    .bins()
                    .filter(|&(f, _)| f >= f0 && f < f1)
                    .fold((0.0, 0usize), |(s, c), (_, p)| (s + p, c + 1));
                ((f0 * f1).sqrt().log2(), 10.0 * (sum / bins as f64).log10())
            })
            .collect();
        let mean = |pick: fn(&(f64, f64)) -> f64| {
            points.iter().map(pick).sum::<f64>() / points.len() as f64
        };
        let (mx, my) = (mean(|p| p.0), mean(|p| p.1));
        let sxy: f64 = points.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
        let sxx: f64 = points.iter().map(|(x, _)| (x - mx) * (x - mx)).sum();
        sxy / sxx
    }
}

#[cfg(test)]
mod tests {
    use pixtuoid_lofi::dsp::NoiseStream;
    use pixtuoid_lofi::{Frame, Performance};

    use super::*;

    fn sine(hz: f64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f64::consts::TAU * hz * i as f64 / SR).sin() as f32)
            .collect()
    }

    #[test]
    fn a_sweep_or_more_songs_is_refused() {
        for args in [["--sweep", "dusk"], ["--songs", "2"]] {
            assert!(
                parse(args.into_iter().map(str::to_owned)).is_err(),
                "{args:?} parsed"
            );
        }
    }

    #[test]
    fn the_tempo_rail_is_lofi_bibles_row() {
        let bible = include_str!("../../.claude/skills/procedural-lofi/reference/LOFI-BIBLE.md");
        let [lo, hi] = BPM_RAIL;
        let rail = format!("BPM: {lo}–{hi} (MEASURED");
        assert!(
            bible.contains(&rail),
            "LOFI-BIBLE.md no longer reads {rail:?}"
        );
    }

    #[test]
    fn the_usage_names_every_flag_and_a_run_of_songs_is_refused() {
        let usage: String = (include_str!("main.rs").lines())
            .take_while(|line| line.starts_with("//!"))
            .collect();
        for flag in valid_flags(&OWN_FLAGS).split(' ') {
            assert!(
                usage.contains(&format!("[{flag}")) || usage.contains(&format!("| {flag}")),
                "the usage leaves out {flag}"
            );
        }
        let args = |args: &[&str]| parse(args.iter().map(|&a| a.to_owned()));
        assert!(args(&["--sweep", "rain"]).is_ok());
        assert!(args(&["--sweep", "dusk"]).is_err());
        assert!(args(&["--songs", "2"]).is_err());
    }

    #[test]
    fn the_corpus_was_read_on_the_axes_a_take_is_read_on() {
        use corpus::MEASURED_ON;
        assert_eq!(MEASURED_ON.band_edges_hz, BAND_EDGES_HZ);
        assert_eq!(MEASURED_ON.ltas_span_hz, LTAS_SPAN_HZ);
        assert_eq!(MEASURED_ON.ltas_bands_per_octave, LTAS_BANDS_PER_OCTAVE);
        assert_eq!(MEASURED_ON.sample_rate, SAMPLE_RATE);
    }

    #[test]
    fn a_spread_is_numpys_linear_percentile() {
        // numpy.percentile([1, 2, 3, 4], [10, 25, 50, 75, 90])
        let got = spread(vec![4.0, 1.0, 3.0, 2.0]);
        for (g, w) in got.iter().zip([1.3, 1.75, 2.5, 3.25, 3.7]) {
            assert!((g - w).abs() < 1e-12, "{got:?}");
        }
    }

    #[test]
    fn the_corpus_is_read_alone() {
        let args = |args: &[&str]| mode(args.iter().map(|&a| a.to_owned()));
        assert!(matches!(args(&["--corpus", "refs"]), Ok(Mode::Corpus(_))));
        assert!(args(&["--corpus", "refs", "--tier", "busy"]).is_err());
        assert!(matches!(args(&["--tier", "busy"]), Ok(Mode::Takes(_))));
    }

    #[test]
    fn the_bass_window_and_its_limit_are_lofi_bibles_r7() {
        let bible = include_str!("../../.claude/skills/procedural-lofi/reference/LOFI-BIBLE.md");
        let low = audition::BASS_LOW;
        let r7 = [
            format!("bass root **{low}–{}**", low + 11),
            format!("no interval <P5 below {}", audition::LOW_INTERVAL_LIMIT),
        ];
        for row in &r7 {
            assert!(bible.contains(row), "LOFI-BIBLE.md no longer reads {row:?}");
        }
    }

    #[test]
    fn the_bands_split_the_top_at_4_and_6_5_khz() {
        let labels: Vec<String> = bands().into_iter().map(band_label).collect();
        let want = [
            "<62", "62-125", "125-250", "250-500", "500-1k", "1k-2k", "2k-4k", "4k-6.5k", ">6.5k",
        ];
        assert_eq!(labels, want);
    }

    #[test]
    fn every_corpus_spread_rises() {
        let spreads =
            (CORPUS.shares_pct.iter()).chain([&CORPUS.centroid_hz, &CORPUS.ltas_db_per_oct]);
        for spread in spreads {
            assert!(spread.is_sorted_by(|a, b| a < b), "{spread:?}");
        }
    }

    #[test]
    fn a_reading_falls_between_the_percentiles_around_it() {
        let spread = [1.0, 2.0, 3.0, 4.0, 5.0];
        for (value, want) in [
            (0.5, "<p10"),
            (1.0, "p10-25"),
            (2.5, "p25-50"),
            (5.0, "p75-90"),
            (5.5, ">p90"),
        ] {
            assert_eq!(percentile(value, spread), want, "{value}");
        }
    }

    #[test]
    fn a_takes_stem_rows_sum_to_its_mix_row_through_either_engine() {
        // f32's rounding, of each stem and of the frame it sums to
        let rounding = 8.0 * f64::from(f32::EPSILON);
        let (batch, seed) = (Batch::default(), 0);
        let mut frames = vec![Frame::default(); 10 * SAMPLE_RATE as usize];
        let mut stems = audition::Stems::default();
        stems.render(&mut Performance::new(seed, batch.controls()), &mut frames);
        let song = batch.whole(seed, lane_gains).remove(0);
        for (engine, mix, rows) in [
            ("stream", fold(&frames), stream_rows(&stems)),
            ("whole", song.mix(), whole_rows(&song)),
        ] {
            let mix = Transform::of(&mix);
            let spectrum = Spectrum::cross(&mix, &mix);
            let rows: Vec<Vec<f64>> = (rows.iter())
                .map(|row| shares_of(&Spectrum::cross(&Transform::of(&row.stem), &mix), &spectrum))
                .collect();
            for (k, (lo, hi)) in bands().into_iter().enumerate() {
                let sum: f64 = rows.iter().map(|row| row[k]).sum();
                let want = spectrum.share(lo, hi);
                assert!(
                    (sum - want).abs() < rounding,
                    "{engine} {}: the stems sum to {sum}, the mix {want}",
                    band_label((lo, hi))
                );
            }
        }
    }

    #[test]
    fn a_tone_lands_in_its_band_at_its_frequency() {
        let s = Spectrum::of(&sine(100.0, 1 << 16));
        assert!(s.share(62.0, 125.0) > 0.99, "{}", s.share(62.0, 125.0));
        assert!((s.centroid() - 100.0).abs() < 1.0, "{}", s.centroid());
    }

    #[test]
    fn white_noise_has_a_flat_ltas_density() {
        let mut rng = NoiseStream::new(1);
        let noise: Vec<f32> = (0..1 << 20).map(|_| rng.unit() - 0.5).collect();
        let slope = Spectrum::of(&noise).ltas_slope();
        assert!(slope.abs() < 0.5, "{slope}");
    }

    #[test]
    fn brown_noise_falls_6_db_per_octave() {
        let mut rng = NoiseStream::new(1);
        let mut walk = 0.0f64;
        let brown: Vec<f32> = (0..1 << 20)
            .map(|_| {
                walk += f64::from(rng.unit() - 0.5);
                walk as f32
            })
            .collect();
        let slope = Spectrum::of(&brown).ltas_slope();
        assert!((slope - (-6.0)).abs() < 0.5, "{slope}");
    }

    #[test]
    fn the_whole_lane_mix_reads_at_the_stage_the_streams_does() {
        let (batch, seed) = (Batch::default(), 0);
        let whole: Vec<f32> = batch
            .whole(seed, lane_gains)
            .iter()
            .flat_map(WholeSong::mix)
            .collect();
        let whole = gated_lufs(&[&whole, &whole], SAMPLE_RATE);
        let [left, right] = channels(&batch.perform(seed));
        let stream = gated_lufs(&[&left, &right], SAMPLE_RATE);
        // half the staging in dB, which a mix staged once less or once more
        // reads past
        let bound = -10.0 * Performance::MASTER_GAIN.log10();
        assert!(
            (whole - stream).abs() < bound,
            "the whole-lane mix reads {whole:.2} LUFS, the stream's {stream:.2}"
        );
    }

    #[test]
    fn only_a_loop_cut_mid_cycle_reads_a_seam() {
        let n = 1 << 16;
        // a whole number of cycles per loop
        let whole = sine(SR * 64.0 / n as f64, n);
        let cut = sine(SR * 64.5 / n as f64, n);
        assert!(seam_ratio(&whole) <= 1.0, "{}", seam_ratio(&whole));
        assert!(seam_ratio(&cut) > 1.0, "{}", seam_ratio(&cut));
    }
}
