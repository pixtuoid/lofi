//! The listening loop's numbers: what a sound change moves in a pixtuoid
//! tier's mix and in each of its lanes, and where the mix's spectrum sits in
//! the reference corpus. They steer a change; `ab_judge.py` judges it.
//! `--engine stream`, the default, measures a `Performance`'s song, which is
//! what lofi ships, its lanes as the stems its mix sums; `--engine whole`
//! measures the same score's lanes rendered whole and their mix looped as
//! long, at the stage where the stream's mix meets its ceiling, so the two
//! mixes compare.
//!
//! Usage:
//!   cargo run --release --example lofi_measure -- \
//!     [--engine whole|stream] [--mood day|night] [--seeds N] [--start S] [--tier empty|moderate|busy] \
//!     [--batch judge]

#[path = "common/batch.rs"]
mod batch;
#[path = "common/corpus.rs"]
mod corpus;
#[path = "common/loudness.rs"]
mod loudness;

use std::process::ExitCode;
use std::time::{Duration, Instant};

use pixtuoid_lofi::audition::{self, Stems};
use pixtuoid_lofi::dsp::SAMPLE_RATE;

use batch::{Batch, Engine, LANE_NAMES, lane_gains, render_take, soak_mix, usage_error};
use corpus::{BAND_EDGES_HZ, CORPUS, LTAS_BANDS_PER_OCTAVE, LTAS_SPAN_HZ, percentile};
use loudness::{fold, gated_lufs};

const SR: f64 = SAMPLE_RATE as f64;

/// `LOFI-BIBLE.md`'s MEASURED tempo, printed as a guard rail beside the take's.
const BPM_RAIL: [f64; 2] = [68.0, 90.0];

/// The seam ratio's reference: this quantile of the lane's own second
/// differences.
const SEAM_QUANTILE: f64 = 0.999;

/// The batch `args` select.
fn parse(mut args: impl Iterator<Item = String>) -> std::io::Result<Batch> {
    let mut batch = Batch::default();
    while let Some(a) = args.next() {
        if !batch.parse_flag(&a, &mut args)? {
            return Err(usage_error(format!(
                "unknown argument {a:?}; valid: --engine --mood --seeds --start --tier --batch"
            )));
        }
    }
    // its whole-lane rows measure one song's lanes under fixed controls
    if batch.sweep.is_some() || batch.songs.is_some() {
        return Err(usage_error(
            "lofi_measure measures one song under fixed controls, so it takes no --sweep or --songs"
                .into(),
        ));
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

fn run() -> std::io::Result<()> {
    let batch = parse(std::env::args().skip(1))?;
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
            let (lanes, mix, took, rendered) = match batch.engine {
                Engine::WholeLane => {
                    let (beds, took) = render_take(&score);
                    let len = audition::song_len(seed, 0, batch.controls());
                    let mix = soak_mix(&beds, &lane_gains(batch.controls()), len);
                    let (rendered, all) = (beds[0].len(), took.iter().sum());
                    (Lanes::Whole(beds, took), mix, all, rendered)
                }
                Engine::Stream => {
                    let started = Instant::now();
                    let mix = fold(&batch.perform(seed));
                    let took = started.elapsed();
                    // rendered again, so the render's time is the shipped one's
                    let mut stems = Stems::default();
                    batch.perform_with(seed, |p, out| stems.render(p, out));
                    let rendered = mix.len();
                    (Lanes::Stems(stems), mix, took, rendered)
                }
            };
            // the score's loop, rounded in f32 as the engine rounds it
            let loop_len = (score.loop_secs() * SAMPLE_RATE as f32).round() as usize;
            println!(
                "{} seed {seed}: {}; lead {}, loop {:.2} s; song {:.1} s: {}",
                batch.mood_name(),
                rail(f64::from(score.bpm()), BPM_RAIL, " bpm"),
                score.lead_voice_name(),
                loop_len as f64 / SR,
                score.song_secs(),
                score.form()
            );
            let header = cells(&|lo, hi| band_label((lo, hi)));
            print_row("lane", "LUFS", &header, "centroid", "seam", "ms");
            let transform = Transform::of(&mix);
            let spectrum = Spectrum::cross(&transform, &transform);
            match &lanes {
                Lanes::Whole(beds, took) => {
                    for ((name, bed), took) in LANE_NAMES.iter().zip(beds).zip(took) {
                        let own = Spectrum::of(bed);
                        print_row(
                            name,
                            &format!("{:.1}", gated_lufs(bed, SAMPLE_RATE)),
                            &cells(&|lo, hi| pct(own.share(lo, hi))),
                            &format!("{:.0}", own.centroid()),
                            &format!("{:.2}", seam_ratio(bed)),
                            &format!("{:.0}", took.as_secs_f64() * 1e3),
                        );
                    }
                }
                Lanes::Stems(stems) => {
                    println!("  stems in shares of the mix's power, which they sum to");
                    for (name, stem) in named(stems) {
                        let stem_transform = Transform::of(stem);
                        let own = Spectrum::cross(&stem_transform, &stem_transform);
                        let row =
                            shares_of(&Spectrum::cross(&stem_transform, &transform), &spectrum);
                        print_row(
                            name,
                            &format!("{:.1}", gated_lufs(stem, SAMPLE_RATE)),
                            &row.into_iter().map(pct).collect::<Vec<_>>(),
                            &format!("{:.0}", own.centroid()),
                            "-",
                            "-",
                        );
                    }
                }
            }
            let lufs = gated_lufs(&mix, SAMPLE_RATE);
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

/// What a take's lane rows read: each lane rendered whole with its render
/// time, or the stream's stems.
enum Lanes {
    Whole([Vec<f32>; LANE_NAMES.len()], [Duration; LANE_NAMES.len()]),
    Stems(Stems),
}

/// One row of the table: a cell per band between its LUFS and centroid.
fn print_row(name: &str, lufs: &str, bands: &[String], centroid: &str, seam: &str, ms: &str) {
    let bands: String = bands.iter().map(|cell| format!(" {cell:>7}")).collect();
    println!("  {name:<8} {lufs:>6}{bands} {centroid:>8} {seam:>6} {ms:>6}");
}

/// Each stem with the name it prints under: the lanes', then the plate's.
fn named(stems: &Stems) -> impl Iterator<Item = (&str, &[f32])> {
    let lanes = stems.lanes().iter().map(Vec::as_slice);
    LANE_NAMES
        .into_iter()
        .zip(lanes)
        .chain([("plate", stems.plate())])
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
        let n = buf.len().next_power_of_two();
        let mut re: Vec<f64> = buf.iter().map(|&v| f64::from(v)).collect();
        re.resize(n, 0.0);
        let mut im = vec![0.0; n];
        fft(&mut re, &mut im);
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

/// In-place iterative radix-2 FFT; `re.len()` is a power of two.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -std::f64::consts::TAU / len as f64;
        let (wr0, wi0) = (angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let (mut wr, mut wi) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * wr - im[b] * wi;
                let ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                (wr, wi) = (wr * wr0 - wi * wi0, wr * wi0 + wi * wr0);
            }
        }
        len <<= 1;
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
        let bible = include_str!("../.claude/skills/procedural-lofi/reference/LOFI-BIBLE.md");
        let [lo, hi] = BPM_RAIL;
        let rail = format!("BPM: {lo}–{hi} (MEASURED");
        assert!(
            bible.contains(&rail),
            "LOFI-BIBLE.md no longer reads {rail:?}"
        );
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
    fn a_takes_stem_rows_sum_to_its_mix_row() {
        // f32's rounding, of each stem and of the frame it sums to
        let rounding = 8.0 * f64::from(f32::EPSILON);
        let (batch, seed) = (Batch::default(), 0);
        let mut frames = vec![Frame::default(); 10 * SAMPLE_RATE as usize];
        let mut stems = audition::Stems::default();
        stems.render(&mut Performance::new(seed, batch.controls()), &mut frames);
        let mix = Transform::of(&fold(&frames));
        let spectrum = Spectrum::cross(&mix, &mix);
        let rows: Vec<Vec<f64>> = (named(&stems))
            .map(|(_, stem)| shares_of(&Spectrum::cross(&Transform::of(stem), &mix), &spectrum))
            .collect();
        for (k, (lo, hi)) in bands().into_iter().enumerate() {
            let sum: f64 = rows.iter().map(|row| row[k]).sum();
            let want = spectrum.share(lo, hi);
            assert!(
                (sum - want).abs() < rounding,
                "{}: the stems sum to {sum}, the mix {want}",
                band_label((lo, hi))
            );
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
        let (beds, _) = render_take(&batch.score(seed));
        let len = audition::song_len(seed, 0, batch.controls());
        let whole = gated_lufs(
            &soak_mix(&beds, &lane_gains(batch.controls()), len),
            SAMPLE_RATE,
        );
        let stream = gated_lufs(&fold(&batch.perform(seed)), SAMPLE_RATE);
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
