//! The listening loop's numbers: what a sound change moves, per lane and in a
//! pixtuoid tier's mix. They steer a change; only the owner's LISTEN judges it.
//! Renders through the REAL `render_lane` chain, so it measures what ships.
//!
//! Usage:
//!   cargo run --release --example lofi_measure -- \
//!     [--mood day|night] [--seeds N] [--start S] [--tier empty|moderate|busy]

#[path = "common/batch.rs"]
mod batch;
#[path = "common/loudness.rs"]
mod loudness;

use std::process::ExitCode;

use pixtuoid_lofi::dsp::SAMPLE_RATE;
use pixtuoid_lofi::{Lane, compose};

use batch::{Batch, LANE_NAMES, render_take, soak_mix, tier_gain, usage_error};
use loudness::gated_lufs;

const SR: f64 = SAMPLE_RATE as f64;

/// The band shares' inner edges in Hz; the first is the guard rails' sub band.
const BAND_EDGES_HZ: [f64; 6] = [62.0, 125.0, 250.0, 500.0, 1000.0, 2000.0];

/// `LOFI-BIBLE.md`'s MEASURED rows, each printed as a guard rail beside the
/// take's value: the tempo in BPM, the percent of power under
/// [`BAND_EDGES_HZ`]`[0]`, and the centroid in Hz.
const BPM_RAIL: [f64; 2] = [68.0, 90.0];
const SUB_SHARE_RAIL_PCT: [f64; 2] = [55.0, 65.0];
const CENTROID_RAIL_HZ: [f64; 2] = [140.0, 160.0];

/// The LTAS slope's span in Hz, fitted over the power density of its fractional
/// octave bands.
const LTAS_SPAN_HZ: [f64; 2] = [250.0, 4000.0];
const LTAS_BANDS_PER_OCTAVE: f64 = 3.0;

/// The seam ratio's reference: this quantile of the lane's own second
/// differences.
const SEAM_QUANTILE: f64 = 0.999;

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
    let mut batch = Batch::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if !batch.parse_flag(&a, &mut args)? {
            return Err(usage_error(format!(
                "unknown argument {a:?}; valid: --mood --seeds --start --tier"
            )));
        }
    }
    let tier = batch.tier();
    let bands = bands();
    let header: Vec<String> = bands.iter().map(|&b| band_label(b)).collect();
    for seed in batch.seeds() {
        let score = compose(batch.mood(), seed);
        let take = render_take(&score);
        let loop_len = take[0].0.len();
        println!(
            "{} seed {seed}: {}; lead {}, loop {:.2} s",
            batch.mood_name(),
            rail(f64::from(score.bpm()), BPM_RAIL, " bpm"),
            score.lead_voice_name(),
            loop_len as f64 / SR
        );
        println!(
            "  {:<8} {:>6}{} {:>8} {:>6} {:>6}",
            "lane",
            "LUFS",
            header
                .iter()
                .map(|h| format!(" {h:>7}"))
                .collect::<String>(),
            "centroid",
            "seam",
            "ms"
        );
        for (name, (bed, took)) in LANE_NAMES.iter().zip(&take) {
            let spectrum = Spectrum::of(bed);
            let shares: String = bands
                .iter()
                .map(|&(lo, hi)| format!(" {:>6.1}%", 100.0 * spectrum.share(lo, hi)))
                .collect();
            println!(
                "  {name:<8} {:>6.1}{shares} {:>8.0} {:>6.2} {:>6.0}",
                gated_lufs(bed),
                spectrum.centroid(),
                seam_ratio(bed),
                took.as_secs_f64() * 1e3
            );
        }
        let beds = take.map(|(bed, _)| bed);
        let mix = soak_mix(&beds, &Lane::ALL.map(|lane| tier_gain(lane, tier)));
        let spectrum = Spectrum::of(&mix);
        let sub = BAND_EDGES_HZ[0];
        println!(
            "  {} mix: {:.1} LUFS; below {sub} Hz {}; centroid {}; LTAS density slope {}-{} Hz {:.1} dB/oct; autocorrelation at the loop {:.4}",
            tier.name(),
            gated_lufs(&mix),
            rail(100.0 * spectrum.share(0.0, sub), SUB_SHARE_RAIL_PCT, "%"),
            rail(spectrum.centroid(), CENTROID_RAIL_HZ, " Hz"),
            LTAS_SPAN_HZ[0],
            LTAS_SPAN_HZ[1],
            spectrum.ltas_slope(),
            loop_autocorrelation(&mix, loop_len)
        );
    }
    Ok(())
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

/// A buffer's power spectrum, zero-padded to a power of two: bins `0..=n/2`.
struct Spectrum {
    power: Vec<f64>,
    hz_per_bin: f64,
}

impl Spectrum {
    fn of(buf: &[f32]) -> Self {
        let n = buf.len().next_power_of_two();
        let mut re: Vec<f64> = buf.iter().map(|&v| f64::from(v)).collect();
        re.resize(n, 0.0);
        let mut im = vec![0.0; n];
        fft(&mut re, &mut im);
        let power = (0..=n / 2).map(|k| re[k] * re[k] + im[k] * im[k]).collect();
        Self {
            power,
            hz_per_bin: SR / n as f64,
        }
    }

    /// `(Hz, power)` per bin, DC left out: it carries no pitch.
    fn bins(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        (1..self.power.len()).map(|k| (k as f64 * self.hz_per_bin, self.power[k]))
    }

    /// The fraction of the power in `[lo, hi)` Hz.
    fn share(&self, lo: f64, hi: f64) -> f64 {
        let (mut band, mut total) = (0.0, 0.0);
        for (f, p) in self.bins() {
            total += p;
            if f >= lo && f < hi {
                band += p;
            }
        }
        band / total
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

    use super::*;

    fn sine(hz: f64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f64::consts::TAU * hz * i as f64 / SR).sin() as f32)
            .collect()
    }

    #[test]
    fn the_guard_rails_are_lofi_bibles_measured_rows() {
        let bible = include_str!("../.claude/skills/procedural-lofi/reference/LOFI-BIBLE.md");
        let [bpm_lo, bpm_hi] = BPM_RAIL;
        let ([sub_lo, sub_hi], [c_lo, c_hi]) = (SUB_SHARE_RAIL_PCT, CENTROID_RAIL_HZ);
        let rails = [
            format!("BPM: {bpm_lo}–{bpm_hi} (MEASURED"),
            format!(
                "{sub_lo}–{sub_hi}% of power below {} Hz, centroid {c_lo}–{c_hi} Hz",
                BAND_EDGES_HZ[0]
            ),
        ];
        for rail in &rails {
            assert!(
                bible.contains(rail),
                "LOFI-BIBLE.md no longer reads {rail:?}"
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
    fn only_a_loop_cut_mid_cycle_reads_a_seam() {
        let n = 1 << 16;
        // a whole number of cycles per loop
        let whole = sine(SR * 64.0 / n as f64, n);
        let cut = sine(SR * 64.5 / n as f64, n);
        assert!(seam_ratio(&whole) <= 1.0, "{}", seam_ratio(&whole));
        assert!(seam_ratio(&cut) > 1.0, "{}", seam_ratio(&cut));
    }
}
