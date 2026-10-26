//! Minimal DSP kernel for the procedural synth — brickwall band filters,
//! spectral-envelope noise shaping, its spectral measures, and a deterministic
//! noise stream. The filters take a whole buffer — one voice's note, one foley
//! one-shot — never a stream, so none runs per audio frame.
//! Dependency-free on purpose: a consumer's wasm ships this module, and an FFT
//! crate is a download every visitor pays.

/// The one sample rate every buffer this crate makes or measures uses
/// (CD-standard mono).
pub const SAMPLE_RATE: u32 = 44_100;

/// ⌊2⁶⁴/φ⌋, φ the golden ratio: splitmix64's increment.
const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX64_M1: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX64_M2: u64 = 0x94d0_49bb_1331_11eb;

/// The splitmix64 finalizer.
pub(crate) fn splitmix64(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX64_M1);
    let z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX64_M2);
    z ^ (z >> 31)
}

/// Draw `n` of the splitmix64 stream seeded at `seed`, counting from 1.
pub(crate) fn splitmix_draw(seed: u64, n: u64) -> u64 {
    splitmix64(seed.wrapping_add(n.wrapping_mul(GOLDEN_GAMMA)))
}

/// Deterministic noise stream over the splitmix64 finalizer — seedable, so
/// synthesized assets are reproducible run-to-run. No `Clone`: a clone replays
/// the same draws, and the draw order is the sound.
#[derive(Debug)]
pub struct NoiseStream {
    seed: u64,
    counter: u64,
}

impl NoiseStream {
    /// A stream seeded at `seed`; equal seeds draw equal sequences.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    fn next_u64(&mut self) -> u64 {
        self.counter = self.counter.wrapping_add(1);
        splitmix_draw(self.seed, self.counter)
    }

    /// A stream seeded by this one's next draw, so what it draws moves no
    /// sibling fork's draws.
    pub(crate) fn fork(&mut self) -> Self {
        Self::new(self.next_u64())
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Approximately standard-normal (Irwin–Hall n=4, unit variance).
    pub fn norm(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.732_051
    }
}

/// The gain that keeps [`NoiseStream::norm`]'s draws, one a sample, as loud in
/// each hertz at `sr` as at [`SAMPLE_RATE`], where every noise level was set:
/// white noise's power spectral density "is a constant"
/// (<https://ccrma.stanford.edu/~jos/sasp/White_Noise_I.html>), so a draw's
/// variance spreads evenly up to the rate's Nyquist.
pub(crate) fn white_noise_gain(sr: u32) -> f32 {
    (sr as f32 / SAMPLE_RATE as f32).sqrt()
}

/// The float an [`fft`] runs in: `f32` on the render path, `f64` where a tool
/// sums millions of bins.
pub(crate) trait Real:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
{
    const ONE: Self;

    /// `x`, rounded to this float.
    fn of(x: f64) -> Self;
}

impl Real for f32 {
    const ONE: Self = 1.0;

    fn of(x: f64) -> Self {
        x as f32
    }
}

impl Real for f64 {
    const ONE: Self = 1.0;

    fn of(x: f64) -> Self {
        x
    }
}

/// `e^(−2πik/n)` for `k` in `0..n/2` as `(cos, sin)` rows, built per transform
/// pair and dropped: a kept table for the longest beds pins megabytes.
pub(crate) type Twiddles<T = f32> = Vec<(T, T)>;

/// [`Twiddles`] for `n`, in f64: a recurrence across a stage drifts by its
/// length.
pub(crate) fn twiddles<T: Real>(n: usize) -> Twiddles<T> {
    (0..n / 2)
        .map(|k| {
            let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
            (T::of(a.cos()), T::of(a.sin()))
        })
        .collect()
}

/// In-place iterative radix-2 FFT (Cooley–Tukey), unscaled. `re`/`im` len
/// must be a power of two no longer than the table: `tw` is [`twiddles`] for
/// a power of two at least `re.len()`.
pub(crate) fn fft<T: Real>(re: &mut [T], im: &mut [T], tw: &[(T, T)], inverse: bool) {
    let n = re.len();
    debug_assert!(n.is_power_of_two() && im.len() == n && 2 * tw.len() >= n);
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
    let sign = if inverse { -T::ONE } else { T::ONE };
    let mut len = 2;
    // Two radix-2 stages per pass where two remain (radix-2²): half the passes
    // over a buffer that outgrows the cache, and the second stage's odd
    // twiddles are the even ones turned a quarter.
    while 2 * len <= n {
        let h = len / 2;
        let (step1, step2) = (2 * tw.len() / len, tw.len() / len);
        for (re, im) in re
            .chunks_exact_mut(2 * len)
            .zip(im.chunks_exact_mut(2 * len))
        {
            for k in 0..h {
                let (w1r, w1i) = tw[k * step1];
                let (w2r, w2i) = tw[k * step2];
                let (w1i, w2i) = (sign * w1i, sign * w2i);
                let [i0, i1, i2, i3] = [k, k + h, k + 2 * h, k + 3 * h];
                let (t1r, t1i) = (re[i1] * w1r - im[i1] * w1i, re[i1] * w1i + im[i1] * w1r);
                let (t3r, t3i) = (re[i3] * w1r - im[i3] * w1i, re[i3] * w1i + im[i3] * w1r);
                let (y0r, y0i) = (re[i0] + t1r, im[i0] + t1i);
                let (y1r, y1i) = (re[i0] - t1r, im[i0] - t1i);
                let (y2r, y2i) = (re[i2] + t3r, im[i2] + t3i);
                let (y3r, y3i) = (re[i2] - t3r, im[i2] - t3i);
                let (u2r, u2i) = (y2r * w2r - y2i * w2i, y2r * w2i + y2i * w2r);
                // w2 · e^(∓iπ/2): the twiddle `h` further along the stage
                let (u3r, u3i) = {
                    let (ar, ai) = (y3r * w2r - y3i * w2i, y3r * w2i + y3i * w2r);
                    (sign * ai, -sign * ar)
                };
                (re[i0], im[i0]) = (y0r + u2r, y0i + u2i);
                (re[i2], im[i2]) = (y0r - u2r, y0i - u2i);
                (re[i1], im[i1]) = (y1r + u3r, y1i + u3i);
                (re[i3], im[i3]) = (y1r - u3r, y1i - u3i);
            }
        }
        len <<= 2;
    }
    if len <= n {
        let half = len / 2;
        let step = 2 * tw.len() / len;
        for (re, im) in re.chunks_exact_mut(len).zip(im.chunks_exact_mut(len)) {
            let (ur, vr) = re.split_at_mut(half);
            let (ui, vi) = im.split_at_mut(half);
            let w = tw.iter().step_by(step);
            for ((((ur, ui), vr), vi), &(wr, wi)) in ur.iter_mut().zip(ui).zip(vr).zip(vi).zip(w) {
                let wi = sign * wi;
                let (tr, ti) = (*vr * wr - *vi * wi, *vr * wi + *vi * wr);
                (*vr, *vi) = (*ur - tr, *ui - ti);
                (*ur, *ui) = (*ur + tr, *ui + ti);
            }
        }
    }
}

/// Bins `0..=n/2` of the spectrum of `buf` zero-padded to `n`, `tw` being
/// [`twiddles`]`(n)`, through one `n/2`-point complex FFT of its even and odd
/// samples (a real signal's spectrum is conjugate-symmetric, so the upper half
/// is redundant).
fn spectrum(buf: &[f32], tw: &[(f32, f32)]) -> (Vec<f32>, Vec<f32>) {
    let m = tw.len();
    let (mut zr, mut zi) = (vec![0.0f32; m], vec![0.0f32; m]);
    for (k, pair) in buf.chunks(2).enumerate() {
        zr[k] = pair[0];
        zi[k] = pair.get(1).copied().unwrap_or(0.0);
    }
    fft(&mut zr, &mut zi, tw, false);
    let (mut xr, mut xi) = (vec![0.0f32; m + 1], vec![0.0f32; m + 1]);
    for k in 0..=m {
        let (ar, ai) = (zr[k % m], zi[k % m]);
        let (br, bi) = (zr[(m - k) % m], -zi[(m - k) % m]);
        // the even samples' spectrum, and the odd samples'
        let (er, ei) = (f32::midpoint(ar, br), f32::midpoint(ai, bi));
        let (or, oi) = ((ai - bi) * 0.5, (br - ar) * 0.5);
        let (wr, wi) = tw.get(k).copied().unwrap_or((-1.0, 0.0));
        xr[k] = er + wr * or - wi * oi;
        xi[k] = ei + wr * oi + wi * or;
    }
    (xr, xi)
}

/// The `n`-sample real signal whose bins `0..=n/2` are `xr`/`xi`, `tw` being
/// [`twiddles`]`(n)`: [`spectrum`] inverted, scaled back to its input.
fn signal(xr: &[f32], xi: &[f32], tw: &[(f32, f32)]) -> Vec<f32> {
    let m = tw.len();
    let (mut zr, mut zi) = (vec![0.0f32; m], vec![0.0f32; m]);
    for k in 0..m {
        let (ar, ai) = (xr[k], xi[k]);
        let (br, bi) = (xr[m - k], -xi[m - k]);
        let (er, ei) = (f32::midpoint(ar, br), f32::midpoint(ai, bi));
        let (dr, di) = ((ar - br) * 0.5, (ai - bi) * 0.5);
        let (wr, wi) = tw[k];
        let (or, oi) = (dr * wr + di * wi, di * wr - dr * wi);
        (zr[k], zi[k]) = (er - oi, ei + or);
    }
    fft(&mut zr, &mut zi, tw, true);
    let scale = 1.0 / m as f32;
    zr.iter()
        .zip(&zi)
        .flat_map(|(&e, &o)| [e * scale, o * scale])
        .collect()
}

/// `buf`'s half spectrum at the next power of two, with the twiddles it was
/// taken with (half that length) for the [`signal`] back, and the bin width
/// at `sr`.
fn forward_spectrum(buf: &[f32], sr: u32) -> (Vec<f32>, Vec<f32>, Twiddles, f32) {
    let n = buf.len().next_power_of_two().max(2);
    let tw = twiddles(n);
    let (re, im) = spectrum(buf, &tw);
    (re, im, tw, sr as f32 / n as f32)
}

/// Mirror-aware frequency of FFT bin `k` in an `n`-point spectrum: bins above
/// `n/2` are NEGATIVE frequencies, folded back to `n − k`. A copy that drops the
/// fold band-passes the wrong half of the spectrum.
fn bin_freq(k: usize, n: usize, hz_per_bin: f32) -> f32 {
    (if k <= n / 2 { k } else { n - k }) as f32 * hz_per_bin
}

/// Brickwall band-pass via FFT bin zeroing. Construction-time only — a
/// linear-phase FIR would be overkill for pre-rendered assets. Keeps
/// `buf.len()`; `buf` is zero-padded to the next power of two, so a
/// power-of-two buffer filters circularly (a seamless loop stays seamless) and
/// any other length does not.
///
/// # Panics
///
/// If `lo_hz` is over `hi_hz`, or either is NaN.
#[must_use]
pub fn bandpass(buf: &[f32], lo_hz: f32, hi_hz: f32) -> Vec<f32> {
    bandpass_at(buf, lo_hz, hi_hz, SAMPLE_RATE)
}

/// [`bandpass`] of a buffer sampled at `sr`.
pub(crate) fn bandpass_at(buf: &[f32], lo_hz: f32, hi_hz: f32, sr: u32) -> Vec<f32> {
    assert!(
        lo_hz <= hi_hz,
        "a band-pass keeps the band from lo_hz up to hi_hz"
    );
    let (mut re, mut im, tw, hz_per_bin) = forward_spectrum(buf, sr);
    let n = 2 * tw.len();
    for (k, (r, i)) in re.iter_mut().zip(&mut im).enumerate() {
        let f = bin_freq(k, n, hz_per_bin);
        if f < lo_hz || f > hi_hz {
            *r = 0.0;
            *i = 0.0;
        }
    }
    let mut out = signal(&re, &im, &tw);
    out.truncate(buf.len());
    out
}

/// Brickwall low-pass at `cutoff_hz`.
///
/// # Panics
///
/// If `cutoff_hz` is negative or NaN.
#[must_use]
pub fn lowpass(buf: &[f32], cutoff_hz: f32) -> Vec<f32> {
    bandpass(buf, 0.0, cutoff_hz)
}

pub(crate) fn highpass(buf: &[f32], cutoff_hz: f32, sr: u32) -> Vec<f32> {
    bandpass_at(buf, cutoff_hz, sr as f32, sr)
}

/// Tape wow/flutter — resample `buf` along a sinusoidally warped time axis
/// (linear interpolation, edge-clamped). Each `(hz, dev)` pair contributes a
/// pitch deviation of ±`dev` (fractional) by displacing the read head
/// `dev·SR/(2π·hz)` samples at rate `hz`.
pub(crate) fn warp_resample(buf: &[f32], warps: &[(f32, f32)]) -> Vec<f32> {
    let n = buf.len();
    let amp: Vec<(f32, f32)> = warps
        .iter()
        .map(|&(hz, dev)| {
            (
                hz,
                dev * SAMPLE_RATE as f32 / (2.0 * std::f32::consts::PI * hz),
            )
        })
        .collect();
    (0..n)
        .map(|i| {
            let mut p = i as f32;
            for &(hz, a) in &amp {
                p += a * (2.0 * std::f32::consts::PI * hz * i as f32 / SAMPLE_RATE as f32).sin();
            }
            let p = p.clamp(0.0, (n - 1) as f32);
            let lo = p.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = p - lo as f32;
            buf[lo] * (1.0 - frac) + buf[hi] * frac
        })
        .collect()
}

/// One band of a spectral envelope: its share of the power, as
/// [`band_energy_share`] measures it, from `lo_hz` up to `hi_hz`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Band {
    lo_hz: f32,
    hi_hz: f32,
    share: f32,
}

impl Band {
    /// The band from `lo_hz` up to `hi_hz` holding `share` of the power.
    ///
    /// # Panics
    ///
    /// If `lo_hz` is negative or not under `hi_hz`, or `share` is outside
    /// `[0, 1]`, NaN included; in a `const`, when it compiles.
    #[must_use]
    pub const fn new(lo_hz: f32, hi_hz: f32, share: f32) -> Self {
        assert!(
            0.0 <= lo_hz && lo_hz < hi_hz,
            "a band runs up from a frequency of at least 0"
        );
        assert!(
            0.0 <= share && share <= 1.0,
            "a band's share is a fraction of the power"
        );
        Self {
            lo_hz,
            hi_hz,
            share,
        }
    }

    /// Where the band starts, in Hz.
    #[must_use]
    pub const fn lo_hz(self) -> f32 {
        self.lo_hz
    }

    /// Where the band ends, in Hz, outside it.
    #[must_use]
    pub const fn hi_hz(self) -> f32 {
        self.hi_hz
    }

    /// The band's share of the power, a fraction.
    #[must_use]
    pub const fn share(self) -> f32 {
        self.share
    }
}

/// The span a [`shaped_noise_loop`]'s envelope is smoothed over, so the stairs
/// between bands don't ring: the 201 bins pixtuoid's rain was ratified with,
/// at its loop of 2^19 samples, in Hz, so a loop of any length smooths as much.
const EDGE_SMOOTHING_HZ: f32 = 201.0 * SAMPLE_RATE as f32 / (1 << 19) as f32;

/// A power-of-two block of noise FFT-shaped to a measured envelope —
/// CIRCULARLY seamless by construction (FFT-domain shaping is periodic in the
/// block), so the returned block loops without a click. The block is
/// normalized to peak 1.0.
///
/// ```
/// use pixtuoid_lofi::dsp::{Band, NoiseStream, band_energy_share, shaped_noise_loop};
///
/// let bands = [Band::new(100.0, 1000.0, 0.7), Band::new(1000.0, 8000.0, 0.3)];
/// let wash = shaped_noise_loop(1 << 16, &bands, &mut NoiseStream::new(7));
/// for band in bands {
///     let share = band_energy_share(&wash, band.lo_hz(), band.hi_hz());
///     assert!((share - band.share()).abs() < 0.05);
/// }
/// ```
///
/// # Panics
///
/// If `n_pow2` is not a power of two of at least 2.
#[must_use]
pub fn shaped_noise_loop(n_pow2: usize, bands: &[Band], rng: &mut NoiseStream) -> Vec<f32> {
    assert!(
        n_pow2 >= 2 && n_pow2.is_power_of_two(),
        "a noise loop is a power-of-two block of at least 2 samples"
    );
    let noise: Vec<f32> = (0..n_pow2).map(|_| rng.norm()).collect();
    let tw = twiddles(n_pow2);
    let (mut re, mut im) = spectrum(&noise, &tw);
    let hz_per_bin = SAMPLE_RATE as f32 / n_pow2 as f32;
    // per-bin amplitude gain: sqrt(band power share / band bin count)
    let mut gain = vec![0.0f32; n_pow2];
    for band in bands {
        let bins = ((band.hi_hz - band.lo_hz) / hz_per_bin).max(1.0);
        let g = (band.share / bins).sqrt();
        for (k, gk) in gain.iter_mut().enumerate() {
            let f = bin_freq(k, n_pow2, hz_per_bin);
            if f >= band.lo_hz && f < band.hi_hz {
                *gk = g;
            }
        }
    }
    let smoothed = moving_average(&gain, (EDGE_SMOOTHING_HZ / hz_per_bin).round() as usize);
    for ((r, i), s) in re.iter_mut().zip(&mut im).zip(&smoothed) {
        *r *= s;
        *i *= s;
    }
    let mut out = signal(&re, &im, &tw);
    let peak = out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-9);
    out.iter_mut().for_each(|v| *v /= peak);
    out
}

/// `x` averaged over `window / 2` neighbours on each side, fewer at its ends.
fn moving_average(x: &[f32], window: usize) -> Vec<f32> {
    let half = window / 2;
    let n = x.len();
    let mut out = vec![0.0f32; n];
    let mut acc = 0.0f32;
    let mut count = 0usize;
    let mut lo = 0usize;
    let mut hi = 0usize; // exclusive
    for (i, o) in out.iter_mut().enumerate() {
        let want_lo = i.saturating_sub(half);
        let want_hi = (i + half + 1).min(n);
        while hi < want_hi {
            acc += x[hi];
            count += 1;
            hi += 1;
        }
        while lo < want_lo {
            acc -= x[lo];
            count -= 1;
            lo += 1;
        }
        *o = acc / count as f32;
    }
    out
}

/// Spectral centroid of `buf` in Hz: its bins' frequencies weighted by their
/// power, over the bins [`band_energy_share`] reads.
#[must_use]
pub fn centroid_hz(buf: &[f32]) -> f32 {
    let (re, im, _, hz_per_bin) = forward_spectrum(buf, SAMPLE_RATE);
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (k, (r, i)) in re.iter().zip(&im).enumerate() {
        let p = f64::from(r * r + i * i);
        num += k as f64 * f64::from(hz_per_bin) * p;
        den += p;
    }
    (num / den.max(1e-12)) as f32
}

/// Fraction of `buf`'s spectral power inside `[lo_hz, hi_hz)`, over the bins
/// from 0 Hz to Nyquist of `buf` zero-padded to the next power of two.
#[must_use]
pub fn band_energy_share(buf: &[f32], lo_hz: f32, hi_hz: f32) -> f32 {
    let (re, im, tw, hz_per_bin) = forward_spectrum(buf, SAMPLE_RATE);
    let (mut band, mut total) = (0.0f64, 0.0f64);
    for k in 0..=tw.len() {
        let p = f64::from(re[k] * re[k] + im[k] * im[k]);
        let f = k as f32 * hz_per_bin;
        total += p;
        if f >= lo_hz && f < hi_hz {
            band += p;
        }
    }
    (band / total.max(1e-12)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bands pixtuoid's rain plays
    /// (crates/pixtuoid-scene/src/audio/foley.rs at a63f178).
    const RAIN: [Band; 9] = [
        Band::new(20.0, 60.0, 0.015),
        Band::new(60.0, 125.0, 0.035),
        Band::new(125.0, 250.0, 0.069),
        Band::new(250.0, 500.0, 0.143),
        Band::new(500.0, 1000.0, 0.244),
        Band::new(1000.0, 2000.0, 0.253),
        Band::new(2000.0, 4000.0, 0.102),
        Band::new(4000.0, 8000.0, 0.089),
        Band::new(8000.0, 16000.0, 0.04),
    ];

    #[test]
    #[should_panic(expected = "a noise loop is a power-of-two block of at least 2 samples")]
    fn a_one_sample_noise_loop_is_refused_at_the_guard() {
        let _ = shaped_noise_loop(1, &[Band::new(100.0, 200.0, 1.0)], &mut NoiseStream::new(1));
    }

    #[test]
    #[should_panic(expected = "a band runs up from a frequency of at least 0")]
    fn a_band_that_ends_where_it_starts_is_refused() {
        let _ = Band::new(200.0, 200.0, 0.5);
    }

    #[test]
    #[should_panic(expected = "a band's share is a fraction of the power")]
    fn a_band_s_share_in_percent_is_refused() {
        let _ = Band::new(100.0, 200.0, 25.0);
    }

    #[test]
    #[should_panic(expected = "a band-pass keeps the band from lo_hz up to hi_hz")]
    fn an_inverted_band_pass_is_refused() {
        let _ = bandpass(&[0.0; 64], 800.0, 300.0);
    }

    /// The shares a loop of each length plays on average over seeds: one
    /// loop's share of a band of few bins is as random as their noise. Bins'
    /// smoothing biased a loop of 2^12 by 4.7 points over these seeds, and the
    /// smoothing in Hz by 1.3 (opinion, between the two).
    #[test]
    fn shaped_noise_plays_its_envelope_on_average_at_every_loop_length() {
        const SEEDS: u64 = 32;
        const TOLERANCE: f32 = 0.02;
        for n in [1 << 12, 1 << 14, 1 << 16] {
            let loops: Vec<_> = (0..SEEDS)
                .map(|seed| shaped_noise_loop(n, &RAIN, &mut NoiseStream::new(seed)))
                .collect();
            for band in RAIN {
                let mean = loops
                    .iter()
                    .map(|buf| band_energy_share(buf, band.lo_hz(), band.hi_hz()))
                    .sum::<f32>()
                    / SEEDS as f32;
                assert!(
                    (mean - band.share()).abs() < TOLERANCE,
                    "{n} samples, {band:?}: {mean}"
                );
            }
        }
    }

    #[test]
    fn a_constant_s_power_is_all_at_0_hz() {
        let dc = [0.5; 1024];
        assert_eq!(band_energy_share(&dc, 0.0, 1.0), 1.0);
        assert_eq!(centroid_hz(&dc), 0.0);
    }

    #[test]
    fn fft_round_trips() {
        let mut rng = NoiseStream::new(1);
        let orig: Vec<f32> = (0..256).map(|_| rng.norm()).collect();
        let tw = twiddles(256);
        let (re, im) = spectrum(&orig, &tw);
        for (a, b) in orig.iter().zip(&signal(&re, &im, &tw)) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    /// Every power of two, odd and even exponents alike (the radix-2² passes
    /// end on a lone radix-2 stage only for the odd ones), against the DFT's
    /// own sum.
    #[test]
    fn the_spectrum_is_the_direct_dft_at_every_length() {
        for n in (1..=10).map(|e| 1usize << e) {
            let mut rng = NoiseStream::new(n as u64);
            let x: Vec<f32> = (0..n).map(|_| rng.norm()).collect();
            let tw = twiddles(n);
            let (re, im) = spectrum(&x, &tw);
            for k in 0..=n / 2 {
                let (mut dr, mut di) = (0.0f64, 0.0f64);
                for (t, &v) in x.iter().enumerate() {
                    let a = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                    dr += f64::from(v) * a.cos();
                    di += f64::from(v) * a.sin();
                }
                let tol = 1e-3 * (n as f64).sqrt();
                assert!(
                    (f64::from(re[k]) - dr).abs() < tol && (f64::from(im[k]) - di).abs() < tol,
                    "n {n} bin {k}: ({}, {}) vs ({dr}, {di})",
                    re[k],
                    im[k]
                );
            }
            let back = signal(&re, &im, &tw);
            assert!(
                x.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-4),
                "n {n} round trip"
            );
        }
    }

    #[test]
    fn bandpass_keeps_in_band_and_kills_out_of_band() {
        let t: Vec<f32> = (0..8192)
            .map(|i| (2.0 * std::f32::consts::PI * 500.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let kept = bandpass(&t, 300.0, 800.0);
        let killed = bandpass(&t, 2000.0, 4000.0);
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        assert!(rms(&kept) > 0.5, "in-band tone survives");
        assert!(rms(&killed) < 0.01, "out-of-band tone dies");
        // a bin's own frequency at 48 kHz, so none leaks into the band below,
        // which holds where that bin falls at `SAMPLE_RATE`
        let (sr, n) = (48_000, 8192);
        let hz = 85.0 * sr as f32 / n as f32;
        let t: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / sr as f32).sin())
            .collect();
        let kept = bandpass_at(&t, hz - 20.0, hz + 20.0, sr);
        let killed = bandpass_at(&t, hz - 200.0, hz - 20.0, sr);
        assert!(rms(&kept) > 0.5, "in-band tone survives at {sr}");
        assert!(rms(&killed) < 0.01, "out-of-band tone dies at {sr}");
    }

    #[test]
    fn warp_resample_preserves_a_tone_and_is_identity_at_zero_dev() {
        let tone: Vec<f32> = (0..44_100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        // zero deviation is bit-exact identity: the read head never moves
        let same = warp_resample(&tone, &[(0.7, 0.0)]);
        assert_eq!(same, tone);
        // the shipped wow/flutter pair — a deliberately subtle warble
        let warped = warp_resample(&tone, &[(0.7, 0.0025), (8.0, 0.0006)]);
        let c = centroid_hz(&warped);
        assert!((400.0..480.0).contains(&c), "centroid {c} strayed from 440");
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        assert!(
            rms(&warped) > 0.9 * rms(&tone),
            "warble must not lose power"
        );
    }

    /// The octave edges a fingerprint reads power shares between, in Hz.
    const OCTAVE_EDGES_HZ: [f32; 10] = [
        0.0,
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
        16000.0,
        SAMPLE_RATE as f32 / 2.0,
    ];

    /// `buf`'s fingerprint: its RMS in dB, its peak, its centroid and its
    /// power's share in each octave.
    fn fingerprint(buf: &[f32]) -> Vec<f32> {
        let power = buf.iter().map(|&v| f64::from(v * v)).sum::<f64>() / buf.len() as f64;
        let peak = buf.iter().fold(0.0f32, |p, v| p.max(v.abs()));
        let shares = (OCTAVE_EDGES_HZ.windows(2)).map(|w| band_energy_share(buf, w[0], w[1]));
        [(10.0 * power.log10()) as f32, peak, centroid_hz(buf)]
            .into_iter()
            .chain(shares)
            .collect()
    }

    #[test]
    fn the_public_filters_keep_their_fingerprints() {
        // over a platform's last-bit `sin`, and under any change pixtuoid's
        // foley, tuned on these outputs, could hear (opinion)
        const TOLERANCE: f32 = 1e-4;
        // a keystroke's length, not a power of two, so the padding shows
        let mut rng = NoiseStream::new(11);
        let noise: Vec<f32> = (0..2646).map(|_| rng.norm()).collect();
        // at the length pixtuoid's rain loops at
        let mut drops = NoiseStream::new(5);
        let wash = shaped_noise_loop(1 << 19, &RAIN, &mut drops);
        // RMS, peak and centroid, then a share per octave
        type Fingerprint = [f32; 3 + OCTAVE_EDGES_HZ.len() - 1];
        let cases: [(&str, Vec<f32>, usize, Fingerprint); 3] = [
            (
                "bandpass",
                bandpass(&noise, 900.0, 3000.0),
                noise.len(),
                [
                    -10.244952,
                    1.0600876,
                    1877.087,
                    0.00018664123,
                    0.00019315626,
                    0.00044841098,
                    0.06135948,
                    0.49538866,
                    0.44221273,
                    0.00014493613,
                    4.79502e-5,
                    1.800845e-5,
                ],
            ),
            (
                "lowpass",
                lowpass(&noise, 4000.0),
                noise.len(),
                [
                    -7.590567,
                    1.2667441,
                    1829.902,
                    0.03587059,
                    0.04643017,
                    0.056849808,
                    0.15276156,
                    0.26877767,
                    0.43810514,
                    0.0011964117,
                    8.134562e-6,
                    5.2381813e-7,
                ],
            ),
            (
                "shaped_noise_loop",
                wash,
                1 << 19,
                [
                    -13.252275,
                    1.0,
                    1975.7383,
                    0.050717503,
                    0.06665233,
                    0.14755078,
                    0.24297257,
                    0.2585963,
                    0.10297062,
                    0.09013539,
                    0.04040117,
                    3.3380925e-6,
                ],
            ),
        ];
        let mut off = Vec::new();
        for (name, buf, len, want) in cases {
            let got = fingerprint(&buf);
            let strays = (got.iter().zip(want)).any(|(g, w)| (g - w).abs() > TOLERANCE * w.abs());
            if buf.len() != len || strays {
                off.push(format!("{name}: {} samples, {got:?}", buf.len()));
            }
        }
        // the draws it took, which every later draw of pixtuoid's stream follows
        let next = drops.unit();
        if next != 0.172_583_52 {
            off.push(format!("shaped_noise_loop's stream draws {next} next"));
        }
        assert!(off.is_empty(), "off their fingerprints: {off:#?}");
    }
}
