//! The texture's tape hiss: noise under a linear-phase low-pass. Hiss is
//! stationary noise, so the kernel's symmetric pre-ringing has no transient
//! to expose, and only an FIR reaches the whole-lane chain's brickwall. Its
//! noise is drawn ahead in blocks of its own, so no chunking of the render
//! moves a draw.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the leak proof plays the hiss through the cascade instead"
    )
)]

use std::f64::consts::{PI, TAU};

use crate::dsp::{NoiseStream, Twiddles, fft, twiddles, white_noise_gain};

/// The low-pass's stopband, in dB under its passband, and its transition's
/// width in Hz, centred on the corner: the controller's proposal (opinion),
/// which the hiss test holds against the whole-lane chain's brickwall.
const STOPBAND_DB: f64 = 80.0;
const TRANSITION_HZ: f64 = 200.0;

/// Kaiser's attenuation for an order, `a = SLOPE·(N − 1)·π·width + OFFSET`
/// with the width a fraction of Nyquist, as SciPy's `kaiser_atten` gives it
/// (<https://github.com/scipy/scipy/blob/v1.18.0/scipy/signal/_fir_filter_design.py#L76-L116>).
const KAISER_ORDER_SLOPE: f64 = 2.285;
const KAISER_ORDER_OFFSET_DB: f64 = 7.95;
/// Kaiser's β for a stopband over [`KAISER_BETA_HIGH_DB`],
/// `β = SLOPE·(a − OFFSET)`, the branch of SciPy's `kaiser_beta` that
/// [`kernel`] takes
/// (<https://github.com/scipy/scipy/blob/v1.18.0/scipy/signal/_fir_filter_design.py#L38-L73>).
const KAISER_BETA_SLOPE: f64 = 0.1102;
const KAISER_BETA_OFFSET_DB: f64 = 8.7;
const KAISER_BETA_HIGH_DB: f64 = 50.0;
const _: () = assert!(STOPBAND_DB > KAISER_BETA_HIGH_DB);

/// A transform is at least this many kernels long, so most of it is output
/// rather than the kernel's memory (opinion).
const KERNELS_PER_TRANSFORM: usize = 4;

/// I₀, the zeroth-order modified Bessel function of the first kind, by its
/// series Σ ((x/2)ᵏ / k!)² (eq. 3.12 of
/// <https://ccrma.stanford.edu/~jos/sasp/Kaiser_Window.html>), summed until
/// a term no longer moves the sum.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 0.0);
    loop {
        k += 1.0;
        term *= (x / (2.0 * k)).powi(2);
        if sum + term == sum {
            return sum;
        }
        sum += term;
    }
}

/// The taps of a low-pass at `corner_hz` at `sr`, [`STOPBAND_DB`] down past
/// [`TRANSITION_HZ`]: Eq. 16-4's windowed sinc, whose "cutoff frequency ...
/// is measured at the one-half amplitude point"
/// (<https://www.dspguide.com/ch16.htm>), under the Kaiser window
/// (<https://ccrma.stanford.edu/~jos/sasp/Kaiser_Window.html>) in place of
/// its Blackman one. The taps sum to 1, unity gain at DC, which absorbs the
/// window's `1/I₀(β)`. The order solves [`KAISER_ORDER_SLOPE`]'s formula
/// for [`STOPBAND_DB`], rounded up to even so the kernel has a centre tap.
pub(crate) fn kernel(corner_hz: f32, sr: u32) -> Vec<f64> {
    let rate = f64::from(sr);
    let width = TRANSITION_HZ / (rate / 2.0);
    let order = ((STOPBAND_DB - KAISER_ORDER_OFFSET_DB) / (KAISER_ORDER_SLOPE * PI * width)).ceil()
        as usize;
    let order = order + order % 2;
    let beta = KAISER_BETA_SLOPE * (STOPBAND_DB - KAISER_BETA_OFFSET_DB);
    let (fc, half) = (f64::from(corner_hz) / rate, order as f64 / 2.0);
    let taps: Vec<f64> = (0..=order)
        .map(|i| {
            let x = i as f64 - half;
            let sinc = if x == 0.0 {
                TAU * fc
            } else {
                (TAU * fc * x).sin() / x
            };
            sinc * bessel_i0(beta * (1.0 - (x / half).powi(2)).sqrt())
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    taps.into_iter().map(|t| t / sum).collect()
}

/// Noise under [`kernel`], from a song's first frame on: overlap–save over the
/// crate's FFT, convolving "by multiplying their frequency spectra"
/// (<https://www.dspguide.com/ch18.htm>), where of each segment's circular
/// convolution "the subregion [M + 1, L + M] is appended to the output
/// stream, and the other values are discarded"
/// (<https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method>).
#[derive(Debug)]
pub(crate) struct Hiss {
    /// The kernel's spectrum over a transform, real then imaginary, with the
    /// inverse's 1/N and the rate's [`white_noise_gain`] in it.
    kernel: [Vec<f32>; 2],
    twiddles: Twiddles,
    /// The noise the next two segments read: the kernel's memory, then
    /// their draws.
    input: Vec<f32>,
    /// A transform's real and imaginary parts.
    re: Vec<f32>,
    im: Vec<f32>,
    /// Hiss filtered ahead, two segments of it, and how much has played.
    ahead: Vec<f32>,
    played: usize,
    /// Whether `input` holds the kernel's memory yet.
    primed: bool,
}

impl Hiss {
    /// Hiss under a low-pass at `corner_hz` at `sr`.
    pub(crate) fn new(corner_hz: f32, sr: u32) -> Self {
        let taps = kernel(corner_hz, sr);
        let memory = taps.len() - 1;
        let n = (KERNELS_PER_TRANSFORM * taps.len()).next_power_of_two();
        let twiddles = twiddles(n);
        let (mut re, mut im) = (vec![0.0f32; n], vec![0.0f32; n]);
        for (r, &t) in re.iter_mut().zip(&taps) {
            *r = t as f32;
        }
        fft(&mut re, &mut im, &twiddles, false);
        let scale = white_noise_gain(sr) / n as f32;
        for v in re.iter_mut().chain(&mut im) {
            *v *= scale;
        }
        let segment = n - memory;
        Self {
            kernel: [re, im],
            twiddles,
            input: vec![0.0; memory + 2 * segment],
            re: vec![0.0; n],
            im: vec![0.0; n],
            ahead: vec![0.0; 2 * segment],
            played: 2 * segment,
            primed: false,
        }
    }

    /// Fills `out` with the next hiss, its noise drawn from `noise`.
    pub(crate) fn fill(&mut self, out: &mut [f64], noise: &mut NoiseStream) {
        let mut done = 0;
        while done < out.len() {
            if self.played == self.ahead.len() {
                self.filter_ahead(noise);
            }
            let n = (out.len() - done).min(self.ahead.len() - self.played);
            for (o, &h) in out[done..done + n]
                .iter_mut()
                .zip(&self.ahead[self.played..])
            {
                *o = f64::from(h);
            }
            (done, self.played) = (done + n, self.played + n);
        }
    }

    /// Draws the next two segments' noise and filters them, the first in a
    /// transform's real part and the second in its imaginary part: the
    /// kernel is real, so it convolves them apart.
    fn filter_ahead(&mut self, noise: &mut NoiseStream) {
        let n = self.re.len();
        let segment = self.ahead.len() / 2;
        let memory = n - segment;
        // the song's first segments read a kernel's worth of earlier noise,
        // so its first frame plays at steady state, not out of silence
        let fresh = if self.primed {
            self.input.copy_within(2 * segment.., 0);
            memory
        } else {
            0
        };
        self.primed = true;
        for x in &mut self.input[fresh..] {
            *x = noise.norm();
        }
        self.re.copy_from_slice(&self.input[..n]);
        self.im.copy_from_slice(&self.input[segment..segment + n]);
        fft(&mut self.re, &mut self.im, &self.twiddles, false);
        let [kr, ki] = &self.kernel;
        for (((r, i), &hr), &hi) in self.re.iter_mut().zip(&mut self.im).zip(kr).zip(ki) {
            (*r, *i) = (*r * hr - *i * hi, *r * hi + *i * hr);
        }
        fft(&mut self.re, &mut self.im, &self.twiddles, true);
        self.ahead[..segment].copy_from_slice(&self.re[memory..]);
        self.ahead[segment..].copy_from_slice(&self.im[memory..]);
        self.played = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{SAMPLE_RATE, band_energy_share, lowpass};
    use crate::synth::HISS_HZ;

    /// The centre of the third octave `k` steps from 1 kHz's, `1000·2^(k/3)` Hz.
    fn centre(k: i32) -> f32 {
        1000.0 * 2f32.powf(k as f32 / 3.0)
    }

    /// `buf`'s level in the third octave at `k`, in dB over its 1 kHz one's,
    /// under a Hann window: a rectangular one leaks the passband's edges into
    /// the stopband far above [`STOPBAND_DB`].
    fn levels(buf: &[f64], ks: &[i32]) -> Vec<f64> {
        let n = buf.len() as f64;
        let windowed: Vec<f32> = (buf.iter().enumerate())
            .map(|(i, &v)| (v * (0.5 - 0.5 * (TAU * i as f64 / n).cos())) as f32)
            .collect();
        let edge = 2f32.powf(1.0 / 6.0);
        let share = |k| {
            f64::from(band_energy_share(
                &windowed,
                centre(k) / edge,
                centre(k) * edge,
            ))
        };
        let reference = share(0);
        ks.iter()
            .map(|&k| 10.0 * (share(k) / reference).log10())
            .collect()
    }

    #[test]
    fn the_hiss_keeps_the_whole_lane_passband_and_is_its_stopband_down_past_it() {
        // the controller's proposal (opinion): the passband through the
        // corner's band within this of the whole-lane brickwall, and every
        // band past it the stopband down, less what a band's edges leak
        const PASSBAND_DB: f64 = 0.5;
        const EDGE_LEAKAGE_DB: f64 = 3.0;
        let (sr, n, seed) = (SAMPLE_RATE, 1 << 20, 17);
        let mut stream = vec![0.0; n];
        Hiss::new(HISS_HZ, sr).fill(&mut stream, &mut NoiseStream::new(seed));
        // the same draws, from as far on as the kernel's centre lags its
        // newest
        let mut noise = NoiseStream::new(seed);
        for _ in 0..kernel(HISS_HZ, sr).len() / 2 {
            noise.norm();
        }
        let raw: Vec<f32> = (0..n).map(|_| noise.norm()).collect();
        let whole: Vec<f64> = lowpass(&raw, HISS_HZ).into_iter().map(f64::from).collect();
        let edge = 2f32.powf(1.0 / 6.0);
        let corner = (-14..=12)
            .find(|&k| (centre(k) / edge..centre(k) * edge).contains(&HISS_HZ))
            .expect("a third octave holds the corner");
        let ks: Vec<i32> = (-14..=12).collect();
        let (old, new) = (levels(&whole, &ks), levels(&stream, &ks));
        let mut off = Vec::new();
        for ((&k, o), v) in ks.iter().zip(&old).zip(&new) {
            let hz = centre(k).round();
            if k <= corner && (v - o).abs() > PASSBAND_DB {
                off.push(format!("{hz} Hz: {v:+.2} vs the whole lane's {o:+.2} dB"));
            }
            if k > corner && *v > EDGE_LEAKAGE_DB - STOPBAND_DB {
                off.push(format!("{hz} Hz: {v:+.2} dB"));
            }
        }
        assert!(off.is_empty(), "bands re 1 kHz: {off:#?}");
    }

    #[test]
    fn the_hiss_is_its_kernels_convolution_of_its_draws_from_the_first_frame() {
        // f32's transform rounding, far under the hiss's own level (opinion)
        const TOLERANCE: f64 = 1e-4;
        let sr = SAMPLE_RATE;
        let taps = kernel(HISS_HZ, sr);
        let memory = taps.len() - 1;
        let mut hiss = Hiss::new(HISS_HZ, sr);
        let segment = hiss.ahead.len() / 2;
        // through both segments of one transform and into the next's
        let frames = 3 * segment;
        let mut out = vec![0.0; frames];
        hiss.fill(&mut out, &mut NoiseStream::new(4));
        let mut noise = NoiseStream::new(4);
        let draws: Vec<f64> = (0..memory + frames)
            .map(|_| f64::from(noise.norm()))
            .collect();
        let off: Vec<String> = [
            0,
            1,
            segment - 1,
            segment,
            2 * segment - 1,
            2 * segment,
            frames - 1,
        ]
        .into_iter()
        .filter_map(|j| {
            let want: f64 = (taps.iter().enumerate())
                .map(|(k, h)| h * draws[memory + j - k])
                .sum();
            ((out[j] - want).abs() > TOLERANCE).then(|| format!("frame {j}: {} vs {want}", out[j]))
        })
        .collect();
        assert!(
            off.is_empty(),
            "the hiss strays from its convolution: {off:#?}"
        );
    }

    #[test]
    fn the_hiss_is_chunk_invariant() {
        crate::stream::assert_chunk_invariant(
            || (Hiss::new(HISS_HZ, SAMPLE_RATE), NoiseStream::new(3)),
            |(hiss, noise), buf| hiss.fill(buf, noise),
        );
    }
}
