//! The lane chains' filters, Butterworth slopes and the cookbook's band-pass:
//! causal, so nothing rings before an onset. `dsp`'s public brickwalls keep
//! their output: pixtuoid's foley hears it.

use std::f64::consts::{LN_2, PI, TAU};

/// One second-order section over `a0 = 1`, with its Direct Form 1 state.
#[derive(Debug)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    /// A section from the [RBJ cookbook](https://www.w3.org/TR/audio-eq-cookbook/)'s
    /// `b0 b1 b2 a0 a1 a2`, at rest. Its filters are "derived from analog
    /// prototypes" and "digitized using the Bilinear Transform", prewarped so
    /// the given frequency stays put.
    fn cookbook([b0, b1, b2, a0, a1, a2]: [f64; 6]) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// The cookbook's LPF at `hz` at `sr`.
    fn lowpass(hz: f32, q: f64, sr: u32) -> Self {
        let w0 = TAU * f64::from(hz) / f64::from(sr);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let b0 = (1.0 - cos) / 2.0;
        Self::cookbook([b0, 1.0 - cos, b0, 1.0 + alpha, -2.0 * cos, 1.0 - alpha])
    }

    /// The state is f64: a 60 Hz section's poles sit near enough the unit
    /// circle to amplify f32 rounding in the feedback.
    fn process(&mut self, buf: &mut [f64]) {
        let (mut x1, mut x2, mut y1, mut y2) = (self.x1, self.x2, self.y1, self.y2);
        for v in buf {
            let x0 = *v;
            *v = super::flush(
                self.b0 * x0 + self.b1 * x1 + self.b2 * x2 - self.a1 * y1 - self.a2 * y2,
            );
            (x2, x1, y2, y1) = (x1, x0, y1, *v);
        }
        (self.x1, self.x2, self.y1, self.y2) = (x1, x2, y1, y2);
    }

    fn at_rest(&self) -> bool {
        [self.x1, self.x2, self.y1, self.y2] == [0.0; 4]
    }
}

/// Second-order sections run one after another.
#[derive(Debug)]
pub(crate) struct Cascade(Vec<Biquad>);

impl Cascade {
    pub(crate) fn process(&mut self, buf: &mut [f64]) {
        for s in &mut self.0 {
            s.process(buf);
        }
    }

    /// Whether its state is all zero, so silence in plays silence out.
    pub(crate) fn at_rest(&self) -> bool {
        self.0.iter().all(Biquad::at_rest)
    }
}

/// A Butterworth low-pass of even `order` at `hz` at `sr`: one section per
/// quadratic factor of the normalized Butterworth polynomial,
/// `s² − 2s·cos((2k + n − 1)π / 2n) + 1` for `k` in `1..=n/2`
/// (<https://en.wikipedia.org/wiki/Butterworth_filter>).
pub(crate) fn butter_lowpass(hz: f32, order: usize, sr: u32) -> Cascade {
    debug_assert!(
        order.is_multiple_of(2),
        "an odd order needs a first-order section"
    );
    let n = order as f64;
    Cascade(
        (1..=order / 2)
            .map(|k| {
                let q = -1.0 / (2.0 * ((2.0 * k as f64 + n - 1.0) * PI / (2.0 * n)).cos());
                Biquad::lowpass(hz, q, sr)
            })
            .collect(),
    )
}

/// The cookbook's BPF with "constant 0 dB peak gain", centered between
/// `lo_hz` and `hi_hz` on a log scale, with "the bandwidth in octaves (between
/// -3 dB frequencies)" between them. The bilinear image of
/// `(s/Q) / (s² + s/Q + 1)`, whose real part is never negative, it peaks at
/// exactly 1 and, added to its input, cancels it nowhere.
pub(crate) fn cookbook_bandpass(lo_hz: f32, hi_hz: f32, sr: u32) -> Cascade {
    let (lo, hi) = (f64::from(lo_hz), f64::from(hi_hz));
    let w0 = TAU * (lo * hi).sqrt() / f64::from(sr);
    let (sin, cos) = w0.sin_cos();
    let alpha = sin * (LN_2 / 2.0 * (hi / lo).log2() * w0 / sin).sinh();
    Cascade(vec![Biquad::cookbook([
        alpha,
        0.0,
        -alpha,
        1.0 + alpha,
        -2.0 * cos,
        1.0 - alpha,
    ])])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::SAMPLE_RATE;

    /// The rate lofi renders at, and another, so a filter that reads
    /// [`SAMPLE_RATE`] instead of its own rate misses one of them.
    const RATES: [u32; 2] = [SAMPLE_RATE, 48_000];

    /// `H(e^{jω})` of `cascade`, run at `sr`, at `hz`, as `(re, im)`.
    fn response(cascade: &Cascade, hz: f64, sr: u32) -> (f64, f64) {
        let w = TAU * hz / f64::from(sr);
        let (c1, s1, c2, s2) = (w.cos(), -w.sin(), (2.0 * w).cos(), -(2.0 * w).sin());
        cascade.0.iter().fold((1.0, 0.0), |(re, im), s| {
            let (nr, ni) = (s.b0 + s.b1 * c1 + s.b2 * c2, s.b1 * s1 + s.b2 * s2);
            let (dr, di) = (1.0 + s.a1 * c1 + s.a2 * c2, s.a1 * s1 + s.a2 * s2);
            let d = dr * dr + di * di;
            let (hr, hi) = ((nr * dr + ni * di) / d, (ni * dr - nr * di) / d);
            (re * hr - im * hi, re * hi + im * hr)
        })
    }

    /// `|H(e^{jω})|²` of `cascade`, run at `sr`, at `hz`.
    fn power_gain(cascade: &Cascade, hz: f64, sr: u32) -> f64 {
        let (re, im) = response(cascade, hz, sr);
        re * re + im * im
    }

    #[test]
    fn a_low_pass_has_its_butterworth_prototypes_magnitude() {
        // the bilinear transform maps the analog prototype's frequency to
        // tan(ω/2), prewarped to the corner, so the digital magnitude is the
        // analog Butterworth one in that variable
        let corner = 6500.0;
        for (sr, order) in RATES.into_iter().flat_map(|sr| [(sr, 2), (sr, 4)]) {
            let warped = |hz: f64| (PI * hz / f64::from(sr)).tan();
            let n = order as i32;
            let cascade = butter_lowpass(corner as f32, order, sr);
            for hz in [20.0, 60.0, 90.0, 1000.0, 6500.0, 12000.0] {
                let want = 1.0 / (1.0 + (warped(hz) / warped(corner)).powi(2 * n));
                let got = power_gain(&cascade, hz, sr);
                assert!(
                    (got / want - 1.0).abs() < 1e-6,
                    "order {order} at {sr}, {hz} Hz: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn a_band_pass_peaks_at_one_halves_at_its_edges_and_never_opposes() {
        let (lo, hi) = (60.0, 120.0);
        for sr in RATES {
            let band = cookbook_bandpass(lo as f32, hi as f32, sr);
            let center = power_gain(&band, f64::sqrt(lo * hi), sr);
            assert!((center - 1.0).abs() < 1e-6, "{sr} center: {center}");
            for edge in [lo, hi] {
                let got = power_gain(&band, edge, sr);
                assert!((got - 0.5).abs() < 1e-4, "{sr}, {edge} Hz: {got}");
            }
            for hz in (1..2200).map(|k| f64::from(k) * 10.0) {
                let (re, _) = response(&band, hz, sr);
                assert!(re >= 0.0, "{sr}, {hz} Hz: real part {re}");
            }
        }
    }

    #[test]
    fn processing_a_sine_scales_its_power_by_the_magnitude() {
        // a whole second holds whole cycles of an integer-Hz sine, so its mean
        // square is exactly half its squared amplitude
        let sr = SAMPLE_RATE as usize;
        for (mut cascade, hz) in [
            (cookbook_bandpass(60.0, 120.0, SAMPLE_RATE), 30.0),
            (cookbook_bandpass(60.0, 120.0, SAMPLE_RATE), 90.0),
            (butter_lowpass(6500.0, 4, SAMPLE_RATE), 1000.0),
            (butter_lowpass(6500.0, 4, SAMPLE_RATE), 12000.0),
        ] {
            let want = power_gain(&cascade, hz, SAMPLE_RATE) / 2.0;
            let mut buf: Vec<f64> = (0..2 * sr)
                .map(|i| (TAU * hz * i as f64 / f64::from(SAMPLE_RATE)).sin())
                .collect();
            cascade.process(&mut buf);
            // the first second settles the transient
            let got = buf[sr..].iter().map(|v| v * v).sum::<f64>() / sr as f64;
            assert!((got / want - 1.0).abs() < 1e-6, "{hz} Hz: {got} vs {want}");
        }
    }

    #[test]
    fn silence_settles_a_cascade_to_exact_zero() {
        // long enough for the head bump, the chain's slowest section, to reach
        // the subnormal range
        let mut cascade = cookbook_bandpass(60.0, 120.0, SAMPLE_RATE);
        let mut burst = crate::stream::burst_then_silence(10);
        let (first, rest) = burst.split_at_mut(1);
        cascade.process(first);
        assert!(!cascade.at_rest());
        cascade.process(rest);
        assert!(cascade.at_rest());
    }

    #[test]
    fn a_cascade_is_chunk_invariant() {
        // the chain's slowest section, the head bump, and its steepest slope
        crate::stream::assert_chunk_invariant(
            || cookbook_bandpass(60.0, 120.0, SAMPLE_RATE),
            Cascade::process,
        );
        crate::stream::assert_chunk_invariant(
            || butter_lowpass(6500.0, 4, SAMPLE_RATE),
            Cascade::process,
        );
    }
}
