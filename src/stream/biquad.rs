//! Butterworth slopes for the lane chains: causal, so nothing rings before an
//! onset. `dsp`'s public brickwalls keep their output: pixtuoid's foley hears it.

use std::f64::consts::{PI, TAU};

use crate::dsp::SAMPLE_RATE;

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

#[derive(Clone, Copy)]
enum Pass {
    Low,
    High,
}

impl Biquad {
    /// The [RBJ cookbook](https://www.w3.org/TR/audio-eq-cookbook/)'s LPF or
    /// HPF at `hz`: "derived from analog prototypes" and "digitized using the
    /// Bilinear Transform", prewarped so the corner stays at `hz`.
    fn rbj(pass: Pass, hz: f32, q: f64) -> Self {
        let w0 = TAU * f64::from(hz) / f64::from(SAMPLE_RATE);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let (b0, b1) = match pass {
            Pass::Low => ((1.0 - cos) / 2.0, 1.0 - cos),
            Pass::High => (f64::midpoint(1.0, cos), -(1.0 + cos)),
        };
        let a0 = 1.0 + alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b0 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
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
}

/// A Butterworth slope of even `order` at `hz`: one section per quadratic
/// factor of the normalized Butterworth polynomial,
/// `s² − 2s·cos((2k + n − 1)π / 2n) + 1` for `k` in `1..=n/2`
/// (<https://en.wikipedia.org/wiki/Butterworth_filter>).
fn butterworth(pass: Pass, hz: f32, order: usize) -> impl Iterator<Item = Biquad> {
    debug_assert!(
        order.is_multiple_of(2),
        "an odd order needs a first-order section"
    );
    let n = order as f64;
    (1..=order / 2).map(move |k| {
        let q = -1.0 / (2.0 * ((2.0 * k as f64 + n - 1.0) * PI / (2.0 * n)).cos());
        Biquad::rbj(pass, hz, q)
    })
}

pub(crate) fn butter_lowpass(hz: f32, order: usize) -> Cascade {
    Cascade(butterworth(Pass::Low, hz, order).collect())
}

/// A high-pass at `lo_hz` into a low-pass at `hi_hz`, each of `order`.
pub(crate) fn butter_bandpass(lo_hz: f32, hi_hz: f32, order: usize) -> Cascade {
    Cascade(
        butterworth(Pass::High, lo_hz, order)
            .chain(butterworth(Pass::Low, hi_hz, order))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `|H(e^{jω})|²` of `cascade` at `hz`.
    fn power_gain(cascade: &Cascade, hz: f64) -> f64 {
        let w = TAU * hz / f64::from(SAMPLE_RATE);
        let (c1, s1, c2, s2) = (w.cos(), -w.sin(), (2.0 * w).cos(), -(2.0 * w).sin());
        cascade.0.iter().fold(1.0, |g, s| {
            let num = (s.b0 + s.b1 * c1 + s.b2 * c2).powi(2) + (s.b1 * s1 + s.b2 * s2).powi(2);
            let den = (1.0 + s.a1 * c1 + s.a2 * c2).powi(2) + (s.a1 * s1 + s.a2 * s2).powi(2);
            g * num / den
        })
    }

    fn butter_highpass(hz: f32, order: usize) -> Cascade {
        Cascade(butterworth(Pass::High, hz, order).collect())
    }

    #[test]
    fn a_cascade_has_its_butterworth_prototypes_magnitude() {
        // the bilinear transform maps the analog prototype's frequency to
        // tan(ω/2), prewarped to the corner, so the digital magnitude is the
        // analog Butterworth one in that variable
        let warped = |hz: f64| (PI * hz / f64::from(SAMPLE_RATE)).tan();
        let (lo, hi) = (60.0, 6500.0);
        for order in [2, 4] {
            let n = order as i32;
            for hz in [20.0, 60.0, 90.0, 1000.0, 6500.0, 12000.0] {
                let low = 1.0 / (1.0 + (warped(hz) / warped(hi)).powi(2 * n));
                let high = 1.0 / (1.0 + (warped(lo) / warped(hz)).powi(2 * n));
                for (name, got, want) in [
                    (
                        "low",
                        power_gain(&butter_lowpass(hi as f32, order), hz),
                        low,
                    ),
                    (
                        "high",
                        power_gain(&butter_highpass(lo as f32, order), hz),
                        high,
                    ),
                    (
                        "band",
                        power_gain(&butter_bandpass(lo as f32, hi as f32, order), hz),
                        low * high,
                    ),
                ] {
                    assert!(
                        (got / want - 1.0).abs() < 1e-6,
                        "order {order} {name} at {hz} Hz: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn processing_a_sine_scales_its_power_by_the_magnitude() {
        // a whole second holds whole cycles of an integer-Hz sine, so its mean
        // square is exactly half its squared amplitude
        let sr = SAMPLE_RATE as usize;
        for (mut cascade, hz) in [
            (butter_bandpass(60.0, 120.0, 4), 30.0),
            (butter_bandpass(60.0, 120.0, 4), 90.0),
            (butter_lowpass(6500.0, 4), 1000.0),
            (butter_lowpass(6500.0, 4), 12000.0),
        ] {
            let want = power_gain(&cascade, hz) / 2.0;
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
        // long enough for the head bump's slowest section to reach the subnormal range
        let mut cascade = butter_bandpass(60.0, 120.0, 4);
        cascade.process(&mut crate::stream::burst_then_silence(10));
        for s in &cascade.0 {
            assert_eq!([s.x1, s.x2, s.y1, s.y2], [0.0; 4]);
        }
    }

    #[test]
    fn a_cascade_is_chunk_invariant() {
        // the slowest sections in the chain, the head bump's
        crate::stream::assert_chunk_invariant(|| butter_bandpass(60.0, 120.0, 4), Cascade::process);
    }
}
