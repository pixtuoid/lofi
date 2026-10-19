//! The one-pole low-pass, and the tilt it makes, which brightens a lane's
//! top.

use std::f64::consts::TAU;

/// A one-pole low-pass at a corner, `y = (1 − a)·x + a·y₋₁`.
#[derive(Debug)]
pub(crate) struct OnePole {
    pole: f64,
    y: f64,
}

impl OnePole {
    pub(crate) fn new(corner_hz: f32, sr: u32) -> Self {
        Self {
            // the analog pole at −2π·corner mapped by impulse invariance,
            // z = e^{sT} (<https://ccrma.stanford.edu/~jos/pasp/Impulse_Invariant_Method.html>)
            pole: (-TAU * f64::from(corner_hz) / f64::from(sr)).exp(),
            y: 0.0,
        }
    }

    /// The next output, `x` in.
    pub(crate) fn lowpass(&mut self, x: f64) -> f64 {
        self.y = super::flush(self.pole * self.y + (1.0 - self.pole) * x);
        self.y
    }

    /// Whether its state is zero, so silence in plays silence out.
    pub(crate) fn at_rest(&self) -> bool {
        self.y == 0.0
    }
}

/// A one-pole high shelf, `x + lift·(x − lp(x))`, `lp` a one-pole low-pass at
/// the corner.
#[derive(Debug)]
pub(crate) struct Tilt {
    lp: OnePole,
    lift: f64,
}

impl Tilt {
    pub(crate) fn new(corner_hz: f32, lift: f64, sr: u32) -> Self {
        Self {
            lp: OnePole::new(corner_hz, sr),
            lift,
        }
    }

    pub(crate) fn process(&mut self, buf: &mut [f64]) {
        let lift = self.lift;
        self.process_lifted(buf, |_| lift);
    }

    /// [`process`](Self::process) with the lift on `buf[i]` at `lift(i)`: the
    /// lift is memoryless, so it may move per sample.
    pub(crate) fn process_lifted(&mut self, buf: &mut [f64], lift: impl Fn(usize) -> f64) {
        for (i, x) in buf.iter_mut().enumerate() {
            let lp = self.lp.lowpass(*x);
            *x += lift(i) * (*x - lp);
        }
    }

    /// Whether its state is zero, so silence in plays silence out.
    pub(crate) fn at_rest(&self) -> bool {
        self.lp.at_rest()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::SAMPLE_RATE;

    #[test]
    fn a_one_pole_settles_a_step_as_its_closed_form() {
        // y[n] = (1 − a)·Σ aᵏ = 1 − aⁿ⁺¹
        for sr in [SAMPLE_RATE, 48_000] {
            let mut lp = OnePole::new(5000.0, sr);
            let a = (-TAU * 5000.0 / f64::from(sr)).exp();
            for n in 0..200 {
                let (got, want) = (lp.lowpass(1.0), 1.0 - a.powi(n + 1));
                assert!((got - want).abs() < 1e-12, "{sr} at {n}: {got} vs {want}");
            }
            assert!(!lp.at_rest());
        }
    }

    #[test]
    fn a_tilt_has_its_closed_form_step_response() {
        // its low-pass settles a unit step as 1 − aⁿ⁺¹, so the shelf's output
        // is 1 + lift·aⁿ⁺¹
        let (corner, lift) = (1200.0, 0.8);
        // a second rate, so a tilt that reads `SAMPLE_RATE` instead of its
        // own misses one
        for sr in [SAMPLE_RATE, 48_000] {
            let a = (-TAU * f64::from(corner) / f64::from(sr)).exp();
            let mut buf = vec![1.0; 2000];
            Tilt::new(corner, lift, sr).process(&mut buf);
            for (n, &got) in buf.iter().enumerate() {
                let want = 1.0 + lift * a.powi(n as i32 + 1);
                assert!((got - want).abs() < 1e-12, "{sr} at {n}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn silence_settles_a_tilt_to_exact_zero() {
        let mut tilt = Tilt::new(1200.0, 0.8, SAMPLE_RATE);
        let mut burst = crate::stream::burst_then_silence(1);
        let (first, rest) = burst.split_at_mut(1);
        tilt.process(first);
        assert!(!tilt.at_rest());
        tilt.process(rest);
        assert!(tilt.at_rest());
    }

    #[test]
    fn a_tilt_is_chunk_invariant() {
        crate::stream::assert_chunk_invariant(
            1,
            || Tilt::new(1200.0, 0.8, SAMPLE_RATE),
            Tilt::process,
        );
    }
}
