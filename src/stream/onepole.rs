//! The one-pole tilt that brightens a lane's top.

use std::f64::consts::TAU;

/// A one-pole high shelf, `x + lift·(x − lp(x))`, `lp` a one-pole low-pass at
/// the corner.
#[derive(Debug)]
pub(crate) struct Tilt {
    pole: f64,
    pub(crate) lift: f64,
    lp: f64,
}

impl Tilt {
    pub(crate) fn new(corner_hz: f32, lift: f64, sr: u32) -> Self {
        Self {
            // the analog pole at −2π·corner mapped by impulse invariance,
            // z = e^{sT} (<https://ccrma.stanford.edu/~jos/pasp/Impulse_Invariant_Method.html>)
            pole: (-TAU * f64::from(corner_hz) / f64::from(sr)).exp(),
            lift,
            lp: 0.0,
        }
    }

    pub(crate) fn process(&mut self, buf: &mut [f64]) {
        let mut lp = self.lp;
        for x in buf {
            lp = super::flush(self.pole * lp + (1.0 - self.pole) * *x);
            *x += self.lift * (*x - lp);
        }
        self.lp = lp;
    }

    /// [`process`](Self::process) with the lift on `buf[i]` at `lift(i)`: the
    /// lift is memoryless, so it may move per sample.
    pub(crate) fn process_lifted(&mut self, buf: &mut [f64], lift: impl Fn(usize) -> f64) {
        let mut lp = self.lp;
        for (i, x) in buf.iter_mut().enumerate() {
            lp = super::flush(self.pole * lp + (1.0 - self.pole) * *x);
            *x += lift(i) * (*x - lp);
        }
        self.lp = lp;
    }

    /// Whether its state is zero, so silence in plays silence out.
    pub(crate) fn at_rest(&self) -> bool {
        self.lp == 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::SAMPLE_RATE;

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
            || Tilt::new(1200.0, 0.8, SAMPLE_RATE),
            Tilt::process,
        );
    }
}
