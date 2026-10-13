//! The one-pole tilt that brightens a lane's top.

use std::f64::consts::TAU;

use crate::dsp::SAMPLE_RATE;

/// A one-pole high shelf, `x + lift·(x − lp(x))`, `lp` a one-pole low-pass at
/// the corner.
#[derive(Debug)]
pub(crate) struct Tilt {
    pole: f64,
    lift: f64,
    lp: f64,
}

impl Tilt {
    pub(crate) fn new(corner_hz: f32, lift: f64) -> Self {
        Self {
            // the analog pole at −2π·corner mapped by impulse invariance,
            // z = e^{sT} (<https://ccrma.stanford.edu/~jos/pasp/Impulse_Invariant_Method.html>)
            pole: (-TAU * f64::from(corner_hz) / f64::from(SAMPLE_RATE)).exp(),
            lift,
            lp: 0.0,
        }
    }

    pub(crate) fn process(&mut self, buf: &mut [f64]) {
        let mut lp = self.lp;
        for x in buf {
            lp = self.pole * lp + (1.0 - self.pole) * *x;
            *x += self.lift * (*x - lp);
        }
        self.lp = lp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tilt_has_its_closed_form_step_response() {
        // its low-pass settles a unit step as 1 − aⁿ⁺¹, so the shelf's output
        // is 1 + lift·aⁿ⁺¹
        let (corner, lift) = (1200.0, 0.8);
        let a = (-TAU * f64::from(corner) / f64::from(SAMPLE_RATE)).exp();
        let mut buf = vec![1.0; 2000];
        Tilt::new(corner, lift).process(&mut buf);
        for (n, &got) in buf.iter().enumerate() {
            let want = 1.0 + lift * a.powi(n as i32 + 1);
            assert!((got - want).abs() < 1e-12, "at {n}: {got} vs {want}");
        }
    }

    #[test]
    fn a_tilt_is_chunk_invariant() {
        crate::stream::assert_chunk_invariant(|| Tilt::new(1200.0, 0.8), Tilt::process);
    }
}
