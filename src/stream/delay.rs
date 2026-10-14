//! A fractional delay line: tape wow and flutter as a moving read head.

/// Reads its input back `delay(n)` samples late, at fractional delays.
#[derive(Debug)]
pub(crate) struct DelayLine {
    /// The latest input, a power of two long so a position masks into it.
    ring: Vec<f64>,
    max_delay: f64,
    /// Every sample taken so far: the read head's timeline, which no loop wraps.
    written: u64,
}

impl DelayLine {
    /// A line that can read up to `max_delay` samples back.
    pub(crate) fn new(max_delay: f64) -> Self {
        // the sample being written, and the earlier neighbor of the longest read
        let len = (max_delay.ceil() as usize + 2).next_power_of_two();
        Self {
            ring: vec![0.0; len],
            max_delay,
            written: 0,
        }
    }

    /// Replaces each sample with the input `delay(n)` samples before it, `n`
    /// counting every sample this line has taken; before the first, it reads
    /// silence. A delay out of range reads at the nearer end and NaN at 0,
    /// since a panic on the render path costs more than a misread delay.
    pub(crate) fn process(&mut self, buf: &mut [f64], mut delay: impl FnMut(u64) -> f64) {
        let mask = self.ring.len() - 1;
        for x in buf {
            let n = self.written;
            self.ring[n as usize & mask] = *x;
            // `max` returns the non-NaN operand
            // (<https://doc.rust-lang.org/std/primitive.f64.html#method.max>)
            let d = delay(n).max(0.0).min(self.max_delay);
            // the whole part comes off the integer count, so the fraction keeps
            // its precision however long the line runs
            let whole = d.floor();
            let frac = d - whole;
            let at = n.wrapping_sub(whole as u64);
            // linear between n − M and n − M − 1, "a one-zero FIR filter"
            // (<https://ccrma.stanford.edu/~jos/pasp/Fractional_Delay_Filtering_Linear.html>),
            // so a moving delay leaves no recursive state behind
            *x = (1.0 - frac) * self.ring[at as usize & mask]
                + frac * self.ring[at.wrapping_sub(1) as usize & mask];
            self.written += 1;
        }
    }

    /// Whether it holds only zeros, so silence in plays silence out.
    pub(crate) fn at_rest(&self) -> bool {
        self.ring.iter().all(|&v| v == 0.0)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;
    use crate::dsp::{NoiseStream, SAMPLE_RATE};

    #[test]
    fn a_constant_integer_delay_is_exact() {
        let max = 64;
        let mut rng = NoiseStream::new(1);
        let input: Vec<f64> = (0..1000).map(|_| f64::from(rng.norm())).collect();
        for k in [0, 1, 37, max] {
            let mut line = DelayLine::new(f64::from(max));
            let mut got = input.clone();
            line.process(&mut got, |_| f64::from(k));
            let k = k as usize;
            for (n, &g) in got.iter().enumerate() {
                let want = if n < k { 0.0 } else { input[n - k] };
                assert_eq!(g, want, "delay {k} at {n}");
            }
        }
    }

    #[test]
    fn a_ramp_reads_back_at_its_fractional_delay() {
        // linear interpolation is exact on a ramp, and these dyadic values keep
        // every product exact, so any slip in the neighbors, the weights or
        // the instant the delay is read shows as an inequality
        let delay = |n: u64| 3.0 + (n % 4) as f64 * 0.25;
        let mut buf: Vec<f64> = (0..1000).map(f64::from).collect();
        let mut line = DelayLine::new(4.0);
        line.process(&mut buf, delay);
        for (n, &got) in buf.iter().enumerate().skip(4) {
            assert_eq!(got, n as f64 - delay(n as u64), "at {n}");
        }
    }

    #[test]
    fn an_out_of_range_delay_reads_at_the_nearest_end() {
        let max = 32.0;
        let input = crate::stream::burst_then_silence(0);
        let read = |d: f64| {
            let mut buf = input.clone();
            DelayLine::new(max).process(&mut buf, |_| d);
            buf
        };
        for (d, end) in [
            (-5.0, 0.0),
            (f64::NEG_INFINITY, 0.0),
            (f64::NAN, 0.0),
            (max + 0.5, max),
            (f64::INFINITY, max),
        ] {
            assert!(
                read(d)
                    .iter()
                    .zip(read(end))
                    .all(|(g, w)| g.to_bits() == w.to_bits()),
                "delay {d} should read as {end}"
            );
        }
    }

    #[test]
    fn a_line_rests_once_its_input_has_passed() {
        let mut line = DelayLine::new(32.0);
        let mut burst = crate::stream::burst_then_silence(1);
        let (first, rest) = burst.split_at_mut(1);
        line.process(first, |_| 32.0);
        assert!(!line.at_rest());
        line.process(rest, |_| 32.0);
        assert!(line.at_rest());
    }

    #[test]
    fn a_modulated_delay_is_chunk_invariant() {
        let sr = f64::from(SAMPLE_RATE);
        crate::stream::assert_chunk_invariant(
            || DelayLine::new(32.0),
            |line, buf| {
                line.process(buf, |n| {
                    let t = n as f64 / sr;
                    16.0 + 12.0 * (TAU * 3.0 * t).sin() + 3.0 * (TAU * 40.0 * t).sin()
                });
            },
        );
    }
}
