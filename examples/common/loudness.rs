//! [BS.1770-5](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf)
//! gated loudness of a buffer that plays on repeat: the K-filter and the gating
//! blocks both wrap, so the reading is the endlessly looping buffer's.

use std::f64::consts::PI;

use pixtuoid_lofi::dsp::SAMPLE_RATE;

/// A second-order section, `[b0, b1, b2, a1, a2]` over `a0 = 1`.
type Biquad = [f64; 5];

/// The gating block's duration, `T_g`, in seconds.
const BLOCK_S: f64 = 0.4;
/// The gating blocks' overlap, as a fraction of `T_g`.
const OVERLAP: f64 = 0.75;
/// The absolute gate `Γ_a`, LKFS.
const ABSOLUTE_GATE: f64 = -70.0;
/// The relative gate, LU against the absolutely gated loudness.
const RELATIVE_GATE: f64 = -10.0;
/// Equation (2)'s offset, which cancels the K-filter's gain at 997 Hz.
const OFFSET: f64 = -0.691;

/// The K-weighting pre-filter at `sr` Hz, shelf then high-pass. The standard
/// tabulates it at 48 kHz only; these are libebur128's analog parameters
/// through its bilinear transform
/// (<https://github.com/jiixyj/libebur128/blob/67b33abe1558160ed76ada1322329b0e9e058b02/ebur128/ebur128.c#L251-L283>),
/// which reproduce that table.
fn k_filter(sr: f64) -> [Biquad; 2] {
    const SHELF_HZ: f64 = 1681.974450955533;
    const SHELF_GAIN_DB: f64 = 3.999843853973347;
    const SHELF_Q: f64 = 0.7071752369554196;
    const SHELF_VB_EXPONENT: f64 = 0.4996667741545416;
    const HIGHPASS_HZ: f64 = 38.13547087602444;
    const HIGHPASS_Q: f64 = 0.5003270373238773;

    let k = (PI * SHELF_HZ / sr).tan();
    let vh = 10f64.powf(SHELF_GAIN_DB / 20.0);
    let vb = vh.powf(SHELF_VB_EXPONENT);
    let a0 = 1.0 + k / SHELF_Q + k * k;
    let shelf = [
        (vh + vb * k / SHELF_Q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / SHELF_Q + k * k) / a0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / SHELF_Q + k * k) / a0,
    ];
    let k = (PI * HIGHPASS_HZ / sr).tan();
    let a0 = 1.0 + k / HIGHPASS_Q + k * k;
    // Table 2's numerator, which the standard leaves unscaled by `a0`.
    let highpass = [
        1.0,
        -2.0,
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / HIGHPASS_Q + k * k) / a0,
    ];
    [shelf, highpass]
}

/// `x` through each of `filter`'s sections in turn, each run over the loop
/// once to warm up and once to keep. A loop outlasts the filter's settling, so
/// the kept pass starts in the state the loop ends in and the output wraps.
fn run_circular(x: &[f64], filter: &[Biquad]) -> Vec<f64> {
    let mut y = x.to_vec();
    for &[b0, b1, b2, a1, a2] in filter {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        let mut out = vec![0.0; y.len()];
        for pass in 0..2 {
            for (i, &x0) in y.iter().enumerate() {
                let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                (x2, x1, y2, y1) = (x1, x0, y1, y0);
                if pass == 1 {
                    out[i] = y0;
                }
            }
        }
        y = out;
    }
    y
}

/// `buf`'s gated loudness as a loop, in LKFS: −∞ when no block passes the
/// absolute gate.
pub(crate) fn gated_lufs(buf: &[f32]) -> f64 {
    let sr = f64::from(SAMPLE_RATE);
    let x: Vec<f64> = buf.iter().map(|&v| f64::from(v)).collect();
    let y = run_circular(&x, &k_filter(sr));
    let n = y.len();
    let block = (BLOCK_S * sr).round() as usize;
    let step = (BLOCK_S * (1.0 - OVERLAP) * sr).round() as usize;
    // equation (3), each block wrapping past the loop's end
    let z: Vec<f64> = (0..n.div_ceil(step))
        .map(|j| {
            (0..block)
                .map(|i| y[(j * step + i) % n].powi(2))
                .sum::<f64>()
                / block as f64
        })
        .collect();
    let loudness = |z: f64| OFFSET + 10.0 * z.log10();
    // equation (5) over the blocks louder than `gate`
    let gated = |gate: f64| {
        let (sum, count) = z
            .iter()
            .filter(|&&zj| loudness(zj) > gate)
            .fold((0.0, 0usize), |(s, c), &zj| (s + zj, c + 1));
        loudness(sum / count.max(1) as f64)
    };
    let relative = gated(ABSOLUTE_GATE) + RELATIVE_GATE;
    gated(relative.max(ABSOLUTE_GATE))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a full-scale 997 Hz sine reads, so a sine's amplitude sets its loudness.
    const FULL_SCALE_LKFS: f64 = -3.01;

    fn amplitude_for(lkfs: f64) -> f64 {
        10f64.powf((lkfs - FULL_SCALE_LKFS) / 20.0)
    }

    fn sine_997_hz(n: usize, amplitude: impl Fn(usize) -> f64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(SAMPLE_RATE);
                (amplitude(i) * (std::f64::consts::TAU * 997.0 * t).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn a_full_scale_997_hz_sine_reads_minus_3_01_lkfs() {
        let l = gated_lufs(&sine_997_hz(SAMPLE_RATE as usize * 10, |_| 1.0));
        assert!((l - FULL_SCALE_LKFS).abs() <= 0.005, "{l}");
    }

    #[test]
    fn at_48_khz_the_k_filter_is_the_standards_table() {
        // BS.1770-5 Tables 1 and 2, as [b0, b1, b2, a1, a2]
        let tables = [
            [
                1.53512485958697,
                -2.69169618940638,
                1.19839281085285,
                -1.69065929318241,
                0.73248077421585,
            ],
            [1.0, -2.0, 1.0, -1.99004745483398, 0.99007225036621],
        ];
        for (got, want) in k_filter(48_000.0).iter().zip(tables) {
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 1e-12, "{got:?} vs {want:?}");
            }
        }
    }

    #[test]
    fn a_quiet_half_falls_under_the_relative_gate() {
        let n = SAMPLE_RATE as usize * 10;
        let l = gated_lufs(&sine_997_hz(n, |i| if i < n / 2 { 1.0 } else { 0.1 }));
        // ungated, the half 20 dB down would pull it to about −6
        assert!((l - FULL_SCALE_LKFS).abs() < 0.5, "{l}");
    }

    #[test]
    fn blocks_under_the_absolute_gate_stay_out_of_the_relative_gate() {
        let s = SAMPLE_RATE as usize;
        let level = |i: usize| match i / s {
            0..4 => -20.0,
            4..8 => -38.0,
            _ => -80.0,
        };
        let l = gated_lufs(&sine_997_hz(40 * s, |i| amplitude_for(level(i))));
        // the −80 blocks would sink the relative gate under the −38 ones, which
        // would then pull the reading to about −23
        assert!((l - (-20.0)).abs() < 0.5, "{l}");
    }

    #[test]
    fn a_burst_across_the_wrap_reads_as_it_does_mid_loop() {
        let n = SAMPLE_RATE as usize * 10;
        let burst = SAMPLE_RATE as usize / 4;
        let across = sine_997_hz(n, |i| f64::from(u8::from(i < burst || i >= n - burst)));
        // a whole number of the standard's 100 ms block steps
        let shift = SAMPLE_RATE as usize / 10 * 37;
        let mid: Vec<f32> = (0..n).map(|i| across[(i + shift) % n]).collect();
        let (a, b) = (gated_lufs(&across), gated_lufs(&mid));
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}
