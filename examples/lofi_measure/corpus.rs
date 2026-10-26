//! The reference corpus a take's spectrum is read against, and the axes it
//! was read on: moving one means reading [`CORPUS`] again, with
//! `lofi_measure --corpus DIR`.

/// The share bands' inner edges in Hz; the first is the sub band's.
pub(crate) const BAND_EDGES_HZ: [f64; 8] =
    [62.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 6500.0];

/// The LTAS slope's span in Hz, fitted over the power density of its
/// fractional octave bands.
pub(crate) const LTAS_SPAN_HZ: [f64; 2] = [250.0, 4000.0];
pub(crate) const LTAS_BANDS_PER_OCTAVE: f64 = 3.0;

/// The axes [`CORPUS`] was read on, which `lofi_measure --corpus` prints with
/// it: a change to any of them leaves [`CORPUS`] stale until it is read again.
#[cfg(test)]
pub(crate) struct Axes {
    pub(crate) band_edges_hz: [f64; BAND_EDGES_HZ.len()],
    pub(crate) ltas_span_hz: [f64; 2],
    pub(crate) ltas_bands_per_octave: f64,
    pub(crate) sample_rate: u32,
}

/// What [`CORPUS`] was read on.
#[cfg(test)]
pub(crate) const MEASURED_ON: Axes = Axes {
    band_edges_hz: [62.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 6500.0],
    ltas_span_hz: [250.0, 4000.0],
    ltas_bands_per_octave: 3.0,
    sample_rate: 44_100,
};

/// The percentiles a [`Spread`] holds.
pub(crate) const PERCENTILES: [u8; 5] = [10, 25, 50, 75, 90];

/// A reading's [`PERCENTILES`] across the corpus's tracks.
pub(crate) type Spread = [f64; PERCENTILES.len()];

/// What [`CORPUS`] holds.
pub(crate) struct Corpus {
    /// Each band's percent of the power, from DC up across
    /// [`BAND_EDGES_HZ`].
    pub(crate) shares_pct: [Spread; BAND_EDGES_HZ.len() + 1],
    pub(crate) centroid_hz: Spread,
    /// The LTAS density slope across [`LTAS_SPAN_HZ`], in dB per octave.
    pub(crate) ltas_db_per_oct: Spread,
}

/// 100 lofi tracks from archive.org netlabel uploads, 5.98 h, each item
/// declaring a CC BY-NC licence in its `licenseurl`, each read whole as
/// `lofi_measure` reads a take: decoded at lofi's sample rate, folded to mono
/// as `fold` does, and read by `Spectrum`, one rectangular FFT over the track
/// with DC left out. Each spread is linear between ranks (numpy's
/// `percentile`). Measured 2026-10-19.
#[expect(
    clippy::approx_constant,
    reason = "a slope's p90 reads -3.14 dB/oct, not π"
)]
pub(crate) const CORPUS: Corpus = Corpus {
    shares_pct: [
        [3.90, 8.09, 14.75, 26.20, 36.85],
        [18.64, 23.56, 35.37, 47.61, 59.81],
        [6.23, 10.95, 15.38, 22.67, 29.49],
        [2.90, 5.54, 8.52, 13.07, 19.09],
        [2.03, 2.73, 5.95, 9.97, 14.19],
        [0.96, 1.57, 3.05, 5.70, 8.84],
        [0.61, 1.05, 2.00, 3.55, 5.26],
        [0.31, 0.55, 1.11, 1.84, 2.57],
        [0.15, 0.30, 0.69, 1.60, 2.64],
    ],
    centroid_hz: [206.3, 310.3, 445.3, 579.3, 763.2],
    ltas_db_per_oct: [-7.00, -6.15, -5.07, -3.89, -3.14],
};

/// `values`' [`PERCENTILES`], each linear between ranks, as numpy's
/// `percentile` reads them.
pub(crate) fn spread(mut values: Vec<f64>) -> Spread {
    values.sort_by(f64::total_cmp);
    let last = values.len() - 1;
    PERCENTILES.map(|p| {
        let at = f64::from(p) / 100.0 * last as f64;
        let (lo, frac) = (at.floor() as usize, at.fract());
        values[lo] + (values[(lo + 1).min(last)] - values[lo]) * frac
    })
}

/// Which two of `spread`'s percentiles `value` falls between, as `p25-50`,
/// or which end it falls past, as `<p10` or `>p90`.
pub(crate) fn percentile(value: f64, spread: Spread) -> String {
    let [first, .., last] = PERCENTILES;
    if value < spread[0] {
        return format!("<p{first}");
    }
    (PERCENTILES.windows(2).zip(spread.windows(2)))
        .find(|(_, s)| value <= s[1])
        .map_or(format!(">p{last}"), |(p, _)| format!("p{}-{}", p[0], p[1]))
}
