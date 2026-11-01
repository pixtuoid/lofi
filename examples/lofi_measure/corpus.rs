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

/// 108 tracks of Lofi Girl's own lofi hip hop playlist on YouTube, 5.00 h,
/// the most viewed there under a cap of three to an artist and to a release,
/// each read whole as `lofi_measure` reads a take: decoded at lofi's sample
/// rate, folded to mono as `fold` does, and read by `Spectrum`, one rectangular
/// FFT over the track with DC left out. Each spread is linear between ranks
/// (numpy's `percentile`). Measured 2026-11-01.
pub(crate) const CORPUS: Corpus = Corpus {
    shares_pct: [
        [11.48, 18.17, 26.19, 37.78, 46.83],
        [15.78, 20.26, 29.12, 36.53, 44.48],
        [6.94, 10.45, 15.21, 19.25, 24.75],
        [5.72, 10.38, 14.79, 20.65, 28.99],
        [2.29, 3.36, 5.09, 8.07, 12.55],
        [0.55, 0.76, 1.44, 2.53, 4.30],
        [0.06, 0.16, 0.40, 0.63, 1.08],
        [0.00, 0.02, 0.06, 0.18, 0.34],
        [0.00, 0.00, 0.01, 0.05, 0.10],
    ],
    centroid_hz: [146.9, 169.3, 211.2, 264.1, 314.3],
    ltas_db_per_oct: [-11.28, -9.57, -8.31, -7.28, -6.40],
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
