//! Streaming building blocks for the lane chains: each keeps its state across
//! `process` calls, so a chain renders the same samples however
//! [`Performance::render`](crate::Performance::render) is chunked.

pub(crate) mod biquad;
pub(crate) mod delay;
pub(crate) mod onepole;

/// Below this a feedback state is silence: "a number whose absolute value is
/// below 10⁻²⁰ can be considered as 0" ([de Soras][denormal], §3).
///
/// [denormal]: http://ldesoras.free.fr/doc/articles/denormal-en.pdf
const FLUSH_FLOOR: f64 = 1e-20;

/// `v`, or 0 below [`FLUSH_FLOOR`]. In silence a feedback state otherwise
/// sticks at a subnormal, slow on a general-purpose CPU, because
/// round-to-nearest can return a tiny product unchanged: "Output stays
/// denormalized until input stops being 0" ([de Soras][denormal], §2.1). Per
/// sample, so no chunking moves it; in code, because `forbid(unsafe_code)`
/// keeps the FPU's DAZ flag out of reach.
///
/// [denormal]: http://ldesoras.free.fr/doc/articles/denormal-en.pdf
fn flush(v: f64) -> f64 {
    if v.abs() < FLUSH_FLOOR { 0.0 } else { v }
}

/// A noise burst, then `silence_secs` of zeros for a feedback state to decay
/// over.
#[cfg(test)]
fn burst_then_silence(silence_secs: u32) -> Vec<f64> {
    let mut rng = crate::dsp::NoiseStream::new(7);
    let mut buf: Vec<f64> = (0..10_000).map(|_| f64::from(rng.norm())).collect();
    buf.resize(
        buf.len() + (silence_secs * crate::dsp::SAMPLE_RATE) as usize,
        0.0,
    );
    buf
}

/// Panics unless a fresh `make()` driven by `process` gives bit-identical
/// output over one call and over every chunking the render path can see.
#[cfg(test)]
fn assert_chunk_invariant<P>(make: impl Fn() -> P, mut process: impl FnMut(&mut P, &mut [f64])) {
    // the silence decays every state past the flush floor, so a flush
    // placed per call rather than per sample shows
    let input = burst_then_silence(1);
    let mut whole = input.clone();
    process(&mut make(), &mut whole);
    for chunk in [1, 127, 4096] {
        let mut p = make();
        let mut got = input.clone();
        for c in got.chunks_mut(chunk) {
            process(&mut p, c);
        }
        let first_diff = whole
            .iter()
            .zip(&got)
            .position(|(w, g)| w.to_bits() != g.to_bits());
        assert_eq!(first_diff, None, "chunks of {chunk}");
    }
}
