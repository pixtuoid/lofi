//! Streaming building blocks for the lane chains: each keeps its state across
//! `process` calls, so a chain renders the same samples however
//! [`Performance::render`](crate::Performance::render) is chunked.

pub(crate) mod biquad;
pub(crate) mod delay;
pub(crate) mod onepole;

/// Panics unless a fresh `make()` driven by `process` gives bit-identical
/// output over one call and over every chunking the render path can see.
#[cfg(test)]
fn assert_chunk_invariant<P>(make: impl Fn() -> P, mut process: impl FnMut(&mut P, &mut [f64])) {
    let mut rng = crate::dsp::NoiseStream::new(7);
    let input: Vec<f64> = (0..10_000).map(|_| f64::from(rng.norm())).collect();
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
