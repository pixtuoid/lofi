# pixtuoid-lofi

A seed composes a lofi song; the song is synthesized from constants. No samples, no dependencies, deterministic. Cut from [pixtuoid](https://github.com/IvanWng97/pixtuoid), whose terminal office it scores; history kept.

```sh
cargo add pixtuoid-lofi
```

```rust
use pixtuoid_lofi::dsp::NoiseStream;
use pixtuoid_lofi::{Lane, Mood, compose, render_lane};

let score = compose(Mood::Night, 7);
let mut rng = NoiseStream::new(1);
let bass = render_lane(&score, Lane::Bass, &mut rng);
assert!(!bass.is_empty());
```

Each lane renders to mono `f32` at `dsp::SAMPLE_RATE`; mixing and playback are the caller's.

## Determinism

The same seed on the same platform gives the same bits. `+ - * /` are exact per [Rust RFC 3514](https://rust-lang.github.io/rfcs/3514-float-semantics.html), but `sin` and `exp` may differ across platforms, so cross-platform output matches by spectral fingerprint, not bit for bit.

## Listening

```sh
cargo run --release --example lofi_audition -- --seeds 4
```

Writes one wav per seed; `--solo <lane>` isolates a lane.

## License

MIT.
