# pixtuoid-lofi

A seed composes an endless run of lofi songs, synthesized from constants. No samples, no dependencies, deterministic. Cut from [pixtuoid](https://github.com/pixtuoid/pixtuoid), whose terminal office it scores; history kept.

```sh
cargo add pixtuoid-lofi
```

```rust
use pixtuoid_lofi::{Controls, Frame, Performance};

let mut music = Performance::new(7, Controls::default().with_intensity(1.0));
let mut chunk = [Frame::default(); 1024];
music.render(&mut chunk);
assert_eq!(music.position(), 1024);
```

`render` fills `[left, right]` frames at `dsp::SAMPLE_RATE`, mixed and under lofi's own ceiling; playback is the caller's.

## Determinism

The same seed gives the same score, event for event and bit for bit, on every platform: the composer uses no transcendental function, and `+ - * /` are exact IEEE 754-2008 per [Rust RFC 3514](https://rust-lang.github.io/rfcs/3514-float-semantics.html). Rendered audio is not bit-identical: `sin` and `exp` have "non-deterministic" precision that "varies by platform, Rust version, and can even differ within the same execution" ([`f32::sin`](https://doc.rust-lang.org/std/primitive.f32.html#method.sin)), so renders match by spectral fingerprint, not by bit.

## Listening

```sh
cargo run --release --example lofi_audition -- --seeds 4
```

Writes one wav per seed, as pixtuoid's office plays it; the example's header lists its flags.

## License

MIT.
