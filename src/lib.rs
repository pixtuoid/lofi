//! Procedural lofi: a seed composes a song, and the song is synthesized from
//! constants — no samples, no dependencies, deterministic. Cut from
//! [pixtuoid](https://github.com/IvanWng97/pixtuoid), whose terminal office it
//! scores.
//!
//! ```
//! use pixtuoid_lofi::dsp::NoiseStream;
//! use pixtuoid_lofi::{Lane, Mood, compose, render_lane};
//!
//! let score = compose(Mood::Night, 7);
//! let mut rng = NoiseStream::new(1);
//! let bass = render_lane(&score, Lane::Bass, &mut rng);
//! assert!(!bass.is_empty());
//! ```

pub mod dsp;

mod compose;
mod score;
mod synth;

pub use compose::{Mood, Score, compose};
pub use synth::{Lane, render_lane};
