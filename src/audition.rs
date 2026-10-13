//! Score access and per-lane renders for lofi's own tools (`examples/`) to
//! solo, mute and measure. Not a contract: any release may change it.

pub use crate::compose::{Mood, Score, compose};
pub use crate::synth::{Lane, render_lane};
