//! Score access and per-lane renders for lofi's own tools (`examples/`) to
//! solo, mute and measure. Not a contract: any release may change it.

pub use crate::compose::{Mood, Score, compose};
pub use crate::stream::master::{TIER_INTENSITIES, lane_gain};
pub use crate::synth::{Lane, render_lane};

use crate::{Controls, Frame, Performance};

/// Song `k` of `seed`'s run, from its first bar to the next song's, played
/// under `controls` with no earlier song ringing into it.
pub fn song(seed: u64, k: u64, controls: Controls) -> Vec<Frame> {
    Performance::from_song(seed, k, controls).finish_song()
}
