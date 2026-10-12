#![doc = include_str!("../README.md")]

pub mod dsp;

mod compose;
#[cfg(test)]
mod score;
mod synth;

pub use compose::{Mood, Score, compose};
pub use synth::{Lane, render_lane};
