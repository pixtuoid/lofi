#![doc = include_str!("../README.md")]

pub mod dsp;

#[doc(hidden)]
pub mod audition;
mod compose;
#[cfg(test)]
#[path = "../examples/common/loudness.rs"]
mod loudness;
mod perform;
#[cfg(test)]
mod score;
mod stream;
mod synth;

pub use perform::{Controls, Frame, Performance};
