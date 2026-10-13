#![doc = include_str!("../README.md")]

pub mod dsp;

#[doc(hidden)]
pub mod audition;
mod compose;
mod perform;
#[cfg(test)]
mod score;
#[cfg_attr(not(test), expect(dead_code, reason = "wired by Task 4"))]
mod stream;
mod synth;

pub use perform::{Controls, Frame, Performance};
