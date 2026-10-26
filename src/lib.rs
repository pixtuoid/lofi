#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod dsp;

#[doc(hidden)]
pub mod audition;
mod compose;
mod perform;
#[cfg(test)]
mod score;
mod stream;
mod synth;

pub use perform::{Controls, Frame, Performance};
