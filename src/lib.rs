#![doc = include_str!("../README.md")]

pub mod dsp;

#[doc(hidden)]
pub mod audition;
mod compose;
mod perform;
#[cfg(test)]
mod score;
mod synth;

pub use perform::{Controls, Frame, Performance};
