//! `Performance::render`'s cost against two local gates, never run in CI, whose
//! shared runners' clocks time their neighbors too (opinion). Exits non-zero
//! when a gate fails, and names the machine, since the numbers are only its.
//!
//! Usage, natively, then on wasm32 under Node, through a harness that lends the
//! build a clock and a console, which that target's std lacks:
//!   cargo run --release --example lofi_bench
//!   cargo build --release --example lofi_bench --target wasm32-unknown-unknown
//!   node examples/lofi_bench.mjs target/wasm32-unknown-unknown/release/examples/lofi_bench.wasm

use std::hint::black_box;
use std::ops::Range;
use std::process::ExitCode;

use pixtuoid_lofi::dsp::SAMPLE_RATE;
use pixtuoid_lofi::{Controls, Frame, Performance, audition};

use host::{Clock, machine, say};

/// The span a player renders ahead in, in seconds.
const CHUNK_S: f64 = 0.5;

/// The most of a chunk's duration that rendering it may take. Opinion: it
/// keeps a player's catch-up after its slowest chunk inside its underrun
/// bound.
const WORST_CHUNK_SHARE: f64 = 0.5;

/// The most of one core the render may take on average: a proposal the
/// owner hasn't ruled on, which the measured figures go to.
const MEAN_RTF: f64 = 0.05;

/// The seeds the slowest chunk is sought over.
const SEEDS: Range<u64> = 0..3;

/// How long each room's mean is taken over, in seconds: an hour.
const MEAN_OVER_S: u32 = 60 * 60;

fn main() -> ExitCode {
    if cfg!(debug_assertions) {
        say("lofi_bench: a debug build times nothing a listener runs; build with --release");
        return ExitCode::FAILURE;
    }
    say(&format!("machine: {}", machine()));
    let chunk = (CHUNK_S * f64::from(SAMPLE_RATE)) as usize;
    let mut buf = vec![Frame::default(); chunk];
    let worst = slowest_chunk(&mut buf);
    let share = worst / CHUNK_S;
    say(&format!(
        "slowest {CHUNK_S} s chunk over two songs and their transition, at intensity 1 and rain 1: {:.2} ms, {:.1} % of the chunk (gate {:.0} %)",
        worst * 1e3,
        share * 1e2,
        WORST_CHUNK_SHARE * 1e2
    ));
    let mut pass = share <= WORST_CHUNK_SHARE;
    for (room, intensity) in [("empty", 0.0), ("busy", 1.0)] {
        let rtf = mean_rtf(&mut buf, Controls::default().with_intensity(intensity));
        say(&format!(
            "mean real-time factor over an hour, {room} room: {rtf:.4} (gate {MEAN_RTF})"
        ));
        pass &= rtf <= MEAN_RTF;
    }
    if pass {
        ExitCode::SUCCESS
    } else {
        say("lofi_bench: a gate failed");
        ExitCode::FAILURE
    }
}

/// The longest one `render` of a `buf`-long chunk took, in seconds, over each
/// of [`SEEDS`]' first two songs: a busy, rainy day's, then a night's, whose
/// first bars the day's tails ring over.
fn slowest_chunk(buf: &mut [Frame]) -> f64 {
    let day = Controls::default().with_intensity(1.0).with_rain(1.0);
    let night = day.with_daylight(0.0);
    let mut worst = 0.0f64;
    for seed in SEEDS {
        let span = audition::song(seed, 0, day).len() + audition::song(seed, 1, night).len();
        let mut music = Performance::new(seed, day);
        music.set_controls(night);
        while music.position() < span as u64 {
            let n = buf.len().min(span - music.position() as usize);
            let clock = Clock::start();
            music.render(&mut buf[..n]);
            worst = worst.max(clock.secs());
            black_box(&buf);
        }
    }
    worst
}

/// The seconds rendering [`MEAN_OVER_S`] took under `controls`, in `buf`-long
/// chunks, per second rendered.
fn mean_rtf(buf: &mut [Frame], controls: Controls) -> f64 {
    let mut music = Performance::new(SEEDS.start, controls);
    let frames = u64::from(MEAN_OVER_S) * u64::from(SAMPLE_RATE);
    let clock = Clock::start();
    while music.position() < frames {
        music.render(buf);
        black_box(&buf);
    }
    clock.secs() / f64::from(MEAN_OVER_S)
}

/// What the host lends the bench: std's clock and stdout natively.
#[cfg(not(target_arch = "wasm32"))]
mod host {
    use std::process::Command;
    use std::time::Instant;

    /// A stopwatch.
    pub(crate) struct Clock(Instant);

    impl Clock {
        pub(crate) fn start() -> Self {
            Self(Instant::now())
        }

        /// Seconds since [`start`](Self::start).
        pub(crate) fn secs(&self) -> f64 {
            self.0.elapsed().as_secs_f64()
        }
    }

    pub(crate) fn say(line: &str) {
        println!("{line}");
    }

    /// The model and chip `sysctl` names on a Mac, else what `uname` names,
    /// else the target's.
    pub(crate) fn machine() -> String {
        let asks: [(&str, &[&str]); 2] = [
            ("sysctl", &["-n", "hw.model", "machdep.cpu.brand_string"]),
            ("uname", &["-srm"]),
        ];
        for (program, args) in asks {
            if let Ok(out) = Command::new(program).args(args).output()
                && out.status.success()
            {
                let named = String::from_utf8_lossy(&out.stdout);
                return named.split_whitespace().collect::<Vec<_>>().join(" ");
            }
        }
        format!("{} {}", std::env::consts::OS, std::env::consts::ARCH)
    }
}

/// What the host lends the bench: on wasm32, whose std has no clock and whose
/// stdout discards, `lofi_bench.mjs`'s.
#[cfg(target_arch = "wasm32")]
mod host {
    #[expect(
        unsafe_code,
        reason = "an import's signature is a promise the compiler can't check; the harness keeps it"
    )]
    #[link(wasm_import_module = "bench")]
    unsafe extern "C" {
        /// `performance.now()`, in milliseconds.
        safe fn now_ms() -> f64;
        /// Prints the UTF-8 `len` bytes at `ptr` as a line.
        safe fn print(ptr: *const u8, len: usize);
    }

    /// A stopwatch.
    pub(crate) struct Clock(f64);

    impl Clock {
        pub(crate) fn start() -> Self {
            Self(now_ms())
        }

        /// Seconds since [`start`](Self::start).
        pub(crate) fn secs(&self) -> f64 {
            (now_ms() - self.0) / 1e3
        }
    }

    pub(crate) fn say(line: &str) {
        print(line.as_ptr(), line.len());
    }

    /// The harness names the host before the bench runs.
    pub(crate) fn machine() -> String {
        "wasm32-unknown-unknown, on the host named above".into()
    }
}
