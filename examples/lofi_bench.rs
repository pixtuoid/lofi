//! Local gates for what CI can't afford: `Performance::render`'s cost, and a
//! run of each room playing finite at a steady level. Never in CI, which
//! would spend minutes on them and whose shared runners' clocks time their
//! neighbors too (opinion). Exits non-zero when a gate fails, and names the
//! machine, since the times are only its.
//!
//! Usage, natively, then on wasm32 under Node, through a harness that lends the
//! build what its wasm32 `host` imports:
//!   cargo run --release --example lofi_bench
//!   cargo build --release --example lofi_bench --target wasm32-unknown-unknown
//!   node examples/lofi_bench.mjs target/wasm32-unknown-unknown/release/examples/lofi_bench.wasm

use std::hint::black_box;
use std::ops::Range;
use std::process::ExitCode;

use pixtuoid_lofi::audition::{self, TIER_INTENSITIES};
use pixtuoid_lofi::dsp::SAMPLE_RATE;
use pixtuoid_lofi::{Controls, Frame, Performance};

use host::{Clock, machine, say};

/// The span a player renders ahead in, in seconds.
const CHUNK_S: f64 = 0.5;

/// The most of a chunk's duration that rendering it may take. Opinion: it
/// keeps a player's catch-up after its slowest chunk inside the render-ahead
/// pixtuoid's player keeps, `MUSIC_AHEAD` in
/// crates/pixtuoid-scene/src/audio/mod.rs.
const WORST_CHUNK_SHARE: f64 = 0.5;

/// The most of one core the render may take on average: the owner's gate
/// ("保留 0.05 作为 gate", 2026-10-26), which the measured figures go to.
const MEAN_RTF: f64 = 0.05;

/// The seeds the slowest chunk is sought over (opinion).
const SEEDS: Range<u64> = 0..3;

/// How many times each seed's chunks are timed, each chunk read at its least:
/// a render is a pure function of its seed and timeline, so every run does
/// the same work and only the machine's other load sets the rest, and "using
/// the minimum estimator for the true run time of a benchmark, rather than
/// the mean or median, is robust to nonideal statistics and also provides the
/// smallest error" (Chen & Revels 2016, <https://arxiv.org/abs/1608.04295>).
/// The count is opinion.
const REPEATS: usize = 5;

/// The daylight the timeline's day song grades to: halfway from
/// [`Controls::DUSK`] to full day, so the song grades within the hiphop style
/// rather than past it.
const MID_SONG_DAYLIGHT: f32 = f32::midpoint(Controls::DUSK, 1.0);

/// How long each room plays, in seconds, to the end of its last song.
/// The mean settles within minutes, and this spans many song transitions.
const ROOM_S: u32 = 20 * 60;

/// The most a song of a room's run may sit from its own level played alone,
/// in LU: EBU R 128's permitted deviation from a target, which "exists to
/// allow for minor variations of meter calibrations and errors"
/// (<https://tech.ebu.ch/docs/r/r128-2014.pdf>).
const SONG_LEVEL_TOLERANCE_LU: f64 = 0.5;

fn main() -> ExitCode {
    if cfg!(debug_assertions) {
        say("lofi_bench: a debug build times nothing a listener runs; build with --release");
        return ExitCode::FAILURE;
    }
    say(&format!("machine: {}", machine()));
    let chunk = (CHUNK_S * f64::from(SAMPLE_RATE)) as usize;
    let mut buf = vec![Frame::default(); chunk];
    let (worst, seed, at) = match slowest_chunk(&mut buf) {
        Ok(worst) => worst,
        Err(e) => {
            say(&format!("lofi_bench: {e}"));
            return ExitCode::FAILURE;
        }
    };
    let share = worst / CHUNK_S;
    say(&format!(
        "slowest {CHUNK_S} s chunk over two songs, a mid-song timeline and their transition, from intensity 1: {:.2} ms, {:.1} % of the chunk (gate {:.0} %), seed {seed} at {:.1} s",
        worst * 1e3,
        share * 1e2,
        WORST_CHUNK_SHARE * 1e2,
        at as f64 / f64::from(SAMPLE_RATE)
    ));
    let mut pass = share <= WORST_CHUNK_SHARE;
    for (room, intensity) in [
        ("empty", TIER_INTENSITIES[0]),
        ("busy", TIER_INTENSITIES[2]),
    ] {
        let controls = Controls::default().with_intensity(intensity);
        pass &= play_room(&mut buf, room, controls);
    }
    if pass {
        ExitCode::SUCCESS
    } else {
        say("lofi_bench: a gate failed");
        ExitCode::FAILURE
    }
}

/// The longest one `render` of a `buf`-long chunk takes, in seconds, over each
/// of [`SEEDS`]' first two songs: a busy day's, whose bars empty the room,
/// fill it again and grade toward night, then a rainy night's, whose first
/// bars the day's tails ring over. Each chunk takes its least over
/// [`REPEATS`] runs. With it, the seed and the frame the chunk started at,
/// which tell a cold start from a transition. Errs unless the runs played
/// those two songs.
fn slowest_chunk(buf: &mut [Frame]) -> Result<(f64, u64, u64), String> {
    let day = Controls::default().with_intensity(1.0);
    let night = day.with_daylight(0.0).with_rain(1.0);
    // songs with no form would time a render of less than the forms play
    let forms: Vec<_> = (SEEDS.flat_map(|seed| [(seed, 0, day), (seed, 1, night)]))
        .map(|(seed, k, c)| audition::song_form(seed, k, c))
        .collect();
    audition::spans_a_form(&forms).map_err(|e| format!("the timed songs: {e}"))?;
    let mut worst = (0.0, SEEDS.start, 0);
    for seed in SEEDS {
        let (runs, ends): (Vec<_>, Vec<_>) = (0..REPEATS)
            .map(|_| timed_chunks(seed, day, night, buf))
            .unzip();
        // every run renders a pure function of the same seed and timeline,
        // so the first plays the two songs if any does
        played(seed, day, night, &ends[0])?;
        for (at, took) in least(&runs) {
            if took > worst.0 {
                worst = (took, seed, at);
            }
        }
    }
    Ok(worst)
}

/// Each chunk's frame and its least time over `runs` of one seed, which
/// chunk alike: a render is a pure function of its seed and timeline.
fn least(runs: &[Vec<(u64, f64)>]) -> Vec<(u64, f64)> {
    let mut least = runs[0].clone();
    for run in &runs[1..] {
        for ((_, took), &(_, now)) in least.iter_mut().zip(run) {
            *took = took.min(now);
        }
    }
    least
}

/// Each chunk's frame and how long its `render` took, in seconds, over one
/// run of `seed`'s day song, then its rainy `night` one, and what the run's
/// first chunk and its last played.
fn timed_chunks(
    seed: u64,
    day: Controls,
    night: Controls,
    buf: &mut [Frame],
) -> (Vec<(u64, f64)>, [Vec<Frame>; 2]) {
    let span = audition::song_len(seed, 0, day) + audition::song_len(seed, 1, night);
    // each mid-bar, so it latches at the next bar line: the night is asked
    // for in song 0's second-to-last bar, so its last bar grades toward
    // night and song 1 starts a night song
    let bars = audition::bar_frames(seed, 0, day);
    let mid = |bar: usize| bars[bar].midpoint(bars[bar + 1]);
    let calls = [
        (mid(8), day.with_intensity(TIER_INTENSITIES[0])),
        (mid(16), day.with_daylight(MID_SONG_DAYLIGHT)),
        (mid(bars.len() - 2), night),
    ];
    let mut calls = calls.iter().peekable();
    let mut music = Performance::new(seed, day);
    let (mut opening, mut last, mut timed) = (Vec::new(), 0, Vec::new());
    while music.position() < span as u64 {
        let at = music.position();
        let until = calls.peek().map_or(span, |&&(call, _)| call);
        let n = buf.len().min(until - at as usize);
        let clock = Clock::start();
        music.render(&mut buf[..n]);
        timed.push((at, clock.secs()));
        black_box(&buf);
        if at == 0 {
            opening = buf[..n].to_vec();
        }
        if let Some(&&(call, controls)) = calls.peek()
            && music.position() == call as u64
        {
            music.set_controls(controls);
            calls.next();
        }
        last = n;
    }
    (timed, [opening, buf[..last].to_vec()])
}

/// Errs unless the run of `seed` whose first chunk played `opening` and whose
/// last played `closing` played its day song, then its rainy `night` one.
fn played(
    seed: u64,
    day: Controls,
    night: Controls,
    [opening, closing]: &[Vec<Frame>; 2],
) -> Result<(), String> {
    let mut day_alone = vec![Frame::default(); opening.len()];
    Performance::new(seed, day).render(&mut day_alone);
    // past its first loop, song 1 plays as it does alone
    let night_alone = audition::song(seed, 1, night, &[]);
    if day_alone != *opening || night_alone[night_alone.len() - closing.len()..] != closing[..] {
        return Err(format!(
            "seed {seed} didn't play a day's song, then a rainy night's"
        ));
    }
    Ok(())
}

/// Plays [`ROOM_S`] of `room` under `controls` in `buf`-long chunks, and
/// says whether it passed its gates: the mean real-time factor within
/// [`MEAN_RTF`], every frame finite, and each song's gated loudness within
/// [`SONG_LEVEL_TOLERANCE_LU`] of the same song played alone. Each song is
/// held to itself, not to other songs' spread, so what the run's sequencing
/// does to a level shows, and how far a seed's songs differ doesn't.
fn play_room(buf: &mut [Frame], room: &str, controls: Controls) -> bool {
    let seed = SEEDS.start;
    let mut music = Performance::new(seed, controls);
    let (mut took, mut songs) = (0.0, Vec::new());
    while music.position() < u64::from(ROOM_S) * u64::from(SAMPLE_RATE) {
        let len = audition::song_len(seed, songs.len() as u64, controls);
        let mut song = Vec::with_capacity(len);
        while song.len() < len {
            let n = buf.len().min(len - song.len());
            let clock = Clock::start();
            music.render(&mut buf[..n]);
            took += clock.secs();
            if let Some(i) = buf[..n]
                .iter()
                .position(|f| !f.iter().all(|v| v.is_finite()))
            {
                let frame = music.position() - (n - i) as u64;
                say(&format!("{room} room: frame {frame} plays {:?}", buf[i]));
                return false;
            }
            song.extend_from_slice(&buf[..n]);
        }
        songs.push(lufs(&song));
    }
    let played = music.position() as f64 / f64::from(SAMPLE_RATE);
    let rtf = took / played;
    let offs: Vec<f64> = (songs.iter().enumerate())
        .map(|(k, &run)| (run - lufs(&audition::song(seed, k as u64, controls, &[]))).abs())
        .collect();
    let worst = worst(&offs);
    say(&format!(
        "{room} room, {:.1} min: mean real-time factor {rtf:.4} (gate {MEAN_RTF}); {} songs, every frame finite, each within {worst:.3} LU of itself played alone (gate {SONG_LEVEL_TOLERANCE_LU})",
        played / 60.0,
        songs.len()
    ));
    // `<=`, which the NaN a silent song's -inf LUFS leaves fails
    rtf <= MEAN_RTF && offs.iter().all(|&off| off <= SONG_LEVEL_TOLERANCE_LU)
}

/// The largest of `offs`, or NaN where one is: `f64::max` drops a NaN, which
/// the gate fails on, and the room's line would print a passing figure.
fn worst(offs: &[f64]) -> f64 {
    (offs.iter().copied()).fold(0.0, |a, b| if b.is_nan() || b > a { b } else { a })
}

/// `frames`' gated loudness, in LUFS.
fn lufs(frames: &[Frame]) -> f64 {
    let [left, right] = audition::channels(frames);
    audition::gated_lufs(&[&left, &right], SAMPLE_RATE)
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

/// What the host lends the bench: on wasm32, `lofi_bench.mjs`'s. That
/// target's std reads no clock, falling to an `Instant::now` that panics
/// (<https://github.com/rust-lang/rust/blob/1.98.1/library/std/src/sys/time/mod.rs#L47-L50>,
/// <https://github.com/rust-lang/rust/blob/1.98.1/library/std/src/sys/time/unsupported.rs#L12-L14>),
/// and its `println!` "does nothing"
/// (<https://doc.rust-lang.org/1.98.1/rustc/platform-support/wasm32-unknown-unknown.html>).
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

#[cfg(test)]
mod tests {
    use super::{least, worst};

    #[test]
    fn the_worst_offset_is_the_largest_and_a_nan_wins() {
        assert_eq!(worst(&[0.1, 0.3, 0.2]), 0.3);
        assert_eq!(worst(&[0.1, f64::INFINITY]), f64::INFINITY);
        assert!(worst(&[0.1, f64::NAN, 0.2]).is_nan());
        assert!(worst(&[f64::NAN, 0.2]).is_nan());
    }

    #[test]
    fn each_chunk_reads_at_its_least_over_the_runs() {
        // each run holds one chunk's least, so dropping any run moves one
        let runs = [
            vec![(0, 1.0), (10, 5.0), (20, 6.0)],
            vec![(0, 4.0), (10, 2.0), (20, 7.0)],
            vec![(0, 5.0), (10, 6.0), (20, 3.0)],
        ];
        assert_eq!(least(&runs), [(0, 1.0), (10, 2.0), (20, 3.0)]);
    }
}
