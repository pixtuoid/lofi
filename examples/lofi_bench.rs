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

#[path = "common/loudness.rs"]
mod loudness;

use std::hint::black_box;
use std::ops::Range;
use std::process::ExitCode;

use pixtuoid_lofi::audition::{self, Edge, TIER_INTENSITIES, WHOLE_LANE_LUFS_SD};
use pixtuoid_lofi::dsp::SAMPLE_RATE;
use pixtuoid_lofi::{Controls, Frame, Performance};

use host::{Clock, machine, say};
use loudness::{fold, gated_lufs};

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

/// The daylight the timeline's day song grades to: halfway from
/// [`Controls::DUSK`] to full day, so the song grades within the day family
/// rather than past it.
const MID_SONG_DAYLIGHT: f32 = f32::midpoint(Controls::DUSK, 1.0);

/// How long each room plays, in seconds, to the end of its last song.
/// The mean settles within minutes, and this spans many song transitions.
const ROOM_S: u32 = 20 * 60;

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

/// The longest one `render` of a `buf`-long chunk took, in seconds, over each
/// of [`SEEDS`]' first two songs: a busy day's, whose bars empty the room,
/// fill it again and grade toward night, then a rainy night's, whose first
/// bars the day's tails ring over. With it, the seed and the frame the chunk started
/// at, which tell a cold start from a transition. Errs unless the run played
/// those two songs.
fn slowest_chunk(buf: &mut [Frame]) -> Result<(f64, u64, u64), String> {
    let day = Controls::default().with_intensity(1.0);
    let night = day.with_daylight(0.0).with_rain(1.0);
    // a song that mutes no lane or no song that fades or filters would time
    // a render of less than the forms the run plays
    let forms: Vec<_> = (SEEDS.flat_map(|seed| [(seed, 0, day), (seed, 1, night)]))
        .map(|(seed, k, c)| audition::song_score(seed, k, c).form().clone())
        .collect();
    let edged = |f: &audition::Form| {
        f.entry() != Edge::Plain || f.exit() != Edge::Plain || f.dip().is_some()
    };
    if forms.iter().any(|f| f.intro() + f.outro() == 0) || !forms.iter().any(edged) {
        return Err(format!("the timed songs play too little form: {forms:?}"));
    }
    let mut worst = (0.0, SEEDS.start, 0);
    for seed in SEEDS {
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
        let (mut opening, mut last) = (Vec::new(), 0);
        while music.position() < span as u64 {
            let at = music.position();
            let until = calls.peek().map_or(span, |&&(call, _)| call);
            let n = buf.len().min(until - at as usize);
            let clock = Clock::start();
            music.render(&mut buf[..n]);
            let took = clock.secs();
            black_box(&buf);
            if took > worst.0 {
                worst = (took, seed, at);
            }
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
        let mut day_alone = vec![Frame::default(); opening.len()];
        Performance::new(seed, day).render(&mut day_alone);
        // past its first loop, song 1 plays as it does alone
        let night_alone = audition::song(seed, 1, night, &[]);
        if day_alone != opening || night_alone[night_alone.len() - last..] != buf[..last] {
            return Err(format!(
                "seed {seed} didn't play a day's song, then a rainy night's"
            ));
        }
    }
    Ok(worst)
}

/// Plays [`ROOM_S`] of `room` under `controls` in `buf`-long chunks, and
/// says whether it passed its gates: the mean real-time factor within
/// [`MEAN_RTF`], every frame finite, and its songs' gated loudness varying
/// within [`audition::lufs_sd_bound`] of the whole-lane mix's at its controls.
fn play_room(buf: &mut [Frame], room: &str, controls: Controls) -> bool {
    let Some(&(.., whole_lane)) = (WHOLE_LANE_LUFS_SD.iter())
        .find(|&&(i, d, _)| i == controls.intensity && d == controls.daylight)
    else {
        say(&format!(
            "{room} room: no whole-lane loudness to hold it to"
        ));
        return false;
    };
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
            song.extend(fold(&buf[..n]));
        }
        songs.push(gated_lufs(&song, SAMPLE_RATE));
    }
    let played = music.position() as f64 / f64::from(SAMPLE_RATE);
    let rtf = took / played;
    let (sd, bound) = (
        audition::lufs_sd(&songs),
        audition::lufs_sd_bound(whole_lane),
    );
    say(&format!(
        "{room} room, {:.1} min: mean real-time factor {rtf:.4} (gate {MEAN_RTF}); {} songs, every frame finite, gated loudness sd {sd:.4} LU (gate {bound:.4})",
        played / 60.0,
        songs.len()
    ));
    // `<=`, which the NaN sd a silent song's -inf LUFS makes fails
    rtf <= MEAN_RTF && sd <= bound
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
