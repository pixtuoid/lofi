//! `Performance`'s heap, counted by a global allocator: one test in a binary
//! of its own, so no other test's allocations land in the count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use pixtuoid_lofi::dsp::SAMPLE_RATE;
use pixtuoid_lofi::{Controls, Frame, Performance, audition};

/// The most heap a `Performance` may hold at once, in bytes: the highest
/// high-water this test measured plus 25 %, both of which the commit that set
/// it records.
const BUDGET: usize = 8_424_525;

/// Bytes allocated and not yet freed, and the most there have been.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// The system allocator, counting.
struct Counting;

#[expect(
    unsafe_code,
    reason = "a `GlobalAlloc` is an unsafe trait, and only one sees every allocation"
)]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let live = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
        PEAK.fetch_max(live, Relaxed);
        // SAFETY: `alloc`'s contract is the caller's, passed on unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        // SAFETY: `ptr` came from `alloc` above, which `System` served.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// `frames`' bits, hashed without a heap, so a chunk is checked after the
/// count without copying it during it.
fn fingerprint(frames: &[Frame]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for frame in frames {
        frame.map(f32::to_bits).hash(&mut hasher);
    }
    hasher.finish()
}

/// The most heap a `Performance` holds playing song 0 of `seed`'s run under
/// `a`, then song 1 under `b`; panics unless it played those two songs.
fn high_water(seed: u64, a: Controls, b: Controls) -> usize {
    let span = audition::song_len(seed, 0, a) + audition::song_len(seed, 1, b);
    // a chunk as a player renders ahead, allocated before the count starts
    let mut chunk = vec![Frame::default(); SAMPLE_RATE as usize / 2];
    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    let mut music = Performance::new(seed, a);
    let (mut opening, mut closing) = ((0, 0), (0, 0));
    while music.position() < span as u64 {
        let at = music.position();
        let n = chunk.len().min(span - at as usize);
        music.render(&mut chunk[..n]);
        if at == 0 {
            opening = (n, fingerprint(&chunk[..n]));
            // a song latches the controls asked for before its first frame,
            // so `b` is asked for once `a`'s song plays
            music.set_controls(b);
        }
        closing = (n, fingerprint(&chunk[..n]));
    }
    let high_water = PEAK.load(Relaxed) - before;
    drop(music);
    let mut first_alone = vec![Frame::default(); opening.0];
    Performance::new(seed, a).render(&mut first_alone);
    // past its first loop, song 1 plays as it does alone
    let second_alone = audition::song(seed, 1, b);
    assert_eq!(
        (
            fingerprint(&first_alone),
            fingerprint(&second_alone[second_alone.len() - closing.0..])
        ),
        (opening.1, closing.1),
        "seed {seed} didn't play song 0 under {a:?}, then song 1 under {b:?}"
    );
    high_water
}

#[test]
fn a_performance_plays_two_songs_and_a_transition_within_its_heap_budget() {
    let busy_day = Controls::default().with_intensity(1.0);
    let night = Controls::default().with_daylight(0.0);
    // a night song after a busy day's, and the heaviest pair the sweep that
    // set the budget found
    let runs = [
        (6, busy_day, busy_day.with_daylight(0.0)),
        (6, night, night),
    ];
    let over: Vec<String> = runs
        .into_iter()
        .map(|(seed, a, b)| (seed, a, b, high_water(seed, a, b)))
        .filter(|&(.., bytes)| bytes > BUDGET)
        .map(|(seed, a, b, bytes)| format!("seed {seed}, {a:?} into {b:?}: {bytes} bytes"))
        .collect();
    assert!(
        over.is_empty(),
        "the heap went over the {BUDGET}-byte budget: {over:#?}"
    );
}
