//! `Performance`'s heap, counted by a global allocator: one test in a binary
//! of its own, so no other test's allocations land in the count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use pixtuoid_lofi::dsp::SAMPLE_RATE;
use pixtuoid_lofi::{Controls, Frame, Performance, audition};

/// The most heap a `Performance` may hold at once, in bytes: the high-water
/// this test measured plus 25 %, both of which the commit that set it records.
const BUDGET: usize = 7_720_305;

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

#[test]
fn a_performance_plays_two_songs_and_a_transition_within_its_heap_budget() {
    // a night song after a busy, rainy day's, the heaviest pair the sweep that
    // set the budget found
    let (seed, day) = (6, Controls::default().with_intensity(1.0).with_rain(1.0));
    let night = day.with_daylight(0.0);
    let span = audition::song(seed, 0, day).len() + audition::song(seed, 1, night).len();
    // a chunk as a player renders ahead, allocated before the count starts
    let mut chunk = vec![Frame::default(); SAMPLE_RATE as usize / 2];
    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    let mut music = Performance::new(seed, day);
    music.set_controls(night);
    while music.position() < span as u64 {
        let n = chunk.len().min(span - music.position() as usize);
        music.render(&mut chunk[..n]);
    }
    let high_water = PEAK.load(Relaxed) - before;
    assert!(
        high_water <= BUDGET,
        "the heap reached {high_water} bytes, over the {BUDGET}-byte budget"
    );
}
