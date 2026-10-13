//! A seed's songs, one after another, each note struck into its lane's ring
//! as the render reaches its onset.

use std::collections::VecDeque;

use crate::Controls;
use crate::compose::{Mood, Score, compose};
use crate::dsp::{NoiseStream, splitmix_draw};
use crate::synth::{Lane, Part, n_samples, part};

const LANES: usize = Lane::ALL.len();

/// The most frames one step plays; any size plays the same frames.
pub(crate) const BLOCK: usize = 1024;

/// A song plays its loop this many times.
pub(crate) const LOOPS_PER_SONG: usize = 4;

/// The daylight below which a song is composed for the night.
const DUSK: f32 = 0.5;

/// Song `k`'s seed, which composes its score and seeds its noise: any song is
/// reached without composing the ones before it.
fn song_seed(seed: u64, k: u64) -> u64 {
    splitmix_draw(seed, k + 1)
}

/// The songs a seed composes, played one after another.
pub(crate) struct Schedule {
    seed: u64,
    /// The index of the next song to start.
    next_song: u64,
    /// The song playing; `None` before the first.
    song: Option<Song>,
    /// Each lane's notes summed from the next frame on, which ring on past the
    /// song that struck them.
    rings: [Ring; LANES],
    /// Each lane's sum over the step just played.
    sums: [Vec<f64>; LANES],
}

impl Schedule {
    /// A schedule whose first song is song `first` of `seed`'s run.
    pub(crate) fn new(seed: u64, first: u64) -> Self {
        Self {
            seed,
            next_song: first,
            song: None,
            rings: Default::default(),
            sums: std::array::from_fn(|_| Vec::with_capacity(BLOCK)),
        }
    }

    /// Frames left in the playing song: 0 when the next song's first bar is
    /// due.
    pub(crate) fn song_left(&self) -> usize {
        self.song.as_ref().map_or(0, |song| song.len - song.played)
    }

    /// Starts the next song, composed for `controls`.
    pub(crate) fn start_song(&mut self, controls: Controls) {
        self.song = Some(Song::new(self.seed, self.next_song, controls));
        self.next_song += 1;
    }

    /// Plays the next `n` frames, at most [`BLOCK`] and
    /// [`song_left`](Self::song_left), and returns each lane's sum over them.
    pub(crate) fn play(&mut self, n: usize) -> &[Vec<f64>; LANES] {
        debug_assert!(n <= BLOCK && n <= self.song_left());
        if let Some(song) = &mut self.song {
            song.play(&mut self.rings, &mut self.sums, n);
        }
        &self.sums
    }
}

/// One song, played from its first bar.
struct Song {
    score: Score,
    /// Frames in the song, and played so far.
    len: usize,
    played: usize,
    /// Each lane's notes.
    cues: [Cues; LANES],
}

impl Song {
    fn new(seed: u64, k: u64, controls: Controls) -> Self {
        let seed = song_seed(seed, k);
        let mood = if controls.daylight < DUSK {
            Mood::Night
        } else {
            Mood::Day
        };
        let score = compose(mood, seed);
        // as long as `render_lane` renders a loop, so each note lands where it
        // always has
        let loop_len = n_samples(score.loop_secs());
        let mut noise = NoiseStream::new(seed);
        let cues = Lane::ALL.map(|lane| Cues::new(&part(&score, lane), loop_len, noise.fork()));
        Self {
            score,
            len: LOOPS_PER_SONG * loop_len,
            played: 0,
            cues,
        }
    }

    /// Strikes each note whose onset falls in the next `n` frames, then moves
    /// those frames of each lane into `sums`.
    fn play(&mut self, rings: &mut [Ring; LANES], sums: &mut [Vec<f64>; LANES], n: usize) {
        let end = self.played + n;
        let lanes = Lane::ALL.into_iter().zip(&mut self.cues).zip(rings);
        for (((lane, cues), ring), sum) in lanes.zip(sums) {
            let part = part(&self.score, lane);
            while let Some(&(at, i)) = cues.order.get(cues.struck)
                && at < end
            {
                let (snippet, gain) = part.note(i, &mut cues.noise);
                ring.add(at - self.played, &snippet, gain);
                cues.struck += 1;
            }
            sum.resize(n, 0.0);
            ring.take(sum);
            if matches!(part, Part::Bed) {
                for v in sum.iter_mut() {
                    *v += f64::from(cues.noise.norm());
                }
            }
        }
        self.played = end;
    }
}

/// A lane's notes over one song, in the order they sound: the order its
/// noise is drawn in, whatever the chunking.
struct Cues {
    /// `(onset in the song, index into the lane's events)`, by onset.
    order: Vec<(usize, usize)>,
    /// Notes struck so far.
    struck: usize,
    /// The lane's draws of the song's noise.
    noise: NoiseStream,
}

impl Cues {
    fn new(part: &Part<'_>, loop_len: usize, noise: NoiseStream) -> Self {
        let onsets = part.onsets();
        let mut order: Vec<(usize, usize)> = (0..LOOPS_PER_SONG)
            .flat_map(|lap| {
                onsets
                    .iter()
                    .enumerate()
                    .map(move |(i, &at)| (lap * loop_len + at, i))
            })
            .collect();
        order.sort_unstable();
        Self {
            order,
            struck: 0,
            noise,
        }
    }
}

/// One lane's notes summed from the next frame to play on.
#[derive(Default)]
struct Ring(VecDeque<f32>);

impl Ring {
    /// Mixes `snippet` in at `gain`, `offset` frames on, as `render_lane`
    /// places a note.
    fn add(&mut self, offset: usize, snippet: &[f32], gain: f32) {
        let end = offset + snippet.len();
        if self.0.len() < end {
            self.0.resize(end, 0.0);
        }
        for (slot, &s) in self.0.range_mut(offset..end).zip(snippet) {
            *slot += s * gain;
        }
    }

    /// Moves the next `out.len()` frames into `out`: silence past the last
    /// tail.
    fn take(&mut self, out: &mut [f64]) {
        let n = out.len().min(self.0.len());
        for (o, s) in out.iter_mut().zip(self.0.drain(..n)) {
            *o = f64::from(s);
        }
        out[n..].fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_song_of_a_run_is_composed_anew() {
        let day = Controls::default();
        let scores: Vec<Score> = (0..8).map(|k| Song::new(1, k, day).score).collect();
        for (k, a) in scores.iter().enumerate() {
            assert!(!scores[k + 1..].contains(a), "song {k} recurs");
            assert!(
                *a != Song::new(2, k as u64, day).score,
                "seeds 1 and 2 share song {k}"
            );
        }
    }
}
