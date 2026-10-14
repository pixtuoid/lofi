//! A seed's songs, one after another: each note struck into its lane's ring
//! as the render reaches its onset, and each song played through chains of
//! its own, which ring on past its last bar until they rest.

use std::collections::VecDeque;

use super::lanes::Chain;
use crate::Controls;
use crate::compose::{Mood, Score, compose};
use crate::dsp::{NoiseStream, splitmix_draw};
use crate::synth::{CRACKLE_POPS_PER_SEC, Lane, Part, crackle_pop, part, samples_at};

const LANES: usize = Lane::ALL.len();

/// The most frames one step plays; any size plays the same frames.
pub(crate) const BLOCK: usize = 1024;

/// A song plays its loop this many times.
pub const LOOPS_PER_SONG: usize = 4;

/// The daylight below which a song is composed for the night.
const DUSK: f32 = 0.5;

/// Song `k`'s seed, which composes its score and seeds its noise: any song is
/// reached without composing the ones before it.
fn song_seed(seed: u64, k: u64) -> u64 {
    splitmix_draw(seed, k + 1)
}

/// The score song `k` of `seed`'s run plays under `controls`.
pub fn song_score(seed: u64, k: u64, controls: Controls) -> Score {
    let mood = if controls.daylight < DUSK {
        Mood::Night
    } else {
        Mood::Day
    };
    compose(mood, song_seed(seed, k))
}

/// The songs a seed composes, played one after another.
pub(crate) struct Schedule {
    seed: u64,
    sr: u32,
    /// The index of the next song to start.
    next_song: u64,
    /// The song playing; `None` before the first.
    song: Option<Song>,
    /// Songs past their last bar whose chains still sound.
    fading: Vec<Song>,
    /// Each lane's sum over the step just played.
    sums: [Vec<f64>; LANES],
    /// One song's lane over a step.
    lane: Vec<f64>,
}

impl Schedule {
    /// A schedule at `sr` whose first song is song `first` of `seed`'s run.
    pub(crate) fn new(seed: u64, first: u64, sr: u32) -> Self {
        Self {
            seed,
            sr,
            next_song: first,
            song: None,
            fading: Vec::new(),
            sums: std::array::from_fn(|_| Vec::with_capacity(BLOCK)),
            lane: Vec::with_capacity(BLOCK),
        }
    }

    /// Frames left in the playing song: 0 when the next song's first bar is
    /// due.
    pub(crate) fn song_left(&self) -> usize {
        self.song.as_ref().map_or(0, |song| song.len - song.played)
    }

    /// The score of the song playing.
    #[cfg(test)]
    pub(crate) fn playing(&self) -> Option<&Score> {
        self.song.as_ref().map(|song| &song.score)
    }

    /// Starts the next song, composed for `controls`, each lane mixed in at
    /// its gain in `levels`.
    pub(crate) fn start_song(&mut self, controls: Controls, levels: [f64; LANES]) {
        let song = Song::new(self.seed, self.next_song, controls, levels, self.sr);
        self.fading.extend(self.song.replace(song));
        self.next_song += 1;
    }

    /// Plays the next `n` frames, at most [`BLOCK`] and
    /// [`song_left`](Self::song_left), and returns each lane's sum over them,
    /// each song's at its level.
    pub(crate) fn play(&mut self, n: usize) -> &[Vec<f64>; LANES] {
        debug_assert!(n <= BLOCK && n <= self.song_left());
        for sum in &mut self.sums {
            sum.clear();
            sum.resize(n, 0.0);
        }
        for song in self.song.iter_mut().chain(&mut self.fading) {
            song.play(&mut self.sums, &mut self.lane, n);
        }
        // a song at rest adds only zeros, so dropping it moves no frame
        self.fading.retain(|song| !song.at_rest());
        &self.sums
    }
}

/// One song, played from its first bar.
struct Song {
    score: Score,
    sr: u32,
    /// Frames in the song, and played so far.
    len: usize,
    played: usize,
    /// Each lane's notes.
    cues: [Cues; LANES],
    /// Each lane's notes summed from the next frame on.
    rings: [Ring; LANES],
    chains: [Chain; LANES],
    crackle: Crackle,
    /// Each lane's gain into the mix: the song's own, which its tails keep
    /// past the next song's first bar.
    levels: [f64; LANES],
}

impl Song {
    fn new(seed: u64, k: u64, controls: Controls, levels: [f64; LANES], sr: u32) -> Self {
        let score = song_score(seed, k, controls);
        let (mood, seed) = (score.mood, song_seed(seed, k));
        // as long as `render_lane` renders a loop, so each note lands where it
        // always has
        let loop_len = samples_at(score.loop_secs(), sr);
        let len = LOOPS_PER_SONG * loop_len;
        let mut noise = NoiseStream::new(seed);
        let cues = Lane::ALL.map(|lane| Cues::new(&part(&score, lane), loop_len, noise.fork(), sr));
        // forked after every lane's, so it moves none of their draws
        let crackle = Crackle::new(noise.fork(), sr);
        // where the whole-lane night texture ducks, rounded as it rounds them
        let mut kicks: Vec<usize> = (score.kick_times.iter())
            .flat_map(|&at| every_loop(samples_at(at, sr), loop_len))
            .collect();
        kicks.sort_unstable();
        let chains =
            Lane::ALL.map(|lane| Chain::new(mood, lane, &part(&score, lane), len, &kicks, sr));
        Self {
            score,
            sr,
            len,
            played: 0,
            cues,
            rings: Default::default(),
            chains,
            crackle,
            levels,
        }
    }

    /// Strikes each note whose onset falls in the next `n` frames, then adds
    /// those frames of each lane, through its chain and at its level, into
    /// `sums`.
    fn play(&mut self, sums: &mut [Vec<f64>; LANES], buf: &mut Vec<f64>, n: usize) {
        let end = self.played + n;
        let lanes = (Lane::ALL.into_iter().zip(&mut self.cues))
            .zip(self.rings.iter_mut().zip(&mut self.chains))
            .zip(sums.iter_mut().zip(self.levels));
        for (((lane, cues), (ring, chain)), (sum, level)) in lanes {
            let part = part(&self.score, lane);
            cues.strike(&part, self.played, end, ring, self.sr);
            if matches!(part, Part::Bed) {
                self.crackle
                    .strike(self.played, end.min(self.len), ring, self.sr);
            }
            buf.resize(n, 0.0);
            ring.take(buf);
            chain.process(buf, &mut cues.noise);
            for (s, x) in sum.iter_mut().zip(buf.iter()) {
                *s += level * x;
            }
        }
        self.played = end;
    }

    /// Whether the song has played out: past its last bar, every tail taken
    /// and every chain at rest, so all it can add is silence.
    fn at_rest(&self) -> bool {
        self.played >= self.len
            && self.rings.iter().all(|ring| ring.0.is_empty())
            && self.chains.iter().all(Chain::at_rest)
    }
}

/// `at`, a frame in the loop, in each of the song's loops.
fn every_loop(at: usize, loop_len: usize) -> impl Iterator<Item = usize> {
    (0..LOOPS_PER_SONG).map(move |lap| lap * loop_len + at)
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
    fn new(part: &Part<'_>, loop_len: usize, noise: NoiseStream, sr: u32) -> Self {
        let mut order: Vec<(usize, usize)> = (part.onsets(sr).into_iter().enumerate())
            .flat_map(|(i, at)| every_loop(at, loop_len).map(move |at| (at, i)))
            .collect();
        order.sort_unstable();
        Self {
            order,
            struck: 0,
            noise,
        }
    }

    /// Strikes into `ring`, which starts at frame `from`, each note whose
    /// onset is before `to`.
    fn strike(&mut self, part: &Part<'_>, from: usize, to: usize, ring: &mut Ring, sr: u32) {
        while let Some(&(at, i)) = self.order.get(self.struck)
            && at < to
        {
            let (snippet, gain) = part.note(i, &mut self.noise, sr);
            ring.add(at - from, &snippet, gain);
            self.struck += 1;
        }
    }
}

/// The texture's vinyl crackle over one song: pops at Poisson times, drawn
/// from a stream of their own, so when a step strikes a pop moves no other
/// draw.
struct Crackle {
    rng: NoiseStream,
    /// The next pop's onset in the song.
    next: usize,
}

impl Crackle {
    fn new(mut rng: NoiseStream, sr: u32) -> Self {
        let next = Self::gap(&mut rng, sr);
        Self { rng, next }
    }

    /// Frames to the next pop: exponential, the distribution of "the distance
    /// between events in a Poisson point process", drawn by its quantile
    /// function −ln(1 − p)/λ (<https://en.wikipedia.org/wiki/Exponential_distribution>).
    fn gap(rng: &mut NoiseStream, sr: u32) -> usize {
        let secs = -f64::from(1.0 - rng.unit()).ln() / f64::from(CRACKLE_POPS_PER_SEC);
        (secs * f64::from(sr)) as usize
    }

    /// Strikes into `ring`, which starts at frame `from`, each pop before `to`.
    fn strike(&mut self, from: usize, to: usize, ring: &mut Ring, sr: u32) {
        while self.next < to {
            let (pop, gain) = crackle_pop(&mut self.rng, sr);
            ring.add(self.next - from, &pop, gain);
            self.next += Self::gap(&mut self.rng, sr);
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

/// Every lane at full level, as `render_lane` renders it.
#[cfg(test)]
pub(crate) const UNITY: [f64; LANES] = [1.0; LANES];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::SAMPLE_RATE;

    #[test]
    fn each_song_of_a_run_is_composed_anew() {
        let day = Controls::default();
        let scores: Vec<Score> = (0..8)
            .map(|k| Song::new(1, k, day, UNITY, SAMPLE_RATE).score)
            .collect();
        for (k, a) in scores.iter().enumerate() {
            assert!(!scores[k + 1..].contains(a), "song {k} recurs");
            assert!(
                *a != Song::new(2, k as u64, day, UNITY, SAMPLE_RATE).score,
                "seeds 1 and 2 share song {k}"
            );
        }
    }

    #[test]
    fn a_run_plays_each_songs_score() {
        for c in [Controls::default(), Controls::default().with_daylight(0.0)] {
            let mut run = Schedule::new(6, 0, SAMPLE_RATE);
            for k in 0..3 {
                run.start_song(c, UNITY);
                let played = run.song.as_ref().map(|song| &song.score);
                assert_eq!(played, Some(&song_score(6, k, c)), "song {k} at {c:?}");
            }
        }
    }

    #[test]
    fn a_run_is_its_songs_each_played_alone() {
        let (seed, c, sr) = (3, Controls::default(), SAMPLE_RATE);
        // levels that differ in every lane, so a tail heard at the next
        // song's levels shows
        let levels = [UNITY, [0.5; LANES]];
        let mut run = Schedule::new(seed, 0, sr);
        let mut first = Song::new(seed, 0, c, levels[0], sr);
        let mut second = Song::new(seed, 1, c, levels[1], sr);
        let mut buf = Vec::new();
        // through song 1's first loop, past every tail of song 0
        let span = first.len + second.len / LOOPS_PER_SONG;
        let mut at = 0;
        let mut levels = levels.into_iter();
        while at < span {
            if run.song_left() == 0 {
                run.start_song(c, levels.next().expect("two songs start"));
            }
            let n = run.song_left().min(BLOCK);
            let mut alone: [Vec<f64>; LANES] = std::array::from_fn(|_| vec![0.0; n]);
            // as the run sums them: the playing song, then the fading one,
            // which plays here past its rest rather than being dropped
            if at >= first.len {
                second.play(&mut alone, &mut buf, n);
            }
            first.play(&mut alone, &mut buf, n);
            for ((lane, got), want) in Lane::ALL.iter().zip(run.play(n)).zip(&alone) {
                let diff = got
                    .iter()
                    .zip(want)
                    .position(|(g, w)| g.to_bits() != w.to_bits());
                assert_eq!(
                    diff.map(|i| at + i),
                    None,
                    "{lane:?}: the first frame that differs"
                );
            }
            at += n;
        }
        assert!(
            run.fading.is_empty(),
            "song 0 never rested, so the run keeps playing its chains"
        );
    }

    #[test]
    fn a_night_bed_ducks_after_each_kick() {
        let sr = SAMPLE_RATE;
        let night = Controls::default().with_daylight(0.0);
        let mut song = Song::new(4, 0, night, UNITY, sr);
        let loop_len = song.len / LOOPS_PER_SONG;
        let texture = Lane::ALL.iter().position(|&l| l == Lane::Texture).unwrap();
        let (mut bed, mut buf) = (Vec::with_capacity(loop_len), Vec::new());
        while bed.len() < loop_len {
            let n = BLOCK.min(loop_len - bed.len());
            let mut sums: [Vec<f64>; LANES] = std::array::from_fn(|_| vec![0.0; n]);
            song.play(&mut sums, &mut buf, n);
            bed.extend_from_slice(&sums[texture]);
        }
        let window = samples_at(0.05, sr);
        let energy = |from: usize| bed[from..from + window].iter().map(|v| v * v).sum::<f64>();
        let kicks: Vec<usize> = (song.score.kick_times.iter())
            .map(|&at| samples_at(at, sr))
            .filter(|&k| k >= window)
            .collect();
        assert!(!kicks.is_empty());
        let before: f64 = kicks.iter().map(|&k| energy(k - window)).sum();
        let after: f64 = kicks.iter().map(|&k| energy(k)).sum();
        let ratio = (after / before).sqrt();
        assert!(ratio < 0.85, "the bed must dip after the kicks: {ratio:.3}");
    }

    #[test]
    fn crackle_pops_at_its_rate_at_poisson_times() {
        let secs = 3600.0;
        // a second rate, so a gap or a pop that counts `SAMPLE_RATE`'s frames
        // misses it
        let [ratified, at] = [SAMPLE_RATE, 96_000].map(|sr| {
            let mut crackle = Crackle::new(NoiseStream::new(5), sr);
            let (mut gaps, mut last, mut pops) = (Vec::new(), 0, Vec::new());
            while crackle.next < samples_at(secs, sr) {
                gaps.push((crackle.next - last) as f64 / f64::from(sr));
                last = crackle.next;
                let mut ring = Ring::default();
                crackle.strike(last, last + 1, &mut ring, sr);
                let mut pop = vec![0.0; ring.0.len()];
                ring.take(&mut pop);
                pops.extend(pop.iter().map(|&v| v as f32));
            }
            // a Poisson count's standard deviation is the square root of its mean
            let want = f64::from(CRACKLE_POPS_PER_SEC * secs);
            let count = gaps.len() as f64;
            assert!(
                (count - want).abs() < 4.0 * want.sqrt(),
                "{count} pops in an hour at {sr} Hz vs {want}"
            );
            // an exponential's standard deviation is its mean; a grid's is 0
            let mean = gaps.iter().sum::<f64>() / count;
            let sd = (gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / count).sqrt();
            assert!((sd / mean - 1.0).abs() < 0.15, "gaps vary by {}", sd / mean);
            // the pops' power over every third octave, which averages more of
            // their draws than any one band does
            let power: f64 = third_octaves(&pops, sr)
                .iter()
                .map(|db| 10f64.powf(db / 10.0))
                .sum();
            10.0 * power.log10()
        });
        assert!(
            (at - ratified).abs() < 0.5,
            "an hour's pops play at {at:.2} dB at 96 kHz, {ratified:.2} dB at {SAMPLE_RATE} Hz"
        );
    }

    /// Seed 0's first song under `c` at `sr`: its loop's length in seconds,
    /// and each lane's first loop.
    fn lanes_at(c: Controls, sr: u32) -> (f64, [Vec<f32>; LANES]) {
        let mut run = Schedule::new(0, 0, sr);
        run.start_song(c, UNITY);
        let loop_len = run.song_left() / LOOPS_PER_SONG;
        let mut lanes: [Vec<f32>; LANES] = Default::default();
        while lanes[0].len() < loop_len {
            let n = BLOCK.min(loop_len - lanes[0].len());
            for (lane, sum) in lanes.iter_mut().zip(run.play(n)) {
                lane.extend(sum.iter().map(|&v| v as f32));
            }
        }
        (loop_len as f64 / f64::from(sr), lanes)
    }

    /// `buf`'s mean square at `sr` in each third octave from 25 Hz to 4 kHz,
    /// in dB, which a sound keeps however it is sampled.
    fn third_octaves(buf: &[f32], sr: u32) -> Vec<f64> {
        let n = buf.len().next_power_of_two();
        let (mut re, mut im) = (buf.to_vec(), vec![0.0; n]);
        re.resize(n, 0.0);
        crate::dsp::fft(&mut re, &mut im, &crate::dsp::twiddles(n), false);
        let hz_per_bin = f64::from(sr) / n as f64;
        let edge = 2f64.powf(1.0 / 6.0);
        (0..=22)
            .map(|k| {
                let centre = 25.0 * 2f64.powf(f64::from(k) / 3.0);
                let bins = (centre / edge / hz_per_bin).ceil() as usize
                    ..=(centre * edge / hz_per_bin) as usize;
                let power: f64 = (re[bins.clone()].iter().zip(&im[bins]))
                    .map(|(&r, &i)| f64::from(r).powi(2) + f64::from(i).powi(2))
                    .sum();
                // Parseval's, the mirrored half included
                10.0 * (2.0 * power / (n * buf.len()) as f64).log10()
            })
            .collect()
    }

    /// `buf`'s mean square at `sr` over each 50 ms, in dB.
    fn envelope(buf: &[f32], sr: u32) -> Vec<f64> {
        let window = samples_at(0.05, sr);
        (buf.chunks_exact(window))
            .map(|w| {
                10.0 * (w.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / window as f64)
                    .log10()
            })
            .collect()
    }

    #[test]
    fn a_song_at_96_khz_plays_as_it_does_at_the_ratified_rate() {
        const RATE: u32 = 96_000;
        // per lane in `Lane::ALL` order, how far a third octave, and the
        // envelope at its 90th percentile, may stray between the rates: just
        // past what they stray by, the noise lanes' draws being apart
        // (opinion)
        const STRAY_DB: [(f64, f64); LANES] = [
            (0.5, 0.08),
            (1.0, 0.2),
            (0.5, 0.08),
            (2.0, 1.6),
            (1.2, 0.7),
            (0.5, 0.05),
        ];
        // a band or a window this far under the lane's loudest is left out,
        // too quiet to compare (opinion)
        const FLOOR_DB: f64 = 30.0;
        let loudest = |levels: &[f64]| levels.iter().copied().fold(f64::MIN, f64::max);
        let mut off = Vec::new();
        for c in [Controls::default(), Controls::default().with_daylight(0.0)] {
            let [(ratified_loop, ratified), (loop_at, at)] =
                [SAMPLE_RATE, RATE].map(|sr| lanes_at(c, sr));
            // each loop rounds to a whole frame at its rate
            if (ratified_loop - loop_at).abs() > 1.0 / f64::from(SAMPLE_RATE) {
                off.push(format!(
                    "{c:?}: a loop lasts {loop_at} s, not {ratified_loop} s"
                ));
            }
            let lanes = Lane::ALL.iter().zip(ratified.iter().zip(&at)).zip(STRAY_DB);
            for ((lane, (ratified, at)), (band_db, envelope_db)) in lanes {
                let (want, got) = (
                    third_octaves(ratified, SAMPLE_RATE),
                    third_octaves(at, RATE),
                );
                let top = loudest(&want);
                for (k, (w, g)) in want.iter().zip(&got).enumerate() {
                    if *w > top - FLOOR_DB && (g - w).abs() > band_db {
                        off.push(format!(
                            "{c:?} {lane:?}, third octave {k}: {g:.2} vs {w:.2} dB"
                        ));
                    }
                }
                let (want, got) = (envelope(ratified, SAMPLE_RATE), envelope(at, RATE));
                let top = loudest(&want);
                let mut strays: Vec<f64> = (want.iter().zip(&got))
                    .filter(|(w, _)| **w > top - FLOOR_DB)
                    .map(|(w, g)| (g - w).abs())
                    .collect();
                strays.sort_by(f64::total_cmp);
                let stray = strays[(strays.len() - 1) * 9 / 10];
                if stray > envelope_db {
                    off.push(format!("{c:?} {lane:?}: the envelope strays {stray:.2} dB"));
                }
            }
        }
        assert!(
            off.is_empty(),
            "the song plays apart at {RATE} Hz: {off:#?}"
        );
    }
}
