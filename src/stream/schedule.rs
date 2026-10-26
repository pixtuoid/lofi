//! A seed's songs, one after another: each note struck into its lane's ring
//! as the render reaches its onset, and each song played through chains of
//! its own, which ring on past its last bar until they rest.

use std::collections::VecDeque;
use std::ops::Range;

use super::lanes::{Automation, Chain, Sweep};
use super::master::{lane_gain, resting_gain};
use super::plate::{self, Plate};
use super::rain::{self, RAIN_RAMP_BARS};
use super::ramp::{Ramp, Shape};
use super::{BLOCK, LANES, gain_of_db};
use crate::Controls;
use crate::compose::{
    Edge, Form, GEN_LOOP_BARS, HAT_JITTER_S, Layer, Mood, Score, arrange, compose,
};
use crate::dsp::{NoiseStream, SAMPLE_RATE, splitmix_draw};
use crate::synth::{CRACKLE_POPS_PER_SEC, Lane, Part, bar_onset, crackle_pop, part, samples_at};

/// How long before its bar line a bar latches the controls asked for: the
/// composer's earliest anticipation, a downbeat hat's half of
/// [`HAT_JITTER_S`], so an anticipated downbeat joins the bar it anticipates.
pub(crate) const ANTICIPATION_S: f32 = HAT_JITTER_S / 2.0;

/// The level a fading edge starts a music lane from or ends it at, in dB:
/// silence a `Shape::Db` ramp can start from. Opinion.
pub(crate) const FADE_FLOOR_DB: f64 = -36.0;

/// How long, in beats, a lane's level takes to reach the one a latch asks
/// for: no longer than a beat, which gated lanes fade over at most. Opinion.
const LEVEL_RAMP_BEATS: f32 = 1.0;

/// How long, in bars, a lane's colour takes to grade to the daylight a latch
/// asks for. Opinion.
const GRADE_RAMP_BARS: f32 = 1.0;

/// Song `k`'s seed, which composes its score and seeds its noise: any song is
/// reached without composing the ones before it.
fn song_seed(seed: u64, k: u64) -> u64 {
    splitmix_draw(seed, k + 1)
}

/// The score song `k` of `seed`'s run plays under `controls`.
pub fn song_score(seed: u64, k: u64, controls: Controls) -> Score {
    let mood = if controls.daylight < Controls::DUSK {
        Mood::Night
    } else {
        Mood::Day
    };
    compose(mood, song_seed(seed, k))
}

/// The form song `k` of `seed`'s run plays under `controls`, which its first
/// bar latches: the one it draws, or with no beat at their intensity that
/// form's `beatless` one.
pub fn song_form(seed: u64, k: u64, controls: Controls) -> Form {
    let form = arrange(song_seed(seed, k));
    if lane_gain(Lane::Drums, controls.intensity) > 0.0 {
        form
    } else {
        form.beatless()
    }
}

/// The window song `k` of `seed`'s run plays under `controls` at `sr`.
fn song_window(seed: u64, k: u64, controls: Controls, sr: u32) -> Window {
    Window::new(
        &song_score(seed, k, controls),
        &song_form(seed, k, controls),
        sr,
    )
}

/// The frame each bar of song `k` of `seed`'s run starts on under
/// `controls`, at [`SAMPLE_RATE`]: the song's own bar grid.
pub fn bar_frames(seed: u64, k: u64, controls: Controls) -> Vec<usize> {
    song_window(seed, k, controls, SAMPLE_RATE).bars
}

/// How many frames song `k` of `seed`'s run lasts under `controls` at `sr`:
/// where the next song's first bar falls.
pub(crate) fn song_len(seed: u64, k: u64, controls: Controls, sr: u32) -> usize {
    song_window(seed, k, controls, sr).len
}

/// Each bar's first frame in the song `score` plays as `form` at `sr`, and
/// the song's length.
#[cfg(test)]
pub(crate) fn bars_and_len(score: &Score, form: &Form, sr: u32) -> (Vec<usize>, usize) {
    let window = Window::new(score, form, sr);
    (window.bars, window.len)
}

/// How many bars song `k` of `seed`'s run lasts.
pub fn song_bars(seed: u64, k: u64) -> usize {
    arrange(song_seed(seed, k)).bars()
}

/// The frame bar `k` of the endless tiling of a loop `loop_len` long starts
/// on: each loop's bars where `render_lane` places them, loop after loop.
fn tiled(k: usize, bar_s: f32, loop_len: usize, sr: u32) -> usize {
    (k / GEN_LOOP_BARS) * loop_len + bar_onset(k % GEN_LOOP_BARS, bar_s, sr)
}

/// The stretch of its loop's endless tiling a song plays: from its form's
/// lead-in bar, as many bars as its form has.
struct Window {
    /// Where the song starts in the tiling, and the tiling's loop length.
    start: usize,
    loop_len: usize,
    /// Each bar's first frame in the song, and the song's length.
    bars: Vec<usize>,
    len: usize,
    /// How far before its line a bar latches.
    anticipation: usize,
    /// The frames the form's intro, body and outro span, and its dip's bar.
    parts: [Range<usize>; 3],
    dip: Option<Range<usize>>,
}

impl Window {
    fn new(score: &Score, form: &Form, sr: u32) -> Self {
        let bar_s = score.bar_s();
        let loop_len = samples_at(score.loop_secs(), sr);
        let tile = |bar: usize| tiled(form.lead_in() + bar, bar_s, loop_len, sr);
        let start = tile(0);
        // a bar's line in the song, the end of the song past its last bar
        let at = |bar: usize| tile(bar) - start;
        let span = |bars: Range<usize>| at(bars.start)..at(bars.end);
        Self {
            start,
            loop_len,
            bars: (0..form.bars()).map(at).collect(),
            len: at(form.bars()),
            anticipation: samples_at(ANTICIPATION_S, sr),
            parts: form.parts().map(span),
            dip: form.dip().map(|bar| span(bar..bar + 1)),
        }
    }

    /// The frame bar `bar` starts on, or the song's end past its last bar.
    fn line(&self, bar: usize) -> usize {
        self.bars.get(bar).copied().unwrap_or(self.len)
    }

    /// The frame bar `bar` latches on.
    fn latch(&self, bar: usize) -> usize {
        self.latch_of(self.line(bar))
    }

    /// The frame a bar starting on `line` latches on: the line less the
    /// anticipation, from the song's first frame on.
    fn latch_of(&self, line: usize) -> usize {
        line.saturating_sub(self.anticipation)
    }

    /// Each of `onsets`, frames in the loop, in every loop the window spans:
    /// `(frame in the song, bar, index into onsets)` for each in the song. A
    /// note belongs to the bar whose latch frame is the last at or before it,
    /// as a latch hears it; one anticipating the song's first bar plays on its
    /// first frame, as `compose` clamps a downbeat hat to its loop's start.
    fn tile(&self, onsets: &[usize]) -> Vec<(usize, usize, usize)> {
        let laps = (self.start + self.len).div_ceil(self.loop_len);
        let mut tiled = Vec::new();
        for lap in 0..laps {
            for (i, &at) in onsets.iter().enumerate() {
                let Some(frame) = (lap * self.loop_len + at + self.anticipation)
                    .checked_sub(self.start)
                    .map(|late| late.saturating_sub(self.anticipation))
                else {
                    continue;
                };
                // the song's end line is the next song's first latch
                if frame >= self.latch(self.bars.len()) {
                    continue;
                }
                let bar = self
                    .bars
                    .partition_point(|&line| self.latch_of(line) <= frame)
                    - 1;
                tiled.push((frame, bar, i));
            }
        }
        tiled.sort_unstable();
        tiled
    }
}

/// A step's working buffers, sized to [`BLOCK`] once when the schedule is
/// made, as CLAP's `activate` is where "the plugin may allocate memory and
/// prepare everything needed for the process call"
/// (<https://github.com/free-audio/clap/blob/main/include/clap/plugin.h>):
/// songs play one step after another, so one serves them all.
pub(crate) struct Scratch {
    /// A lane's frames over a step, its gains, and its chain's own work.
    lane: Vec<f64>,
    gains: Vec<f64>,
    work: Vec<f64>,
    /// The music lanes' edge gain on each frame where an edge ramps.
    edged: Vec<f64>,
    /// The plate's send and its stereo return.
    send: Vec<f64>,
    wet: [Vec<f64>; 2],
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

impl Scratch {
    pub(crate) fn new() -> Self {
        let block = || Vec::with_capacity(BLOCK);
        Self {
            lane: block(),
            gains: block(),
            work: block(),
            edged: block(),
            send: block(),
            wet: [block(), block()],
        }
    }
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
    /// The plates' stereo return over the step just played.
    plates: [Vec<f64>; 2],
    scratch: Scratch,
    /// The form every song plays instead of its own, for a test of a
    /// mechanism a form would mute.
    #[cfg(test)]
    form: Option<Form>,
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
            plates: std::array::from_fn(|_| Vec::with_capacity(BLOCK)),
            scratch: Scratch::new(),
            #[cfg(test)]
            form: None,
        }
    }

    /// The schedule with every song playing `form` instead of its own.
    #[cfg(test)]
    pub(crate) fn with_form(mut self, form: Form) -> Self {
        self.form = Some(form);
        self
    }

    /// A schedule whose every song plays [`Form::plain`], for a test of a
    /// mechanism a form would mute.
    #[cfg(test)]
    pub(crate) fn plain(seed: u64, first: u64, sr: u32) -> Self {
        Self::new(seed, first, sr).with_form(Form::plain())
    }

    /// The next song, composed for `controls`, its first bar latching them.
    fn next_song(&self, controls: Controls) -> Song {
        let form = song_form(self.seed, self.next_song, controls);
        #[cfg(test)]
        let form = self.form.clone().unwrap_or(form);
        Song::of(
            song_score(self.seed, self.next_song, controls),
            form,
            song_seed(self.seed, self.next_song),
            controls,
            self.sr,
        )
    }

    /// Frames left in the playing song: 0 when the next song's first bar is
    /// due.
    pub(crate) fn song_left(&self) -> usize {
        self.song
            .as_ref()
            .map_or(0, |song| song.window.len - song.played)
    }

    /// The score of the song playing.
    #[cfg(test)]
    pub(crate) fn playing(&self) -> Option<&Score> {
        self.song.as_ref().map(|song| &song.score)
    }

    /// Plays the next step under `controls`, the ones asked for last: the
    /// next song started if its first bar is due and the playing song's next
    /// bar latched if this is its latch frame, then at most `max` frames,
    /// [`BLOCK`] and the frames to the next latch, as CLAP's process loop
    /// handles "every events that happrens at the frame "i"" and then
    /// processes "every samples until the next event"
    /// (<https://github.com/free-audio/clap/blob/main/src/plugin-template.c>).
    /// Returns the frames it played, whose sums [`sums`](Self::sums) and
    /// [`returned`](Self::returned) read.
    pub(crate) fn step(&mut self, controls: Controls, max: usize) -> usize {
        self.start_if_due(controls);
        self.latch_due(controls);
        let n = max.min(BLOCK).min(self.until_latch());
        self.play(n);
        n
    }

    /// Starts the next song under `controls` if its first bar is due.
    pub(crate) fn start_if_due(&mut self, controls: Controls) {
        if self.song_left() == 0 {
            self.start_song(controls);
        }
    }

    /// Starts the next song, composed for `controls`, which its first bar
    /// latches.
    fn start_song(&mut self, controls: Controls) {
        let song = self.next_song(controls);
        self.start(song);
    }

    /// Starts the next song, composed for `controls`, with every lane present
    /// at its gain in `levels` for good.
    #[cfg(test)]
    pub(crate) fn start_song_at(&mut self, controls: Controls, levels: [f64; LANES]) {
        let song = self.next_song(controls).at_levels(levels);
        self.start(song);
    }

    fn start(&mut self, song: Song) {
        let mut ended = self.song.replace(song);
        if let Some(ended) = &mut ended {
            ended.freeze();
        }
        self.fading.extend(ended);
        self.next_song += 1;
    }

    /// Frames until the playing song's next latch frame or its end, whichever
    /// comes first: as far as one step may play.
    fn until_latch(&self) -> usize {
        self.song.as_ref().map_or(0, |song| {
            let next = song.next_latch().unwrap_or(song.window.len);
            next.min(song.window.len) - song.played
        })
    }

    /// Latches `controls` for the playing song's next bar if this is its latch
    /// frame.
    fn latch_due(&mut self, controls: Controls) {
        if let Some(song) = &mut self.song
            && song.next_latch() == Some(song.played)
        {
            song.latch(controls);
        }
    }

    /// Plays the next `n` frames, at most [`BLOCK`] and
    /// [`song_left`](Self::song_left), and returns each lane's sum over them,
    /// each song's at its level.
    pub(crate) fn play(&mut self, n: usize) -> &[Vec<f64>; LANES] {
        debug_assert!(n <= BLOCK && n <= self.song_left());
        for sum in self.sums.iter_mut().chain(&mut self.plates) {
            sum.clear();
            sum.resize(n, 0.0);
        }
        for song in self.song.iter_mut().chain(&mut self.fading) {
            song.play(&mut self.sums, &mut self.plates, &mut self.scratch, n);
        }
        // a song at rest adds only zeros, so dropping it moves no frame
        self.fading.retain(|song| !song.at_rest());
        &self.sums
    }

    /// Each lane's sum over the step [`play`](Self::play) played last.
    pub(crate) fn sums(&self) -> &[Vec<f64>; LANES] {
        &self.sums
    }

    /// The plates' stereo return over the step [`play`](Self::play) played
    /// last, beside its lanes' sums.
    pub(crate) fn returned(&self) -> &[Vec<f64>; 2] {
        &self.plates
    }
}

/// One song, played from its first bar.
struct Song {
    score: Score,
    /// The stretch of the loop's tiling the song plays.
    window: Window,
    sr: u32,
    /// Frames played so far.
    played: usize,
    /// Each lane's notes.
    cues: [Cues; LANES],
    /// Each lane's notes summed from the next frame on.
    rings: [Ring; LANES],
    chains: [Chain; LANES],
    crackle: Crackle,
    /// How many bars have latched.
    latched: usize,
    /// How long a level ramps and a colour grades, in frames.
    level_ramp: usize,
    grade_ramp: usize,
    /// Whether each lane plays the notes struck in the latched bar.
    present: [bool; LANES],
    /// Each lane's gain into the mix, whose ramps all end within the song,
    /// so its tails ring on at the level it ended on.
    levels: [Ramp; LANES],
    /// The daylight, the rain and the sweep the chains play under, and how
    /// long rain takes to arrive, in frames.
    automation: Automation,
    rain_ramp: usize,
    /// The song's plate, which rings out with its tails.
    plate: Plate,
    /// The music lanes' level through the intro's edge and the outro's, each
    /// held at 1 where its edge is plain.
    edges: [Ramp; 2],
}

/// The music lanes' gain through `edges` on frame `n`.
fn edge_at(edges: &[Ramp; 2], n: usize) -> f64 {
    edges[0].at(n) * edges[1].at(n)
}

/// Whether a form's edges move `lane`: the lanes it arranges do, and the room
/// plays through.
fn fades(lane: Lane) -> bool {
    Layer::of(lane).is_some()
}

/// The level of `form`'s music lanes through its edges in `window`: a fade
/// rises from [`FADE_FLOOR_DB`] across the intro, or falls to it across the
/// outro, linear in dB.
fn edges(form: &Form, window: &Window) -> [Ramp; 2] {
    let floor = gain_of_db(FADE_FLOOR_DB);
    let [intro, _, outro] = window.parts.clone();
    let mut entry = Ramp::held(1.0, Shape::Db);
    if form.entry() == Some(Edge::Fade) {
        entry = Ramp::held(floor, Shape::Db);
        entry.toward(intro.start, 1.0, intro.len());
    }
    let mut exit = Ramp::held(1.0, Shape::Db);
    if form.exit() == Some(Edge::Fade) {
        exit.toward(outro.start, floor, outro.len());
    }
    [entry, exit]
}

/// `form`'s low-pass automation in `window`, if it has one: a filtered intro
/// opening to the body's first line, a dip shutting over `ramp` frames from
/// its bar's line and opening over as many from the next, and a filtered
/// outro shutting to the song's end.
fn sweep(form: &Form, window: &Window, ramp: usize) -> Option<Sweep> {
    let [intro, _, outro] = window.parts.clone();
    let mut points = Vec::new();
    if form.entry() == Some(Edge::Filter) {
        points.extend([(intro.start, 0.0), (intro.end, 1.0)]);
    }
    if let Some(Range {
        start: line,
        end: next,
    }) = window.dip.clone()
    {
        points.extend([
            (line, 1.0),
            (line + ramp, 0.0),
            (next, 0.0),
            (next + ramp, 1.0),
        ]);
    }
    if form.exit() == Some(Edge::Filter) {
        points.extend([(outro.start, 1.0), (outro.end, 0.0)]);
    }
    (!points.is_empty()).then(|| Sweep::new(points))
}

impl Song {
    /// Song `k` of `seed`'s run at `sr`, its first bar latching `controls`.
    #[cfg(test)]
    fn new(seed: u64, k: u64, controls: Controls, sr: u32) -> Self {
        Self::with_form(seed, k, controls, song_form(seed, k, controls), sr)
    }

    /// [`new`](Self::new) with the song playing `form` instead of its own.
    #[cfg(test)]
    fn with_form(seed: u64, k: u64, controls: Controls, form: Form, sr: u32) -> Self {
        let score = song_score(seed, k, controls);
        Self::of(score, form, song_seed(seed, k), controls, sr)
    }

    /// `score` played as `form` at `sr`, as the song seeded `seed`, which
    /// seeds its noise, its first bar latching `controls`.
    fn of(score: Score, form: Form, seed: u64, controls: Controls, sr: u32) -> Self {
        let mood = score.mood;
        let window = Window::new(&score, &form, sr);
        let len = window.len;
        let mut noise = NoiseStream::new(seed);
        let cues = Lane::ALL.map(|lane| Cues::new(&score, &form, lane, &window, noise.fork(), sr));
        // forked after every lane's, so it moves none of their draws
        let crackle = Crackle::new(noise.fork(), sr);
        // where the whole-lane night texture ducks, rounded as it rounds them:
        // under each kick the form strikes, since the duck is the groove's
        let onsets: Vec<usize> = (score.kick_times.iter())
            .map(|&at| samples_at(at, sr))
            .collect();
        let kicks: Vec<usize> = (window.tile(&onsets).into_iter())
            .filter(|&(_, bar, _)| form.plays(bar).strikes(Lane::Drums))
            .map(|(at, ..)| at)
            .collect();
        let level_ramp = samples_at(LEVEL_RAMP_BEATS * score.beat_s(), sr);
        let automation = Automation::new(controls, sweep(&form, &window, level_ramp));
        let chains = Lane::ALL.map(|lane| {
            Chain::new(
                mood,
                lane,
                &part(&score, lane),
                len,
                &kicks,
                &automation,
                sr,
            )
        });
        let present = Lane::ALL.map(|lane| lane_gain(lane, controls.intensity) > 0.0);
        let levels = Lane::ALL
            .map(|lane| Ramp::held(f64::from(resting_gain(lane, controls.intensity)), Shape::Db));
        Self {
            edges: edges(&form, &window),
            latched: 1,
            level_ramp,
            grade_ramp: samples_at(GRADE_RAMP_BARS * score.bar_s(), sr),
            automation,
            rain_ramp: samples_at(RAIN_RAMP_BARS * score.bar_s(), sr),
            plate: Plate::new(sr),
            score,
            window,
            sr,
            played: 0,
            cues,
            rings: Default::default(),
            chains,
            crackle,
            present,
            levels,
        }
    }

    /// The song with every lane present at its gain in `levels` for good.
    #[cfg(test)]
    fn at_levels(mut self, levels: [f64; LANES]) -> Self {
        self.present = [true; LANES];
        self.levels = levels.map(|level| Ramp::held(level, Shape::Db));
        self.latched = self.window.bars.len();
        self
    }

    /// The frame the next bar latches on. The first bar latched when the song
    /// started.
    fn next_latch(&self) -> Option<usize> {
        (self.latched < self.window.bars.len()).then(|| self.window.latch(self.latched))
    }

    /// Latches `controls` for the next bar, on its latch frame: each lane
    /// present where its gain is above 0 and ramping to it, or else holding
    /// its level, and each chain grading to the daylight.
    fn latch(&mut self, controls: Controls) {
        let at = self.played;
        for ((lane, present), level) in Lane::ALL
            .iter()
            .zip(&mut self.present)
            .zip(&mut self.levels)
        {
            let gain = f64::from(lane_gain(*lane, controls.intensity));
            *present = gain > 0.0;
            if *present {
                level.toward(at, gain, self.level_ramp);
            } else {
                level.hold(at);
            }
        }
        (self.automation).latch(at, controls, self.grade_ramp, self.rain_ramp);
        self.latched += 1;
    }

    /// Holds every ramp where it is, at the song's end: a fading song's tails
    /// latch nothing, and rain's phrase can outlast the last bar.
    fn freeze(&mut self) {
        let at = self.played;
        for level in &mut self.levels {
            level.hold(at);
        }
        self.automation.freeze(at);
    }

    /// The music lanes' edge gain over the `n` frames from `from`: held, or
    /// each frame's in `edged`, computed once for every music lane.
    fn edge_over(&self, from: usize, n: usize, edged: &mut Vec<f64>) -> Option<f64> {
        let (edges, end) = (self.edges, from + n);
        let flat = (edges.iter())
            .all(|edge| edge.flat_over(from, end))
            .then(|| edge_at(&edges, from));
        edged.clear();
        if flat.is_none() {
            edged.extend((from..end).map(|at| edge_at(&edges, at)));
        }
        flat
    }

    /// Lane `i`'s gain into the mix on each of `out`'s frames from `from`:
    /// its intensity's level, rain's on it and, on a music lane, the edges'
    /// ([`edge_over`](Self::edge_over)'s `edge` or `edged` over the same
    /// frames), multiplied in that order, once where all three hold.
    fn gain_over(
        &self,
        i: usize,
        from: usize,
        (edge, edged): (Option<f64>, &[f64]),
        out: &mut [f64],
    ) {
        let lane = Lane::ALL[i];
        let (level, rain) = (&self.levels[i], self.automation.rain());
        let wet = (rain.end() <= from).then(|| rain::level(lane, rain.to()));
        let edge = if fades(lane) { edge } else { Some(1.0) };
        match (level.end() <= from, wet, edge) {
            (true, Some(wet), Some(edge)) => out.fill(level.to() * wet * edge),
            _ => {
                for (k, (gain, at)) in out.iter_mut().zip(from..).enumerate() {
                    let wet = wet.unwrap_or_else(|| rain::level(lane, rain.at(at)));
                    let edge = edge.unwrap_or_else(|| edged[k]);
                    *gain = level.at(at) * wet * edge;
                }
            }
        }
    }

    /// Lane `i`'s gain into the mix on frame `n`, as [`play`](Self::play)
    /// takes it: one frame's edge is [`edge_over`](Self::edge_over)'s,
    /// held or not.
    #[cfg(test)]
    fn gain(&self, i: usize, n: usize) -> f64 {
        let mut gain = [0.0];
        self.gain_over(i, n, (Some(edge_at(&self.edges, n)), &[]), &mut gain);
        gain[0]
    }

    /// Strikes each note whose onset falls in the next `n` frames, then adds
    /// those frames of each lane, through its chain and at its gain, into
    /// `sums`, working in `scratch`.
    fn play(
        &mut self,
        sums: &mut [Vec<f64>; LANES],
        plates: &mut [Vec<f64>; 2],
        scratch: &mut Scratch,
        n: usize,
    ) {
        let Scratch {
            lane: buf,
            gains,
            work,
            edged,
            send: sent,
            wet,
        } = scratch;
        let end = self.played + n;
        sent.clear();
        sent.resize(n, 0.0);
        let rain = *self.automation.rain();
        let hit = |at: usize| rain::hit(rain.at(at));
        let unity = |_: usize| 1.0;
        let edge = self.edge_over(self.played, n, edged);
        for (i, lane) in Lane::ALL.into_iter().enumerate() {
            let part = part(&self.score, lane);
            let voice: &dyn Fn(usize) -> f32 = match part {
                Part::Hits(_) => &hit,
                _ => &unity,
            };
            let (ring, present) = (&mut self.rings[i], self.present[i]);
            let cues = &mut self.cues[i];
            cues.strike(
                &part,
                self.played,
                end,
                ring,
                present.then_some(voice),
                self.sr,
            );
            if matches!(part, Part::Bed) {
                (self.crackle).strike(
                    self.played,
                    end.min(self.window.len),
                    ring,
                    present,
                    self.sr,
                );
            }
            buf.resize(n, 0.0);
            ring.take(buf);
            (self.chains[i]).process(buf, self.played, &self.automation, &mut cues.noise, work);
            // the lane at its gain, once: what both its sum and its send hear
            gains.resize(n, 0.0);
            self.gain_over(i, self.played, (edge, edged), gains);
            for (x, gain) in buf.iter_mut().zip(gains.iter()) {
                *x *= gain;
            }
            for (s, x) in sums[i].iter_mut().zip(buf.iter()) {
                *s += x;
            }
            if let Some(send) = plate::send(lane) {
                for (s, x) in sent.iter_mut().zip(buf.iter()) {
                    *s += send * x;
                }
            }
        }
        self.plate.process(sent, wet);
        for (sum, wet) in plates.iter_mut().zip(wet.iter()) {
            for (s, x) in sum.iter_mut().zip(wet) {
                *s += x;
            }
        }
        self.played = end;
    }

    /// Whether the song has played out: past its last bar, every tail taken
    /// and every chain and its plate at rest, so all it can add is silence.
    fn at_rest(&self) -> bool {
        self.played >= self.window.len
            && self.rings.iter().all(|ring| ring.0.is_empty())
            && self.chains.iter().all(|chain| chain.at_rest(self.played))
            && self.plate.at_rest()
    }
}

/// A lane's notes over one song, in the order they sound: the order its
/// noise is drawn in, whatever the chunking.
struct Cues {
    /// `(onset in the song, index into the lane's events, muted)`, by onset:
    /// a note is muted where the form doesn't strike its lane in its bar.
    order: Vec<(usize, usize, bool)>,
    /// Notes struck so far.
    struck: usize,
    /// The lane's draws of the song's noise.
    noise: NoiseStream,
}

impl Cues {
    fn new(
        score: &Score,
        form: &Form,
        lane: Lane,
        window: &Window,
        noise: NoiseStream,
        sr: u32,
    ) -> Self {
        let order = (window.tile(&part(score, lane).onsets(sr)).into_iter())
            .map(|(at, bar, i)| (at, i, !form.plays(bar).strikes(lane)))
            .collect();
        Self {
            order,
            struck: 0,
            noise,
        }
    }

    /// Strikes into `ring`, which starts at frame `from`, each note whose
    /// onset is before `to`, at the gain `voice` gives its onset, or silent
    /// if there is no `voice` or the note is muted: the lane is absent. A
    /// silent hit still draws its noise, so no later hit's draws move; no
    /// other note draws any, so it needs no synthesis.
    fn strike(
        &mut self,
        part: &Part<'_>,
        from: usize,
        to: usize,
        ring: &mut Ring,
        voice: Option<&dyn Fn(usize) -> f32>,
        sr: u32,
    ) {
        while let Some(&(at, i, muted)) = self.order.get(self.struck)
            && at < to
        {
            let voice = voice.filter(|_| !muted);
            if voice.is_some() || matches!(part, Part::Hits(_)) {
                let (snippet, gain) = part.note(i, &mut self.noise, sr);
                if let Some(voice) = voice {
                    ring.add(at - from, &snippet, gain * voice(at));
                }
            }
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

    /// Strikes into `ring`, which starts at frame `from`, each pop before
    /// `to`, silent unless its lane is `present`, which draws the same.
    fn strike(&mut self, from: usize, to: usize, ring: &mut Ring, present: bool, sr: u32) {
        while self.next < to {
            let (pop, gain) = crackle_pop(&mut self.rng, sr);
            if present {
                ring.add(self.next - from, &pop, gain);
            }
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
    use crate::compose::{Plays, spans_a_form};
    use crate::stream::lanes::{COEFF_BLOCK, FILTERED_HZ};
    use crate::stream::master::TIER_INTENSITIES;

    /// Song `k` of `seed`'s run under `c` at `sr`, playing [`Form::plain`].
    fn plain_song(seed: u64, k: u64, c: Controls, sr: u32) -> Song {
        Song::with_form(seed, k, c, Form::plain(), sr)
    }

    /// The frame each bar of [`plain_song`] starts on at [`SAMPLE_RATE`].
    fn plain_bars(seed: u64, k: u64, c: Controls) -> Vec<usize> {
        plain_song(seed, k, c, SAMPLE_RATE).window.bars
    }

    /// The playing song's loop length at `sr`.
    fn loop_len(run: &Schedule, sr: u32) -> usize {
        samples_at(run.playing().expect("a song plays").loop_secs(), sr)
    }

    #[test]
    fn a_song_that_starts_with_no_beat_plays_its_body_whole_and_moves_no_edge() {
        let empty = Controls::default().with_intensity(TIER_INTENSITIES[0]);
        assert_eq!(lane_gain(Lane::Drums, empty.intensity), 0.0);
        let (mut entries, mut exits, mut dips, mut thinned) = (0, 0, 0, 0);
        for seed in 0..512 {
            let (drawn, beatless) = (
                song_form(seed, 0, Controls::default()),
                song_form(seed, 0, empty),
            );
            let at = |what: &str| format!("seed {seed}: {what}: {drawn} / {beatless}");
            entries += usize::from(drawn.entry().is_some());
            exits += usize::from(drawn.exit().is_some());
            dips += usize::from(drawn.dip().is_some());
            let [intro, body, outro] = drawn.parts();
            assert_eq!(beatless.parts(), drawn.parts(), "{}", at("parts"));
            assert_eq!(
                (beatless.entry(), beatless.exit(), beatless.dip()),
                (None, None, None),
                "{}",
                at("an edge")
            );
            thinned += usize::from(body.clone().any(|bar| *drawn.plays(bar) != Plays::ALL));
            for bar in body {
                assert_eq!(*beatless.plays(bar), Plays::ALL, "{}", at("the body"));
            }
            for bar in intro.chain(outro) {
                assert_eq!(beatless.plays(bar), drawn.plays(bar), "{}", at("an end"));
            }
        }
        // the seeds draw every edge and a thinned body, so each clear is proven
        assert!(
            [entries, exits, dips, thinned].iter().all(|&n| n > 0),
            "{entries} entries, {exits} exits, {dips} dips, {thinned} thinned bodies"
        );
    }

    #[test]
    fn the_room_is_the_one_lane_no_layer_holds_and_no_tape_plays() {
        // the form's layers, the voices' parts and the chains each name the
        // room, and a sweep or an edge handed to it would be lost
        for daylight in [0.0, 1.0] {
            let c = Controls::default().with_daylight(daylight);
            let song = Song::new(1, 0, c, SAMPLE_RATE);
            for (lane, chain) in Lane::ALL.iter().zip(&song.chains) {
                let room = matches!(part(&song.score, *lane), Part::Bed);
                assert_eq!(fades(*lane), !room, "{lane:?} at daylight {daylight}");
                assert_eq!(matches!(chain, Chain::Room(_)), room, "{lane:?}'s chain");
            }
        }
    }

    #[test]
    fn a_song_is_composed_for_the_night_below_dusk_and_the_day_from_it() {
        let at = |daylight: f32| song_score(1, 0, Controls::default().with_daylight(daylight)).mood;
        assert_eq!(at(Controls::DUSK), Mood::Day);
        assert_eq!(at(Controls::DUSK.next_down()), Mood::Night);
    }

    #[test]
    fn each_song_of_a_run_is_composed_anew() {
        let day = Controls::default();
        let scores: Vec<Score> = (0..8)
            .map(|k| Song::new(1, k, day, SAMPLE_RATE).at_levels(UNITY).score)
            .collect();
        for (k, a) in scores.iter().enumerate() {
            assert!(!scores[k + 1..].contains(a), "song {k} recurs");
            assert!(
                *a != Song::new(2, k as u64, day, SAMPLE_RATE)
                    .at_levels(UNITY)
                    .score,
                "seeds 1 and 2 share song {k}"
            );
        }
    }

    /// Each lane of `seed`'s run from frame 0 to `to`, starting under
    /// `start` and each of `calls` made once its frame has played, latched as
    /// a `Performance` latches them; and the schedule it played.
    fn latched(
        seed: u64,
        start: Controls,
        calls: &[(usize, Controls)],
        to: usize,
    ) -> (Schedule, [Vec<f64>; LANES]) {
        let mut run = Schedule::plain(seed, 0, SAMPLE_RATE);
        let mut lanes: [Vec<f64>; LANES] = Default::default();
        let (mut asked, mut calls, mut at) = (start, calls.iter().peekable(), 0);
        while at < to {
            while let Some(&&(call, c)) = calls.peek()
                && call == at
            {
                asked = c;
                calls.next();
            }
            let next_call = calls.peek().map_or(to, |&&(call, _)| call);
            let n = run.step(asked, (next_call - at).min(to - at));
            for (lane, sum) in lanes.iter_mut().zip(run.sums()) {
                lane.extend_from_slice(sum);
            }
            at += n;
        }
        (run, lanes)
    }

    fn energy(buf: &[f64]) -> f64 {
        buf.iter().map(|x| x * x).sum()
    }

    /// The drums' index in [`Lane::ALL`].
    fn drums() -> usize {
        Lane::ALL
            .iter()
            .position(|&lane| lane == Lane::Drums)
            .expect("a drums lane")
    }

    #[test]
    fn a_song_starts_at_the_levels_1a_played_it_at() {
        for intensity in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let c = Controls::default().with_intensity(intensity);
            let song = plain_song(3, 0, c, SAMPLE_RATE);
            for ((lane, present), level) in Lane::ALL.iter().zip(song.present).zip(&song.levels) {
                let gain = f64::from(lane_gain(*lane, intensity));
                // an absent lane plays zeros, as 1a's played at a gain of 0
                assert_eq!(present, gain > 0.0, "{lane:?} at {intensity}");
                if present {
                    assert_eq!(level.at(0), gain, "{lane:?} at {intensity}");
                }
            }
        }
    }

    #[test]
    fn the_drums_strike_a_kick_on_every_bars_first_beat() {
        let mut missing = Vec::new();
        for seed in 0..64 {
            for c in [Controls::default(), Controls::default().with_daylight(0.0)] {
                let score = song_score(seed, 0, c);
                let kicks: Vec<f32> = (score.drums.iter())
                    .filter(|&&(_, kind, _)| kind == crate::compose::DrumKind::Kick)
                    .map(|&(at, ..)| at)
                    .collect();
                for bar in 0..GEN_LOOP_BARS {
                    let line = bar as f32 * score.bar_s();
                    // within a thirty-second of the line, or anticipating it
                    let first = (line - ANTICIPATION_S)..(line + score.beat_s() / 8.0);
                    if !kicks.iter().any(|at| first.contains(at)) {
                        missing.push(format!("seed {seed} daylight {}: bar {bar}", c.daylight));
                    }
                }
            }
        }
        assert!(
            missing.is_empty(),
            "bars with no first-beat kick: {missing:#?}"
        );
    }

    #[test]
    fn a_silent_hit_draws_the_noise_it_would_have_played() {
        let (seed, bars) = (7, 6);
        let [mut empty, mut moderate] = [0.0, 0.5].map(|intensity| {
            plain_song(
                seed,
                0,
                Controls::default().with_intensity(intensity),
                SAMPLE_RATE,
            )
        });
        let to = empty.window.bars[bars];
        let (mut sums, mut buf) = (std::array::from_fn(|_| Vec::new()), Scratch::new());
        for song in [&mut empty, &mut moderate] {
            while song.played < to {
                song.play(
                    &mut sums,
                    &mut [Vec::new(), Vec::new()],
                    &mut buf,
                    BLOCK.min(to - song.played),
                );
            }
        }
        assert!(!empty.present[drums()] && moderate.present[drums()]);
        let [a, b] = [&mut empty, &mut moderate].map(|song| song.cues[drums()].noise.unit());
        assert_eq!(a, b, "a silent hit moved the drums' later draws");
    }

    #[test]
    fn a_lane_enters_on_its_first_note_after_the_line_at_six_db_or_more() {
        let (seed, empty) = (2, Controls::default().with_intensity(0.0));
        let moderate = empty.with_intensity(0.5);
        let bars = plain_bars(seed, 0, empty);
        let beat = samples_at(song_score(seed, 0, empty).beat_s(), SAMPLE_RATE);
        let anticipation = samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let (before, line) = (bars[4], bars[5]);
        // anywhere in the bar before, up to the line's own latch frame
        for call in [before, before.midpoint(line), line - anticipation] {
            let (_, lanes) = latched(seed, empty, &[(call, moderate)], line + beat);
            let drums = &lanes[drums()];
            let (was, now) = (energy(&drums[line - beat..line]), energy(&drums[line..]));
            assert!(
                now > 0.0 && now >= 4.0 * was,
                "called at {call}: {was} then {now}"
            );
        }
    }

    #[test]
    fn a_levels_ramp_lands_on_its_target_one_beat_from_its_latch() {
        let (seed, moderate) = (4, Controls::default());
        let busy = moderate.with_intensity(1.0);
        let bars = plain_bars(seed, 0, moderate);
        let latch = bars[3] - samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let (run, _) = latched(seed, moderate, &[(bars[2], busy)], latch + 1);
        let song = run.song.as_ref().expect("a song plays");
        let level = &song.levels[drums()];
        let beat = samples_at(song.score.beat_s(), SAMPLE_RATE);
        let [from, to] = [moderate, busy].map(|c| f64::from(lane_gain(Lane::Drums, c.intensity)));
        assert_eq!(level.at(latch), from);
        assert!(level.at(latch + beat - 1) < to, "the ramp ends early");
        assert_eq!(level.at(latch + beat), to);
    }

    #[test]
    fn a_daylight_latch_grades_each_graded_lane_over_a_bar() {
        let (seed, day) = (4, Controls::default());
        let bars = plain_bars(seed, 0, day);
        let latch = bars[3] - samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let (run, _) = latched(seed, day, &[(bars[2], day.with_daylight(0.6))], latch + 1);
        let song = run.song.as_ref().expect("a song plays");
        let bar = samples_at(song.score.bar_s(), SAMPLE_RATE);
        assert_eq!(song.automation.graded_until(), latch + bar);
        let grading: Vec<bool> = song.chains.iter().map(Chain::grades).collect();
        let want = Lane::ALL.map(|lane| ![Lane::Pad, Lane::Texture].contains(&lane));
        assert_eq!(grading, want);
    }

    #[test]
    fn the_last_bars_grade_and_level_ramps_end_within_the_song() {
        for seed in 0..8 {
            for start in [Controls::default(), Controls::default().with_daylight(0.0)] {
                let bars = plain_bars(seed, 0, start);
                let last = bars[bars.len() - 1] - samples_at(ANTICIPATION_S, SAMPLE_RATE);
                let into = start.with_daylight(0.4).with_intensity(1.0);
                let (run, _) = latched(seed, start, &[(last, into)], last + 1);
                let song = run.song.as_ref().expect("a song plays");
                let ends = [song.automation.graded_until()].into_iter();
                let ends = ends.chain(song.levels.iter().map(Ramp::end));
                assert!(
                    ends.clone().all(|end| end <= song.window.len),
                    "seed {seed} from {start:?}: ramps end at {:?}, past {}",
                    ends.collect::<Vec<_>>(),
                    song.window.len
                );
            }
        }
    }

    #[test]
    fn rain_leaves_the_score_its_tempo_and_its_bars_alone() {
        for seed in 0..8 {
            for c in [Controls::default(), Controls::default().with_daylight(0.0)] {
                let wet = c.with_rain(1.0);
                assert_eq!(
                    song_score(seed, 0, wet),
                    song_score(seed, 0, c),
                    "seed {seed}"
                );
                assert_eq!(
                    plain_bars(seed, 0, wet),
                    plain_bars(seed, 0, c),
                    "seed {seed}"
                );
            }
        }
    }

    #[test]
    fn a_rain_latch_turns_the_sparkle_down_and_the_pad_up_over_a_phrase() {
        let (seed, empty) = (3, Controls::default().with_intensity(0.0));
        let bars = plain_bars(seed, 0, empty);
        let latch = bars[3] - samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let (run, _) = latched(seed, empty, &[(bars[2], empty.with_rain(1.0))], latch + 1);
        let song = run.song.as_ref().expect("a song plays");
        let phrase = samples_at(RAIN_RAMP_BARS * song.score.bar_s(), SAMPLE_RATE);
        for (i, lane) in Lane::ALL.iter().enumerate() {
            let level = song.levels[i].at(latch);
            assert_eq!(song.gain(i, latch), level, "{lane:?} at the latch");
            let wet = level * rain::level(*lane, 1.0);
            assert_eq!(song.gain(i, latch + phrase), wet, "{lane:?} a phrase on");
            if wet != level {
                assert!(
                    song.gain(i, latch + phrase - 1) != wet,
                    "{lane:?} lands early"
                );
            }
        }
    }

    #[test]
    fn a_rain_latch_darkens_the_sparkle_keys_and_drums_over_a_phrase() {
        let (seed, c) = (3, Controls::default());
        let bars = plain_bars(seed, 0, c);
        let latch = bars[3] - samples_at(ANTICIPATION_S, SAMPLE_RATE);
        // past the later bars' latches, which ask for the same rain
        let (run, _) = latched(seed, c, &[(bars[2], c.with_rain(1.0))], bars[7]);
        let song = run.song.as_ref().expect("a song plays");
        let phrase = samples_at(RAIN_RAMP_BARS * song.score.bar_s(), SAMPLE_RATE);
        assert_eq!(
            song.automation.rain().end(),
            latch + phrase,
            "a later latch restarted rain"
        );
        for (lane, chain) in Lane::ALL.iter().zip(&song.chains) {
            assert_eq!(chain.darkens(), rain::DARKENED.contains(lane), "{lane:?}");
        }
    }

    #[test]
    fn the_pads_rain_level_plays_on_the_lane_it_mixes() {
        // the pad's tone takes no rain, so its lane is its dry one, louder
        let pad = Lane::ALL
            .iter()
            .position(|&lane| lane == Lane::Pad)
            .expect("a pad lane");
        let c = Controls::default();
        let to = plain_bars(1, 0, c)[2];
        let [(_, dry), (_, wet)] = [0.0, 1.0].map(|rain| latched(1, c.with_rain(rain), &[], to));
        let louder = rain::level(Lane::Pad, 1.0);
        assert!(dry[pad].iter().any(|&x| x != 0.0));
        let off = (dry[pad].iter().zip(&wet[pad]))
            .position(|(&d, &w)| (w - d * louder).abs() > 1e-12 * d.abs().max(1e-12));
        assert_eq!(off, None, "the pad's wet lane isn't its dry one, louder");
    }

    #[test]
    fn rain_softens_each_drum_hit_as_it_strikes() {
        let seed = 6;
        let [mut dry, mut wet] = [0.0, 1.0]
            .map(|rain| plain_song(seed, 0, Controls::default().with_rain(rain), SAMPLE_RATE));
        let (mut sums, mut buf) = (std::array::from_fn(|_| Vec::new()), Scratch::new());
        for song in [&mut dry, &mut wet] {
            song.play(&mut sums, &mut [Vec::new(), Vec::new()], &mut buf, BLOCK);
        }
        let [dry, wet] =
            [&dry, &wet].map(|song| song.rings[drums()].0.iter().copied().collect::<Vec<f32>>());
        assert!(dry.iter().any(|&s| s != 0.0), "no hit rang");
        // within f32's rounding of the sums the ring holds
        let loudest = dry.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let off = (dry.iter().zip(&wet))
            .find(|&(&d, &w)| (w - d * rain::hit(1.0)).abs() > 1e-6 * loudest);
        assert_eq!(off, None, "a wet hit isn't the dry one, softened");
    }

    #[test]
    fn a_fading_songs_rain_holds_where_it_ended() {
        let (seed, c) = (2, Controls::default().with_intensity(0.0));
        let bars = plain_bars(seed, 0, c);
        let len = audition_len(seed, c);
        let last = bars[bars.len() - 1] - samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let (run, _) = latched(seed, c, &[(last, c.with_rain(1.0))], len + 1);
        let fading = run.fading.first().expect("song 0 fades");
        let sparkle = Lane::ALL
            .iter()
            .position(|&lane| lane == Lane::Sparkle)
            .expect("a sparkle lane");
        let held = fading.gain(sparkle, len);
        assert!(
            held < fading.levels[sparkle].at(len),
            "the rain didn't start"
        );
        assert!(
            [len + 1, len + 10_000]
                .iter()
                .all(|&n| fading.gain(sparkle, n) == held)
        );
        assert_eq!(
            fading.automation.rain().end(),
            len,
            "the tilts' rain kept ramping"
        );
    }

    /// Song 0's length under `c`.
    fn audition_len(seed: u64, c: Controls) -> usize {
        plain_song(seed, 0, c, SAMPLE_RATE).window.len
    }

    #[test]
    fn a_latch_that_asks_for_nothing_new_leaves_every_ramp_settled() {
        for c in [
            Controls::default(),
            Controls::default().with_daylight(0.25).with_rain(0.6),
            Controls::default().with_intensity(0.0),
        ] {
            let bars = plain_bars(1, 0, c);
            let latch = bars[5] - samples_at(ANTICIPATION_S, SAMPLE_RATE);
            let (run, _) = latched(1, c, &[], latch + 1);
            let song = run.song.as_ref().expect("a song plays");
            let ends: Vec<usize> = (song.levels.iter().map(Ramp::end))
                .chain([song.automation.rain().end(), song.automation.graded_until()])
                .collect();
            assert!(
                ends.iter().all(|&end| end < latch),
                "{c:?}: ramps restarted, ending {ends:?}"
            );
        }
    }

    #[test]
    fn a_dropped_lane_holds_its_level_and_rings_out_what_it_struck() {
        let (seed, busy) = (5, Controls::default().with_intensity(1.0));
        let empty = busy.with_intensity(0.0);
        let anticipation = samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let bars = plain_bars(seed, 0, busy);
        let (_, kept) = latched(seed, busy, &[], bars[bars.len() - 1]);
        let onsets: Vec<usize> = (plain_song(seed, 0, busy, SAMPLE_RATE).cues[drums()].order)
            .iter()
            .map(|&(at, ..)| at)
            .collect();
        // a bar whose latch a hit of the bar before still rings over, alone
        // until the bar's own first hit
        let (bar, first) = (1..bars.len() - 2)
            .find_map(|bar| {
                let latch = bars[bar] - anticipation;
                let first = onsets.iter().copied().find(|&at| at >= latch)?;
                (energy(&kept[drums()][latch..first]) > 0.0).then_some((bar, first))
            })
            .expect("a bar a hit rings into");
        let latch = bars[bar] - anticipation;
        let to = bars[bar + 2];
        let (run, dropped) = latched(seed, busy, &[(bars[bar - 1], empty)], to);
        let level = &run.song.as_ref().expect("a song plays").levels[drums()];
        let held = f64::from(lane_gain(Lane::Drums, busy.intensity));
        assert!([latch, latch + 1, to].iter().all(|&n| level.at(n) == held));
        assert_eq!(
            &dropped[drums()][latch..first],
            &kept[drums()][latch..first],
            "the tails rang at another level"
        );
        // and nothing struck after the drop sounds, once those tails are out
        assert_eq!(energy(&dropped[drums()][bars[bar + 1]..to]), 0.0);
    }

    #[test]
    fn every_onset_a_bar_anticipates_lies_at_or_after_its_latch_frame() {
        let anticipation = samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let mut early = Vec::new();
        for seed in 0..64 {
            for c in [Controls::default(), Controls::default().with_daylight(0.0)] {
                let song = Song::new(seed, 0, c, SAMPLE_RATE);
                // nearer a bar line than any onset the bar before plays late,
                // whose latest sits a sixteenth or more before it
                let window = samples_at(song.score.beat_s() / 8.0, SAMPLE_RATE);
                for (lane, cues) in Lane::ALL.iter().zip(&song.cues) {
                    for &(at, ..) in &cues.order {
                        // the song's end line too, the next song's first
                        let line = song
                            .window
                            .bars
                            .iter()
                            .chain([&song.window.len])
                            .find(|&&bar| bar > at);
                        if let Some(&line) = line
                            && line - at < window
                            && line - at > anticipation
                        {
                            early.push(format!(
                                "seed {seed} {:?} {lane:?}: {} frames early",
                                c.daylight,
                                line - at
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            early.is_empty(),
            "onsets before their bar's latch frame: {early:#?}"
        );
    }

    #[test]
    fn a_songs_bars_start_where_its_pad_strikes_each_chord() {
        let pad = Lane::ALL.iter().position(|&lane| lane == Lane::Pad);
        for (seed, controls) in [
            (1, Controls::default()),
            (2, Controls::default().with_daylight(0.0)),
        ] {
            let song = Song::new(seed, 0, controls, SAMPLE_RATE).at_levels(UNITY);
            let strikes: Vec<usize> = pad
                .map(|i| song.cues[i].order.iter().map(|&(at, ..)| at).collect())
                .unwrap_or_default();
            assert_eq!(bar_frames(seed, 0, controls), strikes, "seed {seed}");
        }
    }

    #[test]
    fn a_run_plays_each_songs_score() {
        for c in [Controls::default(), Controls::default().with_daylight(0.0)] {
            let mut run = Schedule::new(6, 0, SAMPLE_RATE);
            for k in 0..3 {
                run.start_song_at(c, UNITY);
                let played = run.song.as_ref().map(|song| &song.score);
                assert_eq!(played, Some(&song_score(6, k, c)), "song {k} at {c:?}");
            }
        }
    }

    #[test]
    fn the_plate_answers_the_pads_first_chord_only_after_its_pre_delay() {
        // the pad strikes on the song's first frame, before any other wet lane
        let (seed, c) = (2, Controls::default());
        let (mut run, mut plate) = (
            Schedule::plain(seed, 0, SAMPLE_RATE),
            [Vec::new(), Vec::new()],
        );
        run.start_song(c);
        let loop_len = loop_len(&run, SAMPLE_RATE);
        while plate[0].len() < loop_len {
            run.play(BLOCK);
            for (side, got) in plate.iter_mut().zip(run.returned()) {
                side.extend_from_slice(got);
            }
        }
        let first = plate
            .iter()
            .filter_map(|side| side.iter().position(|&x| x != 0.0))
            .min();
        let earliest = samples_at(super::plate::PLATE_PREDELAY_S, SAMPLE_RATE);
        assert!(
            first.is_some_and(|f| f >= earliest),
            "the plate answered at {first:?}, before {earliest}"
        );
    }

    #[test]
    fn a_run_is_its_songs_each_played_alone() {
        // a seed whose first song fades and filters, so its tails cross the
        // join through every edge
        let (seed, c, sr) = (1, Controls::default(), SAMPLE_RATE);
        // levels that differ in every lane, so a tail heard at the next
        // song's levels shows
        let levels = [UNITY, [0.5; LANES]];
        let mut run = Schedule::new(seed, 0, sr);
        let mut first = Song::new(seed, 0, c, sr).at_levels(levels[0]);
        let mut second = Song::new(seed, 1, c, sr).at_levels(levels[1]);
        spans_a_form(&[song_form(seed, 0, c), song_form(seed, 1, c)]).unwrap();
        let mut buf = Scratch::new();
        // through song 1's first loop, past every tail of song 0
        let span = first.window.len + samples_at(second.score.loop_secs(), sr);
        let mut at = 0;
        let mut levels = levels.into_iter();
        while at < span {
            if run.song_left() == 0 {
                run.start_song_at(c, levels.next().expect("two songs start"));
            }
            let n = run.song_left().min(BLOCK);
            let mut alone: [Vec<f64>; LANES] = std::array::from_fn(|_| vec![0.0; n]);
            let mut plates: [Vec<f64>; 2] = std::array::from_fn(|_| vec![0.0; n]);
            // as the run sums them: the playing song, then the fading one,
            // which plays here past its rest rather than being dropped
            if at >= first.window.len {
                second.play(&mut alone, &mut plates, &mut buf, n);
            }
            first.play(&mut alone, &mut plates, &mut buf, n);
            run.play(n);
            let returned = run.returned().iter().zip(&plates).enumerate();
            for (side, (got, want)) in returned {
                let diff = got
                    .iter()
                    .zip(want)
                    .position(|(g, w)| g.to_bits() != w.to_bits());
                assert_eq!(
                    diff.map(|i| at + i),
                    None,
                    "the plates' side {side}: the first frame that differs"
                );
            }
            for ((lane, got), want) in Lane::ALL.iter().zip(run.sums()).zip(&alone) {
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
        let mut song = plain_song(4, 0, night, sr).at_levels(UNITY);
        let loop_len = samples_at(song.score.loop_secs(), sr);
        let texture = Lane::ALL.iter().position(|&l| l == Lane::Texture).unwrap();
        let (mut bed, mut buf) = (Vec::with_capacity(loop_len), Scratch::new());
        while bed.len() < loop_len {
            let n = BLOCK.min(loop_len - bed.len());
            let mut sums: [Vec<f64>; LANES] = std::array::from_fn(|_| vec![0.0; n]);
            song.play(&mut sums, &mut [vec![0.0; n], vec![0.0; n]], &mut buf, n);
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

    /// FNV-1a over 64-bit words, as `compose`'s goldens hash.
    fn fnv1a(words: impl IntoIterator<Item = u64>) -> u64 {
        words.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, w| {
            (h ^ w).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Every bar frame and every cue of seeds 0..4's first two songs by day
    /// and by night, played plain: recorded on 527372e, before songs had a
    /// form, so a window that moves a note moves this. The frames are
    /// exact-rounded f32 products, so they hold on every platform.
    #[test]
    fn a_plain_form_plays_todays_frames() {
        const TODAYS_FRAMES_GOLDEN: u64 = 0x0069_6c09_00db_6ad9;
        let mut words: Vec<u64> = Vec::new();
        for daylight in [1.0, 0.0] {
            for seed in 0..4 {
                for k in 0..2 {
                    let song = plain_song(
                        seed,
                        k,
                        Controls::default().with_daylight(daylight),
                        SAMPLE_RATE,
                    );
                    words.push(song.window.len as u64);
                    words.extend(song.window.bars.iter().map(|&b| b as u64));
                    for cues in &song.cues {
                        words.push(cues.order.len() as u64);
                        for &(at, i, _) in &cues.order {
                            words.extend([at as u64, i as u64]);
                        }
                    }
                }
            }
        }
        let h = fnv1a(words);
        assert_eq!(h, TODAYS_FRAMES_GOLDEN, "{h:#018x}");
    }

    /// The index of `lane` in [`Lane::ALL`].
    fn index(lane: Lane) -> usize {
        Lane::ALL.iter().position(|&l| l == lane).expect("a lane")
    }

    /// `song`'s lanes from its first frame to `to`, each at its own level.
    fn play_to(song: &mut Song, to: usize) -> [Vec<f64>; LANES] {
        let (mut lanes, mut buf): ([Vec<f64>; LANES], _) = Default::default();
        while song.played < to {
            let n = BLOCK.min(to - song.played);
            let mut sums: [Vec<f64>; LANES] = std::array::from_fn(|_| vec![0.0; n]);
            song.play(&mut sums, &mut [vec![0.0; n], vec![0.0; n]], &mut buf, n);
            for (lane, sum) in lanes.iter_mut().zip(sums) {
                lane.extend(sum);
            }
        }
        lanes
    }

    #[test]
    fn a_muted_note_strikes_nothing_and_its_tail_rings_on() {
        let (seed, c, bar) = (3, Controls::default(), 9);
        let mut form = Form::plain();
        form.plays_mut(bar).set(Layer::Bass, false);
        let mut muted = Song::with_form(seed, 0, c, form, SAMPLE_RATE).at_levels(UNITY);
        let mut whole = plain_song(seed, 0, c, SAMPLE_RATE).at_levels(UNITY);
        let bass = index(Lane::Bass);
        let anticipation = samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let (latch, next) = (
            muted.window.bars[bar] - anticipation,
            muted.window.bars[bar + 1] - anticipation,
        );
        let in_bar: Vec<(usize, bool)> = (muted.cues[bass].order.iter())
            .filter(|&&(at, ..)| (latch..next).contains(&at))
            .map(|&(at, _, muted)| (at, muted))
            .collect();
        assert!(
            !in_bar.is_empty() && in_bar.iter().all(|&(_, m)| m),
            "bar {bar}'s notes: {in_bar:?}"
        );
        let first = in_bar[0].0;
        let to = muted.window.bars[bar + 1];
        let [got, want] = [&mut muted, &mut whole].map(|song| play_to(song, to)[bass].clone());
        assert_eq!(got[..first], want[..first], "the bar before moved");
        let beat = samples_at(muted.score.beat_s(), SAMPLE_RATE);
        assert!(
            energy(&got[muted.window.bars[bar]..muted.window.bars[bar] + beat]) > 0.0,
            "the bar-8 tail was cut"
        );
        assert!(
            energy(&got[first..to]) < energy(&want[first..to]),
            "the muted bar still struck"
        );
    }

    #[test]
    fn a_muted_hit_still_draws_its_noise() {
        let (seed, c) = (7, Controls::default());
        let mut form = Form::plain();
        for bar in 4..8 {
            form.plays_mut(bar).set(Layer::Drums, false);
        }
        let mut muted = Song::with_form(seed, 0, c, form, SAMPLE_RATE).at_levels(UNITY);
        let mut whole = plain_song(seed, 0, c, SAMPLE_RATE).at_levels(UNITY);
        let to = muted.window.bars[8];
        play_to(&mut muted, to);
        play_to(&mut whole, to);
        let [a, b] = [&mut muted, &mut whole].map(|song| song.cues[drums()].noise.unit());
        assert_eq!(a, b, "a muted hit moved the drums' later draws");
    }

    #[test]
    fn a_lane_plays_where_the_form_and_the_intensity_both_let_it() {
        let (seed, empty) = (2, Controls::default().with_intensity(0.0));
        let moderate = empty.with_intensity(0.5);
        let mut form = Form::plain();
        for bar in 0..4 {
            form.plays_mut(bar).set(Layer::Drums, false);
        }
        let bars = plain_bars(seed, 0, empty);
        let anticipation = samples_at(ANTICIPATION_S, SAMPLE_RATE);
        // asked inside the muted bars, latched at bar 3's line, heard from 4's
        let (call, entry) = (bars[2].midpoint(bars[3]), bars[4] - anticipation);
        let mut run = Schedule::new(seed, 0, SAMPLE_RATE).with_form(form);
        let mut heard = Vec::new();
        let mut asked = empty;
        while heard.len() < bars[6] {
            let at = heard.len();
            if at == call {
                asked = moderate;
            }
            let next_call = if at < call { call } else { bars[6] };
            run.step(asked, (next_call - at).min(bars[6] - at));
            heard.extend_from_slice(&run.sums()[drums()]);
        }
        assert_eq!(
            energy(&heard[..entry]),
            0.0,
            "the drums struck in a muted bar"
        );
        assert!(energy(&heard[entry..]) > 0.0, "the drums never entered");
        let song = run.song.as_ref().expect("a song plays");
        assert_eq!(
            song.levels[drums()].at(bars[4]),
            f64::from(lane_gain(Lane::Drums, moderate.intensity)),
            "the drums entered at a stale level"
        );
    }

    #[test]
    fn a_note_past_the_last_line_never_plays() {
        let anticipation = samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let mut ending_mid_loop = 0;
        for seed in 0..32 {
            for c in [Controls::default(), Controls::default().with_daylight(0.0)] {
                let (song, form) = (Song::new(seed, 0, c, SAMPLE_RATE), song_form(seed, 0, c));
                ending_mid_loop +=
                    usize::from(!(form.lead_in() + form.bars()).is_multiple_of(GEN_LOOP_BARS));
                for (lane, cues) in Lane::ALL.iter().zip(&song.cues) {
                    let late = cues
                        .order
                        .iter()
                        .find(|&&(at, ..)| at + anticipation >= song.window.len);
                    assert_eq!(late, None, "seed {seed} {lane:?}: a note of the next bar");
                }
            }
        }
        assert!(
            ending_mid_loop > 0,
            "no song ends mid-loop: the end proves nothing"
        );
    }

    #[test]
    fn a_break_does_not_duck_the_room() {
        let (seed, night) = (4, Controls::default().with_daylight(0.0));
        let mut form = Form::plain();
        let (from, to) = (20, 24);
        for bar in from..to {
            form.plays_mut(bar).set(Layer::Drums, false);
        }
        let texture = index(Lane::Texture);
        let mut broken = Song::with_form(seed, 0, night, form, SAMPLE_RATE).at_levels(UNITY);
        let mut whole = plain_song(seed, 0, night, SAMPLE_RATE).at_levels(UNITY);
        let end = broken.window.bars[to];
        let [got, want] = [&mut broken, &mut whole].map(|song| play_to(song, end)[texture].clone());
        let start = broken.window.bars[from];
        // the room before the break, past any kick's release, plays alike
        assert_eq!(got[..start / 2], want[..start / 2]);
        assert!(
            energy(&got[start..end]) > energy(&want[start..end]),
            "the room ducked under kicks the break never struck"
        );
    }

    /// The first seed whose song 0 under the default controls has a form
    /// `want` holds.
    fn seed_where(want: impl Fn(&Form) -> bool) -> u64 {
        (0..512)
            .find(|&seed| want(&song_form(seed, 0, Controls::default())))
            .expect("a seed draws the form")
    }

    #[test]
    fn an_edge_fades_monotonically_between_its_ends() {
        let c = Controls::default();
        let floor = gain_of_db(FADE_FLOOR_DB);
        let seed = seed_where(|f| f.entry() == Some(Edge::Fade) && f.exit() == Some(Edge::Fade));
        let (song, form) = (Song::new(seed, 0, c, SAMPLE_RATE), song_form(seed, 0, c));
        let (body, outro) = (
            song.window.bars[form.intro()],
            song.window.bars[form.intro() + form.body()],
        );
        for (i, lane) in Lane::ALL.iter().enumerate() {
            let full = song.gain(i, body);
            let at = |n: usize| song.gain(i, n) / full;
            if *lane == Lane::Texture {
                assert!(
                    [0, body, outro, song.window.len]
                        .iter()
                        .all(|&n| at(n) == 1.0),
                    "the room faded"
                );
                continue;
            }
            assert_eq!(at(0), floor, "{lane:?} enters off the floor");
            assert_eq!(at(outro), 1.0, "{lane:?} leaves the body below it");
            assert_eq!(at(song.window.len), floor, "{lane:?} ends off the floor");
            let rising: Vec<f64> = (0..=body).step_by(97).map(at).collect();
            let falling: Vec<f64> = (outro..=song.window.len).step_by(97).map(at).collect();
            assert!(rising.is_sorted(), "{lane:?}'s fade-in falls somewhere");
            assert!(
                falling.is_sorted_by(|a, b| a >= b),
                "{lane:?}'s fade-out rises somewhere"
            );
        }
    }

    /// Each lane's tuned tone corner after each step from `song`'s first
    /// frame to `to`, at most a [`COEFF_BLOCK`] long, with the frame it ended
    /// on, latching `c` at every latch frame it reaches, as a `Performance`
    /// asked for `c` does.
    fn corners(song: &mut Song, to: usize, c: Controls) -> Vec<(usize, [Option<f32>; LANES])> {
        let (mut sums, mut buf): ([Vec<f64>; LANES], _) = Default::default();
        let mut out = Vec::new();
        while song.played < to {
            if song.next_latch() == Some(song.played) {
                song.latch(c);
            }
            let next = song.next_latch().unwrap_or(to).min(to);
            let n = COEFF_BLOCK.min(next - song.played);
            for sum in &mut sums {
                sum.clear();
                sum.resize(n, 0.0);
            }
            song.play(&mut sums, &mut [vec![0.0; n], vec![0.0; n]], &mut buf, n);
            out.push((song.played, song.chains.each_ref().map(Chain::tuned_hz)));
        }
        out
    }

    /// Lane `i`'s corner over the step of `path` that plays frame `n`.
    fn corner_at(path: &[(usize, [Option<f32>; LANES])], n: usize, i: usize) -> f32 {
        let step = path
            .iter()
            .find(|&&(end, _)| end > n)
            .expect("a step plays the frame");
        step.1[i].expect("a tape tunes")
    }

    /// Song 0 of the first seed whose form under `c` has a filtered intro,
    /// its twin with a plain one, both latching as a `Performance` does, and
    /// the frame the body starts on.
    fn filtered_and_open(c: Controls) -> (Song, Song, usize) {
        let seed = seed_where(|f| f.entry() == Some(Edge::Filter));
        let form = song_form(seed, 0, c);
        let open = form.clone().with_edges(None, form.exit());
        let filtered = Song::new(seed, 0, c, SAMPLE_RATE);
        let body = filtered.window.line(form.intro());
        (
            filtered,
            Song::with_form(seed, 0, c, open, SAMPLE_RATE),
            body,
        )
    }

    #[test]
    fn a_filtered_intro_opens_to_the_lanes_own_corner() {
        let c = Controls::default();
        let (mut filtered, mut open, body) = filtered_and_open(c);
        let to = body + BLOCK;
        let (got, want) = (corners(&mut filtered, to, c), corners(&mut open, to, c));
        for (i, lane) in Lane::ALL
            .iter()
            .enumerate()
            .filter(|(_, l)| **l != Lane::Texture)
        {
            let own = corner_at(&want, 0, i);
            let path: Vec<f32> = got
                .iter()
                .map(|&(_, c)| c[i].expect("a tape tunes"))
                .collect();
            assert_eq!(
                path[0],
                FILTERED_HZ.min(own),
                "{lane:?} opens from elsewhere"
            );
            assert!(path.is_sorted(), "{lane:?} closes somewhere in the intro");
            assert_eq!(
                path[path.len() - 1],
                own,
                "{lane:?} never reaches its own corner"
            );
        }
    }

    #[test]
    fn a_sweep_and_a_grade_take_the_lower_corner() {
        // a day song graded brighter from its first latch on, across its
        // filtered intro, and a night one whose bass sits under the sweep's
        // corner
        let day = Controls::default().with_daylight(0.6);
        let night = Controls::default().with_daylight(0.0);
        for (c, asked) in [(day, day.with_daylight(1.0)), (night, night)] {
            let (mut filtered, mut open, body) = filtered_and_open(c);
            let to = body + BLOCK;
            let (got, want) = (
                corners(&mut filtered, to, asked),
                corners(&mut open, to, asked),
            );
            for (i, lane) in Lane::ALL
                .iter()
                .enumerate()
                .filter(|(_, l)| **l != Lane::Texture)
            {
                if asked != c && filtered.chains[i].grades() {
                    let moved = filtered.automation.graded_until() > 0;
                    assert!(moved, "{lane:?}'s grade never moved");
                }
                for &(end, g) in &got {
                    let (g, w) = (g[i].expect("a tape"), corner_at(&want, end - 1, i));
                    assert!(g <= w, "{c:?} {lane:?} by {end}: {g} over its own {w}");
                    if w <= FILTERED_HZ {
                        assert_eq!(g, w, "{lane:?} under the sweep moved");
                    }
                }
            }
        }
    }

    #[test]
    fn the_dip_closes_and_reopens_within_a_beat() {
        let c = Controls::default();
        let seed = seed_where(|f| f.dip().is_some());
        let mut song = Song::new(seed, 0, c, SAMPLE_RATE);
        let dip = song_form(seed, 0, c).dip().expect("a dip");
        let beat = samples_at(song.score.beat_s(), SAMPLE_RATE);
        let (line, next) = (song.window.bars[dip], song.window.bars[dip + 1]);
        let keys = index(Lane::Keys);
        let path = corners(&mut song, next + beat + 2 * COEFF_BLOCK, c);
        let own = corner_at(&path, line - COEFF_BLOCK, keys);
        assert_eq!(
            corner_at(&path, line + beat + COEFF_BLOCK, keys),
            FILTERED_HZ,
            "the dip never closed"
        );
        assert_eq!(
            corner_at(&path, next + beat + COEFF_BLOCK, keys),
            own,
            "the dip never reopened"
        );
    }

    #[test]
    fn every_lane_plays_through_the_edges_but_the_room() {
        let c = Controls::default();
        let seed = seed_where(|f| f.entry() == Some(Edge::Fade) && f.exit() == Some(Edge::Fade));
        let form = song_form(seed, 0, c);
        let mut faded = Song::new(seed, 0, c, SAMPLE_RATE).at_levels(UNITY);
        let plain = form.with_edges(None, None);
        let mut twin = Song::with_form(seed, 0, c, plain, SAMPLE_RATE).at_levels(UNITY);
        let edges = faded.edges;
        let len = faded.window.len;
        let [got, want] = [&mut faded, &mut twin].map(|song| play_to(song, len));
        for (i, lane) in Lane::ALL.iter().enumerate() {
            // the room by name, not by `fades`, so a room that fades shows
            let edge = |n: usize| {
                if *lane == Lane::Texture {
                    1.0
                } else {
                    edge_at(&edges, n)
                }
            };
            let off = (got[i].iter().zip(&want[i]).enumerate())
                .position(|(n, (g, w))| g.to_bits() != (w * edge(n)).to_bits());
            assert_eq!(off, None, "{lane:?} plays off its edges' gain");
        }
    }

    #[test]
    fn a_fade_out_leaves_its_tails_at_its_floor() {
        let c = Controls::default();
        let floor = gain_of_db(FADE_FLOOR_DB);
        let seed = seed_where(|f| f.exit() == Some(Edge::Fade));
        let form = song_form(seed, 0, c);
        let mut faded = Song::new(seed, 0, c, SAMPLE_RATE).at_levels(UNITY);
        let plain_exit = form.clone().with_edges(form.entry(), None);
        let mut twin = Song::with_form(seed, 0, c, plain_exit, SAMPLE_RATE).at_levels(UNITY);
        let (len, tail) = (faded.window.len, SAMPLE_RATE as usize);
        let [got, want] = [&mut faded, &mut twin].map(|song| play_to(song, len + tail));
        let mut ringing = 0;
        for (i, lane) in Lane::ALL
            .iter()
            .enumerate()
            .filter(|(_, l)| **l != Lane::Texture)
        {
            let (got, want) = (&got[i][len..], &want[i][len..]);
            // a lane whose last note died before the end rings nothing past it
            ringing += usize::from(energy(want) > 0.0);
            let off = got
                .iter()
                .zip(want)
                .position(|(g, w)| g.to_bits() != (w * floor).to_bits());
            assert_eq!(off, None, "{lane:?}'s tail plays off the fade's floor");
        }
        assert!(ringing > 0, "no lane rings a tail to fade");
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
                crackle.strike(last, last + 1, &mut ring, true, sr);
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
        let mut run = Schedule::plain(0, 0, sr);
        run.start_song_at(c, UNITY);
        let loop_len = loop_len(&run, sr);
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
