//! A song's FORM: which lanes strike in each bar of the window a song cuts
//! from its tiled loop, and how its edges fade or filter. The grammar's
//! weights are the measured corpora's (LOFI-BIBLE R11); a lane only ever
//! drops out over the loop, so a form adds nothing to the music.

use std::fmt;

use super::{GEN_LOOP_BARS, chance, pick};
use crate::dsp::{NoiseStream, splitmix64};

/// Seeds the form's own stream, so a change to the loop's draws never moves a
/// form's, nor the reverse.
const FORM_SALT: u64 = 0xF0_4A_55_ED;

/// Which lanes strike in one bar, named as the `Score` names their parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plays {
    pub(crate) pad: bool,
    pub(crate) sparkle: bool,
    pub(crate) keys: bool,
    pub(crate) drums: bool,
    pub(crate) texture: bool,
    pub(crate) bass: bool,
}

impl Plays {
    pub(crate) const ALL: Self = Self {
        pad: true,
        sparkle: true,
        keys: true,
        drums: true,
        texture: true,
        bass: true,
    };

    /// The bed alone: the pad's chords over the room.
    const BED: Self = Self {
        pad: true,
        texture: true,
        sparkle: false,
        keys: false,
        drums: false,
        bass: false,
    };

    /// Every lane but the lead, which sings in the dense cores only.
    const BAND: Self = Self {
        sparkle: false,
        ..Self::ALL
    };
}

/// How a song's intro or outro moves its music lanes' level or colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    /// The lanes enter and leave at their own level and colour.
    Plain,
    /// The music lanes' level rises across the intro, or falls across the outro.
    Fade,
    /// The music lanes' low-pass opens across the intro, or closes across the
    /// outro.
    Filter,
}

/// One song's form: an intro, a body of whole loops starting on loop bar 0,
/// and an outro, each bar naming the lanes that strike in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Form {
    intro: usize,
    body: usize,
    outro: usize,
    plays: Vec<Plays>,
    entry: Edge,
    exit: Edge,
    dip: Option<usize>,
}

impl Form {
    /// Bars in the song.
    pub fn bars(&self) -> usize {
        self.plays.len()
    }

    /// Bars before the body.
    pub fn intro(&self) -> usize {
        self.intro
    }

    /// Bars in the body, a whole number of loops.
    pub fn body(&self) -> usize {
        self.body
    }

    /// Bars after the body.
    pub fn outro(&self) -> usize {
        self.outro
    }

    /// How the intro moves the music lanes.
    pub fn entry(&self) -> Edge {
        self.entry
    }

    /// How the outro moves the music lanes.
    pub fn exit(&self) -> Edge {
        self.exit
    }

    /// The song bar, phrase 1's last, whose music lanes low-pass for one bar,
    /// if any.
    pub fn dip(&self) -> Option<usize> {
        self.dip
    }

    /// The loop bar the song starts on: the intro ends where the loop does, so
    /// the body's first beat is the loop's.
    pub(crate) fn lead_in(&self) -> usize {
        (GEN_LOOP_BARS - self.intro % GEN_LOOP_BARS) % GEN_LOOP_BARS
    }

    /// The lanes that strike in song bar `bar`.
    pub(crate) fn plays(&self, bar: usize) -> &Plays {
        &self.plays[bar]
    }

    /// Whether some bar mutes some lane: a song whose invariance a test
    /// proves must change somewhere.
    #[cfg(test)]
    pub(crate) fn mutes_any(&self) -> bool {
        self.plays.iter().any(|&plays| plays != Plays::ALL)
    }

    /// Whether the form fades or filters somewhere: an edge or a dip.
    #[cfg(test)]
    pub(crate) fn edged(&self) -> bool {
        self.entry != Edge::Plain || self.exit != Edge::Plain || self.dip.is_some()
    }

    /// The form with its intro's edge `entry` and its outro's `exit`, the
    /// rest kept, for a test's twin.
    #[cfg(test)]
    pub(crate) fn with_edges(mut self, entry: Edge, exit: Edge) -> Self {
        (self.entry, self.exit) = (entry, exit);
        self
    }

    /// The lanes song bar `bar` strikes, to change for a test.
    #[cfg(test)]
    pub(crate) fn plays_mut(&mut self, bar: usize) -> &mut Plays {
        &mut self.plays[bar]
    }

    /// `loops` loops of every lane, with no intro or outro: a song as it was
    /// before its form.
    #[cfg(test)]
    pub(crate) fn plain(loops: usize) -> Self {
        Self {
            intro: 0,
            body: loops * GEN_LOOP_BARS,
            outro: 0,
            plays: vec![Plays::ALL; loops * GEN_LOOP_BARS],
            entry: Edge::Plain,
            exit: Edge::Plain,
            dip: None,
        }
    }
}

/// A lane's name and whether a bar's [`Plays`] strikes it.
type Named = (&'static str, fn(&Plays) -> bool);

/// Every lane, as [`Form`]'s display names it.
const LANE_NAMES: [Named; 6] = [
    ("pad", |p| p.pad),
    ("sparkle", |p| p.sparkle),
    ("keys", |p| p.keys),
    ("drums", |p| p.drums),
    ("texture", |p| p.texture),
    ("bass", |p| p.bass),
];

/// Each part's bar count, then each lane out in it as the bars it is out:
/// `intro 4: drums 0-3 · body 40: sparkle 20-27 · outro 4 fade`.
impl fmt::Display for Form {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let body = self.intro + self.body;
        let parts = [
            ("intro", 0..self.intro, self.entry),
            ("body", self.intro..body, Edge::Plain),
            ("outro", body..self.bars(), self.exit),
        ];
        for (i, (name, bars, edge)) in parts.into_iter().enumerate() {
            if i > 0 {
                write!(f, " · ")?;
            }
            write!(f, "{name} {}", bars.len())?;
            match edge {
                Edge::Plain => {}
                Edge::Fade => write!(f, " fade")?,
                Edge::Filter => write!(f, " filter")?,
            }
            let mut sep = ":";
            for (lane, plays) in LANE_NAMES {
                let out: Vec<usize> = bars.clone().filter(|&b| !plays(&self.plays[b])).collect();
                for run in out.chunk_by(|a, b| a + 1 == *b) {
                    let (first, last) = (run[0], run[run.len() - 1]);
                    if first == last {
                        write!(f, "{sep} {lane} {first}")?;
                    } else {
                        write!(f, "{sep} {lane} {first}-{last}")?;
                    }
                    sep = ",";
                }
            }
        }
        if let Some(dip) = self.dip {
            write!(f, " · dip {dip}")?;
        }
        Ok(())
    }
}

/// How a song opens.
#[derive(Clone, Copy)]
enum Opening {
    /// Some lanes sit out the intro.
    Sparse(Plays),
    /// The band plays the intro through the edge.
    Edged(Edge),
    /// No intro: the body's first bar is the song's.
    Cold,
}

/// How a song closes.
#[derive(Clone, Copy)]
enum Closing {
    Sparse(Plays),
    /// The loop's first chord on the pad, then a bar of the room alone, which
    /// outlasts the chord's tail.
    RingOut,
    Edged(Edge),
    Cold,
}

/// The intros and their weights: the judge corpus's start types, counted
/// (research-form-corpus.md). The lead sits out every intro.
const INTRO_KINDS: [(Opening, u32); 8] = [
    (
        Opening::Sparse(Plays {
            keys: true,
            ..Plays::BED
        }),
        27,
    ),
    (
        Opening::Sparse(Plays {
            keys: true,
            bass: true,
            ..Plays::BED
        }),
        13,
    ),
    (
        Opening::Sparse(Plays {
            keys: true,
            drums: true,
            ..Plays::BED
        }),
        10,
    ),
    (Opening::Sparse(Plays::BED), 7),
    (
        Opening::Sparse(Plays {
            drums: true,
            bass: true,
            ..Plays::BED
        }),
        5,
    ),
    (Opening::Edged(Edge::Fade), 13),
    (Opening::Edged(Edge::Filter), 10),
    (Opening::Cold, 5),
];

/// The outros and their weights: the corpus's end types, counted; the ring-out
/// takes its ring-outs and its noise-only outros.
const OUTRO_KINDS: [(Closing, u32); 7] = [
    (
        Closing::Sparse(Plays {
            keys: true,
            ..Plays::BED
        }),
        14,
    ),
    (
        Closing::Sparse(Plays {
            keys: true,
            drums: true,
            ..Plays::BED
        }),
        11,
    ),
    (
        Closing::Sparse(Plays {
            keys: true,
            bass: true,
            ..Plays::BED
        }),
        10,
    ),
    (Closing::RingOut, 18),
    (Closing::Edged(Edge::Fade), 31),
    (Closing::Edged(Edge::Filter), 7),
    (Closing::Cold, 4),
];

/// How the body's middle phrase thins, and its weight in tenths (opinion,
/// fitted to the corpus's drop-out shares).
#[derive(Clone, Copy, PartialEq)]
enum Thinning {
    /// The beat drops out.
    Break,
    /// The bass drops out.
    BassDrop,
    /// Only the lead rests.
    Rest,
}

const THINNINGS: [(Thinning, u32); 3] = [
    (Thinning::Break, 3),
    (Thinning::BassDrop, 5),
    (Thinning::Rest, 2),
];

/// Short and long intros' bars, and the long one's share: the corpus counts 26
/// eight-bar intros to 14 four-bar ones.
const INTRO_BARS: [usize; 2] = [4, 8];
const LONG_INTRO_P: f32 = 0.65;

/// The body's phrases, each a loop: drawn at even odds (opinion), which puts
/// the median song between the two corpora's.
const PHRASES: [usize; 3] = [4, 5, 6];

/// A sparse outro's bars, an edged one's (short, long), and the long one's
/// share; the ring-out's bars. Opinion within the corpus's outro lengths.
const SPARSE_OUTRO_BARS: usize = 4;
const EDGED_OUTRO_BARS: [usize; 2] = [4, 8];
const LONG_EDGED_OUTRO_P: f32 = 0.3;
const RING_OUT_BARS: usize = 2;

/// The thinning's span: the phrase's last half, or with this chance all of it.
const WHOLE_THINNING_P: f32 = 0.4;
/// The bass leaves with the beat in a break: the corpus's 52%.
const LOW_WITH_BREAK_P: f32 = 0.5;

/// A dense phrase cuts its beat for its last bar or two with this chance,
/// the two-bar cut with the second.
const CUT_P: f32 = 0.15;
const LONG_CUT_P: f32 = 0.3;
const CUT_BARS: [usize; 2] = [1, 2];

/// The bass sits out the last half of the phrase after the thinning with this
/// chance, in bodies of at least `BASS_DROP_MIN_PHRASES`.
const BASS_DROP_P: f32 = 0.2;
const BASS_DROP_MIN_PHRASES: usize = 5;

/// A song low-passes phrase 1's last bar with this chance: the corpus's 20% of
/// tracks with a mid-track filter moment, mostly one bar long.
const DIP_P: f32 = 0.2;

/// The band one unit draw `u` falls in, each band as wide as its weight.
fn band(u: f32, weights: &[u32]) -> usize {
    let total: u32 = weights.iter().sum();
    let at = ((u * total as f32) as u32).min(total - 1);
    let mut below = 0;
    weights
        .iter()
        .position(|&w| {
            below += w;
            at < below
        })
        .unwrap_or(weights.len() - 1)
}

/// The row of `table` one unit draw falls in.
fn kind<T: Copy>(u: f32, table: &[(T, u32)]) -> T {
    let weights: Vec<u32> = table.iter().map(|&(_, w)| w).collect();
    table[band(u, &weights)].0
}

/// Song seed `seed`'s form, from its own stream. Every draw is made whether or
/// not its branch applies, so a change to one row of the grammar moves no
/// other row's draws.
pub(crate) fn arrange(seed: u64) -> Form {
    let mut rng = NoiseStream::new(splitmix64(seed ^ FORM_SALT));
    let opening = kind(rng.unit(), &INTRO_KINDS);
    let long_intro = chance(&mut rng, LONG_INTRO_P);
    let phrases = PHRASES[pick(&mut rng, PHRASES.len())];
    let thinning = kind(rng.unit(), &THINNINGS);
    let whole = chance(&mut rng, WHOLE_THINNING_P);
    let low_with_break = chance(&mut rng, LOW_WITH_BREAK_P);
    let cuts: [Option<usize>; PHRASES[PHRASES.len() - 1]] = std::array::from_fn(|_| {
        let cut = chance(&mut rng, CUT_P);
        let long = chance(&mut rng, LONG_CUT_P);
        cut.then_some(CUT_BARS[usize::from(long)])
    });
    let bass_drop = chance(&mut rng, BASS_DROP_P);
    let dip = chance(&mut rng, DIP_P);
    let closing = kind(rng.unit(), &OUTRO_KINDS);
    let long_outro = chance(&mut rng, LONG_EDGED_OUTRO_P);

    let (intro_plays, intro, entry) = match opening {
        Opening::Sparse(plays) => (plays, INTRO_BARS[usize::from(long_intro)], Edge::Plain),
        Opening::Edged(edge) => (Plays::BAND, INTRO_BARS[usize::from(long_intro)], edge),
        Opening::Cold => (Plays::BAND, 0, Edge::Plain),
    };
    let mut plays = vec![intro_plays; intro];

    let mut body = vec![Plays::ALL; phrases * GEN_LOOP_BARS];
    let thin = phrases / 2;
    let phrase = |p: usize| p * GEN_LOOP_BARS..(p + 1) * GEN_LOOP_BARS;
    let half = |p: usize| p * GEN_LOOP_BARS + GEN_LOOP_BARS / 2..(p + 1) * GEN_LOOP_BARS;
    for bar in &mut body[phrase(thin)] {
        bar.sparkle = false;
    }
    let span = if whole { phrase(thin) } else { half(thin) };
    let keeps_low = match thinning {
        Thinning::Break => {
            for bar in &mut body[span] {
                bar.drums = false;
                bar.bass &= !low_with_break;
            }
            !low_with_break
        }
        Thinning::BassDrop => {
            for bar in &mut body[span] {
                bar.bass = false;
            }
            false
        }
        Thinning::Rest => true,
    };
    // a dense phrase cuts only into a phrase that opens on the beat, and never
    // into the outro, where a cut would just start a beatless outro early
    let opens_on_beat = |p: usize| p != thin || !(thinning == Thinning::Break && whole);
    let mut phrase_1_cut = false;
    for (p, cut) in cuts.iter().enumerate().take(phrases - 1) {
        if let Some(len) = cut
            && p != thin
            && opens_on_beat(p + 1)
        {
            let end = (p + 1) * GEN_LOOP_BARS;
            for bar in &mut body[end - len..end] {
                bar.drums = false;
            }
            phrase_1_cut |= p == 1;
        }
    }
    if bass_drop && phrases >= BASS_DROP_MIN_PHRASES && keeps_low {
        for bar in &mut body[half(thin + 1)] {
            bar.bass = false;
        }
    }
    let dip = (dip && !phrase_1_cut).then_some(intro + 2 * GEN_LOOP_BARS - 1);
    plays.extend(body);

    let (outro, exit) = match closing {
        Closing::Sparse(p) => {
            plays.extend([p; SPARSE_OUTRO_BARS]);
            (SPARSE_OUTRO_BARS, Edge::Plain)
        }
        Closing::RingOut => {
            plays.extend([
                Plays::BED,
                Plays {
                    pad: false,
                    ..Plays::BED
                },
            ]);
            (RING_OUT_BARS, Edge::Plain)
        }
        Closing::Edged(edge) => {
            let bars = EDGED_OUTRO_BARS[usize::from(long_outro)];
            plays.extend(std::iter::repeat_n(Plays::BAND, bars));
            (bars, edge)
        }
        Closing::Cold => (0, Edge::Plain),
    };
    Form {
        intro,
        body: phrases * GEN_LOOP_BARS,
        outro,
        plays,
        entry,
        exit,
        dip,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::{Mood, compose};

    /// The forms every test here sweeps: [`arrange`] reads no mood, so these
    /// are the population whatever the tempo.
    const FORMS: u64 = 512;

    fn forms() -> impl Iterator<Item = (u64, Form)> {
        (0..FORMS).map(|seed| (seed, arrange(seed)))
    }

    /// The body phrase that thins: the middle one, rounded down.
    fn thinner(form: &Form) -> usize {
        (form.body() / GEN_LOOP_BARS) / 2
    }

    /// The body's bars, by index into the song.
    fn body(form: &Form) -> std::ops::Range<usize> {
        form.intro()..form.intro() + form.body()
    }

    /// Runs `(first, len)` of `bars` where `out` holds, within `range`.
    fn runs(
        form: &Form,
        range: std::ops::Range<usize>,
        out: impl Fn(&Plays) -> bool,
    ) -> Vec<(usize, usize)> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for bar in range {
            if out(form.plays(bar)) {
                match runs.last_mut() {
                    Some((first, len)) if *first + *len == bar => *len += 1,
                    _ => runs.push((bar, 1)),
                }
            }
        }
        runs
    }

    #[test]
    fn every_form_is_well_formed() {
        for (seed, form) in forms() {
            let at = |what: &str| format!("seed {seed}: {what}: {form}");
            assert_eq!(
                form.plays.len(),
                form.intro() + form.body() + form.outro(),
                "{}",
                at("rows")
            );
            assert_eq!(form.bars(), form.plays.len(), "{}", at("bars"));
            assert_eq!(
                form.body() % GEN_LOOP_BARS,
                0,
                "{}",
                at("body in whole loops")
            );
            assert_eq!(
                (form.lead_in() + form.intro()) % GEN_LOOP_BARS,
                0,
                "{}",
                at("body on loop bar 0")
            );
            assert!([0, 4, 8].contains(&form.intro()), "{}", at("intro bars"));
            assert!([0, 2, 4, 8].contains(&form.outro()), "{}", at("outro bars"));
            let ring_out = form.outro() == 2;
            let last = form.bars() - 1;
            for bar in 0..form.bars() {
                let plays = form.plays(bar);
                assert!(plays.texture, "{}", at(&format!("texture out at {bar}")));
                assert!(
                    plays.pad || (ring_out && bar == last),
                    "{}",
                    at(&format!("pad out at {bar}"))
                );
            }
            let t = thinner(&form);
            for bar in 0..form.bars() {
                let phrase = bar.checked_sub(form.intro()).map(|b| b / GEN_LOOP_BARS);
                let dense = body(&form).contains(&bar) && phrase != Some(t);
                assert_eq!(
                    form.plays(bar).sparkle,
                    dense,
                    "{}",
                    at(&format!("lead at {bar}"))
                );
            }
            // a row changes on a 4-bar line, but for a cut or a ring-out
            let cuts: Vec<(usize, usize)> = runs(&form, body(&form), |p| !p.drums)
                .into_iter()
                .filter(|&(_, len)| len <= 2)
                .collect();
            for bar in 1..form.bars() {
                if ring_out && bar == last {
                    continue;
                }
                let (mut a, b) = (*form.plays(bar - 1), *form.plays(bar));
                if cuts.iter().any(|&(first, _)| first == bar) {
                    a.drums = b.drums;
                }
                if a != b {
                    assert_eq!(
                        bar % 4,
                        0,
                        "{}",
                        at(&format!("a change off the line at {bar}"))
                    );
                }
            }
            let last_phrase = body(&form).end - GEN_LOOP_BARS..body(&form).end;
            for &(first, len) in &cuts {
                assert!(
                    !last_phrase.contains(&first),
                    "{}",
                    at("a cut before the outro")
                );
                let before = form.plays(first - 1).drums;
                let after = form.plays(first + len).drums;
                assert!(
                    before && after,
                    "{}",
                    at(&format!("a cut at {first} touches a beat-out bar"))
                );
                assert_eq!(
                    (first + len - form.intro()) % GEN_LOOP_BARS,
                    0,
                    "{}",
                    at("a cut ends a phrase")
                );
            }
            for (edge, part) in [
                (form.entry(), 0..form.intro()),
                (form.exit(), body(&form).end..form.bars()),
            ] {
                if edge != Edge::Plain {
                    assert!(!part.is_empty(), "{}", at("an edge with no bars"));
                    for bar in part {
                        let mut band = *form.plays(bar);
                        band.sparkle = true;
                        assert_eq!(
                            band,
                            Plays::ALL,
                            "{}",
                            at(&format!("an edge's band at {bar}"))
                        );
                    }
                }
            }
            if let Some(dip) = form.dip() {
                assert_eq!(
                    dip,
                    form.intro() + 2 * GEN_LOOP_BARS - 1,
                    "{}",
                    at("the dip's bar")
                );
                assert!(form.plays(dip).drums, "{}", at("a dip on a cut"));
            }
        }
    }

    #[test]
    fn the_kinds_take_their_measured_weights() {
        for (name, weights) in [
            ("intro", INTRO_KINDS.map(|(_, w)| w).to_vec()),
            ("outro", OUTRO_KINDS.map(|(_, w)| w).to_vec()),
        ] {
            let total: u32 = weights.iter().sum();
            let mut below = 0;
            for (row, &w) in weights.iter().enumerate() {
                // half a unit inside each edge of the row's band
                for at in [below as f32 + 0.5, (below + w) as f32 - 0.5] {
                    let u = at / total as f32;
                    assert_eq!(band(u, &weights), row, "{name} row {row} at {u}");
                }
                below += w;
            }
        }
        assert_eq!(INTRO_KINDS.map(|(_, w)| w), [27, 13, 10, 7, 5, 13, 10, 5]);
        assert_eq!(OUTRO_KINDS.map(|(_, w)| w), [14, 11, 10, 18, 31, 7, 4]);
    }

    fn median(mut xs: Vec<f32>) -> f32 {
        xs.sort_by(f32::total_cmp);
        xs[xs.len() / 2]
    }

    fn share(hits: usize, of: usize) -> f32 {
        hits as f32 / of as f32
    }

    /// The bands are the measured corpora's (LOFI-BIBLE R11): Fields-Meyer's
    /// 150 Lofi Girl tracks and the judge's 100-track corpus. How far a
    /// statistic may sit from its corpus is opinion.
    #[test]
    fn the_forms_land_in_the_measured_corpus() {
        let (mut songs, mut outros) = (Vec::new(), Vec::new());
        let (mut intros, mut eight, mut drum, mut bass, mut first) = (0, 0, 0, 0, Vec::new());
        for (seed, form) in forms() {
            for mood in [Mood::Day, Mood::Night] {
                let bar_s = compose(mood, seed).bar_s();
                songs.push(form.bars() as f32 * bar_s);
                outros.push(form.outro() as f32 * bar_s);
            }
            intros += usize::from(form.intro() > 0);
            eight += usize::from(form.intro() == 8);
            let drops = runs(&form, body(&form), |p| !p.drums);
            drum += usize::from(!drops.is_empty());
            bass += usize::from(!runs(&form, body(&form), |p| !p.bass).is_empty());
            if let Some(&(bar, _)) = drops.first() {
                first.push(bar as f32 / form.bars() as f32);
            }
        }
        let n = FORMS as usize;
        let checks = [
            ("median song s", median(songs), 147.0, 212.0),
            ("intro share", share(intros, n), 0.9, 1.0),
            ("8-bar share of intros", share(eight, intros), 0.55, 0.75),
            ("median outro s", median(outros), 7.0, 28.0),
            ("drum-drop share", share(drum, n), 0.5, 0.7),
            ("median first-drop position", median(first), 0.38, 0.56),
            ("bass-drop share", share(bass, n), 0.64, 0.84),
        ];
        for (name, value, lo, hi) in checks {
            assert!(
                (lo..=hi).contains(&value),
                "{name} {value} outside [{lo}, {hi}]"
            );
        }
    }

    /// FNV-1a over 64-bit words, as `compose`'s goldens hash.
    fn fnv1a(words: impl IntoIterator<Item = u64>) -> u64 {
        words.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, w| {
            (h ^ w).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Seeds 0..64's forms, every row and edge: a change to the grammar, its
    /// weights or its draw order moves this, a sound change, never a test fix.
    #[test]
    fn every_seed_arranges_the_form_it_did() {
        let mut words = Vec::new();
        for seed in 0..64 {
            let form = arrange(seed);
            words.extend([form.intro(), form.body(), form.outro()].map(|n| n as u64));
            for p in &form.plays {
                let bits = [p.pad, p.sparkle, p.keys, p.drums, p.texture, p.bass];
                words.push(bits.iter().fold(0, |w, &b| w << 1 | u64::from(b)));
            }
            words.extend([form.entry() as u64, form.exit() as u64]);
            words.push(form.dip().map_or(u64::MAX, |d| d as u64));
        }
        let h = fnv1a(words);
        assert_eq!(h, FORM_GOLDEN, "{h:#018x}");
    }

    const FORM_GOLDEN: u64 = 0x369c_4d6c_bf26_9767;

    #[test]
    fn a_form_prints_its_parts() {
        assert_eq!(Form::plain(4).to_string(), "intro 0 · body 32 · outro 0");
        let mut cut = Form::plain(2);
        cut.plays[7].drums = false;
        assert_eq!(cut.to_string(), "intro 0 · body 16: drums 7 · outro 0");
    }
}
