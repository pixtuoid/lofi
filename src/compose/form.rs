//! A song's FORM: which layers strike in each bar of the window a song cuts
//! from its tiled loop, and how its edges fade or filter. The grammar's
//! weights are the measured corpora's (LOFI-BIBLE R11); a layer only ever
//! drops out over the loop, so a form adds nothing to the music.

use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Range;

use super::{GEN_LOOP_BARS, Lane, chance, kind, pick};
use crate::dsp::{NoiseStream, splitmix64};

/// Seeds the form's own stream, so a change to the loop's draws never moves a
/// form's, nor the reverse.
const FORM_SALT: u64 = 0xF0_4A_55_ED;

/// What a form arranges: R11's measured layers, whose "keys" band holds the
/// lead too. The texture is in none: the room plays every bar of every song
/// and through every edge, which links one song to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layer {
    Pad,
    Keys,
    Drums,
    Bass,
}

impl Layer {
    /// Every layer, each at its index.
    const ALL: [Layer; 4] = [Layer::Pad, Layer::Keys, Layer::Drums, Layer::Bass];

    /// Each layer's name in [`Layer::ALL`] order, as a form prints it.
    const NAMES: [&'static str; Layer::ALL.len()] = ["pad", "keys", "drums", "bass"];

    /// The layer `lane` plays in, or none for the room.
    pub(crate) const fn of(lane: Lane) -> Option<Layer> {
        match lane {
            Lane::Pad => Some(Layer::Pad),
            Lane::Sparkle | Lane::Keys => Some(Layer::Keys),
            Lane::Drums => Some(Layer::Drums),
            Lane::Bass => Some(Layer::Bass),
            Lane::Texture => None,
        }
    }

    const fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }
}

/// Which layers strike in one bar, each at its index in [`Layer::ALL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plays([bool; Layer::ALL.len()]);

impl Plays {
    pub(crate) const ALL: Self = Self([true; Layer::ALL.len()]);

    /// The bed alone: the pad's chords over the room.
    const BED: Self = Self([false; Layer::ALL.len()]).with(Layer::Pad, true);

    /// These layers, with `layer` striking or not.
    const fn with(mut self, layer: Layer, strikes: bool) -> Self {
        self.0[layer as usize] = strikes;
        self
    }

    /// Whether `layer` strikes.
    const fn has(self, layer: Layer) -> bool {
        self.0[layer as usize]
    }

    /// Whether `lane` strikes: its layer's say, and the room always.
    pub(crate) const fn strikes(self, lane: Lane) -> bool {
        match Layer::of(lane) {
            Some(layer) => self.has(layer),
            None => true,
        }
    }

    /// Sets whether `layer` strikes.
    pub(crate) fn set(&mut self, layer: Layer, strikes: bool) {
        self.0[layer as usize] = strikes;
    }
}

/// How a song's intro or outro moves its music lanes' level or colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Edge {
    /// The music lanes' level rises across the intro, or falls across the outro.
    Fade,
    /// The music lanes' low-pass opens across the intro, or closes across the
    /// outro.
    Filter,
}

/// A song's intro, or an outro that doesn't ring out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    /// None: the body's first bar is the song's, or its last.
    Cold,
    /// Bars at the music lanes' own level and colour, whichever layers play.
    Plain(NonZeroUsize),
    /// Bars every layer plays through an edge.
    Edged(NonZeroUsize, Edge),
}

impl End {
    const fn bars(self) -> usize {
        match self {
            Self::Cold => 0,
            Self::Plain(bars) | Self::Edged(bars, _) => bars.get(),
        }
    }

    const fn edge(self) -> Option<Edge> {
        match self {
            Self::Edged(_, edge) => Some(edge),
            _ => None,
        }
    }

    /// The end with `edge` over its bars, or none.
    fn with_edge(self, edge: Option<Edge>) -> Self {
        match (self, edge) {
            (Self::Plain(bars) | Self::Edged(bars, _), Some(edge)) => Self::Edged(bars, edge),
            (Self::Plain(bars) | Self::Edged(bars, _), None) => Self::Plain(bars),
            (Self::Cold, _) => Self::Cold,
        }
    }
}

/// A song's outro.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outro {
    Ends(End),
    /// The loop's first chord on the pad, then a bar of the room alone, which
    /// outlasts the chord's tail.
    RingOut,
}

impl Outro {
    const fn bars(self) -> usize {
        match self {
            Self::Ends(end) => end.bars(),
            Self::RingOut => RING_OUT_BARS,
        }
    }

    const fn edge(self) -> Option<Edge> {
        match self {
            Self::Ends(end) => end.edge(),
            Self::RingOut => None,
        }
    }
}

/// `n` bars, of a part that has some.
const fn bars(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(bars) => bars,
        None => panic!("a part with bars has some"),
    }
}

/// One song's form: an intro, a body of whole loops starting on loop bar 0,
/// and an outro, each bar naming the layers that strike in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Form {
    intro: End,
    phrases: usize,
    outro: Outro,
    /// Whether the last bar of phrase [`DIP_PHRASE`] low-passes.
    dip: bool,
    plays: Vec<Plays>,
}

impl Form {
    fn new(intro: End, phrases: usize, outro: Outro, dip: bool, plays: Vec<Plays>) -> Self {
        let form = Self {
            intro,
            phrases,
            outro,
            dip,
            plays,
        };
        debug_assert_eq!(form.bars(), form.parts()[2].end);
        form
    }

    /// Bars in the song.
    pub fn bars(&self) -> usize {
        self.plays.len()
    }

    /// Bars before the body.
    pub fn intro(&self) -> usize {
        self.intro.bars()
    }

    /// Bars in the body, a whole number of loops.
    pub fn body(&self) -> usize {
        self.phrases * GEN_LOOP_BARS
    }

    /// Bars after the body.
    pub fn outro(&self) -> usize {
        self.outro.bars()
    }

    /// The intro's, the body's and the outro's bars.
    pub(crate) fn parts(&self) -> [Range<usize>; 3] {
        let body = self.intro()..self.intro() + self.body();
        [
            0..body.start,
            body.clone(),
            body.end..body.end + self.outro(),
        ]
    }

    /// How the intro moves the music lanes, if it does.
    pub(crate) fn entry(&self) -> Option<Edge> {
        self.intro.edge()
    }

    /// How the outro moves the music lanes, if it does.
    pub(crate) fn exit(&self) -> Option<Edge> {
        self.outro.edge()
    }

    /// The song bar whose music lanes low-pass for one bar, if any.
    pub(crate) fn dip(&self) -> Option<usize> {
        (self.dip).then(|| self.intro() + (DIP_PHRASE + 1) * GEN_LOOP_BARS - 1)
    }

    /// Whether the song closes on a ring-out, whose last bar the pad sits out.
    #[cfg(test)]
    pub(crate) fn rings_out(&self) -> bool {
        self.outro == Outro::RingOut
    }

    /// The loop bar the song starts on: the intro ends where the loop does, so
    /// the body's first beat is the loop's.
    pub(crate) fn lead_in(&self) -> usize {
        (GEN_LOOP_BARS - self.intro() % GEN_LOOP_BARS) % GEN_LOOP_BARS
    }

    /// The layers that strike in song bar `bar`.
    pub(crate) fn plays(&self, bar: usize) -> &Plays {
        &self.plays[bar]
    }

    /// Whether some bar mutes some layer.
    fn mutes_any(&self) -> bool {
        self.plays.iter().any(|&plays| plays != Plays::ALL)
    }

    /// Whether the form fades or filters somewhere: an edge or a dip.
    pub(crate) fn edged(&self) -> bool {
        self.entry().is_some() || self.exit().is_some() || self.dip
    }

    /// The form a band with no beat plays: its intro and outro layered as
    /// drawn, its body whole, and no edge or dip. A body's drops and an edge
    /// move a groove; with no beat they thin R11's drums-out band further,
    /// which the judge's Empty takes heard as worse and the corpus as no
    /// closer (the figures are in the commit that made this rule).
    pub(crate) fn beatless(mut self) -> Self {
        let body = self.parts()[1].clone();
        self.plays[body].fill(Plays::ALL);
        self.intro = self.intro.with_edge(None);
        if let Outro::Ends(end) = self.outro {
            self.outro = Outro::Ends(end.with_edge(None));
        }
        self.dip = false;
        self
    }

    /// The form with its intro's edge `entry` and its outro's `exit`, the
    /// rest kept, for a test's twin.
    #[cfg(test)]
    pub(crate) fn with_edges(mut self, entry: Option<Edge>, exit: Option<Edge>) -> Self {
        self.intro = self.intro.with_edge(entry);
        if let Outro::Ends(end) = self.outro {
            self.outro = Outro::Ends(end.with_edge(exit));
        }
        self
    }

    /// The layers song bar `bar` strikes, to change for a test.
    #[cfg(test)]
    pub(crate) fn plays_mut(&mut self, bar: usize) -> &mut Plays {
        &mut self.plays[bar]
    }

    /// The song a test of a mechanism a form would mute plays: every layer
    /// for [`PLAIN_LOOPS`] loops, as every song played before it had a form.
    #[cfg(test)]
    pub(crate) fn plain() -> Self {
        let plays = vec![Plays::ALL; PLAIN_LOOPS * GEN_LOOP_BARS];
        Self::new(End::Cold, PLAIN_LOOPS, Outro::Ends(End::Cold), false, plays)
    }
}

/// The loops a song played before it had a form.
#[cfg(test)]
const PLAIN_LOOPS: usize = 4;

/// Whether `forms` span a form: a proof over songs that mute no layer, or
/// that no edge moves, proves nothing of one.
///
/// # Errors
///
/// If no form of `forms` mutes a layer somewhere, or none fades or filters.
pub fn spans_a_form<'a>(forms: impl IntoIterator<Item = &'a Form>) -> Result<(), String> {
    let (mut mutes, mut edged, mut seen) = (false, false, Vec::new());
    for form in forms {
        mutes |= form.mutes_any();
        edged |= form.edged();
        seen.push(form.to_string());
    }
    match (mutes, edged) {
        (true, true) => Ok(()),
        (false, _) => Err(format!("no song mutes a layer: {seen:?}")),
        (true, false) => Err(format!("no song fades or filters: {seen:?}")),
    }
}

/// Each part's bar count, then each layer out in it as the bars it is out:
/// `intro 4: drums 0-3 · body 40: bass 20-27 · outro 4 fade`.
impl fmt::Display for Form {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [intro, body, outro] = self.parts();
        let parts = [
            ("intro", intro, self.entry()),
            ("body", body, None),
            ("outro", outro, self.exit()),
        ];
        for (i, (name, bars, edge)) in parts.into_iter().enumerate() {
            if i > 0 {
                write!(f, " · ")?;
            }
            write!(f, "{name} {}", bars.len())?;
            match edge {
                None => {}
                Some(Edge::Fade) => write!(f, " fade")?,
                Some(Edge::Filter) => write!(f, " filter")?,
            }
            let mut sep = ":";
            for layer in Layer::ALL {
                let out: Vec<usize> = bars
                    .clone()
                    .filter(|&b| !self.plays[b].has(layer))
                    .collect();
                for run in out.chunk_by(|a, b| a + 1 == *b) {
                    let (first, last) = (run[0], run[run.len() - 1]);
                    if first == last {
                        write!(f, "{sep} {} {first}", layer.name())?;
                    } else {
                        write!(f, "{sep} {} {first}-{last}", layer.name())?;
                    }
                    sep = ",";
                }
            }
        }
        if let Some(dip) = self.dip() {
            write!(f, " · dip {dip}")?;
        }
        Ok(())
    }
}

/// How a song opens.
#[derive(Clone, Copy)]
enum Opening {
    /// Some layers sit out the intro.
    Sparse(Plays),
    /// The band plays the intro through the edge.
    Edged(Edge),
    /// No intro: the body's first bar is the song's.
    Cold,
}

/// How a song closes.
#[derive(Clone, Copy)]
enum Closing {
    /// Some layers sit out the outro.
    Sparse(Plays),
    /// [`Outro::RingOut`].
    RingOut,
    /// The band plays the outro through the edge.
    Edged(Edge),
    /// No outro: the body's last bar is the song's.
    Cold,
}

/// The intros and their weights: LOFI-BIBLE R11's start counts, the sparse
/// ones split by the lanes out and its unsplit ones left out.
const INTRO_KINDS: [(Opening, u32); 8] = [
    (Opening::Sparse(Plays::BED.with(Layer::Keys, true)), 27),
    (
        Opening::Sparse(Plays::BED.with(Layer::Keys, true).with(Layer::Bass, true)),
        13,
    ),
    (
        Opening::Sparse(Plays::BED.with(Layer::Keys, true).with(Layer::Drums, true)),
        10,
    ),
    (Opening::Sparse(Plays::BED), 7),
    (
        Opening::Sparse(Plays::BED.with(Layer::Drums, true).with(Layer::Bass, true)),
        5,
    ),
    (Opening::Edged(Edge::Fade), 13),
    (Opening::Edged(Edge::Filter), 10),
    (Opening::Cold, 5),
];

/// The outros and their weights: R11's end counts, the sparse ones split by
/// the lanes out, its noise-only ones joining the ring-out and its unsplit
/// ones left out.
const OUTRO_KINDS: [(Closing, u32); 7] = [
    (Closing::Sparse(Plays::BED.with(Layer::Keys, true)), 14),
    (
        Closing::Sparse(Plays::BED.with(Layer::Keys, true).with(Layer::Drums, true)),
        11,
    ),
    (
        Closing::Sparse(Plays::BED.with(Layer::Keys, true).with(Layer::Bass, true)),
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
    /// Nothing drops out.
    Steady,
}

const THINNINGS: [(Thinning, u32); 3] = [
    (Thinning::Break, 3),
    (Thinning::BassDrop, 5),
    (Thinning::Steady, 2),
];

/// Short and long intros' bars, and the long one's share: R11's count of
/// eight-bar intros against four-bar ones.
const INTRO_BARS: [NonZeroUsize; 2] = [bars(4), bars(8)];
const LONG_INTRO_P: f32 = 0.65;

/// The body's phrases, each a loop: drawn at even odds (opinion), which puts
/// the median song between the two corpora's.
const PHRASES: [usize; 3] = [4, 5, 6];

/// A sparse or edged outro's bars, short or long, the long one's share for
/// each, and the ring-out's bars: fitted to R11's outro lengths (opinion).
const OUTRO_BARS: [NonZeroUsize; 2] = [bars(4), bars(8)];
const LONG_SPARSE_OUTRO_P: f32 = 0.4;
const LONG_EDGED_OUTRO_P: f32 = 0.5;
const RING_OUT_BARS: usize = RING_OUT.len();

/// A ring-out's bars: the pad's chord, then the room alone.
const RING_OUT: [Plays; 2] = [Plays::BED, Plays::BED.with(Layer::Pad, false)];

/// The thinning's span: the phrase's last half, or with this chance all of it.
const WHOLE_THINNING_P: f32 = 0.4;
/// The bass leaves with the beat in a break, as R11's drum drop-outs take it.
const LOW_WITH_BREAK_P: f32 = 0.5;

/// Whether a song cuts its beat at all, then each phrase from
/// `FIRST_CUT_PHRASE` on but the thinning's cuts its last bars with `CUT_P`,
/// so the cuts cluster as R11's drops per track do; the first core plays
/// through, as R11's first drop sits near the track's middle. Fitted to R11
/// (opinion).
const CUTTING_P: f32 = 0.5;
const CUT_P: f32 = 0.75;
const FIRST_CUT_PHRASE: usize = 1;

/// A cut's bars and their weights, fitted to R11's drop lengths with a break
/// making the rest, and the chance the bass leaves with it, fitted to R11's
/// bass drops that keep the beat (opinion).
const CUT_BARS: [(usize, u32); 3] = [(1, 55), (2, 20), (3, 25)];
const LOW_WITH_CUT_P: f32 = 0.3;

/// The bass sits out the last half of the phrase after the thinning with this
/// chance, in bodies of at least `BASS_DROP_MIN_PHRASES`.
const BASS_DROP_P: f32 = 0.2;
const BASS_DROP_MIN_PHRASES: usize = 5;

/// A song low-passes the last bar of phrase `DIP_PHRASE` with this chance:
/// R11's mid-track filter moments.
const DIP_P: f32 = 0.2;
const DIP_PHRASE: usize = 1;

/// The cut draws every form makes, one per phrase a body may have: as many
/// whatever the body, so a longer body in `PHRASES` moves no later draw.
const CUT_SLOTS: usize = 6;
const _: () = assert!(PHRASES[PHRASES.len() - 1] <= CUT_SLOTS);

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
    let cutting = chance(&mut rng, CUTTING_P);
    let cuts: [Option<(usize, bool)>; CUT_SLOTS] = std::array::from_fn(|_| {
        let cut = chance(&mut rng, CUT_P);
        let bars = kind(rng.unit(), &CUT_BARS);
        let low = chance(&mut rng, LOW_WITH_CUT_P);
        (cutting && cut).then_some((bars, low))
    });
    let bass_drop = chance(&mut rng, BASS_DROP_P);
    let dip = chance(&mut rng, DIP_P);
    let closing = kind(rng.unit(), &OUTRO_KINDS);
    let long_outro = rng.unit();

    let intro_bars = INTRO_BARS[usize::from(long_intro)];
    let (intro_plays, intro) = match opening {
        Opening::Sparse(plays) => (plays, End::Plain(intro_bars)),
        Opening::Edged(edge) => (Plays::ALL, End::Edged(intro_bars, edge)),
        Opening::Cold => (Plays::ALL, End::Cold),
    };
    let mut plays = vec![intro_plays; intro.bars()];

    let mut body = vec![Plays::ALL; phrases * GEN_LOOP_BARS];
    let thin = phrases / 2;
    let phrase = |p: usize| p * GEN_LOOP_BARS..(p + 1) * GEN_LOOP_BARS;
    let half = |p: usize| p * GEN_LOOP_BARS + GEN_LOOP_BARS / 2..(p + 1) * GEN_LOOP_BARS;
    let span = if whole { phrase(thin) } else { half(thin) };
    let keeps_low = match thinning {
        Thinning::Break => {
            for bar in &mut body[span] {
                bar.set(Layer::Drums, false);
                bar.set(Layer::Bass, bar.has(Layer::Bass) && !low_with_break);
            }
            !low_with_break
        }
        Thinning::BassDrop => {
            for bar in &mut body[span] {
                bar.set(Layer::Bass, false);
            }
            false
        }
        Thinning::Steady => true,
    };
    // a phrase cuts only into a phrase that opens on the beat, and never
    // into the outro, where a cut would just start a beatless outro early
    let opens_on_beat = |p: usize| p != thin || !(thinning == Thinning::Break && whole);
    let mut dip_phrase_cut = false;
    for (p, cut) in cuts.iter().enumerate().take(phrases - 1) {
        if let Some((len, low)) = cut
            && p >= FIRST_CUT_PHRASE
            && p != thin
            && opens_on_beat(p + 1)
        {
            let end = (p + 1) * GEN_LOOP_BARS;
            for bar in &mut body[end - len..end] {
                bar.set(Layer::Drums, false);
                bar.set(Layer::Bass, bar.has(Layer::Bass) && !low);
            }
            dip_phrase_cut |= p == DIP_PHRASE;
        }
    }
    if bass_drop && phrases >= BASS_DROP_MIN_PHRASES && keeps_low {
        for bar in &mut body[half(thin + 1)] {
            bar.set(Layer::Bass, false);
        }
    }
    plays.extend(body);

    let outro = match closing {
        Closing::Sparse(p) => {
            let bars = OUTRO_BARS[usize::from(long_outro < LONG_SPARSE_OUTRO_P)];
            plays.extend(std::iter::repeat_n(p, bars.get()));
            Outro::Ends(End::Plain(bars))
        }
        Closing::RingOut => {
            plays.extend(RING_OUT);
            Outro::RingOut
        }
        Closing::Edged(edge) => {
            let bars = OUTRO_BARS[usize::from(long_outro < LONG_EDGED_OUTRO_P)];
            plays.extend(std::iter::repeat_n(Plays::ALL, bars.get()));
            Outro::Ends(End::Edged(bars, edge))
        }
        Closing::Cold => Outro::Ends(End::Cold),
    };
    Form::new(intro, phrases, outro, dip && !dip_phrase_cut, plays)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::{Mood, band, compose, fnv1a};

    /// The forms every test here sweeps: [`arrange`] reads no mood, so these
    /// are the population whatever the tempo.
    const FORMS: u64 = 512;

    fn forms() -> impl Iterator<Item = (u64, Form)> {
        (0..FORMS).map(|seed| (seed, arrange(seed)))
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
    fn every_lane_and_layer_sits_at_its_index() {
        // `name` reads a lane's or a layer's name, and `Plays` a layer, at its
        // discriminant, which must be its place
        for (i, lane) in Lane::ALL.into_iter().enumerate() {
            assert_eq!(lane as usize, i, "{lane:?}");
            assert_eq!(lane.name(), Lane::NAMES[i]);
        }
        for (i, layer) in Layer::ALL.into_iter().enumerate() {
            assert_eq!(layer as usize, i, "{layer:?}");
        }
    }

    #[test]
    fn every_form_is_well_formed() {
        for (seed, form) in forms() {
            let at = |what: &str| format!("seed {seed}: {what}: {form}");
            assert_eq!(
                (form.lead_in() + form.intro()) % GEN_LOOP_BARS,
                0,
                "{}",
                at("body on loop bar 0")
            );
            assert!([0, 4, 8].contains(&form.intro()), "{}", at("intro bars"));
            assert!([0, 2, 4, 8].contains(&form.outro()), "{}", at("outro bars"));
            let [intro, body, outro] = form.parts();
            let (ring_out, last) = (form.rings_out(), form.bars() - 1);
            for bar in 0..form.bars() {
                assert!(
                    form.plays(bar).has(Layer::Pad) || (ring_out && bar == last),
                    "{}",
                    at(&format!("pad out at {bar}"))
                );
            }
            // a row changes on a 4-bar line, but for a cut or a ring-out
            let cuts: Vec<(usize, usize)> = runs(&form, body.clone(), |p| !p.has(Layer::Drums))
                .into_iter()
                .filter(|&(_, len)| len <= CUT_BARS[CUT_BARS.len() - 1].0)
                .collect();
            for bar in 1..form.bars() {
                if ring_out && bar == last {
                    continue;
                }
                let (mut a, b) = (*form.plays(bar - 1), *form.plays(bar));
                // a cut may take the bass with the beat
                if cuts.iter().any(|&(first, _)| first == bar) {
                    a.set(Layer::Drums, b.has(Layer::Drums));
                    a.set(Layer::Bass, b.has(Layer::Bass));
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
            let last_phrase = body.end - GEN_LOOP_BARS..body.end;
            for &(first, len) in &cuts {
                assert!(
                    !last_phrase.contains(&first),
                    "{}",
                    at("a cut before the outro")
                );
                let before = form.plays(first - 1).has(Layer::Drums);
                let after = form.plays(first + len).has(Layer::Drums);
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
            for (edge, part) in [(form.entry(), intro), (form.exit(), outro)] {
                if edge.is_some() {
                    for bar in part {
                        assert_eq!(
                            *form.plays(bar),
                            Plays::ALL,
                            "{}",
                            at(&format!("an edge's band at {bar}"))
                        );
                    }
                }
            }
            if let Some(dip) = form.dip() {
                assert!(
                    form.plays(dip).has(Layer::Drums),
                    "{}",
                    at("a dip on a cut")
                );
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

    /// The value a share `p` of `xs` sits at or under.
    fn percentile(mut xs: Vec<f32>, p: f32) -> f32 {
        xs.sort_by(f32::total_cmp);
        xs[((xs.len() as f32 * p) as usize).min(xs.len() - 1)]
    }

    fn median(xs: Vec<f32>) -> f32 {
        percentile(xs, 0.5)
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
        let (mut counts, mut lengths, mut long, mut alone, mut with) =
            (Vec::new(), Vec::new(), 0, 0, 0);
        for (seed, form) in forms() {
            for mood in [Mood::Day, Mood::Night] {
                let bar_s = compose(mood, seed).bar_s();
                songs.push(form.bars() as f32 * bar_s);
                outros.push(form.outro() as f32 * bar_s);
            }
            intros += usize::from(form.intro() > 0);
            eight += usize::from(form.intro() == 8);
            let drops = runs(&form, form.parts()[1].clone(), |p| !p.has(Layer::Drums));
            drum += usize::from(!drops.is_empty());
            counts.push(drops.len() as f32);
            lengths.extend(drops.iter().map(|&(_, len)| len));
            long += usize::from(drops.iter().any(|&(_, len)| len >= 4));
            let bass_drops = runs(&form, form.parts()[1].clone(), |p| !p.has(Layer::Bass));
            bass += usize::from(!bass_drops.is_empty());
            for (at, len) in bass_drops {
                let beat_kept = (at..at + len).all(|bar| form.plays(bar).has(Layer::Drums));
                (alone, with) = (
                    alone + usize::from(beat_kept),
                    with + usize::from(!beat_kept),
                );
            }
            if let Some(&(bar, _)) = drops.first() {
                first.push(bar as f32 / form.bars() as f32);
            }
        }
        let bars = |n: usize| {
            share(
                lengths.iter().filter(|&&len| len == n).count(),
                lengths.len(),
            )
        };
        let n = FORMS as usize;
        let at_least = |k: f32| share(counts.iter().filter(|&&c| c >= k).count(), n);
        let checks = [
            ("median song s", median(songs), 147.0, 212.0),
            ("intro share", share(intros, n), 0.9, 1.0),
            ("8-bar share of intros", share(eight, intros), 0.55, 0.75),
            ("median outro s", median(outros.clone()), 7.0, 28.0),
            ("drum-drop share", share(drum, n), 0.5, 0.7),
            ("median first-drop position", median(first), 0.38, 0.56),
            ("bass-drop share", share(bass, n), 0.64, 0.84),
            // shares, not order statistics, which sit on their bin edges
            ("share with 2+ drum drops", at_least(2.0), 0.18, 0.36),
            ("share with 3+ drum drops", at_least(3.0), 0.06, 0.18),
            ("1-bar share of drum drops", bars(1), 0.3, 0.5),
            ("3-bar share of drum drops", bars(3), 0.08, 0.25),
            ("share with a 4+ bar drum drop", share(long, n), 0.15, 0.35),
            (
                "bass drops that keep the beat",
                share(alone, alone + with),
                0.45,
                0.7,
            ),
            ("outro s, P75", percentile(outros.clone(), 0.75), 20.0, 42.0),
        ];
        for (name, value, lo, hi) in checks {
            assert!(
                (lo..=hi).contains(&value),
                "{name} {value} outside [{lo}, {hi}]"
            );
        }
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
                let bits = Lane::ALL.map(|lane| p.strikes(lane));
                words.push(bits.iter().fold(0, |w, &b| w << 1 | u64::from(b)));
            }
            // an edge's word as it was while a plain end had one
            let edge = |edge: Option<Edge>| edge.map_or(0, |edge| edge as u64 + 1);
            words.extend([edge(form.entry()), edge(form.exit())]);
            words.push(form.dip().map_or(u64::MAX, |d| d as u64));
        }
        let h = fnv1a(words);
        assert_eq!(h, FORM_GOLDEN, "{h:#018x}");
    }

    const FORM_GOLDEN: u64 = 0x3dec_540e_1881_cac9;

    #[test]
    fn a_form_prints_its_parts() {
        assert_eq!(Form::plain().to_string(), "intro 0 · body 32 · outro 0");
        let mut cut = Form::plain();
        cut.plays[7].set(Layer::Drums, false);
        assert_eq!(cut.to_string(), "intro 0 · body 32: drums 7 · outro 0");
    }
}
