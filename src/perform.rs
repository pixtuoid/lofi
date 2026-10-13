//! The public engine: a [`Performance`] streams a seed's songs as [`Frame`]s
//! under the listener's [`Controls`].

/// One stereo frame, `[left, right]`, at [`SAMPLE_RATE`](crate::dsp::SAMPLE_RATE).
pub type Frame = [f32; 2];

/// What the listener's world asks of the music; how it sounds is lofi's call.
/// Out-of-range values clamp; NaN reads as the default.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Controls {
    /// 0 = night … 1 = full day.
    pub daylight: f32,
    /// 0 = an empty room … 1 = a busy one.
    pub intensity: f32,
    /// 0 = dry … 1 = a storm.
    pub rain: f32,
}

impl Default for Controls {
    /// Full day, a moderately busy room, no rain.
    fn default() -> Self {
        Self {
            daylight: 1.0,
            intensity: 0.5,
            rain: 0.0,
        }
    }
}

impl Controls {
    /// These controls with `daylight` replaced.
    #[must_use]
    pub const fn with_daylight(self, daylight: f32) -> Self {
        Self { daylight, ..self }
    }

    /// These controls with `intensity` replaced.
    #[must_use]
    pub const fn with_intensity(self, intensity: f32) -> Self {
        Self { intensity, ..self }
    }

    /// These controls with `rain` replaced.
    #[must_use]
    pub const fn with_rain(self, rain: f32) -> Self {
        Self { rain, ..self }
    }
}

/// An endless run of the songs a seed composes. Its output is a pure function
/// of the seed, the starting controls, and each
/// [`set_controls`](Self::set_controls) call with the
/// [`position`](Self::position) it is made at.
#[derive(Debug)]
pub struct Performance {
    position: u64,
}

// Losing `Send` is silent here and breaks only a consumer that renders off-thread.
const _: fn() = || {
    fn s<T: Send>() {}
    s::<Performance>();
};

impl Performance {
    /// `seed` picks the whole sequence of songs; lofi reads no clock, so the
    /// caller supplies it.
    pub fn new(seed: u64, controls: Controls) -> Self {
        let _ = (seed, controls);
        Self { position: 0 }
    }

    /// Each field applies at its first latch boundary at or after
    /// [`position`](Self::position), and of several calls before one boundary
    /// the last wins. `intensity` and `rain` latch at bar lines; `daylight`
    /// grades at bar lines, and the mood it picks follows at a song's first
    /// bar.
    pub fn set_controls(&mut self, controls: Controls) {
        let _ = controls;
    }

    /// Fills `out` with the next frames, identical however the calls chunk it.
    /// May allocate, so callers render ahead, off the real-time thread.
    pub fn render(&mut self, out: &mut [Frame]) {
        out.fill(Frame::default());
        self.position += out.len() as u64;
    }

    /// Frames rendered so far.
    pub fn position(&self) -> u64 {
        self.position
    }
}
