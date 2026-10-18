//! A latched control's ramp: from its value at the latch frame to the new
//! latch's over a fixed length, closed-form in the frame, so no chunking moves
//! it and no state outlives the song.

/// How a [`Ramp`] moves between its ends.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Shape {
    Linear,
    /// Linear in dB, so a level change sounds even across its length.
    Db,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Ramp {
    from: f64,
    to: f64,
    start: usize,
    len: usize,
    shape: Shape,
}

impl Ramp {
    /// `value` from frame 0 on.
    pub(crate) fn held(value: f64, shape: Shape) -> Self {
        Self {
            from: value,
            to: value,
            start: 0,
            len: 0,
            shape,
        }
    }

    /// The value on frame `n`: exactly `to` from the ramp's end on.
    pub(crate) fn at(&self, n: usize) -> f64 {
        if n >= self.end() {
            return self.to;
        }
        if n <= self.start {
            return self.from;
        }
        let x = (n - self.start) as f64 / self.len as f64;
        match self.shape {
            Shape::Linear => self.from * (1.0 - x) + self.to * x,
            Shape::Db => self.from * (self.to / self.from).powf(x),
        }
    }

    /// Ramps from the value on frame `n` to `to` over `len` frames; one
    /// already settled on `to` stays put, so a latch asking for nothing new
    /// starts no ramp.
    pub(crate) fn toward(&mut self, n: usize, to: f64, len: usize) {
        if n >= self.end() && self.to == to {
            return;
        }
        self.from = self.at(n);
        self.to = to;
        self.start = n;
        self.len = len;
    }

    /// Holds the value on frame `n` from then on.
    pub(crate) fn hold(&mut self, n: usize) {
        let held = self.at(n);
        self.toward(n, held, 0);
    }

    /// The frame from which it holds `to`.
    pub(crate) fn end(&self) -> usize {
        self.start + self.len
    }

    pub(crate) fn to(&self) -> f64 {
        self.to
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ramp_starts_at_its_value_and_lands_exactly_on_its_target() {
        for shape in [Shape::Linear, Shape::Db] {
            let mut ramp = Ramp::held(0.35, shape);
            ramp.toward(100, 0.6, 1000);
            assert_eq!(ramp.at(100), 0.35, "{shape:?} at its start");
            assert_eq!(ramp.at(1100), 0.6, "{shape:?} at its end");
            assert_eq!(ramp.at(5000), 0.6, "{shape:?} past its end");
            let mid = ramp.at(600);
            assert!(0.35 < mid && mid < 0.6, "{shape:?} halfway: {mid}");
            // a latch mid-ramp starts from where the ramp is
            ramp.toward(600, 0.1, 50);
            assert_eq!(ramp.at(600), mid, "{shape:?} retargeted");
            assert_eq!(ramp.at(650), 0.1, "{shape:?} retargeted end");
        }
    }

    #[test]
    fn a_db_ramp_moves_evenly_in_db() {
        let mut ramp = Ramp::held(0.1, Shape::Db);
        ramp.toward(0, 1.0, 4);
        let db: Vec<f64> = (0..=4).map(|n| 20.0 * ramp.at(n).log10()).collect();
        for step in db.windows(2) {
            assert!((step[1] - step[0] - 5.0).abs() < 1e-9, "{db:?}");
        }
    }

    #[test]
    fn a_settled_ramp_asked_for_its_own_value_stays_settled() {
        for shape in [Shape::Linear, Shape::Db] {
            let mut ramp = Ramp::held(0.35, shape);
            ramp.toward(100, 0.6, 1000);
            ramp.toward(5000, 0.6, 1000);
            assert_eq!(ramp.end(), 1100, "{shape:?} restarted at its own value");
            // but a ramp in flight, or asked elsewhere, moves
            ramp.toward(6000, 0.2, 10);
            assert_eq!(ramp.end(), 6010, "{shape:?}");
        }
    }

    #[test]
    fn a_held_ramp_holds_the_value_it_had() {
        let mut ramp = Ramp::held(0.2, Shape::Db);
        ramp.toward(0, 0.8, 100);
        let at = ramp.at(40);
        ramp.hold(40);
        assert!([40, 41, 10_000].iter().all(|&n| ramp.at(n) == at));
    }
}
