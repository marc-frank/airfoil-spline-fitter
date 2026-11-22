//! Undo and redo of settings: changes within 500 ms form one step.

use std::time::{Duration, Instant};

const MERGE: Duration = Duration::from_millis(500);
const LIMIT: usize = 200;

struct Pending<T> {
    deadline: Instant,
    quiet: bool,
    latest: T,
}

/// A list of states with a current index and a pending, not yet settled
/// change. Quiet changes (made by the program, not the user) replace the
/// current state instead of adding a step.
pub struct History<T: Clone + PartialEq> {
    states: Vec<T>,
    index: usize,
    pending: Option<Pending<T>>,
}

impl<T: Clone + PartialEq> History<T> {
    pub fn new(initial: T) -> Self {
        History { states: vec![initial], index: 0, pending: None }
    }

    /// Records a change; it settles 500 ms after the last one.
    pub fn on_change(&mut self, state: T, quiet: bool, now: Instant) {
        if let Some(p) = self.pending.take() {
            if p.quiet != quiet {
                self.settle(p.latest, p.quiet);
            }
        }
        self.pending = Some(Pending { deadline: now + MERGE, quiet, latest: state });
    }

    /// Settles a pending change whose time has come; true when it did.
    pub fn tick(&mut self, now: Instant, current: &T) -> bool {
        if self.pending.as_ref().is_some_and(|p| now >= p.deadline) {
            self.commit(current);
            return true;
        }
        false
    }

    /// Time until the pending change settles.
    pub fn time_left(&self, now: Instant) -> Option<Duration> {
        self.pending.as_ref().map(|p| p.deadline.saturating_duration_since(now))
    }

    /// Settles a pending change now with the current state.
    pub fn commit(&mut self, current: &T) {
        if let Some(p) = self.pending.take() {
            self.settle(current.clone(), p.quiet);
        }
    }

    fn settle(&mut self, state: T, quiet: bool) {
        if quiet {
            self.states[self.index] = state;
        } else if state != self.states[self.index] {
            self.states.truncate(self.index + 1);
            self.states.push(state);
            if self.states.len() > LIMIT {
                self.states.remove(0);
            }
            self.index = self.states.len() - 1;
        }
    }

    pub fn can_undo(&self) -> bool {
        self.index > 0 || self.pending.as_ref().is_some_and(|p| !p.quiet)
    }

    pub fn can_redo(&self) -> bool {
        self.index + 1 < self.states.len() && self.pending.is_none()
    }

    /// The state to restore for an undo, if any.
    pub fn undo(&mut self, current: &T) -> Option<T> {
        self.commit(current);
        (self.index > 0).then(|| {
            self.index -= 1;
            self.states[self.index].clone()
        })
    }

    /// The state to restore for a redo, if any.
    pub fn redo(&mut self, current: &T) -> Option<T> {
        self.commit(current);
        (self.index + 1 < self.states.len()).then(|| {
            self.index += 1;
            self.states[self.index].clone()
        })
    }

    /// Stores the values a restore actually set (limits may clamp them).
    pub fn restored(&mut self, actual: T) {
        self.states[self.index] = actual;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_and_undoes() {
        let t0 = Instant::now();
        let mut h = History::new(0);
        h.on_change(1, false, t0);
        h.on_change(2, false, t0 + Duration::from_millis(100));
        assert!(h.can_undo() && !h.can_redo());
        assert!(!h.tick(t0 + Duration::from_millis(300), &2));
        assert!(h.tick(t0 + Duration::from_millis(700), &2));
        assert_eq!(h.undo(&2), Some(0));
        assert!(h.can_redo());
        assert_eq!(h.redo(&0), Some(2));
        assert_eq!(h.redo(&2), None);
    }

    #[test]
    fn quiet_changes_replace_the_current_state() {
        let t0 = Instant::now();
        let mut h = History::new(0);
        h.on_change(5, true, t0);
        h.commit(&5);
        assert!(!h.can_undo());
        h.on_change(6, false, t0);
        h.commit(&6);
        assert_eq!(h.undo(&6), Some(5));
    }
}
