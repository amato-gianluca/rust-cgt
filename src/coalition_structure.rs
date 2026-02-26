use super::*;

#[derive(Clone, Debug)]
pub struct CoalitionStructure<'a> {
    pub game: &'a HedonicGame,
    pub cos: Vec<usize>,
    cos_size: Vec<usize>,
}

pub struct Utility {
    ut: Weight,
    size: usize,
    is_fractional: bool,
}

impl Utility {
    pub fn to_float(&self) -> f64 {
        if self.is_fractional {
            if self.size == 0 { 0.0 } else { self.ut as f64 / self.size as f64 }
        } else {
            self.ut as f64
        }
    }
}

impl<'a> CoalitionStructure<'a> {
    pub fn new(game: &'a HedonicGame, cos: Vec<Agent>, cos_size: Vec<Coalition>) -> Self {
        debug_assert_eq!(
            cos.len(),
            game.agent_count(),
            "The coalition structure should have the same size as the number of agents."
        );
        debug_assert_eq!(
            cos_size.len(),
            game.agent_count(),
            "The cos_size field should have the same size as the number of agents."
        );
        debug_assert!(
            {
                let mut seen = vec![false; cos_size.len()];
                for &c in &cos {
                    seen[c] = true;
                }
                seen.iter().all(|&v| v)
            },
            "The coalition structure should contain all integers from `0` to `max(cs)`."
        );
        CoalitionStructure { game, cos, cos_size }
    }

    pub fn from_vec(game: &'a HedonicGame, cos: Vec<Agent>) -> Self {
        let mut size = vec![0; cos.len()];
        for &c in &cos {
            size[c] += 1;
        }
        Self::new(game, cos, size)
    }

    pub fn size(&self) -> usize {
        *self.cos.iter().max().unwrap() + 1
    }

    pub fn agent_count(&self) -> usize {
        self.cos.len()
    }

    pub fn coalition_size(&self, co: Coalition) -> usize {
        debug_assert!(co < self.size(), "Coalition number out of range.");
        self.cos_size[co]
    }

    pub fn agent_coalition(&self, ag: Agent) -> Coalition {
        debug_assert!(ag < self.cos.len(), "Agent number out of range.");
        self.cos[ag]
    }

    pub fn coalition_agent_utility(&self, co: Coalition, ag: Agent) -> Utility {
        let mut ut = 0;
        for j in 0..self.game.agent_count() {
            if self.cos[j] == co {
                ut += self.game[(ag, j)];
            }
        }
        Utility {
            ut,
            size: self.cos_size[co],
            is_fractional: self.game.is_fractional(),
        }
    }

    pub fn coalition_social_welfare(&self, co: Coalition) -> Utility {
        let mut ut = 0;
        for i in 0..self.game.agent_count() {
            if self.cos[i] == co {
                for j in 0..self.game.agent_count() {
                    if self.cos[j] == co {
                        ut += self.game[(i, j)];
                    }
                }
            }
        }
        Utility {
            ut,
            size: self.cos_size[co],
            is_fractional: self.game.is_fractional(),
        }
    }

    pub fn agent_utility(&self, ag: Agent) -> Utility {
        let co = self.cos[ag];
        self.coalition_agent_utility(co, ag)
    }

    pub fn social_welfare(&self) -> f64 {
        (0..self.size())
            .map(|co| self.coalition_social_welfare(co).to_float())
            .sum()
    }

    pub fn has_improving_deviation(&self) -> bool {
        self.improving_deviations().next().is_some()
    }

    pub fn is_improving_deviation(&self, ag: Agent, co_new: Coalition) -> bool {
        debug_assert!(ag < self.cos.len(), "Agent number out of range.");
        debug_assert!(co_new <= self.size(), "Coalition number out of range.");
        let co_old = self.cos[ag];
        if co_old == co_new {
            return false;
        }
        if let Some(k) = self.game.k {
            if co_new < self.cos_size.len() && self.cos_size[co_new] == k {
                return false;
            }
        }
        let Utility {
            ut: ut_old,
            size: size_old,
            is_fractional: _,
        } = self.agent_utility(ag);
        let Utility {
            ut: ut_new,
            size: size_new,
            is_fractional: _,
        } = self.coalition_agent_utility(co_new, ag);
        if !self.game.is_fractional() {
            return ut_new > ut_old;
        }
        if ut_old == 0 && ut_new == 0 {
            return size_new + 1 < size_old;
        }
        ut_new * size_old as Weight > ut_old * (size_new as Weight + 1)
    }

    pub fn agent_improving_deviations(&self, ag: Agent) -> impl Iterator<Item = Coalition> {
        debug_assert!(ag < self.cos.len(), "Agent number out of range.");
        (0..=self.agent_count()).filter(move |&co_new| self.is_improving_deviation(ag, co_new))
    }

    pub fn improving_deviations(&self) -> impl Iterator<Item = (Agent, Coalition)> {
        self.game
            .agents()
            .flat_map(|ag| self.agent_improving_deviations(ag).map(move |co| (ag, co)))
    }

    pub fn move_to(&self, ag: Agent, co_new: Coalition) -> CoalitionStructure<'a> {
        debug_assert!(ag < self.cos.len(), "Agent number out of range.");
        debug_assert!(co_new <= self.size(), "Coalition number out of range.");
        debug_assert!(
            if let Some(k) = self.game.k { k < self.size() } else { true },
            "The target coalition size is too large."
        );
        let co_old = self.cos[ag];
        if co_old == co_new {
            return self.clone();
        }
        if co_old < self.cos_size.len() && self.cos_size[co_old] == 1 && co_new == self.size() {
            return self.clone();
        }
        let mut cs_new = self.cos.clone();
        cs_new[ag] = co_new;
        Self::normalize_cs(&mut cs_new);
        CoalitionStructure::from_vec(self.game, cs_new)
    }

    fn normalize_cs(cs: &mut Vec<usize>) {
        let mut current = 0usize;
        let mut map = vec![usize::MAX; cs.len()];
        for i in 0..cs.len() {
            let c = cs[i];
            let tgt = map[c];
            if tgt == usize::MAX {
                map[c] = current;
                cs[i] = current;
                current += 1;
            } else {
                cs[i] = tgt;
            }
        }
    }

    pub fn is_agent_nash_stable(&self, ag: Agent) -> bool {
        assert!(ag < self.cos.len(), "Agent number out of range.");
        for co_new in 0..=self.size() {
            if co_new == self.cos[ag] {
                continue;
            }
            if self.is_improving_deviation(ag, co_new) {
                return false;
            }
        }
        true
    }

    pub fn is_nash_stable(&self) -> bool {
        self.game.agents().all(|ag| self.is_agent_nash_stable(ag))
    }

    pub fn to_list(&self) -> Vec<Vec<usize>> {
        let mut res = vec![Vec::new(); self.size()];
        for ag in self.game.agents() {
            let co = self.cos[ag];
            res[co].push(ag);
        }
        res
    }
}

impl<'a> PartialEq for CoalitionStructure<'a> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.game, other.game) && self.cos == other.cos
    }
}
