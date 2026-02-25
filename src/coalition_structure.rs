use super::*;

#[derive(Clone, Debug)]
pub struct CoalitionStructure<'a> {
    pub game: &'a HedonicGame,
    pub cs: Vec<usize>,
    sizes: Vec<usize>,
}

pub struct Utility {
    ut: Weight,
    size: usize,
    is_fractional: bool,
}

impl Utility {
    pub fn to_float(&self) -> f64 {
        if self.is_fractional {
            if self.size == 0 {
                0.0
            } else {
                self.ut as f64 / self.size as f64
            }
        } else {
            self.ut as f64
        }
    }
}

impl<'a> CoalitionStructure<'a> {
    pub fn new(game: &'a HedonicGame, cs: Vec<usize>) -> Self {
        assert!(
            cs.len() == game.agent_count(),
            "The coalition structure should have the same size as the number of agents."
        );
        let max = *cs.iter().max().unwrap_or(&0);
        let mut seen = vec![false; max + 1];
        for &c in &cs {
            if c > max {
                continue;
            }
            seen[c] = true;
        }
        assert!(
            seen.iter().all(|&v| v),
            "The coalition structure should contain all integers from `0` to `max(cs)`."
        );
        let mut sizes = vec![0; cs.len()];
        for &c in &cs {
            sizes[c] += 1;
        }
        CoalitionStructure { game, cs, sizes }
    }

    pub fn size(&self) -> usize {
        self.cs.iter().max().unwrap_or(&0) + 1
    }

    pub fn coalition_size(&self, co: Coalition) -> usize {
        assert!(co < self.size(), "Coalition number out of range.");
        self.sizes[co]
    }

    pub fn agent_coalition(&self, ag: Agent) -> Coalition {
        assert!(ag < self.cs.len(), "Agent number out of range.");
        self.cs[ag]
    }

    pub fn coalition_agent_utility(&self, co: Coalition, ag: Agent) -> Utility {
        let mut ut = 0;
        let mut size = 0;
        for j in 0..self.game.agent_count() {
            if self.cs[j] == co {
                ut += self.game[(ag, j)];
                size += 1;
            }
        }
        Utility {
            ut,
            size,
            is_fractional: self.game.is_fractional()
        }
    }

    pub fn coalition_social_welfare(&self, co: Coalition) -> Utility {
        let mut ut = 0;
        let mut size = 0;
        for i in 0..self.game.agent_count() {
            if self.cs[i] == co {
                size += 1;
                for j in 0..self.game.agent_count() {
                    if self.cs[j] == co {
                        ut += self.game[(i, j)];
                    }
                }
            }
        }
        Utility {
            ut,
            size,
            is_fractional: self.game.is_fractional()
        }
    }

    pub fn agent_utility(&self, ag: Agent) -> Utility {
        let co = self.cs[ag];
        self.coalition_agent_utility(co, ag)
    }

    pub fn social_welfare(&self) -> f64 {
        (0..self.size())
            .map(|co| self.coalition_social_welfare(co).to_float())
            .sum()
    }

    pub fn has_improving_deviation(&self) -> bool {
        !self.improving_deviations().is_empty()
    }

    pub fn is_improving_deviation(&self, ag: Agent, co_new: Coalition) -> bool {
        assert!(ag < self.cs.len(), "Agent number out of range.");
        assert!(co_new <= self.size(), "Coalition number out of range.");
        let co_old = self.cs[ag];
        if co_old == co_new {
            return false;
        }
        if let Some(k) = self.game.k {
            if co_new < self.sizes.len() && self.sizes[co_new] == k {
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

    pub fn agent_improving_deviations(&self, ag: Agent) -> Vec<Coalition> {
        assert!(ag < self.cs.len(), "Agent number out of range.");
        let mut res = Vec::new();
        for co_new in 0..=self.size() {
            if self.is_improving_deviation(ag, co_new) {
                res.push(co_new);
            }
        }
        res
    }

    pub fn improving_deviations(&self) -> Vec<(Agent, Coalition)> {
        let mut res = Vec::new();
        for ag in self.game.agents() {
            for co_new in self.agent_improving_deviations(ag) {
                res.push((ag, co_new));
            }
        }
        res
    }

    pub fn move_to(&self, ag: Agent, co_new: Coalition) -> CoalitionStructure<'a> {
        assert!(ag < self.cs.len(), "Agent number out of range.");
        assert!(co_new <= self.size(), "Coalition number out of range.");
        if let Some(k) = self.game.k {
            if co_new < self.sizes.len() {
                assert!(self.sizes[co_new] < k, "The target coalition size is too large.");
            }
        }
        let co_old = self.cs[ag];
        if co_old == co_new {
            return self.clone();
        }
        if co_old < self.sizes.len() && self.sizes[co_old] == 1 && co_new == self.size() {
            return self.clone();
        }
        let mut cs_new = self.cs.clone();
        cs_new[ag] = co_new;
        Self::normalize_cs(&mut cs_new);
        CoalitionStructure::new(self.game, cs_new)
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
        assert!(ag < self.cs.len(), "Agent number out of range.");
        for co_new in 0..=self.size() {
            if co_new == self.cs[ag] {
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
            let co = self.cs[ag];
            res[co].push(ag);
        }
        res
    }
}

impl<'a> PartialEq for CoalitionStructure<'a> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.game, other.game) && self.cs == other.cs
    }
}
