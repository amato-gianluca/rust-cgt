use std::cmp::min;

use super::*;

#[derive(Clone, Debug)]
pub struct CoalitionStructure<'a> {
    pub game: &'a HedonicGame,
    ag_map: Vec<Coalition>,
    co_sizes: Vec<usize>,
    size: usize,
}

pub struct Utility {
    pub ut: Weight,
    pub size: usize,
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
    fn _co_sizes(ag_map: &[Coalition]) -> Vec<usize> {
        let mut co_sizes = vec![0; ag_map.len()];
        for &c in ag_map {
            co_sizes[c] += 1;
        }
        co_sizes
    }

    fn _size(ag_map: &[Coalition]) -> usize {
        match ag_map.iter().max() {
            None => 0,
            Some(co_max) => co_max + 1,
        }
    }

    fn _is_normalized(ag_map: &[Coalition]) -> bool {
        let size = Self::_size(ag_map);
        let mut seen = vec![false; size];
        for &c in ag_map {
            seen[c] = true;
        }
        seen.iter().all(|&v| v)
    }

    fn _normalize_cs(ag_map: &mut Vec<usize>) {
        let mut current = 0usize;
        let mut map = vec![usize::MAX; ag_map.len()];
        for i in 0..ag_map.len() {
            let c = ag_map[i];
            let tgt = map[c];
            if tgt == usize::MAX {
                map[c] = current;
                ag_map[i] = current;
                current += 1;
            } else {
                ag_map[i] = tgt;
            }
        }
    }

    pub fn new(game: &'a HedonicGame, ag_map: Vec<Coalition>, co_sizes: Vec<usize>, size: usize) -> Self {
        debug_assert_eq!(
            ag_map.len(),
            game.agent_count(),
            "The length of ag_map should be equal to the number of agents."
        );
        debug_assert!(
            ag_map.iter().all(|&x| x < game.agent_count()),
            "Coalition numbers should be less than the number of agents."
        );
        debug_assert!(
            Self::_is_normalized(&ag_map),
            "The vector ag_map is normalized, i.e., it contains all integers from `0` to `max(ag_map)`."
        );
        debug_assert_eq!(
            Self::_size(&ag_map),
            size,
            "The size parameter should be equal to the number of coalition."
        );
        debug_assert_eq!(
            Self::_co_sizes(&ag_map),
            co_sizes,
            "The vector co_sizes contains the size of coalitions."
        );
        Self::new_unchecked(game, ag_map, co_sizes, size)
    }

    pub fn new_unchecked(game: &'a HedonicGame, ag_map: Vec<Coalition>, co_sizes: Vec<usize>, size: usize) -> Self {
        CoalitionStructure {
            game,
            ag_map,
            co_sizes,
            size,
        }
    }

    pub fn from_vec(game: &'a HedonicGame, ag_map: Vec<Agent>) -> Self {
        let co_sizes = Self::_co_sizes(&ag_map);
        let size = Self::_size(&ag_map);
        Self::new(game, ag_map, co_sizes, size)
    }

    pub fn agent_count(&self) -> usize {
        self.ag_map.len()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn agents(&self) -> impl Iterator<Item = Agent> {
        self.game.agents()
    }

    pub fn coalitions(&self) -> impl Iterator<Item = Coalition> {
        0..self.size()
    }

    pub fn agent_coalition(&self, ag: Agent) -> Coalition {
        debug_assert!(ag < self.agent_count(), "Agent number out of range.");
        self.ag_map[ag]
    }

    pub fn coalition_size(&self, co: Coalition) -> usize {
        debug_assert!(co <= self.size(), "Coalition number out of range.");
        self.co_sizes[co]
    }

    pub fn coalition_agent_utility(&self, co: Coalition, ag: Agent) -> Utility {
        debug_assert!(ag < self.agent_count(), "Agent number out of range.");
        debug_assert!(co < self.size(), "Coalition number out of range.");
        let mut ut = 0;
        for j in self.agents() {
            if self.agent_coalition(j) == co {
                ut += self.game[(ag, j)];
            }
        }
        Utility {
            ut,
            size: self.coalition_size(co),
            is_fractional: self.game.is_fractional(),
        }
    }

    pub fn agent_utility(&self, ag: Agent) -> Utility {
        self.coalition_agent_utility(self.agent_coalition(ag), ag)
    }

    pub fn coalition_social_welfare(&self, co: Coalition) -> Utility {
        debug_assert!(co < self.size(), "Coalition number out of range.");
        let mut ut = 0;
        for i in self.agents() {
            if self.agent_coalition(i) == co {
                for j in self.agents() {
                    if self.agent_coalition(j) == co {
                        ut += self.game[(i, j)];
                    }
                }
            }
        }
        Utility {
            ut,
            size: self.coalition_size(co),
            is_fractional: self.game.is_fractional(),
        }
    }

    pub fn social_welfare(&self) -> f64 {
        self.coalitions()
            .map(|co| self.coalition_social_welfare(co).to_float())
            .sum()
    }

    pub fn is_improving_deviation(&self, ag: Agent, co_new: Coalition) -> bool {
        debug_assert!(ag < self.agent_count(), "Agent number out of range.");
        debug_assert!(co_new <= self.size(), "Coalition number out of range.");
        let co_old = self.agent_coalition(ag);
        if co_old == co_new {
            return false;
        }
        if let Some(k) = self.game.k
            && self.coalition_size(co_new) == k
        {
            return false;
        }
        let mut ut_old = 0;
        let mut ut_new = 0;
        for (&w, &co) in self.game.graph.weights.iter_row(ag).zip(&self.ag_map) {
            if co == co_old {
                ut_old += w
            };
            if co == co_new {
                ut_new += w
            };
        }
        if !self.game.is_fractional() {
            return ut_new > ut_old;
        }
        if ut_old == 0 && ut_new == 0 {
            return self.coalition_size(co_new) + 1 < self.coalition_size(co_old);
        }
        ut_new * self.coalition_size(co_old) as Weight > ut_old * (self.coalition_size(co_new) as Weight + 1)
    }

    pub fn improving_deviations_for_agent(&self, ag: Agent) -> impl Iterator<Item = Coalition> {
        debug_assert!(ag < self.agent_count(), "Agent number out of range.");
        (0..min(self.size() + 1, self.agent_count())).filter(move |&co_new| self.is_improving_deviation(ag, co_new))
    }

    pub fn improving_deviations(&self) -> impl Iterator<Item = (Agent, Coalition)> {
        self.agents()
            .flat_map(|ag| self.improving_deviations_for_agent(ag).map(move |co| (ag, co)))
    }

    pub fn has_improving_deviation(&self) -> bool {
        self.improving_deviations().next().is_some()
    }

    pub fn move_to(&self, ag: Agent, co_new: Coalition) -> CoalitionStructure<'a> {
        debug_assert!(ag < self.agent_count(), "Agent number out of range.");
        debug_assert!(co_new <= self.size(), "Coalition number out of range.");
        debug_assert!(
            if let Some(k) = self.game.k { k < self.size() } else { true },
            "The target coalition size is too large."
        );
        let co_old = self.agent_coalition(ag);
        if co_old == co_new {
            return self.clone();
        }
        if self.co_sizes[co_old] == 1 && co_new == self.size() {
            return self.clone();
        }
        let mut ag_map_new = self.ag_map.clone();
        ag_map_new[ag] = co_new;
        Self::_normalize_cs(&mut ag_map_new);
        CoalitionStructure::from_vec(self.game, ag_map_new)
    }

    pub fn to_list(&self) -> Vec<Vec<usize>> {
        let mut res = vec![Vec::new(); self.size()];
        for ag in self.agents() {
            let co = self.agent_coalition(ag);
            res[co].push(ag);
        }
        res
    }
}

impl<'a> PartialEq for CoalitionStructure<'a> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.game, other.game) && self.ag_map == other.ag_map
    }
}

pub struct CoalitionStructures<'a> {
    cs: CoalitionStructure<'a>,
    cs_nums: Vec<usize>,
    size: usize,
    first_unvalid: usize,
    fixed_size: bool,
}

impl<'a> CoalitionStructures<'a> {
    pub fn new(game: &'a HedonicGame, size: Option<usize>) -> CoalitionStructures<'a> {
        let cs = CoalitionStructure::new_unchecked(game, vec![0; game.agent_count()], vec![0; game.agent_count()], 1);
        CoalitionStructures {
            cs,
            cs_nums: vec![0; game.agent_count() + 1],
            size: size.unwrap_or(1),
            first_unvalid: 0,
            fixed_size: size.is_some(),
        }
    }

    fn cs_next_fixedsize(&mut self) -> bool {
        let agent_count = self.cs.agent_count();
        let mut ag = std::cmp::min(self.first_unvalid, agent_count - 1);
        loop {
            if ag == agent_count {
                return true;
            }
            let coalitions_potential = self.cs_nums[ag] + (agent_count - ag);
            let bot = if coalitions_potential > self.size { 0 } else { self.cs_nums[ag] };
            let top = if self.cs_nums[ag] < self.size { self.cs_nums[ag] } else { self.cs_nums[ag] - 1 };
            let mut co_new = if ag < self.first_unvalid {
                let co = self.cs.ag_map[ag];
                self.cs.co_sizes[co] -= 1;
                std::cmp::max(co + 1, bot)
            } else {
                self.first_unvalid += 1;
                bot
            };
            while co_new <= top {
                if let Some(k) = self.cs.game.k {
                    if self.cs.co_sizes[co_new] >= k {
                        co_new += 1;
                        continue;
                    }
                }
                break;
            }
            if co_new <= top {
                self.cs.ag_map[ag] = co_new;
                self.cs.co_sizes[co_new] += 1;
                self.cs_nums[ag + 1] = std::cmp::max(self.cs_nums[ag], co_new + 1);
                self.cs.size = self.cs_nums[ag + 1];
                ag += 1;
            } else {
                self.first_unvalid = ag;
                if ag == 0 {
                    return false;
                } else {
                    self.cs.size = self.cs_nums[ag - 1];
                    ag -= 1;
                }
            }
        }
    }

    pub fn cs_next(&mut self) -> bool {
        while self.size <= self.cs.agent_count() {
            if self.cs_next_fixedsize() {
                return true;
            }
            self.size += 1;
            self.cs.ag_map.fill(0);
            self.cs.co_sizes.fill(0);
            self.cs.size = 0;
            self.cs_nums.fill(0);
            self.first_unvalid = 0
        }
        false
    }

    pub fn next_lending(&mut self) -> Option<&CoalitionStructure<'a>> {
        let res = if self.fixed_size { self.cs_next_fixedsize() } else { self.cs_next() };
        if res { Some(&self.cs) } else { None }
    }
}

impl<'a> Iterator for CoalitionStructures<'a> {
    type Item = CoalitionStructure<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_lending() {
            Some(cs) => Some(cs.clone()),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GameType::*;
    use super::*;
    use grid::*;
    use std::sync::LazyLock;

    static GRAPH1: LazyLock<Graph> = LazyLock::new(|| Graph::from_grid(grid![[0,0,2][1,0,3][2,0,0]]));
    static GAME1_FRAC: LazyLock<HedonicGame> = LazyLock::new(|| HedonicGame::new(GRAPH1.clone(), None, Fractional));
    static GAME1_FRAC_CS1: LazyLock<CoalitionStructure> =
        LazyLock::new(|| CoalitionStructure::from_vec(&GAME1_FRAC, vec![0, 1, 0]));
    static GAME1_FRAC_CS2: LazyLock<CoalitionStructure> =
        LazyLock::new(|| CoalitionStructure::from_vec(&GAME1_FRAC, vec![0, 0, 0]));
    static GAME1_NOFRAC: LazyLock<HedonicGame> = LazyLock::new(|| HedonicGame::new(GRAPH1.clone(), None, Additive));
    static GAME1_NOFRAC_CS1: LazyLock<CoalitionStructure> =
        LazyLock::new(|| CoalitionStructure::from_vec(&GAME1_NOFRAC, vec![0, 1, 0]));

    #[test]
    fn test_size() {
        assert_eq!(GAME1_FRAC_CS1.size(), 2);
        assert_eq!(GAME1_FRAC_CS2.size(), 1);
    }

    #[test]
    fn test_coalition_size() {
        assert_eq!(GAME1_FRAC_CS1.coalition_size(0), 2);
        assert_eq!(GAME1_FRAC_CS2.coalition_size(0), 3);
    }

    #[test]
    fn test_agent_utility() {
        assert_eq!(GAME1_FRAC_CS1.agent_utility(0).to_float(), 1.0);
        assert_eq!(GAME1_FRAC_CS1.agent_utility(1).to_float(), 0.0);
        assert_eq!(GAME1_FRAC_CS1.agent_utility(2).to_float(), 1.0);
        assert_eq!(GAME1_NOFRAC_CS1.agent_utility(0).to_float(), 2.0);
        assert_eq!(GAME1_NOFRAC_CS1.agent_utility(1).to_float(), 0.0);
    }

    #[test]
    fn test_coalition_social_welfare() {
        assert_eq!(GAME1_FRAC_CS1.coalition_social_welfare(0).to_float(), 2.0);
        assert_eq!(GAME1_FRAC_CS1.coalition_social_welfare(1).to_float(), 0.0);
    }

    #[test]
    fn test_social_welfare() {
        assert_eq!(GAME1_FRAC_CS1.social_welfare(), 2.0);
        assert_eq!(GAME1_NOFRAC_CS1.social_welfare(), 4.0);
    }

    #[test]
    fn test_is_improving_deviation() {
        assert!(GAME1_FRAC_CS1.is_improving_deviation(1, 0));
        assert!(GAME1_FRAC_CS1.is_improving_deviation(1, 0));
        assert!(!GAME1_FRAC_CS1.is_improving_deviation(0, 0));
    }

    #[test]
    fn test_equality() {
        assert_eq!(*GAME1_FRAC_CS1, *GAME1_FRAC_CS1);
        assert_ne!(*GAME1_FRAC_CS1, *GAME1_NOFRAC_CS1);
    }

    #[test]
    fn test_move_to() {
        let csa = GAME1_FRAC_CS1.move_to(2, 2);
        assert_eq!(csa, CoalitionStructure::from_vec(&GAME1_FRAC, vec![0, 1, 2]));
        let csa = csa.move_to(0, 2);
        assert_eq!(csa, CoalitionStructure::from_vec(&GAME1_FRAC, vec![0, 1, 0]));
        let csa = csa.move_to(2, 1);
        assert_eq!(csa, CoalitionStructure::from_vec(&GAME1_FRAC, vec![0, 1, 1]));
    }
}
