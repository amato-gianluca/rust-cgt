use std::ops::{Index, IndexMut};
use std::sync::LazyLock;
use std::usize;

use grid::*;

use super::coalition_structure::CoalitionStructures;
use super::graph_enumerator::*;
use super::*;

/// Utility aggregation model for coalition valuation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GameType {
    /// Agent utility is the sum of outgoing weights to coalition members.
    Additive,
    /// Agent utility is averaged by coalition size.
    Fractional,
}

use GameType::*;

/// Hedonic game induced by an interaction graph and optional coalition-size cap.
#[derive(Debug, Clone)]
pub struct HedonicGame {
    /// Underlying weighted graph of agent valuations.
    pub graph: Graph,
    /// Optional upper bound on coalition size.
    pub k: Option<usize>,
    /// Utility model used by agents.
    pub game_type: GameType,
}

impl HedonicGame {
    /// Creates a game from a graph, optional coalition bound, and utility model.
    pub fn new(graph: Graph, k: Option<usize>, game_type: GameType) -> Self {
        HedonicGame {
            graph,
            k,
            game_type,
        }
    }

    /// Creates a game directly from a weight matrix.
    pub fn from_grid(grid: Grid<Weight>, k: Option<usize>, game_type: GameType) -> Self {
        Self::new(Graph::from_grid(grid), k, game_type)
    }

    /// Returns the valuation matrix.
    pub fn valuations(&self) -> &Grid<Weight> {
        &self.graph.weights
    }

    /// Returns whether the valuation matrix is symmetric.
    pub fn is_symmetric(&self) -> bool {
        self.graph.is_symmetric()
    }

    /// Returns whether valuations are directed.
    pub fn is_directed(&self) -> bool {
        self.graph.is_directed()
    }

    /// Returns whether valuations are binary (`0` or `1`).
    pub fn is_simple(&self) -> bool {
        self.graph.is_simple()
    }

    /// Returns whether utilities are fractional.
    pub fn is_fractional(&self) -> bool {
        self.game_type == Fractional
    }

    /// Number of agents in the game.
    pub fn agent_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Iterator over agent identifiers.
    pub fn agents(&self) -> impl Iterator<Item = usize> {
        self.graph.nodes()
    }

    /// Builds a coalition structure from an agent-to-coalition map.
    pub fn coalition_structure<'a>(&'a self, cs: Vec<usize>) -> CoalitionStructure<'a> {
        CoalitionStructure::from_vec(self, cs)
    }

    /// Returns the partition where each agent is isolated.
    pub fn isolated_coalition_structure<'a>(&'a self) -> CoalitionStructure<'a> {
        self.coalition_structure(self.agents().collect())
    }

    /// Returns the grand coalition if it respects the optional size bound.
    pub fn big_coalition_structure<'a>(&'a self) -> Option<CoalitionStructure<'a>> {
        if let Some(k) = self.k {
            if k < self.agent_count() {
                return None;
            }
        }
        let cs = vec![0; self.agent_count()];
        Some(self.coalition_structure(cs))
    }

    /// Enumerates coalition structures, optionally fixing the number of coalitions.
    pub fn coalition_structures<'a>(&'a self, size: Option<usize>) -> CoalitionStructures<'a> {
        CoalitionStructures::new(self, size)
    }

    /// Iterates over Nash-stable coalition structures.
    pub fn nash_stable_coalition_structures<'a>(
        &'a self,
    ) -> impl Iterator<Item = CoalitionStructure<'a>> {
        self.coalition_structures(None)
            .filter(|cs| !cs.has_improving_deviation())
    }

    /// Returns whether at least one Nash-stable coalition structure exists.
    pub fn has_nash_stable_coalition_structure(&self) -> bool {
        let mut cit = CoalitionStructures::new(self, None);
        while let Some(cs) = cit.next_lending() {
            if !cs.has_improving_deviation() {
                return true;
            }
        }
        false
    }

    /// Enumerates all games induced by graphs in the given weight range.
    pub fn enumerate(
        agent_count: usize,
        graph_type: GraphType,
        m_begin: Weight,
        m_end: Weight,
        game_type: GameType,
        k: Option<usize>,
    ) -> impl Iterator<Item = HedonicGame> {
        Enumerate::new(agent_count, graph_type, m_begin, m_end, game_type, k)
    }

    /// Enumerates only unstable games (without Nash-stable coalition structures).
    pub fn enumerate_unstable(
        agent_count: usize,
        graph_type: GraphType,
        m_begin: Weight,
        m_end: Weight,
        game_type: GameType,
        k: Option<usize>,
    ) -> impl Iterator<Item = HedonicGame> {
        EnumerateUnstable::new(agent_count, graph_type, m_begin, m_end, game_type, k)
    }

    /// Counts all games induced by the graph enumeration parameters.
    pub fn count(
        agent_count: usize,
        graph_type: GraphType,
        m_begin: Weight,
        m_end: Weight,
    ) -> usize {
        Graph::count(agent_count, graph_type, m_begin, m_end)
    }

    /// Counts unstable and total games, and optionally returns the first unstable witness.
    pub fn count_unstable(
        agent_count: usize,
        graph_type: GraphType,
        m_begin: Weight,
        m_end: Weight,
        game_type: GameType,
        k: Option<usize>,
    ) -> (usize, usize, Option<Self>) {
        let mut count_total = 0;
        let mut count_unstable = 0;
        let mut result = None;
        let graph = Graph::new(Grid::<Weight>::new(agent_count, agent_count), graph_type);
        let mut game = HedonicGame::new(graph, k, game_type);
        let mut state = GraphEnumeratorState::new(&game.graph, m_begin, m_end, 0);
        while state.next_graph(&mut game.graph) {
            count_total += 1;
            if !game.has_nash_stable_coalition_structure() {
                count_unstable += 1;
                if result.is_none() {
                    result = Some(game.clone());
                }
            }
        }
        (count_unstable, count_total, result)
    }

    // pub fn prices(&self) -> Option<PriceResult> {
    //     let mut poa = f64::NEG_INFINITY;
    //     let mut pos = f64::INFINITY;
    //     let mut cs_worst: Option<CoalitionStructure> = None;
    //     let mut cs_best: Option<CoalitionStructure> = None;
    //     let (_, opt) = self.optimal_coalition_structure();
    //     let mut cs_count = 0usize;
    //     let mut pom = 0.0f64;
    //     for cs in self.nash_stable_coalition_structures() {
    //         cs_count += 1;
    //         let price = opt / cs.social_welfare() as f64;
    //         pom += price;
    //         if price > poa {
    //             poa = price;
    //             cs_worst = Some(cs.clone());
    //         }
    //         if price < pos {
    //             pos = price;
    //             cs_best = Some(cs.clone());
    //         }
    //     }
    //     match (cs_worst, cs_best) {
    //         (Some(worst), Some(best)) => Some(PriceResult {
    //             poa,
    //             pos,
    //             pom: pom / cs_count as f64,
    //             cs_worst: worst,
    //             cs_best: best,
    //             cs_count,
    //         }),
    //         _ => None,
    //     }
}

impl Index<(usize, usize)> for HedonicGame {
    type Output = Weight;

    /// Returns valuation from `i` to `j`.
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.graph[index]
    }
}

impl IndexMut<(usize, usize)> for HedonicGame {
    /// Returns mutable valuation from `i` to `j`.
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.graph[index]
    }
}

/// Iterator over all games matching a set of graph-generation parameters.
pub struct Enumerate {
    game: HedonicGame,
    state: GraphEnumeratorState,
}

impl Enumerate {
    /// Creates a game iterator.
    pub fn new(
        agent_count: usize,
        graph_type: GraphType,
        m_begin: Weight,
        m_end: Weight,
        game_type: GameType,
        k: Option<usize>,
    ) -> Self {
        let graph = Graph::new(Grid::<Weight>::new(agent_count, agent_count), graph_type);
        let game = HedonicGame::new(graph, k, game_type);
        let state = GraphEnumeratorState::new(&game.graph, m_begin, m_end, 0);
        Self { game, state }
    }
}

impl Iterator for Enumerate {
    type Item = HedonicGame;

    /// Returns the next generated game.
    fn next(&mut self) -> Option<Self::Item> {
        if self.state.next_graph(&mut self.game.graph) {
            Some(self.game.clone())
        } else {
            None
        }
    }
}

/// Iterator over only unstable games for a set of graph-generation parameters.
pub struct EnumerateUnstable {
    game: HedonicGame,
    state: GraphEnumeratorState,
}

impl EnumerateUnstable {
    /// Creates an unstable-game iterator.
    pub fn new(
        agent_count: usize,
        graph_type: GraphType,
        m_begin: Weight,
        m_end: Weight,
        game_type: GameType,
        k: Option<usize>,
    ) -> Self {
        let graph = Graph::new(Grid::<Weight>::new(agent_count, agent_count), graph_type);
        let game = HedonicGame::new(graph, k, game_type);
        let state = GraphEnumeratorState::new(&game.graph, m_begin, m_end, 0);
        Self { game, state }
    }
}

impl Iterator for EnumerateUnstable {
    type Item = HedonicGame;

    /// Returns the next generated unstable game.
    fn next(&mut self) -> Option<Self::Item> {
        while self.state.next_graph(&mut self.game.graph) {
            if !self.game.has_nash_stable_coalition_structure() {
                return Some(self.game.clone());
            }
        }
        None
    }
}

#[allow(dead_code)]
mod globals {
    use super::*;

    pub static GAME_K3_NOEQUILIBRIUM_PAPER: LazyLock<HedonicGame> = LazyLock::new(|| {
        HedonicGame::from_grid(
            grid![
                [0, 9, 9, 4]
                [9, 0, 1, 7]
                [9, 1, 0, 7]
                [4, 7, 7, 0]
            ],
            Some(3),
            Fractional,
        )
    });

    pub static GAME_K3_NOEQUILIBRIUM: LazyLock<HedonicGame> = LazyLock::new(|| {
        HedonicGame::from_grid(
            grid![
                [0, 0, 5, 7]
                [0, 0, 5, 7]
                [5, 5, 0, 3]
                [7, 7, 3, 0]
            ],
            Some(3),
            Fractional,
        )
    });

    pub static GAME_K4_NOEQUILIBRIUM: LazyLock<HedonicGame> = LazyLock::new(|| {
        HedonicGame::from_grid(
            grid![
                [0, 0, 0, 5, 10]
                [0, 0, 6, 4, 9]
                [0, 6, 0, 10, 0]
                [5, 4, 10, 0, 10]
                [10, 9, 0, 10, 0]
            ],
            Some(4),
            Fractional,
        )
    });

    pub static GAME_K5_NOEQUILIBRIUM: LazyLock<HedonicGame> = LazyLock::new(|| {
        HedonicGame::from_grid(
            grid![
                [0, 0, 0, 0, 2, 2]
                [0, 0, 0, 2, 0, 2]
                [0, 0, 0, 2, 2, 1]
                [0, 2, 2, 0, 0, 2]
                [2, 0, 2, 0, 0, 2]
                [2, 2, 1, 2, 2, 0]
            ],
            Some(5),
            Fractional,
        )
    });

    pub static GAME_K6_NOEQUILIBRIUM: LazyLock<HedonicGame> = LazyLock::new(|| {
        HedonicGame::from_grid(
            grid![
                [0, 0, 0, 0, 1, 1, 3]
                [0, 0, 1, 3, 0, 1, 2]
                [0, 1, 0, 3, 0, 3, 3]
                [0, 3, 3, 0, 0, 3, 2]
                [1, 0, 0, 0, 0, 3, 1]
                [1, 1, 3, 3, 3, 0, 0]
                [3, 2, 3, 2, 1, 0, 0]
            ],
            Some(6),
            Fractional,
        )
    });

    pub static GAME_K7_NOEQUILIBRIUM: LazyLock<HedonicGame> = LazyLock::new(|| {
        HedonicGame::from_grid(
            grid![
                [0, 0, 0, 0, 0, 0, 1, 2]
                [0, 0, 0, 0, 0, 0, 2, 2]
                [0, 0, 0, 0, 0, 2, 1, 2]
                [0, 0, 0, 0, 1, 2, 1, 0]
                [0, 0, 0, 1, 0, 2, 2, 0]
                [0, 0, 2, 2, 2, 0, 2, 0]
                [1, 2, 1, 1, 2, 2, 0, 2]
                [2, 2, 2, 0, 0, 0, 2, 0]
            ],
            Some(7),
            Fractional,
        )
    });

    pub static GAME_K7_NOEQUILIBRIUM_SIMPLE: LazyLock<HedonicGame> = LazyLock::new(|| {
        HedonicGame::from_grid(
            grid![
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
                [0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
                [0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
                [0, 0, 0, 1, 0, 1, 0, 0, 1, 1]
                [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
                [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
                [0, 1, 1, 0, 0, 0, 1, 0, 1, 1]
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 1]
                [1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
            ],
            Some(7),
            Fractional,
        )
    });

    pub static GAME_K8_NOEQUILIBRIUM: LazyLock<HedonicGame> = LazyLock::new(|| {
        HedonicGame::from_grid(
            grid![
                [0, 0, 0, 0, 0, 0, 0, 1, 2]
                [0, 0, 0, 0, 0, 0, 1, 2, 0]
                [0, 0, 0, 0, 1, 1, 0, 2, 2]
                [0, 0, 0, 0, 1, 1, 1, 1, 0]
                [0, 0, 1, 1, 0, 1, 0, 2, 2]
                [0, 0, 1, 1, 1, 0, 0, 2, 2]
                [0, 1, 0, 1, 0, 0, 0, 2, 0]
                [1, 2, 2, 1, 2, 2, 2, 0, 1]
                [2, 0, 2, 0, 2, 2, 0, 1, 0]
            ],
            Some(8),
            Fractional,
        )
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    static GRAPH1: LazyLock<Graph> =
        LazyLock::new(|| Graph::from_grid(grid![[0,0,2][1,0,3][2,0,0]]));
    static GAME1_FRAC: LazyLock<HedonicGame> =
        LazyLock::new(|| HedonicGame::new(GRAPH1.clone(), None, Fractional));
    static GAME1_FRAC_K1: LazyLock<HedonicGame> =
        LazyLock::new(|| HedonicGame::new(GRAPH1.clone(), Some(1), Fractional));
    static GAME1_FRAC_K2: LazyLock<HedonicGame> =
        LazyLock::new(|| HedonicGame::new(GRAPH1.clone(), Some(2), Fractional));
    static GAME1_FRAC_K3: LazyLock<HedonicGame> =
        LazyLock::new(|| HedonicGame::new(GRAPH1.clone(), Some(3), Fractional));

    fn compare_coalition_structures<'a>(
        game: &'a HedonicGame,
        cs_iter: impl Iterator<Item = CoalitionStructure<'a>>,
        expected: Vec<Vec<usize>>,
    ) {
        let v1 = cs_iter.collect::<Vec<_>>();
        let v2 = expected
            .iter()
            .map(|cs| game.coalition_structure(cs.clone()))
            .collect::<Vec<_>>();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_hedonic_game_coalition_structures_1() {
        compare_coalition_structures(
            &GAME1_FRAC,
            GAME1_FRAC.coalition_structures(Some(1)),
            vec![vec![0, 0, 0]],
        );

        compare_coalition_structures(
            &GAME1_FRAC_K2,
            GAME1_FRAC_K2.coalition_structures(Some(1)),
            vec![],
        );

        compare_coalition_structures(
            &GAME1_FRAC,
            GAME1_FRAC.coalition_structures(Some(2)),
            vec![vec![0, 0, 1], vec![0, 1, 0], vec![0, 1, 1]],
        );

        compare_coalition_structures(
            &GAME1_FRAC_K2,
            GAME1_FRAC_K2.coalition_structures(Some(2)),
            vec![vec![0, 0, 1], vec![0, 1, 0], vec![0, 1, 1]],
        );

        compare_coalition_structures(
            &GAME1_FRAC_K3,
            GAME1_FRAC_K3.coalition_structures(Some(3)),
            vec![vec![0, 1, 2]],
        );

        compare_coalition_structures(
            &GAME1_FRAC_K2,
            GAME1_FRAC_K2.coalition_structures(Some(3)),
            vec![vec![0, 1, 2]],
        );
    }

    #[test]
    fn test_hedonic_game_coalition_structures_2() {
        compare_coalition_structures(
            &GAME1_FRAC,
            GAME1_FRAC.coalition_structures(None),
            vec![
                vec![0, 0, 0],
                vec![0, 0, 1],
                vec![0, 1, 0],
                vec![0, 1, 1],
                vec![0, 1, 2],
            ],
        );

        compare_coalition_structures(
            &GAME1_FRAC_K1,
            GAME1_FRAC_K1.coalition_structures(None),
            vec![vec![0, 1, 2]],
        );

        compare_coalition_structures(
            &GAME1_FRAC_K2,
            GAME1_FRAC_K2.coalition_structures(None),
            vec![vec![0, 0, 1], vec![0, 1, 0], vec![0, 1, 1], vec![0, 1, 2]],
        );

        compare_coalition_structures(
            &GAME1_FRAC_K3,
            GAME1_FRAC_K3.coalition_structures(None),
            vec![
                vec![0, 0, 0],
                vec![0, 0, 1],
                vec![0, 1, 0],
                vec![0, 1, 1],
                vec![0, 1, 2],
            ],
        );
    }

    #[test]
    fn test_hedonic_game_nash_stable_coalition_structures() {
        compare_coalition_structures(
            &GAME1_FRAC,
            GAME1_FRAC.nash_stable_coalition_structures().into_iter(),
            vec![vec![0, 0, 0]],
        );
        compare_coalition_structures(
            &GAME1_FRAC_K1,
            GAME1_FRAC_K1.nash_stable_coalition_structures().into_iter(),
            vec![vec![0, 1, 2]],
        );
    }

    #[test]
    fn test_hedonic_game_has_nash_stable_coalition_structure() {
        use super::globals::*;

        assert!(!GAME_K3_NOEQUILIBRIUM_PAPER.has_nash_stable_coalition_structure());
        assert!(!GAME_K3_NOEQUILIBRIUM.has_nash_stable_coalition_structure());
        assert!(!GAME_K4_NOEQUILIBRIUM.has_nash_stable_coalition_structure());
        assert!(!GAME_K5_NOEQUILIBRIUM.has_nash_stable_coalition_structure());
        assert!(!GAME_K6_NOEQUILIBRIUM.has_nash_stable_coalition_structure());
        assert!(!GAME_K7_NOEQUILIBRIUM.has_nash_stable_coalition_structure());
        assert!(!GAME_K7_NOEQUILIBRIUM_SIMPLE.has_nash_stable_coalition_structure());
        assert!(!GAME_K8_NOEQUILIBRIUM.has_nash_stable_coalition_structure());
    }

    #[test]
    fn test_hedonic_game_no_nash_for_asymmetric_games() {
        for k in [2, 3] {
            let mut edges = Vec::new();
            for i in 0..(k + 1) {
                edges.push((i, (i + 1) % (k + 1), 1));
            }
            let graph = Graph::from_edges(k + 1, &edges, GraphType::Directed);
            let game = HedonicGame::new(graph, Some(k), Fractional);
            assert!(!game.has_nash_stable_coalition_structure());
        }
    }

    #[test]
    fn test_count_games() {
        assert_eq!(HedonicGame::count(4, GraphType::Undirected, 0, 2), 72);
        assert_eq!(HedonicGame::count(4, GraphType::Undirected, 2, 2), 61);
    }

    #[test]
    fn test_count_unstable_games() {
        let (count_unstable, count_total, _) = HedonicGame::count_unstable(
            6,
            GraphType::Undirected,
            2,
            2,
            GameType::Fractional,
            Some(4),
        );
        assert_eq!(count_unstable, 9);
        assert_eq!(count_total, 66515);
        //let weights = vec![0, 1, 4, 7, 9];
        //assert_eq!(HedonicGame::count_unstable(4, graph::GraphType::Undirected, 4, 4, Some(3), GameType::Fractional, Some(&weights), 0), (2, 775));
    }

    // #[test]
    // fn test_hedonic_game_unbound_poa_for_non_simple_games() {
    //     for m in [10, 20] {
    //         let edges = vec![(0, 1, 1), (1, 2, 2 * m), (2, 3, 1)];
    //         let graph = Graph::from_edges(4, &edges, false);
    //         let game = HedonicGame::new(graph, Some(2), true);
    //         let (_cs, opt) = game.optimal_coalition_structure();
    //         let prices = game.prices().expect("prices should exist");
    //         assert!(approx_eq!(f64, opt, (2 * m) as f64));
    //         assert!(approx_eq!(f64, prices.poa, m as f64));
    //     }
    // }

    // #[test]
    // fn test_hedonic_game_prices_for_2ssf() {
    //     let edges = vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)];
    //     let graph = Graph::from_edges(4, &edges, false);
    //     let game = HedonicGame::new(graph, Some(2), true);
    //     let pr = game.prices().expect("prices should exist");
    //     assert_eq!(pr.poa, 2.0);
    //     assert_eq!(pr.pos, 1.0);
    // }

    // #[test]
    // fn test_hedonic_game_optimal_coalition_structure1() {
    //     let game = HedonicGame::new(
    //         grid!vec![
    //             vec![0, 1, 0, 1, 0],
    //             vec![1, 0, 1, 1, 0],
    //             vec![0, 1, 0, 0, 1],
    //             vec![1, 1, 0, 0, 1],
    //             vec![0, 0, 1, 1, 0],
    //         ]),
    //         Some(2),
    //         true,
    //     );
    //     let (_cs, v) = game.optimal_coalition_structure();
    //     assert!(approx_eq!(f64, v, 2.0));
    // }

    // #[test]
    // fn test_hedonic_game_optimal_coalition_structure2() {
    //     let game = HedonicGame::new(
    //         grid!vec![
    //             vec![0, 0, 0, 0, 1, 1],
    //             vec![0, 0, 0, 1, 0, 1],
    //             vec![0, 0, 0, 1, 1, 0],
    //             vec![0, 1, 1, 0, 0, 0],
    //             vec![1, 0, 1, 0, 0, 0],
    //             vec![1, 1, 0, 0, 0, 0],
    //         ]),
    //         Some(2),
    //         true,
    //     );
    //     let (_cs, opt) = game.optimal_coalition_structure();
    //     assert!(approx_eq!(f64, opt, 3.0));
    // }

    #[test]
    fn test_graph_to_from_weights() {
        let edges = vec![
            (0, 1, 9),
            (0, 2, 9),
            (0, 3, 4),
            (1, 2, 1),
            (1, 3, 7),
            (2, 3, 7),
        ];
        let graph = Graph::from_edges(4, &edges, GraphType::Undirected);
        assert_eq!(globals::GAME_K3_NOEQUILIBRIUM_PAPER.graph, graph);
    }
}
