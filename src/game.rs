use std::ops::{Index, IndexMut};
use std::sync::LazyLock;
use std::usize;

use grid::*;

use super::coalition_structure::*;
use super::graph::*;
use super::types::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GameType {
    Additive,
    Fractional,
}

use GameType::*;

#[derive(Debug)]
pub struct HedonicGame {
    pub graph: Graph,
    pub k: Option<usize>,
    pub game_type: GameType,
}

impl HedonicGame {
    pub fn new(graph: Graph, k: Option<usize>, game_type: GameType) -> Self {
        HedonicGame { graph, k, game_type }
    }

    pub fn from_grid(grid: Grid<Weight>, k: Option<usize>, game_type: GameType) -> Self {
        Self::new(Graph::from_grid(grid), k, game_type)
    }

    pub fn valuations(&self) -> &Grid<Weight> {
        &self.graph.weights
    }

    pub fn is_symmetric(&self) -> bool {
        self.graph.is_symmetric()
    }

    pub fn is_directed(&self) -> bool {
        self.graph.is_directed()
    }

    pub fn is_simple(&self) -> bool {
        self.graph.is_simple()
    }

    pub fn is_fractional(&self) -> bool {
        self.game_type == Fractional
    }

    pub fn agent_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn agents(&self) -> impl Iterator<Item = usize> {
        self.graph.nodes()
    }

    pub fn coalition_structure<'a>(&'a self, cs: Vec<usize>) -> CoalitionStructure<'a> {
        CoalitionStructure::from_vec(self, cs)
    }

    pub fn isolated_coalition_structure<'a>(&'a self) -> CoalitionStructure<'a> {
        self.coalition_structure(self.agents().collect())
    }

    pub fn big_coalition_structure<'a>(&'a self) -> Option<CoalitionStructure<'a>> {
        if let Some(k) = self.k {
            if k < self.agent_count() {
                return None;
            }
        }
        let cs = vec![0; self.agent_count()];
        Some(self.coalition_structure(cs))
    }

    pub fn coalition_structures<'a>(&'a self, size: Option<usize>) -> CoalitionStructures<'a> {
        CoalitionStructures::new(self, size)
    }

    pub fn nash_stable_coalition_structures<'a>(&'a self) -> impl Iterator<Item = CoalitionStructure<'a>> {
        self.coalition_structures(None)
            .filter(|cs| !cs.has_improving_deviation())
    }

    pub fn has_nash_stable_coalition_structure(&self) -> bool {
        let mut cit = CoalitionStructures::new(self, None);
        while let Some(cs) = cit.next_lending() {
            if ! cs.has_improving_deviation() {
                return true
            }
        }
        false
    }

    pub fn enumerate(
        agent_count: usize,
        graph_type: GraphType,
        m_begin: Weight,
        m_end: Weight,
        game_type: GameType,
        k: Option<usize>,
    ) -> impl Iterator<Item = HedonicGame> {
        Graph::enumerate(agent_count, graph_type, m_begin, m_end)
            .map(move |g| Self::new(g, k, game_type))
    }

    pub fn enumerate_unstable(
        agent_count: usize,
        graph_type: GraphType,
        m_begin: Weight,
        m_end: Weight,
        game_type: GameType,
        k: Option<usize>,
    ) -> impl Iterator<Item = HedonicGame> {
        Self::enumerate(agent_count, graph_type, m_begin, m_end, game_type, k)
            .filter(|g| !g.has_nash_stable_coalition_structure())
    }

    pub fn count(agent_count: usize, graph_type: GraphType, m_begin: Weight, m_end: Weight) -> usize {
        Graph::count(agent_count, graph_type, m_begin, m_end)
    }

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
        for game in Self::enumerate(agent_count, graph_type, m_begin, m_end, game_type, k) {
            count_total += 1;
            if !game.has_nash_stable_coalition_structure() {
                count_unstable += 1;
                if result.is_none() {
                    result = Some(game);
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

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.graph[index]
    }
}

impl IndexMut<(usize, usize)> for HedonicGame {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.graph[index]
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

    static GRAPH1: LazyLock<Graph> = LazyLock::new(|| Graph::from_grid(grid![[0,0,2][1,0,3][2,0,0]]));
    static GAME1_FRAC: LazyLock<HedonicGame> = LazyLock::new(|| HedonicGame::new(GRAPH1.clone(), None, Fractional));
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

        compare_coalition_structures(&GAME1_FRAC_K2, GAME1_FRAC_K2.coalition_structures(Some(1)), vec![]);

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
        let (count_unstable, count_total, _) =
            HedonicGame::count_unstable(6, GraphType::Undirected, 2, 2, GameType::Fractional, Some(4));
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
        let edges = vec![(0, 1, 9), (0, 2, 9), (0, 3, 4), (1, 2, 1), (1, 3, 7), (2, 3, 7)];
        let graph = Graph::from_edges(4, &edges, GraphType::Undirected);
        assert_eq!(globals::GAME_K3_NOEQUILIBRIUM_PAPER.graph, graph);
    }
}
