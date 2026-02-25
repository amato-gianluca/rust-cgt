use std::cmp::{max, min};
use std::ops::{Index, IndexMut};

use grid::*;

use super::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GraphType {
    Directed,
    Undirected,
}

use GraphType::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub weights: Grid<Weight>,
    pub graph_type: GraphType,
}

impl Graph {
    fn is_symmetric_grid(grid: &Grid<Weight>) -> bool {
        let (rows, cols) = grid.size();
        if rows != cols {
            return false;
        }
        for i in 0..rows {
            for j in 0..cols {
                if grid[(i, j)] != grid[(j, i)] {
                    return false;
                }
            }
        }
        true
    }

    pub fn new(weights: Grid<Weight>, graph_type: GraphType) -> Self {
        debug_assert!(weights.rows() == weights.cols(), "The weight matrix must be square.");
        debug_assert!(
            graph_type == Directed || Self::is_symmetric_grid(&weights),
            "The graph is undirected, but the weights matrix is not symmetric."
        );
        Graph { weights, graph_type }
    }

    pub fn from_grid(weights: Grid<Weight>) -> Self {
        let graph_type = if Self::is_symmetric_grid(&weights) { Undirected } else { Directed };
        Self::new(weights, graph_type)
    }

    pub fn from_edges(num_nodes: usize, edges: &[(usize, usize, Weight)], graph_type: GraphType) -> Self {
        let mut weights = Grid::<Weight>::new(num_nodes, num_nodes);
        for &(i, j, w) in edges {
            debug_assert!(i < num_nodes && j < num_nodes, "Node number out of range");
            weights[(i, j)] = w;
            if graph_type == Undirected {
                debug_assert!(
                    weights[(j, i)] == 0 || weights[(j, i)] == w,
                    "Graph is undirected but weights are not symmetric"
                );
                weights[(j, i)] = w;
            }
        }
        Self::new(weights, graph_type)
    }

    pub fn node_count(&self) -> usize {
        self.weights.rows()
    }

    pub fn edge_count(&self) -> usize {
        let e = self.weights.iter().filter(|&&v| v > 0).count();
        if self.graph_type == Directed { e } else { e / 2 }
    }

    pub fn nodes(&self) -> impl Iterator<Item = usize> {
        0..self.weights.rows()
    }

    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, Weight)> {
        self.weights
            .indexed_iter()
            .filter(|&((i, j), &v)| v > 0 && (self.graph_type == Directed || j > i))
            .map(|((i, j), &v)| (i, j, v))
            .into_iter()
    }

    pub fn is_simple(&self) -> bool {
        self.weights.iter().all(|&v| v <= 1)
    }

    pub fn is_symmetric(&self) -> bool {
        self.graph_type == Undirected || Self::is_symmetric_grid(&self.weights)
    }

    pub fn is_directed(&self) -> bool {
        self.graph_type == Directed
    }

    pub fn enumerate(n: usize, is_symmetric: bool, m_begin: Weight, m_end: Weight) -> Enumerate {
        Enumerate::new(n, is_symmetric, m_begin, m_end, 0)
    }

    pub fn count(n: usize, is_symmetric: bool, m_begin: Weight, m_end: Weight) -> usize {
        Enumerate::new(n, is_symmetric, m_begin, m_end, 0).count()
    }
}

impl Index<(usize, usize)> for Graph {
    type Output = Weight;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.weights[index]
    }
}

impl IndexMut<(usize, usize)> for Graph {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.weights[index]
    }
}

#[derive(Debug)]
pub struct Enumerate {
    weights: Grid<Weight>,
    first_unvalid: usize,
    m_max: Weight,
    m_reached: Option<usize>,
    m_end: Weight,
    is_symmetric: bool,
    debug: usize,
}

impl Enumerate {
    fn new(n: usize, is_symmetric: bool, m_begin: Weight, m_end: Weight, debug: usize) -> Self {
        let weights = Grid::init(n, n, 0);
        if debug > 0 {
            println!("sought_reward: {}", m_begin);
            for col in 1..=min(debug, n) {
                println!("{}[{}] v: 0", "  ".repeat(col), col);
            }
        }
        Self {
            weights: weights,
            first_unvalid: n * n - 1,
            m_max: m_begin,
            m_reached: None,
            m_end: m_end,
            is_symmetric: is_symmetric,
            debug: debug,
        }
    }
}

impl Iterator for Enumerate {
    type Item = Graph;

    fn next(&mut self) -> Option<Graph> {
        fn next_pos(row: usize, col: usize, n: usize) -> (usize, usize) {
            if col < n - 1 { (row, col + 1) } else { (row + 1, 0) }
        }
        fn prev_pos(row: usize, col: usize, n: usize) -> (usize, usize) {
            if col > 0 { (row, col - 1) } else { (row - 1, n - 1) }
        }

        let n = self.weights.rows();
        let pos_final = n * n - 1;
        let mut pos = pos_final;
        let mut row = n - 1;
        let mut col = n - 1;
        while self.m_max <= self.m_end {
            loop {
                let bot = if self.is_symmetric && row > col {
                    self.weights[(col, row)]
                } else if row == 0 && col > 0 {
                    self.weights[(row, col - 1)]
                } else if row > 0 && row != col {
                    self.weights[(0, 1)]
                } else {
                    0
                };
                let top = if row == col {
                    0
                } else if self.is_symmetric && row > col {
                    self.weights[(col, row)]
                } else {
                    self.m_max
                };

                let v_new = if pos < self.first_unvalid { max(self.weights[(row, col)] + 1, bot) } else { bot };
                self.first_unvalid = pos + 1;

                if v_new <= top {
                    self.weights[(row, col)] = v_new;

                    let mut is_invalid_graph = false;
                    if row > 0 && col == n - 1 {
                        for i in 0..row {
                            if i + 2 == row {
                                continue;
                            }
                            for j in 0..n {
                                if j == i || j == row {
                                    continue;
                                }
                                if self.weights[(i, j)] == self.weights[(row, j)] {
                                    continue;
                                }
                                if self.weights[(i, j)] > self.weights[(row, j)] {
                                    is_invalid_graph = true;
                                }
                                break;
                            }
                            if is_invalid_graph {
                                break;
                            }
                        }
                    }

                    if !is_invalid_graph {
                        if self.debug > 0 && row == 0 && col > 0 && col <= self.debug {
                            println!("{}[{}] v: {}", "  ".repeat(col), col, v_new);
                        }
                        if v_new == self.m_max && self.m_reached.is_none() {
                            self.m_reached = Some(pos);
                        }
                        if pos == pos_final {
                            if self.m_reached.is_some() {
                                return Some(Graph::new(
                                    self.weights.clone(),
                                    if self.is_symmetric { Undirected } else { Directed },
                                ));
                            }
                        } else {
                            let (nr, nc) = next_pos(row, col, n);
                            row = nr;
                            col = nc;
                            pos += 1;
                        }
                    }
                } else {
                    if self.m_reached == Some(pos) {
                        self.m_reached = None
                    }
                    if row == 0 && col == 0 {
                        break;
                    }
                    let (pr, pc) = prev_pos(row, col, n);
                    row = pr;
                    col = pc;
                    self.first_unvalid = pos;
                    pos -= 1;
                }

                if row >= n {
                    break;
                }
            }

            self.m_max += 1;
            if self.debug > 0 && self.m_max <= self.m_end {
                println!("sought_reward: {}", self.m_max);
            }
            row = 0;
            col = 0;
            pos = 0;
            self.first_unvalid = 0;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;

    static G1: LazyLock<Graph> = LazyLock::new(|| Graph::from_grid(grid![[0,1][1,0]]));

    #[test]
    fn test_constructors() {
        assert_eq!(G1.weights, grid![[0,1][1,0]]);
        assert_eq!(Graph::new(grid![[0,1][1,0]], Undirected), *G1);
        assert_ne!(Graph::new(grid![[0,1][1,0]], Directed), *G1);
        assert_eq!(Graph::from_edges(2, &[(0, 1, 1)], Undirected), *G1);
    }

    #[test]
    fn test_simple_accessors() {
        assert_eq!(G1.graph_type, Undirected);
        assert_eq!(G1.node_count(), 2);
        assert_eq!(G1.edge_count(), 1);
        assert_eq!(G1.nodes().collect::<Vec<_>>(), [0, 1]);
        assert_eq!(G1.edges().collect::<Vec<_>>(), [(0, 1, 1)]);
        assert!(G1.is_simple());
        assert!(G1.is_symmetric());
        assert!(!G1.is_directed());
    }

    #[test]
    fn test_graphs() {
        let res = [
            Graph::from_grid(grid![[0, 0] [0, 0]]),
            Graph::from_grid(grid![[0, 1] [1, 0]]),
            Graph::from_grid(grid![[0, 2] [2, 0]]),
        ];
        assert_eq!(Graph::enumerate(2, true, 0, 2).collect::<Vec<_>>(), res);
        assert_eq!(Graph::enumerate(2, true, 2, 2).collect::<Vec<_>>(), res[2..]);
    }

    #[test]
    fn test_count_games() {
        assert_eq!(Graph::count(4, true, 0, 2), 72);
        assert_eq!(Graph::count(4, true, 2, 2), 61);
    }
}
