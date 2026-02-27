use std::ops::{Index, IndexMut};

use grid::*;

use super::*;
use super::graph_enumerator::*;

/// Supported graph directionality models.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GraphType {
    /// Arc weights can differ by direction.
    Directed,
    /// Edge weights are mirrored across the diagonal.
    Undirected,
}

/// Weighted graph backed by a square adjacency matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    /// Adjacency/weight matrix.
    pub weights: Grid<Weight>,
    /// Whether the graph is directed or undirected.
    pub graph_type: GraphType,
}

impl Graph {
    /// Returns `true` when the matrix is square and symmetric.
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

    /// Builds a graph from an adjacency matrix and explicit graph type.
    pub fn new(weights: Grid<Weight>, graph_type: GraphType) -> Self {
        debug_assert!(
            weights.rows() == weights.cols(),
            "The weight matrix must be square."
        );
        debug_assert!(
            graph_type == GraphType::Directed || Self::is_symmetric_grid(&weights),
            "The graph is undirected, but the weights matrix is not symmetric."
        );
        Graph {
            weights,
            graph_type,
        }
    }

    /// Builds a graph and infers its type from matrix symmetry.
    pub fn from_grid(weights: Grid<Weight>) -> Self {
        let graph_type = if Self::is_symmetric_grid(&weights) {
            GraphType::Undirected
        } else {
            GraphType::Directed
        };
        Self::new(weights, graph_type)
    }

    /// Builds a graph from an edge list.
    ///
    /// For undirected graphs, each edge is mirrored.
    pub fn from_edges(
        node_count: usize,
        edges: &[(usize, usize, Weight)],
        graph_type: GraphType,
    ) -> Self {
        let mut weights = Grid::<Weight>::new(node_count, node_count);
        for &(i, j, w) in edges {
            debug_assert!(i < node_count && j < node_count, "Node number out of range");
            weights[(i, j)] = w;
            if graph_type == GraphType::Undirected {
                debug_assert!(
                    weights[(j, i)] == 0 || weights[(j, i)] == w,
                    "Graph is undirected but weights are not symmetric"
                );
                weights[(j, i)] = w;
            }
        }
        Self::new(weights, graph_type)
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.weights.rows()
    }

    /// Number of non-zero edges (counted once in undirected graphs).
    pub fn edge_count(&self) -> usize {
        let e = self.weights.iter().filter(|&&v| v > 0).count();
        if self.graph_type == GraphType::Directed { e } else { e / 2 }
    }

    /// Iterator over node identifiers.
    pub fn nodes(&self) -> impl Iterator<Item = usize> {
        0..self.weights.rows()
    }

    /// Iterator over non-zero edges as `(source, target, weight)`.
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, Weight)> {
        self.weights
            .indexed_iter()
            .filter(|&((i, j), &v)| v > 0 && (self.graph_type == GraphType::Directed || j > i))
            .map(|((i, j), &v)| (i, j, v))
            .into_iter()
    }

    /// Returns `true` if all edge weights are either `0` or `1`.
    pub fn is_simple(&self) -> bool {
        self.weights.iter().all(|&v| v <= 1)
    }

    /// Returns `true` if matrix values are symmetric.
    pub fn is_symmetric(&self) -> bool {
        self.graph_type == GraphType::Undirected || Self::is_symmetric_grid(&self.weights)
    }

    /// Returns `true` if graph type is directed.
    pub fn is_directed(&self) -> bool {
        self.graph_type == GraphType::Directed
    }

    /// Enumerates graphs with weights in `[0, m_end]` and at least one edge of weight `m_begin..=m_end`.
    pub fn enumerate(
        node_count: usize,
        graph_type: GraphType,
        m_begin: Weight,
        m_end: Weight,
    ) -> GraphEnumerator {
        GraphEnumerator::new(node_count, graph_type, m_begin, m_end, 0)
    }

    /// Counts the number of graphs produced by [`Self::enumerate`].
    pub fn count(
        node_count: usize,
        graph_type: GraphType,
        m_begin: Weight,
        m_end: Weight,
    ) -> usize {
        let mut graph = Graph::new(Grid::<Weight>::new(node_count, node_count), graph_type);
        let mut state = GraphEnumeratorState::new(&graph, m_begin, m_end, 0);
        let mut count = 0;
        while state.next_graph(&mut graph) {
            count += 1;
        }
        count
    }
}

impl Index<(usize, usize)> for Graph {
    type Output = Weight;

    /// Returns the weight at `(row, col)`.
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.weights[index]
    }
}

impl IndexMut<(usize, usize)> for Graph {
    /// Returns a mutable reference to the weight at `(row, col)`.
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.weights[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use GraphType::*;
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
        assert_eq!(
            Graph::enumerate(2, Undirected, 0, 2).collect::<Vec<_>>(),
            res
        );
        assert_eq!(
            Graph::enumerate(2, Undirected, 2, 2).collect::<Vec<_>>(),
            res[2..]
        );
    }

    #[test]
    fn test_count_graphs() {
        assert_eq!(Graph::count(4, Undirected, 0, 2), 72);
        assert_eq!(Graph::count(4, Undirected, 2, 2), 61);
    }
}
