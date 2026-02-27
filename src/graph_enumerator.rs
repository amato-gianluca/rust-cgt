use super::*;
use grid::*;

/// This is a low-level graph enumerator which changes an external graph. We cannot
/// put the graph's reference inside the enumerator, since this would make impossible
/// to define the GraphEnumerator structl later, hence we provide each time the
/// external graph as a parameter of the external mathod.
///
/// Actually, the performance increase we get with all these intricacies is negligible.
///
#[derive(Debug, Clone)]
pub(crate) struct GraphEnumeratorState {
    first_unvalid: usize,
    m_max: Weight,
    m_reached: Option<usize>,
    m_end: Weight,
    debug: usize,
}

impl GraphEnumeratorState {
    pub(crate) fn new(graoh: &Graph, m_begin: Weight, m_end: Weight, debug: usize) -> Self {
        let n = graoh.node_count();
        if debug > 0 {
            println!("sought_reward: {}", m_begin);
            for col in 1..=std::cmp::min(debug, n) {
                println!("{}[{}] v: 0", "  ".repeat(col), col);
            }
        }
        Self {
            first_unvalid: n * n - 1,
            m_max: m_begin,
            m_reached: None,
            m_end,
            debug,
        }
    }

    pub(crate) fn next_graph(&mut self, graph: &mut Graph) -> bool {
        fn next_pos(row: usize, col: usize, n: usize) -> (usize, usize) {
            if col < n - 1 { (row, col + 1) } else { (row + 1, 0) }
        }
        fn prev_pos(row: usize, col: usize, n: usize) -> (usize, usize) {
            if col > 0 { (row, col - 1) } else { (row - 1, n - 1) }
        }

        let n = graph.weights.rows();
        let pos_final = n * n - 1;
        let mut pos = pos_final;
        let mut row = n - 1;
        let mut col = n - 1;
        while self.m_max <= self.m_end {
            loop {
                let bot = if graph.graph_type == GraphType::Undirected && row > col {
                    graph.weights[(col, row)]
                } else if row == 0 && col > 0 {
                    graph.weights[(row, col - 1)]
                } else if row > 0 && row != col {
                    graph.weights[(0, 1)]
                } else {
                    0
                };
                let top = if row == col {
                    0
                } else if graph.graph_type == GraphType::Undirected && row > col {
                    graph.weights[(col, row)]
                } else {
                    self.m_max
                };

                let v_new = if pos < self.first_unvalid {
                    std::cmp::max(graph.weights[(row, col)] + 1, bot)
                } else {
                    bot
                };
                self.first_unvalid = pos + 1;

                if v_new <= top {
                    graph.weights[(row, col)] = v_new;

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
                                if graph.weights[(i, j)] == graph.weights[(row, j)] {
                                    continue;
                                }
                                if graph.weights[(i, j)] > graph.weights[(row, j)] {
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
                                return true;
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
        false
    }
}

pub struct GraphEnumerator {
    graph: Graph,
    state: GraphEnumeratorState,
}

impl GraphEnumerator {
    pub(crate) fn new(
        node_count: usize,
        graph_type: GraphType,
        m_begin: Weight,
        m_end: Weight,
        debug: usize,
    ) -> Self {
        let graph = Graph::new(Grid::<Weight>::new(node_count, node_count), graph_type);
        let state = GraphEnumeratorState::new(&graph, m_begin, m_end, debug);
        Self { graph, state }
    }
}

impl Iterator for GraphEnumerator {
    type Item = Graph;

    fn next(&mut self) -> Option<Self::Item> {
        if self.state.next_graph(&mut self.graph) {
            Some(self.graph.clone())
        } else {
            None
        }
    }
}
