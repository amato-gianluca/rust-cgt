mod graph;
mod game;
mod coalition_structure;

pub use graph::{Graph, GraphType};
pub use game::{HedonicGame, GameType};
pub use coalition_structure::CoalitionStructure;

type Weight = u64;
type Agent = usize;
type Coalition = usize;
