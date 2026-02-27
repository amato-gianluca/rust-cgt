//! Core data structures and algorithms for hedonic coalition games.

mod types;
mod graph;
mod graph_enumerator;
mod game;
mod coalition_structure;

/// Common scalar and identifier type aliases.
pub use types::*;
/// Graph representation and graph kind metadata.
pub use graph::{Graph, GraphType};
/// Hedonic game representation and utility model variants.
pub use game::{HedonicGame, GameType};
/// Coalition structure over a specific hedonic game instance.
pub use coalition_structure::CoalitionStructure;
