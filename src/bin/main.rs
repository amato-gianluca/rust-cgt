use cgt::*;
use std::time::Instant;

fn main() {

    let k=2;
    let n=7;
    let m=2;

    let now = Instant::now();
    println!("k: {} n: {} m: {} count: {:?}", k, n, m, HedonicGame::count_unstable(n, GraphType::Undirected, m, m, GameType::Fractional, Some(k)));
    println!("Time: {} seconds", now.elapsed().as_secs())
}
