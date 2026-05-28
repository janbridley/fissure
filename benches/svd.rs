#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
#![expect(missing_docs, reason = "benches don't need public documentation")]

use divan::{Bencher, black_box, counter::ItemsCount};
use fissure::{Mat2, svd2};
use rand::{RngExt, SeedableRng, distr::Uniform, rngs::StdRng};

fn main() {
    divan::main();
}

fn random_matrix(rng: &mut StdRng) -> Mat2<f64> {
    let range = Uniform::new(-100.0, 100.0).expect("a valid distribution");
    [
        [rng.sample(range), rng.sample(range)],
        [rng.sample(range), rng.sample(range)],
    ]
}

#[divan::bench]
fn svd2_random(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| random_matrix(&mut rng))
        .bench_local_values(|m| black_box(svd2(m)));
}
