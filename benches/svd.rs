#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
#![expect(missing_docs, reason = "benches don't need public documentation")]
#![expect(clippy::unwrap_used, reason = "benches don't need error handling")]

use divan::{Bencher, black_box, counter::ItemsCount};
use fissure::{Mat2, Mat3};
use rand::{RngExt, SeedableRng, distr::Uniform, rngs::StdRng};

fn main() {
    divan::main();
}

fn random_mat2(rng: &mut StdRng) -> Mat2<f64> {
    let range = Uniform::new(-100.0, 100.0).unwrap();
    [
        [rng.sample(range), rng.sample(range)],
        [rng.sample(range), rng.sample(range)],
    ]
}

fn random_mat3(rng: &mut StdRng) -> Mat3 {
    let range = Uniform::new(-100.0, 100.0).unwrap();
    [
        [rng.sample(range), rng.sample(range), rng.sample(range)],
        [rng.sample(range), rng.sample(range), rng.sample(range)],
        [rng.sample(range), rng.sample(range), rng.sample(range)],
    ]
}

/// Closed-form 2×2 singular values via the quadratic formula on A^T A.
/// Numerically unstable (condition-number squaring) but serves as a speed baseline.
fn svd2_polynomial(m: Mat2<f64>) -> [f64; 2] {
    let [a, b] = m[0];
    let [c, d] = m[1];
    let tr = a * a + c * c + b * b + d * d;
    let det = a * d - b * c;
    let disc = (tr * tr - 4.0 * det * det).sqrt();
    [
        f64::midpoint(tr, disc).sqrt(),
        f64::midpoint(tr, -disc).max(0.0).sqrt(),
    ]
}

/// Closed-form 3×3 singular values via Cardano's formula on A^T A.
/// Numerically unstable (condition-number squaring + casus irreducibilis) but
/// serves as a speed baseline.
fn svd3_polynomial(m: Mat3) -> [f64; 3] {
    let [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]] = m;

    // B = A^T A (symmetric)
    let b00 = a00 * a00 + a10 * a10 + a20 * a20;
    let b01 = a00 * a01 + a10 * a11 + a20 * a21;
    let b02 = a00 * a02 + a10 * a12 + a20 * a22;
    let b11 = a01 * a01 + a11 * a11 + a21 * a21;
    let b12 = a01 * a02 + a11 * a12 + a21 * a22;
    let b22 = a02 * a02 + a12 * a12 + a22 * a22;

    // Characteristic polynomial λ³ − a·λ² + b·λ − c = 0
    let a = b00 + b11 + b22;
    let b = b00 * b11 - b01 * b01 + b00 * b22 - b02 * b02 + b11 * b22 - b12 * b12;
    let c = b00 * (b11 * b22 - b12 * b12)
        - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02);

    // Depressed cubic t³ + pt + q = 0, t = λ − a/3
    let p = b - a * a / 3.0;
    let q = -c + a * b / 3.0 - 2.0 * a * a * a / 27.0;
    let r = (-p * p * p / 27.0).max(0.0).sqrt();
    let cos_arg = if r > 0.0 {
        (-q / (2.0 * r)).clamp(-1.0, 1.0)
    } else {
        1.0
    };
    let phi = cos_arg.acos() / 3.0;
    let amp = 2.0 * (-p / 3.0).max(0.0).sqrt();
    let shift = a / 3.0;
    let sqrt3_half = 0.5 * 3.0_f64.sqrt();
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    let l1 = amp * cos_phi + shift;
    let l2 = amp * (cos_phi * (-0.5) - sqrt3_half * sin_phi) + shift;
    let l3 = amp * (cos_phi * (-0.5) + sqrt3_half * sin_phi) + shift;

    let mut s = [
        l1.max(0.0).sqrt(),
        l2.max(0.0).sqrt(),
        l3.max(0.0).sqrt(),
    ];
    s.sort_unstable_by(|a, b| b.total_cmp(a));
    s
}

// ── Robust implementations ──────────────────────────────────────────

#[divan::bench(name = "svd2")]
fn bench_svd2(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| random_mat2(&mut rng))
        .bench_local_values(|m| black_box(fissure::svd2(m)));
}

#[divan::bench(name = "svd3")]
fn bench_svd3(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| random_mat3(&mut rng))
        .bench_local_values(|m| black_box(fissure::svd3(m)));
}

// ── Closed-form speed-of-light baselines ────────────────────────────

#[divan::bench(name = "svd2_closed")]
fn bench_svd2_closed(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| random_mat2(&mut rng))
        .bench_local_values(|m| black_box(svd2_polynomial(m)));
}

#[divan::bench(name = "svd3_closed")]
fn bench_svd3_closed(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| random_mat3(&mut rng))
        .bench_local_values(|m| black_box(svd3_polynomial(m)));
}
