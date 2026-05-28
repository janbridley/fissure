/*!
Performant and numerically robust matrix decompositions of small matrices.

### lasv2

Faithful port of LAPACK's `DLASV2` / `SLASV2` for the singular value decomposition of
2×2 upper-triangular matrices. Works with any `T: num_traits::Float` (`f32`, `f64`,
SIMD, or custom float types).

- **[`lasv2::lasv2`]** — matches the original LAPACK signature (mutable out-params).

### fissure

- **[`svd2`]** — SVD of a general 2×2 matrix, accepts `[[T; 2]; 2]` row-major input.
- **[`svd3`]** — SVD of a general 3×3 matrix, accepts `[[f64; 3]; 3]` row-major input.
  Based on Vertechy and Parenti-Castelli (2004) with numerical enhancements.

*/
use lasv2::lasv2;
use num_traits::Float;

/// A stack-allocated 2×2 matrix of type T.
pub type Mat2<T> = [[T; 2]; 2];

/// Computes the singular value decomposition of a general 2-by-2 matrix.
///
/// Given a matrix `m`, this function returns `(U, S, V)` where:
/// - `U` is a 2×2 orthogonal matrix (left singular vectors),
/// - `S` is a tuple `(σ_max, σ_min)` of singular values in descending order by absolute value,
/// - `V` is a 2×2 orthogonal matrix (right singular vectors),
///
/// such that `m = U · diag(S) · Vᵀ`.
///
/// # Example
/// ```rust
/// use fissure::svd2;
/// let (u, (smax, smin), v) = svd2([[1.0_f64, 2.0], [3.0, 4.0]]);
/// assert!(smax.abs() >= smin.abs());
/// ```
#[inline]
pub fn svd2<T: Float>(m: Mat2<T>) -> (Mat2<T>, (T, T), Mat2<T>) {
    let (a, b, c, d) = (m[0][0], m[0][1], m[1][0], m[1][1]);

    // Reduce to upper-triangular form via a Givens rotation
    let r = a.hypot(c);
    let tri = if r == T::zero() {
        [[a, b], [T::zero(), d]]
    } else {
        let cos = a / r;
        let sin = c / r;
        [[r, cos * b + sin * d], [T::zero(), cos * d - sin * b]]
    };

    // SVD of the upper-triangular matrix
    let (f, g, h) = (tri[0][0], tri[0][1], tri[1][1]);
    let (mut ssmin, mut ssmax, mut snr, mut csr, mut snl, mut csl) = (
        T::zero(),
        T::zero(),
        T::zero(),
        T::zero(),
        T::zero(),
        T::zero(),
    );
    lasv2(
        &f, &g, &h, &mut ssmin, &mut ssmax, &mut snr, &mut csr, &mut snl, &mut csl,
    );

    let u_tri: Mat2<T> = [[csl, -snl], [snl, csl]];
    let v: Mat2<T> = [[csr, -snr], [snr, csr]];

    if r == T::zero() {
        return (u_tri, (ssmax, ssmin), v);
    }

    let cos = a / r;
    let sin = c / r;

    // Left singular vectors of the original: U_full = G^T * U_tri
    let u_full = [
        [
            cos * u_tri[0][0] - sin * u_tri[1][0],
            cos * u_tri[0][1] - sin * u_tri[1][1],
        ],
        [
            sin * u_tri[0][0] + cos * u_tri[1][0],
            sin * u_tri[0][1] + cos * u_tri[1][1],
        ],
    ];

    (u_full, (ssmax, ssmin), v)
}

/// A stack-allocated 3×3 matrix of f64 values.
pub type Mat3 = [[f64; 3]; 3];

/// Computes the singular value decomposition of a general 3×3 matrix.
///
/// Based on the algorithm of Vertechy and Parenti-Castelli (2004) with numerical
/// enhancements: invariant-based polynomial coefficients to avoid condition number
/// squaring, and Viète's trigonometric method for the cubic eigenvalue solve.
///
/// Returns `(U, σ, V)` such that `m = U · diag(σ) · Vᵀ`, where U and V are
/// proper rotations (det = +1). The third singular value may be negative to
/// accurately represent reflections.
///
/// # Example
/// ```rust
/// use fissure::svd3;
/// let (u, s, v) = svd3([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
/// assert!(s[0].abs() >= s[1].abs());
/// ```
#[inline]
#[expect(clippy::too_many_lines, reason = "unrolled 3×3 SVD is inherently long")]
#[expect(clippy::must_use_candidate, reason = "SVD result should always be used")]
pub fn svd3(m: Mat3) -> (Mat3, [f64; 3], Mat3) {
    let [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]] = m;
    let mut scale = a00.abs();
    scale = scale.max(a01.abs()).max(a02.abs());
    scale = scale.max(a10.abs()).max(a11.abs()).max(a12.abs());
    scale = scale.max(a20.abs()).max(a21.abs()).max(a22.abs());

    if scale == 0.0 {
        return (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        );
    }

    let inv_scale = 1.0 / scale;
    let s00 = a00 * inv_scale;
    let s01 = a01 * inv_scale;
    let s02 = a02 * inv_scale;
    let s10 = a10 * inv_scale;
    let s11 = a11 * inv_scale;
    let s12 = a12 * inv_scale;
    let s20 = a20 * inv_scale;
    let s21 = a21 * inv_scale;
    let s22 = a22 * inv_scale;

    // Characteristic polynomial coefficients of B = A^T A from invariants of A
    let a_coeff = s00 * s00
        + s01 * s01
        + s02 * s02
        + s10 * s10
        + s11 * s11
        + s12 * s12
        + s20 * s20
        + s21 * s21
        + s22 * s22;

    let m00 = s11 * s22 - s12 * s21;
    let m01 = s10 * s22 - s12 * s20;
    let m02 = s10 * s21 - s11 * s20;
    let m10 = s01 * s22 - s02 * s21;
    let m11 = s00 * s22 - s02 * s20;
    let m12 = s00 * s21 - s01 * s20;
    let m20 = s01 * s12 - s02 * s11;
    let m21 = s00 * s12 - s02 * s10;
    let m22 = s00 * s11 - s01 * s10;

    let b_coeff = m00 * m00
        + m01 * m01
        + m02 * m02
        + m10 * m10
        + m11 * m11
        + m12 * m12
        + m20 * m20
        + m21 * m21
        + m22 * m22;
    let det_a = s00 * m00 - s01 * m01 + s02 * m02;
    let c_coeff = det_a * det_a;

    let (lambda1, lambda2, lambda3) =
        solve_characteristic_polynomial(a_coeff, b_coeff, c_coeff);

    // Right singular vectors from eigenvectors of B = A^T A
    let b00 = s00 * s00 + s10 * s10 + s20 * s20;
    let b01 = s00 * s01 + s10 * s11 + s20 * s21;
    let b02 = s00 * s02 + s10 * s12 + s20 * s22;
    let b11 = s01 * s01 + s11 * s11 + s21 * s21;
    let b12 = s01 * s02 + s11 * s12 + s21 * s22;
    let b22 = s02 * s02 + s12 * s12 + s22 * s22;
    let b: Mat3 = [[b00, b01, b02], [b01, b11, b12], [b02, b12, b22]];

    let mut v = sym_evecs_from_evals(&b, [lambda1, lambda2, lambda3]);
    if det3(&v) < 0.0 {
        for row in &mut v {
            row[2] = -row[2];
        }
    }

    // Left singular vectors and signed singular values
    let s_mat: Mat3 = [[s00, s01, s02], [s10, s11, s12], [s20, s21, s22]];
    let mut u_cols = [[0.0; 3]; 3];
    let mut sigmas = [0.0; 3];
    for i in 0..2 {
        let av = mat3_mul_vec(&s_mat, [v[0][i], v[1][i], v[2][i]]);
        let s_i = (av[0] * av[0] + av[1] * av[1] + av[2] * av[2]).sqrt();
        sigmas[i] = s_i * scale;
        if s_i > 1e-10 {
            u_cols[i] = [av[0] / s_i, av[1] / s_i, av[2] / s_i];
        } else {
            let p = if i == 1 {
                u_cols[0]
            } else {
                [1.0, 0.0, 0.0]
            };
            let u_new = cross(
                p,
                if p[0].abs() < 0.8 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                },
            );
            let n = (u_new[0] * u_new[0] + u_new[1] * u_new[1] + u_new[2] * u_new[2]).sqrt();
            u_cols[i] = [u_new[0] / n, u_new[1] / n, u_new[2] / n];
        }
    }
    u_cols[2] = cross(u_cols[0], u_cols[1]);
    let av2 = mat3_mul_vec(&s_mat, [v[0][2], v[1][2], v[2][2]]);
    sigmas[2] = dot(u_cols[2], av2) * scale;

    let u: Mat3 = [
        [u_cols[0][0], u_cols[1][0], u_cols[2][0]],
        [u_cols[0][1], u_cols[1][1], u_cols[2][1]],
        [u_cols[0][2], u_cols[1][2], u_cols[2][2]],
    ];

    (u, sigmas, v)
}

/// Computes the dot product of two 3D vectors.
#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Computes the cross product of two 3D vectors.
#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Multiplies a 3×3 matrix by a 3D vector.
#[inline]
fn mat3_mul_vec(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// Computes the determinant of a 3×3 matrix.
#[inline]
fn det3(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Solves the characteristic polynomial λ³ − a·λ² + b·λ − c = 0
/// using Viète's trigonometric method for the depressed cubic.
#[inline]
fn solve_characteristic_polynomial(
    a_coeff: f64,
    b_coeff: f64,
    c_coeff: f64,
) -> (f64, f64, f64) {
    let a3 = a_coeff / 3.0;
    let p = b_coeff - a_coeff * a3;
    let q = -2.0 * a3 * a3 * a3 + a3 * b_coeff - c_coeff;

    if p >= 0.0 {
        return (a3, a3, a3);
    }

    let s = (-p / 3.0).sqrt();
    let arg = (-q / (2.0 * s * s * s)).clamp(-1.0, 1.0);
    let theta = arg.acos() / 3.0;
    let m = 2.0 * s;

    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let sqrt3_half = 0.5 * 3.0_f64.sqrt();

    let l0 = a3 + m * cos_t;
    let l1 = a3 + m * (cos_t * (-0.5) + sqrt3_half * sin_t);
    let l2 = a3 + m * (cos_t * (-0.5) - sqrt3_half * sin_t);

    (l0, l1, l2)
}

/// Extracts eigenvectors of a symmetric 3×3 matrix from known eigenvalues.
#[inline]
fn sym_evecs_from_evals(m: &Mat3, evals: [f64; 3]) -> Mat3 {
    let off_diag = m[0][1] * m[0][1] + m[0][2] * m[0][2] + m[1][2] * m[1][2];
    if off_diag <= 1e-20 {
        // Diagonal matrix: map eigenvalues to diagonal positions in descending order
        let diag = [m[0][0], m[1][1], m[2][2]];
        let mut idx = [0_usize, 1, 2];
        if diag[idx[0]] < diag[idx[1]] {
            idx.swap(0, 1);
        }
        if diag[idx[1]] < diag[idx[2]] {
            idx.swap(1, 2);
        }
        if diag[idx[0]] < diag[idx[1]] {
            idx.swap(0, 1);
        }
        let mut result = [[0.0; 3]; 3];
        for (col, &row) in idx.iter().enumerate() {
            result[row][col] = 1.0;
        }
        return result;
    }

    let (evec0, evec1, evec2) = if det3(m) >= 0.0 {
        let e0 = sym_evec0(m, evals[0]);
        let e1 = sym_evec1(m, e0, evals[1]);
        let e2 = cross(e1, e0);
        (e0, e1, e2)
    } else {
        let e2 = sym_evec0(m, evals[2]);
        let e1 = sym_evec1(m, e2, evals[1]);
        let e0 = cross(e2, e1);
        (e0, e1, e2)
    };

    [
        [evec0[0], evec1[0], evec2[0]],
        [evec0[1], evec1[1], evec2[1]],
        [evec0[2], evec1[2], evec2[2]],
    ]
}

/// Computes the eigenvector corresponding to the first eigenvalue via adjugate cross-products.
#[inline]
fn sym_evec0(m: &Mat3, eval0: f64) -> [f64; 3] {
    let row0 = [m[0][0] - eval0, m[0][1], m[0][2]];
    let row1 = [m[1][0], m[1][1] - eval0, m[1][2]];
    let row2 = [m[2][0], m[2][1], m[2][2] - eval0];

    let r0xr1 = cross(row0, row1);
    let r0xr2 = cross(row0, row2);
    let r1xr2 = cross(row1, row2);

    let d0 = dot(r0xr1, r0xr1);
    let d1 = dot(r0xr2, r0xr2);
    let d2 = dot(r1xr2, r1xr2);

    if d0 <= 1e-20 && d1 <= 1e-20 && d2 <= 1e-20 {
        [1.0, 0.0, 0.0]
    } else if d0 >= d1 && d0 >= d2 {
        let s = d0.sqrt();
        [r0xr1[0] / s, r0xr1[1] / s, r0xr1[2] / s]
    } else if d1 >= d0 && d1 >= d2 {
        let s = d1.sqrt();
        [r0xr2[0] / s, r0xr2[1] / s, r0xr2[2] / s]
    } else {
        let s = d2.sqrt();
        [r1xr2[0] / s, r1xr2[1] / s, r1xr2[2] / s]
    }
}

/// Computes the eigenvector corresponding to the second eigenvalue in the plane
/// orthogonal to the first eigenvector.
#[inline]
fn sym_evec1(m: &Mat3, evec0: [f64; 3], eval1: f64) -> [f64; 3] {
    let u = if evec0[0].abs() > evec0[1].abs() {
        let s = (evec0[0] * evec0[0] + evec0[2] * evec0[2]).sqrt();
        [-evec0[2] / s, 0.0, evec0[0] / s]
    } else {
        let s = (evec0[1] * evec0[1] + evec0[2] * evec0[2]).sqrt();
        [0.0, evec0[2] / s, -evec0[1] / s]
    };
    let v = cross(evec0, u);
    let mu = mat3_mul_vec(m, u);
    let mv = mat3_mul_vec(m, v);
    let m00 = dot(u, mu) - eval1;
    let m01 = dot(u, mv);
    let m11 = dot(v, mv) - eval1;

    if m00.abs() >= m11.abs() {
        if m00.abs().max(m01.abs()) <= 1e-20 {
            u
        } else if m00.abs() >= m01.abs() {
            let m01_scaled = m01 / m00;
            let s = (1.0 + m01_scaled * m01_scaled).sqrt();
            let c = 1.0 / s;
            [
                m01_scaled * c * u[0] - c * v[0],
                m01_scaled * c * u[1] - c * v[1],
                m01_scaled * c * u[2] - c * v[2],
            ]
        } else {
            let m00_scaled = m00 / m01;
            let s = (1.0 + m00_scaled * m00_scaled).sqrt();
            let c = 1.0 / s;
            [
                c * u[0] - m00_scaled * c * v[0],
                c * u[1] - m00_scaled * c * v[1],
                c * u[2] - m00_scaled * c * v[2],
            ]
        }
    } else if m00.abs().max(m01.abs()) <= 1e-20 {
        u
    } else if m11.abs() >= m01.abs() {
        let m01_scaled = m01 / m11;
        let s = (1.0 + m01_scaled * m01_scaled).sqrt();
        let c = 1.0 / s;
        [
            c * u[0] - m01_scaled * c * v[0],
            c * u[1] - m01_scaled * c * v[1],
            c * u[2] - m01_scaled * c * v[2],
        ]
    } else {
        let m11_scaled = m11 / m01;
        let s = (1.0 + m11_scaled * m11_scaled).sqrt();
        let c = 1.0 / s;
        [
            m11_scaled * c * u[0] - c * v[0],
            m11_scaled * c * u[1] - c * v[1],
            m11_scaled * c * u[2] - c * v[2],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use rstest::rstest;

    #[expect(clippy::missing_docs_in_private_items, reason = "test helper")]
    #[expect(clippy::unwrap_used, reason = "test helper")]
    struct SvdMetrics {
        sv_err: f64,
        recon_err: f64,
        ortho_err: f64,
    }

    #[expect(clippy::missing_docs_in_private_items, reason = "test helper")]
    fn measure_svd3(m: Mat3) -> SvdMetrics {
        let (u, s, v) = svd3(m);

        let faer_svd = faer::mat![
            [m[0][0], m[0][1], m[0][2]],
            [m[1][0], m[1][1], m[1][2]],
            [m[2][0], m[2][1], m[2][2]]
        ]
        .svd()
        .unwrap();

        let mut sv_err = 0.0_f64;
        let mut ours_sorted = [s[0].abs(), s[1].abs(), s[2].abs()];
        ours_sorted.sort_by(|a, b| b.total_cmp(a));
        let mut faer_sorted = [faer_svd.S()[0], faer_svd.S()[1], faer_svd.S()[2]];
        faer_sorted.sort_by(|a, b| b.total_cmp(a));
        let max_sv = ours_sorted[0].max(faer_sorted[0]).max(1e-30);
        for i in 0..3 {
            sv_err = sv_err.max((ours_sorted[i] - faer_sorted[i]).abs() / max_sv);
        }

        let mut recon = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    recon[i][j] += u[i][k] * s[k] * v[j][k];
                }
            }
        }
        let mut diff_sq = 0.0_f64;
        let mut a_sq = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let d = recon[i][j] - m[i][j];
                diff_sq += d * d;
                a_sq += m[i][j] * m[i][j];
            }
        }
        let recon_err = diff_sq.sqrt() / a_sq.sqrt().max(1e-30);

        let mut ortho_err = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let mut uu = 0.0_f64;
                let mut vv = 0.0_f64;
                for k in 0..3 {
                    uu += u[k][i] * u[k][j];
                    vv += v[k][i] * v[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                ortho_err = ortho_err
                    .max((uu - expected).abs())
                    .max((vv - expected).abs());
            }
        }

        SvdMetrics {
            sv_err,
            recon_err,
            ortho_err,
        }
    }

    #[test]
    #[expect(clippy::print_stderr, reason = "prints validation stats")]
    fn validate_svd3_stability() {
        use rand::{RngExt, SeedableRng, distr::Uniform, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(12345);

        let mut sv_errs: Vec<f64> = Vec::new();
        let mut recon_errs: Vec<f64> = Vec::new();
        let mut ortho_errs: Vec<f64> = Vec::new();

        let record = |m: Mat3, sv: &mut Vec<f64>, rc: &mut Vec<f64>, or: &mut Vec<f64>| {
            let metrics = measure_svd3(m);
            sv.push(metrics.sv_err);
            rc.push(metrics.recon_err);
            or.push(metrics.ortho_err);
        };

        // Battery 1: random matrices at various scales
        let scales: &[f64] = &[1.0, 1e3, 1e6, 1e-3, 1e-6, 1e15, 1e-15];
        for &scale in scales {
            let range = Uniform::new(-scale, scale).unwrap();
            for _ in 0..1500 {
                let m: Mat3 = [
                    [
                        rng.sample(range),
                        rng.sample(range),
                        rng.sample(range),
                    ],
                    [
                        rng.sample(range),
                        rng.sample(range),
                        rng.sample(range),
                    ],
                    [
                        rng.sample(range),
                        rng.sample(range),
                        rng.sample(range),
                    ],
                ];
                record(m, &mut sv_errs, &mut recon_errs, &mut ortho_errs);
            }
        }

        // Battery 2: adversarial / degenerate matrices
        let adversarial: Vec<Mat3> = vec![
            // Identity and scalar multiples (all-equal singular values → p ≈ 0)
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            [[1e-10, 0.0, 0.0], [0.0, 1e-10, 0.0], [0.0, 0.0, 1e-10]],
            // Repeated singular values
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
            [[5.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-15]],
            [[1e15, 0.0, 0.0], [0.0, 1e15, 0.0], [0.0, 0.0, 1.0]],
            // Nearly-equal singular values
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0 + 1e-14, 0.0],
                [0.0, 0.0, 1.0 - 1e-14],
            ],
            [
                [1.0, 1e-15, 0.0],
                [1e-15, 1.0, 1e-15],
                [0.0, 1e-15, 1.0],
            ],
            // Rank-deficient
            [[0.0; 3]; 3],
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
            // Extreme dynamic range
            [
                [1e15, 1e-15, 0.0],
                [0.0, 1e15, 1e-15],
                [1e-15, 0.0, 1e15],
            ],
            [[1e-15, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e15]],
            // Reflections (det = −1)
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            // Rotation
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            // Existing regression tests
            [
                [LARGE_F64, 1e8, 1e8],
                [1e8, SMALL_F64, 1e8],
                [1e8, 1e8, LARGE_F64],
            ],
            [
                [SMALL_F64, SMALL_F64, SMALL_F64],
                [SMALL_F64, SMALL_F64, SMALL_F64],
                [SMALL_F64, SMALL_F64, SMALL_F64],
            ],
        ];
        for m in adversarial {
            record(m, &mut sv_errs, &mut recon_errs, &mut ortho_errs);
        }

        // Stats
        let total = sv_errs.len();
        sv_errs.sort_by(|a, b| a.total_cmp(b));
        recon_errs.sort_by(|a, b| a.total_cmp(b));
        ortho_errs.sort_by(|a, b| a.total_cmp(b));

        let pct = |v: &[f64], p: usize| -> f64 { v[p.min(v.len() - 1)] };
        let max_of = |v: &[f64]| -> f64 { v.last().copied().unwrap_or(0.0) };
        let mean_of = |v: &[f64]| -> f64 { v.iter().sum::<f64>() / v.len() as f64 };

        eprintln!("\n=== svd3 validation ({total} matrices) ===");
        eprintln!(
            "  SV rel err:   max={:.2e}  mean={:.2e}  p99={:.2e}",
            max_of(&sv_errs),
            mean_of(&sv_errs),
            pct(&sv_errs, total * 99 / 100),
        );
        eprintln!(
            "  Recon rel err: max={:.2e}  mean={:.2e}  p99={:.2e}",
            max_of(&recon_errs),
            mean_of(&recon_errs),
            pct(&recon_errs, total * 99 / 100),
        );
        eprintln!(
            "  Ortho err:     max={:.2e}  mean={:.2e}  p99={:.2e}",
            max_of(&ortho_errs),
            mean_of(&ortho_errs),
            pct(&ortho_errs, total * 99 / 100),
        );

        assert!(
            max_of(&sv_errs) < 1e-10,
            "max SV error: {:.2e}",
            max_of(&sv_errs),
        );
        assert!(
            max_of(&recon_errs) < 1e-10,
            "max reconstruction error: {:.2e}",
            max_of(&recon_errs),
        );
        assert!(
            max_of(&ortho_errs) < 1e-10,
            "max orthogonality error: {:.2e}",
            max_of(&ortho_errs),
        );
    }

    const SMALL_F64: f64 = 1e-15;
    const LARGE_F64: f64 = 1e30;

    #[rstest]
    #[case::identity([[1.0, 0.0], [0.0, 1.0]])]
    #[case::simple([[1.0, 2.0], [3.0, 4.0]])]
    #[case::negative([[-2.0, 5.0], [-3.0, 7.0]])]
    #[case::rotation([[0.0, -1.0], [1.0, 0.0]])]
    #[case::symmetric([[3.0, 1.0], [1.0, 3.0]])]
    #[case::zero([[0.0, 0.0], [0.0, 0.0]])]
    #[case::near_singular([[1.0, 2.0], [2.0, 4.0]])]
    #[case::degenerate([[LARGE_F64, 1e8], [1e8, SMALL_F64]])]
    #[case::all_tiny_positive([[SMALL_F64, SMALL_F64], [SMALL_F64, SMALL_F64]])]
    #[case::large_mixed_signs([[LARGE_F64, -LARGE_F64], [LARGE_F64, -LARGE_F64]])]
    fn test_svd2(#[case] m: Mat2<f64>) {
        let (u, (smax, smin), v) = svd2(m);

        assert!(smax.abs() >= smin.abs());

        let svd = faer::mat![[m[0][0], m[0][1]], [m[1][0], m[1][1]]]
            .svd()
            .unwrap();
        assert_ulps_eq!(smin.abs(), svd.S()[1], max_ulps = 100);
        assert_ulps_eq!(smax.abs(), svd.S()[0], max_ulps = 100);

        // Orthogonality
        let u_norm = u[0][0] * u[0][0] + u[1][0] * u[1][0];
        let v_norm = v[0][0] * v[0][0] + v[1][0] * v[1][0];
        assert_ulps_eq!(u_norm, 1.0, max_ulps = 4);
        assert_ulps_eq!(v_norm, 1.0, max_ulps = 4);

        // Reconstruction
        let u_m = faer::mat![[u[0][0], u[0][1]], [u[1][0], u[1][1]]];
        let v_m = faer::mat![[v[0][0], v[0][1]], [v[1][0], v[1][1]]];
        let s_m = faer::mat![[smax, 0.0], [0.0, smin]];
        let reconstructed = u_m * s_m * v_m.transpose();

        assert_ulps_eq!(reconstructed[(0, 0)], m[0][0], max_ulps = 100);
        assert_ulps_eq!(reconstructed[(0, 1)], m[0][1], max_ulps = 100);
        assert_ulps_eq!(reconstructed[(1, 0)], m[1][0], max_ulps = 100);
        assert_ulps_eq!(reconstructed[(1, 1)], m[1][1], max_ulps = 100);
    }

    #[rstest]
    #[case::identity([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])]
    #[case::diagonal([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]])]
    #[case::simple([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])]
    #[case::rotation([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])]
    #[case::zero([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])]
    #[case::rank1([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])]
    #[case::symmetric([[3.0, 1.0, 2.0], [1.0, 4.0, 0.0], [2.0, 0.0, 5.0]])]
    #[case::negative([[-2.0, 5.0, -3.0], [7.0, -1.0, 4.0], [0.0, 3.0, -6.0]])]
    #[case::degenerate([
        [LARGE_F64, 1e8, 1e8],
        [1e8, SMALL_F64, 1e8],
        [1e8, 1e8, LARGE_F64]
    ])]
    #[case::all_tiny_positive([
        [SMALL_F64, SMALL_F64, SMALL_F64],
        [SMALL_F64, SMALL_F64, SMALL_F64],
        [SMALL_F64, SMALL_F64, SMALL_F64]
    ])]
    fn test_svd3(#[case] m: Mat3) {
        let (u, s, v) = svd3(m);

        // Singular values in descending order by absolute value
        assert!(s[0].abs() >= s[1].abs(), "|s[0]| < |s[1]|: {}", s[0].abs());
        assert!(s[1].abs() >= s[2].abs(), "|s[1]| < |s[2]|: {}", s[1].abs());

        // Proper rotations
        assert!(
            (det3(&u) - 1.0).abs() < 1e-10,
            "det(U) = {}",
            det3(&u)
        );
        assert!(
            (det3(&v) - 1.0).abs() < 1e-10,
            "det(V) = {}",
            det3(&v)
        );

        // Orthogonality: U^T U = I and V^T V = I
        for i in 0..3 {
            for j in 0..3 {
                let mut u_dot = 0.0;
                let mut v_dot = 0.0;
                for k in 0..3 {
                    u_dot += u[k][i] * u[k][j];
                    v_dot += v[k][i] * v[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (u_dot - expected).abs() < 1e-10,
                    "U^T U[{i}][{j}] = {u_dot}"
                );
                assert!(
                    (v_dot - expected).abs() < 1e-10,
                    "V^T V[{i}][{j}] = {v_dot}"
                );
            }
        }

        // Compare singular values against faer
        let svd = faer::mat![
            [m[0][0], m[0][1], m[0][2]],
            [m[1][0], m[1][1], m[1][2]],
            [m[2][0], m[2][1], m[2][2]]
        ]
        .svd()
        .unwrap();
        let sv_tol = f64::max(s[0].abs(), svd.S()[0]).max(1.0) * f64::EPSILON * 100.0;
        for (i, sv) in s.iter().enumerate() {
            assert!(
                (sv.abs() - svd.S()[i]).abs() < sv_tol,
                "s[{i}]: got {}, expected {}, tol {}",
                sv.abs(),
                svd.S()[i],
                sv_tol
            );
        }

        // Reconstruction: A = U * diag(S) * V^T
        let mut reconstructed = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    reconstructed[i][j] += u[i][k] * s[k] * v[j][k];
                }
            }
        }
        let max_elem = m
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        let tol = (max_elem.max(1.0) * 1e-10).max(1e-12);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (reconstructed[i][j] - m[i][j]).abs() < tol,
                    "reconstruction failed at [{i}][{j}]: got {}, expected {}",
                    reconstructed[i][j],
                    m[i][j]
                );
            }
        }
    }
}
