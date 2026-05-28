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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use rstest::rstest;

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
}
