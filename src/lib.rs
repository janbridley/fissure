pub use lasv2::{Mat22, svd2_tri};

use num_traits::Float;

/// Computes the singular value decomposition of a general 2-by-2 matrix.
///
/// Given a matrix:
/// ```text
/// [ a  b ]
/// [ c  d ]
/// ```
///
/// This function applies a Givens rotation to reduce to upper triangular form,
/// then delegates to [`svd2_tri`].
///
/// # Example
/// ```rust
/// use fissure::svd2;
/// let (u, (smax, smin), v) = svd2(1.0_f64, 2.0, 3.0, 4.0);
/// assert!(smax.abs() >= smin.abs());
/// ```
pub fn svd2<T: Float>(a: T, b: T, c: T, d: T) -> (Mat22<T>, (T, T), Mat22<T>) {
    let r = a.hypot(c);
    if r == T::zero() {
        return svd2_tri(a, b, d);
    }
    let cos = a / r;
    let sin = c / r;

    // T = G * A is upper triangular
    let g = cos * b + sin * d;
    let h = cos * d - sin * b;

    let (u, (smax, smin), v) = svd2_tri(r, g, h);

    // Left singular vectors of A = G^T * U
    let u_full = [
        [cos * u[0][0] - sin * u[1][0], cos * u[0][1] - sin * u[1][1]],
        [sin * u[0][0] + cos * u[1][0], sin * u[0][1] + cos * u[1][1]],
    ];

    (u_full, (smax, smin), v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use rstest::rstest;

    const SMALL_F64: f64 = 1e-15;
    const LARGE_F64: f64 = 1e30;

    #[rstest]
    #[case::identity((1.0, 0.0, 0.0, 1.0))]
    #[case::simple((1.0, 2.0, 3.0, 4.0))]
    #[case::negative((-2.0, 5.0, -3.0, 7.0))]
    #[case::rotation((0.0, -1.0, 1.0, 0.0))]
    #[case::symmetric((3.0, 1.0, 1.0, 3.0))]
    #[case::zero((0.0, 0.0, 0.0, 0.0))]
    #[case::near_singular((1.0, 2.0, 2.0, 4.0))]
    #[case::degenerate((LARGE_F64, 1e8, 1e8, SMALL_F64))]
    #[case::all_tiny_positive((SMALL_F64, SMALL_F64, SMALL_F64, SMALL_F64))]
    #[case::large_mixed_signs((LARGE_F64, -LARGE_F64, LARGE_F64, -LARGE_F64))]
    fn test_svd2(#[case] (a, b, c, d): (f64, f64, f64, f64)) {
        let (u, (smax, smin), v) = svd2(a, b, c, d);

        assert!(smax.abs() >= smin.abs());

        let svd = faer::mat![[a, b], [c, d]].svd().unwrap();
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

        assert_ulps_eq!(reconstructed[(0, 0)], a, max_ulps = 100);
        assert_ulps_eq!(reconstructed[(0, 1)], b, max_ulps = 100);
        assert_ulps_eq!(reconstructed[(1, 0)], c, max_ulps = 100);
        assert_ulps_eq!(reconstructed[(1, 1)], d, max_ulps = 100);
    }
}
