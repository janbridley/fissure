<img src="./_logo/fissure.svg" width="300" align="center" alt="Logo">

Performant and numerically robust matrix decompositions of small matrices.

### lasv2

Faithful port of LAPACK's `DLASV2` / `SLASV2` for the singular value decomposition of
2×2 upper-triangular matrices. Works with any `T: num_traits::Float` (`f32`, `f64`,
SIMD, or custom float types).

- **`lasv2`** — matches the original LAPACK signature (mutable out-params).

### fissure

- **`svd2`** — SVD of a general 2×2 matrix, accepts `[[T; 2]; 2]` row-major input.
