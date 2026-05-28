<img src="./_logo/fissure.svg" width="300" align="center" alt="Logo">

Performant and numerically robust matrix decompositions of small matrices.

### lasv2

Faithful port of LAPACK's `DLASV2` / `SLASV2` for the singular value decomposition of
2×2 upper-triangular matrices. Works with any `T: num_traits::Float` (`f32`, `f64`,
SIMD, or custom float types).

Provides three entry points:

- **`lasv2`** — matches the original LAPACK signature (mutable out-params).
- **`svd2_tri`** — ergonomic wrapper for upper-triangular 2×2 matrices `[f, g; 0, h]`.
  Returns `(U, (σ_max, σ_min), V)`.
- **`svd2`** — extends `svd2_tri` to general 2×2 matrices by first reducing to
  triangular form via a Givens rotation.

```rust
use lasv2::svd2;
let (u, (smax, smin), v) = svd2(1.0_f64, 2.0, 3.0, 4.0);
```
