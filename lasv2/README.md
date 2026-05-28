# LASV2

Faithful port of LAPACK's `DLASV2` / `SLASV2` for the singular value decomposition of
2×2 upper-triangular matrices. Works with any `T: num_traits::Float` (`f32`, `f64`,
SIMD, or custom float types).

- **`lasv2`** — matches the original LAPACK signature (mutable out-params).

```rust
use lasv2::lasv2;

// Decompose the upper-triangular matrix:
// [ 3  5 ]
// [ 0  4 ]
let f = 3.0_f64;
let g = 5.0;
let h = 4.0;
let (mut ssmin, mut ssmax, mut snr, mut csr, mut snl, mut csl) =
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

lasv2(&f, &g, &h, &mut ssmin, &mut ssmax, &mut snr, &mut csr, &mut snl, &mut csl);

// Produces:
//   U = [ csl -snl ]   Σ = [ ssmax  0    ]   V = [ csr -snr ]
//       [ snl  csl ]       [ 0      ssmin ]       [ snr  csr ]
```
