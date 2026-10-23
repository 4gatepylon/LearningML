//! CPU matrix multiplication kernels, starting from a naive baseline.
//!
//! All matrices are dense, row-major `f64` slices.

#![warn(missing_docs)]

/// Computes `C = A * B`, overwriting `c` (it is not accumulated into).
///
/// `a` is `m x k`, `b` is `k x n` and `c` is `m x n`.
///
/// # Panics
///
/// Panics if `a`, `b` or `c` does not have exactly `m * k`, `k * n` or
/// `m * n` elements respectively.
///
/// # Examples
///
/// ```
/// use tiled_matmul_cpu::matmul_naive;
///
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let b = [5.0, 6.0, 7.0, 8.0];
/// let mut c = [0.0; 4];
/// matmul_naive(&a, &b, &mut c, 2, 2, 2);
/// assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
/// ```
pub fn matmul_naive(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    //! The input space is partitioned over shapes `n x k x m`, where an
    //! `n x k` matrix is multiplied by a `k x m` matrix to give `n x m`.
    //! (`matmul_naive` calls the row count `m` and the column count `n`, so
    //! `check` passes `(n, k, m)` as its `(m, k, n)`.)
    //!
    //! 1. Size-1 dimensions (`ones_*`): each of `n`, `k`, `m` is 1 or greater
    //!    than 1, giving 8 cells. The 7 with at least one 1 are tested here,
    //!    with the other dimensions random in `2..=6`. The cell with no 1 is
    //!    covered by group 2.
    //! 2. Size orderings (`*_gt_*_gt_*`): `n`, `k`, `m` are distinct random
    //!    sizes in `2..=6`, one test for each of the 6 relative orders.
    //! 3. Zero matrices (`zero_*`): the left, the right, then both inputs are
    //!    all zeros.
    //!
    //! Across groups 1 and 2 each input is all positive, all negative, or
    //! mixed (positive, negative and zero), on both sides; the comment in each
    //! test names its pattern. Expected outputs are literals computed
    //! independently of this crate. Every test starts `c` filled with garbage,
    //! so each also checks that `c` is overwritten rather than accumulated into.

    use super::*;

    fn check(a: &[f64], b: &[f64], (n, k, m): (usize, usize, usize), expected: &[f64]) {
        let mut c = vec![99.0; n * m];
        matmul_naive(a, b, &mut c, n, k, m);
        assert_eq!(c, expected);
    }

    #[test]
    fn ones_n() {
        // A: pos, B: pos
        let a = [9.0, 1.0];
        let b = [2.0, 4.0, 7.0, 8.0, 3.0, 5.0, 6.0, 9.0, 7.0, 3.0];
        let expected = [23.0, 42.0, 72.0, 79.0, 30.0];
        check(&a, &b, (1, 2, 5), &expected);
    }

    #[test]
    fn ones_k() {
        // A: neg, B: neg
        let a = [-5.0, -7.0, -8.0, -7.0, -7.0];
        let b = [-1.0, -8.0, -2.0, -1.0, -5.0, -5.0];
        let expected = [
            5.0, 40.0, 10.0, 5.0, 25.0, 25.0, 7.0, 56.0, 14.0, 7.0, 35.0, 35.0, 8.0, 64.0, 16.0,
            8.0, 40.0, 40.0, 7.0, 56.0, 14.0, 7.0, 35.0, 35.0, 7.0, 56.0, 14.0, 7.0, 35.0, 35.0,
        ];
        check(&a, &b, (5, 1, 6), &expected);
    }

    #[test]
    fn ones_m() {
        // A: pos, B: neg
        let a = [5.0, 4.0, 7.0, 1.0, 1.0, 8.0, 4.0, 7.0, 2.0, 1.0, 4.0, 7.0];
        let b = [-9.0, -9.0, -7.0];
        let expected = [-130.0, -74.0, -113.0, -94.0];
        check(&a, &b, (4, 3, 1), &expected);
    }

    #[test]
    fn ones_n_k() {
        // A: neg, B: pos
        let a = [-2.0];
        let b = [5.0, 9.0, 2.0];
        let expected = [-10.0, -18.0, -4.0];
        check(&a, &b, (1, 1, 3), &expected);
    }

    #[test]
    fn ones_n_m() {
        // A: mixed, B: mixed
        let a = [-8.0, 2.0, 0.0];
        let b = [-7.0, 9.0, 0.0];
        let expected = [74.0];
        check(&a, &b, (1, 3, 1), &expected);
    }

    #[test]
    fn ones_k_m() {
        // A: mixed, B: pos
        let a = [-1.0, -6.0, 3.0, -8.0, 0.0, 3.0];
        let b = [4.0];
        let expected = [-4.0, -24.0, 12.0, -32.0, 0.0, 12.0];
        check(&a, &b, (6, 1, 1), &expected);
    }

    #[test]
    fn ones_n_k_m() {
        // A: neg, B: neg
        let a = [-3.0];
        let b = [-4.0];
        let expected = [12.0];
        check(&a, &b, (1, 1, 1), &expected);
    }

    #[test]
    fn n_gt_k_gt_m() {
        // A: pos, B: mixed
        let a = [
            8.0, 9.0, 7.0, 1.0, 7.0, 9.0, 4.0, 5.0, 1.0, 3.0, 3.0, 9.0, 8.0, 2.0, 6.0, 9.0, 7.0,
            3.0, 2.0, 9.0, 2.0, 3.0, 2.0, 9.0, 2.0, 9.0, 8.0, 9.0, 3.0, 8.0,
        ];
        let b = [6.0, -8.0, 0.0, -2.0, 6.0, -4.0, 5.0, -9.0, -2.0, -3.0];
        let expected = [
            81.0, -140.0, 83.0, -118.0, 64.0, -110.0, 64.0, -143.0, 65.0, -117.0, 107.0, -175.0,
        ];
        check(&a, &b, (6, 5, 2), &expected);
    }

    #[test]
    fn n_gt_m_gt_k() {
        // A: mixed, B: neg
        let a = [-9.0, -1.0, -2.0, -6.0, 0.0, 8.0, 1.0, 8.0];
        let b = [-1.0, -3.0, -7.0, -8.0, -1.0, -4.0];
        let expected = [
            17.0, 28.0, 67.0, 50.0, 12.0, 38.0, -64.0, -8.0, -32.0, -65.0, -11.0, -39.0,
        ];
        check(&a, &b, (4, 2, 3), &expected);
    }

    #[test]
    fn k_gt_n_gt_m() {
        // A: pos, B: pos
        let a = [
            1.0, 7.0, 6.0, 7.0, 8.0, 9.0, 6.0, 1.0, 9.0, 8.0, 2.0, 4.0, 3.0, 7.0, 7.0, 5.0, 1.0,
            8.0, 1.0, 4.0,
        ];
        let b = [5.0, 5.0, 3.0, 4.0, 3.0, 1.0, 5.0, 8.0, 2.0, 4.0];
        let expected = [95.0, 127.0, 127.0, 174.0, 80.0, 113.0, 65.0, 61.0];
        check(&a, &b, (4, 5, 2), &expected);
    }

    #[test]
    fn k_gt_m_gt_n() {
        // A: neg, B: neg
        let a = [
            -5.0, -5.0, -2.0, -4.0, -8.0, -1.0, -5.0, -8.0, -3.0, -3.0, -8.0, -2.0, -4.0, -9.0,
            -2.0, -8.0, -8.0, -1.0,
        ];
        let b = [
            -9.0, -2.0, -8.0, -1.0, -9.0, -1.0, -1.0, -4.0, -2.0, -7.0, -1.0, -9.0, -2.0, -4.0,
            -5.0, -9.0, -1.0, -6.0, -2.0, -2.0, -2.0, -3.0, -3.0, -3.0,
        ];
        let expected = [
            112.0, 96.0, 86.0, 98.0, 141.0, 105.0, 88.0, 113.0, 147.0, 114.0, 102.0, 149.0,
        ];
        check(&a, &b, (3, 6, 4), &expected);
    }

    #[test]
    fn m_gt_n_gt_k() {
        // A: pos, B: neg
        let a = [8.0, 8.0, 6.0, 3.0, 7.0, 3.0];
        let b = [-8.0, -5.0, -1.0, -7.0, -5.0, -9.0, -6.0, -2.0, -2.0, -6.0];
        let expected = [
            -136.0, -88.0, -24.0, -72.0, -88.0, -75.0, -48.0, -12.0, -48.0, -48.0, -83.0, -53.0,
            -13.0, -55.0, -53.0,
        ];
        check(&a, &b, (3, 2, 5), &expected);
    }

    #[test]
    fn m_gt_k_gt_n() {
        // A: neg, B: pos
        let a = [-2.0, -5.0, -8.0, -1.0, -4.0, -9.0, -4.0, -6.0];
        let b = [
            8.0, 8.0, 2.0, 8.0, 2.0, 2.0, 2.0, 8.0, 1.0, 3.0, 8.0, 4.0, 6.0, 3.0, 4.0, 4.0, 1.0,
            4.0, 2.0, 4.0, 6.0, 6.0, 6.0, 3.0,
        ];
        let expected = [
            -76.0, -84.0, -47.0, -69.0, -58.0, -59.0, -86.0, -140.0, -69.0, -111.0, -120.0, -78.0,
        ];
        check(&a, &b, (2, 4, 6), &expected);
    }

    #[test]
    fn zero_left() {
        // A: zero, B: mixed
        let a = [0.0; 12];
        let b = [4.0, -4.0, 2.0, 0.0, 9.0, 5.0, 2.0, -2.0];
        let expected = [0.0; 6];
        check(&a, &b, (3, 4, 2), &expected);
    }

    #[test]
    fn zero_right() {
        // A: mixed, B: zero
        let a = [
            -1.0, -7.0, 5.0, -2.0, -9.0, 5.0, -7.0, 0.0, -4.0, -4.0, 6.0, 5.0,
        ];
        let b = [0.0; 8];
        let expected = [0.0; 6];
        check(&a, &b, (3, 4, 2), &expected);
    }

    #[test]
    fn zero_both() {
        // A: zero, B: zero
        let a = [0.0; 12];
        let b = [0.0; 8];
        let expected = [0.0; 6];
        check(&a, &b, (3, 4, 2), &expected);
    }
}
