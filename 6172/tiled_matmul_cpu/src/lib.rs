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
    use super::*;

    #[test]
    fn square_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0; 4];
        matmul_naive(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn identity_is_neutral() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = [0.0; 9];
        matmul_naive(&a, &id, &mut c, 3, 3, 3);
        assert_eq!(c, a);
        matmul_naive(&id, &a, &mut c, 3, 3, 3);
        assert_eq!(c, a);
    }

    #[test]
    fn rectangular_2x3_times_3x2() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0; 4];
        matmul_naive(&a, &b, &mut c, 2, 3, 2);
        assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut c = [0.0; 1];
        matmul_naive(&a, &b, &mut c, 1, 3, 1);
        assert_eq!(c, [32.0]);
    }

    #[test]
    fn overwrites_output() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let id = [1.0, 0.0, 0.0, 1.0];
        let mut c = [99.0; 4];
        matmul_naive(&a, &id, &mut c, 2, 2, 2);
        assert_eq!(c, a);
    }
}
