/**
 * @file
 * @brief `wmma_from_vectorized.cu` defines `matmul_wmma_from_vectorized`: `vectorized` with the
 * multiply-adds on tensor cores, through the WMMA API.
 *
 * @par Interface
 * - `matmul_wmma_from_vectorized`: Launches `matmul_wmma_from_vectorized_kernel`.
 *
 * @par Example
 * @code
 * matmul_wmma_from_vectorized(A_device, B_device, C_device, M, N, K);
 * @endcode
 */
#include <mma.h>

#include <cstdint>

#include "common/matmul.h"

namespace {

namespace wmma = nvcuda::wmma;

constexpr int BLOCK_TILE_M = 128;
constexpr int BLOCK_TILE_N = 128;
constexpr int BLOCK_TILE_K = 32;
constexpr int WARP_TILE_M = 64;
constexpr int WARP_TILE_N = 32;
constexpr int MMA_SIZE = 16;
constexpr int MMA_TILES_PER_WARP_M = WARP_TILE_M / MMA_SIZE;
constexpr int MMA_TILES_PER_WARP_N = WARP_TILE_N / MMA_SIZE;
constexpr int WARPS_PER_BLOCK_N = BLOCK_TILE_N / WARP_TILE_N;
constexpr int THREADS_PER_WARP = 32;
constexpr int THREAD_COUNT = (BLOCK_TILE_M / WARP_TILE_M) * WARPS_PER_BLOCK_N * THREADS_PER_WARP;
constexpr int BF16_PER_VECTOR = 8;
constexpr int PADDING_BF16 = 8;
constexpr int A_TILE_STRIDE = BLOCK_TILE_K + PADDING_BF16;
constexpr int B_TILE_STRIDE = BLOCK_TILE_N + PADDING_BF16;
constexpr int A_VECTORS_PER_ROW = BLOCK_TILE_K / BF16_PER_VECTOR;
constexpr int B_VECTORS_PER_ROW = BLOCK_TILE_N / BF16_PER_VECTOR;
constexpr int WMMA_ALIGNMENT_BYTES = 32;

using vector_t = uint4;
using A_fragment_t =
    wmma::fragment<wmma::matrix_a, MMA_SIZE, MMA_SIZE, MMA_SIZE, bf16_t, wmma::row_major>;
using B_fragment_t =
    wmma::fragment<wmma::matrix_b, MMA_SIZE, MMA_SIZE, MMA_SIZE, bf16_t, wmma::row_major>;
using C_fragment_t = wmma::fragment<wmma::accumulator, MMA_SIZE, MMA_SIZE, MMA_SIZE, float>;
using C_warp_tile_t = C_fragment_t[MMA_TILES_PER_WARP_M][MMA_TILES_PER_WARP_N];

/**
 * @brief `load_tiles` loads A's 128 x 32 and B's 32 x 128 tile in 16-byte vectors, 2 of each per
 * thread, zero-filling past the edges.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory.
 * @param[in] B_device Right factor, K x N, row-major, in device memory.
 * @param[out] A_tile A's tile, row-major with `A_TILE_STRIDE`, in shared memory.
 * @param[out] B_tile B's tile, row-major with `B_TILE_STRIDE`, in shared memory.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C, a multiple of 8.
 * @param[in] K Columns of A and rows of B, a multiple of 8.
 * @param[in] block_i Row of C where the block's tile starts.
 * @param[in] block_j Column of C where the block's tile starts.
 * @param[in] k_tile_start Column of A and row of B where the step starts.
 *
 * @par Example
 * @code
 * load_tiles(A_device, B_device, A_tile, B_tile, M, N, K, block_i, block_j, k_tile_start);
 * @endcode
 */
__device__ __forceinline__ void load_tiles(const bf16_t* A_device, const bf16_t* B_device,
                                           bf16_t* A_tile, bf16_t* B_tile, int M, int N, int K,
                                           int64_t block_i, int64_t block_j, int64_t k_tile_start) {
  const vector_t zero_vector = {0, 0, 0, 0};
  for (int v = static_cast<int>(threadIdx.x); v < BLOCK_TILE_M * A_VECTORS_PER_ROW;
       v += THREAD_COUNT) {
    const int row = v / A_VECTORS_PER_ROW;
    const int col = (v % A_VECTORS_PER_ROW) * BF16_PER_VECTOR;
    const int64_t A_i = block_i + row;
    const int64_t A_k = k_tile_start + col;
    *reinterpret_cast<vector_t*>(&A_tile[(row * A_TILE_STRIDE) + col]) =
        (A_i < M && A_k < K) ? *reinterpret_cast<const vector_t*>(&A_device[(A_i * K) + A_k])
                             : zero_vector;
  }
  for (int v = static_cast<int>(threadIdx.x); v < BLOCK_TILE_K * B_VECTORS_PER_ROW;
       v += THREAD_COUNT) {
    const int row = v / B_VECTORS_PER_ROW;
    const int col = (v % B_VECTORS_PER_ROW) * BF16_PER_VECTOR;
    const int64_t B_k = k_tile_start + row;
    const int64_t B_j = block_j + col;
    *reinterpret_cast<vector_t*>(&B_tile[(row * B_TILE_STRIDE) + col]) =
        (B_k < K && B_j < N) ? *reinterpret_cast<const vector_t*>(&B_device[(B_k * N) + B_j])
                             : zero_vector;
  }
}

/**
 * @brief `add_tile_products` adds one step's products to a warp's 64 x 32 tile of C.
 *
 * @param[in] A_tile A's tile, in shared memory.
 * @param[in] B_tile B's tile, in shared memory.
 * @param[in] warp_row_start Row of the block's tile where the warp's tile starts.
 * @param[in] warp_col_start Column of the block's tile where the warp's tile starts.
 * @param[in,out] C_fragments The warp's 4 x 2 accumulator fragments, 16 x 16 each.
 *
 * @par Example
 * @code
 * add_tile_products(A_tile, B_tile, warp_row_start, warp_col_start, C_fragments);
 * @endcode
 *
 * @par Must know
 * - Performance:
 *   - Per 16 of K, it loads 4 fragments of A and 2 of B for 8 tensor-core multiply-adds.
 */
__device__ __forceinline__ void add_tile_products(const bf16_t* A_tile, const bf16_t* B_tile,
                                                  int warp_row_start, int warp_col_start,
                                                  C_warp_tile_t& C_fragments) {
  A_fragment_t A_fragments[MMA_TILES_PER_WARP_M];
  B_fragment_t B_fragments[MMA_TILES_PER_WARP_N];
  for (int k = 0; k < BLOCK_TILE_K; k += MMA_SIZE) {
    for (int m = 0; m < MMA_TILES_PER_WARP_M; ++m) {
      const int row = warp_row_start + (m * MMA_SIZE);
      wmma::load_matrix_sync(A_fragments[m], &A_tile[(row * A_TILE_STRIDE) + k], A_TILE_STRIDE);
    }
    for (int n = 0; n < MMA_TILES_PER_WARP_N; ++n) {
      const int col = warp_col_start + (n * MMA_SIZE);
      wmma::load_matrix_sync(B_fragments[n], &B_tile[(k * B_TILE_STRIDE) + col], B_TILE_STRIDE);
    }
    for (int m = 0; m < MMA_TILES_PER_WARP_M; ++m) {
      for (int n = 0; n < MMA_TILES_PER_WARP_N; ++n) {
        wmma::mma_sync(C_fragments[m][n], A_fragments[m], B_fragments[n], C_fragments[m][n]);
      }
    }
  }
}

/**
 * @brief `store_warp_tile` writes a warp's 64 x 32 tile of C as bf16, skipping elements outside C.
 *
 * @param[in] C_fragments The warp's accumulator fragments.
 * @param[out] scratch The warp's 16 x 16 fp32 scratch tile, in shared memory.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of C.
 * @param[in] N Columns of C.
 * @param[in] i_start Row of C where the warp's tile starts.
 * @param[in] j_start Column of C where the warp's tile starts.
 *
 * @par Example
 * @code
 * store_warp_tile(C_fragments, scratch, C_device, M, N, i_start, j_start);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - A fragment's element order is opaque, so each one goes through `scratch` to reach C.
 */
__device__ __forceinline__ void store_warp_tile(const C_warp_tile_t& C_fragments, float* scratch,
                                                bf16_t* C_device, int M, int N, int64_t i_start,
                                                int64_t j_start) {
  const int lane = static_cast<int>(threadIdx.x) % THREADS_PER_WARP;
  for (int m = 0; m < MMA_TILES_PER_WARP_M; ++m) {
    for (int n = 0; n < MMA_TILES_PER_WARP_N; ++n) {
      wmma::store_matrix_sync(scratch, C_fragments[m][n], MMA_SIZE, wmma::mem_row_major);
      // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN LANES READ UNWRITTEN SCRATCH
      __syncwarp();
      for (int e = lane; e < MMA_SIZE * MMA_SIZE; e += THREADS_PER_WARP) {
        const int64_t i = i_start + static_cast<int64_t>((m * MMA_SIZE) + (e / MMA_SIZE));
        const int64_t j = j_start + static_cast<int64_t>((n * MMA_SIZE) + (e % MMA_SIZE));
        if (i < M && j < N) {
          C_device[(i * N) + j] = __float2bfloat16(scratch[e]);
        }
      }
      // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN LANES OVERWRITE UNREAD SCRATCH
      __syncwarp();
    }
  }
}

/**
 * @brief `matmul_wmma_from_vectorized_kernel` computes a 128 x 128 tile of C on tensor cores.
 *
 * Each of a block's 8 warps computes a 64 x 32 piece of the tile as 4 x 2 WMMA fragments of
 * 16 x 16, accumulating in fp32. For each step of 32 along K, the threads load A's and B's tiles
 * in 16-byte vectors and the block syncs; then each warp loads its fragments from the tiles and
 * multiplies them on the tensor cores.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory.
 * @param[in] B_device Right factor, K x N, row-major, in device memory.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C, a multiple of 8.
 * @param[in] K Columns of A and rows of B, a multiple of 8.
 *
 * @par Example
 * @code
 * matmul_wmma_from_vectorized_kernel<<<grid, THREAD_COUNT>>>(A_device, B_device, C_device, M, N,
 *                                                            K);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - WMMA loads need 32-byte-aligned addresses; the padded strides keep every fragment aligned.
 * - Performance:
 *   - The tile rows are padded by 8 bf16, so a fragment's rows don't start in the same banks.
 */
__global__ void __launch_bounds__(THREAD_COUNT)
    matmul_wmma_from_vectorized_kernel(const bf16_t* A_device, const bf16_t* B_device,
                                       bf16_t* C_device, int M, int N, int K) {
  __shared__ alignas(WMMA_ALIGNMENT_BYTES) bf16_t A_tile[BLOCK_TILE_M * A_TILE_STRIDE];
  __shared__ alignas(WMMA_ALIGNMENT_BYTES) bf16_t B_tile[BLOCK_TILE_K * B_TILE_STRIDE];
  __shared__ alignas(
      WMMA_ALIGNMENT_BYTES) float scratch[THREAD_COUNT / THREADS_PER_WARP][MMA_SIZE * MMA_SIZE];
  const int warp = static_cast<int>(threadIdx.x) / THREADS_PER_WARP;
  const int warp_row_start = (warp / WARPS_PER_BLOCK_N) * WARP_TILE_M;
  const int warp_col_start = (warp % WARPS_PER_BLOCK_N) * WARP_TILE_N;
  const int64_t block_i = static_cast<int64_t>(blockIdx.y) * BLOCK_TILE_M;
  const int64_t block_j = static_cast<int64_t>(blockIdx.x) * BLOCK_TILE_N;
  C_warp_tile_t C_fragments;
  for (int m = 0; m < MMA_TILES_PER_WARP_M; ++m) {
    for (int n = 0; n < MMA_TILES_PER_WARP_N; ++n) {
      wmma::fill_fragment(C_fragments[m][n], 0.0F);
    }
  }
  for (int64_t k_tile_start = 0; k_tile_start < K; k_tile_start += BLOCK_TILE_K) {
    load_tiles(A_device, B_device, A_tile, B_tile, M, N, K, block_i, block_j, k_tile_start);
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A SHARED-MEMORY RACE WILL HAPPEN
    __syncthreads();
    add_tile_products(A_tile, B_tile, warp_row_start, warp_col_start, C_fragments);
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A SHARED-MEMORY RACE WILL HAPPEN
    __syncthreads();
  }
  store_warp_tile(C_fragments, scratch[warp], C_device, M, N, block_i + warp_row_start,
                  block_j + warp_col_start);
}

}  // namespace

/**
 * @brief `matmul_wmma_from_vectorized` launches a 256-thread block per 128 x 128 tile of C.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory, 16-byte aligned.
 * @param[in] B_device Right factor, K x N, row-major, in device memory, 16-byte aligned.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C, a multiple of 8.
 * @param[in] K Columns of A and rows of B, a multiple of 8.
 *
 * @par Example
 * @code
 * matmul_wmma_from_vectorized(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - N and K must be multiples of 8, so that every row starts on a 16-byte boundary.
 * - Gotchas:
 *   - The launch is asynchronous; `cudaGetLastError` reports a failed launch.
 */
void matmul_wmma_from_vectorized(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device,
                                 int M, int N, int K) {
  const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
  matmul_wmma_from_vectorized_kernel<<<grid, THREAD_COUNT>>>(A_device, B_device, C_device, M, N, K);
}
