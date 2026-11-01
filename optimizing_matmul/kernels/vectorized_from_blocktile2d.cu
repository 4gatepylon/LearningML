/**
 * @file
 * @brief `vectorized_from_blocktile2d.cu` defines `matmul_vectorized_from_blocktile2d`:
 * `blocktile2d` with 16-byte loads and stores and A's tile stored transposed.
 *
 * @par Interface
 * - `matmul_vectorized_from_blocktile2d`: Launches `matmul_vectorized_from_blocktile2d_kernel`.
 *
 * @par Example
 * @code
 * matmul_vectorized_from_blocktile2d(A_device, B_device, C_device, M, N, K);
 * @endcode
 */
#include <cstdint>

#include "common/matmul.h"

namespace {

constexpr int BLOCK_TILE_M = 128;
constexpr int BLOCK_TILE_N = 128;
constexpr int BLOCK_TILE_K = 16;
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;
constexpr int THREAD_COUNT = (BLOCK_TILE_M / THREAD_TILE_M) * (BLOCK_TILE_N / THREAD_TILE_N);
constexpr int BF16_PER_VECTOR = 8;
constexpr int A_VECTORS_PER_ROW = BLOCK_TILE_K / BF16_PER_VECTOR;
constexpr int B_VECTORS_PER_ROW = BLOCK_TILE_N / BF16_PER_VECTOR;

static_assert(BLOCK_TILE_M * A_VECTORS_PER_ROW == THREAD_COUNT, "each thread loads 1 vector of A");
static_assert(BLOCK_TILE_K * B_VECTORS_PER_ROW == THREAD_COUNT, "each thread loads 1 vector of B");
static_assert(THREAD_TILE_M == BF16_PER_VECTOR && THREAD_TILE_N == BF16_PER_VECTOR,
              "a thread's 8 rows of A and 8 columns of B are one vector each");

using vector_t = uint4;
using A_tile_t = bf16_t[BLOCK_TILE_K][BLOCK_TILE_M];
using B_tile_t = bf16_t[BLOCK_TILE_K][BLOCK_TILE_N];
using C_thread_tile_t = float[THREAD_TILE_M][THREAD_TILE_N];

/**
 * @brief `load_tiles` loads one 16-byte vector of A and one of B, storing A's transposed.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory.
 * @param[in] B_device Right factor, K x N, row-major, in device memory.
 * @param[out] A_tile A's 128 x 16 tile, transposed to 16 x 128, in shared memory.
 * @param[out] B_tile B's 16 x 128 tile, in shared memory.
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
 *
 * @par Must know
 * - Gotchas:
 *   - With N and K multiples of 8, a vector is wholly inside or wholly outside A or B.
 */
__device__ __forceinline__ void load_tiles(const bf16_t* A_device, const bf16_t* B_device,
                                           A_tile_t& A_tile, B_tile_t& B_tile, int M, int N, int K,
                                           int64_t block_i, int64_t block_j, int64_t k_tile_start) {
  const int thread_index = static_cast<int>(threadIdx.x);
  const vector_t zero_vector = {0, 0, 0, 0};
  const int A_tile_row = thread_index / A_VECTORS_PER_ROW;
  const int A_tile_col_start = (thread_index % A_VECTORS_PER_ROW) * BF16_PER_VECTOR;
  const int64_t A_i = block_i + A_tile_row;
  const int64_t A_k = k_tile_start + A_tile_col_start;
  alignas(sizeof(vector_t)) bf16_t A_values[BF16_PER_VECTOR];
  *reinterpret_cast<vector_t*>(A_values) =
      (A_i < M && A_k < K) ? *reinterpret_cast<const vector_t*>(&A_device[(A_i * K) + A_k])
                           : zero_vector;
  for (int e = 0; e < BF16_PER_VECTOR; ++e) {
    A_tile[A_tile_col_start + e][A_tile_row] = A_values[e];
  }
  const int B_tile_row = thread_index / B_VECTORS_PER_ROW;
  const int B_tile_col_start = (thread_index % B_VECTORS_PER_ROW) * BF16_PER_VECTOR;
  const int64_t B_k = k_tile_start + B_tile_row;
  const int64_t B_j = block_j + B_tile_col_start;
  *reinterpret_cast<vector_t*>(&B_tile[B_tile_row][B_tile_col_start]) =
      (B_k < K && B_j < N) ? *reinterpret_cast<const vector_t*>(&B_device[(B_k * N) + B_j])
                           : zero_vector;
}

/**
 * @brief `add_tile_products` adds one step's products to a thread's 8 x 8 block of C.
 *
 * @param[in] A_tile A's tile, transposed, in shared memory.
 * @param[in] B_tile B's tile, in shared memory.
 * @param[in] C_tile_row_start Row of the block's tile where the thread's block starts.
 * @param[in] C_tile_col_start Column of the block's tile where the thread's block starts.
 * @param[in,out] C_fp32 The thread's 8 x 8 accumulators.
 *
 * @par Example
 * @code
 * add_tile_products(A_tile, B_tile, C_tile_row_start, C_tile_col_start, C_fp32);
 * @endcode
 *
 * @par Must know
 * - Performance:
 *   - Per k, it reads one 16-byte vector of A's tile and one of B's for 64 multiply-adds.
 */
__device__ __forceinline__ void add_tile_products(const A_tile_t& A_tile, const B_tile_t& B_tile,
                                                  int C_tile_row_start, int C_tile_col_start,
                                                  C_thread_tile_t& C_fp32) {
  alignas(sizeof(vector_t)) bf16_t A_values[THREAD_TILE_M];
  alignas(sizeof(vector_t)) bf16_t B_values[THREAD_TILE_N];
  for (int k = 0; k < BLOCK_TILE_K; ++k) {
    *reinterpret_cast<vector_t*>(A_values) =
        *reinterpret_cast<const vector_t*>(&A_tile[k][C_tile_row_start]);
    *reinterpret_cast<vector_t*>(B_values) =
        *reinterpret_cast<const vector_t*>(&B_tile[k][C_tile_col_start]);
    for (int r = 0; r < THREAD_TILE_M; ++r) {
      const float A_fp32 = __bfloat162float(A_values[r]);
      for (int c = 0; c < THREAD_TILE_N; ++c) {
        C_fp32[r][c] += A_fp32 * __bfloat162float(B_values[c]);
      }
    }
  }
}

/**
 * @brief `store_thread_tile` writes a thread's 8 x 8 block of C, one 16-byte vector per row.
 *
 * @param[in] C_fp32 The thread's 8 x 8 accumulators.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of C.
 * @param[in] N Columns of C, a multiple of 8.
 * @param[in] i_start Row of C where the thread's block starts.
 * @param[in] j_start Column of C where the thread's block starts, a multiple of 8.
 *
 * @par Example
 * @code
 * store_thread_tile(C_fp32, C_device, M, N, i_start, j_start);
 * @endcode
 */
__device__ __forceinline__ void store_thread_tile(const C_thread_tile_t& C_fp32, bf16_t* C_device,
                                                  int M, int N, int64_t i_start, int64_t j_start) {
  if (j_start >= N) {
    return;
  }
  alignas(sizeof(vector_t)) bf16_t C_values[THREAD_TILE_N];
  for (int r = 0; r < THREAD_TILE_M && i_start + r < M; ++r) {
    for (int c = 0; c < THREAD_TILE_N; ++c) {
      C_values[c] = __float2bfloat16(C_fp32[r][c]);
    }
    *reinterpret_cast<vector_t*>(&C_device[((i_start + r) * N) + j_start]) =
        *reinterpret_cast<const vector_t*>(C_values);
  }
}

/**
 * @brief `matmul_vectorized_from_blocktile2d_kernel` computes an 8 x 8 block of C per thread.
 *
 * As in `blocktile2d`, each of a block's 256 threads computes an 8 x 8 block of a 128 x 128 tile
 * of C. Each step along K is 16 wide, so each thread loads exactly one 16-byte vector of A and one
 * of B. A's tile is stored transposed, so a thread's 8 rows of it are contiguous for each k.
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
 * matmul_vectorized_from_blocktile2d_kernel<<<grid, THREAD_COUNT>>>(A_device, B_device,
 *                                                                   C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Every thread loads and syncs, even outside C, so no thread may return before the last sync.
 */
__global__ void __launch_bounds__(THREAD_COUNT)
    matmul_vectorized_from_blocktile2d_kernel(const bf16_t* A_device, const bf16_t* B_device,
                                              bf16_t* C_device, int M, int N, int K) {
  __shared__ alignas(sizeof(vector_t)) A_tile_t A_tile;
  __shared__ alignas(sizeof(vector_t)) B_tile_t B_tile;
  const int thread_index = static_cast<int>(threadIdx.x);
  const int C_tile_row_start = (thread_index / (BLOCK_TILE_N / THREAD_TILE_N)) * THREAD_TILE_M;
  const int C_tile_col_start = (thread_index % (BLOCK_TILE_N / THREAD_TILE_N)) * THREAD_TILE_N;
  const int64_t block_i = static_cast<int64_t>(blockIdx.y) * BLOCK_TILE_M;
  const int64_t block_j = static_cast<int64_t>(blockIdx.x) * BLOCK_TILE_N;
  float C_fp32[THREAD_TILE_M][THREAD_TILE_N] = {};
  for (int64_t k_tile_start = 0; k_tile_start < K; k_tile_start += BLOCK_TILE_K) {
    load_tiles(A_device, B_device, A_tile, B_tile, M, N, K, block_i, block_j, k_tile_start);
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A SHARED-MEMORY RACE WILL HAPPEN
    __syncthreads();
    add_tile_products(A_tile, B_tile, C_tile_row_start, C_tile_col_start, C_fp32);
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN A SHARED-MEMORY RACE WILL HAPPEN
    __syncthreads();
  }
  store_thread_tile(C_fp32, C_device, M, N, block_i + C_tile_row_start, block_j + C_tile_col_start);
}

}  // namespace

/**
 * @brief `matmul_vectorized_from_blocktile2d` launches a 256-thread block per 128 x 128 tile.
 *
 * @param[in] A_device Left factor, M x K, row-major, in device memory, 16-byte aligned.
 * @param[in] B_device Right factor, K x N, row-major, in device memory, 16-byte aligned.
 * @param[out] C_device Product, M x N, row-major, in device memory, 16-byte aligned.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C, a multiple of 8.
 * @param[in] K Columns of A and rows of B, a multiple of 8.
 *
 * @par Example
 * @code
 * matmul_vectorized_from_blocktile2d(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - N and K must be multiples of 8, so that every row starts on a 16-byte boundary.
 * - Gotchas:
 *   - The launch is asynchronous; `cudaGetLastError` reports a failed launch.
 */
void matmul_vectorized_from_blocktile2d(const bf16_t* A_device, const bf16_t* B_device,
                                        bf16_t* C_device, int M, int N, int K) {
  const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
  matmul_vectorized_from_blocktile2d_kernel<<<grid, THREAD_COUNT>>>(A_device, B_device, C_device, M,
                                                                    N, K);
}
