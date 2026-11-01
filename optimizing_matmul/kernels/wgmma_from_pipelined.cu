/**
 * @file
 * @brief `wgmma_from_pipelined.cu` defines `matmul_wgmma_from_pipelined`: `pipelined` with TMA
 * copies and wgmma, Hopper's warpgroup-wide tensor-core instructions that read shared memory.
 *
 * @par Interface
 * - `matmul_wgmma_from_pipelined`: Launches `matmul_wgmma_from_pipelined_kernel`.
 * - `WGMMA_FROM_PIPELINED_BLOCK_TILE_N`: Option: the block tile's width, 128 or 256.
 * - `WGMMA_FROM_PIPELINED_STAGE_COUNT`: Option: the K steps in shared memory at once, at least 2.
 *
 * @par Example
 * @code
 * matmul_wgmma_from_pipelined(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - It needs `NVCC_ARCH=sm_90a` (`.env`); built for another architecture, the kernel traps.
 */
#include <cuda.h>

#include <cstdint>

#include "common/matmul.h"

#ifndef WGMMA_FROM_PIPELINED_BLOCK_TILE_N
#define WGMMA_FROM_PIPELINED_BLOCK_TILE_N 256
#endif
#ifndef WGMMA_FROM_PIPELINED_STAGE_COUNT
#define WGMMA_FROM_PIPELINED_STAGE_COUNT 2
#endif

namespace {

constexpr int BLOCK_TILE_M = 128;
constexpr int BLOCK_TILE_N = WGMMA_FROM_PIPELINED_BLOCK_TILE_N;
constexpr int STAGE_COUNT = WGMMA_FROM_PIPELINED_STAGE_COUNT;
constexpr int SWIZZLE_BYTES = 128;
constexpr int BF16_PER_SWIZZLE_ROW = SWIZZLE_BYTES / static_cast<int>(sizeof(bf16_t));
constexpr int BLOCK_TILE_K = BF16_PER_SWIZZLE_ROW;
constexpr int SWIZZLE_ATOM_BYTES = 8 * SWIZZLE_BYTES;
constexpr int WGMMA_M = 64;
constexpr int WGMMA_N = 128;
constexpr int THREADS_PER_WARPGROUP = 128;
constexpr int WARPGROUP_COUNT = BLOCK_TILE_M / WGMMA_M;
constexpr int THREAD_COUNT = WARPGROUP_COUNT * THREADS_PER_WARPGROUP;
constexpr int A_TILE_BYTES = BLOCK_TILE_M * SWIZZLE_BYTES;
constexpr int B_BOX_BYTES = BLOCK_TILE_K * SWIZZLE_BYTES;
constexpr int B_BOX_COUNT = BLOCK_TILE_N / BF16_PER_SWIZZLE_ROW;
constexpr int STAGE_BYTES = A_TILE_BYTES + (B_BOX_COUNT * B_BOX_BYTES);
constexpr int SHARED_BYTES = (STAGE_COUNT * STAGE_BYTES) + SWIZZLE_ATOM_BYTES;
constexpr int MAX_SHARED_BYTES_PER_BLOCK_ACCORDING_TO_CUDA_GUIDE_FOR_SM90 = 227 * 1024;

static_assert(BLOCK_TILE_N % WGMMA_N == 0, "the block tile is a whole number of wgmma tiles");
static_assert(STAGE_COUNT >= 2, "a pipeline needs a stage to load while another computes");
static_assert(SHARED_BYTES <= MAX_SHARED_BYTES_PER_BLOCK_ACCORDING_TO_CUDA_GUIDE_FOR_SM90,
              "the stages must fit in one block's shared memory");
static_assert(STAGE_BYTES % SWIZZLE_ATOM_BYTES == 0, "every tile starts on a swizzle atom");

#if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH_FEAT_SM90_ALL)

constexpr int WGMMA_K = 16;
constexpr int WGMMAS_PER_ROW = BLOCK_TILE_N / WGMMA_N;
constexpr int THREADS_PER_WARP = 32;
constexpr int ACCUMULATOR_COUNT = WGMMA_M * WGMMA_N / THREADS_PER_WARPGROUP;
constexpr uint64_t SWIZZLE_128B_LAYOUT_TYPE = 1;
constexpr uint32_t DESCRIPTOR_ADDRESS_SHIFT = 4;
constexpr uint32_t DESCRIPTOR_LEADING_OFFSET_BIT = 16;
constexpr uint32_t DESCRIPTOR_STRIDE_OFFSET_BIT = 32;
constexpr uint32_t DESCRIPTOR_LAYOUT_TYPE_BIT = 62;
constexpr uint32_t DESCRIPTOR_ADDRESS_MASK = (1U << 18U) - 1U;

/**
 * @brief `get_shared_address` returns a pointer's address in the shared-memory window.
 *
 * @param[in] pointer A pointer into shared memory.
 * @return Its 32-bit shared-memory address, as PTX instructions take it.
 *
 * @par Example
 * @code
 * const uint32_t barrier_address = get_shared_address(&full_barriers[stage]);
 * @endcode
 */
__device__ __forceinline__ uint32_t get_shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

/**
 * @brief `make_descriptor` builds a wgmma shared-memory descriptor for a 128-byte-swizzled tile.
 *
 * @param[in] start_address Shared-memory address of the tile's first element.
 * @param[in] leading_bytes Bytes between 64-element groups along M or N (unused for K-major).
 * @param[in] stride_bytes Bytes between groups of 8 rows.
 * @return The 64-bit descriptor: address, offsets (each in 16-byte units) and the swizzle mode.
 *
 * @par Example
 * @code
 * const uint64_t A_descriptor = make_descriptor(A_address, 16, 1024);
 * @endcode
 */
__device__ __forceinline__ uint64_t make_descriptor(uint32_t start_address, uint32_t leading_bytes,
                                                    uint32_t stride_bytes) {
  return ((start_address & DESCRIPTOR_ADDRESS_MASK) >> DESCRIPTOR_ADDRESS_SHIFT) |
         (static_cast<uint64_t>(leading_bytes >> DESCRIPTOR_ADDRESS_SHIFT)
          << DESCRIPTOR_LEADING_OFFSET_BIT) |
         (static_cast<uint64_t>(stride_bytes >> DESCRIPTOR_ADDRESS_SHIFT)
          << DESCRIPTOR_STRIDE_OFFSET_BIT) |
         (SWIZZLE_128B_LAYOUT_TYPE << DESCRIPTOR_LAYOUT_TYPE_BIT);
}

/**
 * @brief `wgmma_64x128x16` adds A (64 x 16, K-major) times B (16 x 128, N-major) to 64
 * accumulators per thread of the warpgroup.
 *
 * @param[in,out] C_fp32 This thread's 64 accumulators of the 64 x 128 tile.
 * @param[in] A_descriptor Descriptor of A's 64 x 16 slice.
 * @param[in] B_descriptor Descriptor of B's 16 x 128 slice, read transposed (N-major).
 *
 * @par Example
 * @code
 * wgmma_64x128x16(C_fp32, A_descriptor, B_descriptor);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - It only starts the multiply; `wgmma.commit_group` and `wgmma.wait_group` finish it.
 */
__device__ __forceinline__ void wgmma_64x128x16(float (&C_fp32)[ACCUMULATOR_COUNT],
                                                uint64_t A_descriptor, uint64_t B_descriptor) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "%64, %65, 1, 1, 1, 0, 1;\n"
      "}\n"
      : "+f"(C_fp32[0]), "+f"(C_fp32[1]), "+f"(C_fp32[2]), "+f"(C_fp32[3]), "+f"(C_fp32[4]),
        "+f"(C_fp32[5]), "+f"(C_fp32[6]), "+f"(C_fp32[7]), "+f"(C_fp32[8]), "+f"(C_fp32[9]),
        "+f"(C_fp32[10]), "+f"(C_fp32[11]), "+f"(C_fp32[12]), "+f"(C_fp32[13]), "+f"(C_fp32[14]),
        "+f"(C_fp32[15]), "+f"(C_fp32[16]), "+f"(C_fp32[17]), "+f"(C_fp32[18]), "+f"(C_fp32[19]),
        "+f"(C_fp32[20]), "+f"(C_fp32[21]), "+f"(C_fp32[22]), "+f"(C_fp32[23]), "+f"(C_fp32[24]),
        "+f"(C_fp32[25]), "+f"(C_fp32[26]), "+f"(C_fp32[27]), "+f"(C_fp32[28]), "+f"(C_fp32[29]),
        "+f"(C_fp32[30]), "+f"(C_fp32[31]), "+f"(C_fp32[32]), "+f"(C_fp32[33]), "+f"(C_fp32[34]),
        "+f"(C_fp32[35]), "+f"(C_fp32[36]), "+f"(C_fp32[37]), "+f"(C_fp32[38]), "+f"(C_fp32[39]),
        "+f"(C_fp32[40]), "+f"(C_fp32[41]), "+f"(C_fp32[42]), "+f"(C_fp32[43]), "+f"(C_fp32[44]),
        "+f"(C_fp32[45]), "+f"(C_fp32[46]), "+f"(C_fp32[47]), "+f"(C_fp32[48]), "+f"(C_fp32[49]),
        "+f"(C_fp32[50]), "+f"(C_fp32[51]), "+f"(C_fp32[52]), "+f"(C_fp32[53]), "+f"(C_fp32[54]),
        "+f"(C_fp32[55]), "+f"(C_fp32[56]), "+f"(C_fp32[57]), "+f"(C_fp32[58]), "+f"(C_fp32[59]),
        "+f"(C_fp32[60]), "+f"(C_fp32[61]), "+f"(C_fp32[62]), "+f"(C_fp32[63])
      : "l"(A_descriptor), "l"(B_descriptor));
}

/**
 * @brief `wait_barrier` spins until an mbarrier finishes the phase with the given parity.
 *
 * @param[in] barrier_address Shared-memory address of the mbarrier.
 * @param[in] phase_parity Parity of the phase to wait for: 0 for its 1st, 3rd, ... completion.
 *
 * @par Example
 * @code
 * wait_barrier(get_shared_address(&full_barriers[stage]), (step / STAGE_COUNT) % 2);
 * @endcode
 */
__device__ __forceinline__ void wait_barrier(uint32_t barrier_address, uint32_t phase_parity) {
  asm volatile(
      "{\n"
      ".reg .pred is_done;\n"
      "WAIT_LOOP:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 is_done, [%0], %1;\n"
      "@!is_done bra WAIT_LOOP;\n"
      "}\n" ::"r"(barrier_address),
      "r"(phase_parity)
      : "memory");
}

/**
 * @brief `issue_stage_loads` starts the TMA copies of one K step's tiles into a stage.
 *
 * @param[in] A_tensor_map TMA descriptor of A, with 64 x 128 boxes (K x M).
 * @param[in] B_tensor_map TMA descriptor of B, with 64 x 64 boxes (N x K).
 * @param[out] stage The stage: A's tile, then B's boxes, each 128-byte swizzled.
 * @param[in] barrier_address Shared-memory address of the stage's mbarrier.
 * @param[in] block_i Row of C where the block's tile starts.
 * @param[in] block_j Column of C where the block's tile starts.
 * @param[in] k_tile_start Column of A and row of B where the step starts.
 *
 * @par Example
 * @code
 * issue_stage_loads(A_tensor_map, B_tensor_map, stage, barrier_address, block_i, block_j, k);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - One thread calls it. Copies past A's or B's edges land as zeros and still count as bytes.
 */
__device__ __forceinline__ void issue_stage_loads(const CUtensorMap& A_tensor_map,
                                                  const CUtensorMap& B_tensor_map, uint8_t* stage,
                                                  uint32_t barrier_address, int block_i,
                                                  int block_j, int k_tile_start) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier_address),
               "r"(STAGE_BYTES)
               : "memory");
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(get_shared_address(stage)),
      "l"(&A_tensor_map), "r"(k_tile_start), "r"(block_i), "r"(barrier_address)
      : "memory");
  for (int box = 0; box < B_BOX_COUNT; ++box) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
            get_shared_address(stage + A_TILE_BYTES + (static_cast<ptrdiff_t>(box) * B_BOX_BYTES))),
        "l"(&B_tensor_map), "r"(block_j + (box * BF16_PER_SWIZZLE_ROW)), "r"(k_tile_start),
        "r"(barrier_address)
        : "memory");
  }
}

/**
 * @brief `store_accumulators` writes a warpgroup's 64 x 128 accumulator tile to C as bf16.
 *
 * Thread t of the warpgroup holds rows 16 * (t / 32) + (t % 32) / 4 and 8 more, and for each 8
 * columns the 2 at 2 * (t % 4); register r is row +8 when r % 4 >= 2, column +1 when r is odd.
 *
 * @param[in] C_fp32 This thread's 64 accumulators.
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of C.
 * @param[in] N Columns of C, a multiple of 8.
 * @param[in] i_start Row of C where the warpgroup's tile starts.
 * @param[in] j_start Column of C where the warpgroup's tile starts.
 *
 * @par Example
 * @code
 * store_accumulators(C_fp32[n], C_device, M, N, i_start, j_start);
 * @endcode
 */
__device__ __forceinline__ void store_accumulators(const float (&C_fp32)[ACCUMULATOR_COUNT],
                                                   bf16_t* C_device, int M, int N, int64_t i_start,
                                                   int64_t j_start) {
  constexpr int rows_per_warp = 16;
  constexpr int lanes_per_row = 4;
  constexpr int columns_per_group = 8;
  constexpr int registers_per_group = 4;
  constexpr int half_rows_per_warp = 8;
  const int lane = static_cast<int>(threadIdx.x) % THREADS_PER_WARP;
  const int warp = (static_cast<int>(threadIdx.x) % THREADS_PER_WARPGROUP) / THREADS_PER_WARP;
#pragma unroll
  for (int r = 0; r < ACCUMULATOR_COUNT; r += 2) {
    const int tile_row =
        (warp * rows_per_warp) + (lane / lanes_per_row) + (((r / 2) % 2) * half_rows_per_warp);
    const int tile_col =
        ((r / registers_per_group) * columns_per_group) + (2 * (lane % lanes_per_row));
    const int64_t i = i_start + tile_row;
    const int64_t j = j_start + tile_col;
    if (i < M && j < N) {
      *reinterpret_cast<__nv_bfloat162*>(&C_device[(i * N) + j]) =
          __floats2bfloat162_rn(C_fp32[r], C_fp32[r + 1]);
    }
  }
}

/**
 * @brief `matmul_wgmma_from_pipelined_kernel` computes a 128 x `BLOCK_TILE_N` tile of C with TMA
 * and wgmma.
 *
 * Shared memory holds `STAGE_COUNT` stages, each one 64-wide K step's A tile and B boxes, written
 * by TMA with a 128-byte swizzle and tracked by one mbarrier per stage. Thread 0 starts the first
 * stages. For each step, every thread waits on that stage's mbarrier; each of the 2 warpgroups
 * runs 4 x `WGMMAS_PER_ROW` wgmma instructions on its 64 rows straight from shared memory and
 * waits for them; the block syncs; and thread 0 refills the stage with the step `STAGE_COUNT`
 * ahead.
 *
 * @param[in] A_tensor_map TMA descriptor of A (M x K, row-major).
 * @param[in] B_tensor_map TMA descriptor of B (K x N, row-major).
 * @param[out] C_device Product, M x N, row-major, in device memory.
 * @param[in] M Rows of A and C.
 * @param[in] N Columns of B and C, a multiple of 8.
 * @param[in] K Columns of A and rows of B, a multiple of 8.
 *
 * @par Example
 * @code
 * matmul_wgmma_from_pipelined_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(A_tensor_map,
 *     B_tensor_map, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Swizzled tiles must start on 1024-byte boundaries, so the stages start at the next one.
 */
__global__ void __launch_bounds__(THREAD_COUNT, 1)
    matmul_wgmma_from_pipelined_kernel(const __grid_constant__ CUtensorMap A_tensor_map,
                                       const __grid_constant__ CUtensorMap B_tensor_map,
                                       bf16_t* C_device, int M, int N, int K) {
  extern __shared__ uint8_t shared_bytes[];
  __shared__ alignas(8) uint64_t full_barriers[STAGE_COUNT];
  const uint32_t misaligned_bytes = get_shared_address(shared_bytes) % SWIZZLE_ATOM_BYTES;
  uint8_t* stages = shared_bytes + ((SWIZZLE_ATOM_BYTES - misaligned_bytes) % SWIZZLE_ATOM_BYTES);
  const int thread_index = static_cast<int>(threadIdx.x);
  const int warpgroup = thread_index / THREADS_PER_WARPGROUP;
  const int block_i = static_cast<int>(blockIdx.y) * BLOCK_TILE_M;
  const int block_j = static_cast<int>(blockIdx.x) * BLOCK_TILE_N;
  const int step_count = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
  if (thread_index == 0) {
    for (int stage = 0; stage < STAGE_COUNT; ++stage) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(
                       get_shared_address(&full_barriers[stage]))
                   : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    for (int step = 0; step < STAGE_COUNT && step < step_count; ++step) {
      issue_stage_loads(
          A_tensor_map, B_tensor_map, stages + (static_cast<ptrdiff_t>(step) * STAGE_BYTES),
          get_shared_address(&full_barriers[step]), block_i, block_j, step * BLOCK_TILE_K);
    }
  }
  // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN THREADS WAIT ON UNINITIALIZED BARRIERS
  __syncthreads();
  float C_fp32[WGMMAS_PER_ROW][ACCUMULATOR_COUNT] = {};
  for (int step = 0; step < step_count; ++step) {
    const int stage = step % STAGE_COUNT;
    uint8_t* stage_bytes = stages + (static_cast<ptrdiff_t>(stage) * STAGE_BYTES);
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN WGMMA READS TILES STILL IN FLIGHT
    wait_barrier(get_shared_address(&full_barriers[stage]), (step / STAGE_COUNT) % 2);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    const uint32_t A_address =
        get_shared_address(stage_bytes) + (warpgroup * WGMMA_M * SWIZZLE_BYTES);
    const uint32_t B_address = get_shared_address(stage_bytes + A_TILE_BYTES);
#pragma unroll
    for (int k = 0; k < BLOCK_TILE_K; k += WGMMA_K) {
      const uint64_t A_descriptor = make_descriptor(
          A_address + (k * static_cast<int>(sizeof(bf16_t))), 16, SWIZZLE_ATOM_BYTES);
#pragma unroll
      for (int n = 0; n < WGMMAS_PER_ROW; ++n) {
        const uint32_t B_slice_address =
            B_address + (n * (WGMMA_N / BF16_PER_SWIZZLE_ROW) * B_BOX_BYTES) + (k * SWIZZLE_BYTES);
        wgmma_64x128x16(C_fp32[n], A_descriptor,
                        make_descriptor(B_slice_address, B_BOX_BYTES, SWIZZLE_ATOM_BYTES));
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    // DANGEROUS DANGEROUS DANGEROUS IF YOU CHANGE THIS THEN TMA OVERWRITES TILES WGMMA STILL READS
    __syncthreads();
    if (thread_index == 0 && step + STAGE_COUNT < step_count) {
      issue_stage_loads(A_tensor_map, B_tensor_map, stage_bytes,
                        get_shared_address(&full_barriers[stage]), block_i, block_j,
                        (step + STAGE_COUNT) * BLOCK_TILE_K);
    }
  }
#pragma unroll
  for (int n = 0; n < WGMMAS_PER_ROW; ++n) {
    store_accumulators(C_fp32[n], C_device, M, N, block_i + (warpgroup * WGMMA_M),
                       block_j + (n * WGMMA_N));
  }
}

#else

/**
 * @brief `matmul_wgmma_from_pipelined_kernel` traps: this build lacks sm_90a's wgmma and TMA.
 *
 * @param[in] A_tensor_map Unused.
 * @param[in] B_tensor_map Unused.
 * @param[out] C_device Unused.
 * @param[in] M Unused.
 * @param[in] N Unused.
 * @param[in] K Unused.
 *
 * @par Example
 * @code
 * matmul_wgmma_from_pipelined_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(A_tensor_map,
 *     B_tensor_map, C_device, M, N, K);
 * @endcode
 */
__global__ void matmul_wgmma_from_pipelined_kernel(const __grid_constant__ CUtensorMap A_tensor_map,
                                                   const __grid_constant__ CUtensorMap B_tensor_map,
                                                   bf16_t* C_device, int M, int N, int K) {
  __trap();
}

#endif

/**
 * @brief `make_tensor_map` builds a TMA descriptor for a row-major bf16 matrix with 128-byte
 * swizzled boxes.
 *
 * @param[in] X_device The matrix, in device memory, 16-byte aligned.
 * @param[in] row_count Rows of the matrix.
 * @param[in] col_count Columns of the matrix, a multiple of 8.
 * @param[in] box_row_count Rows per box, at most 256.
 * @return The descriptor; boxes are 64 columns (128 bytes) wide.
 *
 * @par Example
 * @code
 * const CUtensorMap A_tensor_map = make_tensor_map(A_device, M, K, BLOCK_TILE_M);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Box elements outside the matrix read as zeros.
 */
CUtensorMap make_tensor_map(const bf16_t* X_device, int row_count, int col_count,
                            int box_row_count) {
  CUtensorMap tensor_map;
  const cuuint64_t global_dims[2] = {
      static_cast<cuuint64_t>(col_count),
      static_cast<cuuint64_t>(row_count),
  };
  const cuuint64_t global_strides_bytes[1] = {static_cast<cuuint64_t>(col_count) * sizeof(bf16_t)};
  const cuuint32_t box_dims[2] = {
      static_cast<cuuint32_t>(BF16_PER_SWIZZLE_ROW),
      static_cast<cuuint32_t>(box_row_count),
  };
  const cuuint32_t element_strides[2] = {1, 1};
  CHECK(cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                               const_cast<bf16_t*>(X_device), global_dims, global_strides_bytes,
                               box_dims, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tensor_map;
}

}  // namespace

/**
 * @brief `matmul_wgmma_from_pipelined` builds A's and B's TMA descriptors and launches a 256-thread
 * block per 128 x `BLOCK_TILE_N` tile of C.
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
 * matmul_wgmma_from_pipelined(A_device, B_device, C_device, M, N, K);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - N and K must be multiples of 8: TMA needs every row to start on a 16-byte boundary.
 * - Gotchas:
 *   - The first call raises the kernel's dynamic shared-memory limit, which defaults to 48 KB.
 */
void matmul_wgmma_from_pipelined(const bf16_t* A_device, const bf16_t* B_device, bf16_t* C_device,
                                 int M, int N, int K) {
  static const bool is_shared_limit_raised_by_first_call = [] {
    CHECK(cudaFuncSetAttribute(matmul_wgmma_from_pipelined_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
    return true;
  }();
  (void)is_shared_limit_raised_by_first_call;
  const CUtensorMap A_tensor_map = make_tensor_map(A_device, M, K, BLOCK_TILE_M);
  const CUtensorMap B_tensor_map = make_tensor_map(B_device, K, N, BLOCK_TILE_K);
  const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
  matmul_wgmma_from_pipelined_kernel<<<grid, THREAD_COUNT, SHARED_BYTES>>>(
      A_tensor_map, B_tensor_map, C_device, M, N, K);
}
