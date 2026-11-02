/**
 * @file
 * @brief `hopper.cuh` wraps the sm_90a PTX that the Hopper kernels share: mbarriers, TMA copies
 * and wgmma on 128-byte-swizzled shared-memory tiles.
 *
 * @par Interface
 * - `hopper::SWIZZLE_BYTES`, `hopper::SWIZZLE_ATOM_BYTES`, `hopper::BF16_PER_SWIZZLE_ROW`: Sizes
 *   of a 128-byte-swizzled tile's rows and 8-row atoms.
 * - `hopper::WGMMA_ACCUMULATOR_COUNT`: Accumulators per thread of one 64 x 128 wgmma tile.
 * - `hopper::get_shared_address`: A shared-memory pointer's 32-bit address.
 * - `hopper::make_descriptor`: A wgmma descriptor of a 128-byte-swizzled tile.
 * - `hopper::init_barrier`, `hopper::arrive_barrier`, `hopper::expect_barrier_bytes`,
 *   `hopper::wait_barrier`: mbarrier operations.
 * - `hopper::load_tile_async`: A 2D TMA copy that completes on an mbarrier.
 * - `hopper::load_tile_multicast_async`: The same copy, landing in several blocks of a cluster.
 * - `hopper::get_cluster_rank`, `hopper::arrive_cluster_barrier`, `hopper::sync_cluster`: Thread
 *   block cluster operations.
 * - `hopper::store_tile_async`, `hopper::commit_bulk_stores`, `hopper::wait_bulk_stores_read`,
 *   `hopper::wait_bulk_stores`, `hopper::fence_async_shared`: TMA stores from shared memory.
 * - `hopper::sync_warpgroup`: A named barrier for the 128 threads of one warpgroup.
 * - `hopper::wgmma_64x128x16`: One wgmma, A K-major and B N-major, both from shared memory.
 * - `hopper::wgmma_64x256x16`: The same, 256 columns wide.
 * - `hopper::fence_wgmma`, `hopper::commit_wgmma_group`, `hopper::wait_wgmma_groups`: wgmma
 *   ordering.
 * - `hopper::set_max_registers`: Changes the warpgroup's register count (`setmaxnreg`).
 * - `hopper::make_tensor_map`: A TMA descriptor of a row-major bf16 matrix (host).
 *
 * @par Example
 * @code
 * hopper::wait_barrier(hopper::get_shared_address(&full_barriers[stage]), phase_parity);
 * @endcode
 *
 * @par Must know
 * - Limitations:
 *   - The device functions compile only for sm_90a; include it inside the sm_90a `#if` too.
 */
#pragma once

#include <cuda.h>

#include <cstdint>

#include "common/matmul.h"

namespace hopper {

inline constexpr int SWIZZLE_BYTES = 128;
inline constexpr int SWIZZLE_ATOM_BYTES = 8 * SWIZZLE_BYTES;
inline constexpr int BF16_PER_SWIZZLE_ROW = SWIZZLE_BYTES / static_cast<int>(sizeof(bf16_t));
inline constexpr int WGMMA_M = 64;
inline constexpr int WGMMA_N = 128;
inline constexpr int WGMMA_K = 16;
inline constexpr int THREADS_PER_WARPGROUP = 128;
inline constexpr int WGMMA_ACCUMULATOR_COUNT = WGMMA_M * WGMMA_N / THREADS_PER_WARPGROUP;

/**
 * @brief `make_tensor_map` builds a TMA descriptor of a row-major bf16 matrix with 128-byte
 * swizzled boxes, 64 columns wide.
 *
 * @param[in] X_device The matrix, in device memory, 16-byte aligned.
 * @param[in] row_count Rows of the matrix.
 * @param[in] col_count Columns of the matrix, a multiple of 8.
 * @param[in] box_row_count Rows per box, at most 256.
 * @return The descriptor. Box elements outside the matrix read as zeros.
 *
 * @par Example
 * @code
 * const CUtensorMap A_tensor_map = hopper::make_tensor_map(A_device, M, K, 128);
 * @endcode
 */
inline CUtensorMap make_tensor_map(const bf16_t* X_device, int row_count, int col_count,
                                   int box_row_count) {
  CUtensorMap tensor_map;
  const cuuint64_t global_dims[2] = {
      static_cast<cuuint64_t>(col_count),
      static_cast<cuuint64_t>(row_count),
  };
  const cuuint64_t global_strides_bytes[1] = {
      static_cast<cuuint64_t>(col_count) * sizeof(bf16_t),
  };
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

#if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH_FEAT_SM90_ALL)

inline constexpr uint64_t SWIZZLE_128B_LAYOUT_TYPE = 1;
inline constexpr uint32_t DESCRIPTOR_ADDRESS_SHIFT = 4;
inline constexpr uint32_t DESCRIPTOR_LEADING_OFFSET_BIT = 16;
inline constexpr uint32_t DESCRIPTOR_STRIDE_OFFSET_BIT = 32;
inline constexpr uint32_t DESCRIPTOR_LAYOUT_TYPE_BIT = 62;
inline constexpr uint32_t DESCRIPTOR_ADDRESS_MASK = (1U << 18U) - 1U;

/**
 * @brief `get_shared_address` returns a pointer's address in the shared-memory window.
 *
 * @param[in] pointer A pointer into shared memory.
 * @return Its 32-bit shared-memory address, as PTX instructions take it.
 *
 * @par Example
 * @code
 * const uint32_t barrier_address = hopper::get_shared_address(&full_barriers[stage]);
 * @endcode
 */
__device__ __forceinline__ uint32_t get_shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

/**
 * @brief `make_descriptor` builds a wgmma descriptor of a 128-byte-swizzled tile.
 *
 * @param[in] start_address Shared-memory address of the tile's first element.
 * @param[in] leading_bytes Bytes between 64-element groups along M or N (unused for K-major).
 * @param[in] stride_bytes Bytes between groups of 8 rows.
 * @return The descriptor: address and offsets in 16-byte units, and the swizzle mode.
 *
 * @par Example
 * @code
 * const uint64_t A_descriptor = hopper::make_descriptor(A_address, 16, 1024);
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
 * @brief `init_barrier` initializes an mbarrier that completes after `arrival_count` arrivals.
 *
 * @param[in] barrier_address Shared-memory address of the mbarrier.
 * @param[in] arrival_count Arrivals per phase.
 *
 * @par Example
 * @code
 * hopper::init_barrier(hopper::get_shared_address(&empty_barriers[stage]), 2);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Follow the inits with `fence.mbarrier_init` and a block sync before any thread uses them.
 */
__device__ __forceinline__ void init_barrier(uint32_t barrier_address, uint32_t arrival_count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(barrier_address),
               "r"(arrival_count)
               : "memory");
}

/**
 * @brief `arrive_barrier` arrives once on an mbarrier.
 *
 * @param[in] barrier_address Shared-memory address of the mbarrier.
 *
 * @par Example
 * @code
 * hopper::arrive_barrier(hopper::get_shared_address(&empty_barriers[stage]));
 * @endcode
 */
__device__ __forceinline__ void arrive_barrier(uint32_t barrier_address) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(barrier_address) : "memory");
}

/**
 * @brief `expect_barrier_bytes` arrives on an mbarrier and adds bytes that must land before the
 * phase completes.
 *
 * @param[in] barrier_address Shared-memory address of the mbarrier.
 * @param[in] byte_count Bytes that TMA copies will complete on it.
 *
 * @par Example
 * @code
 * hopper::expect_barrier_bytes(barrier_address, STAGE_BYTES);
 * @endcode
 */
__device__ __forceinline__ void expect_barrier_bytes(uint32_t barrier_address,
                                                     uint32_t byte_count) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier_address),
               "r"(byte_count)
               : "memory");
}

/**
 * @brief `wait_barrier` spins until an mbarrier finishes the phase with the given parity.
 *
 * @param[in] barrier_address Shared-memory address of the mbarrier.
 * @param[in] phase_parity Parity of the phase to wait for: 0 for its 1st, 3rd, ... completion.
 *
 * @par Example
 * @code
 * hopper::wait_barrier(barrier_address, (step / STAGE_COUNT) % 2);
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
 * @brief `load_tile_async` starts a 2D TMA copy of one box into shared memory.
 *
 * @param[in] tensor_map The source's TMA descriptor, a kernel parameter.
 * @param[in] dst_address Shared-memory address of the box, 1024-byte aligned.
 * @param[in] barrier_address Shared-memory address of the mbarrier that the copy completes on.
 * @param[in] col The box's first column (the inner coordinate).
 * @param[in] row The box's first row.
 *
 * @par Example
 * @code
 * hopper::load_tile_async(A_tensor_map, A_address, barrier_address, k_tile_start, block_i);
 * @endcode
 */
__device__ __forceinline__ void load_tile_async(const CUtensorMap& tensor_map, uint32_t dst_address,
                                                uint32_t barrier_address, int col, int row) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst_address),
      "l"(&tensor_map), "r"(col), "r"(row), "r"(barrier_address)
      : "memory");
}

/**
 * @brief `load_tile_multicast_async` starts a 2D TMA copy of one box into the same shared-memory
 * offset of every block in `cta_mask`, completing on each block's mbarrier at the same offset.
 *
 * @param[in] tensor_map The source's TMA descriptor, a kernel parameter.
 * @param[in] dst_address Shared-memory address of the box in this block, 1024-byte aligned.
 * @param[in] barrier_address Shared-memory address of the mbarrier in this block.
 * @param[in] col The box's first column (the inner coordinate).
 * @param[in] row The box's first row.
 * @param[in] cta_mask Bit r set: the box also lands in the cluster's block of rank r.
 *
 * @par Example
 * @code
 * hopper::load_tile_multicast_async(B_tensor_map, B_address, barrier_address, col, row, 0b11);
 * @endcode
 */
__device__ __forceinline__ void load_tile_multicast_async(const CUtensorMap& tensor_map,
                                                          uint32_t dst_address,
                                                          uint32_t barrier_address, int col,
                                                          int row, uint16_t cta_mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      ".multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(dst_address),
      "l"(&tensor_map), "r"(col), "r"(row), "r"(barrier_address), "h"(cta_mask)
      : "memory");
}

/**
 * @brief `get_cluster_rank` returns this block's rank in its thread block cluster.
 *
 * @return The rank, from 0 to the cluster's size - 1.
 *
 * @par Example
 * @code
 * const int cluster_rank = hopper::get_cluster_rank();
 * @endcode
 */
__device__ __forceinline__ int get_cluster_rank() {
  uint32_t rank = 0;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
  return static_cast<int>(rank);
}

/**
 * @brief `arrive_cluster_barrier` arrives once on the mbarrier at the same offset in the cluster's
 * block of rank `cta_rank`, which may be this block.
 *
 * @param[in] barrier_address Shared-memory address of the mbarrier in this block.
 * @param[in] cta_rank Rank of the block whose mbarrier to arrive on.
 *
 * @par Example
 * @code
 * hopper::arrive_cluster_barrier(empty_barrier_address, peer_rank);
 * @endcode
 */
__device__ __forceinline__ void arrive_cluster_barrier(uint32_t barrier_address, int cta_rank) {
  asm volatile(
      "{\n"
      ".reg .b32 remote_address;\n"
      "mapa.shared::cluster.u32 remote_address, %0, %1;\n"
      "mbarrier.arrive.shared::cluster.b64 _, [remote_address];\n"
      "}\n" ::"r"(barrier_address),
      "r"(cta_rank)
      : "memory");
}

/**
 * @brief `sync_cluster` waits until every thread of every block in the cluster gets here.
 *
 * @par Example
 * @code
 * hopper::sync_cluster();
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Every thread of the cluster must call it, as with `__syncthreads`.
 */
__device__ __forceinline__ void sync_cluster() {
  asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
}

/**
 * @brief `store_tile_async` starts a 2D TMA store of one box from shared memory to global memory.
 *
 * @param[in] tensor_map The destination's TMA descriptor, a kernel parameter.
 * @param[in] src_address Shared-memory address of the box, 1024-byte aligned.
 * @param[in] col The box's first column (the inner coordinate).
 * @param[in] row The box's first row.
 *
 * @par Example
 * @code
 * hopper::store_tile_async(C_tensor_map, staging_address, block_j, block_i);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Box elements outside the matrix are not written. Fence shared memory before it.
 */
__device__ __forceinline__ void store_tile_async(const CUtensorMap& tensor_map,
                                                 uint32_t src_address, int col, int row) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n" ::"l"(
          &tensor_map),
      "r"(col), "r"(row), "r"(src_address)
      : "memory");
}

/**
 * @brief `commit_bulk_stores` groups the TMA stores started since the last commit.
 *
 * @par Example
 * @code
 * hopper::commit_bulk_stores();
 * @endcode
 */
__device__ __forceinline__ void commit_bulk_stores() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

/**
 * @brief `wait_bulk_stores_read` waits until at most `PENDING_COUNT` store groups may still read
 * their shared memory.
 *
 * @tparam PENDING_COUNT Groups that may still read.
 *
 * @par Example
 * @code
 * hopper::wait_bulk_stores_read<0>();
 * @endcode
 */
template <int PENDING_COUNT>
__device__ __forceinline__ void wait_bulk_stores_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(PENDING_COUNT) : "memory");
}

/**
 * @brief `wait_bulk_stores` waits until at most `PENDING_COUNT` store groups are unfinished.
 *
 * @tparam PENDING_COUNT Groups that may still run.
 *
 * @par Example
 * @code
 * hopper::wait_bulk_stores<0>();
 * @endcode
 */
template <int PENDING_COUNT>
__device__ __forceinline__ void wait_bulk_stores() {
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(PENDING_COUNT) : "memory");
}

/**
 * @brief `fence_async_shared` makes this thread's shared-memory writes visible to TMA.
 *
 * @par Example
 * @code
 * hopper::fence_async_shared();
 * @endcode
 */
__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/**
 * @brief `sync_warpgroup` waits until the 128 threads of one warpgroup reach named barrier
 * `barrier_id`.
 *
 * @param[in] barrier_id Named barrier, 1 to 15; barrier 0 is `__syncthreads`'s.
 *
 * @par Example
 * @code
 * hopper::sync_warpgroup(1 + consumer);
 * @endcode
 */
__device__ __forceinline__ void sync_warpgroup(int barrier_id) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(barrier_id), "n"(THREADS_PER_WARPGROUP) : "memory");
}

/**
 * @brief `fence_wgmma` orders earlier register and shared-memory writes before the next wgmmas.
 *
 * @par Example
 * @code
 * hopper::fence_wgmma();
 * @endcode
 */
__device__ __forceinline__ void fence_wgmma() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

/**
 * @brief `commit_wgmma_group` groups the wgmmas started since the last commit.
 *
 * @par Example
 * @code
 * hopper::commit_wgmma_group();
 * @endcode
 */
__device__ __forceinline__ void commit_wgmma_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

/**
 * @brief `wait_wgmma_groups` waits until at most `PENDING_COUNT` wgmma groups are unfinished.
 *
 * @tparam PENDING_COUNT Groups that may still run.
 *
 * @par Example
 * @code
 * hopper::wait_wgmma_groups<0>();
 * @endcode
 */
template <int PENDING_COUNT>
__device__ __forceinline__ void wait_wgmma_groups() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(PENDING_COUNT) : "memory");
}

/**
 * @brief `set_max_registers` changes the calling warpgroup's registers per thread.
 *
 * @tparam REGISTER_COUNT Registers per thread, a multiple of 8 from 24 to 256.
 * @tparam IS_INCREASE Whether to raise the count (consumers) or lower it (producers).
 *
 * @par Example
 * @code
 * hopper::set_max_registers<240, true>();
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - Every thread of the warpgroup must call it; raising waits for registers others released.
 */
template <int REGISTER_COUNT, bool IS_INCREASE>
__device__ __forceinline__ void set_max_registers() {
  if constexpr (IS_INCREASE) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(REGISTER_COUNT));
  } else {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(REGISTER_COUNT));
  }
}

/**
 * @brief `wgmma_64x128x16` adds A (64 x 16, K-major) times B (16 x 128, N-major) to the
 * warpgroup's 64 x 128 accumulators.
 *
 * @param[in,out] C_fp32 This thread's 64 accumulators.
 * @param[in] A_descriptor Descriptor of A's 64 x 16 slice.
 * @param[in] B_descriptor Descriptor of B's 16 x 128 slice, read transposed (N-major).
 *
 * @par Example
 * @code
 * hopper::wgmma_64x128x16(C_fp32, A_descriptor, B_descriptor);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - It only starts the multiply; `commit_wgmma_group` and `wait_wgmma_groups` finish it.
 */
__device__ __forceinline__ void wgmma_64x128x16(float (&C_fp32)[WGMMA_ACCUMULATOR_COUNT],
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
 * @brief `wgmma_64x256x16` adds A (64 x 16, K-major) times B (16 x 256, N-major) to the
 * warpgroup's 64 x 256 accumulators.
 *
 * @param[in,out] C_fp32 This thread's 128 accumulators, laid out as two 64 x 128 tiles' would be.
 * @param[in] A_descriptor Descriptor of A's 64 x 16 slice.
 * @param[in] B_descriptor Descriptor of B's 16 x 256 slice, read transposed (N-major).
 *
 * @par Example
 * @code
 * hopper::wgmma_64x256x16(C_fp32, A_descriptor, B_descriptor);
 * @endcode
 *
 * @par Must know
 * - Gotchas:
 *   - It only starts the multiply; `commit_wgmma_group` and `wait_wgmma_groups` finish it.
 * - Performance:
 *   - It reads A's slice once for all 256 columns, where two 64 x 128 wgmmas read it twice.
 */
__device__ __forceinline__ void wgmma_64x256x16(float (&C_fp32)[2 * WGMMA_ACCUMULATOR_COUNT],
                                                uint64_t A_descriptor, uint64_t B_descriptor) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
      "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
      "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
      "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, "
      "%127"
      "}, "
      "%128, %129, 1, 1, 1, 0, 1;\n"
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
        "+f"(C_fp32[60]), "+f"(C_fp32[61]), "+f"(C_fp32[62]), "+f"(C_fp32[63]), "+f"(C_fp32[64]),
        "+f"(C_fp32[65]), "+f"(C_fp32[66]), "+f"(C_fp32[67]), "+f"(C_fp32[68]), "+f"(C_fp32[69]),
        "+f"(C_fp32[70]), "+f"(C_fp32[71]), "+f"(C_fp32[72]), "+f"(C_fp32[73]), "+f"(C_fp32[74]),
        "+f"(C_fp32[75]), "+f"(C_fp32[76]), "+f"(C_fp32[77]), "+f"(C_fp32[78]), "+f"(C_fp32[79]),
        "+f"(C_fp32[80]), "+f"(C_fp32[81]), "+f"(C_fp32[82]), "+f"(C_fp32[83]), "+f"(C_fp32[84]),
        "+f"(C_fp32[85]), "+f"(C_fp32[86]), "+f"(C_fp32[87]), "+f"(C_fp32[88]), "+f"(C_fp32[89]),
        "+f"(C_fp32[90]), "+f"(C_fp32[91]), "+f"(C_fp32[92]), "+f"(C_fp32[93]), "+f"(C_fp32[94]),
        "+f"(C_fp32[95]), "+f"(C_fp32[96]), "+f"(C_fp32[97]), "+f"(C_fp32[98]), "+f"(C_fp32[99]),
        "+f"(C_fp32[100]), "+f"(C_fp32[101]), "+f"(C_fp32[102]), "+f"(C_fp32[103]),
        "+f"(C_fp32[104]), "+f"(C_fp32[105]), "+f"(C_fp32[106]), "+f"(C_fp32[107]),
        "+f"(C_fp32[108]), "+f"(C_fp32[109]), "+f"(C_fp32[110]), "+f"(C_fp32[111]),
        "+f"(C_fp32[112]), "+f"(C_fp32[113]), "+f"(C_fp32[114]), "+f"(C_fp32[115]),
        "+f"(C_fp32[116]), "+f"(C_fp32[117]), "+f"(C_fp32[118]), "+f"(C_fp32[119]),
        "+f"(C_fp32[120]), "+f"(C_fp32[121]), "+f"(C_fp32[122]), "+f"(C_fp32[123]),
        "+f"(C_fp32[124]), "+f"(C_fp32[125]), "+f"(C_fp32[126]), "+f"(C_fp32[127])
      : "l"(A_descriptor), "l"(B_descriptor));
}

#endif

}  // namespace hopper
