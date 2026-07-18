// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2026, 6sense Insights Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <faiss/gpu/impl/GpuHnswSearchKernel.cuh>
#include <faiss/gpu/impl/GpuHnswTypes.h>

#define GPU_HNSW_CUDA_CHECK(expr)                                     \
    do {                                                              \
        cudaError_t _e = (expr);                                      \
        if (_e != cudaSuccess) {                                      \
            throw std::runtime_error(                                 \
                    std::string("CUDA error: ") +                     \
                    cudaGetErrorString(_e) + " at " + __FILE__ + ":" + \
                    std::to_string(__LINE__));                         \
        }                                                             \
    } while (0)

namespace faiss {
namespace gpu {

inline void gpu_hnsw_search(
        cudaStream_t stream,
        const GpuHnswSearchParams& params,
        const GpuHnswDeviceIndex& idx,
        GpuHnswSearchScratch& sc,
        int num_queries,
        int k) {

    int ef = params.ef;
    int sw = params.search_width;
    // Reject degenerate params up front: ef and search_width feed the
    // shared-memory sizing and the auto-iteration bound (2*ef/sw below), so a
    // value of 0 would divide by zero and a negative value would under-size
    // the buffers.
    if (ef <= 0 || sw <= 0) {
        throw std::runtime_error(
                std::string("gpu_hnsw: ef and search_width must be > 0 (ef=") +
                std::to_string(ef) + ", search_width=" + std::to_string(sw) +
                ")");
    }
    // Auto-clamp ef up to k. The kernel tracks exactly ef candidates in the
    // result beam, so ef < k could only ever return ef real neighbors with the
    // remaining k-ef slots padded with sentinels. Raising ef to k matches the
    // CPU / Knowhere HNSW contract (always return k valid results). The
    // shared-memory-fit check below still throws if even k slots cannot fit.
    if (k > ef) {
        ef = k;
    }
    int max_iter = params.max_iterations > 0
            ? params.max_iterations
            : 2 * ef / sw + 10;
    int dim = static_cast<int>(idx.dim);
    int num_upper_layers = idx.num_upper_layers_built;

    // Layer-0 is templated on the dataset type (DataT), the layer-0 query type
    // (QueryT: float generic, int8_t for the native DP4A path) and USE_DP4A.
    // The upper-layer greedy descent always uses the fp32 queries (sc.d_queries).
    auto launch_kernels = [&]<typename DataT, typename QueryT, bool USE_DP4A>(
                                  const DataT* d_data,
                                  const float* d_inv_norms,
                                  const QueryT* d_layer0_queries) {
        if (num_upper_layers > 0) {
            auto* d_layer_ptrs = static_cast<hnsw_kernel::upper_layer_ptrs*>(
                    idx.d_upper_layer_ptrs);

            int warps_per_block = 4;
            int threads_per_block = warps_per_block * 32;
            int num_blocks =
                    (num_queries + warps_per_block - 1) / warps_per_block;

            hnsw_kernel::upper_layer_search_kernel<DataT>
                    <<<num_blocks, threads_per_block, 0, stream>>>(
                            sc.d_queries,
                            d_data,
                            d_inv_norms,
                            d_layer_ptrs,
                            sc.d_entry_points,
                            idx.entry_point,
                            num_queries,
                            dim,
                            num_upper_layers,
                            idx.use_ip);
            GPU_HNSW_CUDA_CHECK(cudaGetLastError());
        } else {
            std::vector<uint32_t> h_eps(num_queries, idx.entry_point);
            GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
                    sc.d_entry_points,
                    h_eps.data(),
                    num_queries * sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    stream));
            // h_eps is stack-local; synchronize before it is destroyed so the
            // copy never reads freed memory (safe even if the source buffer is
            // ever switched to pinned host memory).
            GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        int block_size =
                params.thread_block_size > 0 ? params.thread_block_size : 128;

        // Per-block dynamic shared-memory budget for this device. The default
        // limit is 48 KiB, but Volta+ GPUs can opt into more via
        // cudaFuncSetAttribute; query the real limit instead of assuming 48 KiB.
        int smem_max = 49152;
        {
            int device = 0;
            int optin = 0;
            if (cudaGetDevice(&device) == cudaSuccess &&
                cudaDeviceGetAttribute(
                        &optin,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin,
                        device) == cudaSuccess &&
                optin > smem_max) {
                smem_max = optin;
            }
        }

        {
            // The bitonic sort in the parallel merge needs a power-of-two
            // staging capacity, one thread per slot. Pad up to the next power
            // of two (handles non-power-of-two 2*M) and grow the block so every
            // staging slot is owned by a thread.
            int max_staging = hnsw_kernel::padded_staging_capacity(
                    sw, idx.max_degree0);
            if (block_size < max_staging) {
                block_size = max_staging;
            }
            if (block_size > 1024) {
                throw std::runtime_error(
                        std::string("gpu_hnsw: padded staging capacity ") +
                        std::to_string(max_staging) +
                        " exceeds the max CUDA block size (1024); reduce "
                        "search_width or M (max_degree0=" +
                        std::to_string(idx.max_degree0) + ")");
            }
            // Fixed overhead: staging (max_staging*8) + parent_ids (sw*4)
            // + meta (12)
            int smem_overhead = max_staging * 8 + sw * 4 + 12;
            // Per-ef cost: 3 result arrays + 3 merge arrays = 6 × 4 = 24 bytes/slot
            int max_ef = (smem_max - smem_overhead) / 24;
            if (max_ef < 1) {
                throw std::runtime_error(
                        std::string("gpu_hnsw: search_width=") +
                        std::to_string(sw) +
                        " too large for device shared memory (" +
                        std::to_string(smem_max) +
                        " bytes); reduce search_width");
            }
            // ef was auto-clamped up to at least k above. If even k candidate
            // slots cannot fit in shared memory, fail loudly rather than
            // silently returning fewer than k valid results.
            if (k > max_ef) {
                throw std::runtime_error(
                        std::string("gpu_hnsw: k=") + std::to_string(k) +
                        " needs k candidate slots (ef>=k), only " +
                        std::to_string(max_ef) +
                        " fit in device shared memory (" +
                        std::to_string(smem_max) +
                        " bytes); reduce k or raise thread_block_size");
            }
            if (ef > max_ef) {
                fprintf(stderr,
                        "[gpu_hnsw] warning: ef=%d exceeds the per-block "
                        "shared-memory budget (%d bytes); clamping ef to %d. "
                        "Recall may be reduced; raise thread_block_size or "
                        "lower search_width to restore the requested ef.\n",
                        ef, smem_max, max_ef);
                ef = max_ef;
            }
        }

        size_t smem_size = hnsw_kernel::calc_layer0_smem_size(
                ef, sw, idx.max_degree0);

        // Opt into >48 KiB dynamic shared memory when the device supports it;
        // without this the kernel launch would fail for large ef. This is
        // cached per kernel instantiation: cudaFuncSetAttribute configures the
        // kernel's max dynamic shared memory globally, so it only needs to run
        // once per (kernel, high-water size) rather than on every search. The
        // statics live in this generic-lambda body, which is instantiated once
        // per <DataT, QueryT, USE_DP4A> combination.
        //
        // The mutex makes the check-set-store atomic and monotonic: without it,
        // two concurrent searches with different smem sizes can interleave so a
        // smaller-ef search's cudaFuncSetAttribute runs *after* a larger one,
        // downgrading the kernel's global attribute below what a recorded
        // high-water mark implies, and a later intermediate search then skips
        // the set (thinks it's configured) and fails to launch. Under the lock
        // the attribute only ever grows, and is always >= the current launch's
        // requirement before we proceed.
        if (smem_size > 49152) {
            static std::mutex configured_smem_mutex;
            static size_t configured_smem = 0;
            std::lock_guard<std::mutex> lock(configured_smem_mutex);
            if (smem_size > configured_smem) {
                GPU_HNSW_CUDA_CHECK(cudaFuncSetAttribute(
                        hnsw_kernel::layer0_beam_search_kernel<
                                DataT, QueryT, USE_DP4A>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(smem_size)));
                configured_smem = smem_size;
            }
        }

        int N_int = static_cast<int>(idx.n_rows);
        size_t bitmap_bytes = hnsw_kernel::calc_visited_bitmap_size(
                num_queries, N_int);

        GPU_HNSW_CUDA_CHECK(
                cudaMemsetAsync(sc.d_visited_bitmaps, 0, bitmap_bytes, stream));

        hnsw_kernel::layer0_beam_search_kernel<DataT, QueryT, USE_DP4A>
                <<<num_queries, block_size, smem_size, stream>>>(
                        d_layer0_queries,
                        d_data,
                        d_inv_norms,
                        idx.d_layer0_graph,
                        sc.d_entry_points,
                        sc.d_visited_bitmaps,
                        sc.d_neighbors,
                        sc.d_distances,
                        num_queries,
                        N_int,
                        dim,
                        idx.max_degree0,
                        k,
                        ef,
                        sw,
                        max_iter,
                        idx.use_ip);
        GPU_HNSW_CUDA_CHECK(cudaGetLastError());
    };

    switch (idx.dataset_type) {
        case GpuHnswDatasetType::INT8:
            // Native DP4A path: requires int8 queries staged on device, an
            // inner-product/cosine metric (DP4A only computes dot products) and
            // dim % 4 == 0. Otherwise fall back to the generic fp32-query path.
            if (sc.d_queries_i8 != nullptr && idx.use_ip && (dim % 4 == 0)) {
                launch_kernels.template operator()<int8_t, int8_t, true>(
                        static_cast<const int8_t*>(idx.d_dataset),
                        idx.d_inv_norms,
                        sc.d_queries_i8);
            } else {
                launch_kernels.template operator()<int8_t, float, false>(
                        static_cast<const int8_t*>(idx.d_dataset),
                        idx.d_inv_norms,
                        sc.d_queries);
            }
            break;
        case GpuHnswDatasetType::FP16:
            launch_kernels.template operator()<half, float, false>(
                    static_cast<const half*>(idx.d_dataset),
                    idx.d_inv_norms,
                    sc.d_queries);
            break;
        case GpuHnswDatasetType::BF16:
            launch_kernels.template operator()<__nv_bfloat16, float, false>(
                    static_cast<const __nv_bfloat16*>(idx.d_dataset),
                    idx.d_inv_norms,
                    sc.d_queries);
            break;
        case GpuHnswDatasetType::FP32:
        default:
            launch_kernels.template operator()<float, float, false>(
                    static_cast<const float*>(idx.d_dataset),
                    idx.d_inv_norms,
                    sc.d_queries);
            break;
    }
}

} // namespace gpu
} // namespace faiss
