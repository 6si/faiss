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
    int max_iter = params.max_iterations > 0
            ? params.max_iterations
            : 2 * ef / sw + 10;
    int dim = static_cast<int>(idx.dim);
    int num_upper_layers = idx.num_upper_layers_built;

    auto launch_kernels = [&]<typename DataT>(
                                  const DataT* d_data,
                                  const float* d_inv_norms) {
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
            int max_staging_check = sw * idx.max_degree0;
            if (max_staging_check > block_size ||
                (max_staging_check & (max_staging_check - 1)) != 0) {
                throw std::runtime_error(
                        std::string("gpu_hnsw: search_width * max_degree0 = ") +
                        std::to_string(max_staging_check) +
                        " must be a power of 2 and <= block_size (" +
                        std::to_string(block_size) + ") for parallel merge");
            }
            // Fixed overhead: staging (sw*deg0*8) + parent_ids (sw*4) + meta (12)
            int smem_overhead = sw * idx.max_degree0 * 8 + sw * 4 + 12;
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
            if (ef > max_ef) {
                ef = max_ef;
            }
        }

        size_t smem_size = hnsw_kernel::calc_layer0_smem_size(
                ef, sw, idx.max_degree0);

        // Opt into >48 KiB dynamic shared memory when the device supports it;
        // without this the kernel launch would fail for large ef.
        if (smem_size > 49152) {
            GPU_HNSW_CUDA_CHECK(cudaFuncSetAttribute(
                    hnsw_kernel::layer0_beam_search_kernel<DataT>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem_size)));
        }

        int N_int = static_cast<int>(idx.n_rows);
        size_t bitmap_bytes = hnsw_kernel::calc_visited_bitmap_size(
                num_queries, N_int);

        GPU_HNSW_CUDA_CHECK(
                cudaMemsetAsync(sc.d_visited_bitmaps, 0, bitmap_bytes, stream));

        hnsw_kernel::layer0_beam_search_kernel<DataT>
                <<<num_queries, block_size, smem_size, stream>>>(
                        sc.d_queries,
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
            launch_kernels(
                    static_cast<const int8_t*>(idx.d_dataset), idx.d_inv_norms);
            break;
        case GpuHnswDatasetType::FP16:
            launch_kernels(
                    static_cast<const half*>(idx.d_dataset), idx.d_inv_norms);
            break;
        case GpuHnswDatasetType::BF16:
            launch_kernels(
                    static_cast<const __nv_bfloat16*>(idx.d_dataset),
                    idx.d_inv_norms);
            break;
        case GpuHnswDatasetType::FP32:
        default:
            launch_kernels(
                    static_cast<const float*>(idx.d_dataset), idx.d_inv_norms);
            break;
    }
}

} // namespace gpu
} // namespace faiss
