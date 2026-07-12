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

// Converts a FAISS HNSW index (CSR graph) + vectors into a
// GpuHnswDeviceIndex on device memory.
//
// This version is adapted for Knowhere's FAISS fork which uses
// faiss::cppcontrib::knowhere::IndexHNSW / HNSW types.

#pragma once

#include <cuda_runtime.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/cppcontrib/knowhere/IndexHNSW.h>
#include <faiss/cppcontrib/knowhere/impl/HNSW.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <faiss/gpu/impl/GpuHnswSearchKernel.cuh>
#include <faiss/gpu/impl/GpuHnswTypes.h>

#define GPU_HNSW_BUILD_CUDA_CHECK(expr)                               \
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

/// Extract HNSW graph layers from a Knowhere HNSW struct.
/// Template parameter HnswT can be faiss::cppcontrib::knowhere::HNSW
/// or faiss::HNSW — any type exposing this interface:
///   - neighbor_range(node, layer, &begin, &end)
///   - nb_neighbors(layer) -> int
///   - neighbors            : flat neighbor array indexed by neighbor_range
///   - levels : per-node array; levels[i] is the 1-based layer count, so
///              node i lives on layers 0..levels[i]-1 and "i is on layer L"
///              is the test levels[i] > L
///   - entry_point, max_level
/// The levels[] convention above matches faiss::HNSW (HNSW.cpp uses
/// pt_level = levels[i] - 1); a variant with different semantics would need a
/// different membership test at the levels[i] > layer check below.
template <typename HnswT>
inline void extract_hnsw_layers(
        const HnswT& hnsw,
        int64_t n_rows,
        std::vector<GpuHnswDeviceUpperLayer>& h_upper_layers,
        std::vector<uint32_t>& h_layer0_flat,
        uint32_t& entry_point,
        int& M,
        int& max_degree0,
        int& num_layers) {
    const int maxM0 = hnsw.nb_neighbors(0);
    const int maxM = hnsw.nb_neighbors(1);
    const int max_lv = hnsw.max_level;

    entry_point = static_cast<uint32_t>(hnsw.entry_point);
    M = maxM;
    max_degree0 = maxM0;
    num_layers = max_lv + 1;

    // Layer 0: dense [n_rows x maxM0]
    h_layer0_flat.assign(n_rows * maxM0, UINT32_MAX);
    for (int64_t i = 0; i < n_rows; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        uint32_t count = static_cast<uint32_t>(end - begin);
        for (uint32_t j = 0; j < count; j++) {
            auto nb = hnsw.neighbors[begin + j];
            if (nb >= 0)
                h_layer0_flat[i * maxM0 + j] = static_cast<uint32_t>(nb);
        }
    }

    // Upper layers (1..max_level): sparse [num_nodes_at_L x maxM]
    h_upper_layers.resize(max_lv);
    for (int layer = 1; layer <= max_lv; layer++) {
        auto& ul = h_upper_layers[layer - 1];
        ul.max_degree = static_cast<uint32_t>(maxM);

        std::vector<uint32_t> node_ids;
        for (int64_t i = 0; i < n_rows; i++) {
            if (hnsw.levels[i] > layer)
                node_ids.push_back(static_cast<uint32_t>(i));
        }
        ul.num_nodes = static_cast<uint32_t>(node_ids.size());

        std::vector<uint32_t> h_neighbors(ul.num_nodes * maxM, UINT32_MAX);
        std::vector<uint32_t> h_node_ids = node_ids;

        for (uint32_t idx = 0; idx < ul.num_nodes; idx++) {
            int64_t i = node_ids[idx];
            size_t begin, end;
            hnsw.neighbor_range(i, layer, &begin, &end);
            uint32_t count = static_cast<uint32_t>(end - begin);
            for (uint32_t j = 0; j < count; j++) {
                auto nb = hnsw.neighbors[begin + j];
                if (nb >= 0)
                    h_neighbors[idx * maxM + j] = static_cast<uint32_t>(nb);
            }
        }

        GPU_HNSW_BUILD_CUDA_CHECK(
                cudaMalloc(&ul.d_node_ids, ul.num_nodes * sizeof(uint32_t)));
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
                ul.d_node_ids,
                h_node_ids.data(),
                ul.num_nodes * sizeof(uint32_t),
                cudaMemcpyHostToDevice));

        GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(
                &ul.d_neighbors,
                ul.num_nodes * maxM * sizeof(uint32_t)));
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
                ul.d_neighbors,
                h_neighbors.data(),
                ul.num_nodes * maxM * sizeof(uint32_t),
                cudaMemcpyHostToDevice));
    }
}

inline void normalize_vectors(
        std::vector<float>& h_vectors,
        int64_t n_rows,
        int64_t dim) {
    for (int64_t i = 0; i < n_rows; i++) {
        float* v = h_vectors.data() + i * dim;
        float sq_norm = 0.0f;
        for (int64_t d = 0; d < dim; d++)
            sq_norm += v[d] * v[d];
        if (sq_norm > 0.0f) {
            float inv = 1.0f / std::sqrt(sq_norm);
            for (int64_t d = 0; d < dim; d++)
                v[d] *= inv;
        }
    }
}

template <typename HnswT>
inline void upload_graph_to_gpu(
        GpuHnswDeviceIndex& idx,
        const HnswT& hnsw,
        int64_t n_rows) {
    std::vector<uint32_t> h_layer0_flat;
    extract_hnsw_layers(
            hnsw,
            n_rows,
            idx.upper_layers,
            h_layer0_flat,
            idx.entry_point,
            idx.M,
            idx.max_degree0,
            idx.num_layers);

    size_t graph0_bytes =
            static_cast<size_t>(n_rows) * idx.max_degree0 * sizeof(uint32_t);
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_layer0_graph, graph0_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_layer0_graph,
            h_layer0_flat.data(),
            graph0_bytes,
            cudaMemcpyHostToDevice));

    int num_upper = static_cast<int>(idx.upper_layers.size());
    idx.num_upper_layers_built = num_upper;
    if (num_upper > 0) {
        using kernel_ptrs = hnsw_kernel::upper_layer_ptrs;
        std::vector<kernel_ptrs> h_ptrs(num_upper);
        for (int i = 0; i < num_upper; i++) {
            const auto& ul = idx.upper_layers[i];
            h_ptrs[i] = {
                    ul.d_node_ids, ul.d_neighbors, ul.num_nodes, ul.max_degree};
        }
        size_t ptrs_bytes = num_upper * sizeof(kernel_ptrs);
        GPU_HNSW_BUILD_CUDA_CHECK(
                cudaMalloc(&idx.d_upper_layer_ptrs, ptrs_bytes));
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
                idx.d_upper_layer_ptrs,
                h_ptrs.data(),
                ptrs_bytes,
                cudaMemcpyHostToDevice));
    }

}

inline void upload_fp32_dataset(
        GpuHnswDeviceIndex& idx,
        std::vector<float>& h_vectors,
        int64_t n_rows,
        bool is_cosine) {
    int64_t dim = idx.dim;
    if (is_cosine)
        normalize_vectors(h_vectors, n_rows, dim);

    size_t dataset_bytes = static_cast<size_t>(n_rows) * dim * sizeof(float);
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_dataset, dataset_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_dataset,
            h_vectors.data(),
            dataset_bytes,
            cudaMemcpyHostToDevice));
    idx.dataset_type = GpuHnswDatasetType::FP32;
}

// Upload fp16 (QT_fp16) or bf16 (QT_bf16) ScalarQuantizer codes to the GPU in
// their native 2-byte layout. faiss stores these codes row-major as raw IEEE
// half / bfloat16, which are bit-compatible with CUDA half / __nv_bfloat16, so
// the bytes are copied verbatim (no up-conversion to fp32). For cosine, inverse
// L2 norms are computed from the fp32-decoded values and applied at search time
// — mirroring the int8 path, since the stored codes are not normalized.
inline void upload_halfwidth_dataset(
        GpuHnswDeviceIndex& idx,
        const uint8_t* codes,
        const float* decoded_for_norms,
        int64_t n_rows,
        bool is_cosine,
        GpuHnswDatasetType dtype) {
    int64_t dim = idx.dim;
    size_t dataset_bytes = static_cast<size_t>(n_rows) * dim * 2;
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_dataset, dataset_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_dataset, codes, dataset_bytes, cudaMemcpyHostToDevice));
    idx.dataset_type = dtype;

    if (is_cosine && decoded_for_norms) {
        std::vector<float> h_inv_norms(n_rows);
        for (int64_t i = 0; i < n_rows; i++) {
            const float* row = decoded_for_norms + i * dim;
            float sq_norm = 0.0f;
            for (int64_t d = 0; d < dim; d++) {
                sq_norm += row[d] * row[d];
            }
            h_inv_norms[i] =
                    (sq_norm > 0.0f) ? (1.0f / std::sqrt(sq_norm)) : 0.0f;
        }
        size_t norms_bytes = static_cast<size_t>(n_rows) * sizeof(float);
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_inv_norms, norms_bytes));
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
                idx.d_inv_norms,
                h_inv_norms.data(),
                norms_bytes,
                cudaMemcpyHostToDevice));
    }
}

inline void upload_int8_dataset(
        GpuHnswDeviceIndex& idx,
        const uint8_t* codes,
        int64_t n_rows,
        bool is_cosine) {
    int64_t dim = idx.dim;
    size_t dataset_bytes = static_cast<size_t>(n_rows) * dim;

    std::vector<int8_t> signed_codes(dataset_bytes);
    for (size_t i = 0; i < dataset_bytes; i++) {
        signed_codes[i] =
                static_cast<int8_t>(static_cast<int>(codes[i]) - 128);
    }

    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_dataset, dataset_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_dataset,
            signed_codes.data(),
            dataset_bytes,
            cudaMemcpyHostToDevice));
    idx.dataset_type = GpuHnswDatasetType::INT8;

    if (is_cosine) {
        std::vector<float> h_inv_norms(n_rows);
        for (int64_t i = 0; i < n_rows; i++) {
            const int8_t* row = signed_codes.data() + i * dim;
            float sq_norm = 0.0f;
            for (int64_t d = 0; d < dim; d++) {
                float v = static_cast<float>(row[d]);
                sq_norm += v * v;
            }
            h_inv_norms[i] =
                    (sq_norm > 0.0f) ? (1.0f / std::sqrt(sq_norm)) : 0.0f;
        }
        size_t norms_bytes = static_cast<size_t>(n_rows) * sizeof(float);
        GPU_HNSW_BUILD_CUDA_CHECK(
                cudaMalloc(&idx.d_inv_norms, norms_bytes));
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
                idx.d_inv_norms,
                h_inv_norms.data(),
                norms_bytes,
                cudaMemcpyHostToDevice));
    }
}

/// Build from Knowhere's HNSW index with SQ storage.
inline std::unique_ptr<GpuHnswDeviceIndex> from_faiss_hnsw_sq(
        const faiss::cppcontrib::knowhere::IndexHNSW& hnsw_index,
        bool use_ip,
        bool is_cosine = false,
        int device = 0) {
    const auto* sq_storage =
            dynamic_cast<const faiss::IndexScalarQuantizer*>(
                    hnsw_index.storage);
    if (!sq_storage)
        throw std::runtime_error(
                "gpu_hnsw: storage is not IndexScalarQuantizer");

    int64_t n_rows = hnsw_index.ntotal;
    int64_t dim = hnsw_index.d;

    auto idx = std::make_unique<GpuHnswDeviceIndex>();
    idx->n_rows = n_rows;
    idx->dim = dim;
    idx->use_ip = use_ip;
    idx->device = device;
    idx->scratch_pool = std::make_unique<GpuHnswScratchPool>(4, device);

    auto qtype = sq_storage->sq.qtype;

    if (qtype == faiss::ScalarQuantizer::QT_8bit_direct_signed) {
        upload_int8_dataset(*idx, sq_storage->codes.data(), n_rows, is_cosine);
    } else if (
            qtype == faiss::ScalarQuantizer::QT_fp16 ||
            qtype == faiss::ScalarQuantizer::QT_bf16) {
        // Keep fp16/bf16 in their native 2-byte layout on the GPU. Decode to
        // fp32 only when cosine needs the row norms.
        GpuHnswDatasetType dtype =
                (qtype == faiss::ScalarQuantizer::QT_fp16)
                        ? GpuHnswDatasetType::FP16
                        : GpuHnswDatasetType::BF16;
        std::vector<float> decoded;
        if (is_cosine) {
            decoded.resize(static_cast<size_t>(n_rows) * dim);
            sq_storage->sa_decode(
                    n_rows, sq_storage->codes.data(), decoded.data());
        }
        upload_halfwidth_dataset(
                *idx,
                sq_storage->codes.data(),
                is_cosine ? decoded.data() : nullptr,
                n_rows,
                is_cosine,
                dtype);
    } else {
        std::vector<float> h_vectors(n_rows * dim);
        sq_storage->sa_decode(
                n_rows, sq_storage->codes.data(), h_vectors.data());
        upload_fp32_dataset(*idx, h_vectors, n_rows, is_cosine);
    }

    upload_graph_to_gpu(*idx, hnsw_index.hnsw, n_rows);
    return idx;
}

/// Build from Knowhere's HNSW index with Flat storage.
inline std::unique_ptr<GpuHnswDeviceIndex> from_faiss_hnsw_flat(
        const faiss::cppcontrib::knowhere::IndexHNSW& hnsw_index,
        bool use_ip,
        bool is_cosine = false,
        int device = 0) {
    const auto* flat_storage =
            dynamic_cast<const faiss::IndexFlat*>(hnsw_index.storage);
    if (!flat_storage)
        throw std::runtime_error("gpu_hnsw: storage is not IndexFlat");

    int64_t n_rows = hnsw_index.ntotal;
    int64_t dim = hnsw_index.d;

    // Use reconstruct_n instead of get_xb() — get_xb() can return a
    // device pointer in GPU querynode context, causing SIGSEGV when
    // accessed from CPU.
    std::vector<float> h_vectors(n_rows * dim);
    flat_storage->reconstruct_n(0, n_rows, h_vectors.data());

    auto idx = std::make_unique<GpuHnswDeviceIndex>();
    idx->n_rows = n_rows;
    idx->dim = dim;
    idx->use_ip = use_ip;
    idx->device = device;
    idx->scratch_pool = std::make_unique<GpuHnswScratchPool>(4, device);

    upload_fp32_dataset(*idx, h_vectors, n_rows, is_cosine);
    upload_graph_to_gpu(*idx, hnsw_index.hnsw, n_rows);
    return idx;
}

} // namespace gpu
} // namespace faiss
