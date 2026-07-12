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
#include <faiss/cppcontrib/knowhere/IndexCosine.h>
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

// GpuHnswUploadFaultInjection (used below) lives in GpuHnswTypes.h so it is
// reachable from host-compiled unit tests without pulling in device kernels.
#define GPU_HNSW_BUILD_CUDA_CHECK(expr)                                  \
    do {                                                                 \
        if (faiss::gpu::GpuHnswUploadFaultInjection::should_fail()) {     \
            throw std::runtime_error(                                    \
                    std::string("CUDA error (injected): simulated ") +   \
                    "upload failure at " + __FILE__ + ":" +              \
                    std::to_string(__LINE__));                           \
        }                                                                \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                         \
            throw std::runtime_error(                                    \
                    std::string("CUDA error: ") +                        \
                    cudaGetErrorString(_e) + " at " + __FILE__ + ":" +   \
                    std::to_string(__LINE__));                           \
        }                                                                \
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
                node_ids.data(),
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

// Upload precomputed per-row inverse L2 norms to the device. These must be the
// norms of the *original* input vectors as recorded by the CPU cosine index
// (HasInverseL2Norms::get_inverse_l2_norms()), NOT norms recomputed from
// lossily-decoded codes — otherwise cosine scores and graph traversal diverge
// from the CPU index for lossy SQ (SQ8 / fp16 / bf16). The search kernel
// computes score = IP(query, decoded_db) * inv_norm[row], mirroring the CPU
// WithCosineNormDistanceComputer.
inline void upload_inv_norms(
        GpuHnswDeviceIndex& idx,
        const float* inv_norms,
        int64_t n_rows) {
    size_t norms_bytes = static_cast<size_t>(n_rows) * sizeof(float);
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_inv_norms, norms_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_inv_norms, inv_norms, norms_bytes, cudaMemcpyHostToDevice));
}

inline void upload_fp32_dataset(
        GpuHnswDeviceIndex& idx,
        std::vector<float>& h_vectors,
        int64_t n_rows,
        bool is_cosine,
        const float* stored_inv_norms = nullptr) {
    int64_t dim = idx.dim;
    // Flat cosine (stored_inv_norms == nullptr): the stored vectors are the
    // exact originals, so normalizing them in place yields cosine via plain
    // inner product. Lossy-SQ cosine decoded to fp32 (stored_inv_norms != null):
    // keep the decoded vectors un-normalized and apply the CPU index's original
    // inverse norms at search time, matching CPU semantics for lossy codes.
    if (is_cosine && stored_inv_norms == nullptr)
        normalize_vectors(h_vectors, n_rows, dim);

    size_t dataset_bytes = static_cast<size_t>(n_rows) * dim * sizeof(float);
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_dataset, dataset_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_dataset,
            h_vectors.data(),
            dataset_bytes,
            cudaMemcpyHostToDevice));
    idx.dataset_type = GpuHnswDatasetType::FP32;

    if (is_cosine && stored_inv_norms != nullptr)
        upload_inv_norms(idx, stored_inv_norms, n_rows);
}

// Upload fp16 (QT_fp16) or bf16 (QT_bf16) ScalarQuantizer codes to the GPU in
// their native 2-byte layout. faiss stores these codes row-major as raw IEEE
// half / bfloat16, which are bit-compatible with CUDA half / __nv_bfloat16, so
// the bytes are copied verbatim (no up-conversion to fp32). For cosine, the CPU
// index's original inverse L2 norms are applied at search time — mirroring the
// int8 path, since the stored codes are not normalized.
inline void upload_halfwidth_dataset(
        GpuHnswDeviceIndex& idx,
        const uint8_t* codes,
        int64_t n_rows,
        bool is_cosine,
        GpuHnswDatasetType dtype,
        const float* stored_inv_norms) {
    int64_t dim = idx.dim;
    size_t dataset_bytes = static_cast<size_t>(n_rows) * dim * 2;
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_dataset, dataset_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(
            idx.d_dataset, codes, dataset_bytes, cudaMemcpyHostToDevice));
    idx.dataset_type = dtype;

    // The stored fp16/bf16 codes are not normalized, so cosine needs the CPU
    // index's original inverse norms (not norms recomputed from the lossy
    // decoded values). Mirrors the int8 path.
    if (is_cosine)
        upload_inv_norms(idx, stored_inv_norms, n_rows);
}

inline void upload_int8_dataset(
        GpuHnswDeviceIndex& idx,
        const uint8_t* codes,
        int64_t n_rows,
        bool is_cosine,
        const float* stored_inv_norms) {
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

    // Apply the CPU index's original inverse norms for cosine. For
    // QT_8bit_direct_signed the codes are the original data (lossless), so these
    // match a recompute; using the stored norms keeps all cosine paths uniform
    // and bit-exact with the CPU index.
    if (is_cosine)
        upload_inv_norms(idx, stored_inv_norms, n_rows);
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

    // For cosine, use the CPU index's inverse L2 norms computed from the
    // *original* input vectors (HasInverseL2Norms::get_inverse_l2_norms()).
    // Recomputing norms from lossily-decoded/quantized codes would diverge from
    // the CPU index's scores and graph traversal for lossy SQ.
    const float* stored_inv_norms = nullptr;
    if (is_cosine) {
        // The cosine norms live on the SQ storage (IndexScalarQuantizerCosine),
        // which implements HasInverseL2Norms — not on the outer IndexHNSW.
        const auto* cos = dynamic_cast<
                const faiss::cppcontrib::knowhere::HasInverseL2Norms*>(
                sq_storage);
        if (!cos || !cos->get_inverse_l2_norms())
            throw std::runtime_error(
                    "gpu_hnsw: cosine SQ index missing inverse L2 norms");
        stored_inv_norms = cos->get_inverse_l2_norms();
    }

    if (qtype == faiss::ScalarQuantizer::QT_8bit_direct_signed) {
        upload_int8_dataset(
                *idx, sq_storage->codes.data(), n_rows, is_cosine,
                stored_inv_norms);
    } else if (
            qtype == faiss::ScalarQuantizer::QT_fp16 ||
            qtype == faiss::ScalarQuantizer::QT_bf16) {
        // Keep fp16/bf16 in their native 2-byte layout on the GPU.
        GpuHnswDatasetType dtype =
                (qtype == faiss::ScalarQuantizer::QT_fp16)
                        ? GpuHnswDatasetType::FP16
                        : GpuHnswDatasetType::BF16;
        upload_halfwidth_dataset(
                *idx,
                sq_storage->codes.data(),
                n_rows,
                is_cosine,
                dtype,
                stored_inv_norms);
    } else {
        // Other SQ types (e.g. unsigned QT_8bit / SQ8) are decoded to fp32 and
        // uploaded un-normalized; the stored original inverse norms are applied
        // in the kernel, matching the CPU cosine computer for lossy codes.
        std::vector<float> h_vectors(n_rows * dim);
        sq_storage->sa_decode(
                n_rows, sq_storage->codes.data(), h_vectors.data());
        upload_fp32_dataset(
                *idx, h_vectors, n_rows, is_cosine, stored_inv_norms);
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
