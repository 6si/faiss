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

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/cppcontrib/knowhere/IndexCosine.h>
#include <faiss/cppcontrib/knowhere/IndexHNSW.h>
#include <faiss/gpu/GpuIndexHNSW.h>
#include <faiss/gpu/impl/GpuHnswTypes.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/GpuHnswBuild.cuh>
#include <faiss/gpu/impl/GpuHnswSearch.cuh>

#include <cstdio>
#include <memory>
#include <stdexcept>

namespace faiss {
namespace gpu {

GpuIndexHNSW::GpuIndexHNSW(
        GpuResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        GpuIndexHNSWConfig config)
        : GpuIndex(provider->getResources(), dims, metric, 0.0f, config),
          hnswConfig_(config) {
    this->is_trained = false;
}

GpuIndexHNSW::~GpuIndexHNSW() = default;

void GpuIndexHNSW::copyFrom(
        const faiss::cppcontrib::knowhere::IndexHNSW* index) {
    FAISS_THROW_IF_NOT_MSG(index, "index must not be null");
    FAISS_THROW_IF_NOT_MSG(index->ntotal > 0, "index must not be empty");

    DeviceScope scope(config_.device);

    this->d = index->d;
    this->metric_type = index->metric_type;
    this->ntotal = index->ntotal;

    // Detect cosine from index type (IndexHNSWFlatCosine / IndexHNSWSQCosine
    // implement HasInverseL2Norms).
    bool is_cosine =
            dynamic_cast<const faiss::cppcontrib::knowhere::HasInverseL2Norms*>(
                    index) != nullptr;
    bool use_ip =
            is_cosine || (index->metric_type == faiss::METRIC_INNER_PRODUCT);

    if (dynamic_cast<const faiss::IndexScalarQuantizer*>(index->storage)) {
        deviceIndex_ = from_faiss_hnsw_sq(*index, use_ip, is_cosine);
    } else {
        deviceIndex_ = from_faiss_hnsw_flat(*index, use_ip, is_cosine);
    }

    this->is_trained = true;
}

void GpuIndexHNSW::copyFromWithMetric(
        const faiss::cppcontrib::knowhere::IndexHNSW* index,
        bool use_ip,
        bool is_cosine) {
    FAISS_THROW_IF_NOT_MSG(index, "index must not be null");
    FAISS_THROW_IF_NOT_MSG(index->ntotal > 0, "index must not be empty");

    DeviceScope scope(config_.device);

    this->d = index->d;
    this->metric_type = index->metric_type;
    this->ntotal = index->ntotal;

    if (dynamic_cast<const faiss::IndexScalarQuantizer*>(index->storage)) {
        deviceIndex_ = from_faiss_hnsw_sq(*index, use_ip, is_cosine);
    } else {
        deviceIndex_ = from_faiss_hnsw_flat(*index, use_ip, is_cosine);
    }

    this->is_trained = true;
}

void GpuIndexHNSW::reset() {
    deviceIndex_.reset();
    this->ntotal = 0;
    this->is_trained = false;
}

void GpuIndexHNSW::setSearchParams(const GpuHnswSearchParams& params) const {
    std::lock_guard<std::mutex> lock(searchParamsMutex_);
    directSearchParams_ = params;
    hasDirectSearchParams_ = true;
}

bool GpuIndexHNSW::addImplRequiresIDs_() const {
    return false;
}

void GpuIndexHNSW::addImpl_(idx_t, const float*, const idx_t*) {
    FAISS_THROW_MSG(
            "GpuIndexHNSW does not support add(). "
            "Build on CPU with IndexHNSW, then call copyFrom().");
}

void GpuIndexHNSW::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* search_params) const {
    FAISS_THROW_IF_NOT_MSG(
            this->is_trained && deviceIndex_,
            "Index not loaded. Call copyFrom() first.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n must be > 0");

    auto& idx = *deviceIndex_;

    GpuHnswSearchParams sp;
    bool got_params = false;

    // Prefer direct params set via setSearchParams() — avoids dynamic_cast.
    {
        std::lock_guard<std::mutex> lock(searchParamsMutex_);
        if (hasDirectSearchParams_) {
            sp = directSearchParams_;
            hasDirectSearchParams_ = false;
            got_params = true;
        }
    }

    // Fallback: try dynamic_cast from SearchParameters.
    if (!got_params && search_params) {
        auto* params =
                dynamic_cast<const SearchParametersGpuHNSW*>(search_params);
        if (params) {
            sp.ef = params->ef;
            sp.search_width = params->search_width;
            sp.max_iterations = params->max_iterations;
            sp.thread_block_size = params->thread_block_size;
            sp.overflow_factor = params->overflow_factor;
            got_params = true;
        }
    }

    ScratchPoolGuard guard(*idx.scratch_pool);
    auto* slot = guard.get();
    auto& sc = slot->scratch;
    cudaStream_t stream = slot->stream;

    int nq = static_cast<int>(n);
    int dim = static_cast<int>(idx.dim);
    int overflow_ef = sp.overflow_factor * sp.ef;
    sc.ensure(nq, k, dim, static_cast<int>(idx.n_rows), overflow_ef);

    // D2D: query vectors (GpuIndex::search passes device pointers)
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            sc.d_queries,
            x,
            static_cast<size_t>(nq) * dim * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));

    gpu_hnsw_search(stream, sp, idx, nq, k);

    // D2D: distances (output is a device pointer from GpuIndex::search)
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            distances,
            sc.d_distances,
            static_cast<size_t>(nq) * k * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));

    // Labels: D2H stage (uint64_t→idx_t conversion), then H2D back
    auto tmp = std::make_unique<uint64_t[]>(nq * k);
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            tmp.get(),
            sc.d_neighbors,
            static_cast<size_t>(nq) * k * sizeof(uint64_t),
            cudaMemcpyDeviceToHost,
            stream));
    GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto h_labels = std::make_unique<idx_t[]>(nq * k);
    for (int i = 0; i < nq * k; i++) {
        h_labels[i] = (tmp[i] == UINT64_MAX) ? -1 : static_cast<idx_t>(tmp[i]);
    }

    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            labels,
            h_labels.get(),
            static_cast<size_t>(nq) * k * sizeof(idx_t),
            cudaMemcpyHostToDevice,
            stream));
    GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void GpuIndexHNSW::searchHost(
        idx_t n,
        const float* x_host,
        int k,
        float* distances_host,
        idx_t* labels_host,
        const GpuHnswSearchParams& sp) const {
    FAISS_THROW_IF_NOT_MSG(
            this->is_trained && deviceIndex_,
            "Index not loaded. Call copyFrom() first.");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n must be > 0");

    GPU_HNSW_CUDA_CHECK(cudaSetDevice(config_.device));
    DeviceScope scope(config_.device);
    auto& idx = *deviceIndex_;

    ScratchPoolGuard guard(*idx.scratch_pool);
    auto* slot = guard.get();
    auto& sc = slot->scratch;
    cudaStream_t stream = slot->stream;

    int nq = static_cast<int>(n);
    int dim = static_cast<int>(idx.dim);
    int overflow_ef = sp.overflow_factor * sp.ef;
    sc.ensure(nq, k, dim, static_cast<int>(idx.n_rows), overflow_ef);

    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            sc.d_queries,
            x_host,
            static_cast<size_t>(nq) * dim * sizeof(float),
            cudaMemcpyDefault,
            stream));

    gpu_hnsw_search(stream, sp, idx, nq, k);

    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            distances_host,
            sc.d_distances,
            static_cast<size_t>(nq) * k * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));

    auto tmp = std::make_unique<uint64_t[]>(nq * k);
    GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(
            tmp.get(),
            sc.d_neighbors,
            static_cast<size_t>(nq) * k * sizeof(uint64_t),
            cudaMemcpyDeviceToHost,
            stream));

    GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < nq * k; i++) {
        labels_host[i] =
                (tmp[i] == UINT64_MAX) ? -1 : static_cast<idx_t>(tmp[i]);
    }
}

} // namespace gpu
} // namespace faiss
