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

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/impl/GpuHnswTypes.h>

#include <atomic>
#include <memory>

namespace faiss {

namespace cppcontrib {
namespace knowhere {
struct IndexHNSW;
} // namespace knowhere
} // namespace cppcontrib

namespace gpu {

struct GpuHnswDeviceIndex;

struct GpuIndexHNSWConfig : public GpuIndexConfig {};

struct SearchParametersGpuHNSW : SearchParameters {
    /// Search ef — number of candidates maintained during search.
    /// Higher values improve recall at the cost of speed.
    int ef = 200;

    /// Number of candidates to expand per iteration.
    int search_width = 4;

    /// Maximum search iterations (0 = auto).
    int max_iterations = 0;

    /// Thread block size (0 = auto, default 128).
    int thread_block_size = 0;

    /// Overflow queue factor: overflow_ef = overflow_factor * ef.
    int overflow_factor = 2;
};

/// GPU implementation of HNSW search.
///
/// This index type does NOT build an HNSW graph on GPU — it takes a
/// CPU-built faiss::IndexHNSW (Flat or SQ storage), converts the graph
/// to a GPU-friendly dense format, and runs the search on GPU using an
/// Overflow Candidate Queue (OCQ) beam search kernel.
///
/// Supports L2, inner product, and cosine metrics.
/// Supports float32 and int8 (QT_8bit_direct_signed) data.
struct GpuIndexHNSW : public GpuIndex {
   public:
    GpuIndexHNSW(
            GpuResourcesProvider* provider,
            int dims,
            faiss::MetricType metric = faiss::METRIC_L2,
            GpuIndexHNSWConfig config = GpuIndexHNSWConfig());

    ~GpuIndexHNSW() override;

    /// Load an HNSW index from CPU to GPU.
    /// The CPU index must have been built and trained already.
    /// Supports IndexHNSWFlat (float32) and IndexHNSWSQ
    /// (QT_8bit_direct_signed for INT8, or dequantized for other SQ types).
    void copyFrom(const faiss::cppcontrib::knowhere::IndexHNSW* index);

    /// Like copyFrom(), but with the metric interpretation supplied by the
    /// caller instead of being detected from the index type.
    /// \param use_ip     treat the metric as inner product
    /// \param is_cosine  the storage carries cosine (inverse L2) norms
    void copyFromWithMetric(
            const faiss::cppcontrib::knowhere::IndexHNSW* index,
            bool use_ip,
            bool is_cosine);

    void reset() override;

    /// Set search parameters directly, bypassing SearchParameters.
    /// Mutex-guarded. The params are sticky: once set they apply to every
    /// subsequent search() until overwritten by another setSearchParams()
    /// call. Prefer passing SearchParametersGpuHNSW per-search (or using
    /// searchHost) when different concurrent searches need different params.
    void setSearchParams(const GpuHnswSearchParams& params) const;

    /// Search with host pointers directly, bypassing GpuIndex::search.
    /// All input/output pointers must be host memory.
    /// This avoids the GpuIndex::search_ex temp allocation chain
    /// which can cause SIGSEGV from pointer lifetime issues.
    void searchHost(
            idx_t n,
            const float* x_host,
            int k,
            float* distances_host,
            idx_t* labels_host,
            const GpuHnswSearchParams& params) const;

   protected:
    bool addImplRequiresIDs_() const override;

    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;

    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* search_params) const override;

   private:
    GpuIndexHNSWConfig hnswConfig_;

    std::unique_ptr<GpuHnswDeviceIndex> deviceIndex_;

    mutable std::mutex searchParamsMutex_;
    mutable GpuHnswSearchParams directSearchParams_;
    mutable bool hasDirectSearchParams_ = false;
};

} // namespace gpu
} // namespace faiss
