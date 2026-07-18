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

#include <memory>
#include <mutex>

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
};

/// GPU implementation of HNSW search.
///
/// This index type does NOT build an HNSW graph on GPU — it takes a
/// CPU-built faiss::IndexHNSW (Flat or SQ storage), converts the graph
/// to a GPU-friendly dense format, and runs the search on GPU using a
/// parallel beam search kernel.
///
/// Supports L2, inner product, and cosine metrics.
/// Supports float32, fp16, bf16, and int8 (QT_8bit_direct_signed) data.
/// Low-precision formats (int8/fp16/bf16) are kept in their native on-device
/// byte layout and up-converted to fp32 per element inside the search kernel.
///
/// Two search entry points exist:
///   - searchHost(): the path Knowhere uses. Host in/out pointers, a single
///     device sync at the end. Preferred for production.
///   - searchImpl_(): the faiss-standard GpuIndex::search() override. It does a
///     D2H copy of labels, a CPU uint64->idx_t conversion, then an H2D copy
///     back, with a stream sync on each side. Correct but not latency-optimal;
///     a GPU-side label-conversion kernel to avoid the round-trip is a
///     documented follow-up. Knowhere does not use this path.
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
    /// (QT_8bit_direct_signed for INT8, QT_fp16/QT_bf16 kept native, or
    /// dequantized for other SQ types).
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
    /// This is the preferred entry point (single device sync); prefer it
    /// over the searchImpl_/GpuIndex::search() path, which round-trips the
    /// labels D2H then H2D.
    ///
    /// Distance convention: inner-product and cosine metrics return the true
    /// similarity score (larger == more similar), matching faiss's
    /// METRIC_INNER_PRODUCT contract. Internally the kernel keeps a negated
    /// (smaller == more similar) score for a uniform min-first ordering and
    /// negates it back on copy-out. L2 distances are returned as-is.
    void searchHost(
            idx_t n,
            const float* x_host,
            int k,
            float* distances_host,
            idx_t* labels_host,
            const GpuHnswSearchParams& params) const;

    /// Search with int8 host query vectors using the native DP4A path.
    /// The queries are the caller's signed int8 values and are uploaded
    /// verbatim — no bias/shift is applied, matching the dataset upload which
    /// already reverses FAISS's +128 SQ bias. When dim % 4 != 0 (DP4A requires
    /// groups of four int8 lanes) this transparently falls back to the fp32
    /// searchHost() path. Same IP/cosine distance convention as searchHost()
    /// (true similarity returned). All input/output pointers must be host
    /// memory.
    void searchHostInt8(
            idx_t n,
            const int8_t* x_host,
            int k,
            float* distances_host,
            idx_t* labels_host,
            const GpuHnswSearchParams& params) const;

    /// Metric interpretation captured at copyFrom()/copyFromWithMetric() time
    /// from the FAISS index type, so callers can post-process results (cosine
    /// query normalization, IP handling) without re-deriving the metric from a
    /// per-search config — the config may omit metric_type and default to L2,
    /// which would silently mishandle a cosine/IP index.
    bool isCosine() const {
        return is_cosine_;
    }
    bool useInnerProduct() const {
        return use_ip_;
    }

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

    // Metric interpretation, set at copy time (see isCosine()/useInnerProduct()).
    bool is_cosine_ = false;
    bool use_ip_ = false;
};

} // namespace gpu
} // namespace faiss
