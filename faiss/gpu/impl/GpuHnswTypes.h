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

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace faiss {
namespace gpu {

struct GpuHnswSearchParams {
    int ef = 200;
    int search_width = 4;
    int max_iterations = 0;
    int thread_block_size = 0;
};

struct GpuHnswDeviceUpperLayer {
    uint32_t* d_node_ids = nullptr;
    uint32_t* d_neighbors = nullptr;
    uint32_t num_nodes = 0;
    uint32_t max_degree = 0;
};

struct GpuHnswSearchScratch {
    float* d_queries = nullptr;
    uint64_t* d_neighbors = nullptr;
    float* d_distances = nullptr;
    uint32_t* d_entry_points = nullptr;
    uint32_t* d_visited_bitmaps = nullptr;

    size_t queries_bytes = 0;
    size_t neighbors_bytes = 0;
    size_t distances_bytes = 0;
    int entry_cap = 0;
    size_t bitmap_bytes = 0;

    // Device this scratch's allocations live on; used to set the CUDA device
    // context before freeing in the destructor (multi-GPU correctness).
    int device = 0;

    void ensure(int nq, int k, int dim, int N);

    ~GpuHnswSearchScratch();

    GpuHnswSearchScratch() = default;
    GpuHnswSearchScratch(const GpuHnswSearchScratch&) = delete;
    GpuHnswSearchScratch& operator=(const GpuHnswSearchScratch&) = delete;
};

/// One slot in the scratch pool: a scratch buffer + its own CUDA stream.
struct GpuHnswScratchSlot {
    GpuHnswSearchScratch scratch;
    cudaStream_t stream = nullptr;

    ~GpuHnswScratchSlot();
    GpuHnswScratchSlot() = default;
    GpuHnswScratchSlot(const GpuHnswScratchSlot&) = delete;
    GpuHnswScratchSlot& operator=(const GpuHnswScratchSlot&) = delete;
};

/// Pool of scratch buffers allowing concurrent GPU searches on the same segment.
/// Each acquire() returns a slot with its own scratch + CUDA stream.
/// Callers block if all slots are in use.
class GpuHnswScratchPool {
 public:
    /// Create a pool. CUDA streams are allocated lazily on first acquire().
    explicit GpuHnswScratchPool(int pool_size = 4, int device = 0);
    ~GpuHnswScratchPool() = default;

    GpuHnswScratchPool(const GpuHnswScratchPool&) = delete;
    GpuHnswScratchPool& operator=(const GpuHnswScratchPool&) = delete;

    /// Acquire a scratch slot (blocks until one is available).
    GpuHnswScratchSlot* acquire();
    /// Release a previously acquired scratch slot back to the pool.
    void release(GpuHnswScratchSlot* slot);

    int pool_size() const { return pool_size_; }

 private:
    void init_once();

    std::mutex mutex_;
    std::condition_variable cv_;
    int pool_size_;
    int device_;
    bool initialized_ = false;
    std::vector<std::unique_ptr<GpuHnswScratchSlot>> slots_;
    std::vector<GpuHnswScratchSlot*> available_;
};

/// RAII guard: acquires a scratch slot on construction, releases on destruction.
class ScratchPoolGuard {
 public:
    ScratchPoolGuard(GpuHnswScratchPool& pool)
            : pool_(pool), slot_(pool.acquire()) {}
    ~ScratchPoolGuard() { pool_.release(slot_); }
    GpuHnswScratchSlot* get() const { return slot_; }

    ScratchPoolGuard(const ScratchPoolGuard&) = delete;
    ScratchPoolGuard& operator=(const ScratchPoolGuard&) = delete;

 private:
    GpuHnswScratchPool& pool_;
    GpuHnswScratchSlot* slot_;
};

struct GpuHnswDeviceIndex {
    void* d_dataset = nullptr;
    bool dataset_int8 = false;
    float* d_inv_norms = nullptr;
    uint32_t* d_layer0_graph = nullptr;
    std::vector<GpuHnswDeviceUpperLayer> upper_layers;

    int64_t n_rows = 0;
    int64_t dim = 0;
    uint32_t entry_point = 0;
    int num_layers = 0;
    int M = 0;
    int max_degree0 = 0;
    bool use_ip = false;

    void* d_upper_layer_ptrs = nullptr;
    int num_upper_layers_built = 0;

    // Device this index's allocations live on; used to set the CUDA device
    // context before freeing in the destructor (multi-GPU correctness).
    int device = 0;

    mutable std::unique_ptr<GpuHnswScratchPool> scratch_pool;

    ~GpuHnswDeviceIndex();
};

} // namespace gpu
} // namespace faiss
