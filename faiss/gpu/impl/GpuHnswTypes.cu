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

#include <faiss/gpu/impl/GpuHnswTypes.h>

#include <stdexcept>
#include <string>

namespace faiss {
namespace gpu {

namespace {

inline void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
                std::string("CUDA error in scratch alloc: ") +
                cudaGetErrorString(err) + " at " + file + ":" +
                std::to_string(line));
    }
}

#define SCRATCH_CUDA_CHECK(expr) check_cuda((expr), __FILE__, __LINE__)

} // namespace

void GpuHnswSearchScratch::ensure(
        int nq,
        int k,
        int dim,
        int N,
        bool use_i8_queries) {
    size_t need_q = static_cast<size_t>(nq) * dim * sizeof(float);
    if (need_q > queries_bytes) {
        if (d_queries)
            cudaFree(d_queries);
        SCRATCH_CUDA_CHECK(cudaMalloc(&d_queries, need_q));
        queries_bytes = need_q;
    }
    size_t need_n = static_cast<size_t>(nq) * k * sizeof(uint64_t);
    if (need_n > neighbors_bytes) {
        if (d_neighbors)
            cudaFree(d_neighbors);
        SCRATCH_CUDA_CHECK(cudaMalloc(&d_neighbors, need_n));
        neighbors_bytes = need_n;
    }
    size_t need_d = static_cast<size_t>(nq) * k * sizeof(float);
    if (need_d > distances_bytes) {
        if (d_distances)
            cudaFree(d_distances);
        SCRATCH_CUDA_CHECK(cudaMalloc(&d_distances, need_d));
        distances_bytes = need_d;
    }
    if (nq > entry_cap) {
        if (d_entry_points)
            cudaFree(d_entry_points);
        SCRATCH_CUDA_CHECK(cudaMalloc(
                &d_entry_points,
                static_cast<size_t>(nq) * sizeof(uint32_t)));
        entry_cap = nq;
    }
    int bitmap_words = (N + 31) / 32;
    size_t need_bm =
            static_cast<size_t>(nq) * bitmap_words * sizeof(uint32_t);
    if (need_bm > bitmap_bytes) {
        if (d_visited_bitmaps)
            cudaFree(d_visited_bitmaps);
        SCRATCH_CUDA_CHECK(cudaMalloc(&d_visited_bitmaps, need_bm));
        bitmap_bytes = need_bm;
    }
    if (use_i8_queries) {
        size_t need_i8 = static_cast<size_t>(nq) * dim * sizeof(int8_t);
        if (need_i8 > queries_i8_bytes) {
            if (d_queries_i8)
                cudaFree(d_queries_i8);
            SCRATCH_CUDA_CHECK(cudaMalloc(&d_queries_i8, need_i8));
            queries_i8_bytes = need_i8;
        }
    }
}

GpuHnswSearchScratch::~GpuHnswSearchScratch() {
    // Set the owning device before freeing so cudaFree runs in the correct
    // context on multi-GPU systems.
    cudaSetDevice(device);
    if (d_queries)
        cudaFree(d_queries);
    if (d_neighbors)
        cudaFree(d_neighbors);
    if (d_distances)
        cudaFree(d_distances);
    if (d_entry_points)
        cudaFree(d_entry_points);
    if (d_visited_bitmaps)
        cudaFree(d_visited_bitmaps);
    if (d_queries_i8)
        cudaFree(d_queries_i8);
}

GpuHnswScratchSlot::~GpuHnswScratchSlot() {
    if (stream)
        cudaStreamDestroy(stream);
}

GpuHnswScratchPool::GpuHnswScratchPool(int pool_size, int device)
        : pool_size_(pool_size), device_(device) {}

void GpuHnswScratchPool::init_once() {
    if (initialized_)
        return;
    slots_.reserve(pool_size_);
    available_.reserve(pool_size_);
    for (int i = 0; i < pool_size_; i++) {
        auto slot = std::make_unique<GpuHnswScratchSlot>();
        slot->scratch.device = device_;
        SCRATCH_CUDA_CHECK(cudaSetDevice(device_));
        SCRATCH_CUDA_CHECK(
                cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking));
        available_.push_back(slot.get());
        slots_.push_back(std::move(slot));
    }
    initialized_ = true;
}

GpuHnswScratchSlot* GpuHnswScratchPool::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    init_once();
    cv_.wait(lock, [this] { return !available_.empty(); });
    auto* slot = available_.back();
    available_.pop_back();
    return slot;
}

void GpuHnswScratchPool::release(GpuHnswScratchSlot* slot) {
    // Safety net: ensure all GPU work on this slot's stream has completed
    // before returning it to the pool. Callers (searchHost / searchImpl_)
    // already synchronize before release, so this is normally a no-op; it
    // guards a future caller that skips its own sync, whose in-flight buffers
    // could otherwise be freed+realloced by the next acquirer's ensure()
    // (use-after-free).
    cudaSetDevice(device_);
    cudaStreamSynchronize(slot->stream);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        available_.push_back(slot);
    }
    cv_.notify_one();
}

GpuHnswDeviceIndex::~GpuHnswDeviceIndex() {
    // Set the owning device before freeing so cudaFree runs in the correct
    // context on multi-GPU systems.
    cudaSetDevice(device);
    if (d_dataset)
        cudaFree(d_dataset);
    if (d_inv_norms)
        cudaFree(d_inv_norms);
    if (d_layer0_graph)
        cudaFree(d_layer0_graph);
    for (auto& ul : upper_layers) {
        if (ul.d_node_ids)
            cudaFree(ul.d_node_ids);
        if (ul.d_neighbors)
            cudaFree(ul.d_neighbors);
    }
    if (d_upper_layer_ptrs)
        cudaFree(d_upper_layer_ptrs);
}

} // namespace gpu
} // namespace faiss
