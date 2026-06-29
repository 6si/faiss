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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace faiss {
namespace gpu {
namespace hnsw_kernel {

// ============================================================================
// Distance computation helpers (templated on dataset element type)
// ============================================================================

__device__ __forceinline__ float load_elem(const float* ptr, int idx) {
    return __ldg(&ptr[idx]);
}
__device__ __forceinline__ float load_elem(const half* ptr, int idx) {
    return __half2float(__ldg(&ptr[idx]));
}
__device__ __forceinline__ float load_elem(const int8_t* ptr, int idx) {
    return static_cast<float>(__ldg(&ptr[idx]));
}

template <typename DataT>
__device__ __forceinline__ float thread_l2_distance(
        const float* __restrict__ query,
        const DataT* __restrict__ vec,
        int dim) {
    float sum = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = query[d] - load_elem(vec, d);
        sum += diff * diff;
    }
    return sum;
}

// int8_t specialization: vectorized char4 loads (4 bytes per load vs 1)
// dim must be divisible by 4 (true for all practical dimensions)
template <>
__device__ __forceinline__ float thread_l2_distance<int8_t>(
        const float* __restrict__ query,
        const int8_t* __restrict__ vec,
        int dim) {
    float sum = 0.0f;
    int d = 0;
    for (; d + 3 < dim; d += 4) {
        char4 v4 = __ldg(reinterpret_cast<const char4*>(vec + d));
        float d0 = query[d] - static_cast<float>(v4.x);
        float d1 = query[d + 1] - static_cast<float>(v4.y);
        float d2 = query[d + 2] - static_cast<float>(v4.z);
        float d3 = query[d + 3] - static_cast<float>(v4.w);
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (; d < dim; d++) {
        float diff = query[d] - static_cast<float>(__ldg(&vec[d]));
        sum += diff * diff;
    }
    return sum;
}

template <typename DataT>
__device__ __forceinline__ float thread_ip_distance(
        const float* __restrict__ query,
        const DataT* __restrict__ vec,
        int dim) {
    float sum = 0.0f;
    for (int d = 0; d < dim; d++) {
        sum += query[d] * load_elem(vec, d);
    }
    return -sum;
}

// int8_t specialization: vectorized char4 loads
template <>
__device__ __forceinline__ float thread_ip_distance<int8_t>(
        const float* __restrict__ query,
        const int8_t* __restrict__ vec,
        int dim) {
    float sum = 0.0f;
    int d = 0;
    for (; d + 3 < dim; d += 4) {
        char4 v4 = __ldg(reinterpret_cast<const char4*>(vec + d));
        sum += query[d] * static_cast<float>(v4.x) +
                query[d + 1] * static_cast<float>(v4.y) +
                query[d + 2] * static_cast<float>(v4.z) +
                query[d + 3] * static_cast<float>(v4.w);
    }
    for (; d < dim; d++) {
        sum += query[d] * static_cast<float>(__ldg(&vec[d]));
    }
    return -sum;
}

// ============================================================================
// Warp-cooperative distance computation
// ============================================================================

__device__ __forceinline__ int select_threads_per_dist(int dim) {
    // Always use 1 thread per distance: 128 concurrent distances per block
    // vs 32 with 4-thread cooperative. For int8 dim=384, each vector is only
    // 384 bytes — fits in L1 cache. The 4x concurrency gain outweighs the
    // cooperative memory coalescing benefit at this data size.
    (void)dim;
    return 1;
}

template <typename DataT>
__device__ __forceinline__ float coop_l2_distance(
        const float* __restrict__ query,
        const DataT* __restrict__ vec,
        int dim,
        int lane_in_group,
        int threads_per_dist,
        uint32_t group_mask) {
    int chunk = dim / threads_per_dist;
    int start = lane_in_group * chunk;
    int end = (lane_in_group == threads_per_dist - 1) ? dim : start + chunk;

    float partial = 0.0f;
    for (int d = start; d < end; d++) {
        float diff = query[d] - load_elem(vec, d);
        partial += diff * diff;
    }

    for (int offset = threads_per_dist / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(group_mask, partial, offset);
    }
    return partial;
}

template <typename DataT>
__device__ __forceinline__ float coop_ip_distance(
        const float* __restrict__ query,
        const DataT* __restrict__ vec,
        int dim,
        int lane_in_group,
        int threads_per_dist,
        uint32_t group_mask) {
    int chunk = dim / threads_per_dist;
    int start = lane_in_group * chunk;
    int end = (lane_in_group == threads_per_dist - 1) ? dim : start + chunk;

    float partial = 0.0f;
    for (int d = start; d < end; d++) {
        partial += query[d] * load_elem(vec, d);
    }

    for (int offset = threads_per_dist / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(group_mask, partial, offset);
    }
    return -partial;
}

// ============================================================================
// Phase 1: Upper-layer greedy search
// ============================================================================

struct upper_layer_ptrs {
    const uint32_t* d_node_ids;
    const uint32_t* d_neighbors;
    uint32_t num_nodes;
    uint32_t max_degree;
};

__device__ __forceinline__ uint32_t
binary_search_node(const uint32_t* d_node_ids, uint32_t n, uint32_t global_id) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (__ldg(&d_node_ids[mid]) < global_id) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < n && __ldg(&d_node_ids[lo]) == global_id)
        return lo;
    return UINT32_MAX;
}

template <typename DataT>
__global__ void upper_layer_search_kernel(
        const float* __restrict__ d_queries,
        const DataT* __restrict__ d_dataset,
        const float* __restrict__ d_inv_norms,
        const upper_layer_ptrs* __restrict__ d_layer_ptrs,
        uint32_t* __restrict__ d_entry_points,
        uint32_t global_entry_point,
        int num_queries,
        int dim,
        int num_upper_layers,
        bool use_inner_product) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= num_queries)
        return;

    const float* query = d_queries + static_cast<int64_t>(warp_id) * dim;
    uint32_t current = global_entry_point;

    float best_dist;
    if (lane == 0) {
        if (use_inner_product) {
            best_dist = thread_ip_distance(
                    query,
                    d_dataset + static_cast<int64_t>(current) * dim,
                    dim);
            if (d_inv_norms)
                best_dist *= __ldg(&d_inv_norms[current]);
        } else {
            best_dist = thread_l2_distance(
                    query,
                    d_dataset + static_cast<int64_t>(current) * dim,
                    dim);
        }
    }
    best_dist = __shfl_sync(0xffffffff, best_dist, 0);

    for (int li = num_upper_layers - 1; li >= 0; li--) {
        const upper_layer_ptrs& lp = d_layer_ptrs[li];
        bool improved = true;
        while (improved) {
            improved = false;
            uint32_t local_idx =
                    binary_search_node(lp.d_node_ids, lp.num_nodes, current);
            if (local_idx == UINT32_MAX)
                break;

            uint32_t best_nbr = UINT32_MAX;
            float best_nbr_dist = best_dist;

            for (uint32_t j = lane; j < lp.max_degree; j += 32) {
                uint32_t nbr = __ldg(&lp.d_neighbors
                                              [static_cast<int64_t>(local_idx) *
                                                       lp.max_degree +
                                               j]);
                float dist = FLT_MAX;
                if (nbr != UINT32_MAX) {
                    const DataT* nbr_vec =
                            d_dataset + static_cast<int64_t>(nbr) * dim;
                    if (use_inner_product) {
                        dist = thread_ip_distance(query, nbr_vec, dim);
                        if (d_inv_norms)
                            dist *= __ldg(&d_inv_norms[nbr]);
                    } else {
                        dist = thread_l2_distance(query, nbr_vec, dim);
                    }
                }
                if (dist < best_nbr_dist) {
                    best_nbr_dist = dist;
                    best_nbr = nbr;
                }
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                float other_dist =
                        __shfl_down_sync(0xffffffff, best_nbr_dist, offset);
                uint32_t other_id =
                        __shfl_down_sync(0xffffffff, best_nbr, offset);
                if (other_dist < best_nbr_dist) {
                    best_nbr_dist = other_dist;
                    best_nbr = other_id;
                }
            }
            best_nbr_dist = __shfl_sync(0xffffffff, best_nbr_dist, 0);
            best_nbr = __shfl_sync(0xffffffff, best_nbr, 0);

            if (best_nbr != UINT32_MAX && best_nbr_dist < best_dist) {
                best_dist = best_nbr_dist;
                current = best_nbr;
                improved = true;
            }
        }
    }

    if (lane == 0) {
        d_entry_points[warp_id] = current;
    }
}

// ============================================================================
// Phase 2: Layer-0 beam search kernel with Overflow Candidate Queue (OCQ)
// ============================================================================

__device__ __forceinline__ bool bitmap_visit(
        uint32_t* bitmap,
        uint32_t node_id) {
    uint32_t word = node_id >> 5;
    uint32_t bit = 1u << (node_id & 31);
    uint32_t old = atomicOr(&bitmap[word], bit);
    return (old & bit) == 0;
}

__device__ __forceinline__ void overflow_insert(
        uint32_t* ovf_ids,
        float* ovf_dists,
        uint32_t* ovf_exp,
        int& ovf_rc,
        int overflow_ef,
        uint32_t id,
        float dist,
        uint32_t expanded) {
    if (ovf_rc >= overflow_ef && dist >= ovf_dists[ovf_rc - 1])
        return;

    int lo = 0, hi = ovf_rc;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (ovf_dists[mid] < dist)
            lo = mid + 1;
        else
            hi = mid;
    }

    int insert_end = ovf_rc < overflow_ef ? ovf_rc : overflow_ef - 1;
    for (int i = insert_end; i > lo; i--) {
        ovf_ids[i] = ovf_ids[i - 1];
        ovf_dists[i] = ovf_dists[i - 1];
        ovf_exp[i] = ovf_exp[i - 1];
    }
    ovf_ids[lo] = id;
    ovf_dists[lo] = dist;
    ovf_exp[lo] = expanded;
    if (ovf_rc < overflow_ef)
        ovf_rc++;
}

template <typename DataT>
__global__ void layer0_beam_search_kernel(
        const float* __restrict__ d_queries,
        const DataT* __restrict__ d_dataset,
        const float* __restrict__ d_inv_norms,
        const uint32_t* __restrict__ d_layer0_graph,
        const uint32_t* __restrict__ d_entry_points,
        uint32_t* __restrict__ d_visited_bitmaps,
        uint64_t* __restrict__ d_neighbors,
        float* __restrict__ d_distances,
        int num_queries,
        int N,
        int dim,
        int max_degree0,
        int k,
        int ef,
        int search_width,
        int max_iterations,
        bool use_inner_product,
        int overflow_ef,
        uint32_t* __restrict__ d_overflow_ids,
        float* __restrict__ d_overflow_dists,
        uint32_t* __restrict__ d_overflow_expanded,
        int* __restrict__ d_overflow_count) {
    int query_idx = blockIdx.x;
    if (query_idx >= num_queries)
        return;

    const float* query = d_queries + static_cast<int64_t>(query_idx) * dim;

    extern __shared__ char smem[];

    int max_staging = search_width * max_degree0;

    uint32_t* result_ids = reinterpret_cast<uint32_t*>(smem);
    float* result_dists = reinterpret_cast<float*>(result_ids + ef);
    uint32_t* is_expanded = reinterpret_cast<uint32_t*>(result_dists + ef);
    uint32_t* staging_ids = is_expanded + ef;
    float* staging_dists = reinterpret_cast<float*>(staging_ids + max_staging);
    uint32_t* parent_ids =
            reinterpret_cast<uint32_t*>(staging_dists + max_staging);
    int* meta = reinterpret_cast<int*>(parent_ids + search_width);

    int bitmap_words = (N + 31) / 32;
    uint32_t* visited_bmap =
            d_visited_bitmaps + static_cast<int64_t>(query_idx) * bitmap_words;

    uint32_t* ovf_ids = overflow_ef > 0
            ? d_overflow_ids + static_cast<int64_t>(query_idx) * overflow_ef
            : nullptr;
    float* ovf_dists = overflow_ef > 0
            ? d_overflow_dists + static_cast<int64_t>(query_idx) * overflow_ef
            : nullptr;
    uint32_t* ovf_exp = overflow_ef > 0
            ? d_overflow_expanded +
                    static_cast<int64_t>(query_idx) * overflow_ef
            : nullptr;

    for (int i = threadIdx.x; i < ef; i += blockDim.x) {
        result_ids[i] = UINT32_MAX;
        result_dists[i] = FLT_MAX;
        is_expanded[i] = 0;
    }
    if (threadIdx.x == 0) {
        meta[0] = 0;
        meta[1] = 0;
        meta[2] = 0;
        if (overflow_ef > 0)
            d_overflow_count[query_idx] = 0;
    }
    __syncthreads();

    // --- Seed with entry point ---
    uint32_t ep = d_entry_points[query_idx];
    if (threadIdx.x == 0) {
        float ep_dist;
        if (use_inner_product) {
            ep_dist = thread_ip_distance(
                    query, d_dataset + static_cast<int64_t>(ep) * dim, dim);
            if (d_inv_norms)
                ep_dist *= __ldg(&d_inv_norms[ep]);
        } else {
            ep_dist = thread_l2_distance(
                    query, d_dataset + static_cast<int64_t>(ep) * dim, dim);
        }
        result_ids[0] = ep;
        result_dists[0] = ep_dist;
        is_expanded[0] = 0;
        meta[0] = 1;
        bitmap_visit(visited_bmap, ep);
    }
    __syncthreads();

    // --- Seed with entry point's neighbors ---
    if (threadIdx.x == 0)
        meta[1] = 0;
    __syncthreads();

    for (int j = threadIdx.x; j < max_degree0; j += blockDim.x) {
        uint32_t nbr = __ldg(
                &d_layer0_graph[static_cast<int64_t>(ep) * max_degree0 + j]);
        if (nbr == UINT32_MAX || nbr >= static_cast<uint32_t>(N))
            continue;
        if (!bitmap_visit(visited_bmap, nbr))
            continue;

        const DataT* nbr_vec = d_dataset + static_cast<int64_t>(nbr) * dim;
        float dist;
        if (use_inner_product) {
            dist = thread_ip_distance(query, nbr_vec, dim);
            if (d_inv_norms)
                dist *= __ldg(&d_inv_norms[nbr]);
        } else {
            dist = thread_l2_distance(query, nbr_vec, dim);
        }

        int slot = atomicAdd(&meta[1], 1);
        if (slot < max_staging) {
            staging_ids[slot] = nbr;
            staging_dists[slot] = dist;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int staging_count = min(meta[1], max_staging);
        int rc = meta[0];

        for (int s = 0; s < staging_count; s++) {
            uint32_t sid = staging_ids[s];
            float sdist = staging_dists[s];
            if (rc >= ef && sdist >= result_dists[rc - 1])
                continue;

            int lo = 0, hi = rc;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (result_dists[mid] < sdist)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            int insert_end = rc < ef ? rc : ef - 1;
            for (int i = insert_end; i > lo; i--) {
                result_ids[i] = result_ids[i - 1];
                result_dists[i] = result_dists[i - 1];
                is_expanded[i] = is_expanded[i - 1];
            }
            result_ids[lo] = sid;
            result_dists[lo] = sdist;
            is_expanded[lo] = 0;
            if (rc < ef)
                rc++;
        }

        for (int i = 0; i < rc; i++) {
            if (result_ids[i] == ep) {
                is_expanded[i] = 1;
                break;
            }
        }
        meta[0] = rc;
    }
    __syncthreads();

    // --- Unified main loop ---
    for (int iter = 0; iter < max_iterations; iter++) {
        if (threadIdx.x == 0) {
            int num_parents = 0;
            int rc = meta[0];

            for (int i = 0; i < rc && num_parents < search_width; i++) {
                if (!is_expanded[i]) {
                    parent_ids[num_parents++] = result_ids[i];
                    is_expanded[i] = 1;
                }
            }

            if (num_parents == 0 && overflow_ef > 0) {
                int ovf_rc = d_overflow_count[query_idx];
                for (int i = 0; i < ovf_rc && num_parents < search_width;
                     i++) {
                    if (!ovf_exp[i]) {
                        parent_ids[num_parents++] = ovf_ids[i];
                        ovf_exp[i] = 1;
                    }
                }
            }
            meta[2] = num_parents;
        }
        __syncthreads();

        int num_parents = meta[2];
        if (num_parents == 0)
            break;

        if (threadIdx.x == 0)
            meta[1] = 0;
        __syncthreads();

        int total_work = num_parents * max_degree0;
        for (int wi = threadIdx.x; wi < total_work; wi += blockDim.x) {
            int parent_idx = wi / max_degree0;
            int nbr_slot = wi % max_degree0;

            uint32_t parent = parent_ids[parent_idx];
            uint32_t nbr =
                    __ldg(&d_layer0_graph
                                  [static_cast<int64_t>(parent) * max_degree0 +
                                   nbr_slot]);
            if (nbr == UINT32_MAX || nbr >= static_cast<uint32_t>(N))
                continue;
            if (!bitmap_visit(visited_bmap, nbr))
                continue;

            const DataT* nbr_vec = d_dataset + static_cast<int64_t>(nbr) * dim;
            float dist;
            if (use_inner_product) {
                dist = thread_ip_distance(query, nbr_vec, dim);
                if (d_inv_norms)
                    dist *= __ldg(&d_inv_norms[nbr]);
            } else {
                dist = thread_l2_distance(query, nbr_vec, dim);
            }

            int slot = atomicAdd(&meta[1], 1);
            if (slot < max_staging) {
                staging_ids[slot] = nbr;
                staging_dists[slot] = dist;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            int staging_count = min(meta[1], max_staging);
            int rc = meta[0];

            for (int s = 0; s < staging_count; s++) {
                uint32_t sid = staging_ids[s];
                float sdist = staging_dists[s];
                if (rc >= ef && sdist >= result_dists[rc - 1])
                    continue;

                int lo = 0, hi = rc;
                while (lo < hi) {
                    int mid = (lo + hi) / 2;
                    if (result_dists[mid] < sdist)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                int insert_end = rc < ef ? rc : ef - 1;
                for (int i = insert_end; i > lo; i--) {
                    result_ids[i] = result_ids[i - 1];
                    result_dists[i] = result_dists[i - 1];
                    is_expanded[i] = is_expanded[i - 1];
                }
                result_ids[lo] = sid;
                result_dists[lo] = sdist;
                is_expanded[lo] = 0;
                if (rc < ef)
                    rc++;
            }

            meta[0] = rc;
        }
        __syncthreads();

        // Stagnation detected when num_parents==0 (all expanded) → break above
    }

    // --- Copy top-k results to global memory ---
    int rc = meta[0];
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        if (i < rc) {
            d_neighbors[static_cast<int64_t>(query_idx) * k + i] =
                    static_cast<uint64_t>(result_ids[i]);
            d_distances[static_cast<int64_t>(query_idx) * k + i] =
                    result_dists[i];
        } else {
            d_neighbors[static_cast<int64_t>(query_idx) * k + i] = UINT64_MAX;
            d_distances[static_cast<int64_t>(query_idx) * k + i] = FLT_MAX;
        }
    }
}

inline size_t calc_layer0_smem_size(int ef, int search_width, int max_degree0) {
    int max_staging = search_width * max_degree0;

    size_t size = 0;
    size += ef * sizeof(uint32_t);
    size += ef * sizeof(float);
    size += ef * sizeof(uint32_t);
    size += max_staging * sizeof(uint32_t);
    size += max_staging * sizeof(float);
    size += search_width * sizeof(uint32_t);
    size += 3 * sizeof(int);
    return size;
}

inline size_t calc_visited_bitmap_size(int num_queries, int N) {
    int bitmap_words = (N + 31) / 32;
    return static_cast<size_t>(num_queries) * bitmap_words * sizeof(uint32_t);
}

} // namespace hnsw_kernel
} // namespace gpu
} // namespace faiss
