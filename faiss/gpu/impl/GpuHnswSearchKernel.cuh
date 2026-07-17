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
__device__ __forceinline__ float load_elem(const __nv_bfloat16* ptr, int idx) {
    return __bfloat162float(__ldg(&ptr[idx]));
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
#pragma unroll 8
    for (int d = 0; d < dim; d++) {
        float diff = query[d] - load_elem(vec, d);
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
#pragma unroll 8
    for (int d = 0; d < dim; d++) {
        sum += query[d] * load_elem(vec, d);
    }
    return -sum;
}

// Native int8 inner-product distance using DP4A (4 int8 MADs/cycle, SM_61+).
// Both query and vec must point at int8 data reinterpreted as packed int32,
// i.e. dim4 == dim / 4 (dim must be a multiple of 4). Returns the negated dot
// product to match thread_ip_distance()'s convention (smaller == closer).
__device__ __forceinline__ float thread_ip_distance_dp4a(
        const int32_t* __restrict__ query_packed,
        const int32_t* __restrict__ vec_packed,
        int dim4) {
    int sum = 0;
#pragma unroll 8
    for (int d = 0; d < dim4; d++) {
        sum = __dp4a(query_packed[d], vec_packed[d], sum);
    }
    return static_cast<float>(-sum);
}

// Smallest power of two >= x (x >= 1). Used to pad the staging capacity so the
// bitonic sort (which requires a power-of-two capacity) never rejects graphs
// whose search_width * max_degree0 is not already a power of two.
__host__ __device__ __forceinline__ int next_pow2_int(int x) {
    int p = 1;
    while (p < x)
        p <<= 1;
    return p;
}

// Padded staging capacity: power-of-two >= search_width * max_degree0.
__host__ __device__ __forceinline__ int padded_staging_capacity(
        int search_width,
        int max_degree0) {
    return next_pow2_int(search_width * max_degree0);
}

// Unified layer-0 distance: DP4A for native int8+IP, generic fp32 path
// otherwise. The branch is resolved at compile time so only the relevant
// distance function is instantiated for each specialization.
template <typename DataT, typename QueryT, bool USE_DP4A>
__device__ __forceinline__ float layer0_distance(
        const QueryT* __restrict__ query,
        const DataT* __restrict__ d_dataset,
        const float* __restrict__ d_inv_norms,
        uint32_t id,
        int dim,
        int dim4,
        bool use_inner_product) {
    if constexpr (USE_DP4A) {
        const int32_t* vec_packed = reinterpret_cast<const int32_t*>(
                d_dataset + static_cast<int64_t>(id) * dim);
        float dist = thread_ip_distance_dp4a(
                reinterpret_cast<const int32_t*>(query), vec_packed, dim4);
        if (d_inv_norms)
            dist *= __ldg(&d_inv_norms[id]);
        return dist;
    } else {
        const DataT* vec = d_dataset + static_cast<int64_t>(id) * dim;
        float dist;
        if (use_inner_product) {
            dist = thread_ip_distance(query, vec, dim);
            if (d_inv_norms)
                dist *= __ldg(&d_inv_norms[id]);
        } else {
            dist = thread_l2_distance(query, vec, dim);
        }
        return dist;
    }
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
                uint32_t nbr = lp.d_neighbors
                                              [static_cast<int64_t>(local_idx) *
                                                       lp.max_degree +
                                               j];
                float dist = FLT_MAX;
                if (nbr != UINT32_MAX) {
                    const DataT* nbr_vec =
                            d_dataset + static_cast<int64_t>(nbr) * dim;
                    if (use_inner_product) {
                        dist = thread_ip_distance(query, nbr_vec, dim);
                        if (d_inv_norms)
                            dist *= d_inv_norms[nbr];
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
// Phase 1: Parallel merge helpers (bitonic sort + parallel merge)
// ============================================================================

__device__ __forceinline__ void bitonic_sort_staging(
        uint32_t* ids, float* dists, int active_count, int capacity) {
    int i = threadIdx.x;
    if (i < capacity && i >= active_count) {
        ids[i] = UINT32_MAX;
        dists[i] = FLT_MAX;
    }
    __syncthreads();

    for (int k = 2; k <= capacity; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = i ^ j;
            // Compare-exchange in two barrier-separated phases. Each thread
            // owns writes to slot i only, but it READS slot `partner` — which
            // another thread owns and may overwrite in the SAME phase. When the
            // partner is in a different warp (j >= warpSize) the read and that
            // write are unsynchronized -> a shared-memory hazard (racecheck
            // flagged ~1.6M in the half kernel). So: (1) every thread reads its
            // pair into registers and decides whether to swap, (2) __syncthreads
            // so all reads complete before any write, (3) threads that must swap
            // write their own slot, (4) __syncthreads before the next phase's
            // reads. Loop bounds (k, j) are uniform across the block, so every
            // thread — including i >= capacity — reaches every barrier.
            float dp = 0.0f;
            uint32_t ip = 0u;
            bool do_swap = false;
            if (i < capacity && partner < capacity) {
                float di = dists[i];
                dp = dists[partner];
                uint32_t ii = ids[i];
                ip = ids[partner];
                // Ascending sort when the direction bit (i & k) is 0.
                // In ascending order: lower index should hold the smaller value.
                bool ascending = ((i & k) == 0);
                bool i_is_lower = (i < partner);
                // i should hold the smaller value iff: ascending and i<partner,
                //                                  or descending and i>partner.
                bool want_smaller = ascending ? i_is_lower : !i_is_lower;
                bool i_is_smaller = (di < dp) || (di == dp && ii < ip);
                do_swap = (want_smaller != i_is_smaller);
            }
            __syncthreads();
            if (do_swap) {
                dists[i] = dp;
                ids[i] = ip;
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void parallel_merge_into_result(
        uint32_t* result_ids,
        float* result_dists,
        uint32_t* is_expanded,
        uint32_t* staging_ids,
        float* staging_dists,
        uint32_t* merged_ids,
        float* merged_dists,
        uint32_t* merged_expanded,
        int* meta,
        int ef,
        int max_staging) {
    int sc = min(meta[1], max_staging);
    int rc = meta[0];

    bitonic_sort_staging(staging_ids, staging_dists, sc, max_staging);

    int T = rc + sc;
    int merge_limit = min(T, ef);

    for (int pos = threadIdx.x; pos < merge_limit; pos += blockDim.x) {
        // Binary search for si: number of staging elements ranked before pos.
        // Valid range: si in [max(0,pos-rc), min(pos,sc)].
        // Half-open: lo inclusive, hi exclusive = min(pos,sc)+1.
        // ri = pos-si is always >= 0 within this range, but after exit si
        // may equal min(pos,sc)+1 (all staging ranked first); guard ri reads.
        int lo = max(0, pos - rc);
        int hi = min(pos, sc) + 1;

        while (lo < hi) {
            int mid = (lo + hi) / 2;
            // Merge-path diagonal test: staging[mid] is taken ahead of the
            // result element it would displace, which is result[pos-mid-1]
            // (NOT result[pos-mid]). Getting this index wrong mis-partitions
            // every merge (empirically ~4% recall) and can promote a padded
            // UINT32_MAX sentinel into the result list, which is later used as
            // a graph/dataset index -> illegal memory access.
            int ri_mid = pos - mid - 1;
            bool ri_mid_valid = (ri_mid >= 0 && ri_mid < rc);
            int ri_mid_safe = max(0, ri_mid);
            float sv = (mid < sc) ? staging_dists[mid] : FLT_MAX;
            uint32_t si_id = (mid < sc) ? staging_ids[mid] : UINT32_MAX;
            float rv;
            uint32_t ri_id;
            if (ri_mid < 0) {
                // No result element precedes this split: staging[mid] must not
                // be taken ahead of it -> compare against -inf (predicate false).
                rv = -FLT_MAX;
                ri_id = 0u;
            } else if (ri_mid_valid) {
                rv = result_dists[ri_mid_safe];
                ri_id = result_ids[ri_mid_safe];
            } else {
                // ri_mid >= rc: no result element there -> +inf (take staging).
                rv = FLT_MAX;
                ri_id = UINT32_MAX;
            }
            bool sv_le = (sv < rv) || (sv == rv && si_id < ri_id);
            if (sv_le)
                lo = mid + 1;
            else
                hi = mid;
        }

        int si = lo;
        // Clamp ri to 0 to prevent negative-index speculative loads;
        // the ri_valid predicate gates whether the value is actually used.
        int ri = pos - si;
        bool ri_valid = (ri >= 0 && ri < rc);
        int ri_safe = max(0, ri);

        float sv = (si < sc) ? staging_dists[si] : FLT_MAX;
        float rv = ri_valid ? result_dists[ri_safe] : FLT_MAX;
        uint32_t si_id = (si < sc) ? staging_ids[si] : UINT32_MAX;
        uint32_t ri_id = ri_valid ? result_ids[ri_safe] : UINT32_MAX;
        bool take_staging = (si < sc) && (!ri_valid || (sv < rv) || (sv == rv && si_id < ri_id));

        if (take_staging) {
            merged_ids[pos] = si_id;
            merged_dists[pos] = sv;
            merged_expanded[pos] = 0;
        } else {
            merged_ids[pos] = ri_id;
            merged_dists[pos] = rv;
            merged_expanded[pos] = ri_valid ? is_expanded[ri_safe] : 0;
        }
    }
    __syncthreads();

    int new_rc = min(T, ef);
    for (int i = threadIdx.x; i < new_rc; i += blockDim.x) {
        result_ids[i] = merged_ids[i];
        result_dists[i] = merged_dists[i];
        is_expanded[i] = merged_expanded[i];
    }
    if (threadIdx.x == 0) {
        meta[0] = new_rc;
    }
    __syncthreads();
}

// ============================================================================
// Phase 2: Layer-0 parallel beam search kernel
// ============================================================================

__device__ __forceinline__ bool bitmap_visit(
        uint32_t* bitmap,
        uint32_t node_id) {
    uint32_t word = node_id >> 5;
    uint32_t bit = 1u << (node_id & 31);
    uint32_t old = atomicOr(&bitmap[word], bit);
    return (old & bit) == 0;
}

// Templated on the dataset element type (DataT), the query element type
// (QueryT: float for the generic path, int8_t for the native DP4A path), and
// USE_DP4A which selects the DP4A int8 inner-product distance at compile time.
template <typename DataT, typename QueryT, bool USE_DP4A>
__global__ void layer0_beam_search_kernel(
        const QueryT* __restrict__ d_queries,
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
        bool use_inner_product) {
    int query_idx = blockIdx.x;
    if (query_idx >= num_queries)
        return;

    const QueryT* query = d_queries + static_cast<int64_t>(query_idx) * dim;
    int dim4 = dim / 4;

    extern __shared__ char smem[];

    // Pad to a power of two so the bitonic sort in parallel_merge_into_result
    // has a valid capacity for any max_degree0 (e.g. non-power-of-two 2*M).
    int max_staging = padded_staging_capacity(search_width, max_degree0);

    uint32_t* result_ids = reinterpret_cast<uint32_t*>(smem);
    float* result_dists = reinterpret_cast<float*>(result_ids + ef);
    uint32_t* is_expanded = reinterpret_cast<uint32_t*>(result_dists + ef);
    uint32_t* staging_ids = is_expanded + ef;
    float* staging_dists = reinterpret_cast<float*>(staging_ids + max_staging);
    uint32_t* parent_ids =
            reinterpret_cast<uint32_t*>(staging_dists + max_staging);
    int* meta = reinterpret_cast<int*>(parent_ids + search_width);
    uint32_t* merged_ids = reinterpret_cast<uint32_t*>(meta + 3);
    float* merged_dists = reinterpret_cast<float*>(merged_ids + ef);
    uint32_t* merged_expanded = reinterpret_cast<uint32_t*>(merged_dists + ef);

    int bitmap_words = (N + 31) / 32;
    uint32_t* visited_bmap =
            d_visited_bitmaps + static_cast<int64_t>(query_idx) * bitmap_words;

    for (int i = threadIdx.x; i < ef; i += blockDim.x) {
        result_ids[i] = UINT32_MAX;
        result_dists[i] = FLT_MAX;
        is_expanded[i] = 0;
    }
    if (threadIdx.x == 0) {
        meta[0] = 0;
        meta[1] = 0;
        meta[2] = 0;
    }
    __syncthreads();

    // --- Seed with entry point ---
    uint32_t ep = d_entry_points[query_idx];
    // Defensive: the entry point must be a real node (< N) before it indexes
    // the dataset (ep_dist) or the layer-0 graph (neighbor seeding). A bad ep
    // would otherwise fault far out of bounds. On the healthy path ep is always
    // valid; if it is not, seed nothing and let the search return sentinels.
    bool ep_valid = (ep < static_cast<uint32_t>(N));
    if (threadIdx.x == 0) {
        if (ep_valid) {
            float ep_dist = layer0_distance<DataT, QueryT, USE_DP4A>(
                    query, d_dataset, d_inv_norms, ep, dim, dim4,
                    use_inner_product);
            result_ids[0] = ep;
            result_dists[0] = ep_dist;
            is_expanded[0] = 0;
            meta[0] = 1;
            bitmap_visit(visited_bmap, ep);
        } else {
            meta[0] = 0;
        }
    }
    __syncthreads();

    // --- Seed with entry point's neighbors ---
    if (threadIdx.x == 0)
        meta[1] = 0;
    __syncthreads();

    for (int j = threadIdx.x; ep_valid && j < max_degree0; j += blockDim.x) {
        uint32_t nbr =
                d_layer0_graph[static_cast<int64_t>(ep) * max_degree0 + j];
        if (nbr == UINT32_MAX || nbr >= static_cast<uint32_t>(N))
            continue;
        if (!bitmap_visit(visited_bmap, nbr))
            continue;

        float dist = layer0_distance<DataT, QueryT, USE_DP4A>(
                query, d_dataset, d_inv_norms, nbr, dim, dim4,
                use_inner_product);

        int slot = atomicAdd(&meta[1], 1);
        if (slot < max_staging) {
            staging_ids[slot] = nbr;
            staging_dists[slot] = dist;
        }
    }
    __syncthreads();

    parallel_merge_into_result(
            result_ids, result_dists, is_expanded,
            staging_ids, staging_dists,
            merged_ids, merged_dists, merged_expanded,
            meta, ef, max_staging);
    if (threadIdx.x == 0) {
        int rc = meta[0];
        for (int i = 0; i < rc; i++) {
            if (result_ids[i] == ep) {
                is_expanded[i] = 1;
                break;
            }
        }
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
            // Defensive: a parent id must be a real node (< N). result_ids only
            // ever holds the entry point or graph neighbors (both < N) once the
            // merge is correct, so this never fires on the healthy path; it caps
            // a stray/garbage id at O(N) instead of letting it index the graph
            // at parent*max_degree0 and fault far out of bounds. Mirrors the
            // neighbor guard below.
            if (parent >= static_cast<uint32_t>(N))
                continue;
            uint32_t nbr =
                    d_layer0_graph
                                  [static_cast<int64_t>(parent) * max_degree0 +
                                   nbr_slot];
            if (nbr == UINT32_MAX || nbr >= static_cast<uint32_t>(N))
                continue;
            if (!bitmap_visit(visited_bmap, nbr))
                continue;

            float dist = layer0_distance<DataT, QueryT, USE_DP4A>(
                    query, d_dataset, d_inv_norms, nbr, dim, dim4,
                    use_inner_product);

            int slot = atomicAdd(&meta[1], 1);
            if (slot < max_staging) {
                staging_ids[slot] = nbr;
                staging_dists[slot] = dist;
            }
        }
        __syncthreads();

        parallel_merge_into_result(
                result_ids, result_dists, is_expanded,
                staging_ids, staging_dists,
                merged_ids, merged_dists, merged_expanded,
                meta, ef, max_staging);

        // Stagnation detected when num_parents==0 (all expanded) → break above
    }

    // --- Copy top-k results to global memory ---
    int rc = meta[0];
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        if (i < rc && result_ids[i] < static_cast<uint32_t>(N)) {
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
    // Must match the kernel's padded staging capacity exactly.
    int max_staging = padded_staging_capacity(search_width, max_degree0);

    size_t size = 0;
    size += ef * sizeof(uint32_t);           // result_ids
    size += ef * sizeof(float);              // result_dists
    size += ef * sizeof(uint32_t);           // is_expanded
    size += max_staging * sizeof(uint32_t);  // staging_ids
    size += max_staging * sizeof(float);     // staging_dists
    size += search_width * sizeof(uint32_t); // parent_ids
    size += 3 * sizeof(int);                 // meta
    size += ef * sizeof(uint32_t);           // merged_ids
    size += ef * sizeof(float);              // merged_dists
    size += ef * sizeof(uint32_t);           // merged_expanded
    return size;
}

inline size_t calc_visited_bitmap_size(int num_queries, int N) {
    int bitmap_words = (N + 31) / 32;
    return static_cast<size_t>(num_queries) * bitmap_words * sizeof(uint32_t);
}

} // namespace hnsw_kernel
} // namespace gpu
} // namespace faiss
