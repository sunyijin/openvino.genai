// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cdpruner_config.hpp"
#include "openvino/runtime/tensor.hpp"
#include <vector>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <mutex>

//#define USE_THREAD
//#define USE_THREAD1
#define USE_OMP


namespace ov::genai::cdpruner {

/**
 * @brief Fast greedy DPP (Determinantal Point Process) algorithm for token selection
 * 
 * This class implements the fast greedy approximation algorithm for maximizing
 * the determinant of a subset selection from a kernel matrix. The algorithm
 * is based on the CDPruner paper and provides O(T²N) complexity where T is
 * the number of tokens to select and N is the total number of tokens.
 * 
 * The core algorithm follows these steps:
 * 1. Initialize diagonal scores (marginal gains)
 * 2. Greedily select tokens with maximum marginal gain
 * 3. Update orthogonalized vectors using Gram-Schmidt process
 * 4. Update marginal gains by subtracting orthogonal projections
 */
class FastGreedyDPP {
public:
    /// @brief Constructor
    /// @param config Configuration for the DPP selector
    explicit FastGreedyDPP(const Config& config);
    ~FastGreedyDPP();
    
    /**
     * @brief Select diverse tokens using fast greedy DPP algorithm
     * @param kernel Conditional kernel matrix [B, N, N]
     * @param num_tokens Number of tokens to select
     * @return Selected token indices for each batch [B, T]
     */
    std::vector<std::vector<size_t>> select(const ov::Tensor& kernel, size_t num_tokens, size_t num_images=1);

    /**
     * @brief Create boolean mask from selected indices
     * @param selected_indices Selected indices for each batch [B, T]
     * @param total_tokens Total number of tokens
     * @return Boolean mask [B*N] where true indicates selected tokens
     */
    static std::vector<bool> create_mask(const std::vector<std::vector<size_t>>& selected_indices, 
                                       size_t total_tokens);

    /**
     * @brief Compute approximate determinant for validation
     * @param kernel Kernel matrix [1, N, N] (single batch only)
     * @param selected_indices Selected token indices
     * @return Approximated determinant value
     */
    static float compute_determinant_approximation(const ov::Tensor& kernel, 
                                                 const std::vector<size_t>& selected_indices);

private:
    long thread_total_time;
    /**
     * @brief Select tokens for a single batch
     * @param kernel Kernel matrix [B, N, N]
     * @param batch_idx Batch index to process
     * @param num_tokens Number of tokens to select
     * @return Selected token indices for this batch
     */
    std::vector<size_t> select_single_batch(const ov::Tensor& kernel, size_t batch_idx, size_t num_tokens, size_t num_images=1);

    /**
     * @brief Find index with maximum value
     * @param scores Score tensor [N]
     * @return Index of maximum value
     */
    size_t argmax(const ov::Tensor& scores);

    /**
     * @brief Update orthogonal vector using Gram-Schmidt process
     * @param kernel Kernel matrix [B, N, N]
     * @param batch_idx Current batch index
     * @param selected_idx Newly selected token index
     * @param iteration Current iteration (number of previously selected tokens)
     * @param cis Orthogonalized vectors [T, N]
     * @param di2s Current diagonal scores [N]
     */
    void update_orthogonal_vector(const ov::Tensor& kernel, size_t batch_idx, size_t selected_idx, 
                                size_t iteration, ov::Tensor& cis, const ov::Tensor& di2s);

    /**
     * @brief Update marginal gains after selecting a token
     * @param iteration Current iteration
     * @param selected_idx Newly selected token index
     * @param cis Orthogonalized vectors [T, N]
     * @param di2s Diagonal scores to update [N]
     */
    void update_marginal_gains(size_t iteration, size_t selected_idx, 
                             const ov::Tensor& cis, ov::Tensor& di2s);
#ifdef USE_THREAD
    static void thread_worker(const float* kernel_data, const float* di2s_data, float* cis_data,
                       size_t batch_idx, size_t selected_idx, size_t iteration,
                       size_t start_j, size_t end_j, size_t total_tokens, float norm_factor);
    void update_orthogonal_vector_thread(const ov::Tensor& kernel, size_t batch_idx, size_t selected_idx,
                    size_t iteration, ov::Tensor& cis, const ov::Tensor& di2s);
#endif

#ifdef USE_THREAD1
    void thread_main(size_t thread_id);
    void update_orthogonal_vector_thread(const ov::Tensor& kernel, size_t batch_idx, size_t selected_idx,
                    size_t iteration, ov::Tensor& cis, const ov::Tensor& di2s);

    struct ThreadTask {
        const float* kernel_data = nullptr;
        const float* di2s_data = nullptr;
        float* cis_data = nullptr;
        size_t batch_idx = 0;
        size_t selected_idx = 0;
        size_t iteration = 0;
        size_t start_j = 0;
        size_t end_j = 0;
        size_t total_tokens = 0;
        float norm_factor = 0.0f;
        bool has_work = false;
    };

    static constexpr size_t num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<ThreadTask> thread_tasks;
    std::vector<std::mutex> task_mutexes;
    std::vector<std::condition_variable> task_cvs;
    std::atomic<bool> stop_flag;
    std::atomic<size_t> finished_count;
    std::mutex wait_mutex;
    std::condition_variable wait_cv;
#endif

    Config m_config;
};

} // namespace ov::genai::cdpruner 
