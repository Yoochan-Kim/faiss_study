/**
 * HNSWPQ + Refinement Time Profiling Demo
 * 
 * This demo profiles HNSWPQ with refinement operations:
 * - Initial PQ-based search
 * - Loading original vectors for refinement
 * - Recomputing exact distances
 * - Re-ranking results
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <algorithm>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexRefine.h>
#include <faiss/utils/TimeProfiler.h>
#include <faiss/utils/distances.h>

using namespace faiss;

// Read fvecs file format
static void read_fvecs(
    const char* fname,
    std::vector<float>& data,
    size_t& num_vectors,
    size_t& dim
) {
    std::ifstream in(fname, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open file");
    }
    
    // First 4 bytes: dimension (int)
    int d;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    if (!in) throw std::runtime_error("read error");
    dim = static_cast<size_t>(d);

    // Calculate number of vectors from file size
    in.seekg(0, std::ios::end);
    std::streampos fsize = in.tellg();
    std::streampos per = sizeof(int) + sizeof(float) * dim;
    num_vectors = static_cast<size_t>(fsize / per);

    data.resize(num_vectors * dim);

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; i++) {
        int di;
        in.read(reinterpret_cast<char*>(&di), sizeof(int));
        in.read(reinterpret_cast<char*>(data.data() + i * dim), sizeof(float) * dim);
        if (!in) throw std::runtime_error("read error");
    }
    in.close();
}

// Custom refinement search function with profiling
void search_with_refinement(
    const IndexHNSWPQ& index,
    const std::vector<float>& base_data,
    size_t nq,
    const float* queries,
    size_t k,
    size_t k_refine,  // number of candidates to refine
    float* distances,
    idx_t* labels
) {
    SCOPED_TIMER("HNSWPQ_refinement::search_total");
    
    // Allocate space for initial search results
    std::vector<float> coarse_distances(k_refine * nq);
    std::vector<idx_t> coarse_labels(k_refine * nq);
    
    // Step 1: Initial PQ-based search with more candidates
    {
        SCOPED_TIMER("HNSWPQ_refinement::coarse_search");
        index.search(nq, queries, k_refine, coarse_distances.data(), coarse_labels.data());
    }
    
    // Step 2: Refinement with exact distances
    {
        SCOPED_TIMER("HNSWPQ_refinement::refinement");
        
#pragma omp parallel for
        for (int64_t i = 0; i < nq; i++) {
            const float* query = queries + i * index.d;
            float* query_distances = distances + i * k;
            idx_t* query_labels = labels + i * k;
            
            // Get candidates for this query
            const idx_t* candidates = coarse_labels.data() + i * k_refine;
            
            // Temporary storage for exact distances
            std::vector<std::pair<float, idx_t>> exact_results;
            exact_results.reserve(k_refine);
            
            // Step 2a: Load vectors and compute exact distances
            {
                SCOPED_TIMER("HNSWPQ_refinement::compute_exact_distances");
                
                for (size_t j = 0; j < k_refine; j++) {
                    idx_t id = candidates[j];
                    if (id < 0) continue;  // Skip invalid results
                    
                    // Load original vector
                    const float* vec = base_data.data() + id * index.d;
                    
                    // Compute exact distance
                    float dist = fvec_L2sqr(query, vec, index.d);
                    exact_results.push_back({dist, id});
                }
            }
            
            // Step 2b: Sort by exact distance
            {
                SCOPED_TIMER("HNSWPQ_refinement::rerank");
                std::partial_sort(
                    exact_results.begin(),
                    exact_results.begin() + std::min(k, exact_results.size()),
                    exact_results.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; }
                );
            }
            
            // Step 2c: Copy final results
            {
                SCOPED_TIMER("HNSWPQ_refinement::copy_results");
                size_t n_copy = std::min(k, exact_results.size());
                for (size_t j = 0; j < n_copy; j++) {
                    query_distances[j] = exact_results[j].first;
                    query_labels[j] = exact_results[j].second;
                }
                // Fill remaining slots if needed
                for (size_t j = n_copy; j < k; j++) {
                    query_distances[j] = HUGE_VALF;
                    query_labels[j] = -1;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    printf("=== HNSWPQ + Refinement Time Profiling Demo ===\n\n");
    
    try {
        // Parameters
        const char* base_path = "sift/sift_base.fvecs";
        const char* query_path = "sift/sift_query.fvecs";
        const char* learn_path = "sift/sift_learn.fvecs";
        
        if (argc > 3) {
            base_path = argv[1];
            query_path = argv[2];
            learn_path = argv[3];
        }
        
        // Read data
        size_t d_base, nb;
        std::vector<float> base_data;
        read_fvecs(base_path, base_data, nb, d_base);
        
        size_t d_query, nq;
        std::vector<float> query_data;
        read_fvecs(query_path, query_data, nq, d_query);
        
        size_t d_learn, n_learn;
        std::vector<float> learn_data;
        read_fvecs(learn_path, learn_data, n_learn, d_learn);
        
        if (d_base != d_query || d_base != d_learn) {
            throw std::runtime_error("dimension mismatch");
        }
        
        size_t d = d_base;
        const size_t k = 10;              // final number of neighbors
        const size_t k_refine = 100;      // number of candidates for refinement
        const int M_hnsw = 32;            // HNSW connectivity parameter
        const int M_pq = 8;               // PQ subquantizers
        const int nbits_pq = 8;           // bits per subquantizer
        const int efConstruction = 200;   // construction parameter
        const int efSearch = 50;          // search parameter
        
        printf("Parameters:\n");
        printf("  d=%zu, nb=%zu, nq=%zu, n_learn=%zu\n", d, nb, nq, n_learn);
        printf("  M_hnsw=%d, M_pq=%d, nbits=%d\n", M_hnsw, M_pq, nbits_pq);
        printf("  efConstruction=%d, efSearch=%d\n", efConstruction, efSearch);
        printf("  k=%zu, k_refine=%zu\n\n", k, k_refine);
        
        // Create index
        IndexHNSWPQ index(d, M_pq, M_hnsw, nbits_pq);
        index.hnsw.efConstruction = efConstruction;
        
        // Reset profiler
        TimeProfiler::getInstance().reset();
        
        // Train PQ
        printf("Training PQ...\n");
        {
            SCOPED_TIMER("HNSWPQ_refinement::train_total");
            index.train(n_learn, learn_data.data());
        }
        
        printf("\nTraining phase profiling:\n");
        TimeProfiler::getInstance().printReport();
        TimeProfiler::getInstance().reset();
        
        // Add vectors
        printf("\nAdding vectors to HNSWPQ...\n");
        {
            SCOPED_TIMER("HNSWPQ_refinement::add_total");
            index.add(nb, base_data.data());
        }
        
        printf("\nConstruction phase profiling:\n");
        TimeProfiler::getInstance().printReport();
        TimeProfiler::getInstance().reset();
        
        // Prepare search
        index.hnsw.efSearch = efSearch;
        std::vector<idx_t> labels(k * nq);
        std::vector<float> distances(k * nq);
        
        // Warmup search
        search_with_refinement(index, base_data, 1, query_data.data(), 
                             k, k_refine, distances.data(), labels.data());
        TimeProfiler::getInstance().reset();
        
        // Search profiling
        printf("\nPerforming searches with refinement...\n");
        const int num_runs = 100;
        
        for (int run = 0; run < num_runs; run++) {
            search_with_refinement(index, base_data, nq, query_data.data(), 
                                 k, k_refine, distances.data(), labels.data());
        }
        
        printf("\nSearch with refinement phase profiling:\n");
        TimeProfiler::getInstance().printReport();
        
        // Compare with standard search
        printf("\n\n=== Comparison with standard HNSWPQ search (no refinement) ===\n");
        TimeProfiler::getInstance().reset();
        
        for (int run = 0; run < num_runs; run++) {
            SCOPED_TIMER("HNSWPQ::search_no_refinement");
            index.search(nq, query_data.data(), k, distances.data(), labels.data());
        }
        
        printf("\nStandard search profiling:\n");
        TimeProfiler::getInstance().printReport();
        
        // Analyze with different refinement sizes
        printf("\n\n=== Analysis with different k_refine values ===\n");
        const size_t refine_values[] = {20, 50, 100, 200, 500};
        
        for (size_t kr : refine_values) {
            if (kr > nb) continue;  // Skip if k_refine > database size
            
            TimeProfiler::getInstance().reset();
            
            // Run search multiple times
            for (int run = 0; run < 10; run++) {
                search_with_refinement(index, base_data, nq, query_data.data(), 
                                     k, kr, distances.data(), labels.data());
            }
            
            printf("\nk_refine = %zu:\n", kr);
            TimeProfiler::getInstance().printReport();
        }
        
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
