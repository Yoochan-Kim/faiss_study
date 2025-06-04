/**
 * Comprehensive Index Profiling Demo
 * 
 * This demo compares PQ, IVFPQ, and HNSW performance with detailed time breakdowns
 * Including PQ refinement (dequantization + sorting)
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/TimeProfiler.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/AuxIndexStructures.h>

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
        throw std::runtime_error(std::string("cannot open file: ") + fname);
    }
    
    // Read first dimension
    int d;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    if (!in) throw std::runtime_error("read error");
    dim = static_cast<size_t>(d);

    // Calculate number of vectors
    in.seekg(0, std::ios::end);
    std::streampos fsize = in.tellg();
    std::streampos vec_size = sizeof(int) + sizeof(float) * dim;
    num_vectors = static_cast<size_t>(fsize / vec_size);

    data.resize(num_vectors * dim);

    // Read all vectors
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; i++) {
        int di;
        in.read(reinterpret_cast<char*>(&di), sizeof(int));
        if (di != d) throw std::runtime_error("dimension mismatch in file");
        in.read(reinterpret_cast<char*>(data.data() + i * dim), sizeof(float) * dim);
        if (!in) throw std::runtime_error("read error");
    }
    in.close();
    
    printf("Loaded %zu vectors of dimension %zu from %s\n", num_vectors, dim, fname);
}

// PQ with refinement (dequantization + sorting)
class IndexPQWithRefinement : public IndexPQ {
public:
    IndexPQWithRefinement(int d, size_t M, size_t nbits) 
        : IndexPQ(d, M, nbits) {}
    
    void search_with_refinement(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        idx_t k_factor = 10  // retrieve k_factor * k candidates for refinement
    ) const {
        const idx_t k_expanded = k * k_factor;
        std::vector<float> distances_expanded(n * k_expanded);
        std::vector<idx_t> labels_expanded(n * k_expanded);
        
        // Step 1: Regular PQ search with expanded k
        {
            SCOPED_TIMER("PQ::search_expanded");
            search(n, x, k_expanded, distances_expanded.data(), labels_expanded.data());
        }
        
        // Step 2: Refinement - dequantize and compute exact distances
        {
            SCOPED_TIMER("PQ::refinement");
            
            // Process each query
            #pragma omp parallel for
            for (idx_t i = 0; i < n; i++) {
                std::vector<std::pair<float, idx_t>> refined_results;
                refined_results.reserve(k_expanded);
                
                const float* query = x + i * d;
                
                // Dequantize each candidate and compute exact distance
                for (idx_t j = 0; j < k_expanded; j++) {
                    idx_t candidate_id = labels_expanded[i * k_expanded + j];
                    if (candidate_id < 0) continue;
                    
                    // Dequantize
                    std::vector<float> reconstructed(d);
                    {
                        SCOPED_TIMER("PQ::reconstruct_single");
                        pq.decode(codes.data() + candidate_id * code_size, reconstructed.data());
                    }
                    
                    // Compute exact distance
                    float dist;
                    {
                        SCOPED_TIMER("PQ::exact_distance");
                        dist = fvec_L2sqr(query, reconstructed.data(), d);
                    }
                    
                    refined_results.push_back({dist, candidate_id});
                }
                
                // Sort by distance
                {
                    SCOPED_TIMER("PQ::sort_refined");
                    std::partial_sort(
                        refined_results.begin(),
                        refined_results.begin() + std::min(k, (idx_t)refined_results.size()),
                        refined_results.end(),
                        [](const auto& a, const auto& b) { return a.first < b.first; }
                    );
                }
                
                // Copy top-k results
                for (idx_t j = 0; j < k && j < refined_results.size(); j++) {
                    distances[i * k + j] = refined_results[j].first;
                    labels[i * k + j] = refined_results[j].second;
                }
            }
        }
    }
};

// Benchmark PQ with refinement
void benchmark_pq_sift(
    const std::vector<float>& learn_data,
    size_t n_learn,
    const std::vector<float>& base_data,
    size_t nb,
    const std::vector<float>& query_data,
    size_t nq,
    size_t d
) {
    printf("\n=== PQ Benchmark ===\n");
    
    const size_t M = 8;
    const size_t nbits = 8;
    const size_t k = 10;
    
    IndexPQWithRefinement index(d, M, nbits);
    
    // Training
    {
        SCOPED_TIMER("PQ::train");
        index.train(n_learn, learn_data.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();
    
    // Adding
    {
        SCOPED_TIMER("PQ::add");
        index.add(nb, base_data.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();
    
    std::vector<idx_t> labels(k * nq);
    std::vector<float> distances(k * nq);
    
    // Warmup
    index.search(1, query_data.data(), k, distances.data(), labels.data());
    TimeProfiler::getInstance().reset();
    
    // Regular PQ search
    printf("\nRegular PQ search:\n");
    const int runs = 10;
    for (int i = 0; i < runs; i++) {
        SCOPED_TIMER("PQ::search_total");
        index.search(nq, query_data.data(), k, distances.data(), labels.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();
    
    // PQ search with refinement
    printf("\nPQ search with refinement:\n");
    for (int i = 0; i < runs; i++) {
        SCOPED_TIMER("PQ::search_refined_total");
        index.search_with_refinement(nq, query_data.data(), k, distances.data(), labels.data());
    }
    TimeProfiler::getInstance().printReport();
}

// Benchmark IVFPQ with detailed breakdown
void benchmark_ivfpq_sift(
    const std::vector<float>& learn_data,
    size_t n_learn,
    const std::vector<float>& base_data,
    size_t nb,
    const std::vector<float>& query_data,
    size_t nq,
    size_t d
) {
    printf("\n\n=== IVFPQ Benchmark ===\n");
    
    const size_t nlist = 100;
    const size_t M = 8;
    const size_t nbits = 8;
    const size_t k = 10;
    const size_t nprobe = 10;
    
    IndexFlatL2 quantizer(d);
    IndexIVFPQ index(&quantizer, d, nlist, M, nbits);
    
    // Training
    TimeProfiler::getInstance().reset();
    {
        SCOPED_TIMER("IVFPQ::train_total");
        
        // Train coarse quantizer
        {
            SCOPED_TIMER("IVFPQ::train_coarse");
            quantizer.train(n_learn, learn_data.data());
        }
        
        // Train PQ
        {
            SCOPED_TIMER("IVFPQ::train_pq");
            index.train(n_learn, learn_data.data());
        }
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();
    
    // Adding
    {
        SCOPED_TIMER("IVFPQ::add");
        index.add(nb, base_data.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();
    
    index.nprobe = nprobe;
    std::vector<idx_t> labels(k * nq);
    std::vector<float> distances(k * nq);
    
    // Warmup
    index.search(1, query_data.data(), k, distances.data(), labels.data());
    TimeProfiler::getInstance().reset();
    
    // Search with detailed breakdown
    printf("\nIVFPQ search breakdown:\n");
    const int runs = 10;
    for (int i = 0; i < runs; i++) {
        SCOPED_TIMER("IVFPQ::search_total");
        index.search(nq, query_data.data(), k, distances.data(), labels.data());
    }
    TimeProfiler::getInstance().printReport();
}

// Benchmark HNSW with detailed breakdown
void benchmark_hnsw_sift(
    const std::vector<float>& base_data,
    size_t nb,
    const std::vector<float>& query_data,
    size_t nq,
    size_t d
) {
    printf("\n\n=== HNSW Benchmark ===\n");
    
    const size_t k = 10;
    const int M = 32;
    const int efConstruction = 200;
    const int efSearch = 100;
    
    IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    
    // Building index
    TimeProfiler::getInstance().reset();
    {
        SCOPED_TIMER("HNSW::add");
        index.add(nb, base_data.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();
    
    index.hnsw.efSearch = efSearch;
    std::vector<idx_t> labels(k * nq);
    std::vector<float> distances(k * nq);
    
    // Warmup
    index.search(1, query_data.data(), k, distances.data(), labels.data());
    TimeProfiler::getInstance().reset();
    
    // Search
    printf("\nHNSW search:\n");
    const int runs = 10;
    for (int i = 0; i < runs; i++) {
        SCOPED_TIMER("HNSW::search_total");
        index.search(nq, query_data.data(), k, distances.data(), labels.data());
    }
    TimeProfiler::getInstance().printReport();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <sift_learn.fvecs> <sift_base.fvecs> <sift_query.fvecs>\n", argv[0]);
        return 1;
    }
    
    const char* learn_path = argv[1];
    const char* base_path = argv[2];
    const char* query_path = argv[3];
    
    try {
        // Load data
        size_t d_learn, n_learn;
        std::vector<float> learn_data;
        read_fvecs(learn_path, learn_data, n_learn, d_learn);
        
        size_t d_base, nb;
        std::vector<float> base_data;
        read_fvecs(base_path, base_data, nb, d_base);
        
        size_t d_query, nq;
        std::vector<float> query_data;
        read_fvecs(query_path, query_data, nq, d_query);
        
        if (d_learn != d_base || d_base != d_query) {
            throw std::runtime_error("dimension mismatch among datasets");
        }
        size_t d = d_base;
        
        printf("\nDataset info:\n");
        printf("  Learn: %zu vectors\n", n_learn);
        printf("  Base: %zu vectors\n", nb);
        printf("  Query: %zu vectors\n", nq);
        printf("  Dimension: %zu\n", d);
        
        // Run benchmarks
        benchmark_pq_sift(learn_data, n_learn, base_data, nb, query_data, nq, d);
        benchmark_ivfpq_sift(learn_data, n_learn, base_data, nb, query_data, nq, d);
        benchmark_hnsw_sift(base_data, nb, query_data, nq, d);
        
        printf("\n=== Summary ===\n");
        printf("All benchmarks completed successfully.\n");
        
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    
    return 0;
}
