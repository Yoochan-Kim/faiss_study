/**
 * HNSW Time Profiling Demo
 * 
 * This demo profiles HNSW operations for detailed time breakdown:
 * - Graph construction
 * - Search operations at different levels
 * - Distance computations
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <chrono>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/TimeProfiler.h>

using namespace faiss;

static void compute_ground_truth_labels(
    IndexFlatL2& flat_index,
    const float* query_data,    // 길이 = nq * d
    size_t nq,                  // 쿼리 수
    std::vector<idx_t>& ground_truth_labels  // 크기 = nq
) {
    std::vector<float> distances_gt(nq);
    flat_index.search(
        nq,
        query_data,
        1,                      // Top-1
        distances_gt.data(),
        ground_truth_labels.data()
    );
}

static void compute_recall(
    const std::vector<idx_t>& labels,          // 검색된 k개의 인덱스 (flattened, 크기 = nq * k)
    const std::vector<idx_t>& ground_truth,    // 쿼리당 정답 인덱스 (크기 = nq)
    int k                                             // Top-k
) {
    size_t nq = ground_truth.size();
    size_t correct = 0;

    // 각 쿼리별로 Top-k 내부에 정답이 존재하는지 확인
    for (size_t i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            if (labels[i * k + j] == ground_truth[i]) {
                correct++;
                break;
            }
        }
    }

    float recall = (nq > 0) ? (static_cast<float>(correct) / nq) : 0.0f;
    printf("Recall@%d: %.4f\n", k, recall);
}

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

int main(int argc, char** argv) {
    printf("=== HNSW Time Profiling Demo ===\n\n");
    
    try {
        // Parameters
        const char* base_path = "sift/sift_base.fvecs";
        const char* query_path = "sift/sift_query.fvecs";
        
        if (argc > 2) {
            base_path = argv[1];
            query_path = argv[2];
        }
        
        // Read data
        size_t d_base, nb;
        std::vector<float> base_data;
        read_fvecs(base_path, base_data, nb, d_base);
        
        size_t d_query, nq;
        std::vector<float> query_data;
        read_fvecs(query_path, query_data, nq, d_query);
        
        if (d_base != d_query) {
            throw std::runtime_error("dimension mismatch between base and query");
        }
        
        size_t d = d_base;
        const size_t k = 10;              // number of neighbors
        const int M = 32;                 // HNSW connectivity parameter
        const int efConstruction = 200;   // construction parameter
        const int efSearch = 50;          // search parameter
        
        printf("Parameters:\n");
        printf("  d=%zu, nb=%zu, nq=%zu\n", d, nb, nq);
        printf("  M=%d, efConstruction=%d, efSearch=%d, k=%zu\n\n", 
               M, efConstruction, efSearch, k);
        
        IndexFlatL2 flat_index(d);
        flat_index.add(nb, base_data);

        std::vector<idx_t> ground_truth_labels(nq);
        compute_ground_truth_labels(flat_index, query_data, nq, ground_truth_labels);

        // Create index
        IndexHNSWFlat index(d, M);
        index.hnsw.efConstruction = efConstruction;
        
        // Reset profiler
        TimeProfiler::getInstance().reset();
        
        // Add vectors
        printf("Adding vectors to HNSW...\n");
        {
            SCOPED_TIMER("HNSW::add_total");
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
        index.search(1, query_data.data(), k, distances.data(), labels.data());
        TimeProfiler::getInstance().reset();
        
        // Search profiling
        printf("\nPerforming searches...\n");
        const int num_runs = 100;
        
        for (int run = 0; run < num_runs; run++) {
            {
                SCOPED_TIMER("HNSW::search_total");
                index.search(nq, query_data.data(), k, distances.data(), labels.data());
            }
            compute_recall(labels, ground_truth_labels, k);
        }
        
        printf("\nSearch phase profiling:\n");
        TimeProfiler::getInstance().printReport();
        
        // Analyze search with different ef values
        printf("\n\n=== Analysis with different efSearch values ===\n");
        const int ef_values[] = {16, 32, 64, 128, 256};
        
        for (int ef : ef_values) {
            index.hnsw.efSearch = ef;
            TimeProfiler::getInstance().reset();
            
            // Run search multiple times
            for (int run = 0; run < 10; run++) {
                {
                    SCOPED_TIMER("HNSW::search_total");
                    index.search(nq, query_data.data(), k, distances.data(), labels.data());
                }
                compute_recall(labels, ground_truth_labels, k);
            }
            
            printf("\nefSearch = %d:\n", ef);
            TimeProfiler::getInstance().printReport();
        }
        
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
