/**
 * HNSWPQ Time Profiling Demo
 * 
 * This demo profiles HNSWPQ operations for detailed time breakdown:
 * - PQ training and encoding
 * - Graph construction with PQ codes
 * - Search operations with distance table computation
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <chrono>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexPQ.h>
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
    printf("=== HNSWPQ Time Profiling Demo ===\n\n");
    
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
        const size_t k = 10;              // number of neighbors
        const int M_hnsw = 32;            // HNSW connectivity parameter
        const int M_pq = 8;               // PQ subquantizers
        const int nbits_pq = 8;           // bits per subquantizer
        const int efConstruction = 200;   // construction parameter
        const int efSearch = 50;          // search parameter
        
        printf("Parameters:\n");
        printf("  d=%zu, nb=%zu, nq=%zu, n_learn=%zu\n", d, nb, nq, n_learn);
        printf("  M_hnsw=%d, M_pq=%d, nbits=%d\n", M_hnsw, M_pq, nbits_pq);
        printf("  efConstruction=%d, efSearch=%d, k=%zu\n\n", 
               efConstruction, efSearch, k);

        IndexFlatL2 flat_index(d);
        flat_index.add(nb, base_data.data());

        std::vector<idx_t> ground_truth_labels(nq);
        compute_ground_truth_labels(flat_index, query_data.data(), nq, ground_truth_labels);
        
        // Create index
        IndexHNSWPQ index(d, M_pq, M_hnsw, nbits_pq);
        index.hnsw.efConstruction = efConstruction;
        
        // Reset profiler
        TimeProfiler::getInstance().reset();
        
        // Train PQ
        printf("Training PQ...\n");
        {
            SCOPED_TIMER("HNSWPQ::train_total");
            index.train(n_learn, learn_data.data());
        }
        
        printf("\nTraining phase profiling:\n");
        TimeProfiler::getInstance().printReport();
        TimeProfiler::getInstance().reset();
        
        // Add vectors
        printf("\nAdding vectors to HNSWPQ...\n");
        {
            SCOPED_TIMER("HNSWPQ::add_total");
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
                SCOPED_TIMER("HNSWPQ::search_total");
                index.search(nq, query_data.data(), k, distances.data(), labels.data());
            }
            compute_recall(labels, ground_truth_labels, k);
        }
        
        printf("\nSearch phase profiling:\n");
        TimeProfiler::getInstance().printReport();
        
        // Get underlying PQ for detailed analysis
        IndexPQ* pq_index = dynamic_cast<IndexPQ*>(index.storage);
        if (pq_index) {
            printf("\n\n=== PQ-specific operations analysis ===\n");
            TimeProfiler::getInstance().reset();
            
            // Test distance table computation separately
            std::vector<float> dis_tables(nq * pq_index->pq.ksub * pq_index->pq.M);
            for (int i = 0; i < 10; i++) {
                SCOPED_TIMER("PQ::compute_distance_tables_standalone");
                pq_index->pq.compute_distance_tables(nq, query_data.data(), dis_tables.data());
            }
            
            TimeProfiler::getInstance().printReport();
        }
        
        // Analyze search with different ef values
        printf("\n\n=== Analysis with different efSearch values ===\n");
        const int ef_values[] = {16, 32, 64, 128, 256};
        
        for (int ef : ef_values) {
            index.hnsw.efSearch = ef;
            TimeProfiler::getInstance().reset();
            
            // Run search multiple times
            for (int run = 0; run < 10; run++) {
                {
                    SCOPED_TIMER("HNSWPQ::search_total");
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
