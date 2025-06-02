/**
 * PQ Time Profiling Demo
 * 
 * This demo shows how to profile different parts of PQ operations:
 * - Distance table computation
 * - Code computation 
 * - Search operations
 * - Memory access patterns
 */

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <chrono>

#include <faiss/IndexPQ.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/TimeProfiler.h>

using namespace faiss;

// Generate random data
void generate_random_data(
    float* data, 
    size_t n, 
    size_t d,
    unsigned seed = 1234
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> distrib;
    
    for (size_t i = 0; i < n * d; i++) {
        data[i] = distrib(rng);
    }
}

void benchmark_pq_operations() {
    printf("=== PQ Time Profiling Demo ===\n\n");
    
    // Parameters
    const size_t d = 128;        // dimension
    const size_t nb = 100000;     // database size  
    const size_t nq = 1000;       // number of queries
    const size_t M = 8;           // number of subquantizers
    const size_t nbits = 8;       // bits per subquantizer
    const size_t k = 10;          // number of neighbors
    
    printf("Parameters:\n");
    printf("  d=%zu, nb=%zu, nq=%zu\n", d, nb, nq);
    printf("  M=%zu, nbits=%zu, k=%zu\n\n", M, nbits, k);
    
    // Generate data
    std::vector<float> database(nb * d);
    std::vector<float> queries(nq * d);
    
    generate_random_data(database.data(), nb, d);
    generate_random_data(queries.data(), nq, d);
    
    // Reset profiler
    TimeProfiler::getInstance().reset();
    
    // Create and train index
    IndexPQ index(d, M, nbits);
    
    printf("Training index...\n");
    {
        SCOPED_TIMER("IndexPQ::train");
        index.train(nb, database.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();
    
    printf("Adding vectors...\n");
    {
        SCOPED_TIMER("IndexPQ::add");
        index.add(nb, database.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();
    
    // Benchmark search operations
    printf("Performing searches...\n");
    std::vector<idx_t> labels(k * nq);
    std::vector<float> distances(k * nq);
    
    // Warmup
    index.search(1, queries.data(), k, distances.data(), labels.data());
    
    TimeProfiler::getInstance().reset();
    // Actual benchmark
    const int num_runs = 100;
    for (int run = 0; run < num_runs; run++) {
        index.search(nq, queries.data(), k, distances.data(), labels.data());
    }
    
    // // Print profiling report
    TimeProfiler::getInstance().printReport();
    
    // Additional detailed profiling
    // printf("\n=== Detailed PQ Operations Analysis ===\n");
    
    // // Profile individual operations
    // ProductQuantizer& pq = index.pq;
    
    // // 1. Distance table computation
    // std::vector<float> dis_tables(nq * pq.ksub * pq.M);
    // for (int i = 0; i < 5; i++) {
    //     TIME_START();
    //     pq.compute_distance_tables(nq, queries.data(), dis_tables.data());
    //     TIME_END("compute_distance_tables_only");
    // }
    
    // // 2. Code computation
    // std::vector<uint8_t> codes(nq * pq.code_size);
    // for (int i = 0; i < 5; i++) {
    //     TIME_START();
    //     pq.compute_codes(queries.data(), codes.data(), nq);
    //     TIME_END("compute_codes_only");
    // }
    
    // // 3. Direct search timing (which internally uses distance tables)
    // for (int i = 0; i < 5; i++) {
    //     float_maxheap_array_t res = {
    //         size_t(nq), size_t(k), labels.data(), distances.data()
    //     };
        
    //     TIME_START();
    //     pq.search(
    //         queries.data(),
    //         nq,
    //         index.codes.data(),
    //         nb,
    //         &res,
    //         true
    //     );
    //     TIME_END("pq_search_direct");
    // }
    
    // printf("\n");
    // TimeProfiler::getInstance().printReport();
}

void benchmark_memory_access_patterns() {
    printf("\n\n=== Memory Access Pattern Analysis ===\n");
    
    const size_t d = 128;
    const size_t M = 8;
    const size_t nbits = 8;
    const size_t ksub = 1 << nbits;
    const size_t nb = 1000000;  // 1M vectors
    
    ProductQuantizer pq(d, M, nbits);
    
    // Generate synthetic centroids
    for (size_t i = 0; i < pq.centroids.size(); i++) {
        pq.centroids[i] = (float)i / pq.centroids.size();
    }
    
    // Test different memory layouts
    printf("\nTesting regular vs transposed centroids:\n");
    
    std::vector<float> query(d);
    generate_random_data(query.data(), 1, d);
    
    std::vector<float> dis_table(M * ksub);
    
    // Regular layout
    TimeProfiler::getInstance().reset();
    for (int i = 0; i < 100; i++) {
        TIME_START();
        pq.compute_distance_table(query.data(), dis_table.data());
        TIME_END("distance_table_regular_layout");
    }
    
    // Transposed layout
    pq.sync_transposed_centroids();
    for (int i = 0; i < 100; i++) {
        TIME_START();
        pq.compute_distance_table(query.data(), dis_table.data());
        TIME_END("distance_table_transposed_layout");
    }
    
    TimeProfiler::getInstance().printReport();
    
    // Test search with different batch sizes
    printf("\n\nTesting different batch sizes:\n");
    
    std::vector<uint8_t> codes(nb * pq.code_size);
    for (size_t i = 0; i < codes.size(); i++) {
        codes[i] = i % 256;
    }
    
    const size_t batch_sizes[] = {1, 10, 100, 1000, 10000};
    const size_t k = 10;
    
    for (size_t bs : batch_sizes) {
        std::vector<float> queries(bs * d);
        generate_random_data(queries.data(), bs, d);
        
        std::vector<idx_t> labels(bs * k);
        std::vector<float> distances(bs * k);
        
        float_maxheap_array_t res = {
            size_t(bs), size_t(k), labels.data(), distances.data()
        };
        
        TimeProfiler::getInstance().reset();
        
        for (int i = 0; i < 10; i++) {
            pq.search(queries.data(), bs, codes.data(), nb, &res, true);
        }
        
        printf("\nBatch size: %zu\n", bs);
        TimeProfiler::getInstance().printReport();
    }
}

int main() {
    benchmark_pq_operations();
    // benchmark_memory_access_patterns();
    
    return 0;
}
