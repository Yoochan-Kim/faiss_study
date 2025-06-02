#ifndef FAISS_PQ_PROFILING_UTILS_H
#define FAISS_PQ_PROFILING_UTILS_H

#include <faiss/utils/TimeProfiler.h>
#include <sstream>

// Helper macros for detailed PQ profiling
#define PQ_PROFILE_SUBSPACE(m) \
    do { \
        std::stringstream ss; \
        ss << "PQ::subspace_" << m; \
        SCOPED_TIMER(ss.str()); \
    } while(0)

#define PQ_PROFILE_BATCH(batch_id, batch_size) \
    do { \
        std::stringstream ss; \
        ss << "PQ::batch_" << batch_id << "_size_" << batch_size; \
        SCOPED_TIMER(ss.str()); \
    } while(0)

// Memory access profiling helpers
class MemoryAccessProfiler {
private:
    size_t cache_misses_estimate = 0;
    size_t total_accesses = 0;
    
public:
    void recordAccess(const void* ptr, size_t size) {
        total_accesses++;
        // Simple cache miss estimation based on access pattern
        // This is a heuristic - real cache misses would need hardware counters
        static const void* last_ptr = nullptr;
        static size_t last_size = 0;
        
        if (!last_ptr || 
            (char*)ptr < (char*)last_ptr || 
            (char*)ptr >= (char*)last_ptr + last_size + 64) { // 64 byte cache line
            cache_misses_estimate++;
        }
        
        last_ptr = ptr;
        last_size = size;
    }
    
    void reset() {
        cache_misses_estimate = 0;
        total_accesses = 0;
    }
    
    void report() const {
        if (total_accesses > 0) {
            printf("Memory Access Pattern:\n");
            printf("  Total accesses: %zu\n", total_accesses);
            printf("  Estimated cache misses: %zu (%.2f%%)\n", 
                   cache_misses_estimate, 
                   100.0 * cache_misses_estimate / total_accesses);
        }
    }
};

// Example usage in ProductQuantizer functions:
// For compute_distance_table with memory profiling:
/*
void ProductQuantizer::compute_distance_table_profiled(const float* x, float* dis_table) const {
    SCOPED_TIMER("PQ::compute_distance_table_detailed");
    MemoryAccessProfiler mem_prof;
    
    if (transposed_centroids.empty()) {
        for (size_t m = 0; m < M; m++) {
            PQ_PROFILE_SUBSPACE(m);
            
            const float* centroids_m = get_centroids(m, 0);
            mem_prof.recordAccess(centroids_m, ksub * dsub * sizeof(float));
            
            fvec_L2sqr_ny(
                dis_table + m * ksub,
                x + m * dsub,
                centroids_m,
                dsub,
                ksub);
        }
    } else {
        // Similar for transposed case
    }
    
    mem_prof.report();
}
*/

#endif // FAISS_PQ_PROFILING_UTILS_H
