/*
 * SIFT10M PQ Benchmark Demo
 * 
 * This demo benchmarks Product Quantization (PQ) with different nbits values
 * on the SIFT10M dataset, comparing recall and speedup against flat search.
 * 
 * Prerequisites:
 * - SIFT10M dataset extracted to ./SIFT10M/
 * - HDF5 library installed for reading .mat files
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <unordered_set>

#include <hdf5.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>

using namespace faiss;

/**
 * Read SIFT features from HDF5 .mat file
 * Returns transposed data if needed (n_samples x d)
 */
float* read_sift10m_features(const char* filename, size_t* n_out, size_t* d_out) {
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Could not open file %s\n", filename);
        abort();
    }
    
    // Try different possible dataset names
    const char* dataset_names[] = {"features", "data", "fea"};
    hid_t dataset_id = -1;
    
    for (const char* name : dataset_names) {
        H5E_BEGIN_TRY {
            dataset_id = H5Dopen2(file_id, name, H5P_DEFAULT);
        } H5E_END_TRY;
        
        if (dataset_id >= 0) {
            printf("Found dataset: %s\n", name);
            break;
        }
    }
    
    // If not found, try to get the first dataset
    if (dataset_id < 0) {
        hsize_t num_objs;
        H5Gget_num_objs(file_id, &num_objs);
        
        for (hsize_t i = 0; i < num_objs; i++) {
            char name[256];
            H5Gget_objname_by_idx(file_id, i, name, sizeof(name));
            
            // Skip metadata objects
            if (name[0] != '#') {
                dataset_id = H5Dopen2(file_id, name, H5P_DEFAULT);
                if (dataset_id >= 0) {
                    printf("Found dataset: %s\n", name);
                    break;
                }
            }
        }
    }
    
    if (dataset_id < 0) {
        fprintf(stderr, "Could not find feature dataset in %s\n", filename);
        H5Fclose(file_id);
        abort();
    }
    
    // Get dataset dimensions
    hid_t space_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(space_id);
    assert(ndims == 2);
    
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space_id, dims, NULL);
    
    // Read data
    size_t total_size = dims[0] * dims[1];
    float* data = new float[total_size];
    
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, 
                           H5P_DEFAULT, data);
    assert(status >= 0);
    
    // Check if we need to transpose (MATLAB stores in column-major order)
    // SIFT features are 128-dimensional, so the smaller dimension should be 128
    size_t n_samples, d;
    float* result;
    
    if (dims[0] < dims[1]) {
        // Data is (d x n_samples), need to transpose
        d = dims[0];
        n_samples = dims[1];
        result = new float[n_samples * d];
        
        // Transpose from column-major to row-major
        for (size_t i = 0; i < n_samples; i++) {
            for (size_t j = 0; j < d; j++) {
                result[i * d + j] = data[j * n_samples + i];
            }
        }
        delete[] data;
    } else {
        // Data is already (n_samples x d)
        n_samples = dims[0];
        d = dims[1];
        result = data;
    }
    
    *n_out = n_samples;
    *d_out = d;
    
    printf("Loaded %zu samples of dimension %zu\n", n_samples, d);
    
    // Cleanup
    H5Sclose(space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    
    return result;
}

/**
 * Calculate Recall@k between PQ and Flat predictions
 */
float calculate_recall(const idx_t* pq_predictions, const idx_t* flat_predictions, 
                      size_t nq, size_t k) {
    float total_recall = 0.0;
    
    for (size_t i = 0; i < nq; i++) {
        std::unordered_set<idx_t> flat_set;
        for (size_t j = 0; j < k; j++) {
            flat_set.insert(flat_predictions[i * k + j]);
        }
        
        int intersection = 0;
        for (size_t j = 0; j < k; j++) {
            idx_t pq_idx = pq_predictions[i * k + j];
            if (pq_idx >= 0 && flat_set.count(pq_idx) > 0) {
                intersection++;
            }
        }
        
        total_recall += (float)intersection / k;
    }
    
    return total_recall / nq;
}

/**
 * Timer utility
 */
class Timer {
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        return diff.count();
    }
};

int main() {
    printf("SIFT10M FAISS PQ Benchmark\n");
    printf("==================================================\n\n");
    
    // Load SIFT10M features
    size_t n_total, d;
    float* features = read_sift10m_features("SIFT10M/SIFT10Mfeatures.mat", &n_total, &d);
    
    // Split data into train and query
    const size_t n_train = 3000000;  // 3M training vectors
    const size_t n_query = 1;        // 1 query as in Python code
    const size_t k = 10;             // top-k results
    
    assert(n_total >= n_train + n_query);
    
    float* train_data = features;
    float* query_data = features + n_train * d;
    
    printf("Train data: %zu x %zu\n", n_train, d);
    printf("Query data: %zu x %zu\n\n", n_query, d);
    
    // Results storage
    std::vector<idx_t> flat_labels(n_query * k);
    std::vector<float> flat_distances(n_query * k);
    std::vector<idx_t> pq_labels(n_query * k);
    std::vector<float> pq_distances(n_query * k);
    
    Timer timer;
    
    // 1. Flat (exact) search as baseline
    printf("Running flat search...\n");
    IndexFlatL2 index_flat(d);
    
    timer.start();
    index_flat.add(n_train, train_data);
    double flat_train_time = timer.elapsed();
    
    timer.start();
    index_flat.search(n_query, query_data, k, flat_distances.data(), flat_labels.data());
    double flat_search_time = timer.elapsed();
    
    // Print results header
    printf("\nResults:\n");
    printf("====================================================================================================\n");
    printf("%-15s %-12s %-14s %-8s %-10s\n", 
           "Method", "Train Time(s)", "Search Time(s)", "Speedup", "Recall@10");
    printf("----------------------------------------------------------------------------------------------------\n");
    printf("%-15s %-12.3f %-14.6f %-8s %-10s\n", 
           "Flat", flat_train_time, flat_search_time, "1.00x", "1.0000");
    
    // 2. PQ search with different nbits
    const size_t m = 8;  // number of subquantizers
    
    for (int nbits = 16; nbits <= 18; nbits += 2) {
        printf("Running PQ search with %d bits...\n", nbits);
        
        try {
            IndexPQ index_pq(d, m, nbits);
            
            timer.start();
            index_pq.train(n_train, train_data);
            index_pq.add(n_train, train_data);
            double pq_train_time = timer.elapsed();
            
            timer.start();
            index_pq.search(n_query, query_data, k, pq_distances.data(), pq_labels.data());
            double pq_search_time = timer.elapsed();
            
            // Calculate recall
            float recall = calculate_recall(pq_labels.data(), flat_labels.data(), n_query, k);
            
            // Calculate speedup
            double speedup = flat_search_time / pq_search_time;
            
            // Print results
            char method_name[20];
            sprintf(method_name, "PQ-%d", nbits);
            printf("%-15s %-12.3f %-14.6f %-7.2fx %-10.4f\n", 
                   method_name, pq_train_time, pq_search_time, speedup, recall);
            
        } catch (const std::exception& e) {
            fprintf(stderr, "Error with PQ-%d: %s\n", nbits, e.what());
            continue;
        }
    }
    
    printf("====================================================================================================\n");
    
    // Cleanup
    delete[] features;
    
    return 0;
}
