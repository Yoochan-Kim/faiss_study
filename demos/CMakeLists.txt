# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

add_executable(demo_imi_flat EXCLUDE_FROM_ALL demo_imi_flat.cpp)
target_link_libraries(demo_imi_flat PRIVATE faiss)

add_executable(demo_imi_pq EXCLUDE_FROM_ALL demo_imi_pq.cpp)
target_link_libraries(demo_imi_pq PRIVATE faiss)

add_executable(demo_ivfpq_indexing EXCLUDE_FROM_ALL demo_ivfpq_indexing.cpp)
target_link_libraries(demo_ivfpq_indexing PRIVATE faiss)

add_executable(demo_nndescent EXCLUDE_FROM_ALL demo_nndescent.cpp)
target_link_libraries(demo_nndescent PRIVATE faiss)

add_executable(demo_sift1M EXCLUDE_FROM_ALL demo_sift1M.cpp)
target_link_libraries(demo_sift1M PRIVATE faiss)

add_executable(demo_weighted_kmeans EXCLUDE_FROM_ALL demo_weighted_kmeans.cpp)
target_link_libraries(demo_weighted_kmeans PRIVATE faiss)

add_executable(demo_residual_quantizer EXCLUDE_FROM_ALL demo_residual_quantizer.cpp)
target_link_libraries(demo_residual_quantizer PRIVATE faiss)

add_executable(demo_pq_profiling EXCLUDE_FROM_ALL demo_pq_profiling.cpp)
target_link_libraries(demo_pq_profiling PRIVATE faiss_avx2)

add_executable(demo_yoochan EXCLUDE_FROM_ALL demo_yoochan.cpp)
target_link_libraries(demo_yoochan PRIVATE faiss_avx2)

add_executable(demo_hnsw_profiling EXCLUDE_FROM_ALL demo_hnsw_profiling.cpp)
target_link_libraries(demo_hnsw_profiling PRIVATE faiss_avx2)

add_executable(demo_hnswpq_profiling EXCLUDE_FROM_ALL demo_hnswpq_profiling.cpp)
target_link_libraries(demo_hnswpq_profiling PRIVATE faiss_avx2)

add_executable(demo_hnswpq_refinement_profiling EXCLUDE_FROM_ALL demo_hnswpq_refinement_profiling.cpp)
target_link_libraries(demo_hnswpq_refinement_profiling PRIVATE faiss_avx2)

add_executable(demo_sift10m_pq_benchmark EXCLUDE_FROM_ALL demo_sift10m_pq_benchmark.cpp)
target_link_libraries(demo_sift10m_pq_benchmark PRIVATE faiss_avx2)
find_package(HDF5 REQUIRED)
target_include_directories(demo_sift10m_pq_benchmark PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(demo_sift10m_pq_benchmark PRIVATE ${HDF5_LIBRARIES})