#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/utils/TimeProfiler.h>

using namespace faiss;

// fvecs 파일 포맷을 읽어서 벡터(행렬)를 반환하는 함수
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
    // 첫 4바이트: 차원 (int)
    int d;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    if (!in) throw std::runtime_error("read error");
    dim = static_cast<size_t>(d);

    // 파일 크기로부터 벡터 개수 계산
    in.seekg(0, std::ios::end);
    std::streampos fsize = in.tellg();
    // 첫 벡터의 4바이트 차원 정보를 제외한 나머지 총 바이트 수
    std::streampos rest = fsize - sizeof(int);
    // 각 벡터당 (1 int + d floats) 바이트 크기
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

// ivecs 파일 포맷(groundtruth)에서 첫값은 k이고, 뒤에 k개의 int가 저장됨.
// 여기서는 groundtruth를 벤치마크 용도로 쓰지 않으므로 생략.

void benchmark_hnsw_sift(
    const char* base_path,    // 예: "sift/sift_base.fvecs"
    const char* query_path    // 예: "sift/sift_query.fvecs"
) {
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
    const size_t k = 10;          // 반환할 이웃 개수
    const int M = 32;             // HNSW 그래프 파라미터
    const int efConstruction = 200;
    const int efSearch = 50;

    IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;

    {
        SCOPED_TIMER("HNSW::add");
        index.add(nb, base_data.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();

    index.hnsw.efSearch = efSearch;
    std::vector<idx_t> labels(k * nq);
    std::vector<float> distances(k * nq);

    // 워밍업
    index.search(1, query_data.data(), k, distances.data(), labels.data());
    TimeProfiler::getInstance().reset();

    const int runs = 100;
    for (int i = 0; i < runs; i++) {
        SCOPED_TIMER("HNSW::search");
        index.search(nq, query_data.data(), k, distances.data(), labels.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();
}

void benchmark_ivfpq_sift(
    const char* learn_path,   // 예: "sift/sift_learn.fvecs"
    const char* base_path,    // 예: "sift/sift_base.fvecs"
    const char* query_path    // 예: "sift/sift_query.fvecs"
) {
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
        throw std::runtime_error("dimension mismatch among learn, base, query");
    }
    size_t d = d_base;
    const size_t nlist = 100;
    const size_t M = 8;
    const size_t nbits = 8;
    const size_t k = 10;
    const size_t nprobe = 10;

    IndexFlatL2 quantizer(d);
    IndexIVFPQ index(&quantizer, d, nlist, M, nbits);

    {
        SCOPED_TIMER("IVFPQ::train");
        index.train(n_learn, learn_data.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();

    {
        SCOPED_TIMER("IVFPQ::add");
        index.add(nb, base_data.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();

    index.nprobe = nprobe;
    std::vector<idx_t> labels(k * nq);
    std::vector<float> distances(k * nq);

    // 워밍업
    index.search(1, query_data.data(), k, distances.data(), labels.data());
    TimeProfiler::getInstance().reset();

    const int runs = 100;
    for (int i = 0; i < runs; i++) {
        SCOPED_TIMER("IVFPQ::search");
        index.search(nq, query_data.data(), k, distances.data(), labels.data());
    }
    TimeProfiler::getInstance().printReport();
    TimeProfiler::getInstance().reset();
}

int main() {
    try {
        // HNSW 벤치마크: base, query만 사용
        benchmark_hnsw_sift(
            "sift/sift_base.fvecs",
            "sift/sift_query.fvecs"
        );

        // IVFPQ 벤치마크: learn, base, query 사용
        benchmark_ivfpq_sift(
            "sift/sift_learn.fvecs",
            "sift/sift_base.fvecs",
            "sift/sift_query.fvecs"
        );
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
