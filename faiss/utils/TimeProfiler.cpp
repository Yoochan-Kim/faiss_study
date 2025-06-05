#include <faiss/utils/TimeProfiler.h>

namespace faiss {

// Static member definitions
TimeProfiler* TimeProfiler::instance = nullptr;
std::once_flag TimeProfiler::init_flag;

} // namespace faiss
