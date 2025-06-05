#ifndef FAISS_TIME_PROFILER_H
#define FAISS_TIME_PROFILER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <thread>
#include <sstream>
#include <atomic>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace faiss {

class TimeProfiler {
public:
    struct TimingInfo {
        std::atomic<double> total_time{0.0};
        std::atomic<size_t> count{0};
        std::atomic<double> min_time{std::numeric_limits<double>::max()};
        std::atomic<double> max_time{0.0};
        std::atomic<double> first_start_time{-1.0};
        
        // For atomic updates
        void updateTiming(double duration, double start_time) {
            // Update total time
            double current = total_time.load();
            while (!total_time.compare_exchange_weak(current, current + duration));
            
            // Update count
            count.fetch_add(1);
            
            // Update min/max (not perfectly atomic but good enough for profiling)
            double current_min = min_time.load();
            while (duration < current_min && 
                   !min_time.compare_exchange_weak(current_min, duration));
            
            double current_max = max_time.load();
            while (duration > current_max && 
                   !max_time.compare_exchange_weak(current_max, duration));
            
            // Set first start time if not set
            double expected = -1.0;
            first_start_time.compare_exchange_strong(expected, start_time);
        }
    };
    
    // Simple copyable struct for printing
    struct TimingSnapshot {
        double total_time;
        size_t count;
        double min_time;
        double max_time;
        double first_start_time;
        
        TimingSnapshot(const TimingInfo& info) 
            : total_time(info.total_time.load())
            , count(info.count.load())
            , min_time(info.min_time.load())
            , max_time(info.max_time.load())
            , first_start_time(info.first_start_time.load()) {}
    };
    
private:
    // Thread-safe timing storage
    mutable std::mutex timings_mutex;
    std::unordered_map<std::string, TimingInfo> timings;
    
    // Singleton instance
    static TimeProfiler* instance;
    static std::once_flag init_flag;
    
    TimeProfiler() = default;
    
    // Get thread-specific name
    std::string getThreadSpecificName(const std::string& base_name) const {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#else
        static std::atomic<int> thread_counter{0};
        static thread_local int tl_id = -1;
        if (tl_id == -1) {
            tl_id = thread_counter.fetch_add(1);
        }
        thread_id = tl_id;
#endif
        return base_name + "#" + std::to_string(thread_id);
    }

public:
    static TimeProfiler& getInstance() {
        std::call_once(init_flag, []() {
            instance = new TimeProfiler();
        });
        return *instance;
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(timings_mutex);
        timings.clear();
    }
    
    void printReport() const {
        std::lock_guard<std::mutex> lock(timings_mutex);

        std::cout << "\n=== Time Profiling Report ===" << std::endl;
        std::cout << std::setw(50) << "Operation"
                << std::setw(15) << "Total (ms)"
                << std::setw(15) << "Avg (ms)"
                << std::setw(15) << "Min (ms)"
                << std::setw(15) << "Max (ms)"
                << std::setw(10) << "Count" << std::endl;
        std::cout << std::string(130, '-') << std::endl;

        /* utility lambdas */
        auto baseName = [](const std::string& n) {
            size_t p = n.find('#');
            return (p != std::string::npos) ? n.substr(0, p) : n;
        };
        auto threadNum = [](const std::string& n) {
            size_t p = n.find('#');
            return (p != std::string::npos) ? std::stoi(n.substr(p + 1)) : 0;
        };

        /* 1) bucket by base name */
        using SnapPair = std::pair<std::string, TimingSnapshot>;
        std::unordered_map<std::string, std::vector<SnapPair>> buckets;
        for (const auto& [name, info] : timings) {
            buckets[baseName(name)].emplace_back(name, TimingSnapshot(info));
        }

        /* 2) determine order of buckets by earliest start time */
        std::vector<std::pair<std::string, double>> order;
        for (const auto& [bname, vec] : buckets) {
            double earliest = std::numeric_limits<double>::max();
            for (const auto& p : vec) {
                earliest = std::min(earliest, p.second.first_start_time);
            }
            order.emplace_back(bname, earliest);
        }
        std::sort(order.begin(), order.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

        /* 3) output: each bucket internally sorted by thread id */
        for (const auto& [bname, _] : order) {
            auto& vec = buckets[bname];
            std::sort(vec.begin(), vec.end(),
                    [&](const SnapPair& a, const SnapPair& b) {
                        return threadNum(a.first) < threadNum(b.first);
                    });

            for (const auto& [name, snapshot] : vec) {
                if (snapshot.count == 0)
                    continue;
                double avg = snapshot.total_time / snapshot.count;
                std::cout << std::setw(50) << name
                        << std::setw(15) << std::fixed << std::setprecision(3) << snapshot.total_time
                        << std::setw(15) << avg
                        << std::setw(15) << snapshot.min_time
                        << std::setw(15) << snapshot.max_time
                        << std::setw(10) << snapshot.count << std::endl;
            }
        }
        std::cout << std::string(130, '-') << std::endl;
    }
    
    // Thread-safe timing update
    void recordTiming(const std::string& name, double duration, double start_time) {
        std::string thread_name = getThreadSpecificName(name);
        
        // Try to update existing entry first (common case)
        {
            std::lock_guard<std::mutex> lock(timings_mutex);
            auto it = timings.find(thread_name);
            if (it != timings.end()) {
                it->second.updateTiming(duration, start_time);
                return;
            }
        }
        
        // Create new entry if needed
        {
            std::lock_guard<std::mutex> lock(timings_mutex);
            timings[thread_name].updateTiming(duration, start_time);
        }
    }
};

// RAII-based timer for automatic timing
class ScopedTimer {
private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start_time;
    double start_time_ms;
    bool is_active;
    
public:
    ScopedTimer(const std::string& n) : name(n), is_active(true) {
        start_time = std::chrono::high_resolution_clock::now();
        // Store start time in milliseconds since epoch for sorting
        auto epoch = std::chrono::time_point<std::chrono::high_resolution_clock>{};
        start_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            start_time - epoch).count();
    }
    
    ~ScopedTimer() {
        if (is_active) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count() / 1000.0;
            
            TimeProfiler::getInstance().recordTiming(name, duration, start_time_ms);
        }
    }
    
    // Prevent copy/move
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;
};

// Backward compatibility macros
#define TIME_START() do {} while(0)
#define TIME_END(name) do {} while(0)
#define SCOPED_TIMER(name) faiss::ScopedTimer _timer_##__LINE__(name)

} // namespace faiss

#endif // FAISS_TIME_PROFILER_H