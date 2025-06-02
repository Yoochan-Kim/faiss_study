#ifndef FAISS_TIME_PROFILER_H
#define FAISS_TIME_PROFILER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>

namespace faiss {

class TimeProfiler {
public:
    struct TimingInfo {
        double total_time = 0.0;
        size_t count = 0;
        double min_time = std::numeric_limits<double>::max();
        double max_time = 0.0;
    };
    
private:
    std::unordered_map<std::string, TimingInfo> timings;
    std::chrono::high_resolution_clock::time_point start_time;
    
    // Singleton instance
    static TimeProfiler* instance;
    
    TimeProfiler() = default;
    
    friend class ScopedTimer;

public:
    static TimeProfiler& getInstance() {
        if (!instance) {
            instance = new TimeProfiler();
        }
        return *instance;
    }
    
    void startTimer() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void endTimer(const std::string& name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count() / 1000.0; // Convert to milliseconds
        
        auto& info = timings[name];
        info.total_time += duration;
        info.count++;
        info.min_time = std::min(info.min_time, duration);
        info.max_time = std::max(info.max_time, duration);
    }
    
    void reset() {
        timings.clear();
    }
    
    void printReport() const {
        std::cout << "\n=== PQ Time Profiling Report ===" << std::endl;
        std::cout << std::setw(40) << "Operation" 
                  << std::setw(15) << "Total (ms)"
                  << std::setw(15) << "Avg (ms)"
                  << std::setw(15) << "Min (ms)"
                  << std::setw(15) << "Max (ms)"
                  << std::setw(10) << "Count" << std::endl;
        std::cout << std::string(105, '-') << std::endl;
        
        for (const auto& [name, info] : timings) {
            double avg = info.total_time / info.count;
            std::cout << std::setw(40) << name
                      << std::setw(15) << std::fixed << std::setprecision(3) << info.total_time
                      << std::setw(15) << avg
                      << std::setw(15) << info.min_time
                      << std::setw(15) << info.max_time
                      << std::setw(10) << info.count << std::endl;
        }
        std::cout << std::string(105, '-') << std::endl;
    }
    
    // Add this method to allow ScopedTimer access
    TimingInfo& getTimingInfo(const std::string& name) {
        return timings[name];
    }
};

// Macro for easy timing
#define TIME_START() faiss::TimeProfiler::getInstance().startTimer()
#define TIME_END(name) faiss::TimeProfiler::getInstance().endTimer(name)

// RAII-based timer for automatic timing
class ScopedTimer {
private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    ScopedTimer(const std::string& n) : name(n) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    ~ScopedTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count() / 1000.0;
        
        auto& profiler = TimeProfiler::getInstance();
        auto& info = profiler.getTimingInfo(name);
        info.total_time += duration;
        info.count++;
        info.min_time = std::min(info.min_time, duration);
        info.max_time = std::max(info.max_time, duration);
    }
};

#define SCOPED_TIMER(name) faiss::ScopedTimer _timer_##__LINE__(name)

} // namespace faiss

#endif // FAISS_TIME_PROFILER_H
