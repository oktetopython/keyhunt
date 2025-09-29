/*
 * Simple benchmark without external dependencies
 * Tests optimization concepts for PUZZLE71
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cstring>
#include <thread>
#include <cstdint>

// Simulate target hash for Puzzle #71
const uint32_t TARGET_HASH[5] = {
    0x225b45f8,  // Bytes 0-3 (little-endian)
    0x409a46fa,  // Bytes 4-7
    0xd3504465,  // Bytes 8-11
    0x3b9a9563,  // Bytes 12-15
    0xb4242993   // Bytes 16-19
};

// Simple hash simulation (not real crypto, just for benchmarking)
void compute_hash_simple(const uint64_t* input, uint32_t* output) {
    // Simple mixing function for simulation
    output[0] = (uint32_t)(input[0] ^ (input[0] >> 32));
    output[1] = (uint32_t)(input[1] ^ (input[1] >> 32));
    output[2] = (uint32_t)(input[2] ^ (input[2] >> 32));
    output[3] = (uint32_t)(input[3] ^ (input[3] >> 32));
    output[4] = (uint32_t)((input[0] + input[1]) ^ (input[2] + input[3]));
    
    // Additional mixing
    for (int i = 0; i < 5; i++) {
        output[i] = (output[i] * 0x9E3779B1) ^ (output[i] >> 16);
    }
}

// Check hash with memory indirection (baseline)
bool check_hash_baseline(const uint32_t* hash, const uint32_t* target) {
    for (int i = 0; i < 5; i++) {
        if (hash[i] != target[i]) return false;
    }
    return true;
}

// Check hash with hardcoded values (optimized)
bool check_hash_hardcoded(const uint32_t* hash) {
    return hash[0] == 0x225b45f8 &&
           hash[1] == 0x409a46fa &&
           hash[2] == 0xd3504465 &&
           hash[3] == 0x3b9a9563 &&
           hash[4] == 0xb4242993;
}

// Baseline benchmark - single key processing
double benchmark_baseline(uint64_t num_keys) {
    std::cout << "Running baseline benchmark (single key processing)..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::mt19937_64 rng(12345);
    uint64_t matches = 0;
    
    for (uint64_t k = 0; k < num_keys; k++) {
        // Generate random key
        uint64_t key[4];
        key[0] = rng();
        key[1] = rng();
        key[2] = rng();
        key[3] = rng();
        
        // Compute hash
        uint32_t hash[5];
        compute_hash_simple(key, hash);
        
        // Check against target
        if (check_hash_baseline(hash, TARGET_HASH)) {
            matches++;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double keys_per_second = num_keys / seconds;
    
    std::cout << "  Keys tested: " << num_keys << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << seconds << " seconds" << std::endl;
    std::cout << "  Speed: " << std::fixed << std::setprecision(0) << keys_per_second << " keys/sec" << std::endl;
    
    return keys_per_second;
}

// Optimized with hardcoded target
double benchmark_hardcoded(uint64_t num_keys) {
    std::cout << "Running hardcoded target benchmark..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::mt19937_64 rng(12345);
    uint64_t matches = 0;
    
    for (uint64_t k = 0; k < num_keys; k++) {
        uint64_t key[4];
        key[0] = rng();
        key[1] = rng();
        key[2] = rng();
        key[3] = rng();
        
        uint32_t hash[5];
        compute_hash_simple(key, hash);
        
        // Direct comparison without memory indirection
        if (check_hash_hardcoded(hash)) {
            matches++;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double keys_per_second = num_keys / seconds;
    
    std::cout << "  Keys tested: " << num_keys << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << seconds << " seconds" << std::endl;
    std::cout << "  Speed: " << std::fixed << std::setprecision(0) << keys_per_second << " keys/sec" << std::endl;
    
    return keys_per_second;
}

// Batch processing optimization
double benchmark_batch(uint64_t num_keys, size_t batch_size = 16) {
    std::cout << "Running batch processing benchmark (batch=" << batch_size << ")..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::mt19937_64 rng(12345);
    uint64_t matches = 0;
    
    // Process in batches
    std::vector<uint64_t> batch_keys(batch_size * 4);
    std::vector<uint32_t> batch_hashes(batch_size * 5);
    
    for (uint64_t batch_start = 0; batch_start < num_keys; batch_start += batch_size) {
        size_t current_batch = std::min((uint64_t)batch_size, num_keys - batch_start);
        
        // Generate batch of keys
        for (size_t i = 0; i < current_batch; i++) {
            batch_keys[i * 4 + 0] = rng();
            batch_keys[i * 4 + 1] = rng();
            batch_keys[i * 4 + 2] = rng();
            batch_keys[i * 4 + 3] = rng();
        }
        
        // Compute batch of hashes
        for (size_t i = 0; i < current_batch; i++) {
            compute_hash_simple(&batch_keys[i * 4], &batch_hashes[i * 5]);
        }
        
        // Check batch results
        for (size_t i = 0; i < current_batch; i++) {
            if (check_hash_hardcoded(&batch_hashes[i * 5])) {
                matches++;
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double keys_per_second = num_keys / seconds;
    
    std::cout << "  Keys tested: " << num_keys << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << seconds << " seconds" << std::endl;
    std::cout << "  Speed: " << std::fixed << std::setprecision(0) << keys_per_second << " keys/sec" << std::endl;
    
    return keys_per_second;
}

// Multi-threaded benchmark
double benchmark_multithread(uint64_t num_keys, int num_threads) {
    std::cout << "Running multi-threaded benchmark (" << num_threads << " threads)..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    std::vector<uint64_t> thread_matches(num_threads, 0);
    uint64_t keys_per_thread = num_keys / num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([t, keys_per_thread, &thread_matches]() {
            std::mt19937_64 rng(12345 + t);
            uint64_t matches = 0;
            
            for (uint64_t k = 0; k < keys_per_thread; k++) {
                uint64_t key[4];
                key[0] = rng();
                key[1] = rng();
                key[2] = rng();
                key[3] = rng();
                
                uint32_t hash[5];
                compute_hash_simple(key, hash);
                
                if (check_hash_hardcoded(hash)) {
                    matches++;
                }
            }
            
            thread_matches[t] = matches;
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double keys_per_second = num_keys / seconds;
    
    std::cout << "  Keys tested: " << num_keys << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << seconds << " seconds" << std::endl;
    std::cout << "  Speed: " << std::fixed << std::setprecision(0) << keys_per_second << " keys/sec" << std::endl;
    
    return keys_per_second;
}

// Combined optimizations
double benchmark_combined(uint64_t num_keys) {
    std::cout << "Running fully optimized benchmark (batch + multithread)..." << std::endl;
    
    int num_threads = std::thread::hardware_concurrency();
    const size_t batch_size = 32;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    uint64_t keys_per_thread = num_keys / num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([t, keys_per_thread, batch_size]() {
            std::mt19937_64 rng(12345 + t);
            uint64_t matches = 0;
            
            std::vector<uint64_t> batch_keys(batch_size * 4);
            std::vector<uint32_t> batch_hashes(batch_size * 5);
            
            for (uint64_t batch_start = 0; batch_start < keys_per_thread; batch_start += batch_size) {
                size_t current_batch = std::min((uint64_t)batch_size, keys_per_thread - batch_start);
                
                // Generate and process batch
                for (size_t i = 0; i < current_batch; i++) {
                    batch_keys[i * 4 + 0] = rng();
                    batch_keys[i * 4 + 1] = rng();
                    batch_keys[i * 4 + 2] = rng();
                    batch_keys[i * 4 + 3] = rng();
                    
                    compute_hash_simple(&batch_keys[i * 4], &batch_hashes[i * 5]);
                    
                    if (check_hash_hardcoded(&batch_hashes[i * 5])) {
                        matches++;
                    }
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double keys_per_second = num_keys / seconds;
    
    std::cout << "  Keys tested: " << num_keys << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << seconds << " seconds" << std::endl;
    std::cout << "  Speed: " << std::fixed << std::setprecision(0) << keys_per_second << " keys/sec" << std::endl;
    
    return keys_per_second;
}

int main(int argc, char* argv[]) {
    std::cout << "\n================================================" << std::endl;
    std::cout << "     PUZZLE71 CPU Benchmark Simulation" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    uint64_t num_keys = 10000000;  // 10 million keys
    if (argc > 1) {
        num_keys = std::stoull(argv[1]);
    }
    
    int num_cores = std::thread::hardware_concurrency();
    std::cout << "System info:" << std::endl;
    std::cout << "  CPU cores: " << num_cores << std::endl;
    std::cout << "  Test size: " << num_keys << " keys\n" << std::endl;
    
    // Run benchmarks
    double baseline = benchmark_baseline(num_keys);
    std::cout << std::endl;
    
    double hardcoded = benchmark_hardcoded(num_keys);
    std::cout << std::endl;
    
    double batch16 = benchmark_batch(num_keys, 16);
    std::cout << std::endl;
    
    double batch32 = benchmark_batch(num_keys, 32);
    std::cout << std::endl;
    
    double multithread = benchmark_multithread(num_keys, num_cores);
    std::cout << std::endl;
    
    double combined = benchmark_combined(num_keys);
    std::cout << std::endl;
    
    // Results
    std::cout << "================================================" << std::endl;
    std::cout << "           OPTIMIZATION RESULTS" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    std::cout << "Speedup relative to baseline:" << std::endl;
    std::cout << "  Baseline:            1.00x" << std::endl;
    std::cout << "  Hardcoded target:    " << std::fixed << std::setprecision(2) 
              << hardcoded / baseline << "x" << std::endl;
    std::cout << "  Batch (16):          " << std::fixed << std::setprecision(2) 
              << batch16 / baseline << "x" << std::endl;
    std::cout << "  Batch (32):          " << std::fixed << std::setprecision(2) 
              << batch32 / baseline << "x" << std::endl;
    std::cout << "  Multi-thread:        " << std::fixed << std::setprecision(2) 
              << multithread / baseline << "x" << std::endl;
    std::cout << "  Combined (all):      " << std::fixed << std::setprecision(2) 
              << combined / baseline << "x\n" << std::endl;
    
    std::cout << "Note: This is CPU simulation. GPU would provide:" << std::endl;
    std::cout << "  - Much higher base throughput" << std::endl;
    std::cout << "  - Better memory bandwidth utilization" << std::endl;
    std::cout << "  - Additional optimizations (warp-level, tensor cores)" << std::endl;
    std::cout << "  - Expected GPU speedup: 3-5x with all optimizations\n" << std::endl;
    
    return 0;
}