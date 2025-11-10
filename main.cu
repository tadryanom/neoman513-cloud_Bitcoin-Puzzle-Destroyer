// author: https://t.me/biernus
// Modified: Configurable step size with wrap-around from max to min
#include "secp256k1.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <fstream>
#include <stdint.h>
#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <inttypes.h>
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#include <chrono>
#pragma once

__device__ __host__ __forceinline__ uint8_t hex_char_to_byte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

// Convert hex string to bytes
__device__ __host__ void hex_string_to_bytes(const char* hex_str, uint8_t* bytes, int num_bytes) {
    #pragma unroll 8
    for (int i = 0; i < num_bytes; i++) {
        bytes[i] = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                   hex_char_to_byte(hex_str[i * 2 + 1]);
    }
}

// Convert hex string to BigInt - optimized
__device__ __host__ void hex_to_bigint(const char* hex_str, BigInt* bigint) {
    // Initialize all data to 0
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }
    
    int len = 0;
    while (hex_str[len] != '\0' && len < 64) len++;
    
    // Process hex string from right to left
    int word_idx = 0;
    int bit_offset = 0;
    
    for (int i = len - 1; i >= 0 && word_idx < 8; i--) {
        uint8_t val = hex_char_to_byte(hex_str[i]);
        
        bigint->data[word_idx] |= ((uint32_t)val << bit_offset);
        
        bit_offset += 4;
        if (bit_offset >= 32) {
            bit_offset = 0;
            word_idx++;
        }
    }
}

// Convert BigInt to hex string - optimized
__device__ void bigint_to_hex(const BigInt* bigint, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int idx = 0;
    bool leading_zero = true;
    
    // Process from most significant word to least
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        for (int j = 28; j >= 0; j -= 4) {
            uint8_t nibble = (bigint->data[i] >> j) & 0xF;
            if (nibble != 0 || !leading_zero || (i == 0 && j == 0)) {
                hex_str[idx++] = hex_chars[nibble];
                leading_zero = false;
            }
        }
    }
    
    // Handle case where number is 0
    if (idx == 0) {
        hex_str[idx++] = '0';
    }
    
    hex_str[idx] = '\0';
}

// Optimized byte to hex conversion
__device__ __forceinline__ void byte_to_hex(uint8_t byte, char* out) {
    const char hex_chars[] = "0123456789abcdef";
    out[0] = hex_chars[(byte >> 4) & 0xF];
    out[1] = hex_chars[byte & 0xF];
}

__device__ void hash160_to_hex(uint8_t* hash, char* hex_str) {
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        byte_to_hex(hash[i], &hex_str[i * 2]);
    }
    hex_str[40] = '\0';
}

__device__ __forceinline__ bool compare_hash160_fast(const uint8_t* hash1, const uint8_t* hash2) {
    uint64_t a1, a2, b1, b2;
    uint32_t c1, c2;
    
    memcpy(&a1, hash1, 8);
    memcpy(&a2, hash1 + 8, 8);
    memcpy(&c1, hash1 + 16, 4);

    memcpy(&b1, hash2, 8);
    memcpy(&b2, hash2 + 8, 8);
    memcpy(&c2, hash2 + 16, 4);

    return (a1 == b1) && (a2 == b2) && (c1 == c2);
}

__device__ void hash160_to_hex(const uint8_t *hash, char *out_hex) {
    const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 20; ++i) {
        out_hex[i * 2]     = hex_chars[hash[i] >> 4];
        out_hex[i * 2 + 1] = hex_chars[hash[i] & 0x0F];
    }
    out_hex[40] = '\0';
}

// Global device constants for min/max as BigInt
__constant__ BigInt d_min_bigint;
__constant__ BigInt d_max_bigint;
__constant__ BigInt d_step_size;
__constant__ BigInt d_range_size; // NEW: For wrap-around calculation

__device__ volatile int g_found = 0;
__device__ char g_found_hex[65] = {0};
__device__ char g_found_hash160[41] = {0};

// Per-thread persistent state (stored in global memory)
__device__ BigInt* d_thread_keys;

// Helper function to add with wrap-around
__device__ void add_with_wraparound(BigInt* result, const BigInt* a, const BigInt* b) {
    // result = a + b
    ptx_u256Add(result, a, b);
    
    // If result > max, wrap: result = min + (result - max - 1)
    if (compare_bigint(result, &d_max_bigint) > 0) {
        // Calculate overflow: result - max - 1
        BigInt overflow;
        ptx_u256Sub(&overflow, result, &d_max_bigint);
        
        // Subtract 1 from overflow
        BigInt one;
        init_bigint(&one, 1);
        ptx_u256Sub(&overflow, &overflow, &one);
        
        // Wrap: result = min + overflow
        ptx_u256Add(result, &d_min_bigint, &overflow);
    }
}

__global__ void start(const uint8_t* target, uint64_t iteration)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load this thread's current private key from global memory
    BigInt priv_base = d_thread_keys[tid];
    
    // Early exit: Check if already found by another thread
    if (g_found) return;
    
    // Array to hold batch of points
    ECPointJac result_jac_batch[BATCH_SIZE];
    uint8_t hash160_batch[BATCH_SIZE][20];
    
    // --- Compute base point: P = priv_base * G ---
    scalar_multiply_multi_base_jac(&result_jac_batch[0], &priv_base);
    
    // --- Generate sequential points with step_size: P+step, P+2*step, P+3*step, ... ---
    BigInt current_key = priv_base;
    #pragma unroll
    for (int i = 1; i < BATCH_SIZE; ++i) {
        // current_key = current_key + step_size (with wrap-around)
        BigInt next_key;
        add_with_wraparound(&next_key, &current_key, &d_step_size);
        
        // Reduce modulo n if needed
        if (compare_bigint(&next_key, &const_n) >= 0) {
            ptx_u256Sub(&next_key, &next_key, &const_n);
        }
        
        current_key = next_key;
        
        // Compute point for this key
        scalar_multiply_multi_base_jac(&result_jac_batch[i], &current_key);
    }
    
    // --- Convert the entire batch to hash160s with ONE inverse ---
    jacobian_batch_to_hash160(result_jac_batch, hash160_batch);
    
    // Debug print for first thread only (every 1000 iterations)
    if (tid == 0 && iteration % 1000 == 0) {
        char hash160_str[41];
        char hex_key[65];
        bigint_to_hex(&priv_base, hex_key);
        hash160_to_hex(hash160_batch[0], hash160_str);
        printf("Thread %d walking - Key: %s -> %s\n", tid, hex_key, hash160_str);
    }
    
    // --- Optimized batch checking with early exit ---
    BigInt key_for_check = priv_base;
    #pragma unroll
    for (int i = 0; i < BATCH_SIZE; ++i) {
        // Check if another thread already found it
        if (g_found) return;
        
        if (compare_hash160_fast(hash160_batch[i], target)) {
            if (atomicCAS((int*)&g_found, 0, 1) == 0) {
                // The actual private key for this index
                char found_hex[65];
                bigint_to_hex(&key_for_check, found_hex);
                hash160_to_hex(hash160_batch[i], g_found_hash160);
                memcpy(g_found_hex, found_hex, 65);
                return;
            }
        }
        
        // Update key for next iteration
        if (i < BATCH_SIZE - 1) {
            BigInt next_key;
            add_with_wraparound(&next_key, &key_for_check, &d_step_size);
            if (compare_bigint(&next_key, &const_n) >= 0) {
                ptx_u256Sub(&next_key, &next_key, &const_n);
            }
            key_for_check = next_key;
        }
    }
    
    // --- Update this thread's key for next iteration: priv_base + (BATCH_SIZE * step_size) ---
    BigInt total_offset;
    
    // Calculate total_offset = BATCH_SIZE * step_size
    init_bigint(&total_offset, 0);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        BigInt temp;
        ptx_u256Add(&temp, &total_offset, &d_step_size);
        if (compare_bigint(&temp, &const_n) >= 0) {
            ptx_u256Sub(&temp, &temp, &const_n);
        }
        total_offset = temp;
    }
    
    BigInt next_key;
    add_with_wraparound(&next_key, &priv_base, &total_offset);
    
    // Reduce modulo n if needed
    if (compare_bigint(&next_key, &const_n) >= 0) {
        ptx_u256Sub(&next_key, &next_key, &const_n);
    }
    
    // Store updated key back to global memory
    d_thread_keys[tid] = next_key;
}

bool run_with_quantum_data(const char* min, const char* max, const char* target, 
                           uint64_t step_size, int blocks, int threads, int device_id) {
    uint8_t shared_target[20];
    hex_string_to_bytes(target, shared_target, 20);
    uint8_t *d_target;
    cudaMalloc(&d_target, 20);
    cudaMemcpy(d_target, shared_target, 20, cudaMemcpyHostToDevice);
    
    // Convert min and max hex strings to BigInt and copy to device
    BigInt min_bigint, max_bigint;
    hex_to_bigint(min, &min_bigint);
    hex_to_bigint(max, &max_bigint);
    
    cudaMemcpyToSymbol(d_min_bigint, &min_bigint, sizeof(BigInt));
    cudaMemcpyToSymbol(d_max_bigint, &max_bigint, sizeof(BigInt));
    
    // Calculate range size: max - min + 1
    BigInt range_size;
    bool borrow = false;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t diff = (uint64_t)max_bigint.data[i] - (uint64_t)min_bigint.data[i] - (borrow ? 1 : 0);
        range_size.data[i] = (uint32_t)diff;
        borrow = (diff > 0xFFFFFFFFULL);
    }
    // Add 1 to get inclusive range (host-compatible addition)
    bool carry = true; // Adding 1
    for (int i = 0; i < BIGINT_WORDS && carry; ++i) {
        uint64_t sum = (uint64_t)range_size.data[i] + 1;
        range_size.data[i] = (uint32_t)sum;
        carry = (sum > 0xFFFFFFFFULL);
    }
    
    cudaMemcpyToSymbol(d_range_size, &range_size, sizeof(BigInt));
    
    // Convert step_size to BigInt and copy to device
    BigInt step_bigint;
    init_bigint(&step_bigint, step_size);
    cudaMemcpyToSymbol(d_step_size, &step_bigint, sizeof(BigInt));
    
    int total_threads = blocks * threads;
    int found_flag;
    
    // Calculate keys processed per kernel launch
    uint64_t keys_per_kernel = (uint64_t)total_threads * BATCH_SIZE * step_size;
    
    printf("Searching in range (with wrap-around enabled):\n");
    printf("Min: %s\n", min);
    printf("Max: %s\n", max);
    printf("Target: %s\n", target);
    printf("Step size: %llu\n", (unsigned long long)step_size);
    printf("Blocks: %d, Threads: %d, Batch size: %d\n", blocks, threads, BATCH_SIZE);
    printf("Total threads: %d\n", total_threads);
    printf("Keys per kernel: %llu (each thread checks %d keys with step +%llu)\n", 
           (unsigned long long)keys_per_kernel, BATCH_SIZE, (unsigned long long)step_size);
    printf("Note: When reaching max, keys wrap around to min automatically\n\n");
    
    // Allocate device memory for per-thread persistent keys
    BigInt* d_thread_keys_ptr;
    cudaMalloc(&d_thread_keys_ptr, total_threads * sizeof(BigInt));
    cudaMemcpyToSymbol(d_thread_keys, &d_thread_keys_ptr, sizeof(BigInt*));
    
    // Pre-calculate range
    BigInt range;
    borrow = false;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t diff = (uint64_t)max_bigint.data[i] - (uint64_t)min_bigint.data[i] - (borrow ? 1 : 0);
        range.data[i] = (uint32_t)diff;
        borrow = (diff > 0xFFFFFFFFULL);
    }
    
    // Find highest non-zero word and create mask
    int highest_word = BIGINT_WORDS - 1;
    while (highest_word >= 0 && range.data[highest_word] == 0) {
        highest_word--;
    }
    
    uint32_t mask = 0xFFFFFFFF;
    if (highest_word >= 0) {
        mask = range.data[highest_word];
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
    }
    
    // Initialize random starting points for each thread
    std::vector<BigInt> thread_keys(total_threads);
    uint64_t seed;
    BCryptGenRandom(NULL, (PUCHAR)&seed, sizeof(seed), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    std::knuth_b gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);
    
    printf("Initializing %d random starting points for threads...\n", total_threads);
    
    for (int tid = 0; tid < total_threads; ++tid) {
        BigInt random_val;
        
        // Generate random value in [0, range]
        for (int i = 0; i < BIGINT_WORDS; ++i) {
            random_val.data[i] = dis(gen);
        }
        
        // Apply mask to fit within range
        if (highest_word >= 0) {
            random_val.data[highest_word] &= mask;
            for (int i = highest_word + 1; i < BIGINT_WORDS; ++i) {
                random_val.data[i] = 0;
            }
        }
        
        // thread_keys[tid] = random_val + min
        bool carry = false;
        for (int i = 0; i < BIGINT_WORDS; ++i) {
            uint64_t sum = (uint64_t)random_val.data[i] + (uint64_t)min_bigint.data[i] + (carry ? 1 : 0);
            thread_keys[tid].data[i] = (uint32_t)sum;
            carry = (sum > 0xFFFFFFFFULL);
        }
    }
    
    // Copy initialized keys to device
    cudaMemcpy(d_thread_keys_ptr, thread_keys.data(), total_threads * sizeof(BigInt), cudaMemcpyHostToDevice);
    
    printf("Starting persistent walking search with wrap-around (step size %llu)...\n\n", (unsigned long long)step_size);
    
    // Performance tracking variables
    uint64_t total_keys_checked = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_report_time = start_time;
    uint64_t iteration = 0;
    
    // Create CUDA stream for async operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    while(true) {
        // Launch kernel with stream
        start<<<blocks, threads, 0, stream>>>(d_target, iteration);
        
        // Asynchronous check - only sync when needed
        cudaError_t err = cudaStreamQuery(stream);
        if (err == cudaSuccess) {
            // Update counters
            total_keys_checked += keys_per_kernel;
            iteration++;
            
            // Check if found
            cudaMemcpyFromSymbol(&found_flag, g_found, sizeof(int));
            if (found_flag) {
                cudaStreamSynchronize(stream);
                printf("\n\n");
                
                char found_hex[65], found_hash160[41];
                cudaMemcpyFromSymbol(found_hex, g_found_hex, 65);
                cudaMemcpyFromSymbol(found_hash160, g_found_hash160, 41);
                
                double total_time = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - start_time
                ).count();
                
                printf("FOUND!\n");
                printf("Private Key: %s\n", found_hex);
                printf("Hash160: %s\n", found_hash160);
                printf("Step size: %llu\n", (unsigned long long)step_size);
                printf("Iterations: %llu\n", iteration);
                printf("Total time: %.2f seconds\n", total_time);
                printf("Total keys checked: %llu (%.2f million)\n", 
                       (unsigned long long)total_keys_checked,
                       total_keys_checked / 1000000.0);
                printf("Average speed: %.2f MK/s\n", total_keys_checked / total_time / 1000000.0);
                
                std::ofstream outfile("result.txt", std::ios::app);
                if (outfile.is_open()) {
                    std::time_t now = std::time(nullptr);
                    char timestamp[100];
                    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                    outfile << "[" << timestamp << "] Found: " << found_hex << " -> " << found_hash160 << std::endl;
                    outfile << "Step size: " << step_size << std::endl;
                    outfile << "Iterations: " << iteration << std::endl;
                    outfile << "Total keys checked: " << total_keys_checked << std::endl;
                    outfile << "Time taken: " << total_time << " seconds" << std::endl;
                    outfile << "Average speed: " << (total_keys_checked / total_time / 1000000.0) << " MK/s" << std::endl;
                    outfile << std::endl;
                    outfile.close();
                    std::cout << "Result appended to result.txt" << std::endl;
                }
                
                cudaStreamDestroy(stream);
                cudaFree(d_thread_keys_ptr);
                cudaFree(d_target);
                return true;
            }
            
            // Progress report every 10 seconds
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_since_report = std::chrono::duration<double>(now - last_report_time).count();
            if (elapsed_since_report >= 10.0) {
                double total_time = std::chrono::duration<double>(now - start_time).count();
                double speed = total_keys_checked / total_time / 1000000.0;
                printf("Progress: %llu iterations, %.2f MK checked (step: %llu), %.2f MK/s\n", 
                       iteration, total_keys_checked / 1000000.0, (unsigned long long)step_size, speed);
                last_report_time = now;
            }
        } else if (err != cudaErrorNotReady) {
            // Handle actual errors
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            cudaStreamSynchronize(stream);
        }
    }
    
    cudaStreamDestroy(stream);
    cudaFree(d_thread_keys_ptr);
    cudaFree(d_target);
    return false;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <min> <max> <target> <step_size> [device_id]" << std::endl;
        std::cerr << "Example: " << argv[0] << " 8000000000000000 ffffffffffffffff <target_hash> 10" << std::endl;
        std::cerr << "  step_size: increment between checked keys (1=sequential, 10=skip 9, etc.)" << std::endl;
        std::cerr << "  Note: Keys automatically wrap from max to min" << std::endl;
        return 1;
    }
    int blocks = 256;
    int threads = 256;
    uint64_t step_size = std::stoull(argv[4]);
    int device_id = (argc > 5) ? std::stoi(argv[5]) : 0;
    
    if (step_size == 0) {
        std::cerr << "Error: step_size must be greater than 0" << std::endl;
        return 1;
    }
    
    // Set GPU device
    cudaSetDevice(device_id);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error setting device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Validate input lengths match
    if (strlen(argv[1]) != strlen(argv[2])) {
        std::cerr << "Error: min and max must have the same length" << std::endl;
        return 1;
    }
    init_gpu_constants();
    cudaDeviceSynchronize();
    bool result = run_with_quantum_data(argv[1], argv[2], argv[3], step_size, blocks, threads, device_id);
    
    return 0;
}