// Memory Optimization Header for Arduino UNO Q4GB
// Optimizes memory usage for TinyML models on constrained devices

#ifndef MEMORY_OPT_H
#define MEMORY_OPT_H

#include <stdint.h>

// Memory configuration for Arduino UNO Q4GB
#define TOTAL_RAM_KB 4096           // 4GB RAM available
#define STACK_SIZE_BYTES 2048       // Reserve 2KB for stack
#define HEAP_SIZE_BYTES (4096 - 2)  // 4GB - stack (simplified)
#define MODEL_RAM_KB 512            // Allocate 512KB for AI models
#define SENSOR_BUFFER_BYTES 1024     // 1KB for sensor data
#define TEMP_BUFFER_BYTES 512       // 512KB for temporary calculations

// Memory-efficient data types
typedef int8_t q7_t;     // 8-bit quantized values
typedef int16_t q15_t;    // 16-bit quantized values
typedef uint8_t uint8;

// Optimized structure for detection results
typedef struct {
    q7_t x, y, w, h;     // Bounding box (quantized)
    q7_t confidence;      // Confidence score (quantized)
    q7_t class_id;       // Class ID
} TinyDetection;

// Packed sensor data structure
typedef struct {
    uint16_t center_dist : 10;  // 0-1023 cm (10 bits)
    uint16_t left_dist : 10;    // 0-1023 cm
    uint16_t right_dist : 10;   // 0-1023 cm
    uint8_t validity : 3;        // Which readings are valid (3 bits)
    uint8_t safety_level : 2;   // Safety level (2 bits)
} PackedSensorData;

// Memory pool for dynamic allocation
typedef struct {
    uint8_t* start;
    uint32_t size;
    uint32_t used;
} MemoryPool;

// Global memory pools
extern MemoryPool model_pool;
extern MemoryPool temp_pool;
extern MemoryPool sensor_pool;

// Memory management functions
void init_memory_pools();
void* pool_alloc(MemoryPool* pool, uint16_t size);
void pool_reset(MemoryPool* pool);
void print_memory_usage();

// Optimized math operations for quantized data
static inline q15_t q7_mul(q7_t a, q7_t b) {
    return ((q15_t)a * b) >> 7;  // Shift back to maintain scale
}

static inline q7_t q7_add_sat(q7_t a, q7_t b) {
    q15_t sum = (q15_t)a + b;
    if (sum > 127) return 127;
    if (sum < -128) return -128;
    return (q7_t)sum;
}

static inline q7_t q7_relu(q7_t x) {
    return (x > 0) ? x : 0;
}

// Optimized buffer operations
void buffer_copy_8(const uint8_t* src, uint8_t* dst, uint16_t len);
void buffer_copy_16(const uint16_t* src, uint16_t* dst, uint16_t len);
void buffer_clear(uint8_t* buffer, uint16_t len);

// Memory-efficient image processing
void downsample_image(const uint8_t* src, uint16_t src_w, uint16_t src_h,
                     uint8_t* dst, uint16_t dst_w, uint16_t dst_h);

void extract_roi(const uint8_t* image, uint16_t img_w, uint16_t img_h,
                uint16_t x, uint16_t y, uint16_t w, uint16_t h,
                uint8_t* roi);

// Circular buffer for sensor history
typedef struct {
    PackedSensorData* buffer;
    uint8_t size;
    uint8_t head;
    uint8_t count;
} CircularBuffer;

void circ_buffer_init(CircularBuffer* cb, PackedSensorData* buffer, uint8_t size);
void circ_buffer_push(CircularBuffer* cb, PackedSensorData data);
PackedSensorData circ_buffer_get(const CircularBuffer* cb, uint8_t index);
uint8_t circ_buffer_count(const CircularBuffer* cb);

// Fixed-size array for detections
typedef struct {
    TinyDetection detections[5];  // Max 5 detections
    uint8_t count;
} DetectionArray;

void detection_add(DetectionArray* arr, TinyDetection det);
void detection_clear(DetectionArray* arr);
void detection_sort_by_confidence(DetectionArray* arr);

// Bit manipulation utilities
static inline uint8_t bit_count(uint8_t x) {
    uint8_t count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

static inline uint8_t find_first_set(uint8_t x) {
    if (x == 0) return 0xFF;
    uint8_t pos = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        pos++;
    }
    return pos;
}

// Memory-efficient string operations
uint8_t string_length(const char* str);
int string_compare(const char* a, const char* b);
void string_copy(char* dst, const char* src);

// CRC calculation for data integrity
uint8_t crc8(const uint8_t* data, uint16_t len);

// Flash memory utilities for model storage
void flash_read_model(const void* address, void* buffer, uint16_t size);
uint8_t flash_verify_model(const void* address, uint16_t size, uint8_t expected_crc);

#endif // MEMORY_OPT_H