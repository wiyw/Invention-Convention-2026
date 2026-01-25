// Memory Optimization Implementation for Arduino UNO Q4GB
// Implements memory-efficient operations for TinyML models

#include "memory_opt.h"
#include <string.h>
// Note: Arduino.h should be included in the main sketch using this header

// Global memory pool definitions
static uint8_t model_heap[512 * 1024];  // 512KB model pool
static uint8_t temp_heap[TEMP_BUFFER_BYTES];
static uint8_t sensor_heap[SENSOR_BUFFER_BYTES];

MemoryPool model_pool = {model_heap, 524288, 0};
MemoryPool temp_pool = {temp_heap, TEMP_BUFFER_BYTES, 0};
MemoryPool sensor_pool = {sensor_heap, SENSOR_BUFFER_BYTES, 0};

void init_memory_pools() {
    model_pool.used = 0;
    temp_pool.used = 0;
    sensor_pool.used = 0;
    
    // Remove Serial prints for header-only compatibility
    // Actual prints would be in main sketch
}

void* pool_alloc(MemoryPool* pool, uint16_t size) {
    // Align to 4-byte boundary
    size = (size + 3) & ~3;
    
    if (pool->used + size > pool->size) {
        return NULL;  // Out of memory
    }
    
    void* ptr = pool->start + pool->used;
    pool->used += size;
    return ptr;
}

void pool_reset(MemoryPool* pool) {
    pool->used = 0;
}

void print_memory_usage() {
    // Print statements removed for header compatibility
    // Use this function in your main sketch with Serial.print() calls
}

// Optimized buffer operations
void buffer_copy_8(const uint8_t* src, uint8_t* dst, uint16_t len) {
    // Use word-sized operations when possible
    uint16_t i = 0;
    
    // Copy 4 bytes at a time
    while (i + 4 <= len) {
        *((uint32_t*)(dst + i)) = *((const uint32_t*)(src + i));
        i += 4;
    }
    
    // Copy remaining bytes
    while (i < len) {
        dst[i] = src[i];
        i++;
    }
}

void buffer_copy_16(const uint16_t* src, uint16_t* dst, uint16_t len) {
    uint16_t i = 0;
    
    while (i + 2 <= len) {
        *((uint32_t*)(dst + i)) = *((const uint32_t*)(src + i));
        i += 2;
    }
    
    while (i < len) {
        dst[i] = src[i];
        i++;
    }
}

void buffer_clear(uint8_t* buffer, uint16_t len) {
    uint16_t i = 0;
    
    // Clear 4 bytes at a time
    while (i + 4 <= len) {
        *((uint32_t*)(buffer + i)) = 0;
        i += 4;
    }
    
    // Clear remaining bytes
    while (i < len) {
        buffer[i] = 0;
        i++;
    }
}

// Memory-efficient image downsampling
void downsample_image(const uint8_t* src, uint16_t src_w, uint16_t src_h,
                     uint8_t* dst, uint16_t dst_w, uint16_t dst_h) {
    
    uint16_t x_ratio = (src_w << 8) / dst_w;
    uint16_t y_ratio = (src_h << 8) / dst_h;
    
    for (uint16_t y = 0; y < dst_h; y++) {
        uint16_t src_y = (y * y_ratio) >> 8;
        
        for (uint16_t x = 0; x < dst_w; x++) {
            uint16_t src_x = (x * x_ratio) >> 8;
            
            uint32_t sum = 0;
            uint16_t count = 0;
            
            // Simple box filter for better quality
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int sx = src_x + dx;
                    int sy = src_y + dy;
                    
                    if (sx >= 0 && sx < src_w && sy >= 0 && sy < src_h) {
                        sum += src[sy * src_w + sx];
                        count++;
                    }
                }
            }
            
            dst[y * dst_w + x] = count > 0 ? (sum / count) : 0;
        }
    }
}

void extract_roi(const uint8_t* image, uint16_t img_w, uint16_t img_h,
                uint16_t x, uint16_t y, uint16_t w, uint16_t h,
                uint8_t* roi) {
    
    // Clamp coordinates to image bounds
    if (x >= img_w || y >= img_h) return;
    if (x + w > img_w) w = img_w - x;
    if (y + h > img_h) h = img_h - y;
    
    for (uint16_t dy = 0; dy < h; dy++) {
        uint16_t src_y = y + dy;
        for (uint16_t dx = 0; dx < w; dx++) {
            uint16_t src_x = x + dx;
            roi[dy * w + dx] = image[src_y * img_w + src_x];
        }
    }
}

// Circular buffer implementation
void circ_buffer_init(CircularBuffer* cb, PackedSensorData* buffer, uint8_t size) {
    cb->buffer = buffer;
    cb->size = size;
    cb->head = 0;
    cb->count = 0;
}

void circ_buffer_push(CircularBuffer* cb, PackedSensorData data) {
    cb->buffer[cb->head] = data;
    cb->head = (cb->head + 1) % cb->size;
    
    if (cb->count < cb->size) {
        cb->count++;
    }
}

PackedSensorData circ_buffer_get(const CircularBuffer* cb, uint8_t index) {
    if (index >= cb->count) {
        PackedSensorData empty = {0};
        return empty;
    }
    
    uint8_t pos = (cb->head - cb->count + index + cb->size) % cb->size;
    return cb->buffer[pos];
}

uint8_t circ_buffer_count(const CircularBuffer* cb) {
    return cb->count;
}

// Detection array operations
void detection_add(DetectionArray* arr, TinyDetection det) {
    if (arr->count < 5) {
        arr->detections[arr->count] = det;
        arr->count++;
    }
}

void detection_clear(DetectionArray* arr) {
    arr->count = 0;
}

void detection_sort_by_confidence(DetectionArray* arr) {
    // Simple bubble sort for small array
    for (uint8_t i = 0; i < arr->count - 1; i++) {
        for (uint8_t j = 0; j < arr->count - i - 1; j++) {
            if (arr->detections[j].confidence < arr->detections[j + 1].confidence) {
                // Swap
                TinyDetection temp = arr->detections[j];
                arr->detections[j] = arr->detections[j + 1];
                arr->detections[j + 1] = temp;
            }
        }
    }
}

// Memory-efficient string operations
uint8_t string_length(const char* str) {
    uint8_t len = 0;
    while (str[len] != '\0') len++;
    return len;
}

int string_compare(const char* a, const char* b) {
    while (*a && *b && *a == *b) {
        a++;
        b++;
    }
    return *a - *b;
}

void string_copy(char* dst, const char* src) {
    while ((*dst++ = *src++) != '\0');
}

// CRC8 calculation for data integrity
uint8_t crc8(const uint8_t* data, uint16_t len) {
    uint8_t crc = 0;
    
    for (uint16_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t bit = 0; bit < 8; bit++) {
            if (crc & 0x80) {
                crc = (crc << 1) ^ 0x07;  // CRC8 polynomial
            } else {
                crc <<= 1;
            }
        }
    }
    
    return crc;
}

// Flash memory utilities (simplified)
void flash_read_model(const void* address, void* buffer, uint16_t size) {
    // This would interface with actual flash memory
    // For now, just copy from program memory
    const uint8_t* src = (const uint8_t*)address;
    uint8_t* dst = (uint8_t*)buffer;
    
    for (uint16_t i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

uint8_t flash_verify_model(const void* address, uint16_t size, uint8_t expected_crc) {
    // Read model from flash and verify CRC
    uint8_t* buffer = (uint8_t*)pool_alloc(&temp_pool, size);
    if (!buffer) return 0;  // Memory allocation failed
    
    flash_read_model(address, buffer, size);
    uint8_t actual_crc = crc8(buffer, size);
    
    return (actual_crc == expected_crc) ? 1 : 0;
}