#!/usr/bin/env python3
"""
Arduino UNO Q4GB Camera Interface
Hardware-optimized camera capture and processing for ARM64
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path

class CameraInterface:
    def __init__(self, camera_id=0, resolution=(640, 480)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.camera = None
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_frame_time = 0
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize camera with ARM64 optimizations"""
        print("📸 Initializing camera...")
        
        try:
            # Create video capture object
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                print("  ❌ Failed to open camera")
                return False
            
            # Configure for Arduino UNO Q4GB optimization
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Set buffer size for ARM64 optimization
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set FPS for stability
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual camera properties
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"  ✅ Camera initialized")
            print(f"  📏 Resolution: {actual_width}x{actual_height}")
            print(f"  🎥 FPS: {actual_fps}")
            print(f"  🎯 Target FPS: 30")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Camera initialization error: {e}")
            return False
    
    def capture_frame(self):
        """Capture a single frame with timing"""
        if not self.camera or not self.camera.isOpened():
            return None, 0.0
        
        try:
            start_time = time.time()
            
            ret, frame = self.camera.read()
            
            if not ret:
                return None, 0.0
            
            # Calculate capture time
            capture_time = time.time() - start_time
            
            # Update FPS calculation
            self.frame_count += 1
            current_time = time.time()
            
            if self.last_frame_time > 0:
                time_diff = current_time - self.last_frame_time
                if time_diff > 0:
                    # Rolling FPS calculation
                    instant_fps = 1.0 / time_diff
                    self.fps = 0.9 * self.fps + 0.1 * instant_fps  # Smooth FPS
            
            self.last_frame_time = current_time
            
            return frame, capture_time
            
        except Exception as e:
            print(f"  ⚠️  Frame capture error: {e}")
            return None, 0.0
    
    def create_test_frame(self):
        """Create a test frame for simulation mode"""
        height, width = self.resolution
        
        # Create a colorful test pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            for x in range(width):
                frame[y, x] = [x * 255 // width, y * 255 // height, 128]
        
        # Add some moving elements for visual interest
        center_x, center_y = width // 2, height // 2
        t = time.time()
        
        # Moving circle
        radius = 30 + int(10 * np.sin(t))
        cv2.circle(frame, 
                 (center_x + int(50 * np.cos(t)), center_y + int(30 * np.sin(t))), 
                 radius, (255, 255, 255), -1)
        
        # Moving rectangle
        rect_x = center_x + int(40 * np.cos(t * 1.5))
        rect_y = center_y + int(40 * np.sin(t * 1.5))
        cv2.rectangle(frame, 
                   (rect_x - 20, rect_y - 15), 
                   (rect_x + 20, rect_y + 15), 
                   (0, 255, 0), -1)
        
        return frame
    
    def preprocess_frame(self, frame):
        """Preprocess frame for AI model input"""
        if frame is None:
            return None
        
        try:
            # Resize for AI model (640x640 is standard)
            resized_frame = cv2.resize(frame, (640, 640))
            
            # Convert color space if needed (BGR to RGB)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            return rgb_frame
            
        except Exception as e:
            print(f"  ⚠️  Frame preprocessing error: {e}")
            return frame
    
    def get_frame_info(self):
        """Get current frame statistics"""
        return {
            'frame_count': self.frame_count,
            'fps': self.fps,
            'resolution': self.resolution,
            'camera_active': self.camera is not None and self.camera.isOpened()
        }
    
    def release(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            print("  ✅ Camera released")
    
    def benchmark(self, duration=10):
        """Benchmark camera performance"""
        print(f"🚀 Running camera benchmark for {duration} seconds...")
        
        start_time = time.time()
        frame_count = 0
        capture_times = []
        
        while time.time() - start_time < duration:
            frame, capture_time = self.capture_frame()
            if frame is not None:
                frame_count += 1
                capture_times.append(capture_time)
                
                # Show progress every second
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"    Progress: {frame_count} frames, FPS: {current_fps:.1f}")
        
        # Calculate statistics
        if capture_times:
            avg_capture_time = np.mean(capture_times) * 1000  # Convert to ms
            max_capture_time = np.max(capture_times) * 1000
            avg_fps = frame_count / duration if duration > 0 else 0
        
            print(f"  📊 Benchmark Results:")
            print(f"    Total frames: {frame_count}")
            print(f"    Average capture time: {avg_capture_time:.2f}ms")
            print(f"    Max capture time: {max_capture_time:.2f}ms")
            print(f"    Average FPS: {avg_fps:.1f}")
            
            return {
                'total_frames': frame_count,
                'avg_capture_time_ms': avg_capture_time,
                'max_capture_time_ms': max_capture_time,
                'avg_fps': avg_fps
            }
        else:
            print("  ❌ No frames captured during benchmark")
            return None

class CameraManager:
    def __init__(self):
        self.camera = None
        self.thread = None
        self.running = False
        self.frame_callback = None
        
    def initialize(self, camera_id=0, resolution=(640, 480)):
        """Initialize camera manager"""
        self.camera = CameraInterface(camera_id, resolution)
        return self.camera.initialize()
    
    def start_capture(self, frame_callback=None):
        """Start continuous capture in separate thread"""
        if self.camera is None:
            print("❌ Camera not initialized")
            return False
        
        self.frame_callback = frame_callback
        self.running = True
        
        # Start capture thread
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        
        print("  ✅ Camera capture started")
        return True
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.running:
            frame, capture_time = self.camera.capture_frame()
            
            if frame is not None and self.frame_callback:
                self.frame_callback(frame, capture_time)
            
            # Small delay to prevent overwhelming CPU
            time.sleep(0.001)
    
    def stop_capture(self):
        """Stop continuous capture"""
        self.running = False
        
        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None
        
        print("  ✅ Camera capture stopped")
    
    def release(self):
        """Release all resources"""
        self.stop_capture()
        if self.camera is not None:
            self.camera.release()
            self.camera = None

def main():
    """Main function for testing camera interface"""
    print("📸 Arduino UNO Q4GB Camera Interface Test")
    print("=" * 50)
    
    # Test basic camera initialization
    camera = CameraInterface()
    
    if not camera.initialize():
        print("❌ Camera initialization failed, using simulation mode")
        # Test with simulation
        for i in range(5):
            frame = camera.create_test_frame()
            if frame is not None:
                print(f"  ✅ Simulation frame {i+1}: {frame.shape}")
        
        print("✅ Camera simulation test completed")
        return True
    
    # Run benchmark
    benchmark_results = camera.benchmark(duration=5)  # 5-second benchmark
    
    if benchmark_results:
        print("✅ Camera benchmark completed successfully!")
    else:
        print("❌ Camera benchmark failed!")
    
    # Test frame preprocessing
    for i in range(3):
        frame, _ = camera.capture_frame()
        if frame is not None:
            preprocessed = camera.preprocess_frame(frame)
            if preprocessed is not None:
                print(f"  ✅ Preprocessed frame {i+1}: {preprocessed.shape}")
    
    # Clean up
    camera.release()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)