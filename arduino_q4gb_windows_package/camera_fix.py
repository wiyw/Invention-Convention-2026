#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Robot - Windows Camera Fix
Comprehensive camera initialization with fallback options
"""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import time

class CameraTest:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("📷 Camera Fix & Test")
        self.root.geometry("600x400")
        
        self.cap = None
        self.camera_index = 0
        self.test_results = {}
        
        self.create_gui()
        self.test_all_cameras()
    
    def create_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Test results area
        results_frame = ttk.LabelFrame(main_frame, text="🔍 Camera Test Results", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="🔄 Retest Cameras", command=self.test_all_cameras).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🎥 Test Selected", command=self.test_selected_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📷 Test with Different Methods", command=self.test_advanced_methods).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Close", command=self.root.quit).pack(side=tk.RIGHT, padx=5)
        
        # Camera selection
        select_frame = ttk.Frame(main_frame)
        select_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(select_frame, text="Select Camera:").pack(side=tk.LEFT, padx=5)
        
        self.camera_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(select_frame, textvariable=self.camera_var, 
                                      values=[str(i) for i in range(10)], width=10)
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        
        # Backend selection
        ttk.Label(select_frame, text="Backend:").pack(side=tk.LEFT, padx=5)
        self.backend_var = tk.StringVar(value="DShow")
        backend_combo = ttk.Combobox(select_frame, textvariable=self.backend_var,
                                  values=["DShow", "Media Foundation", "V4L2"], width=15)
        backend_combo.pack(side=tk.LEFT, padx=5)
    
    def log_message(self, message):
        """Add message to results window"""
        self.results_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def test_all_cameras(self):
        """Test all available cameras"""
        self.results_text.delete(1.0, tk.END)
        self.log_message("🔍 Starting comprehensive camera test...")
        
        # Test standard OpenCV backends
        backends = [
            (cv2.CAP_DSHOW, "DShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_FFMPEG, "FFMPEG"),
            (cv2.CAP_IMAGES, "Images")
        ]
        
        working_cameras = []
        
        for backend_idx, (backend, name) in enumerate(backends):
            self.log_message(f"\n📡 Testing {name} backend...")
            
            backend_working = False
            for camera_idx in range(10):
                try:
                    cap = cv2.VideoCapture(camera_idx + backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            h, w = frame.shape[:2]
                            self.log_message(f"  ✅ Camera {camera_idx} works! ({w}x{h})")
                            working_cameras.append((camera_idx, name, backend))
                            backend_working = True
                        else:
                            self.log_message(f"  ⚠️  Camera {camera_idx}: Opens but no frame")
                        cap.release()
                    else:
                        cap.release()
                except Exception as e:
                    self.log_message(f"  ❌ Camera {camera_idx}: Error - {str(e)[:50]}")
            
            if not backend_working:
                self.log_message(f"  ❌ {name} backend: No working cameras")
        
        # Summary
        self.log_message(f"\n📊 SUMMARY: Found {len(working_cameras)} working camera configurations")
        
        if working_cameras:
            self.log_message("\n🎯 Recommended configurations:")
            for cam_idx, backend_name, backend in working_cameras[:5]:
                self.log_message(f"  • Camera {cam_idx} with {backend_name}")
            
            # Update combo box with working cameras
            camera_indices = list(set([str(cam[0]) for cam in working_cameras]))
            self.camera_combo['values'] = camera_indices
            if camera_indices:
                self.camera_combo.set(camera_indices[0])
        else:
            self.log_message("\n❌ NO WORKING CAMERAS FOUND")
            self.log_message("Troubleshooting tips:")
            self.log_message("  • Close Zoom, Teams, Skype, Discord")
            self.log_message("  • Check Windows Privacy → Camera settings")
            self.log_message("  • Update camera drivers")
            self.log_message("  • Try different USB ports")
            self.log_message("  • Restart computer")
    
    def test_selected_camera(self):
        """Test the selected camera configuration"""
        try:
            camera_idx = int(self.camera_var.get())
            backend_name = self.backend_var.get()
            
            # Get backend constant
            backend_map = {
                "DShow": cv2.CAP_DSHOW,
                "Media Foundation": cv2.CAP_MSMF,
                "V4L2": cv2.CAP_V4L2,
                "FFMPEG": cv2.CAP_FFMPEG,
                "Images": cv2.CAP_IMAGES
            }
            backend = backend_map.get(backend_name, cv2.CAP_DSHOW)
            
            self.log_message(f"\n🎥 Testing Camera {camera_idx} with {backend_name}...")
            
            cap = cv2.VideoCapture(camera_idx + backend)
            if not cap.isOpened():
                self.log_message("  ❌ Failed to open camera")
                return
            
            self.log_message("  ✅ Camera opened successfully")
            
            # Test reading multiple frames
            frames_read = 0
            start_time = time.time()
            
            for i in range(30):  # Test 30 frames
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames_read += 1
                    if frames_read == 1:
                        h, w, c = frame.shape
                        self.log_message(f"  ✅ First frame: {w}x{h}, {c} channels")
                time.sleep(0.033)  # ~30 FPS
            
            elapsed = time.time() - start_time
            fps = frames_read / elapsed if elapsed > 0 else 0
            
            self.log_message(f"  📊 Read {frames_read}/30 frames in {elapsed:.1f}s ({fps:.1f} FPS)")
            
            if frames_read > 25:
                self.log_message("  ✅ EXCELLENT: Camera working perfectly!")
            elif frames_read > 15:
                self.log_message("  ✅ GOOD: Camera working well")
            elif frames_read > 5:
                self.log_message("  ⚠️  FAIR: Camera working but slow")
            else:
                self.log_message("  ❌ POOR: Camera not working properly")
            
            cap.release()
            
            # Test if this config works with main app
            self.log_message("\n🔧 Testing with main robot application...")
            success = self.test_with_robot_app(camera_idx, backend)
            if success:
                self.log_message("  ✅ This configuration works with the robot app!")
            else:
                self.log_message("  ❌ Issues detected with robot app")
        
        except Exception as e:
            self.log_message(f"❌ Test failed: {e}")
    
    def test_advanced_methods(self):
        """Test advanced camera initialization methods"""
        self.log_message("\n🔬 Testing advanced camera methods...")
        
        methods = [
            # Method 1: Standard with buffer reduction
            lambda idx: self.test_method("Standard + Buffer", idx, lambda: cv2.VideoCapture(cv2.CAP_DSHOW + idx)),
            
            # Method 2: MSMF with specific settings
            lambda idx: self.test_method("MSMF + Settings", idx, lambda: cv2.VideoCapture(cv2.CAP_MSMF + idx)),
            
            # Method 3: DirectShow with properties
            lambda idx: self.test_method("DirectShow + Properties", idx, lambda: cv2.VideoCapture(cv2.CAP_DSHOW + idx)),
            
            # Method 4: Auto-detection
            lambda idx: self.test_method("Auto-detect", idx, lambda: cv2.VideoCapture(idx)),
        ]
        
        working_methods = []
        
        for method in methods:
            for idx in range(3):  # Test first 3 cameras
                try:
                    if method(idx):
                        working_methods.append((method.__name__, idx))
                        break
                except:
                    pass
        
        self.log_message(f"\n🎯 Found {len(working_methods)} working methods")
    
    def test_method(self, name, idx, initializer):
        """Test a specific camera initialization method"""
        try:
            cap = initializer()
            if cap.isOpened():
                # Configure camera
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.log_message(f"  ✅ {name}: Camera {idx} works ({frame.shape[1]}x{frame.shape[0]})")
                    cap.release()
                    return True
                cap.release()
        except Exception as e:
            pass
        return False
    
    def test_with_robot_app(self, camera_idx, backend):
        """Test if configuration works with main robot app"""
        try:
            # Create a minimal version of the robot's camera setup
            cap = cv2.VideoCapture(camera_idx + backend)
            if not cap.isOpened():
                return False
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test reading for 3 seconds
            start_time = time.time()
            frames_read = 0
            
            while time.time() - start_time < 3:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames_read += 1
                    # Simulate some processing (like edge detection)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                time.sleep(0.1)
            
            cap.release()
            return frames_read > 10
            
        except Exception as e:
            return False
    
    def run(self):
        """Run the camera test application"""
        self.root.mainloop()

def main():
    print("📷 Arduino UNO Q4GB - Camera Fix & Test Tool")
    print("=" * 50)
    print("This tool will help diagnose and fix camera issues")
    print("and find the best configuration for your system.")
    print("=" * 50)
    
    app = CameraTest()
    app.run()

if __name__ == "__main__":
    main()