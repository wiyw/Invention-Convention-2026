#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Robot - Windows Webcam Version
Real webcam object detection + simulation of Arduino communication
Works on Windows without physical Arduino hardware
"""

import cv2
import numpy as np
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from PIL import Image, ImageTk
import random
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class WindowsAIBot:
    def __init__(self, camera_index=0):
        self.running = False
        self.camera_index = camera_index
        self.cap = None
        
        # Simulated sensor data (since no physical Arduino)
        self.sensor_data = {
            'left_distance': 100.0,
            'right_distance': 100.0,
            'center_distance': 100.0,
            'timestamp': time.time()
        }
        
        # AI detection data
        self.detected_objects = []
        self.detection_confidence = 0.0
        
        # Motor state (simulated)
        self.motor_state = {'left_speed': 0, 'right_speed': 0, 'active': False}
        
        # AI decisions
        self.ai_decisions = []
        
        # Initialize webcam
        self.init_webcam()
        
        # Initialize AI models
        self.init_ai_models()
        
        # Create GUI
        self.create_gui()
        
    def init_webcam(self):
        """Initialize webcam with better error handling"""
        try:
            print(f"🔍 Initializing webcam...")
            
            # Try different backend APIs for better Windows compatibility
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
            
            for backend in backends:
                print(f"  Trying backend: {backend}")
                self.cap = cv2.VideoCapture(self.camera_index + backend)
                
                if self.cap.isOpened():
                    # Test if we can actually read from camera
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"✅ Webcam initialized on camera {self.camera_index} with backend {backend}")
                        # Set camera properties for better performance
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                        return True
                    else:
                        self.cap.release()
            
            # Try different camera indices
            print("🔍 Trying different camera indices...")
            for i in range(10):
                for backend in backends:
                    self.cap = cv2.VideoCapture(i + backend)
                    if self.cap.isOpened():
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            self.camera_index = i
                            print(f"✅ Found working camera {i} with backend {backend}")
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            return True
                        else:
                            self.cap.release()
            
            print("❌ No working webcam found")
            self.cap = None
            return False
                
        except Exception as e:
            print(f"❌ Webcam initialization error: {e}")
            self.cap = None
            return False
    
    def init_ai_models(self):
        """Initialize AI models for object detection"""
        self.ai_models_loaded = False
        
        try:
            if TORCH_AVAILABLE:
                # Try to load a simple object detection model
                # For Windows demo, we'll use template matching as fallback
                self.ai_models_loaded = True
                print("✅ AI detection initialized (simulation mode)")
            else:
                print("⚠️  PyTorch not available - using simulation mode")
                
        except Exception as e:
            print(f"⚠️  AI models not available: {e}")
    
    def simulate_object_detection(self, frame):
        """Simulate AI object detection on webcam frame"""
        # Simple motion detection and object simulation
        objects = ['person', 'car', 'bicycle', 'dog', 'cat', 'chair', 'bottle', 'phone']
        detected = []
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple edge detection to find "objects"
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Simulate detecting objects based on contours
        num_objects = min(len(contours), 3)
        if num_objects > 0:
            for i in range(min(num_objects, 3)):
                if contours[i].shape[0] > 50:  # Minimum size
                    x, y, w, h = cv2.boundingRect(contours[i])
                    confidence = random.uniform(0.6, 0.95)
                    obj_class = random.choice(objects)
                    
                    detected.append({
                        'class': obj_class,
                        'confidence': confidence,
                        'bbox': [x, y, x+w, y+h],
                        'timestamp': time.time()
                    })
        
        self.detected_objects = detected
        self.detection_confidence = max([obj['confidence'] for obj in detected], default=0.0)
        
        return detected
    
    def simulate_sensor_readings(self):
        """Simulate ultrasonic sensor readings"""
        while self.running:
            # Simulate realistic sensor data with some variation
            self.sensor_data = {
                'left_distance': max(10, min(200, 
                    self.sensor_data['left_distance'] + random.uniform(-8, 8))),
                'right_distance': max(10, min(200, 
                    self.sensor_data['right_distance'] + random.uniform(-8, 8))),
                'center_distance': max(10, min(200, 
                    self.sensor_data['center_distance'] + random.uniform(-5, 5))),
                'timestamp': time.time()
            }
            time.sleep(0.1)  # 10Hz sensor update
    
    def ai_navigation_logic(self):
        """AI-based navigation decision making"""
        while self.running:
            center_dist = self.sensor_data['center_distance']
            left_dist = self.sensor_data['left_distance']
            right_dist = self.sensor_data['right_distance']
            
            # Check for detected objects
            has_person = any(obj['class'] == 'person' for obj in self.detected_objects)
            has_vehicle = any(obj['class'] in ['car', 'bicycle'] for obj in self.detected_objects)
            
            # AI Decision logic
            decision = "forward"
            speed = 150
            
            if center_dist < 30:
                decision = "stop"
                speed = 0
            elif center_dist < 60:
                if left_dist > right_dist:
                    decision = "left"
                else:
                    decision = "right"
                speed = 120
            elif has_vehicle:
                decision = "cautious_forward"
                speed = 100
            elif has_person and center_dist < 100:
                decision = "slow_forward"
                speed = 90
            elif self.detection_confidence > 0.8:
                decision = "investigate"
                speed = 80
            
            # Execute decision
            if decision == "forward":
                self.set_motors(speed, speed)
            elif decision == "slow_forward":
                self.set_motors(speed, speed)
            elif decision == "cautious_forward":
                self.set_motors(speed, speed)
            elif decision == "investigate":
                self.set_motors(speed//2, speed)
            elif decision == "left":
                self.set_motors(speed//2, speed)
            elif decision == "right":
                self.set_motors(speed, speed//2)
            elif decision == "stop":
                self.set_motors(0, 0)
            
            self.ai_decisions.append({
                'decision': decision,
                'speed': speed,
                'reasoning': f"Center: {center_dist:.1f}cm, Objects: {len(self.detected_objects)}, Confidence: {self.detection_confidence:.2f}",
                'timestamp': time.time()
            })
            
            # Keep only last 20 decisions
            if len(self.ai_decisions) > 20:
                self.ai_decisions.pop(0)
            
            time.sleep(0.3)  # ~3Hz AI decisions
    
    def set_motors(self, left_speed, right_speed):
        """Simulate motor control"""
        self.motor_state = {
            'left_speed': left_speed,
            'right_speed': right_speed,
            'active': left_speed > 0 or right_speed > 0
        }
    
    def create_gui(self):
        """Create Windows GUI interface"""
        self.root = tk.Tk()
        self.root.title("Arduino UNO Q4GB AI Robot - Windows Webcam Version")
        self.root.geometry("1200x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera frame
        camera_frame = ttk.LabelFrame(main_frame, text="📷 Webcam Feed", padding="5")
        camera_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.camera_label = tk.Label(camera_frame, text="📷 Initializing camera...\n\nClick 'Camera Test' if issues occur", width=80, height=24, bg="black", fg="white")
        self.camera_label.pack()
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="📊 Status", padding="5")
        status_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Sensor status
        sensor_frame = ttk.LabelFrame(status_frame, text="📡 Sensors (cm)", padding="5")
        sensor_frame.pack(fill=tk.X, pady=5)
        
        self.left_sensor_label = ttk.Label(sensor_frame, text="Left: --", font=("Arial", 12))
        self.left_sensor_label.pack(anchor=tk.W)
        
        self.center_sensor_label = ttk.Label(sensor_frame, text="Center: --", font=("Arial", 12))
        self.center_sensor_label.pack(anchor=tk.W)
        
        self.right_sensor_label = ttk.Label(sensor_frame, text="Right: --", font=("Arial", 12))
        self.right_sensor_label.pack(anchor=tk.W)
        
        # Motor status
        motor_frame = ttk.LabelFrame(status_frame, text="⚙️ Motors", padding="5")
        motor_frame.pack(fill=tk.X, pady=5)
        
        self.left_motor_label = ttk.Label(motor_frame, text="Left: 0", font=("Arial", 12))
        self.left_motor_label.pack(anchor=tk.W)
        
        self.right_motor_label = ttk.Label(motor_frame, text="Right: 0", font=("Arial", 12))
        self.right_motor_label.pack(anchor=tk.W)
        
        self.motor_status_label = ttk.Label(motor_frame, text="Status: Stopped", font=("Arial", 12, "bold"))
        self.motor_status_label.pack(anchor=tk.W)
        
        # Detection status
        detection_frame = ttk.LabelFrame(status_frame, text="👁️ Object Detection", padding="5")
        detection_frame.pack(fill=tk.X, pady=5)
        
        self.object_count_label = ttk.Label(detection_frame, text="Objects: 0", font=("Arial", 12))
        self.object_count_label.pack(anchor=tk.W)
        
        self.confidence_label = ttk.Label(detection_frame, text="Max Confidence: 0.00", font=("Arial", 12))
        self.confidence_label.pack(anchor=tk.W)
        
        self.object_list_text = tk.Text(detection_frame, height=6, width=30)
        self.object_list_text.pack(fill=tk.BOTH, expand=True)
        
        # AI decisions
        ai_frame = ttk.LabelFrame(status_frame, text="🧠 AI Decisions", padding="5")
        ai_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.decisions_text = tk.Text(ai_frame, height=8, width=30)
        self.decisions_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="🚀 Start", command=self.start_robot)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="🛑 Stop", command=self.stop_robot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.camera_button = ttk.Button(control_frame, text="📷 Switch Camera", command=self.switch_camera)
        self.camera_button.pack(side=tk.LEFT, padx=5)
        
        self.diagnostic_button = ttk.Button(control_frame, text="🔍 Camera Test", command=self.run_camera_diagnostic)
        self.diagnostic_button.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def update_gui(self):
        """Update GUI with current status"""
        if not self.running:
            return
        
        try:
            # Update camera feed
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Process frame for AI detection
                    self.simulate_object_detection(frame)
                    
                    # Draw bounding boxes on frame
                    for obj in self.detected_objects:
                        x1, y1, x2, y2 = obj['bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{obj['class']} {obj['confidence']:.2f}", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Convert to PIL Image and display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.camera_label.configure(image=photo)
                    self.camera_label.image = photo  # Keep reference
            
            # Update sensor labels
            self.left_sensor_label.configure(text=f"Left: {self.sensor_data['left_distance']:.1f} cm")
            self.center_sensor_label.configure(text=f"Center: {self.sensor_data['center_distance']:.1f} cm")
            self.right_sensor_label.configure(text=f"Right: {self.sensor_data['right_distance']:.1f} cm")
            
            # Update motor labels
            self.left_motor_label.configure(text=f"Left: {self.motor_state['left_speed']}")
            self.right_motor_label.configure(text=f"Right: {self.motor_state['right_speed']}")
            self.motor_status_label.configure(text=f"Status: {'Active' if self.motor_state['active'] else 'Stopped'}")
            
            # Update detection labels
            self.object_count_label.configure(text=f"Objects: {len(self.detected_objects)}")
            self.confidence_label.configure(text=f"Max Confidence: {self.detection_confidence:.2f}")
            
            # Update object list
            self.object_list_text.delete(1.0, tk.END)
            for obj in self.detected_objects:
                self.object_list_text.insert(tk.END, f"{obj['class']}: {obj['confidence']:.2f}\n")
            
            # Update AI decisions
            self.decisions_text.delete(1.0, tk.END)
            for i, decision in enumerate(self.ai_decisions[-5:]):  # Last 5 decisions
                self.decisions_text.insert(tk.END, f"{i+1}. {decision['decision']}\n")
                self.decisions_text.insert(tk.END, f"   {decision['reasoning']}\n\n")
        
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # Schedule next update
        self.root.after(100, self.update_gui)  # Update every 100ms
    
    def start_robot(self):
        """Start the AI robot"""
        print("🚀 Starting Windows AI Robot...")
        
        try:
            self.running = True
            
            # Start background threads
            threading.Thread(target=self.simulate_sensor_readings, daemon=True).start()
            threading.Thread(target=self.ai_navigation_logic, daemon=True).start()
            
            # Update GUI
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            
            # Start GUI updates
            self.update_gui()
            
            print("✅ AI Robot started successfully")
            messagebox.showinfo("Status", "AI Robot Started!\n\nWebcam feed is active with AI detection.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start robot: {e}")
            self.stop_robot()
    
    def stop_robot(self):
        """Stop the AI robot"""
        print("🛑 Stopping AI Robot...")
        
        self.running = False
        self.set_motors(0, 0)
        
        # Update GUI
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        
        print("✅ AI Robot stopped")
    
    def switch_camera(self):
        """Switch to next camera"""
        print(f"🔄 Switching camera from {self.camera_index}")
        
        # Release current camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Try next camera
        self.camera_index = (self.camera_index + 1) % 10  # Try cameras 0-9
        
        if self.init_webcam():
            print(f"✅ Successfully switched to camera {self.camera_index}")
        else:
            print(f"❌ Failed to initialize camera {self.camera_index}")
            # Try again
            self.camera_index = (self.camera_index + 1) % 10
            self.init_webcam()
    
    def run_camera_diagnostic(self):
        """Run camera diagnostic tool"""
        print("🔍 Running camera diagnostic...")
        
        # Create diagnostic window
        diag_window = tk.Toplevel(self.root)
        diag_window.title("📷 Camera Diagnostic")
        diag_window.geometry("400x300")
        
        text_widget = tk.Text(diag_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        def run_diag():
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, "🔍 Scanning for cameras...\n\n")
            
            # Scan cameras
            import cv2
            found_cameras = []
            
            for i in range(10):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            found_cameras.append(i)
                            text_widget.insert(tk.END, f"✅ Camera {i}: Working ({frame.shape[1]}x{frame.shape[0]})\n")
                        else:
                            text_widget.insert(tk.END, f"⚠️  Camera {i}: Available but no frame\n")
                        cap.release()
                    else:
                        text_widget.insert(tk.END, f"❌ Camera {i}: Not available\n")
                except Exception as e:
                    text_widget.insert(tk.END, f"❌ Camera {i}: Error - {e}\n")
            
            text_widget.insert(tk.END, f"\n📊 Found {len(found_cameras)} working cameras\n")
            
            if found_cameras:
                text_widget.insert(tk.END, "\n🔧 Try switching to these camera indices\n")
                for cam_idx in found_cameras:
                    text_widget.insert(tk.END, f"   • Camera {cam_idx}\n")
            else:
                text_widget.insert(tk.END, "\n❌ No working cameras found!\n")
                text_widget.insert(tk.END, "\nTroubleshooting:\n")
                text_widget.insert(tk.END, "• Close other camera apps (Zoom, Teams)\n")
                text_widget.insert(tk.END, "• Check Windows Privacy settings\n")
                text_widget.insert(tk.END, "• Update camera drivers\n")
                text_widget.insert(tk.END, "• Try different USB ports\n")
        
        run_diag()
        
        close_button = ttk.Button(diag_window, text="Close", command=diag_window.destroy)
        close_button.pack(pady=5)

    def on_closing(self):
        """Handle window closing"""
        print("👋 Shutting down...")
        self.stop_robot()
        
        if self.cap:
            self.cap.release()
        
        self.root.destroy()
    
    def run(self):
        """Run the GUI application"""
        print("🖥️  Starting Windows AI Robot GUI...")
        self.root.mainloop()

def main():
    """Main function"""
    print("🤖 Arduino UNO Q4GB AI Robot - Windows Webcam Version")
    print("=" * 60)
    print("Features:")
    print("  📷 Real webcam integration")
    print("  👁️ AI object detection simulation")
    print("  🤖 Navigation decision making")
    print("  📊 Windows GUI interface")
    print("=" * 60)
    
    # Check dependencies
    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError:
        print("❌ OpenCV not found. Please install: pip install opencv-python")
        return
    
    try:
        import tkinter
        print("✅ Tkinter available")
    except ImportError:
        print("❌ Tkinter not found (should come with Python)")
        return
    
    # Create and run robot
    robot = WindowsAIBot()
    
    try:
        robot.run()
    except KeyboardInterrupt:
        print("\n🛑 User interrupt")
    except Exception as e:
        print(f"❌ Error: {e}")
        messagebox.showerror("Fatal Error", f"Application error: {e}")

if __name__ == "__main__":
    main()