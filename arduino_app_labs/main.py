#!/usr/bin/env python3
"""
Arduino App Labs Ultrasonic Trash Collector
Main Python script - must be in root directory
"""

import os
import sys
import json
import time
import numpy as np

class ArduinoAppLabsRobot:
    def __init__(self):
        self.running = False
        self.frame_count = 0
        self.detection_count = 0
        
        # Arduino App Labs configuration
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Ultrasonic sensor configuration
        self.sensors = {
            'front': {'threshold': 30, 'angle': 0, 'color': (0, 255, 0)},
            'front_left': {'threshold': 25, 'angle': -30, 'color': (255, 255, 0)},
            'front_right': {'threshold': 25, 'angle': 30, 'color': (0, 255, 255)},
            'left': {'threshold': 20, 'angle': -60, 'color': (255, 0, 255)},
            'right': {'threshold': 20, 'angle': 60, 'color': (255, 0, 0)}
        }
        
        self.sensor_data = {key: 100.0 for key in self.sensors.keys()}
        
        print("=== Arduino App Labs Ultrasonic Robot ===")
        print("Version: 1.0.0")
        print("Sensors: 5x Ultrasonic")
        print("Detection: Real-time")
        print("Arduino: UNO R4")
    
    def generate_sensor_data(self):
        """Generate realistic ultrasonic sensor data"""
        import random
        
        # Simulate room with objects
        for sensor_name in self.sensors.keys():
            # 70% chance of clear path, 30% chance of object
            if random.random() < 0.7:
                self.sensor_data[sensor_name] = random.uniform(50, 200)
            else:
                threshold = self.sensors[sensor_name]['threshold']
                self.sensor_data[sensor_name] = random.uniform(5, threshold - 5)
        
        return self.sensor_data
    
    def detect_objects(self):
        """Detect objects from sensor data"""
        objects = []
        
        for sensor_name, distance in self.sensor_data.items():
            threshold = self.sensors[sensor_name]['threshold']
            
            if distance < threshold:
                objects.append({
                    'type': 'trash',
                    'sensor': sensor_name,
                    'distance': distance,
                    'angle': self.sensors[sensor_name]['angle']
                })
        
        return objects
    
    def create_visualization(self):
        """Create ASCII visualization for Arduino App Labs"""
        width, height = 80, 24
        grid = [['.' for _ in range(width)] for _ in range(height)]
        
        # Place robot at center
        cx, cy = width // 2, height // 2
        grid[cy][cx] = 'R'
        
        # Draw sensor rays
        for sensor_name, sensor_info in self.sensors.items():
            angle = sensor_info['angle']
            distance = self.sensor_data[sensor_name]
            threshold = sensor_info['threshold']
            color = sensor_info['color']
            
            # Convert angle to grid coordinates
            angle_rad = np.radians(angle)
            max_dist = min(distance, threshold) // 4
            
            for step in range(1, max_dist):
                x = int(cx + step * np.cos(angle_rad))
                y = int(cy + step * np.sin(angle_rad))
                
                if 0 <= x < width and 0 <= y < height:
                    if distance < threshold and step == max_dist - 1:
                        grid[y][x] = 'X'  # Object detected
                    else:
                        grid[y][x] = '-'  # Sensor ray
        
        return grid
    
    def print_visualization(self, grid):
        """Print ASCII visualization"""
        print("\n" + "=" * 82)
        print("ULTRASONIC SENSOR VISUALIZATION")
        print("=" * 82)
        for row in grid:
            print(''.join(row))
        print("=" * 82)
        
        # Print sensor data
        print(f"Frame: {self.frame_count} | Objects: {self.detection_count}")
        for sensor_name, distance in self.sensor_data.items():
            threshold = self.sensors[sensor_name]['threshold']
            status = "DETECT" if distance < threshold else "CLEAR"
            print(f"{sensor_name:10s}: {distance:5.1f}cm [{status}]")
    
    def communicate_with_arduino(self, objects):
        """Simulate Arduino communication"""
        if objects:
            nearest = min(objects, key=lambda x: x['distance'])
            print(f"Arduino Command: MOVE_TO:{nearest['sensor']}@{nearest['distance']:.1f}cm")
        else:
            print("Arduino Command: STOP")
    
    def run(self):
        """Main robot loop"""
        print("Starting ultrasonic robot loop...")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running:
                # Generate sensor data
                self.generate_sensor_data()
                
                # Detect objects
                objects = self.detect_objects()
                self.detection_count = len(objects)
                
                # Create visualization
                grid = self.create_visualization()
                
                # Clear screen and print visualization
                os.system('clear' if os.name == 'posix' else 'cls')
                self.print_visualization(grid)
                
                # Communicate with Arduino
                self.communicate_with_arduino(objects)
                
                # Update counters
                self.frame_count += 1
                
                # Small delay
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nRobot stopped by user")
        
        finally:
            # Print final statistics
            elapsed_time = time.time() - start_time
            print(f"\nFinal Statistics:")
            print(f"Total Frames: {self.frame_count}")
            print(f"Runtime: {elapsed_time:.1f} seconds")
            print(f"Average FPS: {self.frame_count / elapsed_time:.1f}")
            print(f"Total Detections: {self.detection_count}")

def main():
    """Main entry point"""
    robot = ArduinoAppLabsRobot()
    
    try:
        robot.run()
    except Exception as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()