#!/usr/bin/env python3
"""
Arduino App Labs Ultrasonic Trash Collector
3-SENSOR VERSION: Front, Left 45°, Right 45°
"""

import time
import random
import math

class ThreeSensorTrashCollector:
    def __init__(self):
        self.running = False
        self.frame_count = 0
        
        # EXACT 3 SENSOR CONFIGURATION
        self.sensors = {
            'front': {'threshold': 30, 'angle': 0, 'name': 'FRONT'},
            'left_45': {'threshold': 25, 'angle': -45, 'name': 'LEFT 45°'},  
            'right_45': {'threshold': 25, 'angle': 45, 'name': 'RIGHT 45°'}
        }
        
        self.sensor_data = {key: 100.0 for key in self.sensors.keys()}
        self.objects_detected = []
        
        print("=== Arduino App Labs 3-Sensor Robot ===")
        print("Configuration: 3 Ultrasonic Sensors")
        print("Sensors: Front, Left 45°, Right 45°")
        print("Detection: Real-time Object Detection")
        print("Arduino: UNO R4 Motor Control")
    
    def update_sensor_data(self):
        """Generate realistic sensor data for 3 sensors"""
        import random
        
        for sensor_name in self.sensors.keys():
            threshold = self.sensors[sensor_name]['threshold']
            
            # 40% chance of detecting object
            if random.random() < 0.4:
                # Object detected within threshold
                self.sensor_data[sensor_name] = random.uniform(5, threshold - 5)
            else:
                # Clear path
                self.sensor_data[sensor_name] = random.uniform(40, 150)
    
    def detect_objects(self):
        """Detect objects from 3 sensor readings"""
        self.objects_detected = []
        
        for sensor_name, distance in self.sensor_data.items():
            threshold = self.sensors[sensor_name]['threshold']
            
            if distance < threshold:
                # Object detected
                self.objects_detected.append({
                    'sensor': sensor_name,
                    'name': self.sensors[sensor_name]['name'],
                    'distance': distance,
                    'angle': self.sensors[sensor_name]['angle'],
                    'confidence': 0.9 if distance < 15 else 0.8
                })
    
    def create_visualization(self):
        """Create ASCII visualization for 3 sensors"""
        width, height = 60, 20
        grid = [['.' for _ in range(width)] for _ in range(height)]
        
        # Place robot at center
        cx, cy = width // 2, height // 2
        grid[cy][cx] = 'R'
        
        # Draw 3 sensor rays
        for sensor_name, sensor_info in self.sensors.items():
            angle = sensor_info['angle']
            distance = self.sensor_data[sensor_name]
            threshold = sensor_info['threshold']
            
            # Convert angle to coordinates
            angle_rad = math.radians(angle)
            # Ensure max_dist is an int and at least 1
            max_dist = max(1, int(min(distance, threshold) / 3))
            
            for step in range(1, max_dist + 1):
                x = int(cx + step * math.cos(angle_rad))
                y = int(cy + step * math.sin(angle_rad))
                
                if 0 <= x < width and 0 <= y < height:
                    if distance < threshold and step == max_dist - 1:
                        grid[y][x] = 'X'  # Object detected
                    else:
                        grid[y][x] = '-'  # Clear path
            
            # Draw sensor label
            label = self.sensors[sensor_name]['name'][:3]
            if angle < 0:
                grid[max(0, cy-1)][cx-8:cx-8+len(label)] = list(label)
            elif angle > 0:
                grid[max(0, cy-1)][cx+2:cx+2+len(label)] = list(label)
        
        return grid
    
    def print_visualization(self, grid):
        """Print ASCII visualization"""
        print("\n" + "=" * 62)
        print("3-SENSOR ULTRASONIC VISUALIZATION")
        print("=" * 62)
        for row in grid:
            print(''.join(row))
        print("=" * 62)
        
        # Print sensor data
        print(f"Frame: {self.frame_count} | Objects: {len(self.objects_detected)}")
        for sensor_name, distance in self.sensor_data.items():
            threshold = self.sensors[sensor_name]['threshold']
            status = "DETECT" if distance < threshold else "CLEAR"
            sensor_display = self.sensors[sensor_name]['name']
            print(f"{sensor_display:8s}: {distance:5.1f}cm [{status}]")
    
    def generate_arduino_command(self):
        """Generate Arduino command for detected objects"""
        if self.objects_detected:
            # Find nearest object
            nearest = min(self.objects_detected, key=lambda x: x['distance'])
            command = f"MOVE:{nearest['sensor']}:{nearest['distance']:.1f}:{nearest['angle']}"
            print(f"Arduino Command: {command}")
            return command
        else:
            print("Arduino Command: STOP")
            return "STOP"
    
    def run(self):
        """Main robot loop"""
        print("Starting 3-sensor robot loop...")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running:
                # Update sensors
                self.update_sensor_data()
                
                # Detect objects
                self.detect_objects()
                
                # Create visualization
                grid = self.create_visualization()
                
                # Clear screen and print
                print("\033[2J\033[H", end="")  # Clear screen
                self.print_visualization(grid)
                
                # Generate Arduino command
                command = self.generate_arduino_command()
                
                # Update frame counter
                self.frame_count += 1
                
                # Small delay
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nRobot stopped by user")
        
        finally:
            # Final statistics
            elapsed_time = time.time() - start_time
            print(f"\n=== FINAL STATISTICS ===")
            print(f"Total Frames: {self.frame_count}")
            print(f"Runtime: {elapsed_time:.1f} seconds")
            if self.frame_count > 0:
                print(f"Average FPS: {self.frame_count / elapsed_time:.1f}")
            print(f"Total Detections: {len(self.objects_detected)}")
            print("=========================")

def main():
    """Main entry point"""
    robot = ThreeSensorTrashCollector()
    
    try:
        robot.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
