#!/usr/bin/env python3
"""
Arduino UNO Q4GB Web Interface
Lightweight web interface for robot control and monitoring
"""

import json
import time
import threading
from pathlib import Path
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs
    import socketserver
except ImportError:
    print("⚠️  HTTP server modules not available")

class ArduinoWebHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.robot_controller = kwargs.pop('robot_controller', None)
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_index()
        elif parsed_path.path == '/api/status':
            self.serve_status()
        elif parsed_path.path == '/api/detections':
            self.serve_detections()
        elif parsed_path.path == '/control':
            self.serve_control_page()
        elif parsed_path.path.startswith('/static/'):
            self.serve_static(parsed_path.path[8:])
        else:
            self.send_response(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/move':
            self.handle_move_control()
        elif parsed_path.path == '/api/led':
            self.handle_led_control()
        elif parsed_path.path == '/api/servo':
            self.handle_servo_control()
        else:
            self.send_response(404, "Not Found")
    
    def serve_index(self):
        """Serve main web interface"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Arduino UNO Q4GB AI Robot</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
        .header { background: #2196F3; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .control-group { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        button { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #45a049; }
        button.danger { background: #f44336; }
        button.danger:hover { background: #d32f2f; }
        input { width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        select { width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
        .info { background: #e3f2fd; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .detection { background: #ffeb3b; padding: 10px; border-radius: 4px; margin: 5px 0; border-left: 4px solid #f9a825; }
        h2 { color: #333; margin-top: 0; }
        h3 { color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Arduino UNO Q4GB AI Robot</h1>
        <p>Hardware-specific optimized AI robot control interface</p>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <h3>🎮 Robot Control</h3>
            <button onclick="moveRobot('forward')">⬆️ Forward</button>
            <button onclick="moveRobot('backward')">⬇️ Backward</button>
            <button onclick="moveRobot('left')">⬅️ Left</button>
            <button onclick="moveRobot('right')">➡️ Right</button>
            <button onclick="stopRobot()" class="danger">⛑ Stop</button>
        </div>
        
        <div class="control-group">
            <h3>⚙️ Settings</h3>
            <label>Speed: <input type="range" id="speed" min="0" max="100" value="50" oninput="updateSpeedDisplay()"></label>
            <div>Speed: <span id="speedDisplay">50</span>%</div>
            
            <label>Turn Angle: <input type="range" id="turnAngle" min="-180" max="180" value="45" oninput="updateAngleDisplay()"></label>
            <div>Angle: <span id="angleDisplay">45</span>°</div>
        </div>
    </div>
    
    <div class="status">
        <h3>📊 System Status</h3>
        <div id="statusContent">Loading...</div>
        
        <h3>🤖 AI Detections</h3>
        <div id="detectionContent">No recent detections</div>
    </div>
    
    <script>
        function moveRobot(direction) {
            fetch('/api/move', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: 'direction=' + direction + '&speed=' + document.getElementById('speed').value
            })
            .then(response => response.text())
            .then(data => console.log('Move response:', data));
        }
        
        function stopRobot() {
            fetch('/api/move', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: 'direction=stop'
            })
            .then(response => response.text())
            .then(data => console.log('Stop response:', data));
        }
        
        function updateSpeedDisplay() {
            document.getElementById('speedDisplay').textContent = document.getElementById('speed').value;
        }
        
        function updateAngleDisplay() {
            document.getElementById('angleDisplay').textContent = document.getElementById('turnAngle').value;
        }
        
        // Auto-refresh status
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('statusContent').innerHTML = formatStatus(data);
                });
        }
        
        function updateDetections() {
            fetch('/api/detections')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('detectionContent').innerHTML = formatDetections(data);
                });
        }
        
        function formatStatus(data) {
            if (data.error) {
                return '<div class="info">❌ Error: ' + data.error + '</div>';
            }
            
            let html = '<div class="info">✅ Connected to Arduino</div>';
            html += '<div class="info">🔌 Port: ' + (data.port || 'Unknown') + '</div>';
            html += '<div class="info">📶 Baud: ' + (data.baudrate || 'Unknown') + '</div>';
            html += '<div class="info">⏱️ Uptime: ' + (data.uptime || 'Unknown') + '</div>';
            
            return html;
        }
        
        function formatDetections(data) {
            if (!data.detections || data.detections.length === 0) {
                return '<div class="info">No recent detections</div>';
            }
            
            let html = '';
            data.detections.forEach((detection, index) => {
                html += '<div class="detection">';
                html += '<strong>Detection ' + (index + 1) + ':</strong> ' + detection.class;
                html += '<br>Confidence: ' + (detection.confidence ? detection.confidence.toFixed(2) : 'N/A') + '%';
                html += '</div>';
            });
            
            return html;
        }
        
        // Auto-update status every 2 seconds
        setInterval(updateStatus, 2000);
        setInterval(updateDetections, 1000);
        
        // Initial load
        updateStatus();
        updateDetections();
    </script>
</body>
</html>
        """
        
        self.send_response(200, html_content, 'text/html')
    
    def serve_status(self):
        """Serve system status"""
        if self.robot_controller:
            status = self.robot_controller.get_system_status()
            self.send_json_response(status)
        else:
            self.send_json_response({'error': 'Robot controller not available'})
    
    def serve_detections(self):
        """Serve detection results"""
        if self.robot_controller:
            detections = self.robot_controller.get_recent_detections()
            self.send_json_response(detections)
        else:
            self.send_json_response({'error': 'Robot controller not available', 'detections': []})
    
    def handle_move_control(self):
        """Handle robot movement commands"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        params = parse_qs(post_data)
        direction = params.get('direction', [''])[0]
        speed = params.get('speed', ['50'])[0]
        
        response = {"action": "move", "direction": direction, "speed": speed}
        
        if self.robot_controller:
            try:
                if direction == 'forward':
                    self.robot_controller.move_forward(int(speed))
                elif direction == 'backward':
                    self.robot_controller.move_backward(int(speed))
                elif direction == 'left':
                    self.robot_controller.turn_left(45)
                elif direction == 'right':
                    self.robot_controller.turn_right(45)
                elif direction == 'stop':
                    self.robot_controller.stop_robot()
                
                response['success'] = True
            except Exception as e:
                response['success'] = False
                response['error'] = str(e)
        
        self.send_json_response(response)
    
    def handle_led_control(self):
        """Handle LED control"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        params = parse_qs(post_data)
        state = params.get('state', ['0'])[0]
        
        response = {"action": "led", "state": state}
        
        if self.robot_controller:
            try:
                self.robot_controller.set_led(state == '1')
                response['success'] = True
            except Exception as e:
                response['success'] = False
                response['error'] = str(e)
        
        self.send_json_response(response)
    
    def handle_servo_control(self):
        """Handle servo control"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        params = parse_qs(post_data)
        servo_id = params.get('servo', ['0'])[0]
        angle = params.get('angle', ['90'])[0]
        
        response = {"action": "servo", "servo_id": servo_id, "angle": angle}
        
        if self.robot_controller:
            try:
                self.robot_controller.set_servo(int(servo_id), int(angle))
                response['success'] = True
            except Exception as e:
                response['success'] = False
                response['error'] = str(e)
        
        self.send_json_response(response)
    
    def send_json_response(self, data):
        """Send JSON response"""
        json_data = json.dumps(data, indent=2)
        self.send_response(200, json_data, 'application/json')
    
    def send_response(self, code, content, content_type='text/plain'):
        """Send HTTP response"""
        self.send_response(code, content_type)
        self.end_headers()
        self.wfile.write(content.encode())

class WebInterface:
    def __init__(self, robot_controller=None, port=8080):
        self.robot_controller = robot_controller
        self.port = port
        self.server = None
        self.server_thread = None
    
    def start(self):
        """Start web interface"""
        print(f"🌐 Starting web interface on port {self.port}...")
        
        try:
            handler_class = lambda *args, **kwargs: ArduinoWebHandler(*args, robot_controller=self.robot_controller, **kwargs)
            self.server = HTTPServer(('0.0.0.0', self.port), handler_class)
            
            # Start server in separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            print(f"  ✅ Web interface started")
            print(f"  🌍 Access at: http://localhost:{self.port}")
            print(f"  📱 Control your robot from web browser!")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to start web interface: {e}")
            return False
    
    def stop(self):
        """Stop web interface"""
        if self.server:
            print("🌐 Stopping web interface...")
            self.server.shutdown()
            self.server.server_close()
            print("  ✅ Web interface stopped")

def main():
    """Main function for testing web interface"""
    print("🌐 Arduino UNO Q4GB Web Interface Test")
    print("=" * 50)
    
    # Create mock robot controller for testing
    class MockRobotController:
        def __init__(self):
            self.start_time = time.time()
            self.detections = []
        
        def get_system_status(self):
            return {
                'connected': True,
                'port': 'mock_port',
                'baudrate': 115200,
                'uptime': f"{int(time.time() - self.start_time)}s"
            }
        
        def get_recent_detections(self):
            return self.detections
        
        def move_forward(self, speed):
            print(f"  🎮 Moving forward at speed {speed}")
        
        def move_backward(self, speed):
            print(f"  🎮 Moving backward at speed {speed}")
        
        def turn_left(self, angle):
            print(f"  🎮 Turning left {angle}°")
        
        def turn_right(self, angle):
            print(f"  🎮 Turning right {angle}°")
        
        def stop_robot(self):
            print("  🛑 Stopping robot")
        
        def set_led(self, state):
            print(f"  💡 LED {'ON' if state else 'OFF'}")
        
        def set_servo(self, servo_id, angle):
            print(f"  ⚙️  Servo {servo_id} to {angle}°")
    
    # Create mock controller
    mock_controller = MockRobotController()
    
    # Create and start web interface
    web_interface = WebInterface(robot_controller=mock_controller, port=8080)
    
    if web_interface.start():
        print("✅ Web interface test completed!")
        print("  🌐 Open http://localhost:8080 in your browser")
        print("  ⏱️  Press Ctrl+C to stop")
        
        try:
            # Keep server running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⚡ Web interface stopped by user")
        
        web_interface.stop()
        return True
    else:
        print("❌ Failed to start web interface!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)