#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Robot - Arduino Sketch for STM32 Coprocessor
Handles motor control and responds to AI detection commands
Compatible with Arduino UNO Q4GB hybrid Linux+Arduino system
*/

#include <Wire.h>
#include <Servo.h>
#include <SoftwareSerial.h>

// Motor control pins
#define ENA 5   // Enable A (Left motor speed)
#define ENB 6   // Enable B (Right motor speed)
#define IN1 7   // Input 1 (Left motor direction)
#define IN2 8   // Input 2 (Left motor direction)
#define IN3 9   // Input 3 (Right motor direction)
#define IN4 10  // Input 4 (Right motor direction)

// Sensor pins
#define ULTRASONIC_TRIG 11
#define ULTRASONIC_ECHO 12
#define FRONT_IR A0
#define LEFT_IR A1
#define RIGHT_IR A2

// LED indicators
#define STATUS_LED 13
#define ERROR_LED 3

// Motor speed limits
#define MIN_SPEED 30
#define MAX_SPEED 255
#define DEFAULT_SPEED 150

// Communication settings
#define BAUD_RATE 115200
#define COMMAND_TIMEOUT 1000  // milliseconds
#define STATUS_UPDATE_INTERVAL 500  // milliseconds

// Command structure
struct MotorCommand {
  String action;
  int left_speed;
  int right_speed;
  unsigned long timestamp;
};

// Global variables
MotorCommand current_command;
String last_command = "";
unsigned long last_command_time = 0;
unsigned long last_status_update = 0;
bool emergency_stop = false;
bool motors_enabled = true;

// Status tracking
struct RobotStatus {
  bool moving;
  bool emergency_stopped;
  int left_speed;
  int right_speed;
  String current_action;
  unsigned long uptime;
  int front_distance;
  int left_distance;
  int right_distance;
  int battery_voltage;
  float temperature;
};

RobotStatus robot_status;

void setup() {
  // Initialize serial communication
  Serial.begin(BAUD_RATE);
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  // Initialize motor control pins
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  
  // Initialize sensor pins
  pinMode(ULTRASONIC_TRIG, OUTPUT);
  pinMode(ULTRASONIC_ECHO, INPUT);
  pinMode(FRONT_IR, INPUT);
  pinMode(LEFT_IR, INPUT);
  pinMode(RIGHT_IR, INPUT);
  
  // Initialize LED pins
  pinMode(STATUS_LED, OUTPUT);
  pinMode(ERROR_LED, OUTPUT);
  
  // Initialize status
  initialize_status();
  
  // Signal ready
  digitalWrite(STATUS_LED, HIGH);
  Serial.println("READY:Arduino UNO Q4GB AI Robot Motor Controller");
  
  // Calibrate motors briefly
  calibrate_motors();
  
  Serial.println("OK:System ready");
}

void loop() {
  // Process incoming commands
  process_commands();
  
  // Update status
  update_status();
  
  // Send periodic status updates
  if (millis() - last_status_update > STATUS_UPDATE_INTERVAL) {
    send_status_update();
    last_status_update = millis();
  }
  
  // Check for command timeout
  if (millis() - last_command_time > COMMAND_TIMEOUT && 
      robot_status.moving && !emergency_stop) {
    Serial.println("WARN:Command timeout - stopping motors");
    stop_motors();
  }
  
  // Small delay for stability
  delay(10);
}

void initialize_status() {
  robot_status.moving = false;
  robot_status.emergency_stopped = false;
  robot_status.left_speed = 0;
  robot_status.right_speed = 0;
  robot_status.current_action = "STOP";
  robot_status.uptime = millis();
  robot_status.front_distance = 0;
  robot_status.left_distance = 0;
  robot_status.right_distance = 0;
  robot_status.battery_voltage = 0;
  robot_status.temperature = 0.0;
}

void process_commands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.length() > 0) {
      last_command = command;
      last_command_time = millis();
      
      Serial.println("RECV:" + command);
      execute_command(command);
    }
  }
}

void execute_command(String command) {
  // Parse command
  MotorCommand cmd = parse_command(command);
  
  if (cmd.action == "") {
    Serial.println("ERROR:Invalid command format");
    return;
  }
  
  // Check emergency stop
  if (cmd.action == "ESTOP") {
    emergency_stop = true;
    stop_motors();
    digitalWrite(ERROR_LED, HIGH);
    Serial.println("OK:Emergency stop activated");
    return;
  }
  
  if (cmd.action == "CLEAR_ESTOP") {
    emergency_stop = false;
    digitalWrite(ERROR_LED, LOW);
    Serial.println("OK:Emergency stop cleared");
    return;
  }
  
  // Don't execute other commands during emergency stop
  if (emergency_stop) {
    Serial.println("ERROR:Emergency stop active");
    return;
  }
  
  // Execute specific commands
  bool success = false;
  
  if (cmd.action == "FORWARD") {
    success = move_forward(cmd.left_speed, cmd.right_speed);
  } else if (cmd.action == "BACKWARD") {
    success = move_backward(cmd.left_speed, cmd.right_speed);
  } else if (cmd.action == "LEFT") {
    success = turn_left(cmd.left_speed, cmd.right_speed);
  } else if (cmd.action == "RIGHT") {
    success = turn_right(cmd.left_speed, cmd.right_speed);
  } else if (cmd.action == "STOP") {
    success = stop_motors();
  } else if (cmd.action == "SPEED") {
    success = set_speed(cmd.left_speed, cmd.right_speed);
  } else if (cmd.action == "STATUS") {
    send_status_update();
    success = true;
  } else if (cmd.action == "CALIBRATE") {
    success = calibrate_motors();
  } else {
    Serial.println("ERROR:Unknown command: " + cmd.action);
    return;
  }
  
  if (success) {
    current_command = cmd;
    robot_status.current_action = cmd.action;
    Serial.println("OK:Command executed");
  } else {
    Serial.println("ERROR:Command execution failed");
  }
}

MotorCommand parse_command(String command) {
  MotorCommand cmd;
  cmd.action = "";
  cmd.left_speed = 0;
  cmd.right_speed = 0;
  cmd.timestamp = millis();
  
  // Split command by colon
  int first_colon = command.indexOf(':');
  if (first_colon == -1) {
    cmd.action = command.toUpperCase();
    return cmd;
  }
  
  cmd.action = command.substring(0, first_colon).toUpperCase();
  String params = command.substring(first_colon + 1);
  
  // Parse speed parameters
  int second_colon = params.indexOf(':');
  if (second_colon == -1) {
    cmd.left_speed = params.toInt();
    cmd.right_speed = cmd.left_speed;
  } else {
    cmd.left_speed = params.substring(0, second_colon).toInt();
    cmd.right_speed = params.substring(second_colon + 1).toInt();
  }
  
  // Validate speed limits
  cmd.left_speed = constrain(cmd.left_speed, 0, MAX_SPEED);
  cmd.right_speed = constrain(cmd.right_speed, 0, MAX_SPEED);
  
  return cmd;
}

bool move_forward(int left_speed, int right_speed) {
  if (!motors_enabled) return false;
  
  // Set motor directions
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  
  // Set motor speeds
  analogWrite(ENA, left_speed);
  analogWrite(ENB, right_speed);
  
  // Update status
  robot_status.moving = true;
  robot_status.left_speed = left_speed;
  robot_status.right_speed = right_speed;
  
  return true;
}

bool move_backward(int left_speed, int right_speed) {
  if (!motors_enabled) return false;
  
  // Set motor directions
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  
  // Set motor speeds
  analogWrite(ENA, left_speed);
  analogWrite(ENB, right_speed);
  
  // Update status
  robot_status.moving = true;
  robot_status.left_speed = left_speed;
  robot_status.right_speed = right_speed;
  
  return true;
}

bool turn_left(int left_speed, int right_speed) {
  if (!motors_enabled) return false;
  
  // Set motor directions for left turn
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  
  // Set motor speeds
  analogWrite(ENA, left_speed);
  analogWrite(ENB, right_speed);
  
  // Update status
  robot_status.moving = true;
  robot_status.left_speed = left_speed;
  robot_status.right_speed = right_speed;
  
  return true;
}

bool turn_right(int left_speed, int right_speed) {
  if (!motors_enabled) return false;
  
  // Set motor directions for right turn
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  
  // Set motor speeds
  analogWrite(ENA, left_speed);
  analogWrite(ENB, right_speed);
  
  // Update status
  robot_status.moving = true;
  robot_status.left_speed = left_speed;
  robot_status.right_speed = right_speed;
  
  return true;
}

bool stop_motors() {
  // Stop motors by setting speed to 0
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  
  // Update status
  robot_status.moving = false;
  robot_status.left_speed = 0;
  robot_status.right_speed = 0;
  robot_status.current_action = "STOP";
  
  return true;
}

bool set_speed(int left_speed, int right_speed) {
  if (!motors_enabled || !robot_status.moving) return false;
  
  // Set new speeds while maintaining current direction
  analogWrite(ENA, left_speed);
  analogWrite(ENB, right_speed);
  
  // Update status
  robot_status.left_speed = left_speed;
  robot_status.right_speed = right_speed;
  
  return true;
}

bool calibrate_motors() {
  Serial.println("INFO:Starting motor calibration");
  
  // Test each motor direction briefly
  for (int speed = 50; speed <= 100; speed += 25) {
    // Forward test
    move_forward(speed, speed);
    delay(200);
    stop_motors();
    delay(100);
    
    // Backward test
    move_backward(speed, speed);
    delay(200);
    stop_motors();
    delay(100);
  }
  
  Serial.println("INFO:Motor calibration complete");
  return true;
}

void update_status() {
  robot_status.uptime = millis();
  robot_status.emergency_stopped = emergency_stop;
  
  // Read sensors
  robot_status.front_distance = read_ultrasonic_distance();
  robot_status.left_distance = read_ir_distance(LEFT_IR);
  robot_status.right_distance = read_ir_distance(RIGHT_IR);
  
  // Simulate battery voltage (would need voltage divider in real implementation)
  robot_status.battery_voltage = map(analogRead(A3), 0, 1023, 0, 120);  // 0-12V
  
  // Simulate temperature (would need temperature sensor in real implementation)
  robot_status.temperature = 25.0 + (random(-50, 50) / 10.0);  // 20-30°C
}

int read_ultrasonic_distance() {
  digitalWrite(ULTRASONIC_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG, LOW);
  
  long duration = pulseIn(ULTRASONIC_ECHO, HIGH, 30000);  // 30ms timeout
  int distance = duration * 0.034 / 2;  // Convert to cm
  
  return constrain(distance, 0, 300);  // Limit to 0-300cm
}

int read_ir_distance(int pin) {
  int value = analogRead(pin);
  // Convert analog reading to distance (calibration needed for specific IR sensor)
  int distance = map(value, 0, 1023, 30, 0);  // Simple linear approximation
  return constrain(distance, 0, 30);
}

void send_status_update() {
  // Create JSON status string
  String status_json = "{";
  status_json += "\"moving\":" + String(robot_status.moving ? "true" : "false") + ",";
  status_json += "\"emergency_stopped\":" + String(robot_status.emergency_stopped ? "true" : "false") + ",";
  status_json += "\"left_speed\":" + String(robot_status.left_speed) + ",";
  status_json += "\"right_speed\":" + String(robot_status.right_speed) + ",";
  status_json += "\"current_action\":\"" + robot_status.current_action + "\",";
  status_json += "\"uptime\":" + String(robot_status.uptime) + ",";
  status_json += "\"front_distance\":" + String(robot_status.front_distance) + ",";
  status_json += "\"left_distance\":" + String(robot_status.left_distance) + ",";
  status_json += "\"right_distance\":" + String(robot_status.right_distance) + ",";
  status_json += "\"battery_voltage\":" + String(robot_status.battery_voltage) + ",";
  status_json += "\"temperature\":" + String(robot_status.temperature, 1);
  status_json += "}";
  
  Serial.println("STATUS:" + status_json);
}