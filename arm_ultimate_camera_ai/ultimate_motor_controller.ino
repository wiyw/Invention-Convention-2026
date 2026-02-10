/*
  Arduino UNO Q4GB Ultimate Motor Controller
  Ultimate version with ultrasonic sensors and intelligent navigation
*/

#include <Servo.h>

// Ultrasonic sensor pins
#define ULTRASONIC_TRIG_PIN   11
#define ULTRASONIC_ECHO_PIN   12
#define LEFT_ULTRASONIC_TRIG   9
#define LEFT_ULTRASONIC_ECHO   10
#define RIGHT_ULTRASONIC_TRIG  13
#define RIGHT_ULTRASONIC_ECHO  8

// Motor control pins
#define ENA_PIN  5
#define ENB_PIN  6
#define IN1_PIN  7
#define IN2_PIN  8
#define IN3_PIN  9
#define IN4_PIN  10

// LED indicators
#define STATUS_LED_PIN  13
#define ERROR_LED_PIN  3

// Robot state
struct RobotState {
  bool emergency_stopped;
  String current_action;
  int left_speed;
  int right_speed;
  unsigned long last_sensor_read;
  int front_distance;
  int left_distance;
  int right_distance;
};

RobotState robot;

void setup() {
  Serial.begin(115200);
  
  // Initialize pins
  pinMode(ULTRASONIC_TRIG_PIN, OUTPUT);
  pinMode(ULTRASONIC_ECHO_PIN, INPUT);
  pinMode(LEFT_ULTRASONIC_TRIG, OUTPUT);
  pinMode(LEFT_ULTRASONIC_ECHO, INPUT);
  pinMode(RIGHT_ULTRASONIC_TRIG, OUTPUT);
  pinMode(RIGHT_ULTRASONIC_ECHO, INPUT);
  
  // Motor pins
  pinMode(ENA_PIN, OUTPUT);
  pinMode(ENB_PIN, OUTPUT);
  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);
  pinMode(IN3_PIN, OUTPUT);
  pinMode(IN4_PIN, OUTPUT);
  
  // LED pins
  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(ERROR_LED_PIN, OUTPUT);
  
  // Initialize robot state
  robot.emergency_stopped = false;
  robot.current_action = "STOP";
  robot.left_speed = 0;
  robot.right_speed = 0;
  robot.last_sensor_read = 0;
  robot.front_distance = 999;
  robot.left_distance = 999;
  robot.right_distance = 999;
  
  // Stop motors initially
  stop_motors();
  
  // Signal ready
  digitalWrite(STATUS_LED_PIN, HIGH);
  digitalWrite(ERROR_LED_PIN, LOW);
  
  Serial.println("READY:Arduino UNO Q4GB Ultimate Motor Controller");
  Serial.println("OK:Ultimate system initialized");
}

void loop() {
  // Check for incoming commands
  if (Serial.available() > 0) {
    process_command();
  }
  
  // Update sensors periodically
  if (millis() - robot.last_sensor_read > 500) {  // Read every 500ms
    read_ultrasonic_sensors();
    robot.last_sensor_read = millis();
  }
  
  // Small delay for stability
  delay(10);
}

void process_command() {
  String command = Serial.readStringUntil('\n');
  command.trim();
  
  if (command.length() == 0) return;
  
  Serial.print("RECV:");
  Serial.println(command);
  
  // Parse and execute command
  if (command.startsWith("FORWARD:")) {
    int colon_index = command.indexOf(':');
    int comma_index = command.indexOf(':', colon_index + 1);
    
    if (colon_index > 0 && comma_index > colon_index) {
      String left_speed_str = command.substring(colon_index + 1, comma_index);
      String right_speed_str = command.substring(comma_index + 1);
      
      int left_speed = left_speed_str.toInt();
      int right_speed = right_speed_str.toInt();
      
      forward_motors(left_speed, right_speed);
    }
  } else if (command == "STOP") {
    stop_motors();
  } else if (command == "ESTOP") {
    emergency_stop();
  } else if (command == "GET_SENSORS")) {
    send_sensor_readings();
  } else if (command == "CENTER") {
    center_robot_45_degrees();
  } else if (command.startsWith("TURN_LEFT")) {
    int colon_index = command.indexOf(':');
    int comma_index = command.indexOf(':', colon_index + 1);
    
    if (colon_index > 0 && comma_index > colon_index) {
      String left_speed_str = command.substring(colon_index + 1, comma_index);
      String right_speed_str = command.substring(comma_index + 1);
      
      int left_speed = left_speed_str.toInt();
      int right_speed = right_speed_str.toInt();
      
      turn_left(left_speed, right_speed);
    }
  } else if (command.startsWith("TURN_RIGHT")) {
    int colon_index = command.indexOf(':');
    int comma_index = command.indexOf(':', colon_index + 1);
    
    if (colon_index > 0 && comma_index > colon_index) {
      String left_speed_str = command.substring(colon_index + 1, comma_index);
      String right_speed_str = command.substring(comma_index + 1);
      
      int left_speed = left_speed_str.toInt();
      int right_speed = right_speed_str.toInt();
      
      turn_right(left_speed, right_speed);
    }
  } else {
    Serial.print("ERROR:Unknown command:");
    Serial.println(command);
  }
}

void read_ultrasonic_sensors() {
  // Read front sensor
  robot.front_distance = read_ultrasonic_distance(ULTRASONIC_TRIG_PIN, ULTRASONIC_ECHO_PIN);
  
  // Read left sensor
  robot.left_distance = read_ultrasonic_distance(LEFT_ULTRASONIC_TRIG, LEFT_ULTRASONIC_ECHO);
  
  // Read right sensor
  robot.right_distance = read_ultrasonic_distance(RIGHT_ULTRASONIC_TRIG, RIGHT_ULTRASONIC_ECHO);
}

int read_ultrasonic_distance(int trig_pin, int echo_pin) {
  // Send pulse
  digitalWrite(trig_pin, LOW);
  delayMicroseconds(2);
  digitalWrite(trig_pin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig_pin, LOW);
  
  // Read echo
  long duration = pulseIn(echo_pin, HIGH, 30000);  // 30ms timeout
  
  // Calculate distance (speed of sound = 343 m/s, divide by 2 for round trip)
  int distance = duration * 0.034 / 2;
  
  return constrain(distance, 0, 500);  // Limit to 0-5m range
}

void send_sensor_readings() {
  Serial.print("SENSORS:");
  Serial.print(robot.front_distance);
  Serial.print(",");
  Serial.print(robot.left_distance);
  Serial.print(",");
  Serial.println(robot.right_distance);
}

void forward_motors(int left_speed, int right_speed) {
  if (robot.emergency_stopped) return;
  
  // Set motor direction for forward
  digitalWrite(IN1_PIN, HIGH);
  digitalWrite(IN2_PIN, LOW);
  digitalWrite(IN3_PIN, HIGH);
  digitalWrite(IN4_PIN, LOW);
  
  // Set motor speeds
  analogWrite(ENA_PIN, left_speed);
  analogWrite(ENB_PIN, right_speed);
  
  robot.current_action = "FORWARD";
  robot.left_speed = left_speed;
  robot.right_speed = right_speed;
  
  digitalWrite(STATUS_LED_PIN, HIGH);
  digitalWrite(ERROR_LED_PIN, LOW);
  
  Serial.print("OK:FORWARD:");
  Serial.print(left_speed);
  Serial.print(",");
  Serial.println(right_speed);
}

void turn_left(int forward_speed, int turn_speed) {
  if (robot.emergency_stopped) return;
  
  // Set motor direction for left turn (left motor slower)
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, HIGH);
  digitalWrite(IN3_PIN, HIGH);
  digitalWrite(IN4_PIN, LOW);
  
  // Set motor speeds (right motor faster)
  analogWrite(ENA_PIN, turn_speed);
  analogWrite(ENB_PIN, forward_speed);
  
  robot.current_action = "LEFT";
  robot.left_speed = turn_speed;
  robot.right_speed = forward_speed;
  
  Serial.print("OK:LEFT:");
  Serial.print(turn_speed);
  Serial.print(",");
  Serial.println(forward_speed);
}

void turn_right(int forward_speed, int turn_speed) {
  if (robot.emergency_stopped) return;
  
  // Set motor direction for right turn (left motor faster)
  digitalWrite(IN1_PIN, HIGH);
  digitalWrite(IN2_PIN, LOW);
  digitalWrite(IN3_PIN, LOW);
  digitalWrite(IN4_PIN, HIGH);
  
  // Set motor speeds (left motor faster)
  analogWrite(ENA_PIN, forward_speed);
  analogWrite(ENB_PIN, turn_speed);
  
  robot.current_action = "RIGHT";
  robot.left_speed = forward_speed;
  robot.right_speed = turn_speed;
  
  Serial.print("OK:RIGHT:");
  Serial.print(forward_speed);
  Serial.print(",");
  Serial.println(turn_speed);
}

void stop_motors() {
  // Stop motors
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);
  digitalWrite(IN3_PIN, LOW);
  digitalWrite(IN4_PIN, LOW);
  
  analogWrite(ENA_PIN, 0);
  analogWrite(ENB_PIN, 0);
  
  robot.current_action = "STOP";
  robot.left_speed = 0;
  robot.right_speed = 0;
  
  digitalWrite(STATUS_LED_PIN, LOW);
  digitalWrite(ERROR_LED_PIN, LOW);
  
  Serial.println("OK:STOP");
}

void emergency_stop() {
  // Emergency stop - all motors off
  stop_motors();
  robot.emergency_stopped = true;
  
  // Flash error LED
  for (int i = 0; i < 10; i++) {
    digitalWrite(ERROR_LED_PIN, HIGH);
    delay(100);
    digitalWrite(ERROR_LED_PIN, LOW);
    delay(100);
  }
  
  digitalWrite(ERROR_LED_PIN, HIGH);
  
  Serial.println("OK:ESTOP");
}

void center_robot_45_degrees() {
  // Center robot and turn 45 degrees for optimal sensor positioning
  
  // Stop first
  stop_motors();
  delay(500);
  
  // Turn left 45 degrees (approximate timing)
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, HIGH);
  digitalWrite(IN3_PIN, HIGH);
  digitalWrite(IN4_PIN, LOW);
  
  analogWrite(ENA_PIN, 80);
  analogWrite(ENB_PIN, 100);
  
  // Calculate approximate 45-degree turn time
  int turn_duration = 1500;  // Adjust based on your robot
  unsigned long start_time = millis();
  
  while (millis() - start_time < turn_duration && !robot.emergency_stopped) {
    // Keep turning
    if (Serial.available() > 0) {
      String cmd = Serial.readStringUntil('\n');
      if (cmd.startsWith("ESTOP")) {
        emergency_stop();
        break;
      }
    }
  }
  
  // Stop
  stop_motors();
  delay(500);
  
  Serial.println("OK:CENTERED");
}