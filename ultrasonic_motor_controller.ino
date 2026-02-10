/*
  Arduino UNO R4 Ultrasonic Trash Collector Controller
  Receives commands from AI Robot via serial communication
  Commands: ULTRASONIC:count:type:distance:angle
  
  Ultrasonic Sensor Connections (if using real sensors):
  - Front Sensor: Pin 2 (Trigger), Pin 8 (Echo)
  - Front Left Sensor: Pin 3 (Trigger), Pin 9 (Echo)
  - Front Right Sensor: Pin 4 (Trigger), Pin 10 (Echo)
  - Left Sensor: Pin 5 (Trigger), Pin 11 (Echo)
  - Right Sensor: Pin 6 (Trigger), Pin 12 (Echo)
  
  Motor Connections:
  - Left Motor: Pin 3 (PWM), Pin 4 (Direction)
  - Right Motor: Pin 5 (PWM), Pin 6 (Direction)
  - Collection Servo: Pin 9
  - LED Indicator: Pin 13
  
  Created for Invention Convention 2026
*/

// Motor pins
const int LEFT_MOTOR_PWM = 3;
const int LEFT_MOTOR_DIR = 4;
const int RIGHT_MOTOR_PWM = 5;
const int RIGHT_MOTOR_DIR = 6;

// Servo and LED
const int COLLECTION_SERVO = 9;
const int LED_PIN = 13;

// Ultrasonic sensor pins (if using real sensors)
const int ULTRASONIC_TRIG[5] = {2, 3, 4, 5, 6};
const int ULTRASONIC_ECHO[5] = {8, 9, 10, 11, 12};

// Robot parameters
const int MOTOR_SPEED = 180;  // 0-255
const int TURN_SPEED = 120;
const int COLLECTION_ANGLE = 90;
const int REST_ANGLE = 0;
const int COLLISION_THRESHOLD = 15; // cm

// Variables
bool collecting = false;
bool obstacle_detected = false;
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 3000; // 3 seconds

void setup() {
  Serial.begin(115200);
  
  // Initialize motor pins
  pinMode(LEFT_MOTOR_PWM, OUTPUT);
  pinMode(LEFT_MOTOR_DIR, OUTPUT);
  pinMode(RIGHT_MOTOR_PWM, OUTPUT);
  pinMode(RIGHT_MOTOR_DIR, OUTPUT);
  
  // Initialize servo and LED
  pinMode(COLLECTION_SERVO, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  
  // Initialize ultrasonic sensors (if using real hardware)
  for (int i = 0; i < 5; i++) {
    pinMode(ULTRASONIC_TRIG[i], OUTPUT);
    pinMode(ULTRASONIC_ECHO[i], INPUT);
  }
  
  // Initial state
  stopMotors();
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("Ultrasonic Trash Collector Motor Controller Ready");
  Serial.println("Waiting for ultrasonic AI commands...");
  Serial.println("Format: ULTRASONIC:count:type:distance:angle");
}

void loop() {
  checkSerialCommands();
  
  // Safety: Stop if no commands for timeout period
  if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
    stopMotors();
    collecting = false;
    obstacle_detected = false;
  }
  
  delay(50); // Small delay for stability
}

void checkSerialCommands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    Serial.print("Received: ");
    Serial.println(command);
    
    // Parse command: ULTRASONIC:count:type:distance:angle
    if (command.startsWith("ULTRASONIC:")) {
      parseUltrasonicCommand(command);
    }
  }
}

void parseUltrasonicCommand(String command) {
  // Parse: ULTRASONIC:count:type:distance:angle
  int firstColon = command.indexOf(':');
  int secondColon = command.indexOf(':', firstColon + 1);
  int thirdColon = command.indexOf(':', secondColon + 1);
  int fourthColon = command.indexOf(':', thirdColon + 1);
  
  if (firstColon > 0 && secondColon > 0 && thirdColon > 0 && fourthColon > 0) {
    int count = command.substring(firstColon + 1, secondColon).toInt();
    String type = command.substring(secondColon + 1, thirdColon);
    float distance = command.substring(thirdColon + 1, fourthColon).toFloat();
    int angle = command.substring(fourthColon + 1).toInt();
    
    Serial.print("Parsed - Count: ");
    Serial.print(count);
    Serial.print(", Type: ");
    Serial.print(type);
    Serial.print(", Distance: ");
    Serial.print(distance);
    Serial.print("cm, Angle: ");
    Serial.print(angle);
    Serial.println("°");
    
    if (count > 0 && !type.equals("NONE") && distance < 50.0) {
      moveToUltrasonicTarget(distance, angle, type);
      startCollection();
    } else {
      stopMotors();
      collecting = false;
      digitalWrite(LED_PIN, LOW);
      obstacle_detected = false;
    }
    
    lastCommandTime = millis();
  }
}

void moveToUltrasonicTarget(float distance, int angle, String type) {
  Serial.print("Moving to ultrasonic target: ");
  Serial.print(type);
  Serial.print(" at ");
  Serial.print(distance);
  Serial.print("cm, ");
  Serial.print(angle);
  Serial.println("°");
  
  digitalWrite(LED_PIN, HIGH);
  obstacle_detected = true;
  
  // Navigation based on angle
  if (angle >= -30 && angle <= 30) {
    // Front area - move forward
    Serial.println("Moving forward to front object");
    moveForward(800); // Move based on distance
  }
  else if (angle < -30) {
    // Left side
    Serial.println("Turning left to reach object");
    turnLeft(400);
    moveForward(600);
  }
  else if (angle > 30) {
    // Right side
    Serial.println("Turning right to reach object");
    turnRight(400);
    moveForward(600);
  }
  
  // Adjust movement based on object type and distance
  if (type.equals("small_trash")) {
    // Small object - precise approach
    Serial.println("Small object detected - precise approach");
    moveForward(300);
  }
  else if (type.equals("large_trash")) {
    // Large object - slower approach
    Serial.println("Large object detected - cautious approach");
    moveForward(400);
    delay(200); // Brief pause for safety
  }
}

void startCollection() {
  Serial.println("Starting ultrasonic-guided collection sequence");
  collecting = true;
  
  // Move collection servo
  analogWrite(COLLECTION_SERVO, COLLECTION_ANGLE);
  delay(1000);
  
  // Small backing up to clear object
  moveBackward(200);
  
  // Reset servo
  analogWrite(COLLECTION_SERVO, REST_ANGLE);
  delay(500);
  
  Serial.println("Collection complete");
  collecting = false;
  digitalWrite(LED_PIN, LOW);
}

// Motor control functions
void moveForward(int duration) {
  Serial.println("Moving forward");
  digitalWrite(LEFT_MOTOR_DIR, HIGH);
  digitalWrite(RIGHT_MOTOR_DIR, HIGH);
  analogWrite(LEFT_MOTOR_PWM, MOTOR_SPEED);
  analogWrite(RIGHT_MOTOR_PWM, MOTOR_SPEED);
  delay(duration);
  stopMotors();
}

void moveBackward(int duration) {
  Serial.println("Moving backward");
  digitalWrite(LEFT_MOTOR_DIR, LOW);
  digitalWrite(RIGHT_MOTOR_DIR, LOW);
  analogWrite(LEFT_MOTOR_PWM, MOTOR_SPEED);
  analogWrite(RIGHT_MOTOR_PWM, MOTOR_SPEED);
  delay(duration);
  stopMotors();
}

void turnLeft(int duration) {
  Serial.println("Turning left");
  digitalWrite(LEFT_MOTOR_DIR, LOW);
  digitalWrite(RIGHT_MOTOR_DIR, HIGH);
  analogWrite(LEFT_MOTOR_PWM, TURN_SPEED);
  analogWrite(RIGHT_MOTOR_PWM, TURN_SPEED);
  delay(duration);
  stopMotors();
}

void turnRight(int duration) {
  Serial.println("Turning right");
  digitalWrite(LEFT_MOTOR_DIR, HIGH);
  digitalWrite(RIGHT_MOTOR_DIR, LOW);
  analogWrite(LEFT_MOTOR_PWM, TURN_SPEED);
  analogWrite(RIGHT_MOTOR_PWM, TURN_SPEED);
  delay(duration);
  stopMotors();
}

void stopMotors() {
  analogWrite(LEFT_MOTOR_PWM, 0);
  analogWrite(RIGHT_MOTOR_PWM, 0);
  digitalWrite(LEFT_MOTOR_DIR, LOW);
  digitalWrite(RIGHT_MOTOR_DIR, LOW);
}

// Real ultrasonic sensor reading (if using actual hardware)
float readUltrasonicSensor(int sensorIndex) {
  if (sensorIndex < 0 || sensorIndex >= 5) return 999.0;
  
  // Send pulse
  digitalWrite(ULTRASONIC_TRIG[sensorIndex], LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG[sensorIndex], HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG[sensorIndex], LOW);
  
  // Read echo
  long duration = pulseIn(ULTRASONIC_ECHO[sensorIndex], HIGH);
  
  // Calculate distance (cm)
  float distance = duration * 0.034 / 2;
  
  return distance;
}

// Status functions
void sendUltrasonicStatus() {
  Serial.print("ULTRASONIC_STATUS:collecting=");
  Serial.print(collecting ? "true" : "false");
  Serial.print(":obstacle=");
  Serial.print(obstacle_detected ? "true" : "false");
  Serial.print(":uptime=");
  Serial.print(millis());
  Serial.println();
}

void testAllUltrasonicSensors() {
  Serial.println("Testing all ultrasonic sensors:");
  
  for (int i = 0; i < 5; i++) {
    float distance = readUltrasonicSensor(i);
    
    String sensorNames[] = {"Front", "Front Left", "Front Right", "Left", "Right"};
    Serial.print("  ");
    Serial.print(sensorNames[i]);
    Serial.print(": ");
    Serial.print(distance);
    Serial.println(" cm");
    
    delay(100); // Small delay between readings
  }
  
  Serial.println("Sensor test complete");
}