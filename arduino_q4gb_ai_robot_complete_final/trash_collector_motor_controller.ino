/*
  Arduino UNO R4 Trash Collector Motor Controller
  Receives commands from AI Robot via serial communication
  Commands: TRASH:count:type:x:y
  
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

// Robot parameters
const int MOTOR_SPEED = 180;  // 0-255
const int TURN_SPEED = 120;
const int COLLECTION_ANGLE = 90;
const int REST_ANGLE = 0;

// Variables
bool collecting = false;
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 5000; // 5 seconds

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
  
  // Initial state
  stopMotors();
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("Trash Collector Motor Controller Ready");
  Serial.println("Waiting for AI commands...");
}

void loop() {
  checkSerialCommands();
  
  // Safety: Stop if no commands for timeout period
  if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
    stopMotors();
    collecting = false;
  }
  
  delay(50); // Small delay for stability
}

void checkSerialCommands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    Serial.print("Received: ");
    Serial.println(command);
    
    // Parse command: TRASH:count:type:x:y
    if (command.startsWith("TRASH:")) {
      parseTrashCommand(command);
    }
  }
}

void parseTrashCommand(String command) {
  // Parse: TRASH:count:type:x:y
  int firstColon = command.indexOf(':');
  int secondColon = command.indexOf(':', firstColon + 1);
  int thirdColon = command.indexOf(':', secondColon + 1);
  int fourthColon = command.indexOf(':', thirdColon + 1);
  
  if (firstColon > 0 && secondColon > 0 && thirdColon > 0 && fourthColon > 0) {
    int count = command.substring(firstColon + 1, secondColon).toInt();
    String type = command.substring(secondColon + 1, thirdColon);
    int x = command.substring(thirdColon + 1, fourthColon).toInt();
    int y = command.substring(fourthColon + 1).toInt();
    
    Serial.print("Parsed - Count: ");
    Serial.print(count);
    Serial.print(", Type: ");
    Serial.print(type);
    Serial.print(", X: ");
    Serial.print(x);
    Serial.print(", Y: ");
    Serial.println(y);
    
    if (count > 0 && !type.equals("NONE")) {
      moveToTrash(x, y);
      startCollection();
    } else {
      stopMotors();
      collecting = false;
      digitalWrite(LED_PIN, LOW);
    }
    
    lastCommandTime = millis();
  }
}

void moveToTrash(int x, int y) {
  // Camera resolution is 640x480
  // Center is at (320, 240)
  
  Serial.println("Moving to trash...");
  digitalWrite(LED_PIN, HIGH);
  
  // Calculate direction based on X position
  if (x < 200) {
    // Trash is on the left
    Serial.println("Turning left");
    turnLeft(500);
  } else if (x > 440) {
    // Trash is on the right
    Serial.println("Turning right");
    turnRight(500);
  } else {
    // Trash is roughly centered, move forward
    Serial.println("Moving forward");
    moveForward(1000);
  }
  
  // Consider Y position for distance
  if (y > 300) {
    // Object is far, move forward longer
    Serial.println("Object far, moving forward more");
    moveForward(500);
  }
}

void startCollection() {
  Serial.println("Starting collection sequence");
  collecting = true;
  
  // Move collection servo
  analogWrite(COLLECTION_SERVO, COLLECTION_ANGLE);
  delay(1000);
  
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

// Status functions
void sendStatus() {
  Serial.print("STATUS:collecting=");
  Serial.print(collecting ? "true" : "false");
  Serial.print(":uptime=");
  Serial.print(millis());
  Serial.println();
}