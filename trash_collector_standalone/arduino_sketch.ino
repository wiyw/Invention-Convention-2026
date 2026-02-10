/*
 * Trash Collector Robot - Arduino Standalone Sketch
 * Direct serial communication with Python
 * No App Labs bridge dependency
 */

#include <Servo.h>

// Pin definitions
#define SERVO_LEFT_PIN   9
#define SERVO_RIGHT_PIN  10
#define ULTRASOUND_FRONT_TRIG_PIN  2
#define ULTRASOUND_FRONT_ECHO_PIN  3
#define ULTRASOUND_LEFT_TRIG_PIN   4
#define ULTRASOUND_LEFT_ECHO_PIN   5
#define ULTRASOUND_RIGHT_TRIG_PIN  6
#define ULTRASOUND_RIGHT_ECHO_PIN  7

// Servo objects
Servo servoLeft;
Servo servoRight;

// Global variables
int servoLeftAngle = 90;
int servoRightAngle = 90;
unsigned long lastSensorUpdate = 0;
const unsigned long SENSOR_UPDATE_INTERVAL = 50;

// Sensor readings
int ultrasoundFront = 0;
int ultrasoundLeft = 0;
int ultrasoundRight = 0;

// Function prototypes
void updateServos();
void readUltrasoundSensors();
int readUltrasound(int trigPin, int echoPin);
void processCommand(String command);
void sendResponse(String response);

void setup() {
  Serial.begin(115200);
  Serial.println("Trash Collector Robot - Arduino Standalone");
  
  // Initialize servos
  servoLeft.attach(SERVO_LEFT_PIN);
  servoRight.attach(SERVO_RIGHT_PIN);
  servoLeft.write(90);
  servoRight.write(90);
  
  // Initialize ultrasound pins
  pinMode(ULTRASOUND_FRONT_TRIG_PIN, OUTPUT);
  pinMode(ULTRASOUND_FRONT_ECHO_PIN, INPUT);
  pinMode(ULTRASOUND_LEFT_TRIG_PIN, OUTPUT);
  pinMode(ULTRASOUND_LEFT_ECHO_PIN, INPUT);
  pinMode(ULTRASOUND_RIGHT_TRIG_PIN, OUTPUT);
  pinMode(ULTRASOUND_RIGHT_ECHO_PIN, INPUT);
  
  Serial.println("Ready for commands");
  Serial.println("Available commands:");
  Serial.println("  TEST - Test connection");
  Serial.println("  GET_FRONT - Get front ultrasound");
  Serial.println("  GET_LEFT - Get left ultrasound");
  Serial.println("  GET_RIGHT - Get right ultrasound");
  Serial.println("  SERVO_LEFT:<angle> - Set left servo");
  Serial.println("  SERVO_RIGHT:<angle> - Set right servo");
  Serial.println("  STOP - Stop both servos");
  
  // Initial sensor reading
  readUltrasoundSensors();
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    processCommand(command);
  }
  
  // Update sensors periodically
  unsigned long currentTime = millis();
  if (currentTime - lastSensorUpdate >= SENSOR_UPDATE_INTERVAL) {
    readUltrasoundSensors();
    lastSensorUpdate = currentTime;
  }
  
  // Update servos
  updateServos();
  
  delay(10);
}

void processCommand(String command) {
  command.toUpperCase();
  
  if (command == "TEST") {
    Serial.println("OK: Connection test successful");
  }
  else if (command == "GET_FRONT") {
    Serial.print("FRONT:");
    Serial.println(ultrasoundFront);
  }
  else if (command == "GET_LEFT") {
    Serial.print("LEFT:");
    Serial.println(ultrasoundLeft);
  }
  else if (command == "GET_RIGHT") {
    Serial.print("RIGHT:");
    Serial.println(ultrasoundRight);
  }
  else if (command.startsWith("SERVO_LEFT:")) {
    int angle = command.substring(11).toInt();
    if (angle >= 60 && angle <= 120) {
      servoLeftAngle = angle;
      servoLeft.write(angle);
      Serial.print("OK: Left servo set to ");
      Serial.println(angle);
    } else {
      Serial.println("ERROR: Invalid angle (60-120)");
    }
  }
  else if (command.startsWith("SERVO_RIGHT:")) {
    int angle = command.substring(12).toInt();
    if (angle >= 60 && angle <= 120) {
      servoRightAngle = angle;
      servoRight.write(angle);
      Serial.print("OK: Right servo set to ");
      Serial.println(angle);
    } else {
      Serial.println("ERROR: Invalid angle (60-120)");
    }
  }
  else if (command == "STOP") {
    servoLeftAngle = 90;
    servoRightAngle = 90;
    servoLeft.write(90);
    servoRight.write(90);
    Serial.println("OK: Both servos stopped");
  }
  else {
    Serial.print("ERROR: Unknown command - ");
    Serial.println(command);
  }
}

void readUltrasoundSensors() {
  ultrasoundFront = readUltrasound(ULTRASOUND_FRONT_TRIG_PIN, ULTRASOUND_FRONT_ECHO_PIN);
  ultrasoundLeft = readUltrasound(ULTRASOUND_LEFT_TRIG_PIN, ULTRASOUND_LEFT_ECHO_PIN);
  ultrasoundRight = readUltrasound(ULTRASOUND_RIGHT_TRIG_PIN, ULTRASOUND_RIGHT_ECHO_PIN);
}

int readUltrasound(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  long duration = pulseIn(echoPin, HIGH, 30000);
  int distance = duration * 0.0343 / 2;
  
  if (distance < 2) distance = 400;
  if (distance > 400) distance = 400;
  
  return distance;
}

void updateServos() {
  // Servos are updated directly in command processing
}