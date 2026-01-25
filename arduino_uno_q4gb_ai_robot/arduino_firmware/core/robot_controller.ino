// Arduino Uno Q4GB - YOLO26n + Qwen2.5-0.5B-Instruct Robot Controller
// Controls 2 servo motors (wheels) with 3 ultrasonic sensors and camera integration

#include <Servo.h>

// Ultrasonic sensor pins
#define LEFT45_TRIG_PIN 2
#define LEFT45_ECHO_PIN 3
#define RIGHT45_TRIG_PIN 4
#define RIGHT45_ECHO_PIN 5
#define CENTER_TRIG_PIN 6
#define CENTER_ECHO_PIN 7

// Servo motor pins
#define LEFT_SERVO_PIN 9
#define RIGHT_SERVO_PIN 10

// Motor speed and direction control
Servo leftServo;
Servo rightServo;

// Command parsing
String inputCommand = "";
boolean commandReady = false;

// Ultrasonic sensor readings
struct SensorData {
  float left45;
  float right45;
  float center;
  unsigned long timestamp;
};

SensorData currentSensors;

void setup() {
  Serial.begin(115200);
  
  // Initialize ultrasonic sensors
  pinMode(LEFT45_TRIG_PIN, OUTPUT);
  pinMode(LEFT45_ECHO_PIN, INPUT);
  pinMode(RIGHT45_TRIG_PIN, OUTPUT);
  pinMode(RIGHT45_ECHO_PIN, INPUT);
  pinMode(CENTER_TRIG_PIN, OUTPUT);
  pinMode(CENTER_ECHO_PIN, INPUT);
  
  // Initialize servos
  leftServo.attach(LEFT_SERVO_PIN);
  rightServo.attach(RIGHT_SERVO_PIN);
  
  // Stop motors initially
  stopMotors();
  
  Serial.println("Arduino Robot Controller Ready");
  Serial.println("Format: CMD L<speed> R<speed> T<duration_ms>");
}

void loop() {
  // Read sensors continuously
  updateSensorReadings();
  
  // Check for incoming commands
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputCommand.length() > 0) {
        commandReady = true;
      }
    } else {
      inputCommand += c;
    }
  }
  
  // Process commands when ready
  if (commandReady) {
    processCommand(inputCommand);
    inputCommand = "";
    commandReady = false;
  }
  
  // Send sensor data periodically
  static unsigned long lastSensorReport = 0;
  if (millis() - lastSensorReport > 100) { // Report every 100ms
    sendSensorData();
    lastSensorReport = millis();
  }
  
  delay(10);
}

void updateSensorReadings() {
  currentSensors.left45 = readUltrasonicDistance(LEFT45_TRIG_PIN, LEFT45_ECHO_PIN);
  currentSensors.right45 = readUltrasonicDistance(RIGHT45_TRIG_PIN, RIGHT45_ECHO_PIN);
  currentSensors.center = readUltrasonicDistance(CENTER_TRIG_PIN, CENTER_ECHO_PIN);
  currentSensors.timestamp = millis();
}

float readUltrasonicDistance(int trigPin, int echoPin) {
  // Send 10us pulse to trigger
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Read echo pulse
  long duration = pulseIn(echoPin, HIGH, 30000); // 30ms timeout
  
  // Convert to cm (sound speed = 343 m/s)
  float distance = (duration * 0.0343) / 2.0;
  
  // Filter invalid readings
  if (distance < 2.0 || distance > 400.0) {
    return -1; // Invalid reading
  }
  
  return distance;
}

void processCommand(String command) {
  command.trim();
  
  if (command.startsWith("CMD")) {
    // Parse: CMD L<left_speed> R<right_speed> T<duration>
    int leftSpeed = 0;
    int rightSpeed = 0;
    int duration = 0;
    
    // Extract left speed
    int leftIndex = command.indexOf('L');
    int rightIndex = command.indexOf('R');
    int timeIndex = command.indexOf('T');
    
    if (leftIndex != -1 && rightIndex != -1 && timeIndex != -1) {
      leftSpeed = command.substring(leftIndex + 1, rightIndex).toInt();
      rightSpeed = command.substring(rightIndex + 1, timeIndex).toInt();
      duration = command.substring(timeIndex + 1).toInt();
      
      // Execute motor command
      executeMotorCommand(leftSpeed, rightSpeed, duration);
    }
  }
}

void executeMotorCommand(int leftSpeed, int rightSpeed, int duration) {
  // Convert speed (0-255) to servo pulse width
  // 90 = stop, 0 = full reverse, 180 = full forward
  int leftServoValue = map(leftSpeed, 0, 255, 0, 180);
  int rightServoValue = map(rightSpeed, 0, 255, 0, 180);
  
  // Apply motor control
  leftServo.write(leftServoValue);
  rightServo.write(rightServoValue);
  
  // Send confirmation
  Serial.print("MOTORS L:");
  Serial.print(leftSpeed);
  Serial.print(" R:");
  Serial.print(rightSpeed);
  Serial.print(" T:");
  Serial.println(duration);
  
  // Run for specified duration
  delay(duration);
  
  // Stop motors after duration
  stopMotors();
}

void stopMotors() {
  leftServo.write(90); // Center position = stop
  rightServo.write(90);
}

void sendSensorData() {
  // Send JSON formatted sensor data
  Serial.print("SENSORS {");
  Serial.print("\"left45\":");
  Serial.print(currentSensors.left45);
  Serial.print(",\"right45\":");
  Serial.print(currentSensors.right45);
  Serial.print(",\"center\":");
  Serial.print(currentSensors.center);
  Serial.print(",\"timestamp\":");
  Serial.print(currentSensors.timestamp);
  Serial.println("}");
}