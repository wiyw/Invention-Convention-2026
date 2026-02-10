/*
  Arduino UNO R4 Ultrasonic Trash Collector
  For Arduino App Labs deployment
  
  Receives commands and controls motors
  Baud rate: 115200
*/

// Motor pins
const int LEFT_MOTOR = 3;
const int RIGHT_MOTOR = 5;
const int LED_PIN = 13;

void setup() {
  Serial.begin(115200);
  
  // Initialize motor pins
  pinMode(LEFT_MOTOR, OUTPUT);
  pinMode(RIGHT_MOTOR, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  
  // Initialize motors to stopped state
  digitalWrite(LEFT_MOTOR, LOW);
  digitalWrite(RIGHT_MOTOR, LOW);
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("=== Arduino App Labs Ultrasonic Controller ===");
  Serial.println("Version: 1.0.0");
  Serial.println("Motors: Ready");
  Serial.println("========================================");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("MOVE_TO:")) {
      // Parse move command: MOVE_TO:sensor@distance
      int colon = command.indexOf(':');
      int at = command.indexOf('@');
      
      if (colon > 0 && at > colon) {
        String sensor = command.substring(colon + 1, at);
        String distance_str = command.substring(at + 1);
        float distance = distance_str.toFloat();
        
        Serial.print("Moving to ");
        Serial.print(sensor);
        Serial.print(" at ");
        Serial.print(distance, 1);
        Serial.println("cm");
        
        // Move motors based on sensor
        if (sensor == "front") {
          moveForward(1000);
        } else if (sensor == "front_left") {
          turnLeft(300);
          moveForward(800);
        } else if (sensor == "front_right") {
          turnRight(300);
          moveForward(800);
        } else if (sensor == "left") {
          turnLeft(500);
          moveForward(600);
        } else if (sensor == "right") {
          turnRight(500);
          moveForward(600);
        }
        
        digitalWrite(LED_PIN, HIGH);
        delay(1000);
        digitalWrite(LED_PIN, LOW);
      }
    } else if (command == "STOP") {
      stopMotors();
      digitalWrite(LED_PIN, LOW);
    } else if (command == "STATUS") {
      sendStatus();
    } else if (command == "TEST") {
      runTest();
    } else if (command.length() > 0) {
      Serial.print("Unknown command: ");
      Serial.println(command);
    }
  }
  
  delay(50);
}

void moveForward(int duration) {
  Serial.println("Moving forward");
  digitalWrite(LEFT_MOTOR, HIGH);
  digitalWrite(RIGHT_MOTOR, HIGH);
  delay(duration);
  stopMotors();
}

void turnLeft(int duration) {
  Serial.println("Turning left");
  digitalWrite(LEFT_MOTOR, LOW);
  digitalWrite(RIGHT_MOTOR, HIGH);
  delay(duration);
  stopMotors();
}

void turnRight(int duration) {
  Serial.println("Turning right");
  digitalWrite(LEFT_MOTOR, HIGH);
  digitalWrite(RIGHT_MOTOR, LOW);
  delay(duration);
  stopMotors();
}

void stopMotors() {
  digitalWrite(LEFT_MOTOR, LOW);
  digitalWrite(RIGHT_MOTOR, LOW);
}

void sendStatus() {
  Serial.println("=== Status ===");
  Serial.print("Left Motor: ");
  Serial.println(digitalRead(LEFT_MOTOR) == HIGH ? "ON" : "OFF");
  Serial.print("Right Motor: ");
  Serial.println(digitalRead(RIGHT_MOTOR) == HIGH ? "ON" : "OFF");
  Serial.print("LED: ");
  Serial.println(digitalRead(LED_PIN) == HIGH ? "ON" : "OFF");
  Serial.print("Uptime: ");
  Serial.print(millis() / 1000);
  Serial.println("s");
  Serial.println("===============");
}

void runTest() {
  Serial.println("=== Motor Test ===");
  
  Serial.println("Testing forward...");
  moveForward(500);
  delay(500);
  
  Serial.println("Testing left turn...");
  turnLeft(300);
  delay(500);
  
  Serial.println("Testing right turn...");
  turnRight(300);
  delay(500);
  
  Serial.println("Test complete");
  Serial.println("================");
}