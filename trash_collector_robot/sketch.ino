/*
  Arduino UNO R4 3-Sensor Ultrasonic Controller
  For Arduino App Labs - Front, Left 45°, Right 45°
  
  Motor pins
*/
const int LEFT_MOTOR = 3;
const int RIGHT_MOTOR = 5;
const int LED_PIN = 13;

void setup() {
  Serial.begin(115200);
  
  // Initialize motor pins
  pinMode(LEFT_MOTOR, OUTPUT);
  pinMode(RIGHT_MOTOR, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  
  // Initialize motors to stopped
  digitalWrite(LEFT_MOTOR, LOW);
  digitalWrite(RIGHT_MOTOR, LOW);
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("=== Arduino 3-Sensor Controller ===");
  Serial.println("Motors: 2x DC Motors");
  Serial.println("Sensors: Front, Left 45°, Right 45°");
  Serial.println("Baud: 115200");
  Serial.println("================================");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // Parse MOVE commands: MOVE:sensor:distance:angle
    if (command.startsWith("MOVE:")) {
      parseMoveCommand(command);
    }
    else if (command.equals("STOP")) {
      stopMotors();
      Serial.println("Motors STOPPED");
      digitalWrite(LED_PIN, LOW);
    }
    else if (command.equals("TEST")) {
      runMotorTest();
    }
    else if (command.equals("STATUS")) {
      sendStatus();
    }
    else if (command.length() > 0) {
      Serial.print("Unknown command: ");
      Serial.println(command);
    }
  }
  
  delay(50);
}

void parseMoveCommand(String command) {
  // Parse: MOVE:sensor:distance:angle
  int colon1 = command.indexOf(':');
  int colon2 = command.indexOf(':', colon1 + 1);
  int colon3 = command.indexOf(':', colon2 + 1);
  
  if (colon1 > 0 && colon2 > 0 && colon3 > 0) {
    String sensor = command.substring(colon1 + 1, colon2);
    float distance = command.substring(colon2 + 1, colon3).toFloat();
    int angle = command.substring(colon3 + 1).toInt();
    
    Serial.print("Moving to ");
    Serial.print(sensor);
    Serial.print(" at ");
    Serial.print(distance, 1);
    Serial.print("cm, ");
    Serial.print(angle);
    Serial.println("°");
    
    digitalWrite(LED_PIN, HIGH);
    
    // Move based on sensor
    if (sensor.equals("front")) {
      moveForward(1000);
    }
    else if (sensor.equals("left_45")) {
      turnLeft(400);
      moveForward(800);
    }
    else if (sensor.equals("right_45")) {
      turnRight(400);
      moveForward(800);
    }
    
    delay(500);
    digitalWrite(LED_PIN, LOW);
    Serial.println("Movement complete");
  }
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

void runMotorTest() {
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
  Serial.println("===================");
}

void sendStatus() {
  Serial.println("=== 3-Sensor Status ===");
  Serial.print("Left Motor: ");
  Serial.println(digitalRead(LEFT_MOTOR) == HIGH ? "ON" : "OFF");
  Serial.print("Right Motor: ");
  Serial.println(digitalRead(RIGHT_MOTOR) == HIGH ? "ON" : "OFF");
  Serial.print("LED: ");
  Serial.println(digitalRead(LED_PIN) == HIGH ? "ON" : "OFF");
  Serial.print("Uptime: ");
  Serial.print(millis() / 1000);
  Serial.println("s");
  Serial.print("Sensors: Front, Left 45°, Right 45°");
  Serial.println("===================");
}