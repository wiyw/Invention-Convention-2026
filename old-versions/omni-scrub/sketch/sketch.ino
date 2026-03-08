// =============================================================================
//  OmniScrub — sketch.ino
//  Arduino Uno Q MCU (STM32U585)
//
//  Pin assignments:
//    Wheels:   L=3,  R=5  (continuous rotation servos)
//    Mop:      L=6,  R=9
//    Scoop:    L=10, R=11
//    Vacuum:   ESC=7
//    Ultrasonic FRONT: TRIG=A0, ECHO=A1
//    Ultrasonic LEFT:  TRIG=A2, ECHO=A3
//    Ultrasonic RIGHT: TRIG=A4, ECHO=A5
// =============================================================================

#include <Servo.h>                  // Library Manager: "Servo" by Arduino
#include <Arduino_RouterBridge.h>   // Built-in to App Lab

// -------- Pin Definitions --------
#define PIN_SERVO_WHEEL_L   3
#define PIN_SERVO_WHEEL_R   5
#define PIN_SERVO_MOP_L     6
#define PIN_SERVO_MOP_R     9
#define PIN_SERVO_SCOOP_L  10
#define PIN_SERVO_SCOOP_R  11
#define PIN_VACUUM_PWM      7

#define US_FRONT_TRIG  A0
#define US_FRONT_ECHO  A1
#define US_LEFT_TRIG   A2
#define US_LEFT_ECHO   A3
#define US_RIGHT_TRIG  A4
#define US_RIGHT_ECHO  A5

// -------- Servo Objects --------
Servo wheelL, wheelR;
Servo mopL,   mopR;
Servo scoopL, scoopR;
Servo vacESC;

// -------- State --------
int    currentSpeed = 50;
bool   mopOn        = false;
bool   vacOn        = false;
bool   scoopOpen    = false;
String driveDir     = "stop";

unsigned long lastSensorMs = 0;
const unsigned long SENSOR_INTERVAL_MS = 200;

// -------- Wheel helpers --------
// Continuous-rotation: 1000µs=full reverse, 1500µs=stop, 2000µs=full forward
void setWheels(int leftUs, int rightUs) {
  wheelL.writeMicroseconds(leftUs);
  wheelR.writeMicroseconds(rightUs);
}

int speedToUs(int pct) {
  return map(pct, 0, 100, 1500, 2000);
}

void applyDrive() {
  int fwd  = speedToUs(currentSpeed);
  int rev  = map(currentSpeed, 0, 100, 1500, 1000);
  int slow = speedToUs(currentSpeed / 2);

  if      (driveDir == "forward")  setWheels(fwd,  2000 - (fwd  - 1500));
  else if (driveDir == "backward") setWheels(rev,  2000 - (rev  - 1500));
  else if (driveDir == "left")     setWheels(1500 - (slow - 1500), 2000 - (slow - 1500));
  else if (driveDir == "right")    setWheels(slow, 2000 - (slow - 1500) - (slow - 1500));
  else                             setWheels(1500, 1500);
}

// -------- Ultrasonic helper --------
long readUltrasonic(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  long duration = pulseIn(echoPin, HIGH, 30000);
  if (duration == 0) return 999;
  return duration * 0.034 / 2;
}

// =============================================================================
// BRIDGE FUNCTIONS  (Python calls these)
// =============================================================================

void cmd_drive(String dir) {
  driveDir = dir;
  applyDrive();
  Bridge.call("servo_ack", "drive:" + dir);
  Monitor.print("Drive: " + dir);
}

void cmd_speed(String pctStr) {
  currentSpeed = constrain(pctStr.toInt(), 0, 100);
  applyDrive();
  Monitor.print("Speed: " + String(currentSpeed) + "%");
}

void cmd_mop(String state) {
  mopOn = (state == "on");
  int angle = mopOn ? 90 : 0;
  mopL.write(angle);
  mopR.write(180 - angle);
  Bridge.call("servo_ack", "mop:" + state);
  Monitor.print("Mop: " + state);
}

void cmd_vacuum(String state) {
  vacOn = (state == "on");
  if (vacOn) {
    vacESC.writeMicroseconds(1500);
    delay(100);
    vacESC.writeMicroseconds(map(currentSpeed, 0, 100, 1500, 2000));
  } else {
    vacESC.writeMicroseconds(1500);
  }
  Bridge.call("servo_ack", "vacuum:" + state);
  Monitor.print("Vacuum: " + state);
}

void cmd_scoop(String state) {
  scoopOpen = (state == "open");
  int angle = scoopOpen ? 120 : 10;
  scoopL.write(angle);
  scoopR.write(180 - angle);
  Bridge.call("servo_ack", "scoop:" + state);
  Monitor.print("Scoop: " + state);
}

void get_sensors() {
  long f = readUltrasonic(US_FRONT_TRIG, US_FRONT_ECHO);
  long l = readUltrasonic(US_LEFT_TRIG,  US_LEFT_ECHO);
  long r = readUltrasonic(US_RIGHT_TRIG, US_RIGHT_ECHO);
  char buf[64];
  snprintf(buf, sizeof(buf), "{\"f\":%ld,\"l\":%ld,\"r\":%ld}", f, l, r);
  Bridge.call("sensor_data", String(buf));
}

// =============================================================================
// SETUP & LOOP
// =============================================================================

void setup() {
  Serial.begin(115200);

  wheelL.attach(PIN_SERVO_WHEEL_L);
  wheelR.attach(PIN_SERVO_WHEEL_R);
  mopL  .attach(PIN_SERVO_MOP_L);
  mopR  .attach(PIN_SERVO_MOP_R);
  scoopL.attach(PIN_SERVO_SCOOP_L);
  scoopR.attach(PIN_SERVO_SCOOP_R);
  vacESC.attach(PIN_VACUUM_PWM);

  // Safe initial positions
  setWheels(1500, 1500);
  mopL.write(0);    mopR.write(180);
  scoopL.write(10); scoopR.write(170);
  vacESC.writeMicroseconds(1500);

  // Ultrasonic pins
  pinMode(US_FRONT_TRIG, OUTPUT); pinMode(US_FRONT_ECHO, INPUT);
  pinMode(US_LEFT_TRIG,  OUTPUT); pinMode(US_LEFT_ECHO,  INPUT);
  pinMode(US_RIGHT_TRIG, OUTPUT); pinMode(US_RIGHT_ECHO, INPUT);

  // Register Bridge functions
  Bridge.provide("cmd_drive",   cmd_drive);
  Bridge.provide("cmd_speed",   cmd_speed);
  Bridge.provide("cmd_mop",     cmd_mop);
  Bridge.provide("cmd_vacuum",  cmd_vacuum);
  Bridge.provide("cmd_scoop",   cmd_scoop);
  Bridge.provide("get_sensors", get_sensors);

  Monitor.print("OmniScrub sketch ready");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSensorMs >= SENSOR_INTERVAL_MS) {
    lastSensorMs = now;
    get_sensors();
  }
  delay(10);
}