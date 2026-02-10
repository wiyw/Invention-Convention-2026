/*
  Test Arduino App Labs Structure
  Minimal working example
*/

void setup() {
  Serial.begin(115200);
  Serial.println("=== Arduino App Labs Test ===");
  Serial.println("Arduino sketch working...");
}

void loop() {
  Serial.println("Test loop running...");
  delay(1000);
}