// Arduino UNO Q4GB - On-Device AI Robot Controller
// Integrates TinyYOLO and TinyQwen for complete on-board intelligence

#include <Servo.h>
#include <stdint.h>
#include <math.h>
#include "tiny_qwen.h"
#include "tiny_yolo_weights.h"

// Hardware configuration
#define LEFT45_TRIG_PIN 2
#define LEFT45_ECHO_PIN 3
#define RIGHT45_TRIG_PIN 4
#define RIGHT45_ECHO_PIN 5
#define CENTER_TRIG_PIN 6
#define CENTER_ECHO_PIN 7

#define LEFT_SERVO_PIN 9
#define RIGHT_SERVO_PIN 10

#define CAMERA_CS_PIN 8  // If using SPI camera
#define LED_STATUS_PIN 13

// System configuration
#define MODEL_INPUT_WIDTH 160
#define MODEL_INPUT_HEIGHT 120
#define MAX_DETECTIONS 5
#define DECISION_CYCLE_MS 100
#define ULTRASONIC_TIMEOUT_MS 30

// Servo motor control
Servo leftServo;
Servo rightServo;

// State structures
struct Detection {
  int8_t class_id;
  uint8_t confidence;
  uint8_t x_center;
  uint8_t y_center;
  uint8_t width;
  uint8_t height;
};

struct SensorData {
  float left45;
  float right45;
  float center;
  uint32_t timestamp;
};

struct RobotState {
  Detection detections[MAX_DETECTIONS];
  uint8_t detection_count;
  SensorData sensors;
  uint8_t safety_level;
  uint32_t last_decision_time;
  uint8_t current_action;
  uint16_t cycle_count;
};

// Global state
RobotState robot_state;

// TinyYOLO inference (simplified)
class TinyYOLO {
private:
  // Simplified neural network implementation
  int8_t* weights;
  uint16_t input_size;
  uint16_t output_size;
  
public:
  TinyYOLO() {
    weights = (int8_t*)model_weights;
    input_size = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3;
    output_size = 9 * 8 * 10;  // Simplified output grid
  }
  
  // Simplified convolution implementation
  int8_t convolve(uint8_t* input, int8_t* kernel, uint8_t input_size, uint8_t kernel_size) {
    int32_t sum = 0;
    for (int i = 0; i < kernel_size * kernel_size; i++) {
      sum += ((int16_t)input[i] - 128) * kernel[i];
    }
    return (int8_t)(sum >> 8);  // Scale back to int8
  }
  
  // Simplified activation function (ReLU)
  int8_t relu(int8_t x) {
    return x > 0 ? x : 0;
  }
  
  // Very simplified YOLO inference (demonstration)
  uint8_t detect_objects(uint8_t* image_data, Detection* detections) {
    uint8_t detected_count = 0;
    
    // Edge detection as simple object detection
    for (int y = 10; y < MODEL_INPUT_HEIGHT - 10; y += 20) {
      for (int x = 10; x < MODEL_INPUT_WIDTH - 10; x += 20) {
        
        // Calculate local variance (edge detection)
        int32_t sum = 0;
        int32_t sum_sq = 0;
        int pixel_count = 0;
        
        for (int dy = -5; dy <= 5; dy++) {
          for (int dx = -5; dx <= 5; dx++) {
            int px = x + dx;
            int py = y + dy;
            if (px >= 0 && px < MODEL_INPUT_WIDTH && py >= 0 && py < MODEL_INPUT_HEIGHT) {
              uint8_t pixel = image_data[py * MODEL_INPUT_WIDTH + px];
              sum += pixel;
              sum_sq += (int32_t)pixel * pixel;
              pixel_count++;
            }
          }
        }
        
        if (pixel_count > 0) {
          int32_t mean = sum / pixel_count;
          int32_t variance = (sum_sq / pixel_count) - (mean * mean);
          
          // If high variance, likely an object edge
          if (variance > 1000 && detected_count < MAX_DETECTIONS) {
            detections[detected_count].x_center = (uint8_t)x;
            detections[detected_count].y_center = (uint8_t)y;
            detections[detected_count].width = 20;
            detections[detected_count].height = 20;
            detections[detected_count].confidence = (uint8_t)min(255, variance / 10);
            detections[detected_count].class_id = 0;  // Default class
            
            detected_count++;
          }
        }
      }
    }
    
    return detected_count;
  }
};

// TinyQwen decision engine
class TinyQwenEngine {
private:
  int8_t feature_vector[9];
  int8_t decision_scores[4];
  int16_t decision_weights[4];
  
public:
  TinyQwenEngine() {
    // Initialize decision weights
    for (int i = 0; i < 4; i++) {
      decision_weights[i] = 0;
    }
  }
  
  void quantize_input(float value, float min_val, float max_val, int8_t* result) {
    float normalized = (value - min_val) / (max_val - min_val);
    *result = (int8_t)(normalized * 255 - 128);
    if (*result < -128) *result = -128;
    if (*result > 127) *result = 127;
  }
  
  void process_detections(Detection* detections, uint8_t count) {
    if (count == 0) {
      feature_vector[F_HAS_OBJECT] = 0;
      feature_vector[F_PRIORITY] = 0;
      feature_vector[F_CENTER_OFFSET] = 0;
      feature_vector[F_SIZE_SCORE] = 0;
      feature_vector[F_CONFIDENCE] = 0;
      return;
    }
    
    // Find best detection
    uint8_t best_idx = 0;
    int16_t best_score = 0;
    
    for (uint8_t i = 0; i < count; i++) {
      int16_t score = (int16_t)detections[i].confidence * 100;
      if (score > best_score) {
        best_score = score;
        best_idx = i;
      }
    }
    
    // Extract features
    feature_vector[F_HAS_OBJECT] = 127;
    feature_vector[F_PRIORITY] = (int8_t)(best_score / 2);
    
    // Center offset from image center
    int16_t center_x = MODEL_INPUT_WIDTH / 2;
    int16_t offset = detections[best_idx].x_center - center_x;
    quantize_input((float)offset, -80.0, 80.0, &feature_vector[F_CENTER_OFFSET]);
    
    // Size score
    float size = (float)(detections[best_idx].width * detections[best_idx].height);
    quantize_input(size, 0.0, 400.0, &feature_vector[F_SIZE_SCORE]);
    
    // Confidence
    quantize_input(detections[best_idx].confidence, 0.0, 255.0, &feature_vector[F_CONFIDENCE]);
  }
  
  void process_ultrasonics(SensorData* sensors) {
    feature_vector[F_CENTER_SAFE] = (sensors->center > CRITICAL_DISTANCE) ? 127 : -127;
    feature_vector[F_LEFT_SAFE] = (sensors->left45 > CRITICAL_DISTANCE) ? 127 : -127;
    feature_vector[F_RIGHT_SAFE] = (sensors->right45 > CRITICAL_DISTANCE) ? 127 : -127;
    
    // Calculate safety level
    float min_dist = sensors->center;
    if (sensors->left45 > 0 && sensors->left45 < min_dist) min_dist = sensors->left45;
    if (sensors->right45 > 0 && sensors->right45 < min_dist) min_dist = sensors->right45;
    
    if (min_dist < CRITICAL_DISTANCE) {
      feature_vector[F_SAFETY_LEVEL] = 96;  // Critical
    } else if (min_dist < DANGER_DISTANCE) {
      feature_vector[F_SAFETY_LEVEL] = 64;  // Danger
    } else if (min_dist < CAUTION_DISTANCE) {
      feature_vector[F_SAFETY_LEVEL] = 32;  // Caution
    } else {
      feature_vector[F_SAFETY_LEVEL] = 0;    // Safe
    }
  }
  
  void compute_decision() {
    // Safety override
    if (feature_vector[F_SAFETY_LEVEL] >= 96) {
      decision_scores[ACTION_STOP] = 127;
      decision_scores[ACTION_FORWARD] = -127;
      decision_scores[ACTION_TURN_LEFT] = -127;
      decision_scores[ACTION_TURN_RIGHT] = -127;
      return;
    }
    
    // Initialize scores
    for (int i = 0; i < 4; i++) {
      decision_scores[i] = 0;
    }
    
    // Forward decision logic
    if (feature_vector[F_HAS_OBJECT] > 0 && 
        abs(feature_vector[F_CENTER_OFFSET]) < 32 &&
        feature_vector[F_CENTER_SAFE] > 0 &&
        feature_vector[F_SIZE_SCORE] < 64) {
      decision_scores[ACTION_FORWARD] = 100 + (feature_vector[F_CONFIDENCE] / 2);
    }
    
    // Turn left logic
    if (feature_vector[F_CENTER_OFFSET] < -32) {
      decision_scores[ACTION_TURN_LEFT] = 80 + abs(feature_vector[F_CENTER_OFFSET]);
    }
    
    // Turn right logic
    if (feature_vector[F_CENTER_OFFSET] > 32) {
      decision_scores[ACTION_TURN_RIGHT] = 80 + feature_vector[F_CENTER_OFFSET];
    }
    
    // Stop logic
    if (feature_vector[F_SAFETY_LEVEL] >= 64 || feature_vector[F_SIZE_SCORE] > 80) {
      decision_scores[ACTION_STOP] = 120;
    }
  }
  
  uint8_t get_best_action() {
    int8_t best_score = -128;
    uint8_t best_action = ACTION_STOP;
    
    for (uint8_t i = 0; i < 4; i++) {
      if (decision_scores[i] > best_score) {
        best_score = decision_scores[i];
        best_action = i;
      }
    }
    
    return best_action;
  }
  
  void get_motor_command(uint8_t action, uint8_t* left_speed, uint8_t* right_speed, uint16_t* duration) {
    uint8_t base_speed = 120;
    float confidence = (feature_vector[F_CONFIDENCE] + 128) / 256.0;
    confidence = max(0.5f, confidence);  // Ensure minimum confidence
    
    switch (action) {
      case ACTION_FORWARD:
        *left_speed = (uint8_t)(base_speed * confidence);
        *right_speed = (uint8_t)(base_speed * confidence);
        *duration = 500;
        break;
        
      case ACTION_TURN_LEFT:
        *left_speed = (uint8_t)max(0, base_speed * confidence - 30);
        *right_speed = (uint8_t)min(255, base_speed * confidence + 30);
        *duration = 300;
        break;
        
      case ACTION_TURN_RIGHT:
        *left_speed = (uint8_t)min(255, base_speed * confidence + 30);
        *right_speed = (uint8_t)max(0, base_speed * confidence - 30);
        *duration = 300;
        break;
        
      case ACTION_STOP:
      default:
        *left_speed = 0;
        *right_speed = 0;
        *duration = 200;
        break;
    }
  }
};

// Global AI engines
TinyYOLO yolo_engine;
TinyQwenEngine qwen_engine;

// Camera interface (simplified)
class SimpleCamera {
private:
  uint8_t frame_buffer[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT];
  
public:
  bool capture_frame() {
    // Simplified camera capture
    // In reality, this would interface with the camera hardware
    
    // Generate synthetic image data for demonstration
    for (int i = 0; i < MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT; i++) {
      frame_buffer[i] = (uint8_t)(analogRead(A0) / 4);  // Read from analog pin as fake camera
    }
    return true;
  }
  
  uint8_t* get_frame_data() {
    return frame_buffer;
  }
};

// Motor control
void execute_motor_command(uint8_t left_speed, uint8_t right_speed, uint16_t duration) {
  // Convert speed to servo angles (0-255 to 0-180)
  uint8_t left_angle = map(left_speed, 0, 255, 90, 180);  // 90=stop, 180=forward
  uint8_t right_angle = map(right_speed, 0, 255, 90, 180);
  
  // Apply to servos
  leftServo.write(left_angle);
  rightServo.write(right_angle);
  
  // Status indication
  digitalWrite(LED_STATUS_PIN, HIGH);
  
  // Execute for duration
  delay(duration);
  
  // Stop motors
  leftServo.write(90);
  rightServo.write(90);
  digitalWrite(LED_STATUS_PIN, LOW);
}

// Ultrasonic sensor functions
float readUltrasonicDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  long duration = pulseIn(echoPin, HIGH, ULTRASONIC_TIMEOUT_MS * 100);
  float distance = (duration * 0.0343) / 2.0;
  
  if (distance < 2.0 || distance > 400.0) {
    return -1.0;  // Invalid reading
  }
  
  return distance;
}

void updateSensorReadings() {
  robot_state.sensors.left45 = readUltrasonicDistance(LEFT45_TRIG_PIN, LEFT45_ECHO_PIN);
  robot_state.sensors.right45 = readUltrasonicDistance(RIGHT45_TRIG_PIN, RIGHT45_ECHO_PIN);
  robot_state.sensors.center = readUltrasonicDistance(CENTER_TRIG_PIN, CENTER_ECHO_PIN);
  robot_state.sensors.timestamp = millis();
}

// Main AI decision cycle
void ai_decision_cycle() {
  // Capture camera frame
  SimpleCamera camera;
  if (!camera.capture_frame()) {
    return;  // Skip cycle if camera fails
  }
  
  // Run TinyYOLO inference
  uint8_t* frame_data = camera.get_frame_data();
  robot_state.detection_count = yolo_engine.detect_objects(frame_data, robot_state.detections);
  
  // Process features with TinyQwen
  qwen_engine.process_detections(robot_state.detections, robot_state.detection_count);
  qwen_engine.process_ultrasonics(&robot_state.sensors);
  qwen_engine.compute_decision();
  
  // Get best action
  uint8_t action = qwen_engine.get_best_action();
  robot_state.current_action = action;
  
  // Execute motor command
  uint8_t left_speed, right_speed;
  uint16_t duration;
  qwen_engine.get_motor_command(action, &left_speed, &right_speed, &duration);
  
  execute_motor_command(left_speed, right_speed, duration);
  
  // Update state
  robot_state.last_decision_time = millis();
  robot_state.cycle_count++;
  
  // Status output (optional)
  outputStatus();
}

void outputStatus() {
  // Simple status output via serial
  Serial.print("Cycle ");
  Serial.print(robot_state.cycle_count);
  Serial.print(": Action=");
  
  switch (robot_state.current_action) {
    case ACTION_FORWARD: Serial.print("FORWARD"); break;
    case ACTION_TURN_LEFT: Serial.print("LEFT"); break;
    case ACTION_TURN_RIGHT: Serial.print("RIGHT"); break;
    case ACTION_STOP: Serial.print("STOP"); break;
    default: Serial.print("UNKNOWN"); break;
  }
  
  Serial.print(", Detections=");
  Serial.print(robot_state.detection_count);
  Serial.print(", CenterDist=");
  Serial.print(robot_state.sensors.center);
  Serial.println("cm");
}

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) delay(10);
  
  Serial.println("=== Arduino UNO Q4GB AI Robot ===");
  Serial.println("On-device TinyYOLO + TinyQwen inference");
  
  // Initialize pins
  pinMode(LEFT45_TRIG_PIN, OUTPUT);
  pinMode(LEFT45_ECHO_PIN, INPUT);
  pinMode(RIGHT45_TRIG_PIN, OUTPUT);
  pinMode(RIGHT45_ECHO_PIN, INPUT);
  pinMode(CENTER_TRIG_PIN, OUTPUT);
  pinMode(CENTER_ECHO_PIN, INPUT);
  pinMode(LED_STATUS_PIN, OUTPUT);
  
  // Initialize servos
  leftServo.attach(LEFT_SERVO_PIN);
  rightServo.attach(RIGHT_SERVO_PIN);
  
  // Stop motors initially
  leftServo.write(90);
  rightServo.write(90);
  
  // Initialize robot state
  memset(&robot_state, 0, sizeof(robot_state));
  robot_state.last_decision_time = millis();
  
  Serial.println("AI Robot Ready!");
  Serial.println("Starting autonomous navigation...");
}

void loop() {
  uint32_t current_time = millis();
  
  // Update sensor readings
  updateSensorReadings();
  
  // Run AI decision cycle
  if (current_time - robot_state.last_decision_time >= DECISION_CYCLE_MS) {
    ai_decision_cycle();
  }
  
  // Small delay to prevent overwhelming the system
  delay(10);
}