#include <SPI.h>
#include <SD.h>


const int SENSOR_PIN = A0;           // analog input pin
const int SD_CS_PIN  = 10;           // chip Select pin for SD module
const int WINDOW_SIZE = 12;          // must match training
const unsigned long SAMPLE_INTERVAL = 5000; // ms between samples
const char *WEIGHT_FILE = "/weights.txt";

//sample model weights
float W[8] = { -0.12, 0.85, -0.05, 0.07, -0.03, 0.01, 0.02, -0.005 };

//data buffers
float y_vals[WINDOW_SIZE];
float slopes[WINDOW_SIZE - 1];
unsigned long lastSampleTime = 0;

// shift array and append new value
void pushValue(float *arr, int len, float val) {
  for (int i = 0; i < len - 1; i++) arr[i] = arr[i + 1];
  arr[len - 1] = val;
}

// Compute slopes
void computeSlopes() {
  for (int i = 0; i < WINDOW_SIZE - 1; i++) {
    slopes[i] = y_vals[i + 1] - y_vals[i];
  }
}

//Feature functions
float computeVelocity() {
  float sum = 0;
  for (int i = 0; i < WINDOW_SIZE - 1; i++) sum += slopes[i];
  return sum / (WINDOW_SIZE - 1);
}

float computeAcceleration() {
  float acc = 0;
  for (int i = 1; i < WINDOW_SIZE - 1; i++) acc += (slopes[i] - slopes[i - 1]);
  return acc / (WINDOW_SIZE - 2);
}

float computeCurvature() {
  float v = computeVelocity();
  float var = 0;
  for (int i = 0; i < WINDOW_SIZE - 1; i++) {
    float d = slopes[i] - v;
    var += d * d;
  }
  return var / (WINDOW_SIZE - 1);
}


// added momentum component here
float computeMomentum() {
  float sum = 0, wsum = 0;
  for (int i = 0; i < WINDOW_SIZE - 1; i++) {
    float w = (i + 1);
    sum += w * slopes[i];
    wsum += w;
  }
  return sum / wsum;
}

// prediction
float predictNext() {
  computeSlopes();
  float V = computeVelocity();
  float A = computeAcceleration();
  float C = computeCurvature();
  float M = computeMomentum();

  float spoke_sum = 0;
  for (int i = 0; i < WINDOW_SIZE - 1; i++) {
    float s = slopes[i];
    float g = W[0] * s + W[1] * V + W[2] * A + W[3] * C + W[4] * M
            + W[5] * (s * C) + W[6] * (s * M) + W[7] * (s * s);
    spoke_sum += g;
  }
  float next_delta = spoke_sum / (WINDOW_SIZE - 1);
  return y_vals[WINDOW_SIZE - 1] + next_delta;
}

//cold loading from SD
// exception is thrown if file not found
bool loadWeights() {
  File f = SD.open(WEIGHT_FILE, FILE_READ);
  if (!f) {
    Serial.println("Failed to open weights.txt");
    return false;
  }
  int i = 0;
  while (f.available() && i < 8) {
    String line = f.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      W[i] = line.toFloat();
      i++;
    }
  }
  f.close();
  if (i != 8) {
    Serial.println("Weights file incomplete!");
    return false;
  }
  Serial.println("Weights loaded from SD:");
  for (int j = 0; j < 8; j++) {
    Serial.print("W[");
    Serial.print(j);
    Serial.print("] = ");
    Serial.println(W[j], 6);
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("SD init failed! Using default weights.");
  } else {
    loadWeights();
  }

  // fill buffer with initial readings
  for (int i = 0; i < WINDOW_SIZE; i++) {
    int raw = analogRead(SENSOR_PIN);
    y_vals[i] = raw / 1023.0;
    delay(100);
  }
  computeSlopes();
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = now;

    int raw = analogRead(SENSOR_PIN);
    float val = raw / 1023.0;
    pushValue(y_vals, WINDOW_SIZE, val);

    float prediction = predictNext();

    Serial.print("Current moisture: ");
    Serial.print(val, 3);
    Serial.print("  Predicted next: ");
    Serial.println(prediction, 3);
  }
}
