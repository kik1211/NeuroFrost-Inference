#include <Wire.h>
#include <Adafruit_MLX90614.h>

// Include your Edge Impulse model library
#include <Neuro_Frost_inferencing.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];  // should be 8
uint16_t feature_ix = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!mlx.begin()) {
    Serial.println("Error connecting to MLX90614 sensor!");
    while (1);
  }

  Serial.println("MLX90614 + Edge Impulse inference started...");
}

void loop() {
  // Collect samples every 500 ms until buffer full
  while (feature_ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
    float tempC = mlx.readObjectTempC();
    features[feature_ix] = tempC;   // store in feature array

    Serial.print("Sample ");
    Serial.print(feature_ix);
    Serial.print(": ");
    Serial.print(tempC);
    Serial.println(" C");

    feature_ix++;
    delay(EI_CLASSIFIER_INTERVAL_MS); // wait 500 ms
  }

  // Once buffer full -> Run inference
  ei_impulse_result_t result = { 0 };
  signal_t signal;

  int err = numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
  if (err != 0) {
    Serial.print("Failed to create signal: ");
    Serial.println(err);
    return;
  }

  err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) {
    Serial.print("ERROR: Failed to run classifier (");
    Serial.print(err);
    Serial.println(")");
    return;
  }

  // Print results
  Serial.println("Inference results:");
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    Serial.print(result.classification[ix].label);
    Serial.print(": ");
    Serial.println(result.classification[ix].value, 4);
  }

  Serial.println("--------------------------");

  // Reset buffer index for next inference
  feature_ix = 0;
}
