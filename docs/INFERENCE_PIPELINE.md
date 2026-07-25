# 🧠 Technical Inference Pipeline Specification

This document provides a technical specification of the signal acquisition, pre-processing, feature extraction, and neural network inference pipeline implemented in `neuro.ino` and the `Neuro_Frost_inferencing` C++ SDK.

---

## 1. Sensor Data Acquisition & Buffering

* **Sensor Module:** MLX90614 Non-Contact Infrared Thermometer.
* **Interface Protocol:** Standard I²C / SMBus (`0x5A` default address).
* **Sensor Read Method:** `mlx.readObjectTempC()` (reads object temperature in degrees Celsius).
* **Sampling Rate:** 2 Hz (1 sample every 500 ms, defined by `EI_CLASSIFIER_INTERVAL_MS = 500`).
* **Buffer Size:** 8 raw single-precision floating-point samples (`EI_CLASSIFIER_RAW_SAMPLE_COUNT = 8`).
* **Window Duration:** 4.0 seconds (8 samples × 0.5 seconds).
* **Buffering Mechanism:** **Synchronous Tumbling Window (Fixed Block Buffer)**. The firmware collects 8 sequential temperature samples in a synchronous `while` loop with a 500 ms delay per sample (`delay(EI_CLASSIFIER_INTERVAL_MS)`). Once 8 samples are collected, inference is triggered, the buffer index resets (`feature_ix = 0`), and acquisition of the next 8-sample block begins.

---

## 2. DSP Feature Extraction (Spectral Analysis)

The 8-sample raw temperature block is wrapped in an Edge Impulse signal structure (`numpy::signal_from_buffer`) and passed to the DSP extraction engine (`extract_spectral_analysis_features`):

```text
┌──────────────────────────────────────────────┐
│ Raw Temporal Window (8 Samples @ 2 Hz)       │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Spectral DSP Processing                      │
│ • Analysis Type : FFT                        │
│ • FFT Length    : 16                         │
│ • Spectral Peaks: 3                          │
│ • Power Edges   : 0.1, 0.5, 1.0, 2.0, 5.0 Hz  │
│ • Log Scaling   : Enabled                    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Extracted Feature Vector (13 Features)       │
└──────────────────────────────────────────────┘
```

The DSP block transforms the 8 temporal temperature values into a 13-element feature array (`EI_CLASSIFIER_NN_INPUT_FRAME_SIZE = 13`).

---

## 3. Neural Network Topology & Quantization

The 13-feature array is evaluated locally by a compiled TensorFlow Lite for Microcontrollers (EON Compiler) neural network:

| Layer Stage | Layer Type | Input Shape | Output Shape | Tensor Type & Quantization |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Feature Vector | (13,) | (13,) | `kTfLiteInt8` Affine Quantization |
| **Dense 1** | Fully Connected + ReLU | (13,) | (20,) | `kTfLiteInt8` (260 weights + 20 biases) |
| **Dense 2** | Fully Connected + ReLU | (20,) | (10,) | `kTfLiteInt8` (200 weights + 10 biases) |
| **Output** | Fully Connected + Softmax | (10,) | (2,) | `kTfLiteInt8` (20 weights + 2 biases) |

### Output Class Labels
The output tensor maps to two classification categories defined in `model_variables.h`:
1. `normal`: Thermal temporal profile matching baseline thermal stability.
2. `stroke`: Thermal temporal profile exhibiting spectral/drift variations associated with anomalous patterns.

---

## 4. Execution Flow in Firmware (`neuro.ino`)

```text
 ┌────────────────────────────────────────┐
 │ setup()                                │
 │ 1. Initialize Serial @ 115200 baud     │
 │ 2. Initialize Wire & MLX90614 sensor   │
 └───────────────────┬────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │ loop() - Data Acquisition              │
 │ Synchronous while loop (8 samples)     │ ──> Reads mlx.readObjectTempC()
 │ Delays 500 ms between samples          │     (Blocks for 4.0s total)
 └───────────────────┬────────────────────┘
                     │ Buffer Full (8 samples)
                     ▼
 ┌────────────────────────────────────────┐
 │ Signal Preparation                     │
 │ numpy::signal_from_buffer(features, 8) │
 └───────────────────┬────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │ Classifier Execution                   │
 │ run_classifier(&signal, &result, false)│ ──> Executes Spectral DSP + TFLite NN
 └───────────────────┬────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │ Result Output                          │
 │ Prints label & value to Serial         │ ──> Output: normal: 0.xxxx
 └───────────────────┬────────────────────┘            stroke: 0.xxxx
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │ Buffer Reset                           │
 │ feature_ix = 0                         │ ──> Ready for next 8-sample block
 └────────────────────────────────────────┘
```
