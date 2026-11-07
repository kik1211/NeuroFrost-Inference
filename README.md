# 🧠 In-Ear Neurothermal Anomaly Detection via Embedded ML

[![CI](https://github.com/kik1211/NeuroFrost-Inference/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kik1211/NeuroFrost-Inference/actions/workflows/ci.yml)

An experimental embedded machine learning proof-of-concept that classifies temporal infrared thermal patterns from the ear canal using non-contact thermometry and a quantized neural network running entirely on a microcontroller.

> **⚠️ This is a research prototype. It is NOT a medical device. See [Disclaimer](docs/DISCLAIMER.md).**

---

## 🔬 Project Overview

This repository contains firmware and a compiled inference SDK that continuously sample tympanic-region temperature, extract spectral features from the time series, and classify each window as `normal` or `stroke` using an on-device neural network.

The tympanic region is anatomically adjacent to cerebral arterial pathways. This project investigates whether temporal patterns in ear-canal infrared readings—processed locally on a microcontroller—can distinguish baseline thermal profiles from anomalous ones.

**Scope:** This codebase covers the **embedded inference stage only**. Model training was performed externally using [Edge Impulse](https://www.edgeimpulse.com/). The training dataset and training scripts are not included.

---

## 🛠️ Hardware

| Component | Part | Details |
| :--- | :--- | :--- |
| **IR Sensor** | MLX90614 | Non-contact infrared thermometer. Factory-calibrated. I²C address `0x5A`. |
| **Interface** | I²C / SMBus | Two-wire serial bus. Requires `SDA`, `SCL`, `VCC`, `GND`. |
| **MCU** | Arduino-compatible board | Any board supporting `<Wire.h>` and the Edge Impulse C++ SDK (e.g., ESP32, SAMD21, RP2040). |

### Wiring

| MLX90614 Pin | MCU Pin | Notes |
| :--- | :--- | :--- |
| VCC | 3.3 V or 5 V | Per sensor module board specification |
| GND | GND | — |
| SDA | SDA | 4.7 kΩ pull-up to VCC if not on breakout board |
| SCL | SCL | 4.7 kΩ pull-up to VCC if not on breakout board |

---

## 💻 Software Stack

| Layer | Component | Notes |
| :--- | :--- | :--- |
| **Firmware** | `neuro.ino` (Arduino C++) | Sensor polling, sample buffering, inference invocation, serial logging |
| **Sensor Driver** | Adafruit MLX90614 Library | I²C driver; provides `readObjectTempC()` |
| **Inference SDK** | `Neuro_Frost_inferencing/` | Auto-generated Edge Impulse C++ library (v1.0.2) containing DSP and NN code |
| **NN Runtime** | TensorFlow Lite for Microcontrollers | EON-compiled int8 model; no interpreter overhead |
| **DSP** | Edge Impulse Spectral Analysis | FFT-based feature extraction built into the SDK |

---

## 🧠 Machine Learning Pipeline

### 1. Data Acquisition

The firmware reads `mlx.readObjectTempC()` in a synchronous loop with `delay(EI_CLASSIFIER_INTERVAL_MS)` between samples.

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Sampling interval | 500 ms (2 Hz) | `EI_CLASSIFIER_INTERVAL_MS` · `model_metadata.h:96` |
| Samples per window | 8 | `EI_CLASSIFIER_RAW_SAMPLE_COUNT` · `model_metadata.h:88` |
| Axes per sample | 1 (`temperature`) | `EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME` · `model_metadata.h:89` |
| Window duration | 4.0 s | 8 × 500 ms |
| Windowing strategy | Tumbling (non-overlapping) | `neuro.ino:26-38,70` — buffer fills, infers, resets index to 0 |

### 2. DSP — Spectral Feature Extraction

The 8-sample buffer is passed to the Edge Impulse spectral analysis block, which produces **13 features**.

| DSP Parameter | Value | Source |
| :--- | :--- | :--- |
| Analysis type | FFT | `model_variables.h:63` |
| FFT length | 16 | `model_variables.h:64` |
| Spectral peaks | 3 | `model_variables.h:65` |
| Power edge frequencies | 0.1, 0.5, 1.0, 2.0, 5.0 Hz | `model_variables.h:67` |
| Log scaling | Enabled | `model_variables.h:68` |
| Output feature count | 13 | `EI_CLASSIFIER_NN_INPUT_FRAME_SIZE` · `model_metadata.h:87` |

### 3. Neural Network

| Layer | Type | Shape | Activation | Quantization |
| :--- | :--- | :--- | :--- | :--- |
| Input | — | (1, 13) | — | int8 affine |
| Dense 1 | Fully Connected | (13) → (20) | ReLU | int8 |
| Dense 2 | Fully Connected | (20) → (10) | ReLU | int8 |
| Output | Fully Connected + Softmax | (10) → (2) | Softmax | int8 |

**Classification labels:** `normal`, `stroke` — defined in `model_variables.h:50`.

**Tensor arena size:** 368 bytes (default) / 2944 bytes (largest arena) — `tflite_learn_768102_9_compiled.cpp:99`, `model_metadata.h:112`.

**Registered ops:** `FULLY_CONNECTED`, `SOFTMAX` — `tflite_learn_768102_9_compiled.cpp:568-569`.

### Inference Flow

```text
MLX90614 (I2C)                    Edge Impulse SDK              Serial Output
─────────────                     ────────────────              ─────────────
readObjectTempC()  ──►  features[0..7]  ──►  Spectral DSP (13 features)
                                             │
                                             ▼
                                        Dense NN (int8)
                                        13 → 20 → 10 → 2
                                             │
                                             ▼
                                        Softmax probabilities
                                             │
                                             ▼
                                        Serial.println()
                                        normal: 0.xxxx
                                        stroke: 0.xxxx
```

For a detailed technical breakdown, see [`docs/INFERENCE_PIPELINE.md`](docs/INFERENCE_PIPELINE.md).

---

## 📂 Repository Structure

```
├── neuro.ino                   Arduino sketch — sensor read loop, buffering, inference
├── Neuro_Frost_inferencing/    Edge Impulse C++ SDK (auto-generated, read-only)
│   ├── library.properties      Arduino library metadata
│   ├── examples/               Board-specific example sketches (from Edge Impulse)
│   └── src/
│       ├── model-parameters/   Model constants, DSP config, label definitions
│       ├── tflite-model/       EON-compiled int8 neural network weights and graph
│       └── edge-impulse-sdk/   TFLite Micro runtime, CMSIS-DSP, classifier engine
├── docs/
│   ├── DISCLAIMER.md           Research and medical disclaimer
│   └── INFERENCE_PIPELINE.md   Detailed sampling, DSP, and NN specification
├── LICENSE                     Apache License 2.0
└── .gitignore                  Build artifact exclusions
```

---

## 🚀 Build Instructions

### Prerequisites

1. **Arduino IDE** v1.8.x or v2.x.
2. **Adafruit MLX90614 Library** — install via *Sketch → Include Library → Manage Libraries*.

### Steps

1. Copy `Neuro_Frost_inferencing/` into your Arduino libraries folder  
   (e.g., `~/Documents/Arduino/libraries/Neuro_Frost_inferencing/`).
2. Open `neuro.ino` in Arduino IDE.
3. Select your board and port under **Tools**.
4. Click **Upload**.
5. Open **Serial Monitor** at **115200 baud**.

### Expected Serial Output

```
MLX90614 + Edge Impulse inference started...
Sample 0: 36.47 C
Sample 1: 36.51 C
...
Sample 7: 36.49 C
Inference results:
normal: 0.9412
stroke: 0.0588
--------------------------
```

---

## ⚠️ Limitations

| Limitation | Detail |
| :--- | :--- |
| **Not a medical device** | This is an engineering proof-of-concept. No clinical validation has been performed. |
| **Blocking acquisition loop** | `neuro.ino` uses `delay(500)` inside a `while` loop, blocking the MCU for ~4 s per inference cycle. |
| **No sensor error handling** | I²C read failures or NaN returns from the sensor are not checked before buffering. |
| **Tumbling window only** | Non-overlapping 8-sample blocks; no sliding window or overlap. Temporal discontinuities at block boundaries are not captured. |
| **No ambient compensation** | Readings may drift with ambient temperature, sensor repositioning, or body movement. |
| **Training data not included** | The dataset and Edge Impulse project configuration used to train this model are not in this repository. Model performance metrics cannot be independently verified from repository contents alone. |
| **Boot lockup on sensor failure** | If `mlx.begin()` fails at startup, the MCU enters `while(1)` and cannot recover without a hardware reset. |

---

## ⚖️ Disclaimer

This project is an **experimental prototype and research exploration**. It is **NOT** a clinically validated medical diagnostic device.

Do not use this system to diagnose, treat, cure, or prevent any disease. See [`docs/DISCLAIMER.md`](docs/DISCLAIMER.md) for the full disclaimer.

---

## 📚 References & Acknowledgements

| Resource | Role in This Project |
| :--- | :--- |
| [Edge Impulse](https://www.edgeimpulse.com/) | Model training platform, DSP block generation, C++ SDK export |
| [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) | On-device neural network inference runtime |
| [CMSIS-DSP](https://arm-software.github.io/CMSIS_5/DSP/html/index.html) | Arm DSP library for signal processing acceleration |
| [Adafruit MLX90614 Library](https://github.com/adafruit/Adafruit-MLX90614-Library) | Arduino I²C driver for the MLX90614 sensor |
