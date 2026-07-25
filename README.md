# 🧠 In-Ear Neurothermal Anomaly Detection via Embedded ML

An experimental embedded machine learning proof-of-concept investigating the classification of temporal infrared thermal patterns in the ear canal using non-contact infrared thermometry and quantized neural networks.

---

## 🔬 Project Overview

This repository contains firmware and an embedded inference SDK designed to evaluate temporal thermal dynamics in the ear canal. The tympanic region is anatomically adjacent to major cerebral arterial pathways. This project explores whether continuous, local temporal temperature readings—processed via on-device spectral feature extraction and a lightweight neural network—can distinguish baseline thermal dynamics from anomalous temporal profiles.

This codebase represents the **embedded deployment and inference stage**. Model generation and training were conducted externally using Edge Impulse.

---

## 🛠️ Hardware

### Component Specifications

| Component | Part / Library | Specification / Role |
| :--- | :--- | :--- |
| **Infrared Sensor** | MLX90614 | Medical-grade non-contact IR thermometer, factory-calibrated |
| **Bus Interface** | I²C / SMBus | Default bus address `0x5A`, 2 Hz sampling frequency |
| **Microcontroller** | Arduino-compatible MCU | Any C++ board supporting `<Wire.h>` and standard C++11 (e.g., SAMD21, ESP32, RP2040) |

### Hardware Setup Placeholder
> `[ Hardware Setup Photograph Placeholder ]`  
> *Connect MLX90614 VCC to 3.3V/5V, GND to GND, SDA to MCU SDA, and SCL to MCU SCL (with 4.7 kΩ pull-up resistors on SDA/SCL lines if required).*

---

## 💻 Software Stack

| Layer | Component / Library | Source / Notes |
| :--- | :--- | :--- |
| **Firmware** | Arduino C++ (`neuro.ino`) | Main application sketch handling I²C sampling, buffering, and inference calls |
| **Sensor Driver** | `Adafruit_MLX90614` | I²C driver library for reading object temperatures (`readObjectTempC`) |
| **ML Engine** | Edge Impulse C++ SDK | Auto-generated SDK (`Neuro_Frost_inferencing/`) containing DSP and NN routines |
| **Inference Runtime** | TensorFlow Lite for Microcontrollers | Embedded C++ neural network runtime compiled via Edge Impulse EON compiler |
| **DSP Library** | CMSIS-DSP / Spectral Analysis | Performs 16-point FFT and spectral feature extraction on raw input frames |

---

## 🧠 Machine Learning Pipeline

### 1. Data Acquisition & Buffering
* **Sampling Rate:** 2 Hz (1 sample every 500 ms / `EI_CLASSIFIER_INTERVAL_MS`).
* **Input Window:** 8 raw temporal temperature readings (`EI_CLASSIFIER_RAW_SAMPLE_COUNT = 8`).
* **Window Duration:** 4.0 seconds total temporal span.
* **Windowing Mechanism:** **Synchronous Tumbling Window (Fixed Block Buffer)**. The firmware collects 8 sequential samples in a `while` loop with `delay(EI_CLASSIFIER_INTERVAL_MS)` between readings. Once 8 samples are collected, inference is executed, the buffer index resets (`feature_ix = 0`), and acquisition of the next 8-sample block begins.

### 2. DSP Spectral Feature Extraction
The 8 temporal temperature samples are transformed by an Edge Impulse Spectral Analysis DSP block:
* **Analysis Type:** Fast Fourier Transform (FFT, length 16)
* **Features Extracted:** 13 spectral metrics (spectral peaks, power edges, log scale)
* **DSP Output Size:** 13 features (`EI_CLASSIFIER_NN_INPUT_FRAME_SIZE = 13`)

### 3. Neural Network Architecture
The 13 spectral features feed into an `int8` fully-quantized neural network:

```text
Input (13 Spectral Features)
          │
          ▼
Dense Layer 1 (20 Neurons, ReLU, int8)
          │
          ▼
Dense Layer 2 (10 Neurons, ReLU, int8)
          │
          ▼
Output Layer (2 Neurons, Softmax, int8)
          │
          ▼
Classification Labels: ["normal", "stroke"]
```

For full signal processing mechanics and data flow, see [`docs/INFERENCE_PIPELINE.md`](docs/INFERENCE_PIPELINE.md).

### System Workflow Diagram
```text
┌────────────────────────────────┐
│ Data Acquisition (I2C)         │  ──> Polls MLX90614 over I2C every 500 ms
│ Buffers 8 temporal samples     │      Fixed 4.0s tumbling window
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ Spectral DSP Extraction        │  ──> Translates 8 temporal samples
│ (Edge Impulse DSP Block)       │      into 13 spectral features
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ TFLite Micro Inference         │  ──> Executes int8 Dense NN
│ (Local On-Device Execution)    │      Inputs: 13 ──> 20 ──> 10 ──> 2
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ Serial Output                  │  ──> Prints label confidences:
│ (115200 Baud Log)              │      normal: 0.xxxx | stroke: 0.xxxx
└────────────────────────────────┘
```

> `[ System Architecture Diagram Placeholder ]`

---

## 📂 Repository Structure

```plaintext
├── neuro.ino                 # Main Arduino sketch (I2C sampling, buffer management & inference)
├── Neuro_Frost_inferencing/  # Compiled C++ Edge Impulse SDK (TFLite Micro + CMSIS-DSP)
└── docs/                     # Project documentation
    ├── DISCLAIMER.md         # Research & medical disclaimers
    └── INFERENCE_PIPELINE.md # Detailed specification of sampling, DSP, and NN pipeline
```

---

## 🚀 Build Instructions

### Prerequisites
1. **Arduino IDE** (v1.8.x or v2.x).
2. **Adafruit MLX90614 Library** installed via Arduino Library Manager.

### Installation & Flashing
1. **Import Inference Library:**
   - Copy the `Neuro_Frost_inferencing` folder into your Arduino `libraries` folder (e.g., `Documents/Arduino/libraries/`).
2. **Open Sketch:**
   - Open `neuro.ino` in Arduino IDE.
3. **Configure & Upload:**
   - Select your target Arduino-compatible board and COM port under **Tools > Board** and **Tools > Port**.
   - Click **Upload**.
4. **Monitor Output:**
   - Open **Serial Monitor** and set the baud rate to **115200 baud**.

---

## ⚠️ Limitations

1. **Experimental Prototype:** This software is an engineering proof-of-concept and has not undergone clinical trials or medical validation.
2. **Synchronous Tumbling Buffer:** The current `neuro.ino` sketch uses a blocking `delay(500)` loop during data acquisition. It evaluates non-overlapping 8-sample blocks rather than a continuous sliding window.
3. **Environmental Drift:** Infrared thermometry can be affected by ambient room temperature variations and physical sensor placement stability.
4. **Model Dataset Excluded:** The raw training dataset and model generation scripts are not hosted in this repository; only the compiled inference SDK is included.

---

## ⚖️ Disclaimer

This project is an **experimental prototype and research exploration**. It is **NOT** a clinically validated medical diagnostic device. 

For full legal and medical disclaimers, please read [`docs/DISCLAIMER.md`](docs/DISCLAIMER.md).

---

## 📚 References & Acknowledgements

* **[Edge Impulse](https://www.edgeimpulse.com/):** For model training, DSP block generation, and C++ SDK export.
* **[TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers):** On-device neural network runtime engine.
* **[CMSIS-DSP](https://arm-software.github.io/CMSIS_5/DSP/html/index.html):** Arm CMSIS-DSP library for accelerated embedded signal processing.
* **[Adafruit MLX90614 Library](https://github.com/adafruit/Adafruit-MLX90614-Library):** Arduino I²C driver for the MLX90614 IR sensor.

---

## 🏷️ Suggested GitHub Topics

For optimal repository discovery on GitHub, the following topic tags are recommended:
`tinyml`, `edge-impulse`, `tflite-micro`, `embedded-ml`, `mlx90614`, `arduino`, `edge-ai`, `cpp`
