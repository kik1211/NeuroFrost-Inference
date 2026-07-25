# 🧠 NeuroFrost: Non-Invasive Neurothermal Monitoring System for Early Cerebrovascular Risk Detection

[![Patent Pending](https://img.shields.io/badge/Patent-Pending_IN--2025-blue.svg)](docs/DISCLAIMER.md)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-Edge%20Impulse-orange.svg)](#-machine-learning-architecture)
[![License](https://img.shields.io/badge/Status-Research%20Prototype-green.svg)](docs/DISCLAIMER.md)

> **Proactive, real-time in-ear thermal anomaly detection leveraging edge machine learning for early cerebrovascular stress indication.**

---

## 🔬 Overview & Scientific Rationale
Cerebrovascular accidents (strokes) are a leading cause of permanent disability worldwide. Existing diagnostic modalities like CT and MRI are inherently reactive—conducted only after overt neurological symptoms manifest.

**NeuroFrost** introduces a non-invasive, continuous neuro-thermal monitoring approach based on the physiological relationship between the ear canal and cerebral circulation. Because the tympanic region shares vascular pathways with major cerebral arterial supplies, subtle thermal variations, drift, and rate-of-change patterns can serve as early physiological indicators of altered cerebral blood flow or inflammation.

*Developed at PSG Institute of Technology and Applied Research.*

---

## ⚖️ Patent & Intellectual Property
* **Invention Title:** A Non-Invasive Neurothermal Monitoring System For Early Cerebrovascular Risk Detection
* **Applicant:** PSG Institute of Technology and Applied Research (Coimbatore, India)
* **Status:** Patent Application Filed (Indian Patents Act, 1970)

---

## 🛠️ System Architecture

### 1. 🌡️ Hardware Sensing & Processing
* **In-Ear Sensor:** MLX90614 Non-Contact Infrared Thermometer (Medical-grade accuracy, factory-calibrated, internal DSP).
* **Communication:** SMBus / I²C Bus Protocol (`0x5A` default address) running at 500ms sampling intervals.
* **Compute Platform:** Microcontroller platform (ESP32 / ESP-EYE) executing bare-metal C++ inference.

### 2. 🧠 Machine Learning Architecture
The core analytics engine utilizes an optimized **int8 quantized** neural network generated via Edge Impulse and executed locally via TensorFlow Lite for Microcontrollers:

* **DSP Block:** Spectral Feature Extraction (Translates time-series temperature windows into 13 spectral features).
* **Network Topology:**
  * **Input Layer:** 13 Features
  * **Dense Layer 1:** 20 Neurons
  * **Dense Layer 2:** 10 Neurons
  * **Output Layer:** Softmax Classification (`NORMAL` vs `STROKE`)

### 📊 Validation & Model Performance
* **Validation Accuracy:** `84.0%`
* **Model Loss:** `0.35`
* **F1 Score / AUC:** `0.84`
* **Quantization:** `int8` fully quantized for low-power embedded microcontrollers.

---

## 🔄 End-to-End Operational Workflow
```text
┌────────────────────────────────┐
│ Data Acquisition via MCU       │  ---> Polls MLX90614 over I2C
│ (Collects & buffers readings) │       Interval: 500ms
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ Edge AI Processing             │  ---> Spectral DSP Feature Extraction
│ (Local TFLite Inference)       │       int8 Quantized Neural Network
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ Risk Estimation                │  ---> Calculates continuous confidence
│ (Classifies Thermal Pattern)   │       score & thermal drift
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ Alert & Display Output         │  ---> Local OLED / Wireless Notification
└────────────────────────────────┘
```

## 📂 Repository Layout
```plaintext
├── neuro.ino                 # Main Arduino firmware (I2C reading, buffering & inference)
├── Neuro_Frost_inferencing/  # Compiled C++ Edge Impulse SDK (TFLite Micro + CMSIS-DSP)
└── docs/                     # Research documentation and legal disclaimers
    ├── DISCLAIMER.md         # Experimental research disclaimer
    └── INFERENCE_PIPELINE.md # Deep-dive into time-series windowing mechanics
```
