# Early Brain Stroke Detection via Inner Ear Temperature

An embedded machine learning proof-of-concept exploring the detection of abnormal cerebral blood flow patterns using non-contact infrared thermometry. 

Please read the [Experimental Research Disclaimer](docs/DISCLAIMER.md) before reviewing this repository.

---

## 🔬 Project Overview
This project investigates whether subtle temperature variations in the ear canal—which is anatomically close to major blood vessels supplying the brain—can serve as an early physiological indicator of a stroke. 

Rather than relying on a single static temperature reading, this system acts as a continuous inference engine. It collects a sliding window of thermal data, processes it locally on the microcontroller, and uses a neural network to predict whether the thermal signature matches normal or abnormal profiles.

## ⚙️ Hardware Architecture
*   **Sensor:** MLX90614 Infrared Thermometer
    *   *Why:* Medical-grade accuracy, factory-calibrated, internal DSP for ambient compensation, and an accessible I²C/SMBus interface.
*   **Compute:** Microcontroller capable of running Edge Impulse / TFLite Micro C++ libraries.

## 🧠 Machine Learning Pipeline
The inference pipeline runs entirely on the edge (no cloud connectivity required):
1.  **Data Acquisition:** The MCU reads the object temperature via I²C every 500 milliseconds.
2.  **Windowing:** Samples are buffered into a 4-second temporal window (8 sequential readings).
3.  **DSP & Inference:** The 8-feature array is passed to an embedded TensorFlow Lite model (via the Edge Impulse SDK).
4.  **Classification:** The neural network outputs a confidence probability distribution across the trained labels.

## 📂 Repository Structure
*   `neuro.ino`: Main application firmware handling I²C sensor reads, buffer management, and ML inference triggering.
*   `Neuro_Frost_inferencing/`: The compiled C++ machine learning SDK containing the neural network weights and CMSIS-DSP acceleration libraries.
*   `docs/`: Deep-dive engineering documentation covering IR physics, I²C protocol analysis, and embedded architecture.

## 🚀 Getting Started
*(Deployment instructions to be added)*

---
*Note: The original training dataset and model generation scripts are not included in this repository. This codebase represents the final embedded deployment/inference stage.*
