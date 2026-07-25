# Technical Inference Pipeline Specification

Detailed specification of the data acquisition, signal processing, feature extraction, and neural network inference pipeline. Every parameter cited below is traceable to a specific source file in the repository.

---

## 1. Sensor Data Acquisition

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Sensor | MLX90614 non-contact IR thermometer | `neuro.ino:2,7` |
| Bus protocol | I²C / SMBus, address `0x5A` | Adafruit MLX90614 library default |
| Read function | `mlx.readObjectTempC()` | `neuro.ino:27` |
| Sampling interval | 500 ms (2 Hz) | `EI_CLASSIFIER_INTERVAL_MS = 500` · `model_metadata.h:96` |
| Samples per frame | 1 axis (`temperature`) | `EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME = 1` · `model_metadata.h:89` |
| Buffer length | 8 samples | `EI_CLASSIFIER_RAW_SAMPLE_COUNT = 8` · `model_metadata.h:88` |
| DSP input frame size | 8 floats (8 × 1) | `EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE` · `model_metadata.h:90` |
| Window duration | 4.0 seconds | 8 samples × 0.5 s |

### Buffering Mechanism

The firmware uses a **synchronous tumbling window**:

1. A `while` loop in `loop()` calls `mlx.readObjectTempC()` and stores the result in `features[feature_ix]`.
2. After each sample, `delay(EI_CLASSIFIER_INTERVAL_MS)` blocks for 500 ms.
3. When `feature_ix` reaches `EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE` (8), the loop exits and inference begins.
4. After inference completes and results are printed, `feature_ix` is reset to 0.
5. The next `loop()` iteration begins filling a fresh 8-sample block.

There is no overlap between consecutive windows. Each block is independent.

---

## 2. DSP Feature Extraction

The 8-sample buffer is wrapped in an Edge Impulse signal structure via `numpy::signal_from_buffer()` and passed to `run_classifier()`, which internally invokes the spectral analysis DSP block.

### DSP Configuration

All parameters are defined in the `ei_dsp_config_spectral_analysis_t` struct in `model_variables.h:54-73`:

| Parameter | Value | Field |
| :--- | :--- | :--- |
| Analysis type | FFT | `analysis-type` · line 63 |
| FFT length | 16 | `fft-length` · line 64 |
| Spectral peaks count | 3 | `spectral-peaks-count` · line 65 |
| Spectral peaks threshold | 0.1 | `spectral-peaks-threshold` · line 66 |
| Power edge frequencies | 0.1, 0.5, 1.0, 2.0, 5.0 Hz | `spectral-power-edges` · line 67 |
| Log scaling | Enabled | `do-log` · line 68 |
| FFT overlap | Enabled | `do-fft-overlap` · line 69 |
| Filter type | None | `filter-type` · line 60 |
| Input decimation ratio | 1 | `input-decimation-ratio` · line 59 |

### Output

The DSP block produces **13 spectral features** (`EI_CLASSIFIER_NN_INPUT_FRAME_SIZE = 13`, `model_metadata.h:87`; DSP output size = 13, `model_variables.h:79`).

```text
  8 raw temperature samples (float32, 2 Hz)
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │  Spectral Analysis DSP Block        │
  │  FFT(16) + 3 peaks + 5 power edges │
  │  + log scaling                      │
  └─────────────────┬───────────────────┘
                    │
                    ▼
         13 spectral features (float32)
```

---

## 3. Neural Network Architecture

The 13 features are evaluated by an EON-compiled TensorFlow Lite for Microcontrollers model. The model graph is defined in `tflite_learn_768102_9_compiled.cpp`.

### Layer Topology

| # | Operation | Input Tensor | Output Tensor | Weights | Activation | Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Fully Connected | (1,13) → tensor 0 | (1,20) → tensor 7 | tensor 6: (20,13) int8 = 260 B | ReLU | `opdata0` · line 234 |
| 1 | Fully Connected | (1,20) → tensor 7 | (1,10) → tensor 8 | tensor 4: (10,20) int8 = 200 B | ReLU | `opdata1` · line 237 |
| 2 | Fully Connected | (1,10) → tensor 8 | (1,2) → tensor 9 | tensor 2: (2,10) int8 = 20 B | None | `opdata2` · line 240 |
| 3 | Softmax | (1,2) → tensor 9 | (1,2) → tensor 10 | — | Softmax | `opdata3` · line 243 |

### Quantization

| Property | Value | Source |
| :--- | :--- | :--- |
| Input data type | int8 | `EI_CLASSIFIER_TFLITE_INPUT_DATATYPE = EI_CLASSIFIER_DATATYPE_INT8` · `model_metadata.h:109` |
| Output data type | int8 | `EI_CLASSIFIER_TFLITE_OUTPUT_DATATYPE = EI_CLASSIFIER_DATATYPE_INT8` · `model_metadata.h:110` |
| Quantization enabled | Yes | `EI_CLASSIFIER_QUANTIZATION_ENABLED = 1` · `model_metadata.h:118` |
| Tensor arena (default) | 368 bytes | `kTensorArenaSize = 368` · `tflite_learn_768102_9_compiled.cpp:99` |

### Classification Labels

Defined in `model_variables.h:50`:

```c
const char* ei_classifier_inferencing_categories_768102_1[] = { "normal", "stroke" };
```

| Index | Label |
| :--- | :--- |
| 0 | `normal` |
| 1 | `stroke` |

---

## 4. Firmware Execution Flow

```text
┌──────────────────────────────────────┐
│ setup()                              │
│  Serial.begin(115200)                │
│  Wire.begin()                        │
│  mlx.begin()                         │  ← halts (while(1)) on failure
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ loop()                               │
│                                      │
│  while (feature_ix < 8):             │
│    features[ix] = readObjectTempC()  │
│    Serial.print(sample)              │
│    feature_ix++                      │
│    delay(500)                        │  ← blocks 500 ms per sample
│                                      │
│  signal_from_buffer(features, 8)     │
│  run_classifier(&signal, &result)    │  ← DSP + NN inference
│                                      │
│  for each label:                     │
│    Serial.print(label: confidence)   │
│                                      │
│  feature_ix = 0                      │  ← reset for next window
└──────────────────────────────────────┘
```

### Timing

| Phase | Duration | Notes |
| :--- | :--- | :--- |
| Data acquisition | ~4.0 s | 8 × delay(500 ms); actual period slightly > 500 ms due to I²C read + Serial print overhead |
| DSP + Inference | Platform-dependent | Typically < 50 ms on Cortex-M4 class MCUs |
| Total cycle | ~4.0 s + inference | Continuous loop, no idle period between windows |
