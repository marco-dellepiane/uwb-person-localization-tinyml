# 📡 TinyML: Multi-Person UWB Radar Localization

An Embedded Edge AI project focused on localizing multiple subjects (up to 4) inside a room using Ultra-Wideband (UWB) radar sensors. The system is designed to run inference directly on severely constrained microcontrollers (**ESP32-S3**), processing raw Channel Impulse Response (CIR) data.

## 🚀 Key Features & Innovations
- **Permutation Invariant Training (Hungarian Loss):** Resolves the target assignment problem dynamically across frames by calculating the optimal geometric match among all 24 possible permutations.
- **Hardware-Aware Architecture:** An optimized multi-head CNN (EEAI-Net) specifically designed to stay well below the ESP32-S3 limits, utilizing *Residual Reduction Modules (RRM)*, *Depthwise Separable Convolutions*, and *Squeeze-and-Excitation*.
- **Advanced Data Pipeline:** Implements Exponential Moving Average (EMA) for static clutter removal, Normalization, and Radar Dropout for robust data augmentation.
- **Full INT8 Quantization:** The final `.tflite` model relies strictly on INT8 operations for both weights and activations to maximize speed and minimize memory footprint.

## 💻 Hardware Constraints (ESP32-S3)
The model has been architected to strictly adhere to the following memory budgets (assuming no external PSRAM):
- **Flash Memory (Model Size):** < 800 KB
- **SRAM (Activation Arena):** < 300 KB
- **Forbidden Operations:** No LSTM, GRU, RNN, CUSTOM, or Flex ops (Not supported by TFLM).

## 📡 Dataset Configuration
- **Sensor:** TSRR250 Ultra-Wideband radar (6 radars positioned around the room perimeter).
- **Antennas:** 3 per radar (providing angular diversity).
- **Range Bins:** 120 bins.
- **Sampling Rate:** 25 Hz.
- **Room Dimensions:** 4.8 m × 7.2 m.
- **Data Format:** `(T, 6, 3, 120, 2)` representing `T frames × 6 radars × 3 antennas × 120 bins × 2 (I, Q)`.

## 📁 Repository Structure
```text
.
├── evaluation/                 # Evaluation scripts (F1-score, RMSE, constraints checker)
├── submission/                 # Final deliverables
│   ├── code.py                 # TFLite inference script
│   └── model.tflite            # The fully quantized INT8 model
├── other_files/                # Jupyter Notebooks containing the training evolution and visualizations
└── requirements.txt            # Python dependencies
```

## 🛠️ Usage & Evaluation

### 1. Run Hardware Constraints Check
To verify that the model fits the ESP32-S3 constraints:
```bash
python evaluation/evaluate_constraint.py --model-path submission/model.tflite
```

### 2. Run Inference
To execute the TFLite model on a `.npz` radar window and output the tracking sequence in `.jsonl`:
```bash
python submission/code.py --input-path path/to/input.npz --output-path output.jsonl
```

### 3. Evaluate Performance
To compare the model's predictions against the Ground Truth (computing F1-Score, RMSE, Precision, and Recall):
```bash
python evaluation/evaluate_performance.py --gt-path path/to/gt.jsonl --pred-path output.jsonl
```

## 📊 Evaluation Metrics
The primary evaluation metric is the **F1-Score**, calculated using a Hungarian matching algorithm per frame. A match is considered a True Positive (TP) only if the predicted position is within **1.0 m** of the ground-truth position.