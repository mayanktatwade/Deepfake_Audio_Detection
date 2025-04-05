# 🎙️ Audio Deepfake Detection

This project focuses on detecting audio deepfakes using machine learning and deep learning models, leveraging MFCC features and CNN architectures. Audio deepfakes pose significant societal risks including misinformation, scams, and impersonation threats.

---

## 📌 Overview

Deepfake audio is synthetically generated using techniques like **Text-to-Speech (TTS)** and **Voice Conversion**, often powered by **GANs** (Generative Adversarial Networks) and **VAEs** (Variational Autoencoders). This project combines **signal processing**, **machine learning**, and **deep learning** to detect such manipulations with high accuracy.

---

## 🧠 Techniques Used

### 1. **Handcrafted Feature-Based Methods**
- Extracted **MFCCs (Mel-Frequency Cepstral Coefficients)** from audio samples.
- Converted MFCCs into 2D images (spectrogram-like) and trained models on these.

### 2. **Machine Learning Classifier**
- Implemented a **QSVM (Quadratic Support Vector Machine)** model.
- Achieved accuracy: **97.56%**

### 3. **Deep Learning with CNN**
- Designed a custom CNN model trained on MFCC image plots.
- Achieved accuracy: **98.5%**

---

## 🔬 Key Research Conclusions

- Combining **signal processing**, **machine learning**, and **perceptual insights** yields robust performance.
- State-of-the-art models like **DeepSonar (98.10%)** and **QSVM (97.56%)** show excellent results on large datasets.
- Audio deepfakes are complex due to:
  - Variability in **languages, accents, tones, and emotions**
  - Need for **large, clean datasets**
  - **Background noise** interference

---

## ⚙️ Project Structure

- `mfcc_extraction.py`: Extracts MFCCs from real/fake audio samples and saves them as `.npy` arrays.
- `mfcc_training_model.py`: Trains a model using raw MFCC arrays.
- `mfcc_image_training.py`: Generates MFCC images, trains CNN model directly on image batches.
- `mfcc_features.npy`, `labels.npy`: Processed data for training/testing.

---

## ✅ Results

| Method              | Accuracy |
|---------------------|----------|
| QSVM                | 97.56%   |
| DeepSonar (paper)   | 98.10%   |
| CNN on MFCC Images  | 98.5%    |

---

## 🚨 Societal Risks

1. **Fake News Generation**
2. **Voice-based Scams or Frauds**
3. **Deepfake Calls for Misinformation**

---

## 🛠️ Requirements

- Python 3.8+
- TensorFlow
- Librosa
- NumPy
- Matplotlib
- scikit-learn (for SVM)

Install via:

```bash
pip install -r requirements.txt
