# 🎙️ Audio Deepfake Detection

This project focuses on detecting audio deepfakes and differentiate them from real audio using machine learning and deep learning models, leveraging MFCC features and CNN architectures. Audio deepfakes pose significant societal risks including misinformation, scams, and impersonation threats.

---

## 📌 Overview

Deepfake audio is synthetically generated using techniques like **Text-to-Speech (TTS)** and **Voice Conversion**, often powered by **GANs** (Generative Adversarial Networks) and **VAEs** (Variational Autoencoders). This project combines **signal processing**, **machine learning**, and **deep learning** to detect such manipulations with high accuracy.

---
## 🛡️ Requirement of Deepfake Detection

1. **Prevent Misinformation:** Avoid spread of fake news and manipulated speeches.
2. **Avoid Scams:** Stop fraud calls that impersonate trusted individuals.
3. **Protect Media Integrity:** Preserve trust in audio and video content.
4. **Secure Voice Authentication:** Prevent misuse in voice-based security systems.
5. **Support Legal Verification:** Ensure authenticity of audio evidence in legal settings.

---

## 📂 Dataset Used

We used the **FoR (Fake or Real) Audio Deepfake Detection Dataset**, a publicly available dataset designed for evaluating audio deepfake detection methods. It contains a variety of **real and fake audio samples**, generated using modern speech synthesis and voice conversion techniques.

For faster training and experimentation, we utilized the **2-second clipped version** of the FoR dataset, where:

- Each audio file is exactly **2 seconds long**.
- The dataset is organized into separate folders for **real** and **fake** audio samples.
- The short duration helps speed up **MFCC extraction**, **model training**, and **evaluation** without compromising detection performance for initial benchmarking.

You can find more about the original dataset [here](https://github.com/nii-yamagishilab/Fake-or-Real-Audio-Detection).


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
