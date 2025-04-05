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

**TO DOWNLOAD THE DATASET**: [here](https://drive.google.com/file/d/1H9rlZc83QO1XdouU2CBCnaLZuJBqjJv-/view?usp=sharing)

We used the **FoR (Fake or Real) Audio Deepfake Detection Dataset**, a publicly available dataset designed for evaluating audio deepfake detection methods. It contains a variety of **real and fake audio samples**, generated using modern speech synthesis and voice conversion techniques.

For faster training and experimentation, we utilized the **2-second clipped version** of the FoR dataset, where:

- Each audio file is exactly **2 seconds long**.
- The dataset is organized into separate folders for **real** and **fake** audio samples.
- The short duration helps speed up **MFCC extraction**, **model training**, and **evaluation** without compromising detection performance for initial benchmarking.

You can find more about the original dataset [here](https://github.com/nii-yamagishilab/Fake-or-Real-Audio-Detection).
---

## 🔍 Research and Selection

During the research phase of this project, various techniques for interpreting and analyzing audio signals were explored. Among the different representations like **raw waveforms**, **spectrograms**, **chroma features**, and **MFCCs (Mel Frequency Cepstral Coefficients)**, **MFCCs stood out as the most effective**. They provide a compact and perceptually-relevant representation of audio, which makes them highly suitable for distinguishing between real and fake speech.

### 📌 Key Insights:
- **MFCCs** effectively capture the timbral texture of speech, which is often subtly distorted in deepfakes.
- **Spectrograms** are visually rich but computationally heavier and less compact.
- **Raw waveforms** require deep models like WaveNet or raw audio transformers, which are resource-intensive.

### 📊 Top Performing Models Explored:
1. **CNN + LSTM:**  
   Combines spatial feature extraction (CNN) with temporal context understanding (LSTM). Achieved high performance in modeling time-series patterns in MFCCs.

2. **DeepSonar:**  
   A BiLSTM-based model that uses frequency-aware features to detect fake audio. Its structure mirrors how humans perceive speech changes.

3. **Quadratic SVM (QSVM):**  
   A lightweight and efficient model using MFCC-based handcrafted features. Offers fast training and decent accuracy (~97%).

4. **CNN on MFCC Images:**  
   Translates MFCCs into visual data, allowing image-based CNNs to classify speech samples with high accuracy.

---

## 🔍 Conclusions

- **CNN+LSTM on MFCC Arrays Achieved the Best Performance**  
  The hybrid CNN+LSTM model trained directly on MFCC arrays achieved the highest accuracy of **99.28%**, effectively capturing both spatial and temporal audio patterns.

- **DeepSonar Performed Exceptionally Well**  
  The DeepSonar model reached **97.64% accuracy**, demonstrating strong capability in detecting deepfakes using bidirectional LSTM layers and speaker verification cues.

- **QSVM Offered Moderate Accuracy but was the fastest**  
  The Quantum Support Vector Machine (QSVM) model achieved **82.41% accuracy**. Though not as powerful as deep learning models, it presents a promising quantum-based alternative.

- **CNN on MFCC Images Showed Lower Performance and longest training duration**  
  The CNN model trained on MFCC spectrogram images resulted in **83.46% accuracy**. It is easier to implement but can lose audio features during image transformation.

- **MFCC Was the Most Reliable Audio Feature**  
  Across all models, MFCCs consistently offered meaningful representations of audio data, proving to be the most effective feature for deepfake detection.


---
## ⚠️ Key Problems Faced

- **Lack of Dataset Variety**  
  The dataset had limited speakers, accents, and noise profiles, which could affect generalization to real-world deepfakes.

- **Time-Consuming Feature Extraction**  
  Extracting and saving MFCC features or spectrogram images for thousands of audio clips was computationally intensive.

- **Unbalanced Labels**  
  Initially, real and fake audio labels were not randomized, which affected training performance until addressed through shuffling.



- **Hyperparameter Tuning Challenges**  
  Finding the optimal combination of layers, units, and dropout rates for each model required multiple iterations and validations.

- **Mainting storage paths**  
  Keeping a track on file directories, during importing saving was initially challenging.
---

## ⚙️ Project Structure

- `main.ipynb`: Main code file containing all the model training and accuracy scores
- `Download the Dataset`: Instructions to download dataset
- `README.md`: All the description about the project

---

## ✅ Results

| Method              | Accuracy |
|---------------------|----------|
| CNN+LTSM on MFCC arrays   | 99.2837%   |
| QSVM                | 82.414%   |
| DeepSonar   | 97.64%   |
| CNN on MFCC Images  | 83.463%    |

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
pip install numpy matplotlib librosa tensorflow scikit-learn
