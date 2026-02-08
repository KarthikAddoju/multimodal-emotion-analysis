# Multimodal Emotion Analysis 🎭🎧

A real-time and offline **multimodal emotion recognition system** that analyzes human emotions by fusing **speech (audio)** and **facial expressions (video)**.  
This project demonstrates the complete AI pipeline — from data preprocessing and model inference to multimodal fusion and real-time deployment.

---

## 🔍 Overview

Human emotions are complex and cannot be reliably inferred from a single modality.  
This project leverages **audio and visual cues together** to improve robustness and accuracy in emotion recognition.

The system:
- Extracts emotions from **speech signals**
- Detects emotions from **facial expressions**
- Combines predictions using a **confidence-weighted late fusion strategy**

---

## ✨ Key Features

- 🎙️ **Speech Emotion Recognition**
  - Uses a pretrained **Wav2Vec2** model fine-tuned for emotion classification
  - Supports offline audio inference

- 📹 **Facial Emotion Recognition**
  - Face detection using OpenCV
  - Modular design to plug in pretrained CNN-based FER models

- 🔗 **Multimodal Late Fusion**
  - Confidence-weighted fusion of audio and video predictions
  - Handles noisy or missing modalities gracefully

- ⚡ **Real-Time Inference**
  - Live microphone input
  - Webcam-based facial analysis
  - Temporal-friendly architecture for future smoothing extensions

---

## 🧠 Architecture

