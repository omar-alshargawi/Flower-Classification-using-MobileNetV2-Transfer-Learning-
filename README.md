# 🌸 Flower Classification using MobileNetV2 (Transfer Learning)

## 📌 Project Objective
The goal of this project is to build a deep learning model capable of accurately classifying flower images into five categories. Instead of training a Convolutional Neural Network (CNN) from scratch, I leveraged **Transfer Learning** with **MobileNetV2** to achieve high accuracy with significantly reduced training time and better generalization.

---

## 📊 Dataset Overview
The model was trained on a dataset containing **3,670 images** categorized into 5 distinct classes:
* **Classes:** Roses, Dandelion, Tulips, Sunflowers, Daisy.
* **Training Set:** 2,569 images.
* **Validation Set:** 1,101 images.

---

## 🧠 Model Architecture & Strategy
I utilized **MobileNetV2** (pretrained on ImageNet) for its efficiency and high performance on mobile/embedded vision tasks.

1.  **Phase 1: Feature Extraction** – The base layers were frozen to preserve the pretrained weights while training a custom classification head.
2.  **Phase 2: Fine-Tuning** – Specific layers were unthawed to fine-tune the model on the unique features of the flower dataset.
3.  **Final Layers:** Added a Global Average Pooling layer followed by a Dense layer with **Softmax activation** for multi-class prediction.

---

## 📈 Training Performance
The model was trained for 6 epochs, showing a strong convergence between training and validation accuracy.

<img width="848" alt="Training Curves" src="https://github.com/user-attachments/assets/135046e9-b77d-4eb1-a516-091e22c4c532" />

### Final Metrics:
* **Training Accuracy:** ~91%
* **Validation Accuracy:** ~85.5%
* **Validation Loss:** ~0.42

---

## 🖼️ Sample Predictions
Below are some visual results from the test set. The model shows high confidence in identifying the correct flower species, even with complex backgrounds.

<img width="925" alt="Sample Predictions" src="https://github.com/user-attachments/assets/407eeb23-9baf-4230-9881-11d5d376cd53" />

> **Note:** Some misclassifications occur between visually similar classes (e.g., specific colors of Roses vs. Tulips), which is a common challenge in fine-grained visual categorization.

---

## 🛠️ Tools & Technologies
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Model:** MobileNetV2 (Transfer Learning)
* **Data Handling:** NumPy, Pandas
* **Visualization:** Matplotlib

## 💡 Key Learnings
* Implementing **Transfer Learning** to save computational resources.
* Distinguishing between **feature extraction** and **fine-tuning** workflows.
* Monitoring and interpreting **overfitting** through loss/accuracy curves.
* Optimizing multi-class classification using Softmax.

---
📅 **Analysis Date:** February 2026 | 🛠️ **Technique:** Transfer Learning (MobileNetV2)

👤 **Author:** Omar Alshargawi
