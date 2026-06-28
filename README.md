---
title: CodeAlpha Character Recognition
emoji: 📝
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---

# Task 3: Character Recognition from Images

🎉 **Live Interactive Demo:** [CodeAlpha Character Recognition App on Hugging Face](https://huggingface.co/spaces/rexrevanth/CodeAlpha-Character-Recognition)

This folder contains the solution for the Character Recognition from Images task.

## Objective
Identify handwritten characters or alphabets using deep learning.

## Features & Uniqueness
- **Extended Dataset**: Instead of just using the standard MNIST (10 digits), we use the **EMNIST Balanced** dataset via `tensorflow_datasets`, which contains 47 classes (digits, uppercase, and lowercase letters).
- **Robust Model**: A Convolutional Neural Network (CNN) built with **Data Augmentation** layers (Random Rotation and Zoom) to ensure the model generalizes well to different handwriting styles.
- **Explainable AI (Grad-CAM)**: We implement Gradient-weighted Class Activation Mapping (Grad-CAM) to generate heatmaps that highlight which parts of the drawn character the model focused on to make its prediction.
- **Interactive Sketch App**: We use **Gradio** to provide an interactive canvas where you can draw a letter or digit with your mouse, and the model will instantly predict it while displaying the Grad-CAM heatmap overlay!

## How to Run

### Step 1: Train the Model
Run the training script. This will download the EMNIST dataset (around 500MB), build the CNN, apply data augmentation, and train it. The best model will be saved as `emnist_model.h5`.
```bash
python train_model.py
```

### Step 2: Launch the Drawing App
Once training is complete, start the interactive web application:
```bash
python app.py
```
This will open a local web server in your browser. Use the sketchpad to draw a character, click "Predict Character", and see the results!
