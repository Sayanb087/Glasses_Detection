# 👓 Glasses Detection Using CNN

## 🌐 Project Overview

This project involves building a deep learning model to automatically detect whether a person is wearing glasses or not from facial images. Such systems can be useful in biometric authentication, access control, surveillance, and social media filters. The model is built using a Convolutional Neural Network (CNN) architecture, implemented with TensorFlow and Keras, two popular libraries for machine learning and deep learning.

## 🔎 Problem Statement

Glasses detection is a binary image classification problem where each image is categorized into one of the two classes:
glasses: The person in the image is wearing eyeglasses.
no_glasses: The person is not wearing eyeglasses.
This task can be challenging due to variations in lighting conditions, face orientation, image resolution, and the type of glasses.

## 📅 Dataset Description

The dataset used consists of two folders:
glasses/: Contains images of people wearing glasses.
no_glasses/: Contains images of people without glasses.
The images are of varying sizes and conditions, hence preprocessing is required before feeding them to the model. Each image is resized to 128x128 pixels.

## 🧐 What is CNN (Convolutional Neural Network)?

CNN is a class of deep neural networks most commonly applied to analyzing visual imagery. It is especially effective for tasks like image classification, object detection, and facial recognition. Key components of CNNs include:
Convolutional layers: Apply filters to extract spatial features.
Pooling layers: Downsample feature maps to reduce dimensionality.
Fully connected layers: Learn complex patterns and perform classification.

## 💡 About TensorFlow and Keras

TensorFlow is an open-source end-to-end machine learning platform developed by Google.
Keras is a high-level API built on top of TensorFlow for building and training neural networks more easily.
We used tensorflow.keras for defining and training the CNN model.

## 🔢 Data Preprocessing

Preprocessing is critical for improving the model's performance:
Resizing: All images are resized to 128x128 pixels.
Normalization: Pixel values are scaled to a [0, 1] range.
One-hot encoding: The labels are encoded into two-element vectors for categorical classification.

## 🌈 Image Augmentation

Image augmentation increases the diversity of the training data by applying transformations such as:
* Rotation
* Zooming
* Horizontal flipping
* Shifting
This helps prevent overfitting and improves the model's generalization to unseen data.

## 🎓 Model Architecture

### 🔹 Layer Details
  Conv2D: Detects features such as edges and textures.
  MaxPooling2D: Reduces dimensionality while preserving key features.
  Flatten: Converts the 3D feature map into a 1D vector.
  Dense: Fully connected layer for decision making.
  Dropout: Prevents overfitting by randomly deactivating neurons during training.

### 🔹 Activation Functions
  ReLU (Rectified Linear Unit): Introduces non-linearity.
  Softmax: Converts output scores into class probabilities.
### 🔹 Optimizer
  Adam: An adaptive optimizer that combines the benefits of RMSprop and SGD with momentum.
### 🔹 Loss Function
  Categorical Crossentropy: Suitable for multi-class classification with one-hot encoded labels.

## 🏋️‍⚖️ Model Training

* Batch Size: 16 (number of images processed together in a batch).
* Epochs: Multiple passes through the dataset.
* Validation Split: A portion of data reserved for model validation.
The model is trained using augmented data to improve robustness and reduce overfitting.

## 🔮 Evaluation Metrics

### ✅ Accuracy

The ratio of correctly predicted images to the total number of images.
### 👁️ Confusion Matrix
A matrix showing actual vs. predicted class counts:
* True Positive (TP): Correctly predicted glasses.
* True Negative (TN): Correctly predicted no_glasses.
* False Positive (FP): Incorrectly predicted glasses.
* False Negative (FN): Incorrectly predicted no_glasses.

Useful for calculating:
* Precision = TP / (TP + FP)
* Recall = TP / (TP + FN)
* F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

## 📊 Classification Report

Displays precision, recall, f1-score, and support for each class.
### 🌐 Visualization
* Training/Validation Accuracy and Loss graphs.
* Confusion matrix heatmap.

### 🤖 Predictions
* Sample predictions are made on test images.
* Results are visualized with actual and predicted labels.
* Helps evaluate model generalization.

## 🌟 Final Observations

* The model performs well in classifying glasses vs. no_glasses.
* Image augmentation improves accuracy and robustness.
* Dropout helps reduce overfitting.

## 🚀 Future Improvements

* Use transfer learning with pre-trained models (like MobileNet or ResNet).
* Real-time glasses detection using a webcam.
* Extend to detect sunglasses, transparent glasses, and different shapes.

## 📃 Summary

This project demonstrates how a CNN can be effectively used for binary image classification. By leveraging TensorFlow/Keras and applying proper image preprocessing and augmentation techniques, the model can distinguish glasses with high accuracy. The complete pipeline from loading images to evaluating performance is implemented and visualized for end-to-end understanding.



