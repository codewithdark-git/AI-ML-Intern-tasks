# Breast Cancer Classification

This project focuses on classifying breast cancer as either benign or malignant using machine learning. The model is trained on histopathological images of breast cancer cells and uses deep learning techniques for image classification.

## Table of Contents

- [Breast Cancer Classification](#breast-cancer-classification)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dependencies](#dependencies)
  - [Data](#data)
  - [Model](#model)
  - [Training](#training)
    - [Hyperparameters:](#hyperparameters)
  - [Evaluation](#evaluation)
    - [Performance Metrics:](#performance-metrics)
  - [Results](#results)
  - [License](#license)

## Overview

The goal of this project is to build a machine learning model that can predict whether a given breast cancer tumor is benign or malignant based on histopathological image data. We employ Convolutional Neural Networks (CNNs) to perform image classification, and the model is trained using the well-known breast cancer dataset.



## Dependencies

The following Python libraries are required to run the project:

- `torch` (PyTorch) - for deep learning model building and training
- `torchvision` - for image transformation and pre-processing
- `numpy` - for numerical operations
- `matplotlib` - for plotting and visualizations
- `pandas` - for data manipulation
- `scikit-learn` - for performance metrics and evaluation

Install the dependencies by running:

```
pip install -r requirements.txt
```

## Data

The dataset used for training the model is the [Breast Cancer Histopathological Database](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images), which contains labeled images of breast cancer cells. Each image is labeled as either **benign** or **malignant**.

You can place the dataset in the `/data` folder of the project.

## Model

This project uses a Convolutional Neural Network (CNN) for classifying breast cancer images. The CNN model includes several convolutional layers followed by pooling and fully connected layers for prediction. The model is trained using the Adam optimizer and categorical cross-entropy loss function.

The model architecture is as follows:
1. Convolutional layers for feature extraction
2. Max-pooling layers to reduce dimensionality
3. Fully connected layers for classification
4. Output layer with a softmax activation function for binary classification (benign or malignant)

## Training

To train the model, run the following script:

```
python scripts/train.py
```

This will load the dataset, preprocess the images, train the model, and save the trained model to the `/models` folder.

### Hyperparameters:
- **Batch size**: 32
- **Epochs**: 20
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Loss function**: Cross-entropy loss

## Evaluation

To evaluate the model, you can run the following script:

```
python scripts/evaluate.py
```

This will load the trained model, run it on the test dataset, and output evaluation metrics such as accuracy, precision, recall, F1 score, and confusion matrix.

### Performance Metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

## Results

After training and evaluation, you can visualize the results of the model's performance. The accuracy, loss, and other evaluation metrics can be found in the `/results` folder.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


