# SVM_Model

---

# HW3 - Iris Dataset Classification

This project compares the performance of **Support Vector Machine (SVM)** and **Multilayer Perceptron (MLP, using PyTorch)** on the classic Iris flower dataset. The workflow includes data preprocessing, model training, performance evaluation, and visualization of decision boundaries.

## Project Overview

The Jupyter Notebook (`HW3.ipynb`) performs the following main steps:

1. **Data Loading & Preprocessing**:
* Loads the Iris dataset from `sklearn`.
* Applies `StandardScaler` for feature standardization.
* Splits the data into an 80% training set and a 20% testing set.


2. **SVM Model (Support Vector Machine)**:
* Trains an SVM classifier using the **RBF kernel** (Radial Basis Function).
* Evaluates training accuracy, testing accuracy, and execution time.
* Analyzes **SVM Classification Confidence** using the decision function.


3. **Dimensionality Reduction & Visualization (PCA)**:
* Reduces the 4-dimensional features to 2 dimensions using **Principal Component Analysis (PCA)**.
* Visualizes the SVM decision boundaries in the 2D plane.


4. **MLP Model (Multilayer Perceptron)**:
* Builds a neural network using **PyTorch**.
* **Architecture**: Input Layer (2 PCA features)  Hidden Layer (16 neurons, ReLU)  Output Layer (3 classes).
* Visualizes the MLP decision boundaries on the PCA-reduced data.


5. **Results Comparison**:
* Compares the accuracy and runtime of SVM and MLP.



## Requirements

To run this project, you need a Python 3 environment with the following packages installed:

* numpy
* pandas
* matplotlib
* scikit-learn
* torch (PyTorch)

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib scikit-learn torch

```

## Results Summary

Based on the execution logs in the Notebook, the performance of the models on the Iris dataset is as follows:

| Model | Training Accuracy | Testing Accuracy | Runtime (seconds) |
| --- | --- | --- | --- |
| **SVM** | 97.50% | 100.0% | ~0.03 s |
| **MLP** | 98.33% | 100.0% | ~0.59 s |

*Note: The MLP runtime includes the training process over 200 epochs.*

## File Structure

* `HW3.ipynb`: The main Jupyter Notebook containing the complete code, model training process, and visualization charts.
