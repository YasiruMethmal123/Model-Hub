# Handwritten Digit Classification - Optimized SVM Model

A high-performance machine learning pipeline for classifying handwritten digits using a Support Vector Machine (SVM) combined with Principal Component Analysis (PCA) for dimensionality reduction.

## Overview

This project implements a multi-class classification model to identify handwritten digits (0-9). The pipeline uses an RBF-kernel SVM optimized via Grid Search. To ensure production-level efficiency and handle the computational complexity of SVMs, the model utilizes PCA to compress the feature space without losing critical structural information, significantly reducing inference time.

## Model Performance

The optimized pipeline achieves the following metrics on the 2,000-sample test set:

* **Accuracy**: 95.65%
* **F1 Score (Macro)**: 0.96
* **Precision (Macro)**: 0.96
* **Recall (Macro)**: 0.96
* **Inference Time**: 3.23 seconds

### Performance Highlights

* **Optimal Hyperparameters**: `C=10`, `gamma='scale'`, `kernel='rbf'`.
* **Speed Optimization**: By implementing PCA, inference time was reduced from 11.71 seconds (raw features) to 3.23 seconds, making the model much more viable for production use.
* **Improved Edge Cases**: The RBF kernel successfully resolved linear blind spots, effectively distinguishing visually similar digits like 4s vs. 9s and 3s vs. 8s.

## Dataset

**MNIST Database of Handwritten Digits** - Contains pre-centered, grayscale images of handwritten digits.

### Dataset Features:

* **Raw Format**: 28x28 pixel 2D arrays.
* **Engineered Features**: Flattened into 1D arrays of 784 features per image.
* **Values**: Scaled from 0-255 down to 0.0-1.0 to ensure proper distance calculations for the SVM.
* **Class**: Target variable (0 through 9).

### Data Characteristics:

To handle the $O(n^3)$ training time complexity inherent to SVMs, a representative subset of the data was used for this specific model:

* **Training Set**: 10,000 samples
* **Test Set**: 2,000 samples

## Technologies Used

* **Python 3.x**
* **scikit-learn** - Machine learning algorithms (SVC), PCA, and GridSearchCV
* **numpy** - Numerical matrix operations
* **matplotlib & seaborn** - Visualization of confusion matrices
* **joblib** - Efficient model serialization
* **uv** - Fast Python package and environment manager

## Methodology

### 1. Data Preprocessing

* Flattened the 28x28 images into 784-length vectors.
* Normalized pixel intensities by dividing by 255.0.

### 2. Dimensionality Reduction (PCA)

* Applied Principal Component Analysis (PCA) to retain 95% of the variance.
* Stripped empty background noise, reducing the feature space from 784 dimensions to approximately 150 dimensions.

### 3. Hyperparameter Tuning

* Built a Scikit-Learn `Pipeline` linking PCA directly to the SVM to prevent data leakage.
* Executed `GridSearchCV` with 3-fold cross-validation to test combinations of `C` (regularization) and `gamma` (kernel coefficient).

### 4. Model Evaluation

* Evaluated the best estimator on the holdout test set.
* Generated a classification report and visualized the errors using a seaborn heatmap (Confusion Matrix).

## Getting Started

### Prerequisites

```bash
uv pip install numpy pandas scikit-learn matplotlib seaborn joblib

```

### Loading and Running the Model

1. **Load the saved pipeline**:

```python
import joblib
import numpy as np

# Load the PCA+SVM pipeline
model_path = "../models/optimized_svm_pipeline.joblib"
pipeline = joblib.load(model_path)

```

2. **Preprocess new images and predict**:

```python
# Assuming 'new_image' is a 28x28 numpy array
# 1. Flatten
flat_image = new_image.reshape(1, -1)

# 2. Scale
scaled_image = flat_image.astype('float32') / 255.0

# 3. Predict (The pipeline automatically applies PCA before the SVM)
prediction = pipeline.predict(scaled_image)
print(f"Predicted Digit: {prediction[0]}")

```

## Model Architecture

```text
Input Features (784 Pixels: 0.0 - 1.0)
        ↓
Principal Component Analysis (PCA)
    - Retains 95% variance
    - Compresses to ~150 features
        ↓
Support Vector Machine (SVC)
    - kernel: 'rbf'
    - C: 10
    - gamma: 'scale'
        ↓
Predictions (Digits 0-9)

```

## Key Insights

1. **Non-Linearity is Required**: The jump from a Linear kernel (89.75%) to an RBF kernel (95.65%) proved that handwritten digits are not linearly separable in their raw pixel space.
2. **Efficiency vs. Accuracy**: PCA is an essential step for traditional machine learning on image data. It provided a ~70% reduction in prediction latency without sacrificing accuracy.
3. **Regularization Matters**: The Grid Search selected a higher `C` value (10), indicating that strictly penalizing misclassifications during training helped the model draw tighter, more accurate boundaries.

## Future Improvements

* [ ] Test polynomial kernels (`kernel='poly'`) to see if specific degree functions map the digit curves better.
* [ ] Expand the training subset from 10k to the full 60k dataset using a cloud compute instance to see the absolute upper limit of SVM accuracy.
* [ ] Compare this baseline against ensemble methods (Random Forest) and Deep Learning architectures (CNN).
