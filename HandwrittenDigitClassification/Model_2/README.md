# Handwritten Digit Classification - Optimized Random Forest

A high-performance ensemble machine learning model for classifying handwritten digits using a Random Forest Classifier. This implementation focuses on parallel processing, computational efficiency, and hyperparameter tuning at scale.

## Overview

This project implements a multi-class classification model to identify handwritten digits (0-9) from the MNIST dataset. By leveraging a Random Forest—an ensemble of decision trees—this model effortlessly scales to the full 60,000-image training dataset without the need for dimensionality reduction (PCA), achieving high accuracy and lightning-fast inference times.

## Model Performance

The tuned Random Forest model achieves the following metrics on the full 10,000-sample test set:

* **Accuracy**: ~97.04%+
* **F1 Score (Macro)**: ~0.97
* **Inference Time**: 0.40 seconds (for 10,000 images)
* **Training Time**: ~48 seconds (for 60,000 images)

### Performance Highlights

* **Blazing Fast Inference**: The model predicts 10,000 images in under half a second, making it roughly 40x faster at inference than the optimized SVM pipeline.
* **Highly Parallelizable**: Utilizing `n_jobs=-1`, the model distributes tree building across all available CPU cores, drastically reducing training time.
* **Robust to Noise**: Naturally handles uninformative background pixels without requiring explicit feature selection or PCA.

## Dataset

**MNIST Database of Handwritten Digits** - Contains pre-centered, grayscale images of handwritten digits.

### Dataset Features:

* **Data Size**: Full dataset utilized (60,000 training samples, 10,000 test samples).
* **Format**: 28x28 pixel images flattened into 1D arrays of 784 features.
* **Scaling**: Pixel intensities normalized to a 0.0-1.0 range.
* **Class**: Target variable containing 10 classes (digits 0 through 9).

## Technologies Used

* **Python 3.x**
* **scikit-learn** - Machine learning algorithm (`RandomForestClassifier`), hyperparameter tuning (`RandomizedSearchCV`)
* **numpy & pandas** - Data matrix manipulation and evaluation
* **matplotlib & seaborn** - Visualization of confusion matrices
* **joblib** - Efficient model serialization

## Methodology

### 1. Data Preprocessing

* Flattened the 28x28 images into 784-length vectors.
* Scaled features by dividing by 255.0.
* *Note: Unlike the SVM implementation, PCA was omitted as Random Forests naturally perform feature selection during node splitting.*

### 2. Baseline Establishment

* Trained an initial ensemble of 100 trees to establish a baseline accuracy of 97.04%.

### 3. Hyperparameter Tuning at Scale

* Implemented `RandomizedSearchCV` to efficiently search the hyperparameter space without the massive compute cost of an exhaustive Grid Search.
* **Parameters Tuned**: `n_estimators` (number of trees), `max_depth` (tree depth limit), and `min_samples_split`.
* Utilized 3-fold cross-validation to ensure model generalizability.

### 4. Evaluation

* Generated a full classification report and confusion matrix to identify edge-case misclassifications (e.g., 4s vs. 9s, 7s vs. 2s).

## Getting Started

### Prerequisites

```bash
uv pip install numpy pandas scikit-learn matplotlib seaborn joblib

```

### Loading and Running the Model

1. **Load the saved model**:

```python
import joblib
import numpy as np

# Load the optimized Random Forest model
model_path = "../models/optimized_random_forest.joblib"
rf_model = joblib.load(model_path)

```

2. **Preprocess new images and predict**:

```python
# Assuming 'new_image' is a 28x28 numpy array
# 1. Flatten
flat_image = new_image.reshape(1, -1)

# 2. Scale
scaled_image = flat_image.astype('float32') / 255.0

# 3. Predict
prediction = rf_model.predict(scaled_image)
print(f"Predicted Digit: {prediction[0]}")

```

## Model Architecture

```text
Input Features (784 Pixels: 0.0 - 1.0)
        ↓
Random Forest Classifier Ensemble
    - n_estimators: [Optimized]
    - max_depth: [Optimized]
    - min_samples_split: [Optimized]
    - Bootstrapping: True
    - CPU Cores: All (n_jobs=-1)
        ↓
Majority Vote Aggregation
        ↓
Predictions (Digits 0-9)

```

## Key Insights

1. **The Power of Ensembles**: By aggregating the predictions of hundreds of independent decision trees, the model overcame the high-variance (overfitting) problem inherent to single decision trees.
2. **Speed is King**: The $O(n \log n)$ training complexity of trees proved vastly superior to the SVM's $O(n^3)$ complexity when scaling to the full 60,000-image dataset.
3. **The Limits of 1D Data**: Despite the high accuracy, the confusion matrix revealed persistent struggles with visually similar digits (like 4s and 9s). Because the images are flattened into 1D arrays, the model loses spatial context (the 2D shape of loops and lines), highlighting the need for Deep Learning (CNNs) to push accuracy further.

## Future Improvements

* [ ] Extract and plot `feature_importances_` to visualize exactly which pixels the Random Forest relies on the most.
* [ ] Experiment with Gradient Boosting frameworks (like XGBoost or LightGBM) to see if sequential tree building outperforms parallel tree building.