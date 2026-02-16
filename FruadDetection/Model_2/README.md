# Fraud Detection Model

A machine learning project for detecting fraudulent credit card transactions using Random Forest Classification with SMOTE for handling class imbalance.

## Overview

This project implements a binary classification model to identify fraudulent credit card transactions. The model uses a Random Forest Classifier trained on PCA-transformed transaction features to achieve high accuracy in fraud detection while maintaining a low false positive rate.

## Model Performance

The model achieves the following metrics on the test set:

- **Accuracy**: 99.14%
- **F1 Score**: 99.13%
- **Precision**: 99.89%
- **Recall**: 98.39%

## Dataset

The project uses the **Credit Card Fraud Detection** dataset, which contains transactions made by European cardholders in September 2013. 

### Dataset Features:
- **Time**: Seconds elapsed between this transaction and the first transaction
- **V1-V28**: PCA-transformed features (anonymized due to confidentiality)
- **Amount**: Transaction amount
- **Class**: Target variable (0 = legitimate, 1 = fraudulent)

### Class Distribution:
The dataset is highly imbalanced, with fraudulent transactions representing a small minority of the total transactions.

## Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning algorithms and tools
- **imbalanced-learn (imblearn)** - SMOTE for handling class imbalance

## Methodology

### 1. Data Preprocessing
- Load the credit card transaction dataset
- Exploratory data analysis and visualization
- Feature scaling using StandardScaler

### 2. Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset
- SMOTE generates synthetic samples for the minority class (fraudulent transactions)

### 3. Model Training
- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - `max_depth=10` - Limits tree depth to prevent overfitting
  - `n_jobs=-1` - Uses all available CPU cores for parallel processing
  - `random_state=42` - Ensures reproducibility

### 4. Model Evaluation
- Split data into training and testing sets
- Evaluated using multiple metrics:
  - Accuracy
  - F1 Score
  - Precision
  - Recall

## Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Running the Model

1. **Mount Google Drive** (if using Google Colab):
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Load the dataset**:
```python
df = pd.read_csv("path/to/creditcard.csv")
```

3. **Preprocess and train**:
- Apply feature scaling
- Use SMOTE for class balancing
- Train the Random Forest model

4. **Evaluate**:
```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

## 📈 Model Architecture

```
Input Features (V1-V28, Time, Amount)
        ↓
StandardScaler (Feature Scaling)
        ↓
SMOTE (Oversampling)
        ↓
Train/Test Split
        ↓
Random Forest Classifier
    - max_depth: 10
    - n_jobs: -1
    - random_state: 42
        ↓
Predictions & Evaluation
```

##  Key Insights

1. **High Precision (99.89%)**: The model rarely flags legitimate transactions as fraudulent, minimizing false positives
2. **Strong Recall (98.39%)**: Catches the vast majority of actual fraud cases
3. **SMOTE Effectiveness**: Successfully addresses class imbalance without data loss
4. **Random Forest Benefits**: Handles non-linear relationships and provides robust predictions

## Important Considerations

- **Real-time Performance**: Consider model inference time for production deployment
- **Model Updates**: Regularly retrain with new fraud patterns as they evolve
- **Threshold Tuning**: Adjust classification threshold based on business requirements (cost of false positives vs. false negatives)
- **Feature Engineering**: The V1-V28 features are PCA components; original features are confidential

## Future Improvements

- [ ] Implement cross-validation for more robust performance estimates
- [ ] Experiment with other algorithms (XGBoost, LightGBM, Neural Networks)
- [ ] Feature importance analysis to identify key fraud indicators
- [ ] Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- [ ] Deploy as a REST API for real-time fraud detection
- [ ] Add model interpretability using SHAP or LIME
- [ ] Implement anomaly detection techniques

## License

This project uses a public dataset and is intended for educational and research purposes.

##  Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This model was trained on historical data. Always validate model performance on current data before deploying to production.