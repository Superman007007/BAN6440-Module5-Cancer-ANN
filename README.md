# BAN6440 Module 5 - Cancer Diagnosis Using Artificial Neural Networks

## Project Overview

This project develops and evaluates an Artificial Neural Network (ANN) for binary cancer diagnosis using TensorFlow and Keras.

The model predicts whether a diagnosis is:

- 0 = Benign
- 1 = Malignant

The project was developed for BAN6440 Applied Machine Learning for Business Analytics.

## Dataset

Dataset:

BAN6440 M5 mdanderson_cancer_diagnosis.csv

Records: 600 patients

- Benign: 320 (53.33%)
- Malignant: 280 (46.67%)
- Missing values: 0
- Duplicate rows: 0

The `patient_id` field was excluded from model training because it is an identifier rather than a predictive radiological characteristic.

## Input Features

The ANN uses 12 numeric radiological features:

1. mean_radius
2. mean_texture
3. mean_perimeter
4. mean_area
5. mean_smoothness
6. mean_compactness
7. mean_concavity
8. mean_concave_points
9. mean_symmetry
10. mean_fractal_dimension
11. worst_radius
12. worst_area

## Data Preprocessing

The dataset was divided using stratified sampling:

- Training: 420 patients (70%)
- Validation: 90 patients (15%)
- Testing: 90 patients (15%)

StandardScaler was fitted only to the training dataset to avoid data leakage.

## Baseline ANN

Architecture:

12 Inputs
→ Dense 32 (ReLU)
→ Dense 16 (ReLU)
→ Dense 1 (Sigmoid)

Training configuration:

- Optimizer: Adam
- Learning rate: 0.001
- Loss: Binary Cross-Entropy
- Batch size: 16
- Maximum epochs: 150
- Early stopping enabled

### Baseline Test Results

- Accuracy: 96.67%
- Precision: 95.35%
- Recall: 97.62%
- F1 Score: 96.47%
- ROC-AUC: 99.65%

Confusion matrix:

- True Negatives: 46
- False Positives: 2
- False Negatives: 1
- True Positives: 41

## Improved ANN

The improved model added:

- 20% Dropout after the first hidden layer
- 10% Dropout after the second hidden layer
- Class weighting with additional emphasis on malignant cases
- ReduceLROnPlateau
- Early stopping

Training stopped after 73 epochs.

### Improved Test Results

- Accuracy: 97.78%
- Precision: 95.45%
- Recall: 100.00%
- F1 Score: 97.67%
- ROC-AUC: 99.85%

Confusion matrix:

- True Negatives: 46
- False Positives: 2
- False Negatives: 0
- True Positives: 42

The improved model detected all 42 malignant cases in the held-out test sample.

## Model Improvement

The main improvement objective was reducing malignant false negatives while maintaining strong overall classification performance.

Malignant recall improved:

97.62% → 100.00%

False negatives improved:

1 → 0

Overall accuracy improved:

96.67% → 97.78%

## Unit Testing

Nine automated tests were implemented with pytest.

Tests validate:

- Dataset structure
- Missing-value handling
- Patient ID exclusion
- Stratified split sizes
- Feature scaling
- Baseline ANN architecture
- Improved ANN architecture
- Probability output range
- Saved model availability and loading

All nine tests passed successfully.

## Project Files

- inspect_data.py - Dataset inspection and exploratory checks
- main.py - Data loading, stratified splitting, and preprocessing
- train_ann.py - Baseline ANN training
- evaluate_ann.py - Baseline test-set evaluation
- improve_ann.py - Improved ANN training and evaluation
- test_ann.py - Automated unit tests
- output/ - Model results, plots, predictions, and saved models

## Important Limitation

This project is an educational machine learning exercise. Strong performance on the provided dataset does not establish clinical validity.

A real cancer diagnostic system would require external validation, larger and more diverse patient populations, bias assessment, calibration, regulatory review, and qualified clinician oversight before clinical use.

## Author

Kyaw Swar

BAN6440 Applied Machine Learning for Business Analytics