# Project 2: Data Classification Using AI

**DecodeLabs · Artificial Intelligence Track · Batch 2026**

A supervised machine learning project that classifies iris flowers into three species (*Setosa*, *Versicolor*, *Virginica*) from four measurements, using K-Nearest Neighbors.

## Objective

Build a basic classification model on a small dataset and demonstrate the core supervised learning pipeline: load and understand the data, split it, train a model, and validate the results.

## Dataset

The **Iris benchmark** dataset (bundled with scikit-learn):

| Property | Value |
|---|---|
| Samples | 150 (balanced, 50 per class) |
| Features | 4 (sepal length, sepal width, petal length, petal width, in cm) |
| Classes | 3 (setosa, versicolor, virginica) |
| Missing values | 0 |

## Methodology

1. **Exploratory analysis:** class balance, pairplot, correlation heatmap, and feature ranges by species.
2. **Feature scaling:** `StandardScaler` (mean 0, variance 1), placed inside a `Pipeline` so it is fitted on training data only (no data leakage).
3. **Train/test split:** 80/20, shuffled and stratified (`random_state=42`).
4. **Choosing K:** 5-fold cross-validation on the training set, using the elbow method. Best **K = 3**.
5. **Model:** `KNeighborsClassifier` (instantiate, fit, predict).
6. **Validation:** confusion matrix, precision, recall, and F1 score.
7. **Robustness check:** 10-fold cross-validation and comparison with other classifiers.

## Results

| Metric (test set, 30 samples) | Score |
|---|---|
| Accuracy | 0.933 |
| Precision (macro) | 0.944 |
| Recall (macro) | 0.933 |
| F1 score (macro) | 0.933 |

**Model comparison** (10-fold cross-validation, macro F1):

| Model | CV Macro F1 | Std | Test Accuracy |
|---|---|---|---|
| KNN (K=3) | 0.946 | 0.072 | 0.933 |
| Logistic Regression | 0.953 | 0.053 | 0.933 |
| SVM (RBF) | 0.960 | 0.044 | 0.967 |
| Random Forest | 0.945 | 0.060 | 0.900 |

**Key findings**
- *Setosa* is classified perfectly.
- The few errors occur between *Versicolor* and *Virginica*, whose measurements overlap.
- Petal length and petal width are the most informative features.

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Google Colab

## How to Run

1. Open `Project2_Iris_Classification.ipynb` in [Google Colab](https://colab.research.google.com) (File → Upload notebook).
2. Choose **Runtime → Run all**.

No dataset upload is needed, since Iris ships with scikit-learn. To run locally:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn notebook
jupyter notebook Project2_Iris_Classification.ipynb
```

## Repository Structure

```
├── Project2_Iris_Classification.ipynb   # Full notebook with code, outputs and plots
└── README.md
```

## Author

**I. Rihana Sulthana**
AI Intern, DecodeLabs (Batch 2026)
