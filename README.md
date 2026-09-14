# Iris Flower Species Classification AI

An end-to-end supervised machine learning pipeline for classifying Iris flower species using **Scikit-Learn**. This project covers exploratory data analysis (EDA), data preprocessing, feature scaling, model training via **Logistic Regression**, model evaluation, and inference on unseen custom measurements.

---

## Project Overview
Automated flower species identification is a classic baseline benchmark in machine learning and exploratory data analysis. This project implements a clean, robust machine learning workflow using Python and `scikit-learn` to process data, train a multi-class Logistic Regression classifier, and systematically evaluate predictive performance.

---

## Dataset
The model uses the classic built-in **Iris dataset** via `sklearn.datasets`, containing 150 samples split evenly across **3 distinct flower species**:
1. **Setosa**
2. **Versicolor**
3. **Virginica**

**Features Tracked:**
* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

---

## Pipeline Workflow

Load Raw Dataset (Iris)
     ↓
Exploratory Data Analysis (EDA & Seaborn Pairplot)
     ↓
Train-Test Split (80% Training, 20% Testing)
     ↓
Feature Scaling (StandardScaler fitted on Train, applied to Test)
     ↓
Model Training (Logistic Regression)
     ↓
Evaluation (Accuracy, Classification Report, Confusion Matrix)
     ↓
Inference (Predicting species for custom new flower measurements)

---

## Model Training & Evaluation Strategy
* **Train/Test Split:** Data is partitioned using an 80/20 ratio (`train_test_split`) with a fixed random state for reproducibility.
* **Feature Scaling:** Uses `StandardScaler` to normalize feature columns. Crucially, the scaler is *fitted exclusively on the training data* and *transformed* onto the testing data to prevent data leakage.
* **Classifier:** Multi-class **Logistic Regression** configured with an optimized maximum iteration limit (`max_iter=200`).
* **Metrics Tracked:** 
  * Predictive Accuracy Score
  * Precision, Recall, and F1-Score (via Classification Report)
  * Confusion Matrix breakdown across all 3 classes

---

## Inference: Predicting New Samples
The pipeline includes a dedicated inference block that safely accepts raw feature arrays for brand-new flowers, applies the exact same preprocessing/scaling parameters learned during training, and outputs the predicted target index and species string name.

---

## Project Objectives
* Load and format tabular multi-class datasets using `pandas` and `scikit-learn`.
* Conduct visual exploratory data analysis utilizing `matplotlib` and `seaborn`.
* Prevent data leakage by correctly decoupling training and testing scaling transformations.
* Build, fit, and evaluate a multi-class Logistic Regression model.
* Implement a structured inference mechanism to classify unobserved real-world inputs.

---

## Getting Started

### Prerequisites
Ensure you have Python installed along with the essential data science libraries:
pip install scikit-learn pandas matplotlib seaborn

### Running the Code
1. Clone the repository:
   git clone https://github.com/Aaryam-S/Flower-Species-Classification-AI.git
   cd Flower-Species-Classification-AI
2. Run the Python script or open the Jupyter Notebook containing the workflow to train the model and test out custom predictions.
