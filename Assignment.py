#!/usr/bin/env python3

# === IMPORTS ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- CLASSIFICATION IMPORTS ---
# Metrics appropriate for classification tasks (Accuracy, detailed report, Confusion Matrix)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Ensemble model (Classification version)
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC  #Support Vector Classifier (SVC)

# Set visual style and ensure reproducibility
sns.set(style="whitegrid", palette="deep")
np.random.seed(42)


# === MAIN FUNCTION ===
def run_classification():

    print("=== Classification Task: California Housing (High/Low Value) ===")

    # Load dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # 1. --- CONVERT REGRESSION TARGET TO CLASSIFICATION TARGET ---
    # The original continuous target ('MedHouseVal') is binarized by the median.
    # This transforms the problem from 'predict the price' (regression) to
    # 'predict if the price is high or low' (classification).
    median_price = df['MedHouseVal'].median()
    df['HighValue'] = (df['MedHouseVal'] >= median_price).astype(int)  # 1 if price is high, 0 otherwise

    # Add a categorical feature (retained from original script)
    df['LatitudeZone'] = pd.cut(df['Latitude'], bins=5, labels=[f'Zone{i}' for i in range(1, 6)])

    # --- Simulate imbalance ---
    # Imbalance is simulated based on the NEW binary target ('HighValue').
    # All majority class samples (HighValue = 1) are kept, but only 30% of
    # the minority class (HighValue = 0) are sampled. This forces the models
    # to be evaluated under challenging, biased conditions.
    df_major = df[df['HighValue'] == 1]
    df_minor = df[df['HighValue'] == 0].sample(frac=0.3, random_state=42)
    df_imbal = pd.concat([df_major, df_minor])
    print(f"Original samples: {len(df)}, After imbalance: {len(df_imbal)}")

    # Define features and target (Dropping both original and new targets from features)
    X = df_imbal.drop(['MedHouseVal', 'HighValue'], axis=1)
    y = df_imbal['HighValue']  # The new binary target

    # Train-test split (Standard 80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing (Remains the same as features are the same)
    cat_cols = ['LatitudeZone']
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer([
        # Numerical features are scaled (essential for distance-based SVC)
        ('num', StandardScaler(), num_cols),
        # Categorical feature is one-hot encoded
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # 2. --- DEFINE CLASSIFICATION MODELS (SVC and Random Forest) ---
    models = {
        # Support Vector Classifier: Uses a linear kernel to find the maximum-margin hyperplane.
        # Since it is sensitive to feature scale, the StandardScaler step in the pipeline is critical.
        "Support Vector Classifier (SVC)": SVC(kernel='linear', random_state=42),
        # Random Forest Classifier: A robust, non-linear ensemble model that uses majority voting.
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    # Store results
    results = {}

    for name, model in models.items():
        # Build the pipeline: Preprocessing -> Classification Algorithm
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Train the model on the preprocessed training data
        pipeline.fit(X_train, y_train)
        # Predict the classes on the unseen test data
        y_pred = pipeline.predict(X_test)

        # 3. --- CLASSIFICATION EVALUATION ---
        # Evaluate performance using classification metrics
        acc = accuracy_score(y_test, y_pred)
        # Get a detailed report (includes Precision, Recall, F1-score for all classes)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {'pred': y_pred, 'accuracy': acc, 'report': report, 'model': pipeline}
        print(f"\n{name} → Accuracy: {acc:.3f}")
        # Focus on Class 1 (High Value) metrics due to the induced imbalance
        print(f"  Precision (Class 1): {report['1']['precision']:.3f}")
        print(f"  Recall (Class 1): {report['1']['recall']:.3f}")

    # --- Model tuning (Random Forest Classifier) ---
    print("\n--- Tuning Random Forest Classifier ---")
    param_grid = {
        # Tuning n_estimators (number of trees) and max_depth (tree complexity)
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20, None]
    }
    grid = GridSearchCV(
        Pipeline([('preprocessor', preprocessor),
                  ('model', RandomForestClassifier(random_state=42))]),
        param_grid,
        cv=3,
        scoring='accuracy',  # The hyperparameter optimization goal is maximizing accuracy
        n_jobs=-1
    )
    # Execute the grid search on the training data
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    tuned_pred = best_rf.predict(X_test)

    # 3. --- CLASSIFICATION EVALUATION FOR TUNED MODEL ---
    tuned_acc = accuracy_score(y_test, tuned_pred)
    tuned_report = classification_report(y_test, tuned_pred, output_dict=True)

    results['Tuned Random Forest'] = {'pred': tuned_pred, 'accuracy': tuned_acc, 'report': tuned_report,
                                      'model': best_rf}
    print(f"Best Params: {grid.best_params_}")
    print(f"Tuned Random Forest → Accuracy: {tuned_acc:.3f}")
    print(f"  Precision (Class 1): {tuned_report['1']['precision']:.3f}")
    print(f"  Recall (Class 1): {tuned_report['1']['recall']:.3f}")

    # --- Visualization ---
    plot_results_classification(y_test, results)


def plot_results_classification(y_test, results):
    """Visualize classification model performance in a multi-plot window.
    Focus is placed on Accuracy and class-specific metrics (Precision/Recall)
    due to the imbalanced nature of the dataset."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Classification Model Comparison and Evaluation", fontsize=16, fontweight='bold')

    model_names = list(results.keys())

    # === 1️⃣ Accuracy Score Comparison ===
    # Measures overall correct predictions (TP + TN) / Total
    accuracies = [res['accuracy'] for res in results.values()]
    sns.barplot(x=model_names, y=accuracies, ax=axes[0, 0], palette="Purples_d")
    axes[0, 0].set_title("Accuracy Score Comparison")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0.5, 1.0)

    # === 2️⃣ Precision Comparison (Focus on Class 1: High Value) ===
    # Precision: Out of all predictions made as High Value (Class 1), how many were actually correct?
    # High precision minimizes False Positives.
    precision_class1 = [res['report']['1']['precision'] for res in results.values()]
    sns.barplot(x=model_names, y=precision_class1, ax=axes[0, 1], palette="Blues_d")
    axes[0, 1].set_title("Precision (High Value Class) Comparison")
    axes[0, 1].set_ylabel("Precision")

    # === 3️⃣ Recall Comparison (Focus on Class 1: High Value) ===
    # Recall: Out of all actual High Value cases (Class 1), how many were correctly identified?
    # High recall minimizes False Negatives (missing a high-value home).
    recall_class1 = [res['report']['1']['recall'] for res in results.values()]
    sns.barplot(x=model_names, y=recall_class1, ax=axes[1, 0], palette="Greens_d")
    axes[1, 0].set_title("Recall (High Value Class) Comparison")
    axes[1, 0].set_ylabel("Recall")

    # === 4️⃣ Feature Importance (Random Forest Classifier) ===
    # Visualizes the top features used by the non-linear Random Forest Classifier
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model'].named_steps['model']
        # Dynamically extracts feature names after preprocessing/encoding
        feature_names = np.concatenate([
            results['Random Forest']['model'].named_steps['preprocessor']
            .transformers_[0][2],
            results['Random Forest']['model'].named_steps['preprocessor']
            .named_transformers_['cat']
            .get_feature_names_out(['LatitudeZone'])
        ])
        importances = rf_model.feature_importances_
        idx = np.argsort(importances)[-10:]
        sns.barplot(x=importances[idx], y=feature_names[idx], ax=axes[1, 1], palette="Reds_d")
        axes[1, 1].set_title("Top 10 Feature Importances (Random Forest)")
    else:
        axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# === ENTRY POINT ===
if __name__ == "__main__":
    run_classification()