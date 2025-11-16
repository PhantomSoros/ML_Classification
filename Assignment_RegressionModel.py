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

# --- REGRESSION IMPORTS ---
# Metrics appropriate for regression tasks (Mean Squared Error, R2)
from sklearn.metrics import mean_squared_error, r2_score
# Ensemble model (Regression version)
from sklearn.ensemble import RandomForestRegressor
# Support Vector Regressor (SVR)
from sklearn.svm import SVR

# Set visual style and ensure reproducibility
sns.set(style="whitegrid", palette="deep")
np.random.seed(42)


# === MAIN FUNCTION ===
def run_regression():

    print("=== Regression Task: California Housing (Predict MedHouseVal) ===")

    # Load dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # --- DEFINE FEATURES AND TARGET ---
    # The original continuous target is kept.
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    # Add a categorical feature (retained from original script)
    X['LatitudeZone'] = pd.cut(X['Latitude'], bins=5, labels=[f'Zone{i}' for i in range(1, 6)])

    print(f"Total samples: {len(df)}")

    # Train-test split (Standard 80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing (Remains the same as features are the same)
    cat_cols = ['LatitudeZone']
    # The 'MedHouseVal' column is no longer in X, so no need to drop it here.
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer([
        # Numerical features are scaled (essential for distance-based SVR)
        ('num', StandardScaler(), num_cols),
        # Categorical feature is one-hot encoded
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # 1. --- DEFINE REGRESSION MODELS (SVR and Random Forest Regressor) ---
    models = {
        # Support Vector Regressor: Sensitive to feature scale, hence the StandardScaler is critical.
        "Support Vector Regressor (SVR)": SVR(kernel='linear'),
        # Random Forest Regressor: A robust, non-linear ensemble model.
        "Random Forest Regressor": RandomForestRegressor(random_state=42)
    }

    # Store results
    results = {}

    for name, model in models.items():
        # Build the pipeline: Preprocessing -> Regression Algorithm
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Train the model on the preprocessed training data
        pipeline.fit(X_train, y_train)
        # Predict the continuous values on the unseen test data
        y_pred = pipeline.predict(X_test)

        # 2. --- REGRESSION EVALUATION ---
        # Evaluate performance using regression metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {'pred': y_pred, 'mse': mse, 'r2': r2, 'model': pipeline}
        print(f"\n{name} → R² Score: {r2:.3f}, MSE: {mse:.3f}")

    # --- Model tuning (Random Forest Regressor) ---
    print("\n--- Tuning Random Forest Regressor ---")
    param_grid = {
        # Tuning n_estimators (number of trees) and max_depth (tree complexity)
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20, None]
    }
    grid = GridSearchCV(
        Pipeline([('preprocessor', preprocessor),
                  ('model', RandomForestRegressor(random_state=42))]),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',  # Optimization goal is to minimize MSE (maximize negative MSE)
        n_jobs=-1
    )
    # Execute the grid search on the training data
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    tuned_pred = best_rf.predict(X_test)

    # 2. --- REGRESSION EVALUATION FOR TUNED MODEL ---
    tuned_mse = mean_squared_error(y_test, tuned_pred)
    tuned_r2 = r2_score(y_test, tuned_pred)

    results['Tuned Random Forest Regressor'] = {'pred': tuned_pred, 'mse': tuned_mse, 'r2': tuned_r2,
                                               'model': best_rf}
    print(f"Best Params: {grid.best_params_}")
    print(f"Tuned Random Forest Regressor → R² Score: {tuned_r2:.3f}, MSE: {tuned_mse:.3f}")

    # --- Visualization ---
    plot_results_regression(y_test, results)


def plot_results_regression(y_test, results):
    """Visualize regression model performance using R-squared, MSE, and residuals."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Regression Model Comparison and Evaluation", fontsize=16, fontweight='bold')

    model_names = list(results.keys())

    # === 1️⃣ R-squared ($R^2$) Comparison ===
    # Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
    r2_scores = [res['r2'] for res in results.values()]
    sns.barplot(x=model_names, y=r2_scores, ax=axes[0, 0], palette="Purples_d")
    axes[0, 0].set_title("$R^2$ Score Comparison (Closer to 1 is better)")
    axes[0, 0].set_ylabel("$R^2$ Score")
    axes[0, 0].set_ylim(0.0, 1.0)

    # === 2️⃣ Mean Squared Error (MSE) Comparison ===
    # Measures the average squared difference between the estimated values and the actual value.
    # Lower values are better.
    mse_scores = [res['mse'] for res in results.values()]
    sns.barplot(x=model_names, y=mse_scores, ax=axes[0, 1], palette="Blues_d")
    axes[0, 1].set_title("Mean Squared Error (MSE) Comparison (Lower is better)")
    axes[0, 1].set_ylabel("MSE")

    # === 3️⃣ Residual Plot (Tuned Random Forest) ===
    # Plots the prediction error (Residual = Actual - Predicted). Ideal: Random scattering around y=0.
    if 'Tuned Random Forest Regressor' in results:
        tuned_preds = results['Tuned Random Forest Regressor']['pred']
        residuals = y_test - tuned_preds
        sns.scatterplot(x=tuned_preds, y=residuals, ax=axes[1, 0])
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title("Residual Plot (Tuned Random Forest)")
        axes[1, 0].set_xlabel("Predicted Values")
        axes[1, 0].set_ylabel("Residuals (Actual - Predicted)")
    else:
        axes[1, 0].axis('off')

    # === 4️⃣ Feature Importance (Random Forest Regressor) ===
    # Visualizes the top features used by the non-linear Random Forest Regressor
    if 'Random Forest Regressor' in results:
        rf_model = results['Random Forest Regressor']['model'].named_steps['model']
        # Dynamically extracts feature names after preprocessing/encoding
        feature_names = np.concatenate([
            results['Random Forest Regressor']['model'].named_steps['preprocessor']
            .transformers_[0][2],
            results['Random Forest Regressor']['model'].named_steps['preprocessor']
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
    run_regression()