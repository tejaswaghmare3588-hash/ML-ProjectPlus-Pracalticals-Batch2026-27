"""
HomeValue AI - Model Training and Evaluation Pipeline
Trains, compares, and evaluates 4 regression models on the Indian Housing Dataset.
Saves the best preprocessing and prediction pipeline using Joblib.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(DATASET_DIR, "housing_data.csv")

os.makedirs(MODELS_DIR, exist_ok=True)


def load_or_generate_dataset() -> pd.DataFrame:
    """Load cleaned housing dataset or trigger preparation if missing."""
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Preparing dataset now...")
        from dataset.prepare_dataset import prepare_housing_data
        df = prepare_housing_data()
    else:
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded existing housing dataset from {DATASET_PATH} with shape: {df.shape}")
    return df


def train_and_evaluate_models():
    """Execute complete preprocessing, training, model comparison, and pipeline export."""
    df = load_or_generate_dataset()

    print("\n--- Feature Engineering & Locality Grouping ---")
    # Identify popular localities per city (frequency >= 12)
    locality_counts = df["Locality"].value_counts()
    frequent_localities = locality_counts[locality_counts >= 12].index.tolist()
    
    # Create Locality_Grouped feature
    df["Locality_Grouped"] = df["Locality"].apply(lambda loc: loc if loc in frequent_localities else "Other")

    # Build City -> Localities dictionary for UI dropdowns
    city_locality_map = {}
    for city in sorted(df["City"].unique()):
        city_sub = df[df["City"] == city]
        locs = sorted(city_sub["Locality_Grouped"].unique().tolist())
        if "Other" in locs:
            locs.remove("Other")
            locs.append("Other")
        city_locality_map[city] = locs

    locality_map_file = os.path.join(MODELS_DIR, "locality_mapping.json")
    with open(locality_map_file, "w") as f:
        json.dump(city_locality_map, f, indent=2)
    print(f"Saved locality mapping to {locality_map_file}")

    # Define Feature Sets
    numeric_features = ["Area_sqft", "BHK", "Bathrooms", "Balconies", "Property_Age", "Parking"]
    categorical_features = ["City", "Locality_Grouped", "Furnishing_Status", "Property_Type"]
    feature_columns = categorical_features + numeric_features

    X = df[feature_columns]
    y = df["Price"]

    print(f"Features: {feature_columns}")
    print(f"Total instances: {len(X)}")

    # 80/20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Build Sklearn Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    # Candidate Regressors
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=4,
            random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=110,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.85,
            random_state=42
        )
    }

    metrics_summary = {}
    fitted_pipelines = {}
    test_predictions_dict = {}

    print("\n--- Training and Evaluating Models ---")
    best_name = None
    best_r2 = -float("inf")
    best_pipeline = None

    for name, regressor in candidate_models.items():
        print(f"\nTraining [{name}]...")
        start_time = time.time()

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        pipeline.fit(X_train, y_train)
        train_duration = round(time.time() - start_time, 2)

        # Evaluation on held-out test set
        y_pred = pipeline.predict(X_test)
        
        # Guard against rare negative predictions in edge cases
        y_pred = np.maximum(y_pred, 500000.0)

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        metrics_summary[name] = {
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2_Score": round(r2, 4),
            "Training_Time_Sec": train_duration
        }

        fitted_pipelines[name] = pipeline
        test_predictions_dict[name] = y_pred[:300].tolist()  # Sample for interactive charts

        print(f"Results for {name}:")
        print(f"  R2 Score : {r2:.4f}")
        print(f"  MAE      : ₹{mae:,.2f}")
        print(f"  RMSE     : ₹{rmse:,.2f}")
        print(f"  Train Time: {train_duration}s")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipeline = pipeline

    print(f"\n==========================================")
    print(f"BEST MODEL SELECTED: {best_name} (R2 = {best_r2:.4f})")
    print(f"==========================================")

    # Calculate residual standard error for confidence interval estimation
    best_pred_test = best_pipeline.predict(X_test)
    residuals = y_test - best_pred_test
    residual_std = float(np.std(residuals))
    print(f"Residual Standard Deviation (Best Model): ₹{residual_std:,.2f}")

    # Export best model pipeline
    best_pipeline_path = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
    joblib.dump(best_pipeline, best_pipeline_path)
    print(f"Saved best model pipeline to: {best_pipeline_path}")

    # Export overall metrics and metadata
    metadata = {
        "best_model": best_name,
        "best_r2": round(best_r2, 4),
        "residual_std": round(residual_std, 2),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "models_evaluated": metrics_summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_file = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metrics metadata to: {metrics_file}")

    # Save actual test sample slice for the interactive analytics dashboard
    test_sample_data = {
        "actual": y_test.iloc[:300].tolist(),
        "predictions": {name: preds for name, preds in test_predictions_dict.items()},
        "area": X_test["Area_sqft"].iloc[:300].tolist(),
        "bhk": X_test["BHK"].iloc[:300].tolist(),
        "city": X_test["City"].iloc[:300].tolist()
    }
    sample_file = os.path.join(MODELS_DIR, "test_sample_predictions.json")
    with open(sample_file, "w") as f:
        json.dump(test_sample_data, f, indent=2)
    print(f"Saved evaluation samples to: {sample_file}")

    print("\nTraining and pipeline packaging completed successfully!")
    return metadata


if __name__ == "__main__":
    train_and_evaluate_models()
