"""
HomeValue AI - Automated Comprehensive Test Suite
Validates model training, inference, database storage, input validation, and artifact integrity.
"""

import os
import json
import unittest
import pandas as pd
import numpy as np
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "housing_data.csv")
MODEL_PIPELINE_PATH = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
LOCALITY_MAP_PATH = os.path.join(MODELS_DIR, "locality_mapping.json")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Import DB
from database.db_handler import (
    init_db,
    save_prediction,
    get_prediction_history,
    get_summary_metrics,
    clear_history
)


class TestHomeValueAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test environment and initialize database."""
        init_db()

    def test_01_dataset_integrity(self):
        """Verify cleaned dataset existence, shape, and essential columns."""
        self.assertTrue(os.path.exists(DATASET_PATH), "Dataset housing_data.csv must exist.")
        df = pd.read_csv(DATASET_PATH)
        self.assertGreaterEqual(len(df), 25000, "Dataset should have at least 25,000 records.")
        required_cols = [
            "City", "Locality", "Area_sqft", "BHK", "Bathrooms", "Balconies",
            "Property_Age", "Parking", "Furnishing_Status", "Property_Type", "Price"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Column '{col}' must exist in dataset.")
        # Ensure no negative prices or areas
        self.assertTrue((df["Price"] > 0).all(), "All prices must be positive.")
        self.assertTrue((df["Area_sqft"] > 0).all(), "All areas must be positive.")

    def test_02_model_artifacts_and_metrics(self):
        """Verify that the model pipeline and evaluation metrics exist and are non-empty."""
        self.assertTrue(os.path.exists(MODEL_PIPELINE_PATH), "Pipeline artifact must exist.")
        self.assertTrue(os.path.exists(METRICS_PATH), "Metrics metadata must exist.")
        self.assertTrue(os.path.exists(LOCALITY_MAP_PATH), "Locality map must exist.")

        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

        self.assertIn("best_model", metrics)
        self.assertIn("models_evaluated", metrics)
        models_eval = metrics["models_evaluated"]
        # Must compare all 4 required regressors
        required_models = [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "Gradient Boosting Regressor"
        ]
        for m in required_models:
            self.assertIn(m, models_eval, f"Model '{m}' must be evaluated.")
            self.assertIn("R2_Score", models_eval[m])
            self.assertIn("MAE", models_eval[m])
            self.assertIn("RMSE", models_eval[m])

    def test_03_inference_pipeline_execution(self):
        """Test inference execution for multiple cities and property configurations."""
        pipeline = joblib.load(MODEL_PIPELINE_PATH)
        self.assertIsNotNone(pipeline, "Loaded pipeline should not be None.")

        test_cases = [
            {
                "City": "Bangalore",
                "Locality_Grouped": "Whitefield",
                "Furnishing_Status": "Semi-Furnished",
                "Property_Type": "Apartment",
                "Area_sqft": 1450.0,
                "BHK": 3,
                "Bathrooms": 3,
                "Balconies": 2,
                "Property_Age": 2,
                "Parking": 1
            },
            {
                "City": "Mumbai",
                "Locality_Grouped": "Bandra West",
                "Furnishing_Status": "Furnished",
                "Property_Type": "Apartment",
                "Area_sqft": 1100.0,
                "BHK": 2,
                "Bathrooms": 2,
                "Balconies": 1,
                "Property_Age": 5,
                "Parking": 1
            },
            {
                "City": "Delhi",
                "Locality_Grouped": "Other",
                "Furnishing_Status": "Unfurnished",
                "Property_Type": "Independent House",
                "Area_sqft": 2200.0,
                "BHK": 4,
                "Bathrooms": 4,
                "Balconies": 2,
                "Property_Age": 8,
                "Parking": 1
            }
        ]

        for tc in test_cases:
            input_df = pd.DataFrame([tc])
            pred = pipeline.predict(input_df)[0]
            self.assertIsInstance(float(pred), float)
            self.assertGreater(pred, 500000, f"Predicted price {pred} should be above 5 Lakhs floor.")

    def test_04_database_operations(self):
        """Test SQLite prediction saving, history retrieval, metrics computation, and cleanup."""
        initial_metrics = get_summary_metrics()
        initial_count = initial_metrics["total_predictions"]

        # Insert a sample test prediction
        new_id = save_prediction(
            city="Bangalore",
            locality="Whitefield",
            area_sqft=1450.0,
            bhk=3,
            bathrooms=3,
            balconies=2,
            property_age=2,
            parking="Yes",
            furnishing_status="Semi-Furnished",
            property_type="Apartment",
            predicted_price=9500000.0,
            lower_bound=8500000.0,
            upper_bound=10500000.0,
            model_used="Gradient Boosting Regressor"
        )
        self.assertGreater(new_id, 0, "New record ID should be positive.")

        # Verify record appeared in history
        history_df = get_prediction_history(limit=10)
        self.assertFalse(history_df.empty, "History should not be empty.")
        self.assertEqual(history_df.iloc[0]["city"], "Bangalore")
        self.assertEqual(history_df.iloc[0]["locality"], "Whitefield")
        self.assertEqual(history_df.iloc[0]["bhk"], 3)

        updated_metrics = get_summary_metrics()
        self.assertEqual(updated_metrics["total_predictions"], initial_count + 1)

    def test_05_documentation_and_deliverables(self):
        """Verify presence of technical diagrams and documentation."""
        diagram_files = [
            "system_architecture.png",
            "project_flowchart.png",
            "uml_usecase.png",
            "uml_class.png",
            "uml_sequence.png"
        ]
        for img in diagram_files:
            img_path = os.path.join(IMAGES_DIR, img)
            self.assertTrue(os.path.exists(img_path), f"Diagram {img} must exist.")
            self.assertGreater(os.path.getsize(img_path), 5000, f"Diagram {img} must be non-empty.")


if __name__ == "__main__":
    unittest.main()
