"""
HomeValue AI - Database Management Module
Handles SQLite operations for storing, retrieving, and exporting house price prediction history.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "predictions.db")


def get_connection():
    """Establish and return SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the predictions database and create tables if they do not exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                city TEXT NOT NULL,
                locality TEXT NOT NULL,
                area_sqft REAL NOT NULL,
                bhk INTEGER NOT NULL,
                bathrooms INTEGER NOT NULL,
                balconies INTEGER NOT NULL,
                property_age INTEGER NOT NULL,
                parking TEXT NOT NULL,
                furnishing_status TEXT NOT NULL,
                property_type TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                predicted_price_lakhs REAL NOT NULL,
                price_per_sqft REAL NOT NULL,
                lower_bound REAL NOT NULL,
                upper_bound REAL NOT NULL,
                model_used TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_prediction(
    city: str,
    locality: str,
    area_sqft: float,
    bhk: int,
    bathrooms: int,
    balconies: int,
    property_age: int,
    parking: str,
    furnishing_status: str,
    property_type: str,
    predicted_price: float,
    lower_bound: float,
    upper_bound: float,
    model_used: str = "Gradient Boosting Regressor"
) -> int:
    """Save a new prediction record into SQLite database."""
    init_db()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    price_lakhs = round(predicted_price / 100000.0, 2)
    price_per_sqft = round(predicted_price / area_sqft, 2)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                timestamp, city, locality, area_sqft, bhk, bathrooms, balconies,
                property_age, parking, furnishing_status, property_type,
                predicted_price, predicted_price_lakhs, price_per_sqft,
                lower_bound, upper_bound, model_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str, city, locality, area_sqft, bhk, bathrooms, balconies,
                property_age, parking, furnishing_status, property_type,
                predicted_price, price_lakhs, price_per_sqft,
                lower_bound, upper_bound, model_used
            )
        )
        conn.commit()
        return cursor.lastrowid


def get_prediction_history(limit: int = 500) -> pd.DataFrame:
    """Retrieve historical prediction records from SQLite as a DataFrame."""
    init_db()
    conn = get_connection()
    try:
        query = f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_summary_metrics() -> dict:
    """Compute aggregate metrics from real stored prediction records."""
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), AVG(predicted_price), MIN(predicted_price), MAX(predicted_price) FROM predictions")
        row = cursor.fetchone()
        count = row[0] or 0
        avg_price = row[1] or 0.0
        min_price = row[2] or 0.0
        max_price = row[3] or 0.0
        return {
            "total_predictions": count,
            "avg_price": avg_price,
            "min_price": min_price,
            "max_price": max_price
        }
    finally:
        conn.close()


def clear_history():
    """Clear all records from prediction history."""
    init_db()
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized at:", DB_PATH)
