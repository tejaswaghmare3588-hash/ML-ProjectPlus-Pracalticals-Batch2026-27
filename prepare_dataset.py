"""
Dataset Preparation and Cleaning Script for HomeValue AI
Builds a unified, standardized Indian Housing Dataset from raw metropolitan records.
"""

import os
import glob
import pandas as pd
import numpy as np

# Set base paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(CURRENT_DIR, "raw_cities")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "housing_data.csv")

def prepare_housing_data():
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not raw_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    print(f"Found {len(raw_files)} city raw datasets: {[os.path.basename(f) for f in raw_files]}")
    
    dfs = []
    for f in raw_files:
        city_name = os.path.basename(f).replace(".csv", "")
        df_city = pd.read_csv(f)
        df_city["City"] = city_name
        dfs.append(df_city)

    raw_df = pd.concat(dfs, ignore_index=True)
    print(f"Raw merged shape: {raw_df.shape}")

    # Set fixed random seed for reproducibility
    np.random.seed(42)

    # 1. Car Parking: Kaggle dataset uses '9' for unrecorded/missing
    def resolve_parking(row):
        cp = row.get("CarParking", 9)
        if cp in [0, 1]:
            return int(cp)
        # Logical estimation for missing values: larger flats/villas or clubhouses usually provide parking
        if row.get("Area", 0) >= 1100 or row.get("No. of Bedrooms", 0) >= 3 or row.get("ClubHouse", 0) == 1:
            return 1
        return 0

    # 2. Furnishing Status based on appliance and furniture inventory
    furnish_cols = ['BED', 'Sofa', 'DiningTable', 'Wardrobe', 'Refrigerator', 'AC', 'WashingMachine', 'TV', 'Microwave']
    def resolve_furnishing(row):
        score = sum(1 for c in furnish_cols if row.get(c, 0) == 1)
        if score >= 4:
            return "Furnished"
        elif score >= 1:
            return "Semi-Furnished"
        else:
            return "Unfurnished"

    # 3. Bathrooms: standard Indian architectural norms
    def resolve_bathrooms(row):
        bhk = int(row["No. of Bedrooms"])
        area = float(row["Area"])
        if bhk == 1:
            return 1 if area < 700 else int(np.random.choice([1, 2], p=[0.85, 0.15]))
        elif bhk == 2:
            return 2 if area < 1300 else int(np.random.choice([2, 3], p=[0.85, 0.15]))
        elif bhk == 3:
            return 2 if area < 1250 else (3 if area < 2200 else int(np.random.choice([3, 4], p=[0.8, 0.2])))
        elif bhk == 4:
            return 3 if area < 2000 else (4 if area < 3500 else int(np.random.choice([4, 5], p=[0.8, 0.2])))
        else:
            return min(bhk, 6)

    # 4. Balconies
    def resolve_balconies(row):
        bhk = int(row["No. of Bedrooms"])
        if bhk == 1:
            return int(np.random.choice([1, 0], p=[0.80, 0.20]))
        elif bhk == 2:
            return int(np.random.choice([1, 2], p=[0.45, 0.55]))
        elif bhk == 3:
            return int(np.random.choice([2, 3], p=[0.55, 0.45]))
        else:
            return int(np.random.choice([2, 3, 4], p=[0.30, 0.50, 0.20]))

    # 5. Property Age in years
    # Resale == 0 means brand-new / ready-to-move (0-2 years)
    # Resale == 1 means existing property (3-25 years)
    def resolve_property_age(row):
        resale = row.get("Resale", 0)
        if resale == 0:
            return int(np.random.choice([0, 1, 2], p=[0.60, 0.30, 0.10]))
        else:
            age = int(np.random.exponential(scale=6) + 3)
            return min(max(age, 3), 28)

    # 6. Property Type (Apartment, Independent House, Villa)
    def resolve_property_type(row):
        area = float(row["Area"])
        bhk = int(row["No. of Bedrooms"])
        lift = row.get("LiftAvailable", 0)
        gardens = row.get("LandscapedGardens", 0)
        pool = row.get("SwimmingPool", 0)

        if area >= 2400 and bhk >= 3 and (gardens == 1 or pool == 1):
            return "Villa"
        elif (lift == 0 or lift == 9) and area >= 1200 and np.random.rand() < 0.25:
            return "Independent House"
        else:
            return "Apartment"

    # Assemble cleaned DataFrame
    df = pd.DataFrame()
    df["City"] = raw_df["City"]
    df["Locality"] = raw_df["Location"].astype(str).str.strip()
    df["Area_sqft"] = raw_df["Area"].astype(float)
    df["BHK"] = raw_df["No. of Bedrooms"].astype(int)
    df["Bathrooms"] = [resolve_bathrooms(r) for _, r in raw_df.iterrows()]
    df["Balconies"] = [resolve_balconies(r) for _, r in raw_df.iterrows()]
    df["Property_Age"] = [resolve_property_age(r) for _, r in raw_df.iterrows()]
    df["Parking"] = [resolve_parking(r) for _, r in raw_df.iterrows()]
    df["Furnishing_Status"] = [resolve_furnishing(r) for _, r in raw_df.iterrows()]
    df["Property_Type"] = [resolve_property_type(r) for _, r in raw_df.iterrows()]
    df["Price"] = raw_df["Price"].astype(float)

    # Realistic outlier filtering based on real-estate domains
    # Area: 300 to 8,000 sq ft
    # BHK: 1 to 6
    # Price: ₹15 Lakhs to ₹15 Crores
    # Price per sq ft: ₹2,000 to ₹50,000 / sqft
    df = df[(df["Area_sqft"] >= 300) & (df["Area_sqft"] <= 8000)]
    df = df[(df["BHK"] >= 1) & (df["BHK"] <= 6)]
    df = df[(df["Price"] >= 1500000) & (df["Price"] <= 150000000)]
    
    df["Price_per_sqft"] = df["Price"] / df["Area_sqft"]
    df = df[(df["Price_per_sqft"] >= 2000) & (df["Price_per_sqft"] <= 50000)]

    # Clean locality name strings
    df["Locality"] = df["Locality"].str.title().str.replace(r"\s+", " ", regex=True)

    # Save to disk
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned housing data successfully saved to: {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")
    print("City distribution:\n", df["City"].value_counts())
    print("Property Type distribution:\n", df["Property_Type"].value_counts())
    print("Furnishing Status distribution:\n", df["Furnishing_Status"].value_counts())
    return df

if __name__ == "__main__":
    prepare_housing_data()
