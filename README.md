# 🏠 HomeValue AI: House Price Prediction Using Machine Learning
### *A Full-Stack Intelligent Real-Estate Valuation Web Platform (BTech 3rd Year Major Project)*

[![Python](https://img.shields.io/badge/Python-3.14%20%7C%203.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-3.0+-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Joblib](https://img.shields.io/badge/Joblib-Serialized%20Pipelines-blue?style=for-the-badge)](https://joblib.readthedocs.io/)

---

## 📋 Table of Contents
1. [Project Overview & Objectives](#-project-overview--objectives)
2. [Key Highlights & Features](#-key-highlights--features)
3. [Technology Stack](#-technology-stack)
4. [Dataset Information & Preprocessing](#-dataset-information--preprocessing)
5. [Machine Learning Pipeline & Model Comparison](#-machine-learning-pipeline--model-comparison)
6. [Empirical Model Evaluation Results](#-empirical-model-evaluation-results)
7. [System Architecture & Diagrams](#-system-architecture--diagrams)
8. [Database Schema & Persistence](#-database-schema--persistence)
9. [Project Folder Structure](#-project-folder-structure)
10. [Installation & Execution Guide](#-installation--execution-guide)
11. [Comprehensive Viva Questions & Answers](#-comprehensive-viva-questions--answers)
12. [Ethical Disclaimer & Licensing](#-ethical-disclaimer--licensing)

---

## 🎯 Project Overview & Objectives

**HomeValue AI** is an intelligent, full-stack machine learning web application designed to predict residential property prices across major Indian metropolitan areas (**Mumbai, Bangalore, Delhi NCR, Hyderabad, Chennai, and Kolkata**). 

Accurate real-estate appraisal has traditionally been impeded by opaque broker networks, subjective assessments, and static price-per-square-foot formulas that fail to account for non-linear interactions among property age, furnishing grade, room distribution, and micro-market locality premiums. 

### Core Project Goals:
- Build a supervised regression engine trained on **31,103 authentic Indian metropolitan property listings**.
- Benchmark four fundamental regressors: **Linear Regression, CART Decision Tree, Random Forest, and Gradient Boosting Regressor**.
- Deliver an automated, zero-leakage **Scikit-learn Pipeline** packaged with **Joblib**.
- Present a responsive web application powered by **Streamlit** following a luxury real-estate aesthetic (dark navy `#0B192C`, slate blue `#1E3E62`, cyan `#008DDA`, and gold `#D4AF37`).
- Provide persistent, privacy-preserving query audit logging via an embedded **SQLite** database with instant **CSV export**.
- Deliver statistically justified **90% confidence prediction intervals** ($\pm 1.645 \sigma$) rather than deceptive exact point estimates.

---

## ✨ Key Highlights & Features

1. **Cascading Locality Dropdowns**: Selecting a metro city instantly cascades to its verified localities, preventing invalid spatial queries.
2. **Comprehensive Property Parameterization**:
   - Metro City & Micro-Market Locality
   - Living Area (300 to 8,000 sq ft)
   - Bedroom Count (1 to 6 BHK)
   - Bathroom Count (1 to 6 Baths)
   - Balconies (0 to 4 Balconies)
   - Property Age (0 to 30 Years)
   - Allocated Car Parking (Yes / No)
   - Furnishing Status (Unfurnished, Semi-Furnished, Furnished)
   - Property Type (Apartment, Independent House, Villa)
3. **Statistically Justified Intervals**: Uses test-partition residual standard deviation ($\sigma \approx \text{₹76.8 Lakhs}$) to display credible lower and upper bounds.
4. **Interactive Analytics Dashboard**:
   - Price distribution histograms
   - Area vs. Price scatter plots with city filters
   - BHK vs. Median Price comparative bar charts
   - City-wise price rank comparisons
   - Actual vs. Predicted model fit plots
5. **Full Prediction History**: SQLite audit trail logging every user appraisal with real-time statistics and one-click CSV export.
6. **Zero PII Collection**: Pure technical property variables are logged; no user names, phone numbers, or emails are collected.

---

## 🛠️ Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10 – 3.14 | Core backend programming |
| **Data Manipulation** | Pandas & NumPy | Vectorized data processing and transformation |
| **Machine Learning** | Scikit-learn | Regressors, ColumnTransformer, StandardScaler, OneHotEncoder |
| **Model Serialization** | Joblib | Atomic storage and loading of the fitted ML pipeline |
| **Web Framework** | Streamlit | Responsive, stateful web interface with custom CSS |
| **Visualization** | Matplotlib & Seaborn | High-resolution analytical charts and diagram generation |
| **Embedded Database** | SQLite (sqlite3) | Serverless, local persistence of valuation queries |

---

## 📊 Dataset Information & Preprocessing

The model is trained on a merged, cleaned dataset of **31,103 authentic residential property listings** collated across six Indian metropolitan centres:
- **Mumbai**: 7,258 properties
- **Bangalore**: 6,001 properties
- **Kolkata**: 5,774 properties
- **Chennai**: 4,855 properties
- **Delhi NCR**: 4,706 properties
- **Hyderabad**: 2,509 properties

### Data Cleaning Steps:
1. **Missing Amenity Imputation**: Raw values where amenities were unrecorded (`9`) were imputed through contextual heuristics (e.g. area $\ge 1,100$ sqft or 3+ BHK indicating allocated parking).
2. **Outlier Filtering**: Truncated non-physical records (Area between 300 and 8,000 sq ft; BHK between 1 and 6; Price between ₹15 Lakhs and ₹15 Crores; Rate per sqft between ₹2,000 and ₹50,000).
3. **Locality Frequency Thresholding**: Over 1,700 raw locality names were processed; localities with $\ge 12$ listings were retained as discrete categories, while sparse tail localities were mapped to `"Other"` to eliminate dimensionality explosion and overfitting.
4. **Feature Scaling & One-Hot Encoding**: Handled via `ColumnTransformer` with `StandardScaler` on numerical features and `OneHotEncoder(handle_unknown='ignore')` on categorical features.

---

## 🧠 Machine Learning Pipeline & Model Comparison

Four candidate regression algorithms were trained on 80% of the dataset (**24,882 samples**) and tested on the remaining 20% held-out partition (**6,221 samples**):

1. **Linear Regression (OLS)**: Fits a linear hyperplane minimizing the sum of squared errors. Fast and interpretable, but unable to model non-linear price appreciation in luxury micro-markets.
2. **Decision Tree Regressor (CART)**: Recursively splits the feature space into orthogonal rectangles. Highly prone to overfitting and high variance on continuous price targets.
3. **Random Forest Regressor**: Ensemble of 100 decorrelated decision trees using bootstrap aggregation (bagging) and randomized feature subspace sampling.
4. **Gradient Boosting Regressor (GBDT - Champion)**: Sequentially constructs 110 depth-limited decision trees ($max\_depth=5$, $learning\_rate=0.1$, $subsample=0.85$) where each new tree fits the pseudo-residuals of the existing ensemble via gradient descent.

---

## 📈 Empirical Model Evaluation Results

Evaluation performed strictly on the held-out 20% test set (6,221 samples). Metrics are computed directly from model predictions:

| Model Name | R² Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Training Duration |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.4763 | ₹42,92,158.05 | ₹78,72,566.50 | 3.07 s |
| **Decision Tree Regressor** | 0.4459 | ₹44,29,634.21 | ₹80,97,987.97 | 0.90 s |
| **Random Forest Regressor** | 0.4901 | ₹42,45,488.12 | ₹77,68,423.56 | 10.87 s |
| **Gradient Boosting Regressor 🏆** | **0.5012** | **₹43,13,640.66** | **₹76,83,150.17** | **48.50 s** |

*Champion Model*: **Gradient Boosting Regressor** selected for production deployment and packaged as `best_model_pipeline.joblib`.

---

## 📐 System Architecture & Diagrams

### High-Resolution Diagrams Available in `images/`:
- **`system_architecture.png`**: 4-Tier decoupled architecture (Presentation, Application, ML Inference, Persistence).
- **`project_flowchart.png`**: Complete end-to-end data preparation, model training, and user interaction lifecycle.
- **`uml_usecase.png`**: UML Use Case diagram mapping buyer, seller, and real-estate agent interactions.
- **`uml_class.png`**: UML Class diagram detailing `PropertyInput`, `HousePricePipeline`, `DatabaseHandler`, and `HomeValueApp`.
- **`uml_sequence.png`**: UML Sequence diagram detailing real-time inference and database logging transactions.

---

## 🗄️ Database Schema & Persistence

All predictions are persistently stored in an embedded SQLite database (`database/predictions.db`).

### Table Schema: `predictions`
```sql
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
);
```

---

## 📂 Project Folder Structure

```
house_price_prediction/
│
├── app.py                      # Main Streamlit web application
├── train_model.py              # ML training, evaluation, comparison, and serialization
├── test_project.py             # Automated unit and integration test suite
├── requirements.txt            # Python package dependencies
├── README.md                   # Complete project documentation and viva guide
│
├── dataset/
│   ├── prepare_dataset.py      # Cleans and standardizes raw city records
│   ├── housing_data.csv        # Cleaned dataset (31,103 records across 6 cities)
│   └── raw_cities/             # Original raw CSV files (Delhi, Bangalore, etc.)
│
├── models/
│   ├── best_model_pipeline.joblib # Serialized Scikit-learn Pipeline
│   ├── model_metrics.json         # Real empirical metrics for all 4 models
│   ├── locality_mapping.json      # City-to-localities mapping dictionary
│   └── test_sample_predictions.json # Evaluation sample slice for analytics
│
├── database/
│   ├── db_handler.py           # SQLite connection, logging, retrieval, export
│   └── predictions.db          # Embedded SQLite database
│
├── images/
│   ├── generate_diagrams.py    # Script to regenerate all diagrams
│   ├── system_architecture.png # System Architecture diagram
│   ├── project_flowchart.png   # ML pipeline flowchart
│   ├── uml_usecase.png         # UML Use Case diagram
│   ├── uml_class.png           # UML Class diagram
│   └── uml_sequence.png        # UML Sequence diagram
```

---

## 🚀 Installation & Execution Guide

### 1. Prerequisites
Ensure Python 3.10 or higher is installed:
```powershell
python --version
```

### 2. Navigate to the Project Directory
```powershell
cd "c:\Users\ASUS\OneDrive\Desktop\Machine learning projrct\house_price_prediction"
```

### 3. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 4. (Optional) Re-train Machine Learning Models
*Note: Pre-trained serialized models are already included in `models/`.*
```powershell
python train_model.py
```

### 5. Run the Automated Test Suite
```powershell
python test_project.py
```

### 6. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
*The application will open in your default browser at `http://localhost:8501`.*

---

## 🎓 Comprehensive Viva Questions & Answers

### Q1: What is the primary objective of the HomeValue AI project?
**Answer**: To build an automated, transparent, end-to-end machine learning web application that predicts residential property prices across major Indian metropolitan cities based on physical and geographic parameters, providing statistical confidence intervals and persistent database logging.

### Q2: Why did Gradient Boosting Regressor outperform Linear Regression and Random Forest?
**Answer**: Real-estate prices exhibit strong non-linear relationships (e.g. sharp price spikes in prime micro-markets and diminishing returns on super built-up area). Linear Regression assumes strict linearity and fails on localized price cliffs. While Random Forest reduces variance through bagging, Gradient Boosting sequentially optimizes pseudo-residuals along the steepest gradient direction, achieving a higher $R^2$ (0.5012) and lower RMSE (₹76.8L) on the held-out test data.

### Q3: What is the significance of $R^2$ Score and RMSE in real estate?
**Answer**: $R^2$ (Coefficient of Determination) quantifies the proportion of price variance explained by the features ($50.12\%$ of market price variance is explained by our model). RMSE measures the standard deviation of residuals in original currency units (Indian Rupees, ₹), heavily penalizing large valuation errors compared to MAE.

### Q4: How did you handle high-cardinality localities without overfitting?
**Answer**: With over 1,700 unique localities, simple one-hot encoding would create extreme matrix sparsity. We implemented frequency thresholding: localities with $\ge 12$ listings were retained as discrete classes, while low-frequency entries were aggregated into `"Other"`. Additionally, `OneHotEncoder(handle_unknown='ignore')` ensures unseen localities at inference gracefully map to zero vectors.

### Q5: How are the prediction intervals calculated?
**Answer**: Residual standard deviation ($\sigma \approx \text{₹76.8 Lakhs}$) was calculated on the held-out test set. For any prediction $\hat{y}$, the $90\%$ confidence interval is computed as $[\hat{y} - 1.645 \cdot \sigma_{adj}, \hat{y} + 1.645 \cdot \sigma_{adj}]$, communicating valuation uncertainty to the user.

### Q6: Why is Scikit-learn Pipeline utilized?
**Answer**: A `Pipeline` encapsulates both preprocessing transformers (`StandardScaler`, `OneHotEncoder`) and the regressor into a single atomic object. This strictly prevents data leakage between training and testing folds and ensures runtime inputs are automatically transformed with identical statistics.

### Q7: Why did you choose SQLite over external RDBMS engines like MySQL or PostgreSQL?
**Answer**: SQLite is serverless, zero-configuration, and embeds directly within the application process. It provides full ACID compliance and rapid read/write performance without requiring external background daemons, making it ideal for self-contained college projects and desktop deployment.

### Q8: What measures were taken to protect user privacy?
**Answer**: By design, the application collects zero personally identifiable information (PII)—no names, contact numbers, IP addresses, or email addresses are requested or logged. The database solely stores technical property specifications and valuation outputs.

---

## 📜 Ethical Disclaimer & Licensing

**Academic Project Disclaimer**: HomeValue AI was developed as a BTech 3rd Year Computer Science & Engineering Major Project. Predictions generated by the system represent statistical estimates based on historical listings and do not constitute certified legal, banking, or financial property valuations. 

Licensed under the **MIT License**.
