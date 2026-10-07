import os
import pandas as pd
import numpy as np
joblib_installed = True
try:
    import joblib
except ImportError:
    joblib_installed = False

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate():
    # 1. Load training data
    data_path = os.path.join("data", "student_performance_2.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "student_performance.csv")
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset from {data_path} with shape {df.shape}")

    # 2. Define features (X) and target (y)
    # Drop 'ID' and 'FinalExamScore' from features
    target_col = "FinalExamScore"
    drop_cols = ["ID", target_col]
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target_col]

    # Identify numeric features
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print(f"Numeric features to process: {numeric_features}")

    # 3. Build Preprocessing Pipeline using ColumnTransformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handle missing values safely
        ('scaler', StandardScaler())                    # Normalize features for stability
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )

    # 4. Create Full Pipeline with a baseline model (Linear Regression)
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # 5. Train-Test Split (80% train, 20% test for internal evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Fit and Evaluate Linear Regression
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)

    print("\n" + "="*40)
    print("BASELINE MODEL: Linear Regression")
    print("="*40)
    print(f"RMSE: {rmse_lr:.4f}")
    print(f"R2 Score: {r2_lr:.4f}")

    # 7. Try a more powerful model: Random Forest Regressor
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)

    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print("\n" + "="*40)
    print("ADVANCED MODEL: Random Forest Regressor")
    print("="*40)
    print(f"RMSE: {rmse_rf:.4f}")
    print(f"R2 Score: {r2_rf:.4f}")

    # 8. Save the best model artifact using joblib
    best_pipeline = rf_pipeline if rmse_rf < rmse_lr else lr_pipeline
    model_name = "Random Forest" if rmse_rf < rmse_lr else "Linear Regression"
    
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_model.pkl")
    
    if joblib_installed:
        joblib.dump(best_pipeline, model_path)
        print(f"\nSuccessfully saved the best model ({model_name}) to {model_path}!")
    else:
        print("\nJoblib is not installed, skipping model save.")

if __name__ == "__main__":
    train_and_evaluate()