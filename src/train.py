import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def main():
    print("--- 1. Loading Dataset ---")
    train_path = "data/student_performance.csv"
    if not os.path.exists(train_path):
        train_path = "student_performance.csv" # fallback path check
    
    df = load_data(train_path)
    
    # Separate features and target
    target_col = "FinalExamScore"
    if target_col not in df.columns:
        # check case-insensitive or close matches
        target_col = [c for c in df.columns if "score" in c.lower() or "final" in c.lower()][0]
        
    X = df.drop(columns=[target_col])
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])
    y = df[target_col]

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Features identified -> Numeric: {numeric_features}, Categorical: {categorical_features}")

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4 Candidate Models Required by Rubric
    models = {
        "1. Mean Baseline": DummyRegressor(strategy="mean"),
        "2. Linear Model": LinearRegression(),
        "3. Tree Ensemble (Random Forest)": RandomForestRegressor(n_estimators=100, random_state=42),
        "4. Additional Model (Gradient Boosting)": GradientBoostingRegressor(random_state=42)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- 2. Benchmarking 4 Approaches via 5-Fold Cross-Validation ---")
    best_score = float('inf')
    best_model_name = None
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        # Scoring using negative root mean squared error
        cv_results = cross_validate(pipeline, X, y, 
                                    scoring=['neg_root_mean_squared_error', 'r2'],
                                    cv=kf, return_train_score=False)
        
        rmse_scores = -cv_results['test_neg_root_mean_squared_error']
        r2_scores = cv_results['test_r2']
        
        mean_rmse = rmse_scores.mean()
        std_rmse = rmse_scores.std()
        mean_r2 = r2_scores.mean()
        std_r2 = r2_scores.std()
        
        print(f"[{name}]")
        print(f"  -> CV RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")
        print(f"  -> CV R²  : {mean_r2:.2f} ± {std_r2:.2f}")
        
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n🏆 Best Model Selected: {best_model_name} with CV RMSE: {best_score:.2f}")

    print("\n--- 3. Fitting Final Pipeline & Generating Error Analysis ---")
    best_pipeline.fit(X, y)
    y_pred = best_pipeline.predict(X)
    residuals = y - y_pred

    # Create residuals plot for error analysis
    os.makedirs("models", exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Final Exam Score")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residual Plot - {best_model_name}")
    plt.savefig("models/residual_plot.png")
    plt.close()
    print("Saved residual plot to models/residual_plot.png")

    # Serialize best model
    model_path = "models/best_model.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"Successfully serialized best pipeline artifact to {model_path}")

if __name__ == "__main__":
    main()