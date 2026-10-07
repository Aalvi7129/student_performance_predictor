import os
import argparse
import pandas as pd
import joblib

def make_predictions(input_path, output_path):
    # 1. Load the model artifact
    model_path = os.path.join("models", "best_model.pkl")
    if not os.path.exists(model_path):
        print(f"Error: Model artifact not found at {model_path}. Please run train.py first!")
        return

    print(f"Loading trained model from {model_path}...")
    pipeline = joblib.load(model_path)

    # 2. Load input test data
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}!")
        return

    print(f"Loading input data from {input_path}...")
    df_test = pd.read_csv(input_path)
    
    # Keep track of IDs for final submission
    if "ID" in df_test.columns:
        student_ids = df_test["ID"]
    else:
        student_ids = pd.Series(range(len(df_test)))

    # 3. Drop ID if present for prediction
    X_inference = df_test.drop(columns=["ID"], errors="ignore")

    # 4. Generate predictions using the pipeline
    print("Generating predictions...")
    predictions = pipeline.predict(X_inference)

    # 5. Create submission dataframe
    submission_df = pd.DataFrame({
        "ID": student_ids,
        "FinalExamScore": predictions
    })

    # 6. Save output CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    submission_df.to_csv(output_path, index=False)
    print(f"Success! Predictions saved to {output_path}")
    print(submission_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction CLI for Student Performance Predictor")
    parser.add_argument("--input", type=str, default="data/student_performance_test.csv", help="Path to input test CSV")
    parser.add_argument("--output", type=str, default="submission.csv", help="Path to save output predictions CSV")
    
    args = parser.parse_args()
    make_predictions(args.input, args.output)