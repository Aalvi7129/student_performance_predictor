# 🎓 Student Performance Predictor

An end-to-end machine learning pipeline and interactive web application designed to predict student final exam scores based on study habits, attendance, and academic history. Developed as part of a technical challenge.


## 📂 Project Structure

```text
student-performance-predictor/
├── data/
│   ├── student_performance.csv       # Training dataset with target variable
│   └── student_performance_test.csv    # Unseen test records for batch inference
├── models/
│   └── best_model.pkl                  # Serialized Scikit-Learn pipeline artifact
├── src/
│   ├── audit.py                        # Data audit and validation script
│   ├── train.py                        # Training pipeline script (Linear Regression vs. Random Forest)
│   └── predict.py                      # CLI batch prediction script
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Project dependencies
├── app.py                              # Interactive Streamlit web application
└── README.md                           # Project documentation

🛠️ Setup & Installation

1.Clone the Repository:
  git clone [https://github.com/Aalvi7129/student_performance_predictor.git](https://github.com/Aalvi7129/student_performance_predictor.git)
  cd student-performance-predictor
2.Create and Activate a Virtual Environment:
  python -m venv venv
  # On Windows (PowerShell):
  .\venv\Scripts\Activate
3.Install Dependencies:
  pip install -r requirements.txt

🚀 Usage Guide

1. Train the Model
Run the training script to preprocess data, evaluate models (Linear Regression vs. Random Forest), and serialize the best pipeline artifact to models/best_model.pkl:
  python src/train.py

2.Run Batch Predictions
Generate predictions for unseen test records and export a submission file:
  python src/predict.py --input data/student_performance_test.csv --output submissions.csv

3.Launch the Interactive Web App
Explore real-time predictions using the Streamlit dashboard:
  streamlit run app.py

📊 Model Performance
Selected Best Model: Gradient Boosting Regressor
Evaluation Metrics:
  RMSE: 7.29 ± 0.77
  R² Score: 0.74 ± 0.05