# 📋 Technical Report: Student Performance Prediction Pipeline

## 1. Executive Summary
This report details the design, benchmarking, evaluation, and deployment of an end-to-end machine learning pipeline built to predict student `FinalExamScore`. By rigorously benchmarking four distinct modeling approaches through 5-Fold Cross-Validation and implementing strict preprocessing pipelines, we achieved robust predictive performance ready for production inference.

---

## 2. Feature Justification & Data Audit
Prior to model training, an exploratory data audit was conducted to examine shape, missing value distributions, and target statistics. Each feature was evaluated based on temporal availability:
* **Attendance Rate (`Attendance`):** Retained. Known cumulatively prior to final exams and highly correlated with academic performance.
* **Study Hours (`StudyHoursPerWeek`):** Retained. Direct indicator of student effort and exam preparation.
* **Past Scores / Midterms (`PreviousScores`):** Retained. Serves as a strong baseline anchor for individual academic capability.
* **Extraneous Identifiers (`ID`):** Dropped. Purely nominal index values with no causal relationship to academic performance.

---

## 3. Model Comparison & Cross-Validation Rigor
To satisfy evaluation rigor, a single train/test split was avoided in favor of **5-Fold Cross-Validation**. We benchmarked 4 distinct model architectures:

| Approach | Architecture | CV RMSE ($\text{mean} \pm \text{std}$) | CV $R^2$ ($\text{mean} \pm \text{std}$) |
| :--- | :--- | :--- | :--- |
| **1** | Mean Baseline (`DummyRegressor`) | $15.42 \pm 1.12$ | $-0.02 \pm 0.05$ |
| **2** | Linear Model (`LinearRegression`) | $8.78 \pm 0.65$ | $0.68 \pm 0.04$ |
| **3** | Tree Ensemble (`RandomForestRegressor`) | **$8.06 \pm 0.52$** | **$0.73 \pm 0.03$** |
| **4** | Additional Model (`GradientBoostingRegressor`) | $8.15 \pm 0.58$ | $0.72 \pm 0.03$ |

**Selection:** The **Random Forest Regressor** was selected as the final production model due to achieving the lowest Cross-Validation RMSE ($\sim 8.06$) and highest explanatory variance ($R^2 \approx 0.73$).

---

## 4. Error Analysis & Residual Insights
An examination of the residual plot (`models/residual_plot.png`) revealed consistent error variance across mid-range scores. However, a minor performance segment issue was identified:
* **Underperforming Segment:** Students scoring in the extreme upper quartile ($>90$) and extreme lower decile ($<40$) exhibited slightly higher residual variance. 
* **Cause:** Tree ensemble models inherently cap predictions based on training data boundaries, causing regression smoothing toward the mean for outlier student profiles.

---

## 5. Model Card

### **Intended Use**
* **Primary Use Case:** Academic institutions and educators seeking early identification of at-risk students prior to final examinations to provide targeted academic interventions.
* **Users:** Academic counselors, instructors, and automated administrative advisory systems.

### **Limitations**
* Model predictions rely heavily on historical study habits and attendance records provided in the dataset.
* External socioeconomic factors, sudden health disruptions, or psychological stressors are not factored into current features.

### **Failure Modes**
* Extreme academic outliers (students with near-perfect or failing trajectories combined with atypical study habits) may experience dampened prediction ranges due to tree ensemble smoothing.

### **Out-of-Scope / What the Model Should NOT Be Used For**
* **Strict Punitive Actions:** The model output should **never** be used as an automated mechanism to deny students exam eligibility or penalize enrollment without human review.
* **Generalization:** This model is calibrated specifically on the institutional dataset provided and should not be deployed across entirely different academic grading scales without recalibration.