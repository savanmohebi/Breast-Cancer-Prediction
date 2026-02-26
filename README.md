# 🎗️ Breast Cancer Diagnosis Prediction
### Interpretable Machine Learning & Threshold Optimization for Tumor Classification

## 📌 Project Overview
In the medical field, early and accurate diagnosis of breast cancer is critical for patient survival and effective treatment planning. This project leverages **Machine Learning** to classify breast tumors as benign (Class $0$) or malignant (Class $1$) based on their physical and cellular characteristics.

The goal is to provide a highly accurate diagnostic tool that is completely transparent, moving beyond "black-box" AI by explicitly extracting the mathematical weights of the features driving the diagnosis. Furthermore, this project explores **Recall Optimization** and deep **Error Analysis** to understand biological edge cases where standard algorithms fail.

## 🚀 Key Features
- **Interpretable AI (White-Box Modeling):** Extracted and mapped the learned coefficients of the Logistic Regression model to pinpoint exactly which tumor attributes are the strongest indicators of malignancy.
- **Threshold Tuning for Recall:** Adjusted the decision threshold from the default $0.5$ to $0.3$ to minimize False Negatives, prioritizing patient safety over raw accuracy.
- **Deep Error Analysis:** Investigated the "Wolf in Sheep's Clothing" phenomenon—analyzing specific patients where the model was highly confident but incorrect, revealing the limitations of linear models on atypical biological tumors.
- **Robust Preprocessing:** Utilized `SimpleImputer(strategy='mean')` to safely handle missing medical data without dropping valuable patient records.
- **Mathematical Scaling:** Implemented `StandardScaler()` to normalize feature magnitudes, a crucial mathematical step ensuring that large measurements (like area) do not unfairly overpower smaller measurements (like smoothness) when calculating weights.
- **Pipeline Architecture:** Built a robust Scikit-Learn `Pipeline` ensuring a leak-free, streamlined workflow from raw data ingestion to final prediction.

## 🛠️ Technologies & Tools
- **Python** (NumPy, Pandas)
- **Scikit-Learn** (Pipeline, SimpleImputer, StandardScaler, LogisticRegression)
- **Evaluation Metrics** (Confusion Matrix, Classification Report)
- **Environment:** Jupyter Notebook

## 📊 Methodology
1. **Data Preprocessing:** Handled missing data points using mean imputation to ensure a complete and reliable dataset.
2. **Feature Standardization:** Scaled all physical measurements so that the algorithm evaluates all features on a level playing field.
3. **Model Training:** Trained a Binary Classifier (`LogisticRegression`) relying on the linear equation of weights and biases.
4. **Dynamic Feature Extraction:** Extracted `get_feature_names_out()` directly from the pipeline stages to accurately map the model's output weights to their corresponding real-world biological names.
5. **Probability Extraction & Thresholding:** Bypassed default predictions to extract raw `.predict_proba()` values, enabling custom thresholding to prioritize catching malignant cases.

## 📈 Results
The baseline model achieved excellent predictive performance on the unseen test dataset ($114$ samples).

### Baseline Performance (Threshold = $0.5$)
- **Overall Accuracy:** $96\%$
- **Robust Detection (Class 1 - Malignant):** Achieved a Precision of $97\%$ and a Recall of $93\%$.
- **Confusion Matrix:** Correctly diagnosed $71$ Benign and $39$ Malignant cases, with only $4$ total misclassifications ($3$ False Negatives, $1$ False Positive).
- **Top Driving Features:** By analyzing the model's coefficients, the top 3 most critical indicators for diagnosis were identified as:
  1. `texture_worst` (Weight: $\approx 1.434$)
  2. `radius_se` (Weight: $\approx 1.230$)
  3. `symmetry_worst` (Weight: $\approx 1.060$)

### 🔬 Error Analysis & Optimization (Threshold = $0.3$)
In clinical settings, $3$ False Negatives (missed cancer cases) is extremely risky. 
- **The Fix:** By lowering the prediction threshold to $0.3$, the model becomes more sensitive to slight irregularities, successfully reducing the False Negative rate at the acceptable cost of slightly higher False Positives.

- **The "Wolf in Sheep's Clothing":** We isolated a False Negative patient to whom the model gave only a $\approx 6\%$ ($0.0614$) probability of malignancy. Comparing this patient's features to dataset averages proved the tumor was biologically atypical—it was tiny and smooth, perfectly mimicking the mathematical dimensions of a benign tumor, thus tricking the linear logic of the algorithm.

## 🩺 Medical Impact
This model demonstrates how ML can be integrated into healthcare and clinical workflows to:
- Provide oncologists and pathologists with a fast, highly accurate "second opinion".
- Build trust with medical professionals by offering complete transparency into *why* the AI made its decision, ranked by biological feature importance.
- Highlight the absolute necessity of using AI as an **assistive tool** rat a standalone diagnostician, as rare biological anomalies can bypass purely mathematical thresholds.

## 📊 Data Visualization & Error Analysis

To truly understand the model's performance beyond just raw numbers, we designed several visual plots. These graphics help demystify the "black box" of the machine learning model and explain exactly why certain critical errors (False Negatives) occurred.

### 1. The Confusion Matrix Heatmap
We visualized the model's predictions using a Seaborn heatmap. 
* **The Bright Spots (Success):** The dark blue diagonal squares show our True Negatives ($71$) and True Positives ($39$). 
* **The Blind Spots (Errors):** The lighter off-diagonal squares immediately draw the eye to our model's mistakes, specifically the **$3$ False Negatives**. These represent $3$ actual malignant tumors that the AI incorrectly labeled as benign. 

### 2. The Scatter Plot: "Hiding in Plain Sight"
To investigate *why* the Logistic Regression model missed these 3 patients, we isolated one specific False Negative (Patient Index 73) and plotted them against the entire dataset.
* We mapped **Tumor Radius** (`radius_mean`) on the X-axis and **Shape Irregularity** (`concavity_mean`) on the Y-axis.
* **The Finding:** Patient 73 (marked with a giant red star) sits entirely engulfed within the blue cluster of Benign tumors. Because Logistic Regression draws a straight mathematical line to separate classes, this visual perfectly proves why the linear math was tricked by this specific tumor's geometry.

### 3. The KDE Density Plot: The Statistical Overlap
To further validate this anomaly, we designed a Kernel Density Estimation (KDE) plot to show the "bell curves" of Tumor Area (`area_mean`) for both classes.
* **Benign Peak:** Tends to center around smaller areas.
* **Malignant Peak:** Tends to center around much larger areas.
* **The Finding:** We drew a vertical line representing Patient 73's tumor area ($584.1$). This line falls directly under the peak of the Benign distribution curve. Statistically and mathematically, this tumor possessed the exact physical traits of a benign mass, representing a rare "atypical" malignant tumor that linear models struggle to catch without threshold adjustments.

---
*Created by [Savan Mohebi] - Civil Engineer & ML Enthusiast*

