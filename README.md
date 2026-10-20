# Student Performance Prediction

Predicting students' final math grade from the UCI Student Performance dataset, using both regression (exact score) and classification (pass/fail).

**Live demo:** [student-performance-app.streamlit.app](https://student-performance-app-hbpkxgdplcam4xody6cara.streamlit.app/)

## Problem statement

Given a student's demographic, family, and school-related information, can we predict their final grade (**G3**, 0-20)? The project frames this two ways:

1. **Regression:** predict the exact G3 score.
2. **Classification:** predict **Pass / Fail** (pass = G3 >= 10).

There are two modelling goals:

- **Option A (with G1/G2):** the most accurate model, once earlier period grades exist.
- **Option B (without G1/G2):** an early-warning model that flags at-risk students before any grades exist.

## Dataset

- **Source:** UCI Machine Learning Repository, *Student Performance* dataset (Cortez & Silva, 2008), math course file (`student-mat`).
- **Size:** 395 students, 33 raw columns.
- **Target:** `G3` (final grade, 0-20). Class balance for Pass/Fail: 265 pass, 130 fail.

## Project structure

 Student's performace project.ipynb   # full analysis and modelling
 encoding_per_column.py               # per-column encoding (fit + transform)
 app.py                               # Streamlit app (to be added)
 student-mat-clean.csv                # input data
 student_mat_processed.csv            # encoded data
 requirements.txt
 README.md


## Preprocessing

- **One-hot encoding**, one `OneHotEncoder` per column: `school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian`. One reference dummy is dropped per group to avoid the dummy variable trap.
- **Label encoding** (no=0, yes=1) for `schoolsup, famsup, paid, activities, higher, internet, romantic, nursery`, stored as `en_*`.
- **Outliers:** `absences` capped at the IQR upper fence (Q3 + 1.5 x IQR).
- **Scaling:** `StandardScaler`, fit on the training set only, for Lasso, Ridge, Logistic Regression, and scaled KNN.
- **Split:** 80/20 train/test, `random_state=42`.
- All fitted encoders are kept in a dictionary and saved with `joblib`, so new input in the app is encoded exactly like the training data.

## Results

### Regression (test set)

| Model | R2 | MAE | RMSE | MSE |
| Simple linear (G1 only) | 69.9% | 1.86 | 2.48 | 6.16   |
| Multiple linear | 72.6% | 1.67 | 2.37 | 5.62  |
| Polynomial (degree 2, G1) | 69.9% | 1.86 | 2.48 | 6.17 |
| Ridge (alpha=0.5) | 72.6% | 1.67 | 2.37 |  5.61  |
| **Lasso (alpha=0.5)** | **79.2%** | **1.20** | **2.06** | **4.63** |

**Best: Lasso.** Polynomial features add nothing over the simple model, so G1 vs G3 is essentially linear.

### Classification (test set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (all features) | 91.1% | 95.9 | 90.4 | 93.1 |
| Logistic Regression (no G1/G2) | 70.9% | 73.8 | 86.5 | 79.6 |
| Gaussian Naive Bayes | 84.8% | 88.5 | 88.5 | 88.5 |
| Multinomial Naive Bayes | 86.1% | 87.3 | 92.3 | 89.7 |
| Bernoulli Naive Bayes | 65.8% | 68.7 | 88.5 | 77.3 |
| Decision Tree (max_depth=2) | 91.1% | 97.9 | 88.5 | 92.9 |
| KNN (scaled, k=18) | 72.2% | 71.4 | 96.2 | 82.0 |
| KNN (unscaled, k=7) | 88.6% | 93.9 | 88.5 | 91.1 |
| KNN (G1+G2 only, scaled, k=12) | 91.1% | 100.0 | 86.5 | 92.8 |
| **SVM (linear)** | **92.4%** | 97.9 | 90.4 | **94.0** |

**Best: linear SVM**, with logistic regression and the shallow decision tree close behind.

## Key findings

- **G1 and G2 carry most of the predictive signal.** Removing them drops logistic regression accuracy from 91% to 71%.
- **Regularization helps regression:** Lasso beats plain linear regression by about 7 R2 points.
- **Unscaled KNN beat scaled KNN.** The grade columns have larger numeric ranges, so they dominate the distance and act as implicit feature weighting.
- **Option B is the more useful real-world model.** Its lower accuracy is the honest cost of predicting before any grades exist.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the notebook
jupyter notebook "Student's performace project.ipynb"

# 3. Run the Streamlit app (once app.py is added)
streamlit run app.py
```

`requirements.txt`:

```
pandas
numpy
scikit-learn
seaborn
matplotlib
joblib
streamlit
```

## Limitations

- Hyperparameters (k, alpha, max_depth) were chosen using test-set scores. Cross-validation (`GridSearchCV`) would give unbiased estimates.
- The test set has only 79 rows, so one or two students shift accuracy by about 1.3 points.
- `class_weight='balanced'` was used for only the first logistic model, so comparisons are not strictly like-for-like.
- The data comes from two Portuguese schools, so results may not generalize elsewhere.

## Author

**Hishmat Rai**
GitHub: [https://student-performance-app-hbpkxgdplcam4xody6cara.streamlit.app/] | LinkedIn: [www.linkedin.com/in/hishmat-rai-361919378]


## Acknowledgements

P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, 2008.
