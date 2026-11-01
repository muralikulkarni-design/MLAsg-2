# Student Outcome Classifier – ML Assignment 2

M.Tech (AIML), BITS WILP – Machine Learning, Assignment 2
Murali Kulkarni

- **GitHub repo:** https://github.com/muralikulkarni-design/MLAsg-2
- **Live Streamlit app:** _<add link after deploying on Streamlit Community Cloud>_

---

## a. Problem statement

Universities lose a lot of students in the first couple of years, and by the time a
student formally drops out it is usually too late to help. The aim here is to predict,
from what is known at enrolment plus the first two semesters of results, which of three
outcomes a student is heading towards:

| Class | Meaning |
|---|---|
| **Dropout** | left the course before finishing |
| **Enrolled** | still enrolled after the normal duration of the course |
| **Graduate** | finished in the normal duration |

This is a **multi-class classification** problem. Five classifiers (Logistic Regression,
Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same data, compared on
six metrics, and served through a Streamlit app where a test CSV can be uploaded and
scored with any of the saved models.

## b. Dataset description

**Predict Students' Dropout and Academic Success** – UCI Machine Learning Repository, dataset id 697
(Realinho, Vieira Martins, Machado & Baptista, Polytechnic Institute of Portalegre, Portugal, 2021).
Source: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
License: CC BY 4.0

| Property | Value |
|---|---|
| Instances | 4,424 students |
| Features | 36 (+ 1 target column `Target`) |
| Missing values | none |
| Classes | Graduate 2,209 (49.9%) · Dropout 1,421 (32.1%) · Enrolled 794 (17.9%) |

The features fall into a few groups:

- **Demographic** – marital status, nationality, gender, age at enrolment, displaced, international
- **Socio-economic** – mother's/father's qualification and occupation, debtor, tuition fees up to date, scholarship holder
- **Academic at entry** – application mode and order, course, day/evening attendance, previous qualification and its grade, admission grade
- **Academic performance** – curricular units credited / enrolled / evaluated / approved / grade / without evaluation, for the 1st and the 2nd semester (12 columns)
- **Macro-economic** – unemployment rate, inflation rate, GDP at the time of enrolment

Notes on preparation (see `train_models.py`):

- The raw file is `;` separated and one header has a trailing tab; column names are stripped.
- Nine columns are integer *codes* rather than quantities (course, application mode,
  parents' occupation, etc.). These are one-hot encoded – codes seen fewer than 10 times are
  grouped into an "infrequent" bucket. The remaining 27 columns are standardised.
  Both steps live inside each saved sklearn `Pipeline`, so the app can feed raw columns in.
- Stratified **80/20 train/test split** (`random_state=26`): 3,539 rows for training, 885
  for testing. The test part is saved as `test_data.csv` and is what the app uses by default.
- The classes are imbalanced (Enrolled is only 18%), so the grid search optimises
  **macro F1** with 5-fold stratified CV, and Precision / Recall / F1 below are **macro averaged**.
  AUC is the macro one-vs-rest ROC AUC.

## c. GitHub repository link

https://github.com/muralikulkarni-design/MLAsg-2

```
MLAsg-2/
├── app.py                  # Streamlit app (loads saved models, no retraining)
├── train_models.py         # preprocessing, grid search, evaluation, saving models
├── requirements.txt
├── README.md
├── test_data.csv           # held-out 20% test split (36 features + Target)
├── data/
│   └── students_raw.csv    # original UCI file
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   ├── metrics.csv         # comparison table below
│   └── model_info.json     # feature list, classes, best params, confusion matrices
└── .streamlit/config.toml  # theme
```

## d. Models used

All five models are sklearn pipelines of *(one-hot + scaling) → classifier*, tuned with
`GridSearchCV` on the training set only. Hyper-parameters chosen:

| Model | Best parameters |
|---|---|
| Logistic Regression | C = 5.0 (multinomial, lbfgs) |
| Decision Tree | criterion = gini, max_depth = 6, min_samples_leaf = 1 |
| KNN | k = 5, weights = distance |
| Naive Bayes (Gaussian) | var_smoothing = 0.01 |
| Support Vector Machine | RBF kernel, C = 3, gamma = scale, isotonic-calibrated probabilities |

### Comparison table (test set, 885 students)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7763 | **0.8859** | 0.7317 | 0.6957 | 0.7075 | 0.6294 |
| Decision Tree | 0.7322 | 0.8410 | 0.6877 | 0.6589 | 0.6671 | 0.5610 |
| KNN | 0.7186 | 0.8089 | 0.6676 | 0.6230 | 0.6344 | 0.5316 |
| Naive Bayes | 0.6034 | 0.7751 | 0.5869 | 0.5778 | 0.5686 | 0.3930 |
| Support Vector Machine | **0.7842** | 0.8740 | **0.7388** | **0.6976** | **0.7099** | **0.6425** |

Recall per class on the test set (from the confusion matrices) helps explain the numbers:

| Model | Dropout (284) | Enrolled (159) | Graduate (442) |
|---|---|---|---|
| Logistic Regression | 73.6% | 42.1% | 93.0% |
| Decision Tree | 65.1% | 43.4% | 89.1% |
| KNN | 65.1% | 30.8% | 91.0% |
| Naive Bayes | 54.6% | 51.6% | 67.2% |
| Support Vector Machine | 76.4% | 39.0% | 93.9% |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very close second overall (F1 0.708, MCC 0.629) and the **best AUC (0.886)**, i.e. its probabilities rank students best. The strongest signals – approved units and grades in the 1st/2nd semester, tuition fees up to date – are roughly monotonic, so a linear boundary already captures most of the problem. It wanted weak regularisation (C = 5) and trains in a fraction of the SVM's time, and its coefficients are easy to explain. |
| Decision Tree | Mid table (F1 0.667, MCC 0.561). The CV picked a fairly shallow tree (depth 6); deeper trees overfit the training folds. The top splits are on 2nd-semester approved units and tuition fees, which makes the model very readable, but the axis-aligned, step-like boundaries are too coarse for the grade features, so it misses more dropouts (65%) than LR or SVM. Its AUC (0.841) suffers because leaf probabilities take only a few distinct values. |
| KNN | Weakest of the "reasonable" models (F1 0.634). After one-hot encoding there are 130 input dimensions, many of them sparse dummies, so Euclidean distance gets noisy. It leans heavily towards the majority class: Graduate recall is 91% but only 31% of Enrolled students are recognised – the lowest of all models. Distance weighting with small k worked best, which hints that only very local neighbourhoods are informative. |
| Naive Bayes | Lowest scores on almost every metric (accuracy 0.603, MCC 0.393). The independence assumption is clearly broken here – the 1st- and 2nd-semester columns are strongly correlated with each other – and one-hot / count features are not Gaussian, so it double counts evidence. Interestingly it has the **highest Enrolled recall (52%)**, but only because it predicts "Enrolled" far too often (it calls 93 graduates and 100 dropouts "Enrolled"). |
| Support Vector Machine | **Best accuracy (0.784), precision, recall, F1 (0.710) and MCC (0.643)**, and the best dropout recall (76%), which is the class that matters most for an early-warning use case. The RBF kernel picks up some non-linear interaction the linear model misses, but the margin over Logistic Regression is small (~0.3% F1). AUC is a bit lower than LR because SVC has no native probabilities – they come from isotonic calibration, which flattens the ranking slightly. |
| **Overall winner** | **Support Vector Machine** wins on 5 of the 6 metrics, with **Logistic Regression** effectively tied and better on AUC. For deployment I would honestly pick either; LR is the cheaper and more explainable choice, SVM gets the most dropouts right. Every model finds the **Enrolled** class hard (best recall ≈ 40–50%) – these students sit between the two other groups academically and are under-represented, so class weighting or extra features (3rd semester results, attendance) would be the next thing to try. |

---

## Streamlit app

Features (all inside `app.py`):

1. **CSV upload** of test data in the sidebar (comma or semicolon separated; the bundled `test_data.csv` is used when nothing is uploaded and can also be downloaded from the sidebar). Uploaded files are validated for the 36 feature columns + `Target`.
2. **Model dropdown** to pick one of the five saved models.
3. **Evaluation metrics** – Accuracy, AUC, Precision, Recall, F1, MCC for the chosen model.
4. **Confusion matrix and classification report** for the chosen model.
5. **Compare all models** tab – metrics table for all five models on the uploaded file, a bar chart per metric and every confusion matrix.
6. **Predictions** tab – row-level predictions with class probabilities, a filter for misclassified rows and a CSV download.

The app only calls `predict` / `predict_proba` on the saved `.joblib` pipelines – nothing is retrained on the uploaded data.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py      # optional - regenerates model/ and test_data.csv
streamlit run app.py
```

Deployment: Streamlit Community Cloud → New app → repo `muralikulkarni-design/MLAsg-2`, branch `main`,
file `app.py` (Python 3.12 or 3.13 in *Advanced settings*).
