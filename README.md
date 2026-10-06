# Insurance Claim Severity Prediction

A machine learning project that predicts the cost of insurance claims (`loss`) for
customers of Allstate. Built with scikit-learn, XGBoost, LightGBM, and PyTorch on the
Kaggle "Allstate Claims Severity" dataset.

## Problem

The goal is to predict the `loss` (claim severity) for each claim in the test set, using
116 anonymized categorical features and 14 continuous features. Model performance is
evaluated using MAE (Mean Absolute Error) between predicted and actual loss — a lower
score is better.

## Project Structure

```
.
├── data/
│   ├── train.csv
│   └── test.csv
├── insurance-claim-severity-prediction.ipynb   # Main notebook: EDA, preprocessing, modeling, tuning
├── submission.csv                              # Final predictions
├── requirements.txt
└── README.md
```

Running the notebook also generates `model_comparison.csv`.

## Approach

1. **Exploratory Data Analysis** — inspect shape, types, and missing values (none in
   train or test), categorical cardinality (72 binary and 44 multi-level columns, up to
   326 levels in `cat116`), and the target distribution. `loss` is heavily right-skewed
   (skewness 3.79); a `log1p` transform reduces it to 0.10.
2. **Preprocessing** —
   - Applies `log1p` to the target; predictions are converted back with `expm1`
   - Drops `cont12` (0.99 correlation with `cont11`) and the non-predictive `id`
   - Label-encodes all categorical features, fitting the encoders on train and test
     combined to avoid unseen-label errors
   - Standardizes features for the neural network only
3. **Model Selection** — compares regressors on an 80/20 train/validation split and
   5-fold cross-validation, scored by MAE on the original loss scale:
   - Ridge Regression (baseline)
   - Random Forest
   - XGBoost Regressor
   - LightGBM Regressor
   - PyTorch MLP (129 → 256 → 128 → 64 → 1, ReLU, Adam, MSE loss)
4. **Hyperparameter Tuning** — `RandomizedSearchCV` (10 candidates, 5-fold CV) for the
   XGBoost model, scored by negative MAE on the log-transformed target. Best parameters:
   `n_estimators=200`, `max_depth=7`, `learning_rate=0.1`, `subsample=0.8`,
   `colsample_bytree=0.9`.
5. **Final Model** — the tuned XGBoost Regressor, used to generate `submission.csv`.

### Model Comparison

| Model | Validation MAE | Mean CV MAE |
|---|---|---|
| **XGBoost** | **1149.19** | **1158.07** |
| PyTorch MLP | 1246.22 | 1165.12 |
| LightGBM | 1183.88 | 1194.38 |
| Random Forest | 1233.00 | 1245.49 |
| Ridge (baseline) | 1287.21 | 1290.52 |

The MLP's CV score was computed on the 80% training split only, with a different batch
size and epoch count than its validation run, so it is not directly comparable to the
other CV scores. After tuning, XGBoost reached a validation MAE of **1141.78**.

## Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the data
Download `train.csv` and `test.csv` from the
[Kaggle competition page](https://www.kaggle.com/competitions/allstate-claims-severity/data)
and place them in a `data/` folder at the project root.

### 4. Update the data path
In the notebook, change `data_path` (currently
`/kaggle/input/competitions/allstate-claims-severity`) to `data`.

### 5. Run the notebook
```bash
jupyter notebook insurance-claim-severity-prediction.ipynb
```

## Output

Running the notebook end-to-end produces `submission.csv`, containing predicted `loss`
values for each `id` in the test set (125,546 rows), ready for submission to Kaggle.

## Results

Submitted to the [Kaggle "Allstate Claims Severity"](https://www.kaggle.com/competitions/allstate-claims-severity) competition:

| Metric | Score |
|---|---|
| Private score (MAE) | **1144.91916** |
| Public score (MAE) | **1134.71335** |

## License

This project is open source and available under the MIT License.
