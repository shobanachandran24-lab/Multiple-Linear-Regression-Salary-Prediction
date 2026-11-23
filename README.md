# Multiple Linear Regression: Salary Prediction

A beginner-friendly machine learning project that predicts an employee's **salary** from their **years of experience**, **age**, and **training hours** using multiple linear regression (scikit-learn).

## Dataset

`data/multiple_linear_regression_dataset.csv` has 12 rows and 4 columns:

| Column | Description |
|---|---|
| `Experience_Years` | Years of work experience (feature) |
| `Age` | Age of the employee (feature) |
| `Training_Hours` | Hours of training completed (feature) |
| `Salary` | Salary (target) |

## What the notebook does

1. Imports libraries
2. Loads the dataset
3. Checks data types and missing values
4. Splits features (X) and target (y)
5. Checks correlations between variables
6. Train/test split (80/20, `random_state=42`)
7. Trains a `LinearRegression` model
8. Inspects coefficients
9. Predicts on the test set and compares actual vs predicted
10. Calculates residuals
11. Evaluates with MAE, MSE, RMSE and R²
12. Predicts salary for a new employee

## Results

| Metric | Value |
|---|---|
| MAE | 1,071 |
| MSE | 1,225,447 |
| RMSE | 1,107 |
| R² | 0.997 |

Learned coefficients: Experience_Years ≈ 1962.65, Age ≈ 860.26, Training_Hours ≈ 256.18.

Example: an employee with 5 years of experience, age 30 and 100 training hours is predicted to earn about **57,929**.

> **Note:** The dataset is very small (12 rows) and the features are highly correlated with each other (0.87 to 0.97), so the high R² and the individual coefficients should be interpreted with caution. This is a learning project, not a production model.

## How to run

```bash
git clone https://github.com/<your-username>/multiple-linear-regression.git
cd multiple-linear-regression
pip install -r requirements.txt
jupyter notebook Multi_Linear_regression.ipynb
```

## Project structure

```
.
├── data/
│   └── multiple_linear_regression_dataset.csv
├── Multi_Linear_regression.ipynb
└── README.md
```

## Tech stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn
