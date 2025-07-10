# 🏡 Housing Regression - MLOps

This repository contains a complete Machine Learning workflow to predict house prices using the Boston Housing dataset. It demonstrates modular code development, regression modeling, hyperparameter tuning, and CI/CD integration via GitHub Actions as part of an academic MLOps assignment.

---

## 📂 Project Structure

```
HousingRegression/
├── .github/workflows/ci.yml       # CI pipeline for automated testing
├── regression.py                  # Main script for training models
├── utils.py                       # Utility functions for data loading, modeling, tuning
├── requirements.txt               # Environment dependencies
├── README.md                      # This file
```

---

## 🔧 Environment Setup

Create a clean Conda environment and install the required packages:

```bash
conda create -n houseprice python=3.10 -y
conda activate houseprice
pip install -r requirements.txt
```

---

##  Dataset

The **Boston Housing dataset** is used in this project. Due to deprecation in scikit-learn, the dataset is manually fetched from:

📎 `http://lib.stat.cmu.edu/datasets/boston`

The `load_data()` function in `utils.py` handles all loading and preprocessing.

---

## 📊 Models Implemented

### In `reg` branch:
- Linear Regression
- Ridge Regression
- Lasso Regression

### In `hyper` branch:
- Same models with hyperparameter tuning using `GridSearchCV`

---

## ⚙️ Evaluation Metrics

Each model is evaluated using:
- **Mean Squared Error (MSE)**
- **R² Score (coefficient of determination)**

---

## 🚀 How to Run

Run the following command to execute training and evaluation:

```bash
python regression.py
```

This script prints performance metrics for each model.

---

## 🤖 GitHub Actions (CI/CD)

Every push to the `reg` and `hyper` branches triggers the CI pipeline (`ci.yml`) which:

1. Sets up Python 3.10
2. Installs dependencies
3. Runs `regression.py` to validate the models

You can view the logs under the **Actions** tab in this repo.

---

## 🧠 Branch Details

| Branch   | Description                               |
|----------|-------------------------------------------|
| `main`   | Final merged project                      |
| `reg`    | Baseline regression models (no tuning)    |
| `hyperb`  | Models with hyperparameter tuning         |

---

## ✅ Deliverables

- Modular code (utils.py, regression.py)
- requirements.txt generated from pip freeze
- CI pipeline using GitHub Actions
- Reproducible environment via Conda
- Evaluation results for both baseline and tuned models

---

## 📬 Contact

For any academic references, reach out to:  
**Sai Swaroop** 
g24ai1026@iitj.ac.in

