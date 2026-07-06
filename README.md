# Housing Price Regression - MLOps

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Ridge%20%7C%20Lasso%20%7C%20Linear-F7931E)

A modular MLOps assignment demonstrating regression modelling and hyperparameter tuning on the Boston Housing dataset, with CI/CD automation via GitHub Actions.

---

## Models & Results

Three regression models evaluated on an 80/20 train/test split:

| Model | Branch | Tuning |
|---|---|---|
| Linear Regression | `reg` | None |
| Ridge Regression | `reg` | — |
| Lasso Regression | `reg` | — |
| Ridge (tuned) | `hyper` | GridSearchCV (`alpha`: 0.1, 1.0, 10.0, 100.0) |
| Lasso (tuned) | `hyper` | GridSearchCV (`alpha`: 0.01, 0.1, 1.0, 10.0) |

Evaluated with **MSE** and **R²** on each.

---

## Repository Structure

```
HousingRegression/
├── .github/workflows/ci.yml   # GitHub Actions CI — runs on push to reg/hyper
├── regression.py              # Entry point: train + evaluate all models
├── utils.py                   # load_data(), train_models(), tune_and_train_models()
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/saiswaroopkakarla/HousingRegression.git
cd HousingRegression
conda create -n houseprice python=3.10 -y
conda activate houseprice
pip install -r requirements.txt
```

---

## Run

```bash
# Train all models + print MSE and R² for each
python regression.py
```

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) triggers on every push to the `reg` or `hyper` branches:

```
Checkout → Set up Python 3.10 → Install dependencies → Run regression.py
```

This validates that the training pipeline runs end-to-end without errors on every commit.

---

## Dataset Note

Uses the **Boston Housing dataset** loaded directly from the CMU StatLib archive (`http://lib.stat.cmu.edu/datasets/boston`) — scikit-learn removed this dataset in v1.2 due to ethical concerns around the `B` (racial composition) feature. The `load_data()` function in `utils.py` fetches it manually and handles the raw file parsing.

---

## Author

**Kakarla Sai Swaroop**, M25DE1023, IIT Jodhpur M.Tech Data Engineering
