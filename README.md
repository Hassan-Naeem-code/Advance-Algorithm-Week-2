# Credit Approval with Logistic Regression

This repository contains an end-to-end example for Week 2 Assignment: building a logistic regression model to predict credit application approval.

Contents
- `src/main.py`: main script that runs EDA, builds a preprocessing pipeline (ColumnTransformer), trains a logistic regression model, interprets coefficients (odds ratios), and evaluates with threshold analysis. The script supports a synthetic generator (default) and an optional real dataset loader (UCI Default of Credit Card Clients).
- `requirements.txt`: Python package requirements.
- `.gitignore`: common ignores.
- `figures/`: saved EDA and evaluation plots created by `src/main.py`.

Dataset
- By default the repository uses a synthetic dataset generated via scikit-learn `make_classification` (documented parameters are in `src/main.py`).
- Optional real dataset (UCI): the script can load the UCI "Default of Credit Card Clients" dataset if you run with `--use-real` (see below). Dataset URL:

	https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls

Quick setup
1. Create and activate a virtual environment in the project root (macOS / Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the main script (synthetic data, default):

```bash
python src/main.py
```

3. To run using the UCI real dataset (the script will attempt to download it):

```bash
python src/main.py --use-real
```

Notes
- If the real dataset download fails, the script will fall back to the synthetic generator and print a warning.
- See inline comments in `src/main.py` for assumptions, feature selection decisions, and hyperparameter rationale.

References (APA style)
- scikit-learn developers. (2024). scikit-learn: Machine Learning in Python. https://scikit-learn.org
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
