import numpy as np
import pandas as pd

from src.main import (
    DatasetParams,
    generate_synthetic_credit,
    build_pipeline,
    map_feature_names,
)


def test_synthetic_generator_basic():
    params = DatasetParams(n_samples=100)
    df = generate_synthetic_credit(params)
    assert "approved" in df.columns
    assert df.shape[0] == 100


def test_pipeline_smoke_fit_predict():
    params = DatasetParams(n_samples=200)
    df = generate_synthetic_credit(params)
    X = df.drop(columns=["approved"])
    y = df["approved"].values

    numeric = ["income", "age", "credit_score", "debt_ratio"]
    categorical = ["employment_status", "home_ownership"]

    pipeline = build_pipeline(numeric, categorical)
    pipeline.fit(X, y)

    # predict_proba should return a column for positive class
    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape[0] == X.shape[0]

    # map_feature_names should return a list and its length should match transformed features
    features = map_feature_names(pipeline, numeric, categorical)
    assert isinstance(features, list)
    assert len(features) >= len(numeric)
