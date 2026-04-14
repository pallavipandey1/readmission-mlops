# Unit tests for feature engineering
import pandas as pd
import sys
sys.path.insert(0, ".")
from src.features.engineer import engineer_features, load_and_clean

# ── Helper: build a minimal fake patient row ──────────
def make_fake_df(n=10):
    return pd.DataFrame({
        "age":                    ["[50-60)"] * n,
        "time_in_hospital":       [5] * n,
        "num_lab_procedures":     [40] * n,
        "num_procedures":         [1] * n,
        "num_medications":        [15] * n,
        "number_outpatient":      [0] * n,
        "number_emergency":       [1] * n,
        "number_inpatient":       [2] * n,
        "number_diagnoses":       [7] * n,
        "A1Cresult":              [">8"] * n,
        "discharge_disposition_id": [1] * n,
        "admission_source_id":    [7] * n,
        "metformin":              ["No"] * n,
        "repaglinide":            ["No"] * n,
        "nateglinide":            ["No"] * n,
        "chlorpropamide":         ["No"] * n,
        "glimepiride":            ["No"] * n,
        "glipizide":              ["Steady"] * n,
        "glyburide":              ["No"] * n,
        "pioglitazone":           ["No"] * n,
        "rosiglitazone":          ["No"] * n,
        "acarbose":               ["No"] * n,
        "insulin":                ["Ch"] * n,
        "diag_1":                 ["250"] * n,
        "diag_2":                 ["401"] * n,
        "diag_3":                 ["0"] * n,
        "readmitted_30d":         [1] * n,
    })

# ── Test 1: output shape ──────────────────────────────
#  catches if you accidentally drop rows or return the wrong number of features
def test_engineer_features_shape():
    df = make_fake_df(10)
    X, y = engineer_features(df)
    assert X.shape[0] == 10, "Should have 10 rows"
    assert len(X.columns) > 0, "Should have feature columns"

# ── Test 2: no NaN values in output ──────────────────
# catches missing value bugs in the feature engineering code
def test_no_nulls_in_features():
    df = make_fake_df(10)
    X, y = engineer_features(df)
    assert X.isnull().sum().sum() == 0, "Features should have no nulls"

# ── Test 3: age conversion works correctly ────────────
#  catches if the age mapping breaks
def test_age_conversion():
    df = make_fake_df(5)
    X, y = engineer_features(df)
    assert (X["age_num"] == 55).all(), "[50-60) should map to midpoint 55"

# ── Test 4: insulin 'Ch' flag increments meds_changed ─
# catches if the meds_changed logic breaks and doesn't count medication changes correctly
def test_meds_changed_counts():
    df = make_fake_df(5)
    X, y = engineer_features(df)
    # insulin is set to 'Ch' — meds_changed should be 1
    assert (X["meds_changed"] == 1).all()

# ── Test 5: target column is binary 0/1 ──────────────
# catches if the target variable is not properly encoded as binary
def test_target_is_binary():
    df = make_fake_df(10)
    _, y = engineer_features(df)
    assert set(y.unique()).issubset({0, 1}), "Target must be 0 or 1 only"