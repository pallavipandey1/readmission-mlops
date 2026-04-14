# Creates a synthetic sample dataset for CI
# Real data stays local — CI gets a small fake version
import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 2000   # small enough to train fast in CI

os.makedirs("data", exist_ok=True)

df = pd.DataFrame({
    "age":               np.random.choice(
                           ["[40-50)","[50-60)","[60-70)","[70-80)"],n),
    "time_in_hospital":  np.random.randint(1, 14, n),
    "num_lab_procedures":np.random.randint(1, 100, n),
    "num_procedures":    np.random.randint(0, 6, n),
    "num_medications":   np.random.randint(1, 30, n),
    "number_outpatient": np.random.randint(0, 5, n),
    "number_emergency":  np.random.randint(0, 3, n),
    "number_inpatient":  np.random.randint(0, 5, n),
    "number_diagnoses":  np.random.randint(1, 9, n),
    "A1Cresult":         np.random.choice(["None",">7",">8","Norm"],n),
    "discharge_disposition_id": np.random.choice([1,2,3,6],n),
    "admission_source_id":      np.random.choice([1,4,7],n),
    "metformin":         np.random.choice(["No","Steady","Up","Down"],n),
    "repaglinide":       ["No"] * n,
    "nateglinide":       ["No"] * n,
    "chlorpropamide":    ["No"] * n,
    "glimepiride":       ["No"] * n,
    "glipizide":         np.random.choice(["No","Steady"],n),
    "glyburide":         ["No"] * n,
    "pioglitazone":      ["No"] * n,
    "rosiglitazone":     ["No"] * n,
    "acarbose":          ["No"] * n,
    "insulin":           np.random.choice(["No","Steady","Ch"],n),
    "diag_1":            np.random.choice(["250","401","428","786"],n),
    "diag_2":            np.random.choice(["250","401","0"],n),
    "diag_3":            ["0"] * n,
    "readmitted_30d":    np.random.choice([0,1], n, p=[0.89,0.11]),
})

df.to_csv("data/sample_data.csv", index=False)
print(f"Sample dataset created: {len(df)} rows")