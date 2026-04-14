# Feature engineering for readmission prediction
import pandas as pd
import numpy as np

def load_and_clean(path):
    df = pd.read_csv(path, na_values='?')

    # Drop columns that would cause data leakage
    # (encounter_id is just a row number, not a real feature)
    df = df.drop(columns=['encounter_id', 'patient_nbr'])

    # Target: simplify to binary — readmitted within 30 days yes/no
    df['readmitted_30d'] = (df['readmitted'] == '<30').astype(int)
    df = df.drop(columns=['readmitted'])

    return df

# def engineer_features(df):
#     # Age: convert text range '[70-80)' to midpoint number 75
#     age_map = {
#         '[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35,
#         '[40-50)':45, '[50-60)':55, '[60-70)':65, '[70-80)':75,
#         '[80-90)':85, '[90-100)':95
#     }
#     df['age_num'] = df['age'].map(age_map)

#     # Count how many diabetes meds were changed
#     med_cols = [
#         'metformin','repaglinide','nateglinide','chlorpropamide',
#         'glimepiride','glipizide','glyburide','pioglitazone',
#         'rosiglitazone','acarbose','insulin'
#     ]
#     df['meds_changed'] = (df[med_cols] == 'Ch').sum(axis=1)
#     df['meds_on']      = (df[med_cols] != 'No').sum(axis=1)

#     # Binary: was HbA1c tested this visit?
#     df['a1c_tested'] = (df['A1Cresult'] != 'None').astype(int)

#     # Select final features for the model
#     features = [
#         'age_num', 'time_in_hospital', 'num_lab_procedures',
#         'num_procedures', 'num_medications', 'number_outpatient',
#         'number_emergency', 'number_inpatient', 'number_diagnoses',
#         'meds_changed', 'meds_on', 'a1c_tested'
#     ]
#     X = df[features].fillna(0)
#     y = df['readmitted_30d']
#     return X, y

def engineer_features(df):
    # Age: convert text range to midpoint
    age_map = {
        '[0-10)':5,'[10-20)':15,'[20-30)':25,'[30-40)':35,
        '[40-50)':45,'[50-60)':55,'[60-70)':65,'[70-80)':75,
        '[80-90)':85,'[90-100)':95
    }
    df['age_num'] = df['age'].map(age_map)

    # Medication features
    med_cols = [
        'metformin','repaglinide','nateglinide','chlorpropamide',
        'glimepiride','glipizide','glyburide','pioglitazone',
        'rosiglitazone','acarbose','insulin'
    ]
    df['meds_changed'] = (df[med_cols] == 'Ch').sum(axis=1)
    df['meds_on']      = (df[med_cols] != 'No').sum(axis=1)

    # HbA1c tested this visit
    df['a1c_tested'] = (df['A1Cresult'] != 'None').astype(int)
    df['a1c_high']   = (df['A1Cresult'] == '>8').astype(int)

    # Discharge — was patient discharged to another facility?
    # (discharge to home vs transferred is strong readmission signal)
    df['discharged_home'] = (df['discharge_disposition_id'] == 1).astype(int)

    # Admission source
    df['admitted_emergency'] = (df['admission_source_id'] == 7).astype(int)

    # Prior healthcare utilisation — strongest predictors
    df['total_visits'] = (df['number_outpatient']
                        + df['number_emergency']
                        + df['number_inpatient'])

    # Diagnosis codes — convert to numeric, flag circulatory/diabetes
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['has_circulatory'] = (
        ((df['diag_1'] >= 390) & (df['diag_1'] <= 459)) |
        ((df['diag_2'] >= 390) & (df['diag_2'] <= 459)) |
        ((df['diag_3'] >= 390) & (df['diag_3'] <= 459))
    ).astype(int)

    df['has_diabetes_diag'] = (
        ((df['diag_1'] >= 250) & (df['diag_1'] < 251)) |
        ((df['diag_2'] >= 250) & (df['diag_2'] < 251)) |
        ((df['diag_3'] >= 250) & (df['diag_3'] < 251))
    ).astype(int)

    features = [
        'age_num', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_diagnoses',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'total_visits', 'meds_changed', 'meds_on',
        'a1c_tested', 'a1c_high', 'discharged_home',
        'admitted_emergency', 'has_circulatory', 'has_diabetes_diag',
        'diag_1', 'diag_2', 'diag_3'
    ]
    X = df[features].fillna(0)
    y = df['readmitted_30d']
    return X, y