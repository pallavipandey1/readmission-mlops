# This notebook is for checking the feature importance of the XGBoost model. It loads the data, engineers the features,
#  trains the model, and then prints the feature importance scores. This is no longer required as we log the importance scores in MLflow, 
# but it's useful for a quick check.


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, ".")
from src.features.engineer import load_and_clean, engineer_features

df = load_and_clean("data/dataset_diabetes/diabetic_data.csv")
X, y = engineer_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(n_estimators=300, max_depth=5,
                           learning_rate=0.05, subsample=0.8,
                           eval_metric="auc")
model.fit(X_train, y_train)

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print(importance.to_string())