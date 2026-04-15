# Model training with full MLflow tracking
from xml.parsers.expat import model

from check_importance import X_train
from check_importance import X_train
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score,
                              precision_score, recall_score)
import sys
import pandas as pd
import joblib, os
from src.features.engineer import load_and_clean, engineer_features

def train(data_path):
    # ── 1. Load and prepare data ──────────────────────────
    df = load_and_clean(data_path)
    X, y = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y      # keeps class balance equal in both splits
    )

    # ── 2. Define model settings ──────────────────────────
    params = {
        "n_estimators":   300,
        "max_depth":      5,
        "learning_rate":  0.05,
        "subsample":      0.8,
        "eval_metric":    "auc",
        "scale_pos_weight": 7, #The value 7 means the positive class (readmitted) is roughly 7x rarer — XGBoost will weight it accordingly. Then retrain.
    }

    # ── 3. Start the MLflow recording session ─────────────
    mlflow.set_experiment("readmission-prediction")

    with mlflow.start_run(run_name="baseline-xgboost"):

        # Log every parameter so we can reproduce this run
        mlflow.log_params(params)

        # Train the model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate on the held-out test set
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc       = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)

        # Log the results
        mlflow.log_metrics({
            "auc":       round(auc, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4)
        })

        # Save the model file as an artifact
        mlflow.sklearn.log_model(model, "model")

        # Save model file for Docker container to use
        os.makedirs("model_artifacts", exist_ok=True)
        joblib.dump(model, "model_artifacts/model.pkl")
        joblib.dump(list(X_train.columns),
            "model_artifacts/feature_columns.pkl")
        print("Model saved to model_artifacts/")

        # ── Feature importance ────────────────────────────────
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        for feat, score in importance.items():
            mlflow.log_metric(f"importance_{feat}", round(float(score), 4))

        importance.to_csv("feature_importance.csv")
        mlflow.log_artifact("feature_importance.csv")


        print(f"AUC: {auc:.4f} | Precision: {precision:.4f}"
              f" | Recall: {recall:.4f}")
        
        print("\nTop 10 features:")
        print(importance.head(10).to_string())

        # Quality gate — used later by the CI pipeline
        if auc < 0.50:
            print("AUC below threshold 0.50 — failing build")
            sys.exit(1)

# if __name__ == "__main__":
#     train("data/diabetic_data.csv")




if __name__ == "__main__":
    import sys
    train(sys.argv[1]) # allows us to specify the data path when running the script.sys.argv[1] means "use whatever path I typed in the terminal" — so command line argument actually gets used instead of the hardcoded one.