import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_save_model(data_path: str, model_path: str, metrics_path: str | None, save_retrain: bool, n_estimators: int):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")

    score = clf.score(X_test, y_test)
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump({"accuracy": score}, mf)
        print(f"Metrics saved to {metrics_path}")

    if save_retrain:
        # Optional non-deterministic artifact; disabled by default for DVC pipelines
        from datetime import datetime

        Path("retrained_models").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        retrain_filename = f"retrained_models/random_forest_retrain_{timestamp}.pkl"
        with open(retrain_filename, "wb") as f:
            pickle.dump(clf, f)
        print(f"Retrained model saved to {retrain_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest on Iris and save model")
    parser.add_argument("--data", default="data/iris.csv", help="Input CSV path")
    parser.add_argument("--model", default="models/random_forest.pkl", help="Output model path")
    parser.add_argument("--metrics", default="metrics.json", help="Path to write metrics JSON")
    parser.add_argument("--n-estimators", type=int, default=100, help="RandomForest n_estimators")
    parser.add_argument("--save-retrain", action="store_true", help="Also write a timestamped retrained model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save_model(
        data_path=args.data,
        model_path=args.model,
        metrics_path=args.metrics,
        save_retrain=args.save_retrain,
        n_estimators=args.n_estimators,
    )
