import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def create_drifted_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df_drifted = df.copy()
    # Add noise to numerical columns to simulate drift
    for col in df_drifted.columns:
        if col != "target" and np.issubdtype(df_drifted[col].dtype, np.number):
            noise = np.random.normal(0, 0.5, df_drifted.shape[0])
            df_drifted[col] = df_drifted[col] + noise
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_drifted.to_csv(output_path, index=False)
    print(f"Drifted data saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate drifted version of a dataset")
    parser.add_argument("--in", dest="input_path", default="data/iris.csv", help="Input CSV path")
    parser.add_argument("--out", dest="output_path", default="data/iris_drifted.csv", help="Output CSV path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_drifted_data(input_path=args.input_path, output_path=args.output_path)