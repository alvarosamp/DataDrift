import pandas as pd
from sklearn.datasets import load_iris

def carregar_dataset():
    """Carrega o dataset Iris e retorna como um DataFrame do pandas."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv("data/iris.csv", index=False)
    print("Iris dataset saved to data/iris.csv")

if __name__ == "__main__":
    carregar_dataset()