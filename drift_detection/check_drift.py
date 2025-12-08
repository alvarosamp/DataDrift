import pandas as pd
import numpy as np
from alibi_detect.cd import KSDrift 


def check_data_drift(reference_data_path, target_data_path, threshold=0.05):
    #Carregando o dado de treinamento
    ref_df = pd.read_csv(reference_data_path)
    #Carregando o dado de teste
    ref_X = ref_df.drop(columns=["target"]).values

    #New df
    new_df = pd.read_csv(target_data_path)
    new_X = new_df.drop(columns=["target"]).values

    #Inicializando o detector de drift
    detector = KSDrift(ref_X, p_val=threshold)

    #Check drift
    preds = detector.predict(new_X)
    drift = preds["data"]["is_drift"]
    p_value = preds["data"]["p_val"]

    if drift:
        print(f"Drift detected (p-value: {p_value})")
    else:
        print(f"No drift detected (p-value: {p_value})")
    return drift

if __name__ == "__main__":
    reference_path = r"C:\Users\alvaro.careli\Documents\DataDrift\data\iris.csv"
    target_path = r"C:\Users\alvaro.careli\Documents\DataDrift\data\iris_drifted.csv"
    check_data_drift(reference_data_path=reference_path, target_data_path=target_path)