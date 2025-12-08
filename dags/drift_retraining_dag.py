from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime
import subprocess

def decide_retrain_branch():
    #Rodando o drift e decidindo se retreina ou nao

    result = subprocess.run(
        ["python", "drift_detection/check_drift.py"],
        capture_output=True,
        text=True,
    )
    output = result.stdout
    print("Drift detection output:", output)

    #Determinando o branch baseado na saida
    if "Drift detected" in output:
        return "retrain_model"
    else:
        return "no_retrain"
    
def retrain_model_task():
    #Aqui voce colocaria o codigo real de retreinamento
    print("Retraining the model...")
    subprocess.run(
        ["python", "models/retrain_model.py"]
    )
    print("Model retraining completed.")

def skip_retraining_task():
    print("Skipping model retraining as no drift was detected.")


with DAG(
    dag_id="data_model_drift_detection_and_retraining",
    schedule=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    description="Detect model drift and retrain if needed",
    tags=["ml", "ai", "mlops", "drift detection"]
) as dag:

    decide_branch = BranchPythonOperator(
        task_id="decide_retrain_branch",
        python_callable=decide_retrain_branch,
    )

    retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model_task,
    )

    no_retrain = PythonOperator(
        task_id="no_retrain",
        python_callable=skip_retraining_task,  # Corrigido aqui
    )

    decide_branch >> [retrain_model, no_retrain]
