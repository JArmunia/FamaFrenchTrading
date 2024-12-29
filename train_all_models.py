import subprocess
from pathlib import Path
from time import time



def ejecutar_model_training(data_store, data_store_item, results_path):    
    print(f"\nEjecutando model_training.py con data_store_item={data_store_item} y results_path={results_path}...")
    subprocess.run([
        "python", 
        "model_training.py",
        "--data_store", str(data_store),
        "--data_store_item", data_store_item,
        "--results_path", str(results_path)
    ])

# Ejecutar predict_OOS.py
def ejecutar_predict_oos(data_store, data_store_item, results_path, predictions_store):
    print(f"\nEjecutando predict_OOS.py con data_store_item={data_store_item}, results_path={results_path} y predictions_store={predictions_store}...")
    subprocess.run([
        "python",
        "predict_OOS.py", 
        "--data_store", str(data_store),
        "--data_store_item", data_store_item,
        "--results_path", str(results_path),
        "--predictions_store", str(predictions_store)
    ])


if __name__ == "__main__":
    
    modelos = [
        # {
    #     "data_store":"data_neutralized/assets.h5",
    #     "data_store_item":"engineered_features",
    #     "results_path":"results_neutralized_todo/us_stocks",
    #     "predictions_store":"data_neutralized/predictions_neutralized_todo.h5"
    # },
    {
        "data_store":"data_normalized/assets.h5",
        "data_store_item":"engineered_features",
        "results_path":"results_normalized_todo/us_stocks",
        "predictions_store":"data_normalized/predictions_normalized_todo.h5"
    },
    # {
    #     "data_store":"data_neutralized/assets.h5",
    #     "data_store_item":"engineered_features_trimmed",
    #     "results_path":"results_neutralized_trimmed/us_stocks",
    #     "predictions_store":"data_neutralized/predictions_neutralized_trimmed.h5"
    # },
    {
        "data_store":"data_normalized/assets.h5",
        "data_store_item":"engineered_features_trimmed",
        "results_path":"results_normalized_trimmed/us_stocks",
        "predictions_store":"data_normalized/predictions_normalized_trimmed.h5"
    },
    # {
    #     "data_store":"data_neutralized/assets.h5",
    #     "data_store_item":"engineered_features_pca",
    #     "results_path":"results_neutralized_pca/us_stocks",
    #     "predictions_store":"data_neutralized/predictions_neutralized_pca.h5"
    # },
    # {
    #     "data_store":"data_normalized/assets.h5",
    #     "data_store_item":"engineered_features_pca",
    #     "results_path":"results_normalized_pca/us_stocks",
    #     "predictions_store":"data_normalized/predictions_normalized_pca.h5"
    # }
    ]

    for modelo in modelos:
        time_start = time()
        print("*"*100)
        print(f"\nEjecutando modelo {modelo['data_store_item']}...")
        ejecutar_model_training(modelo["data_store"], modelo["data_store_item"], modelo["results_path"])
        ejecutar_predict_oos(modelo["data_store"], modelo["data_store_item"], modelo["results_path"], modelo["predictions_store"])
        print("*"*45, f"Tiempo transcurrido: {time() - time_start:.2f} segundos", "*"*45)