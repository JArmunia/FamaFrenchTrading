import subprocess
from pathlib import Path

# Definir argumentos comunes
data_store = Path("data/assets_neutralized.h5")
data_store_item = "engineered_features_trimmed"
results_path = Path("results/us_stocks")
predictions_store = Path("data/prueba_predictions.h5")

# Ejecutar model_training.py
print("\nEjecutando model_training.py...")
subprocess.run([
    "python", 
    "model_training.py",
    "--data_store", str(data_store),
    "--data_store_item", data_store_item,
    "--results_path", str(results_path)
])

# Ejecutar predict_OOS.py
print("\nEjecutando predict_OOS.py...")
subprocess.run([
    "python",
    "predict_OOS.py", 
    "--data_store", str(data_store),
    "--data_store_item", data_store_item,
    "--results_path", str(results_path),
    "--predictions_store", str(predictions_store)
])
