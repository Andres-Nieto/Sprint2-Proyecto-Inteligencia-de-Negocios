import os
import joblib

# Este archivo tiene una función simple para cargar modelos entrenados desde la carpeta /models.
# Su objetivo es mantener centralizada la forma de cargar los archivos .pkl que fueron generados con el script train_save_models.py.

# Definimos las rutas base a partir de la ubicación del archivo actual
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_model(name):
    """
    Carga un modelo previamente guardado (por ejemplo: scaler.pkl, kmeans.pkl o feature_cols.pkl).
    Si el modelo existe en la carpeta /models, lo devuelve; si no, muestra una advertencia.
    """
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    else:
        print(f"⚠ Modelo no encontrado: {path}")
        return None