import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

#  Rutas 
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

#  Selección del dataset limpio (prioriza SaberPro)
files = sorted([f for f in os.listdir(DATA_DIR) if "LIMPIO" in f])
priority = [f for f in files if "SaberPro" in f]
path = os.path.join(DATA_DIR, priority[0] if priority else files[0])
print("Usando dataset:", path)

#  Lectura 
df = pd.read_csv(path)

#  Usaremos SOLO las 4 columnas que la API recibe 
#   API -> dataset
#   punt_matematicas        -> mod_razona_cuantitat_punt
#   punt_lectura_critica    -> mod_lectura_critica_punt
#   punt_ingles             -> mod_ingles_punt
#   punt_global             -> punt_global
feature_cols = [
    "punt_global",
    "mod_razona_cuantitat_punt",
    "mod_lectura_critica_punt",
    "mod_ingles_punt",
]

# Filtro y limpieza
X = df[feature_cols].select_dtypes(include=[np.number]).dropna()
X = X.sample(min(80000, len(X)), random_state=42)

# Escalado 
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

# KMeans
kmeans = MiniBatchKMeans(n_clusters=6, batch_size=512, random_state=42, max_iter=300, n_init=20).fit(Xs)

# Guardar artefactos 
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
joblib.dump(kmeans, f"{MODELS_DIR}/kmeans.pkl")
joblib.dump(feature_cols, f"{MODELS_DIR}/feature_cols.pkl")

print("✅ Modelos guardados correctamente en la carpeta /models/")