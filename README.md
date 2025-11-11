# ICFES Analytics – Sprint 2

Clustering y series de tiempo para datos ICFES (Saber 11 / Saber Pro). Incluye:
- Clusterización (particional, densidad, jerárquica) con seis gráficos compactos.
- Serie de tiempo: agregación, diagnóstico (STL + ADF), ARIMA y RNN mínima (NumPy, sin TensorFlow).
- Módulos en `src/` reutilizables desde notebooks o scripts.

## Datos (datasets grandes y Git)
- Los CSV del Sprint 1 pueden superar el límite de 100 MB de GitHub; por eso NO se incluyen en el repo.
- Coloca los archivos CSV (por ejemplo `Dataset1–Saber11(2020-2)_LIMPIO.csv` y `Dataset2–SaberPro(2021–2024)_LIMPIO.csv`) en la carpeta del proyecto.
- Link de los datasets: https://drive.google.com/drive/folders/1O49JVxhRDbB1oaLek9JYvX1UWl59MEmo?usp=drive_link
- Este repo ignora `*.csv` y `data/` por defecto.
- El notebook detecta automáticamente los archivos limpios por patrón: `Dataset1*LIMPIO.csv` y `Dataset2*LIMPIO.csv` en el directorio actual.

## Requisitos
- Python 3.10–3.12 (Windows recomendado).
- pip y venv (incluido con Python) para gestionar dependencias.

## Setup del proyecto con venv + pip (Windows)
1. Abre PowerShell en la carpeta del proyecto.
2. Crea y activa un entorno virtual:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
3. Instala las dependencias:
  ```powershell
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
4. Registra el kernel de Jupyter:
  ```powershell
  python -m ipykernel install --user --name icfes-analytics
  ```

## Estructura
```
Proyecto/
├─ src/icfes_analytics/           → Módulos analíticos (creados en Sprint 2)
│   ├─ clustering.py
│   ├─ timeseries.py
│   ├─ rnn_numpy.py
│   ├─ plots.py
│   └─ _init_.py
├─ api/app/                       → Lógica y estructura del sistema web (Sprint 3)
│   ├─ main.py
│   ├─ services.py
│   ├─ schemas.py
│   └─ models_loader.py
├─ models/                        → Modelos entrenados y serializados
│   ├─ scaler.pkl
│   ├─ kmeans.pkl
│   └─ feature_cols.pkl
├─ data/                          → Datasets limpios del Sprint 1
│   ├─ Dataset1–Saber11(2020-2)_LIMPIO.csv
│   └─ Dataset2–SaberPro(2021–2024)_LIMPIO.csv
├─ train_save_models.py           → Entrenamiento y guardado de modelos
├─ Sprint2_ICFES.ipynb            → Notebook base del Sprint 2
├─  requirements-api.txt
└─ requirements.txt

```

## Uso de los módulos
Importa desde `icfes_analytics`:
```python
from icfes_analytics import (
    apply_btc_style,
    parse_periodo_flexible, aggregate_series,
    fit_arima_small_grid, plot_arima_forecast,
    forecast_one_step_numpy,
)
```

### Clustering
```python
from icfes_analytics.clustering import run_six_clustering_plots
# X_scaled: ndarray con features estandarizadas (StandardScaler)
resumen = run_six_clustering_plots(X_scaled, n_clusters=5)
print(resumen)
```

### Series de tiempo
```python
# df: DataFrame con columnas 'periodo' y 'punt_global'
agg = aggregate_series(df, period_col='periodo', value_col='punt_global')
apply_btc_style()
# Plot simple
agg.plot(figsize=(12,4))
```

#### ARIMA
```python
order_sel, res_sel, y_pred, y_true, train, test, metrics = fit_arima_small_grid(agg)
print("ARIMA", order_sel, metrics)
plot_arima_forecast(train, test, y_pred, order_sel)
```

#### RNN (NumPy)
```python
y_pred, y_true, metrics = forecast_one_step_numpy(agg, freq='QS-MAR', window=4, hidden_size=16, epochs=600)
print(metrics)
```

## Ejecutar el notebook
1. Abre VS Code y el notebook `Sprint2_ICFES.ipynb`.
2. Selecciona el intérprete del venv `.venv`.
3. Ejecuta las celdas en orden. El notebook ya usa los módulos de `src/`.



# API de Recomendaciones (Sprint 3)

Este módulo implementa una API con FastAPI que utiliza los modelos entrenados para predecir el clúster académico de un estudiante y generar recomendaciones automáticas basadas en sus puntajes Saber 11 / Saber Pro.


# Activar entorno y dependencias

Si no tienes el entorno virtual activo, ejecútalo desde la raíz del proyecto:

1. .\.venv\Scripts\Activate.ps1
2. pip install -r requirements-api.txt

#  Iniciar la API localmente

Desde la raíz del proyecto (donde está train_save_models.py):

- uvicorn api.app.main:app --reload

Esto iniciará el servidor en:

- http://127.0.0.1:8000


# Documentación automática con FastAPI

FastAPI genera una interfaz interactiva automáticamente:

- Documentación Swagger UI → http://127.0.0.1:8000/docs

Desde ahí puedes probar cada endpoint directamente, sin usar Postman ni consola.




# Endpoints principales

GET	/health	Verifica el estado de la API
POST	/predict	Predice el clúster y devuelve recomendaciones automáticas
POST	/cluster	Predice el clúster y guarda la consulta en el historial
GET	/history	Muestra el historial de consultas recientes
GET	/student/{id_student}	Muestra el historial de un estudiante específico
GET	/summary	Muestra el conteo de estudiantes por clúster
DELETE	/clear-history	Limpia el historial almacenado en memoria


# Entrenamiento de modelos (previo)

Antes de ejecutar la API, asegúrate de haber generado los modelos con:

- python train_save_models.py

Esto crea los archivos serializados en /models:

- scaler.pkl
- kmeans.pkl
- feature_cols.pkl

Estos archivos son cargados automáticamente al iniciar la API.


# Notas finales

Todos los datasets deben estar ubicados en la carpeta /data/.

La API usa los mismos puntajes procesados en el Sprint 2 para generar sus recomendaciones.

El sistema está completamente modularizado: los modelos, el procesamiento y la interfaz (FastAPI) son reutilizables para futuros sprints o despliegues en la nube.