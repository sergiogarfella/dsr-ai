# Sistema de Detección de Sentimiento en Reseñas de Usuarios (DSR-AI)

## Objetivo

Desarrollar un modelo en Python capaz de clasificar las reseñas de los usuarios como positivas o negativas usando Doc2Vec y K-Nearest Neighbors (k-NN). Todo ello presentado a través de una sencilla interfaz web hecha con Streamlit.

## Estructura del Proyecto

```
dsr-ai/
├── dsr/                          # Paquete de producción (modelos + predicción)
│   ├── __init__.py
│   ├── dsr.py                    # Clase Dsr — interfaz principal de predicción
│   └── text_clean.py             # Limpieza de texto para inferencia
├── src/                          # Pipeline de entrenamiento
│   ├── config.py                 # Rutas y parámetros centralizados
│   ├── data_preprocessing/       # Limpieza y preparación de datos
│   ├── features/                 # Vectorización con Doc2Vec
│   └── models/                   # Entrenamiento KNN y Regresión Logística
├── web-app/                      # Interfaz web con Streamlit
│   ├── app.py                    # Aplicación principal
│   └── requirements.txt          # Dependencias de producción
├── data/
│   ├── raw/                      # Datasets originales (no trackeados en git)
│   └── processed/                # Datos procesados intermedios
├── docs/                         # Documentación del proyecto
└── tools/                        # Scripts de análisis auxiliares
```

## Requisitos

- Python 3.11+
- Dependencias listadas en `web-app/requirements.txt`

## Instalación y Ejecución

### Interfaz web (Streamlit)

```bash
cd web-app
python -m venv .venv
source .venv/bin/activate        # En macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

### Pipeline de entrenamiento

> **Nota:** Los datasets y modelos no están en el repositorio git por su tamaño.
> Contactar a los autores para obtenerlos.

Los scripts de `src/` deben ejecutarse **desde la raíz del repositorio** en este orden:

1. `python src/data_preprocessing/limpieza_datos.py` — Limpieza y tokenización
2. `python src/features/vectorizacion_train_test.py` — Vectorización con Doc2Vec
3. `python src/models/knn_training_and_metrics.py` — Entrenamiento KNN
4. `python src/models/lr_training_metrics.py` — Entrenamiento Regresión Logística

## Modelos Utilizados

| Modelo | Descripción |
|---|---|
| **Doc2Vec** | Representación vectorial de documentos (dim=384) |
| **K-Nearest Neighbors** | Clasificador basado en distancia coseno (k=71) |
| **Regresión Logística** | Clasificador lineal con regularización L2 |

## Autores del Proyecto

- Adrián A. Acosta Villegas
- Sergio Garfella Pérez
- Ainara Sanfélix Ruiz
- Jairo E. Urdaneta Colmenares

Proyecto desarrollado para la asignatura **Proyecto I. Introducción a la IA** de la **Universitat Politècnica de València (UPV)**.