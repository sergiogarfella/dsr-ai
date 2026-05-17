"""
Configuración centralizada de rutas del proyecto DSR-AI.
Todas las rutas son relativas a la raíz del repositorio.
"""
from pathlib import Path

# Raíz del repositorio (dos niveles arriba de src/config.py)
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Datasets ──
DATASETS_DIR = ROOT_DIR / "data" / "raw"
DATASET_CSV = DATASETS_DIR / "dataset_unificado_corregido.csv"

# ── Datos procesados ──
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TOKENIZED_TRAIN_PKL = PROCESSED_DIR / "tokenized_docs_train_clean.pkl"
TOKENIZED_TEST_PKL = PROCESSED_DIR / "tokenized_docs_test_clean.pkl"
X_TRAIN_PKL = PROCESSED_DIR / "X_train.pkl"
X_TEST_PKL = PROCESSED_DIR / "X_test.pkl"
Y_TRAIN_PKL = PROCESSED_DIR / "y_train.pkl"
Y_TEST_PKL = PROCESSED_DIR / "y_test.pkl"

# ── Modelos (paquete de producción) ──
PACKAGE_DIR = ROOT_DIR / "dsr"
DOC2VEC_MODEL_PATH = PACKAGE_DIR / "doc2vec_model"
KNN_MODEL_PATH = PACKAGE_DIR / "knn_model.pkl"
LR_MODEL_PATH = PACKAGE_DIR / "lr_model.pkl"

# ── Parámetros de entrenamiento ──
TRAIN_SAMPLES_PER_CLASS = 5000
TEST_SAMPLES_PER_CLASS = 2500
RANDOM_STATE = 42
