import pandas as pd
import os
from scipy.stats import levene

print("--- HOMOGENEIDAD DE VARIANZAS (TEST DE LEVENE) ---\n")

script_dir = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.normpath(os.path.join(script_dir, '../../datasets/dataset_unificado_analisis.csv'))

if not os.path.exists(ruta_csv):
    print("Error: No se encuentra el dataset_unificado_analisis.csv.")
    exit(1)

df = pd.read_csv(ruta_csv)

# Variables a evaluar y sus nombres para la tabla
variables = {
    'num_caracteres': 'N. Caracteres',
    'longitud': 'Longitud Reseñas',
    'vocabulario': 'Vocabulario'
}

# Separar los grupos por sentimiento
grupo_pos = df[df['sentimiento'] == 'pos']
grupo_neg = df[df['sentimiento'] == 'neg']

# Calcular el test de Levene para cada variable
print("Varianzas de variables del dataset\n")
print(f"{'Variable':<20} {'F-value (Levene)':<20} {'p-valor':<20} {'¿Homogéneas?'}")
print("-" * 75)

for col, nombre in variables.items():
    stat, p_valor = levene(grupo_pos[col].dropna(), grupo_neg[col].dropna())
    homogeneas = 'Sí' if p_valor >= 0.05 else 'No'
    print(f"{nombre:<20} {stat:<20.3f} {p_valor:<20.4e} {homogeneas}")

print()
