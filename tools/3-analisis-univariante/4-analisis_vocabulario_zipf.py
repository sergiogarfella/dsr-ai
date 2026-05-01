import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

print("--- ANÁLISIS DE VOCABULARIO Y LEY DE ZIPF ---\n")

# Configurar fuente Inter
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'sans-serif']

script_dir = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.normpath(os.path.join(script_dir, '../../datasets/dataset_unificado.csv'))
dir_graficos = os.path.join(script_dir, 'graficos')

if not os.path.exists(ruta_csv):
    print("Error: No se encuentra el dataset_unificado.csv.")
    exit(1)

os.makedirs(dir_graficos, exist_ok=True)

df = pd.read_csv(ruta_csv)

# 1. Estadísticas de Vocabulario por Dataset y Global
print("ESTADÍSTICAS DE VOCABULARIO:")
print("-" * 85)

datasets = df['dataset'].unique().tolist() + ['Unificado']

for ds in datasets:
    if ds == 'Unificado':
        textos = df['texto'].astype(str)
    else:
        textos = df[df['dataset'] == ds]['texto'].astype(str)
    
    todas_palabras = ' '.join(textos).lower().split()
    total_palabras = len(todas_palabras)
    vocabulario = set(todas_palabras)
    tam_vocabulario = len(vocabulario)
    
    # Frecuencias
    frecuencias = Counter(todas_palabras)
    hapax_legomena = sum(1 for f in frecuencias.values() if f == 1)  # Palabras que aparecen solo 1 vez
    top_10 = frecuencias.most_common(10)
    
    print(f"\nDataset: {ds}")
    print(f"  - Total de palabras (tokens):       {total_palabras:,}")
    print(f"  - Vocabulario (palabras únicas):    {tam_vocabulario:,}")
    print(f"  - Ratio tipo/token (TTR):           {(tam_vocabulario / total_palabras):.4f}")
    print(f"  - Hapax legomena (frecuencia = 1):  {hapax_legomena:,} ({(hapax_legomena / tam_vocabulario) * 100:.1f}% del vocabulario)")
    print(f"  - Top 10 palabras más frecuentes:")
    for palabra, freq in top_10:
        print(f"      '{palabra}': {freq:,}")

# 2. Ley de Zipf
print("\n\nLEY DE ZIPF:")
print("-" * 85)

for ds in datasets:
    if ds == 'Unificado':
        textos = df['texto'].astype(str)
    else:
        textos = df[df['dataset'] == ds]['texto'].astype(str)
    
    todas_palabras = ' '.join(textos).lower().split()
    frecuencias = Counter(todas_palabras)
    
    # Ordenar frecuencias de mayor a menor
    frecuencias_ordenadas = sorted(frecuencias.values(), reverse=True)
    rangos = np.arange(1, len(frecuencias_ordenadas) + 1)
    
    # Calcular la pendiente de Zipf (en escala log-log debería ser cercana a -1)
    log_rangos = np.log10(rangos)
    log_frecuencias = np.log10(frecuencias_ordenadas)
    pendiente, intercepto = np.polyfit(log_rangos, log_frecuencias, 1)
    
    print(f"\nDataset: {ds}")
    print(f"  - Pendiente de Zipf (ideal ≈ -1.0): {pendiente:.3f}")
    print(f"  - Intercepto:                        {intercepto:.3f}")

# 3. Gráfico de Zipf (todos los datasets en un único gráfico)
print("\nGenerando gráfico de Zipf...")

plt.figure(figsize=(12, 7))
colores = {'stanfordSentimentTreebank': '#2ecc71', 'aclImdb': '#3498db', 'review_polarity': '#e74c3c', 'Unificado': '#9b59b6'}

for ds in datasets:
    if ds == 'Unificado':
        textos = df['texto'].astype(str)
    else:
        textos = df[df['dataset'] == ds]['texto'].astype(str)
    
    todas_palabras = ' '.join(textos).lower().split()
    frecuencias = Counter(todas_palabras)
    frecuencias_ordenadas = sorted(frecuencias.values(), reverse=True)
    rangos = np.arange(1, len(frecuencias_ordenadas) + 1)
    
    plt.loglog(rangos, frecuencias_ordenadas, label=ds, color=colores.get(ds, '#333'), alpha=0.8, linewidth=1.5)

# Línea de referencia Zipf ideal (pendiente -1)
max_freq = max(Counter(' '.join(df['texto'].astype(str)).lower().split()).values())
x_ref = np.arange(1, 100000)
y_ref = max_freq / x_ref
plt.loglog(x_ref, y_ref, '--', color='gray', alpha=0.5, linewidth=1, label='Zipf ideal (α = -1)')

plt.title('Ley de Zipf: Frecuencia vs. Rango', fontsize=14, fontweight='bold')
plt.xlabel('Rango (log)', fontsize=12, fontweight='medium')
plt.ylabel('Frecuencia (log)', fontsize=12, fontweight='medium')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()

ruta_grafico = os.path.join(dir_graficos, '5-zipf.png')
plt.savefig(ruta_grafico, dpi=300)
plt.close()

print(f"\n¡Análisis completado! El gráfico se ha guardado en:\n{os.path.abspath(ruta_grafico)}")
