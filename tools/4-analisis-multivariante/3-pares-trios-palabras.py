import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter

print("--- PARES Y TRÍOS DE PALABRAS MÁS FRECUENTES ---\n")

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

# Función auxiliar para extraer n-gramas de una lista de textos
def extraer_ngramas(textos, n):
    contador = Counter()
    for texto in textos:
        palabras = str(texto).lower().split()
        ngramas = [' '.join(palabras[i:i+n]) for i in range(len(palabras) - n + 1)]
        contador.update(ngramas)
    return contador

# 1. Pares de palabras
print("TOP 10 PARES DE PALABRAS:")
print("=" * 60)

for sentimiento, etiqueta in [('pos', 'POSITIVOS'), ('neg', 'NEGATIVOS')]:
    textos = df[df['sentimiento'] == sentimiento]['texto']
    pares = extraer_ngramas(textos, 2)
    top_10 = pares.most_common(10)
    
    print(f"\n  {etiqueta}:")
    print(f"  {'Rango':<6} {'Par de palabras':<30} {'Frecuencia':<12}")
    print("  " + "-" * 50)
    for i, (par, freq) in enumerate(top_10, 1):
        print(f"  {i:<6} {par:<30} {freq:,}")

# 2. Tríos de palabras
print(f"\n\nTOP 10 TRÍOS DE PALABRAS:")
print("=" * 60)

for sentimiento, etiqueta in [('pos', 'POSITIVOS'), ('neg', 'NEGATIVOS')]:
    textos = df[df['sentimiento'] == sentimiento]['texto']
    trios = extraer_ngramas(textos, 3)
    top_10 = trios.most_common(10)
    
    print(f"\n  {etiqueta}:")
    print(f"  {'Rango':<6} {'Trío de palabras':<35} {'Frecuencia':<12}")
    print("  " + "-" * 55)
    for i, (trio, freq) in enumerate(top_10, 1):
        print(f"  {i:<6} {trio:<35} {freq:,}")

# 3. Gráficos comparativos
print("\nGenerando gráficos...")

for n, nombre_tipo in [(2, 'Pares de palabras'), (3, 'Tríos de palabras')]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (sentimiento, etiqueta, color) in enumerate([('pos', 'Positivos', '#2ecc71'), ('neg', 'Negativos', '#ff4d4d')]):
        textos = df[df['sentimiento'] == sentimiento]['texto']
        ngramas = extraer_ngramas(textos, n)
        top_10 = ngramas.most_common(10)
        
        etiquetas_grafico = [ng for ng, _ in top_10][::-1]
        frecuencias = [f for _, f in top_10][::-1]
        
        axes[idx].barh(etiquetas_grafico, frecuencias, color=color, edgecolor='white')
        axes[idx].set_title(f'{nombre_tipo} - {etiqueta}', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Frecuencia', fontsize=11)
        
        # Añadir valores en las barras
        for j, v in enumerate(frecuencias):
            axes[idx].text(v + max(frecuencias) * 0.01, j, f'{v:,}', va='center', fontsize=9)
    
    plt.suptitle(f'Top 10 {nombre_tipo} más frecuentes por Sentimiento', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    nombre_archivo = '5-pares_palabras_frecuentes.png' if n == 2 else '6-trios_palabras_frecuentes.png'
    ruta_grafico = os.path.join(dir_graficos, nombre_archivo)
    plt.savefig(ruta_grafico, dpi=300)
    plt.close()

print(f"\n¡Análisis completado! Los gráficos se han guardado en:\n{os.path.abspath(dir_graficos)}")
