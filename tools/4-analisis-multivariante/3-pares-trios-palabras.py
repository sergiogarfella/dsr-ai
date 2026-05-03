import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from collections import Counter

print("--- PARES Y TRÍOS DE PALABRAS SIGNIFICATIVOS ---\n")

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

# Extraer n-gramas de una lista de textos
def extraer_ngramas(textos, n):
    contador = Counter()
    for texto in textos:
        # Limpiar etiquetas HTML y caracteres residuales
        texto = re.sub(r'<[^>]+>', ' ', str(texto))
        texto = re.sub(r'[/>]+', ' ', texto)
        palabras = texto.lower().split()
        ngramas = [' '.join(palabras[i:i+n]) for i in range(len(palabras) - n + 1)]
        contador.update(ngramas)
    return contador

# Para encontrar n-gramas significativos, comparamos la frecuencia relativa
# de cada n-grama en positivos vs negativos. Si un par aparece mucho más
# en positivos que en negativos (proporcionalmente), es significativo para positivos.
MIN_FRECUENCIA = 50  # Mínimo de apariciones para evitar ruido

textos_pos = df[df['sentimiento'] == 'pos']['texto']
textos_neg = df[df['sentimiento'] == 'neg']['texto']

for n, nombre_tipo in [(2, 'Pares de palabras'), (3, 'Tríos de palabras')]:
    ngramas_pos = extraer_ngramas(textos_pos, n)
    ngramas_neg = extraer_ngramas(textos_neg, n)
    
    total_pos = sum(ngramas_pos.values())
    total_neg = sum(ngramas_neg.values())
    
    # Calcular ratio de sobrerrepresentación para cada n-grama
    # ratio = (freq_pos / total_pos) / (freq_neg / total_neg)
    # ratio > 1 = más frecuente en positivos
    # ratio < 1 = más frecuente en negativos
    todos_ngramas = set(ngramas_pos.keys()) | set(ngramas_neg.keys())
    
    scores = []
    for ng in todos_ngramas:
        freq_pos = ngramas_pos.get(ng, 0)
        freq_neg = ngramas_neg.get(ng, 0)
        
        # Filtrar n-gramas poco frecuentes
        if freq_pos + freq_neg < MIN_FRECUENCIA:
            continue
        
        # Añadir suavizado (+1) para evitar divisiones por cero
        ratio = (freq_pos / total_pos) / ((freq_neg + 1) / total_neg)
        scores.append((ng, ratio, freq_pos, freq_neg))
    
    # Ordenar: los más significativos para positivos (ratio alto) y negativos (ratio bajo)
    scores.sort(key=lambda x: x[1], reverse=True)
    top_positivos = scores[:10]
    top_negativos = scores[-10:][::-1]  # Invertir para que el más significativo vaya primero
    
    print(f"\n{nombre_tipo.upper()} SIGNIFICATIVOS PARA POSITIVOS:")
    print("=" * 70)
    print(f"  {'Rango':<6} {nombre_tipo:<30} {'Freq pos':<10} {'Freq neg':<10} {'Ratio':<8}")
    print("  " + "-" * 64)
    for i, (ng, ratio, fp, fn) in enumerate(top_positivos, 1):
        print(f"  {i:<6} {ng:<30} {fp:<10,} {fn:<10,} {ratio:<8.2f}")
    
    print(f"\n{nombre_tipo.upper()} SIGNIFICATIVOS PARA NEGATIVOS:")
    print("=" * 70)
    print(f"  {'Rango':<6} {nombre_tipo:<30} {'Freq pos':<10} {'Freq neg':<10} {'Ratio':<8}")
    print("  " + "-" * 64)
    for i, (ng, ratio, fp, fn) in enumerate(top_negativos, 1):
        inv_ratio = 1/ratio if ratio > 0 else float('inf')
        print(f"  {i:<6} {ng:<30} {fp:<10,} {fn:<10,} {inv_ratio:<8.2f}")

# Gráficos
print("\nGenerando gráficos...")

for n, nombre_tipo, num_archivo in [(2, 'Pares de palabras', '5'), (3, 'Tríos de palabras', '6')]:
    ngramas_pos = extraer_ngramas(textos_pos, n)
    ngramas_neg = extraer_ngramas(textos_neg, n)
    total_pos = sum(ngramas_pos.values())
    total_neg = sum(ngramas_neg.values())
    
    todos_ngramas = set(ngramas_pos.keys()) | set(ngramas_neg.keys())
    scores = []
    for ng in todos_ngramas:
        fp = ngramas_pos.get(ng, 0)
        fn = ngramas_neg.get(ng, 0)
        if fp + fn < MIN_FRECUENCIA:
            continue
        ratio = (fp / total_pos) / ((fn + 1) / total_neg)
        scores.append((ng, ratio, fp, fn))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    top_pos = scores[:10]
    top_neg = scores[-10:][::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Positivos - ordenar por frecuencia para el gráfico
    top_pos_sorted = sorted(top_pos, key=lambda x: x[2])  # Ascendente para barh
    etiquetas = [ng for ng, _, _, _ in top_pos_sorted]
    valores = [fp for _, _, fp, _ in top_pos_sorted]
    axes[0].barh(etiquetas, valores, color='#2ecc71', edgecolor='white')
    axes[0].set_title(f'{nombre_tipo} - Positivos', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Frecuencia', fontsize=11)
    for j, v in enumerate(valores):
        axes[0].text(v + max(valores) * 0.01, j, f'{v:,}', va='center', fontsize=9)
    
    # Negativos - ordenar por frecuencia para el gráfico
    top_neg_sorted = sorted(top_neg, key=lambda x: x[3])  # Ascendente para barh
    etiquetas = [ng for ng, _, _, _ in top_neg_sorted]
    valores = [fn for _, _, _, fn in top_neg_sorted]
    axes[1].barh(etiquetas, valores, color='#ff4d4d', edgecolor='white')
    axes[1].set_title(f'{nombre_tipo} - Negativos', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Frecuencia', fontsize=11)
    for j, v in enumerate(valores):
        axes[1].text(v + max(valores) * 0.01, j, f'{v:,}', va='center', fontsize=9)
    
    plt.suptitle(f'Top 10 {nombre_tipo} significativos por Sentimiento', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    nombre_archivo = f'{num_archivo}-{nombre_tipo.lower().replace(" ", "_")}_significativos.png'
    ruta_grafico = os.path.join(dir_graficos, nombre_archivo)
    plt.savefig(ruta_grafico, dpi=300)
    plt.close()

print(f"\n¡Análisis completado! Los gráficos se han guardado en:\n{os.path.abspath(dir_graficos)}")
