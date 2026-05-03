import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import chi2_contingency

print("--- VOCABULARIO SIGNIFICATIVO POR SENTIMIENTO ---\n")

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

print("Leyendo dataset...")
df = pd.read_csv(ruta_csv)

# =====================================================================
# 1. Conteo de palabras por sentimiento
# =====================================================================
print("\nContando palabras por sentimiento...\n")

textos_pos = df[df['sentimiento'] == 'pos']['texto'].astype(str)
textos_neg = df[df['sentimiento'] == 'neg']['texto'].astype(str)

palabras_pos = ' '.join(textos_pos).lower().split()
palabras_neg = ' '.join(textos_neg).lower().split()

freq_pos = Counter(palabras_pos)
freq_neg = Counter(palabras_neg)

total_pos = len(palabras_pos)
total_neg = len(palabras_neg)

# Vocabulario común (palabras que aparecen en ambos sentimientos)
vocabulario_comun = set(freq_pos.keys()) & set(freq_neg.keys())
print(f"Total de palabras positivas (tokens): {total_pos:,}")
print(f"Total de palabras negativas (tokens): {total_neg:,}")
print(f"Vocabulario exclusivo positivo:       {len(set(freq_pos.keys()) - set(freq_neg.keys())):,}")
print(f"Vocabulario exclusivo negativo:       {len(set(freq_neg.keys()) - set(freq_pos.keys())):,}")
print(f"Vocabulario compartido:               {len(vocabulario_comun):,}")

# =====================================================================
# 2. Log-Odds Ratio con suavizado (palabras más discriminativas)
# =====================================================================
# El log-odds ratio mide cuánto más probable es una palabra en un
# sentimiento vs el otro. Usamos suavizado de Laplace (+1) para evitar
# divisiones por cero y reducir el ruido de palabras muy infrecuentes.
# =====================================================================
print("\n\nLOG-ODDS RATIO (palabras más discriminativas):")
print("=" * 85)

# Solo consideramos palabras con frecuencia mínima para evitar ruido
FREQ_MINIMA = 50

palabras_candidatas = [p for p in vocabulario_comun
                       if freq_pos[p] >= FREQ_MINIMA and freq_neg[p] >= FREQ_MINIMA]

log_odds = {}
for palabra in palabras_candidatas:
    # Proporción suavizada (Laplace smoothing)
    prop_pos = (freq_pos[palabra] + 1) / (total_pos + len(vocabulario_comun))
    prop_neg = (freq_neg[palabra] + 1) / (total_neg + len(vocabulario_comun))
    log_odds[palabra] = np.log2(prop_pos / prop_neg)

# Ordenar por log-odds ratio
log_odds_ordenado = sorted(log_odds.items(), key=lambda x: x[1], reverse=True)

TOP_N = 20

print(f"\nTop {TOP_N} palabras más asociadas a POSITIVO (log-odds > 0):")
print("-" * 85)
print(f"{'Palabra':<20} | {'Freq Pos':>10} | {'Freq Neg':>10} | {'Log-Odds':>10}")
print("-" * 85)
for palabra, lor in log_odds_ordenado[:TOP_N]:
    print(f"{palabra:<20} | {freq_pos[palabra]:>10,} | {freq_neg[palabra]:>10,} | {lor:>+10.3f}")

print(f"\nTop {TOP_N} palabras más asociadas a NEGATIVO (log-odds < 0):")
print("-" * 85)
print(f"{'Palabra':<20} | {'Freq Pos':>10} | {'Freq Neg':>10} | {'Log-Odds':>10}")
print("-" * 85)
for palabra, lor in log_odds_ordenado[-TOP_N:]:
    print(f"{palabra:<20} | {freq_pos[palabra]:>10,} | {freq_neg[palabra]:>10,} | {lor:>+10.3f}")

# =====================================================================
# 3. Test Chi-Cuadrado de independencia para las top palabras
# =====================================================================
# Para cada palabra, construimos una tabla de contingencia 2×2:
#   [freq_palabra_pos, freq_palabra_neg]
#   [total_pos - freq_palabra_pos, total_neg - freq_palabra_neg]
# y aplicamos chi² para comprobar si la asociación palabra-sentimiento
# es estadísticamente significativa.
# =====================================================================
print("\n\nTEST CHI-CUADRADO DE INDEPENDENCIA:")
print("=" * 85)

# Top palabras de ambos extremos
top_positivas = [p for p, _ in log_odds_ordenado[:TOP_N]]
top_negativas = [p for p, _ in log_odds_ordenado[-TOP_N:]]
palabras_test = top_positivas + top_negativas

print(f"\n{'Palabra':<20} | {'χ²':>10} | {'p-valor':>12} | {'V de Cramér':>12} | {'Asociación':<10}")
print("-" * 85)

for palabra in palabras_test:
    # Tabla de contingencia 2×2
    a = freq_pos[palabra]              # palabra en positivo
    b = freq_neg[palabra]              # palabra en negativo
    c = total_pos - a                  # no-palabra en positivo
    d = total_neg - b                  # no-palabra en negativo

    tabla = np.array([[a, b], [c, d]])
    chi2, p_valor, _, _ = chi2_contingency(tabla)

    # V de Cramér (tamaño del efecto para chi²)
    n = tabla.sum()
    v_cramer = np.sqrt(chi2 / n)

    # Interpretación del tamaño del efecto
    if v_cramer < 0.01:
        efecto = "Nulo"
    elif v_cramer < 0.03:
        efecto = "Pequeño"
    elif v_cramer < 0.05:
        efecto = "Medio"
    else:
        efecto = "Grande"

    sig = "***" if p_valor < 0.001 else "**" if p_valor < 0.01 else "*" if p_valor < 0.05 else "ns"
    print(f"{palabra:<20} | {chi2:>10.1f} | {p_valor:>12.2e} | {v_cramer:>12.4f} | {efecto:<10} {sig}")

print("\nSignificación: *** p<0.001, ** p<0.01, * p<0.05, ns = no significativo")

# =====================================================================
# 4. Análisis por Dataset
# =====================================================================
print("\n\nTOP 10 PALABRAS DISCRIMINATIVAS POR DATASET:")
print("=" * 85)

for ds in df['dataset'].unique():
    df_ds = df[df['dataset'] == ds]

    textos_pos_ds = df_ds[df_ds['sentimiento'] == 'pos']['texto'].astype(str)
    textos_neg_ds = df_ds[df_ds['sentimiento'] == 'neg']['texto'].astype(str)

    fp = Counter(' '.join(textos_pos_ds).lower().split())
    fn = Counter(' '.join(textos_neg_ds).lower().split())

    tp = sum(fp.values())
    tn = sum(fn.values())

    vocab_ds = set(fp.keys()) & set(fn.keys())
    candidatas_ds = [p for p in vocab_ds if fp[p] >= 20 and fn[p] >= 20]

    lo_ds = {}
    for palabra in candidatas_ds:
        pp = (fp[palabra] + 1) / (tp + len(vocab_ds))
        pn = (fn[palabra] + 1) / (tn + len(vocab_ds))
        lo_ds[palabra] = np.log2(pp / pn)

    lo_ds_ord = sorted(lo_ds.items(), key=lambda x: x[1], reverse=True)

    print(f"\nDataset: {ds}")
    print(f"  → Más positivas: {', '.join([p for p, _ in lo_ds_ord[:10]])}")
    print(f"  → Más negativas: {', '.join([p for p, _ in lo_ds_ord[-10:]])}")

# =====================================================================
# 5. Gráficos
# =====================================================================
print("\n\nGenerando gráficos...")

# --- Gráfico 1: Barras horizontales de log-odds ratio (top global) ---
top_pos_graf = log_odds_ordenado[:15]
top_neg_graf = log_odds_ordenado[-15:]
datos_grafico = top_neg_graf + top_pos_graf

palabras_g = [p for p, _ in datos_grafico]
valores_g = [v for _, v in datos_grafico]
colores_g = ['#e74c3c' if v < 0 else '#2ecc71' for v in valores_g]

plt.figure(figsize=(10, 10))
bars = plt.barh(range(len(palabras_g)), valores_g, color=colores_g, edgecolor='white', linewidth=0.5)

plt.yticks(range(len(palabras_g)), palabras_g, fontsize=10)
plt.axvline(x=0, color='#333', linewidth=0.8)
plt.xlabel('Log-Odds Ratio (log₂)', fontsize=12, fontweight='medium')
plt.title('Palabras Más Discriminativas por Sentimiento', fontsize=14, fontweight='bold')

# Leyenda manual
from matplotlib.patches import Patch
leyenda = [Patch(facecolor='#2ecc71', label='Asociada a Positivo'),
           Patch(facecolor='#e74c3c', label='Asociada a Negativo')]
plt.legend(handles=leyenda, loc='lower right', fontsize=10)

plt.tight_layout()
ruta_1 = os.path.join(dir_graficos, '1-palabras_discriminativas_global.png')
plt.savefig(ruta_1, dpi=300)
plt.close()

# --- Gráfico 2: Log-odds ratio por dataset (subplots) ---
datasets_unicos = df['dataset'].unique().tolist()
fig, axes = plt.subplots(1, len(datasets_unicos), figsize=(7 * len(datasets_unicos), 9), sharey=False)

if len(datasets_unicos) == 1:
    axes = [axes]

for idx, ds in enumerate(datasets_unicos):
    ax = axes[idx]

    df_ds = df[df['dataset'] == ds]
    textos_pos_ds = df_ds[df_ds['sentimiento'] == 'pos']['texto'].astype(str)
    textos_neg_ds = df_ds[df_ds['sentimiento'] == 'neg']['texto'].astype(str)

    fp = Counter(' '.join(textos_pos_ds).lower().split())
    fn = Counter(' '.join(textos_neg_ds).lower().split())
    tp = sum(fp.values())
    tn = sum(fn.values())
    vocab_ds = set(fp.keys()) & set(fn.keys())
    candidatas_ds = [p for p in vocab_ds if fp[p] >= 20 and fn[p] >= 20]

    lo_ds = {}
    for palabra in candidatas_ds:
        pp = (fp[palabra] + 1) / (tp + len(vocab_ds))
        pn = (fn[palabra] + 1) / (tn + len(vocab_ds))
        lo_ds[palabra] = np.log2(pp / pn)

    lo_ds_ord = sorted(lo_ds.items(), key=lambda x: x[1], reverse=True)

    top_ds = lo_ds_ord[-10:] + lo_ds_ord[:10]
    palabras_ds = [p for p, _ in top_ds]
    valores_ds = [v for _, v in top_ds]
    colores_ds = ['#e74c3c' if v < 0 else '#2ecc71' for v in valores_ds]

    ax.barh(range(len(palabras_ds)), valores_ds, color=colores_ds, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(palabras_ds)))
    ax.set_yticklabels(palabras_ds, fontsize=9)
    ax.axvline(x=0, color='#333', linewidth=0.8)
    ax.set_title(ds, fontsize=13, fontweight='bold')
    ax.set_xlabel('Log-Odds Ratio (log₂)', fontsize=10, fontweight='medium')

fig.suptitle('Palabras Discriminativas por Dataset', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
ruta_2 = os.path.join(dir_graficos, '2-palabras_discriminativas_por_dataset.png')
plt.savefig(ruta_2, dpi=300, bbox_inches='tight')
plt.close()

# --- Gráfico 3: Heatmap de frecuencia relativa (top palabras × sentimiento) ---
top_ambos = [p for p, _ in log_odds_ordenado[:10]] + [p for p, _ in log_odds_ordenado[-10:]]

freq_relativa_pos = [freq_pos[p] / total_pos * 10000 for p in top_ambos]  # por cada 10.000 palabras
freq_relativa_neg = [freq_neg[p] / total_neg * 10000 for p in top_ambos]

matriz = np.array([freq_relativa_pos, freq_relativa_neg])

plt.figure(figsize=(14, 5))
sns.heatmap(matriz, annot=True, fmt='.1f', cmap='RdYlGn',
            xticklabels=top_ambos, yticklabels=['Positivo', 'Negativo'],
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Frecuencia por cada 10.000 palabras'})

plt.title('Frecuencia Relativa de Palabras Discriminativas por Sentimiento', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=11)

plt.tight_layout()
ruta_3 = os.path.join(dir_graficos, '3-heatmap_frecuencia_relativa.png')
plt.savefig(ruta_3, dpi=300)
plt.close()

print(f"\n¡Análisis completado! Se han guardado 3 gráficos (.png) en la carpeta:\n{os.path.abspath(dir_graficos)}")
