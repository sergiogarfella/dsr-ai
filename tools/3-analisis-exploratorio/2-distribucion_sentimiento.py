import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("--- DISTRIBUCIÓN DE SENTIMIENTO ---\n")

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

# Crear un duplicado de todo el dataframe pero con dataset="Unificado"
df_unificado_copia = df.copy()
df_unificado_copia['dataset'] = 'Unificado'

# Concatenar para tener los 3 originales + el unificado
df_completo = pd.concat([df, df_unificado_copia], ignore_index=True)

# Imprimir distribución en consola
datasets = df['dataset'].unique().tolist() + ['Unificado']

print("DISTRIBUCIÓN POR DATASET:")
print("-" * 50)
for ds in datasets:
    df_ds = df_completo[df_completo['dataset'] == ds]
    conteo = df_ds['sentimiento'].value_counts()
    negativas = conteo.get('neg', 0)
    positivas = conteo.get('pos', 0)
    total = negativas + positivas
    
    print(f"Dataset: {ds}")
    print(f"  - Negativas: {negativas:,} ({(negativas/total)*100:.1f}%)")
    print(f"  - Positivas: {positivas:,} ({(positivas/total)*100:.1f}%)")
    print(f"  - Total:     {total:,}\n")

# Único gráfico comparativo
plt.figure(figsize=(12, 6))
ax = sns.countplot(data=df_completo, x='dataset', hue='sentimiento', palette={'neg': '#ff4d4d', 'pos': '#2ecc71'}, order=datasets)
plt.title('Distribución de Sentimiento por Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Dataset', fontsize=12, fontweight='medium')
plt.ylabel('Cantidad de Reseñas', fontsize=12, fontweight='medium')
plt.legend(title='Sentimiento', fontsize=10, title_fontsize=11)

# Aumentar margen superior para que los textos no se corten
y_max = df_completo.groupby(['dataset', 'sentimiento']).size().max()
plt.ylim(0, y_max * 1.15)

# Añadir valores sobre las barras
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height):,}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=10, fontweight='medium')

plt.tight_layout()
ruta_grafico = os.path.join(dir_graficos, '1-distribucion_sentimiento.png')
plt.savefig(ruta_grafico, dpi=300)
plt.close()

print(f"¡Análisis completado! El gráfico se ha guardado en:\n{os.path.abspath(ruta_grafico)}")
