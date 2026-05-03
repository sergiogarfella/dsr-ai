import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("--- DISTRIBUCIÓN DE LONGITUD ---\n")

# Configurar fuente Inter
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'sans-serif']

script_dir = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.normpath(os.path.join(script_dir, '../../datasets/dataset_unificado_analisis.csv'))
dir_graficos = os.path.join(script_dir, 'graficos')

if not os.path.exists(ruta_csv):
    print("Error: No se encuentra el dataset_unificado_analisis.csv. Por favor, ejecuta primero la unificación.")
    exit(1)

os.makedirs(dir_graficos, exist_ok=True)

print("Leyendo dataset y calculando métricas...")
df = pd.read_csv(ruta_csv)

metricas = {
    'longitud': 'Número de Palabras',
    'num_caracteres': 'Número de Caracteres',
    'vocabulario': 'Vocabulario'
}

# 1. Estadísticas Descriptivas en Consola
for col, nombre in metricas.items():
    print(f"\n{nombre.upper()}:")
    print("-" * 85)
    
    # Calcular estadísticas por dataset
    stats = df.groupby('dataset')[col].describe(percentiles=[.25, .5, .75, .95])[['mean', 'std', 'min', '25%', '50%', '75%', '95%', 'max']].round(1)
    stats.columns = ['Media', 'Std', 'Mínimo', 'P25', 'Mediana', 'P75', 'P95', 'Máximo']
    
    # Calcular estadísticas globales (Unificado)
    global_stats = df[col].describe(percentiles=[.25, .5, .75, .95])[['mean', 'std', 'min', '25%', '50%', '75%', '95%', 'max']].round(1).to_frame().T
    global_stats.index = ['Unificado']
    global_stats.columns = ['Media', 'Std', 'Mínimo', 'P25', 'Mediana', 'P75', 'P95', 'Máximo']
    
    # Unir e imprimir
    stats_completas = pd.concat([stats, global_stats])
    print(stats_completas.to_string())

# 2. Generación de Gráficos (Boxplots comparativos)
print("\nGenerando gráficos...")

# Añadir un bloque "Unificado" para que salga en el mismo gráfico
df_unificado = df.copy()
df_unificado['dataset'] = 'Unificado'
df_completo = pd.concat([df, df_unificado], ignore_index=True)

# El orden en el que queremos que aparezcan en el gráfico
orden_datasets = df['dataset'].unique().tolist() + ['Unificado']

nombres_archivos = {
    'longitud': '2-distribucion_palabras.png',
    'num_caracteres': '3-distribucion_caracteres.png',
    'vocabulario': '4-distribucion_vocabulario.png'
}

for col, nombre in metricas.items():
    plt.figure(figsize=(10, 6))
    
    # Utilizamos showfliers=False para omitir los valores atípicos extremos 
    # y que se vea clara la distribución principal (las cajas).
    sns.boxplot(data=df_completo, x='dataset', y=col, hue='dataset', palette='Set2', order=orden_datasets, showfliers=False, legend=False)
    
    plt.title(f'Distribución de {nombre} por Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12, fontweight='medium')
    plt.ylabel(nombre, fontsize=12, fontweight='medium')
    
    plt.tight_layout()
    ruta_grafico = os.path.join(dir_graficos, nombres_archivos[col])
    plt.savefig(ruta_grafico, dpi=300)
    plt.close()

print(f"\n¡Análisis completado! Se han guardado 3 gráficos (.png) en la carpeta:\n{os.path.abspath(dir_graficos)}")