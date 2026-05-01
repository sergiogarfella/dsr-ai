import pandas as pd
import os


ruta_csv = 'datasets/dataset_unificado.csv'

if not os.path.exists(ruta_csv):
    print("Error: No se encuentra el dataset_unificado.csv.")
    exit(1)

df = pd.read_csv(ruta_csv)

# Distribución de Volumen por Dataset y Sentimiento
distribucion = df.groupby(['dataset', 'sentimiento']).size().unstack(fill_value=0)
if 'neg' not in distribucion.columns: distribucion['neg'] = 0
if 'pos' not in distribucion.columns: distribucion['pos'] = 0
distribucion['Total'] = distribucion['neg'] + distribucion['pos']
distribucion['Ratio pos/neg'] = (distribucion['pos'] / distribucion['neg'].replace(0, 1)).round(3)

print("DISTRIBUCIÓN DE VOLUMEN POR DATASET Y SENTIMIENTO:")
print(f"{'Dataset':<26} | {'Negativas':<9} | {'Positivas':<9} | {'Total':<9} | {'Ratio pos/neg'}")
print("-" * 80)
for dataset, row in distribucion.iterrows():
    print(f"{dataset:<26} | {int(row['neg']):<9,} | {int(row['pos']):<9,} | {int(row['Total']):<9,} | {row['Ratio pos/neg']:.3f}")

total_negativas = int(distribucion['neg'].sum())
total_positivas = int(distribucion['pos'].sum())
total_general = int(distribucion['Total'].sum())
ratio_general = round(total_positivas / (total_negativas if total_negativas > 0 else 1), 3)

print("-" * 80)
print(f"{'Total':<26} | {total_negativas:<9,} | {total_positivas:<9,} | {total_general:<9,} | {ratio_general:.3f}\n")