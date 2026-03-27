"""
═══════════════════════════════════════════════════════════════════════
GENERADOR DE TABLAS VISUALES (HEATMAPS) PARA EL TFG
Extrae los datos de final_summary.json y crea tablas coloreadas
═══════════════════════════════════════════════════════════════════════
"""

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_heatmaps():
    summary_path = 'results/final_summary.json'
    
    # 1. Comprobamos que el archivo de resultados exista
    if not os.path.exists(summary_path):
        print(f"No se encuentra el archivo {summary_path}. ¡Entrena los modelos primero!")
        return

    # 2. Cargamos los datos del JSON
    with open(summary_path, 'r') as f:
        data = json.load(f)

    # 3. Preparamos los "Excel" vacíos (DataFrames de Pandas)
    modelos = list(data.keys())
    
    # Listas para guardar las filas de nuestras tablas
    rmse_rows = []
    r2_rows = []

    # 4. Rellenamos las filas con los datos de cada modelo
    for model in modelos:
        # Extraemos los datos de los 5 Folds y la Media Global
        rmse_folds = data[model]['per_fold']['rmse']
        r2_folds = data[model]['per_fold']['r2']
        
        rmse_mean = data[model]['rmse_mean']
        r2_mean = data[model]['r2_mean']
        
        # Creamos la fila para RMSE
        rmse_row = {f'Fold {i+1}': rmse_folds[i] for i in range(5)}
        rmse_row['Media Global'] = rmse_mean
        rmse_rows.append(rmse_row)
        
        # Creamos la fila para R2
        r2_row = {f'Fold {i+1}': r2_folds[i] for i in range(5)}
        r2_row['Media Global'] = r2_mean
        r2_rows.append(r2_row)

    # Convertimos las listas en Tablas Oficiales de Pandas
    df_rmse = pd.DataFrame(rmse_rows, index=modelos)
    df_r2 = pd.DataFrame(r2_rows, index=modelos)

    # =================================================================
    # DIBUJAR TABLA 1: RMSE (Error - Menor es mejor)
    # =================================================================
    plt.figure(figsize=(10, 4)) # Tamaño de la imagen
    
    # cmap="RdYlGn_r" significa: Red (Rojo) -> Yellow (Amarillo) -> Green (Verde). 
    # La "_r" final invierte la paleta para que los números BAJOS sean VERDES (porque es un error).
    sns.heatmap(df_rmse, annot=True, fmt=".4f", cmap="RdYlGn_r", cbar_kws={'label': 'RMSE (Menor es mejor)'}, linewidths=1, linecolor='black')
    
    plt.title("Evaluación 5-Fold CV: Raíz del Error Cuadrático Medio (RMSE)", fontsize=14, pad=15, fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/tabla_RMSE_colores.png', dpi=300)
    plt.close()
    print("Tabla RMSE guardada en: results/tabla_RMSE_colores.png")

    # =================================================================
    # DIBUJAR TABLA 2: R² (Precisión - Mayor es mejor)
    # =================================================================
    plt.figure(figsize=(10, 4))
    
    # cmap="RdYlGn" (SIN la _r). Aquí los números ALTOS son VERDES.
    sns.heatmap(df_r2, annot=True, fmt=".4f", cmap="RdYlGn", cbar_kws={'label': 'R² (Mayor es mejor)'}, linewidths=1, linecolor='black')
    
    plt.title("Evaluación 5-Fold CV: Coeficiente de Determinación (R²)", fontsize=14, pad=15, fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/tabla_R2_colores.png', dpi=300)
    plt.close()
    print("Tabla R² guardada en: results/tabla_R2_colores.png")

if __name__ == "__main__":
    print("\nGenerando tablas visuales para el TFG...")
    generate_heatmaps()