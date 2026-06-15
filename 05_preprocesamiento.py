import pandas as pd
import numpy as np

INPUT_FILE  = 'data/dataset_final_tfm.csv'
OUTPUT_FILE = 'data/dataset_final_tfm_limpio.csv'

df = pd.read_csv(INPUT_FILE, parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Imputación del apagón 
df = df.set_index('datetime')

inicio_apagon = '2025-04-28 12:00:00'
fin_apagon    = '2025-04-29 06:00:00'
mascara = (df.index >= inicio_apagon) & (df.index <= fin_apagon)

for ts in df[mascara].index:
    ventana = df[
        (df.index >= ts - pd.Timedelta(days=7)) &
        (df.index <= ts + pd.Timedelta(days=7)) &
        (df.index.hour == ts.hour) &
        ~mascara
    ]
    df.loc[ts, 'value'] = ventana['value'].median()

df = df.reset_index()

# Eliminar es_domingo 
df = df.drop(columns=['es_domingo'], errors='ignore')

df.to_csv(OUTPUT_FILE, index=False)
print(f"Guardado: {OUTPUT_FILE} ({len(df)} registros, {len(df.columns)} columnas)")