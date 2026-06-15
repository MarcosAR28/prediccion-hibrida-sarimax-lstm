import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.makedirs('figures/comparativa', exist_ok=True)
os.makedirs('results/comparativa', exist_ok=True)

COLOR_SARIMAX = '#1a6faf'
COLOR_DIRECTO = '#e67e22'
COLOR_HIBRIDO = '#27ae60'
COLOR_REAL    = '#2c3e50'


# CARGA Y CONSTRUCCIÓN DEL DATASET DE COMPARATIVA

df_sarimax      = pd.read_csv('results/sarimax/resultados_sarimax.csv',              parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
df_lstm_hibrido = pd.read_csv('results/lstm/lstm_pred_completo.csv',                 parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
df_lstm_directo = pd.read_csv('results/lstm_directo/lstm_directo_pred_completo.csv', parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
df_main         = pd.read_csv('data/dataset_final_tfm_limpio.csv',                   parse_dates=['datetime'])

sarimax_eval = df_sarimax[df_sarimax['conjunto'].isin(['validacion', 'test'])].copy()

df_comp = (sarimax_eval
           .merge(df_lstm_hibrido[['datetime', 'pred_lstm_mw']], on='datetime', how='inner')
           .merge(df_lstm_directo[['datetime', 'pred_lstm_directo']], on='datetime', how='inner')
           .merge(df_main[['datetime', 'hora', 'es_festivo_nacional', 'es_sabado',
                            'es_puente', 'es_vispera_festivo', 'peso_festivo_auto']], on='datetime', how='left'))

df_comp['pred_hibrido'] = df_comp['predicho'] + df_comp['pred_lstm_mw']

print(f"Dataset de comparativa: {len(df_comp):,} instancias  "
      f"(val={( df_comp['conjunto']=='validacion').sum():,}  "
      f"test={(df_comp['conjunto']=='test').sum():,})")


# MÉTRICAS

def metricas(y_real, y_pred):
    y_real, y_pred = np.array(y_real).flatten(), np.array(y_pred).flatten()
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mask = y_real > 0
    mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

resultados = {}
for conj in ['validacion', 'test']:
    sub = df_comp[df_comp['conjunto'] == conj]
    y   = sub['real'].values
    resultados[(conj, 'SARIMAX')]      = metricas(y, sub['predicho'].values)
    resultados[(conj, 'LSTM directo')] = metricas(y, sub['pred_lstm_directo'].values)
    resultados[(conj, 'Híbrido')]      = metricas(y, sub['pred_hibrido'].values)

df_tabla = pd.DataFrame([
    {'Conjunto': conj.capitalize(), 'Modelo': modelo,
     'MAE (MW)': round(met['MAE'], 2), 'RMSE (MW)': round(met['RMSE'], 2), 'MAPE (%)': round(met['MAPE'], 3)}
    for (conj, modelo), met in resultados.items()
])
print(df_tabla.to_string(index=False))

# Mejoras respecto al SARIMAX
df_mejora = pd.DataFrame([
    {'Conjunto': conj.capitalize(), 'Modelo': modelo,
     'Δ MAE (%)':  round((resultados[(conj, 'SARIMAX')]['MAE']  - resultados[(conj, modelo)]['MAE'])  / resultados[(conj, 'SARIMAX')]['MAE']  * 100, 2),
     'Δ RMSE (%)': round((resultados[(conj, 'SARIMAX')]['RMSE'] - resultados[(conj, modelo)]['RMSE']) / resultados[(conj, 'SARIMAX')]['RMSE'] * 100, 2),
     'Δ MAPE (%)': round((resultados[(conj, 'SARIMAX')]['MAPE'] - resultados[(conj, modelo)]['MAPE']) / resultados[(conj, 'SARIMAX')]['MAPE'] * 100, 2)}
    for conj in ['validacion', 'test']
    for modelo in ['LSTM directo', 'Híbrido']
])
print("\nMejora (%) respecto al SARIMAX:")
print(df_mejora.to_string(index=False))


# FIGURA 1: BARRAS COMPARATIVAS (MAE, RMSE, MAPE)

modelos = ['SARIMAX', 'LSTM directo', 'Híbrido']
colores = [COLOR_SARIMAX, COLOR_DIRECTO, COLOR_HIBRIDO]
x, w    = np.arange(len(modelos)), 0.35

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metrica, ylabel in zip(axes,
    ['MAE (MW)', 'RMSE (MW)', 'MAPE (%)'],
    ['MAE (MW)', 'RMSE (MW)', 'MAPE (%)']):

    key = metrica.split()[0]
    v_val  = [resultados[('validacion', m)][key] for m in modelos]
    v_test = [resultados[('test',       m)][key] for m in modelos]

    bv = ax.bar(x - w/2, v_val,  w, color=colores, alpha=0.65, edgecolor='white')
    bt = ax.bar(x + w/2, v_test, w, color=colores, alpha=0.95, edgecolor='white')
    for bar in bv:
        bar.set_hatch('//')

    ax.set_xticks(x); ax.set_xticklabels(modelos, fontsize=10)
    ax.set_ylabel(ylabel); ax.set_title(ylabel)
    for bar, val in zip(list(bv) + list(bt), v_val + v_test):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(v_val + v_test) * 0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

fig.legend(handles=[
    mpatches.Patch(facecolor='grey', hatch='//', alpha=0.65, label='Validación'),
    mpatches.Patch(facecolor='grey', alpha=0.95,             label='Test'),
], loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('figures/comparativa/fig_metricas_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()


# FIGURA 2: PREDICCIONES HORARIAS (VENTANA DE 2 SEMANAS)

for conj, titulo in [('validacion', 'Validación — enero 2024'), ('test', 'Test — enero 2025')]:
    sub = df_comp[df_comp['conjunto'] == conj].copy()

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(sub['datetime'].iloc[1152:1200], sub['real'].iloc[1152:1200]             / 1000, color=COLOR_REAL,    lw=1.6, label='Real',         zorder=4)
    ax.plot(sub['datetime'].iloc[1152:1200], sub['predicho'].iloc[1152:1200]         / 1000, color=COLOR_SARIMAX, lw=1.2, linestyle='--',        label='SARIMAX',      zorder=3)
    ax.plot(sub['datetime'].iloc[1152:1200], sub['pred_lstm_directo'].iloc[1152:1200]/ 1000, color=COLOR_DIRECTO, lw=1.2, linestyle=':',         label='LSTM directo', zorder=3)
    ax.plot(sub['datetime'].iloc[1152:1200], sub['pred_hibrido'].iloc[1152:1200]     / 1000, color=COLOR_HIBRIDO, lw=1.4, linestyle='-.',         label='Híbrido',      zorder=3)
    ax.set_xlabel('Fecha'); ax.set_ylabel('Demanda (GW)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'figures/comparativa/fig_pred_{conj}.png', dpi=150, bbox_inches='tight')
    plt.show()


# FIGURA 3: VISTA GLOBAL (MEDIA DIARIA)

for conj, titulo in [('validacion', 'Validación 2024'), ('test', 'Test 2025')]:
    sub = df_comp[df_comp['conjunto'] == conj].copy()
    sub['fecha'] = sub['datetime'].dt.normalize()
    diario = sub.groupby('fecha')[['real', 'predicho', 'pred_lstm_directo', 'pred_hibrido']].mean()

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(diario.index, diario['real']              / 1000, color=COLOR_REAL,    lw=1.6, label='Real')
    ax.plot(diario.index, diario['predicho']          / 1000, color=COLOR_SARIMAX, lw=1.2, linestyle='--', label='SARIMAX')
    ax.plot(diario.index, diario['pred_lstm_directo'] / 1000, color=COLOR_DIRECTO, lw=1.2, linestyle=':',  label='LSTM directo')
    ax.plot(diario.index, diario['pred_hibrido']      / 1000, color=COLOR_HIBRIDO, lw=1.4, linestyle='-.', label='Híbrido')
    ax.set_ylabel('Demanda media diaria (GW)')
    ax.set_title(f'Vista global (media diaria) — {titulo}')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=30, ha='right'); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'figures/comparativa/fig_global_{conj}.png', dpi=150, bbox_inches='tight')
    plt.show()


# EXPORTACIÓN

df_tabla.to_csv('results/comparativa/tabla_metricas_comparativa.csv', index=False)
df_mejora.to_csv('results/comparativa/tabla_mejoras_sobre_sarimax.csv', index=False)

cols_export = ['datetime', 'conjunto', 'real', 'predicho', 'pred_lstm_directo', 'pred_lstm_mw', 'pred_hibrido']
df_comp[cols_export].to_csv('results/comparativa/predicciones_comparativa.csv', index=False)

print("Guardado: results/comparativa/")
