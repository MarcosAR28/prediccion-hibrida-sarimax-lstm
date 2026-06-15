import os
import gc
import joblib
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import boxcox, levene

warnings.filterwarnings('ignore')

INPUT_FILE  = 'data/dataset_final_tfm_limpio.csv'
os.makedirs('models/sarimax', exist_ok=True)
os.makedirs('results/sarimax', exist_ok=True)

FECHA_FIN_TRAIN = '2023-12-31 23:00:00'
FECHA_FIN_VAL   = '2024-12-31 23:00:00'
HORAS_CONTEXTO  = 4 * 24

EXOGENAS_GRID = [
    'hdd', 'cdd', 'radiacion_solar', 'humedad', 'viento',
    'es_lunes', 'es_martes', 'es_miercoles', 'es_jueves', 'es_viernes', 'es_sabado',
    'es_festivo_nacional', 'peso_festivo_auto', 'es_vispera_festivo', 'es_puente',
    'navidad', 'semana_santa', 'agosto', 'horario_verano',
]

EXOGENAS = [
    'es_lunes', 'es_martes', 'es_miercoles', 'es_jueves', 'es_viernes', 'es_sabado',
    'navidad',
]


def mape(y_real, y_pred):
    y_real, y_pred = np.array(y_real), np.array(y_pred)
    mask = y_real != 0
    return np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100

def metricas(nombre, y_real, y_pred):
    y_real, y_pred = np.array(y_real), np.array(y_pred)
    err = y_real - y_pred
    mae_v  = np.mean(np.abs(err))
    rmse_v = np.sqrt(np.mean(err ** 2))
    mape_v = mape(y_real, y_pred)
    print(f'{nombre:<20}  MAE={mae_v:.2f} MW   RMSE={rmse_v:.2f} MW   MAPE={mape_v:.3f}%')
    return {'MAE': mae_v, 'RMSE': rmse_v, 'MAPE': mape_v}


# 1. Carga y partición de datos

df = pd.read_csv(INPUT_FILE, parse_dates=['datetime'])
df = df.sort_values('datetime')
df = df[['datetime', 'value'] + EXOGENAS_GRID].copy()
df.set_index('datetime', inplace=True)
df = df.asfreq('h')

# dividimos conjuntos de datos
train = df[df.index <= FECHA_FIN_TRAIN].copy()
val   = df[(df.index > FECHA_FIN_TRAIN) & (df.index <= FECHA_FIN_VAL)].copy()
test  = df[df.index > FECHA_FIN_VAL].copy()

val_extendido  = df.loc[val.index.min()  - pd.Timedelta(hours=HORAS_CONTEXTO) : val.index.max()].copy()
test_extendido = df.loc[test.index.min() - pd.Timedelta(hours=HORAS_CONTEXTO) : test.index.max()].copy()

print(f'Train : {train.index.min().date()} → {train.index.max().date()}  ({len(train):,} horas)')
print(f'Val   : {val.index.min().date()} → {val.index.max().date()}  ({len(val):,} horas)')
print(f'Test  : {test.index.min().date()} → {test.index.max().date()}  ({len(test):,} horas)')


# =============================================================================
# 2. Transformación raíz cuadrada y análisis de estacionariedad
# =============================================================================

_, lambda_mle = boxcox(train['value'].values)
print(f"Lambda óptimo (MLE): {lambda_mle:.4f}")   # optimo cercano a raiz cuadrada

train_sqrt = np.sqrt(train['value'])

# Contrastes de homocedasticidad
lm_orig, p_orig, _, _ = het_arch(train['value'] - train['value'].mean(), nlags=24)
lm_sqrt, p_sqrt, _, _ = het_arch(train_sqrt - train_sqrt.mean(), nlags=24)
print(f"Test ARCH-LM (nlags=24):")
print(f"   Original  — LM={lm_orig:.2f}  p={p_orig:.4f}")
print(f"   Sqrt      — LM={lm_sqrt:.2f}  p={p_sqrt:.4f}")

meses_inv = [12, 1, 2]
stat_orig, p_lev_orig = levene(train[train.index.month.isin(meses_inv)]['value'],
                                train[~train.index.month.isin(meses_inv)]['value'])
stat_sqrt, p_lev_sqrt = levene(train_sqrt[train_sqrt.index.month.isin(meses_inv)],
                                train_sqrt[~train_sqrt.index.month.isin(meses_inv)])
print(f"Test de Levene (invierno vs. resto):")
print(f"   Original  — W={stat_orig:.2f}  p={p_lev_orig:.4f}")
print(f"   Sqrt      — W={stat_sqrt:.2f}  p={p_lev_sqrt:.4f}")

# Contrastes ADF/KPSS
def test_estacionariedad(serie, nombre):
    adf = adfuller(serie.dropna(), autolag='AIC')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kp = kpss(serie.dropna(), regression='c', nlags='auto')
    print(f"{nombre}: ADF p={adf[1]:.4f}  KPSS p={kp[1]:.4f}")

for serie, nombre in [
    (train_sqrt,                          '1. Nivel (d=0, D=0)'),
    (train_sqrt.diff(1).dropna(),         '2. d=1, D=0'),
    (train_sqrt.diff(24).dropna(),        '3. d=0, D=1'),
    (train_sqrt.diff(24).diff(1).dropna(),'4. d=1, D=1'),
]:
    test_estacionariedad(serie, nombre)

# Lags clave de ACF/PACF
nlags    = 72
serie_id = train_sqrt.diff(1).diff(24).dropna()
acf_vals  = acf(serie_id,  nlags=nlags, fft=True)
pacf_vals = pacf(serie_id, nlags=nlags)

print(f"\n{'Lag':>5}  {'ACF':>8}  {'PACF':>8}")
for lag in [1, 2, 3, 4, 5, 24, 48, 72]:
    print(f"{lag:>5}  {acf_vals[lag]:>8.4f}  {pacf_vals[lag]:>8.4f}")


# 3. Grid search

train_sqrt_full    = np.sqrt(train['value'])
val_ext_sqrt_endog = np.sqrt(val_extendido['value'])
val_ext_exog       = val_extendido[EXOGENAS_GRID]
y_real_val         = val['value'].values

# aqui poner candidatos
candidatos = [
    # (p, d, q, P, D, Q)
]

resultados = []

for p, d, q, P, D, Q in candidatos:
    fit = model_val = None
    try:
        modelo = SARIMAX(
            train_sqrt_full,
            exog=train[EXOGENAS_GRID],
            order=(p, d, q),
            seasonal_order=(P, D, Q, 24),
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend='n',
        )
        fit = modelo.fit(disp=False, maxiter=200)

        lb_p = acorr_ljungbox(
            np.array(fit.resid.dropna()), lags=[24], model_df=p + q + P + Q,
        )['lb_pvalue'].iloc[0]

        model_val   = fit.apply(endog=val_ext_sqrt_endog, exog=val_ext_exog, refit=False)
        pred_mw_val = model_val.fittedvalues.iloc[HORAS_CONTEXTO:].values ** 2

        resultados.append({
            'Modelo'  : f'({p},{d},{q})({P},{D},{Q})24',
            'AIC'     : round(fit.aic, 2),
            'BIC'     : round(fit.bic, 2),
            'LB_p24'  : round(lb_p, 4),
            'MAE_val' : round(np.mean(np.abs(y_real_val - pred_mw_val)), 2),
            'RMSE_val': round(np.sqrt(np.mean((y_real_val - pred_mw_val) ** 2)), 2),
            'MAPE_val': round(mape(y_real_val, pred_mw_val), 3),
        })
        print(f"({p},{d},{q})({P},{D},{Q})24  AIC={fit.aic:.2f}  MAPE_val={resultados[-1]['MAPE_val']:.3f}%")

    except Exception as e:
        print(f"Error ({p},{d},{q})({P},{D},{Q})24: {e}")
    finally:
        del fit, model_val
        gc.collect()

if resultados:
    df_grid = pd.DataFrame(resultados).sort_values('MAPE_val').reset_index(drop=True)
    print(df_grid.to_string(index=False))
    df_grid.to_csv('results/sarimax/resultados_grid.csv', index=False)


# el grid search elige el modelo definitivo

# 4. Entrenamiento del modelo definitivo


p_f, d_f, q_f = 4, 1, 3
P_f, D_f, Q_f = 2, 1, 2
MODEL_PATH = f'models/sarimax/sarimax_{p_f}{d_f}{q_f}_{P_f}{D_f}{Q_f}_24.joblib'

if os.path.exists(MODEL_PATH):
    model_fit = joblib.load(MODEL_PATH)
    print(f"Modelo cargado: {MODEL_PATH}")
else:
    modelo_def = SARIMAX(
        train_sqrt_full,
        exog=train[EXOGENAS],
        order=(p_f, d_f, q_f),
        seasonal_order=(P_f, D_f, Q_f, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
        trend='n',
    )
    model_fit = modelo_def.fit(disp=True, maxiter=200)
    joblib.dump(model_fit, MODEL_PATH)
    print(f"Modelo entrenado y guardado: {MODEL_PATH}")

print(model_fit.summary())


# 5. Diagnóstico de residuos in-sample

residuos = pd.Series(model_fit.resid).iloc[HORAS_CONTEXTO:].dropna()

lb = acorr_ljungbox(residuos, lags=[24, 48, 168], model_df=p_f + q_f + P_f + Q_f)
print("Ljung-Box:")
for lag in [24, 48, 168]:
    print(f"   Lag {lag:3d}: Q={lb.loc[lag,'lb_stat']:.2f}  p={lb.loc[lag,'lb_pvalue']:.4e}")

jb_stat, jb_p = stats.jarque_bera(residuos)
print(f"\nJarque-Bera: JB={jb_stat:.2f}  p={jb_p:.4e}")
print(f"   Asimetría: {stats.skew(residuos):.3f}  Curtosis: {stats.kurtosis(residuos)+3:.3f}")

arch = het_arch(residuos, nlags=12)
print(f"\nARCH-LM: LM={arch[0]:.2f}  p={arch[1]:.4e}")


# 6. Exportación de resultados

# Train
fechas_train  = train.index[HORAS_CONTEXTO:]
y_real_train  = train['value'].iloc[HORAS_CONTEXTO:].values
resid_sqrt    = np.array(model_fit.resid)[HORAS_CONTEXTO:]
fitted_sqrt   = model_fit.fittedvalues.iloc[HORAS_CONTEXTO:].values
resid_mw      = resid_sqrt * 2 * fitted_sqrt
pred_mw_train = y_real_train - resid_mw

df_train_out = pd.DataFrame({
    'real': y_real_train, 'predicho': pred_mw_train,
    'residuo': resid_mw,  'conjunto': 'train',
}, index=fechas_train)

# Validación
model_v   = model_fit.apply(endog=np.sqrt(val_extendido['value']), exog=val_extendido[EXOGENAS], refit=False)
pred_mw_v = model_v.fittedvalues.iloc[HORAS_CONTEXTO:].values ** 2
y_real_v  = val['value'].values
df_val_out = pd.DataFrame({
    'real': y_real_v, 'predicho': pred_mw_v,
    'residuo': y_real_v - pred_mw_v, 'conjunto': 'validacion',
}, index=val.index)

# Test
model_t   = model_fit.apply(endog=np.sqrt(test_extendido['value']), exog=test_extendido[EXOGENAS], refit=False)
pred_mw_t = model_t.fittedvalues.iloc[HORAS_CONTEXTO:].values ** 2
y_real_t  = test['value'].values
df_test_out = pd.DataFrame({
    'real': y_real_t, 'predicho': pred_mw_t,
    'residuo': y_real_t - pred_mw_t, 'conjunto': 'test',
}, index=test.index)

df_global = pd.concat([df_train_out, df_val_out, df_test_out]).sort_index()

df_global[['real', 'predicho', 'conjunto']].to_csv(
    'results/sarimax/resultados_sarimax.csv', index=True, index_label='datetime')
df_global[['residuo', 'conjunto']].to_csv(
    'results/sarimax/residuos_sarimax.csv', index=True, index_label='datetime')

df_test_out['error_abs'] = np.abs(df_test_out['residuo'])
df_test_out.to_csv('results/sarimax/resultados_sarimax_test.csv', index=True, index_label='datetime')

metricas('Test 2025', y_real_t, pred_mw_t)
