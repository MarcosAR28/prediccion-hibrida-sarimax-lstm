import os
import copy
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = 'data/dataset_final_tfm_limpio.csv'

EXOGENAS = [
    'hdd', 'cdd', 'radiacion_solar', 'humedad', 'viento',
    'es_lunes', 'es_martes', 'es_miercoles', 'es_jueves', 'es_viernes', 'es_sabado',
    'es_festivo_nacional', 'peso_festivo_auto', 'es_vispera_festivo', 'es_puente',
    'navidad', 'semana_santa', 'agosto', 'horario_verano',
]

FECHA_FIN_TRAIN = '2023-12-31 23:00:00'
FECHA_FIN_VAL   = '2024-12-31 23:00:00'


# Se define el modelo a entrenar previamente
LOOK_BACK    = 24
BATCH_SIZE   = 64
N_EPOCHS     = 300
PATIENCE     = 20
HIDDEN_SIZE  = 256
NUM_LAYERS   = 2
FC_UNITS     = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.002
N_FEATURES   = 1 + len(EXOGENAS)

MODEL_PATH = 'models/lstm_directo/mejor_lstm_directo.pt'

os.makedirs('models/lstm_directo',  exist_ok=True)
os.makedirs('results/lstm_directo', exist_ok=True)


def fijar_semilla(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# CARGA Y PARTICIÓN DE DATOS

df = pd.read_csv(DATASET_PATH, parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

cols_usar = ['datetime', 'value'] + EXOGENAS
if df[cols_usar].isnull().sum().sum() > 0:
    df[cols_usar] = df[cols_usar].interpolate(method='linear')

train_df = df[df['datetime'] <= FECHA_FIN_TRAIN].copy()
val_df   = df[(df['datetime'] > FECHA_FIN_TRAIN) & (df['datetime'] <= FECHA_FIN_VAL)].copy()
test_df  = df[df['datetime'] > FECHA_FIN_VAL].copy()


# ESCALADO Y CONSTRUCCIÓN DE SECUENCIAS

idx_val  = df[df['datetime'] > FECHA_FIN_TRAIN].index
idx_test = df[df['datetime'] > FECHA_FIN_VAL].index

val_df_ext  = df.loc[max(0, idx_val[0]  - LOOK_BACK) : idx_val[-1]].copy()
test_df_ext = df.loc[max(0, idx_test[0] - LOOK_BACK) : idx_test[-1]].copy()


def extraer_arrays(sub_df):
    return sub_df['value'].values.reshape(-1, 1), sub_df[EXOGENAS].values


dem_train, exog_train = extraer_arrays(train_df)
dem_val,   exog_val   = extraer_arrays(val_df_ext)
dem_test,  exog_test  = extraer_arrays(test_df_ext)

scaler_demanda = MinMaxScaler(feature_range=(-1, 1))
scaler_exog    = MinMaxScaler(feature_range=(0, 1))

dem_train_sc  = scaler_demanda.fit_transform(dem_train).flatten()
dem_val_sc    = scaler_demanda.transform(dem_val).flatten()
dem_test_sc   = scaler_demanda.transform(dem_test).flatten()
exog_train_sc = scaler_exog.fit_transform(exog_train)
exog_val_sc   = scaler_exog.transform(exog_val)
exog_test_sc  = scaler_exog.transform(exog_test)


class DemandaLSTMDataset(Dataset):
    def __init__(self, demanda_sc, exogenas_sc, look_back):
        demanda  = torch.from_numpy(demanda_sc.reshape(-1, 1)).float()
        exogenas = torch.from_numpy(exogenas_sc).float()
        X_list, y_list = [], []
        for i in range(len(demanda) - look_back):
            seq = torch.cat([demanda[i : i + look_back],
                             exogenas[i + 1 : i + look_back + 1]], dim=1)
            X_list.append(seq)
            y_list.append(demanda[i + look_back, 0])
        self.X = torch.stack(X_list)
        self.y = torch.stack(y_list)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


ds_train = DemandaLSTMDataset(dem_train_sc, exog_train_sc, LOOK_BACK)
ds_val   = DemandaLSTMDataset(dem_val_sc,   exog_val_sc,   LOOK_BACK)
ds_test  = DemandaLSTMDataset(dem_test_sc,  exog_test_sc,  LOOK_BACK)

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ARQUITECTURA

class LSTMDirecto(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, fc_units, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.bn      = nn.BatchNorm1d(hidden_size)
        self.drop_fc = nn.Dropout(dropout_rate)
        self.fc1     = nn.Linear(hidden_size, fc_units)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(fc_units, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        out    = self.bn(out)
        out    = self.drop_fc(out)
        out    = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


def evaluar(modelo, dataloader, criterio, device):
    modelo.eval()
    total = 0.0
    with torch.no_grad():
        for X_b, y_b in dataloader:
            total += criterio(modelo(X_b.to(device)), y_b.to(device)).item()
    return total / len(dataloader)


# ENTRENAMIENTO O CARGA DE PESOS

fijar_semilla(42)

modelo = LSTMDirecto(
    n_features=N_FEATURES, hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS, fc_units=FC_UNITS, dropout_rate=DROPOUT_RATE,
).to(DEVICE)

criterio  = nn.MSELoss()
optimizer = Adam(modelo.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)

if os.path.exists(MODEL_PATH):
    modelo.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    mejor_val_loss = evaluar(modelo, dl_val, criterio, DEVICE)
    print(f"Modelo cargado. MSE validación: {mejor_val_loss:.6f}")
else:
    mejor_val_loss    = float('inf')
    epochs_sin_mejora = 0

    print(f"{'Época':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}")
    print("-" * 45)

    for epoch in range(1, N_EPOCHS + 1):
        modelo.train()
        train_loss = 0.0
        for X_b, y_b in dl_train:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterio(modelo(X_b), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(dl_train)

        val_loss = evaluar(modelo, dl_val, criterio, DEVICE)
        scheduler.step(val_loss)
        lr_actual = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  {lr_actual:>10.2e}")

        if val_loss < mejor_val_loss:
            mejor_val_loss    = val_loss
            epochs_sin_mejora = 0
            torch.save(modelo.state_dict(), MODEL_PATH)
        else:
            epochs_sin_mejora += 1

        if epochs_sin_mejora >= PATIENCE:
            print(f"Early stopping en época {epoch}")
            break

    modelo.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"\nMejor val_loss: {mejor_val_loss:.6f}")


# EVALUACIÓN

def predecir(modelo, dataloader, device):
    modelo.eval()
    preds = []
    with torch.no_grad():
        for X_b, _ in dataloader:
            preds.append(modelo(X_b.to(device)).cpu().numpy())
    return np.concatenate(preds)


pred_val_sc  = predecir(modelo, dl_val,  DEVICE)
pred_test_sc = predecir(modelo, dl_test, DEVICE)

pred_val_mw  = scaler_demanda.inverse_transform(pred_val_sc.reshape(-1, 1)).flatten()
pred_test_mw = scaler_demanda.inverse_transform(pred_test_sc.reshape(-1, 1)).flatten()

real_val_mw  = dem_val[LOOK_BACK:].flatten()
real_test_mw = dem_test[LOOK_BACK:].flatten()


def calcular_metricas(y_real, y_pred, nombre=''):
    y_real, y_pred = y_real.flatten(), y_pred.flatten()
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mask = y_real > 0
    mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100
    print(f"LSTM directo — {nombre:<15}  MAE={mae:.2f} MW   RMSE={rmse:.2f} MW   MAPE={mape:.3f}%")
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


metricas_val  = calcular_metricas(real_val_mw,  pred_val_mw,  'Validación')
metricas_test = calcular_metricas(real_test_mw, pred_test_mw, 'Test')

errores_val  = real_val_mw  - pred_val_mw
errores_test = real_test_mw - pred_test_mw
print(f"Validación — Error medio: {errores_val.mean():+.2f} MW   Std: {errores_val.std():.2f} MW")
print(f"Test       — Error medio: {errores_test.mean():+.2f} MW   Std: {errores_test.std():.2f} MW")


# EXPORTACIÓN

fechas_val_pred  = val_df_ext['datetime'].values[LOOK_BACK:]
fechas_test_pred = test_df_ext['datetime'].values[LOOK_BACK:]

df_directo_val  = pd.DataFrame({'datetime': fechas_val_pred,  'real_mw': real_val_mw,  'pred_lstm_directo': pred_val_mw,  'conjunto': 'validacion'})
df_directo_test = pd.DataFrame({'datetime': fechas_test_pred, 'real_mw': real_test_mw, 'pred_lstm_directo': pred_test_mw, 'conjunto': 'test'})
df_directo_all  = pd.concat([df_directo_val, df_directo_test], ignore_index=True)

df_directo_val.to_csv('results/lstm_directo/lstm_directo_pred_validacion.csv', index=False)
df_directo_test.to_csv('results/lstm_directo/lstm_directo_pred_test.csv', index=False)
df_directo_all.to_csv('results/lstm_directo/lstm_directo_pred_completo.csv', index=False)

print(f"Guardado: lstm_directo_pred_completo.csv ({len(df_directo_all):,} registros)")