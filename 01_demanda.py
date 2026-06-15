import pandas as pd
import numpy as np
from datetime import datetime


def corregir_cambios_hora(df):
    df = df.copy()
    df['year']  = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day']   = df['datetime'].dt.day
    df['hour']  = df['datetime'].dt.hour

    id_borrar = []

    for year in df['year'].unique():
        # Octubre: hora duplicada
        for dia in range(31, 24, -1):    # cogemos la última semana
            d = datetime(year, 10, dia)  
            if d.weekday() == 6:        # cogemos el domingo de esa última semana
                repes = (df['year'] == year) & (df['month'] == 10) & (df['day'] == dia) & (df['hour'] == 2)
                duplicados = df[repes]
                if len(duplicados) >= 2:
                    df.loc[duplicados.index[0], 'value'] = duplicados['value'].mean()
                    id_borrar.append(duplicados.index[1])   
                break

        # Marzo: falta una hora -> interpolar
        for dia in range(31, 24, -1):
            d = datetime(year, 3, dia)
            if d.weekday() == 6:
                h1 = (df['year'] == year) & (df['month'] == 3) & (df['day'] == dia) & (df['hour'] == 1)
                h3 = (df['year'] == year) & (df['month'] == 3) & (df['day'] == dia) & (df['hour'] == 3)
                h2 = (df['year'] == year) & (df['month'] == 3) & (df['day'] == dia) & (df['hour'] == 2)

                #if df[h1].empty or df[h3].empty or not df[h2].empty:
                #    break

                val_interp = (df.loc[df[h1].index[0], 'value'] + df.loc[df[h3].index[0], 'value']) / 2
                fila = df.loc[df[h1].index[0]].copy()
                fila['datetime'] = pd.Timestamp(year, 3, dia, 2, 0)
                fila['value'] = val_interp
                fila['hour'] = 2
                df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
                break

    if id_borrar:
        df = df.drop(id_borrar)

    df = df.sort_values('datetime').reset_index(drop=True)
    return df[['datetime', 'value']]


def main():
    df = pd.read_csv('demanda_horaria_peninsular.csv', parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    df = corregir_cambios_hora(df)

    # Asegurar continuidad horaria completa
    # rango = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='h')
    # df = pd.DataFrame({'datetime': rango}).merge(df, on='datetime', how='left')

    df.to_csv('data/01_demanda_horaria_peninsular.csv', index=False)

if __name__ == '__main__':
    main()