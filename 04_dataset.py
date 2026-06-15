import pandas as pd
import numpy as np

DEMANDA     = '01_demanda.csv'
METEO       = '02_meteo.csv'
CALENDARIO  = '03_calendario.csv'
OUTPUT_FILE = 'dataset_final_tfm.csv'

def main():
    df_dem = pd.read_csv(DEMANDA,    parse_dates=['datetime'])
    df_met = pd.read_csv(METEO,      parse_dates=['datetime'])
    df_cal = pd.read_csv(CALENDARIO, parse_dates=['datetime'])

    df = df_dem.merge(df_met, on='datetime', how='left')
    df = df.merge(df_cal, on='datetime', how='left')

    df.to_csv(OUTPUT_FILE, index=False)


if __name__ == '__main__':
    main()