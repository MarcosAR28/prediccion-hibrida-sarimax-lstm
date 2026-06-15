import requests
import pandas as pd
import numpy as np
import time

CIUDADES = {
    'Madrid':    {'lat': 40.4165, 'lon': -3.7026},
    'Barcelona': {'lat': 41.3888, 'lon':  2.1590},
    'Valencia':  {'lat': 39.4697, 'lon': -0.3774},
    'Sevilla':   {'lat': 37.3828, 'lon': -5.9732},
    'Bilbao':    {'lat': 43.2627, 'lon': -2.9253},
    'Zaragoza':  {'lat': 41.6561, 'lon': -0.8773},
}

MAPA_ZONAS = {
    'Madrid':    ['Madrid', 'Toledo', 'Ciudad Real', 'Cuenca', 'Guadalajara',
                  'Ávila', 'Segovia', 'Salamanca', 'Valladolid', 'Zamora', 'Cáceres'],
    'Barcelona': ['Barcelona', 'Tarragona', 'Lleida', 'Girona'],
    'Valencia':  ['Alicante', 'Castellón', 'Valencia', 'Murcia', 'Almería', 'Albacete'],
    'Sevilla':   ['Sevilla', 'Córdoba', 'Jaén', 'Huelva', 'Cádiz', 'Málaga', 'Badajoz', 'Granada'],
    'Bilbao':    ['Coruña', 'Lugo', 'Ourense', 'Pontevedra', 'Álava', 'Vizcaya',
                  'Guipúzcoa', 'Asturias', 'Cantabria', 'León', 'Palencia'],
    'Zaragoza':  ['Zaragoza', 'Huesca', 'Teruel', 'Navarra', 'Rioja', 'Soria', 'Burgos'],
}


def obtener_pesos():
    url = 'https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/2852?tip=AM'
    datos = requests.get(url, timeout=30).json()

    pob = {z: 0 for z in CIUDADES}
    total = 0

    for serie in datos:
        nombre = serie.get('Nombre', '')
        if not serie.get('Data'):
            continue
        if 'Total' not in nombre or 'Nacional' in nombre or 'Hombres' in nombre or 'Mujeres' in nombre:
            continue
        pob_val = serie['Data'][0]['Valor']
        for zona, provincias in MAPA_ZONAS.items():
            if any(p.lower() in nombre.lower() for p in provincias):
                pob[zona] += pob_val
                total += pob_val
                break

    return {z: round(pob[z] / total, 5) for z in CIUDADES}    # dividimos por el total para que el vector sume 1


def descargar_meteo():
    pesos = obtener_pesos()
    df_final = None

    for ciudad, coords in CIUDADES.items():
        params = {
            'latitude':   coords['lat'],
            'longitude':  coords['lon'],
            'start_date': '2022-01-01',
            'end_date':   '2025-12-31',
            'hourly':     'temperature_2m,shortwave_radiation,relative_humidity_2m,wind_speed_10m',
            'timezone':   'Europe/Madrid',
        }
        r = requests.get('https://archive-api.open-meteo.com/v1/archive', params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        if df_final is None:
            df_final = pd.DataFrame({'datetime': pd.to_datetime(data['hourly']['time'])})

        df_final[f'temp_{ciudad}']     = data['hourly']['temperature_2m']
        df_final[f'rad_{ciudad}']      = data['hourly']['shortwave_radiation']
        df_final[f'hum_{ciudad}']      = data['hourly']['relative_humidity_2m']
        df_final[f'viento_{ciudad}']   = data['hourly']['wind_speed_10m']

        print(f"{ciudad}: descargado")
        time.sleep(0.8)

    # Ponderación por población
    df_final['temperatura']     = sum(df_final[f'temp_{c}']   * w for c, w in pesos.items())
    df_final['radiacion_solar'] = sum(df_final[f'rad_{c}']    * w for c, w in pesos.items())
    df_final['humedad']         = sum(df_final[f'hum_{c}']    * w for c, w in pesos.items())
    df_final['viento']          = sum(df_final[f'viento_{c}'] * w for c, w in pesos.items())

    df_final['radiacion_solar'] = df_final['radiacion_solar'].clip(lower=0)  # Por si da valores negativos

    # Creamos hdd y cdd
    df_final['hdd'] = np.maximum(0, 18.0 - df_final['temperatura'])   
    df_final['cdd'] = np.maximum(0, df_final['temperatura'] - 18.0)

    for col in ['temperatura', 'radiacion_solar', 'humedad', 'viento', 'hdd', 'cdd']:
        df_final[col] = df_final[col].round(2)

    return df_final[['datetime', 'temperatura', 'humedad', 'viento', 'radiacion_solar', 'hdd', 'cdd']]


def main():
    df = descargar_meteo()
    df = df.sort_values('datetime').reset_index(drop=True)
    df.to_csv('data/02_meteorologia_horaria_peninsular.csv', index=False)


if __name__ == '__main__':
    main()