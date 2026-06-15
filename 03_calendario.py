import pandas as pd
import numpy as np
import holidays
import requests
from datetime import date, datetime, timedelta
from dateutil.easter import easter

ANIOS  = [2022, 2023, 2024, 2025]
CCAA   = ['AN', 'AR', 'AS', 'CB', 'CL', 'CM', 'CT', 'EX', 'GA', 'MD', 'MC', 'NC', 'PV', 'RI', 'VC']
UMBRAL = 0.25

MAPA_PROVINCIAS = {
    'Almería': 'AN', 'Cádiz': 'AN', 'Córdoba': 'AN', 'Granada': 'AN', 'Huelva': 'AN',
    'Jaén': 'AN', 'Málaga': 'AN', 'Sevilla': 'AN',
    'Huesca': 'AR', 'Teruel': 'AR', 'Zaragoza': 'AR',
    'Asturias': 'AS',
    'Cantabria': 'CB',
    'Ávila': 'CL', 'Burgos': 'CL', 'León': 'CL', 'Palencia': 'CL', 'Salamanca': 'CL',
    'Segovia': 'CL', 'Soria': 'CL', 'Valladolid': 'CL', 'Zamora': 'CL',
    'Albacete': 'CM', 'Ciudad Real': 'CM', 'Cuenca': 'CM', 'Guadalajara': 'CM', 'Toledo': 'CM',
    'Barcelona': 'CT', 'Girona': 'CT', 'Lleida': 'CT', 'Tarragona': 'CT',
    'Badajoz': 'EX', 'Cáceres': 'EX',
    'A Coruña': 'GA', 'Coruña': 'GA', 'Lugo': 'GA', 'Ourense': 'GA', 'Pontevedra': 'GA',
    'Madrid': 'MD',
    'Murcia': 'MC',
    'Navarra': 'NC',
    'Álava': 'PV', 'Araba': 'PV', 'Gipuzkoa': 'PV', 'Guipúzcoa': 'PV', 'Bizkaia': 'PV', 'Vizcaya': 'PV',
    'La Rioja': 'RI', 'Rioja': 'RI',
    'Alicante': 'VC', 'Castellón': 'VC', 'Valencia': 'VC',
}

DIAS = {0: 'lunes', 1: 'martes', 2: 'miercoles', 3: 'jueves', 4: 'viernes', 5: 'sabado', 6: 'domingo'}


def obtener_pesos_ccaa():
    try:
        datos = requests.get('https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/2852?tip=AM', timeout=25).json()
        pob = {ccaa: 0.0 for ccaa in CCAA}
        total = 0.0
        for serie in datos:
            nombre = serie.get('Nombre', '').lower()
            if not serie.get('Data'):
                continue
            if 'total' not in nombre or 'nacional' in nombre or 'hombres' in nombre or 'mujeres' in nombre:
                continue
            try:
                p = float(serie['Data'][0]['Valor'])
            except (ValueError, KeyError):
                continue
            for prov, ccaa in MAPA_PROVINCIAS.items():
                if prov.lower() in nombre:
                    pob[ccaa] += p
                    total += p
                    break
        if total < 1e6:
            raise ValueError("Datos insuficientes")
        return {ccaa: round(pob[ccaa] / total, 5) for ccaa in CCAA}
    except Exception as e:   # Por si no funciona, aquí están los pesos obtenidos
        print(f"Error al obtener pesos del INE ({e}), usando valores de respaldo")
        return {'AN': 0.19313, 'CT': 0.17697, 'MD': 0.15389, 'VC': 0.11530, 'GA': 0.06145,
                'CL': 0.05432, 'PV': 0.05047, 'CM': 0.04672, 'MC': 0.03461, 'AR': 0.03023,
                'EX': 0.02415, 'AS': 0.02306, 'NC': 0.01508, 'CB': 0.01332, 'RI': 0.00729}


def calcular_festivos(pesos):
    festivos_nac = set()
    pesos_auto   = {}
    for anio in ANIOS:
        h_nac = holidays.Spain(years=anio)
        for d in h_nac:
            festivos_nac.add(d)
        for ccaa in CCAA:
            for d in holidays.Spain(years=anio, subdiv=ccaa):
                if d not in festivos_nac:
                    pesos_auto[d] = round(pesos_auto.get(d, 0.0) + pesos[ccaa], 5)
    return festivos_nac, pesos_auto


def calcular_visperas_puentes(festivos_nac, pesos_auto):
    relevantes = set(festivos_nac) | {d for d, p in pesos_auto.items() if p >= UMBRAL}

    def no_lab(d):
        return d in relevantes or d.weekday() >= 5

    visperas, puentes = set(), set()
    d = date(2022, 1, 1)
    while d <= date(2025, 12, 31):
        sig = d + timedelta(days=1)
        ant = d - timedelta(days=1)
        if d.weekday() < 5 and d not in relevantes:
            if sig in relevantes:
                visperas.add(d)
            if no_lab(ant) and no_lab(sig):
                puentes.add(d)
        d += timedelta(days=1)
    return visperas, puentes


def main():
    pesos = obtener_pesos_ccaa()
    festivos_nac, pesos_auto = calcular_festivos(pesos)
    visperas, puentes = calcular_visperas_puentes(festivos_nac, pesos_auto)

    df = pd.DataFrame({'datetime': pd.date_range('2022-01-01', '2025-12-31 23:00:00', freq='h')})

    df['hora']             = df['datetime'].dt.hour
    df['mes']              = df['datetime'].dt.month
    df['dia_semana_num']   = df['datetime'].dt.weekday
    df['dia_semana_nombre'] = df['dia_semana_num'].map(DIAS)

    for num, nombre in DIAS.items():
        df[f'es_{nombre}'] = (df['dia_semana_num'] == num).astype(int)

    fecha = df['datetime'].dt.date
    df['es_festivo_nacional'] = fecha.apply(lambda d: 1 if d in festivos_nac else 0)
    df['peso_festivo_auto']   = fecha.apply(lambda d: pesos_auto.get(d, 0.0))
    df['es_vispera_festivo']  = fecha.apply(lambda d: 1 if d in visperas else 0)
    df['es_puente']           = fecha.apply(lambda d: 1 if d in puentes else 0)

    # Horario de verano: último domingo de marzo → último domingo de octubre
    df['horario_verano'] = 0
    for anio in ANIOS:
        inicio_v = fin_v = None
        for dia in range(31, 24, -1):
            if datetime(anio, 3, dia).weekday() == 6:
                inicio_v = pd.Timestamp(anio, 3, dia)
                break
        for dia in range(31, 24, -1):
            if datetime(anio, 10, dia).weekday() == 6:
                fin_v = pd.Timestamp(anio, 10, dia)
                break
        df.loc[(df['datetime'] >= inicio_v) & (df['datetime'] < fin_v), 'horario_verano'] = 1

    # Períodos especiales
    df['agosto']  = (df['mes'] == 8).astype(int)
    df['navidad'] = (((df['mes'] == 12) & (df['datetime'].dt.day >= 24)) |
                     ((df['mes'] == 1)  & (df['datetime'].dt.day <= 6))).astype(int)

    df['semana_santa'] = 0
    for anio in ANIOS:
        domingo = pd.Timestamp(easter(anio))
        lunes   = domingo - pd.Timedelta(days=6)
        df.loc[(df['datetime'] >= lunes) & (df['datetime'] < domingo + pd.Timedelta(days=1)), 'semana_santa'] = 1

    columnas = [
        'datetime', 'hora', 'mes', 'dia_semana_num', 'dia_semana_nombre',
        'es_lunes', 'es_martes', 'es_miercoles', 'es_jueves', 'es_viernes', 'es_sabado', 'es_domingo',
        'es_festivo_nacional', 'peso_festivo_auto', 'es_vispera_festivo', 'es_puente',
        'navidad', 'semana_santa', 'agosto', 'horario_verano',
    ]
    df = df[columnas].sort_values('datetime').reset_index(drop=True)
    df.to_csv('data/03_calendario_horario_peninsular.csv', index=False)


if __name__ == '__main__':
    main()