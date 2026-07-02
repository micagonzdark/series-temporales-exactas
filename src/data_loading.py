"""Carga y armado de series a partir del CSV de Spotify Charts.

Columnas esperadas: title, rank, date, artist, url, region, chart, trend, streams.
"""

import pandas as pd


def load_charts(path="data/charts_ar_global.csv"):
    """Carga el CSV chico (ya filtrado a Argentina/Global) y lo ordena por fecha."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_song_series(df, cancion, region="Argentina", chart="top200"):
    """Serie diaria de streams/rank de una canción puntual."""
    serie = df[
        (df["title"] == cancion) & (df["region"] == region) & (df["chart"] == chart)
    ].sort_values("date").reset_index(drop=True)
    return serie


def get_genre_series(df, artistas_genero, region="Argentina", chart="top200", freq="W"):
    """Serie agregada: cantidad de canciones de un género (dado por su lista de
    artistas) presentes en el chart, agrupada por período (semana por defecto).
    """
    subset = df[df["artist"].isin(artistas_genero) & (df["region"] == region) & (df["chart"] == chart)]
    conteo = (
        subset.drop_duplicates(subset=["date", "title"])
        .groupby(pd.Grouper(key="date", freq=freq))["title"]
        .nunique()
    )
    return conteo.rename("cantidad_canciones").reset_index()


def reindex_daily(serie, date_col="date", value_col="streams", fill_value=0.0):
    """Reindexa una serie a frecuencia diaria completa, rellenando los días
    'silencio' (fuera del chart) con `fill_value`.
    """
    fechas_completas = pd.date_range(serie[date_col].min(), serie[date_col].max(), freq="D")
    completa = (
        serie.set_index(date_col)[value_col]
        .reindex(fechas_completas, fill_value=fill_value)
        .rename_axis(date_col)
        .reset_index()
    )
    return completa
