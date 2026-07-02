"""Comparación / correlación entre las series de los 4 integrantes."""

import numpy as np
import pandas as pd


def alinear_series(series_dict, date_col="date", value_col="streams", fill_value=0.0):
    """Recibe un dict {nombre: dataframe con columnas date/value} y devuelve un
    único DataFrame alineado por fecha (unión de todas las fechas), con NaN/0
    en los días donde esa serie no tiene dato.
    """
    df = None
    for nombre, serie in series_dict.items():
        s = serie.set_index(date_col)[value_col].rename(nombre)
        df = s.to_frame() if df is None else df.join(s, how="outer")
    return df.fillna(fill_value).sort_index()


def correlacion_cruzada(x, y, max_lag=14):
    """Correlación cruzada normalizada entre dos series para lags de
    -max_lag a +max_lag días. Lag positivo: `x` adelanta a `y`.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    norm = np.std(x) * np.std(y) * len(x)

    lags = range(-max_lag, max_lag + 1)
    valores = []
    for lag in lags:
        if lag < 0:
            corr = np.sum(x[-lag:] * y[: len(y) + lag])
        elif lag > 0:
            corr = np.sum(x[: len(x) - lag] * y[lag:])
        else:
            corr = np.sum(x * y)
        valores.append(corr / norm if norm else 0.0)
    return pd.Series(valores, index=list(lags), name="correlacion")


def matriz_correlacion(df_alineado):
    """Matriz de correlación de Pearson entre todas las series ya alineadas."""
    return df_alineado.corr()
