"""Funciones auxiliares compartidas del TP de Series Temporales (charts de Spotify).

Implementación de todo el contrato usado por los notebooks. Agrupadas por bloque:

    1. Carga y armado de series
    2. Descripción / estadística
    3. Fourier y filtrado
    4. Energía
    5. Correlación
    6. Estacionariedad y forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Carga y armado de series
# ---------------------------------------------------------------------------

def cargar_dataset_completo(path_csv_grande):
    """Carga el CSV completo de Spotify Charts y lo reduce a las regiones de interés.

    Parameters
    ----------
    path_csv_grande : str
        Ruta al CSV original de Kaggle (~26M filas, todas las regiones).

    Returns
    -------
    pandas.DataFrame
        DataFrame filtrado a `region in ["Argentina", "Global"]`, con las mismas
        columnas que el original y `date` parseada como datetime.
    """
    regiones = ["Argentina", "Global"]
    # El archivo es grande: leemos en chunks y nos quedamos solo con las regiones.
    partes = []
    for chunk in pd.read_csv(path_csv_grande, parse_dates=["date"], chunksize=500_000):
        partes.append(chunk[chunk["region"].isin(regiones)])
    df = pd.concat(partes, ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


def construir_serie(df, filtro):
    """Arma una serie temporal indexada por fecha a partir del DataFrame de charts.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de charts.
    filtro : dict
        Claves reconocidas: "title", "artist", "region", "chart",
        "valor" ("streams" o "rank", default "streams") y "agregado"
        (cómo colapsar varias filas por fecha: "sum" default, "mean", "count").

    Returns
    -------
    pandas.Series
        Serie indexada por fecha (DatetimeIndex) con los valores pedidos.
    """
    valor = filtro.get("valor", "streams")
    agregado = filtro.get("agregado", "sum")

    mask = pd.Series(True, index=df.index)
    for col in ("title", "artist", "region", "chart"):
        if col in filtro and filtro[col] is not None:
            mask &= df[col] == filtro[col]

    sub = df.loc[mask, ["date", valor]]
    serie = sub.groupby("date")[valor].agg(agregado).sort_index()
    serie.name = valor
    return serie


def interpolar_serie(serie):
    """Completa los huecos de fechas faltantes con interpolación lineal.

    Reindexa a un rango diario continuo (primera a última fecha) e interpola
    linealmente los días faltantes.

    Parameters
    ----------
    serie : pandas.Series
        Serie indexada por fecha, posiblemente con días faltantes.

    Returns
    -------
    pandas.Series
        Serie con índice de fechas continuo (frecuencia diaria) y sin NaN.
    """
    serie = serie.sort_index()
    idx_completo = pd.date_range(serie.index.min(), serie.index.max(), freq="D")
    completa = serie.reindex(idx_completo).interpolate(method="linear")
    # Bordes: si quedaron NaN al inicio, los completamos hacia atrás/adelante.
    completa = completa.bfill().ffill()
    completa.name = serie.name
    return completa


# ---------------------------------------------------------------------------
# 2. Descripción / estadística
# ---------------------------------------------------------------------------

def resumen_estadistico(serie):
    """Estadísticos descriptivos básicos de la serie.

    Returns
    -------
    dict
        Claves "media", "desvio" (desvío estándar) y "rango" (máx - mín).
    """
    valores = np.asarray(serie, dtype=float)
    return {
        "media": float(np.mean(valores)),
        "desvio": float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0,
        "rango": float(np.max(valores) - np.min(valores)),
    }


def graficar_serie(serie, titulo):
    """Grafica la serie temporal (valor vs. fecha).

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(serie.index, serie.values)
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(serie.name if serie.name else "Valor")
    fig.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# 3. Fourier y filtrado
# ---------------------------------------------------------------------------

def calcular_fft(serie):
    """Transformada de Fourier (FFT) de la serie.

    Usa np.fft.rfft sobre la serie sin su media (se descarta la componente DC).
    El muestreo se asume de 1 muestra/día, por lo que las frecuencias quedan en
    ciclos/día.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        (frecuencias, magnitudes) de una cara (frecuencias positivas, ciclos/día),
        descartando la componente DC (f = 0).
    """
    x = np.asarray(serie, dtype=float)
    n = len(x)
    espectro = np.fft.rfft(x - np.mean(x))
    frecuencias = np.fft.rfftfreq(n, d=1.0)  # d=1 día -> ciclos/día
    magnitudes = np.abs(espectro)
    # Descartamos la componente DC (índice 0).
    return frecuencias[1:], magnitudes[1:]


def graficar_espectro(frecuencias, magnitudes, titulo):
    """Grafica el espectro de magnitud (magnitud vs. frecuencia).

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frecuencias, magnitudes)
    ax.set_title(titulo)
    ax.set_xlabel("Frecuencia (ciclos/día)")
    ax.set_ylabel("|X(f)|")
    fig.tight_layout()
    return ax


def filtro_media_movil(serie, ventana):
    """Aplica un filtro de media móvil (MA) centrada de la ventana dada.

    Returns
    -------
    pandas.Series
        Serie suavizada, con el mismo índice que la entrada.
    """
    if not isinstance(serie, pd.Series):
        serie = pd.Series(serie)
    return serie.rolling(window=ventana, center=True, min_periods=1).mean()


def descomponer_MA7_MA21(serie):
    """Descompone la serie en tendencia, componente estacional y residuo.

    Algoritmo del apunte con medias móviles:
    - MA21 sobre la serie -> tendencia (baja frecuencia).
    - MA7 sobre la serie -> quita el ruido de alta frecuencia (tendencia+estacional).
    - estacional = MA7 - MA21 (lo que queda entre ambas bandas).
    - residuo = serie - MA7 (alta frecuencia que MA7 filtró de la original).

    Returns
    -------
    dict
        Claves "tendencia" (MA21), "estacional" (MA7 - MA21) y "residuo" (serie - MA7).
    """
    if not isinstance(serie, pd.Series):
        serie = pd.Series(serie)
    ma7 = filtro_media_movil(serie, 7)
    ma21 = filtro_media_movil(serie, 21)
    return {
        "tendencia": ma21,
        "estacional": ma7 - ma21,
        "residuo": serie - ma7,
    }


def filtro_pasabanda_frecuencia(serie, f_low, f_high):
    """Filtro pasa-banda en el dominio de frecuencia (FFT -> máscara -> IFFT).

    Lleva la serie (sin su media) al dominio de frecuencia, pone en cero todas las
    componentes cuya frecuencia (en valor absoluto) cae fuera de [f_low, f_high] y
    vuelve al tiempo con la IFFT real. Se re-suma la media al final.

    Parameters
    ----------
    serie : pandas.Series
        Serie a filtrar (muestreo diario).
    f_low, f_high : float
        Frecuencias de corte inferior/superior (ciclos/día).

    Returns
    -------
    pandas.Series
        Serie filtrada, con el mismo índice de entrada.
    """
    x = np.asarray(serie, dtype=float)
    n = len(x)
    media = np.mean(x)
    espectro = np.fft.fft(x - media)
    frecuencias = np.fft.fftfreq(n, d=1.0)

    mascara = (np.abs(frecuencias) >= f_low) & (np.abs(frecuencias) <= f_high)
    espectro_filtrado = np.where(mascara, espectro, 0.0)
    filtrada = np.real(np.fft.ifft(espectro_filtrado)) + media

    if isinstance(serie, pd.Series):
        return pd.Series(filtrada, index=serie.index, name=serie.name)
    return filtrada


# ---------------------------------------------------------------------------
# 4. Energía
# ---------------------------------------------------------------------------

def energia_serie(serie):
    """Energía de la señal en tiempo y en frecuencia (chequeo de Parseval).

    Energía en tiempo: sum(x(n)^2).
    Energía en frecuencia: sum(|X(k)|^2) / N, con X = FFT completa.
    (Teorema de Parseval: ambas deberían coincidir.)

    Returns
    -------
    dict
        Claves "energia_tiempo", "energia_frecuencia", "diferencia_relativa"
        (|Et - Ef| / Et) y "parseval_ok" (bool, diferencia < 1e-8).
    """
    x = np.asarray(serie, dtype=float)
    energia_tiempo = float(np.sum(x ** 2))
    X = np.fft.fft(x)
    energia_frecuencia = float(np.sum(np.abs(X) ** 2) / len(x))
    denom = energia_tiempo if energia_tiempo != 0 else 1.0
    dif_rel = abs(energia_tiempo - energia_frecuencia) / denom
    return {
        "energia_tiempo": energia_tiempo,
        "energia_frecuencia": energia_frecuencia,
        "diferencia_relativa": dif_rel,
        "parseval_ok": dif_rel < 1e-8,
    }


# ---------------------------------------------------------------------------
# 5. Correlación
# ---------------------------------------------------------------------------

def autocorrelacion(serie, nlags):
    """Función de autocorrelación (ACF) de la serie.

    Returns
    -------
    numpy.ndarray
        Vector de ACF de longitud nlags+1 (lag 0 = 1.0).
    """
    from statsmodels.tsa.stattools import acf

    x = np.asarray(serie, dtype=float)
    return acf(x, nlags=nlags, fft=True)


def correlacion_cruzada(serie1, serie2, max_lag):
    """Correlación cruzada normalizada entre dos series para lags en [-max_lag, +max_lag].

    Las series se alinean por índice común antes de correlacionar. Lag positivo:
    `serie1` adelanta a `serie2`.

    Returns
    -------
    pandas.Series
        Correlación para cada lag, indexada por el lag.
    """
    if isinstance(serie1, pd.Series) and isinstance(serie2, pd.Series):
        alineadas = pd.concat([serie1, serie2], axis=1).dropna()
        x = alineadas.iloc[:, 0].to_numpy(dtype=float)
        y = alineadas.iloc[:, 1].to_numpy(dtype=float)
    else:
        x = np.asarray(serie1, dtype=float)
        y = np.asarray(serie2, dtype=float)

    x = x - np.mean(x)
    y = y - np.mean(y)
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
    return pd.Series(valores, index=list(lags), name="correlacion_cruzada")


# ---------------------------------------------------------------------------
# 6. Estacionariedad y forecasting
# ---------------------------------------------------------------------------

def test_estacionariedad(serie):
    """Tests de estacionariedad: Dickey-Fuller aumentado (ADF) y KPSS.

    En ADF, H0 = raíz unitaria (no estacionaria): p < 0.05 => estacionaria.
    En KPSS, H0 = estacionaria: p < 0.05 => NO estacionaria.

    Returns
    -------
    dict
        {"adf": {...}, "kpss": {...}} con estadístico, p_valor y estacionaria (bool).
    """
    import warnings
    from statsmodels.tsa.stattools import adfuller, kpss

    x = np.asarray(serie, dtype=float)

    adf_stat, adf_p, *_ = adfuller(x)
    with warnings.catch_warnings():
        # kpss emite InterpolationWarning cuando el p-valor cae fuera de la tabla.
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")

    return {
        "adf": {"estadistico": float(adf_stat), "p_valor": float(adf_p), "estacionaria": adf_p < 0.05},
        "kpss": {"estadistico": float(kpss_stat), "p_valor": float(kpss_p), "estacionaria": kpss_p >= 0.05},
    }


def ajustar_arima(serie, order):
    """Ajusta un modelo ARIMA a la serie.

    Parameters
    ----------
    serie : pandas.Series
    order : tuple(int, int, int)
        Orden (p, d, q).

    Returns
    -------
    dict
        {"modelo": result ajustado, "aic": float, "bic": float}.
    """
    from statsmodels.tsa.arima.model import ARIMA

    modelo = ARIMA(np.asarray(serie, dtype=float), order=order).fit()
    return {"modelo": modelo, "aic": float(modelo.aic), "bic": float(modelo.bic)}


def ajustar_prophet(serie):
    """Ajusta un modelo Prophet a la serie.

    Parameters
    ----------
    serie : pandas.Series
        Serie indexada por fecha (se convierte a formato ds/y internamente).

    Returns
    -------
    prophet.Prophet
        Modelo ya ajustado (fit).
    """
    from prophet import Prophet

    df = pd.DataFrame({"ds": pd.to_datetime(serie.index), "y": np.asarray(serie, dtype=float)})
    modelo = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    modelo.fit(df)
    return modelo


def backtesting_forecast(serie, modelo_fn, ventana_train, horizonte):
    """Backtesting con ventana móvil (walk-forward) de un modelo de pronóstico.

    Desliza una ventana de entrenamiento de `ventana_train` días; en cada paso
    llama `modelo_fn(train, horizonte)` (que debe devolver un array de `horizonte`
    predicciones) y lo compara contra los valores reales siguientes.

    Parameters
    ----------
    serie : pandas.Series
    modelo_fn : callable
        `modelo_fn(train_series, horizonte) -> array-like` de longitud `horizonte`.
    ventana_train : int
        Tamaño (días) de la ventana de entrenamiento.
    horizonte : int
        Pasos hacia adelante a pronosticar en cada iteración.

    Returns
    -------
    dict
        {"errores": array de errores punto a punto, "mae", "rmse", "mape"}.
    """
    valores = np.asarray(serie, dtype=float)
    n = len(valores)
    errores = []

    inicio = 0
    while inicio + ventana_train + horizonte <= n:
        train = serie.iloc[inicio: inicio + ventana_train]
        real = valores[inicio + ventana_train: inicio + ventana_train + horizonte]
        pred = np.asarray(modelo_fn(train, horizonte), dtype=float)[:horizonte]
        errores.extend((real - pred).tolist())
        inicio += horizonte

    errores = np.asarray(errores, dtype=float)
    if len(errores) == 0:
        return {"errores": errores, "mae": np.nan, "rmse": np.nan, "mape": np.nan}

    reales = np.asarray(serie, dtype=float)[ventana_train: ventana_train + len(errores)]
    no_cero = reales != 0
    mape = float(np.mean(np.abs(errores[no_cero] / reales[no_cero])) * 100) if no_cero.any() else np.nan
    return {
        "errores": errores,
        "mae": float(np.mean(np.abs(errores))),
        "rmse": float(np.sqrt(np.mean(errores ** 2))),
        "mape": mape,
    }
