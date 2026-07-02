"""Filtros de media móvil (pasa-bajos) para separar tendencia/estacionalidad."""

import numpy as np
import pandas as pd


def media_movil(serie, ventana):
    """Media móvil simple (MA) de `ventana` días, centrada."""
    if isinstance(serie, pd.Series):
        return serie.rolling(window=ventana, center=True, min_periods=1).mean()
    return pd.Series(serie).rolling(window=ventana, center=True, min_periods=1).mean().to_numpy()


def componente_estacional(serie, ventana_corta=7, ventana_larga=21):
    """Aproxima la componente semanal como la diferencia entre una MA corta
    (ruido+estacional) y una MA larga (tendencia).
    """
    tendencia = media_movil(serie, ventana_larga)
    suavizada = media_movil(serie, ventana_corta)
    return suavizada - tendencia, tendencia


def pasa_banda_semanal(serie, fs=1.0, periodo=7, ancho=0.15):
    """Filtro pasa-banda en frecuencia alrededor de 1/periodo (ej. componente
    semanal), implementado recortando el espectro FFT y volviendo al tiempo.
    """
    valores = np.asarray(serie, dtype=float)
    n = len(valores)
    espectro = np.fft.rfft(valores - np.mean(valores))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    freq_objetivo = 1.0 / periodo
    banda = (freqs >= freq_objetivo * (1 - ancho)) & (freqs <= freq_objetivo * (1 + ancho))

    espectro_filtrado = np.where(banda, espectro, 0)
    return np.fft.irfft(espectro_filtrado, n=n)
