"""Análisis en frecuencia (FFT) de series diarias."""

import numpy as np


def compute_fft(valores, fs=1.0):
    """FFT de una señal real. `fs` es la frecuencia de muestreo (1/día por defecto).

    Devuelve (frecuencias positivas, magnitud) descartando la componente DC.
    """
    n = len(valores)
    espectro = np.fft.rfft(valores - np.mean(valores))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitud = np.abs(espectro) / n
    return freqs[1:], magnitud[1:]


def periodo_dominante(valores, fs=1.0):
    """Período (en días, si fs=1/día) del pico principal del espectro."""
    freqs, magnitud = compute_fft(valores, fs=fs)
    freq_pico = freqs[np.argmax(magnitud)]
    return 1.0 / freq_pico if freq_pico > 0 else np.inf


def potencia_en_periodo(valores, periodo_dias, fs=1.0, tolerancia=0.1):
    """Suma de |X(f)|^2 en una banda alrededor de 1/periodo_dias (ej. 7 días)."""
    freqs, magnitud = compute_fft(valores, fs=fs)
    freq_objetivo = 1.0 / periodo_dias
    banda = (freqs >= freq_objetivo * (1 - tolerancia)) & (freqs <= freq_objetivo * (1 + tolerancia))
    return np.sum(magnitud[banda] ** 2)
