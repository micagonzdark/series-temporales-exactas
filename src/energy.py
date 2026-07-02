"""Energía y potencia de una señal (streams diarios)."""

import numpy as np


def energia(valores):
    """Energía total de la señal: suma de los cuadrados de las muestras."""
    valores = np.asarray(valores, dtype=float)
    return np.sum(valores ** 2)


def potencia_media(valores):
    """Potencia media: energía dividida por la cantidad de muestras."""
    valores = np.asarray(valores, dtype=float)
    return energia(valores) / len(valores) if len(valores) else 0.0


def energia_por_ventana(valores, ventana):
    """Energía calculada en ventanas no solapadas de tamaño `ventana` (ej. por
    semana), útil para ver cómo evoluciona la energía de la señal en el tiempo.
    """
    valores = np.asarray(valores, dtype=float)
    n_ventanas = len(valores) // ventana
    recortado = valores[: n_ventanas * ventana].reshape(n_ventanas, ventana)
    return np.sum(recortado ** 2, axis=1)
