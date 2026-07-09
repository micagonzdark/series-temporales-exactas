# TP Series Temporales — Análisis de charts de Spotify

Trabajo práctico de series temporales: tratamos la evolución diaria de
streams/posición de canciones en el Top de Spotify (Argentina / Global) como una
señal real y le aplicamos las herramientas de la materia (muestreo, Fourier,
filtros, energía, correlación, estacionariedad y forecasting).

## Pregunta central

**¿Se puede caracterizar y predecir el comportamiento de una canción en los
charts de Spotify tratándola como una serie temporal / señal?**

### Sub-preguntas

1. **Estructura de la señal** — ¿Qué componentes tiene la serie (tendencia,
   estacionalidad semanal, saltos abruptos) y cómo se ven en el dominio del
   tiempo y de la frecuencia (FFT, filtros MA, energía)?
2. **Comparación entre series** — ¿El chart argentino sigue al global? ¿Con qué
   desfasaje y qué correlación (autocorrelación, correlación cruzada)?
3. **Predicción** — ¿Es la serie estacionaria y qué tan bien se puede pronosticar
   su evolución (ARIMA / Prophet, validado con backtesting)?

## Estructura de carpetas

```text
.
├── data/               # CSVs (no versionados, ver .gitignore)
├── resultados/         # gráficos finales e informe
├── src/
│   ├── __init__.py
│   └── utils.py        # contrato (stubs) de todas las funciones compartidas
├── notebooks/
│   ├── 00_definiciones.ipynb     # definiciones, alcance y convenciones del TP
│   ├── 01_carga_y_datos.ipynb    # carga del dataset y armado de series
│   ├── 02_resumen_general.ipynb  # exploratorio y estadística descriptiva
│   ├── 03_ciclo_de_vida.ipynb    # ciclo de vida de una canción (forma de onda)
│   ├── 04_comparacion_global.ipynb  # Argentina vs. Global (correlación)
│   ├── 05_forecasting.ipynb      # estacionariedad + ARIMA/Prophet + backtesting
│   └── 06_conclusiones.ipynb     # cierre e integración de resultados
├── requirements.txt
├── .gitignore
└── README.md
```

Todas las funciones auxiliares viven en un único módulo `src/utils.py` (por ahora
como stubs con firma + docstring). Los notebooks las importan con
`from src import utils`.

## Dataset

Usamos [Spotify Charts (dhruvildave)](https://www.kaggle.com/datasets/dhruvildave/spotify-charts)
de Kaggle: por día, el Top 200 / Viral 50 de Spotify en decenas de países
(incluye `Argentina` y `Global`), con columnas `title, rank, date, artist, url,
region, chart, trend, streams`. El CSV completo es grande (~26M filas); una sola
persona lo descarga, corre `cargar_dataset_completo` para quedarse con Argentina +
Global y comparte el CSV chico. Los CSV **no se versionan** (ver `.gitignore`).

## Cómo correr los notebooks

1. Clonar el repo.
2. Crear un entorno virtual e instalar dependencias:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

3. Correr `nbstripout --install` una sola vez después de clonar, para que git
   limpie automáticamente los outputs de los notebooks al commitear (excepto
   `notebooks/00_definiciones.ipynb`, que conserva los suyos como referencia).
4. Poner el CSV en `data/` (ver sección Dataset).
5. Abrir los notebooks de `notebooks/` con Jupyter/VS Code y correrlos en orden,
   empezando por `00_definiciones`. Cada notebook agrega `..` al `sys.path` e
   importa las funciones compartidas con `from src import utils`.
