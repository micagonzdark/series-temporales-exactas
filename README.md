# TP Procesamiento de Señales — Series de tiempo en charts de Spotify

## Pregunta de investigación

¿Se puede tratar la evolución diaria de streams/posición de una canción (o de un
género agregado) en el Top de Spotify como una señal real, y aplicarle las
herramientas de la materia (muestreo, Fourier, filtros, energía, correlación,
forecasting) para caracterizar su comportamiento — estacionalidad semanal,
tendencia, saltos abruptos (virilidad) y comparación entre series distintas?

## Equipo — quién trabaja qué serie

| Persona   | Notebook                      | Serie asignada (canción o género) | Región | Chart |
|-----------|--------------------------------|------------------------------------|--------|-------|
| Persona A | `notebooks/persona_a.ipynb`   | _completar_                         | _completar_ | _completar_ |
| Persona B | `notebooks/persona_b.ipynb`   | _completar_                         | _completar_ | _completar_ |
| Persona C | `notebooks/persona_c.ipynb`   | _completar_                         | _completar_ | _completar_ |
| Persona D | `notebooks/persona_d.ipynb`   | _completar_                         | _completar_ | _completar_ |

Cada persona trabaja en su propia rama (`persona-a`, `persona-b`, `persona-c`,
`persona-d`), partiendo del notebook base `notebooks/00_base_template.ipynb`,
y cambia los parámetros (`CANCION`/`GENERO`, `REGION`, `CHART`) por la serie
que le toca. La sección 5 (comparación/correlación entre series del equipo) se
arma después, juntando las 4 series ya procesadas.

## Estructura del repo

```
.
├── data/               # CSVs (no versionados, ver .gitignore)
├── notebooks/
│   ├── 00_base_template.ipynb   # notebook base compartido
│   ├── persona_a.ipynb
│   ├── persona_b.ipynb
│   ├── persona_c.ipynb
│   └── persona_d.ipynb
├── src/                # funciones compartidas (carga de datos, FFT, filtros MA, energía, correlación)
├── resultados/         # gráficos finales e informe
├── requirements.txt
└── README.md
```

## Dataset

Usamos [Spotify Charts (dhruvildave)](https://www.kaggle.com/datasets/dhruvildave/spotify-charts)
de Kaggle: contiene, por día, el Top 200 / Viral 50 de Spotify en decenas de
países (incluye `Argentina` y `Global`), con columnas `title, rank, date,
artist, url, region, chart, trend, streams`.

### Cómo conseguir el CSV (una sola vez para todo el grupo)

1. Crear una cuenta gratuita en Kaggle y entrar a la página del dataset.
2. Descargar `charts.csv` (el archivo completo es grande: ~26 millones de
   filas, puede superar 1 GB descomprimido). Alternativa: usar la API de
   Kaggle (`kaggle datasets download -d dhruvildave/spotify-charts`).
3. Una sola persona lo descarga, corre la celda de "Filtrado inicial" de
   `notebooks/00_base_template.ipynb` (se queda solo con `Argentina` +
   `Global`), y comparte el CSV chico resultante con el resto del equipo
   (Drive u otro medio — **no subirlo a git**).
4. Cada integrante coloca ese archivo en `data/charts_ar_global.csv`.

Los CSV **no se versionan** (ver `.gitignore`): cada uno los descarga/copia
localmente en su carpeta `data/`.

## Cómo correr los notebooks

1. Clonar el repo y pararse en la rama que corresponda (`persona-a`,
   `persona-b`, `persona-c` o `persona-d`).
2. Crear un entorno virtual e instalar dependencias:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
3. Poner `charts_ar_global.csv` en `data/` (ver sección Dataset).
4. Abrir el notebook propio (`notebooks/persona_x.ipynb`) con Jupyter/VS
   Code y correr las celdas en orden. Las funciones compartidas de `src/`
   se importan directamente desde los notebooks (carga de datos, FFT,
   filtros de media móvil, energía y correlación entre series).
