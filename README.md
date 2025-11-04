# App de Predicción - Liga Argentina ⚽

Aplicación Streamlit para predecir resultados de partidos de la Liga Argentina basada en machine learning.

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecutar la app:
```bash
streamlit run streamlit_app.py
```

**Nota:** El modelo ya está entrenado (`model.pkl`). Se entrena en Google Colab y se incluye en el repositorio.

## Uso

La aplicación sigue un flujo secuencial para asegurar que se usen datos históricos válidos:

1. **Selecciona la Temporada**: Solo están disponibles temporadas de 2024 o posteriores (el modelo se entrenó con datos de 2015-2023)
2. **Selecciona la Subtemporada**: Basada en la temporada elegida
3. **Selecciona la Fecha del Partido**: Fecha mínima 2024-01-01, sin límite máximo (permite fechas futuras)
4. **Selecciona los Equipos**: Solo se muestran equipos que participaron en la temporada/subtemporada seleccionada
5. **Haz clic en "Predecir Resultado"**: Visualiza la predicción y probabilidades

### Lógica de Normalización de Datos

El modelo fue entrenado con valores normalizados por **subseason** (subtemporada). Por lo tanto:

- **Primero**: Se busca el último partido de cada equipo en la subseason seleccionada antes de la fecha elegida
- **Si no encuentra**: Se busca el último partido disponible del equipo (sin importar subseason, pero >= 2024 y antes de la fecha)
- **Valores utilizados**: Se usan siempre los valores normalizados del último partido encontrado

Esto asegura que las predicciones usen datos históricos realistas y coherentes con el entrenamiento del modelo.

## Estructura del Proyecto

```
app-prediccion-liga-argentina/
├── streamlit_app.py                    # Aplicación Streamlit
├── model.pkl                           # Modelo entrenado (de Colab)
├── equipos.pkl                         # Lista de equipos disponibles
├── liga_argentina_features_v3.csv     # Dataset completo con features normalizadas
├── .streamlit/
│   └── config.toml                    # Configuración de Streamlit
├── requirements.txt                   # Dependencias
└── README.md
```

## Modelo

- **Algoritmo**: Logistic Regression con regularización ElasticNet
- **Entrenamiento**: Partidos de 2015-2023 (los datos se normalizan por subseason)
- **Evaluación**: Partidos de 2024-2025 (usados para testear)
- **Métrica**: F1-Score ponderado
- **Normalización**: Los valores se normalizan por subseason para mantener coherencia temporal

### Restricciones de Predicción

- **No se pueden predecir partidos anteriores a 2024**: Estos datos se usaron para entrenar el modelo
- **Solo equipos con datos históricos**: Se validan que ambos equipos tengan partidos previos disponibles
- **Valores normalizados**: Se usan siempre los valores normalizados del último partido disponible para mantener consistencia con el entrenamiento

