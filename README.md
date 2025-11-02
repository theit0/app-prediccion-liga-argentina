# App de Predicción - Liga Argentina ⚽

Aplicación Streamlit para predecir resultados de partidos de la Liga Argentina basada en machine learning.

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Entrenar el modelo:
```bash
python -m app.train_model
```

3. Ejecutar la app:
```bash
streamlit run streamlit_app.py
```

## Uso

1. Selecciona el equipo local
2. Selecciona el equipo visitante  
3. Haz clic en "Predecir Resultado"
4. Visualiza la predicción y probabilidades

## Estructura del Proyecto

```
app-prediccion-liga-argentina/
├── streamlit_app.py        # Aplicación Streamlit
├── app/
│   ├── train_model.py      # Script de entrenamiento
│   ├── preprocessing.py    # Transformadores de preprocesamiento
│   └── models/             # Modelos guardados
├── .streamlit/
│   └── config.toml         # Configuración de Streamlit
├── requirements.txt        # Dependencias
├── liga_argentina_features_v2 (1).csv  # Dataset
└── README.md
```

## Modelo

- **Algoritmo**: Logistic Regression con regularización ElasticNet
- **Entrenamiento**: 3095 partidos (2015-2022)
- **Evaluación**: 1011 partidos (2023-2025)
- **Métrica**: F1-Score ponderado

