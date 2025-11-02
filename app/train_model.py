"""
Script para entrenar el modelo de predicción de resultados de la Liga Argentina.
Guarda el pipeline completo con pickle/joblib para uso posterior en la app Streamlit.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib
from app.preprocessing import DropColumns

# Configuración
RANDOM_STATE = 42
TARGET = 'Resultado'
CUTOFF_YEAR = 2023

# Columnas a eliminar
columns_to_drop = [
    "Forma_local_ultimos5",
    "Forma_visitante_ultimos5",
    "fixture_id",
    "Equipo_local_id",
    "Equipo_visitante_id",
    "season",
    "sub_season",
    "round",
    "Partidos_jugados_visitante_previos",
    "Partidos_jugados_local_previos"
]


def main():
    # Cargar datos
    print("Cargando datos...")
    data_path = Path("liga_argentina_features_v2 (1).csv")
    df = pd.read_csv(data_path, sep=';')
    print(f"Dataset cargado: {df.shape}")
    
    # Procesar fechas y crear features temporales
    df['fecha_del_partido'] = pd.to_datetime(df['fecha_del_partido'], errors='coerce', utc=True)
    df['partido_anio'] = df['fecha_del_partido'].dt.year
    df['partido_mes'] = df['fecha_del_partido'].dt.month
    df['partido_dia_semana'] = df['fecha_del_partido'].dt.dayofweek
    df = df.sort_values('fecha_del_partido').reset_index(drop=True)
    df = df.drop(columns=['fecha_del_partido'])
    
    # Separar X e y
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    
    # División temporal
    print("Dividiendo datos en entrenamiento y prueba...")
    train_mask = X['partido_anio'] < CUTOFF_YEAR
    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    print(f"Entrenamiento: {X_train.shape[0]} partidos")
    print(f"Prueba: {(~train_mask).sum()} partidos")
    
    # Construir pipeline de preprocesamiento
    print("Construyendo pipeline de preprocesamiento...")
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    column_transform = ColumnTransformer([
        ("num", numeric_pipeline, make_column_selector(dtype_include=["int64", "float64"])),
        ("cat", categorical_pipeline, make_column_selector(dtype_include=["object", "category"]))
    ])
    
    preprocessor = Pipeline([
        ("drop_columns", DropColumns(columns_to_drop=columns_to_drop)),
        ("column_transform", column_transform)
    ])
    
    # Pipeline completo con modelo
    print("Entrenando modelo...")
    model = LogisticRegression(
        class_weight='balanced',
        C=0.01,
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Entrenar
    pipeline.fit(X_train, y_train)
    print("Modelo entrenado exitosamente")
    
    # Guardar modelo
    output_dir = Path("app/models")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Modelo guardado en: {model_path}")
    
    # Guardar lista de equipos
    equipos = sorted(df['Equipo_local'].unique().tolist())
    equipos_path = output_dir / "equipos.pkl"
    joblib.dump(equipos, equipos_path)
    print(f"Lista de equipos guardada en: {equipos_path}")
    
    # Guardar valores de mercado 2025
    team_values_2025 = {
        "River Plate": 86.25,
        "Boca Juniors": 83.35,
        "Racing Club": 77.23,
        "Independiente": 56.95,
        "Velez Sarsfield": 55.03,
        "Estudiantes L.P.": 46.58,
        "Rosario Central": 39.40,
        "Talleres Cordoba": 38.95,
        "Argentinos JRS": 37.58,
        "San Lorenzo": 35.08,
        "Lanus": 32.93,
        "Belgrano": 31.45,
        "Godoy Cruz": 30.60,
        "Huracan": 29.80,
        "Tigre": 28.15,
        "Defensa Y Justicia": 28.08,
        "Platense": 27.43,
        "Instituto Cordoba": 26.40,
        "Union Santa Fe": 25.50,
        "Independ. Rivadavia": 23.95,
        "Barracas Central": 21.30,
        "Newells Old Boys": 18.70,
        "Central Cordoba de Santiago": 17.48,
        "Banfield": 17.30,
        "Gimnasia L.P.": 16.88,
        "Atletico Tucuman": 15.98,
        "Sarmiento Junin": 12.63,
        "Aldosivi": 10.89,
        "San Martin S.J.": 10.13,
        "Deportivo Riestra": 7.99
    }
    values_path = output_dir / "team_values_2025.pkl"
    joblib.dump(team_values_2025, values_path)
    print(f"Valores de mercado guardados en: {values_path}")


if __name__ == "__main__":
    main()
