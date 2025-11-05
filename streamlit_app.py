"""
App Streamlit para predecir resultados de partidos de la Liga Argentina
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import altair as alt
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date
from sklearn.base import BaseEstimator, TransformerMixin

# Utilidades para compatibilidad con pickles creados bajo versiones previas.
from numpy.random import PCG64
import numpy.random._pickle as np_random_pickle

st.set_page_config(
    page_title="Predicción Liga Argentina",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class DropColumns(BaseEstimator, TransformerMixin):
    """Pequeño bloque reutilizado por los modelos al entrenar.

    Los pipelines guardados eliminan un conjunto fijo de columnas al comenzar;
    para que ese paso funcione al reabrir el .pkl, volvemos a declarar la clase
    exactamente igual. Recibe la lista de columnas a descartar y, durante la
    predicción, quita esas columnas del DataFrame que entra al pipeline.
    """
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = list(columns_to_drop or [])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors="ignore")

# --- Registro de la clase en __main__ para que joblib la encuentre al deserializar.
# Al momento de entrenar los pipelines, Python guardó que DropColumns vivía
# en el módulo "__main__". Cuando ejecutamos la app ese módulo ya no es el
# mismo, por lo que forzamos a que el nombre apunte al módulo actual y
# agregamos la clase manualmente. Así, al abrir el .pkl, joblib localiza
# de nuevo la definición y puede reconstruir el pipeline sin fallar.
sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "DropColumns", DropColumns)

# --- Parche para cargar estados antiguos del generador PCG64 sin errores.
class LegacyPCG64(PCG64):
    """Adaptación del generador aleatorio PCG64 para viejas versiones de NumPy.

    Algunos modelos fueron guardados con una versión anterior de NumPy.
    Al actualizarnos, el formato del estado interno cambió.
    Este método reacomoda los datos para que el nuevo NumPy los entienda.
    """

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
            head, _ = state
            state = dict(head)
            state["bit_generator"] = "LegacyPCG64"
        super().__setstate__(state)


np_random_pickle.BitGenerators["PCG64"] = LegacyPCG64
np_random_pickle.BitGenerators["LegacyPCG64"] = LegacyPCG64
np_random_pickle.BitGenerators.setdefault("numpy.random._pcg64.PCG64", LegacyPCG64)
np_random_pickle.BitGenerators.setdefault(PCG64, LegacyPCG64)
np_random_pickle.BitGenerators.setdefault(LegacyPCG64, LegacyPCG64)

# --- Helper que busca múltiples alias para reconstruir el bit generator correcto - Reproducibilidad.
def _compatible_bit_generator_ctor(bit_generator_name="MT19937"):
    """Devuelve una clase de bit generator compatible con pickles antiguos.

    Los ficheros serializados pueden haber guardado el generador de números
    aleatorios usando nombres diferentes (la clase, la ruta completa, etc.).
    Exploramos todas las variantes posibles hasta encontrar una coincidencia
    en la tabla de bit generators. Si ninguna calza, caemos en el constructor
    original de NumPy.
    """

    candidates: List[object] = []
    if isinstance(bit_generator_name, type):
        candidates.extend(
            [
                bit_generator_name,
                getattr(bit_generator_name, "__qualname__", ""),
                getattr(bit_generator_name, "__name__", ""),
                f"{bit_generator_name.__module__}.{bit_generator_name.__name__}",
            ]
        )
    else:
        candidates.append(bit_generator_name)
        module = getattr(bit_generator_name, "__module__", "")
        name = getattr(bit_generator_name, "__name__", "")
        if module and name:
            candidates.append(f"{module}.{name}")
        cls = bit_generator_name.__class__
        candidates.append(cls)
        candidates.append(getattr(cls, "__name__", ""))
        candidates.append(f"{cls.__module__}.{cls.__name__}")

    for candidate in candidates:
        if candidate in np_random_pickle.BitGenerators:
            generator = np_random_pickle.BitGenerators[candidate]
            return generator()

    return np_random_pickle.__original_bit_generator_ctor(bit_generator_name)


if not hasattr(np_random_pickle, "__original_bit_generator_ctor"):
    np_random_pickle.__original_bit_generator_ctor = np_random_pickle.__bit_generator_ctor
    np_random_pickle.__bit_generator_ctor = _compatible_bit_generator_ctor

# --- Replica el constructor de Generator para que use el bit generator parcheado - Reproducibilidad.
def _compatible_generator_ctor(
    bit_generator_name="MT19937",
    bit_generator_ctor=_compatible_bit_generator_ctor,
):
    """Construye un Generator de NumPy usando el bit generator adaptado.

    garantizamos que el pipeline use la misma secuencia pseudoaleatoria que en el entrenamiento,
    manteniendo el comportamiento reproducible del modelo cuando lo cargás en otra máquina o con otra versión de NumPy.
    """

    return np_random_pickle.Generator(bit_generator_ctor(bit_generator_name))


if not hasattr(np_random_pickle, "__original_generator_ctor"):
    np_random_pickle.__original_generator_ctor = np_random_pickle.__generator_ctor
    np_random_pickle.__generator_ctor = _compatible_generator_ctor

@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except FileNotFoundError:
        st.error("No se encontró el archivo del modelo (model.pkl).")
        st.stop()

@st.cache_data
def load_auxiliary_data():
    try:
        equipos = joblib.load("equipos.pkl")
        return equipos
    except FileNotFoundError:
        st.error("No se encontraron los archivos auxiliares.")
        st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("liga_argentina_features_v3.csv", sep=';')
        df['fecha_del_partido'] = pd.to_datetime(df['fecha_del_partido'], errors='coerce', utc=True)
        return df
    except Exception as e:
        st.error(f"No se pudo cargar el archivo de datos: {e}")
        st.stop()

@st.cache_data
def load_team_urls():
    try:
        df = pd.read_csv("liga_argentina_features_v3.csv", sep=';')
        team_urls = {}
        for equipo, url in zip(df['Equipo_local'], df['local_team_url']):
            if pd.notna(equipo) and pd.notna(url) and url:
                team_urls[equipo] = url
        for equipo, url in zip(df['Equipo_visitante'], df['visitante_team_url']):
            if pd.notna(equipo) and pd.notna(url) and url:
                if equipo not in team_urls:
                    team_urls[equipo] = url
        return team_urls
    except Exception as e:
        st.warning(f"No se pudieron cargar las URLs de los equipos: {e}")
        return {}

# --- Rutas y metadatos base para localizar modelos y columnas relevantes.
# DATA_DIR: Carpeta donde está este archivo. Desde allí buscamos CSV y modelos.
# MODEL_FILES: Diccionario que traduce un nombre amigable a un archivo .pkl específico.
# RESULT_LABELS: Posibles resultados que devuelven los modelos (local, empate, visita).
# ODDS_COLUMN_BY_RESULT: Para cada resultado, cuál columna del CSV contiene la cuota.
DATA_DIR = Path(__file__).resolve().parent
MODEL_FILES: Dict[str, str] = {
    "Regresion Logistica": "model.pkl",
    "Random Forest": "model-rf.pkl",
    "SVC": "model-svc.pkl",
    "HistGradientBoosting": "model-hgb.pkl",
}
RESULT_LABELS = ["Ganador local", "Empate", "Ganador visitante"]
ODDS_COLUMN_BY_RESULT = {
    "Ganador local": "Cuota1",
    "Empate": "CuotaX",
    "Ganador visitante": "Cuota2",
}

# --- Lectura y limpieza de cuotas: fechas, cuotas numéricas y resultado real.
@st.cache_data(show_spinner=False)
def load_odds(path: Path) -> pd.DataFrame:
    """Lee el archivo de cuotas y lo deja listo para usar.

    Pasos principales:
    1. Renombrar columnas para que coincidan con los nombres del dataset de features.
    2. Convertir fechas y cuotas a tipos numéricos/fecha reales.
    3. A partir del marcador "X:Y", deducir si ganó el local, el visitante o empataron.
    """

    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Equipo1": "Equipo_local",
            "Equipo2": "Equipo_visitante",
            "Resultado": "marcador",
        }
    )
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d.%m.%Y", errors="coerce")
    for col in ("Cuota1", "CuotaX", "Cuota2"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def parse_score(value: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        try:
            goles_local, goles_visita = value.split(":")
            home = int(goles_local)
            away = int(goles_visita)
        except (AttributeError, ValueError):
            return None, None, None
        if home > away:
            outcome = "Ganador local"
        elif away > home:
            outcome = "Ganador visitante"
        else:
            outcome = "Empate"
        return home, away, outcome

    parsed = df["marcador"].apply(parse_score)
    df["goles_local"] = parsed.map(lambda x: x[0])
    df["goles_visitante"] = parsed.map(lambda x: x[1])
    df["resultado_real_desde_marcador"] = parsed.map(lambda x: x[2])
    df["match_date"] = df["Fecha"]
    return df

# --- Carga de features históricas más lista de columnas que esperan los modelos.
@st.cache_data(show_spinner=False)
def load_features(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Abre el dataset con las características utilizadas para entrenar los modelos.

    Devuelve dos cosas:
    1. Un DataFrame que mantiene intactas todas las columnas, pero añade dos
       versiones de la fecha del partido:
       - `match_datetime`: con zona horaria, igual que en el dataset original.
       - `match_date`: sin zona horaria y reducida al día, ideal para emparejar.
    2. La lista de columnas que componen las features (todas menos `Resultado`),
       que más adelante se usará para extraer exactamente lo que esperan
       los pipelines al momento de predecir.
    """

    features_raw = pd.read_csv(path, sep=";")
    feature_columns = [col for col in features_raw.columns if col != "Resultado"]
    features = features_raw.copy()
    features["match_datetime"] = pd.to_datetime(
        features["fecha_del_partido"], errors="coerce"
    )
    features["match_date"] = (
        features["match_datetime"].dt.tz_convert(None).dt.normalize()
    )
    return features, feature_columns

# --- Estructura que agrupa dataset preparado y metadata auxiliar de simulación.
@dataclass
class SimulationDataset:
    """Devolver de una sola vez el dataset listo con todo lo necesario para ejecutar la simulación."""

    data: pd.DataFrame
    feature_columns: List[str]
    unmatched: pd.DataFrame
    features_lookup: pd.DataFrame

# --- Empareja cuotas 2024 con features 2024 seleccionando la fecha más cercana.
@st.cache_data(show_spinner=False)
def build_simulation_dataset(
    odds_path: Path, features_path: Path
) -> SimulationDataset:
    """Junta en una sola tabla la información de cuotas y la de features para 2024.

    Cómo funciona:
    1. Filtra ambos orígenes (cuotas y features) al año 2024.
    2. Agrupa las features por enfrentamiento (par `Equipo_local`, `Equipo_visitante`).
    3. Recorre cada partido con cuotas y busca dentro de ese grupo la fila cuya fecha
       esté más cerca de la fecha del partido en las cuotas. Esto nos asegura que
       las estadísticas utilizadas correspondan al mismo cruce (o al más cercano).
    4. Marca los índices de features que ya usó para evitar reutilizarlos en partidos
       duplicados.
    5. Si no encuentra ningún candidato compatible (por equipo o por fecha), anota
       ese partido dentro de `unmatched` para informar en la interfaz qué quedó fuera.
     
     Proporcionando  
    .El DataFrame ya combinado (cuotas + features) que alimenta la simulación histórica.
    .La lista de columnas de entrada que los modelos esperan.
    .El listado de partidos sin match para avisarlo en la interfaz.
    .Las features 2024 "crudas" para reutilizarlas en la predicción rápida.
    """

    odds = load_odds(odds_path)
    features, feature_columns = load_features(features_path)

    features_2024_full = features[
        features["match_datetime"].dt.year == 2024
    ].copy()
    features_2024 = features_2024_full.reset_index(drop=False)
    odds_2024 = odds[odds["Fecha"].dt.year == 2024].reset_index(drop=False)

    # Tabla auxiliar: para cada combinación local/visitante guardamos
    # la lista de índices disponibles en el dataset de features.
    feature_map: Dict[Tuple[str, str], List[int]] = {}
    for _, row in features_2024.iterrows():
        key = (row["Equipo_local"], row["Equipo_visitante"])
        feature_map.setdefault(key, []).append(row["index"])

    used_features: set[int] = set()
    records: List[Dict] = []
    unmatched_rows: List[int] = []

    for _, odds_row in odds_2024.iterrows():
        key = (odds_row["Equipo_local"], odds_row["Equipo_visitante"])
        # Traemos los candidatos que coinciden en equipos.
        candidates = feature_map.get(key, [])
        available = [idx for idx in candidates if idx not in used_features]
        # Si todavía no usamos alguno de ellos, priorizamos los que están libres.
        search_space = available or candidates

        best_idx: Optional[int] = None
        best_delta: Optional[pd.Timedelta] = None
        for candidate_idx in search_space:
            candidate = features.loc[candidate_idx]
            candidate_date = candidate.get("match_date")
            # Cuando falta la fecha en cualquiera de las tablas, rechazamos el match.
            if pd.isna(candidate_date) or pd.isna(odds_row["match_date"]):
                delta = pd.Timedelta.max
            else:
                # Distancia absoluta en días para identificar la fecha más cercana.
                delta = abs(candidate_date - odds_row["match_date"])
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = candidate_idx

        if best_idx is None:
            # No hay features compatibles: el partido se mostrará como omitido.
            unmatched_rows.append(odds_row["index"])
            continue

        used_features.add(best_idx)
        feature_row = features.loc[best_idx].to_dict()
        resultado_real = feature_row.pop("Resultado", None)

        record = {
            **feature_row,
            "Resultado_real": resultado_real,
            "Fecha": odds_row["Fecha"],
            "marcador": odds_row["marcador"],
            "Cuota1": odds_row["Cuota1"],
            "CuotaX": odds_row["CuotaX"],
            "Cuota2": odds_row["Cuota2"],
            "goles_local": odds_row["goles_local"],
            "goles_visitante": odds_row["goles_visitante"],
            "resultado_real_desde_marcador": odds_row[
                "resultado_real_desde_marcador"
            ],
        }
        records.append(record)

    dataset = pd.DataFrame(records)
    dataset = dataset.sort_values("Fecha").reset_index(drop=True)

    unmatched = odds.loc[unmatched_rows].reset_index(drop=True)
    return SimulationDataset(
        data=dataset,
        feature_columns=feature_columns,
        unmatched=unmatched,
        features_lookup=features_2024_full,
    )

@st.cache_resource(show_spinner=False)
def load_models(model_files: Dict[str, str]) -> Dict[str, object]:
    """Abre cada archivo .pkl y devuelve los modelos listos para predecir.

    La clave del diccionario es el nombre legible (ej. "Random Forest")
    y el valor es el pipeline completo con preprocesamiento + modelo.
    """

    models = {}
    for name, filename in model_files.items():
        model_path = DATA_DIR / filename
        try:
            models[name] = joblib.load(model_path)
        except FileNotFoundError:
            # Si un modelo no existe, lo omitimos silenciosamente
            continue
    return models

# --- Ejecuta las predicciones y calcula ganancias, saldo y métricas de desempeño.
def simulate_betting(
    model,
    dataset: pd.DataFrame,
    feature_columns: List[str],
    stake: float,
    initial_balance: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Corre la simulación de apuestas para un modelo concreto.

    Produce dos cosas:
    - Un DataFrame con el detalle de cada partido (predicción, cuota aplicada, saldo, etc.).
    - Un resumen con métricas agregadas (ganancia total, ROI, precisión de aciertos).

    Se asume una estrategia muy sencilla: apostar siempre el mismo monto
    (stake) a lo que el modelo predice, sin aplicar martingalas ni ajustes.
    """

    if dataset.empty:
        return pd.DataFrame(), {
            "final_balance": initial_balance,
            "profit": 0.0,
            "aciertos": 0,
            "total_bets": 0,
            "accuracy": np.nan,
            "roi": np.nan,
        }

    features = dataset[feature_columns].copy()
    predictions = model.predict(features)

    results = dataset[
        [
            "Fecha",
            "Equipo_local",
            "Equipo_visitante",
            "marcador",
            "Resultado_real",
            "Cuota1",
            "CuotaX",
            "Cuota2",
        ]
    ].copy()
    results["Prediccion"] = predictions
    results["Acierto"] = results["Prediccion"] == results["Resultado_real"]
    results["Resultado_match"] = results["Acierto"].map(
        {True: "✅", False: "❌"}
    )
    results["Cuota_usada"] = results.apply(
        lambda row: row.get(ODDS_COLUMN_BY_RESULT.get(row["Prediccion"]), np.nan),
        axis=1,
    )
    results["Stake"] = stake
    results["Ganancia"] = np.where(
        results["Acierto"],
        stake * (results["Cuota_usada"] - 1.0),
        -stake,
    )

    balances: List[float] = []
    balance = initial_balance
    for profit in results["Ganancia"]:
        balance += profit
        balances.append(balance)
    results["Balance"] = balances

    total_bets = len(results)
    aciertos = int(results["Acierto"].sum())
    profit = balance - initial_balance
    accuracy = aciertos / total_bets if total_bets else np.nan
    roi = profit / initial_balance if initial_balance else np.nan

    summary = {
        "final_balance": balance,
        "profit": profit,
        "aciertos": aciertos,
        "total_bets": total_bets,
        "accuracy": accuracy,
        "roi": roi,
    }
    return results, summary

# --- Formato consistente para mostrar montos de dinero.
def format_currency(value: float) -> str:
    """Convierte números en un string amigable con símbolo de moneda."""

    return f"${value:,.2f}"

# --- Función para generar gráfico radar de comparación de equipos
def crear_grafico_radar_equipos(df: pd.DataFrame):
    """Crea un gráfico radar comparativo de equipos usando Altair."""
    try:
        # Datos como local
        local_metrics = df.groupby('Equipo_local').agg({
            'local_team_value': 'mean',
            'Victorias_local_en_casa_tasa_normalizada': 'mean',
            'Promedio_Goles_marcados_local_en_casa_normalizado': 'mean',
            'Promedio_Goles_recibidos_local_en_casa_normalizado': 'mean',
            'Valla_invicta_local_tasa_normalizada': 'mean',
            'Promedio_Puntuacion_total_local_normalizado': 'mean'
        }).reset_index()

        local_metrics.columns = [
            'Equipo', 'Valor', 'Victorias', 'Goles_Marcados',
            'Goles_Recibidos', 'Valla_Invicta', 'Puntuacion'
        ]

        # Datos como visitante
        visit_metrics = df.groupby('Equipo_visitante').agg({
            'visitante_team_value': 'mean',
            'Victorias_visitante_fuera_tasa_normalizada': 'mean',
            'Promedio_Goles_marcados_visitante_fuera_normalizado': 'mean',
            'Promedio_Goles_recibidos_visitante_fuera_normalizado': 'mean',
            'Valla_invicta_visitante_tasa_normalizada': 'mean',
            'Promedio_Puntuacion_total_visitante_normalizado': 'mean'
        }).reset_index()

        visit_metrics.columns = [
            'Equipo', 'Valor', 'Victorias', 'Goles_Marcados',
            'Goles_Recibidos', 'Valla_Invicta', 'Puntuacion'
        ]

        # Combinar y promediar ambas condiciones
        all_metrics = pd.concat([local_metrics, visit_metrics])
        radar_data = all_metrics.groupby('Equipo').agg({
            'Valor': 'mean',
            'Victorias': 'mean',
            'Goles_Marcados': 'mean',
            'Goles_Recibidos': 'mean',
            'Valla_Invicta': 'mean',
            'Puntuacion': 'mean'
        }).reset_index()

        # Normalizar el valor del equipo a escala 0-1
        if radar_data['Valor'].max() != radar_data['Valor'].min():
            radar_data['Valor_Norm'] = (radar_data['Valor'] - radar_data['Valor'].min()) / (
                radar_data['Valor'].max() - radar_data['Valor'].min()
            )
        else:
            radar_data['Valor_Norm'] = 0.5

        radar_data = radar_data.dropna()

        if radar_data.empty:
            return None

        # Lista de equipos
        equipos_disponibles = sorted(radar_data['Equipo'].unique().tolist())

        if len(equipos_disponibles) == 0:
            return None

        # Preparar métricas (ahora todas normalizadas entre 0 y 1)
        metrics = ['Victorias', 'Goles_Marcados', 'Goles_Recibidos', 'Valla_Invicta', 'Puntuacion', 'Valor_Norm']
        metric_labels = ['Victorias', 'Goles A Favor', 'Goles Recibidos', 'Vallas Invictas', 'Puntuación', 'Valor Equipo']

        # Crear datos en formato largo
        radar_long = []
        for _, row in radar_data.iterrows():
            for metric, label in zip(metrics, metric_labels):
                radar_long.append({
                    'Equipo': row['Equipo'],
                    'Metrica': label,
                    'Valor': row[metric]
                })

        radar_df = pd.DataFrame(radar_long)

        # Calcular coordenadas polares
        metric_order = {label: i for i, label in enumerate(metric_labels)}
        radar_df['orden'] = radar_df['Metrica'].map(metric_order)
        radar_df['angulo'] = (radar_df['orden'] / len(metric_labels)) * 2 * np.pi
        radar_df['x'] = radar_df['Valor'] * np.cos(radar_df['angulo'])
        radar_df['y'] = radar_df['Valor'] * np.sin(radar_df['angulo'])

        # Cerrar el polígono para cada equipo
        radar_df_closed = []
        for equipo in radar_df['Equipo'].unique():
            equipo_data = radar_df[radar_df['Equipo'] == equipo].copy()
            first_point = equipo_data.iloc[0:1].copy()
            radar_df_closed.append(pd.concat([equipo_data, first_point]))

        radar_df_closed = pd.concat(radar_df_closed, ignore_index=True)

        # Crear datos auxiliares
        axis_data = []
        for label in metric_labels:
            orden = metric_order[label]
            angulo = (orden / len(metric_labels)) * 2 * np.pi
            axis_data.append({
                'Metrica': label,
                'x': 0, 'y': 0,
                'x2': np.cos(angulo),
                'y2': np.sin(angulo)
            })
        axis_df = pd.DataFrame(axis_data)

        label_data = []
        for label in metric_labels:
            orden = metric_order[label]
            angulo = (orden / len(metric_labels)) * 2 * np.pi
            label_data.append({
                'Metrica': label,
                'x': 1.2 * np.cos(angulo),
                'y': 1.2 * np.sin(angulo)
            })
        label_df = pd.DataFrame(label_data)

        circle_data = []
        for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
            for i in range(len(metric_labels) + 1):
                angulo = (i / len(metric_labels)) * 2 * np.pi
                circle_data.append({
                    'radio': r,
                    'x': r * np.cos(angulo),
                    'y': r * np.sin(angulo)
                })
        circle_df = pd.DataFrame(circle_data)

        # Crear selecciones de equipos
        equipo1_select = alt.selection_point(
            name='equipo1',
            fields=['Equipo'],
            bind=alt.binding_select(
                options=[None] + equipos_disponibles,
                labels=['Seleccionar...'] + equipos_disponibles,
                name='Equipo 1: '
            ),
            value=[{'Equipo': equipos_disponibles[0]}]
        )

        equipo2_select = alt.selection_point(
            name='equipo2',
            fields=['Equipo'],
            bind=alt.binding_select(
                options=[None] + equipos_disponibles,
                labels=['Seleccionar...'] + equipos_disponibles,
                name='Equipo 2: '
            ),
            value=[{'Equipo': equipos_disponibles[1] if len(equipos_disponibles) > 1 else equipos_disponibles[0]}]
        )

        # Círculos de referencia
        circles = alt.Chart(circle_df).mark_line(
            strokeWidth=0.5, stroke='lightgray', opacity=0.5
        ).encode(
            x=alt.X('x:Q', scale=alt.Scale(domain=[-1.4, 1.4]), axis=None),
            y=alt.Y('y:Q', scale=alt.Scale(domain=[-1.4, 1.4]), axis=None),
            detail='radio:Q'
        )

        # Ejes
        axes = alt.Chart(axis_df).mark_rule(
            strokeWidth=1, stroke='gray', opacity=0.5
        ).encode(x='x:Q', y='y:Q', x2='x2:Q', y2='y2:Q')

        # Labels
        labels = alt.Chart(label_df).mark_text(
            fontSize=12, fontWeight='bold'
        ).encode(x='x:Q', y='y:Q', text='Metrica:N')

        # Polígono Equipo 1
        radar1 = alt.Chart(radar_df_closed).mark_line(
            point=True, strokeWidth=3, filled=True, opacity=0.3
        ).encode(
            x='x:Q', y='y:Q',
            order='orden:Q',
            color=alt.value('#1f77b4'),  # Color for team 1
            tooltip=[
                alt.Tooltip('Equipo:N'),
                alt.Tooltip('Metrica:N'),
                alt.Tooltip('Valor:Q', format='.3f', title='Valor (0-1)')
            ]
        ).add_params(
            equipo1_select
        ).transform_filter(
            equipo1_select
        )

        # Polígono Equipo 2
        radar2 = alt.Chart(radar_df_closed).mark_line(
            point=True, strokeWidth=3, filled=True, opacity=0.3
        ).encode(
            x='x:Q', y='y:Q',
            order='orden:Q',
            color=alt.value('#ff7f0e'),  # Color for team 2
            tooltip=[
                alt.Tooltip('Equipo:N'),
                alt.Tooltip('Metrica:N'),
                alt.Tooltip('Valor:Q', format='.3f', title='Valor (0-1)')
            ]
        ).add_params(
            equipo2_select
        ).transform_filter(
            equipo2_select
        )

        # Combinar todo
        radar_chart = (circles + axes + labels + radar1 + radar2).properties(
            width=600,
            height=600,
            title={
                'text': 'Comparación de Equipos',
            }
        )

        return radar_chart

    except Exception as e:
        return None

# --- Función para generar gráfico de comparación de forma reciente entre equipos
def crear_grafico_comparacion_forma(df: pd.DataFrame, equipo1: str, equipo2: str, fecha: str):
    """Crea un gráfico de comparación de forma reciente entre dos equipos usando matplotlib."""
    try:
        # Preparar datos de forma reciente
        forma_data = df[['fecha_del_partido', 'Equipo_local', 'Equipo_visitante',
                         'Forma_local_puntos_ultimos5', 'Forma_visitante_puntos_ultimos5',
                         'Forma_local_ultimos5', 'Forma_visitante_ultimos5']].copy()

        forma_data['fecha'] = pd.to_datetime(forma_data['fecha_del_partido'])

        # Crear dataset combinado
        forma_local = forma_data[['fecha', 'Equipo_local', 'Forma_local_puntos_ultimos5', 'Forma_local_ultimos5']].copy()
        forma_local.columns = ['fecha', 'Equipo', 'Puntos_Ultimos5', 'Forma_Ultimos5']

        forma_visitante = forma_data[['fecha', 'Equipo_visitante', 'Forma_visitante_puntos_ultimos5', 'Forma_visitante_ultimos5']].copy()
        forma_visitante.columns = ['fecha', 'Equipo', 'Puntos_Ultimos5', 'Forma_Ultimos5']

        forma_combined = pd.concat([forma_local, forma_visitante])
        forma_combined = forma_combined.dropna(subset=['Puntos_Ultimos5'])
        forma_combined = forma_combined.sort_values('fecha')
        forma_combined['fecha_str'] = forma_combined['fecha'].dt.strftime('%Y-%m-%d')

        # Filtrar datos
        data_eq1 = forma_combined[(forma_combined['Equipo'] == equipo1) &
                                  (forma_combined['fecha_str'] == fecha)]
        data_eq2 = forma_combined[(forma_combined['Equipo'] == equipo2) &
                                  (forma_combined['fecha_str'] == fecha)]

        if data_eq1.empty:
            data_eq1 = forma_combined[forma_combined['Equipo'] == equipo1].tail(1)
            if not data_eq1.empty:
                fecha_mostrada_eq1 = data_eq1['fecha_str'].values[0]
                st.warning(f"No hay datos para {equipo1} en la fecha {fecha}. Mostrando último dato disponible: {fecha_mostrada_eq1}")

        if data_eq2.empty:
            data_eq2 = forma_combined[forma_combined['Equipo'] == equipo2].tail(1)
            if not data_eq2.empty:
                fecha_mostrada_eq2 = data_eq2['fecha_str'].values[0]
                st.warning(f"No hay datos para {equipo2} en la fecha {fecha}. Mostrando último dato disponible: {fecha_mostrada_eq2}")

        if data_eq1.empty or data_eq2.empty:
            st.error("❌ No se pueden mostrar datos para la comparación")
            return None

        # Crear figura
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1, 1]})
        fig.suptitle('Comparación de Forma Reciente entre Equipos', fontsize=16, fontweight='bold')

        # GRÁFICO 1: Barras de puntos
        ax1 = axes[0]

        puntos_eq1 = data_eq1['Puntos_Ultimos5'].values[0]
        puntos_eq2 = data_eq2['Puntos_Ultimos5'].values[0]

        bars = ax1.barh([equipo1, equipo2], [puntos_eq1, puntos_eq2],
                        color=['#4472C4', '#ED7D31'], height=0.5)

        ax1.set_xlim(0, 15)
        ax1.set_xlabel('Puntos (últimos 5 partidos)', fontsize=12)
        ax1.set_title('Puntos Obtenidos', fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Añadir valores en las barras
        for i, (bar, puntos) in enumerate(zip(bars, [puntos_eq1, puntos_eq2])):
            ax1.text(puntos + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{int(puntos)} pts', va='center', fontweight='bold', fontsize=11)

        # Líneas de referencia
        ax1.axvline(x=15, color='green', linestyle='--', alpha=0.4, linewidth=1)
        ax1.axvline(x=10, color='blue', linestyle='--', alpha=0.3, linewidth=1)
        ax1.axvline(x=7.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        # GRÁFICO 2: Racha Equipo 1
        ax2 = axes[1]
        forma_eq1 = str(data_eq1['Forma_Ultimos5'].values[0]) if pd.notna(data_eq1['Forma_Ultimos5'].values[0]) else 'N/A'

        ax2.set_xlim(0, 5)
        ax2.set_ylim(0, 1)
        ax2.set_title(f'Racha Equipo 1: {equipo1}', fontsize=12, fontweight='bold')
        ax2.axis('off')

        if forma_eq1 != 'N/A':
            for i, resultado in enumerate(forma_eq1):
                color = {'W': '#2ca02c', 'D': '#ffcc00', 'L': '#d62728'}.get(resultado, 'gray')
                rect = mpatches.Rectangle((i, 0.2), 0.8, 0.6, linewidth=2,
                                         edgecolor='white', facecolor=color)
                ax2.add_patch(rect)
                texto = {'W': 'V', 'D': 'E', 'L': 'D'}.get(resultado, '?')
                ax2.text(i + 0.4, 0.5, texto, ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white')

        ax2.text(2.5, -0.2, '← Más antiguo    Más reciente →', ha='center',
                fontsize=9, style='italic', color='gray')

        # GRÁFICO 3: Racha Equipo 2
        ax3 = axes[2]
        forma_eq2 = str(data_eq2['Forma_Ultimos5'].values[0]) if pd.notna(data_eq2['Forma_Ultimos5'].values[0]) else 'N/A'

        ax3.set_xlim(0, 5)
        ax3.set_ylim(0, 1)
        ax3.set_title(f'Racha Equipo 2: {equipo2}', fontsize=12, fontweight='bold')
        ax3.axis('off')

        if forma_eq2 != 'N/A':
            for i, resultado in enumerate(forma_eq2):
                color = {'W': '#2ca02c', 'D': '#ffcc00', 'L': '#d62728'}.get(resultado, 'gray')
                rect = mpatches.Rectangle((i, 0.2), 0.8, 0.6, linewidth=2,
                                         edgecolor='white', facecolor=color)
                ax3.add_patch(rect)
                texto = {'W': 'V', 'D': 'E', 'L': 'D'}.get(resultado, '?')
                ax3.text(i + 0.4, 0.5, texto, ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white')

        ax3.text(2.5, -0.2, '← Más antiguo    Más reciente →', ha='center',
                fontsize=9, style='italic', color='gray')

        # Leyenda
        victoria_patch = mpatches.Patch(color='#2ca02c', label='Victoria (V)')
        empate_patch = mpatches.Patch(color='#ffcc00', label='Empate (E)')
        derrota_patch = mpatches.Patch(color='#d62728', label='Derrota (D)')
        fig.legend(handles=[victoria_patch, empate_patch, derrota_patch],
                  loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        return fig, puntos_eq1, puntos_eq2, forma_eq1, forma_eq2

    except Exception as e:
        st.error(f"Error al generar el gráfico de comparación de forma: {str(e)}")
        return None, None, None, None, None

# --- Función helper para preparar datos de forma
def preparar_datos_forma(df: pd.DataFrame):
    """Prepara los datos de forma reciente para los selectores."""
    try:
        forma_data = df[['fecha_del_partido', 'Equipo_local', 'Equipo_visitante',
                         'Forma_local_puntos_ultimos5', 'Forma_visitante_puntos_ultimos5',
                         'Forma_local_ultimos5', 'Forma_visitante_ultimos5']].copy()

        forma_data['fecha'] = pd.to_datetime(forma_data['fecha_del_partido'])

        # Crear dataset combinado
        forma_local = forma_data[['fecha', 'Equipo_local', 'Forma_local_puntos_ultimos5', 'Forma_local_ultimos5']].copy()
        forma_local.columns = ['fecha', 'Equipo', 'Puntos_Ultimos5', 'Forma_Ultimos5']

        forma_visitante = forma_data[['fecha', 'Equipo_visitante', 'Forma_visitante_puntos_ultimos5', 'Forma_visitante_ultimos5']].copy()
        forma_visitante.columns = ['fecha', 'Equipo', 'Puntos_Ultimos5', 'Forma_Ultimos5']

        forma_combined = pd.concat([forma_local, forma_visitante])
        forma_combined = forma_combined.dropna(subset=['Puntos_Ultimos5'])
        forma_combined = forma_combined.sort_values('fecha')
        forma_combined['fecha_str'] = forma_combined['fecha'].dt.strftime('%Y-%m-%d')

        equipos_disponibles = sorted(forma_combined['Equipo'].unique().tolist())
        fechas_disponibles = sorted(forma_combined['fecha_str'].unique().tolist())

        return forma_combined, equipos_disponibles, fechas_disponibles
    except Exception as e:
        return None, [], []

# --- Función para generar gráfico de relación entre valor de mercado y rendimiento
def crear_grafico_valor_rendimiento(df: pd.DataFrame):
    """Crea un scatter plot mostrando la relación entre valor de mercado y porcentaje de victorias."""
    try:
        equipos_stats = []
        
        for season in df['season'].unique():
            season_df = df[df['season'] == season]
            for equipo in season_df['Equipo_local'].unique():
                local_games = season_df[season_df['Equipo_local'] == equipo]
                away_games = season_df[season_df['Equipo_visitante'] == equipo]
                
                wins_local = (local_games['Resultado'] == 'Ganador local').sum()
                wins_away = (away_games['Resultado'] == 'Ganador visitante').sum()
                
                total_games = len(local_games) + len(away_games)
                total_wins = wins_local + wins_away
                
                valor = local_games['local_team_value'].iloc[0] if len(local_games) > 0 else (
                    away_games['visitante_team_value'].iloc[0] if len(away_games) > 0 else None)
                
                if total_games > 0 and pd.notna(valor):
                    equipos_stats.append({
                        'equipo': equipo, 'season': season, 'victorias': total_wins, 'partidos': total_games,
                        'porcentaje_victorias': (total_wins / total_games) * 100, 'valor_mercado': valor
                    })
        
        equipos_df = pd.DataFrame(equipos_stats)
        
        if equipos_df.empty:
            return None
        
        cinco_grandes = ['River Plate', 'Boca Juniors', 'Racing Club', 'San Lorenzo', 'Independiente']
        equipos_df['categoria'] = equipos_df['equipo'].apply(lambda x: '5 Grandes' if x in cinco_grandes else 'Otros')
        
        # Selector de temporada con dropdown
        seasons_available = sorted(equipos_df['season'].unique().tolist())
        if not seasons_available:
            return None
        
        season_dropdown = alt.binding_select(
            options=seasons_available,
            name='Temporada: '
        )
        season_select = alt.selection_point(
            fields=['season'], 
            bind=season_dropdown, 
            value=seasons_available[-1] if seasons_available else None
        )
        
        # Scatter plot que solo muestra la temporada seleccionada
        scatter = alt.Chart(equipos_df).mark_circle(size=150, opacity=0.85).encode(
            x=alt.X('valor_mercado:Q', title='Valor de Mercado (millones €)', scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
            y=alt.Y('porcentaje_victorias:Q', title='Porcentaje de Victorias (%)',
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
            color=alt.Color('categoria:N',
                           scale=alt.Scale(domain=['5 Grandes', 'Otros'], range=['#e74c3c', '#3498db']),
                           legend=alt.Legend(title='Categoría', titleFontSize=14, labelFontSize=12)),
            size=alt.Size('partidos:Q', scale=alt.Scale(range=[100, 500]),
                         legend=alt.Legend(title='Partidos', titleFontSize=12)),
            tooltip=[
                alt.Tooltip('equipo:N', title='Equipo'),
                alt.Tooltip('season:O', title='Temporada'),
                alt.Tooltip('valor_mercado:Q', title='Valor (M€)', format='.1f'),
                alt.Tooltip('porcentaje_victorias:Q', title='% Victorias', format='.1f'),
                alt.Tooltip('victorias:Q', title='Victorias'),
                alt.Tooltip('partidos:Q', title='Partidos')
            ]
        ).transform_filter(
            season_select
        ).add_params(season_select)
        
        # Línea de regresión para la temporada seleccionada
        regression = scatter.transform_regression(
            'valor_mercado', 'porcentaje_victorias', method='linear'
        ).mark_line(color='#e67e22', strokeWidth=3, strokeDash=[5, 5])
        
        chart = (scatter + regression).properties(
            width=750, height=500,
            title=alt.TitleParams(
                text='Relación entre Valor de Mercado y Rendimiento',
                subtitle='Selecciona una temporada del menú | Tamaño del círculo = cantidad de partidos',
                fontSize=18, fontWeight='bold', anchor='middle'
            )
        ).configure_view(strokeWidth=0)
        
        return chart
        
    except Exception as e:
        return None

# --- Función helper para mostrar estadísticas normalizadas con barras de progreso y colores
def mostrar_estadistica_normalizada(st, label: str, valor: float, es_inverso: bool = False):
    """Muestra una estadística normalizada con barra de progreso y colores según umbrales.
    
    Args:
        st: Instancia de streamlit
        label: Etiqueta de la estadística
        valor: Valor normalizado (0-1)
        es_inverso: Si True, valores más bajos son mejores (ej: goles recibidos)
    """
    # Convertir valor a porcentaje (0-100)
    porcentaje = valor * 100
    
    # Determinar color según umbrales
    if es_inverso:
        # Para métricas inversas (goles recibidos): más bajo es mejor
        if valor <= 0.4:
            color = "#28a745"  # Verde
        elif valor <= 0.6:
            color = "#ffc107"  # Amarillo
        else:
            color = "#dc3545"  # Rojo
    else:
        # Para métricas normales (victorias, goles marcados, etc.): más alto es mejor
        if valor >= 0.6:
            color = "#28a745"  # Verde
        elif valor >= 0.4:
            color = "#ffc107"  # Amarillo
        else:
            color = "#dc3545"  # Rojo
    
    # Mostrar el valor y la barra de progreso con color
    st.write(f"**{label}:** {valor:.3f} ({porcentaje:.1f}%)")
    st.markdown(
        f"""
        <div style="background-color: #e0e0e0; border-radius: 10px; padding: 3px; margin-bottom: 15px; height: 25px;">
            <div style="background-color: {color}; width: {porcentaje}%; height: 19px; border-radius: 7px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 11px; transition: width 0.3s ease;">
                {porcentaje:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

model = load_model()
equipos = load_auxiliary_data()
team_urls = load_team_urls()
df_data = load_data()

# Mostrar logo y título lado a lado
col_logo, col_titulo = st.columns([1, 3])
with col_logo:
    st.image("logo.png", use_container_width=False, width=150)
with col_titulo:
    st.title("Predicción de Resultados - Liga Argentina")

# Dividir pantalla en dos columnas: izquierda (predicción) y derecha (simulador)
col_left, col_right = st.columns([1, 1])

# ============================================
# COLUMNA IZQUIERDA: PREDICCIÓN DE PARTIDOS
# ============================================
with col_left:
    # Crear tabs
    tab1, tab2 = st.tabs(["Predecir", "Análisis"])
    
    with tab1:
        st.subheader("Predicción de Partidos")
        st.markdown("Selecciona la temporada, subtemporada y equipos para predecir el resultado")

        # Paso 1: Seleccionar Temporada (solo 2024 o posteriores, ya que 2015-2023 se usaron para entrenar)
        seasons_todas = sorted(df_data['season'].unique(), reverse=True)
        seasons_disponibles = [s for s in seasons_todas if s >= 2024]

        if not seasons_disponibles:
            st.warning("⚠️ No hay temporadas disponibles para predicción (2024 o posteriores). El modelo se entrenó con datos de 2015-2023.")
            st.stop()

        # Selectores en una sola línea
        col_temp, col_subtemp, col_fecha = st.columns(3)
        
        with col_temp:
            season = st.selectbox("Temporada", options=seasons_disponibles, index=0)

        # Paso 2: Seleccionar Subtemporada basada en la temporada
        subseasons_disponibles = sorted(df_data[df_data['season'] == season]['sub_season'].unique())
        if not subseasons_disponibles:
            st.error(f"No hay subtemporadas disponibles para la temporada {season}")
            st.stop()

        with col_subtemp:
            sub_season = st.selectbox("Subtemporada", options=subseasons_disponibles, index=len(subseasons_disponibles)-1)

        # Obtener equipos que jugaron en esa season/sub_season
        df_subseason = df_data[(df_data['season'] == season) & (df_data['sub_season'] == sub_season)]
        equipos_local = sorted(df_subseason['Equipo_local'].unique())
        equipos_visitante = sorted(df_subseason['Equipo_visitante'].unique())
        equipos_disponibles = sorted(set(equipos_local + equipos_visitante))

        if not equipos_disponibles:
            st.error(f"No hay equipos disponibles para la temporada {season} subtemporada {sub_season}")
            st.stop()

        # Obtener rango de fechas de la subseason (solo desde 2024-01-01 en adelante)
        fecha_limite_entrenamiento = pd.Timestamp('2024-01-01').date()
        fechas_subseason = df_subseason['fecha_del_partido'].dropna()
        if not fechas_subseason.empty:
            fecha_min_data = fechas_subseason.min().date()
            # Asegurar que la fecha mínima sea desde 2024
            fecha_min = max(fecha_min_data, fecha_limite_entrenamiento)
        else:
            fecha_min = fecha_limite_entrenamiento

        # Ajustar fecha al año de la temporada seleccionada
        if 'last_season' not in st.session_state or st.session_state.last_season != season:
            hoy = pd.Timestamp.now().date()
            fecha_default = pd.Timestamp(year=season, month=hoy.month, day=min(hoy.day, 28)).date()
            fecha_default = max(fecha_default, fecha_min)
            st.session_state.last_season = season
        else:
            fecha_default = st.session_state.get('fecha_partido', max(pd.Timestamp.now().date(), fecha_min))

        # Paso 3: Seleccionar fecha
        with col_fecha:
            fecha_partido = st.date_input("Fecha del partido", value=fecha_default, min_value=fecha_min)
        st.session_state.fecha_partido = fecha_partido

        # Paso 4: Seleccionar equipos
        subseason_key = f"{season}_{sub_season}"
        if 'last_subseason_key' not in st.session_state or st.session_state.last_subseason_key != subseason_key:
            st.session_state.last_subseason_key = subseason_key
            st.session_state.equipo_local = equipos_disponibles[0]
            st.session_state.equipo_visitante = equipos_disponibles[1] if len(equipos_disponibles) > 1 else equipos_disponibles[0]

        # Obtener equipos actuales del session_state para mostrar logos
        equipo_local_actual = st.session_state.get('equipo_local', equipos_disponibles[0])
        equipo_visitante_actual = st.session_state.get('equipo_visitante', equipos_disponibles[1] if len(equipos_disponibles) > 1 else equipos_disponibles[0])
        
        # Contenedor con logos grandes centrados (arriba de los selectores)
        col_logo_local, col_vs, col_logo_visitante = st.columns([1, 0.2, 1])
        
        with col_logo_local:
            if equipo_local_actual in team_urls:
                st.image(team_urls[equipo_local_actual], width=120)
        
        with col_vs:
            st.markdown("""
                <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                    <h2 style="margin: 0; color: #666;">vs.</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col_logo_visitante:
            if equipo_visitante_actual in team_urls:
                st.image(team_urls[equipo_visitante_actual], width=120)
        
        # Selectores de equipos (debajo de los logos)
        col1, col2, col3 = st.columns([2.5, 0.5, 2.5])

        with col1:
            st.write("Equipo Local")
            equipo_local = st.selectbox("", equipos_disponibles, 
                                       index=equipos_disponibles.index(st.session_state.equipo_local) if st.session_state.equipo_local in equipos_disponibles else 0,
                                       key="select_local", label_visibility="collapsed")
            st.session_state.equipo_local = equipo_local

        with col2:
            st.write("")
            st.write("")
            if st.button("⇄", key="switch_teams", use_container_width=True, help="Intercambiar equipos"):
                st.session_state.equipo_local, st.session_state.equipo_visitante = st.session_state.equipo_visitante, st.session_state.equipo_local
                st.rerun()

        with col3:
            st.write("Equipo Visitante")
            equipo_visitante = st.selectbox("", equipos_disponibles,
                                           index=equipos_disponibles.index(st.session_state.equipo_visitante) if st.session_state.equipo_visitante in equipos_disponibles else 0,
                                           key="select_visitante", label_visibility="collapsed")
            st.session_state.equipo_visitante = equipo_visitante


        if st.button("Predecir Resultado", type="primary", use_container_width=True, key="predict_button_left"):
            if equipo_local == equipo_visitante:
                st.error("Por favor selecciona dos equipos diferentes")
            else:
                # Validar que la fecha sea desde 2024 en adelante
                if fecha_partido < pd.Timestamp('2024-01-01').date():
                    st.error("❌ No se pueden predecir partidos anteriores a 2024, ya que esos datos se usaron para entrenar el modelo.")
                    st.stop()
                
                fecha_actual = pd.to_datetime(fecha_partido, utc=True)
                
                # Buscar último partido del equipo local como local en la subseason actual (antes de la fecha seleccionada)
                df_local = df_data[(df_data['Equipo_local'] == equipo_local) & 
                                  (df_data['season'] == season) & 
                                  (df_data['sub_season'] == sub_season) &
                                  (df_data['fecha_del_partido'] <= fecha_actual)]
                
                # Si no hay en la subseason actual, buscar el último partido sin importar subseason (>= 2024 y antes de la fecha)
                if df_local.empty:
                    df_local = df_data[(df_data['Equipo_local'] == equipo_local) & 
                                      (df_data['season'] >= 2024) &
                                      (df_data['fecha_del_partido'] <= fecha_actual)]
                
                df_local = df_local.sort_values('fecha_del_partido', ascending=False)
                
                # Buscar último partido del equipo visitante como visitante en la subseason actual (antes de la fecha seleccionada)
                df_visitante = df_data[(df_data['Equipo_visitante'] == equipo_visitante) & 
                                      (df_data['season'] == season) & 
                                      (df_data['sub_season'] == sub_season) &
                                      (df_data['fecha_del_partido'] <= fecha_actual)]
                
                # Si no hay en la subseason actual, buscar el último partido sin importar subseason (>= 2024 y antes de la fecha)
                if df_visitante.empty:
                    df_visitante = df_data[(df_data['Equipo_visitante'] == equipo_visitante) & 
                                          (df_data['season'] >= 2024) &
                                          (df_data['fecha_del_partido'] <= fecha_actual)]
                
                df_visitante = df_visitante.sort_values('fecha_del_partido', ascending=False)
                
                # Validar que ambos equipos tengan datos
                if df_local.empty:
                    st.error(f"❌ No se encontraron partidos previos para {equipo_local} como local.")
                    st.stop()
                
                if df_visitante.empty:
                    st.error(f"❌ No se encontraron partidos previos para {equipo_visitante} como visitante.")
                    st.stop()
                
                # Informar si se usaron datos de otra subseason o temporada
                partido_local_found = df_local.iloc[0]
                partido_visitante_found = df_visitante.iloc[0]
                
                if partido_local_found['season'] != season or partido_local_found['sub_season'] != sub_season:
                    st.info(f"ℹ️ Se usaron datos de {equipo_local} de la temporada {partido_local_found['season']} subtemporada {partido_local_found['sub_season']} (no se encontraron datos en {season}-{sub_season})")
                
                if partido_visitante_found['season'] != season or partido_visitante_found['sub_season'] != sub_season:
                    st.info(f"ℹ️ Se usaron datos de {equipo_visitante} de la temporada {partido_visitante_found['season']} subtemporada {partido_visitante_found['sub_season']} (no se encontraron datos en {season}-{sub_season})")
                
                # Construir valores usando los últimos partidos encontrados (con valores normalizados)
                def get_val(series, col, default):
                    return series[col] if col in series and pd.notna(series[col]) else default
                
                partido_local = partido_local_found
                partido_visitante = partido_visitante_found
                
                default_values = {
                    'fecha_del_partido': fecha_actual,
                    'season': season,
                    'sub_season': sub_season,
                    'fixture_id': 0,
                    'round': get_val(partido_local, 'round', 'Regular Season - 1'),
                    'Equipo_local_id': get_val(partido_local, 'Equipo_local_id', 0),
                    'Equipo_local': equipo_local,
                    'Equipo_visitante_id': get_val(partido_visitante, 'Equipo_visitante_id', 0),
                    'Equipo_visitante': equipo_visitante,
                    'Partidos_jugados_local_previos': get_val(partido_local, 'Partidos_jugados_local_previos', 0),
                    'Partidos_jugados_visitante_previos': get_val(partido_visitante, 'Partidos_jugados_visitante_previos', 0),
                    'Forma_local_ultimos5': get_val(partido_local, 'Forma_local_ultimos5', ''),
                    'Forma_visitante_ultimos5': get_val(partido_visitante, 'Forma_visitante_ultimos5', ''),
                    'Forma_local_puntos_ultimos5': get_val(partido_local, 'Forma_local_puntos_ultimos5', 7),
                    'Forma_visitante_puntos_ultimos5': get_val(partido_visitante, 'Forma_visitante_puntos_ultimos5', 7),
                    'Victorias_local_en_casa_tasa_normalizada': get_val(partido_local, 'Victorias_local_en_casa_tasa_normalizada', 0.5),
                    'Victorias_visitante_fuera_tasa_normalizada': get_val(partido_visitante, 'Victorias_visitante_fuera_tasa_normalizada', 0.5),
                    'Empates_local_en_casa_tasa_normalizada': get_val(partido_local, 'Empates_local_en_casa_tasa_normalizada', 0.5),
                    'Empates_visitante_fuera_tasa_normalizada': get_val(partido_visitante, 'Empates_visitante_fuera_tasa_normalizada', 0.5),
                    'Derrotas_local_en_casa_tasa_normalizada': get_val(partido_local, 'Derrotas_local_en_casa_tasa_normalizada', 0.5),
                    'Derrotas_visitante_fuera_tasa_normalizada': get_val(partido_visitante, 'Derrotas_visitante_fuera_tasa_normalizada', 0.5),
                    'Promedio_Goles_marcados_totales_local_normalizado': get_val(partido_local, 'Promedio_Goles_marcados_totales_local_normalizado', 0.5),
                    'Promedio_Goles_marcados_totales_visitante_normalizado': get_val(partido_visitante, 'Promedio_Goles_marcados_totales_visitante_normalizado', 0.5),
                    'Promedio_Goles_recibidos_totales_local_normalizado': get_val(partido_local, 'Promedio_Goles_recibidos_totales_local_normalizado', 0.5),
                    'Promedio_Goles_recibidos_totales_visitante_normalizado': get_val(partido_visitante, 'Promedio_Goles_recibidos_totales_visitante_normalizado', 0.5),
                    'Promedio_Goles_marcados_local_en_casa_normalizado': get_val(partido_local, 'Promedio_Goles_marcados_local_en_casa_normalizado', 0.5),
                    'Promedio_Goles_marcados_visitante_fuera_normalizado': get_val(partido_visitante, 'Promedio_Goles_marcados_visitante_fuera_normalizado', 0.5),
                    'Promedio_Goles_recibidos_local_en_casa_normalizado': get_val(partido_local, 'Promedio_Goles_recibidos_local_en_casa_normalizado', 0.5),
                    'Promedio_Goles_recibidos_visitante_fuera_normalizado': get_val(partido_visitante, 'Promedio_Goles_recibidos_visitante_fuera_normalizado', 0.5),
                    'Valla_invicta_local_tasa_normalizada': get_val(partido_local, 'Valla_invicta_local_tasa_normalizada', 0.5),
                    'Valla_invicta_visitante_tasa_normalizada': get_val(partido_visitante, 'Valla_invicta_visitante_tasa_normalizada', 0.5),
                    'Promedio_Diferencia_gol_total_local_normalizado': get_val(partido_local, 'Promedio_Diferencia_gol_total_local_normalizado', 0.0),
                    'Promedio_Diferencia_gol_total_visitante_normalizado': get_val(partido_visitante, 'Promedio_Diferencia_gol_total_visitante_normalizado', 0.0),
                    'Promedio_Puntuacion_total_local_normalizado': get_val(partido_local, 'Promedio_Puntuacion_total_local_normalizado', 0.5),
                    'Promedio_Puntuacion_total_visitante_normalizado': get_val(partido_visitante, 'Promedio_Puntuacion_total_visitante_normalizado', 0.5),
                    'local_team_value': get_val(partido_local, 'local_team_value', 0.0),
                    'visitante_team_value': get_val(partido_visitante, 'visitante_team_value', 0.0),
                    'local_team_url': team_urls.get(equipo_local, ''),
                    'visitante_team_url': team_urls.get(equipo_visitante, ''),
                    'local_team_value_normalized': get_val(partido_local, 'local_team_value_normalized', 0.5),
                    'visitante_team_value_normalized': get_val(partido_visitante, 'visitante_team_value_normalized', 0.5),
                }
                
                df_pred = pd.DataFrame([default_values])
                
                # Generar features temporales como en el entrenamiento
                df_pred['fecha_del_partido'] = pd.to_datetime(df_pred['fecha_del_partido'], errors='coerce', utc=True)
                df_pred['partido_anio'] = df_pred['fecha_del_partido'].dt.year
                df_pred['partido_mes'] = df_pred['fecha_del_partido'].dt.month
                df_pred['partido_dia_semana'] = df_pred['fecha_del_partido'].dt.dayofweek

                try:
                    prediccion = model.predict(df_pred)[0]
                    probabilidades = model.predict_proba(df_pred)[0]
                    
                    st.markdown("---")
                    
                    if prediccion == "Ganador local":
                        imagen_html = ""
                        if equipo_local in team_urls:
                            imagen_html = f'<img src="{team_urls[equipo_local]}" width="40" style="vertical-align: middle; margin-right: 10px;">'
                            st.markdown(
                                f"""
                                <div style='margin-bottom: 20px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 0.5rem; padding: .5rem; display: flex; justify-content: center; align-items: center; gap: .5rem;'>
                                    {imagen_html}
                                    <span><strong>{equipo_local}</strong> gana</span>
                                </div>
                                """,
                            unsafe_allow_html=True
                        )
                    elif prediccion == "Ganador visitante":
                        imagen_html = ""
                        if equipo_visitante in team_urls:
                            imagen_html = f'<img src="{team_urls[equipo_visitante]}" width="40" style="vertical-align: middle; margin-right: 10px;">'
                            st.markdown(
                                f"""
                                <div style='margin-bottom: 20px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 0.5rem; padding: .5rem; display: flex; justify-content: center; align-items: center; gap: .5rem;'>
                                    {imagen_html}
                                    <span><strong>{equipo_visitante}</strong> gana</span>
                                </div>
                                """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.info(" **Empate**")
                    
                    
                    resultados_labels = model.named_steps['model'].classes_
                    prob_df = pd.DataFrame({
                        'Resultado': resultados_labels,
                        'Probabilidad': probabilidades
                    })
                    
                    for i, row in prob_df.iterrows():
                        prob_pct = row['Probabilidad'] * 100
                        st.progress(float(row['Probabilidad']), text=f"{row['Resultado']}: {prob_pct:.1f}%")
                    
                    # Desplegable con valores de entrada al modelo
                    with st.expander("Ver valores de entrada al modelo"):
                        # Obtener fechas de los partidos usados
                        fecha_partido_local = partido_local['fecha_del_partido'].date() if pd.notna(partido_local['fecha_del_partido']) else None
                        fecha_partido_visitante = partido_visitante['fecha_del_partido'].date() if pd.notna(partido_visitante['fecha_del_partido']) else None
                        
                        col_info1, col_info2 = st.columns(2)
                        
                        with col_info1:
                            # Mostrar imagen del equipo local
                            if equipo_local in team_urls:
                                st.image(team_urls[equipo_local], width=60)
                            st.subheader("Equipo Local")
                            # Obtener información del partido usado para el equipo local
                            equipo_local_partido = partido_local.get('Equipo_local', '')
                            equipo_visitante_partido_local = partido_local.get('Equipo_visitante', '')
                            if fecha_partido_local:
                                st.write(f"**Equipo:** {equipo_local} vs. {equipo_visitante_partido_local}")
                                st.write(f"**Datos del partido:** {fecha_partido_local.strftime('%d/%m/%Y')}")
                            else:
                                st.write(f"**Equipo:** {equipo_local}")
                            st.write(f"**Partidos jugados:** {default_values['Partidos_jugados_local_previos']}")
                            st.write(f"**Forma últimos 5:** {default_values['Forma_local_ultimos5']} ({default_values['Forma_local_puntos_ultimos5']} pts)")
                            st.markdown("---")
                            mostrar_estadistica_normalizada(st, "Victorias en casa (norm)", default_values['Victorias_local_en_casa_tasa_normalizada'])
                            mostrar_estadistica_normalizada(st, "Goles marcados (norm)", default_values['Promedio_Goles_marcados_totales_local_normalizado'])
                            mostrar_estadistica_normalizada(st, "Goles recibidos (norm)", default_values['Promedio_Goles_recibidos_totales_local_normalizado'], es_inverso=True)
                            mostrar_estadistica_normalizada(st, "Valla invicta (norm)", default_values['Valla_invicta_local_tasa_normalizada'])
                            mostrar_estadistica_normalizada(st, "Puntuación total (norm)", default_values['Promedio_Puntuacion_total_local_normalizado'])
                            mostrar_estadistica_normalizada(st, "Valor equipo (norm)", default_values['local_team_value_normalized'])
                        
                        with col_info2:
                            # Mostrar imagen del equipo visitante
                            if equipo_visitante in team_urls:
                                st.image(team_urls[equipo_visitante], width=60)
                            st.subheader("Equipo Visitante")
                            # Obtener información del partido usado para el equipo visitante
                            equipo_local_partido_visitante = partido_visitante.get('Equipo_local', '')
                            equipo_visitante_partido = partido_visitante.get('Equipo_visitante', '')
                            if fecha_partido_visitante:
                                st.write(f"**Equipo:** {equipo_visitante} vs. {equipo_local_partido_visitante}")
                                st.write(f"**Datos del partido:** {fecha_partido_visitante.strftime('%d/%m/%Y')}")
                            else:
                                st.write(f"**Equipo:** {equipo_visitante}")
                            st.write(f"**Partidos jugados:** {default_values['Partidos_jugados_visitante_previos']}")
                            st.write(f"**Forma últimos 5:** {default_values['Forma_visitante_ultimos5']} ({default_values['Forma_visitante_puntos_ultimos5']} pts)")
                            st.markdown("---")
                            mostrar_estadistica_normalizada(st, "Victorias fuera (norm)", default_values['Victorias_visitante_fuera_tasa_normalizada'])
                            mostrar_estadistica_normalizada(st, "Goles marcados (norm)", default_values['Promedio_Goles_marcados_totales_visitante_normalizado'])
                            mostrar_estadistica_normalizada(st, "Goles recibidos (norm)", default_values['Promedio_Goles_recibidos_totales_visitante_normalizado'], es_inverso=True)
                            mostrar_estadistica_normalizada(st, "Valla invicta (norm)", default_values['Valla_invicta_visitante_tasa_normalizada'])
                            mostrar_estadistica_normalizada(st, "Puntuación total (norm)", default_values['Promedio_Puntuacion_total_visitante_normalizado'])
                            mostrar_estadistica_normalizada(st, "Valor equipo (norm)", default_values['visitante_team_value_normalized'])
                    
                except Exception as e:
                    st.error(f"Error al hacer la predicción: {str(e)}")
                    st.exception(e)
    
    with tab2:
        st.subheader("Análisis de Equipos")
        
        # Sub-tabs dentro del tab Análisis
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Gráfico Radar", "Comparación de Forma", "Valor vs Rendimiento"])
        
        with sub_tab1:
            st.markdown("Compara estadísticas de diferentes equipos usando el gráfico radar")
            
            # Gráfico radar de comparación de equipos
            radar_chart = crear_grafico_radar_equipos(df_data)
            if radar_chart is not None:
                st.altair_chart(radar_chart, use_container_width=True)
            else:
                st.info("No se pudo generar el gráfico de comparación. Verifica que los datos estén disponibles.")
        
        with sub_tab2:
            st.markdown("Compara la forma reciente (últimos 5 partidos) entre dos equipos")
            
            # Preparar datos
            forma_combined, equipos_disponibles, fechas_disponibles = preparar_datos_forma(df_data)
            
            if forma_combined is not None and len(equipos_disponibles) > 0 and len(fechas_disponibles) > 0:
                # Valores por defecto
                equipo1_default = equipos_disponibles[0]
                equipo2_default = equipos_disponibles[1] if len(equipos_disponibles) > 1 else equipos_disponibles[0]
                fecha_default = fechas_disponibles[-1]
                
                # Selectores
                col_forma1, col_forma2, col_forma3 = st.columns(3)
                
                with col_forma1:
                    equipo1 = st.selectbox(
                        "Equipo 1",
                        options=equipos_disponibles,
                        index=equipos_disponibles.index(equipo1_default) if equipo1_default in equipos_disponibles else 0,
                        key="forma_equipo1"
                    )
                
                with col_forma2:
                    equipo2 = st.selectbox(
                        "Equipo 2",
                        options=equipos_disponibles,
                        index=equipos_disponibles.index(equipo2_default) if equipo2_default in equipos_disponibles else 0,
                        key="forma_equipo2"
                    )
                
                with col_forma3:
                    # Mostrar solo las últimas 100 fechas para no sobrecargar el selector
                    fechas_recientes = fechas_disponibles[-100:] if len(fechas_disponibles) > 100 else fechas_disponibles
                    fecha_seleccionada = st.selectbox(
                        "Fecha",
                        options=fechas_recientes,
                        index=len(fechas_recientes) - 1,
                        key="forma_fecha"
                    )
                
                # Generar gráfico
                if st.button("Generar Comparación", type="primary", key="generar_comparacion_forma"):
                    fig, puntos_eq1, puntos_eq2, forma_eq1, forma_eq2 = crear_grafico_comparacion_forma(
                        df_data, equipo1, equipo2, fecha_seleccionada
                    )
                    
                    if fig is not None:
                        st.pyplot(fig)
                        
                        # Mostrar estadísticas adicionales
                        st.markdown("---")
                        st.markdown("### Estadísticas Detalladas")
                        col_stat1, col_stat2 = st.columns(2)
                        
                        with col_stat1:
                            st.markdown(f"**{equipo1}**")
                            st.write(f"Puntos: **{int(puntos_eq1)}** puntos")
                            st.write(f"Racha: **{forma_eq1}**")
                        
                        with col_stat2:
                            st.markdown(f"**{equipo2}**")
                            st.write(f"Puntos: **{int(puntos_eq2)}** puntos")
                            st.write(f"Racha: **{forma_eq2}**")
                        
                        diferencia = abs(puntos_eq1 - puntos_eq2)
                        mejor = equipo1 if puntos_eq1 > puntos_eq2 else equipo2
                        if puntos_eq1 == puntos_eq2:
                            st.info("Ambos equipos tienen la misma forma reciente")
                        else:
                            st.success(f"**{mejor}** tiene mejor forma reciente (+{int(diferencia)} puntos)")
            else:
                st.warning("No se pudieron cargar los datos de forma reciente. Verifica que los datos estén disponibles.")
        
        with sub_tab3:
            st.markdown("Analiza la relación entre el valor de mercado de los equipos y su rendimiento en victorias")
            
            # Gráfico de relación entre valor de mercado y rendimiento
            valor_rendimiento_chart = crear_grafico_valor_rendimiento(df_data)
            if valor_rendimiento_chart is not None:
                st.altair_chart(valor_rendimiento_chart, use_container_width=True)
            else:
                st.info("No se pudo generar el gráfico de valor vs rendimiento. Verifica que los datos estén disponibles.")

# ============================================
# COLUMNA DERECHA: SIMULADOR DE APUESTAS
# ============================================
with col_right:
    st.subheader("Simulación de Apuestas 2024")
    st.markdown("Compara el rendimiento de diferentes modelos usando cuotas reales de 2024")
    
    # --- Carga única de cuotas + features ya emparejadas para toda la sesión.
    try:
        dataset_bundle = build_simulation_dataset(
            DATA_DIR / "cuotas_2024.csv",
            DATA_DIR / "liga_argentina_features_v3.csv",
        )

        dataset = dataset_bundle.data
        feature_columns = dataset_bundle.feature_columns
        unmatched = dataset_bundle.unmatched
        features_lookup = dataset_bundle.features_lookup

        if dataset.empty:
            st.error("No se encontraron datos para simular 2024.")
        else:
            # --- Cargamos en memoria los modelos elegidos para evitar repetir lecturas.
            models = load_models(MODEL_FILES)

            if not models:
                st.warning("No se encontraron modelos disponibles para simular.")
            else:
                # --- Módulo de predicción rápida para un partido puntual (colapsable).
                with st.expander("Predicción rápida por partido", expanded=False):
                    if features_lookup.empty:
                        st.info("No hay datos suficientes para generar predicciones manuales.")
                    else:
                        modelos_disponibles = list(models.keys())
                        modelo_seleccionado = st.selectbox(
                            "Modelo a utilizar",
                            modelos_disponibles,
                            index=0,
                            key="quick_model_selector",
                        )

                        equipos_locales = sorted(features_lookup["Equipo_local"].unique())
                        equipo_local = st.selectbox(
                            "Equipo local",
                            equipos_locales,
                            key="quick_home_selector",
                        )

                        visitantes_filtrados = sorted(
                            features_lookup.loc[
                                features_lookup["Equipo_local"] == equipo_local, "Equipo_visitante"
                            ].unique()
                        )
                        if not visitantes_filtrados:
                            visitantes_filtrados = sorted(
                                features_lookup["Equipo_visitante"].unique()
                            )
                        equipo_visitante = st.selectbox(
                            "Equipo visitante",
                            visitantes_filtrados,
                            key="quick_away_selector",
                        )

                        if st.button("Predecir resultado", type="primary", key="quick_predict_button"):
                            partidos_coincidentes = features_lookup[
                                (features_lookup["Equipo_local"] == equipo_local)
                                & (features_lookup["Equipo_visitante"] == equipo_visitante)
                            ].sort_values("match_datetime", ascending=False)

                            if partidos_coincidentes.empty:
                                st.error("No hay datos de características para este enfrentamiento.")
                            else:
                                registro = partidos_coincidentes.iloc[0]
                                X_partido = pd.DataFrame([registro[feature_columns]])
                                modelo = models[modelo_seleccionado]

                                prediccion = modelo.predict(X_partido)[0]
                                if prediccion == "Ganador local":
                                    mensaje = f"{equipo_local} gana"
                                elif prediccion == "Ganador visitante":
                                    mensaje = f"{equipo_visitante} gana"
                                else:
                                    mensaje = "Empate"
                                st.success(f"Pronóstico principal: {mensaje}")

                                if hasattr(modelo, "predict_proba"):
                                    probabilidades = modelo.predict_proba(X_partido)[0]
                                    clase_prob = dict(zip(modelo.classes_, probabilidades))

                                    st.subheader("Probabilidades estimadas")
                                    for etiqueta in RESULT_LABELS:
                                        valor = clase_prob.get(etiqueta, 0.0)
                                        st.write(f"{etiqueta}: {valor:.1%}")
                                        st.progress(float(np.clip(valor, 0.0, 1.0)))
                                else:
                                    st.info(
                                        "Este modelo no expone probabilidades "
                                        "(`predict_proba`)."
                                    )

                min_date = dataset["Fecha"].min()
                max_date = dataset["Fecha"].max()
                equipos = sorted(
                    set(dataset["Equipo_local"]).union(dataset["Equipo_visitante"])
                )

                # --- Controles: capital inicial, stake, modelos y filtros temporales.
                col_saldo, col_monto = st.columns(2)
                
                with col_saldo:
                    initial_balance = st.number_input(
                        "Saldo inicial",
                        min_value=0.0,
                        value=1000.0,
                        step=100.0,
                        key="sim_initial_balance",
                    )
                with col_monto:
                    stake = st.number_input(
                        "Monto fijo por apuesta",
                        min_value=1.0,
                        value=50.0,
                        step=5.0,
                        key="sim_stake",
                    )
                model_selection = st.multiselect(
                    "Modelos a evaluar",
                    options=list(models.keys()),
                    default=list(models.keys()),
                    key="sim_model_selection",
                )
                date_range = st.date_input(
                    "Rango de fechas",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="sim_date_range",
                )
                equipo_filter = st.multiselect(
                    "Filtrar por equipos (opcional)",
                    options=equipos,
                    key="sim_equipo_filter",
                )

                if not model_selection:
                    st.warning("Selecciona al menos un modelo para continuar.")
                else:
                    # Botón para ejecutar la simulación
                    if st.button("Ejecutar Simulación", type="primary", use_container_width=True, key="run_simulation_button"):
                        start_date, end_date = date_range
                        mask_date = dataset["Fecha"].between(
                            pd.to_datetime(start_date), pd.to_datetime(end_date)
                        )
                        df_filtered = dataset[mask_date].copy()

                        # --- Filtro opcional por equipos involucrados.
                        if equipo_filter:
                            df_filtered = df_filtered[
                                df_filtered["Equipo_local"].isin(equipo_filter)
                                | df_filtered["Equipo_visitante"].isin(equipo_filter)
                            ]

                        df_filtered = df_filtered.reset_index(drop=True)

                        if df_filtered.empty:
                            st.warning("No hay partidos que cumplan los filtros seleccionados.")
                        else:
                            # Guardar resultados en session_state para que persistan después del botón
                            st.session_state.sim_df_filtered = df_filtered
                            st.session_state.sim_start_date = start_date
                            st.session_state.sim_end_date = end_date
                            st.session_state.sim_unmatched = unmatched
                            st.rerun()
                    
                    # Mostrar resultados de la simulación si existen en session_state
                    if 'sim_df_filtered' in st.session_state and not st.session_state.sim_df_filtered.empty:
                        df_filtered = st.session_state.sim_df_filtered
                        start_date = st.session_state.sim_start_date
                        end_date = st.session_state.sim_end_date
                        unmatched = st.session_state.sim_unmatched
                        
                        if df_filtered.empty:
                            st.warning("No hay partidos que cumplan los filtros seleccionados.")
                        else:
                            # --- Aviso de partidos descartados por ausencia de features compatibles.
                            if not unmatched.empty:
                                with st.expander("Partidos sin features asociados"):
                                    st.write(
                                        "Los siguientes partidos no pudieron enlazarse con las "
                                        "features disponibles y se excluyeron de la simulación."
                                    )
                                    st.dataframe(
                                        unmatched[
                                            ["Fecha", "Equipo_local", "Equipo_visitante", "marcador"]
                                        ],
                                        use_container_width=True,
                                    )

                            st.markdown(f"**Simulando {len(df_filtered)} partidos entre "
                                      f"{pd.to_datetime(start_date).date()} y {pd.to_datetime(end_date).date()}**")

                            # --- Almacenamos las curvas de saldo de cada modelo para graficar luego.
                            all_results_for_plot: List[pd.DataFrame] = []

                            # --- Simulación y resumen por cada modelo seleccionado.
                            # Aquí hacemos predict, calculamos ganancias y mostramos resultados.
                            for model_name in model_selection:
                                st.markdown(f"### {model_name}")
                                model = models[model_name]
                                results, summary = simulate_betting(
                                    model=model,
                                    dataset=df_filtered,
                                    feature_columns=feature_columns,
                                    stake=stake,
                                    initial_balance=initial_balance,
                                )

                                # --- Cuatro métricas directas para entender de un vistazo el desempeño.
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Saldo final", format_currency(summary["final_balance"]))
                                col2.metric("Ganancia neta", format_currency(summary["profit"]))
                                accuracy = (
                                    f"{summary['accuracy']*100:.1f}%"
                                    if not np.isnan(summary["accuracy"])
                                    else "N/A"
                                )
                                col3.metric("Acierto", f"{summary['aciertos']} / {summary['total_bets']}")
                                col4.metric("Efectividad", accuracy)

                                roi_value = summary["roi"]
                                if np.isnan(roi_value):
                                    st.caption("ROI sobre saldo inicial: N/A")
                                else:
                                    roi_percent = f"{roi_value*100:.1f}%"
                                    if roi_value > 0:
                                        roi_color = "#2ECC71"
                                    elif roi_value < 0:
                                        roi_color = "#E74C3C"
                                    else:
                                        roi_color = "#6C757D"
                                    # Mostramos el resultado en color verde si ganamos, rojo si perdimos.
                                    st.markdown(
                                        f"<p style='color:{roi_color}; font-size:0.9rem; margin-top:0;'>"
                                        f"ROI sobre saldo inicial: {roi_percent}"
                                        "</p>",
                                        unsafe_allow_html=True,
                                    )

                                if results.empty:
                                    st.info("No se generaron resultados para este modelo.")
                                    continue

                                plot_section = results[["Fecha", "Balance"]].copy()
                                plot_section["Modelo"] = model_name
                                all_results_for_plot.append(plot_section)

                                # --- Tabla con detalle de resultados, cuotas y saldo acumulado.
                                display_cols = [
                                    "Fecha",
                                    "Equipo_local",
                                    "Equipo_visitante",
                                    "marcador",
                                    "Resultado_real",
                                    "Prediccion",
                                    "Resultado_match",
                                    "Cuota_usada",
                                    "Ganancia",
                                    "Balance",
                                ]
                                st.dataframe(
                                    results[display_cols],
                                    use_container_width=True,
                                )
                                csv_data = results.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="Descargar detalle (.csv)",
                                    data=csv_data,
                                    file_name=f"simulacion_{model_name.replace(' ', '_').lower()}.csv",
                                    mime="text/csv",
                                    key=f"download_{model_name}",
                                )

                            # --- Visualización comparativa del saldo para los modelos evaluados.
                            if all_results_for_plot:
                                plot_df = pd.concat(all_results_for_plot, ignore_index=True)
                                plot_df["Fecha"] = pd.to_datetime(plot_df["Fecha"])

                                st.subheader("Evolución comparativa del saldo")
                                chart = (
                                    alt.Chart(plot_df)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("Fecha:T", title="Fecha"),
                                        y=alt.Y("Balance:Q", title="Saldo acumulado"),
                                        color=alt.Color("Modelo:N", title="Modelo"),
                                        tooltip=[
                                            alt.Tooltip("Modelo:N", title="Modelo"),
                                            alt.Tooltip("Fecha:T", title="Fecha"),
                                            alt.Tooltip("Balance:Q", title="Saldo"),
                                        ],
                                    )
                                )
                                st.altair_chart(chart, use_container_width=True)
    except FileNotFoundError as e:
        st.warning(f"⚠️ No se encontró el archivo de cuotas 2024. El simulador requiere el archivo 'cuotas_2024.csv' en el directorio del proyecto.")
    except Exception as e:
        st.error(f"Error al cargar el simulador: {str(e)}")
        st.exception(e)
