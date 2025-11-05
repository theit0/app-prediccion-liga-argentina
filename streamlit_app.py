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

model = load_model()
equipos = load_auxiliary_data()
team_urls = load_team_urls()
df_data = load_data()

st.title("⚽ Predicción de Resultados - Liga Argentina")

# Dividir pantalla en dos columnas: izquierda (predicción) y derecha (simulador)
col_left, col_right = st.columns([1, 1])

# ============================================
# COLUMNA IZQUIERDA: PREDICCIÓN DE PARTIDOS
# ============================================
with col_left:
    st.subheader("Predicción de Partidos")
    st.markdown("Selecciona la temporada, subtemporada y equipos para predecir el resultado")

    # Paso 1: Seleccionar Temporada (solo 2024 o posteriores, ya que 2015-2023 se usaron para entrenar)
    seasons_todas = sorted(df_data['season'].unique(), reverse=True)
    seasons_disponibles = [s for s in seasons_todas if s >= 2024]

    if not seasons_disponibles:
        st.warning("⚠️ No hay temporadas disponibles para predicción (2024 o posteriores). El modelo se entrenó con datos de 2015-2023.")
        st.stop()

    season = st.selectbox("Temporada", options=seasons_disponibles, index=0)

    # Paso 2: Seleccionar Subtemporada basada en la temporada
    subseasons_disponibles = sorted(df_data[df_data['season'] == season]['sub_season'].unique())
    if not subseasons_disponibles:
        st.error(f"No hay subtemporadas disponibles para la temporada {season}")
        st.stop()

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
    fecha_partido = st.date_input("Fecha del partido", value=fecha_default, min_value=fecha_min)
    st.session_state.fecha_partido = fecha_partido

    # Paso 4: Seleccionar equipos
    subseason_key = f"{season}_{sub_season}"
    if 'last_subseason_key' not in st.session_state or st.session_state.last_subseason_key != subseason_key:
        st.session_state.last_subseason_key = subseason_key
        st.session_state.equipo_local = equipos_disponibles[0]
        st.session_state.equipo_visitante = equipos_disponibles[1] if len(equipos_disponibles) > 1 else equipos_disponibles[0]

    col1, col2, col3 = st.columns([2.5, 0.5, 2.5])

    with col1:
        st.write("Equipo Local")
        equipo_local = st.selectbox("", equipos_disponibles, 
                                   index=equipos_disponibles.index(st.session_state.equipo_local) if st.session_state.equipo_local in equipos_disponibles else 0,
                                   key="select_local", label_visibility="collapsed")
        st.session_state.equipo_local = equipo_local
        # Mostrar imagen sin columnas anidadas
        if equipo_local in team_urls:
            st.image(team_urls[equipo_local], width=40)

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
        # Mostrar imagen sin columnas anidadas
        if equipo_visitante in team_urls:
            st.image(team_urls[equipo_visitante], width=40)


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
                    st.info("🤝 **Empate**")
                
                
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
                        st.subheader("Equipo Local")
                        st.write(f"**Equipo:** {equipo_local}")
                        if fecha_partido_local:
                            st.write(f"**Datos del partido:** {fecha_partido_local.strftime('%d/%m/%Y')}")
                        st.write(f"**Partidos jugados:** {default_values['Partidos_jugados_local_previos']}")
                        st.write(f"**Forma últimos 5:** {default_values['Forma_local_ultimos5']} ({default_values['Forma_local_puntos_ultimos5']} pts)")
                        st.write(f"**Victorias en casa (norm):** {default_values['Victorias_local_en_casa_tasa_normalizada']:.3f}")
                        st.write(f"**Goles marcados (norm):** {default_values['Promedio_Goles_marcados_totales_local_normalizado']:.3f}")
                        st.write(f"**Goles recibidos (norm):** {default_values['Promedio_Goles_recibidos_totales_local_normalizado']:.3f}")
                        st.write(f"**Valla invicta (norm):** {default_values['Valla_invicta_local_tasa_normalizada']:.3f}")
                        st.write(f"**Valor equipo (norm):** {default_values['local_team_value_normalized']:.3f}")
                    
                    with col_info2:
                        st.subheader("Equipo Visitante")
                        st.write(f"**Equipo:** {equipo_visitante}")
                        if fecha_partido_visitante:
                            st.write(f"**Datos del partido:** {fecha_partido_visitante.strftime('%d/%m/%Y')}")
                        st.write(f"**Partidos jugados:** {default_values['Partidos_jugados_visitante_previos']}")
                        st.write(f"**Forma últimos 5:** {default_values['Forma_visitante_ultimos5']} ({default_values['Forma_visitante_puntos_ultimos5']} pts)")
                        st.write(f"**Victorias fuera (norm):** {default_values['Victorias_visitante_fuera_tasa_normalizada']:.3f}")
                        st.write(f"**Goles marcados (norm):** {default_values['Promedio_Goles_marcados_totales_visitante_normalizado']:.3f}")
                        st.write(f"**Goles recibidos (norm):** {default_values['Promedio_Goles_recibidos_totales_visitante_normalizado']:.3f}")
                        st.write(f"**Valla invicta (norm):** {default_values['Valla_invicta_visitante_tasa_normalizada']:.3f}")
                        st.write(f"**Valor equipo (norm):** {default_values['visitante_team_value_normalized']:.3f}")
                
            except Exception as e:
                st.error(f"Error al hacer la predicción: {str(e)}")
                st.exception(e)

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

                st.markdown("---")

                min_date = dataset["Fecha"].min()
                max_date = dataset["Fecha"].max()
                equipos = sorted(
                    set(dataset["Equipo_local"]).union(dataset["Equipo_visitante"])
                )

                # --- Controles: capital inicial, stake, modelos y filtros temporales.
                initial_balance = st.number_input(
                    "Saldo inicial",
                    min_value=0.0,
                    value=1000.0,
                    step=100.0,
                    key="sim_initial_balance",
                )
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
