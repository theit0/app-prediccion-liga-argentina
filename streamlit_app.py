from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

from numpy.random import PCG64
import numpy.random._pickle as np_random_pickle

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


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop: Optional[Iterable[str]] = None) -> None:
        self.columns_to_drop = list(columns_to_drop or [])

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        return X.drop(columns=self.columns_to_drop, errors="ignore")


sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "DropColumns", DropColumns)

class LegacyPCG64(PCG64):
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


def _compatible_bit_generator_ctor(bit_generator_name="MT19937"):
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


def _compatible_generator_ctor(
    bit_generator_name="MT19937",
    bit_generator_ctor=_compatible_bit_generator_ctor,
):
    return np_random_pickle.Generator(bit_generator_ctor(bit_generator_name))


if not hasattr(np_random_pickle, "__original_generator_ctor"):
    np_random_pickle.__original_generator_ctor = np_random_pickle.__generator_ctor
    np_random_pickle.__generator_ctor = _compatible_generator_ctor


@st.cache_data(show_spinner=False)
def load_odds(path: Path) -> pd.DataFrame:
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


@st.cache_data(show_spinner=False)
def load_features(path: Path) -> Tuple[pd.DataFrame, List[str]]:
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


@dataclass
class SimulationDataset:
    data: pd.DataFrame
    feature_columns: List[str]
    unmatched: pd.DataFrame


@st.cache_data(show_spinner=False)
def build_simulation_dataset(
    odds_path: Path, features_path: Path
) -> SimulationDataset:
    odds = load_odds(odds_path)
    features, feature_columns = load_features(features_path)

    features_2024 = features[
        features["match_datetime"].dt.year == 2024
    ].reset_index(drop=False)
    odds_2024 = odds[odds["Fecha"].dt.year == 2024].reset_index(drop=False)

    feature_map: Dict[Tuple[str, str], List[int]] = {}
    for _, row in features_2024.iterrows():
        key = (row["Equipo_local"], row["Equipo_visitante"])
        feature_map.setdefault(key, []).append(row["index"])

    used_features: set[int] = set()
    records: List[Dict] = []
    unmatched_rows: List[int] = []

    for _, odds_row in odds_2024.iterrows():
        key = (odds_row["Equipo_local"], odds_row["Equipo_visitante"])
        candidates = feature_map.get(key, [])
        available = [idx for idx in candidates if idx not in used_features]
        search_space = available or candidates

        best_idx: Optional[int] = None
        best_delta: Optional[pd.Timedelta] = None
        for candidate_idx in search_space:
            candidate = features.loc[candidate_idx]
            candidate_date = candidate.get("match_date")
            if pd.isna(candidate_date) or pd.isna(odds_row["match_date"]):
                delta = pd.Timedelta.max
            else:
                delta = abs(candidate_date - odds_row["match_date"])
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = candidate_idx

        if best_idx is None:
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
    )


@st.cache_resource(show_spinner=False)
def load_models(model_files: Dict[str, str]) -> Dict[str, object]:
    models = {}
    for name, filename in model_files.items():
        model_path = DATA_DIR / filename
        models[name] = joblib.load(model_path)
    return models


def simulate_betting(
    model,
    dataset: pd.DataFrame,
    feature_columns: List[str],
    stake: float,
    initial_balance: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
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


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def main() -> None:
    st.set_page_config(
        page_title="Simulacion de apuestas Liga Argentina 2024",
        layout="wide",
    )
    st.title("Simulacion de apuestas con modelos entrenados")

    dataset_bundle = build_simulation_dataset(
        DATA_DIR / "cuotas_2024.csv",
        DATA_DIR / "liga_argentina_features_v3.csv",
    )

    dataset = dataset_bundle.data
    feature_columns = dataset_bundle.feature_columns
    unmatched = dataset_bundle.unmatched

    if dataset.empty:
        st.error("No se encontraron datos para simular 2024.")
        return

    min_date = dataset["Fecha"].min()
    max_date = dataset["Fecha"].max()
    equipos = sorted(
        set(dataset["Equipo_local"]).union(dataset["Equipo_visitante"])
    )

    with st.sidebar:
        st.header("Parámetros de simulación")
        initial_balance = st.number_input(
            "Saldo inicial",
            min_value=0.0,
            value=1000.0,
            step=100.0,
        )
        stake = st.number_input(
            "Monto fijo por apuesta",
            min_value=1.0,
            value=50.0,
            step=5.0,
        )
        model_selection = st.multiselect(
            "Modelos a evaluar",
            options=list(MODEL_FILES.keys()),
            default=list(MODEL_FILES.keys()),
        )
        date_range = st.date_input(
            "Rango de fechas",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        equipo_filter = st.multiselect(
            "Filtrar por equipos (opcional)",
            options=equipos,
        )

    if not model_selection:
        st.warning("Selecciona al menos un modelo para continuar.")
        return

    start_date, end_date = date_range
    mask_date = dataset["Fecha"].between(
        pd.to_datetime(start_date), pd.to_datetime(end_date)
    )
    df_filtered = dataset[mask_date].copy()

    if equipo_filter:
        df_filtered = df_filtered[
            df_filtered["Equipo_local"].isin(equipo_filter)
            | df_filtered["Equipo_visitante"].isin(equipo_filter)
        ]

    df_filtered = df_filtered.reset_index(drop=True)

    if df_filtered.empty:
        st.warning("No hay partidos que cumplan los filtros seleccionados.")
        return

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

    models = load_models(MODEL_FILES)

    st.subheader(
        f"Simulando {len(df_filtered)} partidos entre "
        f"{pd.to_datetime(start_date).date()} y {pd.to_datetime(end_date).date()}"
    )

    all_results_for_plot: List[pd.DataFrame] = []

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
        )

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


if __name__ == "__main__":
    main()
