"""
App Streamlit para predecir resultados de partidos de la Liga Argentina
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# Configuración de página
st.set_page_config(
    page_title="Predicción Liga Argentina",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Definir clase DropColumns (necesaria para cargar el modelo)
class DropColumns(BaseEstimator, TransformerMixin):
    """Transformador personalizado para eliminar columnas"""
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Solo eliminar columnas que existen
        columns_to_drop = [col for col in self.columns_to_drop if col in X.columns]
        if columns_to_drop:
            return X.drop(columns=columns_to_drop)
        return X

# Cargar modelo y datos auxiliares
@st.cache_resource
def load_model():
    """Cargar el modelo entrenado"""
    try:
        model = joblib.load("model.pkl")
        return model
    except FileNotFoundError:
        st.error("No se encontró el archivo del modelo (model.pkl).")
        st.stop()

@st.cache_data
def load_auxiliary_data():
    """Cargar lista de equipos"""
    try:
        equipos = joblib.load("equipos.pkl")
        return equipos
    except FileNotFoundError:
        st.error("No se encontraron los archivos auxiliares.")
        st.stop()

@st.cache_data
def load_team_urls():
    """Cargar mapeo de equipos a URLs de iconos desde el CSV"""
    try:
        df = pd.read_csv("liga_argentina_features_v3.csv", sep=';')
        # Crear diccionario único de equipo -> URL
        team_urls = {}
        # Mapear desde Equipo_local
        for equipo, url in zip(df['Equipo_local'], df['local_team_url']):
            if pd.notna(equipo) and pd.notna(url) and url:
                team_urls[equipo] = url
        # Mapear desde Equipo_visitante (para cubrir cualquier caso)
        for equipo, url in zip(df['Equipo_visitante'], df['visitante_team_url']):
            if pd.notna(equipo) and pd.notna(url) and url:
                if equipo not in team_urls:  # Solo agregar si no existe
                    team_urls[equipo] = url
        return team_urls
    except Exception as e:
        st.warning(f"No se pudieron cargar las URLs de los equipos: {e}")
        return {}


# Cargar datos
model = load_model()
equipos = load_auxiliary_data()
team_urls = load_team_urls()

# Interfaz
st.title("⚽ Predicción de Resultados - Liga Argentina")
st.markdown("Selecciona dos equipos para predecir el resultado del partido")

# Obtener valores desde session_state (se actualizan después de los selectores)
equipo_local_actual = st.session_state.get("select_local", equipos[0] if equipos else "")
equipo_visitante_actual = st.session_state.get("select_visitante", equipos[1] if len(equipos) > 1 else equipos[0] if equipos else "")

# Mostrar enfrentamiento con iconos PRIMERO (visualmente arriba)
col_res1, col_res2, col_res3 = st.columns([2, 1, 2])

with col_res1:
    if equipo_local_actual in team_urls:
        st.image(team_urls[equipo_local_actual], width=100)

with col_res2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### VS")
    st.markdown("<br>", unsafe_allow_html=True)

with col_res3:
    if equipo_visitante_actual in team_urls:
        st.image(team_urls[equipo_visitante_actual], width=100)


# Selectores de equipos DESPUÉS (pero se actualizarán los valores arriba)
col1, col2 = st.columns(2)

with col1:
    equipo_local = st.selectbox("Equipo Local", equipos, index=0, key="select_local")

with col2:
    equipo_visitante = st.selectbox("Equipo Visitante", equipos, index=1, key="select_visitante")


# Botón de predicción
if st.button("Predecir Resultado", type="primary", use_container_width=True):
    # Crear fila de datos para la predicción
    # Usamos valores neutros por defecto (como si fuera inicio de temporada)
    default_values = {
        'fecha_del_partido': pd.Timestamp('2025-01-15'),  # Fecha dummy
        'season': 2025,
        'sub_season': 1,
        'fixture_id': 0,  # Dummy ID
        'round': 'Regular Season - 1',
        'Equipo_local_id': 0,  # Dummy ID, se eliminará
        'Equipo_local': equipo_local,
        'Equipo_visitante_id': 0,  # Dummy ID, se eliminará  
        'Equipo_visitante': equipo_visitante,
        'Partidos_jugados_local_previos': 0,
        'Partidos_jugados_visitante_previos': 0,
        'Forma_local_ultimos5': '',
        'Forma_visitante_ultimos5': '',
        'Forma_local_puntos_ultimos5': 7,  # Valor promedio
        'Forma_visitante_puntos_ultimos5': 7,
        # Valores normalizados en 0.5 (medio)
        'Victorias_local_en_casa_tasa_normalizada': 0.5,
        'Victorias_visitante_fuera_tasa_normalizada': 0.5,
        'Empates_local_en_casa_tasa_normalizada': 0.5,
        'Empates_visitante_fuera_tasa_normalizada': 0.5,
        'Derrotas_local_en_casa_tasa_normalizada': 0.5,
        'Derrotas_visitante_fuera_tasa_normalizada': 0.5,
        'Promedio_Goles_marcados_totales_local_normalizado': 0.5,
        'Promedio_Goles_marcados_totales_visitante_normalizado': 0.5,
        'Promedio_Goles_recibidos_totales_local_normalizado': 0.5,
        'Promedio_Goles_recibidos_totales_visitante_normalizado': 0.5,
        'Promedio_Goles_marcados_local_en_casa_normalizado': 0.5,
        'Promedio_Goles_marcados_visitante_fuera_normalizado': 0.5,
        'Promedio_Goles_recibidos_local_en_casa_normalizado': 0.5,
        'Promedio_Goles_recibidos_visitante_fuera_normalizado': 0.5,
        'Valla_invicta_local_tasa_normalizada': 0.5,
        'Valla_invicta_visitante_tasa_normalizada': 0.5,
        'Promedio_Diferencia_gol_total_local_normalizado': 0.0,
        'Promedio_Diferencia_gol_total_visitante_normalizado': 0.0,
        'Promedio_Puntuacion_total_local_normalizado': 0.5,
        'Promedio_Puntuacion_total_visitante_normalizado': 0.5,
        'local_team_value': 0.0,  # Valor dummy, se eliminará
        'visitante_team_value': 0.0,  # Valor dummy, se eliminará
        'local_team_url': team_urls.get(equipo_local, ''),  # URL del equipo local
        'visitante_team_url': team_urls.get(equipo_visitante, ''),  # URL del equipo visitante
        'local_team_value_normalized': 0.5,
        'visitante_team_value_normalized': 0.5,
    }
    
    # Crear DataFrame
    df_pred = pd.DataFrame([default_values])

    # Hacer predicción
    try:
        prediccion = model.predict(df_pred)[0]
        probabilidades = model.predict_proba(df_pred)[0]
        
        # Mostrar resultado con iconos
        st.markdown("---")
        
        # Mostrar predicción
        if prediccion == "Ganador local":
            col_win1, col_win2 = st.columns([0.2, 0.8])
            with col_win1:
                if equipo_local in team_urls:
                    st.image(team_urls[equipo_local], width=60)
            with col_win2:
                st.success(f"🏆 **{equipo_local}** gana")
        elif prediccion == "Ganador visitante":
            col_win1, col_win2 = st.columns([0.2, 0.8])
            with col_win1:
                if equipo_visitante in team_urls:
                    st.image(team_urls[equipo_visitante], width=60)
            with col_win2:
                st.success(f"🏆 **{equipo_visitante}** gana")
        else:
            st.info("🤝 **Empate**")
        
        # Mostrar probabilidades
        st.markdown("### Probabilidades")
        
        resultados_labels = model.named_steps['model'].classes_
        prob_df = pd.DataFrame({
            'Resultado': resultados_labels,
            'Probabilidad': probabilidades
        })
        
        # Barra de probabilidades
        for i, row in prob_df.iterrows():
            prob_pct = row['Probabilidad'] * 100
            st.progress(float(row['Probabilidad']), text=f"{row['Resultado']}: {prob_pct:.1f}%")
        
    except Exception as e:
        st.error(f"Error al hacer la predicción: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("*Modelo entrenado con datos históricos de la Liga Argentina*")

