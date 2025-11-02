"""
App Streamlit para predecir resultados de partidos de la Liga Argentina
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from app.preprocessing import DropColumns

# Configuración de página
st.set_page_config(
    page_title="Predicción Liga Argentina",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
    """Cargar lista de equipos y valores de mercado"""
    try:
        equipos = joblib.load("app/models/equipos.pkl")
        values = joblib.load("app/models/team_values_2025.pkl")
        return equipos, values
    except FileNotFoundError:
        st.error("No se encontraron los archivos auxiliares en app/models/.")
        st.stop()


# Cargar datos
model = load_model()
equipos, team_values = load_auxiliary_data()

# Interfaz
st.title("⚽ Predicción de Resultados - Liga Argentina")
st.markdown("Selecciona dos equipos para predecir el resultado del partido")

# Selectores de equipos
col1, col2 = st.columns(2)

with col1:
    equipo_local = st.selectbox("Equipo Local", equipos, index=0)

with col2:
    equipo_visitante = st.selectbox("Equipo Visitante", equipos, index=1)

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
        'local_team_value_normalized': 0.5,
        'visitante_team_value_normalized': 0.5,
    }
    
    # Crear DataFrame
    df_pred = pd.DataFrame([default_values])
    
    # Hacer predicción
    try:
        prediccion = model.predict(df_pred)[0]
        probabilidades = model.predict_proba(df_pred)[0]
        
        # Mostrar resultado
        st.markdown("---")
        
        if prediccion == "Ganador local":
            st.success(f"🏆 {equipo_local} gana")
        elif prediccion == "Ganador visitante":
            st.success(f"🏆 {equipo_visitante} gana")
        else:
            st.info("🤝 Empate")
        
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

