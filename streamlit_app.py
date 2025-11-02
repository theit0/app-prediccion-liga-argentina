"""
App Streamlit para predecir resultados de partidos de la Liga Argentina
"""

import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(
    page_title="Predicción Liga Argentina",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        columns_to_drop = [col for col in self.columns_to_drop if col in X.columns]
        if columns_to_drop:
            return X.drop(columns=columns_to_drop)
        return X

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

model = load_model()
equipos = load_auxiliary_data()
team_urls = load_team_urls()

st.title("⚽ Predicción de Resultados - Liga Argentina")
st.markdown("Selecciona dos equipos para predecir el resultado del partido")

col1, col2 = st.columns(2)

with col1:
    st.write("Equipo Local")
    col_img1, col_select1 = st.columns([0.8, 4.2], gap="small")
    img_placeholder1 = col_img1.empty()
    with col_select1:
        equipo_local = st.selectbox("", equipos, index=0, key="select_local", label_visibility="collapsed")
    if equipo_local in team_urls:
        img_placeholder1.image(team_urls[equipo_local], width=40)

with col2:
    st.write("Equipo Visitante")
    col_img2, col_select2 = st.columns([0.8, 4.2], gap="small")
    img_placeholder2 = col_img2.empty()
    with col_select2:
        equipo_visitante = st.selectbox("", equipos, index=1, key="select_visitante", label_visibility="collapsed")
    if equipo_visitante in team_urls:
        img_placeholder2.image(team_urls[equipo_visitante], width=40)


if st.button("Predecir Resultado", type="primary", use_container_width=True):
    if equipo_local == equipo_visitante:
        st.error("Por favor selecciona dos equipos diferentes")
    else:
        fecha_actual = pd.Timestamp.now()
        default_values = {
            'fecha_del_partido': fecha_actual,
            'season': fecha_actual.year,
            'sub_season': 1,
            'fixture_id': 0,
            'round': 'Regular Season - 1',
            'Equipo_local_id': 0,
            'Equipo_local': equipo_local,
            'Equipo_visitante_id': 0,
            'Equipo_visitante': equipo_visitante,
            'Partidos_jugados_local_previos': 0,
            'Partidos_jugados_visitante_previos': 0,
            'Forma_local_ultimos5': '',
            'Forma_visitante_ultimos5': '',
            'Forma_local_puntos_ultimos5': 7,
            'Forma_visitante_puntos_ultimos5': 7,
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
            'local_team_value': 0.0,
            'visitante_team_value': 0.0,
            'local_team_url': team_urls.get(equipo_local, ''),
            'visitante_team_url': team_urls.get(equipo_visitante, ''),
            'local_team_value_normalized': 0.5,
            'visitante_team_value_normalized': 0.5,
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
            
        except Exception as e:
            st.error(f"Error al hacer la predicción: {str(e)}")
            st.exception(e)
