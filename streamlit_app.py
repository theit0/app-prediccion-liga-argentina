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

model = load_model()
equipos = load_auxiliary_data()
team_urls = load_team_urls()
df_data = load_data()

st.title("⚽ Predicción de Resultados - Liga Argentina")
st.markdown("Selecciona dos equipos y la fecha del partido para predecir el resultado")

fecha_partido = st.date_input("Fecha del partido", value=pd.Timestamp.now().date())

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
        fecha_actual = pd.to_datetime(fecha_partido, utc=True)
        
        # Determinar season y sub_season basado en la fecha
        df_fecha = df_data[df_data['fecha_del_partido'].notna()]
        df_fecha = df_fecha[df_fecha['fecha_del_partido'] <= fecha_actual]
        
        if df_fecha.empty:
            st.error("No hay datos históricos para la fecha seleccionada")
            st.stop()
        
        # Buscar la subseason más reciente para esa fecha
        df_fecha = df_fecha.sort_values('fecha_del_partido', ascending=False)
        ultima_subseason = df_fecha.iloc[0]
        season = ultima_subseason['season']
        sub_season = ultima_subseason['sub_season']
        
        # Buscar último partido del equipo local como local (primero en subseason actual, luego en anteriores)
        df_local = df_data[(df_data['Equipo_local'] == equipo_local) & 
                          (df_data['fecha_del_partido'] <= fecha_actual)]
        df_local = df_local.sort_values(['season', 'sub_season', 'fecha_del_partido'], ascending=[False, False, False])
        if not df_local.empty:
            # Filtrar por subseason actual primero, si no hay, tomar el más reciente de cualquier subseason
            df_local_actual = df_local[(df_local['season'] == season) & (df_local['sub_season'] == sub_season)]
            df_local = df_local_actual if not df_local_actual.empty else df_local.iloc[:1]
        
        # Buscar último partido del equipo visitante como visitante (primero en subseason actual, luego en anteriores)
        df_visitante = df_data[(df_data['Equipo_visitante'] == equipo_visitante) & 
                              (df_data['fecha_del_partido'] <= fecha_actual)]
        df_visitante = df_visitante.sort_values(['season', 'sub_season', 'fecha_del_partido'], ascending=[False, False, False])
        if not df_visitante.empty:
            # Filtrar por subseason actual primero, si no hay, tomar el más reciente de cualquier subseason
            df_visitante_actual = df_visitante[(df_visitante['season'] == season) & (df_visitante['sub_season'] == sub_season)]
            df_visitante = df_visitante_actual if not df_visitante_actual.empty else df_visitante.iloc[:1]
        
        # Construir valores usando los últimos partidos encontrados
        def get_val(series, col, default):
            return series[col] if col in series and pd.notna(series[col]) else default
        
        # Verificar si se usaron subseasons diferentes
        if not df_local.empty and not df_visitante.empty:
            partido_local = df_local.iloc[0]
            partido_visitante = df_visitante.iloc[0]
            
            local_season = partido_local['season']
            local_sub = partido_local['sub_season']
            visitante_season = partido_visitante['season']
            visitante_sub = partido_visitante['sub_season']
            
            if local_season != season or local_sub != sub_season or visitante_season != season or visitante_sub != sub_season:
                st.info(f"⚠️ Se encontraron partidos en subseasons anteriores: Local en {local_season}-{local_sub}, Visitante en {visitante_season}-{visitante_sub}")
            
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
        else:
            st.warning(f"No se encontraron partidos previos para uno o ambos equipos en la subseason {sub_season} de {season}. Usando valores por defecto.")
            default_values = {
                'fecha_del_partido': fecha_actual,
                'season': season,
                'sub_season': sub_season,
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
