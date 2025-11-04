"""
App Streamlit para predecir resultados de partidos de la Liga Argentina
"""

import streamlit as st
import pandas as pd
import joblib
from datetime import date
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
    col_img1, col_select1 = st.columns([0.8, 4.2], gap="small")
    img_placeholder1 = col_img1.empty()
    with col_select1:
        equipo_local = st.selectbox("", equipos_disponibles, 
                                   index=equipos_disponibles.index(st.session_state.equipo_local) if st.session_state.equipo_local in equipos_disponibles else 0,
                                   key="select_local", label_visibility="collapsed")
        st.session_state.equipo_local = equipo_local
    if equipo_local in team_urls:
        img_placeholder1.image(team_urls[equipo_local], width=40)

with col2:
    st.write("")
    st.write("")
    if st.button("⇄", key="switch_teams", use_container_width=True, help="Intercambiar equipos"):
        st.session_state.equipo_local, st.session_state.equipo_visitante = st.session_state.equipo_visitante, st.session_state.equipo_local
        st.rerun()

with col3:
    st.write("Equipo Visitante")
    col_img2, col_select2 = st.columns([0.8, 4.2], gap="small")
    img_placeholder2 = col_img2.empty()
    with col_select2:
        equipo_visitante = st.selectbox("", equipos_disponibles,
                                       index=equipos_disponibles.index(st.session_state.equipo_visitante) if st.session_state.equipo_visitante in equipos_disponibles else 0,
                                       key="select_visitante", label_visibility="collapsed")
        st.session_state.equipo_visitante = equipo_visitante
    if equipo_visitante in team_urls:
        img_placeholder2.image(team_urls[equipo_visitante], width=40)


if st.button("Predecir Resultado", type="primary", use_container_width=True):
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
