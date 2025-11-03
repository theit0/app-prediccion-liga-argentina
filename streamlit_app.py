"""
App Streamlit para predecir resultados de partidos de la Liga Argentina
"""

import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Predicción Liga Argentina", page_icon="⚽", layout="centered", initial_sidebar_state="collapsed")

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        cols = [col for col in self.columns_to_drop if col in X.columns]
        return X.drop(columns=cols) if cols else X

@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except FileNotFoundError:
        st.error("No se encontró el archivo del modelo (model.pkl).")
        st.stop()

def load_single_model(file_path):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return joblib.load(file_path)
        except:
            try:
                return joblib.load(file_path, mmap_mode=None)
            except:
                return None

def load_all_models():
    models = {}
    model_files = {"Logistic Regression": "model.pkl", "Random Forest": "model-rf.pkl", "SVC": "model-svc.pkl", "HGB": "model-hgb.pkl"}
    failed = []
    for name, file in model_files.items():
        model = load_single_model(file)
        if model:
            models[name] = model
        else:
            failed.append(name)
    if failed:
        st.session_state['failed_models'] = failed
    return models

@st.cache_data
def load_auxiliary_data():
    try:
        return joblib.load("equipos.pkl")
    except FileNotFoundError:
        st.error("No se encontraron los archivos auxiliares.")
        st.stop()

@st.cache_data
def load_team_urls():
    try:
        df = pd.read_csv("liga_argentina_features_v3.csv", sep=';')
        urls = {}
        for col in ['Equipo_local', 'Equipo_visitante']:
            url_col = 'local_team_url' if col == 'Equipo_local' else 'visitante_team_url'
            for eq, url in zip(df[col], df[url_col]):
                if pd.notna(eq) and pd.notna(url) and url and eq not in urls:
                    urls[eq] = url
        return urls
    except:
        return {}

@st.cache_data
def load_cuotas():
    try:
        df = pd.read_csv("cuotas_2024.csv")
        return df[df['Cuota1'].notna() & df['CuotaX'].notna() & df['Cuota2'].notna()]
    except:
        return pd.DataFrame()

def get_default_features(equipo_local, equipo_visitante, fecha, team_urls):
    fecha = pd.to_datetime(fecha, format='%d.%m.%Y', errors='coerce') if isinstance(fecha, str) else fecha
    if pd.isna(fecha):
        fecha = pd.Timestamp.now()
    
    defaults = {
        'fecha_del_partido': fecha, 'season': fecha.year, 'sub_season': 1, 'fixture_id': 0,
        'round': 'Regular Season - 1', 'Equipo_local_id': 0, 'Equipo_local': equipo_local,
        'Equipo_visitante_id': 0, 'Equipo_visitante': equipo_visitante,
        'Partidos_jugados_local_previos': 0, 'Partidos_jugados_visitante_previos': 0,
        'Forma_local_ultimos5': '', 'Forma_visitante_ultimos5': '',
        'Forma_local_puntos_ultimos5': 7, 'Forma_visitante_puntos_ultimos5': 7,
        'Victorias_local_en_casa_tasa_normalizada': 0.5, 'Victorias_visitante_fuera_tasa_normalizada': 0.5,
        'Empates_local_en_casa_tasa_normalizada': 0.5, 'Empates_visitante_fuera_tasa_normalizada': 0.5,
        'Derrotas_local_en_casa_tasa_normalizada': 0.5, 'Derrotas_visitante_fuera_tasa_normalizada': 0.5,
        'Promedio_Goles_marcados_totales_local_normalizado': 0.5, 'Promedio_Goles_marcados_totales_visitante_normalizado': 0.5,
        'Promedio_Goles_recibidos_totales_local_normalizado': 0.5, 'Promedio_Goles_recibidos_totales_visitante_normalizado': 0.5,
        'Promedio_Goles_marcados_local_en_casa_normalizado': 0.5, 'Promedio_Goles_marcados_visitante_fuera_normalizado': 0.5,
        'Promedio_Goles_recibidos_local_en_casa_normalizado': 0.5, 'Promedio_Goles_recibidos_visitante_fuera_normalizado': 0.5,
        'Valla_invicta_local_tasa_normalizada': 0.5, 'Valla_invicta_visitante_tasa_normalizada': 0.5,
        'Promedio_Diferencia_gol_total_local_normalizado': 0.0, 'Promedio_Diferencia_gol_total_visitante_normalizado': 0.0,
        'Promedio_Puntuacion_total_local_normalizado': 0.5, 'Promedio_Puntuacion_total_visitante_normalizado': 0.5,
        'local_team_value': 0.0, 'visitante_team_value': 0.0,
        'local_team_url': team_urls.get(equipo_local, ''), 'visitante_team_url': team_urls.get(equipo_visitante, ''),
        'local_team_value_normalized': 0.5, 'visitante_team_value_normalized': 0.5,
    }
    
    df = pd.DataFrame([defaults])
    df['fecha_del_partido'] = pd.to_datetime(df['fecha_del_partido'], errors='coerce', utc=True)
    df['partido_anio'] = df['fecha_del_partido'].dt.year
    df['partido_mes'] = df['fecha_del_partido'].dt.month
    df['partido_dia_semana'] = df['fecha_del_partido'].dt.dayofweek
    return df

def parse_resultado(resultado_str):
    try:
        g_local, g_visitante = map(int, str(resultado_str).split(':'))
        if g_local > g_visitante:
            return "Ganador local"
        elif g_visitante > g_local:
            return "Ganador visitante"
        return "Empate"
    except:
        return None

def simulate_betting(models, cuotas_df, initial_balance, team_urls):
    resultados = {name: {"balance": initial_balance, "apuestas": [], "errores": [], "historial": [initial_balance]} for name in models.keys()}
    
    for _, row in cuotas_df.iterrows():
        resultado_real = parse_resultado(str(row['Resultado']))
        if not resultado_real:
            continue
        
        equipo_local, equipo_visitante = row['Equipo1'], row['Equipo2']
        cuotas = {'Ganador local': float(row['Cuota1']), 'Empate': float(row['CuotaX']), 'Ganador visitante': float(row['Cuota2'])}
        df_features = get_default_features(equipo_local, equipo_visitante, str(row['Fecha']), team_urls)
        
        for model_name, model in models.items():
            if resultados[model_name]["balance"] <= 0:
                resultados[model_name]["historial"].append(resultados[model_name]["balance"])
                continue
            try:
                prediccion = model.predict(df_features)[0]
                monto = max(10, resultados[model_name]["balance"] * 0.05)
                cuota = cuotas[prediccion]
                
                if prediccion == resultado_real:
                    resultados[model_name]["balance"] += monto * (cuota - 1)
                    ganado = True
                else:
                    resultados[model_name]["balance"] -= monto
                    ganado = False
                
                resultados[model_name]["historial"].append(resultados[model_name]["balance"])
                resultados[model_name]["apuestas"].append({
                    "partido": f"{equipo_local} vs {equipo_visitante}",
                    "prediccion": prediccion, "resultado_real": resultado_real,
                    "apuesta": monto, "cuota": cuota, "ganado": ganado,
                    "balance_actual": resultados[model_name]["balance"]
                })
            except Exception as e:
                resultados[model_name]["historial"].append(resultados[model_name]["balance"])
                resultados[model_name]['errores'].append({
                    "partido": f"{equipo_local} vs {equipo_visitante}",
                    "error": str(e)[:100]
                })
    
    return resultados

equipos = load_auxiliary_data()
team_urls = load_team_urls()
tab1, tab2 = st.tabs(["Predecir", "Modelos"])

with tab1:
    model = load_model()
    st.title("⚽ Predicción de Resultados - Liga Argentina")
    st.markdown("Selecciona dos equipos para predecir el resultado del partido")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Equipo Local")
        col_img1, col_select1 = st.columns([0.8, 4.2], gap="small")
        img_placeholder1 = col_img1.empty()
        equipo_local = st.selectbox("", equipos, index=0, key="select_local", label_visibility="collapsed")
        if equipo_local in team_urls:
            img_placeholder1.image(team_urls[equipo_local], width=40)
    
    with col2:
        st.write("Equipo Visitante")
        col_img2, col_select2 = st.columns([0.8, 4.2], gap="small")
        img_placeholder2 = col_img2.empty()
        equipo_visitante = st.selectbox("", equipos, index=1, key="select_visitante", label_visibility="collapsed")
        if equipo_visitante in team_urls:
            img_placeholder2.image(team_urls[equipo_visitante], width=40)
    
    if st.button("Predecir Resultado", type="primary", use_container_width=True, key="predict_button"):
        if equipo_local == equipo_visitante:
            st.error("Por favor selecciona dos equipos diferentes")
        else:
            df_pred = get_default_features(equipo_local, equipo_visitante, pd.Timestamp.now(), team_urls)
            try:
                prediccion = model.predict(df_pred)[0]
                probabilidades = model.predict_proba(df_pred)[0]
                
                st.markdown("---")
                if prediccion == "Ganador local":
                    img = f'<img src="{team_urls.get(equipo_local, "")}" width="40" style="vertical-align: middle; margin-right: 10px;">' if equipo_local in team_urls else ""
                    st.markdown(f'<div style="margin-bottom: 20px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 0.5rem; padding: .5rem; display: flex; justify-content: center; align-items: center; gap: .5rem;">{img}<span><strong>{equipo_local}</strong> gana</span></div>', unsafe_allow_html=True)
                elif prediccion == "Ganador visitante":
                    img = f'<img src="{team_urls.get(equipo_visitante, "")}" width="40" style="vertical-align: middle; margin-right: 10px;">' if equipo_visitante in team_urls else ""
                    st.markdown(f'<div style="margin-bottom: 20px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 0.5rem; padding: .5rem; display: flex; justify-content: center; align-items: center; gap: .5rem;">{img}<span><strong>{equipo_visitante}</strong> gana</span></div>', unsafe_allow_html=True)
                else:
                    st.info("🤝 **Empate**")
                
                prob_df = pd.DataFrame({'Resultado': model.named_steps['model'].classes_, 'Probabilidad': probabilidades})
                for _, row in prob_df.iterrows():
                    st.progress(float(row['Probabilidad']), text=f"{row['Resultado']}: {row['Probabilidad']*100:.1f}%")
            except Exception as e:
                st.error(f"Error al hacer la predicción: {str(e)}")
                st.exception(e)

with tab2:
    st.title("🎮 Comparación de Modelos")
    st.markdown("Simula apuestas con diferentes modelos y compara su rendimiento")
    
    models = load_all_models()
    
    if 'failed_models' in st.session_state and st.session_state['failed_models']:
        st.warning(f"⚠️ Los siguientes modelos no se pudieron cargar: {', '.join(st.session_state['failed_models'])}. "
                   f"Esto puede deberse a incompatibilidades de versiones entre numpy/joblib.")
        del st.session_state['failed_models']
    
    if len(models) == 0:
        st.error("No se pudieron cargar los modelos.")
        st.info("💡 Sugerencia: `pip install --upgrade numpy joblib`")
    else:
        st.success(f"✅ {len(models)} modelo(s) cargado(s): {', '.join(models.keys())}")
        initial_balance = st.number_input("Saldo inicial por modelo", min_value=100, value=1000, step=100, key="initial_balance")
        
        if st.button("🚀 Iniciar Simulación", type="primary", use_container_width=True):
            cuotas_df = load_cuotas()
            if cuotas_df.empty:
                st.error("No se pudieron cargar las cuotas.")
            else:
                st.info(f"Simulando apuestas para {len(cuotas_df)} partidos...")
                with st.spinner("Calculando resultados..."):
                    resultados = simulate_betting(models, cuotas_df, initial_balance, team_urls)
                
                st.markdown("---")
                st.subheader("💰 Resultados Finales")
                
                resultados_df = pd.DataFrame([{
                    "Modelo": name,
                    "Saldo Final": f"${resultados[name]['balance']:.2f}",
                    "Ganancia/Pérdida": f"${resultados[name]['balance'] - initial_balance:.2f}",
                    "Retorno %": f"{((resultados[name]['balance'] - initial_balance) / initial_balance) * 100:.2f}%",
                    "Apuestas Totales": len(resultados[name]['apuestas'])
                } for name in resultados.keys()])
                
                resultados_df = resultados_df.sort_values('Saldo Final', key=lambda x: x.str.replace('$', '').astype(float), ascending=False)
                st.dataframe(resultados_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.subheader("📈 Evolución del Balance")
                model_names = list(resultados.keys())
                try:
                    import plotly.graph_objects as go
                    cols = st.columns(3)
                    for i in range(min(3, len(model_names))):
                        with cols[i]:
                            model_name = model_names[i]
                            historial = resultados[model_name]["historial"]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=historial,
                                mode='lines+markers',
                                name=model_name,
                                line=dict(color='green' if historial[-1] >= initial_balance else 'red', width=2),
                                marker=dict(size=4)
                            ))
                            fig.add_hline(y=initial_balance, line_dash="dash", line_color="gray", 
                                        annotation_text=f"Saldo inicial: ${initial_balance}")
                            fig.update_layout(
                                title=model_name,
                                xaxis_title="Partido",
                                yaxis_title="Balance ($)",
                                height=300,
                                showlegend=False,
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("💡 Para gráficos interactivos: `pip install plotly`")
                
                st.markdown("---")
                st.subheader("📊 Comparación Visual")
                try:
                    import plotly.express as px
                    fig = px.bar(x=list(resultados.keys()), y=[resultados[n]['balance'] for n in resultados.keys()],
                                labels={'x': 'Modelo', 'y': 'Saldo Final ($)'},
                                color=[resultados[n]['balance'] for n in resultados.keys()], color_continuous_scale='RdYlGn')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("💡 Para gráficos interactivos: `pip install plotly`")
                    balance_data = {name: resultados[name]['balance'] for name in resultados.keys()}
                    max_b, min_b = max(balance_data.values()), min(balance_data.values())
                    for name in sorted(balance_data, key=balance_data.get, reverse=True):
                        bar_width = int((balance_data[name] - min_b) / (max_b - min_b) * 100) if max_b != min_b else 100
                        color = "🟢" if balance_data[name] >= initial_balance else "🔴"
                        st.markdown(f"{color} **{name}**: ${balance_data[name]:.2f} {'█' * bar_width}")
                
                st.markdown("---")
                st.subheader("📋 Detalles por Modelo")
                for model_name in resultados.keys():
                    with st.expander(f"{model_name} - ${resultados[model_name]['balance']:.2f}"):
                        apuestas = resultados[model_name].get('apuestas', [])
                        errores = resultados[model_name].get('errores', [])
                        
                        if errores:
                            st.warning(f"⚠️ Errores en {len(errores)} partido(s).")
                            if errores:
                                st.code(f"Error: {errores[0]['error']}\nPartido: {errores[0]['partido']}")
                        
                        if apuestas:
                            ap_df = pd.DataFrame(apuestas)[['partido', 'prediccion', 'resultado_real', 'apuesta', 'cuota', 'ganado', 'balance_actual']]
                            ap_df['apuesta'] = ap_df['apuesta'].apply(lambda x: f"${x:.2f}")
                            ap_df['balance_actual'] = ap_df['balance_actual'].apply(lambda x: f"${x:.2f}")
                            ap_df['ganado'] = ap_df['ganado'].apply(lambda x: "✅" if x else "❌")
                            ap_df.columns = ['Partido', 'Predicción', 'Resultado Real', 'Apuesta', 'Cuota', 'Resultado', 'Balance Actual']
                            st.dataframe(ap_df, use_container_width=True, hide_index=True)
                            
                            ganadas = sum(1 for a in apuestas if a['ganado'])
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Apuestas Ganadas", ganadas)
                            col2.metric("Apuestas Perdidas", len(apuestas) - ganadas)
                            col3.metric("Tasa de Acierto", f"{(ganadas / len(apuestas) * 100):.1f}%")
                        else:
                            st.error("❌ No se pudo realizar ninguna apuesta.") if errores else st.info("No se realizaron apuestas.")
