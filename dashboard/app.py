import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="ICFES Dashboard", layout="wide")
st.title("ICFES – Dashboard de Estadísticas y Recomendaciones")

# ------------------------- Utilidades API -------------------------

def api_get(path: str) -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=15)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_delete(path: str) -> Dict[str, Any]:
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ------------------------- Sidebar (Filtros / Input) -------------------------
with st.sidebar:
    st.header("Entrada de Puntajes")
    st.caption("Ingresa puntajes para generar recomendaciones.")
    id_student = st.text_input("ID estudiante (opcional)")
    punt_global = st.number_input("Puntaje Global", min_value=0, max_value=300, value=0, step=1, format="%d")
    punt_matematicas = st.number_input("Matemáticas", min_value=0, max_value=300, value=0, step=1, format="%d")
    punt_lectura_critica = st.number_input("Lectura Crítica", min_value=0, max_value=300, value=0, step=1, format="%d")
    punt_ingles = st.number_input("Inglés", min_value=0, max_value=300, value=0, step=1, format="%d")

    st.markdown("---")
    do_predict = st.button("Generar Recomendaciones")

# ------------------------- Layout principal -------------------------
col_left, col_right = st.columns([1.25, 1])

# ------------------------- Columna Izquierda -------------------------
with col_left:
    st.subheader("Perfil y Recomendaciones")
    if do_predict:
        payload = {
            "id_student": id_student or None,
            "punt_global": int(punt_global),
            "punt_matematicas": int(punt_matematicas),
            "punt_lectura_critica": int(punt_lectura_critica),
            "punt_ingles": int(punt_ingles),
        }
        resp = api_post("/predict", payload)
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.success("Predicción generada correctamente")
            st.write(f"ID estudiante: {resp.get('id_student', id_student or '—')}")
            cluster_id = resp.get("cluster_id")  # Se usa internamente pero no se muestra
            st.write(f"Fortaleza principal: {resp.get('main_strength','N/D')}")
            st.write("Carreras sugeridas:")
            for c in resp.get("recommended_careers", []):
                st.markdown(f"- {c}")
            st.write("Refuerzos sugeridos:")
            for r in resp.get("reinforcement_suggestions", []):
                st.markdown(f"- {r}")

            # ------------------------- Radar Comparativo (Global Histórico) -------------------------
            hist_all = api_get("/history")
            if isinstance(hist_all, list) and hist_all:
                df_hist_all = pd.DataFrame(hist_all)
                if "puntajes" in df_hist_all.columns:
                    punt_cols_all = df_hist_all["puntajes"].apply(pd.Series)
                    punt_cols_all = punt_cols_all.loc[:, [c for c in punt_cols_all.columns if c not in df_hist_all.columns]]
                    df_hist_all = pd.concat([df_hist_all.drop(columns=["puntajes"]), punt_cols_all], axis=1)
                score_cols_radar = [c for c in ["punt_global","punt_matematicas","punt_lectura_critica","punt_ingles"] if c in df_hist_all.columns]
                if score_cols_radar:
                    global_means = df_hist_all[score_cols_radar].mean()
                    student_scores = [payload.get(col, 0) for col in score_cols_radar]
                    categories = [
                        "Global" if c == "punt_global" else
                        "Matemáticas" if c == "punt_matematicas" else
                        "Lectura Crítica" if c == "punt_lectura_critica" else
                        "Inglés" if c == "punt_ingles" else c
                        for c in score_cols_radar
                    ]
                    st.markdown("---")
                    st.write("### Comparación Estudiante vs Promedio Histórico")
                    fig_radar = go.Figure()
                    # Colores personalizados: Estudiante (rojo) vs Promedio (azul)
                    fig_radar.add_trace(go.Scatterpolar(
                        r=student_scores,
                        theta=categories,
                        fill='toself',
                        name='Estudiante',
                        line_color='#FF4B4B',
                        fillcolor='rgba(255,75,75,0.40)'
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=global_means.values,
                        theta=categories,
                        fill='toself',
                        name='Promedio Histórico',
                        line_color='#1f77b4',
                        fillcolor='rgba(31,119,180,0.30)'
                    ))
                    max_axis = max(300, int(max(student_scores + list(global_means.values))))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max_axis])), showlegend=True)
                    st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.caption("Historial vacío; genera más consultas para ver comparación.")


    # ------------------------- Estadísticas -------------------------
    st.markdown("---")
    st.subheader("Estadísticas del Historial")
    hist_raw = api_get("/history")
    if isinstance(hist_raw, list) and hist_raw:
        df_hist_full = pd.DataFrame(hist_raw)
        # Expandir puntajes
        if "puntajes" in df_hist_full.columns:
            punt_cols_full = df_hist_full["puntajes"].apply(pd.Series)
            punt_cols_full = punt_cols_full.loc[:, [c for c in punt_cols_full.columns if c not in df_hist_full.columns]]
            df_hist_full = pd.concat([df_hist_full.drop(columns=["puntajes"]), punt_cols_full], axis=1)

        # Filtro simple por rango de puntaje global
        col_r1, col_r2 = st.columns(2)
        if "punt_global" in df_hist_full.columns:
            min_pg, max_pg = int(df_hist_full["punt_global"].min()), int(df_hist_full["punt_global"].max())
            if min_pg == max_pg:
                col_r2.caption(f"Rango Puntaje Global (único valor: {min_pg})")
                range_pg = (min_pg, max_pg)
            else:
                range_pg = col_r2.slider("Rango Puntaje Global", min_pg, max_pg, (min_pg, max_pg))
        else:
            range_pg = (0, 9999)
        df_filt = df_hist_full.copy()
        if "punt_global" in df_filt.columns:
            df_filt = df_filt[(df_filt["punt_global"] >= range_pg[0]) & (df_filt["punt_global"] <= range_pg[1])]
        st.caption(f"Registros filtrados: {len(df_filt)}")

        # Estadísticas generales (sin segmentación por clúster)
        score_cols = [c for c in ["punt_global","punt_matematicas","punt_lectura_critica","punt_ingles"] if c in df_filt.columns]
        if score_cols:
            st.write("Promedios históricos de puntajes:")
            st.dataframe(pd.DataFrame(df_filt[score_cols].mean().round(1), columns=["promedio"]))

        # KPIs generales
        total_consultas = len(df_filt)
        col_kpi1, col_kpi2 = st.columns(2)
        col_kpi1.metric("Consultas totales", total_consultas)
        if score_cols:
            col_kpi2.metric("Materias analizadas", len(score_cols))

        # Distribución promedio general
        if score_cols:
            avg_scores = df_filt[score_cols].mean().round(1)
            fig_avg = go.Figure()
            fig_avg.add_trace(go.Bar(x=avg_scores.index, y=avg_scores.values, text=avg_scores.values, textposition="outside"))
            fig_avg.update_layout(title="Promedio General de Puntajes (Historial Filtrado)")
            st.plotly_chart(fig_avg, use_container_width=True)
    else:
        st.info("Sin datos en el historial para generar estadísticas.")

    st.markdown("---")
    st.subheader("Historial (últimas 50 consultas)")
    hist = api_get("/history")
    if isinstance(hist, list) and hist:
        df_hist = pd.DataFrame(hist)
        if "puntajes" in df_hist.columns:
            punt_cols = df_hist["puntajes"].apply(pd.Series)
            punt_cols = punt_cols.loc[:, [c for c in punt_cols.columns if c not in df_hist.columns]]
            df_hist = pd.concat([df_hist.drop(columns=["puntajes"]), punt_cols], axis=1)
        st.dataframe(df_hist.tail(50), use_container_width=True)
    else:
        st.caption("Sin historial disponible.")

# ------------------------- Columna Derecha -------------------------
with col_right:
    st.subheader("Búsqueda por Estudiante")
    search_id = st.text_input("ID para historial individual")
    if st.button("Buscar historial"):
        if not search_id:
            st.warning("Ingresa un ID válido.")
        else:
            res = api_get(f"/student/{search_id}")
            if isinstance(res, list) and res:
                df_stu = pd.DataFrame(res)
                if "puntajes" in df_stu.columns:
                    punt_cols = df_stu["puntajes"].apply(pd.Series)
                    punt_cols = punt_cols.loc[:, [c for c in punt_cols.columns if c not in df_stu.columns]]
                    df_stu = pd.concat([df_stu.drop(columns=["puntajes"]), punt_cols], axis=1)
                st.write(f"Consultas registradas: {len(df_stu)}")
                st.dataframe(df_stu.tail(20), use_container_width=True)
            else:
                st.info("No hay registros para ese estudiante.")

    st.markdown("---")
    st.subheader("Mantenimiento")
    if st.button("Limpiar historial"):
        r = api_delete("/clear-history")
        if r.get("message"):
            st.success(r["message"])
        else:
            st.error(r.get("error", "Error desconocido"))

    st.markdown("---")
    st.caption(f"API base: {API_BASE}")
    health = api_get("/health")
    if health.get("status") == "ok":
        st.success("API OK")
    else:
        st.error("API no disponible")

st.markdown("---")
st.caption("Sprint 4 – Dashboard interactivo para recomendaciones ICFES")
