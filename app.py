import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="IA - Detección de Barcos UAV", 
    page_icon="🚁", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
    <style>
    .stMetric {
        background-color: #ffffff;
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .card-blind {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 12px;
        background-color: #fafafa;
    }
    .error-card {
        padding: 12px;
        border-radius: 8px;
        background-color: #fff3cd;
        border-left: 5px solid #dc3545;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚁 Sistema de Percepción Autónoma - UAV")
st.markdown("### Clasificación binaria en vivo (Test Set ciego sin sesgo de confirmación)")
st.markdown("---")

# 1. Cargar el modelo
@st.cache_resource
def cargar_modelo():
    return load_model('mejor_modelo_barcos.h5')

try:
    model = cargar_modelo()
    st.sidebar.success("✅ Modelo cargado con éxito.")
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar 'mejor_modelo_barcos.h5': {e}")
    st.stop()

# 2. Panel lateral
st.sidebar.header("📂 1. Carga de Imágenes")
uploaded_files = st.sidebar.file_uploader(
    "Sube el conjunto de prueba (imágenes sueltas o lote):", 
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True
)

threshold = st.sidebar.slider("Umbral de Decisión (Threshold)", 0.0, 1.0, 0.50, 0.01)

# Inicializar estados en session_state
if 'ground_truth' not in st.session_state:
    st.session_state['ground_truth'] = {}
if 'inferencia_ejecutada' not in st.session_state:
    st.session_state['inferencia_ejecutada'] = False

if uploaded_files:
    n_total = len(uploaded_files)
    st.sidebar.info(f"Total imágenes cargadas: {n_total}")
    
    # Procesar imágenes una sola vez
    imagenes_pil = []
    imagenes_array = []
    nombres_archivos = []
    
    for f in uploaded_files:
        img = Image.open(f).convert('RGB')
        img_res = img.resize((80, 80))
        imagenes_pil.append(img)
        imagenes_array.append(np.array(img_res))
        nombres_archivos.append(f.name)
        
    X_eval = np.array(imagenes_array, dtype=np.float32) / 255.0

    # Pestañas de la aplicación
    tab1, tab2, tab3 = st.tabs([
        "🏷️ 1. Etiquetado Ciego (Ground Truth)", 
        "🤖 2. Resultados de Inferencia y Métricas",
        "🚨 3. Auditoría Visual de Errores"
    ])

    # =========================================================
    # TAB 1: ETIQUETADO CIEGO (Sin ver predicciones de la IA)
    # =========================================================
    with tab1:
        st.subheader("Etiquetado Ciego de Imágenes")
        st.caption("Asigna la etiqueta real de cada muestra sin influencia de la red neuronal para evitar sesgo de evaluación.")
        
        # Conteo de avance
        etiquetadas_count = sum(1 for v in st.session_state['ground_truth'].values() if v is not None)
        st.progress(etiquetadas_count / n_total)
        st.write(f"**Progreso de etiquetado:** {etiquetadas_count} de {n_total} imágenes.")

        # Rejilla de 4 columnas para etiquetar
        cols_per_row = 4
        rows = [range(i, min(i + cols_per_row, n_total)) for i in range(0, n_total, cols_per_row)]
        
        for row in rows:
            cols = st.columns(cols_per_row)
            for col_idx, idx in enumerate(row):
                with cols[col_idx]:
                    st.markdown("<div class='card-blind'>", unsafe_allow_html=True)
                    st.image(imagenes_pil[idx], width=130)
                    st.caption(f"📁 `{nombres_archivos[idx]}`")
                    
                    b1, b0 = st.columns(2)
                    with b1:
                        if st.button("🚢 Barco", key=f"gt_b1_{idx}"):
                            st.session_state['ground_truth'][idx] = 1
                            st.session_state['inferencia_ejecutada'] = False
                    with b0:
                        if st.button("🌊 No Barco", key=f"gt_b0_{idx}"):
                            st.session_state['ground_truth'][idx] = 0
                            st.session_state['inferencia_ejecutada'] = False
                            
                    estado = st.session_state['ground_truth'].get(idx, None)
                    if estado == 1:
                        st.success("Asignado: **Barco (1)**")
                    elif estado == 0:
                        st.info("Asignado: **No Barco (0)**")
                    else:
                        st.warning("⚠️ Sin etiquetar")
                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        # Botón para disparar la inferencia
        col_btn, _ = st.columns([2, 3])
        with col_btn:
            if st.button("🚀 Ejecutar Inferencia y Evaluar Modelo", type="primary", use_container_width=True):
                if etiquetadas_count < n_total:
                    st.error(f"Faltan {n_total - etiquetadas_count} imágenes por etiquetar. Completa el lote para calcular métricas válidas.")
                else:
                    st.session_state['inferencia_ejecutada'] = True
                    st.success("¡Inferencia completada! Dirígete a la pestaña '2. Resultados de Inferencia y Métricas'.")

    # =========================================================
    # TAB 2: INFERENCIA Y CONSOLIDADO DE MÉTRICAS
    # =========================================================
    with tab2:
        if not st.session_state['inferencia_ejecutada']:
            st.info("👈 Primero completa el etiquetado ciego en la Pestaña 1 y presiona **'Ejecutar Inferencia y Evaluar Modelo'**.")
        else:
            # Calcular inferencias
            y_probs = model.predict(X_eval, verbose=0).flatten()
            y_pred = (y_probs >= threshold).astype(int)
            y_true = np.array([st.session_state['ground_truth'][i] for i in range(n_total)])

            # Métricas
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            st.subheader("📊 Métricas de Evaluación Oficial (ABET SO6 / RAE-144)")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy en Vivo", f"{acc*100:.2f}%", delta=f"{acc*100 - 98.0:+.2f}% vs Meta (98%)")
            m2.metric("Precision", f"{prec:.4f}")
            m3.metric("Recall", f"{rec:.4f}")
            m4.metric("F1-Score", f"{f1:.4f}")

            # Penalización ABET
            if acc < 0.98:
                penalizacion = ((0.98 - acc) / 0.02) * 0.5
                st.error(f"⚠️ Penalización estimada por rúbrica (-0.5 por cada 2% debajo de 98%): **-{penalizacion:.2f} puntos**.")
            else:
                st.success("🎉 ¡Excelente! Se alcanzó o superó el 98% de accuracy en el conjunto de prueba.")

            col_mat, col_gal = st.columns([1, 1.4])
            
            with col_mat:
                st.markdown("#### Matriz de Confusión Oficial")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3.2))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['No Barco', 'Barco'], 
                            yticklabels=['No Barco', 'Barco'],
                            annot_kws={"size": 14})
                plt.ylabel('Real (Ground Truth)')
                plt.xlabel('Predicción (IA)')
                st.pyplot(fig)

            with col_gal:
                st.markdown("#### Desglose de Inferencia")
                st.write(f"• **Verdaderos Negativos (TN):** {cm[0, 0]}")
                st.write(f"• **Falsos Positivos (FP):** {cm[0, 1]}")
                st.write(f"• **Falsos Negativos (FN):** {cm[1, 0]}")
                st.write(f"• **Verdaderos Positivos (TP):** {cm[1, 1]}")
                st.write(f"• **Umbral aplicado:** {threshold:.2f}")

    # =========================================================
    # TAB 3: AUDITORÍA VISUAL DE ERRORES
    # =========================================================
    with tab3:
        if not st.session_state['inferencia_ejecutada']:
            st.info("Ejecuta la inferencia en la Pestaña 1 para auditar los fallos.")
        else:
            y_probs = model.predict(X_eval, verbose=0).flatten()
            y_pred = (y_probs >= threshold).astype(int)
            y_true = np.array([st.session_state['ground_truth'][i] for i in range(n_total)])
            
            indices_error = np.where(y_true != y_pred)[0]
            
            if len(indices_error) == 0:
                st.balloons()
                st.success("✨ ¡Cero discrepancias! El modelo acertó el 100% de las muestras de este lote.")
            else:
                st.warning(f"Se identificaron **{len(indices_error)} fallos** de clasificación:")
                
                err_cols = st.columns(3)
                for i_pos, idx_err in enumerate(indices_error):
                    with err_cols[i_pos % 3]:
                        st.markdown("<div class='error-card'>", unsafe_allow_html=True)
                        st.image(imagenes_pil[idx_err], width=130)
                        st.write(f"📁 `{nombres_archivos[idx_err]}`")
                        st.write(f"**Real (Humano):** {'Barco (1)' if y_true[idx_err]==1 else 'No Barco (0)'}")
                        st.write(f"**Predicción IA:** {'Barco (1)' if y_pred[idx_err]==1 else 'No Barco (0)'}")
                        st.write(f"**Confianza:** {y_probs[idx_err]*100:.2f}%")
                        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👈 Carga las imágenes de prueba desde la barra lateral izquierda para iniciar la evaluación a ciegas.")