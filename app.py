import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Configuración de la página web
st.set_page_config(
    page_title="IA - Detección de Barcos UAV", 
    page_icon="🚁", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para darle un toque más estético y profesional ("más bonita")
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .error-card {
        padding: 10px;
        border-radius: 8px;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚁 Sistema de Percepción Autónoma - Vehículo Aéreo No Tripulado (UAV)")
st.markdown("### Clasificación binaria de imágenes aéreas satelitales (80x80 px) para inspección portuaria en vivo.")
st.markdown("---")

# 1. Cargar el modelo entrenado previamente
@st.cache_resource
def cargar_modelo():
    return load_model('mejor_modelo_barcos.h5')

try:
    model = cargar_modelo()
    st.sidebar.success("✅ Modelo 'mejor_modelo_barcos.h5' cargado correctamente.")
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar el modelo: Asegúrate de tener 'mejor_modelo_barcos.h5' en la misma carpeta.\nDetalle: {e}")

# 2. Panel lateral para carga de imágenes (Soporta múltiples archivos y carpetas seleccionadas)
st.sidebar.header("📂 Panel de Carga de Test Set")
st.sidebar.markdown("Sube las imágenes de prueba (puedes seleccionar varias o una carpeta completa):")
uploaded_files = st.sidebar.file_uploader(
    "Selecciona las imágenes:", 
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True
)

# Umbral de decisión ajustable
threshold = st.sidebar.slider("Umbral de Decisión (Threshold)", 0.0, 1.0, 0.5, 0.01)

if uploaded_files:
    st.sidebar.info(f"Imágenes cargadas en memoria: {len(uploaded_files)}")
    
    # Procesar imágenes para la IA
    imagenes_cargadas = []
    nombres_archivos = []
    
    for file in uploaded_files:
        img = Image.open(file).convert('RGB')
        img_resized = img.resize((80, 80)) # Obligatorio: 80x80 píxeles especificación técnica
        imagenes_cargadas.append(np.array(img_resized))
        nombres_archivos.append(file.name)
        
    X_test = np.array(imagenes_cargadas) / 255.0
    
    # Ejecutar inferencia de la IA
    predicciones_prob = model.predict(X_test)
    predicciones_clases = (predicciones_prob > threshold).astype(int).flatten()
    
    # Inicializar el estado de sesión para guardar las etiquetas reales mediante botones de forma persistente
    if 'etiquetas_reales' not in st.session_state or len(st.session_state['etiquetas_reales']) != len(uploaded_files):
        st.session_state['etiquetas_reales'] = {i: None for i in range(len(uploaded_files))}

    # Crear Pestañas (Tabs) para ordenar la interfaz de manera elegante
    tab1, tab2 = st.tabs(["🔍 Galería de Inferencia y Etiquetado", "🚨 Análisis Detallado de Fallos"])
    
    with tab1:
        st.markdown("### 🖼️ Panel de Validación Interactiva en Vivo")
        st.markdown("Haz clic en los botones de **'Es Barco'** o **'No es Barco'** debajo de cada imagen para registrar el valor real (Ground Truth) de manera ágil.")
        
        cols_per_row = 4
        rows = [range(i, min(i + cols_per_row, len(uploaded_files))) for i in range(0, len(uploaded_files), cols_per_row)]
        
        for row in rows:
            cols = st.columns(cols_per_row)
            for col_idx, idx in enumerate(row):
                with cols[col_idx]:
                    with st.container():
                        img_arr = imagenes_cargadas[idx]
                        nombre = nombres_archivos[idx]
                        pred_ia = predicciones_clases[idx]
                        confianza = predicciones_prob[idx][0]
                        
                        st.image(img_arr, width=140)
                        st.caption(f"📁 `{nombre}`")
                        
                        if pred_ia == 1:
                            st.markdown("🤖 IA: <span style='color:green; font-weight:bold;'>BARCO 🚢</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("🤖 IA: <span style='color:blue; font-weight:bold;'>NO BARCO 🌊</span>", unsafe_allow_html=True)
                            
                        st.text(f"Confianza: {confianza*100:.1f}%")
                        
                        # Botones rápidos de etiquetado en lugar de selectbox
                        b_col1, b_col2 = st.columns(2)
                        with b_col1:
                            if st.button("🚢 Barco", key=f"b1_{idx}"):
                                st.session_state['etiquetas_reales'][idx] = 1
                        with b_col2:
                            if st.button("🌊 No Barco", key=f"b0_{idx}"):
                                st.session_state['etiquetas_reales'][idx] = 0
                                
                        # Mostrar estado actual del etiquetado
                        current_label = st.session_state['etiquetas_reales'][idx]
                        if current_label is not None:
                            if current_label == pred_ia:
                                st.success("✅ Acierto en vivo")
                            else:
                                st.error("❌ Fallo detectado")
                        else:
                            st.info("⚠️ Pendiente")
                        st.markdown("---")

    # Recopilar datos etiquetados
    y_true_list = []
    y_pred_list = []
    indices_etiquetados = []
    
    for idx in range(len(uploaded_files)):
        val = st.session_state['etiquetas_reales'][idx]
        if val is not None:
            y_true_list.append(val)
            y_pred_list.append(predicciones_clases[idx])
            indices_etiquetados.append(idx)

    # ==========================================
    # 3. CONSOLIDACIÓN DE MÉTRICAS Y PESTAÑA DE FALLOS
    # ==========================================
    with tab2:
        st.markdown("### 🔍 Auditoría Visual de Errores (Falsos Positivos / Negativos)")
        if len(y_true_list) > 0:
            y_true_arr = np.array(y_true_list)
            y_pred_arr = np.array(y_pred_list)
            
            # Filtrar los índices donde hubo errores reales
            errores_locales = [indices_etiquetados[i] for i in range(len(y_true_arr)) if y_true_arr[i] != y_pred_arr[i]]
            
            if len(errores_locales) == 0:
                st.balloons()
                st.success("✨ ¡Extraordinario! No se han registrado fallos en las imágenes etiquetadas hasta el momento.")
            else:
                st.warning(f"⚠️ Se encontraron **{len(errores_locales)} imágenes con errores** de clasificación en el conjunto evaluado.")
                
                # Mostrar en una galería detallada exactamente cuáles imágenes fallaron
                err_cols = st.columns(3)
                for err_i, err_idx in enumerate(errores_locales):
                    with err_cols[err_i % 3]:
                        st.markdown(f"<div class='error-card'>", unsafe_allow_html=True)
                        st.image(imagenes_cargadas[err_idx], width=120)
                        st.write(f"**Archivo:** {nombres_archivos[err_idx]}")
                        st.write(f"🛑 **Real:** {'Barco (1)' if st.session_state['etiquetas_reales'][err_idx] == 1 else 'No Barco (0)'}")
                        st.write(f"🤖 **Predicción IA:** {'Barco (1)' if predicciones_clases[err_idx] == 1 else 'No Barco (0)'}")
                        st.markdown(f"</div>", unsafe_allow_html=True)
        else:
            st.info("💡 Etiqueta al menos algunas imágenes en la pestaña de 'Galería' para visualizar aquí el desglose visual de los errores.")

    # Panel global de métricas oficiales en la parte inferior
    if len(y_true_list) > 0:
        st.markdown("---")
        st.markdown("## 📊 Consolidado de Métricas en Vivo (Evaluación Oficial ABET)")
        
        y_true_arr = np.array(y_true_list)
        y_pred_arr = np.array(y_pred_list)
        
        acc = accuracy_score(y_true_arr, y_pred_arr)
        prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
        rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
        
        # Tarjetas estéticas de métricas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy en Vivo", f"{acc*100:.2f}%", delta=f"Meta >= 98%")
        m2.metric("Precision", f"{prec:.4f}")
        m3.metric("Recall", f"{rec:.4f}")
        m4.metric("F1-Score", f"{f1:.4f}")
        
        # Barra de progreso visual para el Accuracy
        st.progress(float(acc))
        
        if acc < 0.98:
            st.error(f"⚠️ El Accuracy actual ({acc*100:.2f}%) se encuentra por debajo de la meta institucional del 98%.")
        else:
            st.success(f"🎉 ¡Meta cumplida! El Accuracy ({acc*100:.2f}%) supera el umbral del 98% exigido por la rúbrica.")
            
        # Matriz de Confusión en vivo
        st.markdown("#### Matriz de Confusión Oficial")
        cm = confusion_matrix(y_true_arr, y_pred_arr)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Barco', 'Barco'], 
                    yticklabels=['No Barco', 'Barco'])
        plt.ylabel('Real (Ground Truth)')
        plt.xlabel('Predicción (IA)')
        st.pyplot(fig)
        
    else:
        st.info("💡 **Instrucción de Prueba:** Utiliza los botones de etiquetado en la pestaña superior para evaluar las imágenes y generar el reporte oficial.")
else:
    st.info("👈 Por favor, carga las imágenes de prueba usando el panel lateral para iniciar la evaluación interactiva en vivo.")