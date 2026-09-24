import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Configuración de la página web
st.set_page_config(page_title="IA - Detección de Barcos UAV", page_icon="🚢", layout="wide")

st.title("🚁 Sistema de Percepción Autónoma - Vehículo Aéreo No Tripulado (UAV)")
st.markdown("### Clasificación binaria de imágenes aéreas satelitales (80x80 px) para inspección portuaria en vivo.")
st.markdown("---")

# 1. Cargar el modelo entrenado previamente
@st.cache_resource
def cargar_modelo():
    return load_model('mejor_modelo_barcos.h5')

try:
    model = cargar_modelo()
    st.sidebar.success("✅ Modelo 'mejor_modelo_barcos.h5' cargado con éxito.")
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar el modelo: Asegúrate de tener 'mejor_modelo_barcos.h5' en la misma carpeta.\nDetalle: {e}")

# 2. Panel lateral para carga de imágenes de la USB
st.sidebar.header("📂 Carga de Test Set (USB)")
uploaded_files = st.sidebar.file_uploader("Sube las imágenes de prueba (ej. las 40 de la USB):", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Umbral de decisión ajustable
threshold = st.sidebar.slider("Umbral de Decisión (Threshold)", 0.0, 1.0, 0.5, 0.01)

if uploaded_files:
    st.sidebar.info(f"Imágenes cargadas en memoria: {len(uploaded_files)}")
    
    # Procesar imágenes para la IA
    imagenes_cargadas = []
    nombres_archivos = []
    
    for file in uploaded_files:
        img = Image.open(file).convert('RGB')
        img_resized = img.resize((80, 80)) # Obligatorio: 80x80 píxeles como pide la especificación técnica
        imagenes_cargadas.append(np.array(img_resized))
        nombres_archivos.append(file.name)
        
    X_test = np.array(imagenes_cargadas) / 255.0
    
    # Ejecutar inferencia de la IA
    predicciones_prob = model.predict(X_test)
    predicciones_clases = (predicciones_prob > threshold).astype(int).flatten()
    
    st.markdown("---")
    st.subheader("🖼️ Galería de Inferencia y Etiquetado Individual en Vivo")
    st.markdown("Verifica la predicción de la IA y asigna la **Etiqueta Real** de cada imagen para evaluar el desempeño en vivo.")
    
    # Listas para almacenar las respuestas reales provistas interactivamente
    y_true_list = []
    y_pred_list = []
    
    # Mostrar en columnas (rejilla de 4 elementos por fila)
    cols_per_row = 4
    rows = [uploaded_files[i:i + cols_per_row] for i in range(0, len(uploaded_files), cols_per_row)]
    
    global_idx = 0
    for row in rows:
        cols = st.columns(cols_per_row)
        for col_idx, file in enumerate(row):
            with cols[col_idx]:
                img_arr = imagenes_cargadas[global_idx]
                nombre = nombres_archivos[global_idx]
                pred_ia = predicciones_clases[global_idx]
                confianza = predicciones_prob[global_idx][0]
                
                # Mostrar imagen miniatura
                st.image(img_arr, width=130)
                st.caption(f"📁 {nombre}")
                
                # Resultado de la IA
                if pred_ia == 1:
                    st.markdown("🤖 IA: **BARCO** 🚢")
                else:
                    st.markdown("🤖 IA: **NO BARCO** 🌊")
                st.text(f"Confianza: {confianza*100:.1f}%")
                
                # REQUERIMIENTO DE RÚBRICA: Etiquetado individual por imagen
                etiqueta_real_str = st.selectbox(
                    f"Etiqueta real:", 
                    ["Seleccionar...", "Barco (1)", "No Barco (0)"], 
                    key=f"etiqueta_{global_idx}"
                )
                
                if etiqueta_real_str != "Seleccionar...":
                    val_real = 1 if "Barco (1)" in etiqueta_real_str else 0
                    y_true_list.append(val_real)
                    y_pred_list.append(pred_ia)
                    
                    if val_real == pred_ia:
                        st.success("✅ Acierto")
                    else:
                        st.error("❌ Fallo")
                
                st.markdown("---")
                global_idx += 1

    # ==========================================
    # 3. CONSOLIDACIÓN DE MÉTRICAS EN VIVO
    # ==========================================
    if len(y_true_list) > 0 and len(y_true_list) == len(uploaded_files):
        st.markdown("## 📊 Consolidado de Métricas en Vivo (Evaluación Oficial)")
        
        y_true_arr = np.array(y_true_list)
        y_pred_arr = np.array(y_pred_list)
        
        acc = accuracy_score(y_true_arr, y_pred_arr)
        prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
        rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
        
        # Tarjetas de métricas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy en Vivo", f"{acc*100:.2f}%", delta="Meta >= 98%")
        m2.metric("Precision", f"{prec:.4f}")
        m3.metric("Recall", f"{rec:.4f}")
        m4.metric("F1-Score", f"{f1:.4f}")
        
        # Validación de penalización de la rúbrica
        if acc < 0.98:
            penalizacion = ((0.98 - acc) / 0.02) * 0.5
            st.error(f"⚠️ El Accuracy está por debajo del 98%.")
        else:
            st.success("🎉 ¡Excelente! Se alcanzó o superó el 98% de accuracy exigido en el set de prueba.")
            
        # Matriz de Confusión y Análisis de fallos
        col_cm, col_fa = st.columns(2)
        
        with col_cm:
            st.markdown("#### Matriz de Confusión")
            cm = confusion_matrix(y_true_arr, y_pred_arr)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Barco', 'Barco'], 
                        yticklabels=['No Barco', 'Barco'])
            plt.ylabel('Real (Ground Truth)')
            plt.xlabel('Predicción (IA)')
            st.pyplot(fig)
            
        with col_fa:
            st.markdown("#### Resumen de Errores")
            errores = np.where(y_true_arr != y_pred_arr)[0]
            if len(errores) == 0:
                st.success("✨ ¡Cero errores en la clasificación de este lote!")
            else:
                st.warning(f"Se detectaron {len(errores)} errores de clasificación en total.")
                for idx_err in errores:
                    st.text(f"• Archivo: {nombres_archivos[idx_err]} (Real: {y_true_arr[idx_err]} vs Pred: {y_pred_arr[idx_err]})")
    else:
        st.info("💡 **Instrucción:** Asigna la 'Etiqueta real' en el menú desplegable de **todas** las imágenes de arriba para calcular automáticamente el Accuracy, la Matriz de Confusión y las métricas finales.")
else:
    st.info("👈 Por favor, sube las imágenes de prueba usando el panel lateral para iniciar la evaluación en vivo.")