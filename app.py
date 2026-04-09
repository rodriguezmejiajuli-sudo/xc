streamlit
joblib
pandas
numpy
scikit-learn==1.6.1

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# 1. CARGAR EL MODELO
# =============================================================================

@st.cache_resource  # Se carga una sola vez, no en cada interacción
def cargar_modelo():
    artefacto = joblib.load("mejor_modelo_predictivo.pkl")
    return (
        artefacto["pipeline"],
        artefacto["columnas_entrada"],
        artefacto["clases_objetivo"],
        artefacto["metricas"]
    )

pipeline, columnas_entrada, clases_objetivo, metricas = cargar_modelo()

# 2. INTERFAZ GRÁFICA — STREAMLIT
# =============================================================================

st.title("Prediccion de Segmento de Cliente")
st.markdown("Completa los datos del cliente para conocer a que segmento pertenece.")

st.divider()

# Variables categoricas
st.subheader("Datos cualitativos")
col1, col2, col3 = st.columns(3)

with col1:
    Ruteado = st.selectbox("Esta ruteado?", ["Si", "No"])

with col2:
    Sede = st.selectbox("Sede", ["Bogota", "Medellin", "Cali", "Barranquilla", "Otra"])

with col3:
    Empresa = st.selectbox("Empresa", ["Empresa_A", "Empresa_B", "Empresa_C", "Empresa_D", "Otra"])

# Variables numericas
st.subheader("Datos cuantitativos")
col4, col5 = st.columns(2)

with col4:
    Ingresos     = st.number_input("Ingresos ($)",                    min_value=0,   value=500000, step=10000)
    Neg_Fact     = st.number_input("No. Negocios Facturados",         min_value=0,   value=10,     step=1)
    Margen       = st.number_input("Margen (0 a 1)",                  min_value=0.0, max_value=1.0, value=0.30, step=0.01, format="%.2f")
    RIF_Empresa  = st.number_input("RIF x Empresa",                   min_value=0,   value=3,      step=1)
    Integralidad = st.number_input("Integralidad (0 a 1)",            min_value=0.0, max_value=1.0, value=0.70, step=0.01, format="%.2f")
    Permanencia  = st.number_input("Permanencia (meses)",             min_value=0,   value=12,     step=1)

with col5:
    Cartera      = st.number_input("Cartera (0 a 1)",                 min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.2f")
    Pot_Impo     = st.number_input("Potencial Logistico Impo (0-1)",  min_value=0.0, max_value=1.0, value=0.50, step=0.01, format="%.2f")
    Pot_Expo     = st.number_input("Potencial Logistico Expo (0-1)",  min_value=0.0, max_value=1.0, value=0.30, step=0.01, format="%.2f")
    Fidelidad    = st.number_input("Fidelidad (0 a 1)",               min_value=0.0, max_value=1.0, value=0.80, step=0.01, format="%.2f")

st.divider()

# 3. PREPARAR LOS DATOS
#    Se construye el DataFrame con exactamente los mismos nombres de columnas
#    que se usaron en el entrenamiento (guardados en columnas_entrada del .pkl)
# =============================================================================

datos = [[
    Ruteado, Sede, Empresa,
    Ingresos, Neg_Fact, Margen, RIF_Empresa,
    Integralidad, Permanencia, Cartera,
    Pot_Impo, Pot_Expo, Fidelidad
]]

data = pd.DataFrame(datos, columns=columnas_entrada)

# 4. PREDICCION
#    El Pipeline ya contiene el preprocesamiento completo (imputer + scaler + onehot)
#    No se necesita transformar manualmente como en el modelo de regresion
# =============================================================================

if st.button("Predecir segmento", use_container_width=True, type="primary"):

    segmento_pred  = pipeline.predict(data)[0]
    probabilidades = pipeline.predict_proba(data)[0]

    # Resultado principal
    st.subheader("Resultado de la prediccion")

    descripciones_segmento = {
        0: "Segmento 0 — Cliente de bajo valor / reciente",
        1: "Segmento 1 — Cliente de valor medio / en desarrollo",
        2: "Segmento 2 — Cliente de alto valor / estrategico",
    }

    etiqueta = descripciones_segmento.get(segmento_pred, f"Segmento {segmento_pred}")
    st.success(f"**{etiqueta}**")

    # Probabilidades por segmento
    st.subheader("Probabilidades por segmento")
    df_prob = pd.DataFrame({
        "Segmento": [f"Segmento {c}" for c in clases_objetivo],
        "Probabilidad (%)": [round(p * 100, 1) for p in probabilidades]
    })
    st.bar_chart(df_prob.set_index("Segmento"))

    # Datos ingresados (opcional, para verificar)
    with st.expander("Ver datos ingresados"):
        st.dataframe(data.T.rename(columns={0: "Valor"}), use_container_width=True)

    # Referencia del modelo
    st.info(
        f"Modelo: {metricas['mejor_modelo']}  |  "
        f"Accuracy (prueba): {metricas['accuracy'] * 100:.1f}%  |  "
        f"F1 ponderado: {metricas['f1_ponderado'] * 100:.1f}%"
    )
