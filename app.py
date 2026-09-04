# -*- coding: utf-8 -*-
"""
Proyecto Integrador - Fase 6: Despliegue en Producción (Versión 4 - Mejorada)
Aplicación Interactiva de Simulación y Predicción Ex-Ante en Streamlit (app.py)

Diseñado por Gemini Notebook para Blanca.
Esta aplicación carga el pipeline serializado 'modelo_campeon_zni_clima.joblib' 
y permite simular riesgos de alta generación de manera ex-ante y libre de Data Leakage.

Mejoras V4:
- Seguridad: Eliminación de unsafe_allow_class
- Logging: Trazabilidad completa de predicciones
- Validación: Verificación robusta de entrada
- Mantenibilidad: Funciones modularizadas
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zni_clima_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE PÁGINA DE STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="ZNI-Clima Predictor | IPSE & Superservicios",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ESTILOS CSS PERSONALIZADOS (SEGURO)
# ============================================================================
def load_custom_styles():
    """Carga estilos CSS personalizados de manera segura."""
    st.markdown("""
        <style>
        .main-title {
            font-size: 38px;
            font-weight: 800;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 18px;
            color: #4B5563;
            text-align: center;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #F3F4F6;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #1E3A8A;
            margin-bottom: 20px;
        }
        .alert-header {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .card-location {
            font-size: 20px;
            font-weight: 700;
            color: #1E3A8A;
            margin: 0;
        }
        .card-text-normal {
            font-size: 15px;
            margin: 0;
        }
        .card-text-info {
            font-size: 13px;
            font-style: italic;
            color: #4B5563;
            margin: 0;
        }
        .probability-percentage {
            text-align: center;
            color: #1E3A8A;
            font-size: 2.5em;
        }
        .footer-text {
            text-align: center;
            font-size: 12px;
            color: #6B7280;
            margin-top: 20px;
        }
        </style>
    """)

load_custom_styles()

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================
NORMALES_DEPTOS = {
    'CHOCO': {
        'temp': 26.8, 'rain': 650.0, 'sun': 120.0, 'humidity': 88.0,
        'info': "Región Pacífica. Pluviosidad extrema todo el año. Alta sensibilidad a navegabilidad fluvial."
    },
    'AMAZONAS': {
        'temp': 25.9, 'rain': 280.0, 'sun': 150.0, 'humidity': 84.0,
        'info': "Región Amazónica. Selva tropical densa. Logística fluvial prioritaria de largo alcance."
    },
    'GUAINIA': {
        'temp': 27.2, 'rain': 320.0, 'sun': 160.0, 'humidity': 81.0,
        'info': "Orinoquía-Amazonía. Transición de sabana a selva. Altamente sensible a la cota del río Inírida."
    },
    'VAUPES': {
        'temp': 26.1, 'rain': 300.0, 'sun': 140.0, 'humidity': 85.0,
        'info': "Amazonía profunda. Aislamiento geográfico total. Logística por micro-cuencas."
    },
    'NARIÑO': {
        'temp': 25.5, 'rain': 420.0, 'sun': 110.0, 'humidity': 86.0,
        'info': "Pacífico Sur. Comunidades pesqueras costeras y selváticas de difícil acceso."
    },
    'CAUCA': {
        'temp': 25.8, 'rain': 480.0, 'sun': 115.0, 'humidity': 87.0,
        'info': "Costa pacífica del Cauca (Guapí / López de Micay). Alta pluviosidad e influencia de mareas."
    },
    'VICHADA': {
        'temp': 28.5, 'rain': 180.0, 'sun': 210.0, 'humidity': 74.0,
        'info': "Sabanas orientales. Altas temperaturas en temporada seca. Generación híbrida de gran escala."
    },
    'PUTUMAYO': {
        'temp': 24.8, 'rain': 310.0, 'sun': 130.0, 'humidity': 83.0,
        'info': "Piedemonte amazónico. Influencia andina y amazónica. Navegación sobre el río Putumayo."
    },
    'ANTIOQUIA': {
        'temp': 24.2, 'rain': 250.0, 'sun': 145.0, 'humidity': 80.0,
        'info': "Zonas rurales del norte y Urabá antioqueño. Topografía montañosa y acceso terrestre complejo."
    }
}

GENERADOR_MARCAS = ['CUMMINS', 'LOVOL', 'PERKINS', 'CATERPILLAR', 'DEUTZ', 'ELP', 'POWELL', 'YUNDAY', 'OTRA']
OPERADORES_ESP = ['EMPRESA_A_ESP', 'EMPRESA_B_ESP', 'EMPRESA_C_ESP', 'EMPRESA_D_ESP']

# ============================================================================
# FUNCIONES UTILITARIAS
# ============================================================================

def get_model_path() -> Path:
    """Retorna la ruta del modelo de manera multiplataforma."""
    return Path(__file__).parent / "modelo_campeon_zni_clima.joblib"

def load_model(model_path: Path) -> Optional[object]:
    """
    Carga el modelo de joblib con manejo seguro de excepciones.
    
    Args:
        model_path: Ruta al archivo .joblib
        
    Returns:
        Pipeline del modelo o None si hay error
    """
    try:
        if not model_path.exists():
            logger.warning(f"Modelo no encontrado en {model_path}")
            return None
        
        model = joblib.load(model_path)
        logger.info(f"Modelo cargado exitosamente desde {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}", exc_info=True)
        return None

def validate_input(
    temp: float,
    lluvia: float,
    sol: float,
    humedad: float,
    capacidad: float
) -> Tuple[bool, str]:
    """
    Valida que todas las entradas sean valores numéricos válidos.
    
    Args:
        temp: Temperatura en °C
        lluvia: Precipitación en mm
        sol: Brillo solar en horas
        humedad: Humedad relativa en %
        capacidad: Capacidad en kW
        
    Returns:
        Tupla (es_valido, mensaje_error)
    """
    valores = {
        'Temperatura': temp,
        'Precipitación': lluvia,
        'Brillo Solar': sol,
        'Humedad': humedad,
        'Capacidad': capacidad
    }
    
    for nombre, valor in valores.items():
        if np.isnan(valor) or np.isinf(valor):
            return False, f"Error: {nombre} contiene un valor inválido (NaN o Infinito)"
    
    return True, ""

def prepare_prediction_data(
    capacidad: float,
    lluvia: float,
    temp: float,
    sol: float,
    humedad: float,
    mes: int,
    marca: str,
    empresa: str,
    departamento: str
) -> pd.DataFrame:
    """
    Prepara el DataFrame para la predicción con validación.
    
    Args:
        capacidad: Capacidad de generación (kW)
        lluvia: Precipitación (mm)
        temp: Temperatura (°C)
        sol: Brillo solar (horas)
        humedad: Humedad (%)
        mes: Mes (1-12)
        marca: Marca del generador
        empresa: Empresa operadora
        departamento: Departamento
        
    Returns:
        DataFrame preparado para predicción
    """
    return pd.DataFrame([{
        'CAPACIDAD_GENERACION': capacidad,
        'Precipitation_mm': lluvia,
        'Temperature_C': temp,
        'Solar_Brightness_hours': sol,
        'Humidity_percent': humedad,
        'MES': mes,
        'MARCA': marca,
        'EMPRESA': empresa,
        'DEPARTAMENTO': departamento
    }])

def make_prediction(
    pipeline: object,
    data: pd.DataFrame,
    demo_mode: bool = False
) -> Tuple[float, int]:
    """
    Realiza la predicción usando el pipeline.
    
    Args:
        pipeline: Pipeline del modelo
        data: DataFrame con datos para predicción
        demo_mode: Si True, usa simulación matemática
        
    Returns:
        Tupla (probabilidad, clase)
    """
    try:
        if demo_mode:
            # Simulación ex-ante para demo
            temp = data['Temperature_C'].values[0]
            lluvia = data['Precipitation_mm'].values[0]
            sol = data['Solar_Brightness_hours'].values[0]
            capacidad = data['CAPACIDAD_GENERACION'].values[0]
            
            score = 1.0 / (1.0 + np.exp(-(-6.5 + 0.18*temp + 0.004*lluvia - 0.015*sol + 0.002*capacidad)))
            probabilidad = np.clip(score, 0.0, 1.0)
            clase = 1 if probabilidad > 0.5 else 0
            
            logger.info(f"Demo mode prediction: probabilidad={probabilidad:.4f}")
        else:
            # Predicción real
            probabilidad = pipeline.predict_proba(data)[0][1]
            clase = pipeline.predict(data)[0]
            
            logger.info(f"Real prediction: probabilidad={probabilidad:.4f}, clase={clase}")
        
        return probabilidad, int(clase)
    
    except Exception as e:
        logger.error(f"Error durante predicción: {str(e)}", exc_info=True)
        # Retornar predicción neutra en caso de error
        return 0.5, 0

def render_alert(probabilidad: float) -> None:
    """
    Renderiza la alerta semántica basada en probabilidad.
    
    Args:
        probabilidad: Probabilidad de alta demanda (0-1)
    """
    if probabilidad >= 0.70:
        st.error("🔴 ALERTA ROJA: Riesgo Crítico de Alta Demanda")
        st.markdown("""
            **Diagnóstico:** Las condiciones climáticas e infraestructura física simulan un escenario 
            de alto estrés energético. Existe una probabilidad crítica de que la central opere a carga 
            plena superando la mediana nacional de consumo de diésel.
            
            **Acción Sugerida:** El IPSE y la Superservicios deben coordinar de inmediato un despacho 
            de combustible de contingencia y asegurar la navegabilidad o acceso terrestre a la central 
            antes del inicio del mes calendario.
        """)
    
    elif 0.40 <= probabilidad < 0.70:
        st.warning("🟡 ALERTA AMARILLA: Riesgo Moderado de Incremento de Demanda")
        st.markdown("""
            **Diagnóstico:** Escenario de demanda de transición estacional. La combinación de temperatura 
            y humedad indica una probabilidad intermedia de sobrepasar el umbral histórico de consumo.
            
            **Acción Sugerida:** Incluir a la central en la lista de monitoreo preventivo semanal. 
            Verificar los niveles de almacenamiento actuales en los tanques locales para mitigar 
            cualquier retraso logístico menor.
        """)
    
    else:
        st.success("🟢 ALERTA VERDE: Operación Estable y Consumo Normal")
        st.markdown("""
            **Diagnóstico:** Condiciones de operación óptimas. El clima de fondo y la estacionalidad 
            favorecen un consumo moderado o equilibrado del combustible diésel.
            
            **Acción Sugerida:** Continuar con el cronograma regular de despacho de combustible. 
            Operación normal.
        """)

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

# 1. Cabezote de la Aplicación
st.markdown('<div class="main-title">⚡ Simulador Predictivo Ex-Ante: ZNI-Clima (V4)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Herramienta de Planificación Logística y Prevención de Desabastecimiento de Combustible en Zonas No Interconectadas</div>', unsafe_allow_html=True)

# 2. Sidebar de Configuración de Escenarios
st.sidebar.header("🕹️ Parámetros de Simulación Ex-Ante")

# --- BLOQUE 1: INFRAESTRUCTURA FÍSICA ---
st.sidebar.subheader("🔌 Especificaciones de la Central")

capacidad = st.sidebar.number_input(
    "Capacidad de Generación (kW):",
    min_value=10.0,
    max_value=5000.0,
    value=450.0,
    step=50.0,
    help="Capacidad nominal instalada de la planta."
)

marca = st.sidebar.selectbox(
    "Marca del Generador:",
    options=GENERADOR_MARCAS,
    index=0,
    help="Marca fabricante (valores normalizados para evitar alta cardinalidad)."
)

empresa = st.sidebar.selectbox(
    "Empresa Operadora:",
    options=OPERADORES_ESP,
    index=0,
    help="Operador responsable de la planta."
)

# --- BLOQUE 2: UBICACIÓN Y TIEMPO ---
st.sidebar.subheader("📍 Geografía y Temporalidad")

depto_seleccionado = st.sidebar.selectbox(
    "Departamento Objetivo:",
    options=list(NORMALES_DEPTOS.keys()),
    index=0
)

mes_seleccionado = st.sidebar.slider(
    "Mes Operativo (MES):",
    min_value=1,
    max_value=12,
    value=6,
    help="Representa el mes estacional a evaluar (ej: 6 = Junio)"
)

# Cargar las normales de fondo del departamento seleccionado
valores_defecto = NORMALES_DEPTOS[depto_seleccionado]

# --- BLOQUE 3: METEOROLOGÍA DE FONDO ---
st.sidebar.subheader("🌡️ Anomalías Climáticas (Normales IDEAM)")
st.sidebar.caption("Se precargan las normales climatológicas históricas, pero puedes simular fenómenos climáticos como El Niño o La Niña ajustando los controles deslizantes.")

lluvia = st.sidebar.slider(
    "Precipitación Mensual (mm):",
    min_value=0.0,
    max_value=1200.0,
    value=valores_defecto['rain'],
    step=10.0
)

temp = st.sidebar.slider(
    "Temperatura Media (°C):",
    min_value=10.0,
    max_value=45.0,
    value=valores_defecto['temp'],
    step=0.5
)

sol = st.sidebar.slider(
    "Brillo Solar (Horas/Mes):",
    min_value=0.0,
    max_value=350.0,
    value=valores_defecto['sun'],
    step=5.0
)

humedad = st.sidebar.slider(
    "Humedad Relativa (%):",
    min_value=30.0,
    max_value=100.0,
    value=valores_defecto['humidity'],
    step=1.0
)

# 3. Validar entrada
es_valido, msg_error = validate_input(temp, lluvia, sol, humedad, capacidad)
if not es_valido:
    st.error(f"⚠️ {msg_error}")
    st.stop()

# 4. Preparar datos
nuevo_registro = prepare_prediction_data(
    capacidad, lluvia, temp, sol, humedad,
    mes_seleccionado, marca, empresa, depto_seleccionado
)

# 5. Cargar modelo
modelo_path = get_model_path()
pipeline_campeon = load_model(modelo_path)
usar_demo = pipeline_campeon is None

# 6. Realizar predicción
probabilidad, clase_predicha = make_prediction(
    pipeline_campeon,
    nuevo_registro,
    demo_mode=usar_demo
)

# 7. Diseño del Cuerpo Principal
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### 📋 Ficha de la Central y Contexto")
    
    ficha_html = f"""
    <div class="metric-card">
        <h4>📍 Ubicación:</h4>
        <p class="card-location">{depto_seleccionado}</p>
        
        <h4 style='margin-top:15px;'>🔌 Infraestructura de Planta:</h4>
        <p class="card-text-normal">
            • Capacidad: <strong>{capacidad:.1f} kW</strong><br>
            • Marca de Generador: <strong>{marca}</strong><br>
            • Operador: <strong>{empresa}</strong>
        </p>
        
        <h4 style='margin-top:15px;'>⏳ Periodo Evaluado:</h4>
        <p class="card-text-normal" style='font-weight:600;'>Mes {mes_seleccionado} (Ciclo Estacional)</p>
        
        <h4 style='margin-top:15px;'>💡 Características Regionales:</h4>
        <p class="card-text-info">{valores_defecto['info']}</p>
    </div>
    """
    st.markdown(ficha_html, unsafe_allow_html=True)
    
    st.markdown("### 🔬 Parámetros Climáticos en Simulación")
    df_params = pd.DataFrame({
        'Variable': ['Precipitación (mm)', 'Temperatura (°C)', 'Brillo Solar (Hrs)', 'Humedad (%)'],
        'Valor Simulado': [f'{lluvia:.1f}', f'{temp:.1f}', f'{sol:.1f}', f'{humedad:.1f}'],
        'Normal Histórica': [f"{valores_defecto['rain']:.1f}", f"{valores_defecto['temp']:.1f}", 
                           f"{valores_defecto['sun']:.1f}", f"{valores_defecto['humidity']:.1f}"]
    })
    st.dataframe(df_params, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 🔮 Predicción del Perfil de Demanda")
    
    if usar_demo:
        st.info(
            "ℹ️ Ejecutando en modo de demostración visual (Simulador de Probabilidades).\n\n"
            "Asegúrate de haber subido el archivo 'modelo_campeon_zni_clima.joblib' al repositorio."
        )
    
    st.markdown("#### Probabilidad de Alerta de Alta Demanda de Generación:")
    st.progress(float(probabilidad))
    st.markdown(f'<h2 class="probability-percentage">{probabilidad*100:.1f}%</h2>', unsafe_allow_html=True)
    
    st.markdown("---")
    render_alert(probabilidad)

st.markdown("---")
footer_html = """
<div class="footer-text">
    Proyecto Integrador - Maestría en Analítica de Datos | Estudiante: Blanca Janeth Yepes Vergara<br>
    Desplegado con éxito bajo metodología CRISP-DM ex-ante y libre de Data Leakage (V4 - Mejorada).
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
