# 🌍 ZNI-Clima Predictor: Simulador de Demanda Energética

**Herramienta de Planificación Logística y Prevención de Desabastecimiento de Combustible en Zonas No Interconectadas**

---

## 📋 Descripción del Proyecto

ZNI-Clima Predictor es una **aplicación Streamlit** que permite simular y predecir de manera **ex-ante** el riesgo de alta demanda de combustible en centrales de generación ubicadas en zonas no interconectadas (ZNI) de Colombia.

La aplicación utiliza un **pipeline de machine learning entrenado** que integra:
- 🌡️ Variables climáticas (temperatura, precipitación, brillo solar, humedad)
- ⚡ Características de infraestructura (capacidad, marca, operador)
- 📍 Datos geográficos y de estacionalidad

El modelo predice la **probabilidad de alta demanda de diésel** en escenarios específicos, facilitando la toma de decisiones logísticas del IPSE y Superservicios.

---

## 🚀 Características Principales

✅ **Simulación ex-ante** sin data leakage  
✅ **Interfaz interactiva** con parámetros configurables  
✅ **Alertas semánticas** (Rojo/Amarillo/Verde)  
✅ **Logging completo** de predicciones  
✅ **Validación robusta** de entrada  
✅ **Código modularizado** y documentado  
✅ **Modo demostración** cuando el modelo no está disponible  

---

## 📦 Requisitos de Instalación

### Opción 1: Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/Blanquis550/zni-clima-predictor.git
cd zni-clima-predictor

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 2: Despliegue en Streamlit Cloud

1. Sube el repositorio a GitHub
2. Ve a [Streamlit Cloud](https://streamlit.io/cloud)
3. Conecta tu repositorio y selecciona `app.py` como archivo principal
4. Configura las secrets si es necesario

---

## 🎯 Cómo Usar

### Ejecución Local

```bash
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

### Uso de la Interfaz

1. **Panel Lateral (🕹️ Parámetros de Simulación)**
   - Selecciona **capacidad** de generación (10-5000 kW)
   - Elige **marca** del generador (CUMMINS, LOVOL, etc.)
   - Selecciona **empresa operadora**
   - Elige **departamento** (CHOCO, AMAZONAS, etc.)
   - Ajusta **mes** evaluado (1-12)
   - Modifica **parámetros climáticos** (lluvia, temperatura, etc.)

2. **Panel Principal (Central)**
   - Ve la **ficha de la central** con ubicación e información regional
   - Consulta los **parámetros climáticos** simulados vs. históricos
   - Interpreta la **probabilidad de alta demanda** (barra de progreso)
   - Lee la **alerta semántica** con recomendaciones de acción

### Interpretación de Alertas

| Alerta | Rango Probabilidad | Significado | Acción |
|--------|-------------------|-------------|--------|
| 🔴 Rojo | ≥ 70% | Riesgo crítico | Despacho de combustible inmediato |
| 🟡 Amarillo | 40% - 70% | Riesgo moderado | Monitoreo preventivo semanal |
| 🟢 Verde | < 40% | Operación normal | Cronograma regular |

---

## 📊 Estructura de Datos

### Entrada al Modelo

```python
{
    'CAPACIDAD_GENERACION': float,      # kW
    'Precipitation_mm': float,          # mm
    'Temperature_C': float,             # °C
    'Solar_Brightness_hours': float,   # horas
    'Humidity_percent': float,          # %
    'MES': int,                         # 1-12
    'MARCA': str,                       # Categoría normalizada
    'EMPRESA': str,                     # Categoría normalizada
    'DEPARTAMENTO': str                 # Categoría normalizada
}
```

### Salida del Modelo

```python
{
    'probabilidad': float,  # 0.0 - 1.0
    'clase': int            # 0 (Bajo) o 1 (Alto)
}
```

---

## 🔧 Configuración del Modelo

### Carga del Modelo

El archivo `modelo_campeon_zni_clima.joblib` debe colocarse en la **raíz del repositorio**.

```
zni-clima-predictor/
├── app.py
├── requirements.txt
├── README.md
└── modelo_campeon_zni_clima.joblib  # ← Aquí
```

### Generación del Modelo

Si necesitas entrenar un nuevo modelo:

```python
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

# Cargar datos de entrenamiento
X_train, y_train = ...  # Tus datos

# Crear pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LGBMClassifier(...))
])

# Entrenar
pipeline.fit(X_train, y_train)

# Guardar
joblib.dump(pipeline, 'modelo_campeon_zni_clima.joblib')
```

---

## 📝 Logging y Debugging

Los logs se guardan en `zni_clima_app.log` con información:

- ✓ Carga del modelo
- ✓ Predicciones realizadas
- ✓ Errores y excepciones
- ✓ Timestamps completos

### Consultar Logs

```bash
tail -f zni_clima_app.log
```

---

## 🐛 Solución de Problemas

### Problema: "Modelo no encontrado"

**Causa:** El archivo `modelo_campeon_zni_clima.joblib` no existe en la raíz.  
**Solución:**
1. Verifica que el archivo esté presente
2. Usa la opción de demo si no tienes el modelo aún
3. Consulta `zni_clima_app.log` para detalles

### Problema: "Error al cargar el modelo"

**Causa:** Archivo corrupto o incompatible.  
**Solución:**
1. Regenera el modelo con las dependencias actuales
2. Verifica versiones de scikit-learn y LightGBM
3. Revisa los logs para el stack trace completo

### Problema: Predicciones inconsistentes

**Causa:** Valores de entrada inválidos (NaN, Infinito).  
**Solución:**
1. La app valida automáticamente, pero verifica los ranges
2. Consulta la tabla de parámetros climáticos

---

## 📚 Mejoras en V4

### Seguridad
- ❌ Removido: `unsafe_allow_class=True` (deprecated)
- ✅ Agregado: `unsafe_allow_html=True` (parámetro correcto)

### Logging
- ✅ Sistema completo con archivo rotativo
- ✅ Trazabilidad de todas las predicciones
- ✅ Stack traces completos en errores

### Validación
- ✅ Función `validate_input()` centralizada
- ✅ Verificación de NaN e Infinito
- ✅ Manejo de excepciones robusto

### Modularidad
- ✅ Funciones específicas por dominio
- ✅ Type hints en todas las funciones
- ✅ Docstrings completos en español

### Multiplataforma
- ✅ Rutas con `pathlib.Path`
- ✅ Compatible con Windows, macOS, Linux

---

## 👨‍💻 Desarrollo

### Estructura de Código

```
app.py
├── Logging setup
├── Streamlit config
├── CSS styles
├── Constants (NORMALES_DEPTOS, marcas, empresas)
├── Utility functions
│   ├── get_model_path()
│   ├── load_model()
│   ├── validate_input()
│   ├── prepare_prediction_data()
│   ├── make_prediction()
│   └── render_alert()
└── Main UI
    ├── Header
    ├── Sidebar (inputs)
    └── Main (outputs)
```

### Contribuir

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/mejora`
3. Realiza cambios y commitea
4. Push a tu rama: `git push origin feature/mejora`
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto es parte del **Proyecto Integrador** de la Maestría en Analítica de Datos.

**Estudiante:** Blanca Janeth Yepes Vergara  
**Institución:** [Tu Universidad]  
**Año:** 2026

---

## 📧 Contacto

- **GitHub:** [Blanquis550](https://github.com/Blanquis550)
- **Email:** blanca.yepes@upb.edu.co
- **Institución:** IPSE & Superservicios

---

## 🙏 Agradecimientos

- Gemini Notebook por la asistencia en diseño
- IPSE y Superservicios por contexto y datos
- Comunidad de Streamlit y scikit-learn

---

**Última actualización:** Septiembre 4, 2026  
**Versión:** 4.0 (Mejorada)
