# 🤖 Chat Inteligente con PDFs y Resumen Automático (Sistema RAG)

Aplicación web interactiva que permite a los usuarios interactuar de forma natural con documentos PDF extensos, extraer resúmenes ejecutivos automáticos y realizar consultas complejas en segundos, reduciendo hasta en un 80% el tiempo de lectura y análisis de documentación técnica o corporativa.

![Inicio](images/inicio.png)

## 🚀 Características Clave
*   **Procesamiento Multi-Documento:** Carga y lectura simultánea de múltiples archivos PDF de gran escala.
*   **Pipeline RAG Eficiente:** Segmentación de texto (Text Chunking) y generación de embeddings numéricos para búsquedas semánticas precisas.
*   **Generación de Resúmenes:** Extracción automatizada de los puntos clave del documento utilizando modelos de lenguaje avanzados.
*   **Descarga de Historial:** Exportación completa de la sesión de chat para auditoría o consulta posterior.

## 🏭 Casos de Uso en la Industria (Tu toque diferenciador)
*   **Producción y Mantenimiento:** Consulta rápida de manuales de maquinaria pesada en planta para resolver fallas en tiempo real.
*   **Calidad y Normativa:** Auditoría e inspección rápida de normas ISO o procedimientos estandarizados de operación (SOPs).

## 🛠️ Stack Tecnológico
*   **Backend & Lógica:** Python 3.x
*   **Interfaz de Usuario:** Streamlit (UI interactiva y en tiempo real)
*   **Modelos de Lenguaje (LLM):** API de Google Gemini (Gemini Pro)
*   **Procesamiento de Texto:** PyPDF2 (Extracción de texto)
*   **Computación Numérica:** NumPy (Manejo de vectores/embeddings)

## 📐 Arquitectura del Sistema
1. **Ingesta:** Extracción de texto plano desde los archivos PDF cargados.
2. **Procesamiento:** División del texto en fragmentos (chunks) optimizados para el contexto del LLM.
3. **Búsqueda Semántica:** Uso de embeddings y cálculo de similitud (NumPy) para encontrar los fragmentos más relevantes a la pregunta del usuario.
4. **Generación:** Inyección de contexto relevante en el prompt de Google Gemini para obtener respuestas precisas y sin alucinaciones.

## 📸 Demostración Visual

### Interfaz Principal y Resumen Automático
![Resumen](images/resumen.png)

### Sistema de Preguntas y Respuestas (Q&A)
![Pregunta](images/pregunta.png)

## 👩‍💻 Autora
**Cielo Nichool Cuevas Perdomo**  
*Tecnóloga en Producción Industrial & Especialista en Ciencia de Datos e IA Generativa*
