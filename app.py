import sys
import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import numpy as np

# FORZAR EL USO DE LA API ESTABLE DE PRODUCCIÓN (v1)
os.environ["GOOGLE_API_USE_CLIENT_OPTIONS"] = "1"

# ==========================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================

st.set_page_config(
    page_title="Chat Inteligente con PDF",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Chat Inteligente con PDF")
st.markdown(
    "Haz preguntas sobre tus documentos utilizando Inteligencia Artificial rápida y ligera."
)

# ==========================
# CONFIGURAR API KEY DE GEMINI
# ==========================

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.warning("⚠️ Falta configurar la clave GEMINI_API_KEY en los Secrets de Streamlit. El chat con IA no funcionará sin ella.")

# ==========================
# MEMORIA DEL CHAT
# ==========================

if "historial" not in st.session_state:
    st.session_state.historial = []

# ==========================
# SUBIR PDF
# ==========================

archivos = st.file_uploader(
    "📄 Selecciona uno o varios archivos PDF",
    type="pdf",
    accept_multiple_files=True
)

texto_total = ""

if archivos:

    barra = st.progress(0)

    for i, archivo in enumerate(archivos):
        lector = PdfReader(archivo)
        for pagina in lector.pages:
            texto = pagina.extract_text()
            if texto:
                texto_total += texto + "\n"
        barra.progress((i + 1) / len(archivos))

    st.success("✅ PDF cargado correctamente")

    fragmentos = [
        f.strip()
        for f in texto_total.split("\n")
        if len(f.strip()) > 15
    ]

    st.write(f"📑 Fragmentos encontrados: {len(fragmentos)}")

    pregunta = st.chat_input("Escribe tu pregunta...")

# ==========================
# BÚSQUEDA SEMÁNTICA CON GEMINI EMBEDDINGS
# ==========================

    if pregunta:
        with st.spinner("🔎 Buscando información con Gemini Embeddings..."):
            try:
                # Usamos el modelo estable text-embedding-004 de producción
                query_emb = genai.embed_content(
                    model="models/text-embedding-004",
                    content=pregunta,
                    task_type="retrieval_query"
                )["embedding"]

                # Obtener embeddings de los fragmentos
                if len(fragmentos) > 150:
                    st.warning("⚠️ El documento es muy largo. Se analizarán los primeros 150 fragmentos para evitar límites de la API gratuita.")
                    fragmentos_reducidos = fragmentos[:150]
                else:
                    fragmentos_reducidos = fragmentos

                doc_embs = genai.embed_content(
                    model="models/text-embedding-004",
                    content=fragmentos_reducidos,
                    task_type="retrieval_document"
                )["embeddings"]

                # Calcular similitud coseno
                mejor_texto = ""
                mejor_score = -1

                query_vec = np.array(query_emb)
                
                for frag, doc_emb in zip(fragmentos_reducidos, doc_embs):
                    doc_vec = np.array(doc_emb)
                    score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                    
                    if score > mejor_score:
                        mejor_score = score
                        mejor_texto = frag

                st.info(f"📊 Confianza de la búsqueda: {mejor_score:.2f}")

            except Exception as e:
                st.error(f"Error en la búsqueda semántica: {str(e)}")
                mejor_texto = fragmentos[0] if fragmentos else ""
                mejor_score = 0.50

# ==========================
# RESPUESTA CON GEMINI
# ==========================

        with st.spinner("🤖 Pensando con Gemini..."):
            try:
                # El modelo insignia de producción gemini-1.5-flash
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                
                contexto_y_pregunta = f"""
Eres un asistente que responde únicamente usando la información proporcionada del documento.

Información del documento:
{mejor_texto}

Pregunta:
{pregunta}
"""
                response = model.generate_content(contexto_y_pregunta)
                respuesta_ia = response.text
                
            except Exception as e:
                respuesta_ia = f"Error al conectar con Gemini: {str(e)}"

# ==========================
# GUARDAR EN EL HISTORIAL
# ==========================

        st.session_state.historial.append(
            {
                "pregunta": pregunta,
                "respuesta": respuesta_ia,
                "confianza": mejor_score
            }
        )

# ==========================
# MOSTRAR RESPUESTA
# ==========================

        st.subheader("🤖 Respuesta")
        st.write(respuesta_ia)
        st.progress(min(max(float(mejor_score), 0.0), 1.0))
        st.caption(f"Nivel de confianza: {mejor_score:.2f}")

# ==========================
# HISTORIAL DEL CHAT
# ==========================

if st.session_state.historial:
    st.divider()
    st.subheader("💬 Historial")

    for i, chat in enumerate(reversed(st.session_state.historial), start=1):
        with st.expander(f"Pregunta {i}"):
            st.markdown(f"**🙋 Pregunta:** {chat['pregunta']}")
            st.markdown(f"**🤖 Respuesta:** {chat['respuesta']}")
            st.caption(f"Confianza: {chat['confianza']:.2f}")

# ==========================
# BOTÓN LIMPIAR CHAT
# ==========================

if st.button("🗑️ Limpiar conversación"):
    st.session_state.historial = []
    st.rerun()

# ======================================
# SIDEBAR
# ======================================

with st.sidebar:
    st.title("🤖 Chat Inteligente")
    st.markdown("---")
    st.write("### Modelo IA")
    st.success("Google Gemini (gemini-1.5-flash)")
    st.write("### Motor de búsqueda")
    st.success("Gemini Embeddings (text-embedding-004)")
    st.write("### Librerías")
    st.markdown("""
- ✅ Streamlit
- ✅ Gemini API
- ✅ PyPDF2
- ✅ Numpy
- ✅ Python
""")
    st.markdown("---")

    if archivos:
        st.write("### Estadísticas")
        st.metric("PDF cargados", len(archivos))
        st.metric("Fragmentos", len(fragmentos))
        st.metric("Preguntas", len(st.session_state.historial))

# ======================================
# RESUMEN DEL DOCUMENTO
# ======================================

if archivos:
    if st.button("📄 Generar resumen"):
        with st.spinner("Generando resumen con Gemini..."):
            try:
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                prompt_resumen = f"Resume el siguiente documento en máximo 10 líneas:\n\n{texto_total[:12000]}"
                resumen = model.generate_content(prompt_resumen)
                
                st.subheader("📝 Resumen")
                st.write(resumen.text)
            except Exception as e:
                st.error(f"Error al generar el resumen: {str(e)}")

# ======================================
# DESCARGAR CHAT
# ======================================

if st.session_state.historial:
    texto_chat = ""
    for chat in st.session_state.historial:
        texto_chat += f"Pregunta: {chat['pregunta']}\n"
        texto_chat += f"Respuesta: {chat['respuesta']}\n"
        texto_chat += "-"*50+"\n"

    st.download_button(
        "📥 Descargar conversación",
        texto_chat,
        file_name="chat_pdf.txt",
        mime="text/plain"
    )

st.markdown("---")
st.caption("Proyecto desarrollado con ❤️ usando Python, Streamlit y Google Gemini.")