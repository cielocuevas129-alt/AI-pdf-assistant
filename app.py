import sys

print(sys.executable)

import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import ollama

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
    "Haz preguntas sobre tus documentos utilizando Inteligencia Artificial."
)

# ==========================
# CARGAR EL MODELO
# ==========================

@st.cache_resource
def cargar_modelo():
    return SentenceTransformer("all-MiniLM-L6-v2")

modelo = cargar_modelo()

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

    pregunta = st.chat_input(
        "Escribe tu pregunta..."
    )
# ==========================
# BÚSQUEDA SEMÁNTICA
# ==========================

    if pregunta:

        with st.spinner("🔎 Buscando información..."):

            emb_pregunta = modelo.encode(
                pregunta,
                convert_to_tensor=True
            )

            mejor_texto = ""
            mejor_score = -1

            for frag in fragmentos:

                emb_frag = modelo.encode(
                    frag,
                    convert_to_tensor=True
                )

                score = util.cos_sim(
                    emb_pregunta,
                    emb_frag
                ).item()

                if score > mejor_score:
                    mejor_score = score
                    mejor_texto = frag

        st.info(
            f"📊 Confianza de la búsqueda: {mejor_score:.2f}"
        )

# ==========================
# RESPUESTA CON OLLAMA
# ==========================

        with st.spinner("🤖 Pensando..."):

            respuesta = ollama.chat(
                model="gemma3:1b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente que responde "
                            "únicamente usando la información "
                            "proporcionada del documento."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"""
Información del documento:

{mejor_texto}

Pregunta:

{pregunta}
"""
                    }
                ]
            )

        respuesta_ia = respuesta["message"]["content"]
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

        st.progress(
            min(max(mejor_score, 0), 1)
        )

        st.caption(
            f"Nivel de confianza: {mejor_score:.2f}"
        )

# ==========================
# HISTORIAL DEL CHAT
# ==========================

if st.session_state.historial:

    st.divider()

    st.subheader("💬 Historial")

    for i, chat in enumerate(
        reversed(st.session_state.historial),
        start=1
    ):

        with st.expander(
            f"Pregunta {i}"
        ):

            st.markdown(
                f"**🙋 Pregunta:** {chat['pregunta']}"
            )

            st.markdown(
                f"**🤖 Respuesta:** {chat['respuesta']}"
            )

            st.caption(
                f"Confianza: {chat['confianza']:.2f}"
            )

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
    st.success("Gemma 3:1B")

    st.write("### Motor de búsqueda")
    st.success("Sentence Transformers")

    st.write("### Librerías")

    st.markdown("""
- ✅ Streamlit
- ✅ Ollama
- ✅ PyPDF2
- ✅ Sentence Transformers
- ✅ Python
""")

    st.markdown("---")

    if archivos:

        st.write("### Estadísticas")

        st.metric(
            "PDF cargados",
            len(archivos)
        )

        st.metric(
            "Fragmentos",
            len(fragmentos)
        )

        st.metric(
            "Preguntas",
            len(st.session_state.historial)
        )

# ======================================
# RESUMEN DEL DOCUMENTO
# ======================================

if archivos:

    if st.button("📄 Generar resumen"):

        with st.spinner("Generando resumen..."):

            resumen = ollama.chat(

                model="gemma3:1b",

                messages=[

                    {

                        "role":"user",

                        "content":f"""

Resume el siguiente documento en máximo 10 líneas.

Documento:

{texto_total[:12000]}

"""

                    }

                ]

            )

        st.subheader("📝 Resumen")

        st.write(

            resumen["message"]["content"]

        )

# ======================================
# DESCARGAR CHAT
# ======================================

if st.session_state.historial:

    texto_chat=""

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

# ======================================
# FOOTER
# ======================================

st.markdown("---")

st.caption(
    "Proyecto desarrollado con ❤️ usando Python, Streamlit, Ollama y Gemma 3."
)
