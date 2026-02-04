import streamlit as st
import whisper
from whisper.utils import get_writer
import os
import tempfile
import shutil

from transformers import MarianMTModel, MarianTokenizer

# -----------------------------
# CONFIG STREAMLIT
# -----------------------------
st.set_page_config(
    page_title="Cultiver l'Avenir - Sous-titrage",
    page_icon="🌍"
)

st.title("🌍 Plateforme de Transcription & Traduction")
st.markdown("🎧 Vidéo italien → 🇫🇷 Sous-titres français + Texte brut")

# -----------------------------
# CHECK FFMPEG
# -----------------------------
if shutil.which("ffmpeg") is None:
    st.error("❌ FFmpeg n'est pas installé.")
    st.warning("Installe-le avec : winget install ffmpeg")
    st.stop()

# -----------------------------
# LOAD WHISPER
# -----------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("medium")

# -----------------------------
# LOAD TRANSLATOR (MarianMT)
# -----------------------------
@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-it-fr"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

whisper_model = load_whisper()
tokenizer, translator_model = load_translator()

# -----------------------------
# TRANSLATE FUNCTION (chunk safe)
# -----------------------------
def translate_it_to_fr(text):
    """
    Traduction en français phrase par phrase
    (évite les limites de longueur)
    """
    sentences = text.split(".")
    french_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence == "":
            continue

        inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
        translated_tokens = translator_model.generate(**inputs)
        fr = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        french_sentences.append(fr)

    return ". ".join(french_sentences)


# -----------------------------
# UPLOAD FILE
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Importer une vidéo/audio italien",
    type=["mp4", "mkv", "mov", "mp3", "wav"]
)

# -----------------------------
# PROCESS
# -----------------------------
if uploaded_file is not None:

    # Sauvegarde temporaire
    file_extension = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    # Barre de progression
    progress = st.progress(0)
    status = st.empty()

    # -----------------------------
    # 1. TRANSCRIPTION ITALIEN
    # -----------------------------
    status.info("🎧 Transcription italienne en cours...")
    progress.progress(20)

    result = whisper_model.transcribe(temp_path, task="transcribe")

    italian_text = result["text"].strip()
    progress.progress(55)

    # -----------------------------
    # 2. TRADUCTION FRANÇAIS
    # -----------------------------
    status.info("🌍 Traduction italien → français en cours...")
    french_text = translate_it_to_fr(italian_text)

    progress.progress(85)

    # -----------------------------
    # 3. GÉNÉRATION SRT FR
    # -----------------------------
    status.info("📝 Génération des sous-titres français...")

    output_dir = "."
    srt_writer = get_writer("srt", output_dir)

    # Remplacer le texte par la traduction française
    result["text"] = french_text
    srt_writer(result, "subtitles_fr")

    progress.progress(100)
    status.success("✅ Terminé !")

    # -----------------------------
    # DOWNLOAD BUTTONS
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        with open("subtitles_fr.srt", "rb") as f:
            st.download_button(
                "📥 Télécharger SRT (FR)",
                f,
                file_name="sous_titres_fr.srt"
            )

    with col2:
        st.download_button(
            "📥 Télécharger TXT (FR)",
            french_text,
            file_name="transcription_fr.txt"
        )

    # -----------------------------
    # PREVIEW
    # -----------------------------
    with st.expander("👀 Voir le texte traduit en français"):
        st.write(french_text)

    # -----------------------------
    # CLEAN TEMP FILE
    # -----------------------------
    os.remove(temp_path)
