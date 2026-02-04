import streamlit as st
import whisper
from whisper.utils import get_writer
import os

st.set_page_config(page_title="Cultiver l'Avenir", page_icon="🌍")

st.title("🌍 Traduction Automatique des Portraits")


# Utilisation du modèle 'base' pour éviter que Streamlit Cloud ne plante (limite RAM)
@st.cache_resource
def load_model():
    return whisper.load_model("base")


try:
    model = load_model()
    st.success("IA prête à l'emploi !")
except Exception as e:
    st.error(f"Erreur lors du chargement de l'IA : {e}")

uploaded_file = st.file_uploader("Importer la vidéo italienne", type=["mp4", "mov", "mp3"])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Transcription et Traduction en cours... Patientez quelques minutes.")

    try:
        # On demande à Whisper de traduire directement en français (task='translate')
        # Whisper traduit nativement vers l'anglais, pour le français il faut parfois
        # une étape de plus, mais testons la version directe :
        result = model.transcribe("temp_video.mp4", task="translate")

        # Création des fichiers SRT et TXT
        output_dir = "."
        writer = get_writer("srt", output_dir)
        writer(result, "subtitles.srt")

        st.success("Analyse terminée !")

        col1, col2 = st.columns(2)
        with col1:
            with open("subtitles.srt", "rb") as f:
                st.download_button("Télécharger le .SRT", f, file_name="traduction.srt")
        with col2:
            st.download_button("Télécharger le .TXT", result["text"], file_name="transcription.txt")

    except Exception as e:
        st.error(f"Une erreur est survenue pendant le traitement : {e}")