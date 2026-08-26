import streamlit as st

from src.splitter import split_audio, test

app = st.App("main.py")

st.header("DSP - Audio Segmentation")

st.button("test", on_click=test)
# audio_file = st.file_uploader(
#     "Input a sequential conversation (accept .mp3)",
#     accept_multiple_files=False,
#     max_upload_size=5,
# )

# if audio_file:
#     splitted_audio_list = split_audio(audio_file)
#     for audio in splitted_audio_list:
#         st.audio(audio)
