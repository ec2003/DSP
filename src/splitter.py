import streamlit as st
from streamlit.typing import UploadedFile

import time


def split_audio(audio_file: UploadedFile) -> list[UploadedFile]:
    return

def test():
    with st.status("Splitting audio...", expanded=True) as status:
        st.write("Preprocessing...")
        time.sleep(1)
        st.write("Generating embedding...")
        time.sleep(1)
        st.write("Splitting...")
        time.sleep(1)
        status.update(
            label="Download complete!", state="complete", expanded=False
        )