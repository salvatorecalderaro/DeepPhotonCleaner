import streamlit as st

st.set_page_config(page_title="DeepPhotonCleaner", layout="wide")
st.title("🔭 DeepPhotonCleaner")
st.subheader("The software available on this page uses deep learning to ‘clean’ light-curves, enabling the separation of signals from astrophysical sources from noise, such as that caused by solar flares.")
st.write(
    "Upload a `.fits` file to clean noisy photon bins and download the cleaned result."
)


uploaded_file = st.file_uploader(
    "📂 Upload a FITS file",
    type=["fits"],
    key=st.session_state.get("uploader_key", "default_uploader"),
)

if "uploaded_filename" not in st.session_state and not uploaded_file:
    st.info("Please upload a `.fits` file to start.")
    st.stop()

