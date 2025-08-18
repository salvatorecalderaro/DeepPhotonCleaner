import streamlit as st
from tempfile import NamedTemporaryFile
from DeepPhotonCleaner import read_fits, bin_data
from plot_utils import plot_noisy_curve

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

st.sidebar.header("⚙️ Settings")
bin_options = [2**i for i in range(13, 17)]
nt = st.sidebar.selectbox("Select number of bins (power of 2)", bin_options)

if uploaded_file:
    st.success(f"✅ File `{uploaded_file.name}` uploaded successfully.")
    if (
        "uploaded_filename" not in st.session_state
        or st.session_state.uploaded_filename != uploaded_file.name
        or "nt" not in st.session_state
        or st.session_state.nt != nt
    ):

        with NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
            tmp.write(uploaded_file.read())
            tmp_fits_path = tmp.name
        
        
        obs_id = uploaded_file.name.split("_")[0]
        st.session_state.obs_id = obs_id
        
        filename = uploaded_file.name.split("_")[1].split(".")[0]
        st.session_state.uploaded_filename = filename
        st.session_state.nt = nt
        glowcurvedata = read_fits(tmp_fits_path)
        st.session_state.glowcurvedata = glowcurvedata
        st.session_state.uploaded_filename = filename
        st.session_state.nt = nt
        st.session_state.tmp_fits_path = tmp_fits_path
        st.session_state.curve_data = None
        st.session_state.selected_points = []
        st.session_state.curve_created = False
        
    if "uploaded_filename" in st.session_state:
        if not st.session_state.curve_created:
            if st.button("🎨 Create Curve"):
                grid, binned_data = bin_data(st.session_state.glowcurvedata, st.session_state.nt)
                st.session_state.curve_data = binned_data
                st.session_state.curve_created = True
                st.success("✅ Curve successfully created!")
                fig = plot_noisy_curve(st.session_state.obs_id, st.session_state.uploaded_filename, st.session_state.nt, grid, binned_data)
                st.plotly_chart(fig, use_container_width=True)
