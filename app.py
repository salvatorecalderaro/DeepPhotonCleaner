import streamlit as st
from tempfile import NamedTemporaryFile
from DeepPhotonCleaner import identify_device, read_fits, bin_data, create_windows, train_model
from plot_utils import plot_noisy_curve
from model import MultichannelAutoencoder

st.set_page_config(page_title="DeepPhotonCleaner", layout="wide")
st.title("🔭 DeepPhotonCleaner")
st.subheader(
    "The software uses deep learning to clean light-curves, separating signals from noise."
)
st.write("Upload a `.fits` file to clean noisy photon bins and download the cleaned result.")

# --- Upload file ---
uploaded_file = st.file_uploader("📂 Upload a FITS file", type=["fits"])
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

if "uploaded_file" not in st.session_state:
    st.info("Please upload a `.fits` file to start.")
    st.stop()

uploaded_file = st.session_state.uploaded_file

# --- Device ---
if "device" not in st.session_state:
    device, _ = identify_device()
    st.session_state.device = device

# --- Sidebar settings ---
bin_options = [2**i for i in range(13, 17)]
nt = st.sidebar.selectbox("Select number of bins (power of 2)", bin_options)
st.session_state.nt = nt

# --- Initialize session state ---
if "curve_created" not in st.session_state:
    st.session_state.curve_created = False
if "cleaning_done" not in st.session_state:
    st.session_state.cleaning_done = False

# --- Process uploaded file ---
if "glowcurvedata" not in st.session_state:
    with NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
        tmp.write(uploaded_file.read())
        tmp_fits_path = tmp.name

    obs_id = uploaded_file.name.split("_")[0]
    filename = uploaded_file.name.split("_")[1].split(".")[0]

    st.session_state.obs_id = obs_id
    st.session_state.uploaded_filename = filename
    st.session_state.tmp_fits_path = tmp_fits_path
    st.session_state.glowcurvedata = read_fits(tmp_fits_path)
    st.session_state.curve_data = None

# --- Create Curve Button ---
if not st.session_state.curve_created:
    if st.button("🎨 Create Curve"):
        grid, binned_data = bin_data(st.session_state.glowcurvedata, st.session_state.nt)
        st.session_state.curve_data = binned_data
        st.session_state.grid = grid
        st.session_state.curve_created = True
        st.success("✅ Curve successfully created!")

# --- Plot Curve ---
if st.session_state.curve_created and st.session_state.curve_data is not None:
    fig = plot_noisy_curve(
        st.session_state.obs_id,
        st.session_state.uploaded_filename,
        st.session_state.nt,
        st.session_state.grid,
        st.session_state.curve_data,
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Cleaning Button ---
if st.session_state.curve_created and not st.session_state.cleaning_done:
    if st.button("🚀 Run DPC Cleaning"):
        with st.spinner("Running cleaning..."):
            print(st.session_state.device)
            windows = create_windows(st.session_state.curve_data, window_size=16, stride=8)
            model = MultichannelAutoencoder().to(st.session_state.device)
            st.session_state.model = train_model(st.session_state.device, model, windows)
            