import pandas as pd
import numpy as np
from astropy.io import fits
import torch
import random
import os
import platform
import cpuinfo
from torch.utils.data import TensorDataset,DataLoader
from astropy.table import Table
import streamlit as st


seed = 2025 
alpha = 0.7
beta = 0.3
lr = 1e-3
mini_batch_size = 16
gamma = 0.1
epochs = 1000

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def identify_device():
    so = platform.system()
    if so == "Darwin":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_name = (
            torch.cuda.get_device_name()
            if device.type == "cuda"
            else cpuinfo.get_cpu_info()["brand_raw"]
        )
    if device.type == "cuda":
        set_seed(seed)
    return device, dev_name

def convert_endian(df):
    """
    Swap the byte order of the numeric columns in a DataFrame if they are in big-endian byte order.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose columns to check and convert

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns converted to little-endian byte order where necessary
    """
    for col in df.columns:
        dtype = df[col].dtype
        if dtype.byteorder == ">" and np.issubdtype(dtype, np.number):
            swapped = df[col].values.byteswap()
            df[col] = swapped.view(swapped.dtype.newbyteorder()).copy()
    return df


def read_fits(path):
    """
    Reads a FITS file and converts the data into a Pandas DataFrame.

    Parameters
    ----------
    path : str
        The file path to the FITS file.

    Returns
    -------
    glowcurvedata : pandas.DataFrame
        DataFrame containing the data from the FITS file with numeric columns
        converted to little-endian byte order.
    """

    with fits.open(path) as hdul:
        data = hdul[1].data
    glowcurvedata = pd.DataFrame(data)
    glowcurvedata = convert_endian(glowcurvedata)
    return glowcurvedata


def bin_data(glowcurvedata, nbins):
    """
    Bins the data by time and calculates the mean energy for each bin.

    Parameters
    ----------
    glowcurvedata : pandas.DataFrame
        DataFrame containing the data from the FITS file.
    nbins : int
        Number of bins to divide the time range into.

    Returns
    -------
    grid : array_like
        Array of bin edges.
    binned_data : array_like
        Array of shape (2, nbins) where the first row is the counts per bin
        and the second row is the mean energy per bin.

    """
    times = glowcurvedata["TIME"]
    min_t = times.min()
    max_t = times.max()
    grid = np.linspace(min_t, max_t, nbins + 1)
    binned = pd.cut(times, bins=grid, include_lowest=True, right=True)
    counts = binned.value_counts(sort=False)
    energy_mean = glowcurvedata.groupby(binned, observed=True)["PI"].mean()
    energy_mean_aligned = energy_mean.reindex(counts.index, fill_value=0)
    binned_data = np.vstack((counts.values, energy_mean_aligned.values))
    return grid, binned_data

def create_windows(data, window_size, stride):
    """
    Creates overlapping windows of data.

    Parameters
    ----------
    data : array_like
        The input data.
    window_size : int
        The size of each window.
    stride : int
        The step between each window.

    Returns
    -------
    windows : array_like
        The overlapping windows.
    """
    windows = []
    for start in range(0, data.shape[1] - window_size + 1, stride):
        windows.append(data[:, start : start + window_size])
    return np.stack(windows)


def train_model(device, model, windows):
    """
    Train a model to identify noisy bins.

    Parameters
    ----------
    device : torch.device
        The device to use for training.
    model : torch.nn.Module
        The model to train.
    windows : array_like
        The overlapping windows of data.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    poisson_loss_fn = torch.nn.PoissonNLLLoss(log_input=False, full=True)
    mse_loss_fn = torch.nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(windows).float())
    train_loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("🔍 Identifying noisy bins...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out,_ = model(batch)
            loss_poisson = poisson_loss_fn(out, batch)
            loss_mse = mse_loss_fn(out, batch)
            loss = alpha * loss_poisson + beta * loss_mse
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        progress_bar.progress((epoch + 1) / epochs)
    status_text.text("✅ Noisy bins identification complete!")
    return model


def reconstruct_curve(data, model, windows, device):
    """
    Reconstructs the input data from overlapping windows of the data.

    Parameters
    ----------
    data : array_like
        The input data.
    model : torch.nn.Module
        The model to use for reconstructing the data.
    windows : array_like
        The overlapping windows of data.
    device : torch.device
        The device to use for reconstructing the data.

    Returns
    -------
    error : array_like
        The absolute difference between the reconstructed data and the original data.
    threshold : float
        The threshold value to determine if a segment is considered 'good'.
        The threshold is calculated as the 90th percentile of the error array.
    bin_embeddings : array_like
        The embeddings of each bin calculated by summing the embeddings of the overlapping windows and averaged on the windows.
    """
    windows = torch.from_numpy(windows).float().to(device)
    model.eval()
    with torch.no_grad():
        recon_windows,latent_out = model(windows)
        recon_windows = recon_windows.cpu().numpy()
        latent_out = latent_out.cpu().numpy()
    latent_out = latent_out[:,:,0]
    recon_sum = np.zeros_like(data)
    counts_overlap = np.zeros(data.shape[1])
    latent_sum = np.zeros((data.shape[1],latent_out.shape[1]))
    window_size = 16
    stride = 8
    start_indices = range(0, data.shape[1] - window_size + 1, stride)
    for i, start in enumerate(start_indices):
        recon_sum[:, start : start + window_size] += recon_windows[i]
        counts_overlap[start : start + window_size] += 1
        for j in range(start, start + window_size):
            latent_sum[j] += latent_out[i]

    counts_overlap[counts_overlap == 0] = 1
    reconstructed_curve = recon_sum / counts_overlap
    bin_embeddings = latent_sum / counts_overlap[:, None]
    reconstructed_curve = np.clip(reconstructed_curve, a_min=0, a_max=None)
    error = np.abs(reconstructed_curve - data).sum(axis=0)
    threshold = np.percentile(error, 90)
    return error, threshold,bin_embeddings



def longest_good_segment(error_array, threshold):
    """
    Finds the longest contiguous segment where the error is below a given threshold.

    Parameters
    ----------
    error_array : array_like
        Array containing the error values for each segment.
    threshold : float
        The threshold value to determine if a segment is considered 'good'.

    Returns
    -------
    list
        Indices of the longest contiguous segment where the error is below the threshold.
        Returns an empty list if no such segment exists.
    """

    good = error_array < threshold
    best_start, best_end = None, None
    start = None
    for i, val in enumerate(good):
        if val:
            if start is None:
                start = i
        elif start is not None:
            if best_start is None or (i - start) > (best_end - best_start):
                best_start, best_end = start, i
            start = None

    if start is not None:
        if best_start is None or (len(good) - start) > (best_end - best_start):
            best_start, best_end = start, len(good)

    if best_start is not None:
        return list(range(best_start, best_end))
    else:
        return []


def find_noisy_bins(error, threshold):
    """
    Finds the indices of the 'noisy' bins in the given error array.

    Parameters
    ----------
    error : array_like
        Array containing the error values for each segment.
    threshold : float
        The threshold value to determine if a segment is considered 'noisy'.

    Returns
    -------
    tuple
        Tuple of two arrays. The first array contains the indices of the 'noisy'
        bins, and the second array contains the indices of the 'good' bins.
    """
    noisy_bins = np.where(error > threshold)[0]
    good_bins = np.where(error <= threshold)[0]
    return noisy_bins, good_bins

def filter_good_bins(binned_data, good_bins, noisy_bins, good_part):
    """
    Filters out 'good' bins that are not actually part of the longest good segment.

    Parameters
    ----------
    binned_data : array_like
        Array containing the binned data.
    good_bins : array_like
        Array containing the indices of the 'good' bins.
    noisy_bins : array_like
        Array containing the indices of the 'noisy' bins.
    good_part : array_like
        Array containing the indices of the longest good segment.

    Returns
    -------
    array_like
        Updated array of 'noisy' bins after filtering.
    """
    good_counts = binned_data[0, good_bins]
    mean_good_count = good_counts.mean()
    std_good_count = good_counts.std()
    counts = binned_data[0]
    noisy_bins = list(noisy_bins)
    good_bins = list(good_bins)
    for bin_idx in good_bins:
        if bin_idx not in good_part:
            if counts[bin_idx] > mean_good_count + std_good_count:
                noisy_bins.append(bin_idx)
                good_bins.remove(bin_idx)
    noisy_bins = np.array(noisy_bins)
    good_bins = np.array(good_bins) 
    print(f"Number of noisy bins: {len(noisy_bins)}")
    print(f"Number of good bins: {len(good_bins)}")
    return good_bins,noisy_bins

def calculate_reference_features(glowcurvedata, grid, good_part):
    """
    Calculates the reference features for a given set of photons.

    Parameters
    ----------
    glowcurvedata : pandas.DataFrame
        DataFrame containing the data from the FITS file.
    grid : array_like
        Array of bin edges.
    good_part : array_like
        Indices of the good bins.

    Returns
    -------
    tuple
        Tuple of three float values. The first value is the median time difference
        between consecutive photons in the 'good' bins, the second value is the median
        energy of photons in the 'good' bins with energy between 500 and 2000, and the
        third value is the median energy of photons in the 'good' bins with energy
        between 2000 and 10000.
    """
    t_start = grid[good_part[0]]
    t_end = grid[good_part[-1] + 1]
    photons_good = glowcurvedata[(glowcurvedata["TIME"] >= t_start) & (glowcurvedata["TIME"] < t_end)]
    times = np.sort(photons_good["TIME"].values)
    energies = photons_good["PI"].values
    intertbinG = np.diff(times)
    target = np.median(intertbinG) if len(intertbinG) > 0 else 0
    goodElow = energies[(energies > 500) & (energies < 2000)]
    goodEhigh = energies[(energies > 2000) & (energies < 10000)]
    targetElow = np.median(goodElow) if len(goodElow) > 0 else 0
    targetEhigh = np.median(goodEhigh) if len(goodEhigh) > 0 else 0
    return target, targetElow, targetEhigh

def rank_photons_by_similarity(target, targetElow, targetEhigh, noisy_photons, n):
    """
    Ranks the given noisy photons by their similarity to the target photon.

    The similarity is calculated as the sum of two scores: a time score and an energy score.
    The time score is the reciprocal of the absolute difference between the photon's time and
    the target time. The energy score is a normal distribution centered at the target energy
    with a standard deviation of 100 for energies between 500 and 2000 and a standard deviation
    of 500 for energies between 2000 and 10000.

    The top n photons by score are returned as a list of indices.

    Parameters
    ----------
    target : float
        The target time.
    targetElow : float
        The target energy for energies between 500 and 2000.
    targetEhigh : float
        The target energy for energies between 2000 and 10000.
    noisy_photons : pandas.DataFrame
        The DataFrame of noisy photons.
    n : int
        The number of top photons to return.

    Returns
    -------
    list
        A list of indices of the top n photons.
    """
    times = noisy_photons["TIME"].values
    energies = noisy_photons["PI"].values
    time_score = 1 / (1 + np.abs(times - target))
    energy_score = np.zeros_like(energies, dtype=float)

    low_mask = (energies > 500) & (energies <= 2000)
    high_mask = (energies > 2000) & (energies <= 10000)

    std_low = 100
    std_high = 500

    energy_score[low_mask] = np.exp(-((energies[low_mask] - targetElow) ** 2) / (2 * std_low**2))
    energy_score[high_mask] = np.exp( -((energies[high_mask] - targetEhigh) ** 2) / (2 * std_high**2))

    total_score = time_score + energy_score

    noisy_photons = noisy_photons.copy()
    noisy_photons["SCORE"] = total_score

    top_photons = noisy_photons.sort_values(by="SCORE", ascending=False)
    top_indices = top_photons.index[:n].tolist()
    return top_indices

def clean_noisy_bins(
        obs_id,
        filename,
        nbins,
        glowcurvedata,
        binned_data,
        grid,
        noisy_bins,
        good_bins,
        target,
        targetElow,
        targetEhigh,
    ):
    """
    Cleans the noisy bins in the observation data by adjusting the photon counts
    and identifying solar flare contaminated photons.

    Parameters
    ----------
    obs_id : str
        The observation ID.
    filename : str
        The name of the FITS file.
    nbins : int
        The number of bins.
    glowcurvedata : pandas.DataFrame
        DataFrame containing the data from the FITS file.
    binned_data : array_like
        The binned data array.
    grid : array_like
        The grid of time values.
    noisy_bins : array_like
        Indices of the noisy bins.
    good_bins : array_like
        Indices of the good part.
    target : float
        The target time for ranking photon similarity.
    targetElow : float
        The target energy for low energy photons.
    targetEhigh : float
        The target energy for high energy photons.

    Returns
    -------
    clean_curve : array_like
        The cleaned curve with adjusted photon counts for noisy bins.
    """

    good_counts = binned_data[0, good_bins]
    mean_good_count = good_counts.mean()

    noisy_photons = []

    new_noisy_counts = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("🔍 Identifying noisy bins...")

    for i, bin_idx in enumerate(noisy_bins):
        bin_idx = int(bin_idx)
        bin_count = binned_data[0, bin_idx]
        t_start = grid[bin_idx]
        t_end = grid[bin_idx + 1]

        bin_photons = glowcurvedata[(glowcurvedata["TIME"] >= t_start) & (glowcurvedata["TIME"] < t_end)]
        raw_target = max(mean_good_count, gamma * bin_count + (1 - gamma) * mean_good_count)
        n_target = int(np.clip(round(raw_target), 0, len(bin_photons)))
        new_noisy_counts.append(n_target)

        good_bin_photons = rank_photons_by_similarity(target, targetElow, targetEhigh, bin_photons, n_target)
        all_bin_photons_idx = bin_photons.index.values
        noisy_photons_idx = np.setdiff1d(all_bin_photons_idx, good_bin_photons)
        noisy_photons.extend(noisy_photons_idx)
        progress_bar.progress((i + 1) / len(noisy_bins))

    st.success("✅ Noisy bins cleaned!")

    is_flair = np.zeros(len(glowcurvedata))
    is_flair[noisy_photons] = 1

    clean_curve = binned_data[0].copy()
    clean_curve[noisy_bins] = new_noisy_counts
    glowcurvedata["IS_FLAIR"] = is_flair
    st.write(f"Number of photons containated by solar flares: {len(noisy_photons)}")
    t = Table.from_pandas(glowcurvedata)
    path = f"{obs_id}_{filename}_{nbins}_DPC.fits"
    t.write(path, overwrite=True)
    return path,clean_curve