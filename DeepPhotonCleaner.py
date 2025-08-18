import pandas as pd
import numpy as np
from astropy.io import fits
import torch
import random
import os
import platform
import cpuinfo
from torch.utils.data import TensorDataset,DataLoader
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