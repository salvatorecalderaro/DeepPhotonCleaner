import pandas as pd
import numpy as np
from astropy.io import fits
import torch
import random
import os
import platform
import cpuinfo

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