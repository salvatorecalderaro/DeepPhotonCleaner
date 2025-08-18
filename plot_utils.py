import os
import plotly.graph_objects as go

def plot_noisy_curve(obs_id, filename, nbins, grid, binned_data):

    """
    Plot the noisy curve of the observation.

    Parameters
    ----------
    obs_id : str
        The observation ID.
    filename : str
        The name of the FITS file.
    nbins : int
        The number of bins.
    grid : array_like
        The grid of time values.
    binned_data : array_like
        The binned data.

    Returns
    -------
    None
    """
    folder = f"results/{obs_id}"

    os.makedirs(folder, exist_ok=True)

    bin_centers = (grid[:-1] + grid[1:]) / 2
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=bin_centers, y=binned_data[0, :], mode="lines+markers", name="Counts")
    )

    fig.update_layout(
        title=f"Observation {obs_id} - {filename}, Noisy Curve N.Bins={nbins}",
        xaxis_title="Time",
        yaxis_title="Counts",
        template="plotly_white",
    )

    return fig 