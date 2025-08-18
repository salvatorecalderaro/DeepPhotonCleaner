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


def plot_results(obs_id, filename, nbins, grid, data, cleaned_curve, good_bins):
    """
    Plots the original curve and the cleaned curve using DeepPhotonCleaner.
    Adds a green shaded region to highlight the good part of the curve.
    Saves the plot to a HTML file in the results directory.

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
    data : array_like
        The original binned data array.
    cleaned_curve : array_like
        The cleaned binned data array.
    good_bins : array_like
        The indices of the good bins.

    Returns
    -------
    None
    """
    bin_centers = (grid[:-1] + grid[1:]) / 2
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=bin_centers, y=data[0], name="Original curve",mode="lines+markers")
    )

    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=cleaned_curve,
            name="Clean curve (DPC)",
            line=dict(dash="dot"),
            mode="lines+markers"
        )
    )

    if good_bins is not None and len(good_bins) > 0:
        t_start = float(bin_centers[min(good_bins)])
        t_end = float(bin_centers[max(good_bins)])
        fig.add_vrect(
            x0=t_start,
            x1=t_end,
            fillcolor="green",
            opacity=0.4,
            layer="below",
            line_width=0,
        )
    title = f" DeepPhotonCleaner - Observation {obs_id}  {filename}, N.Bins={nbins}"
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Counts",
        template="plotly_white",
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="green", opacity=0.4),
            name="Good part",  
        )
    )
    return fig