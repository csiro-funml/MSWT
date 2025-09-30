import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.fft as fft

import xarray
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_contour(z, func=plt.imshow, **kwargs):
    if isinstance(z, torch.Tensor):
        z = z.cpu().numpy()
    _, ax = plt.subplots(figsize=(3, 3))
    f = func(z, cmap=sns.cm.icefire)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.1)
    cbar = plt.colorbar(f, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.locator_params(nbins=9)
    cbar.update_ticks()


def plot_contour_plotly(
    z,
    colorscale="RdYlBu",
    showscale=False,
    showlabels=False,
    continuous_coloring=False,
    reversescale=True,
    dimensions=(200, 200),
    line_smoothing=0.7,
    ncontours=20,
    **plot_kwargs,
):
    """
    show 2D solution z of its contour
    colorscale: balance (MATLAB new) or Jet (MATLAB old)
    """

    if not plot_kwargs:
        plot_kwargs = dict(
            contour_kwargs=dict(
                colorscale=colorscale,
                line_smoothing=line_smoothing,
                line_width=0.1,
                ncontours=ncontours,
                reversescale=reversescale,
                # )
            ),
            figure_kwargs=dict(
                layout={
                    "xaxis": {
                        "title": "x-label",
                        "visible": False,
                        "showticklabels": False,
                    },
                    "yaxis": {
                        "title": "y-label",
                        "visible": False,
                        "showticklabels": False,
                    },
                }
            ),
            layout_kwargs=dict(
                margin=dict(l=0, r=0, t=0, b=0),
                width=dimensions[0],
                height=dimensions[1],
                template="plotly_white",
            ),
        )

    contour_kwargs = plot_kwargs["contour_kwargs"]
    figure_kwargs = plot_kwargs["figure_kwargs"]
    layout_kwargs = plot_kwargs["layout_kwargs"]
    if showscale:
        contour_kwargs["showscale"] = True
        contour_kwargs["colorbar"] = dict(
            thickness=0.15 * layout_kwargs["height"],
            tickwidth=0.3,
            exponentformat="e",
        )
        layout_kwargs["width"] = 1.32 * layout_kwargs["height"]
    else:
        contour_kwargs["showscale"] = False

    if continuous_coloring:
        contour_kwargs["contours_coloring"] = "heatmap"

    if showlabels:
        contour_kwargs["contours"] = dict(
            coloring="heatmap",
            showlabels=True,  # show labels on contours
            labelfont=dict(  # label font properties
                size=12,
                color="gray",
            ),
        )

    uplot = go.Contour(z=z, **contour_kwargs)
    fig = go.Figure(data=uplot, **figure_kwargs)
    if "template" not in layout_kwargs.keys():
        fig.update_layout(template="plotly_dark", **layout_kwargs)
    else:
        fig.update_layout(**layout_kwargs)
    return fig


def get_enstrophy_spectrum(vorticity, h):
    if isinstance(vorticity, np.ndarray):
        vorticity = torch.from_numpy(vorticity)
    n = vorticity.shape[0]
    kx = fft.fftfreq(n, d=h)
    ky = fft.fftfreq(n, d=h)
    kx, ky = torch.meshgrid([kx, ky], indexing="ij")
    kmax = n // 2
    kx = kx[..., : kmax + 1]
    ky = ky[..., : kmax + 1]
    k2 = (4 * torch.pi**2) * (kx**2 + ky**2)
    k2[0, 0] = 1.0

    wh = fft.rfft2(vorticity)

    tke = (0.5 * wh * wh.conj()).real
    kmod = torch.sqrt(k2)
    k = torch.arange(1, kmax, dtype=torch.float64)  # Nyquist limit for this grid
    Ens = torch.zeros_like(k)
    dk = (torch.max(k) - torch.min(k)) / (2 * n)
    for i in range(len(k)):
        Ens[i] += (tke[(kmod < k[i] + dk) & (kmod >= k[i] - dk)]).sum()

    Ens = Ens / Ens.sum()
    return Ens


def plot_enstrophy_spectrum(
    fields: list,
    h=None,
    slope=5,
    factor=None,
    cutoff=1e-15,
    plot_cutoff_factor=1 / 8,
    labels=None,
    title=None,
    legend_loc="upper right",
    fontsize=15,
    subplot_kw={"figsize": (5, 5), "dpi": 100, "facecolor": "w"},
    log_path=None,
    model_name=None,
    **kwargs,
):
    for k, field in enumerate(fields):
        if isinstance(field, np.ndarray):
            fields[k] = torch.from_numpy(field)
    if labels is None:
        labels = [f"Field {i}" for i in range(len(fields))]
    n = fields[0].shape[0]
    if h is None:
        h = 1 / n
    kmax = n // 2
    k = torch.arange(1, kmax, dtype=torch.float64)  # Nyquist limit for this grid
    Es = [get_enstrophy_spectrum(field, h) for field in fields]
    if factor is None:
        factor = Es[-1].quantile(0.8) / (k[-1] ** (-slope))
        # print(factor)

    fig, ax = plt.subplots(**subplot_kw)
    plot_cutoff = int(n * plot_cutoff_factor)
    for i, E in enumerate(Es):
        if cutoff is not None:
            E[E < cutoff] = np.nan
        E[-plot_cutoff:] = np.nan
        plt.loglog(k, E, label=f"{labels[i]}")

    plt.loglog(
        k[:-plot_cutoff],
        (factor * k ** (-slope))[:-plot_cutoff],
        "b--",
        label=f"$O(k^{{{-slope:.3g}}})$",
    )
    plt.grid(True, which="both", ls="--", linewidth=0.4)
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.legend(fontsize=fontsize, loc=legend_loc)
    plt.title(title, fontsize=fontsize)
    plt.xlabel("Wavenumber", fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    plt.savefig(f'{log_path}/spectral_error/{model_name}_{title}_enstrophy_spectrum.png')


def plot_contour_trajectory(
    field,
    num_snapshots=5,
    col_wrap=5,
    contourf=False,
    T_start=4.5,
    dt=1e-1,
    title=None,
    cb_kws=dict(orientation="vertical", pad=0.01, aspect=10),
    subplot_kws=dict(
        xticks=[],
        yticks=[],
        ylabel="",
        xlabel="",
    ),
    **plot_kws,
):
    """
    plot trajectory using xarray's imshow or contourf wrapper
    """
    field = field.detach().cpu().numpy()
    *size, T = field.shape
    grid = np.linspace(0, 1, size[0] + 1)[:-1]
    time = np.arange(T) * dt + T_start
    coords = {
        "x": grid,
        "y": grid,
        "t": time,
    }
    ds = xarray.DataArray(field, dims=["x", "y", "t"], coords=coords)
    t_steps = T // num_snapshots
    T_rem = T % num_snapshots
    ds = ds.isel(t=slice(T_rem, None)).thin({"t": t_steps})
    plot_func = ds.plot.contourf if contourf else ds.plot.imshow

    # fig = plt.figure()
    _plot_kws = dict(
        col_wrap=col_wrap,
        cmap=sns.cm.icefire,
        interpolation="hermite",
        robust=True,
        add_colorbar=True,
        xticks=None,
        yticks=None,
        size=3,
        aspect=1,
    )
    _plot_kws.update(plot_kws)

    im = plot_func(
        col="t",
        subplot_kws=subplot_kws,
        cbar_kwargs=cb_kws,
        **_plot_kws,
    )
    if title is not None:
        im.fig.suptitle(title, y=0.05)
    # plt.show()

    return im



# Define the function to compute the spectrum
def spectrum_2d(signal, n_observations, normalize=True):
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Paramaters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.
    normalize: bool
        whether to apply normalization to the output of the 2D FFT. 
        If True, normalizes the outputs by ``1/n_observations``
        (actually ``1/sqrt(n_observations * n_observations)``). 
    Returns
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
        The spectrum is computed using a square approximation to radial
        binning, meaning that the wavenumber 'bin' into which a particular 
        coefficient is the coefficient's location along the diagonal, indexed 
        from the top-left corner of the 2d FFT output. 
    """
    T = signal.shape[0]
    signal = signal.view(T, n_observations, n_observations)

    if normalize:
        signal = torch.fft.fft2(signal, norm="ortho")
    else:
        signal = torch.fft.rfft2(
            signal, s=(n_observations, n_observations), norm="backward"
        )

    # 2d wavenumbers following PyTorch fft convention
    k_max = n_observations // 2
    wavenumers = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1),
            torch.arange(start=-k_max, end=0, step=1),
        ),
        0,
    ).repeat(n_observations, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers

    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k

    # Remove symmetric components from wavenumbers
    index = -1.0 * torch.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index[0:k_max1, 0:k_max1] = sum_k[0:k_max1, 0:k_max1]

    spectrum = torch.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = torch.where(index == j)
        spectrum[:, j - 1] = (signal[:, ind[0], ind[1]].sum(dim=1)).abs() ** 2

    spectrum = spectrum.mean(dim=0)
    return spectrum



def compare_two_spectral():
    import scipy.stats as stats

    # function 1 old spectral energy:
    def get_spectral_energy(y, H, W):

        # Use full 2D FFT so the spectrum shape matches (H, W)
        y_fft = torch.abs(torch.fft.fft2(y))

        # Take magnitude and move to numpy for binning
        fourier_amplitudes = y_fft.detach().cpu().numpy()

        # Create the k-frequency grid for rectangular image
        kfreq_x = np.fft.fftfreq(W) * W
        kfreq_y = np.fft.fftfreq(H) * H
        kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

        # Flatten the arrays to use in binning (1D arrays of equal length,  (H*W,) ) 
        knrm = knrm.ravel() # ALL the frequences in the image
        fourier_amplitudes = fourier_amplitudes.ravel() # ALL the fourier amplitudes in the image

        # Define the bins for the wavenumber - use the minimum dimension for binning
        min_dim = min(H, W)
        kbins = np.arange(1, min_dim // 2 + 1, 1.0)

        # Bin the data (radial mean), turn the 2D array into 1D array
        E_freq, _, _ = stats.binned_statistic(
            knrm, fourier_amplitudes, statistic="mean", bins=kbins
        )


        log_E_freq = np.log(E_freq)
        log_freq = np.log(kbins[:len(log_E_freq)])
        k_freq = kbins[:len(E_freq)]
        return E_freq

    

    def get_enstrophy_spectrum_testing(y, h):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        n = y.shape[0]
        kx = fft.fftfreq(n, d=h)
        ky = fft.fftfreq(n, d=h)
        kx, ky = torch.meshgrid([kx, ky], indexing="ij")
        kmax = n // 2
        kx = kx[..., : kmax + 1]
        ky = ky[..., : kmax + 1]
        k2 = (4 * torch.pi**2) * (kx**2 + ky**2)
        k2[0, 0] = 1.0

        wh = fft.rfft2(y)

        tke = (0.5 * wh * wh.conj()).real
        kmod = torch.sqrt(k2)
        k = torch.arange(1, kmax, dtype=torch.float64)  # Nyquist limit for this grid
        Ens = torch.zeros_like(k)
        dk = (torch.max(k) - torch.min(k)) / (2 * n)
        for i in range(len(k)):
            Ens[i] += (tke[(kmod < k[i] + dk) & (kmod >= k[i] - dk)]).sum()

        # Ens = Ens / Ens.sum()
        return Ens
    
    H = 128
    h = 2 * np.pi / H
    y = torch.rand(H, H)
    E_freq = get_spectral_energy(y, H, H)
    Ens = get_enstrophy_spectrum_testing(y, h)
    E_spectrum_fno = spectrum_2d(y.unsqueeze(0), H)
    plt.loglog(E_freq, label='Spectral Energy')
    plt.loglog(Ens, label='Enstrophy Spectrum')
    plt.loglog(E_spectrum_fno, label = 'FNO Spectrum')
    plt.legend()
    plt.show()
    print("")

if __name__ == '__main__':
    compare_two_spectral()





