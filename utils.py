import h5py as h5
import torch


def equal_width_bins(data, n_bins):
    """
    yields bins from [0, nbins) where each bin has equal width
    Accepts data optionally in batch format (n_batch, n_data).
    Bins are calculated on a per-row basis

    Input should be of the form (n_batch, n_data)
    """
    epsilon = 1e-6  # make sure we do not have an extra bin

    edges = torch.vstack(
        [
            torch.linspace(data[i, :].min(), data[i, :].max(), n_bins + 1)
            for i in range(data.shape[0])
        ]
    )
    edges[..., 0] -= epsilon
    return torch.searchsorted(edges, data, side="left") - 1


def equal_area_bins(data, n_bins, xiles=None):
    """
    yields n_bins bins from [0, n_bins) where each bin has equal width

    Accepts data optionally in batch format (n_batch, n_data).
    Bins are calculated on a per-row basis
    """

    epsilon = 1e-6  # make sure we do not have an extra bin

    xiles = torch.linspace(0, 1, n_bins + 1)
    edges = torch.quantile(data, xiles, dim=1).T
    edges[..., 0] -= epsilon
    return torch.searchsorted(edges, data, side="left") - 1


def inspect_h5(h, tab=0):
    print("\t" * tab, h)
    if len(h.attrs.keys()) > 0:
        print("\t" * tab, "  Attrs: ", ", ".join(h.attrs.keys()))
    if isinstance(h, h5.Group) or isinstance(h, h5.File):
        for key in h.keys():
            inspect_h5(h[key], tab + 1)
