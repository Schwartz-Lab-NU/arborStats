from scipy.spatial import ConvexHull
from pywarper.warpers import get_xy_profile
from scipy.ndimage import gaussian_filter
from pywarper.warpers import get_z_profile
import numpy as np
import skeliner as sk
from .utils.gauss import (
    fit_gauss2d_model,
    compute_gauss2d_size,
    compute_gauss2d_log_axis_ratio,
)



def compute_convex_hull_features(points):
    """
    Compute convex hull features for a set of 2D points.

    Parameters:
        points (np.ndarray): Array of shape (n_points, 2)

    Returns:
        dict: Dictionary with hull vertices, area, perimeter, and equivalent diameter
    """
    assert points.ndim == 2, f"Expected 2D points, got {points.ndim}D"
    assert points.shape[1] == 2, f"Expected points with 2 coordinates, got {points.shape[1]}"

    hull = ConvexHull(points)

    perimeter = hull.area  # In 2D, 'area' attribute gives the perimeter
    area = hull.volume  # In 2D, 'volume' attribute gives the area
    hull_points = points[hull.vertices, :2]

    return {
        'hull_points': hull_points,
        'hull_area': area,
        'hull_perimeter': perimeter,
        'hull_diameter': np.sqrt(4 * area / np.pi)
    }

def compute_density_xy(
    skel: sk.Skeleton,
    *,
    bounds: tuple,
    bin_size: float = 10.,
    margin_bins: int = 1,
    blur_sigma: float = 0.,
    **kwargs,
):
    xmin, xmax = skel.nodes[:, 0].min(), skel.nodes[:, 0].max()
    ymin, ymax = skel.nodes[:, 1].min(), skel.nodes[:, 1].max()

    xmin = (np.floor(xmin / bin_size) - margin_bins) * bin_size
    xmax = (np.ceil(xmax / bin_size) + margin_bins) * bin_size
    ymin = (np.floor(ymin / bin_size) - margin_bins) * bin_size
    ymax = (np.ceil(ymax / bin_size) + margin_bins) * bin_size

    extents = [max(xmin, bounds[0]), min(xmax, bounds[1]),
               max(ymin, bounds[2]), min(ymax, bounds[3])]

    xy_profile_dict = get_xy_profile(skel=skel, extents=extents, bin_size=bin_size, **kwargs)
    xmin, xmax, ymin, ymax = xy_profile_dict['extents']
    xy_dens = xy_profile_dict['histogram'].astype(np.float32)

    if blur_sigma > 0:
        xy_dens = gaussian_filter(xy_dens, sigma=blur_sigma, mode='constant', cval=0.0)

    return xy_dens, xmin, xmax, ymin, ymax

def compute_density_z(
    skel: sk.Skeleton,
    **kwargs,
) -> tuple[np.ndarray, float, float]:
    z_profile_dict = get_z_profile(skel=skel, **kwargs)
    zmin, zmax = z_profile_dict['extent']
    z_dens = z_profile_dict['histogram'].astype(np.float32)
    return z_dens, zmin, zmax


def compute_features_xy_gauss(dist2d, ext, plot_init=False, plot_fit=False):
    model, model_params, dist2d_fit, qi = fit_gauss2d_model(
        dist2d=dist2d, n_tries=10, early_stopping=0.9, plot_init=plot_init)

    # Get real coordinates of the fitted model
    model_params['x_mean_um'] = ((model_params['x_mean'] + 0.5) * (ext[1] - ext[0]) / dist2d.shape[1]) + ext[0]
    model_params['y_mean_um'] = ((model_params['y_mean'] + 0.5) * (ext[3] - ext[2]) / dist2d.shape[0]) + ext[2]
    model_params['x_stddev_um'] = model_params['x_stddev'] * (ext[1] - ext[0]) / dist2d.shape[1]
    model_params['y_stddev_um'] = model_params['y_stddev'] * (ext[3] - ext[2]) / dist2d.shape[0]

    r1 = model_params['x_stddev_um'] / model_params['y_stddev_um']
    r2 = model_params['x_stddev'] / model_params['y_stddev']

    assert np.isclose(r1, r2, rtol=1e-2, atol=1e-2), f"{r1} {r2} {dist2d.shape} {ext}"

    #if plot_fit:
    #    plot_dist2d_comparison(dist2d, dist2d_fit, model_params, qi, ext)
    #    plt.show()

    xy_gauss = dict()
    xy_gauss['model'] = model
    xy_gauss['model_params'] = model_params
    xy_gauss['dist2d'] = dist2d
    xy_gauss['dist2d_fit'] = dist2d_fit
    xy_gauss['extents'] = ext
    xy_gauss['qi'] = qi

    xy_gauss['area_um2'], xy_gauss['diameter_um'] = compute_gauss2d_size(
        a=model_params['x_stddev_um'] * 2,
        b=model_params['y_stddev_um'] * 2)

    xy_gauss['log_axis_ratio'] = compute_gauss2d_log_axis_ratio(
        a=model_params['x_stddev_um'],
        b=model_params['y_stddev_um'])

    return xy_gauss


def compute_gauss_xy_stats(xy_soma, xy_gauss_mean, xy_gauss_std, xy_gauss_theta):
    # 1. Get the direction vector from soma to mean
    direction = xy_gauss_mean - xy_soma
    direction_norm = direction / np.linalg.norm(direction)

    xy_gauss_soma_angle_deg = np.rad2deg(np.arctan2(direction[1], direction[0])) % 360

    # 2. Get the perpendicular direction (rotate by 90 degrees)
    perpendicular_norm = np.array([-direction_norm[1], direction_norm[0]])

    # 3. Transform the perpendicular direction to the ellipse's coordinate system
    # The ellipse is defined by a rotation angle xy_gauss_theta
    cos_theta = np.cos(xy_gauss_theta)
    sin_theta = np.sin(xy_gauss_theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Transform perpendicular direction to ellipse coordinates
    perp_in_ellipse = rotation_matrix.T @ perpendicular_norm
    dir_in_ellipse = rotation_matrix.T @ direction_norm

    # 4. Calculate the extent along the perpendicular direction
    # In the ellipse's coordinate system, the semi-axes are xy_gauss_std
    # The extent in any direction (u, v) is: 1/sqrt((u/a)^2 + (v/b)^2)
    a, b = xy_gauss_std[0], xy_gauss_std[1]
    xy_gauss_proj_width = 2 / np.sqrt((perp_in_ellipse[0] / a) ** 2 + (perp_in_ellipse[1] / b) ** 2)
    xy_gauss_proj_length = 2 / np.sqrt((dir_in_ellipse[0] / a) ** 2 + (dir_in_ellipse[1] / b) ** 2)

    results = dict(
        xy_gauss_soma_d_um=float(np.linalg.norm(direction)),
        xy_gauss_soma_dx_um=float(direction[0]),
        xy_gauss_soma_dy_um=float(direction[1]),
        xy_gauss_soma_angle_deg=float(xy_gauss_soma_angle_deg),
        xy_gauss_proj_width=float(xy_gauss_proj_width),
        xy_gauss_proj_length=float(np.asarray(xy_gauss_proj_length)),
    )

    return results


def compute_percentiles_from_histogram(bin_edges, frequencies, percentiles=None):
    """
    Compute percentiles from histogram data using linear interpolation.

    Parameters:
    -----------
    bin_edges : array-like
        The edges of the histogram bins (length = n_bins + 1)
    frequencies : array-like
        The frequency count for each bin (length = n_bins)
    percentiles : array-like, optional
        Percentile values to compute (0-100). If None, computes quartiles.
        Examples: [25, 50, 75] for quartiles, [10, 90] for deciles

    Returns:
    --------
    dict : Dictionary containing computed percentiles
           Keys are 'P{percentile}' (e.g., 'P25', 'P50', 'P95')
    """
    if percentiles is None:
        percentiles = [25, 50, 75]

    percentiles = np.array(percentiles)

    # Validate percentiles
    if np.any(percentiles < 0) or np.any(percentiles > 100):
        raise ValueError("Percentiles must be between 0 and 100")

    frequencies = np.array(frequencies)
    bin_edges = np.array(bin_edges)

    # Calculate bin widths
    bin_widths = np.diff(bin_edges)

    # Calculate cumulative frequencies
    cumsum = np.cumsum(frequencies)
    n = cumsum[-1]

    if n == 0:
        raise ValueError("Total frequency is zero")

    def interpolate_percentile(percentile):
        """Interpolate to find the percentile value"""
        # Calculate the position for this percentile
        position = (percentile / 100) * n

        # Find the bin containing this position
        bin_idx = np.searchsorted(cumsum, position, side='right')

        # Handle edge cases
        if bin_idx == 0:
            cf_prev = 0
        else:
            cf_prev = cumsum[bin_idx - 1]

        # Handle case where position is beyond all data
        if bin_idx >= len(bin_edges) - 1:
            return bin_edges[-1]

        # Linear interpolation within the bin
        L = bin_edges[bin_idx]  # Lower boundary
        f = frequencies[bin_idx]  # Frequency in this bin
        w = bin_widths[bin_idx]  # Width of bin

        if f == 0:
            return L

        percentile_value = L + ((position - cf_prev) / f) * w
        return percentile_value

    results = {}
    for p in percentiles:
        results[p] = interpolate_percentile(p)

    return results
