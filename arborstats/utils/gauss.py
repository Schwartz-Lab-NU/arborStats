import warnings

import numpy as np
from astropy.modeling.fitting import SLSQPLSQFitter
from astropy.modeling.functional_models import Gaussian2D
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter


def compute_explained_var(dist: np.ndarray, dist_fit: np.ndarray):
    var_dist = np.var(dist)
    if var_dist == 0:
        return 0.
    return 1. - (np.var(dist - dist_fit) / var_dist)


def gauss2d_init_params(dist2d, amplimscale=2., plot=False):
    assert dist2d.ndim == 2
    y0, x0 = np.unravel_index(np.argmax(dist2d), shape=dist2d.shape)

    x0 = x0
    y0 = y0

    dist2d_cut = np.clip(dist2d[y0], 0, None)

    std_left = np.append(np.where((dist2d_cut[:x0 + 1][::-1]) < (0.1 * dist2d_cut[x0]))[0], 1)[0]
    std_right = np.append(np.where((dist2d_cut[x0:]) < (0.1 * dist2d_cut[x0]))[0], 1)[0]

    std0 = np.mean([std_left, std_right])

    xlim = (0, dist2d.shape[1])
    ylim = (0, dist2d.shape[0])

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        im = axs[0].imshow(dist2d, vmin=-np.max(np.abs(dist2d)), vmax=np.max(np.abs(dist2d)), cmap='coolwarm',
                           origin='lower')
        plt.colorbar(im, ax=axs[0])
        axs[0].plot(x0, y0, 'kX')

        # Plot ellipse with center x0, y0 and std0 as radius
        axs[0].add_patch(Ellipse(
            xy=(x0, y0),
            width=2 * std0,
            height=2 * std0,
            angle=0, color='k', fill=False, lw=1.5))

        axs[1].plot(dist2d_cut)
        axs[1].axhline(dist2d[y0, x0], c='k')
        plt.axvline(x=x0 - std_left, c='r')
        plt.axvline(x=x0, c='k')
        plt.axvline(x=x0 + std_right, c='r')
        plt.show()

    amp0 = np.max(dist2d)
    assert amp0 > 0, amp0
    amplim = (0., amplimscale * amp0)

    return x0, y0, amp0, std0, xlim, ylim, amplim


def gauss2d_model(dist2d, x0, y0, amp0, std0_x, std0_y, theta0, xlim, ylim, amplim, fixed_params=None):
    assert dist2d.ndim == 2, 'Provide 2d data'

    model = Gaussian2D(
        amplitude=amp0, x_mean=x0, y_mean=y0, x_stddev=std0_x, y_stddev=std0_y, theta=theta0,
        bounds={'amplitude': amplim, 'x_mean': xlim, 'y_mean': ylim},
        fixed=fixed_params if fixed_params is not None else dict())

    return model


def fit_gauss2d_model(dist2d, n_tries=10, early_stopping=0.9, plot_init=False):
    xx, yy = np.meshgrid(np.arange(0, dist2d.shape[1]), np.arange(0, dist2d.shape[0]))

    dist2d_smooth = gaussian_filter(input=dist2d, sigma=1., mode='constant')

    # Get base model for smoothed data
    x0, y0, amp0, std0, xlim, ylim, amplim = gauss2d_init_params(
        dist2d=dist2d_smooth, amplimscale=2., plot=plot_init)

    std0_x = std0
    std0_y = std0
    theta0 = 0.0

    model = gauss2d_model(dist2d, x0=x0, y0=y0, amp0=amp0, std0_x=std0_x, std0_y=std0_y,
                          theta0=theta0, xlim=xlim, ylim=ylim, amplim=amplim)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SLSQPLSQFitter()(model=model, x=xx, y=yy, z=dist2d_smooth, verblevel=0)

    best_model_params = {k: v for k, v in zip(model.param_names, model.param_sets.flatten())}
    best_model_fit = model(xx, yy)
    best_qi = compute_explained_var(dist2d, best_model_fit)
    best_model = model

    for i in range(n_tries):
        x0_n = np.clip(np.random.normal(x0, 0.03 * xlim[1] - xlim[0]), xlim[0], xlim[1])
        y0_n = np.clip(np.random.normal(y0, 0.03 * ylim[1] - ylim[0]), ylim[0], ylim[1])
        std0_x_n = np.random.uniform(0.25 * std0_y, 4 * std0_y)
        std0_y_n = np.random.uniform(0.25 * std0_y, 4 * std0_y)
        theta0_n = np.random.uniform(-np.pi, np.pi)
        amp0_n = np.random.uniform(0.25 * amp0, 4 * amp0)

        model = gauss2d_model(dist2d, x0=x0_n, y0=y0_n, amp0=amp0_n, std0_x=std0_x_n, std0_y=std0_y_n,
                              theta0=theta0_n, xlim=xlim, ylim=ylim, amplim=amplim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SLSQPLSQFitter()(model=model, x=xx, y=yy, z=dist2d_smooth if i == 0 else dist2d,
                                     verblevel=0)

        model_params = {k: v for k, v in zip(model.param_names, model.param_sets.flatten())}
        model_fit = model(xx, yy)
        qi = compute_explained_var(dist2d, model_fit)

        if qi > best_qi:
            best_qi = qi
            best_model = model
            best_model_params = model_params
            best_model_fit = model_fit

        if best_qi > early_stopping:
            break

    return best_model, best_model_params, best_model_fit, best_qi


def get_dist2d_extent(dist2d_shape, pixelsize=None):
    pixelsize = 1 if pixelsize is None else pixelsize
    return np.array(
        [-dist2d_shape[1] / 2., dist2d_shape[1] / 2., -dist2d_shape[0] / 2., dist2d_shape[0] / 2.]) * pixelsize


def compute_gauss2d_size(a, b):
    area = np.pi * a * b
    diameter = np.sqrt(area / np.pi) * 2

    return area, diameter


def compute_gauss2d_log_axis_ratio(a, b):
    if b > a:
        a, b = b, a
    return np.log(a / b)


def plot_dist2d(dist2d, ax=None, vmin=None, vmax=None, pixelsize=None, extent=None, plot_cb=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if vmin is None and vmax is None:
        vabsmax = np.nanmax(np.abs(dist2d))
        vmin = -vabsmax
        vmax = vabsmax
        cmap = 'coolwarm'
    elif vmin == 0 and vmax is None:
        vmin = 0
        vmax = np.nanmax(dist2d)
        cmap = 'viridis'
    else:
        cmap = 'viridis'

    if extent is None and pixelsize is not None:
        extent = get_dist2d_extent(dist2d_shape=dist2d.shape, pixelsize=pixelsize)

    ax.set(title='Dist.')
    im = ax.imshow(dist2d, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='none', origin='lower', extent=extent)
    if plot_cb:
        plt.colorbar(im, ax=ax)

    return ax


def plot_dist2d_gauss_fit(
        ax, dist2d=None, vmin=None, vmax=None, dist2d_params=None, n_std=2, color='k', ms=3,
        plot_cb=False, extent=None, pixelsize=None, **kwargs):
    if dist2d_params is not None:
        ax.plot(dist2d_params['x_mean_um'], dist2d_params['y_mean_um'], zorder=100, marker='x', ms=ms, c=color,
                **kwargs)
        ax.add_patch(Ellipse(
            xy=(dist2d_params['x_mean_um'], dist2d_params['y_mean_um']),
            width=n_std * 2 * dist2d_params['x_stddev_um'],
            height=n_std * 2 * dist2d_params['y_stddev_um'],
            angle=np.rad2deg(dist2d_params['theta']),  # Matplotlib Ellipse expects CCW (anti-clockwise) degrees
            color=color, fill=False, **kwargs))

    if dist2d is not None:
        if vmax is None:
            vmax = np.max(dist2d)
        cmap = 'viridis'

        if extent is None and pixelsize is not None:
            extent = get_dist2d_extent(dist2d_shape=dist2d.shape, pixelsize=pixelsize)

        im = ax.imshow(dist2d, vmin=vmin, vmax=vmax, cmap=cmap, zorder=0, origin='lower', extent=extent)
        if plot_cb:
            plt.colorbar(im, ax=ax)


def plot_dist2d_comparison(dist2d, dist2d_fit, dist2d_params, qi, dist2d_extent, skel=None, soma_xy=None):
    vmax = np.maximum(np.max(np.abs(dist2d)), np.max(np.abs(dist2d_fit)))

    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)

    if soma_xy is not None:
        for ax in axs:
            ax.plot(soma_xy[0], soma_xy[1], 'ro', ms=5, zorder=200)

    ax = axs[0]
    plot_dist2d(dist2d, ax=ax, vmin=0, vmax=vmax, extent=dist2d_extent, plot_cb=False)
    ax.set(title='d (XY data)', ylabel='X', xlabel='Y')
    if skel is not None:
        import skeliner as sk
        sk.plot.projection(ax=ax, skel=skel, plane='xy')

    ax = axs[1]
    plot_dist2d_gauss_fit(ax, dist2d_fit, dist2d_params=dist2d_params, vmin=0, vmax=vmax, plot_cb=False,
                          extent=dist2d_extent)
    ax.set_title('m (model fit)')

    ax = axs[2]
    plot_dist2d(dist2d_fit - dist2d, ax=ax, vmin=None, vmax=None, extent=dist2d_extent)
    ax.set_title(f'm - d: QI={qi:.2f}')

    ax = axs[3]
    plot_dist2d_gauss_fit(ax, dist2d=dist2d, dist2d_params=dist2d_params, vmin=0, vmax=vmax, plot_cb=True,
                          extent=dist2d_extent)
    sd_a = np.maximum(dist2d_params['x_stddev_um'], dist2d_params['y_stddev_um'])
    sd_b = np.minimum(dist2d_params['x_stddev_um'], dist2d_params['y_stddev_um'])
    ax.set(title=f"sd-A={sd_a:.1f}\nsd-B={sd_b:.1f}")

    for ax in axs:
        ax.set_aspect('equal', 'box')
        ax.set(xlim=dist2d_extent[0:2], ylim=dist2d_extent[2:4])

    plt.tight_layout()
    return fig, axs