#!/usr/bin/env python3
"""
Stack (sum) spectra from a rectangular pixel region in an MPDAF cube, with
optional wavelength restriction, de-redshift, and a two-Gaussian + linear
baseline fit overplotted on the flux spectrum.

Requirements: pip install mpdaf matplotlib numpy scipy
"""

import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from mpdaf.obj import Cube

LINE_OFFSET_ANGSTROM = 14.4   # cen2 = cen1 + LINE_OFFSET_ANGSTROM (fixed)
MAX_SIGMA_ANGSTROM = 5.0      # sigma1, sigma2 <= this
MIN_AMP_RATIO = 0.1           # amp2/amp1 bounds; enforces mutual >=0.1x
MAX_AMP_RATIO = 10.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Stack spectra in a rectangular region from MPDAF data and variance cubes."
    )
    p.add_argument("data_cube", help="Path to flux/data cube FITS file")
    p.add_argument("var_cube", help="Path to variance cube FITS file")
    p.add_argument("--data-ext", type=int, required=True, help="FITS extension for flux cube")
    p.add_argument("--var-ext", type=int, required=True, help="FITS extension for variance cube")
    p.add_argument("--x0", type=float, required=True, help="Rectangle center x (pixel)")
    p.add_argument("--y0", type=float, required=True, help="Rectangle center y (pixel)")
    p.add_argument("--width", type=float, required=True, help="Rectangle width (pixels)")
    p.add_argument("--height", type=float, required=True, help="Rectangle height (pixels)")
    p.add_argument("--wmin", type=float, default=None, help="Min wavelength, observed frame")
    p.add_argument("--wmax", type=float, default=None, help="Max wavelength, observed frame")
    p.add_argument("--redshift", type=float, default=0.0,
                   help="Redshift z; rest wavelength = obs / (1+z). Also sets sigma floor.")
    p.add_argument("--fit-gaussians", action="store_true",
                   help="Fit two-Gaussian + linear baseline model and overplot it.")
    p.add_argument("--line1-guess", type=float, default=None,
                   help="Initial guess for Gaussian 1 central wavelength.")
    p.add_argument("--outplot", default="stacked_spectrum.png", help="Output plot filename")
    p.add_argument("--outspec", default="stacked_spectrum.txt", help="Output ASCII spectrum table")
    return p.parse_args()


def rect_to_pixel_indices(x0, y0, width, height, nx, ny):
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0 (got width=%s, height=%s)" % (width, height))

    x1f, x2f = x0 - width / 2.0, x0 + width / 2.0
    y1f, y2f = y0 - height / 2.0, y0 + height / 2.0

    x1, x2 = int(round(x1f)), int(round(x2f))
    y1, y2 = int(round(y1f)), int(round(y2f))

    x1c, x2c = max(0, x1), min(nx, x2)
    y1c, y2c = max(0, y1), min(ny, y2)

    if x2c <= x1c or y2c <= y1c:
        raise ValueError(
            "Requested rectangle does not overlap the cube's pixel grid. "
            "Requested x=[%.2f,%.2f], y=[%.2f,%.2f]; cube bounds x=[0,%d], y=[0,%d]"
            % (x1f, x2f, y1f, y2f, nx, ny)
        )

    if (x1, x2, y1, y2) != (x1c, x2c, y1c, y2c):
        print("[WARN] Rectangle clipped to cube bounds: requested x=[%d,%d] y=[%d,%d] -> "
              "clipped x=[%d,%d] y=[%d,%d]" % (x1, x2, y1, y2, x1c, x2c, y1c, y2c))

    return y1c, y2c, x1c, x2c


def stack_region(cube, y1, y2, x1, x2, wmin=None, wmax=None):
    if (wmin is None) ^ (wmax is None):
        raise ValueError("Please provide both --wmin and --wmax, or neither.")
    if (wmin is not None) and (wmax is not None) and (wmin >= wmax):
        raise ValueError("wmin must be < wmax (got wmin=%s, wmax=%s)" % (wmin, wmax))

    wave_full = cube.wave.coord()

    if wmin is None:
        wave_mask = np.ones_like(wave_full, dtype=bool)
    else:
        wave_mask = (wave_full >= wmin) & (wave_full <= wmax)
        if not np.any(wave_mask):
            raise ValueError(
                "No wavelength points between wmin=%s and wmax=%s. Cube range %.2f-%.2f"
                % (wmin, wmax, wave_full.min(), wave_full.max())
            )

    data = cube.data
    data_region = data[wave_mask, y1:y2, x1:x2]
    if np.ma.isMaskedArray(data_region):
        data_region = data_region.filled(np.nan)
    data_region = np.asarray(data_region, dtype=float)

    spec_sum = np.nansum(data_region, axis=(1, 2))
    wave = wave_full[wave_mask]
    return wave, spec_sum


def deredshift_wavelength(wave_obs, redshift):
    if redshift <= -1.0:
        raise ValueError("redshift must be > -1 (got redshift=%s)" % redshift)
    return np.asarray(wave_obs, dtype=float) / (1.0 + redshift)


def clean_series(wave, y):
    wave = np.asarray(wave, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(wave) & np.isfinite(y)
    return wave[mask], y[mask]


def robust_ylim(y, lo=1.0, hi=99.0, pad_frac=0.10):
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return -1.0, 1.0
    yy = y[finite]
    p_lo, p_hi = np.nanpercentile(yy, lo), np.nanpercentile(yy, hi)
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        ymin, ymax = np.nanmin(yy), np.nanmax(yy)
        if ymax == ymin:
            delta = 1.0 if ymin == 0 else 0.1 * abs(ymin)
            return ymin - delta, ymax + delta
        span = ymax - ymin
        return ymin - pad_frac * span, ymax + pad_frac * span
    span = p_hi - p_lo
    return p_lo - pad_frac * span, p_hi + pad_frac * span


def instrumental_sigma_floor(redshift):
    inst_wave = 6720.0 * (1.0 + redshift)
    sigma_inst = (5.866e-8 * inst_wave**2 - 9.187e-4 * inst_wave + 6.040) / 2.3553

    if sigma_inst <= 0:
        raise ValueError("Computed sigma_inst <= 0 (%.6g); check redshift value." % sigma_inst)
    if sigma_inst > MAX_SIGMA_ANGSTROM:
        raise ValueError(
            "Instrumental sigma floor (%.4f) exceeds MAX_SIGMA_ANGSTROM (%.1f). "
            "Check redshift value." % (sigma_inst, MAX_SIGMA_ANGSTROM)
        )
    return sigma_inst


def two_gaussians_linear_model_reparam(wave, amp1, cen1, sigma1, ratio, sigma2, slope, intercept):
    """cen2 = cen1 + LINE_OFFSET_ANGSTROM (fixed); amp2 = amp1 * ratio (ratio in [0.1,10])."""
    cen2 = cen1 + LINE_OFFSET_ANGSTROM
    amp2 = amp1 * ratio
    g1 = amp1 * np.exp(-0.5 * ((wave - cen1) / sigma1) ** 2)
    g2 = amp2 * np.exp(-0.5 * ((wave - cen2) / sigma2) ** 2)
    return slope * wave + intercept + g1 + g2


def fit_two_gaussians_linear(wave, flux, variance, redshift, line1_guess=None):
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    variance = np.asarray(variance, dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(variance) & (variance > 0)
    if np.count_nonzero(good) < 7:
        raise RuntimeError(
            "Not enough finite valid points to fit (found %d, need >=7)." % np.count_nonzero(good)
        )

    wave_fit, flux_fit, var_fit = wave[good], flux[good], variance[good]
    flux_err = np.sqrt(var_fit)

    sigma_inst = instrumental_sigma_floor(redshift)

    cen1_guess = float(line1_guess) if line1_guess is not None else wave_fit[np.nanargmax(flux_fit)]

    amp1_guess = float(np.nanmax(flux_fit) - np.nanmedian(flux_fit))
    if amp1_guess <= 0:
        amp1_guess = float(np.nanmax(flux_fit)) if np.nanmax(flux_fit) > 0 else 1.0

    ratio_guess = 1.0
    sigma1_guess = min(sigma_inst * 1.5, MAX_SIGMA_ANGSTROM)
    sigma2_guess = min(sigma_inst * 1.5, MAX_SIGMA_ANGSTROM)
    slope_guess = 0.0
    intercept_guess = float(np.nanmedian(flux_fit))

    p0 = [amp1_guess, cen1_guess, sigma1_guess, ratio_guess, sigma2_guess, slope_guess, intercept_guess]

    amp_bound = 10.0 * max(abs(amp1_guess), 1.0)

    lower_bounds = [-amp_bound, wave_fit.min(), sigma_inst, MIN_AMP_RATIO, sigma_inst, -np.inf, -np.inf]
    upper_bounds = [amp_bound, wave_fit.max(), MAX_SIGMA_ANGSTROM, MAX_AMP_RATIO,
                     MAX_SIGMA_ANGSTROM, np.inf, np.inf]

    p0 = np.clip(p0, lower_bounds, upper_bounds)

    popt, pcov = curve_fit(
        two_gaussians_linear_model_reparam, wave_fit, flux_fit,
        p0=p0, sigma=flux_err, absolute_sigma=True,
        bounds=(lower_bounds, upper_bounds), maxfev=20000,
    )

    amp1, cen1, sigma1, ratio, sigma2, slope, intercept = popt
    cen2 = cen1 + LINE_OFFSET_ANGSTROM
    amp2 = amp1 * ratio

    total_flux1 = amp1 * sigma1 * np.sqrt(2.0 * np.pi)
    total_flux2 = amp2 * sigma2 * np.sqrt(2.0 * np.pi)

    return {
        "popt_reparam": popt,
        "pcov_reparam": pcov,
        "sigma_inst": sigma_inst,
        "gaussian1": {"total_flux": total_flux1, "sigma": sigma1,
                      "central_wavelength": cen1, "amplitude": amp1},
        "gaussian2": {"total_flux": total_flux2, "sigma": sigma2,
                      "central_wavelength": cen2, "amplitude": amp2},
    }


def evaluate_fit_model(wave, fit_result):
    g1, g2 = fit_result["gaussian1"], fit_result["gaussian2"]
    slope, intercept = fit_result["popt_reparam"][5], fit_result["popt_reparam"][6]

    gauss1 = g1["amplitude"] * np.exp(-0.5 * ((wave - g1["central_wavelength"]) / g1["sigma"]) ** 2)
    gauss2 = g2["amplitude"] * np.exp(-0.5 * ((wave - g2["central_wavelength"]) / g2["sigma"]) ** 2)
    return slope * wave + intercept + gauss1 + gauss2


def build_title(x1, x2, y1, y2, wrange_txt, redshift):
    """Build the figure suptitle string using simple concatenation to avoid
    any issues with long inline % or f-string expressions."""
    title = "Region x=[" + str(x1) + ":" + str(x2) + "], y=[" + str(y1) + ":" + str(y2) + "] px"
    title += " | lambda: " + wrange_txt
    if redshift != 0.0:
        title += " | z=" + str(redshift)
    return title


def main():
    args = parse_args()

    flux_cube = Cube(args.data_cube, ext=args.data_ext)
    var_cube = Cube(args.var_cube, ext=args.var_ext)

    if flux_cube.shape != var_cube.shape:
        print("[WARN] Flux cube shape %s differs from variance cube shape %s."
              % (flux_cube.shape, var_cube.shape))

    ny, nx = flux_cube.shape[1], flux_cube.shape[2]
    print("[INFO] Cube spatial shape: ny=%d, nx=%d; nlambda=%d" % (ny, nx, flux_cube.shape[0]))

    y1, y2, x1, x2 = rect_to_pixel_indices(args.x0, args.y0, args.width, args.height, nx, ny)
    print("[INFO] Using pixel slice: y=[%d:%d], x=[%d:%d]" % (y1, y2, x1, x2))

    wave_f, flux_sum = stack_region(flux_cube, y1, y2, x1, x2, args.wmin, args.wmax)
    wave_v, var_sum = stack_region(var_cube, y1, y2, x1, x2, args.wmin, args.wmax)

    if len(wave_f) != len(wave_v) or not np.allclose(wave_f, wave_v, rtol=0, atol=1e-8):
        raise RuntimeError("Flux and variance wavelength grids do not match.")

    if args.redshift != 0.0:
        wave_f = deredshift_wavelength(wave_f, args.redshift)
        wave_v = deredshift_wavelength(wave_v, args.redshift)
        print("[INFO] Applied de-redshift correction with z=%s" % args.redshift)

    n_flux_finite = np.count_nonzero(np.isfinite(flux_sum))
    n_var_finite = np.count_nonzero(np.isfinite(var_sum))
    print("[INFO] Wavelength points: %d" % len(wave_f))
    print("[INFO] Flux finite points: %d/%d" % (n_flux_finite, len(flux_sum)))
    print("[INFO] Variance finite points: %d/%d" % (n_var_finite, len(var_sum)))
    if n_flux_finite > 0:
        print("[INFO] Flux min/max: %.6g / %.6g" % (np.nanmin(flux_sum), np.nanmax(flux_sum)))
    if n_var_finite > 0:
        print("[INFO] Variance min/max: %.6g / %.6g" % (np.nanmin(var_sum), np.nanmax(var_sum)))

    if n_flux_finite == 0 and n_var_finite == 0:
        raise RuntimeError("No finite data found. Check extensions and region coordinates.")

    out_arr = np.column_stack([wave_f, flux_sum, var_sum])
    header_txt = "wavelength flux_sum variance_sum"
    if args.redshift != 0.0:
        header_txt += " (rest-frame, z=%s)" % args.redshift
    np.savetxt(args.outspec, out_arr, header=header_txt)
    print("[INFO] Saved spectrum table: %s" % args.outspec)

    wave_plot_f, flux_plot = clean_series(wave_f, flux_sum)
    wave_plot_v, var_plot = clean_series(wave_v, var_sum)

    if len(wave_plot_f) == 0 and len(wave_plot_v) == 0:
        raise RuntimeError("Nothing left to plot after cleaning NaN/Inf values.")

    fit_result = None
    model_curve = None
    wave_model = None
    if args.fit_gaussians:
        try:
            fit_result = fit_two_gaussians_linear(
                wave_f, flux_sum, var_sum, args.redshift, line1_guess=args.line1_guess
            )
            g1 = fit_result["gaussian1"]
            g2 = fit_result["gaussian2"]

            print("")
            print("[FIT RESULT] Two-Gaussian + linear baseline fit")
            print("  sigma_inst=%.4f, sigma bounds=[%.4f, %.1f], amp ratio bounds=[%.2f, %.2f]"
                  % (fit_result["sigma_inst"], fit_result["sigma_inst"], MAX_SIGMA_ANGSTROM,
                     MIN_AMP_RATIO, MAX_AMP_RATIO))
            print("  Gaussian 1 (shorter wavelength):")
            print("      total_flux=%.6g sigma=%.6g central_wavelength=%.6g amplitude=%.6g"
                  % (g1["total_flux"], g1["sigma"], g1["central_wavelength"], g1["amplitude"]))
            print("  Gaussian 2 (longer wavelength, offset=+%.1f A):" % LINE_OFFSET_ANGSTROM)
            print("      total_flux=%.6g sigma=%.6g central_wavelength=%.6g amplitude=%.6g"
                  % (g2["total_flux"], g2["sigma"], g2["central_wavelength"], g2["amplitude"]))

            if len(wave_plot_f) > 0:
                wave_model = np.linspace(np.nanmin(wave_plot_f), np.nanmax(wave_plot_f), 2000)
                model_curve = evaluate_fit_model(wave_model, fit_result)
        except Exception as e:
            print("[WARN] Gaussian fit failed: %s" % e)
            fit_result = None
            model_curve = None
            wave_model = None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)

    if len(wave_plot_f) > 0:
        ax1.plot(wave_plot_f, flux_plot, color="tab:blue", lw=1.2, label="Stacked flux")
    if len(wave_plot_v) > 0:
        ax2.plot(wave_plot_v, var_plot, color="tab:red", lw=1.2)

    if model_curve is not None and wave_model is not None:
        ax1.plot(wave_model, model_curve, color="black", lw=1.5, ls="--", label="2-Gaussian + linear fit")
        ax1.legend(loc="best", fontsize=9)

    ax1.set_ylabel("Summed Flux")
    ax1.set_title("Stacked Spectrum in Rectangular Region")
    ax1.grid(alpha=0.3)

    wave_label = "Wavelength [%s]" % flux_cube.wave.unit
    if args.redshift != 0.0:
        wave_label += " (rest-frame, z=%s)" % args.redshift
    ax2.set_xlabel(wave_label)
    ax2.set_ylabel("Summed Variance")
    ax2.grid(alpha=0.3)

    all_wave_plot = wave_plot_f if len(wave_plot_f) > 0 else wave_plot_v
    if len(all_wave_plot) > 0:
        xmin, xmax = np.nanmin(all_wave_plot), np.nanmax(all_wave_plot)
        if xmax > xmin:
            ax1.set_xlim(xmin, xmax)

    ax1.set_ylim(*robust_ylim(flux_plot))
    ax2.set_ylim(*robust_ylim(var_plot))

    if len(all_wave_plot) > 0:
        wrange_txt = "%.2f - %.2f" % (np.nanmin(all_wave_plot), np.nanmax(all_wave_plot))
    else:
        wrange_txt = "N/A"

    title_str = build_title(x1, x2, y1, y2, wrange_txt, args.redshift)
    fig.suptitle(title_str)

    plt.savefig(args.outplot, dpi=150)
    plt.close(fig)

    if not os.path.exists(args.outplot) or os.path.getsize(args.outplot) == 0:
        raise RuntimeError("Plot file '%s' was not created or is empty." % args.outplot)

    print("[INFO] Saved plot to: %s (%d bytes)" % (args.outplot, os.path.getsize(args.outplot)))


if __name__ == "__main__":
    main()
