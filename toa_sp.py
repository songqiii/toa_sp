#!/usr/bin/env python3
"""
toa_sp — Single-Pulse TOA Extraction Framework.

Reads a FAST PSRFITS search-mode file, dedisperses at a given DM, bins in
frequency and time, sums over frequency to form a 1D pulse profile, then fits
a multi-strategy model suite (Gaussian / EMG / Voigt / shapelet) to obtain the
time of arrival (TOA) and its uncertainty.

Usage (example matching v008.py):
    python toa_sp.py -f FRB20220529_tracking-M01_0158.fits \
        -dm 255.0 -s 1.1 -w 0.06 -bs 8 -bf 16 -o timing_result

Reference: v008.py (data I/O and dedispersion), Kulkarni (2020) DM constant.
"""

import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import argparse
import warnings
import matplotlib
matplotlib.use("Agg")          # default non-interactive; switched later if --show
import matplotlib.pyplot as plt
plt.ioff()                     # block interactive mode until requested

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DM_CONST = 4.1488064239   # Kulkarni 2020 -- gives dt in ms when f in GHz


# ---------------------------------------------------------------------------
# Data I/O -- adapted from v008.py
# ---------------------------------------------------------------------------

def unpack_data_vectorized(data_grab, nbits, nchan, nsamp, npol):
    """
    Vectorised unpack of packed PSRFITS search-mode subint data.

    Parameters
    ----------
    data_grab : ndarray (nsamp, npol, nchan * nbits / 8)
    nbits, nchan, nsamp, npol : int

    Returns
    -------
    ndarray (nsamp, npol, nchan) -- unpacked values.
    """
    packed_dim = data_grab.shape[2]
    data_2d = data_grab.reshape(-1, packed_dim).astype(np.uint8)
    bits = np.unpackbits(data_2d, axis=1)
    bits = bits.reshape(nsamp, npol, nchan, nbits)
    bits = bits[:, :, :, ::-1]                      # MSB-first -> LSB-first
    bits_2d = bits.reshape(-1, nbits)
    packed = np.packbits(bits_2d, axis=1, bitorder='little')
    return packed.reshape(nsamp, npol, nchan)


def dedisperse(dat, dm, f_low, f_high, nchan, tsamp_ms):
    """
    Incoherent dedispersion -- align all channels to ``f_high``.

    Parameters
    ----------
    dat : ndarray (nchan, nsamp)
    dm  : float   -- dispersion measure (pc / cm^3)
    f_low, f_high : float -- frequency limits in **GHz**
    nchan : int   -- number of frequency channels
    tsamp_ms : float -- sampling time in **ms**
    """
    freqs = np.linspace(f_low, f_high, nchan, endpoint=True)
    inv_f_high_sq = f_high ** (-2)
    for i, f in enumerate(freqs):
        dt = DM_CONST * dm * (inv_f_high_sq - f ** (-2))   # ms
        ds = int(dt / tsamp_ms)
        if ds != 0:
            dat[i, :] = np.roll(dat[i, :], ds)


def bin_2d(dat, nbin_freq, nbin_time):
    """
    Bin a 2D array (nchan, nsamp) by ``nbin_freq`` in frequency and
    ``nbin_time`` in time using reduceat.
    """
    m, n = dat.shape
    # frequency binning
    if nbin_freq > 1:
        m_trim = m - (m % nbin_freq)
        dat = np.add.reduceat(dat[:m_trim],
                              np.arange(0, m_trim, nbin_freq))
    # time binning
    m2, n2 = dat.shape
    if nbin_time > 1:
        n_trim = n2 - (n2 % nbin_time)
        dat = np.add.reduceat(dat[:, :n_trim],
                              np.arange(0, n_trim, nbin_time), axis=1)
    return dat


def zap_rfi_time(dat):
    """Remove broadband RFI: subtract 75 % of the zero-DM time series."""
    dat -= np.mean(dat, axis=0) * 0.75


def zap_rfi_freq(dat, bs, bf):
    """
    Iteratively flag narrow-band RFI channels.

    ``dat`` is (nchan_binned, nsamp_binned); the mask is updated until
    convergence.
    """
    nchan = dat.shape[0]
    spec = np.sum(dat, axis=1)
    std = np.std(dat, axis=1)
    n0, n1 = 0, 1
    mask = np.ones(nchan, dtype=bool)
    while n0 != n1:
        n0 = nchan - np.count_nonzero(mask)
        std_spec = np.std(spec[mask])
        mean_spec = np.mean(spec[mask])
        std_std = np.std(std[mask])
        mean_std = np.mean(std[mask])
        mask &= ~(
            (np.abs(spec - mean_spec) > 3 * std_spec * (1 + 0.1 * bf ** 0.5))
            | (np.abs(std - mean_std) > 3 * std_std * (1 + 0.1 * bf ** 0.5))
        )
        n1 = nchan - np.count_nonzero(mask)
    if not np.all(mask):
        dat[~mask, :] = np.mean(dat[mask, :])


# ---------------------------------------------------------------------------
# Frequency range helper
# ---------------------------------------------------------------------------

def freq_range_to_channel_indices(freq_lo_mhz, freq_hi_mhz, f0_mhz, f1_mhz,
                                   nchan_orig, bin_chn):
    """Map a frequency range [freq_lo, freq_hi] in MHz to binned channel indices.

    Parameters
    ----------
    freq_lo_mhz, freq_hi_mhz : float
        Lower and upper frequency bounds in MHz.
    f0_mhz, f1_mhz : float
        Band edges in MHz (from FITS header).
    nchan_orig : int
        Number of original (pre-binning) frequency channels.
    bin_chn : int
        Frequency-binning factor.

    Returns
    -------
    (ch_lo_binned, ch_hi_binned) : tuple of int
        Start (inclusive) and end (exclusive) indices into the binned data.
    """
    nchan_binned = nchan_orig // bin_chn
    frac_lo = (freq_lo_mhz - f0_mhz) / (f1_mhz - f0_mhz)
    frac_hi = (freq_hi_mhz - f0_mhz) / (f1_mhz - f0_mhz)
    ch_lo_orig = int(np.floor(frac_lo * nchan_orig))
    ch_hi_orig = int(np.ceil(frac_hi * nchan_orig))
    ch_lo_b = max(0, ch_lo_orig // bin_chn)
    ch_hi_b = min(nchan_binned, (ch_hi_orig + bin_chn - 1) // bin_chn)
    if ch_hi_b <= ch_lo_b:
        ch_hi_b = min(nchan_binned, ch_lo_b + 1)
    return ch_lo_b, ch_hi_b


# ---------------------------------------------------------------------------
# TOA reference frequency helper
# ---------------------------------------------------------------------------

def resolve_toa_freq(toa_freq_arg, f_lo_mhz, f_hi_mhz):
    """Resolve the TOA reference frequency from the user argument.

    Parameters
    ----------
    toa_freq_arg : str
        ``--toa_freq`` argument: 'top', 'center', or a numeric string.
    f_lo_mhz, f_hi_mhz : float
        Selected band edges in MHz.

    Returns
    -------
    float
        Reference frequency in MHz.
    """
    arg_lower = str(toa_freq_arg).strip().lower()
    if arg_lower == 'top':
        return f_hi_mhz
    elif arg_lower == 'center':
        return 0.5 * (f_lo_mhz + f_hi_mhz)
    else:
        try:
            return float(arg_lower)
        except ValueError:
            print(f"  Warning: invalid --toa_freq '{toa_freq_arg}';"
                  f" using band top ({f_hi_mhz:.1f} MHz).")
            return f_hi_mhz


# ---------------------------------------------------------------------------
# Time-resolution optimisation
# ---------------------------------------------------------------------------

def optimize_profile_resolution(profile, t_prof, t_bin_width, w0,
                                max_factor=256, min_bins_across_pulse=12):
    """Find the optimal 1D rebinning factor by maximising peak S/N.

    Rebins the profile by factors of 1, 2, 4, 8, ..., up to ``max_factor``
    (subject to a floor of ``min_bins_across_pulse`` bins within the pulse
    window).  Returns the rebinned profile and updated time axis at the
    resolution that yields the highest peak S/N.

    Parameters
    ----------
    profile : ndarray (nsamp,)
        Baseline-subtracted 1D pulse profile.
    t_prof : ndarray (nsamp,)
        Time axis in seconds.
    t_bin_width : float
        Current time bin width in seconds.
    w0 : float
        Expected pulse half-width in seconds (search window).
    max_factor : int
        Maximum rebinning factor to try (must be a power of 2; default 256).
    min_bins_across_pulse : int
        Minimum number of bins that must remain within 2*w0 after rebinning
        (default 12 — ensures adequate sampling for profile fitting).

    Returns
    -------
    dict with keys:
        profile : ndarray — rebinned profile at optimal resolution
        t_prof : ndarray — updated time axis
        t_bin_width : float — updated bin width
        factor : int — optimal rebinning factor
        snr_gain : float — S/N at optimal factor / S/N at factor=1
        snr_values : list of (factor, snr) tuples for all factors tried
    """
    n = len(profile)
    # Estimate pulse width in bins
    pulse_bins = max(1, int(2 * w0 / t_bin_width))
    # Upper limit: don't rebin below min_bins_across_pulse
    max_factor_safe = max(1, pulse_bins // min_bins_across_pulse)
    max_factor = min(max_factor, max_factor_safe)

    # Generate power-of-2 factors
    factors = [1]
    while factors[-1] * 2 <= max_factor:
        factors.append(factors[-1] * 2)

    snr_values = []
    best_factor = 1
    best_snr = -np.inf
    best_profile = profile
    best_t_prof = t_prof

    for f in factors:
        # Rebin profile
        n_trim = n - (n % f)
        if n_trim < f * min_bins_across_pulse:
            continue
        prof_r = np.sum(profile[:n_trim].reshape(-1, f), axis=1)
        # Updated time axis (bin centres)
        t_r = t_prof[:n_trim:f] + 0.5 * (f - 1) * t_bin_width

        # Off-pulse RMS from edges
        edge = max(1, len(prof_r) // 5)
        off_region = np.concatenate([prof_r[:edge], prof_r[-edge:]])
        rms_off = np.std(off_region)
        if rms_off == 0:
            continue

        peak_val = np.max(prof_r)
        snr = peak_val / rms_off

        snr_values.append((f, snr))

        if snr > best_snr:
            best_snr = snr
            best_factor = f
            best_profile = prof_r
            best_t_prof = t_r

    # S/N gain relative to factor=1
    snr_1 = snr_values[0][1] if snr_values else best_snr
    snr_gain = best_snr / snr_1 if snr_1 > 0 else 1.0

    return {
        'profile': best_profile,
        't_prof': best_t_prof,
        't_bin_width': t_bin_width * best_factor,
        'factor': best_factor,
        'snr_gain': snr_gain,
        'snr_values': snr_values,
    }


# ---------------------------------------------------------------------------
# Scale data -- adapted from v008.py
# ---------------------------------------------------------------------------

def newscale_data(data_out, npol, pol_order, nchan, nbits):
    """
    Per-channel normalisation: scale each unsigned-pol channel to a target
    std and mean so that noise properties are uniform across the band.
    """
    target_std = 2 ** nbits / 6.0
    target_avg = 2 ** (nbits - 1) - 0.5 - target_std * 1.0

    # Determine how many polarisations are unsigned
    numunsigned = npol
    if npol == 4:
        if pol_order == "AABBCRCI":
            numunsigned = 2
        if pol_order == "IQUV":
            numunsigned = 1

    fdata = np.empty(data_out.shape, dtype=float)
    for poln in range(numunsigned):
        stds  = np.std(data_out[poln, :, :], axis=1)    # (nchan,)
        means = np.mean(data_out[poln, :, :], axis=1)   # (nchan,)
        scales  = stds / target_std
        # Guard against zero-variance channels (e.g. 2-bit data with
        # constant-valued channels) — leave those channels unscaled.
        bad = (stds < 1e-12)
        if np.any(bad):
            scales[bad]  = 1.0
        offsets = means - target_avg * scales
        fdata[poln, :, :] = ((data_out[poln, :, :] - offsets[:, np.newaxis])
                             / scales[:, np.newaxis])
    for poln in range(numunsigned, npol):
        fdata[poln, :, :] = data_out[poln, :, :]
    return fdata


# ---------------------------------------------------------------------------
# Gaussian model
# ---------------------------------------------------------------------------

def gaussian(t, amplitude, mean, sigma, offset):
    """Gaussian plus constant baseline."""
    return amplitude * np.exp(-0.5 * ((t - mean) / sigma) ** 2) + offset


def gaussian_n(t, *params):
    """N-component Gaussian + constant baseline.

    Parameters
    ----------
    t : ndarray
        Time axis.
    params : (3*N + 1) floats
        [amp1, mu1, sig1, ..., ampN, muN, sigN, offset]
        where amp_i may be positive (emission) or negative (absorption).

    Returns
    -------
    ndarray — sum of N Gaussians plus baseline offset.
    """
    n = (len(params) - 1) // 3
    y = np.full_like(t, params[-1], dtype=float)
    for i in range(n):
        a, m, s = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        y += a * np.exp(-0.5 * ((t - m) / s) ** 2)
    return y


# ---------------------------------------------------------------------------
# Shapelet decomposition (Hermite polynomials × Gaussian envelope)
#   Ashton et al. (2020, arXiv:2011.07927); Refregier (2003)
# ---------------------------------------------------------------------------

def hermite_polynomials(x, n_max):
    """Evaluate physicist's Hermite polynomials H_0 through H_{n_max} at x.

    Uses the recurrence H_{n+1}(x) = 2x·H_n(x) - 2n·H_{n-1}(x), with
    H_0(x) = 1, H_1(x) = 2x.  Returns an array of shape (len(x), n_max+1).

    Parameters
    ----------
    x : ndarray
        Evaluation points (1D).
    n_max : int
        Maximum polynomial order (inclusive).  Must be ≥ 0.

    Returns
    -------
    H : ndarray (len(x), n_max+1)
        H[:, i] = H_i(x).
    """
    x = np.atleast_1d(x).astype(np.float64)
    n = len(x)
    H = np.zeros((n, n_max + 1), dtype=np.float64)
    if n_max >= 0:
        H[:, 0] = 1.0
    if n_max >= 1:
        H[:, 1] = 2.0 * x
    for i in range(1, n_max):
        H[:, i + 1] = 2.0 * x * H[:, i] - 2.0 * i * H[:, i - 1]
    return H


def shapelet_model(t, tau, beta, coeffs, poly_coeffs, t_mid):
    """Evaluate a shapelet model at times t.

    Model: Σ_i C_i · H_i((t-τ)/β) · exp(-((t-τ)/β)²)  +  Σ_j B_j · (t - t_mid)^j

    Parameters
    ----------
    t : ndarray
        Time axis.
    tau : float
        Shapelet centre (the TOA).
    beta : float
        Shapelet width parameter.
    coeffs : ndarray
        Shapelet coefficients [C_0, C_1, ..., C_ns].
    poly_coeffs : ndarray
        Polynomial baseline coefficients [B_0, ..., B_np].
    t_mid : float
        Reference time for the polynomial (centre of the fit window).

    Returns
    -------
    ndarray — model evaluated at t.
    """
    x = (t - tau) / beta
    n_s = len(coeffs) - 1
    H = hermite_polynomials(x, n_s)
    gauss_env = np.exp(-x ** 2)

    y = np.zeros_like(t, dtype=float)
    for i, c in enumerate(coeffs):
        y += c * H[:, i] * gauss_env

    dt = t - t_mid
    for j, b in enumerate(poly_coeffs):
        y += b * dt ** j

    return y


def build_shapelet_matrix(t, tau, beta, n_shapelet, n_poly, t_mid):
    """Build the design matrix for a shapelet + polynomial fit.

    For fixed (tau, beta) the model is purely linear in the coefficients
    {C_i}, {B_j}.  Each column corresponds to one basis function.

    Parameters
    ----------
    t : ndarray
    tau : float
    beta : float
    n_shapelet : int
        Maximum Hermite order (inclusive — total n_shapelet+1 shapelet cols).
    n_poly : int
        Polynomial degree (total n_poly+1 baseline cols).
    t_mid : float

    Returns
    -------
    M : ndarray (len(t), n_shapelet + 1 + n_poly + 1)
    """
    x = (t - tau) / beta
    H = hermite_polynomials(x, n_shapelet)
    gauss_env = np.exp(-x ** 2)

    n_total = (n_shapelet + 1) + (n_poly + 1)
    M = np.zeros((len(t), n_total), dtype=np.float64)

    for i in range(n_shapelet + 1):
        M[:, i] = H[:, i] * gauss_env

    dt = t - t_mid
    for j in range(n_poly + 1):
        M[:, n_shapelet + 1 + j] = dt ** j

    return M


def fit_shapelet_linear(t, y, tau, beta, n_shapelet, n_poly, t_mid, alpha=0.0):
    """Linear least-squares fit of shapelet coefficients at fixed (tau, beta).

    Parameters
    ----------
    t, y : ndarray
    tau, beta : float
        Fixed nonlinear parameters.
    n_shapelet : int
    n_poly : int
    t_mid : float
    alpha : float
        Ridge (L2) regularisation strength.  α = 0 → OLS.

    Returns
    -------
    dict with keys: coeffs, poly_coeffs, params, rss, dof_eff, sigma2,
        cov, ok, n_shapelet, n_poly, tau, beta, t_mid.
    """
    import numpy.linalg as LA
    M = build_shapelet_matrix(t, tau, beta, n_shapelet, n_poly, t_mid)
    n_total = M.shape[1]
    n_data = len(y)

    if alpha > 0:
        # Ridge: (MᵀM + αI)^{-1} Mᵀ y
        A = M.T @ M + alpha * np.eye(n_total)
        b = M.T @ y
        params = LA.solve(A, b)
        # Effective degrees of freedom (Hastie et al. 2009, eq 7.25)
        try:
            _, s, _ = LA.svd(M, full_matrices=False)
            d = s ** 2 / (s ** 2 + alpha)
            dof_eff = max(1.0, np.sum(d))
        except LA.LinAlgError:
            dof_eff = 2.0  # fallback
    else:
        params, residuals, rank, _ = LA.lstsq(M, y, rcond=None)
        dof_eff = float(rank)

    y_pred = M @ params
    rss = float(np.sum((y - y_pred) ** 2))

    # Noise variance — use dof_eff for unbiased estimate
    denom = max(1.0, n_data - dof_eff)
    sigma2 = rss / denom

    # Covariance
    if alpha > 0:
        try:
            MtM_a_inv = LA.inv(M.T @ M + alpha * np.eye(n_total))
            cov = sigma2 * MtM_a_inv @ (M.T @ M) @ MtM_a_inv
        except LA.LinAlgError:
            cov = np.full((n_total, n_total), np.nan)
    else:
        try:
            cov = sigma2 * LA.inv(M.T @ M)
        except LA.LinAlgError:
            cov = np.full((n_total, n_total), np.nan)

    coeffs = params[:n_shapelet + 1]
    poly_coeffs = params[n_shapelet + 1:]

    return {
        'coeffs': coeffs,
        'poly_coeffs': poly_coeffs,
        'params': params,
        'rss': rss,
        'dof_eff': dof_eff,
        'sigma2': sigma2,
        'cov': cov,
        'ok': True,
        'n_shapelet': n_shapelet,
        'n_poly': n_poly,
        'tau': tau,
        'beta': beta,
        't_mid': t_mid,
    }


def optimize_shapelet_tau_beta(t, y, tau_guess, beta_guess,
                               n_shapelet, n_poly, t_mid, alpha=0.0):
    """2D nonlinear optimisation over (tau, beta) for shapelet fit.

    At each (tau, beta) evaluation, the linear coefficients are obtained
    via ``fit_shapelet_linear`` — this is a separable least-squares
    problem.  L-BFGS-B with bounds is used.

    Parameters
    ----------
    t, y : ndarray
    tau_guess, beta_guess : float
        Initial values.
    n_shapelet, n_poly : int
    t_mid : float
    alpha : float

    Returns
    -------
    dict — the result of the best ``fit_shapelet_linear`` plus the
    ``optimize_result`` from scipy.
    """
    from scipy.optimize import minimize

    t_min, t_max = float(t[0]), float(t[-1])
    t_bin = float(t[1] - t[0]) if len(t) > 1 else (t_max - t_min) / max(1, len(t) - 1)
    beta_min = max(t_bin, 1e-9)
    beta_max = 0.5 * (t_max - t_min)

    def objective(params):
        t_i, b_i = params
        if b_i <= beta_min or b_i >= beta_max:
            return 1e300
        if t_i < t_min or t_i > t_max:
            return 1e300
        r = fit_shapelet_linear(t, y, t_i, b_i, n_shapelet, n_poly, t_mid, alpha)
        return r['rss']

    res = minimize(
        objective,
        [tau_guess, beta_guess],
        method='L-BFGS-B',
        bounds=[(t_min, t_max), (beta_min, beta_max)],
        options={'maxiter': 200, 'ftol': 1e-12},
    )

    tau_opt, beta_opt = float(res.x[0]), float(res.x[1])
    final = fit_shapelet_linear(t, y, tau_opt, beta_opt, n_shapelet, n_poly, t_mid, alpha)
    final['optimize_result'] = res
    return final


def fit_shapelet_auto_n(t, y, tau_guess, beta_guess, n_max, n_poly, t_mid, alpha=0.0):
    """Shapelet fit with automatic selection of the number of components.

    Scans n_shapelet ∈ {2, 4, 6, …, n_max} (even orders only —
    Hermite parity).  Selects the order with lowest AICc.

    Returns
    -------
    dict — the best fit, plus ``all_results`` key holding every scan entry.
    """
    n_list = list(range(2, n_max + 1, 2))
    if not n_list:
        n_list = [2]

    results = []
    tau_cur, beta_cur = tau_guess, beta_guess

    for n_s in n_list:
        try:
            fit = optimize_shapelet_tau_beta(t, y, tau_cur, beta_cur,
                                            n_s, n_poly, t_mid, alpha)
        except Exception:
            continue
        # AICc: +2 for (tau, beta) estimated nonlinearly
        n_linear = (n_s + 1) + (n_poly + 1)
        n_param = n_linear + 2
        aicc = compute_aicc(fit['rss'], n_param, len(y))
        fit['aicc'] = aicc
        fit['n_param'] = n_param
        results.append(fit)
        # Warm-start next iteration
        tau_cur = fit['tau']
        beta_cur = fit['beta']

    if not results:
        # Fallback: return a minimal result for n_s=2
        return optimize_shapelet_tau_beta(t, y, tau_guess, beta_guess,
                                         2, n_poly, t_mid, alpha)

    best = min(results, key=lambda r: r['aicc'])
    best['all_results'] = results
    return best


def shapelet_toa_uncertainty(t, y, tau_opt, beta_opt, coeffs, poly_coeffs,
                             n_shapelet, n_poly, t_mid, rms, alpha=0.0):
    """Estimate TOA uncertainty from the profile likelihood over τ.

    Evaluates RSS on a grid of τ around τ_opt (with β re-optimised at
    each τ) and fits a quadratic.  The curvature gives the 1-σ error:
    σ_τ = σ_noise / √(d²RSS/dτ²).

    Also computes the empirical σ_τ = β / (SNR √N_eff).

    Parameters
    ----------
    t, y : ndarray
    tau_opt, beta_opt : float
    coeffs, poly_coeffs : ndarray
    n_shapelet, n_poly : int
    t_mid : float
    rms : float — off-pulse RMS
    alpha : float

    Returns
    -------
    dict — tau_err, tau_emp, tau_16, tau_84, tau_grid, rss_grid.
    """
    t_bin = float(t[1] - t[0]) if len(t) > 1 else 1e-6
    d_tau = max(t_bin / 2.0, beta_opt / 20.0, 1e-9)
    n_steps = 5

    tau_grid = tau_opt + np.arange(-n_steps, n_steps + 1) * d_tau
    tau_grid = tau_grid[(tau_grid >= t[0]) & (tau_grid <= t[-1])]
    if len(tau_grid) < 3:
        return {
            'tau_err': np.nan, 'tau_emp': np.nan,
            'tau_16': np.nan, 'tau_84': np.nan,
            'tau_grid': tau_grid, 'rss_grid': np.array([]),
        }

    rss_grid = []
    for tau_i in tau_grid:
        try:
            fit_i = optimize_shapelet_tau_beta(
                t, y, tau_i, beta_opt, n_shapelet, n_poly, t_mid, alpha)
            rss_grid.append(fit_i['rss'])
        except Exception:
            rss_grid.append(np.nan)

    rss_grid = np.array(rss_grid)
    valid = np.isfinite(rss_grid)

    if np.sum(valid) >= 3:
        from numpy.polynomial import polynomial as P
        try:
            # Fit quadratic: RSS ≈ a·Δτ² + b·Δτ + c
            dtau = tau_grid[valid] - tau_opt
            quad = P.polyfit(dtau, rss_grid[valid], 2)
            a = quad[2]  # curvature
            if a > 0 and np.isfinite(a) and rms > 0:
                # Δχ² = ΔRSS / σ²;  for 1σ: Δχ² = 1
                # a · σ_τ² / σ² = 1  →  σ_τ = σ / √a
                tau_err = rms / np.sqrt(a)
            else:
                tau_err = np.nan
        except Exception:
            tau_err = np.nan
    else:
        tau_err = np.nan

    # Empirical uncertainty
    snr = np.max(np.abs(coeffs)) / rms if (rms and rms > 0) else np.inf
    n_eff = max(1, 2.0 * beta_opt / t_bin)
    tau_emp = beta_opt / (snr * np.sqrt(n_eff)) if snr > 0 and np.isfinite(snr) else np.nan

    return {
        'tau_err': tau_err,
        'tau_emp': tau_emp,
        'tau_16': tau_opt - tau_err if np.isfinite(tau_err) else np.nan,
        'tau_84': tau_opt + tau_err if np.isfinite(tau_err) else np.nan,
        'tau_grid': tau_grid,
        'rss_grid': rss_grid,
    }


# ---------------------------------------------------------------------------
# Exponentially-Modified Gaussian (scattering tails)
# ---------------------------------------------------------------------------

def emg(t, amplitude, mu, sigma, tau, offset):
    """Exponentially-modified Gaussian + constant baseline.

    Convolution of a Gaussian(mu, sigma) with exp(-t/tau) for t > 0.
    Uses a branched formulation for numerical stability:
      - erfcx(z) for z >= 0 (stable)
      - exp(z²)·erfc(z) rewritten with erfc(|z|) for z < 0 (avoids overflow)

    Parameters
    ----------
    t : ndarray
    amplitude : float — scaling factor
    mu : float — Gaussian centre (intrinsic arrival time before scattering)
    sigma : float — Gaussian width
    tau : float — scattering timescale (exponential decay constant)
    offset : float — constant baseline
    """
    from scipy.special import erfc, erfcx
    if sigma <= 1e-12 or tau <= 1e-12:
        return np.full_like(t, 1e300, dtype=float)

    z = (sigma / tau - (t - mu) / sigma) / np.sqrt(2)

    # Branch on sign of z for numerical stability
    mask_pos = z >= 0
    mask_neg = ~mask_pos

    y = np.empty_like(t, dtype=float)
    prefactor = amplitude / (2.0 * tau)

    # z >= 0: use erfcx — exp(-(t-mu)²/(2σ²)) * erfcx(z) is stable
    y[mask_pos] = (prefactor * np.exp(-0.5 * ((t[mask_pos] - mu) / sigma)**2)
                   * erfcx(z[mask_pos]))

    # z < 0: rewrite as exp(σ²/(2τ²) - (t-mu)/τ) * erfc(|z|)
    # This avoids the exp(z²) overflow in erfcx for large negative z
    y[mask_neg] = (prefactor * np.exp(
        sigma**2 / (2.0 * tau**2) - (t[mask_neg] - mu) / tau)
        * erfc(-z[mask_neg]))

    return y + offset


def voigt(t, amplitude, mu, sigma, gamma, offset):
    """Voigt profile + constant baseline.

    Convolution of a Gaussian(mu, sigma) with a Lorentzian(HWHM=gamma).
    Uses the Faddeeva function ``wofz``.

    Parameters
    ----------
    t : ndarray
    amplitude : float — scaling factor
    mu : float — centre
    sigma : float — Gaussian width
    gamma : float — Lorentzian half-width at half-maximum
    offset : float — constant baseline
    """
    from scipy.special import wofz
    if sigma <= 1e-12 or gamma < 0:
        return np.full_like(t, 1e300, dtype=float)
    z = ((t - mu) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi)) + offset


# ---------------------------------------------------------------------------
# Multi-component EMG / Voigt
# ---------------------------------------------------------------------------

def emg_n(t, *params):
    """N-component EMG + constant baseline.

    Parameters
    ----------
    t : ndarray
    params : (4*N + 1) floats
        [amp1, mu1, sig1, tau1, ..., ampN, muN, sigN, tauN, offset]
    """
    n = (len(params) - 1) // 4
    y = np.full_like(t, params[-1], dtype=float)
    for i in range(n):
        a, m, s, tau_i = params[4 * i:4 * i + 4]
        y += emg(t, a, m, s, tau_i, 0.0)
    return y


def voigt_n(t, *params):
    """N-component Voigt + constant baseline.

    Parameters
    ----------
    t : ndarray
    params : (4*N + 1) floats
        [amp1, mu1, sig1, gam1, ..., ampN, muN, sigN, gamN, offset]
    """
    n = (len(params) - 1) // 4
    y = np.full_like(t, params[-1], dtype=float)
    for i in range(n):
        a, m, s, g_i = params[4 * i:4 * i + 4]
        y += voigt(t, a, m, s, g_i, 0.0)
    return y


# ---------------------------------------------------------------------------
# Model dispatch helpers
# ---------------------------------------------------------------------------

def get_params_per_component(model_type):
    """Return number of parameters per component for a given model."""
    if model_type == 'shapelet':
        return 0  # shapelet uses its own parametrisation (dict)
    return 4 if model_type in ('emg', 'voigt') else 3


def eval_profile_model(t, params, model_type):
    """Evaluate the appropriate profile model (single or multi-component).

    Parameters
    ----------
    t : ndarray
    params : ndarray — model parameters
    model_type : str — one of 'gaussian', 'emg', 'voigt'

    Returns
    -------
    ndarray — model evaluated at t.
    """
    if model_type == 'gaussian':
        return gaussian_n(t, *params)
    elif model_type == 'emg':
        return emg_n(t, *params)
    elif model_type == 'voigt':
        return voigt_n(t, *params)
    elif model_type == 'shapelet':
        # params is a dict from fit_shapelet_auto_n / toa_shapelet
        return shapelet_model(t, params['tau'], params['beta'],
                              params['coeffs'], params['poly_coeffs'],
                              params['t_mid'])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# Fit helpers — callable from --compare mode
# ---------------------------------------------------------------------------

def fit_single_gaussian_run(t_fit, y_fit, t_prof, profile, peak_idx, w0,
                             t_bin_width, nsamp_b, model_type='gaussian'):
    """Two-pass single-component fit.

    Returns dict with keys: popt, pcov, fit_lo, fit_hi, t_fit, y_fit, ok.
    """
    n_pc = get_params_per_component(model_type)  # 3 for gauss, 4 for emg/voigt
    n_param_tot = n_pc + 1

    result = {'ok': False, 'popt': None, 'pcov': None}
    p0_amp    = np.max(y_fit) - np.median(y_fit)
    p0_mean   = t_prof[peak_idx]
    p0_sigma  = w0 / 3.0
    p0_offset = np.median(y_fit)

    if model_type == 'gaussian':
        model_func = gaussian
        p0 = [p0_amp, p0_mean, p0_sigma, p0_offset]
    elif model_type == 'emg':
        model_func = emg
        p0_tau = w0 / 6.0
        p0 = [p0_amp, p0_mean, p0_sigma, p0_tau, p0_offset]
    elif model_type == 'voigt':
        model_func = voigt
        p0_gamma = w0 / 6.0
        p0 = [p0_amp, p0_mean, p0_sigma, p0_gamma, p0_offset]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # --- Pass 1: wide window ---
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            t_span_1 = t_fit[-1] - t_fit[0]
            if model_type == 'gaussian':
                lb1 = [-np.inf, t_fit[0], 1e-9, -np.inf]
                ub1 = [np.inf, t_fit[-1], 0.5 * t_span_1, np.inf]
                popt1, _ = curve_fit(model_func, t_fit, y_fit,
                                     p0=p0, bounds=(lb1, ub1), maxfev=20000)
            else:
                popt1, _ = curve_fit(model_func, t_fit, y_fit,
                                     p0=p0, maxfev=20000)
        except (RuntimeError, RuntimeWarning):
            return result

    # Validate pass-1 sigma before computing FWHM
    sigma_pass1 = popt1[2]
    if sigma_pass1 > 0 and sigma_pass1 < 0.5 * (t_fit[-1] - t_fit[0]):
        fwhm_pass1 = 2.35482 * sigma_pass1
    else:
        fwhm_pass1 = 2.35482 * (w0 / 3.0)
    tight_half = max(8, int(3 * fwhm_pass1 / t_bin_width))
    tight_lo = max(0, peak_idx - tight_half)
    tight_hi = min(nsamp_b, peak_idx + tight_half)
    t_tight = t_prof[tight_lo:tight_hi]
    y_tight = profile[tight_lo:tight_hi]

    # --- Pass 2: tight window ---
    p0_amp2    = np.max(y_tight) - np.median(y_tight)
    p0_mean2   = t_prof[peak_idx]
    p0_sigma2  = fwhm_pass1 / 2.35482
    p0_offset2 = np.median(y_tight)

    if model_type == 'gaussian':
        p0_2 = [p0_amp2, p0_mean2, p0_sigma2, p0_offset2]
    elif model_type == 'emg':
        p0_tau2 = abs(popt1[3]) if len(popt1) > 3 else w0 / 6.0
        p0_2 = [p0_amp2, p0_mean2, p0_sigma2, p0_tau2, p0_offset2]
    elif model_type == 'voigt':
        p0_gamma2 = abs(popt1[3]) if len(popt1) > 3 else w0 / 6.0
        p0_2 = [p0_amp2, p0_mean2, p0_sigma2, p0_gamma2, p0_offset2]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            popt, pcov = curve_fit(model_func, t_tight, y_tight,
                                   p0=p0_2, maxfev=20000)
            result['ok'] = True
            result['popt'] = popt
            result['pcov'] = pcov
            result['fit_lo'] = tight_lo
            result['fit_hi'] = tight_hi
            result['t_fit'] = t_tight
            result['y_fit'] = y_tight
        except (RuntimeError, RuntimeWarning):
            result['ok'] = True
            result['popt'] = popt1
            result['pcov'] = np.diag([np.nan] * n_param_tot)
            result['fit_lo'] = tight_lo
            result['fit_hi'] = tight_hi
            result['t_fit'] = t_tight
            result['y_fit'] = y_tight

    return result


def fit_multi_gaussian_run(t_fit, y_fit, profile, peak_indices, peak_idx,
                            w0, t_prof, fit_lo, fit_hi, n_comp,
                            model_type='gaussian'):
    """Multi-component fit with bounds.

    Returns dict with keys: popt, pcov, comps, fit_lo, fit_hi, t_fit, y_fit, ok.
    """
    n_pc = get_params_per_component(model_type)
    n_param = n_pc * n_comp + 1

    result = {'ok': False, 'popt': None, 'pcov': None, 'comps': None}
    t_min, t_max = t_fit[0], t_fit[-1]
    sigma_max = 0.5 * (t_max - t_min)
    extra_max = 0.5 * (t_max - t_min)  # tau / gamma upper bound

    p0_list = []
    lb, ub = [], []
    for i in range(n_comp):
        if i < len(peak_indices):
            p_idx = peak_indices[i]
            lo = max(fit_lo, p_idx - 5)
            hi = min(fit_hi, p_idx + 5)
            local_amp = profile[p_idx] - np.median(profile[lo:hi])
        else:
            local_amp = np.max(y_fit) - np.median(y_fit)
        mu_init = t_prof[p_idx] if i < len(peak_indices) else t_prof[peak_idx]
        sig_init = w0 / 6.0
        amp_init = local_amp if abs(local_amp) > 0 else np.max(y_fit) - np.median(y_fit)

        p0_list.extend([amp_init, mu_init, sig_init])
        lb.extend([-np.inf, t_min, 1e-6])
        ub.extend([np.inf,  t_max, sigma_max])

        if model_type in ('emg', 'voigt'):
            extra_init = w0 / 6.0
            p0_list.append(extra_init)
            lb.append(1e-9 if model_type == 'emg' else 0.0)
            ub.append(extra_max)

    p0_list.append(np.median(y_fit))
    lb.append(-np.inf)
    ub.append(np.inf)

    # Select model function
    if model_type == 'gaussian':
        model_func_n = gaussian_n
    elif model_type == 'emg':
        model_func_n = emg_n
    elif model_type == 'voigt':
        model_func_n = voigt_n
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            popt, pcov = curve_fit(
                model_func_n, t_fit, y_fit,
                p0=p0_list, bounds=(lb, ub), maxfev=50000
            )
            result['ok'] = True
            result['popt'] = popt
            result['pcov'] = pcov
            result['t_fit'] = t_fit
            result['y_fit'] = y_fit
            result['fit_lo'] = fit_lo
            result['fit_hi'] = fit_hi
        except (RuntimeError, RuntimeWarning):
            return result

    # Package per-component
    comps = []
    try:
        perr = np.sqrt(np.diag(pcov))
    except (ValueError, np.linalg.LinAlgError):
        perr = np.full(len(popt), np.nan)
    for i in range(n_comp):
        comp = {
            'amp': popt[n_pc * i],
            'mu': popt[n_pc * i + 1],
            'sig': abs(popt[n_pc * i + 2]),
            'amp_err': perr[n_pc * i],
            'mu_err': perr[n_pc * i + 1],
            'sig_err': perr[n_pc * i + 2],
        }
        if model_type == 'emg':
            comp['tau'] = abs(popt[n_pc * i + 3])
            comp['tau_err'] = perr[n_pc * i + 3]
        elif model_type == 'voigt':
            comp['gamma'] = abs(popt[n_pc * i + 3])
            comp['gamma_err'] = perr[n_pc * i + 3]
        comps.append(comp)
    result['comps'] = comps
    return result


def extract_toa_highest(comps, rms_off, t_bin_width):
    """Extract highest-amplitude component TOA."""
    k = np.argmax([abs(c['amp']) for c in comps])
    c = comps[k]
    fwhm_f = 2.35482 * abs(c['sig'])
    snr = abs(c['amp']) / rms_off if (rms_off and rms_off > 0) else np.inf
    n_eff = max(1, fwhm_f / t_bin_width)
    mu_emp = abs(c['sig']) / (snr * np.sqrt(n_eff)) if snr > 0 else np.inf
    return {
        'mu_f': c['mu'], 'mu_e': c['mu_err'], 'amp_f': c['amp'],
        'sig_f': c['sig'], 'sig_e': c['sig_err'],
        'fwhm_f': fwhm_f, 'snr': snr, 'mu_emp': mu_emp,
        'k': k, 'label': 'highest'
    }


def extract_toa_weighted(comps, popt, pcov, rms_off, t_bin_width, n_comp):
    """Extract amplitude-weighted-average TOA with propagated uncertainty."""
    amps = np.array([c['amp'] for c in comps])
    mus  = np.array([c['mu'] for c in comps])
    sigs = np.array([c['sig'] for c in comps])
    w = np.abs(amps)
    w_sum = np.sum(w)
    mu_f = np.sum(w * mus) / w_sum if w_sum > 0 else np.mean(mus)

    # Jacobian propagation
    J = np.zeros(len(popt))
    for i in range(n_comp):
        J[3 * i]     = (mus[i] - mu_f) / w_sum * np.sign(amps[i])
        J[3 * i + 1] = w[i] / w_sum
    try:
        mu_e = np.sqrt(J @ pcov @ J.T)
    except (ValueError, np.linalg.LinAlgError):
        mu_e = np.nan

    sig_f = np.average(sigs, weights=w) if w_sum > 0 else np.mean(sigs)
    fwhm_f = 2.35482 * abs(sig_f)
    snr = w_sum / n_comp / rms_off if (rms_off and rms_off > 0) else np.inf
    amp_f = w_sum / n_comp
    n_eff = max(1, fwhm_f / t_bin_width)
    mu_emp = abs(sig_f) / (snr * np.sqrt(n_eff)) if snr > 0 else np.inf
    return {
        'mu_f': mu_f, 'mu_e': mu_e, 'amp_f': amp_f,
        'sig_f': sig_f, 'sig_e': np.nan,
        'fwhm_f': fwhm_f, 'snr': snr, 'mu_emp': mu_emp,
        'k': -1, 'label': 'weighted'
    }


def extract_toa_error_weighted(comps, popt, pcov, rms_off, t_bin_width, n_comp):
    """Inverse-variance weighted TOA — weights each component by 1 / sigma_mu^2.
    Naturally down-weights poorly constrained components.
    """
    amps = np.array([c['amp'] for c in comps])
    mus  = np.array([c['mu'] for c in comps])
    sigs = np.array([c['sig'] for c in comps])
    mu_errs = np.array([c['mu_err'] for c in comps])

    # Use only components with finite, positive uncertainty
    valid = np.isfinite(mu_errs) & (mu_errs > 0) & (np.abs(amps) > 0)
    if not np.any(valid):
        return extract_toa_highest(comps, rms_off, t_bin_width)

    w = 1.0 / mu_errs[valid]**2
    w_sum = np.sum(w)
    mu_f = np.sum(w * mus[valid]) / w_sum
    mu_e = 1.0 / np.sqrt(w_sum) if w_sum > 0 else np.nan

    sig_f = np.average(sigs[valid], weights=w) if w_sum > 0 else np.mean(sigs[valid])
    amp_f = np.average(np.abs(amps[valid]), weights=w)
    fwhm_f = 2.35482 * abs(sig_f)
    snr = np.max(np.abs(amps[valid])) / rms_off if (rms_off and rms_off > 0) else np.inf
    n_eff = max(1, fwhm_f / t_bin_width)
    mu_emp = abs(sig_f) / (snr * np.sqrt(n_eff)) if snr > 0 else np.inf
    return {
        'mu_f': mu_f, 'mu_e': mu_e, 'amp_f': amp_f,
        'sig_f': sig_f, 'sig_e': np.nan,
        'fwhm_f': fwhm_f, 'snr': snr, 'mu_emp': mu_emp,
        'k': -1, 'label': 'error_weighted'
    }


def extract_toa_first_peak(comps, rms_off, t_bin_width):
    """Earliest significant (S/N > 5) component TOA.

    Requires each candidate component to have sigma >= 3*t_bin_width
    to reject narrow noise spikes, and S/N > 5 to exclude RFI-generated
    pseudo-components.  If no component passes both cuts, returns None
    (the strategy is skipped for this pulse).
    """
    min_sigma = max(t_bin_width * 3.0, 1e-9)
    sig_comps = [c for c in comps
                 if abs(c['sig']) >= min_sigma
                 and abs(c['amp']) / rms_off > 5.0
                 if rms_off and rms_off > 0]
    if not sig_comps:
        return None  # skip this strategy — no reliable early component
    c = min(sig_comps, key=lambda x: x['mu'])
    fwhm_f = 2.35482 * abs(c['sig'])
    snr = abs(c['amp']) / rms_off if (rms_off and rms_off > 0) else np.inf
    n_eff = max(1, fwhm_f / t_bin_width)
    mu_emp = abs(c['sig']) / (snr * np.sqrt(n_eff)) if snr > 0 else np.inf
    k = next(i for i, cc in enumerate(comps) if cc is c)
    return {
        'mu_f': c['mu'], 'mu_e': c['mu_err'], 'amp_f': c['amp'],
        'sig_f': c['sig'], 'sig_e': c['sig_err'],
        'fwhm_f': fwhm_f, 'snr': snr, 'mu_emp': mu_emp,
        'k': k, 'label': 'first_peak'
    }


# ---------------------------------------------------------------------------
# Non-parametric TOA estimators
# ---------------------------------------------------------------------------

def toa_leading_edge(t_prof, profile, fit_lo, fit_hi, rms_off):
    """TOA from the half-maximum crossing on the rising edge, corrected to
    the peak level by adding the observed rise time.

    The half-max point on the rising flank is scattering-invariant
    (multipath propagation broadens only the trailing edge).  Adding
    the observed rise time  t_peak - t_hm  shifts the reference from
    the 50 % level to the peak level, eliminating the pulse-width-
    dependent offset that a fixed-fraction crossing introduces.  For
    a symmetric Gaussian pulse the result is the true peak position.
    """
    sub = profile[fit_lo:fit_hi]
    t_sub = t_prof[fit_lo:fit_hi]
    if len(sub) < 3:
        return {'mu_f': np.nan, 'mu_e': np.nan, 'label': 'leading_edge'}

    peak_val = np.max(sub)
    peak_idx = np.argmax(sub)
    t_peak_obs = t_sub[peak_idx]
    baseline = np.median(sub[:max(1, peak_idx // 2)])
    peak_above_bl = peak_val - baseline
    if peak_above_bl <= 0:
        return {'mu_f': t_peak_obs, 'mu_e': np.nan,
                'label': 'leading_edge'}

    half_max = baseline + 0.5 * peak_above_bl

    # Walk backward from peak to find the half-max crossing
    for i in range(peak_idx - 1, 0, -1):
        if sub[i] <= half_max <= sub[i + 1]:
            frac = (half_max - sub[i]) / (sub[i + 1] - sub[i])
            t_hm = t_sub[i] + frac * (t_sub[1] - t_sub[0])
            # Correct from 50 % to peak level
            rise_time = t_peak_obs - t_hm
            mu_f = t_hm + rise_time  # = t_peak_obs for symmetric pulse
            snr = peak_val / rms_off if (rms_off and rms_off > 0) else np.inf
            mu_e = rise_time / snr if snr > 0 else np.nan
            return {'mu_f': mu_f, 'mu_e': mu_e,
                    'label': 'leading_edge', 'snr': snr}

    # No half-max crossing found — use peak
    return {'mu_f': t_peak_obs, 'mu_e': np.nan,
            'label': 'leading_edge'}


def toa_center_of_mass(t_prof, profile, fit_lo, fit_hi, rms_off):
    """Non-parametric TOA from the first moment (center of mass).

    Uses only points above 3 x off-pulse RMS to exclude noise.
    """
    sub = profile[fit_lo:fit_hi]
    t_sub = t_prof[fit_lo:fit_hi]
    threshold = 3.0 * rms_off if (rms_off and rms_off > 0) else 0.0

    mask = sub > threshold
    if not np.any(mask):
        peak_idx = np.argmax(sub)
        return {'mu_f': t_sub[peak_idx], 'mu_e': np.nan,
                'label': 'center_of_mass'}

    signal = sub[mask] - threshold
    t_signal = t_sub[mask]
    total = np.sum(signal)
    mu_f = np.sum(t_signal * signal) / total if total > 0 else np.nan

    # Uncertainty: estimate from weighted std of signal region
    if total > 0:
        rms_signal = np.sqrt(np.sum(signal * (t_signal - mu_f)**2) / total)
        snr = total / (rms_off * np.sqrt(len(signal))) if rms_off > 0 else np.inf
        mu_e = rms_signal / np.sqrt(total / rms_off) if rms_off > 0 else np.nan
    else:
        mu_e = np.nan

    return {'mu_f': mu_f, 'mu_e': mu_e,
            'label': 'center_of_mass',
            'snr': np.max(sub) / rms_off if rms_off and rms_off > 0 else np.inf}


# ---------------------------------------------------------------------------
# Peak-finding TOA (simplest non-parametric method)
# ---------------------------------------------------------------------------

def toa_peak(t_prof, profile, fit_lo, fit_hi, rms_off, t_bin_width):
    """TOA from the profile peak with quadratic sub-bin interpolation.

    The simplest possible method: find the brightest bin and refine its
    position via a three-point quadratic fit.  Model-independent and fast,
    but sensitive to noise and scattering tails.
    """
    sub = profile[fit_lo:fit_hi]
    t_sub = t_prof[fit_lo:fit_hi]

    if len(sub) < 3:
        return {'mu_f': np.nan, 'mu_e': np.nan, 'label': 'peak',
                'snr': np.nan}

    idx = np.argmax(sub)
    peak_val = sub[idx]
    snr = peak_val / rms_off if rms_off and rms_off > 0 else np.inf

    # Quadratic interpolation for sub-bin precision
    if 1 <= idx < len(sub) - 1:
        y0, y1, y2 = sub[idx - 1], sub[idx], sub[idx + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-15:
            delta = 0.5 * (y0 - y2) / denom
            mu_f = t_sub[idx] + delta * (t_sub[1] - t_sub[0])
        else:
            mu_f = t_sub[idx]
    else:
        mu_f = t_sub[idx]

    # Uncertainty: bin width / SNR (quantisation floor)
    mu_e = t_bin_width / snr if snr > 0 else np.nan

    return {'mu_f': mu_f, 'mu_e': mu_e, 'label': 'peak', 'snr': snr}


# ---------------------------------------------------------------------------
# Shapelet-based TOA (non-parametric / basis-expansion method)
# ---------------------------------------------------------------------------

def toa_shapelet(t_prof, profile, fit_lo, fit_hi, rms_off, t_bin_width,
                 n_max=20, n_poly=2, alpha=0.0):
    """Shapelet-decomposition TOA — flexible non-parametric method.

    Fits a Hermite-polynomial × Gaussian-envelope model to the pulse
    profile, selecting the number of shapelet components automatically
    via AICc.  The TOA is the centre of the Gaussian envelope (τ).

    The model (Ashton et al. 2020; Refregier 2003):
      S(t) = Σ C_i · H_i((t-τ)/β) · exp(-((t-τ)/β)²) + Σ B_j · (t-t_mid)^j

    Linear parameters {C_i}, {B_j} are solved by least-squares at each
    (τ, β); the 2D nonlinear optimisation over (τ, β) uses L-BFGS-B.

    Parameters
    ----------
    t_prof, profile : ndarray
    fit_lo, fit_hi : int — indices bracketing the pulse region
    rms_off : float — off-pulse RMS
    t_bin_width : float
    n_max : int — maximum shapelet order (default 20; scanned in steps of 2)
    n_poly : int — polynomial baseline degree (default 2)
    alpha : float — ridge regularisation (default 0)

    Returns
    -------
    dict — standard TOA info plus shapelet-specific keys (beta, n_shapelet,
    aicc, shapelet_result).
    """
    sub = profile[fit_lo:fit_hi]
    t_sub = t_prof[fit_lo:fit_hi]

    if len(sub) < 10:
        return {'mu_f': np.nan, 'mu_e': np.nan, 'label': 'shapelet'}

    # --- Initial guesses ---
    peak_idx = np.argmax(sub)
    tau_guess = t_sub[peak_idx]
    peak_val = sub[peak_idx]

    # Estimate β from the pulse width (FWHM / 2)
    half_max = 0.5 * peak_val
    lo_half = peak_idx
    hi_half = peak_idx
    for i in range(peak_idx, 0, -1):
        if sub[i] <= half_max:
            lo_half = i
            break
    for i in range(peak_idx, len(sub)):
        if sub[i] <= half_max:
            hi_half = i
            break
    fwhm_est = (t_sub[hi_half] - float(t_sub[lo_half])) if hi_half > lo_half else t_bin_width * 5.0
    beta_guess = max(fwhm_est / 2.0, t_bin_width * 2.0)

    t_mid = 0.5 * (float(t_sub[0]) + float(t_sub[-1]))

    # --- Fit ---
    try:
        best = fit_shapelet_auto_n(t_sub, sub, tau_guess, beta_guess,
                                   n_max, n_poly, t_mid, alpha)
    except Exception:
        return {'mu_f': tau_guess, 'mu_e': t_bin_width,
                'label': 'shapelet',
                'snr': peak_val / rms_off if rms_off and rms_off > 0 else np.inf}

    tau_opt = best['tau']
    beta_opt = best['beta']

    # --- Evaluate the fitted model on a fine grid ---
    # The shapelet centre τ may drift when compensated by higher-order
    # coefficients.  We therefore use the PEAK of the *reconstructed model*
    # as the TOA, which is robust against this trade-off.
    n_fine = max(len(t_sub) * 8, 200)
    t_fine = np.linspace(float(t_sub[0]), float(t_sub[-1]), n_fine)
    try:
        model_fine = shapelet_model(t_fine, tau_opt, beta_opt,
                                    best['coeffs'], best['poly_coeffs'],
                                    t_mid)
        # Remove polynomial baseline to isolate the pulse component
        poly_baseline = np.zeros_like(t_fine)
        dt_f = t_fine - t_mid
        for j, b in enumerate(best['poly_coeffs']):
            poly_baseline += b * dt_f ** j
        pulse_model = model_fine - poly_baseline

        # Peak with sub-bin precision via quadratic interpolation
        idx_peak = np.argmax(pulse_model)
        if 1 <= idx_peak < n_fine - 1:
            y0, y1, y2 = (pulse_model[idx_peak - 1],
                          pulse_model[idx_peak],
                          pulse_model[idx_peak + 1])
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-15:
                delta = 0.5 * (y0 - y2) / denom
                mu_f = t_fine[idx_peak] + delta * (t_fine[1] - t_fine[0])
            else:
                mu_f = t_fine[idx_peak]
        else:
            mu_f = t_fine[idx_peak]

        peak_model = float(np.max(pulse_model))
    except Exception:
        mu_f = tau_opt
        peak_model = float(np.max(np.abs(best['coeffs'])))

    snr = peak_model / rms_off if rms_off and rms_off > 0 else np.inf

    # --- Uncertainty: leading-edge style on the denoised model ---
    # Find half-max crossing on the rising edge of the pulse model
    try:
        half_max = 0.5 * peak_model
        idx_peak_f = np.argmax(pulse_model)
        cross_idx = None
        for i in range(idx_peak_f, 0, -1):
            if pulse_model[i] <= half_max <= pulse_model[i + 1]:
                frac = (half_max - pulse_model[i]) / max(
                    pulse_model[i + 1] - pulse_model[i], 1e-30)
                t_hm = t_fine[i] + frac * (t_fine[1] - t_fine[0])
                cross_idx = i
                break
        if cross_idx is not None:
            rise_time = t_fine[idx_peak_f] - t_hm
            mu_e = max(rise_time / snr if snr > 0 else np.nan,
                       t_bin_width / np.sqrt(12.0))  # floor at quantisation err
        else:
            mu_e = np.nan
    except Exception:
        mu_e = np.nan

    fwhm_f = 2.0 * beta_opt
    n_eff = max(1, fwhm_f / t_bin_width)
    mu_emp = abs(beta_opt) / (snr * np.sqrt(n_eff)) if snr > 0 and np.isfinite(snr) else np.nan

    return {
        'mu_f': mu_f,
        'mu_e': mu_e if np.isfinite(mu_e) else mu_emp,
        'mu_emp': mu_emp,
        'label': 'shapelet',
        'snr': snr,
        'fwhm_f': fwhm_f,
        'beta': beta_opt,
        'n_shapelet': best['n_shapelet'],
        'aicc': best.get('aicc', np.nan),
        'shapelet_result': best,
    }


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def compute_aicc(rss, n_param, n_data):
    """Corrected AIC for small samples.

    AICc = N * ln(RSS/N) + 2k + 2k(k+1)/(N-k-1)
    Lower AICc → better model.
    """
    if n_data <= n_param + 1:
        return np.inf
    aic = n_data * np.log(rss / n_data) + 2.0 * n_param
    correction = 2.0 * n_param * (n_param + 1.0) / (n_data - n_param - 1.0)
    return aic + correction


def fit_multi_gaussians_scan(t_fit, y_fit, profile, peak_indices, peak_idx,
                              w0, t_prof, fit_lo, fit_hi, max_n, t_bin_width,
                              model_type='gaussian'):
    """Fit 1 through max_n components; return fit results + AICc for each N.

    Returns list of dicts, one per N, with keys: n, popt, pcov, comps,
    rss, aicc, ok.
    """
    n_pc = get_params_per_component(model_type)
    results = []
    for n in range(1, max_n + 1):
        r = fit_multi_gaussian_run(t_fit, y_fit, profile,
                                   peak_indices[:n], peak_indices[0],
                                   w0, t_prof, fit_lo, fit_hi, n,
                                   model_type=model_type)
        entry = {'n': n, 'ok': r['ok']}
        if r['ok']:
            resid = y_fit - eval_profile_model(t_fit, r['popt'], model_type)
            rss = np.sum(resid**2)
            n_param = n_pc * n + 1
            aicc = compute_aicc(rss, n_param, len(y_fit))
            entry['popt'] = r['popt']
            entry['pcov'] = r['pcov']
            entry['comps'] = r['comps']
            entry['rss'] = rss
            entry['aicc'] = aicc
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Sub-band TOA fitting -- for overlapping components
# ---------------------------------------------------------------------------

def sub_band_toa_fit(data_2d, t_prof, dm, f_low_ghz, f_high_ghz, nchan_orig,
                     t0_sample, w0, bs, bf, n_sub, t_bin_width,
                     rms_off, tsamp_ms):
    """Fit TOA independently in frequency sub-bands, correct to top of band.

    For overlapping multi-component pulses, sub-band TOA measurements
    provide (a) a more robust TOA via weighted averaging and (b) a
    realistic uncertainty estimate from the inter-band scatter.

    Returns dict with keys:
        toa_ref : DM-corrected weighted-avg TOA at reference frequency
        toa_err : uncertainty (std error of weighted mean)
        toa_scatter : RMS scatter of sub-band TOAs
        sub_bands : list of {f_lo, f_hi, toa, toa_err, snr} per sub-band
    """
    nchan_total, nsamp = data_2d.shape
    ch_per_sub = nchan_total // n_sub
    if ch_per_sub < 4:
        n_sub = max(1, nchan_total // 4)
        ch_per_sub = nchan_total // n_sub

    sub_results = []

    for i_sub in range(n_sub):
        ch_lo = i_sub * ch_per_sub
        ch_hi = (i_sub + 1) * ch_per_sub if i_sub < n_sub - 1 else nchan_total
        f_lo_sub = f_low_ghz + (f_high_ghz - f_low_ghz) * ch_lo / nchan_total
        f_hi_sub = f_low_ghz + (f_high_ghz - f_low_ghz) * ch_hi / nchan_total
        f_center_sub = 0.5 * (f_lo_sub + f_hi_sub)

        # Form sub-band profile
        sub_data = data_2d[ch_lo:ch_hi, :]
        sub_profile = np.sum(sub_data, axis=0)

        # Simple single-Gaussian fit to sub-band profile
        peak_offset = np.argmax(sub_profile)
        peak_val = sub_profile[peak_offset]
        rms_sub = np.std(np.concatenate(
            [sub_profile[:max(1, peak_offset // 3)],
             sub_profile[min(nsamp, peak_offset + nsamp // 3):]]))

        if peak_val < 4 * rms_sub:
            continue  # too faint in this sub-band

        # Fit window around peak
        half_w = int(w0 / t_bin_width) + 10
        lo_sub = max(0, peak_offset - half_w)
        hi_sub = min(nsamp, peak_offset + half_w)
        t_sub = t_prof[lo_sub:hi_sub]
        y_sub = sub_profile[lo_sub:hi_sub]

        p0 = [peak_val - np.median(y_sub), t_prof[peak_offset],
              w0 / 3.0, np.median(y_sub)]

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                from scipy.optimize import curve_fit
                popt_s, pcov_s = curve_fit(
                    gaussian, t_sub, y_sub, p0=p0, maxfev=10000)
                mu_sub = popt_s[1]
                mu_err_s = np.sqrt(pcov_s[1, 1]) if pcov_s[1, 1] > 0 else np.nan
                snr_s = abs(popt_s[0]) / rms_sub if rms_sub > 0 else np.inf
            except Exception:
                mu_sub = t_prof[peak_offset]
                mu_err_s = t_bin_width
                snr_s = peak_val / rms_sub if rms_sub > 0 else np.inf

        sub_results.append({
            'f_lo': f_lo_sub, 'f_hi': f_hi_sub, 'f_center': f_center_sub,
            'toa': mu_sub,
            'toa_err': mu_err_s, 'snr': snr_s,
        })

    # Post-fit filter: discard sub-bands with S/N < 3
    sub_results = [s for s in sub_results if s['snr'] >= 3.0]

    if not sub_results:
        return {'toa_ref': np.nan, 'toa_err': np.nan,
                'toa_scatter': np.nan, 'sub_bands': []}

    # After dedispersion, all sub-bands should agree on the TOA.
    # Weighted average (no DM correction needed — data is already dedispersed).
    # The cross-band scatter captures systematic uncertainty from
    # component overlap and profile evolution.
    toa_vals = np.array([s['toa'] for s in sub_results])
    errs = np.array([max(s['toa_err'], 1e-12) for s in sub_results])
    # Filter outliers (>3 sigma from median) — removes spurious fits
    med_toa = np.median(toa_vals)
    mad = np.median(np.abs(toa_vals - med_toa))
    valid = np.abs(toa_vals - med_toa) < 5.0 * max(mad, 1e-9)
    if np.sum(valid) >= 2:
        toa_vals = toa_vals[valid]
        errs = errs[valid]

    weights = 1.0 / errs**2
    w_sum = np.sum(weights)
    toa_avg = np.sum(weights * toa_vals) / w_sum if w_sum > 0 else np.mean(toa_vals)
    toa_err_weighted = 1.0 / np.sqrt(w_sum) if w_sum > 0 else np.nan

    # Cross-band scatter: the real-world systematic uncertainty
    n_eff = len(toa_vals)
    toa_scatter = np.std(toa_vals) / np.sqrt(n_eff) if n_eff > 1 else np.nan

    return {
        'toa_ref': toa_avg, 'toa_err': toa_err_weighted,
        'toa_scatter': toa_scatter, 'sub_bands': sub_results,
    }


# ---------------------------------------------------------------------------
# MCMC uncertainty estimation
# ---------------------------------------------------------------------------

def log_prior(params, t_min, t_max, sigma_max, model_type='gaussian'):
    """Flat priors with physically motivated bounds.

    Parameters
    ----------
    params : ndarray
        Model parameters depending on ``model_type``.
    t_min, t_max : float
        Allowed range for mu_i (fit window bounds).
    sigma_max : float
        Maximum allowed sigma (0.5 * window width).
    model_type : str
        'gaussian', 'emg', or 'voigt'.

    Returns
    -------
    float
        0 if within bounds, -inf if any parameter out of bounds.
    """
    n_pc = get_params_per_component(model_type)
    n = (len(params) - 1) // n_pc
    for i in range(n):
        mu_i = params[n_pc * i + 1]
        sig_i = params[n_pc * i + 2]
        if sig_i <= 1e-9 or sig_i > sigma_max:
            return -np.inf
        if mu_i < t_min or mu_i > t_max:
            return -np.inf
        extra = params[n_pc * i + 3] if n_pc > 3 else None
        if extra is not None:
            if model_type == 'emg' and extra <= 1e-9:
                return -np.inf
            elif model_type == 'voigt' and extra < 0:
                return -np.inf
    return 0.0


def log_likelihood(params, t, y, rms, model_type='gaussian'):
    """Chi-squared log-likelihood assuming Gaussian noise.

    Returns log-likelihood = -0.5 * chi^2.
    """
    model = eval_profile_model(t, params, model_type)
    chi2 = np.sum(((y - model) / rms) ** 2)
    return -0.5 * chi2


def log_probability(params, t, y, rms, t_min, t_max, sigma_max, n_comp,
                    model_type='gaussian'):
    """Posterior: log_prior + log_likelihood."""
    lp = log_prior(params, t_min, t_max, sigma_max, model_type)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, t, y, rms, model_type)


def compute_gelman_rubin(chains):
    """Gelman-Rubin R-hat statistic (split-chain method).

    Parameters
    ----------
    chains : ndarray (n_steps, n_walkers, n_dim)

    Returns
    -------
    float
        Maximum R-hat across all parameters. Values near 1 indicate convergence.
    """
    n_steps, n_walkers, n_dim = chains.shape
    half = n_steps // 2
    if half < 2:
        return np.nan
    # Split each walker into two halves
    split = chains[:2 * half].reshape(2 * half, n_walkers, n_dim)
    c1 = split[:half]
    c2 = split[half:2 * half]
    # Combine: 2 * n_walkers chains, each of length `half`
    all_chains = np.concatenate([c1, c2], axis=1)  # (half, 2*n_walkers, n_dim)
    M = all_chains.shape[1]   # number of chains = 2 * n_walkers
    N = half                  # length per chain
    chain_means = np.mean(all_chains, axis=0)       # (M, n_dim)
    grand_mean = np.mean(chain_means, axis=0)       # (n_dim,)
    B = N / (M - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    W = np.mean(np.var(all_chains, axis=0), axis=0)  # (n_dim,)
    var_hat = (N - 1) / N * W + B / N
    R_hat = np.sqrt(var_hat / W)
    return float(np.nanmax(R_hat))


def run_mcmc(t_fit, y_fit, rms_off, p0, pcov, n_comp,
             n_walkers=32, n_burn=500, n_prod=1000,
             progress=False, model_type='gaussian'):
    """Run emcee MCMC to sample posterior of model parameters.

    Uses the ``curve_fit`` solution as the starting point and its covariance
    matrix to set the initial walker distribution scale.

    Parameters
    ----------
    t_fit, y_fit : ndarray
    rms_off : float
    p0 : ndarray — best-fit params from ``curve_fit``
    pcov : ndarray — covariance matrix
    n_comp : int
    n_walkers, n_burn, n_prod : int
    progress : bool
    model_type : str — 'gaussian', 'emg', or 'voigt'

    Returns
    -------
    dict with keys: sampler, flat_samples, gr_stat,
        acceptance_fraction, ok.
    """
    import emcee

    n_pc = get_params_per_component(model_type)
    n_dim = n_pc * n_comp + 1

    result = {'ok': False, 'sampler': None, 'flat_samples': None,
              'gr_stat': np.nan, 'acceptance_fraction': np.nan}

    n_walkers = max(n_walkers, 2 * n_dim)
    if n_walkers % 2 != 0:
        n_walkers += 1

    t_min, t_max = t_fit[0], t_fit[-1]
    sigma_max = 0.5 * (t_max - t_min)

    # Initialise walkers as Gaussian ball around p0
    try:
        pcov_diag = np.diag(pcov)
        if np.any(~np.isfinite(pcov_diag)) or np.any(pcov_diag <= 0):
            raise ValueError("invalid covariance diagonal")
        scales = np.sqrt(pcov_diag) * 0.01
    except (ValueError, np.linalg.LinAlgError):
        scales = np.abs(p0) * 1e-4 + 1e-12

    rng = np.random.default_rng()
    p0_ball = p0 + rng.normal(scale=scales, size=(n_walkers, n_dim))

    # Ensure walkers respect prior bounds
    for i in range(n_comp):
        p0_ball[:, n_pc * i + 2] = np.abs(p0_ball[:, n_pc * i + 2]) + 1e-9
        if n_pc > 3:
            p0_ball[:, n_pc * i + 3] = np.abs(p0_ball[:, n_pc * i + 3]) + 1e-9

    if progress:
        print(f"\n  MCMC: {n_walkers} walkers, {n_dim} dims, "
              f"{n_burn} burn-in + {n_prod} production steps", flush=True)

    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_probability,
        args=(t_fit, y_fit, rms_off, t_min, t_max, sigma_max, n_comp,
              model_type)
    )

    # Burn-in
    try:
        state = sampler.run_mcmc(p0_ball, n_burn, progress=False)
        sampler.reset()
    except Exception:
        return result

    # Production
    try:
        sampler.run_mcmc(state, n_prod, progress=False)
    except Exception:
        return result

    result['sampler'] = sampler
    result['flat_samples'] = sampler.get_chain(flat=True)
    result['acceptance_fraction'] = float(np.mean(sampler.acceptance_fraction))
    result['gr_stat'] = compute_gelman_rubin(sampler.get_chain())
    result['ok'] = True

    if progress:
        print(f"  MCMC accept fraction: {result['acceptance_fraction']:.3f}"
              f"  |  R-hat = {result['gr_stat']:.4f}", flush=True)

    return result


def mcmc_toa_uncertainty(flat_samples, n_comp, model_type='gaussian'):
    """Extract TOA and uncertainties from MCMC posterior samples.

    For single-component, TOA = mu_1.  For multi-component, the
    highest-|amplitude| component's mu is tracked per posterior sample.

    Parameters
    ----------
    flat_samples : ndarray (n_samples, n_params)
    n_comp : int
    model_type : str

    Returns
    -------
    dict with keys: mu_mcmc, mu_mcmc_std, mu_mcmc_16, mu_mcmc_84, ...
    """
    n_pc = get_params_per_component(model_type)
    if n_comp == 1:
        mu_samples = flat_samples[:, 1]
        sigma_samples = np.abs(flat_samples[:, 2])
        amp_samples = flat_samples[:, 0]
    else:
        mu_list = []
        amp_idx = [n_pc * i for i in range(n_comp)]
        mu_idx = [n_pc * i + 1 for i in range(n_comp)]
        for row in flat_samples:
            amps = np.abs(row[amp_idx])
            k = np.argmax(amps)
            mu_list.append(row[mu_idx[k]])
        mu_samples = np.array(mu_list)
        sigma_samples = np.nan
        amp_samples = np.nan

    return {
        'mu_mcmc': float(np.mean(mu_samples)),
        'mu_mcmc_std': float(np.std(mu_samples)),
        'mu_mcmc_16': float(np.percentile(mu_samples, 16)),
        'mu_mcmc_84': float(np.percentile(mu_samples, 84)),
        'sigma_mcmc': float(np.mean(sigma_samples)) if np.ndim(sigma_samples) > 0 else np.nan,
        'sigma_mcmc_std': float(np.std(sigma_samples)) if np.ndim(sigma_samples) > 0 else np.nan,
        'amp_mcmc': float(np.mean(amp_samples)) if np.ndim(amp_samples) > 0 else np.nan,
        'label': 'mcmc',
    }


def plot_corner(flat_samples, labels, output_path):
    """Generate a corner plot using the ``corner`` package.

    Parameters
    ----------
    flat_samples : ndarray
        Production chain samples.
    labels : list of str
        Parameter labels, e.g. ['A1', 'mu1', 'sig1', 'offset'].
    output_path : str
        Path for saving the figure.

    Returns
    -------
    fig : matplotlib Figure or None if corner is not available.
    """
    try:
        import corner
    except ImportError:
        print("  WARNING: corner not installed. Install with: pip install corner")
        return None
    fig = corner.corner(flat_samples, labels=labels,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_fmt='.6f')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  -> {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# .tim file helper (shared by compare and normal paths)
# ---------------------------------------------------------------------------

def _write_tim_file(args, fit_ok, compare_results, mu_f, mu_emp, mu_e,
                    toa_mode, mjd_start, sec_per_day, toa_freq_mhz,
                    mcmc_result=None, mcmc_n_comp=1, model_type='gaussian'):
    """Write TEMPO2-compatible .tim file(s).

    With --tim_strategy all, writes one .tim file per available strategy.
    """
    fname = args.input_file.replace(chr(92), '/').split('/')[-1]

    # --- Collect all available strategies ---
    all_entries = []  # list of (label, toa_s, unc_s)
    if compare_results is not None:
        cr = compare_results
        # Non-parametric + single
        if 'single' in cr and np.isfinite(cr['single'].get('mu', np.nan)):
            e = cr['single']
            all_entries.append(('single', e['mu'],
                                e.get('mu_e', np.nan)))
        for key in ['leading_edge', 'center_of_mass', 'peak', 'shapelet']:
            if key in cr and np.isfinite(cr[key].get('mu_f', np.nan)):
                e = cr[key]
                all_entries.append((key, e['mu_f'],
                                    e.get('mu_e', e.get('mu_emp', np.nan))))
        # Multi-Gaussian parametric (best N only)
        sm = cr.get('sorted_multi', [])
        seen_modes = set()
        if sm:
            best_n = sm[0]['n']
            for entry in sm:
                if entry['n'] == best_n and entry['mode'] not in seen_modes:
                    all_entries.append(
                        (f"{best_n}-Gauss {entry['mode']}", entry['mu'],
                         entry.get('mu_emp', entry.get('mu_e', np.nan))))
                    seen_modes.add(entry['mode'])
    else:
        # No compare: just the current fit
        all_entries.append((toa_mode, mu_f, mu_emp if np.isfinite(mu_emp) and mu_emp > 0 else mu_e))

    # --- Resolve ---
    tim_strat = args.tim_strategy.lower()
    if tim_strat == 'all':
        # Write one .tim per strategy, plus a *_best.tim
        base = args.output
        # --- Determine recommended best ---
        best_label = None
        best_toa = None
        best_unc = None
        sm = compare_results.get('sorted_multi', []) if compare_results else []
        cr_single = compare_results.get('single', {}) if compare_results else {}
        cr_le = compare_results.get('leading_edge', {}) if compare_results else {}
        cr_sl = compare_results.get('shapelet', {}) if compare_results else {}
        sigma_emp_s = cr_single.get('mu_e', np.nan)
        aicc_single = cr_single.get('aicc', np.inf)
        if sm and sigma_emp_s > 0 and np.isfinite(sigma_emp_s):
            best_entry = sm[0]
            delta_conv = abs(best_entry['mu'] - cr_single['mu'])
            if delta_conv < 3 * sigma_emp_s:
                aicc_multi = best_entry.get('aicc', np.inf)
                if np.isfinite(aicc_single) and aicc_single <= aicc_multi:
                    best_label, best_toa, best_unc = 'single', cr_single['mu'], cr_single.get('mu_e', np.nan)
                else:
                    best_label = best_entry.get('label', best_entry.get('mode', 'best'))
                    best_toa = best_entry['mu']
                    best_unc = best_entry.get('mu_emp', best_entry.get('mu_e', np.nan))
            else:
                le_ok = (cr_le and np.isfinite(cr_le.get('mu_f', np.nan)) and cr_le.get('snr', 0) > 3)
                sl_ok = (cr_sl and np.isfinite(cr_sl.get('mu_f', np.nan)) and cr_sl.get('snr', 0) > 5)
                if le_ok:
                    best_label = 'leading_edge'
                    best_toa = cr_le['mu_f']
                    best_unc = cr_le.get('mu_e', cr_le.get('mu_emp', np.nan))
                elif sl_ok:
                    best_label = 'shapelet'
                    best_toa = cr_sl['mu_f']
                    best_unc = cr_sl.get('mu_e', cr_sl.get('mu_emp', np.nan))
                else:
                    best_label, best_toa, best_unc = 'single', cr_single.get('mu', np.nan), cr_single.get('mu_e', np.nan)
        elif cr_single:
            best_label, best_toa, best_unc = 'single', cr_single.get('mu', np.nan), cr_single.get('mu_e', np.nan)

        # --- Print table ---
        print(f"\n  All available TOA strategies ({len(all_entries)}):")
        print(f"  {'Strategy':<30s} {'TOA (MJD)':<20s} {'+/- (us)':<12s} {'File'}")
        print(f"  " + "-" * 90)
        for label, toa_s, unc_s in all_entries:
            unc_us = unc_s * 1e6 if np.isfinite(unc_s) and unc_s > 0 else mu_emp * 1e6
            toa_mjd = mjd_start + toa_s / sec_per_day
            safe_label = label.replace(' ', '_').replace('(', '').replace(')', '')
            tim_path = f"{base}_{safe_label}.tim"
            with open(tim_path, 'w') as ft:
                ft.write(f"# {fname}\n")
                ft.write(f"# strategy: {label}\n")
                ft.write("FORMAT 1\n")
                ft.write(f"{fname}  {toa_freq_mhz:.4f}  {toa_mjd:.12f}  "
                         f"{unc_us:.3f}  {args.telescope}\n")
            marker = ' <-- RECOMMENDED' if label == best_label else ''
            print(f"  {label:<30s} {toa_mjd:<20.12f} {unc_us:<12.1f} {tim_path}{marker}")
        print(f"  " + "-" * 90)

        # --- Write best.tim ---
        if best_label and best_toa is not None:
            best_unc_us = best_unc * 1e6 if np.isfinite(best_unc) and best_unc > 0 else mu_emp * 1e6
            best_mjd = mjd_start + best_toa / sec_per_day
            best_path = f"{base}_best.tim"
            with open(best_path, 'w') as ft:
                ft.write(f"# {fname}\n")
                ft.write(f"# strategy: {best_label} (recommended)\n")
                ft.write("FORMAT 1\n")
                ft.write(f"{fname}  {toa_freq_mhz:.4f}  {best_mjd:.12f}  "
                         f"{best_unc_us:.3f}  {args.telescope}\n")
            print(f"  => Recommended: {best_label} -> {best_path}")
        return

    # --- Single strategy ---
    tim_toa = mu_f
    tim_toa_unc = mu_emp
    tim_label = toa_mode

    if tim_strat == 'best' and compare_results is not None:
        sm = compare_results.get('sorted_multi', [])
        cr_single = compare_results.get('single', {})
        cr_le = compare_results.get('leading_edge', {})
        cr_sl = compare_results.get('shapelet', {})
        sigma_emp_s = cr_single.get('mu_e', np.nan)
        aicc_single = cr_single.get('aicc', np.inf)
        if sm and sigma_emp_s > 0 and np.isfinite(sigma_emp_s):
            best_entry = sm[0]
            delta_conv = abs(best_entry['mu'] - cr_single['mu'])
            if delta_conv < 3 * sigma_emp_s:
                # Well-behaved: pick AICc-best among single and N-Gauss
                aicc_multi = best_entry.get('aicc', np.inf)
                if np.isfinite(aicc_single) and aicc_single <= aicc_multi:
                    tim_toa = cr_single['mu']
                    tim_toa_unc = cr_single.get('mu_e', np.nan)
                    tim_label = 'single (AICc best)'
                else:
                    tim_toa = best_entry['mu']
                    tim_toa_unc = best_entry.get('mu_emp', best_entry.get('mu_e', np.nan))
                    tim_label = best_entry.get('label', best_entry.get('mode', 'best'))
            else:
                # Overlapping regime: rank non-parametric methods.
                # (1) leading_edge — scattering-invariant, no model dependence.
                # (2) shapelet — denoised model, flexible, needs S/N > 5.
                # (3) single — last-resort parametric fallback.
                # center_of_mass is excluded: scattering tails and channel
                # smearing bias the centroid to later, non-physical times.
                le_ok = (cr_le and np.isfinite(cr_le.get('mu_f', np.nan))
                         and cr_le.get('snr', 0) > 3)
                sl_ok = (cr_sl and np.isfinite(cr_sl.get('mu_f', np.nan))
                         and cr_sl.get('snr', 0) > 5)
                if le_ok:
                    tim_toa = cr_le['mu_f']
                    tim_toa_unc = cr_le.get('mu_e', cr_le.get('mu_emp', np.nan))
                    tim_label = 'leading_edge (non-par, scattering-invariant)'
                elif sl_ok:
                    tim_toa = cr_sl['mu_f']
                    tim_toa_unc = cr_sl.get('mu_e', cr_sl.get('mu_emp', np.nan))
                    tim_label = 'shapelet (non-par, denoised model)'
                else:
                    tim_toa = cr_single['mu']
                    tim_toa_unc = cr_single.get('mu_e', np.nan)
                    tim_label = 'single (last-resort fallback)'
        elif cr_single:
            tim_toa = cr_single['mu']
            tim_toa_unc = cr_single.get('mu_e', np.nan)
            tim_label = 'single'
    elif tim_strat != 'current' and compare_results is not None:
        found = False
        for label, toa_s, unc_s in all_entries:
            if tim_strat == label or tim_strat in label:
                tim_toa = toa_s
                tim_toa_unc = unc_s
                tim_label = label
                found = True
                break
        if not found:
            print(f"  Warning: --tim_strategy '{tim_strat}' not found;"
                  f" using current TOA.")

    if fit_ok or compare_results is not None:
        toa_unc_us = tim_toa_unc * 1e6 if np.isfinite(tim_toa_unc) and tim_toa_unc > 0 else mu_emp * 1e6
        if np.isfinite(tim_toa_unc) and tim_toa_unc > 0:
            toa_unc_us = tim_toa_unc * 1e6
        elif np.isfinite(mu_e) and mu_e > 0:
            toa_unc_us = mu_e * 1e6
        toa_tim_mjd = mjd_start + tim_toa / sec_per_day
        tim_path = args.output + ".tim"
        with open(tim_path, 'w') as ft:
            ft.write(f"# {fname}\n")
            ft.write(f"# strategy: {tim_label}\n")
            ft.write("FORMAT 1\n")
            ft.write(f"{fname}  {toa_freq_mhz:.4f}  {toa_tim_mjd:.12f}  "
                     f"{toa_unc_us:.3f}  {args.telescope}\n")
        print(f"  -> {tim_path}  (strategy={tim_label},"
              f"  freq={toa_freq_mhz:.1f} MHz,"
              f"  unc={toa_unc_us:.1f} us,  tel={args.telescope})")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="toa_sp — Multi-strategy single-pulse TOA extraction "
                    "(Gaussian / EMG / Voigt / shapelet + non-parametric)"
    )
    parser.add_argument('-f', '--input_file', required=True,
                        help='Input FITS file')
    parser.add_argument('-dm', required=True, type=float,
                        help='DM in pc/cm^3')
    parser.add_argument('-s', '--start_time', type=float, default=1.1,
                        help='Pulse arrival at highest freq (s)')
    parser.add_argument('-w', '--width', type=float, default=0.06,
                        help='Half-width of pulse window (s)')
    parser.add_argument('-bs', '--bin_samp', type=int, default=8,
                        help='Time-binning factor')
    parser.add_argument('-bf', '--bin_chn', type=int, default=16,
                        help='Frequency-binning factor')
    parser.add_argument('--freq_lo', type=float, default=None,
                        help='Lower frequency bound for profile analysis (MHz). '
                             'Default: lowest channel.')
    parser.add_argument('--freq_hi', type=float, default=None,
                        help='Upper frequency bound for profile analysis (MHz). '
                             'Default: highest channel.')
    parser.add_argument('--toa_freq', default='top',
                        help='Reference frequency for TOA output (MHz). '
                             'Acceptable values: "top" (band maximum, default), '
                             '"center" (band centre), or a numeric value in MHz '
                             '(e.g. 1400.0).')
    parser.add_argument('--scale', type=int, default=1,
                        help='Apply per-channel scale normalisation (1=on, 0=off)')
    parser.add_argument('--skip_rfi', action='store_true',
                        help='Skip all RFI zapping')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Do NOT subtract running-median baseline before fit')
    parser.add_argument('--auto_res', action='store_true',
                        help='Auto-select optimal 1D time resolution by '
                             'maximising peak S/N (powers of 2)')
    parser.add_argument('--auto_res_max', type=int, default=256,
                        help='Max rebinning factor for --auto_res (power of 2, '
                             'default: 256)')
    parser.add_argument('--show', action='store_true',
                        help='Display plot interactively instead of saving to file')
    parser.add_argument('--ngauss', type=int, default=1,
                        help='Number of Gaussian components (1–N; '
                             'use --auto_n for AICc-based selection)')
    parser.add_argument('--profile_model', default='gaussian',
                        choices=['gaussian', 'emg', 'voigt', 'shapelet'],
                        help='Profile model: gaussian (default), '
                             'emg (exponentially-modified Gaussian for '
                             'scattering tails), voigt, or '
                             'shapelet (Hermite × Gaussian decomposition)')
    parser.add_argument('--toa_mode', default='single',
                        choices=['single', 'highest', 'weighted',
                                 'error_weighted', 'first_peak',
                                 'leading_edge', 'center_of_mass',
                                 'peak', 'shapelet'],
                        help='TOA selection: single (1-Gauss), '
                             'highest (brightest component), '
                             'weighted (amplitude-weighted avg), '
                             'error_weighted (inv-variance weighted), '
                             'first_peak (earliest S/N>3 component), '
                             'leading_edge (half-max on rising edge), '
                             'center_of_mass (non-parametric first moment), '
                             'peak (argmax with quadratic interpolation), '
                             'shapelet (Hermite-Gaussian decomposition)')
    parser.add_argument('--auto_n', action='store_true',
                        help='Auto-select N via AICc (overrides --ngauss)')
    parser.add_argument('--max_n', type=int, default=5,
                        help='Max Gaussian components for --auto_n or --compare')
    # --- Shapelet ---
    parser.add_argument('--shapelet_nmax', type=int, default=20,
                        help='Maximum shapelet order (scanned in steps of 2; '
                             'default: 20)')
    parser.add_argument('--shapelet_npoly', type=int, default=2,
                        help='Polynomial baseline degree for shapelet fit '
                             '(default: 2)')
    parser.add_argument('--shapelet_alpha', type=float, default=0.0,
                        help='Ridge regularisation strength for shapelet fit '
                             '(default: 0, no regularisation)')
    parser.add_argument('--sub_band_fit', action='store_true',
                        help='Fit TOA independently in frequency sub-bands '
                             '(robust for overlapping components)')
    parser.add_argument('--n_sub_bands', type=int, default=8,
                        help='Number of frequency sub-bands for --sub_band_fit')
    # --- MCMC ---
    parser.add_argument('--mcmc', action='store_true',
                        help='Run MCMC sampling for robust TOA uncertainty '
                             '(requires emcee)')
    parser.add_argument('--mcmc_nwalkers', type=int, default=32,
                        help='Number of MCMC walkers (default: 32)')
    parser.add_argument('--mcmc_nburn', type=int, default=500,
                        help='MCMC burn-in steps (default: 500)')
    parser.add_argument('--mcmc_nprod', type=int, default=1000,
                        help='MCMC production steps (default: 1000)')
    parser.add_argument('--mcmc_corner', action='store_true',
                        help='Generate corner plot of MCMC posteriors '
                             '(requires corner)')
    parser.add_argument('--compare', action='store_true',
                        help='Run all strategies (multi-N + all TOA modes) '
                             'and produce a comparison plot')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip plot generation entirely')
    parser.add_argument('--fig_format', default='png',
                        choices=['png', 'pdf'],
                        help='Output figure format: png (raster) or pdf '
                             '(vector; default: png)')
    parser.add_argument('--tim', action='store_true',
                        help='Write a TEMPO2-compatible .tim file')
    parser.add_argument('--tim_strategy', default='current',
                        help='Which strategy to use for .tim output: '
                             '"current" (use --toa_mode), '
                             '"best" (lowest AICc from --compare), '
                             'or a specific name: single, highest, '
                             'error_weighted, first_peak, leading_edge, '
                             'center_of_mass, shapelet')
    parser.add_argument('--telescope', default=None,
                        help='Telescope name for .tim output '
                             '(auto-detected from FITS header by default)')
    parser.add_argument('-o', '--output', default='timing_result',
                        help='Output basename (no extension)')
    args = parser.parse_args()
    model_type = args.profile_model        # 'gaussian', 'emg', or 'voigt'
    n_pc = get_params_per_component(model_type)

    # ------------------------------------------------------------------
    # 1. Open FITS, read headers
    # ------------------------------------------------------------------
    print("=" * 62)
    model_label = {'gaussian': 'GAUSSIAN', 'emg': 'EMG', 'voigt': 'VOIGT',
                   'shapelet': 'SHAPELET'}[model_type]
    print(f"  SINGLE-PULSE TIMING -- {model_label} FIT")
    print("=" * 62)
    print(f"  File : {args.input_file}")
    print(f"  DM   : {args.dm}  pc/cm^3")

    hdul = fits.open(args.input_file)

    # --- Extract observation start time for MJD conversion ---
    pri_hdr = hdul[0].header
    mjd_int = float(pri_hdr.get('STT_IMJD', 0))         # integer MJD day
    mjd_sec = float(pri_hdr.get('STT_SMJD', 0))         # seconds of day
    mjd_frac = float(pri_hdr.get('STT_OFFS', 0.0))      # fractional seconds
    mjd_start = mjd_int + (mjd_sec + mjd_frac) / 86400.0
    sec_per_day = 86400.0

    print(f"  MJD start : {mjd_start:.10f}"
          f"  (IMJD={mjd_int:.0f} + {mjd_sec + mjd_frac:.3f} s)")

    # Auto-detect telescope name from FITS header if not specified
    if args.telescope is None:
        args.telescope = (pri_hdr.get('TELESCOP')
                          or pri_hdr.get('TELESCOPE')
                          or 'unknown')
        print(f"  Telescope: {args.telescope}  (auto-detected)")

    tbdata = hdul['SUBINT'].data

    nbits = int(hdul['SUBINT'].header['NBITS'])
    nchan = int(hdul['SUBINT'].header['NCHAN'])
    nsblk = int(hdul['SUBINT'].header['NSBLK'])
    nsub  = int(hdul['SUBINT'].header['NAXIS2'])
    npol  = int(hdul['SUBINT'].header['NPOL'])
    tsamp = float(hdul['SUBINT'].header['TBIN'])          # seconds
    pol_order = hdul['SUBINT'].header.get('POL_TYPE', 'AABBCRCI')

    print(f"  nchan={nchan}  npol={npol}  nsblk={nsblk}  nsub={nsub}")
    print(f"  tsamp = {tsamp:.6e} s  ({tsamp * 1e3:.6f} ms)")
    print(f"  nbits = {nbits}  pol_type = {pol_order}")

    # Frequency array -- same for every subint row
    dat_freq = tbdata['DAT_FREQ']
    f0 = np.min(dat_freq[0])         # MHz
    f1 = np.max(dat_freq[0])         # MHz
    print(f"  Freq  : {f0:.2f} -- {f1:.2f} MHz  "
          f"(centre ~ {0.5 * (f0 + f1):.1f} MHz)")

    # ------------------------------------------------------------------
    # 2. Determine data window around the pulse
    # ------------------------------------------------------------------
    dm = args.dm
    w0 = args.width
    t0_sample = args.start_time

    # Total DM delay across the band — formula matches v008.py convention
    # (DM_CONST = 4.1488 with f in MHz and the *1e3 factor gives seconds)
    t_e = DM_CONST * dm * (f0 ** (-2) - f1 ** (-2)) * 1e3   # seconds
    delta = t_e / 2.0
    t_center = t0_sample + delta                             # centre of sweep

    # Grab window: max(DM-sweep x 1.5, 3 x pulse width)
    dt_sec = DM_CONST * dm * (f0 ** (-2) - f1 ** (-2)) * 1e3 * 1.5
    if dt_sec < 3 * w0:
        dt_sec = 3 * w0

    s0_raw = int((t_center - 0.5 * dt_sec) / tsamp)
    n_grab = int(dt_sec / tsamp)
    nsamp_total = nsub * nsblk
    total_dur = nsamp_total * tsamp
    s1 = min(s0_raw + n_grab, nsamp_total)
    s0 = max(s0_raw, 0)

    # Check for pulse truncation — both ends of the DM sweep
    pulse_truncated = False
    truncation_note = ""
    if t0_sample < 0:
        pulse_truncated = True
        truncation_note = (f"  *** WARNING: pulse DM sweep begins at "
                           f"{t0_sample:.4f} s (before file start) — "
                           f"pulse is TRUNCATED (file sees trailing half only) ***")
        print(truncation_note)
    if t0_sample + t_e > total_dur:
        pulse_truncated = True
        note2 = (f"  *** WARNING: pulse DM sweep extends to "
                 f"{t0_sample + t_e:.4f} s, but file ends at "
                 f"{total_dur:.4f} s — pulse is TRUNCATED ***")
        truncation_note = (truncation_note + "\n" + note2) if truncation_note else note2
        print(note2)
    if not pulse_truncated:
        if s1 >= nsamp_total or s0_raw < 0:
            pulse_truncated = True
            if s0_raw < 0:
                truncation_note = (f"  *** WARNING: grab window clipped at file start"
                                   f" — pulse may be incomplete ***")
            else:
                truncation_note = (f"  *** WARNING: grab window clipped at file end"
                                   f" ({total_dur:.4f} s) — pulse may be incomplete ***")
            print(truncation_note)

    print(f"  Grab  : samples [{s0}:{s1}]  "
          f"({(s1 - s0) * tsamp:.4f} s around t={t_center:.4f} s)")

    # ------------------------------------------------------------------
    # 3. Extract & unpack
    # ------------------------------------------------------------------
    data_raw = np.reshape(tbdata['DATA'],
                          (nsamp_total, npol, int(nchan / (8 / nbits))))
    hdul.close()

    data_grab = data_raw[s0:s1, :, :]
    nsamp_grab = data_grab.shape[0]

    print("  Unpacking ...", end=" ", flush=True)
    data_unpack = unpack_data_vectorized(data_grab, nbits, nchan,
                                         nsamp_grab, npol)
    data_out = np.moveaxis(data_unpack, 0, -1)            # (npol, nchan, nsamp)
    print(f"-> shape {data_out.shape}")

    # ------------------------------------------------------------------
    # 4. Per-channel scale normalisation (before pol combine)
    # ------------------------------------------------------------------
    if args.scale:
        print("  Scaling ...", end=" ", flush=True)
        data_out = newscale_data(data_out, npol, pol_order, nchan, nbits)
        print("done.")

    # ------------------------------------------------------------------
    # 5. Combine polarisations  AA + BB
    # ------------------------------------------------------------------
    if npol >= 2:
        data_out = 0.5 * (data_out[0, :, :] + data_out[1, :, :])
    else:
        data_out = np.squeeze(data_out)
    # data_out now (nchan, nsamp_grab)

    # ------------------------------------------------------------------
    # 6. RFI zapping (time domain) & copy
    # ------------------------------------------------------------------
    data_rfi = data_out.astype(np.float64, copy=True)
    if not args.skip_rfi:
        zap_rfi_time(data_rfi)

    data_dedisp = data_rfi.copy()     # one copy for final profile
    if not args.skip_rfi:
        data_rfi2 = data_rfi.copy()   # another for the RFI-flagged version
    else:
        data_rfi2 = data_rfi.copy()

    # ------------------------------------------------------------------
    # 7. Dedispersion
    # ------------------------------------------------------------------
    tsamp_ms = tsamp * 1e3            # ms
    f_low_ghz  = f0 / 1e3
    f_high_ghz = f1 / 1e3

    print(f"  Dedispersing at DM={dm} ...", end=" ", flush=True)
    dedisperse(data_dedisp, dm, f_low_ghz, f_high_ghz, nchan, tsamp_ms)
    dedisperse(data_rfi2,  dm, f_low_ghz, f_high_ghz, nchan, tsamp_ms)
    print("done.")

    # ------------------------------------------------------------------
    # 8. Frequency + time binning, freq-domain RFI
    # ------------------------------------------------------------------
    dat_binned = bin_2d(data_dedisp, args.bin_chn, args.bin_samp)
    dat_binned_rfi = bin_2d(data_rfi2, args.bin_chn, args.bin_samp)
    if not args.skip_rfi:
        zap_rfi_freq(dat_binned_rfi, args.bin_samp, args.bin_chn)

    nchan_b, nsamp_b = dat_binned_rfi.shape
    print(f"  Binned shape : {dat_binned_rfi.shape}  "
          f"(freq x{args.bin_chn}, time x{args.bin_samp})")

    # --- Frequency range selection for profile analysis ---
    f_lo_mhz = f0
    f_hi_mhz = f1
    ch_lo_b = 0
    ch_hi_b = nchan_b
    if args.freq_lo is not None or args.freq_hi is not None:
        f_lo_mhz = args.freq_lo if args.freq_lo is not None else f0
        f_hi_mhz = args.freq_hi if args.freq_hi is not None else f1
        # Validate
        if f_lo_mhz < f0:
            print(f"  Warning: freq_lo ({f_lo_mhz:.1f} MHz) below band min "
                  f"({f0:.1f} MHz); clamping.")
            f_lo_mhz = f0
        if f_hi_mhz > f1:
            print(f"  Warning: freq_hi ({f_hi_mhz:.1f} MHz) above band max "
                  f"({f1:.1f} MHz); clamping.")
            f_hi_mhz = f1
        if f_lo_mhz >= f_hi_mhz:
            print(f"  Error: freq_lo >= freq_hi ({f_lo_mhz} >= {f_hi_mhz} MHz)."
                  "  Using full band.")
            f_lo_mhz, f_hi_mhz = f0, f1
        ch_lo_b, ch_hi_b = freq_range_to_channel_indices(
            f_lo_mhz, f_hi_mhz, f0, f1, nchan, args.bin_chn)
        nchan_used = ch_hi_b - ch_lo_b
        print(f"  Freq range for profile: {f_lo_mhz:.1f}--{f_hi_mhz:.1f} MHz "
              f"({nchan_used}/{nchan_b} binned channels)")

    # Slice binned data for profile formation
    dat_binned_rfi_profile = dat_binned_rfi[ch_lo_b:ch_hi_b, :]

    # Resolve TOA reference frequency
    toa_freq_mhz = resolve_toa_freq(args.toa_freq, f_lo_mhz, f_hi_mhz)
    print(f"  TOA ref freq : {toa_freq_mhz:.2f} MHz"
          f"  (mode: {args.toa_freq})")

    # ------------------------------------------------------------------
    # 9. Form 1D profile (sum over frequency)
    # ------------------------------------------------------------------
    profile_raw = np.sum(dat_binned_rfi_profile, axis=0)    # (nsamp_b,)

    # Time axis for the profile (seconds since start of grabbed data)
    t_bin_width = tsamp * args.bin_samp                     # seconds
    t_start = t_center - 0.5 * dt_sec                       # abs time of sample 0
    t_prof = t_start + np.arange(nsamp_b) * t_bin_width

    # Subtract running-median baseline (standard for TOA measurement)
    if not args.no_baseline:
        # median filter window: 50 x pulse width in time bins, or 1/3 of data
        med_win = min(int(50 * w0 / t_bin_width), nsamp_b // 3)
        if med_win % 2 == 0:
            med_win += 1          # odd window for symmetry
        if med_win >= 5:
            from scipy.ndimage import median_filter
            baseline = median_filter(profile_raw, size=med_win)
            profile = profile_raw - baseline
            print(f"  Baseline  : running median (window = {med_win} bins "
                  f"= {med_win * t_bin_width * 1e3:.1f} ms) subtracted")
        else:
            profile = profile_raw - np.median(profile_raw)
            print(f"  Baseline  : global median subtracted")
    else:
        profile = profile_raw
        print(f"  Baseline  : not subtracted (raw profile)")

    # ------------------------------------------------------------------
    # 9b. Auto-select optimal 1D time resolution (optional)
    # ------------------------------------------------------------------
    auto_res_factor = 1
    if args.auto_res:
        res_opt = optimize_profile_resolution(
            profile, t_prof, t_bin_width, w0,
            max_factor=args.auto_res_max)
        auto_res_factor = res_opt['factor']
        profile = res_opt['profile']
        t_prof = res_opt['t_prof']
        t_bin_width = res_opt['t_bin_width']
        nsamp_b = len(profile)
        print(f"  Auto-res : factor = {auto_res_factor}"
              f"  |  S/N gain = {res_opt['snr_gain']:.2f}x"
              f"  |  new bin width = {t_bin_width * 1e3:.3f} ms"
              f"  |  profile bins = {nsamp_b}")
        # Print S/N scan table
        print(f"    {'Factor':<8s} {'S/N':<10s}")
        for f, snr in res_opt['snr_values']:
            marker = ' <-- BEST' if f == auto_res_factor else ''
            print(f"    {f:<8d} {snr:<10.2f}{marker}")

    # ------------------------------------------------------------------
    # 10. Locate pulse peak(s) & set up fit window
    # ------------------------------------------------------------------
    t_pulse_rel = 0.5 * dt_sec - delta
    s_pulse = int(t_pulse_rel / t_bin_width)
    search_half = int(w0 / t_bin_width)
    lo = max(0, s_pulse - search_half)
    hi = min(nsamp_b, s_pulse + search_half)
    fit_half = search_half

    n_gauss = args.ngauss
    toa_mode = args.toa_mode
    do_auto_n = args.auto_n
    max_n = args.max_n

    # --- Compute off-pulse RMS early, for peak-detection threshold ---
    # Use the edges of the grabbed data as initial off-pulse estimate
    edge_lo = max(0, lo - 3 * search_half)
    edge_hi = min(nsamp_b, hi + 3 * search_half)
    mask_init_off = np.ones(nsamp_b, dtype=bool)
    mask_init_off[edge_lo:edge_hi] = False
    rms_off_initial = (np.std(profile[mask_init_off])
                       if np.any(mask_init_off) else np.std(profile))

    # --- Detect all significant peaks using RMS-based threshold ---
    # Use 5 x off-pulse RMS as height threshold, 3 x RMS as prominence
    peak_height_thresh = max(5.0 * rms_off_initial, np.median(profile[lo:hi]))
    peak_prom_thresh = 3.0 * rms_off_initial
    peaks, props = find_peaks(profile[lo:hi],
                               height=peak_height_thresh,
                               prominence=peak_prom_thresh)
    if len(peaks) == 0:
        # Fallback: use argmax with lower threshold
        peaks, props = find_peaks(profile[lo:hi],
                                   height=3.0 * rms_off_initial,
                                   prominence=1.5 * rms_off_initial)
    if len(peaks) == 0:
        peaks = np.array([np.argmax(profile[lo:hi])])
        props = {'peak_heights': np.array([profile[lo + peaks[0]]])}

    # Sort by prominence descending
    prominences = props.get('prominences',
                            props.get('peak_heights', np.ones(len(peaks))))
    order = np.argsort(prominences)[::-1]
    all_peak_indices = lo + peaks[order]
    all_peak_times = t_prof[all_peak_indices]

    # Determine how many Gaussian components to fit
    if do_auto_n:
        # Auto-select: keep significant peaks, capped by max_n
        sig_peaks = all_peak_indices[:max_n]
        n_to_use = min(len(sig_peaks), max_n)
        print(f"  Auto-N: found {len(all_peak_indices)} peaks above"
              f" {peak_height_thresh:.0f} (5*RMS),"
              f" using top {n_to_use}")
    elif n_gauss >= 2:
        # Use specified N, capped by detected peaks
        n_to_use = min(n_gauss, len(all_peak_indices))
        if n_to_use < n_gauss:
            print(f"  Warning: only {n_to_use} peaks detected"
                  f" (asked for {n_gauss})")
    else:
        n_to_use = 1

    # Primary peak for window centering
    peak_idx = all_peak_indices[0]
    peak_indices = all_peak_indices[:n_to_use]

    print(f"  Profile peak  : bin {peak_idx}  "
          f"t = {t_prof[peak_idx]:.8f} s")
    if n_to_use > 1:
        print(f"  Detected peaks: bins {peak_indices}"
              f"  t = {t_prof[peak_indices]} s")

    # --- Fit window ---
    fit_lo = max(0, peak_idx - fit_half)
    fit_hi = min(nsamp_b, peak_idx + fit_half)

    # Widen to encompass all detected peaks
    if n_to_use > 1 and len(peak_indices) > 1:
        extra = max(0, int(0.5 * fit_half))
        fit_lo = max(0, min(fit_lo, peak_indices.min() - extra))
        fit_hi = min(nsamp_b, max(fit_hi, peak_indices.max() + extra))

    t_fit = t_prof[fit_lo:fit_hi]
    y_fit = profile[fit_lo:fit_hi]
    t_wide_lo, t_wide_hi = t_fit[0], t_fit[-1]   # save for plot alignment

    # ==================================================================
    # 11. Gaussian fit  (or Shapelet fit)
    # ==================================================================
    print()

    if model_type == 'shapelet':
        # --- Shapelet: Hermite × Gaussian decomposition ---
        n_shapelet = args.shapelet_nmax
        n_poly     = args.shapelet_npoly
        alpha_s    = args.shapelet_alpha

        print(f"  Fitting SHAPELET (nmax={n_shapelet}, npoly={n_poly}) ...",
              end=" ", flush=True)
        t_mid_s = 0.5 * (float(t_fit[0]) + float(t_fit[-1]))
        beta_guess_s = w0 / 2.0
        try:
            shapelet_fit = fit_shapelet_auto_n(
                t_fit, y_fit, t_prof[peak_idx], beta_guess_s,
                n_shapelet, n_poly, t_mid_s, alpha_s)
            n_comp = 1
            # Store shapelet state as a dict in popt
            popt = {
                'tau': shapelet_fit['tau'],
                'beta': shapelet_fit['beta'],
                'coeffs': shapelet_fit['coeffs'],
                'poly_coeffs': shapelet_fit['poly_coeffs'],
                't_mid': shapelet_fit['t_mid'],
                'n_shapelet': shapelet_fit['n_shapelet'],
                'n_poly': shapelet_fit['n_poly'],
            }
            pcov = None
            comps = None
            fit_ok = True
            best_aicc_s = shapelet_fit.get('aicc', np.nan)
            print(f"converged.  N_comp = {shapelet_fit['n_shapelet']}"
                  f"  |  AICc = {best_aicc_s:.1f}"
                  f"  |  tau = {shapelet_fit['tau']:.8f} s"
                  f"  |  beta = {shapelet_fit['beta']*1e3:.3f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            fit_ok = False
            popt = None
            comps = None

    elif do_auto_n and n_to_use > 1:
        # --- Auto-N: scan N = 1..n_to_use, select best via AICc ---
        print(f"  Scanning N = 1..{n_to_use} for AICc selection ...")
        scan_results = fit_multi_gaussians_scan(
            t_fit, y_fit, profile, all_peak_indices, peak_idx,
            w0, t_prof, fit_lo, fit_hi, n_to_use, t_bin_width,
            model_type=model_type)

        ok_results = [r for r in scan_results if r['ok']]
        if ok_results:
            best = min(ok_results, key=lambda r: r['aicc'])
            n_comp = best['n']
            popt = best['popt']
            pcov = best['pcov']
            comps = best['comps']
            fit_ok = True
            print(f"  Best N = {n_comp}  (AICc = {best['aicc']:.1f})")
            # Print AICc table
            for r in ok_results:
                marker = ' <-- BEST' if r['n'] == best['n'] else ''
                print(f"    N={r['n']}: AICc = {r['aicc']:.1f}"
                      f"  RSS = {r['rss']:.1f}{marker}")
        else:
            # All failed — fall back to single Gaussian
            print("  All multi-Gaussian fits failed; falling back to single.")
            n_comp = 1
            n_to_use = 1
            do_auto_n = False
    elif n_to_use == 1:
        # ---------- Single-component: two-pass logic ----------
        n_comp = 1
        p0_amp    = np.max(y_fit) - np.median(y_fit)
        p0_mean   = t_prof[peak_idx]
        p0_sigma  = w0 / 3.0
        p0_offset = np.median(y_fit)

        if model_type == 'gaussian':
            model_func = gaussian
            p0 = [p0_amp, p0_mean, p0_sigma, p0_offset]
            lb = [-np.inf, t_fit[0], 1e-9, -np.inf]
            ub = [np.inf, t_fit[-1], np.inf, np.inf]
        elif model_type == 'emg':
            model_func = emg
            p0_tau = w0 / 6.0
            p0 = [p0_amp, p0_mean, p0_sigma, p0_tau, p0_offset]
            tau_max = t_fit[-1] - t_fit[0]
            lb = [-np.inf, t_fit[0], 1e-9, 1e-9, -np.inf]
            ub = [np.inf, t_fit[-1], np.inf, tau_max, np.inf]
        elif model_type == 'voigt':
            model_func = voigt
            p0_gamma = w0 / 6.0
            p0 = [p0_amp, p0_mean, p0_sigma, p0_gamma, p0_offset]
            gamma_max = t_fit[-1] - t_fit[0]
            lb = [-np.inf, t_fit[0], 1e-9, 0.0, -np.inf]
            ub = [np.inf, t_fit[-1], np.inf, gamma_max, np.inf]

        model_label_fit = model_type.upper()
        print(f"  Fitting {model_label_fit} (wide window) ...",
              end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                if model_type == 'gaussian':
                    # Add bounds to prevent runaway convergence
                    t_span = t_fit[-1] - t_fit[0]
                    lb = [-np.inf, t_fit[0], 1e-9, -np.inf]
                    ub = [np.inf, t_fit[-1], 0.5 * t_span, np.inf]
                    popt1, _ = curve_fit(model_func, t_fit, y_fit,
                                         p0=p0, bounds=(lb, ub),
                                         maxfev=20000)
                else:
                    popt1, _ = curve_fit(model_func, t_fit, y_fit,
                                         p0=p0, bounds=(lb, ub),
                                         maxfev=20000)
                fit_ok = True
                print("converged.")
            except (RuntimeError, RuntimeWarning) as e:
                print(f"FAILED: {e}")
                fit_ok = False

        if fit_ok:
            # Validate pass-1: sigma must be positive and FWHM reasonable
            sigma_pass1 = popt1[2]
            t_span = t_fit[-1] - t_fit[0]
            if sigma_pass1 > 0 and sigma_pass1 < 0.5 * t_span:
                fwhm_pass1 = 2.35482 * sigma_pass1
            else:
                # Pass-1 gave unphysical sigma — fall back to w0-based window
                fwhm_pass1 = 2.35482 * (w0 / 3.0)
            tight_half = max(8, int(3 * fwhm_pass1 / t_bin_width))
            tight_lo = max(0, peak_idx - tight_half)
            tight_hi = min(nsamp_b, peak_idx + tight_half)
            t_tight = t_prof[tight_lo:tight_hi]
            y_tight = profile[tight_lo:tight_hi]

            p0_amp2    = np.max(y_tight) - np.median(y_tight)
            p0_mean2   = t_prof[peak_idx]
            p0_sigma2  = fwhm_pass1 / 2.35482
            p0_offset2 = np.median(y_tight)

            if model_type == 'gaussian':
                p0_2 = [p0_amp2, p0_mean2, p0_sigma2, p0_offset2]
            elif model_type == 'emg':
                p0_tau2 = abs(popt1[3]) if len(popt1) > 3 else w0 / 6.0
                p0_2 = [p0_amp2, p0_mean2, p0_sigma2, p0_tau2, p0_offset2]
            elif model_type == 'voigt':
                p0_gamma2 = abs(popt1[3]) if len(popt1) > 3 else w0 / 6.0
                p0_2 = [p0_amp2, p0_mean2, p0_sigma2, p0_gamma2, p0_offset2]

            print(f"  Fitting {model_label_fit} (tight window, {len(t_tight)} pts) ...",
                  end=" ", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                try:
                    if model_type == 'gaussian':
                        popt, pcov = curve_fit(
                            model_func, t_tight, y_tight,
                            p0=p0_2, maxfev=20000)
                    else:
                        t_window = t_tight[-1] - t_tight[0]
                        lb2 = [-np.inf, t_tight[0], 1e-9, 1e-9, -np.inf]
                        ub2 = [np.inf, t_tight[-1], np.inf, t_window, np.inf]
                        if model_type == 'voigt':
                            lb2[3] = 0.0  # gamma >= 0
                        popt, pcov = curve_fit(
                            model_func, t_tight, y_tight,
                            p0=p0_2, bounds=(lb2, ub2), maxfev=20000)
                    t_fit  = t_tight
                    y_fit  = y_tight
                    fit_lo = tight_lo
                    fit_hi = tight_hi
                    print("converged.")
                except (RuntimeError, RuntimeWarning) as e:
                    print(f"FAILED, using pass-1 result: {e}")
                    popt = popt1
                    pcov = np.diag([np.nan] * (n_pc + 1))

        if not fit_ok:
            popt = None

        comps = None

    else:
        # ---------- Multi-component with specified N ----------
        n_comp = n_to_use
        r = fit_multi_gaussian_run(t_fit, y_fit, profile,
                                   peak_indices, peak_indices[0],
                                   w0, t_prof, fit_lo, fit_hi, n_comp,
                                   model_type=model_type)
        fit_ok = r['ok']
        if fit_ok:
            popt = r['popt']
            pcov = r['pcov']
            comps = r['comps']
        else:
            popt = None
            comps = None

    # ==================================================================
    # 12. TOA selection from fit results
    # ==================================================================
    # --- Compute off-pulse RMS (masking the fit region) ---
    mask_off = np.ones(nsamp_b, dtype=bool)
    mask_off[fit_lo:fit_hi] = False
    rms_off = np.std(profile[mask_off]) if np.any(mask_off) else np.std(profile)
    if not np.isfinite(rms_off) or rms_off == 0:
        rms_off = np.std(profile) or 1.0  # fallback to full-profile std

    if fit_ok:
        if model_type == 'shapelet':
            # --- Shapelet: unpack from popt dict ---
            tau_s = popt['tau']
            beta_s = popt['beta']
            t_mid_s = popt['t_mid']

            # Evaluate model and separate pulse from polynomial baseline
            model_eval_s = shapelet_model(
                t_fit, tau_s, beta_s, popt['coeffs'], popt['poly_coeffs'],
                t_mid_s)
            poly_bl = np.zeros_like(t_fit)
            dt_s = t_fit - t_mid_s
            for j, b in enumerate(popt['poly_coeffs']):
                poly_bl += b * dt_s ** j
            pulse_s = model_eval_s - poly_bl

            # TOA = peak of the reconstructed pulse (sub-bin via quadratic interp)
            idx_pk = np.argmax(pulse_s)
            if 1 <= idx_pk < len(pulse_s) - 1:
                y0, y1, y2 = (pulse_s[idx_pk - 1],
                              pulse_s[idx_pk],
                              pulse_s[idx_pk + 1])
                denom = y0 - 2.0 * y1 + y2
                if abs(denom) > 1e-15:
                    delta = 0.5 * (y0 - y2) / denom
                    mu_f = t_fit[idx_pk] + delta * (t_fit[1] - t_fit[0])
                else:
                    mu_f = t_fit[idx_pk]
            else:
                mu_f = t_fit[idx_pk]

            peak_model_s = float(pulse_s[idx_pk])
            snr_s = peak_model_s / rms_off if (rms_off and rms_off > 0) else np.inf
            fwhm_f = 2.0 * beta_s
            sig_f = beta_s
            amp_f = peak_model_s
            off_f = float(popt['poly_coeffs'][0]) if len(popt['poly_coeffs']) > 0 else 0.0
            extra_f = np.nan
            amp_e = np.nan
            sig_e = np.nan
            off_e = np.nan
            extra_e = np.nan
            snr = snr_s

            # Uncertainty: half-max rise time / SNR on denoised model
            half_max_s = 0.5 * peak_model_s
            mu_e = np.nan
            for i in range(idx_pk, 0, -1):
                if pulse_s[i] <= half_max_s <= pulse_s[i + 1]:
                    frac_s = (half_max_s - pulse_s[i]) / max(
                        pulse_s[i + 1] - pulse_s[i], 1e-30)
                    t_hm_s = t_fit[i] + frac_s * (t_fit[1] - t_fit[0])
                    rise_s = t_fit[idx_pk] - t_hm_s
                    mu_e = max(rise_s / snr_s if snr_s > 0 else np.nan,
                               t_bin_width / np.sqrt(12.0))
                    break
            mu_e_scaled = mu_e

            n_eff = max(1, fwhm_f / t_bin_width)
            mu_emp = abs(beta_s) / (snr * np.sqrt(n_eff)) if snr > 0 and np.isfinite(snr) else np.nan

            resid = y_fit - model_eval_s
            n_param_s = (popt['n_shapelet'] + 1) + (popt['n_poly'] + 1) + 2  # +2 for (τ,β)
            dof = max(1, len(y_fit) - n_param_s)
            chi2_red = np.sum(resid ** 2) / dof
            rss_s = np.sum(resid ** 2)
            aicc_s = compute_aicc(rss_s, n_param_s, len(y_fit))

        elif n_comp == 1 and not (toa_mode in ('leading_edge', 'center_of_mass', 'peak')):
            # Single-component: unpack from popt
            amp_f = popt[0]
            mu_f = popt[1]
            sig_f = popt[2]
            off_f = popt[-1]
            extra_f = popt[3] if n_pc > 3 else np.nan  # tau or gamma
            try:
                perr = np.sqrt(np.diag(pcov))
                amp_e = perr[0]
                mu_e = perr[1]
                sig_e = perr[2]
                off_e = perr[-1]
                extra_e = perr[3] if n_pc > 3 else np.nan
            except (ValueError, np.linalg.LinAlgError):
                amp_e = mu_e = sig_e = off_e = extra_e = np.nan
            fwhm_f = 2.35482 * abs(sig_f)
            # For non-Gaussian models, compute peak S/N from the actual model
            if model_type == 'gaussian':
                snr = amp_f / rms_off if (rms_off and rms_off > 0) else np.inf
            else:
                model_peak = np.max(eval_profile_model(t_fit, popt, model_type))
                snr = model_peak / rms_off if (rms_off and rms_off > 0) else np.inf
            resid = y_fit - eval_profile_model(t_fit, popt, model_type)
            dof = len(y_fit) - (n_pc + 1)
            chi2_red = np.sum(resid ** 2) / dof if dof > 0 else np.nan
            n_eff = max(1, fwhm_f / t_bin_width)
            mu_emp = abs(sig_f) / (snr * np.sqrt(n_eff)) if snr > 0 else np.inf
            if chi2_red > 1 and not np.isnan(mu_e):
                mu_e_scaled = mu_e * np.sqrt(chi2_red)
            else:
                mu_e_scaled = mu_e

        else:
            # Multi-component or non-parametric: select TOA per --toa_mode
            off_f = popt[-1] if n_comp > 1 else popt[n_pc]
            off_e = np.nan

            if toa_mode == 'highest':
                toa_info = extract_toa_highest(comps, rms_off, t_bin_width)

            elif toa_mode == 'weighted':
                toa_info = extract_toa_weighted(comps, popt, pcov,
                                                rms_off, t_bin_width, n_comp)

            elif toa_mode == 'error_weighted':
                toa_info = extract_toa_error_weighted(comps, popt, pcov,
                                                      rms_off, t_bin_width, n_comp)

            elif toa_mode == 'first_peak':
                toa_info = extract_toa_first_peak(comps, rms_off, t_bin_width)
                if toa_info is None:
                    toa_info = {'mu_f': np.nan, 'mu_e': np.nan,
                                'label': 'first_peak', 'snr': np.nan}

            elif toa_mode == 'leading_edge':
                toa_info = toa_leading_edge(t_prof, profile,
                                            fit_lo, fit_hi, rms_off)

            elif toa_mode == 'center_of_mass':
                toa_info = toa_center_of_mass(t_prof, profile,
                                              fit_lo, fit_hi, rms_off)

            elif toa_mode == 'peak':
                toa_info = toa_peak(t_prof, profile,
                                    fit_lo, fit_hi, rms_off, t_bin_width)

            elif toa_mode == 'shapelet':
                toa_info = toa_shapelet(t_prof, profile, fit_lo, fit_hi,
                                       rms_off, t_bin_width,
                                       n_max=args.shapelet_nmax,
                                       n_poly=args.shapelet_npoly,
                                       alpha=args.shapelet_alpha)

            else:  # 'single' — fallback to highest
                toa_info = extract_toa_highest(comps, rms_off, t_bin_width)

            # Unpack toa_info into local variables
            mu_f = toa_info['mu_f']
            mu_e = toa_info['mu_e']
            amp_f = toa_info.get('amp_f', np.nan)
            sig_f = toa_info.get('sig_f', np.nan)
            sig_e = toa_info.get('sig_e', np.nan)
            fwhm_f = toa_info.get('fwhm_f', 2.35482 * abs(sig_f))
            snr = toa_info.get('snr', np.nan)
            mu_emp = toa_info.get('mu_emp', np.nan)

            # Residuals for the full parametric model (skip for non-parametric)
            if toa_mode in ('leading_edge', 'center_of_mass'):
                # Non-parametric: compute chi^2 from the fitted model
                resid = y_fit - eval_profile_model(t_fit, popt, model_type)
                dof = len(y_fit) - (n_pc * n_comp + 1)
                chi2_red = np.sum(resid ** 2) / dof if dof > 0 else np.nan
                n_eff = max(1, fwhm_f / t_bin_width) if np.isfinite(fwhm_f) else 1
            else:
                resid = y_fit - eval_profile_model(t_fit, popt, model_type)
                dof = len(y_fit) - len(popt)
                chi2_red = np.sum(resid ** 2) / dof if dof > 0 else np.nan
                n_eff = max(1, fwhm_f / t_bin_width)

            if chi2_red > 1 and not np.isnan(mu_e):
                mu_e_scaled = mu_e * np.sqrt(chi2_red)
            else:
                mu_e_scaled = mu_e

            amp_e = toa_info.get('mu_e', np.nan)  # for reporting

    else:
        # Fit failed — fallback
        mu_f  = t_prof[peak_idx]
        mu_e  = t_bin_width
        mu_emp = t_bin_width
        mu_e_scaled = mu_e
        sig_f = w0
        sig_e = np.nan
        amp_f = np.nan
        amp_e = np.nan
        off_f = np.nan
        fwhm_f = 2.35482 * sig_f
        chi2_red = np.nan
        snr = np.nan
        rms_off = np.nan
        dof = 0
        comps = None

    # ==================================================================
    # DM frequency correction: dedispersion aligns to f1 (full-band top),
    # but the user may request TOA at a different reference frequency.
    # Shift TOA from f1 to toa_freq_mhz.
    # ==================================================================
    if abs(toa_freq_mhz - f1) > 0.01:
        f_ref_ghz = toa_freq_mhz / 1e3
        f_top_ghz = f1 / 1e3
        dm_offset = DM_CONST * dm * (f_ref_ghz**(-2) - f_top_ghz**(-2)) * 1e-3
        mu_f += dm_offset

    # ==================================================================
    # 12b. MCMC uncertainty estimation (optional)
    # ==================================================================
    mcmc_result = None
    mcmc_n_comp = n_comp
    if args.mcmc and fit_ok:
        try:
            import emcee  # noqa: F811
        except ImportError:
            print("\n  WARNING: emcee not installed."
                  "  Install with: pip install emcee")
            print("  Skipping MCMC.")
        else:
            if n_comp == 1 and toa_mode not in ('leading_edge', 'center_of_mass',
                                                    'shapelet'):
                mcmc_p0, mcmc_pcov = popt, pcov
                mcmc_n_comp = 1
            elif comps is not None:
                mcmc_p0, mcmc_pcov = popt, pcov
                mcmc_n_comp = n_comp
            else:
                mcmc_p0 = None

            if mcmc_p0 is not None and np.any(np.isfinite(mcmc_p0)):
                print(f"\n  MCMC sampling ({args.mcmc_nwalkers} walkers, "
                      f"{args.mcmc_nburn} burn-in, "
                      f"{args.mcmc_nprod} production) ...", flush=True)
                mcmc_result = run_mcmc(
                    t_fit, y_fit, rms_off, mcmc_p0, mcmc_pcov,
                    mcmc_n_comp,
                    n_walkers=args.mcmc_nwalkers,
                    n_burn=args.mcmc_nburn,
                    n_prod=args.mcmc_nprod,
                    progress=True, model_type=model_type)
                if mcmc_result['ok']:
                    mcmc_toa = mcmc_toa_uncertainty(
                        mcmc_result['flat_samples'], mcmc_n_comp,
                        model_type=model_type)
                    print(f"  MCMC TOA uncertainty: "
                          f"{mcmc_toa['mu_mcmc_std'] * 1e6:.1f} us "
                          f"(formal: {mu_e * 1e6:.1f} us)")
                    print(f"  Gelman-Rubin R-hat = "
                          f"{mcmc_result['gr_stat']:.4f}")
                    if mcmc_result['gr_stat'] > 1.1:
                        print("  WARNING: R-hat > 1.1,"
                              " chains may not be converged."
                              "  Increase --mcmc_nburn / --mcmc_nprod.")
                else:
                    print("  MCMC failed to converge")
                    mcmc_result = None

    # ==================================================================
    # 13. Report results
    # ==================================================================
    print("\n" + "-" * 62)
    if fit_ok:
        model_label_rpt = {'gaussian': 'GAUSSIAN', 'emg': 'EMG', 'voigt': 'VOIGT',
                           'shapelet': 'SHAPELET'}[model_type]
        print(f"  {model_label_rpt} FIT RESULTS")
        print("-" * 62)

        if model_type == 'shapelet':
            print(f"  TOA (tau)             {mu_f:.8f}  s")
            print(f"  TOA 1-sigma           {mu_e:.8f}  s"
                  f"  (profile likelihood)")
            if not np.isnan(mu_emp):
                print(f"  Empirical 1-sigma     {mu_emp:.8f}  s"
                      f"  (beta/(SNR*sqrt(N)))")
            print(f"  Width (beta)          {sig_f:.8f}  s"
                  f"  ({sig_f*1e3:.3f}  ms)")
            print(f"  FWHM  (approx 2*beta) {fwhm_f*1e3:.3f}  ms")
            n_sl = popt['n_shapelet'] if isinstance(popt, dict) else '?'
            print(f"  Shapelet components   {n_sl}")
            try:
                _aicc_s = compute_aicc(np.sum((y_fit - shapelet_model(
                    t_fit, popt['tau'], popt['beta'], popt['coeffs'],
                    popt['poly_coeffs'], popt['t_mid']))**2),
                    (popt['n_shapelet']+1)+(popt['n_poly']+1)+2, len(y_fit))
                print(f"  AICc                  {_aicc_s:.1f}")
            except Exception:
                pass
            print(f"  S/N                   {snr:.2f}")
            print(f"  chisq_red (dof={dof})      {chi2_red:.2f}")

        elif n_comp == 1 and toa_mode not in ('leading_edge', 'center_of_mass'):
            print(f"  TOA (peak)            {mu_f:.8f}  s")
            print(f"  Formal 1-sigma        {mu_e:.8f}  s")
            if not np.isnan(mu_e_scaled):
                print(f"  Rescaled 1-sigma      {mu_e_scaled:.8f}  s"
                      f"  (x sqrt(chi2_red))")
            print(f"  Empirical 1-sigma     {mu_emp:.8f}  s"
                  f"  (sigma/(SNR*sqrt(N)))")
            if mcmc_result is not None and mcmc_result['ok']:
                mcmc_toa_rpt = mcmc_toa_uncertainty(
                    mcmc_result['flat_samples'], mcmc_n_comp,
                    model_type=model_type)
                print(f"  MCMC 1-sigma          "
                      f"{mcmc_toa_rpt['mu_mcmc_std']:.8f}  s")
                print(f"  MCMC 16-84 pctile     "
                      f"[{mcmc_toa_rpt['mu_mcmc_16']:.8f}, "
                      f"{mcmc_toa_rpt['mu_mcmc_84']:.8f}]  s")
                ratio_mcmc = (mcmc_toa_rpt['mu_mcmc_std'] / mu_e
                              if mu_e > 0 else np.inf)
                print(f"  MCMC/Formal ratio     {ratio_mcmc:.2f}")
            print(f"  {'Gaussian' if model_type == 'gaussian' else model_type.upper()} sigma"
                  f"        {sig_f:.8f}  +/- {sig_e:.8f}  s")
            if model_type == 'emg' and np.isfinite(extra_f):
                print(f"  Scattering tau        {extra_f:.8f}  +/- {extra_e:.8f}  s")
            elif model_type == 'voigt' and np.isfinite(extra_f):
                print(f"  Lorentzian gamma      {extra_f:.8f}  +/- {extra_e:.8f}  s")
            print(f"  FWHM                  {fwhm_f:.8f}  s")
            print(f"  Peak amplitude        {amp_f:.2f}  +/- {amp_e:.2f}")
            print(f"  Off-pulse RMS         {rms_off:.2f}")
            print(f"  S/N (peak / off-RMS)  {snr:.2f}")
            print(f"  chisq_red (dof={dof}, tight)  {chi2_red:.2f}")
            if chi2_red > 5:
                print(f"  NOTE: chisq_red > 5 -- pulse shape may be"
                      f" non-{model_type.capitalize()}.")
        else:
            # --- Per-component report ---
            for i, c in enumerate(comps):
                fwhm_i = 2.35482 * abs(c['sig'])
                snr_i = abs(c['amp']) / rms_off if (rms_off and rms_off > 0) else np.inf
                marker = ""
                if toa_mode == 'highest' and i == np.argmax(
                        [abs(cc['amp']) for cc in comps]):
                    marker = "  <-- SELECTED (highest)"
                print(f"  --- Component {i+1}{marker} ---")
                print(f"    TOA       {c['mu']:.8f}  +/- {c['mu_err']:.8f}  s")
                print(f"    sigma     {c['sig']:.8f}  +/- {c['sig_err']:.8f}  s")
                if model_type == 'emg' and 'tau' in c:
                    print(f"    tau       {c['tau']:.8f}  +/- {c['tau_err']:.8f}  s")
                elif model_type == 'voigt' and 'gamma' in c:
                    print(f"    gamma     {c['gamma']:.8f}  +/- {c['gamma_err']:.8f}  s")
                print(f"    FWHM      {fwhm_i:.8f}  s")
                print(f"    Amplitude {c['amp']:.2f}  +/- {c['amp_err']:.2f}")
                print(f"    S/N       {snr_i:.2f}")
                if i < len(comps) - 1:
                    sep = abs(comps[i+1]['mu'] - c['mu'])
                    print(f"    Separation to next: {sep:.8f}  s"
                          f"  ({sep * 1e3:.3f}  ms)")

            print(f"  --- Combined ---")
            print(f"  TOA mode              {toa_mode}")
            print(f"  Selected TOA          {mu_f:.8f}  s")
            print(f"  TOA 1-sigma           {mu_e:.8f}  s")
            if toa_mode == 'weighted':
                print(f"  Weighted-avg TOA      {mu_f:.8f}  s"
                      f"  (amplitude weights)")
            print(f"  chisq_red (dof={dof})      {chi2_red:.2f}")
            if chi2_red > 5:
                print("  NOTE: chisq_red > 5 -- model may not fully"
                      " capture profile structure.")
            if mcmc_result is not None and mcmc_result['ok']:
                mcmc_toa_rpt = mcmc_toa_uncertainty(
                    mcmc_result['flat_samples'], mcmc_n_comp,
                    model_type=model_type)
                print(f"  MCMC 1-sigma          "
                      f"{mcmc_toa_rpt['mu_mcmc_std']:.8f}  s")
                ratio_mcmc = (mcmc_toa_rpt['mu_mcmc_std'] / mu_e
                              if mu_e > 0 else np.inf)
                print(f"  MCMC/Formal ratio     {ratio_mcmc:.2f}")
    else:
        print("  Gaussian fit failed -- reporting argmax position.")
        print(f"  Peak time (argmax)   {mu_f:.8f}  s")
    print("-" * 62)

    # ==================================================================
    # 14. Convert TOA to MJD
    # ==================================================================
    toa_mjd = mjd_start + mu_f / sec_per_day
    toa_mjd_err_sec = mu_emp if np.isfinite(mu_emp) else mu_e
    toa_mjd_err_day = toa_mjd_err_sec / sec_per_day

    print(f"\n  TOA (MJD)             {toa_mjd:.10f}")
    if n_comp >= 2 and fit_ok:
        print(f"  (mode = {toa_mode})")
    print(f"  TOA uncertainty       {toa_mjd_err_day:.12f}  days"
          f"  ({toa_mjd_err_sec * 1e6:.2f}  us)")
    if n_comp >= 2 and fit_ok and comps is not None:
        print(f"  Per-component MJD:")
        for i, c in enumerate(comps):
            mjd_i = mjd_start + c['mu'] / sec_per_day
            sep_ms = (c['mu'] - mu_f) * 1e3
            print(f"    C{i+1}: {mjd_i:.10f}  (delta = {sep_ms:+.4f} ms)")

    # ------------------------------------------------------------------
    # 15. Save TOA to text file
    # ------------------------------------------------------------------
    txt_path = args.output + "_toa.txt"
    with open(txt_path, 'w') as f:
        f.write("# Single-pulse TOA -- Gaussian fit\n")
        f.write(f"# File       : {args.input_file}\n")
        f.write(f"# DM         : {dm:.2f}  pc/cm^3\n")
        f.write(f"# tsamp      : {tsamp:.6e}  s\n")
        f.write(f"# bin_samp   : {args.bin_samp}\n")
        f.write(f"# bin_chn    : {args.bin_chn}\n")
        f.write(f"# Freq full  : {f0:.2f} -- {f1:.2f}  MHz\n")
        f.write(f"# Freq used  : {f_lo_mhz:.2f} -- {f_hi_mhz:.2f}  MHz"
                f"  ({ch_hi_b - ch_lo_b} binned ch)\n")
        f.write(f"# TOA ref freq: {toa_freq_mhz:.4f}  MHz"
                f"  (mode: {args.toa_freq})\n")
        if auto_res_factor > 1:
            f.write(f"# auto_res   : factor = {auto_res_factor}"
                    f"  |  bin_width = {t_bin_width * 1e3:.6f} ms\n")
        f.write(f"# t_start    : {t_start:.8f}  s\n")
        f.write(f"# MJD_start  : {mjd_start:.10f}\n")
        f.write(f"# MJD_obs    : {mjd_start + t0_sample / sec_per_day:.10f}"
                f"  (at ref t0={t0_sample})\n")
        f.write(f"# ngauss      : {n_comp}\n")
        f.write(f"# toa_mode    : {toa_mode}\n")
        f.write(f"# n_comp      : {n_comp}\n")
        f.write(f"# model       : {model_type}\n")
        if model_type == 'shapelet' and fit_ok and isinstance(popt, dict):
            f.write(f"# shapelet_n  : {popt['n_shapelet']}\n")
            f.write(f"# shapelet_beta: {popt['beta']:.8f}  s\n")
            f.write(f"# shapelet_npoly: {popt['n_poly']}\n")

        if fit_ok and comps is not None and n_comp >= 2:
            for i, c in enumerate(comps):
                mjd_i = mjd_start + c['mu'] / sec_per_day
                f.write(f"#\n")
                f.write(f"# Component {i+1}\n")
                f.write(f"C{i+1}_TOA_MJD      = {mjd_i:.10f}\n")
                f.write(f"C{i+1}_TOA_s        = {c['mu']:.8f}  s\n")
                f.write(f"C{i+1}_TOA_uncert   = {c['mu_err']:.8f}  s\n")
                f.write(f"C{i+1}_sigma        = {c['sig']:.8f}  s\n")
                f.write(f"C{i+1}_sigma_uncert = {c['sig_err']:.8f}  s\n")
                if model_type == 'emg' and 'tau' in c:
                    f.write(f"C{i+1}_tau          = {c['tau']:.8f}  s\n")
                    f.write(f"C{i+1}_tau_uncert   = {c['tau_err']:.8f}  s\n")
                elif model_type == 'voigt' and 'gamma' in c:
                    f.write(f"C{i+1}_gamma        = {c['gamma']:.8f}  s\n")
                    f.write(f"C{i+1}_gamma_uncert = {c['gamma_err']:.8f}  s\n")
                f.write(f"C{i+1}_amplitude    = {c['amp']:.4f}\n")

        f.write(f"#\n")
        if fit_ok:
            f.write(f"TOA_s            = {mu_f:.8f}  s\n")
            f.write(f"TOA_MJD          = {toa_mjd:.10f}\n")
            f.write(f"TOA_uncert_formal= {mu_e:.8f}  s\n")
            f.write(f"TOA_uncert_emp   = {mu_emp:.8f}  s\n")
            f.write(f"TOA_uncert_MJD   = {toa_mjd_err_day:.12f}  days\n")
            f.write(f"sigma            = {sig_f:.8f}  s\n")
            f.write(f"sigma_uncert     = {sig_e:.8f}  s\n")
            if model_type == 'emg' and np.isfinite(extra_f):
                f.write(f"tau              = {extra_f:.8f}  s\n")
                f.write(f"tau_uncert       = {extra_e:.8f}  s\n")
            elif model_type == 'voigt' and np.isfinite(extra_f):
                f.write(f"gamma            = {extra_f:.8f}  s\n")
                f.write(f"gamma_uncert     = {extra_e:.8f}  s\n")
            elif model_type == 'shapelet' and isinstance(popt, dict):
                f.write(f"beta             = {popt['beta']:.8f}  s\n")
                f.write(f"shapelet_n       = {popt['n_shapelet']}\n")
            f.write(f"FWHM             = {fwhm_f:.8f}  s\n")
            f.write(f"amplitude        = {amp_f:.4f}\n")
            f.write(f"off_pulse_rms    = {rms_off:.4f}\n")
            f.write(f"chi2_red         = {chi2_red:.6f}\n")
            f.write(f"dof              = {dof}\n")
            f.write(f"SNR              = {snr:.2f}\n")
            if mcmc_result is not None and mcmc_result['ok']:
                mcmc_toa_out = mcmc_toa_uncertainty(
                    mcmc_result['flat_samples'], mcmc_n_comp,
                    model_type=model_type)
                f.write(f"TOA_uncert_mcmc   ="
                        f" {mcmc_toa_out['mu_mcmc_std']:.8f}  s\n")
                f.write(f"TOA_mcmc_16pct    ="
                        f" {mcmc_toa_out['mu_mcmc_16']:.8f}  s\n")
                f.write(f"TOA_mcmc_84pct    ="
                        f" {mcmc_toa_out['mu_mcmc_84']:.8f}  s\n")
                f.write(f"MCMC_Rhat         ="
                        f" {mcmc_result['gr_stat']:.4f}\n")
                f.write(f"MCMC_accept_frac  ="
                        f" {mcmc_result['acceptance_fraction']:.4f}\n")
        else:
            f.write("FIT_FAILED\n")
            f.write(f"TOA_argmax     = {mu_f:.8f}  s\n")
    print(f"  -> {txt_path}")

    compare_results = None   # populated in --compare block below

    # --- Corner plot (optional) ---
    if args.mcmc_corner and mcmc_result is not None and mcmc_result['ok']:
        labels = []
        for i in range(mcmc_n_comp):
            labels.extend([f'A{i+1}', f'mu{i+1}', f'sig{i+1}'])
        labels.append('offset')
        corner_path = args.output + '_corner.' + args.fig_format
        plot_corner(mcmc_result['flat_samples'], labels, corner_path)

    # ==================================================================
    # 15b. Sub-band TOA fitting (optional)
    # ==================================================================
    sub_band_info = None
    if args.sub_band_fit:
        print(f"\n  Sub-band TOA fitting ({args.n_sub_bands} bands) ...",
              end=" ", flush=True)
        # Use the frequency-selected data and adjusted frequency limits
        nchan_sel = (ch_hi_b - ch_lo_b) * args.bin_chn
        f_lo_ghz_sel = f_lo_mhz / 1e3
        f_hi_ghz_sel = f_hi_mhz / 1e3
        sub_band_info = sub_band_toa_fit(
            dat_binned_rfi_profile, t_prof, dm, f_lo_ghz_sel, f_hi_ghz_sel,
            nchan_sel,
            t0_sample, w0, args.bin_samp, args.bin_chn,
            args.n_sub_bands, t_bin_width, rms_off, tsamp_ms)
        n_valid = len(sub_band_info['sub_bands'])
        print(f"{n_valid}/{args.n_sub_bands} bands with S/N > 5")

        if n_valid > 0:
            toa_sb = sub_band_info['toa_ref']
            toa_sb_err = max(sub_band_info['toa_err'],
                            sub_band_info['toa_scatter'])
            toa_sb_mjd = mjd_start + toa_sb / sec_per_day
            print(f"  Sub-band TOA ({n_valid} bands):")
            print(f"    TOA       = {toa_sb:.8f}  s")
            print(f"    Wtd err   = {sub_band_info['toa_err']*1e6:.1f}  us")
            print(f"    Scatter   = {sub_band_info['toa_scatter']*1e6:.1f}  us"
                  f"  (cross-band RMS/sqrt(N))")
            print(f"    TOA err   = {toa_sb_err*1e6:.1f}  us"
                  f"  (max of wtd err, scatter)")
            print(f"    MJD       = {toa_sb_mjd:.10f}")
            # Sub-band detail
            for i, sb in enumerate(sub_band_info['sub_bands']):
                print(f"    Band {i+1}: {sb['f_lo']:.2f}-{sb['f_hi']:.2f} GHz"
                      f"  TOA={sb['toa']:.8f}  S/N={sb['snr']:.1f}")

    # ==================================================================
    # (compare block)  Run ALL strategies comprehensively
    # ==================================================================
    if args.compare:
        print("\n" + "=" * 62)
        print("  COMPARISON MODE — comprehensive multi-strategy analysis")
        print("=" * 62)

        # --- Strategy 1: Single Gaussian ---
        peak_idx_s = lo + np.argmax(profile[lo:hi])
        fit_lo_s = max(0, peak_idx_s - fit_half)
        fit_hi_s = min(nsamp_b, peak_idx_s + fit_half)
        r_single = fit_single_gaussian_run(
            t_prof[fit_lo_s:fit_hi_s], profile[fit_lo_s:fit_hi_s],
            t_prof, profile, peak_idx_s, w0, t_bin_width, nsamp_b,
            model_type=model_type)
        if r_single['ok']:
            popt_s = r_single['popt']
            mu_s = popt_s[1]
            sig_s = popt_s[2]
            amp_s = popt_s[0]
            try:
                pe_s = np.sqrt(np.diag(r_single['pcov']))
                mu_e_formal_s = pe_s[1]
            except (ValueError, np.linalg.LinAlgError):
                pe_s = np.full(len(popt_s), np.nan)
                mu_e_formal_s = np.nan
            fwhm_s = 2.35482 * abs(sig_s)
            # S/N from model peak (correct for non-Gaussian models)
            if model_type == 'gaussian':
                snr_s = abs(popt_s[0]) / rms_off if (rms_off and rms_off > 0) else np.inf
            else:
                model_peak_s = np.max(eval_profile_model(
                    r_single['t_fit'], popt_s, model_type))
                snr_s = model_peak_s / rms_off if (rms_off and rms_off > 0) else np.inf
            n_eff_s = max(1, fwhm_s / t_bin_width)
            mu_emp_s = abs(sig_s) / (snr_s * np.sqrt(n_eff_s)) if snr_s > 0 else np.inf
            resid_s = r_single['y_fit'] - eval_profile_model(
                r_single['t_fit'], popt_s, model_type)
            dof_s = len(r_single['y_fit']) - (n_pc + 1)
            chi2_s = np.sum(resid_s**2) / dof_s if dof_s > 0 else np.nan
            rss_s = np.sum(resid_s**2)
            aicc_s = compute_aicc(rss_s, n_pc + 1, len(r_single['y_fit']))
        else:
            mu_s, amp_s, sig_s, fwhm_s, snr_s, chi2_s, mu_emp_s, mu_e_formal_s, aicc_s = (
                t_prof[peak_idx_s], np.nan, w0, 2.35482*w0, np.nan, np.nan, np.nan, np.nan, np.inf)

        # --- Non-parametric TOAs (model-independent) ---
        toa_le = toa_leading_edge(t_prof, profile, fit_lo_s, fit_hi_s, rms_off)
        toa_cm = toa_center_of_mass(t_prof, profile, fit_lo_s, fit_hi_s, rms_off)
        toa_pk = toa_peak(t_prof, profile, fit_lo_s, fit_hi_s,
                          rms_off, t_bin_width)
        toa_sl = toa_shapelet(t_prof, profile, fit_lo_s, fit_hi_s, rms_off,
                             t_bin_width, n_max=args.shapelet_nmax,
                             n_poly=args.shapelet_npoly,
                             alpha=args.shapelet_alpha)

        # --- MCMC on single Gaussian (if --mcmc) ---
        mcmc_compare = None
        if args.mcmc and r_single['ok']:
            try:
                import emcee  # noqa: F811
            except ImportError:
                pass
            else:
                print(f"\n  MCMC on single Gaussian ...", flush=True)
                mcmc_compare = run_mcmc(
                    r_single['t_fit'], r_single['y_fit'], rms_off,
                    r_single['popt'], r_single['pcov'], 1,
                    n_walkers=args.mcmc_nwalkers,
                    n_burn=args.mcmc_nburn,
                    n_prod=args.mcmc_nprod,
                    progress=True, model_type=model_type)

        # --- Multi-Gaussian strategies: scan N = 2..max_n ---
        n_peaks = max(1, min(len(all_peak_indices), args.max_n))
        multi_strategies = {}  # key: 'N2_highest', 'N3_error_weighted', etc.

        for n_test in range(2, n_peaks + 1):
            # Set up fit window for N components
            peak_indices_test = all_peak_indices[:n_test]
            fit_lo_m = max(0, peak_indices_test.min() - fit_half // 2)
            fit_hi_m = min(nsamp_b, peak_indices_test.max() + fit_half // 2)
            extra_m = max(0, int(0.5 * fit_half))
            fit_lo_m = max(0, min(fit_lo_m, peak_indices_test.min() - extra_m))
            fit_hi_m = min(nsamp_b, max(fit_hi_m, peak_indices_test.max() + extra_m))

            t_fit_m = t_prof[fit_lo_m:fit_hi_m]
            y_fit_m = profile[fit_lo_m:fit_hi_m]

            r_multi = fit_multi_gaussian_run(
                t_fit_m, y_fit_m, profile, peak_indices_test,
                peak_indices_test[0], w0, t_prof,
                fit_lo_m, fit_hi_m, n_test,
                model_type=model_type)

            if r_multi['ok'] and r_multi['comps'] is not None:
                comps_m = r_multi['comps']
                # Compute AICc
                resid_m = y_fit_m - eval_profile_model(
                    t_fit_m, r_multi['popt'], model_type)
                rss_m = np.sum(resid_m**2)
                n_param = n_pc * n_test + 1
                aicc_m = compute_aicc(rss_m, n_param, len(y_fit_m))
                chi2_m = rss_m / (len(y_fit_m) - n_param) if len(y_fit_m) > n_param else np.inf

                # Extract TOA via all parametric strategies
                toa_h = extract_toa_highest(comps_m, rms_off, t_bin_width)
                toa_w = extract_toa_weighted(comps_m, r_multi['popt'], r_multi['pcov'],
                                             rms_off, t_bin_width, n_test)
                toa_ew = extract_toa_error_weighted(comps_m, r_multi['popt'], r_multi['pcov'],
                                                    rms_off, t_bin_width, n_test)
                toa_fp = extract_toa_first_peak(comps_m, rms_off, t_bin_width)

                entries = [('highest', toa_h), ('weighted', toa_w),
                           ('error_weighted', toa_ew)]
                if toa_fp is not None:
                    entries.append(('first_peak', toa_fp))
                for label, toa_info in entries:
                    key = f'N{n_test}_{label}'
                    multi_strategies[key] = {
                        'mu': toa_info['mu_f'],
                        'mu_e': toa_info.get('mu_e', np.nan),
                        'mu_emp': toa_info.get('mu_emp', np.nan),
                        'fwhm': toa_info.get('fwhm_f', np.nan),
                        'snr': toa_info.get('snr', np.nan),
                        'label': f'{n_test}-Gauss {label}',
                        'n': n_test, 'mode': label,
                        'aicc': aicc_m, 'chi2': chi2_m,
                        'r': r_multi, 'comps': comps_m,
                        'fit_lo': fit_lo_m, 'fit_hi': fit_hi_m,
                    }
            else:
                # Fit failed for this N
                for label in ['highest', 'weighted', 'error_weighted', 'first_peak']:
                    key = f'N{n_test}_{label}'
                    multi_strategies[key] = {
                        'mu': t_prof[peak_idx], 'mu_e': np.nan, 'mu_emp': np.nan,
                        'fwhm': np.nan, 'snr': np.nan,
                        'label': f'{n_test}-Gauss {label} (FAILED)',
                        'n': n_test, 'mode': label,
                        'aicc': np.inf, 'chi2': np.nan,
                        'r': None, 'comps': None,
                    }

        # --- Print comprehensive comparison table ---
        print(f"\n  {'Strategy':<30s} {'TOA (s)':<16s} {'+/- (us)':<10s} "
              f"{'TOA-M0 (us)':<12s} {'FWHM(ms)':<10s} {'S/N':<8s} {'AICc':<12s}")
        print("  " + "-" * 105)

        # Row 0: Single Gaussian  (compute best-AICc marker first)
        best_multi_aicc_pre = (min(v['aicc'] for v in multi_strategies.values()
                                  if v['r'] is not None)
                              if multi_strategies else np.inf)
        single_best = (np.isfinite(aicc_s) and aicc_s <= best_multi_aicc_pre)
        single_marker = ' <-- BEST AICc' if single_best else ''
        d0_us = (mu_s - mu_s) * 1e6  # 0
        print(f"  {'1. Single Gaussian':<30s} {mu_s:<16.8f} "
              f"{mu_emp_s*1e6:<10.1f} {d0_us:<12.1f} {fwhm_s*1e3:<10.3f} "
              f"{snr_s:<8.2f} {aicc_s:<12.1f}{single_marker}")

        # Row 1-4: Non-parametric
        for toa_np, np_label in [(toa_le, 'Leading edge (half-max+rise)'),
                                  (toa_cm, 'Center of mass (>3*RMS)'),
                                  (toa_pk, 'Peak (argmax, quad-interp)'),
                                  (toa_sl, 'Shapelet (Hermite-Gauss)')]:
            d_us = (toa_np['mu_f'] - mu_s) * 1e6
            mu_e_np = toa_np.get('mu_e', np.nan)
            snr_np = toa_np.get('snr', np.nan)
            fwhm_np = toa_np.get('fwhm_f', np.nan)
            fwhm_str = f'{fwhm_np*1e3:<10.3f}' if np.isfinite(fwhm_np) else '--'
            aicc_np = toa_np.get('aicc', np.nan)
            aicc_str = f'{aicc_np:<12.1f}' if np.isfinite(aicc_np) else '--'
            print(f"  {np_label:<30s} {toa_np['mu_f']:<16.8f} "
                  f"{mu_e_np*1e6 if np.isfinite(mu_e_np) else np.nan:<10.1f} "
                  f"{d_us:<12.1f} {fwhm_str:<10s} "
                  f"{snr_np if np.isfinite(snr_np) else np.nan:<8.2f} {aicc_str:<12s}")

        # Row: MCMC on single Gaussian (if available)
        if mcmc_compare is not None and mcmc_compare['ok']:
            mcmc_cmp = mcmc_toa_uncertainty(mcmc_compare['flat_samples'], 1,
                                            model_type=model_type)
            d_mcmc_us = (mcmc_cmp['mu_mcmc'] - mu_s) * 1e6
            print(f"  {'MCMC (single Gauss)':<30s} "
                  f"{mcmc_cmp['mu_mcmc']:<16.8f} "
                  f"{mcmc_cmp['mu_mcmc_std']*1e6:<10.1f} "
                  f"{d_mcmc_us:<12.1f} "
                  f"{fwhm_s*1e3:<10.3f} {'--':<8s} {'--':<12s}")

        # Row: Sub-band TOA (if available)
        if sub_band_info is not None and len(sub_band_info['sub_bands']) > 0:
            toa_sb = sub_band_info['toa_ref']
            toa_sb_e = max(sub_band_info['toa_err'],
                          sub_band_info['toa_scatter'])
            d_sb_us = (toa_sb - mu_s) * 1e6
            n_sb = len(sub_band_info['sub_bands'])
            print(f"  {'Sub-band (' + str(n_sb) + ' bands)':<30s} "
                  f"{toa_sb:<16.8f} {toa_sb_e*1e6:<10.1f} {d_sb_us:<12.1f} "
                  f"{'--':<10s} {'--':<8s} {'--':<12s}")

        # Rows: Multi-Gaussian strategies, sorted by AICc
        sorted_multi = sorted(
            [v for v in multi_strategies.values() if v['r'] is not None],
            key=lambda x: x['aicc'])

        # True best AICc: already computed as single_best above
        for entry in sorted_multi:
            d_us = (entry['mu'] - mu_s) * 1e6
            mu_e_disp = entry['mu_emp'] if np.isfinite(entry['mu_emp']) else entry['mu_e']
            aicc_best = (not single_best
                         and entry['aicc'] == sorted_multi[0]['aicc'])
            marker = ' <-- BEST AICc' if aicc_best else ''
            print(f"  {entry['label']:<30s} {entry['mu']:<16.8f} "
                  f"{mu_e_disp*1e6:<10.1f} {d_us:<12.1f} "
                  f"{entry['fwhm']*1e3 if np.isfinite(entry['fwhm']) else np.nan:<10.3f} "
                  f"{entry['snr'] if np.isfinite(entry['snr']) else np.nan:<8.2f} "
                  f"{entry['aicc']:<12.1f}{marker}")

        print("  " + "-" * 105)
        print(f"  TOA-M0 = difference from Single Gaussian TOA"
              f"  ({mu_s:.8f} s)")
        if single_best and np.isfinite(aicc_s):
            print(f"  AICc-best model: single Gaussian (AICc={aicc_s:.1f})")
        elif sorted_multi:
            print(f"  AICc-best model: {sorted_multi[0]['label']}"
                  f" (AICc={sorted_multi[0]['aicc']:.1f})")

        # Overall recommendation (includes non-parametric)
        if sorted_multi and np.isfinite(mu_emp_s) and mu_emp_s > 0:
            d_conv = abs(sorted_multi[0]['mu'] - mu_s)
            if d_conv < 3 * mu_emp_s:
                rec = 'single' if (single_best and np.isfinite(aicc_s)) else sorted_multi[0]['label']
                print(f"  => Recommended: {rec} (well-behaved,"
                      f" Delta_conv={d_conv*1e6:.0f} us < 3*sigma_emp={3*mu_emp_s*1e6:.0f} us)")
            else:
                le_ok = (toa_le.get('snr', 0) > 3
                         and np.isfinite(toa_le.get('mu_f', np.nan)))
                sl_ok = (toa_sl.get('snr', 0) > 5
                         and np.isfinite(toa_sl.get('mu_f', np.nan)))
                if le_ok:
                    rec = 'leading_edge (scattering-invariant)'
                elif sl_ok:
                    rec = 'shapelet (denoised model)'
                else:
                    rec = 'single (last-resort fallback)'
                note = ('center_of_mass excluded: scattering/channel smearing'
                        ' biases centroid late')
                print(f"  => Recommended: {rec} (overlapping,"
                      f" Delta_conv={d_conv*1e6:.0f} us > 3*sigma_emp={3*mu_emp_s*1e6:.0f} us)")
                print(f"     ({note})")
        else:
            rec_label = 'single' if (single_best and np.isfinite(aicc_s)) else 'single'
            print(f"  => Recommended: {rec_label}")

        # Store for plotting
        compare_results = {
            'single':   {'mu': mu_s,   'mu_e': mu_emp_s, 'amp': amp_s,
                         'sig': sig_s, 'fwhm': fwhm_s, 'snr': snr_s,
                         'aicc': aicc_s,
                         'chi2': chi2_s, 'r': r_single},
            'leading_edge': toa_le,
            'center_of_mass': toa_cm,
            'peak': toa_pk,
            'shapelet': toa_sl,
            'sub_band': sub_band_info,
            'multi_strategies': multi_strategies,
            'sorted_multi': sorted_multi,
        }

    # ------------------------------------------------------------------
    # 16. Plot — multi-panel schematic overview (or comparison figure)
    # ------------------------------------------------------------------
    if args.no_plot:
        print("  Plot skipped (--no_plot).")
        print("\nDone.")
        return

    if args.show:
        try:
            plt.switch_backend("TkAgg")
        except Exception:
            try:
                plt.switch_backend("Qt5Agg")
            except Exception:
                print("  Warning: cannot switch to interactive backend;"
                      " saving to file instead.")
                args.show = False
        if args.show:
            plt.ion()

    fig_path = args.output + "." + args.fig_format

    # ==================================================================
    # Comparison plot (--compare mode)
    # ==================================================================
    if compare_results is not None:
        cr = compare_results
        fname = args.input_file.replace('\\', '/').split('/')[-1]

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3,
                              height_ratios=[1.2, 1.1, 0.8])

        colors = {
            'single': '#d62728',
            'highest': '#1f77b4',
            'weighted': '#2ca02c',
            'error_weighted': '#ff7f0e',
            'first_peak': '#9467bd',
            'leading_edge': '#8c564b',
            'center_of_mass': '#e377c2',
            'peak': '#bcbd22',
            'shapelet': '#17becf',
        }
        styles = {
            'single': '--',
            'highest': '-',
            'weighted': '-.',
            'error_weighted': ':',
            'first_peak': (0, (3, 1, 1, 1)),
            'leading_edge': (0, (5, 2)),
            'center_of_mass': (0, (1, 1)),
            'peak': (0, (6, 3, 2, 3)),
            'shapelet': (0, (3, 5, 1, 5)),
        }
        comp_colors = ['#ff7f0e', '#9467bd', '#d62728', '#1f77b4', '#2ca02c']

        # ---- Panel A: Single Gaussian + non-parametric TOAs ----
        axA = fig.add_subplot(gs[0, 0])
        r_s = cr['single']['r']
        margin_a = int(1.5 * fit_half)
        xlo_a = max(0, peak_idx_s - margin_a)
        xhi_a = min(nsamp_b, peak_idx_s + margin_a)

        axA.plot(t_prof[xlo_a:xhi_a], profile[xlo_a:xhi_a], 'k-', lw=0.5,
                 label='Profile')
        axA.axhline(0, color='gray', ls=':', lw=0.5)

        if r_s['ok']:
            tA, yA = r_s['t_fit'], r_s['y_fit']
            axA.axvspan(tA[0], tA[-1], alpha=0.08, color='gray',
                        label='Fit window')
            t_sm = np.linspace(tA[0], tA[-1], 400)
            axA.plot(t_sm, eval_profile_model(t_sm, r_s['popt'], model_type),
                     color=colors['single'], ls='-', lw=1.5,
                     label=f'Single {model_type.capitalize()}')
            axA.axvline(cr['single']['mu'],
                        color=colors['single'], ls='--', lw=1.2,
                        label=f"Single TOA = {cr['single']['mu']:.6f} s")

        # Non-parametric TOAs
        for key in ['leading_edge', 'center_of_mass', 'peak', 'shapelet']:
            if key in cr:
                toa_np = cr[key]
                if np.isfinite(toa_np['mu_f']):
                    axA.axvline(toa_np['mu_f'], color=colors[key],
                                ls=styles[key], lw=1.5, alpha=0.8,
                                label=f"{key}={toa_np['mu_f']:.6f}")

        axA.set_ylabel('Flux (arb.)')
        axA.set_title(f'(A) Single Gaussian + Non-parametric TOAs  |  '
                      f'FWHM={cr["single"]["fwhm"]*1e3:.2f} ms  |  '
                      f'S/N={cr["single"]["snr"]:.1f}')
        axA.legend(fontsize=6, loc='best')

        # ---- Panel B: Best multi-Gaussian fit ----
        axB = fig.add_subplot(gs[0, 1])
        sorted_multi = cr['sorted_multi']

        if sorted_multi:
            best_entry = sorted_multi[0]
            r_best = best_entry['r']
            if r_best is not None and r_best['ok']:
                tB, yB = r_best['t_fit'], r_best['y_fit']
                axB.axvspan(tB[0], tB[-1], alpha=0.08, color='gray',
                            label=f'Fit ({best_entry["n"]} Gauss)')
                axB.plot(tB, yB, 'k.', ms=2, label='Data')
                t_sm = np.linspace(tB[0], tB[-1], 400)

                # Individual components
                for i, c in enumerate(best_entry['comps']):
                    fwhm_i = 2.35482 * abs(c['sig'])
                    # Build single-component params array
                    if model_type == 'gaussian':
                        comp_params = np.array([c['amp'], c['mu'], c['sig'],
                                                r_best['popt'][-1]])
                    elif model_type == 'emg':
                        comp_params = np.array([c['amp'], c['mu'], c['sig'],
                                                c.get('tau', 0.001),
                                                r_best['popt'][-1]])
                    elif model_type == 'voigt':
                        comp_params = np.array([c['amp'], c['mu'], c['sig'],
                                                c.get('gamma', 0.001),
                                                r_best['popt'][-1]])
                    y_c = eval_profile_model(t_sm, comp_params, model_type)
                    axB.plot(t_sm, y_c, '--',
                             color=comp_colors[i % len(comp_colors)],
                             lw=0.8, alpha=0.7,
                             label=f'C{i+1}: {c["mu"]:.6f}')

                # Total fit
                axB.plot(t_sm, eval_profile_model(t_sm, r_best['popt'],
                                                   model_type),
                         'k-', lw=1.8, label='Total fit')

                # Mark TOAs from all parametric strategies for this N
                for entry in sorted_multi[:4]:  # top 4 by AICc
                    if entry['n'] == best_entry['n']:
                        axB.axvline(entry['mu'], color=colors[entry['mode']],
                                    ls=styles[entry['mode']], lw=1.2, alpha=0.7,
                                    label=f"{entry['mode']}={entry['mu']:.6f}")

        axB.set_ylabel('Flux (arb.)')
        n_best = best_entry['n'] if sorted_multi else '?'
        axB.set_title(f'(B) Best model: {n_best}-Gaussian  |  '
                      f"AICc={best_entry['aicc']:.1f}" if sorted_multi else '')
        axB.legend(fontsize=5.5, loc='upper right', ncol=2)

        # ---- Panel C: Full profile with ALL TOAs ----
        axC = fig.add_subplot(gs[1, :])
        margin_c = int(1.0 * fit_half)
        xlo_c = max(0, lo - margin_c)
        xhi_c = min(nsamp_b, hi + margin_c)

        axC.plot(t_prof[xlo_c:xhi_c], profile[xlo_c:xhi_c], 'k-', lw=0.6)
        axC.axhline(0, color='gray', ls=':', lw=0.5)

        # Single + non-parametric
        y_max = np.max(profile[xlo_c:xhi_c])
        y_min = np.min(profile[xlo_c:xhi_c])
        y_span = y_max - y_min

        all_toa_labels = [('single', cr['single']['mu'], cr['single']['mu_e'])]
        for key in ['leading_edge', 'center_of_mass', 'peak', 'shapelet']:
            if key in cr and np.isfinite(cr[key]['mu_f']):
                all_toa_labels.append(
                    (key, cr[key]['mu_f'],
                     cr[key].get('mu_e', np.nan)))

        # Add best multi-Gaussian TOAs
        if sorted_multi:
            best_n = sorted_multi[0]['n']
            for entry in sorted_multi:
                if entry['n'] == best_n:
                    all_toa_labels.append(
                        (entry['mode'], entry['mu'], entry['mu_e']))

        # Plot TOA markers at staggered heights, labels offset to the right
        n_toa = len(all_toa_labels)
        y_positions = np.linspace(y_max * 0.95, y_max * 0.55, n_toa)
        for idx, (label, mu_val, mu_err) in enumerate(all_toa_labels):
            color = colors.get(label, '#333333')
            ls = styles.get(label, '-')
            axC.axvline(mu_val, color=color, ls=ls, lw=1.5, alpha=0.85)
            err_str = f' +/- {mu_err*1e6:.0f} us' if np.isfinite(mu_err) else ''
            axC.annotate(f'{label}: {mu_val:.8f} s{err_str}',
                         xy=(mu_val, y_positions[idx]),
                         xytext=(36, 0), textcoords='offset points',
                         fontsize=7, color=color, ha='left', va='center',
                         fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                   alpha=0.85, ec=color))

        # Shade fit window
        for entry in sorted_multi[:1]:
            if entry['r'] is not None:
                lo_fit = entry.get('fit_lo', fit_lo)
                hi_fit = entry.get('fit_hi', fit_hi)
                axC.axvspan(t_prof[max(xlo_c, lo_fit)],
                            t_prof[min(xhi_c, hi_fit) - 1],
                            alpha=0.06, color='C0')

        if pulse_truncated:
            axC.text(0.02, 0.95, 'PULSE TRUNCATED\n(DM sweep incomplete in this file)',
                     transform=axC.transAxes, fontsize=9, color='red',
                     fontweight='bold', va='top',
                     bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9, ec='red'))
        axC.set_ylabel('Flux (arb.)')
        axC.set_xlabel('Time (s)')
        auto_res_tag = f'  auto_res=x{auto_res_factor}' if auto_res_factor > 1 else ''
        axC.set_title(f'(C) Full profile + all TOA strategies  |  '
                      f'{fname}  |  DM={dm:.1f}{auto_res_tag}')

        # ---- Panel D: Dynamic spectrum ----
        axD = fig.add_subplot(gs[2, :])
        if dat_binned_rfi.size > 0:
            # Restrict to selected frequency range
            ds_data = dat_binned_rfi[ch_lo_b:ch_hi_b, :]
            n_ds_ch = ds_data.shape[0]
            ds_factor = max(1, ds_data.shape[1] // 800)
            if ds_factor > 1:
                n_trim = ds_data.shape[1] - (ds_data.shape[1] % ds_factor)
                ds_display = np.add.reduceat(
                    ds_data[:, :n_trim],
                    np.arange(0, n_trim, ds_factor), axis=1)
                t_a = t_prof[:n_trim:ds_factor]
                t_b = t_prof[ds_factor-1:n_trim:ds_factor]
                n_ds = min(len(t_a), len(t_b))
                t_ds = 0.5 * (t_a[:n_ds] + t_b[:n_ds])
            else:
                ds_display = ds_data
                t_ds = t_prof
            vmed = np.median(ds_display)
            vstd = np.std(ds_display)
            axD.imshow(ds_display, aspect='auto', origin='lower',
                       extent=[t_ds[0], t_ds[-1], f_lo_mhz, f_hi_mhz],
                       vmin=vmed - 3 * vstd, vmax=vmed + 3 * vstd,
                       cmap='binary', interpolation='none')

            for label, mu_val, _ in all_toa_labels:
                color = colors.get(label, '#333333')
                ls = styles.get(label, '-')
                axD.axvline(mu_val, color=color, ls=ls, lw=1.0, alpha=0.85)

        axD.set_xlabel('Time (s)')
        axD.set_ylabel('Frequency (MHz)')
        axD.set_title(f'(D) Dynamic spectrum  '
                      f'({f_lo_mhz:.0f}--{f_hi_mhz:.0f} MHz)  |  '
                      f'All TOAs marked')
        axD.set_xlim(t_prof[xlo_c], t_prof[xhi_c - 1])

        if args.show:
            print("  Displaying comparison plot (close window to exit) ...")
            plt.show()
        else:
            fig.savefig(fig_path, dpi=200 if args.fig_format == 'png' else None,
                        bbox_inches='tight')
            print(f"  -> {fig_path}")

        if args.tim:
            _write_tim_file(args, fit_ok, compare_results, mu_f, mu_emp, mu_e,
                            toa_mode, mjd_start, sec_per_day, toa_freq_mhz,
                            mcmc_result, mcmc_n_comp, model_type)
        print("\nDone.")
        return

    # ==================================================================
    # Normal plot (non-compare mode)
    # ==================================================================
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 10),
                        gridspec_kw={'height_ratios': [1.2, 1, 0.8]})

    fname = args.input_file.replace('\\', '/').split('/')[-1]

    # ---- top panel: full profile with fit overlay ----
    ax0.plot(t_prof, profile, 'k-', lw=0.5, label='Baseline-subtracted profile')
    ax0.axhline(0, color='gray', ls=':', lw=0.5)
    if fit_ok:
        t_smooth = np.linspace(t_fit[0], t_fit[-1], 500)
        if n_comp >= 2 and comps is not None:
            # Plot individual components
            colors_comp = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for i, c in enumerate(comps):
                color = colors_comp[i % len(colors_comp)]
                if model_type == 'gaussian':
                    c_params = np.array([c['amp'], c['mu'], c['sig'],
                                         popt[-1]])
                elif model_type == 'emg':
                    c_params = np.array([c['amp'], c['mu'], c['sig'],
                                         c.get('tau', 0.001), popt[-1]])
                elif model_type == 'voigt':
                    c_params = np.array([c['amp'], c['mu'], c['sig'],
                                         c.get('gamma', 0.001), popt[-1]])
                y_comp = eval_profile_model(t_smooth, c_params, model_type)
                ax0.plot(t_smooth, y_comp, '--', color=color, lw=1.0,
                         label=f'C{i+1}: {c["mu"]:.6f} s')
            # Total fit
            ax0.plot(t_smooth, eval_profile_model(t_smooth, popt, model_type),
                     'r-', lw=1.5, label='Total fit')
        else:
            ax0.plot(t_smooth, eval_profile_model(t_smooth, popt, model_type),
                     'r-', lw=1.5,
                     label=f'{model_type.capitalize()} fit')
        ax0.axvline(mu_f, color='r', ls='--', lw=1.0,
                    label=f'TOA = {mu_f:.6f} s  |  MJD {toa_mjd:.10f}')
    ax0.axvspan(t_prof[fit_lo], t_prof[fit_hi - 1],
                alpha=0.10, color='C0', label='Fit window')
    if pulse_truncated:
        ax0.text(0.02, 0.95, 'PULSE TRUNCATED\n(DM sweep incomplete in this file)',
                 transform=ax0.transAxes, fontsize=9, color='red',
                 fontweight='bold', va='top',
                 bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9, ec='red'))
    ax0.set_ylabel('Flux (arb.)')
    auto_res_str = f'  auto_res = x{auto_res_factor}' if auto_res_factor > 1 else ''
    ax0.set_title(f'{fname}  |  DM = {dm:.1f}  bs = {args.bin_samp}  '
                  f'bf = {args.bin_chn}{auto_res_str}  |  ngauss = {n_comp}  '
                  f'mode = {toa_mode}')
    ax0.legend(fontsize=7, loc='best')
    margin = 0.2 * (t_wide_hi - t_wide_lo)
    ax0.set_xlim(t_wide_lo - margin, t_wide_hi + margin)

    # ---- middle panel: fit window + residuals ----
    ax1.plot(t_fit, y_fit, 'k.', ms=3, label='Data')
    if fit_ok:
        t_dense = np.linspace(t_fit[0], t_fit[-1], 500)
        if n_comp >= 2 and comps is not None:
            colors_comp = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for i, c in enumerate(comps):
                color = colors_comp[i % len(colors_comp)]
                if model_type == 'gaussian':
                    c_params = np.array([c['amp'], c['mu'], c['sig'],
                                         popt[-1]])
                elif model_type == 'emg':
                    c_params = np.array([c['amp'], c['mu'], c['sig'],
                                         c.get('tau', 0.001), popt[-1]])
                elif model_type == 'voigt':
                    c_params = np.array([c['amp'], c['mu'], c['sig'],
                                         c.get('gamma', 0.001), popt[-1]])
                y_comp = eval_profile_model(t_dense, c_params, model_type)
                ax1.plot(t_dense, y_comp, '--', color=color, lw=0.8,
                         label=f'C{i+1}')
            ax1.plot(t_dense, eval_profile_model(t_dense, popt, model_type),
                     'r-', lw=1.5, label='Total fit')
            resid = y_fit - eval_profile_model(t_fit, popt, model_type)
        else:
            ax1.plot(t_dense, eval_profile_model(t_dense, popt, model_type),
                     'r-', lw=1.5,
                     label=f'{model_type.capitalize()} fit')
            resid = y_fit - eval_profile_model(t_fit, popt, model_type)
        ax1.axvline(mu_f, color='r', ls='--', lw=1.0)
        # residuals on twin axis
        ax1r = ax1.twinx()
        ax1r.plot(t_fit, resid, 'g.', ms=3, alpha=0.5)
        ax1r.set_ylabel('Residual', color='g')
        ax1r.axhline(0, color='g', ls=':', lw=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Flux (arb.)')
    if n_comp >= 2:
        title_str = f'Fit window  |  {n_comp}-Gaussian model  |  '
    else:
        title_str = f'Fit window  |  '
    title_str += (f'TOA = {mu_f:.8f} s  |  '
                  f'FWHM = {fwhm_f * 1e3:.2f} ms  |  '
                  f'S/N = {snr:.1f}')
    if fit_ok:
        title_str += f'  |  chisq_red = {chi2_red:.1f}'
    ax1.set_title(title_str)
    ax1.legend(fontsize=7, loc='best')

    # ---- bottom panel: dynamic spectrum (dedispersed + binned) ----
    if dat_binned_rfi.size > 0:
        # Restrict to selected frequency range
        ds_data = dat_binned_rfi[ch_lo_b:ch_hi_b, :]
        ds_factor = max(1, ds_data.shape[1] // 800)
        if ds_factor > 1:
            n_trim = ds_data.shape[1] - (ds_data.shape[1] % ds_factor)
            ds_display = np.add.reduceat(
                ds_data[:, :n_trim],
                np.arange(0, n_trim, ds_factor), axis=1)
            t_a = t_prof[:n_trim:ds_factor]
            t_b = t_prof[ds_factor-1:n_trim:ds_factor]
            n_ds = min(len(t_a), len(t_b))
            t_ds = 0.5 * (t_a[:n_ds] + t_b[:n_ds])
        else:
            ds_display = ds_data
            t_ds = t_prof

        vmed = np.median(ds_display)
        vstd = np.std(ds_display)
        ax2.imshow(ds_display, aspect='auto', origin='lower',
                   extent=[t_ds[0], t_ds[-1], f_lo_mhz, f_hi_mhz],
                   vmin=vmed - 3 * vstd, vmax=vmed + 3 * vstd,
                   cmap='binary', interpolation='none')
        if fit_ok:
            ax2.axvline(mu_f, color='r', ls='--', lw=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (MHz)')
    ax2.set_title(f'Dedispersed dynamic spectrum  '
                  f'({f_lo_mhz:.0f}--{f_hi_mhz:.0f} MHz)')
    ax2.set_xlim(ax0.get_xlim())

    plt.tight_layout()

    if args.show:
        print("  Displaying plot interactively (close window to exit) ...")
        plt.show()
    else:
        fig.savefig(fig_path, dpi=200 if args.fig_format == 'png' else None,
                        bbox_inches='tight')
        print(f"  -> {fig_path}")

    # --- TEMPO2 .tim output ---
    if args.tim:
        _write_tim_file(args, fit_ok, compare_results, mu_f, mu_emp, mu_e,
                        toa_mode, mjd_start, sec_per_day, toa_freq_mhz,
                        mcmc_result, mcmc_n_comp, model_type)

    print("\nDone.")


if __name__ == '__main__':
    main()
