import numpy as np
import scipy.interpolate as interpolate
import excursion as exc
import cloud as cl
from consts_defaults import pc2cm, Msol2g
import generate_cloudrho0 as gcr
import os 
import matplotlib.pyplot as plt

# ---------- helpers ----------

print('Current issues:')
print('1. The star mass calculation is not working correctly. Needs to calculate turbulent structure around threshold density.')
print('2. The trajectories that cross the threshold need to be resampled to reflect volume-weighting.')

def pick_i_for_radius(grid, R_target_cm):
    r = np.asarray(grid.rlevels, dtype=float)
    return int(np.argmin(np.abs(r - float(R_target_cm))))

def make_rglobal_func(grid):
    """ρ_global(R): initial region density vs scale (from grid.rhocs)."""
    return interpolate.interp1d(np.asarray(grid.rlevels, float),
                                np.asarray(grid.rhocs,   float),
                                kind="linear", fill_value="extrapolate", assume_sorted=False)

def rho_star_from_time(cloud, t_star_seconds):
    """ρ*(t*): initial density that has collapsed by t* (Girichidis inversion)."""
    return float(cloud.calc_rho_acc(t_star_seconds))

def effective_delta(delta_cumulative, i0):
    """
    Remove contributions from scales larger than the original cloud size R0.
    δ_eff[i] = δ[i] - δ[i0-1] for i>=i0; 0 for i<i0.
    """
    delta = np.asarray(delta_cumulative, dtype=float)
    base  = 0.0 if i0 == 0 else float(delta[i0-1])
    out   = np.zeros_like(delta)
    out[i0:] = delta[i0:] - base
    return out

# ---------- core: upcrossing on the correct barrier ----------

def find_largest_upcrossing(traj, grid, rho_star, R0_cm):
    """
    Use ρ_global(R)=grid.rhocs as the baseline, and ρ*(t*) as the collapse threshold.
    Returns crossing info at the largest scale (first from large->small) or None if no hit.
    """
    r       = np.asarray(grid.rlevels,   dtype=float)
    mu_lnr  = np.asarray(grid.mu_lnrho,  dtype=float)        # μ_lnρ(R)
    i0      = pick_i_for_radius(grid, R0_cm)
    rho_g   = make_rglobal_func(grid)(r)                     # ρ_global(R)
    delta_e = effective_delta(traj.delta, i0)                # δ_eff with large scales removed

    # Barrier: δ_th(R) = ln(ρ*/ρ_global(R)) - μ_lnρ(R)
    with np.errstate(divide='ignore'):
        delta_th = np.log(np.clip(rho_star / rho_g, 1e-300, np.inf)) - mu_lnr

    #print(f"R0 = {R0_cm/pc2cm:.3f} pc, i0 = {i0}, rho_star = {rho_star:.3e} g/cm^3, rho_g = {rho_g[i0]:.3e} g/cm^3")

    # First upcrossing from large->small scales
    for i in range(i0, len(r)):
        #print(f"Checking i={i}, R={r[i]/pc2cm:.3f} pc, delta_e={delta_e[i]:.3f}, delta_th={delta_th[i]:.3f}")  # Debug output
        if delta_e[i] >= delta_th[i]:
            R_cross   = float(r[i])
            rho_pre   = float(rho_g[i] * np.exp(mu_lnr[i] + delta_e[i]))  # realized pre-collapse region density
            return dict(index=i, i0=i0, R_cross=R_cross, rho_pre_cross=rho_pre,
                        delta_eff_at_cross=float(delta_e[i]), delta_thresh=float(delta_th[i]))
    return None

# ---------- repeat-draw until upcrossing, then estimate M_* ----------

def draw_trajectory_until_upcrossing(grid, cloud, t_star_seconds, R0_cm,
                                     dt_factor=0.1, max_tries=3000, seed=None, progress=False):
    rho_star = rho_star_from_time(cloud, t_star_seconds)   # collapsed-threshold density (pre-collapse) at t*

    rng = np.random.default_rng(seed)

    # Precompute for speed
    Rmin_grid = float(np.min(grid.rlevels))  # smallest possible scale in the grid


    for n in range(max_tries):
        # draw a trajectory; seed handling depends on excursion.trajectory API
        try:
            traj = exc.trajectory(grid=grid, dt_factor=dt_factor, seed=None if seed is None else seed + n)
        except TypeError:
            if seed is not None:
                np.random.seed(seed + n)
            traj = exc.trajectory(grid=grid, dt_factor=dt_factor)

        hit = find_largest_upcrossing(traj, grid, rho_star, R0_cm)
        if hit is None:
            if progress and (n+1) % 50 == 0:
                print(f"[{n+1} tries] no upcross yet...")
            continue

        R_cross = float(hit["R_cross"])
        # Guard against any numerical weirdness
        denom = max(R_cross, Rmin_grid)
        p_accept = min(1.0, (Rmin_grid / denom) ** 3)

        # Accept–reject
        u = rng.random()
        if u <= p_accept:
            if progress:
                print(f"[try {n+1}] upcross @ R={R_cross/pc2cm:.4f} pc, accepted (p={p_accept:.3g})")
            hit.update({
                "traj": traj,
                "attempt": n + 1,          # try index when we FIRST saw this upcross
                "accepted_on_try": n + 1,  # try index when it was accepted
                "rho_star": rho_star,
                "p_accept": p_accept
            })
            return hit
        else:
            if progress:
                print(f"[try {n+1}] upcross @ R={R_cross/pc2cm:.4f} pc, rejected (p={p_accept:.3g})")

    raise RuntimeError(f"No upcrossing in {max_tries} draws; consider changing t*, R0, or turbulence.")

def star_mass_from_hit(cloud, hit, use_exponential_profile=True):
    """
    Convert a successful upcrossing to gas mass and stellar mass.
    ρ_pre at R_cross is a *region* density (pre-collapse); we map it to mass, then ε_core.
    """
    R = hit["R_cross"]
    rho_pre = hit["rho_pre_cross"]

    if use_exponential_profile:
        M_gas = cloud.calc_Mgas(rho=rho_pre, R=R, update=False)
    else:
        M_gas = (4.0/3.0) * np.pi * rho_pre * (R**3)

    eps = getattr(cloud, "eps_core", 0.5)
    M_star_g   = eps * M_gas
    M_star_Msun= M_star_g / Msol2g

    hit.update({"M_gas": M_gas, "M_star_g": M_star_g, "M_star_Msun": M_star_Msun})
    return hit



# ---- IMFish wrappers ----

def sample_one_star(
    target_radius_cm,
    dt_myr=0.01,
    tmax_myr=10.0,
    base_seed=None,
    dt_factor=0.1,
    max_tries=100000,
    progress=False
):
    """
    Returns (M_star_Msun, hit_dict).
    Uses gcr.build_cloud_baseline_fn to get (grid, cloud, t_star), then draws a trajectory
    until (accepted) upcrossing, converts to star mass via star_mass_from_hit.
    """
    # Build a new cloud+time draw so each star is independent
    # NOTE: we only need the 'data' dict from the builder (grid, cloud, t_star)
    _, data = gcr.build_cloud_baseline_fn(
        rmin = 0.01*pc2cm,  # 0.01 pc minimum radius
        target_radius_cm=target_radius_cm,
        drfact=0.95,
        dt_myr=dt_myr,
        tmax_myr=tmax_myr,
        seed=base_seed
    )

    grid   = data['grid']
    cloud  = data['cloud']
    t_star = data['t_star']
    R0_cm  = target_radius_cm

    # Draw trajectories until upcrossing (and optionally acceptance)
    hit = draw_trajectory_until_upcrossing(
            grid, cloud, t_star, R0_cm,
            dt_factor=dt_factor, max_tries=max_tries,
            seed=base_seed, progress=progress
        )

    # Convert to star mass
    hit = star_mass_from_hit(cloud, hit, use_exponential_profile=True)
    return hit["M_star_Msun"], hit


def sample_imf(
    n_stars=200,
    target_radius_cm=1.0*pc2cm,
    dt_myr=0.01,
    tmax_myr=10.0,
    base_seed=42,
    dt_factor=0.1,
    max_tries=100000,
    verbose_every=20
):
    """
    Sample an IMF by repeating the star-generation workflow n_stars times.
    Returns:
      masses_msun : (n,) array
      hits        : list of per-star hit dicts (diagnostics)
    """
    rng = np.random.default_rng(base_seed)
    masses = []
    hits   = []

    for i in range(n_stars):
        # Derive per-star seeds for reproducibility but independence
        seed_i = None if base_seed is None else int(rng.integers(0, 2**31-1))
        try:
            Mstar, hit = sample_one_star(
                target_radius_cm=target_radius_cm,
                dt_myr=dt_myr,
                tmax_myr=tmax_myr,
                base_seed=seed_i,
                dt_factor=dt_factor,
                max_tries=max_tries,
                progress=False
            )
            masses.append(Mstar)
            hits.append(hit)
        except RuntimeError as e:
            # If no up/accept within max_tries, skip this star (or retry with a new seed)
            # Here we just retry once with a fresh seed:
            seed_retry = None if base_seed is None else int(rng.integers(0, 2**31-1))
            try:
                Mstar, hit = sample_one_star(
                    target_radius_cm=target_radius_cm,
                    dt_myr=dt_myr,
                    tmax_myr=tmax_myr,
                    base_seed=seed_retry,
                    dt_factor=dt_factor,
                    max_tries=max_tries,
                    progress=False
                )
                masses.append(Mstar)
                hits.append(hit)
            except RuntimeError:
                # give up on this index
                if verbose_every and ((i+1) % verbose_every == 0):
                    print(f"[{i+1}/{n_stars}] failed to sample star even after retry")
                continue

        if verbose_every and ((i+1) % verbose_every == 0):
            print(f"[{i+1}/{n_stars}] sampled so far")

    return np.array(masses, dtype=float), hits

def compute_or_load_imf(
    cache_path,
    n_stars=200,
    target_radius_cm=1.0*pc2cm,
    dt_myr=0.01,
    tmax_myr=10.0,
    base_seed=42,
    dt_factor=0.1,
    max_tries=100000,
    overwrite=False,
    verbose_every=25
):
    """
    Load masses from cache if it exists, otherwise sample and save.
    Returns: masses_msun (np.ndarray), meta (dict)
    """
    if (not overwrite) and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        masses_msun = data["masses_msun"]
        meta = dict(data["meta"].item()) if "meta" in data.files else {}
        return masses_msun, meta

    # --- otherwise, compute via your sampler wrappers defined earlier ---
    masses_msun, hits = sample_imf(
        n_stars=n_stars,
        target_radius_cm=target_radius_cm,
        dt_myr=dt_myr,
        tmax_myr=tmax_myr,
        base_seed=base_seed,
        dt_factor=dt_factor,
        max_tries=max_tries,
        verbose_every=verbose_every
    )

    meta = {
        "n_stars": int(n_stars),
        "target_radius_cm": float(target_radius_cm),
        "dt_myr": float(dt_myr),
        "tmax_myr": float(tmax_myr),
        "base_seed": None if base_seed is None else int(base_seed),
        "dt_factor": float(dt_factor),
        "max_tries": int(max_tries),
        "n_sampled": int(len(masses_msun))
    }

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(cache_path, masses_msun=masses_msun, meta=meta)
    return masses_msun, meta

# ===============================
# Analytic IMFs (in log-space)
# ===============================
# We want φ_log10(m) = dN/dlog10 m, normalized so ∫ φ_log10(m) dlog10 m = 1 on [m_min, m_max].

LN10 = np.log(10.0)

def _normalize_logpdf(logm_grid, phi_unnorm):
    """Numerically normalize a log-space PDF over log10(m)."""
    # trapezoidal integral over x=log10 m
    area = np.trapz(phi_unnorm, logm_grid)
    if not np.isfinite(area) or area <= 0:
        return phi_unnorm * 0.0
    return phi_unnorm / area

def _powerlaw_dndm(m, alpha):
    """Salpeter-like: dN/dM ∝ M^{-alpha}."""
    return np.where(m > 0, m**(-alpha), 0.0)

def _to_log10_pdf_from_dndm(m, dndm):
    """Convert dN/dM to dN/dlog10M: φ_log10 = (ln 10) * M * dN/dM."""
    return LN10 * m * dndm

def salpeter_logpdf(logm, alpha=2.35):
    m = 10**logm
    return _to_log10_pdf_from_dndm(m, _powerlaw_dndm(m, alpha))

def kroupa2001_logpdf(logm):
    """
    Kroupa (2001) piecewise power-law (single-star):
    alpha = 0.3 for 0.01–0.08 Msun
    alpha = 1.3 for 0.08–0.5 Msun
    alpha = 2.3 for 0.5–120 Msun
    Continuity enforced between segments.
    Returns UNnormalized φ_log10(m) on any grid.
    """
    m = 10**logm
    phi = np.zeros_like(m)

    # Slopes
    alpha0, alpha1, alpha2 = 0.3, 1.3, 2.3
    b0, b1 = 0.08, 0.5

    # Continuity constants: A1, A2 relative to A0
    A0 = 1.0
    A1 = A0 * (b0)**(alpha1 - alpha0)
    A2 = A1 * (b1)**(alpha2 - alpha1)

    # dN/dM in each segment
    dndm = np.zeros_like(m)
    seg0 = (m >= 0.01) & (m < b0)
    seg1 = (m >= b0)   & (m < b1)
    seg2 = (m >= b1)

    dndm[seg0] = A0 * m[seg0]**(-alpha0)
    dndm[seg1] = A1 * m[seg1]**(-alpha1)
    dndm[seg2] = A2 * m[seg2]**(-alpha2)

    return _to_log10_pdf_from_dndm(m, dndm)

def chabrier2003_logpdf(logm, m_c=0.22, sigma=0.57, alpha_hi=2.3):
    """
    Chabrier (2003, single-star version):
      for m < 1 Msun: log-normal in log10(m) with mean log10(m_c) and scatter sigma (dex)
      for m >= 1 Msun: power-law dN/dM ∝ M^{-alpha_hi}, matched continuously at 1 Msun.
    Returns UNnormalized φ_log10(m).
    """
    m = 10**logm
    phi = np.zeros_like(m)

    # log-normal for x<0 (m<1)
    x = logm
    x0 = np.log10(m_c)
    low = x < 0.0
    phi[low] = np.exp(-0.5 * ((x[low] - x0)/sigma)**2)

    # high-mass power law for m>=1
    high = ~low
    # dN/dM ∝ M^{-alpha}, so φ_log10 ∝ (ln 10) M^{1-alpha}
    phi_hi = (10**(logm[high]))**(1.0 - alpha_hi)

    # continuity at m=1 Msun: match values of φ at x=0
    # compute low-branch value at x=0
    phi_low_at_1 = np.exp(-0.5 * ((0.0 - x0)/sigma)**2)
    # current high-branch value at x=0 is 1^(1-alpha) = 1
    C_hi = phi_low_at_1  # scale factor so branches meet
    phi[high] = C_hi * phi_hi

    # include the (ln 10) factor? Already implicit in both branches because:
    # - low branch defined in log-space,
    # - high branch constructed as φ_log10 ∝ M^{1-α}.
    return phi


# ===============================
# Plotting (normalized histogram + normalized IMF curves)
# ===============================
def plot_imf_with_models(
    masses_msun,
    nbins=25,
    mmin_cut=0.01,      # drop anything below this (Msun)
    mmax=None,          # if None: max of filtered sample
    models=("Kroupa 2001", "Chabrier 2003"), #, "Salpeter 1955"),
    save=None,
    show=True,
    verify_print=True
):
    """
    Area-normalized in log-space:
      - Histogram is a probability density per dex (log10 M),
        so sum(density_i * bin_width_i) = 1.
      - Model curves are renormalized over the same [log10(mmin_cut), log10(mmax)] domain
        so their area (∫ φ_log10 dlog10M) = 1 as plotted.
    """
    m = np.asarray(masses_msun, float)
    m = m[np.isfinite(m) & (m >= float(mmin_cut))]
    if m.size == 0:
        raise ValueError(f"No masses >= {mmin_cut} Msun to plot.")

    # Domain in log10 space
    log_lo = np.log10(mmin_cut)
    log_hi = np.log10(np.max(m) if mmax is None else float(mmax))
    if not np.isfinite(log_hi) or log_hi <= log_lo:
        raise ValueError("Invalid plotting range; check mmin_cut/mmax or sample.")

    # Bins and widths in log10 M
    edges = np.linspace(log_lo, log_hi, nbins + 1)
    widths = np.diff(edges)
    centers = edges[:-1] + 0.5 * widths

    # Histogram as *density per dex*: sum(density * width) = 1
    logm = np.log10(m)
    counts, _ = np.histogram(logm, bins=edges)
    total = counts.sum()
    if total == 0:
        raise ValueError("All counts are zero after filtering.")
    density = counts / (total * widths)  # per-dex PDF

    # Build model curves on same domain, normalize area=1 in log space
    xs = np.linspace(log_lo, log_hi, 800)  # log10 M grid
    model_curves = []
    for name in models:
        key = name.lower()
        if key.startswith("kroupa"):
            phi = kroupa2001_logpdf(xs)           # returns φ_log10 (unnormalized)
        elif key.startswith("chabrier"):
            phi = chabrier2003_logpdf(xs)         # returns φ_log10 (unnormalized)
        elif key.startswith("salpeter"):
            phi = salpeter_logpdf(xs, alpha=2.35) # returns φ_log10 (unnormalized)
        else:
            continue
        phi = _normalize_logpdf(xs, phi)          # area under curve in log-space = 1
        model_curves.append((name, xs, phi))

    # Plot: bars with *exact* bin widths so plotted area equals 1 as well
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.bar(edges[:-1], density, width=widths, align='edge',
           alpha=0.5, edgecolor='k', label=f"Sample ($m \geq {mmin_cut} \,M_\\odot$)")
    for name, xs, phi in model_curves:
        ax.plot(xs, phi, lw=2.0, label=name)

    ax.set_xlim(log_lo, log_hi)
    ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
    #ax.set_yscale('log')  # log scale for density
    #ax.set_ylim([1e-2,])  # log scale for density
    ax.set_ylabel(r'Probability density per dex')  # per unit log10 M
    ax.grid(True, ls=':', alpha=0.4)
    ax.legend(loc='best')
    plt.tight_layout()

    # Optional verification of area normalization
    if verify_print:
        area_hist = float(np.sum(density * widths))
        msg = f"[check] histogram area ≈ {area_hist:.6f}"
        for name, xs, phi in model_curves:
            area = float(np.trapz(phi, xs))
            msg += f" | {name} area ≈ {area:.6f}"
        print(msg)

    if save:
        fig.savefig(save, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig



if __name__ == "__main__":
    # Parameters
    R0_cm   = 1.0 * pc2cm
    N       = 200
    SEED    = 12345

    # Choose cache file (relative path is fine)
    cache_file = "cache/imf_R1pc_weighted_200stars.npz"
    masses_msun, meta = compute_or_load_imf(
         cache_path=cache_file,
         n_stars=1000,
         target_radius_cm=1.0*pc2cm,
         dt_myr=0.01,
         tmax_myr=10.0,
         base_seed=12345,
         dt_factor=0.1,
         max_tries=100000,
         overwrite=False,            # set True to resample and overwrite cache
         verbose_every=25
    )
    print(f"Loaded/sampled {len(masses_msun)} stars; meta={meta}")
    plot_imf_with_models(masses_msun, nbins=15, save="imf_comparison.png", show=True, mmin_cut=0.1)