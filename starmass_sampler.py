import numpy as np
import scipy.interpolate as interpolate
import excursion as exc
import cloud as cl
from consts_defaults import pc2cm, Msol2g, Gcgs
import generate_cloudrho0 as gcr
import gen_sphere_adaptive_unstructured as gsau
import os 
import matplotlib.pyplot as plt
import core_analysis as ca
import plot_jeans_regions as pjm
# ---------- helpers ----------

print('Current issues:')
print('1. The star mass calculation is not working correctly. Needs to calculate turbulent structure around threshold density.')
print('2. The trajectories that cross the threshold need to be resampled to reflect volume-weighting.')

print("EDIT: NEED TO RETHINK PHYSICS --- ACTUALLY, SHOULD FIND JEANS UNSTABLE SUB-REGIONS DURING COLLAPSE")

def pick_i_for_radius(grid, R_target_cm):
    r = np.asarray(grid.rlevels, dtype=float)
    return int(np.argmin(np.abs(r - float(R_target_cm)*2.)))


def pick_i_for_jeans(grid, rho_star, c_s=0.1*1e5, G=Gcgs):
    r = np.asarray(grid.rlevels, dtype=float)
    RJ = jeans_length(rho_star, c_s=c_s, G=G)
    return int(np.argmin(np.abs(r - RJ)))

def make_rglobal_func(grid):
    """ρ_global(R): initial region density vs scale (from grid.rhocs)."""
    return interpolate.interp1d(np.asarray(grid.rlevels, float),
                                np.asarray(grid.rhocs,   float),
                                kind="linear", fill_value="extrapolate", assume_sorted=False)

def rho_star_from_time(cloud, t_star_seconds):
    """ρ*(t*): initial density that has collapsed by t* (Girichidis inversion)."""
    return float(cloud.calc_rho_acc(t_star_seconds))

def jeans_length(rho, c_s=0.1*1e5, G=Gcgs):
    return c_s/np.sqrt(G*rho)



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
    delta   = np.asarray(traj.delta,      dtype=float)  # δ(R)
    rho0_t, v0_t, Lcut_t = traj._get_baseline(0.0)
    rho     = rho0_t*np.exp(delta + grid.mu_lnrho)  # ρ(R) = ρ0 * exp(δ + μ_lnρ)
    i0      = pick_i_for_radius(grid, R0_cm)
    imax    = len(r) #pick_i_for_jeans(grid, rho_star)

    RJ = jeans_length(rho_star)


    #print(f"R0 = {R0_cm/pc2cm:.3f} pc, i0 = {i0}, rho_star = {rho_star:.3e} g/cm^3, rho_g = {rho_g[i0]:.3e} g/cm^3")

    # First upcrossing from large->small scales
    for i in range(i0, imax):
        #print(f"Checking i={i}, R={r[i]/pc2cm:.3f} pc, delta_e={delta_e[i]:.3f}, delta_th={delta_th[i]:.3f}")  # Debug output
        if rho[i] >= rho_star:
            print(f"Crossed threshold (rho, rho_st, R, RJ): {rho[i]:.2e}, {rho_star:.2e}, {r[i]/pc2cm:.2f}, {RJ/pc2cm:.2f}")
            R_cross   = float(r[i])
            """plt.scatter(r/pc2cm, rho, marker='o', s=1, label='$\\rho(R)$')
            plt.axhline(rho_star, color='red', ls='--', label='$\\rho^*(t^*)$')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("$R$ [pc]")
            plt.ylabel("$\\rho$ [g cm$^{-3}$]")
            plt.show()"""

            return dict(index=i, i0=i0, R_cross=R_cross, rho_pre_cross=rho_star,
                        delta_eff_at_cross=float(delta[i]))
    return None

# ---------- repeat-draw until upcrossing, then estimate M_* ----------

def draw_trajectory_until_upcrossing(grid, cloud, t_star_seconds, R0_cm,
                                     dt_factor=0.1, max_tries=3000, seed=None, progress=False, baseline_fn=None):
    rho_star = rho_star_from_time(cloud, t_star_seconds)   # collapsed-threshold density (pre-collapse) at t*

    print(f"Required density: {rho_star:.3e} g/cm^3 at t*={t_star_seconds:.3f} s")
    rng = np.random.default_rng(seed)

    # Precompute for speed
    Rmin_grid = float(np.min(grid.rlevels))  # smallest possible scale in the grid


    for n in range(max_tries):
        # draw a trajectory; seed handling depends on excursion.trajectory API
        try:
            #Note -- t0 is zero because we are mapping densities to t=0! 
            traj = exc.trajectory(grid=grid, dt_factor=dt_factor, baseline_fn=baseline_fn, t0=0.0, seed=None if seed is None else seed + n) 
            '''plt.scatter(traj.grid.rlevels/pc2cm, traj.baseline_fn(0.0)['rho0'] *np.exp(traj.delta+traj.grid.mu_lnrho))
            plt.axvline(traj.baseline_fn(0.0)["Lcut"]/pc2cm)
            plt.axhline(rho_star)
            plt.yscale('log')
            plt.xscale('log')
            plt.show()'''
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
        R_J = jeans_length(rho_star)
        # Guard against any numerical weirdness
        denom = max(R_cross, Rmin_grid)
        num = Rmin_grid
        p_accept = min(1.0, (num / denom) ** 3)
        print(f'Denominator: {R_cross/pc2cm:.2f}, {Rmin_grid/pc2cm:.2f}, {R_J/pc2cm:.2f}')

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

    Rcloud_max = float(np.amax(np.asarray(cloud.Rclouds))) / pc2cm  # max spatial scale in pc
    print("Rcloud_max (pc):", Rcloud_max)
    
    mra, info = init_mra_from_hit_trajectory(
    hit,
    baseline_fn=None,
    mra_kwargs=None,
    cells_per_level=100,
    profile=True,
    maxspatial_pc=Rcloud_max
    )
    
    plot_polar_equatorial_slice_from_mra(
    mra,
    hit,
    Nr=128,
    nside=32,
    theta_tol_deg=5.0,
    vmin=None, vmax=None,
    coords_path=None,
    log_scale="log10",        # 'log10' | 'ln' | 'linear' (for colorbar units)
    Nphi_bins=180,
    cmap="inferno"
    )  

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


def init_mra_from_hit_trajectory(
    hit,
    baseline_fn=None,
    mra_kwargs=None,
    cells_per_level=100,
    profile=True,
    maxspatial_pc=100.0,
    #usegrid=False,
):
    """
    Initialize a MultiResolutionArray and seed scalar density deltas from a successful
    excursion 'hit' (as returned by draw_trajectory_until_upcrossing).

    Parameters
    ----------
    hit : dict
        Output of draw_trajectory_until_upcrossing(...). Must contain:
          - 'traj': the excursion trajectory object with attributes:
                * delta : array-like, shape (Nr_grid,)
                * grid  : the trajectory grid with attributes:
                      - rlevels : array-like (cm), the scales used in the trajectory
        (We do NOT require grid/cloud/time here yet; just the 1D trajectory & scales.)
    baseline_fn : callable or None
        Optional baseline function for MultiResolutionArray (rho0, v0, Lcut). If None,
        MRA falls back to its internal defaults (rho0 from trajectory_grid, zero bulk v).
    mra_kwargs : dict or None
        Extra keyword args for MultiResolutionArray(...) such as snapshot_dir, imaxcoll, etc.
        If None, we infer sensible defaults from the trajectory grid scales.
    cells_per_level : int
        Number of unstructured cells per spatial level.
    profile : bool
        Print a brief apply/report summary.

    Returns
    -------
    mra : MultiResolutionArray
        Fully initialized and seeded with scalar deltas from the hit trajectory.
    info : dict
        Small report with counts and scale summaries.
    """
    # --- Pull trajectory + scales from the hit dict ---
    if "traj" not in hit:
        raise ValueError("hit dict must contain 'traj'.")
    traj = hit["traj"]

    # Trajectory must expose delta array and its grid with rlevels (in cm)
    if not hasattr(traj, "delta"):
        raise ValueError("hit['traj'] must have attribute 'delta'.")
    if not hasattr(traj, "grid") or not hasattr(traj.grid, "rlevels"):
        raise ValueError("hit['traj'] must have 'grid' with 'rlevels' (cm).")

    grid = traj.grid
    rlevels_cm = np.asarray(grid.rlevels, dtype=float)
    Ddeltas_1d  = np.asarray(traj.Ddelta, dtype=float)

    if rlevels_cm.shape[0] != Ddeltas_1d.shape[0]:
        raise ValueError("Length mismatch: traj.grid.rlevels vs traj.delta.")

    # --- Choose MRA scale extents from the trajectory grid ---
    # Convert to pc for MRA API (its __init__ multiplies by pc2cm internally)
    rmin_pc     = max(1e-6, float(np.min(rlevels_cm)) / pc2cm)     # avoid zero

    rspatial_pc = min(hit['R_cross'] * 100.0, maxspatial_pc)  # max 100x R_cross or user-specified
    #rmin_pc = hit['R_cross'] /10.0

    rho_cross = hit['rho_pre_cross']  # pre-collapse density at R_cross

    print(f"R_cross = {hit['R_cross']/pc2cm:.3f} pc, rho_pre_cross = {rho_cross:.3e} g/cm^3")
    
    # Allow some headroom above rspatial as "super" region
    

    # Assemble kwargs for MultiResolutionArray
    mra_kwargs = dict(mra_kwargs or {})
    mra_kwargs.setdefault("grid", grid)  # no snapshots by default
    mra_kwargs.setdefault("rspatial", rspatial_pc)
    mra_kwargs.setdefault("cells_per_level", int(cells_per_level))
    mra_kwargs.setdefault("baseline_fn", baseline_fn)

    # --- Build the MRA (this creates unstructured levels & super levels) ---
    mra = gsau.MultiResolutionArray(**mra_kwargs)

    # --- Build scalar overrides from the 1D trajectory ---
    # Place all seeds at the origin (centered star/high-res element); classification
    # (spatial vs super) is handled by MRA's scale matcher.
    N = rlevels_cm.size
    origins = np.zeros((N, 3), dtype=float)
    scales  = rlevels_cm.copy()
    Ddeltas  = Ddeltas_1d.copy()

    # We assume you've added the convenience wrapper `set_deltas` and the underlying
    # `apply_delta_overrides` per our earlier patch.
    if not hasattr(mra, "apply_delta_overrides"):
        raise RuntimeError("MultiResolutionArray missing 'apply_delta_overrides'. "
                           "Please add the override API we discussed earlier.")

    overrides = [
        {"pos": origins[i], "scale": float(scales[i]), "Ddelta": float(Ddeltas[i]), "kind": "scalar"}
        for i in range(N)
    ]

    """print(f"Applying {len(overrides)} Ddelta overrides to MRA (scales in pc):")
    for i, ov in enumerate(overrides):
        print(f"  {i:3d}: pos={ov['pos']}, scale={ov['scale']/pc2cm:.3f} pc, Ddelta={ov['Ddelta']:.3f}")"""
    
    mra.apply_delta_overrides(overrides, default_kind="scalar", profile=profile)

    # Prepare a small summary
    info = {
        "num_traj_levels": int(N),
        "rspatial_pc": float(rspatial_pc)
    }
    print(info)

    return mra, info




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
    baseline_fn, data = gcr.build_cloud_baseline_fn(
        rmin = 0.001*pc2cm,  # 0.01 pc minimum radius
        target_radius_cm=target_radius_cm,
        drfact=0.8,
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
            seed=base_seed, progress=progress, baseline_fn=baseline_fn
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

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

def _centers_to_edges_geom(r_c):
    """Recover geometric bin edges from geometric centers."""
    r_c = np.asarray(r_c, dtype=float)
    if r_c.size == 1:
        f = 10.0**0.5
        return np.array([r_c[0]/f, r_c[0]*f], dtype=float)
    edges = np.empty(r_c.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(r_c[:-1] * r_c[1:])   # geometric midpoints
    edges[0]    = r_c[0]**2 / edges[1]
    edges[-1]   = r_c[-1]**2 / edges[-2]
    return edges

def plot_polar_equatorial_slice_from_mra(
    mra,
    hit,
    Nr=128,
    nside=32,
    theta_tol_deg=5.0,
    coords_path=None,
    log_scale="log10",        # 'log10' | 'ln' | 'linear' (for colorbar units)
    Nphi_bins=360,
    vmin=-25.0,
    vmax=-19.0,
    cmap="inferno",
    title=None,
    ax=None,
    show=True,
):
    """
    Evaluate ln(rho) on a log-r + HEALPix grid and plot an *equatorial* (theta ≈ π/2)
    polar pcolormesh with log radial axis. Overplots a circle at hit['R_cross'].

    Parameters
    ----------
    mra : MultiResolutionArray
    hit : dict, must contain 'R_cross' in cm
    Nr : int, radial shells
    nside : int, HEALPix nside
    theta_tol_deg : float, half-thickness of equatorial band
    coords_path : str or None, reuse coords file if available
    log_scale : str, colorbar quantity ('log10' of rho, 'ln' of rho, or linear rho)
    Nphi_bins : int, number of azimuth bins around 0..2π
    vmin, vmax : float or None, color limits in chosen log_scale
    cmap : str, matplotlib colormap name
    title : str or None
    ax : matplotlib polar axes or None
    show : bool, call plt.show()

    Returns
    -------
    fig, ax, out
       out is dict with keys: {'r_edges', 'phi_edges', 'Z', 'units'}
    """

    R_cross = float(hit["R_cross"])

    plt.figure(figsize=(8, 6))
    plt.plot(hit['traj'].grid.rlevels/pc2cm, hit['traj'].grid.rho0*np.exp(hit['traj'].delta+hit['traj'].grid.mu_lnrho), marker='o', ls='None')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("R (cm)")
    plt.ylabel("$\\rho$")

    

    # 1) Evaluate on equal-area spherical grid
    lnrho, v = None, None
    lnrho, v, r_c, theta, phi, U = mra.evaluate_to_equal_area(
        Nr=Nr, nside=nside, coords_path=coords_path, return_coords=True
    )  # lnrho shape: (Nr, npix)


    seeds = ca.run_pipeline(r=r_c, theta=theta, phi=phi, lnrho=lnrho)
    
    labels_local, cat_local =  ca.find_bound_region_within_sphere(r=r_c, theta=theta, phi=phi, lnrho=lnrho, v=v,   R_search_cm=R_cross)
    X = ca.build_positions_cartesian(r_c, theta, phi)
    pjm.plot_jeans_regions(X, labels_local, s=1.0, alpha=0.6, elev=20, azim=45,
                       title=None, max_points_per_region=None)

    print('R_c:', np.unique(r_c/pc2cm))

    # 2) Select equatorial band
    theta = np.asarray(theta)
    phi   = np.asarray(phi)
    eq_mask = np.abs(theta - 0.5*np.pi) <= np.deg2rad(theta_tol_deg)
    if not np.any(eq_mask):
        raise RuntimeError("No HEALPix pixels fell within the equatorial tolerance; "
                           "try increasing theta_tol_deg or nside.")
    
    plot_size_linewidth(X, v)
    plot_equatorial_polar_velocity( r_c, theta, phi, v, Nphi_bins=Nphi_bins, theta_tol_deg=theta_tol_deg, R_cross=R_cross)

    # 3) Bin by azimuth (phi) for each radius shell
    phi_sel = phi[eq_mask]
    # Ensure [0, 2π)
    phi_sel = np.mod(phi_sel, 2.0*np.pi)
    phi_edges = np.linspace(0.0, 2.0*np.pi, Nphi_bins + 1)
    phi_bin_idx = np.digitize(phi_sel, phi_edges) - 1
    phi_bin_idx = np.clip(phi_bin_idx, 0, Nphi_bins-1)

    Z = np.full((Nr, Nphi_bins), np.nan, dtype=float)  # will store chosen display quantity
    for ir in range(Nr):
        vals = lnrho[ir, eq_mask]  # ln rho at this radius for equatorial band
        # aggregate by bin (mean)
        for b in range(Nphi_bins):
            m = (phi_bin_idx == b)
            if np.any(m):
                mu = np.nanmean(vals[m])
                Z[ir, b] = mu

    # 4) Choose display units
    if log_scale == "log10":
        units = "$\log \\rho$ [g cm$^{-3}$]"
        Z_disp = Z / np.log(10.0)  # ln→log10
    elif log_scale == "ln":
        units = "$\ln \\rho$ [g cm$^{-3}$]"
        Z_disp = Z
    elif log_scale == "linear":
        units = "$\\rho$ [g cm$^{-3}$]"
        Z_disp = np.exp(Z, dtype=float)
    else:
        raise ValueError("log_scale must be 'log10', 'ln', or 'linear'.")

    # 5) Build r-edges for pcolormesh and set up polar axes
    r_edges = _centers_to_edges_geom(r_c)
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = plt.subplot(111, projection="polar")
    else:
        fig = ax.figure

    # Polar mesh wants (len(r_edges)-1, len(phi_edges)-1)

    PHI, R = np.meshgrid(phi_edges, r_edges)
    pc = ax.pcolormesh(PHI, R/pc2cm, Z_disp, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # 6) Logarithmic radial axis
    try:
        ax.set_rscale("log")  # matplotlib >= 3.1
    except Exception:
        # Fallback: nothing; the plot still shows, just not with true log tick spacing
        pass
    ax.set_theta_zero_location("E")  # 0 at +x
    ax.set_theta_direction(-1)       # clockwise like usual polar plots

    ph = np.linspace(0.0, 2.0*np.pi, 1024)
    ax.plot(ph, np.full_like(ph, R_cross/pc2cm), lw=2.0, color="limegreen", alpha=0.9, label="R_cross")

    # 8) Aesthetics
    cbar = fig.colorbar(pc, ax=ax, pad=0.08, shrink=0.9)
    cbar.set_label(units)
    ax.tick_params(axis="y", colors="green")  
    ax.set_title(title or f"Equatorial polar slice (±{theta_tol_deg:.1f}°), nside={nside}, Nr={Nr}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.15))

    if show:
        plt.show()

    return fig, ax, {"r_edges": r_edges, "phi_edges": phi_edges, "Z": Z_disp, "units": units}

import numpy as np

def build_radial_shell_labels(
    X,
    n_shells=10,
    center="centroid",      # "centroid", "origin", or a 3-vector
    r_edges=None,           # if given, overrides n_shells/log spacing
    logspace=True
):
    """
    Return integer labels for concentric radial shells.

    X : (N,3) positions [cm]
    n_shells : number of shells (ignored if r_edges is provided)
    center : "centroid" | "origin" | array-like(3)
    r_edges : optional array of shell edges [cm], length = n_shells+1
    logspace : use geometric spacing if True, else linear

    Returns
    -------
    labels : (N,) ints in [0, n_shells-1]
    r : (N,) radii from the chosen center [cm]
    edges : (n_shells+1,) shell edges [cm]
    """
    X = np.asarray(X, float)
    if isinstance(center, str):
        if center == "centroid":
            c = X.mean(axis=0)
        elif center == "origin":
            c = np.zeros(3, float)
        else:
            raise ValueError("center must be 'centroid', 'origin', or a 3-vector.")
    else:
        c = np.asarray(center, float)
        if c.shape != (3,):
            raise ValueError("center 3-vector must have shape (3,)")

    r = np.linalg.norm(X - c[None, :], axis=1)

    if r_edges is None:
        rmax = float(np.nanmax(r))
        if rmax <= 0:
            raise ValueError("All radii are zero/nonpositive; cannot build shells.")
        # Avoid log(0): make the first edge exactly 0 so points at r=0 fall into the first bin,
        # and start geometric spacing just above 0.
        if logspace:
            # smallest positive radius as inner >0 reference; fallback to rmax/1e6
            pos = r[r > 0]
            rmin_pos = float(pos.min()) if pos.size else rmax/1e6
            edges = np.geomspace(rmin_pos, rmax, n_shells)   # length n_shells
            edges = np.concatenate([[0.0], edges])           # length n_shells+1, first edge at 0
        else:
            edges = np.linspace(0.0, rmax, n_shells+1)
    else:
        edges = np.asarray(r_edges, float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("r_edges must be a 1D array of length >= 2.")

    # Bin: labels in [0, n_shells-1]; clamp rightmost into last bin
    labels = np.digitize(r, edges, right=False) - 1
    labels = np.clip(labels, 0, edges.size - 2)
    return labels.astype(int), r, edges


def plot_size_linewidth(
    X, v,
    *,
    labels=None,
    pc2cm=3.085677581e18,
    v0=None,                         # optional 3-vector (cm/s) to subtract globally, e.g. origin velocity
    subtract_region_mean=True,       # subtract each region's mean velocity before dispersion
    min_points=50,                   # skip tiny regions
    marker='o', alpha=0.7, ms=6,
    fit=True, fit_color=None,        # log-log OLS fit in log10 space
    title="Size–line width relation",
    show=True,
    n_shells=10,
    shell_center="centroid",     # "centroid" | "origin" | 3-vector
    shell_edges=None,            # optional custom edges [cm]
    shell_logspace=True,
):
    """
    Compute and plot σ_1D vs effective size for labeled regions.

    Parameters
    ----------
    X : (N,3) positions in cm
    v : (Nr,npix,3) or (N,3) velocities in cm/s
    labels : (N,) int labels (negative => ignore)
    v0 : None or (3,) global velocity to subtract (cm/s)
    subtract_region_mean : bool, subtract <v> per region
    min_points : int, minimum members per region to include
    """


    # Optional global subtraction (e.g., 'origin' velocity)
    if v0 is not None:
        v = v - np.asarray(v0)[None, :]
    else:
        v0 = np.nanmean(v[0, :, :], axis=0)  # shape (3,)
        print(f"Zero velocity: {v0}")
        v = v - v0[None, None, :]   # broadcast subtraction

    # Flatten v if needed
    if v.ndim == 3:
        v = v.reshape(-1, 3)

    if labels is None:
        labels, r, used_edges = build_radial_shell_labels(
            X, n_shells=n_shells, center=shell_center,
            r_edges=shell_edges, logspace=shell_logspace
        )
    else:
        labels = np.asarray(labels)
        if labels.shape[0] != X.shape[0]:
            raise ValueError(f"labels length {labels.shape[0]} != X length {X.shape[0]}")
    
    labels = np.asarray(labels)
    m_keep = labels >= 0
    X = np.asarray(X)[m_keep]
    v = np.asarray(v)[m_keep]
    labs = labels[m_keep]

    if X.size == 0:
        raise RuntimeError("No labeled points to analyze (labels >= 0).")

    
    uniq = np.unique(labs)
    Reff_pc, sigma_kms, counts, lab_ids = [], [], [], []

    for L in uniq:
        idx = (labs == L)
        if idx.sum() < min_points:
            continue

        XL = X[idx]
        vL = v[idx]

        # Effective radius from 3D RMS radius (then ×√(5/3))
        xcent = XL.mean(axis=0)
        r2 = np.sum((XL - xcent)**2, axis=1)
        r_rms = np.sqrt(np.mean(r2))
        R_eff = np.sqrt(5.0/3.0) * r_rms           # cm
        R_eff_pc = R_eff / pc2cm

        # Velocity dispersion: subtract region mean if requested
        if subtract_region_mean:
            vcent = vL.mean(axis=0)
            vL = vL - vcent[None, :]

        # 1D dispersion assuming isotropy: average of component variances
        sig2_xyz = np.var(vL, axis=0, ddof=1)      # cm^2/s^2 per component
        sigma_1d = np.sqrt(np.mean(sig2_xyz)) / 1e5  # km/s

        Reff_pc.append(R_eff_pc)
        sigma_kms.append(sigma_1d)
        counts.append(idx.sum())
        lab_ids.append(L)

    Reff_pc = np.asarray(Reff_pc)
    sigma_kms = np.asarray(sigma_kms)

    if Reff_pc.size == 0:
        raise RuntimeError("All regions were filtered out (min_points too high?).")

    print(Reff_pc, sigma_kms)
    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.loglog(Reff_pc, sigma_kms, marker, ms=ms, alpha=alpha, linestyle='None', label="Regions")

    # Optional fit (log10–log10)
    fit_line = None
    if fit and np.isfinite(Reff_pc).all() and np.isfinite(sigma_kms).all():
        m, b = np.polyfit(np.log10(Reff_pc), np.log10(sigma_kms), 1)
        xx = np.logspace(np.log10(Reff_pc.min()), np.log10(Reff_pc.max()), 200)
        yy = 10**b * xx**m
        ax.loglog(xx, yy, '--', lw=2, color=fit_color, label=fr"Fit: $\sigma \propto R^{m:.2f}$".format(m=m))
        fit_line = (m, b)

    ax.set_xlabel("Effective size $R_\\mathrm{eff}$ [pc]")
    ax.set_ylabel("Line width $\\sigma_{1\\,\\mathrm{D}}$ [km s$^{-1}$]")
    ax.set_title(title)
    ax.grid(True, which='both', ls=':')
    ax.legend()

    # Return raw data too
    info = {
        "R_eff_pc": Reff_pc,
        "sigma_1d_kms": sigma_kms,
        "counts": np.asarray(counts),
        "labels": np.asarray(lab_ids),
        "fit_mb" : fit_line,  # (m, b in log10 space) or None
    }

    if show:
        plt.show()

    return fig, ax, info


def plot_equatorial_polar_velocity(
    r_c, theta, phi, v,
    *,
    Nphi_bins=72,
    R_cross = 1.0*pc2cm,
    theta_tol_deg=5.0,
    cmap="coolwarm",
    vmin=None, vmax=None,     # scalar or dict per component, otherwise auto symmetric
    title=None,
    show=True,
):
    """
    Make equatorial polar projections (±theta_tol_deg) of vx, vy, vz
    from vector field v (cm/s) sampled on equal-area spherical grid.

    Parameters
    ----------
    r_c : array, shape (Nr,)
        Radius centers [cm]
    theta, phi : arrays, shape (npix,)
        HEALPix polar/azimuth angles [rad] for a *single* shell; repeated for all shells.
    v : array, shape (Nr, npix, 3)
        Velocity in cm/s with components (vx, vy, vz) in the last dimension.
    Nphi_bins : int
        Number of azimuth bins (columns).
    theta_tol_deg : float
        Half-width of equatorial band in degrees.
    R_cross : float
        Reference radius to overlay [cm].
    pc2cm : float
        Conversion factor.
    cmap : str
        Matplotlib colormap.
    vmin, vmax : None | float | dict
        If None: auto symmetric per-component.
        If float: same for all components.
        If dict: keys "vx","vy","vz" with floats.

    Returns
    -------
    fig, axes, info
        axes is dict {"vx": ax_x, "vy": ax_y, "vz": ax_z}
        info has r_edges, phi_edges, Z_kms (Nr, Nphi_bins, 3), units
    """

    Nr = len(r_c)
    theta = np.asarray(theta)
    phi   = np.asarray(phi)

    # 1) Select equatorial band
    eq_mask = np.abs(theta - 0.5*np.pi) <= np.deg2rad(theta_tol_deg)
    if not np.any(eq_mask):
        raise RuntimeError("No HEALPix pixels fell within the equatorial tolerance; "
                           "increase theta_tol_deg or nside.")

    # 2) Bin by azimuth (phi) for each radius shell
    phi_sel = np.mod(phi[eq_mask], 2.0*np.pi)   # [0, 2π)
    phi_edges = np.linspace(0.0, 2.0*np.pi, Nphi_bins + 1)
    phi_bin_idx = np.digitize(phi_sel, phi_edges) - 1
    phi_bin_idx = np.clip(phi_bin_idx, 0, Nphi_bins-1)

    # Normalize to km/s
    v_kms = v / 1e5  # shape (Nr, npix, 3)

    # 3) Aggregate (mean over pixels in bin) for each component
    Z = np.full((Nr, Nphi_bins, 3), np.nan, dtype=float)
    for ir in range(Nr):
        vals_x = v_kms[ir, eq_mask, 0]
        vals_y = v_kms[ir, eq_mask, 1]
        vals_z = v_kms[ir, eq_mask, 2]
        for b in range(Nphi_bins):
            m = (phi_bin_idx == b)
            if np.any(m):
                Z[ir, b, 0] = np.nanmean(vals_x[m])
                Z[ir, b, 1] = np.nanmean(vals_y[m])
                Z[ir, b, 2] = np.nanmean(vals_z[m])

    # 4) Build r-edges for pcolormesh
    def _centers_to_edges_geom(r):
        """Geometric edges from monotonic centers."""
        r = np.asarray(r, dtype=float)
        if np.any(r <= 0):
            raise ValueError("r must be positive for geometric edges.")
        # interior edges are geometric means
        re = np.sqrt(r[:-1] * r[1:])
        # extrapolate first/last edges
        r0 = r[0]**2 / re[0]
        rN = r[-1]**2 / re[-1]
        edges = np.concatenate([[r0], re, [rN]])
        return edges

    r_edges = _centers_to_edges_geom(r_c)

    v0 = np.nanmean(v[0, :, :], axis=0)  # shape (3,)
    print(f"Zero velocity: {v0}")
    v = v - v0[None, None, :]   # broadcast subtraction

    plt.figure()
    plt.hist(np.linalg.norm(v.reshape(-1,3), axis=-1))
    plt.show()

    # 5) Set up figure with 3 polar subplots
    comp_names = ["vx", "vy", "vz"]
    units = "km s$^{-1}$"
    fig = plt.figure(figsize=(18, 6.5))
    axes = {}

    # Helper to get vmin/vmax per component
    def _get_lims(comp, Z2d):
        # User-specified?
        def pick(val, key):
            if isinstance(val, dict): return val.get(key, None)
            return val
        vmin_c = pick(vmin, comp)
        vmax_c = pick(vmax, comp)
        if (vmin_c is None) or (vmax_c is None):
            finite = np.isfinite(Z2d)
            if not np.any(finite):
                return -1.0, 1.0
            amax = np.nanmax(np.abs(Z2d[finite]))
            if amax == 0 or not np.isfinite(amax):
                amax = 1.0
            vmin_c = -amax if vmin_c is None else vmin_c
            vmax_c =  amax if vmax_c is None else vmax_c
        return vmin_c, vmax_c

    # Common grids for pcolormesh
    PHI, R = np.meshgrid(phi_edges, r_edges)

    for i, comp in enumerate(comp_names):
        ax = plt.subplot(1, 3, i+1, projection="polar")
        axes[comp] = ax

        Zc = Z[:, :, i]  # (Nr, Nphi_bins)
        vmin_c, vmax_c = _get_lims(comp, Zc)

        pc = ax.pcolormesh(PHI, R/pc2cm, Zc, shading="auto", cmap=cmap, vmin=vmin_c, vmax=vmax_c)

        # Log radial axis if available
        try:
            ax.set_rscale("log")
        except Exception:
            pass
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)

        # Reference radius
        ph = np.linspace(0.0, 2.0*np.pi, 1024)
        ax.plot(ph, np.full_like(ph, R_cross/pc2cm), lw=2.0, alpha=0.9, label="R_cross")

        cbar = fig.colorbar(pc, ax=ax, pad=0.08, shrink=0.9)
        cbar.set_label(f"{comp} [{units}]")
        ax.tick_params(axis="y")
        ax.set_title(f"{comp} (±{theta_tol_deg:.1f}°)")

        ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1.12))

    suptitle = title or f"Equatorial polar velocity (nside=auto, Nr={len(r_c)})"
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes, {"r_edges": r_edges, "phi_edges": phi_edges, "Z_kms": Z, "units": units}

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
         target_radius_cm=R0_cm,
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