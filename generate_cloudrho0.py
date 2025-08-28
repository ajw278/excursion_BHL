# --- cloud_baseline_from_collapse.py ---
import numpy as np
import excursion as exc          # your excursion module
import cloud as cl               # where bound_clump lives
from consts_defaults import *    # provides pc2cm, Myr2s, etc.
import starmass_sampler as sms  # your starmass_sampler module
import numpy as np
import matplotlib.pyplot as plt
from consts_defaults import Myr2s

def _sample_baseline_on(times_sec, baseline_fn):
    """Evaluate baseline_fn on a vector of times."""
    rho0 = np.empty_like(times_sec, dtype=float)
    Lcut = np.empty_like(times_sec, dtype=float)
    for i, ts in enumerate(times_sec):
        b = baseline_fn(float(ts))
        rho0[i] = float(b["rho0"])
        Lcut[i] = float(b["Lcut"])
    return rho0, Lcut

def plot_cloud_and_baseline(info, baseline_fn, show_shifted=True, save=None):
    """
    Plot cloud evolution (rho_med, R) and the star's baseline (rho0, Lcut).
    - Baseline curves are shown only during [t_star, t_disp] (masked elsewhere).
    - If show_shifted=True, also show a compact shifted-time view to verify shapes match.
    """
    # Unpack
    t_sec  = np.asarray(info["times_sec"], dtype=float)
    t_myr  = t_sec / Myr2s
    rho    = np.asarray(info["rho_med"], dtype=float)
    R_cm   = np.asarray(info["R_cm"], dtype=float)
    t_star = float(info["t_star"])
    t_disp = float(info["t_disp"])
    grid   = info["grid"]

    # Sample baseline on same time grid
    rho0_all, Lcut_all = _sample_baseline_on(t_sec, baseline_fn)

    # Masks for the star-active window
    active = (t_sec >= t_star) & (t_sec <= t_disp)

    # Prepare masked versions for clean plotting
    rho0_active = np.where(active, rho0_all, np.nan)
    Lcut_active = np.where(active, Lcut_all, np.nan)

    # --- Figure layout ---
    nrows = 2 if show_shifted else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, 4.6*nrows), sharex=False)
    if nrows == 1:
        ax_rho, ax_R = axes
    else:
        (ax_rho, ax_R), (ax_rho_shift, ax_R_shift) = axes

    # ===== Panel 1: Density vs absolute time =====
    ax_rho.plot(t_myr, rho, lw=2.0, label=r"cloud $\rho_{\rm med}(t)$")
    ax_rho.plot(t_myr, rho0_active, lw=2.0, ls="--", label=r"baseline $\rho_0(t)$ (active)")
    # Optional thin line to show default baseline before t_star
    ax_rho.hlines(grid.rho0, xmin=t_myr[0], xmax=t_myr[-1], lw=1.0, ls=":", label=r"default $\rho_0$ (inactive)")

    # Mark t_star and t_disp
    ax_rho.axvline(t_star / Myr2s, color='k', lw=1.0, ls=':', alpha=0.7)
    ax_rho.axvline(t_disp / Myr2s, color='k', lw=1.0, ls=':', alpha=0.7)
    ax_rho.text(t_star / Myr2s, ax_rho.get_ylim()[1], r"$t_\star$", va="top", ha="right", fontsize=9)
    ax_rho.text(t_disp / Myr2s, ax_rho.get_ylim()[1], r"$t_{\rm disp}$", va="top", ha="left", fontsize=9)

    ax_rho.set_yscale("log")
    ax_rho.set_xlabel("time [Myr]")
    ax_rho.set_ylabel(r"density $\rho$ [cgs]")
    ax_rho.legend(loc="best")
    ax_rho.grid(True, ls=":", alpha=0.4)

    # ===== Panel 2: Radius/Scale vs absolute time =====
    ax_R.plot(t_myr, R_cm, lw=2.0, label=r"cloud $R(t)$")
    ax_R.plot(t_myr, Lcut_active, lw=2.0, ls="--", label=r"baseline $L_{\rm cut}(t)$ (active)")

    ax_R.axvline(t_star / Myr2s, color='k', lw=1.0, ls=':', alpha=0.7)
    ax_R.axvline(t_disp / Myr2s, color='k', lw=1.0, ls=':', alpha=0.7)
    ax_R.set_xlabel("time [Myr]")
    ax_R.set_ylabel(r"scale / radius [cm]")
    ax_R.legend(loc="best")
    ax_R.grid(True, ls=":", alpha=0.4)

    if show_shifted:
        # Build shifted time axes:
        #  - cloud: tau = t - t_form (t_form≈0 here)
        #  - baseline: tau_star = t - t_star, and plot only for active window
        tau_cloud_myr = t_myr  # since t_form == 0 in your use
        tau_star_myr  = (t_sec[active] - t_star) / Myr2s

        # Shifted density
        ax_rho_shift.plot(tau_cloud_myr, rho, lw=2.0, label=r"cloud $\rho_{\rm med}(\tau)$")
        ax_rho_shift.plot(tau_star_myr, rho0_all[active], lw=2.0, ls="--", label=r"baseline $\rho_0(\tau)$")
        ax_rho_shift.set_yscale("log")
        ax_rho_shift.set_xlabel(r"shifted time $\tau$ [Myr]")
        ax_rho_shift.set_ylabel(r"density $\rho$ [cgs]")
        ax_rho_shift.legend(loc="best")
        ax_rho_shift.grid(True, ls=":", alpha=0.4)

        # Shifted radius/scale
        ax_R_shift.plot(tau_cloud_myr, R_cm, lw=2.0, label=r"cloud $R(\tau)$")
        ax_R_shift.plot(tau_star_myr, Lcut_all[active], lw=2.0, ls="--", label=r"baseline $L_{\rm cut}(\tau)$")
        ax_R_shift.set_xlabel(r"shifted time $\tau$ [Myr]")
        ax_R_shift.set_ylabel(r"scale / radius [cm]")
        ax_R_shift.legend(loc="best")
        ax_R_shift.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig




def run_cloud_evolution(grid, icol, tform_seconds=0.0, dt_myr=0.01, tmax_myr=10.0,
                        seed=None):
    """
    Form a clump at (icol) with r=v=0 and evolve until dispersal or tmax.
    Returns a dict with time series arrays (in seconds) and some scalars.
    """
    rng = np.random.default_rng(seed)

    # Form the clump at the requested grid level, zero pos/vel, tform
    cloud = cl.bound_clump(eps_core=1.0)
    position = np.zeros(3, dtype=float)
    velocity = np.zeros(3, dtype=float)
    cloud.form_nontraj(icol=icol, grid=grid, position=position, velocity=velocity, tform=tform_seconds)

    # Evolve
    dt = float(dt_myr) * Myr2s
    tmax = float(tmax_myr) * Myr2s
    t = tform_seconds
    dispersed_at = None

    # Ensure we have at least one sample at tform
    cloud.evolve(t)

    while t < tmax:
        t_next = t + dt
        cloud.evolve(t_next)
        t = t_next
        if cloud.dispersed:
            dispersed_at = t
            break

    # Collect arrays (already tracked inside the cloud object)
    times_sec = np.asarray(cloud.tclouds, dtype=float)
    R_cm      = np.asarray(cloud.Rclouds, dtype=float)
    rho_med   = np.asarray(cloud.rho_meds, dtype=float)
    sfr       = np.asarray(cloud.sfrs, dtype=float)

    # If not dispersed, set end at last time
    if dispersed_at is None:
        dispersed_at = times_sec[-1]

    return dict(
        cloud=cloud,
        times_sec=times_sec,
        R_cm=R_cm,
        rho_med=rho_med,
        sfr=sfr,
        tform=tform_seconds,
        tdisp=dispersed_at,
        rng=rng
    )


def inverse_cdf_sample_time(times_sec, rate, rng=None):
    """
    Draw a time with probability density ∝ rate(t) over the provided time grid.
    We use a trapezoidal cumulative *area* and invert linearly.
    If the integral is zero, return the start time.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.asarray(times_sec, dtype=float)
    r = np.maximum(0.0, np.asarray(rate, dtype=float))  # no negative weights

    if t.size < 2:
        return float(t[0])

    # Trapezoidal cumulative integral
    # area[i] = integral from t[0] to t[i]
    dt = np.diff(t)
    mid = 0.5 * (r[1:] + r[:-1])
    area = np.concatenate([[0.0], np.cumsum(mid * dt)])
    total = area[-1]
    
    if total <= 0.0 or not np.isfinite(total):
        return float(t[0])

    u = rng.random()
    target = u * total

    # Invert: find i s.t. area[i] <= target <= area[i+1]
    j = int(np.searchsorted(area, target, side="right") - 1)
    j = np.clip(j, 0, len(t) - 2)

    interval_area = area[j+1] - area[j]
    if interval_area <= 0:
        return float(t[j])
    frac = (target - area[j]) / interval_area

    return float(t[j] + frac * (t[j+1] - t[j]))


def make_baseline_fn(times_sec, rho_med, R_cm, t_star, t_end, grid):
    """
    Build baseline_fn(t_seconds) -> {rho0, v0, Lcut}:
      - for t in [t_star, t_end]: rho0(t) = interp(rho_med), Lcut(t) = interp(R_cm), v0 = 0
      - otherwise: rho0 = grid.rho0, Lcut = +inf, v0 = 0
    """
    t = np.asarray(times_sec, dtype=float)
    rho = np.asarray(rho_med, dtype=float)
    R   = np.asarray(R_cm, dtype=float)

    # Ensure monotonic increasing time for interp
    idx = np.argsort(t)
    t, rho, R = t[idx], rho[idx], R[idx]


    #Hack: make sure 
    if R[1]>0.0 and np.isfinite(R[1]):
        if R[0]==0.0 or ~np.isfinite(R[0]):
            rho[0] = rho[1]
            R[0] = R[1]

    def baseline_fn(t_seconds):
        ts = float(t_seconds)
        if (ts >= t[0]) and (ts <= t_end):
            rho_t = float(np.interp(ts, t, rho))
            R_t   = float(np.interp(ts, t, R))
            return {"rho0": rho_t, "v0": np.zeros(3, dtype=float), "Lcut": R_t*2.}
        else:
            return {"rho0": float(grid.rho0), "v0": np.zeros(3, dtype=float), "Lcut": np.inf}

    return baseline_fn


def build_cloud_baseline_fn(target_radius_cm=None,
                            rmin=0.01*pc2cm,
                            rmax=10.0*h_*pc2cm,
                            drfact=0.95,
                            dt_myr=0.01,
                            tmax_myr=10.0,
                            seed=None):
    """
    High-level orchestrator:
      1) Build trajectory_grid
      2) Pick icol from target_radius_cm (or middle level)
      3) Form/evolve the clump from tform=0 with v=0, r=0
      4) Sample SF time ~ SFR(t)
      5) Build and return baseline_fn + info dict
    """
    # 1) grid
    grid = exc.trajectory_grid(rmin=rmin, rmax=rmax, drfact=drfact)

    # 2) select level index to represent the desired size scale
    icol = sms.pick_i_for_radius(grid, target_radius_cm)

    # 3) evolve the cloud
    evo = run_cloud_evolution(grid, icol=icol, tform_seconds=0.0,
                              dt_myr=dt_myr, tmax_myr=tmax_myr, seed=seed)

    # 4) draw the star-formation (formation) time weighted by SFR(t)
    t_star = inverse_cdf_sample_time(evo["times_sec"], evo["sfr"], rng=evo["rng"])
    t_end  = float(evo["tdisp"])   # until dispersal

    # 5) build baseline function
    baseline_fn = make_baseline_fn(evo["times_sec"], evo["rho_med"], evo["R_cm"], t_star, t_end, grid)


    # Return both the function and info for diagnostics
    info = dict(
        grid=grid,
        icol=icol,
        times_sec=evo["times_sec"],
        rho_med=evo["rho_med"],
        R_cm=evo["R_cm"],
        sfr=evo["sfr"],
        cloud = evo["cloud"],
        t_star=t_star,
        t_disp=t_end
    )
    return baseline_fn, info


# ---------- example usage ----------
if __name__ == "__main__":
    # Example: pick ~1 pc radius if available, else nearest
    target_R = 1.0 * pc2cm

    baseline_fn, info = build_cloud_baseline_fn(
        target_radius_cm=target_R,
        rmin=0.01*pc2cm,
        rmax=10.0*h_*pc2cm,
        drfact=0.95,
        dt_myr=0.01,
        tmax_myr=10.0,
        seed=42
    )

    plot_cloud_and_baseline(info, baseline_fn, show_shifted=True, save="cloud_vs_baseline.png")

    # Plug into your MultiResolutionArray (requires your MRA to support baseline_fn)
    # from gen_sphere_adaptive_unstructured import MultiResolutionArray
    # mra = MultiResolutionArray(rmax=200.0, rspatial=0.1, rmin=1e-5, dr=0.5, cells_per_level=500,
    #                            baseline_fn=baseline_fn)
    # mra.evolve(Tend=5.0, fraction_of_tau=0.1, dt_snap=0.05)

    # Quick sanity print
    print("Picked grid level index icol =", info["icol"])
    print("Star-formation time [Myr]    =", info["t_star"]/Myr2s)
    print("Dispersal time [Myr]         =", info["t_disp"]/Myr2s)
