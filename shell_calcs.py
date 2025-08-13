import os
import numpy as np
import matplotlib.pyplot as plt

def get_or_compute_v_ref(
    snapshot_dir: str,
    coords_file: str = "sph_coords_equalarea.npz",
    lnrho_pattern: str = "snapshot_{:04d}_lnrho_eq.npy",
    v_pattern: str = "snapshot_{:04d}_v_eq.npy",
    v_ref_path: str = "v_ref.npy",
    mode: str | None = None,              # None | "first_snapshot_inner"
    init_snapshot_idx: int = 0,           # which snapshot to use for the reference
    shell_index: int = 0                  # which shell (0 = smallest r)
) -> np.ndarray:
    """
    Returns a 3-vector star-frame reference velocity v_ref.

    If v_ref_path exists -> loads and returns it.
    If mode == "first_snapshot_inner" -> computes from given (init_snapshot_idx, shell_index),
        then saves to v_ref_path and returns it.
    If mode is None and file doesn't exist -> returns zeros (no correction).
    """
    # absolute path for the cache file
    vref_abs = v_ref_path if os.path.isabs(v_ref_path) else os.path.join(snapshot_dir, v_ref_path)

    # 1) If already saved, reuse
    if os.path.exists(vref_abs):
        v_ref = np.load(vref_abs)
        return v_ref

    # 2) If asked to compute from a snapshot & shell
    if mode == "first_snapshot_inner":
        from math import isfinite
        v_shell, speed, r_c, M_shell, dr = compute_shell_density_weighted_velocity(
            snapshot_dir=snapshot_dir,
            snapshot_idx=init_snapshot_idx,
            coords_file=coords_file,
            lnrho_pattern=lnrho_pattern,
            v_pattern=v_pattern,
            v_ref=None,  # compute in lab frame first
        )
        if shell_index < 0 or shell_index >= r_c.size:
            raise IndexError(f"shell_index {shell_index} out of range (Nr={r_c.size})")
        v_ref = v_shell[shell_index].astype(np.float64)
        # sanitize
        if not np.all(np.isfinite(v_ref)):
            v_ref = np.zeros(3, dtype=np.float64)
        # persist
        os.makedirs(snapshot_dir, exist_ok=True)
        np.save(vref_abs, v_ref)
        return v_ref

    # 3) Default: no correction
    return np.zeros(3, dtype=np.float64)



def compute_shell_J_and_dm_dr(
    snapshot_dir: str,
    snapshot_idx: int,
    coords_file: str = "sph_coords_equalarea.npz",
    lnrho_pattern: str = "snapshot_{:04d}_lnrho_eq.npy",
    v_pattern: str = "snapshot_{:04d}_v_eq.npy"
):
    """
    Load one equal-area spherical snapshot and compute per-shell:
      - specific angular momentum vector j(r) = (1/M) ∑ [ (r x v) dm ]
      - dM/dr = M_shell / Δr

    Assumes you saved:
      - lnrho_eq:  shape (Nr, npix)    (log-density perturbation)
      - v_eq:      shape (Nr, npix, 3) (velocity components, Cartesian)
      - coords npz with:
          r (Nr,), theta (npix,), phi (npix,), nside (int)
          optionally U (npix,3) (unit vectors) and/or r_edges (Nr+1,)

    Args
    ----
    snapshot_dir  : folder containing snapshot files
    snapshot_idx  : which snapshot (integer index in filenames)
    coords_file   : coordinate npz (as saved by save_equal_area_coords)
    lnrho_pattern : filename pattern for lnrho (format with snapshot_idx)
    v_pattern     : filename pattern for velocity (format with snapshot_idx)

    Returns
    -------
    j_vec   : (Nr, 3)  specific angular momentum vector for each shell
    dm_dr   : (Nr,)    radial mass density (M_shell / Δr)
    M_shell : (Nr,)    shell mass
    r_c     : (Nr,)    shell center radii (cm)
    Δr      : (Nr,)    shell thickness (cm)
    """
    # ---- load coords ----
    cc_path = coords_file if os.path.isabs(coords_file) else os.path.join(snapshot_dir, coords_file)
    cc = np.load(cc_path)
    r_c = np.asarray(cc["r"], dtype=np.float64)          # (Nr,)
    Nr = r_c.size

    # Prefer saved unit vectors U; otherwise build from theta/phi
    if "U" in cc:
        U = np.asarray(cc["U"], dtype=np.float64)        # (npix, 3)
    else:
        theta = np.asarray(cc["theta"], dtype=np.float64)
        phi   = np.asarray(cc["phi"], dtype=np.float64)
        st = np.sin(theta)
        U = np.stack([st*np.cos(phi), st*np.sin(phi), np.cos(theta)], axis=-1)  # (npix,3)

    npix = U.shape[0]
    dOmega = 4.0 * np.pi / float(npix)

    # Reconstruct Δr from edges if available, else from log-uniform assumption
    if "r_edges" in cc:
        r_edges = np.asarray(cc["r_edges"], dtype=np.float64)  # (Nr+1,)
    else:
        # Centers are geometric means: r_c[i]^2 = e[i]*e[i+1], with constant ratio q between edges.
        # For log-uniform bins, center ratios are constant: r_c[i+1]/r_c[i] = q.
        # Then e[0] = r_c[0]/sqrt(q), e[k] = e[0]*q^k.
        if Nr < 2:
            raise ValueError("Need at least two radial bins to reconstruct edges from centers.")
        q = r_c[1] / r_c[0]
        if not np.allclose(r_c[1:] / r_c[:-1], q, rtol=1e-6, atol=0.0):
            raise ValueError("Radial centers are not log-uniform; please save r_edges in the coord file.")
        e0 = r_c[0] / np.sqrt(q)
        r_edges = e0 * (q ** np.arange(Nr + 1, dtype=np.float64))
    dr = r_edges[1:] - r_edges[:-1]   # (Nr,)

    # ---- load fields ----
    lnrho = np.load(os.path.join(snapshot_dir, lnrho_pattern.format(snapshot_idx)))  # (Nr, npix)
    v     = np.load(os.path.join(snapshot_dir, v_pattern.format(snapshot_idx)))      # (Nr, npix, 3)

    if lnrho.shape != (Nr, npix):
        raise ValueError(f"lnrho shape {lnrho.shape} != (Nr={Nr}, npix={npix})")
    if v.shape != (Nr, npix, 3):
        raise ValueError(f"v shape {v.shape} != (Nr={Nr}, npix={npix}, 3)")

    rho = np.exp(lnrho, dtype=np.float64)   # (Nr, npix)

    # ---- mass per pixel in each shell: dM = rho * (r^2 dΩ dr) ----
    r2 = (r_c**2).reshape(Nr, 1)                   # (Nr,1)
    vol_weight = r2 * dOmega * dr.reshape(Nr, 1)   # (Nr,1)
    dM = rho * vol_weight                          # (Nr, npix)

    # ---- shell mass and dm/dr ----
    M_shell = np.sum(dM, axis=1)                   # (Nr,)
    dm_dr   = M_shell / dr                         # (Nr,)

    # ---- specific angular momentum vector per shell ----
    # r_vec (per pixel) = r * U
    rU = (r_c.reshape(Nr, 1, 1)) * U.reshape(1, npix, 3)          # (Nr, npix, 3)
    # cross product r x v for each pixel
    rxv = np.cross(rU, v, axis=2)                                  # (Nr, npix, 3)
    # mass-weighted sum over pixels
    rxv_weighted_sum = np.sum(rxv * dM[:, :, None], axis=1)        # (Nr, 3)
    # normalize by shell mass (avoid divide-by-zero)
    with np.errstate(invalid='ignore', divide='ignore'):
        j_vec = rxv_weighted_sum / M_shell[:, None]                # (Nr, 3)
        j_vec[~np.isfinite(j_vec)] = 0.0

    return j_vec, dm_dr, M_shell, r_c, dr

import os
import numpy as np

def compute_shell_density_weighted_velocity(
    snapshot_dir: str,
    snapshot_idx: int,
    coords_file: str = "sph_coords_equalarea.npz",
    lnrho_pattern: str = "snapshot_{:04d}_lnrho_eq.npy",
    v_pattern: str = "snapshot_{:04d}_v_eq.npy",
    v_ref: np.ndarray | None = None,
    v_ref_mode: str | None = None,     # NEW: e.g. "first_snapshot_inner"
    v_ref_path: str = "v_ref.npy",     # NEW: persistent cache file
    init_snapshot_idx: int = 0,        # used if v_ref_mode set
    shell_index_for_ref: int = 0       # used if v_ref_mode set
):
    # Resolve v_ref if needed
    if v_ref is None and v_ref_mode is not None:
        v_ref = get_or_compute_v_ref(
            snapshot_dir=snapshot_dir,
            coords_file=coords_file,
            lnrho_pattern=lnrho_pattern,
            v_pattern=v_pattern,
            v_ref_path=v_ref_path,
            mode=v_ref_mode,
            init_snapshot_idx=init_snapshot_idx,
            shell_index=shell_index_for_ref
        )
    """
    For each radial shell, compute the density-weighted mean velocity:
        v_shell(r) = (1/M_shell) * sum_pixels[ v(r, Ω) * dm(r, Ω) ]
    where dm = rho * r^2 * dΩ * dr, rho =  exp(lnrho).

    Assumes equal-area angular pixels (HEALPix) saved as (Nr, npix):
      - lnrho_eq:  (Nr, npix)     (log-density perturbation)
      - v_eq:      (Nr, npix, 3)  (Cartesian velocity components)
      - coords npz with at least: r (Nr,), theta (npix,) or U (npix,3).
        If r_edges (Nr+1,) is present, it is used; otherwise edges are
        reconstructed assuming log-uniform radial bins.

    Returns
    -------
    v_shell : (Nr, 3)  density-weighted mean velocity per shell
    speed   : (Nr,)    |v_shell|
    r_c     : (Nr,)    shell-center radii [cm]
    M_shell : (Nr,)    shell mass
    dr      : (Nr,)    shell thickness [cm]
    """
    # ----- load coords -----
    cc_path = coords_file if os.path.isabs(coords_file) else os.path.join(snapshot_dir, coords_file)
    cc = np.load(cc_path)
    r_c = np.asarray(cc["r"], dtype=np.float64)      # (Nr,)
    Nr  = r_c.size

    # unit directions (not needed for this calculation, but we need npix)
    if "U" in cc:
        U = np.asarray(cc["U"], dtype=np.float64)    # (npix,3)
        npix = U.shape[0]
    else:
        theta = np.asarray(cc["theta"], dtype=np.float64)
        phi   = np.asarray(cc["phi"], dtype=np.float64)
        npix  = theta.size

    dOmega = 4.0 * np.pi / float(npix)

    # radial thickness from edges (preferred) or reconstruct (log-uniform)
    if "r_edges" in cc:
        r_edges = np.asarray(cc["r_edges"], dtype=np.float64)  # (Nr+1,)
    else:
        if Nr < 2:
            raise ValueError("Need at least two radial bins to reconstruct edges.")
        q = r_c[1] / r_c[0]
        if not np.allclose(r_c[1:] / r_c[:-1], q, rtol=1e-6, atol=0.0):
            raise ValueError("Radial centers are not log-uniform; please save r_edges in coord file.")
        e0 = r_c[0] / np.sqrt(q)
        r_edges = e0 * (q ** np.arange(Nr + 1, dtype=np.float64))
    dr = r_edges[1:] - r_edges[:-1]                   # (Nr,)

    # ----- load fields -----
    lnrho = np.load(os.path.join(snapshot_dir, lnrho_pattern.format(snapshot_idx)))  # (Nr, npix)
    v     = np.load(os.path.join(snapshot_dir, v_pattern.format(snapshot_idx)))      # (Nr, npix, 3)

    if lnrho.shape[0] != Nr or lnrho.shape[1] != npix:
        raise ValueError(f"lnrho shape {lnrho.shape} != (Nr={Nr}, npix={npix})")
    if v.shape != (Nr, npix, 3):
        raise ValueError(f"v shape {v.shape} != (Nr={Nr}, npix={npix}, 3)")

    # optional velocity reference subtraction (e.g., stellar COM frame)
    if v_ref is None:
        v_ref = np.zeros(3, dtype=np.float64)
    v = v - np.asarray(v_ref, dtype=np.float64).reshape(1, 1, 3)

    # ----- weights and averages -----
    rho = np.exp(lnrho, dtype=np.float64)     # (Nr, npix)
    r2  = (r_c**2).reshape(Nr, 1)                    # (Nr,1)
    vol_weight = r2 * dOmega * dr.reshape(Nr, 1)     # (Nr,1)
    dM  = rho * vol_weight                           # (Nr, npix)

    M_shell = np.sum(dM, axis=1)                     # (Nr,)

    # mass-weighted mean velocity per shell:
    # sum_over_pixels( v * dM ) / M_shell
    num = np.sum(v * dM[:, :, None], axis=1)         # (Nr, 3)
    with np.errstate(invalid='ignore', divide='ignore'):
        v_shell = num / M_shell[:, None]             # (Nr, 3)
        v_shell[~np.isfinite(v_shell)] = 0.0

    speed = np.linalg.norm(v_shell, axis=1)          # (Nr,)

    return v_shell, speed, r_c, M_shell, dr

import os
import numpy as np
import matplotlib.pyplot as plt

# Gravitational constant in CGS
G_CGS = 6.67430e-8
MSUN = 1.98847e33
PC2CM = 3.086e18

def compute_bhl_radius_per_shell(
    snapshot_dir: str,
    snapshot_idx: int,
    M_star: float,
    mass_units: str = "Msun",
    coords_file: str = "sph_coords_equalarea.npz",
    lnrho_pattern: str = "snapshot_{:04d}_lnrho_eq.npy",
    v_pattern: str = "snapshot_{:04d}_v_eq.npy",
    v_ref: np.ndarray | None = None,
    v_ref_mode: str | None = None,       # NEW
    v_ref_path: str = "v_ref.npy",       # NEW
    init_snapshot_idx: int = 0,          # NEW
    shell_index_for_ref: int = 0,        # NEW
    c_s: float | np.ndarray | None = None
):
    # Resolve v_ref if needed (computed once & cached)
    if v_ref is None and v_ref_mode is not None:
        v_ref = get_or_compute_v_ref(
            snapshot_dir=snapshot_dir,
            coords_file=coords_file,
            lnrho_pattern=lnrho_pattern,
            v_pattern=v_pattern,
            v_ref_path=v_ref_path,
            mode=v_ref_mode,
            init_snapshot_idx=init_snapshot_idx,
            shell_index=shell_index_for_ref
        )

    v_shell, speed, r_c, M_shell, dr = compute_shell_density_weighted_velocity(
        snapshot_dir=snapshot_dir,
        snapshot_idx=snapshot_idx,
        coords_file=coords_file,
        lnrho_pattern=lnrho_pattern,
        v_pattern=v_pattern,
        v_ref=v_ref
    )
    """
    For each shell, compute the Bondi–Hoyle–Lyttleton radius:
        r_BHL(r) = 2 G M_star / ( v_rel(r)^2 + c_s(r)^2 ).

    Inputs:
      - M_star: stellar mass (Msun by default; set mass_units='g' for grams).
      - v_rel(r): taken as the density-weighted mean speed of the shell, computed from the snapshot.
                  If you want star-frame, pass v_ref (the star velocity) so velocities are shifted first.
      - c_s: optional sound speed (scalar or array length Nr). If None, uses 0.

    Returns:
      r_c     : (Nr,)  shell center radii [cm]
      r_BHL   : (Nr,)  BHL radius per shell [cm]
      v_shell : (Nr,3) density-weighted mean velocity vector [cm/s]
      speed   : (Nr,)  |v_shell| [cm/s]
    """
    # --- get density-weighted shell velocity (and radii) ---
    v_shell, speed, r_c, M_shell, dr = compute_shell_density_weighted_velocity(
        snapshot_dir=snapshot_dir,
        snapshot_idx=snapshot_idx,
        coords_file=coords_file,
        lnrho_pattern=lnrho_pattern,
        v_pattern=v_pattern,
        v_ref=v_ref
    )

    Nr = r_c.size

    # --- mass in grams ---
    if mass_units.lower() == "msun":
        M = M_star * MSUN
    elif mass_units.lower() in ("g", "gram", "grams"):
        M = float(M_star)
    else:
        raise ValueError("mass_units must be 'Msun' or 'g'")

    # --- sound speed array ---
    if c_s is None:
        cs2 = 0.0
    else:
        c_s = np.asarray(c_s, dtype=np.float64)
        if c_s.ndim == 0:
            cs2 = float(c_s)**2
        else:
            if c_s.shape != (Nr,):
                raise ValueError(f"c_s has shape {c_s.shape}, expected scalar or (Nr,) with Nr={Nr}")
            cs2 = c_s**2  # (Nr,)

    v2 = speed**2                    # (Nr,)
    denom = v2 + cs2                 # (Nr,) or scalar
    with np.errstate(divide='ignore', invalid='ignore'):
        r_BHL = 2.0 * G_CGS * M / denom
        r_BHL[~np.isfinite(r_BHL)] = 0.0

    return r_c, r_BHL, v_shell, speed

PC2CM = 3.086e18

def plot_bhl_and_shell_velocity(
    snapshot_dir: str,
    snapshot_idx: int,
    M_star: float,
    mass_units: str = "Msun",
    v_ref: np.ndarray | None = None,
    c_s: float | np.ndarray | None = None,
    coords_file: str = "sph_coords_equalarea.npz",
    lnrho_pattern: str = "snapshot_{:04d}_lnrho_eq.npy",
    v_pattern: str = "snapshot_{:04d}_v_eq.npy",
    r_units: str = "pc",        # 'pc' or 'cm' (x-axis)
    rBHL_units: str = "pc",     # 'pc' or 'cm' (top y-axis)
    v_units: str = "km/s",      # 'cm/s' or 'km/s' (bottom y-axis)
    figsize=(7.8, 6.8),
    save: str | None = None,
    v_ref_mode: str | None = None,     # NEW: e.g. "first_snapshot_inner"
    v_ref_path: str = "v_ref.npy",     # NEW: persistent cache file
    init_snapshot_idx: int = 0,        # used if v_ref_mode set
    shell_index_for_ref: int = 0       # used if v_ref_mode set
):
    # --- compute BHL radius and shell velocities ---
    r_c, r_BHL, v_shell, speed = compute_bhl_radius_per_shell(
        snapshot_dir=snapshot_dir,
        snapshot_idx=snapshot_idx,
        M_star=M_star,
        mass_units=mass_units,
        coords_file=coords_file,
        lnrho_pattern=lnrho_pattern,
        v_pattern=v_pattern,
        v_ref=v_ref,
        v_ref_mode=v_ref_mode,
        v_ref_path=v_ref_path,
        init_snapshot_idx=init_snapshot_idx,
        shell_index_for_ref=shell_index_for_ref,
        c_s=c_s,
    )

    # units for radius
    if r_units.lower() == "pc":
        x = r_c / PC2CM
        xlabel = r"$r\ \mathrm{[pc]}$"
    elif r_units.lower() == "cm":
        x = r_c
        xlabel = r"$r\ \mathrm{[cm]}$"
    else:
        raise ValueError("r_units must be 'pc' or 'cm'")

    # units for r_BHL
    if rBHL_units.lower() == "pc":
        yB = r_BHL / PC2CM
        yB_label = r"$r_{\rm BHL}\ \mathrm{[pc]}$"
        yR = r_c / PC2CM
    elif rBHL_units.lower() == "cm":
        yB = r_BHL
        yB_label = r"$r_{\rm BHL}\ \mathrm{[cm]}$"
        yR = r_c
    else:
        raise ValueError("rBHL_units must be 'pc' or 'cm'")

    # units for velocity
    if v_units.lower() == "km/s":
        v_scale = 1e-5  # cm/s -> km/s
        vlabel = r"$\langle |\mathbf{v}| \rangle\ \mathrm{[km\,s^{-1}]}$"
        vclabel = r"$\langle v_i \rangle\ \mathrm{[km\,s^{-1}]}$"
    elif v_units.lower() == "cm/s":
        v_scale = 1.0
        vlabel = r"$\langle |\mathbf{v}| \rangle\ \mathrm{[cm\,s^{-1}]}$"
        vclabel = r"$\langle v_i \rangle\ \mathrm{[cm\,s^{-1}]}$"
    else:
        raise ValueError("v_units must be 'km/s' or 'cm/s'")

    jx, jy, jz = v_shell[:,0], v_shell[:,1], v_shell[:,2]
    speed_plot = speed * v_scale
    vx_plot, vy_plot, vz_plot = jx*v_scale, jy*v_scale, jz*v_scale

    # --- figure ---
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2.0, 2.0])

    # Top: r_BHL and r
    ax0.plot(x, yB, lw=2.2, label=r"$r_{\rm BHL}(r)$")
    ax0.plot(x, yR, lw=1.5, ls="--", label=r"$r$")
    ax0.set_xscale("log"); ax0.set_yscale("log")
    ax0.set_ylabel(yB_label)
    ax0.grid(True, ls=":", alpha=0.6)
    ax0.legend(frameon=False)

    # Bottom: density-weighted mean shell velocity
    ax1.plot(x, speed_plot, lw=2.0, label=r"$\langle |\mathbf{v}| \rangle$")
    ax1.plot(x, vx_plot, lw=1.5, ls="-.", label=r"$\langle v_x \rangle$")
    ax1.plot(x, vy_plot, lw=1.5, ls="--", label=r"$\langle v_y \rangle$")
    ax1.plot(x, vz_plot, lw=1.5, ls=":", label=r"$\langle v_z \rangle$")
    ax1.set_xscale("log")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(vclabel if v_units.lower()=="km/s" else vlabel)
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(frameon=False, ncol=2)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_shell_angular_momentum(
    snapshot_dir: str,
    snapshot_idx: int,
    coords_file: str = "sph_coords_equalarea.npz",
    lnrho_pattern: str = "snapshot_{:04d}_lnrho_eq.npy",
    v_pattern: str = "snapshot_{:04d}_v_eq.npy",
    x_units: str = "pc",         # 'pc' or 'cm'
    logx: bool = True,
    figsize=(7.5, 6.0),
    save: str | None = None,
):
    """
    Loads one snapshot, computes j(r), and plots |j| plus components vs r.

    Args:
        snapshot_dir : folder with snapshots
        snapshot_idx : which snapshot number to load
        coords_file  : coord file saved by equal-area grid builder
        lnrho_pattern, v_pattern : filename patterns
        x_units      : 'pc' or 'cm' for x-axis
        logx         : use log scale on r-axis
        figsize      : matplotlib figure size
        save         : path to save the figure; if None, show()
    """
    # compute j and get r
    j_vec, dm_dr, M_shell, r_c, dr = compute_shell_J_and_dm_dr(
        snapshot_dir=snapshot_dir,
        snapshot_idx=snapshot_idx,
        coords_file=coords_file,
        lnrho_pattern=lnrho_pattern,
        v_pattern=v_pattern
    )

    # x-axis in requested units
    pc2cm = 3.086e18
    if x_units.lower() == "pc":
        r_plot = r_c / pc2cm
        r_label = r"$r\ \mathrm{[pc]}$"
    elif x_units.lower() == "cm":
        r_plot = r_c
        r_label = r"$r\ \mathrm{[cm]}$"
    else:
        raise ValueError("x_units must be 'pc' or 'cm'")

    # magnitudes
    j_mag = np.linalg.norm(j_vec, axis=1)  # (Nr,)
    jx, jy, jz = j_vec[:,0], j_vec[:,1], j_vec[:,2]

    # plot
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1.8])
    ax0, ax1 = ax

    # top: |j|
    ax0.plot(r_plot, j_mag, lw=2)
    ax0.set_ylabel(r"$|{\bf j}(r)|$")
    ax0.set_yscale("log")
    ax0.grid(True, ls=":", lw=0.5, alpha=0.7)

    # bottom: components
    ax1.plot(r_plot, jx, label=r"$j_x$", lw=1.8)
    ax1.plot(r_plot, jy, label=r"$j_y$", lw=1.8)
    ax1.plot(r_plot, jz, label=r"$j_z$", lw=1.8)
    ax1.set_yscale("log")
    ax1.set_xlabel(r_label)
    ax1.set_ylabel(r"$j_i(r)$")
    ax1.grid(True, ls=":", lw=0.5, alpha=0.7)
    ax1.legend(loc="best", frameon=False)

    if logx:
        ax0.set_xscale("log")
        ax1.set_xscale("log")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":

    # Example for snapshot #12
    plot_shell_angular_momentum(
        snapshot_dir="snapshots",
        snapshot_idx=0,
        coords_file="sph_coords_equalarea.npz",
        x_units="pc",
        logx=True,
        save=None
    )

    plot_bhl_and_shell_velocity(
        snapshot_dir="snapshots",
        snapshot_idx=0,
        M_star=1.0,
        mass_units = "Msun",
        v_ref=None,
        c_s=0.1*1e5,
        v_ref_mode = "first_snapshot_inner",  # compute v_ref from snapshot 0, shell 0
        coords_file = "sph_coords_equalarea.npz")


    # Example for snapshot 12:
    j_vec, dm_dr, M_shell, r_c, dr = compute_shell_J_and_dm_dr(
        snapshot_dir="snapshots",
        snapshot_idx=12,
        coords_file="sph_coords_equalarea.npz"
    )

    # Magnitude of specific angular momentum:
    j_mag = np.linalg.norm(j_vec, axis=1)   # (Nr,)