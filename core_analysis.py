#!/usr/bin/env python3
"""
phi_minima_jeans_catalog.py

Step 1: Build a catalog of gravitational-potential minima on a spherical (log-r) × HEALPix grid.
Step 2: At those seeds, perform a 3-D Jeans mass check using an isothermal sound speed.

Inputs (default names match your files):
  - /mnt/data/sph_coords_equalarea.npz   (contains: r [cm], theta [rad], phi [rad], nside)
  - /mnt/data/snapshot_0000_lnrho_eq.npy (shape: Nr × Npix, values are ln(rho) with rho in g cm^-3)

Outputs:
  - CSV with columns: ir, ipix, r_cm, theta, phi, phi (potential), lnrho, R_kernel_cm, M_sphere_g, M_over_MJ
"""

import argparse
import numpy as np
import pandas as pd
from scipy.special import sph_harm
from scipy.spatial import cKDTree

# ------------------------------- Utilities -------------------------------- #

G   = 6.67430e-8       # cgs
k_B = 1.380649e-16     # cgs
m_H = 1.6735575e-24    # g

def load_grid(grid_path, rho_path):
    grid = np.load(grid_path, allow_pickle=True)
    r = grid["r"]                         # (Nr,)
    theta = grid["theta"]                 # (Npix,)
    phi = grid["phi"]                     # (Npix,)
    nside = int(grid["nside"])
    lnrho = np.load(rho_path)             # (Nr, Npix), ln(rho) with rho in g cm^-3
    assert lnrho.shape[0] == r.shape[0]
    assert lnrho.shape[1] == theta.shape[0] == phi.shape[0]
    return r, theta, phi, nside, lnrho

def reconstruct_edges_log(r):
    """Given log-spaced centers r[i], reconstruct shell edges r_{i-1/2}, r_{i+1/2}."""
    Nr = r.size
    logr = np.log(r)
    redges = np.empty(Nr+1)
    redges[1:-1] = np.exp(0.5*(logr[:-1] + logr[1:]))
    redges[0] = np.exp(logr[0] - 0.5*(logr[1]-logr[0]))
    redges[-1] = np.exp(logr[-1] + 0.5*(logr[-1]-logr[-2]))
    return redges

def spherical_harmonics_cache(theta, phi, lmax):
    """Precompute Y_lm(theta,phi) for all pixels up to lmax. Returns (Y, lm_list)."""
    cols, lm_list = [], []
    for l in range(lmax+1):
        for m in range(-l, l+1):
            # Note: scipy.sph_harm(m, l, phi, theta)
            cols.append(sph_harm(m, l, phi, theta))
            lm_list.append((l, m))
    Y = np.column_stack(cols)  # (Npix, (lmax+1)^2), complex
    return Y, lm_list

def project_rho_lm(rho_shells, Y, dOmega):
    """Angular integration rho_lm(r_i) = ∫ rho(r_i,Ω) Y*_lm(Ω) dΩ."""
    Nr = rho_shells.shape[0]
    nH = Y.shape[1]
    rho_lm = np.zeros((Nr, nH), dtype=np.complex128)
    Yc = np.conj(Y)
    for i in range(Nr):
        rho_lm[i] = (rho_shells[i, :, None] * Yc).sum(axis=0) * dOmega
    return rho_lm

def build_radial_integrals(r_in, r_out, rho_lm, lm_list):
    """Compute A_lm(i) and B_lm(i) shell-by-shell for all (l,m)."""
    Nr, nH = rho_lm.shape
    A = np.zeros_like(rho_lm, dtype=np.complex128)
    B = np.zeros_like(rho_lm, dtype=np.complex128)

    # Map (l,m) -> column index
    lm_to_col = {lm: j for j, lm in enumerate(lm_list)}

    for l in range(max(l for l,_ in lm_list)+1):
        # Inner weight per shell: ∫ r'^{l+2} dr' over shell
        W_inner = (r_out**(l+3) - r_in**(l+3)) / (l+3)  # (Nr,)
        # Outer weight per shell: ∫ r'^{1-l} dr'
        if l == 2:
            W_outer = np.log(r_out / r_in)
        else:
            W_outer = (r_out**(2-l) - r_in**(2-l)) / (2 - l)

        cols_l = [lm_to_col[(l, m)] for m in range(-l, l+1)]
        rl = rho_lm[:, cols_l]  # (Nr, 2l+1), complex

        # A[i] = sum_{j<=i-1} rl[j]*W_inner[j]
        A[:, cols_l] = np.vstack([np.zeros((1, rl.shape[1]), dtype=complex),
                                  np.cumsum(rl[:-1] * W_inner[:-1, None], axis=0)])

        # B[i] = sum_{j>=i} rl[j]*W_outer[j]
        B[:, cols_l] = np.flipud(np.cumsum(np.flipud(rl * W_outer[:, None]), axis=0))

    return A, B

def potential_on_shell(i, r_i, Y, A_i, B_i, lm_list):
    """Assemble Φ(Ω) on shell i from A_i, B_i (vectors over (l,m))."""
    two_l_plus1 = np.array([2*lm[0] + 1 for lm in lm_list], dtype=float)
    l_arr = np.array([lm[0] for lm in lm_list], dtype=float)

    pref = -4.0 * np.pi * G / two_l_plus1
    term_inner = (r_i ** (-(l_arr + 1.0))) * A_i
    term_outer = (r_i ** ( l_arr        )) * B_i
    C_i = pref * (term_inner + term_outer)  # (nH,), complex
    Phi_i = (Y @ C_i).real                  # (Npix,), real
    return Phi_i

def angular_neighbors(theta, phi, Npix, factor=1.25):
    """Neighbor list on the unit sphere using a chordal radius ~ pixel size."""
    ux = np.sin(theta)*np.cos(phi)
    uy = np.sin(theta)*np.sin(phi)
    uz = np.cos(theta)
    X = np.column_stack([ux, uy, uz])
    tree = cKDTree(X)
    # Pixel angular scale ~ sqrt(4π/Npix), chordal ≈ 2 sin(alpha/2).
    alpha_pix = np.sqrt(4*np.pi / Npix)
    rchord = factor * 2.0 * np.sin(alpha_pix/2.0)
    nbrs = tree.query_ball_point(X, r=rchord)
    return nbrs

def local_minima_on_sphere(values, neighbors):
    """Strict local minima: value < all neighbor values (excluding self)."""
    N = values.size
    is_min = np.zeros(N, dtype=bool)
    for p, nbrs in enumerate(neighbors):
        nbrs_wo_self = [q for q in nbrs if q != p]
        if not nbrs_wo_self:
            continue
        if np.all(values[p] < values[nbrs_wo_self]):
            is_min[p] = True
    return np.where(is_min)[0]

def build_positions_cartesian(r, theta, phi):
    """3-D Cartesian positions for all (ir, ipix)."""
    Nr, Npix = r.size, theta.size
    # Unit vectors for pixels
    ux = np.sin(theta)*np.cos(phi)
    uy = np.sin(theta)*np.sin(phi)
    uz = np.cos(theta)
    ux_t = np.tile(ux, Nr)
    uy_t = np.tile(uy, Nr)
    uz_t = np.tile(uz, Nr)
    R_flat = np.repeat(r, Npix)
    X = np.column_stack([R_flat*ux_t, R_flat*uy_t, R_flat*uz_t])  # (Nr*Npix, 3)
    return X

# ------------------------------- Main pipeline ----------------------------- #

def run_pipeline(
    r=None, theta=None, phi=None, lnrho=None,
    grid_path="snapshots/sph_coords_equalarea.npz",
    rho_path="snapshots/snapshot_0000_lnrho_eq.npy",
    out_csv="core_candidates_from_phi_minima.csv",
    file_load=False,
    lmax=12,
    T=10.0, mu=2.33,
    f_kernel=0.5,
    max_eval=2000,
    fallback_if_no_minima=True,
):
    
    if r is None or theta is None or phi is None or lnrho is None:
        if file_load:
            # Load
            r, theta, phi, nside, lnrho = load_grid(grid_path, rho_path)
        else:
            raise Exception("Require either file or input for finding cores")
    Nr, Npix = lnrho.shape
    dOmega = 4.0*np.pi / Npix

    # Density (convert from ln rho -> rho)
    rho = np.exp(lnrho)  # g cm^-3

    # Radial edges & cell volumes
    redges = reconstruct_edges_log(r)
    shell_volumes = (redges[1:]**3 - redges[:-1]**3) / 3.0          # per steradian
    V_cell = shell_volumes[:, None] * np.full((Nr, Npix), dOmega)    # cm^3

    # Angular basis
    Y, lm_list = spherical_harmonics_cache(theta, phi, lmax)
    # HEALPix angular neighbor graph (kNN radius ~ pixel size)
    neighbors = angular_neighbors(theta, phi, Npix, factor=1.25)

    # Project density on each shell
    rho_lm = project_rho_lm(rho, Y, dOmega)

    # Radial integrals for all (l,m)
    r_in, r_out = redges[:-1], redges[1:]
    A, B = build_radial_integrals(r_in, r_out, rho_lm, lm_list)

    # Find Φ minima on each shell
    records = []
    for i in range(Nr):
        Phi_i = potential_on_shell(i, r[i], Y, A[i], B[i], lm_list)  # (Npix,)
        mins_idx = local_minima_on_sphere(Phi_i, neighbors)
        if mins_idx.size:
            rec = np.column_stack([
                np.full(mins_idx.size, i, dtype=int),
                mins_idx.astype(int),
                np.full(mins_idx.size, r[i]),
                theta[mins_idx],
                phi[mins_idx],
                Phi_i[mins_idx],
                lnrho[i, mins_idx],
            ])
            records.append(rec)

    if len(records):
        Phi_minima = np.vstack(records)
        Phi_df = pd.DataFrame(Phi_minima, columns=["ir","ipix","r_cm","theta","phi","phi","lnrho"])
        Phi_df["ir"] = Phi_df["ir"].astype(int)
        Phi_df["ipix"] = Phi_df["ipix"].astype(int)
    else:
        # Fallback: one densest pixel per shell (keeps Step 2 useful)
        if not fallback_if_no_minima:
            raise RuntimeError("No Φ-minima found; try increasing lmax or neighbor radius.")
        ip_dense = np.argmax(rho, axis=1)
        Phi_df = pd.DataFrame({
            "ir": np.arange(Nr, dtype=int),
            "ipix": ip_dense.astype(int),
            "r_cm": r,
            "theta": theta[ip_dense],
            "phi": phi[ip_dense],
            "phi": np.zeros(Nr),  # placeholder
            "lnrho": lnrho[np.arange(Nr), ip_dense],
        })

    # -------- Step 2: Jeans mass check at seeds -------- #
    cs = np.sqrt(k_B*T/(mu*m_H))  # cm/s, isothermal
    R_flat = np.repeat(r, Npix)
    rho_flat = rho.reshape(-1)
    V_flat = V_cell.reshape(-1)

    # 3D positions & KD-tree
    X = build_positions_cartesian(r, theta, phi)
    tree3d = cKDTree(X)

    # Local Jeans quantities at every cell
    lamJ_flat = cs*np.sqrt(np.pi/(G*rho_flat))
    MJ_flat   = (np.pi**(2.5)/6.0) * cs**3 / (G**1.5) / np.sqrt(rho_flat)

    # Evaluate (possibly cap for speed)
    n_eval = min(len(Phi_df), max_eval)
    R_kernel = np.zeros(n_eval)
    M_sphere = np.zeros(n_eval)
    ratio    = np.zeros(n_eval)

    for k in range(n_eval):
        ir = int(Phi_df.iloc[k]["ir"]); ip = int(Phi_df.iloc[k]["ipix"])
        idx = ir*Npix + ip
        Rk = f_kernel * lamJ_flat[idx]
        nbrs = tree3d.query_ball_point(X[idx], r=Rk)
        Ms = np.dot(rho_flat[nbrs], V_flat[nbrs])
        R_kernel[k] = Rk
        M_sphere[k] = Ms
        ratio[k]    = Ms / MJ_flat[idx]

    Phi_df_sub = Phi_df.iloc[:n_eval].copy()
    Phi_df_sub["R_kernel_cm"] = R_kernel
    Phi_df_sub["M_sphere_g"]  = M_sphere
    Phi_df_sub["M_over_MJ"]   = ratio


    # Sort by M/MJ (descending) and save
    out = Phi_df_sub.sort_values("M_over_MJ", ascending=False).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"Seeds evaluated: {n_eval}  |  Max M/MJ: {out['M_over_MJ'].max():.3e}  |  Median: {out['M_over_MJ'].median():.3e}")
    return out


def assemble_phi_all_shells(r, theta, phi, rho, lmax=12):
    """
    Compute Φ at every cell (Nr*Npix) using the same Y_lm/A/B machinery.

    Returns
    -------
    Phi_flat : (Nr*Npix,) array
    """
    Nr, Npix = rho.shape
    dOmega = 4*np.pi/Npix
    Y, lm_list = spherical_harmonics_cache(theta, phi, lmax)
    rho_lm = project_rho_lm(rho, Y, dOmega)
    redges = reconstruct_edges_log(r)
    A, B = build_radial_integrals(redges[:-1], redges[1:], rho_lm, lm_list)

    Phi_flat = np.empty(Nr*Npix, dtype=np.float64)
    two_l_plus1 = np.array([2*l+1 for l,_ in lm_list], dtype=float)
    l_arr = np.array([l for l,_ in lm_list], dtype=float)
    pref = -4.0*np.pi*G / two_l_plus1

    for i in range(Nr):
        ri = r[i]
        term_inner = (ri ** (-(l_arr + 1.0))) * A[i]
        term_outer = (ri ** ( l_arr        )) * B[i]
        C_i = pref * (term_inner + term_outer)  # (nH,)
        Phi_i = (Y @ C_i).real                  # (Npix,)
        Phi_flat[i*Npix:(i+1)*Npix] = Phi_i
    return Phi_flat


def grow_bound_region(seed_idx, X, vxyz, Phi, rho_flat, V_flat,
                      R_init, assign_mask):
    """
    Iterative unbinding around a seed (Subfind-like).

    Parameters
    ----------
    seed_idx : int
        Flat index of the seed (ir*Npix + ipix).
    X : (N,3) cell positions
    vxyz : (N,3) cell velocities (Cartesian)
    Phi : (N,) gravitational potential
    rho_flat : (N,) densities
    V_flat : (N,) cell volumes
    R_init : float [cm], initial spherical radius to gather candidates
    assign_mask : (N,) bool, True for cells already assigned to previous regions

    Returns
    -------
    bound_indices : 1D int array of the cells bound to this seed (disjoint from already assigned)
    """
    tree = cKDTree(X)
    # Initial candidate set: within R_init of the seed, excluding already assigned
    cand = np.array([i for i in tree.query_ball_point(X[seed_idx], r=R_init)
                     if not assign_mask[i]], dtype=int)
    if cand.size == 0:
        return np.array([], dtype=int)

    # Iterative unbinding
    cand_set = cand
    max_iter = 30
    tol_remove = 0  # remove all with positive energy
    for _ in range(max_iter):
        # mass weights
        m = rho_flat[cand_set] * V_flat[cand_set]
        M = m.sum()
        if M <= 0:
            cand_set = np.array([], dtype=int)
            break
        # center-of-mass velocity
        vcm = (m[:,None] * vxyz[cand_set]).sum(axis=0) / M
        # specific energy relative to CM; Φ is absolute (zero at infinity)
        dv = vxyz[cand_set] - vcm[None,:]
        e_spec = 0.5*np.einsum("ij,ij->i", dv, dv) + Phi[cand_set]
        keep = e_spec < 0.0
        if keep.all():
            break
        new_set = cand_set[keep]
        if new_set.size == cand_set.size:
            break
        cand_set = new_set

    # Final pass: ensure connectedness around the seed (optional, helps avoid flyaways)
    if cand_set.size == 0:
        return cand_set
    # Keep only the connected component that contains the seed
    # Build a small KD-tree on the survivors with link length = 0.5 * R_init
    Xu = X[cand_set]
    tree_u = cKDTree(Xu)
    nbrs = tree_u.query_ball_point(Xu, r=0.5*R_init)
    # BFS/DFS from the seed
    idx_map = {idx:i for i,idx in enumerate(cand_set)}
    root = idx_map.get(seed_idx, None)
    if root is None:
        # if the seed was unbound itself, just return empty
        return np.array([], dtype=int)
    stack = [root]; visited = set([root])
    while stack:
        i = stack.pop()
        for j in nbrs[i]:
            if j not in visited:
                visited.add(j); stack.append(j)
    comp = np.array([cand_set[i] for i in visited], dtype=int)
    return comp

def coerce_velocity_to_NrNpix3(v, Nr, Npix):
    """
    Try to coerce 'v' into shape (Nr, Npix, 3) from common permutations.
    Raises ValueError if it can't.
    """
    v = np.asarray(v)
    if v.ndim == 3:
        if v.shape == (Nr, Npix, 3):
            return v
        if v.shape == (3, Nr, Npix):
            return np.moveaxis(v, 0, -1).transpose(1, 2, 0)  # -> (Nr, Npix, 3)
        if v.shape == (Nr, 3, Npix):
            return np.transpose(v, (0, 2, 1))                 # -> (Nr, Npix, 3)
        if v.shape == (Npix, Nr, 3):
            return np.transpose(v, (1, 0, 2))                 # -> (Nr, Npix, 3)
        if v.shape == (Nr*Npix, 3):
            return v.reshape(Nr, Npix, 3)
        if v.shape == (3, Nr*Npix):
            return np.moveaxis(v, 0, -1).reshape(Nr, Npix, 3)
    if v.ndim == 2 and v.shape == (Nr*Npix, 3):
        return v.reshape(Nr, Npix, 3)
    raise ValueError(f"Unrecognized velocity shape {v.shape}; "
                     f"expected something convertible to (Nr, Npix, 3).")


import heapq

def velocities_to_cartesian(r, theta, phi, v, basis="spherical"):
    """
    Convert velocity components to Cartesian at every cell.

    Parameters
    ----------
    r : (Nr,) array [cm]
    theta, phi : (Npix,) arrays [rad]
    v : (Nr, Npix, 3) array
        If basis='spherical', v[...,0]=v_r, v[...,1]=v_theta, v[...,2]=v_phi [cm/s].
        If basis='cartesian', v[...,0:3] are already (vx, vy, vz) [cm/s].
    basis : 'spherical' or 'cartesian'

    Returns
    -------
    vxyz : (Nr*Npix, 3) array of Cartesian velocities [cm/s] aligned with the global frame.
    """
    Nr, Npix, _ = v.shape
    if basis == "cartesian":
        return v.reshape(-1, 3).copy()

    # spherical -> cartesian: v = v_r rhat + v_theta thetahat + v_phi phihat
    v_r = v[..., 0]
    v_t = v[..., 1]
    v_p = v[..., 2]

    # unit vectors at each pixel
    st, ct = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    # broadcast to (Nr, Npix)
    st = np.broadcast_to(st, (Nr, Npix))
    ct = np.broadcast_to(ct, (Nr, Npix))
    cp = np.broadcast_to(cp, (Nr, Npix))
    sp = np.broadcast_to(sp, (Nr, Npix))

    # rhat, thetahat, phihat in Cartesian
    rx = st*cp; ry = st*sp; rz = ct
    tx = ct*cp; ty = ct*sp; tz = -st
    px = -sp;   py =  cp;   pz =  0.0

    vx = v_r*rx + v_t*tx + v_p*px
    vy = v_r*ry + v_t*ty + v_p*py
    vz = v_r*rz + v_t*tz + v_p*pz
    return np.column_stack([vx.reshape(-1), vy.reshape(-1), vz.reshape(-1)])


def find_bound_region_within_sphere(
    r=None, theta=None, phi=None, lnrho=None, v=None,
    file_load=False,
    grid_path="snapshots/sph_coords_equalarea.npz",
    rho_path ="snapshots/snapshot_0000_lnrho_eq.npy",
    vel_path ="snapshots/snapshot_0000_v_eq.npy",
    out_labels="local_bound_labels.npy",
    out_catalog="local_bound_catalog.csv",
    # --- sphere definition ---
    center_mode="cartesian",     # "cartesian" or "index"
    center_xyz_cm=(0.0, 0.0, 0.0), # used if center_mode="cartesian"
    center_index=(0, 0),         # (ir, ipix) if center_mode="index"
    R_search_cm=1.0e18,          # search radius (cm)
    # --- physics / numerics ---
    lmax=12,
    velocity_basis='cartesian',
    # initial gather radius for unbinding:
    use_fixed_Rinit=False,
    R_init_cm=None,              # if use_fixed_Rinit=True, use this value
    T=10.0, mu=2.33,             # else R_init = f_init * lambda_J(seed) (scale only)
    f_init=2.0,
):
    """
    1) Compute Φ everywhere.
    2) Restrict to cells within a user-defined sphere.
    3) Pick the *deepest* Φ minimum (most negative Φ) inside that sphere.
    4) Iterative unbinding around that seed to return the locally bound set.
    """

    if r is None or theta is None or phi is None or lnrho is None:
        if file_load:
            # Load
            grid = np.load(grid_path, allow_pickle=True)
            r = grid["r"]; theta = grid["theta"]; phi = grid["phi"]

            lnrho = np.load(rho_path)
            v = np.load(vel_path)   
        else:
            raise Exception("Require either file or input for finding cores")

    rho = np.exp(lnrho)
    Nr, Npix = len(r), len(theta)
        # volumes
    Nr, Npix = r.size, phi.size
    dOmega = 4*np.pi / Npix
    redges = reconstruct_edges_log(r)
    shell_vol = (redges[1:]**3 - redges[:-1]**3) / 3.0        # (Nr,)
    V_flat = np.repeat(shell_vol * dOmega, Npix)              # (Nr*Npix,)

    print('Nr, Npix', Nr, Npix, r.shape, theta.shape)

    # ---- volumes, flat arrays, positions, velocities ----
    #redges = reconstruct_edges_log(r)
    #dOmega = 4*np.pi/Npix
    #shell_vol = (redges[1:]**3 - redges[:-1]**3)/3.0
    #V_cell = shell_vol[:,None] * dOmega          # (Nr, Npix)
    #V_flat = V_cell.reshape(-1)
    rho_flat = rho.reshape(-1)

    X = build_positions_cartesian(r, theta, phi)  # (N,3)
    print(v.shape)
    v = coerce_velocity_to_NrNpix3(v, Nr, Npix)
    vxyz = velocities_to_cartesian(r, theta, phi, v, basis=velocity_basis)
    print(vxyz.shape)

    # ---- gravitational potential everywhere ----
    Phi_flat = assemble_phi_all_shells(r, theta, phi, rho, lmax=lmax)  # (N,)

    # ---- define spherical mask ----
    if center_mode == "cartesian":
        cx, cy, cz = map(float, center_xyz_cm)
        C = np.array([cx, cy, cz], dtype=float)
    elif center_mode == "index":
        ir0, ip0 = center_index
        # center at that cell's position
        cx, cy, cz = X[ir0*Npix + ip0]
        C = np.array([cx, cy, cz], dtype=float)
    else:
        raise ValueError("center_mode must be 'cartesian' or 'index'.")

    d2 = np.sum((X - C[None,:])**2, axis=1)
    in_sphere = d2 <= (R_search_cm**2)

    if not np.any(in_sphere):
        raise RuntimeError("No cells fall inside the requested search sphere. Check center/radius.")

    # ---- pick the deepest Φ *inside* the sphere ----
    masked_idx = np.where(in_sphere)[0]
    seed_idx = masked_idx[np.argmin(Phi_flat[masked_idx])]
    seed_ir, seed_ip = divmod(seed_idx, Npix)

    # ---- choose initial gather radius for unbinding ----
    if use_fixed_Rinit and (R_init_cm is not None):
        R_init = float(R_init_cm)
    else:
        # set by local Jeans length at the seed (scale only; not used in binding energy)
        G   = 6.67430e-8
        k_B = 1.380649e-16
        m_H = 1.6735575e-24
        cs  = np.sqrt(k_B*T/(mu*m_H))
        lamJ_seed = cs*np.sqrt(np.pi/(G*rho_flat[seed_idx]))
        R_init = float(f_init * lamJ_seed)
        # ensure it's not trivially small; cap by the user sphere
        R_init = min(max(R_init, 0.05*R_search_cm), R_search_cm)

    # ---- iterative unbinding restricted to the search sphere ----
    # We pass an "assigned" mask that forbids grabbing cells OUTSIDE the sphere
    assigned = ~in_sphere.copy()   # True outside => excluded

    N = X.shape[0]
    assert rho_flat.size == N and V_flat.size == N and vxyz.shape[0] == N, \
        f"Shape mismatch: N={N}, rho={rho_flat.size}, V={V_flat.size}, v={vxyz.shape[0]}"

    bound_comp = grow_bound_region(seed_idx, X, vxyz, Phi_flat, rho_flat, V_flat, R_init, assigned)

    # If seed ended up unbound or pruning removed everything, return empty outputs but still save files
    labels = np.full(Nr*Npix, -1, dtype=int)
    catalog_rows = []
    if bound_comp.size > 0:
        labels[bound_comp] = 0  # single local region id
        m = rho_flat[bound_comp] * V_flat[bound_comp]
        M = float(m.sum())
        rc = (m[:,None] * X[bound_comp]).sum(axis=0) / max(M, 1e-99)
        vc = (m[:,None] * vxyz[bound_comp]).sum(axis=0) / max(M, 1e-99)
        catalog_rows.append({
            "region_id": 0,
            "n_cells": int(bound_comp.size),
            "mass_g": M,
            "x_cm": float(rc[0]), "y_cm": float(rc[1]), "z_cm": float(rc[2]),
            "vx_cm": float(vc[0]), "vy_cm": float(vc[1]), "vz_cm": float(vc[2]),
            "seed_ir": int(seed_ir), "seed_ipix": int(seed_ip),
            "phi_seed": float(Phi_flat[seed_idx]),
            "R_search_cm": float(R_search_cm),
            "R_init_cm": float(R_init),
        })

    # ---- save & return ----
    np.save(out_labels, labels.reshape(Nr, Npix))
    cat = pd.DataFrame(catalog_rows)
    cat.to_csv(out_catalog, index=False)
    print(f"[local] bound cells inside sphere: {(labels>=0).sum()}  | saved:")
    print("  - labels:", out_labels)
    print("  - catalog:", out_catalog)
    return labels.reshape(Nr, Npix), cat



# ------------------------------- CLI -------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Φ-minima catalog + Jeans check on spherical HEALPix grid.")
    p.add_argument("--grid", default="/mnt/data/sph_coords_equalarea.npz", help="Path to grid .npz")
    p.add_argument("--rho",  default="/mnt/data/snapshot_0000_lnrho_eq.npy", help="Path to ln-rho .npy")
    p.add_argument("--out",  default="/mnt/data/core_candidates_from_phi_minima.csv", help="Output CSV path")
    p.add_argument("--lmax", type=int, default=12, help="Max spherical-harmonic degree for Φ")
    p.add_argument("--T", type=float, default=10.0, help="Isothermal temperature [K]")
    p.add_argument("--mu", type=float, default=2.33, help="Mean molecular weight")
    p.add_argument("--f_kernel", type=float, default=0.5, help="Kernel radius factor: R = f_kernel * lambda_J")
    p.add_argument("--max_eval", type=int, default=2000, help="Max seeds to evaluate for Jeans check")
    p.add_argument("--no_fallback", action="store_true", help="Disable densest-per-shell fallback if no Φ-minima")
    args = p.parse_args()

    run_pipeline(
        grid_path=args.grid,
        rho_path=args.rho,
        out_csv=args.out,
        lmax=args.lmax,
        T=args.T,
        mu=args.mu,
        f_kernel=args.f_kernel,
        max_eval=args.max_eval,
        fallback_if_no_minima=not args.no_fallback
    )

if __name__ == "__main__":
    main()
