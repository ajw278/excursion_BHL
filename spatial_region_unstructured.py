# binary_unstructured_field.py  (extended)
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import generate_cloudrho0 as gcr
from consts_defaults import year2s, pc2cm, Gcgs as G  # your constants

print('Next steps: ADD RECURSIVE CLOUD GENERATION')


def _ratio() -> float:
    return 2.0 ** (-1.0 / 3.0)

def _rand_in_ball(R: float, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3); n = np.linalg.norm(v)
    if n == 0: v = np.array([1.0, 0.0, 0.0]); n = 1.0
    v /= n
    u = rng.random()
    return v * (R * (u ** (1.0 / 3.0)))


@dataclass
class LevelView:
    level: int
    n: int
    offset: int
    grid_index: int


class BinaryUnstructuredField:
    """
    Unstructured, fully populated binary hierarchy on a geometric grid.

    New features:
      - evolve() to step OU fields in time and save snapshots
      - density_at_level(), velocity_at_level()
      - make_video() to render 3D colored-scatter of log10(density)
    """
    def __init__(
        self,
        grid,
        *,
        start_grid_index: int = 0,
        seed: Optional[int] = None,
        root_pos_cm: Optional[np.ndarray] = None,
        max_levels: Optional[int] = None,
        verbose: bool = False,
        collapse: bool = False,
        evolve0d: bool = False,          
        n_super: Optional[int] = None,   
    ):
        self.grid = grid
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

        self.root_pos = np.zeros(3) if root_pos_cm is None else np.asarray(root_pos_cm, float)
        self.start = int(start_grid_index)
        
        self.L0_cm = float(self.grid.rlevels[self.start])  # store L0 scale
        self.R0_cm = self.L0_cm/2.  # store L0 scale
        self.collapse = bool(collapse)
        self.evolve0d = bool(evolve0d)

        self.baseline_fn = None
        self.baseline_info = {}
        if self.collapse:
            try:
                self.baseline_fn, self.baseline_info = gcr.build_cloud_baseline_fn(target_radius_cm=self.R0_cm)
            except Exception:
                # collapse mode requested but builder missing; proceed without baseline
                self.collapse = False
                if self.verbose:
                    print("[collapse] generate_cloudrho0 not available; collapse disabled.")

        # depth rule: STOP at first level where t_ff < t_cross
        i_end = self._find_end_grid_index(self.start)
        if i_end is None:
            i_end = min(self.start + 1, len(self.grid.rlevels) - 1)

        n_levels_nat = i_end - self.start + 1
        L = int(max(1, min(max_levels, n_levels_nat))) if max_levels is not None else int(max(1, n_levels_nat))
        

        # level views and flattened storage
        self.levels: List[LevelView] = []
        offset = 0
        for ℓ in range(L):
            n = 2 ** ℓ  # 2, 4, 8, ...
            gi = self.start + ℓ
            self.levels.append(LevelView(level=ℓ, n=n, offset=offset, grid_index=gi))
            offset += n
        self.N_total = offset

        self.delta   = np.zeros(self.N_total, dtype=float)
        self.delta_v = np.zeros((self.N_total, 3), dtype=float)
        self.pos     = np.zeros((self.N_total, 3), dtype=float)

        self.parent = -np.ones(self.N_total, dtype=int)
        self._build_topology_and_positions()
        self._initialize_gaussian_increments()

        self.super_grid_indices: np.ndarray = np.array([], dtype=int)
        self.super_delta: np.ndarray = np.zeros(0, float)
        self.super_delta_v: np.ndarray = np.zeros((0, 3), float)

        if self.evolve0d and self.start > 0:
            self._init_super_fields(n_super)

        # cache: repeated aggregations are common
        self._agg_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


    def _super_active_mask(self, Lcut: float) -> np.ndarray:
        """
        True for super scales with R <= Lcut (same rule as 0D filtering).
        If no super levels, returns empty bool array.
        """
        if not (self.evolve0d and self.super_grid_indices.size):
            return np.zeros(0, dtype=bool)
        r = np.asarray(self.grid.rlevels, float)
        return r[self.super_grid_indices] <= float(Lcut)

    def _super_sums(self):
        if not getattr(self, "evolve0d", False) or self.super_grid_indices.size == 0:
            return 0.0, np.zeros(3, float)
        return float(self.super_delta.sum()), self.super_delta_v.sum(axis=0)


    # ---------- construction helpers ----------
    def _branch_loc_for_gridlevel(self, branch_idx, target_gi):
        """
        branch_idx : 1D array of global node indices for this finest point's ancestors (coarse→fine)
        target_gi  : integer grid level (the 'gi_first' for Lcut)
        returns: integer ℓ such that branch_idx[ℓ] has grid_index == target_gi, or None
        """
        gi_on_branch = self.grid_index[branch_idx]  # vector of grid-level ids along branch
        hits = np.where(gi_on_branch == int(target_gi))[0]
        return int(hits[0]) if hits.size else None


    def _find_end_grid_index(self, i_start: int, t_factor: float=1.0) -> Optional[int]:
        """Return the smallest i >= i_start with t_ff(rhocs[i]) < tau_R[i]."""
        for i in range(i_start, len(self.grid.rlevels)):
            t_cross = float(self.grid.tau_R[i])
            rho_c   = float(self.grid.rhocs[i])
            if not (np.isfinite(t_cross) and np.isfinite(rho_c) and rho_c > 0.0):
                continue
            t_ff = np.sqrt(3.0 * np.pi / (32.0 * G * rho_c))
            if t_ff < t_cross*t_factor:
                if self.verbose:
                    print(f"[depth] stop at grid index ({i-i_start}): t_ff={t_ff:.3e}s < t_cross={t_cross:.3e}s *factor")
                return i
           
        if self.verbose:
            print(f"[depth] include all indices ({i-i_start}): t_ff={t_ff:.3e}s > t_cross={t_cross:.3e}s *factor")
                
        return None

    def _build_topology_and_positions(self):
        # level 0 children around root
        v0 = self.levels[0]
        self.pos[v0.offset + 0] = self.root_pos
        self.parent[v0.offset + 0] = -1

        # deeper levels: unchanged in structure, but now they start from ℓ=1
        for ℓ in range(1, len(self.levels)):
            v  = self.levels[ℓ]
            vp = self.levels[ℓ - 1]
            Rℓ = float(self.grid.rlevels[v.grid_index])
            for i_parent in range(vp.n):
                p_glob   = vp.offset + i_parent
                r_parent = self.pos[p_glob]
                d = _rand_in_ball(0.5 * Rℓ, self.rng)        # <-- radius = grid scale / 2
                c0 = v.offset + 2 * i_parent
                c1 = c0 + 1
                self.pos[c0]   = r_parent + d
                self.pos[c1]   = r_parent - d
                self.parent[c0] = p_glob
                self.parent[c1] = p_glob

    
    def _initialize_gaussian_increments(self):
        for v in self.levels:
            s = slice(v.offset, v.offset + v.n)
            self.delta[s]   = self.rng.normal(0.0, np.sqrt(float(self.grid.Delta_S[v.grid_index])),  size=v.n)
            self.delta_v[s] = self.rng.normal(0.0, np.sqrt(float(self.grid.Delta_Sv[v.grid_index])), size=(v.n, 3))


    def _init_super_fields(self, n_super: Optional[int]):
        """
        Initialize 0-D (no position) stochastic increments on coarser-than-level-0
        grid scales. Each super scale contributes globally to ALL points.
        """
        if self.start <= 0:
            return  # nothing coarser than level-0

        if n_super is None:
            idx = np.arange(0, self.start, dtype=int)  # all coarser grid indices
        else:
            n_super = int(max(1, n_super))
            idx = np.arange(max(0, self.start - n_super), self.start, dtype=int)

        self.super_grid_indices = idx

        # allocate arrays
        ns = idx.size
        self.super_delta = np.zeros(ns, dtype=float)
        self.super_delta_v = np.zeros((ns, 3), dtype=float)

        # one-off Gaussian draw with the grid variances at each coarse index
        for j, gi in enumerate(idx):
            self.super_delta[j] = self.rng.normal(0.0, np.sqrt(float(self.grid.Delta_S[gi])))
            self.super_delta_v[j, :] = self.rng.normal(0.0, np.sqrt(float(self.grid.Delta_Sv[gi])), size=3)

    # ---------- ancestry & aggregation ----------

    def level_slice(self, level: int) -> slice:
        v = self.levels[level]
        return slice(v.offset, v.offset + v.n)

    def ancestors(self, idx_global: int) -> List[int]:
        chain = []
        i = int(idx_global)
        while i != -1:
            chain.append(i)
            i = int(self.parent[i])
        chain.reverse()
        return chain

    def field_at_level(self, level: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sum (delta, delta_v) down branches up to `level` (cached)."""
        if level in self._agg_cache:
            return (self._agg_cache[level][0].copy(),
                    self._agg_cache[level][1].copy())

        if level == 0:
            s0 = self.level_slice(0)
            out = (self.delta[s0].copy(), self.delta_v[s0].copy())
            self._agg_cache[level] = (out[0].copy(), out[1].copy())
            return out

        # dynamic program up the levels
        s0 = self.level_slice(0)
        agg_scalar_prev = self.delta[s0].copy()
        agg_vector_prev = self.delta_v[s0].copy()
        for ℓ in range(1, level + 1):
            sℓ = self.level_slice(ℓ)
            rep_s = np.repeat(agg_scalar_prev, 2, axis=0)
            rep_v = np.repeat(agg_vector_prev, 2, axis=0)
            agg_scalar_prev = rep_s + self.delta[sℓ]
            agg_vector_prev = rep_v + self.delta_v[sℓ]

        self._agg_cache[level] = (agg_scalar_prev.copy(), agg_vector_prev.copy())
        return agg_scalar_prev, agg_vector_prev

    def _invalidate_cache(self):
        self._agg_cache.clear()

    # ---------- densities & velocities ----------
    def density_at_level(self, level: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if level is None:
            level = len(self.levels) - 1
        dsum, _ = self.field_at_level(level)

        # add 0-D supers (unfiltered in the non-collapse path)
        s_super, _ = self._super_sums()
        if s_super != 0.0:
            dsum = dsum + float(s_super)

        gi = self.levels[level].grid_index
        rho0 = float(self.grid.rho0)
        mu = float(np.atleast_1d(self.grid.mu_lnrho)[gi]) if hasattr(self.grid, "mu_lnrho") else 0.0
        log_rho = mu + dsum
        rho = rho0 * np.exp(log_rho)
        return rho, log_rho

    def velocity_at_level(self, level: Optional[int] = None) -> np.ndarray:
        if level is None:
            level = len(self.levels) - 1
        _, vsum = self.field_at_level(level)

        # add 0-D supers (unfiltered in the non-collapse path)
        _, v_super = self._super_sums()
        if np.any(v_super):
            vsum = vsum + v_super[None, :]

        return vsum


    # ---------- OU time stepping ----------

    def step(self, dt_seconds: float):
        dt = float(dt_seconds)

        # --- 0-D super scales (if enabled) ---
        if self.evolve0d and self.super_grid_indices.size:
            for j, gi in enumerate(self.super_grid_indices):
                tau = float(self.grid.tau_R[gi])
                a = np.exp(-dt / tau)
                var_s  = float(self.grid.Delta_S[gi])  * (1.0 - np.exp(-2.0 * dt / tau))
                var_sv = float(self.grid.Delta_Sv[gi]) * (1.0 - np.exp(-2.0 * dt / tau))
                self.super_delta[j]     = self.super_delta[j]     * a + self.rng.normal(0.0, np.sqrt(var_s))
                self.super_delta_v[j,:] = self.super_delta_v[j,:] * a + self.rng.normal(0.0, np.sqrt(var_sv), size=3)

        for v in self.levels:
            s = slice(v.offset, v.offset + v.n)
            tau = float(self.grid.tau_R[v.grid_index])

            a = np.exp(-dt / tau)
            var_b  = float(self.grid.Delta_S[v.grid_index])  * (1.0 - np.exp(-2.0 * dt / tau))
            var_bv = float(self.grid.Delta_Sv[v.grid_index]) * (1.0 - np.exp(-2.0 * dt / tau))

            self.delta[s]   = self.delta[s]   * a + self.rng.normal(0.0, np.sqrt(var_b),  size=v.n)
            self.delta_v[s] = self.delta_v[s] * a + self.rng.normal(0.0, np.sqrt(var_bv), size=(v.n, 3))

        self._invalidate_cache()

    # ---------- snapshots & evolution ----------

    def save_snapshot(
        self,
        t_seconds: float,
        snapshot_dir: str,
        *,
        level: Optional[int] = None,
        frame_idx: Optional[int] = None,
        percent_clip: Tuple[float, float] = (2.0, 98.0),
    ):
        """
        Save a snapshot at `level` (default: finest) with positions, rho, log_rho, velocity.
        """
        if level is None:
            level = len(self.levels) - 1
        os.makedirs(snapshot_dir, exist_ok=True)
        s = self.level_slice(level)

        posL = self.pos[s].copy()
        # use the finest-level helpers with t_seconds for collapse filtering
        if level is None:
            rho, log_rho = self.density_at_finest(t_seconds=t_seconds)
            vel = self.velocity_at_finest(t_seconds=t_seconds)
        else:
            # for non-finest, you can keep the old behavior or add a level-aware variant
            rho, log_rho = self.density_at_level(level)   # unchanged
            vel = self.velocity_at_level(level)

        # helpful color limits persisted in file for consistent videos
        lo, hi = np.percentile(np.log10(rho), percent_clip)

        if frame_idx is None:
            # auto-infer by counting files
            existing = [f for f in os.listdir(snapshot_dir) if f.startswith("frame_") and f.endswith(".npz")]
            frame_idx = 0 if not existing else (max(int(f[6:10]) for f in existing) + 1)

        fpath = os.path.join(snapshot_dir, f"frame_{frame_idx:04d}.npz")
        np.savez_compressed(
            fpath,
            t_seconds=np.array([t_seconds], float),
            pos_cm=posL.astype(float),
            rho=rho.astype(float),
            log_rho=log_rho.astype(float),
            vel_cm_s=vel.astype(float),
            log10rho_clip=np.array([lo, hi], float),
        )

        # maintain a times index
        times_path = os.path.join(snapshot_dir, "times.npy")
        if os.path.exists(times_path):
            times = np.load(times_path)
            times = np.append(times, t_seconds)
        else:
            times = np.array([t_seconds], float)
        np.save(times_path, times)

    def evolve(
        self,
        Tend_s: float,
        dt_s: float,
        dt_snap_s: float,
        *,
        snapshot_dir: str = "snapshots_unstructured",
        start_time_s: float = 0.0,
        percent_clip: Tuple[float, float] = (2.0, 98.0),
        verbose: bool = True,
    ):
        """
        Time-march OU fields, writing snapshots of the finest level every dt_snap_s.
        """
        os.makedirs(snapshot_dir, exist_ok=True)
        t = float(start_time_s)
        next_snap = t

        # initial snapshot
        self.save_snapshot(t, snapshot_dir, percent_clip=percent_clip, frame_idx=0)
        next_snap += dt_snap_s
        frame = 1

        while t < Tend_s:
            dt = min(dt_s, Tend_s - t)
            self.step(dt)
            t += dt

            if t + 1e-12 >= next_snap:
                self.save_snapshot(t, snapshot_dir, percent_clip=percent_clip, frame_idx=frame)
                frame += 1
                next_snap += dt_snap_s
                if verbose:
                    print(f"[snap] t = {t/year2s:.3f} yr, frame {frame-1}")

    # ---------- baseline and collapse helpers --------

    def _get_baseline(self, t_seconds: float):
        """
        Returns (rho0_t, v0_t, Lcut_t) from self.baseline_fn or sensible defaults.
        """
        if not callable(self.baseline_fn):
            return float(self.grid.rho0), np.zeros(3, float), np.inf
        out = self.baseline_fn(float(t_seconds)) or {}
        rho0_t = float(out.get("rho0", self.grid.rho0))
        v0_t   = np.asarray(out.get("v0", np.zeros(3, float)), float).reshape(3,)
        Lcut_t = float(out.get("Lcut", np.inf))
        return rho0_t, v0_t, Lcut_t
    
    def _grid_index_first_active(self, Lcut_t: float) -> Optional[int]:
        """First grid index whose scale is <= Lcut_t."""
        for gi in range(self.start, self.start + len(self.levels)):
            if float(self.grid.rlevels[gi]) <= Lcut_t:
                return gi
        return None

    def _sum_branch_collapsing(self, j_finest: int, t_seconds: float):
        # --- Baseline & geometry ---
        rho0_t, v0_t, Lcut_t = self._get_baseline(t_seconds)
        R_affect = 0.5 * float(Lcut_t)   # radius of influence

        s_finest = self.level_slice(len(self.levels) - 1)
        rj = self.pos[s_finest][j_finest]
        inside = (np.linalg.norm(rj - self.root_pos) <= R_affect)

        # Ancestor nodes (global indices) along this finest point's path
        j_global = s_finest.start + j_finest
        branch = self.ancestors(j_global)  # coarse → fine global node indices

        # Super terms are ALWAYS added
        s_super, v_super = self._super_sums()

        if not inside:
            # OUTSIDE cloud: plain OU (grid + super)
            dsum = float(self.delta[branch].sum()) + s_super
            vsum = self.delta_v[branch, :].sum(axis=0) + v_super
            return dsum, vsum

        # INSIDE cloud: filter by Lcut and pin FIRST ACTIVE grid level on the branch
        gi_first = self._grid_index_first_active(Lcut_t)
        if gi_first is None:
            # no active grid level → only super survives
            return float(s_super), v_super

        # ★ locate that grid level INSIDE THIS BRANCH
        ℓ_first = self._branch_loc_for_gridlevel(branch, gi_first)
        if ℓ_first is None:
            # defensive: if the branch doesn't carry that level, fall back to normal
            dsum = float(self.delta[branch].sum()) + s_super
            vsum = self.delta_v[branch, :].sum(axis=0) + v_super
            return dsum, vsum

        # Work on COPIES (no state mutation during evaluation)
        Dd = np.array([ self.delta[idx]   for idx in branch ], float)
        Dv = np.array([ self.delta_v[idx] for idx in branch ], float)

        # Zero out *coarser than cloud* grid contributions
        if ℓ_first > 0:
            Dd[:ℓ_first] = 0.0
            Dv[:ℓ_first] = 0.0

        # ★ Pin the FIRST ACTIVE level so that (grid + super) at the cloud scale
        #    matches the baseline cloud density/velocity:
        #    cumulative(delta_grid) at ℓ_first + s_super = ln(rho0_t/rho_ref) - mu(gi_first)
        #    ⇒ set the increment at ℓ_first to:
        rho_ref = float(self.grid.rho0)
        mu_ifirst = float(np.atleast_1d(self.grid.mu_lnrho)[gi_first]) if hasattr(self.grid, "mu_lnrho") else 0.0

        delta_target_total = np.log(rho0_t / rho_ref) - mu_ifirst          # desired TOTAL at cloud scale
        delta_target_grid  = delta_target_total - float(s_super)           # ★ subtract super (added later)
        Dd[ℓ_first] = delta_target_grid

        # For velocity, make the *total* (grid + super) = v0_t at cloud scale
        Dv[ℓ_first] = np.asarray(v0_t, float) - v_super                    # ★ subtract super vector

        # Finer-than-cloud contributions are left as drawn (they add substructure)
        dsum = float(Dd.sum()) + s_super
        vsum = Dv.sum(axis=0) + v_super
        return dsum, vsum

    def density_at_finest(self, t_seconds: Optional[float] = None):
        level = len(self.levels) - 1
        gi = self.levels[level].grid_index
        rho0_ref = float(self.grid.rho0)
        mu = float(np.atleast_1d(self.grid.mu_lnrho)[gi]) if hasattr(self.grid, "mu_lnrho") else 0.0

        s = self.level_slice(level)
        N = s.stop - s.start

        if not self.collapse:
            # fall back to non-collapse path but still include 0-D supers (already handled there)
            dsum, _ = self.field_at_level(level)
            s_super, _ = self._super_sums()
            log_rho = mu + (dsum + s_super)
            rho = rho0_ref * np.exp(log_rho)
            return rho, log_rho

        if t_seconds is None:
            t_seconds = 0.0

        log_rho = np.empty(N, float)
        for j in range(N):
            dsum, _ = self._sum_branch_collapsing(j, t_seconds)
            log_rho[j] = mu + dsum
        rho = rho0_ref * np.exp(log_rho)
        return rho, log_rho

    def velocity_at_finest(self, t_seconds: Optional[float] = None):
        level = len(self.levels) - 1
        s = self.level_slice(level)
        N = s.stop - s.start

        if not self.collapse:
            _, v = self.field_at_level(level)
            _, v_super = self._super_sums()
            if np.any(v_super):
                v = v + v_super[None, :]
            return v

        if t_seconds is None:
            t_seconds = 0.0

        v = np.empty((N, 3), float)
        for j in range(N):
            _, vsum = self._sum_branch_collapsing(j, t_seconds)
            v[j] = vsum
        return v

    # ---------- visualization (video) ----------

    def _project_xy(self, P, axis="z"):
        """
        Return 2D coords (X, Y) by dropping the chosen LOS axis.
        axis ∈ {"x","y","z"} means LOS is that axis.
        """
        P = np.asarray(P, float)
        if axis == "z":
            return P[:, 0], P[:, 1]
        elif axis == "y":
            return P[:, 0], P[:, 2]
        elif axis == "x":
            return P[:, 1], P[:, 2]
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

    def _column_map_from_points(self,
                            P_cm, rho, r_cell_cm,
                            *,
                            axis="z",
                            nxy=512,
                            fov_cm=None,
                            sigma_fac=0.6,
                            kappa=3.0,
                            to_NH=False,
                            mu_mass=2.33,   # mean molecular weight if converting
                            mH=1.6735575e-24,
                            ):
        """
        Build a single column-density image from unstructured points.

        P_cm      : (N,3) positions
        rho       : (N,)   volume density [g cm^-3]
        r_cell_cm : float or (N,) radius of each cell [cm]
        axis      : LOS ("x","y","z")
        nxy       : pixels per side
        fov_cm    : full field of view (edge-to-edge) [cm]; if None, auto from data
        sigma_fac : Gaussian σ as fraction of r_cell (σ = sigma_fac * r_cell)
        kappa     : kernel half-width in σ units (patch radius = ceil(kappa*σ))
        to_NH     : if True, convert Σ [g cm^-2] → N_H [cm^-2]
        """
        import numpy as np

        P = np.asarray(P_cm, float)
        rho = np.asarray(rho, float)
        if np.isscalar(r_cell_cm):
            r_cell = np.full(P.shape[0], float(r_cell_cm), float)
        else:
            r_cell = np.asarray(r_cell_cm, float)

        # masses per “particle/cell”
        M = rho * (4.0/3.0) * np.pi * r_cell**3  # [g]

        X, Y = self._project_xy(P, axis=axis)

        # choose FOV if not provided
        if fov_cm is None:
            pad = 2.5 * np.max(r_cell) if P.shape[0] else 0.0
            xmin, xmax = float(np.min(X) - pad), float(np.max(X) + pad)
            ymin, ymax = float(np.min(Y) - pad), float(np.max(Y) + pad)
        else:
            # center on data centroid
            xc, yc = float(np.mean(X)), float(np.mean(Y))
            half = 0.5 * float(fov_cm)
            xmin, xmax = xc - half, xc + half
            ymin, ymax = yc - half, yc + half

        # pixel grid
        nx = int(nxy); ny = int(nxy)
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        img = np.zeros((ny, nx), float)

        # deposit mass-conserving Gaussians
        inv_two_pi = 1.0 / (2.0 * np.pi)

        for Xi, Yi, Mi, ri in zip(X, Y, M, r_cell):
            if not np.isfinite(Mi) or Mi <= 0.0:
                continue
            sigma = float(sigma_fac) * float(ri)
            if not np.isfinite(sigma) or sigma <= 0.0:
                continue

            # pixel index of the center
            cx = int(np.floor((Xi - xmin) / dx))
            cy = int(np.floor((Yi - ymin) / dy))
            if cx < -1 or cx > nx or cy < -1 or cy > ny:
                continue  # well outside domain

            # half-window in pixels
            rad_x = int(np.ceil(kappa * sigma / dx))
            rad_y = int(np.ceil(kappa * sigma / dy))
            if rad_x <= 0 or rad_y <= 0:
                continue

            # window bounds (clipped)
            x0, x1 = max(0, cx - rad_x), min(nx - 1, cx + rad_x)
            y0, y1 = max(0, cy - rad_y), min(ny - 1, cy + rad_y)
            if x1 < x0 or y1 < y0:
                continue

            # coordinates of the window pixel centers in cm
            xs = xmin + (np.arange(x0, x1 + 1) + 0.5) * dx
            ys = ymin + (np.arange(y0, y1 + 1) + 0.5) * dy
            XX, YY = np.meshgrid(xs - Xi, ys - Yi, indexing="xy")

            # continuous 2-D Gaussian evaluated at centers
            sig2 = sigma * sigma
            G = inv_two_pi / sig2 * np.exp(-0.5 * (XX*XX + YY*YY) / sig2)

            # mass-conserving deposition ⇒ Σ += M * G   (units: g cm^-2)
            img[y0:y1+1, x0:x1+1] += Mi * G

        if to_NH:
            img = img / (float(mu_mass) * float(mH))  # [cm^-2]

        extent = [xmin, xmax, ymin, ymax]  # for imshow
        return img, extent
    
    def make_column_video(self,
                      snapshot_dir: str = "snapshots_unstructured",
                      outfile: str = "column.mp4",
                      *,
                      axis: str = "z",
                      nxy: int = 512,
                      fov_cm: float = None,
                      sigma_fac: float = 0.6,
                      kappa: float = 3.0,
                      to_NH: bool = True,
                      mu_mass: float = 2.33,
                      mH: float = 1.6735575e-24,
                      fps: int = 12,
                      cmap: str = "inferno",
                      vmin: float = None,
                      vmax: float = None,
                      show_cloud_radius: bool = True):
        """
        Render a time series of column-density maps from saved frames (frame_*.npz).
        Requires that your save_snapshot() stored pos_cm and rho (it does).
        """
        import os, numpy as np, matplotlib.pyplot as plt
        from matplotlib import animation

        frames = sorted(f for f in os.listdir(snapshot_dir) if f.startswith("frame_") and f.endswith(".npz"))
        if not frames:
            raise RuntimeError("No frames found; run evolve() first (it writes frame_XXXX.npz).")

        # Finest-cell radius (constant per run): r_cell = 0.5 * R_finest
        gi_finest = self.levels[-1].grid_index
        r_cell = 0.5 * float(self.grid.rlevels[gi_finest])

        # First pass: determine global color limits if not provided
        if vmin is None or vmax is None:
            vals = []
            for f in frames:
                d = np.load(os.path.join(snapshot_dir, f))
                P = d["pos_cm"]; rho = d["rho"]
                img, _ = self._column_map_from_points(P, rho, r_cell,
                                                    axis=axis, nxy=nxy, fov_cm=fov_cm,
                                                    sigma_fac=sigma_fac, kappa=kappa,
                                                    to_NH=to_NH, mu_mass=mu_mass, mH=mH)
                finite = np.isfinite(img)
                if np.any(finite):
                    lo, hi = np.nanpercentile(img[finite], [2.0, 98.0])
                    vals.append((lo, hi))
            if vals:
                vmin = float(min(lo for lo, _ in vals))
                vmax = float(max(hi for _, hi in vals))
            else:
                vmin, vmax = 0.0, 1.0

        # Set up animation
        fig, ax = plt.subplots(figsize=(6, 5))
        d0 = np.load(os.path.join(snapshot_dir, frames[0]))
        img0, extent = self._column_map_from_points(d0["pos_cm"], d0["rho"], r_cell,
                                                    axis=axis, nxy=nxy, fov_cm=fov_cm,
                                                    sigma_fac=sigma_fac, kappa=kappa,
                                                    to_NH=to_NH, mu_mass=mu_mass, mH=mH)
        im = ax.imshow(np.log10(img0 + 1e-99) if to_NH else np.log10(img0 + 1e-99),
                    extent=extent, origin="lower", cmap=cmap,
                    vmin=np.log10(vmin + 1e-99), vmax=np.log10(vmax + 1e-99))
        cb = plt.colorbar(im, ax=ax, pad=0.01)
        cb.set_label(r"$\log_{10} N_{\mathrm{H}}\,[\mathrm{cm}^{-2}]$" if to_NH else r"$\log_{10}\Sigma\,[\mathrm{g\,cm^{-2}}]$")
        ax.set_xlabel("x [cm]" if axis in ("z","y") else "y [cm]")
        ax.set_ylabel("y [cm]" if axis in ("z","x") else "z [cm]")

        title = ax.set_title("")

        # Optional overlay: projected cloud radius
        ring = None
        def _lcut(t_seconds):
            if not (show_cloud_radius and getattr(self, "collapse", False) and callable(self.baseline_fn)):
                return None
            out = self.baseline_fn(float(t_seconds)) or {}
            return float(out.get("Lcut", np.nan))

        def update(k):
            nonlocal ring
            d = np.load(os.path.join(snapshot_dir, frames[k]))
            P = d["pos_cm"]; rho = d["rho"]; t_s = float(d["t_seconds"][0])
            img, _ = self._column_map_from_points(P, rho, r_cell,
                                                axis=axis, nxy=nxy, fov_cm=fov_cm,
                                                sigma_fac=sigma_fac, kappa=kappa,
                                                to_NH=to_NH, mu_mass=mu_mass, mH=mH)
            im.set_data(np.log10(img + 1e-99))
            title.set_text(f"t = {t_s/3.154e7/1e6:.3f} Myr")

            # overlay cloud radius (circle on the plane)
            if ring is not None:
                try: ring.remove()
                except Exception: pass
                ring = None
            R = _lcut(t_s)/2.
            if np.isfinite(R):
                # circle at root_pos projected
                x0, y0 = self._project_xy(self.root_pos[None,:], axis=axis)
                circ = plt.Circle((x0[0], y0[0]), R, fill=False, color="white", alpha=0.5, linewidth=1.0)
                ring = ax.add_patch(circ)
            return im, title

        ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
        ani.save(os.path.join(snapshot_dir, outfile), writer="ffmpeg", fps=fps)
        plt.close(fig)



    def make_video(
        self,
        snapshot_dir: str = "snapshots_unstructured",
        outfile: str = "evolution.mp4",
        fps: int = 12,
        point_size: float = 8.0,
        alpha: float = 0.9,
        equal_axes: bool = True,
    ):
        """
        Render a 3D animation of points colored by log10(rho).
        """
        # load frames
        frames = sorted([f for f in os.listdir(snapshot_dir) if f.startswith("frame_") and f.endswith(".npz")])
        if not frames:
            raise RuntimeError("No frames found. Run evolve() first.")

        # set up fig
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

        # load first frame to init
        data0 = np.load(os.path.join(snapshot_dir, frames[0]))
        P = data0["pos_cm"]
        log10rho = np.log10(data0["rho"])
        clip = data0["log10rho_clip"]
        sc = ax.scatter(P[:,0], P[:,1], P[:,2], s=point_size, alpha=alpha, c=log10rho, cmap="viridis", vmin=clip[0], vmax=clip[1])
        cb = plt.colorbar(sc, ax=ax, pad=0.01)
        cb.set_label(r"$\log_{10}\,\rho\ \mathrm{[g\,cm^{-3}]}$")

        ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]"); ax.set_zlabel("z [cm]")
        title = ax.set_title(f"t = {float(data0['t_seconds'][0])/year2s:.3f} yr")

        if equal_axes:
            self._set_axes_equal(ax)

        def update(i):
            d = np.load(os.path.join(snapshot_dir, frames[i]))
            P = d["pos_cm"]; rho = d["rho"]; clip = d["log10rho_clip"]
            sc._offsets3d = (P[:,0], P[:,1], P[:,2])
            sc.set_array(np.log10(rho))
            sc.set_clim(clip[0], clip[1])
            title.set_text(f"t = {float(d['t_seconds'][0])/year2s:.3f} yr")
            return sc, title

        ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
        ani.save(os.path.join(snapshot_dir, outfile), writer="ffmpeg", fps=fps)
        plt.close(fig)

    # ---------- plotting (static) ----------

    def plot3d(
        self,
        level: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        s_points: float = 8.0,
        alpha: float = 0.85,
        show: bool = True,
        equal_axes: bool = True,
        annotate_levels: bool = True,
        nptmax: int = 1000
    ) -> plt.Axes:
        """
        Quick 3D scatter of positions. By default plots *all levels*.
        """
        # Create 3D axes
        if ax is None:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")

        if level is None:
            # plot all levels with level-based marker sizes
            for ℓ, v in enumerate(self.levels):
                s = self.level_slice(ℓ)
                pts = self.pos[s]
                iinc = np.arange(len(pts), dtype=int)
                if len(pts)>nptmax:
                    iinc = np.random.choice(iinc, size=nptmax, replace=False)

                ax.scatter(pts[iinc, 0], pts[iinc, 1], pts[iinc, 2],
                           s=s_points * (1.5 ** (len(self.levels) - ℓ - 1)),
                           alpha=alpha, label=f"level {ℓ}")
        else:
            v = self.levels[level]
            s = self.level_slice(level)
            pts = self.pos[s]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s_points, alpha=alpha, label=f"level {level}")

        if equal_axes:
            self._set_axes_equal(ax)

        if annotate_levels:
            ax.legend(loc="upper right", frameon=False)

        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_zlabel("z [cm]")

        if show:
            plt.show()

        return ax

    @staticmethod
    def _set_axes_equal(ax: plt.Axes):
        xs = ax.get_xlim3d(); ys = ax.get_ylim3d(); zs = ax.get_zlim3d()
        x_range = xs[1] - xs[0]; y_range = ys[1] - ys[0]; z_range = zs[1] - zs[0]
        max_range = max(x_range, y_range, z_range)
        x_mid = 0.5 * (xs[0] + xs[1]); y_mid = 0.5 * (ys[0] + ys[1]); z_mid = 0.5 * (zs[0] + zs[1])
        ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
        ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
        ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])


if __name__=='__main__':
    import excursion as exc

    # Build your grid as usual (must be geometric with ratio 2^(-1/3))
    grid = exc.trajectory_grid(rmax=200.0 * 3.086e18, rmin=0.001 * 3.086e18, drfact=2.0**(-1.0/3.0))
    # Choose level-0 scale ≈ 1 pc
    Rcloud_cm = 2.0 * pc2cm
    start_idx = int(np.argmin(np.abs(grid.rlevels - 2.*Rcloud_cm)))
    print("Chosen level-0 scale:",
        grid.rlevels[start_idx]/pc2cm, "pc  (index =", start_idx, ")")

    # Tie collapse baseline to that level-0 scale (R0 ≈ 1 pc)
    field = BinaryUnstructuredField(
        grid,
        start_grid_index=start_idx,   # <-- key line
        seed=123,
        verbose=True,
        collapse=True
)


    # Evolve & save
    Tend = 10.0e6 * year2s
    dt    = 0.01 * float(np.min(grid.tau_R))
    dt_sn = 0.02e6 * year2s
    field.evolve(Tend, dt, dt_sn, snapshot_dir="snap_super", verbose=True)

    # after you’ve run field.evolve(...):
    field.make_column_video(
        snapshot_dir="snap_super",      # where your frame_*.npz live
        outfile="column.mp4",
        axis="z",                       # LOS
        nxy=512,
        sigma_fac=0.6, kappa=3.0,       # kernel width & stencil radius
        to_NH=True,                     # show as N_H [cm^-2]
        fps=15,
        vmin=1e14, vmax=1e20
    )
    exit()

    # Make the video
    field.make_video(snapshot_dir="snap_collapse", outfile="evolution.mp4", fps=15)
    
    # Plot everything
    field.plot3d()

    # Evolve for 0.5 Myr, snapshots every 0.01 Myr
    Tend = 0.5e6 * year2s
    dt    = 0.0025 * float(np.min(grid.tau_R))
    dt_sn = 0.01e6 * year2s
    field.evolve(Tend, dt, dt_sn, snapshot_dir="snap_u", verbose=True)

    # Make the video
    field.make_video(snapshot_dir="snap_u", outfile="evolution.mp4", fps=15)

    # Get final densities/velocities at the finest level (in-memory, no file I/O)
    rho, log_rho = field.density_at_level()      # default: finest
    vel = field.velocity_at_level()              # (Nfinest, 3)

    # Evolve once
    dt = 0.05 * float(np.min(grid.tau_R))
    field.step(dt_seconds=dt)

    # Aggregated deltas at the deepest level
    deep_level = len(field.levels) - 1
    delta_sum, delta_v_sum = field.field_at_level(deep_level)
    print(delta_sum.shape, delta_v_sum.shape)   # (N_deep,), (N_deep, 3)
