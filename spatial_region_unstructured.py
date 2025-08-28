# binary_unstructured_field.py  (extended)
from __future__ import annotations
from typing import Optional, List, Tuple, Callable, Dict, Any

from dataclasses import dataclass, field, asdict
import os, json, uuid
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
class CloudRecord:
    cloud_id: str
    level: int
    grid_index: int
    R0_cm: float
    pos_cm: np.ndarray          # initial position at spawn  (3,)
    v0_cms: np.ndarray          # constant velocity vector  (3,)
    rho_init: float
    t_init_s: float
    baseline_args: Dict[str, Any]
    baseline_info: Dict[str, Any] = field(default_factory=dict)
    baseline_fn: Optional[Callable[[float], dict]] = None

    # --- live / derived state (not required on disk) ---
    pos_now_cm: Optional[np.ndarray] = None
    t_now_s: Optional[float] = None

    def position_at(self, t_seconds: float) -> np.ndarray:
        dt = float(t_seconds) - float(self.t_init_s)
        return np.asarray(self.pos_cm, float) + np.asarray(self.v0_cms, float) * dt


@dataclass
class LevelView:
    level: int
    n: int
    offset: int
    grid_index: int

def _json_default(obj):
    import numpy as np
    # numpy scalars
    if isinstance(obj, np.generic):
        return obj.item()
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # fall back: stringify the type (avoids crashes on complex objects like grids/functions)
    return f"<{obj.__class__.__name__}>"



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

        self.node_level = np.empty(self.N_total, dtype=np.int32)
        self.grid_index = np.empty(self.N_total, dtype=np.int32)
        for v in self.levels:
            s = slice(v.offset, v.offset + v.n)
            self.node_level[s] = int(v.level)
            self.grid_index[s] = int(v.grid_index)

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


        self.clouds: list[CloudRecord] = []

        self.spawn_check_factor = 0.1   # every 0.1 * tau_R by default
        self._next_cloud_check: dict[int, float] = {}  # level -> next check time (s)

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


    def _branch_loc_for_gridlevel(self, branch_idx, target_gi):
        """
        branch_idx : 1D array of global node indices along this node's ancestors (coarse→fine)
        target_gi  : integer grid level (gi_first from Lcut)
        returns: integer l such that branch_idx[l] maps to target_gi on this branch, else None
        """
        gi_on_branch = self.grid_index[branch_idx]  # same length as branch_idx
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
    def density_at_level(self, level: int, t_seconds: float | None = None):
        if t_seconds is None:
            t_seconds = 0.0
        sL = self.level_slice(level)
        idxs = np.arange(sL.start, sL.stop)
        gi_level = self.levels[level].grid_index
        rho0_ref = float(self.grid.rho0)
        mu_level = float(np.atleast_1d(self.grid.mu_lnrho)[gi_level]) if hasattr(self.grid, "mu_lnrho") else 0.0

        log_rho = np.empty(idxs.size, float)
        for k, jg in enumerate(idxs):
            dsum, _ = self._sum_branch_collapsing_node(int(jg), t_seconds)
            log_rho[k] = mu_level + dsum
        rho = rho0_ref * np.exp(log_rho)
        return rho, log_rho

    def velocity_at_level(self, level: int, t_seconds: float | None = None):
        if t_seconds is None:
            t_seconds = 0.0
        sL = self.level_slice(level)
        idxs = np.arange(sL.start, sL.stop)

        v = np.empty((idxs.size, 3), float)
        for k, jg in enumerate(idxs):
            _, vsum = self._sum_branch_collapsing_node(int(jg), t_seconds)
            v[k, :] = vsum
        return v

    

    # --------- clouds ----------
    def _init_cloud_check_schedule(self, t0: float = 0.0):
        """
        Initialize next-check times for all logical levels at t0 + f * tau_R.
        """
        self._next_cloud_check = {}
        for lv in self.levels:
            gi = int(lv.grid_index)
            tau = float(self.grid.tau_R[gi])
            self._next_cloud_check[int(lv.level)] = float(t0) + self.spawn_check_factor * tau


    def _cloud_radius_at(self, rec: CloudRecord, t_seconds: float) -> float:
        """Return R_affect(t)=Lcut(t)/2 for a stored cloud record; NaN if unknown."""
        if rec.baseline_fn is None:
            return float("nan")
        out = rec.baseline_fn(float(t_seconds)) or {}
        Lcut = float(out.get("Lcut", np.nan))
        return 0.5 * Lcut
    
    def _cloud_center_at(self, rec, t_seconds: float) -> np.ndarray:
        """Ballistic center at time t."""
        return rec.position_at(t_seconds)

    def _cloud_baseline_at(self, rec, t_seconds: float):
        """
        Return (rho0_t, v0_t(3,), Lcut_t) for this cloud at global time t,
        evaluating its baseline at (t - t_init).
        """
        if rec.baseline_fn is None:
            return None, None, None
        tau = float(t_seconds) - float(rec.t_init_s)
        out = rec.baseline_fn(tau) or {}
        rho0_t = float(out.get("rho0", np.nan))
        v0_t   = np.asarray(out.get("v0", np.zeros(3, float)), float).reshape(3,)
        Lcut_t = float(out.get("Lcut", np.nan))
        return rho0_t, v0_t, Lcut_t

    def _cloud_R_affect(self, rec, t_seconds: float) -> float:
        """R_affect = Lcut/2 at time t for this cloud (time-shifted baseline)."""
        _, _, Lcut_t = self._cloud_baseline_at(rec, t_seconds)
        return 0.5 * float(Lcut_t) if np.isfinite(Lcut_t) else float("nan")

    def _smallest_enclosing_cloud(self, pos_cm: np.ndarray, t_seconds: float):
        """
        Return (rec, R_affect_cm) for the smallest cloud whose sphere at time t contains pos_cm,
        or (None, None) if none.
        """
        best = None
        best_R = np.inf
        for rec in getattr(self, "clouds", []):
            c = self._cloud_center_at(rec, t_seconds)
            R = self._cloud_R_affect(rec, t_seconds)
            if not np.isfinite(R):
                continue
            if np.linalg.norm(pos_cm - c) <= R and R < best_R:
                best, best_R = rec, R
        return best, (best_R if np.isfinite(best_R) else None)


    '''def _node_in_any_cloud(self, pos_cm: np.ndarray, t_seconds: float) -> bool:
        """True if pos_cm lies inside any cloud's (moving) sphere of radius Lcut/2 at time t."""
        if not hasattr(self, "clouds"):
            return False
        for rec in self.clouds:
            # current center
            c = rec.position_at(t_seconds)
            # current influence radius
            if rec.baseline_fn is None:
                continue
            out = rec.baseline_fn(float(t_seconds)) or {}
            Lcut = float(out.get("Lcut", np.nan))
            R = 0.5 * Lcut
            if np.isfinite(R) and np.linalg.norm(pos_cm - c) <= R:
                return True
        return False'''

    def _save_cloud_db(self, snapshot_dir: str):
        path = os.path.join(snapshot_dir, "cloud_db.jsonl")
        with open(path, "w") as f:
            for rec in self.clouds:
                # re-derive now if not set
                pos_now = rec.pos_now_cm if rec.pos_now_cm is not None else rec.position_at(rec.t_now_s or rec.t_init_s)
                item = {
                    "cloud_id": rec.cloud_id,
                    "level": int(rec.level),
                    "grid_index": int(rec.grid_index),
                    "R0_cm": float(rec.R0_cm),
                    "pos_cm_init": [float(x) for x in rec.pos_cm],
                    "pos_cm_now":  [float(x) for x in pos_now],
                    "v0_cms": [float(x) for x in rec.v0_cms],
                    "rho_init": float(rec.rho_init),
                    "t_init_s": float(rec.t_init_s),
                    "t_now_s":   float(rec.t_now_s if rec.t_now_s is not None else rec.t_init_s),
                    "baseline_args": rec.baseline_args,
                    "baseline_info": rec.baseline_info,
                }
                f.write(json.dumps(item, default=_json_default) + "\n")


    
    def update_cloud_positions(self, t_seconds: float, *, snapshot_dir: Optional[str] = None, save_state: bool = True):
        """
        Ballistic update for all clouds: x(t)=x0+v0*(t-t_init).
        Optionally persists a compact state file per snapshot.
        """
        if not hasattr(self, "clouds") or len(self.clouds) == 0:
            return

        # update in-memory
        for rec in self.clouds:
            rec.pos_now_cm = rec.position_at(t_seconds)
            rec.t_now_s = float(t_seconds)

        # optionally persist a light-weight state blob for this time
        if save_state and snapshot_dir:
            ids   = [rec.cloud_id for rec in self.clouds]
            level = np.array([rec.level for rec in self.clouds], dtype=np.int32)
            gidx  = np.array([rec.grid_index for rec in self.clouds], dtype=np.int32)
            Xnow  = np.stack([rec.pos_now_cm if rec.pos_now_cm is not None else rec.position_at(t_seconds)
                            for rec in self.clouds], axis=0)
            V0    = np.stack([rec.v0_cms for rec in self.clouds], axis=0)
            # optional: current influence radius = Lcut/2 from each cloud’s baseline
            R_aff = []
            for rec in self.clouds:
                if rec.baseline_fn is None:
                    R_aff.append(np.nan)
                else:
                    out = rec.baseline_fn(float(t_seconds)) or {}
                    Lcut = float(out.get("Lcut", np.nan))
                    R_aff.append(0.5 * Lcut)
            R_aff = np.asarray(R_aff, float)

            # name state by time (or let caller pass an index)
            fname = f"cloud_state_t{int(round(t_seconds)):012d}.npz"
            path  = os.path.join(snapshot_dir, fname)
            np.savez_compressed(path,
                                ids=np.array(ids, dtype=object),
                                level=level, grid_index=gidx,
                                pos_now_cm=Xnow, v0_cms=V0,
                                R_affect_cm=R_aff,
                                t_seconds=float(t_seconds))


    def check_and_spawn_clouds(self, t_seconds: float, levels: Optional[list[int]] = None, verbose: bool = False):
        """
        Scan levels from largest→smallest; if ρ(node) > ρ_crit(level) and node is not
        inside an existing cloud, spawn a new cloud at that node.
        """
        if levels is None:
            levels = list(range(len(self.levels)))  # 0 .. L-1  (0 is largest scale)

        for level in levels:
            gi = int(self.levels[level].grid_index)
            rho_crit = float(self.grid.rhocs[gi])

            # evaluate field at this level (our earlier general API)
            rho, _ = self.density_at_level(level, t_seconds=t_seconds)    # (N_level,)
            v   = self.velocity_at_level(level, t_seconds=t_seconds)      # (N_level,3)
            sL  = self.level_slice(level)
            P   = self.pos[sL]                                            # (N_level,3)

            unstable_idx = np.nonzero(rho > rho_crit)[0]
            if unstable_idx.size == 0:
                continue

            # iterate nodes; skip those already inside any existing cloud at t_seconds
            for k in unstable_idx:
                pos_k = P[k]
                rec_hit, _ = self._smallest_enclosing_cloud(pos_k, t_seconds)
                if rec_hit is not None:
                    continue 

                # spawn new cloud
                R0_cm = 0.5 * float(self.grid.rlevels[gi])   # radius = 1/2 the collapse scale
                v0    = np.asarray(v[k], float)
                rho0  = rho_crit

                baseline_fn = None
                info = {}
                if gcr is not None and hasattr(gcr, "build_cloud_baseline_fn"):
                    try:
                        baseline_fn, info = gcr.build_cloud_baseline_fn(target_radius_cm=R0_cm)
                    except Exception:
                        pass

                rec = CloudRecord(
                    cloud_id=str(uuid.uuid4()),
                    level=int(level),
                    grid_index=gi,
                    R0_cm=float(R0_cm),
                    pos_cm=np.asarray(pos_k, float),
                    v0_cms=v0,
                    rho_init=float(rho0),
                    t_init_s=float(t_seconds),
                    baseline_args={"target_radius_cm": float(R0_cm)},
                    baseline_info=info,
                    baseline_fn=baseline_fn
                )
                self.clouds.append(rec)

                if verbose:
                    pc2cm = 3.086e18
                    print(f"[spawn] t={t_seconds/3.154e13:.3f} Myr  level={level}  "
                        f"R0={R0_cm/pc2cm:.3f} pc  rho_c={rho_crit:.3e}  id={rec.cloud_id}")


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
        overwrite: bool = False, 
    ):
        """
        Time-march OU fields, writing snapshots of the finest level every dt_snap_s.
        """
        
        os.makedirs(snapshot_dir, exist_ok=True)

        # If frames already exist and overwrite=False, don't rerun; just return.
        existing_frames = sorted(f for f in os.listdir(snapshot_dir)
                                if f.startswith("frame_") and f.endswith(".npz"))
        if existing_frames and not overwrite:
            if verbose:
                print(f"[evolve] Found {len(existing_frames)} frames in '{snapshot_dir}'. "
                    "Skipping evolution (set overwrite=True to regenerate).")
            return


        t = float(start_time_s)
        next_snap = t
        # initialize cloud-check schedule (first checks happen at t0 + 0.1*tau_R)
        self._init_cloud_check_schedule(t)

        # initial snapshot
        self.save_snapshot(t, snapshot_dir, percent_clip=percent_clip, frame_idx=0)
        next_snap += dt_snap_s
        frame = 1

        while t < Tend_s:
            dt = min(dt_s, Tend_s - t)
            # advance OU
            dt = min(dt_s, Tend_s - t)
            self.step(dt)
            t += dt

            # update cloud positions (state on disk matched to this snapshot_dir)
            self.update_cloud_positions(t, snapshot_dir=snapshot_dir, save_state=True)

            # throttle cloud spawning checks: only levels whose next_check <= t
            eligible_levels = [lv.level for lv in self.levels
                            if self._next_cloud_check.get(lv.level, -np.inf) <= t]
            if eligible_levels:
                self.check_and_spawn_clouds(t_seconds=t, levels=eligible_levels, verbose=False)
                # bump their next-check times by 0.1 * tau_R(level)
                for lv in self.levels:
                    if lv.level in eligible_levels:
                        gi = int(lv.grid_index)
                        tau = float(self.grid.tau_R[gi])
                        self._next_cloud_check[lv.level] = t + self.spawn_check_factor * tau

            t += dt

            if t + 1e-12 >= next_snap:
                self.save_snapshot(t, snapshot_dir, percent_clip=percent_clip, frame_idx=frame)
                self._save_cloud_db(snapshot_dir)
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

    '''def _sum_branch_collapsing(self, j_finest: int, t_seconds: float):
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
        return dsum, vsum'''
    
    def _sum_branch_collapsing_node(self, node_global_idx: int, t_seconds: float):
        """
        Collapse-aware sum for an arbitrary node, treating this node as if it were 'finest' for the call.
        Picks the smallest enclosing cloud at time t (if any) and applies filtering with that cloud's
        time-shifted baseline. Super-scales are always added.
        Returns (dsum, vsum) of increments over the node's branch + super.
        """
        # branch and position of this node
        node_global_idx = int(node_global_idx)
        branch = self.ancestors(node_global_idx)                  # coarse → fine global node indices
        pos_node = self.pos[node_global_idx]

        # Unconditional super sums
        s_super = float(self.super_delta.sum()) if (getattr(self, "evolve0d", False) and self.super_grid_indices.size) else 0.0
        v_super = self.super_delta_v.sum(axis=0) if (getattr(self, "evolve0d", False) and self.super_grid_indices.size) else np.zeros(3, float)

        # Identify the smallest enclosing cloud at time t
        cloud_rec, R_aff = self._smallest_enclosing_cloud(pos_node, t_seconds)
        if cloud_rec is None:
            # OUTSIDE any cloud → plain OU grid + super
            dsum = float(self.delta[branch].sum()) + s_super
            vsum = self.delta_v[branch, :].sum(axis=0) + v_super
            return dsum, vsum

        # Cloud baseline at (t - t_init)
        rho0_t, v0_t, Lcut_t = self._cloud_baseline_at(cloud_rec, t_seconds)
        if not np.isfinite(Lcut_t) or not np.isfinite(rho0_t):
            # fallback: treat as outside if baseline is invalid
            dsum = float(self.delta[branch].sum()) + s_super
            vsum = self.delta_v[branch, :].sum(axis=0) + v_super
            return dsum, vsum

        # Find the first active grid index gi_first for this cloud scale
        gi_first = self._grid_index_first_active(Lcut_t)
        if gi_first is None:
            # nothing active on grid → only super contributes
            return float(s_super), v_super

        # Map that grid level onto THIS branch
        l_first = self._branch_loc_for_gridlevel(branch, gi_first)
        if l_first is None:
            # This branch doesn't include the cloud scale (node is coarser than cloud).
            # With "treat this node as finest", we cannot pin a level that isn't present → only super stays.
            return float(s_super), v_super

        # Copies (do not mutate state)
        Dd = np.array([ self.delta[idx]   for idx in branch ], float)
        Dv = np.array([ self.delta_v[idx] for idx in branch ], float)

        # Zero out contributions from coarser-than-cloud along this branch
        if l_first > 0:
            Dd[:l_first] = 0.0
            Dv[:l_first] = 0.0

        # Pin the first active grid level so TOTAL (grid + super) at cloud scale matches the cloud baseline
        rho_ref = float(self.grid.rho0)
        mu_ifirst = float(np.atleast_1d(self.grid.mu_lnrho)[gi_first]) if hasattr(self.grid, "mu_lnrho") else 0.0
        delta_target_total = np.log(rho0_t / rho_ref) - mu_ifirst
        Dd[l_first] = delta_target_total - s_super     # subtract super; it's added below
        Dv[l_first] = np.asarray(v0_t, float) - v_super

        # Sum (grid + super)
        dsum = float(Dd.sum()) + s_super
        vsum = Dv.sum(axis=0) + v_super
        return dsum, vsum


    def density_at_finest(self, t_seconds: float | None = None):
        return self.density_at_level(level=len(self.levels)-1, t_seconds=t_seconds)

    def velocity_at_finest(self, t_seconds: float | None = None):
        return self.velocity_at_level(level=len(self.levels)-1, t_seconds=t_seconds)

    # ---------- visualization (video) ----------
    
    def _project_xy(self, X: np.ndarray, axis: str = "z"):
        """
        Project 3D coords X:(N,3) to 2D for the chosen LOS axis.
        axis='z' -> (x,y); 'x'->(y,z); 'y'->(x,z)
        """
        X = np.asarray(X, float)
        if axis == "z":  return X[:, 0], X[:, 1]
        if axis == "x":  return X[:, 1], X[:, 2]
        if axis == "y":  return X[:, 0], X[:, 2]
        raise ValueError("axis must be 'x','y','z'")

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
    
    def make_column_video(self, snapshot_dir="snapshots_unstructured", outfile="column.mp4",
                      *, axis="z", nxy=512, fov_cm=None, sigma_fac=0.6, kappa=3.0,
                      to_NH=True, mu_mass=2.33, mH=1.6735575e-24, fps=12, cmap="inferno",
                      vmin=None, vmax=None, show_cloud_radius=True, draw_cloud_rings=True,
                      ring_kwargs=None, show_ids=False, save_R_series=True,
                      R_series_filename="cloud_radius_vs_time.npy", label_radius_pc=True,
                      id_kwargs=None):
        import os, numpy as np, matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.patches import Circle  # <-- needed for ring overlays
        from consts_defaults import year2s, pc2cm

        frames = sorted(f for f in os.listdir(snapshot_dir) if f.startswith("frame_") and f.endswith(".npz"))
        if not frames:
            raise RuntimeError("No frames found; run evolve() first (it writes frame_XXXX.npz).")

        gi_finest = self.levels[-1].grid_index
        r_cell = 0.5 * float(self.grid.rlevels[gi_finest])

        # global color limits if not provided
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

        # --- single figure/axes ---
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        d0 = np.load(os.path.join(snapshot_dir, frames[0]))
        img0, extent0 = self._column_map_from_points(d0["pos_cm"], d0["rho"], r_cell,
                                                    axis=axis, nxy=nxy, fov_cm=fov_cm,
                                                    sigma_fac=sigma_fac, kappa=kappa,
                                                    to_NH=to_NH, mu_mass=mu_mass, mH=mH)
        # if fov was None, freeze it from first frame for consistent axes
        if fov_cm is None:
            fov_cm = float(extent0[1] - extent0[0])  # assume square
        ax.set_aspect("equal")

        im = ax.imshow(np.log10(img0 + 1e-99),
                    extent=extent0, origin="lower", cmap=cmap,
                    vmin=np.log10(vmin + 1e-99), vmax=np.log10(vmax + 1e-99))
        cb = plt.colorbar(im, ax=ax, pad=0.01)
        cb.set_label(r"$\log_{10} N_{\mathrm{H}}\,[\mathrm{cm}^{-2}]$" if to_NH else r"$\log_{10}\Sigma\,[\mathrm{g\,cm^{-2}}]$")
        ax.set_xlabel("x [cm]" if axis in ("z","y") else "y [cm]")
        ax.set_ylabel("y [cm]" if axis in ("z","x") else "z [cm]")
        title = ax.set_title("")

        ring_kwargs = dict(ec="cyan", lw=1.4, fill=False, alpha=0.9) if ring_kwargs is None else ring_kwargs
        id_kwargs   = dict(color="cyan", fontsize=8) if id_kwargs is None else id_kwargs
        ring_artists = {}
        id_artists = {}

        # use the file you actually wrote in save_snapshot()
        times_path = os.path.join(snapshot_dir, "times.npy")
        t_array = np.load(times_path) if os.path.exists(times_path) else None

        def _update_cloud_overlays(t_s: float):
            if not draw_cloud_rings or not getattr(self, "clouds", None):
                for c in list(ring_artists.values()): c.remove()
                ring_artists.clear()
                for txt in list(id_artists.values()): txt.remove()
                id_artists.clear()
                return
            visible_now = set()
            for rec in self.clouds:
                if t_s < float(rec.t_init_s):  # not formed yet
                    continue
                C3 = self._cloud_center_at(rec, t_s)
                R  = self._cloud_R_affect(rec, t_s)
                if not np.isfinite(R) or R <= 0.0:
                    continue
                cx, cy = self._project_xy(C3[None, :], axis=axis)
                cid = rec.cloud_id
                visible_now.add(cid)
                if cid in ring_artists:
                    r = ring_artists[cid]
                    r.center = (float(cx[0]), float(cy[0])); r.set_radius(float(R))
                else:
                    r = Circle((float(cx[0]), float(cy[0])), float(R), **ring_kwargs)
                    ax.add_patch(r); ring_artists[cid] = r
                if show_ids:
                    label = f"{cid[:6]}"
                    if label_radius_pc:
                        label += f"\nR={R/pc2cm:.2f} pc"
                    if cid in id_artists:
                        txt = id_artists[cid]
                        txt.set_position((float(cx[0]), float(cy[0]))); txt.set_text(label)
                    else:
                        txt = ax.text(float(cx[0]), float(cy[0]), label, ha="center", va="center", **id_kwargs)
                        id_artists[cid] = txt
            for cid in list(ring_artists.keys()):
                if cid not in visible_now:
                    ring_artists[cid].remove(); del ring_artists[cid]
            for cid in list(id_artists.keys()):
                if cid not in visible_now:
                    id_artists[cid].remove(); del id_artists[cid]

        def update(i: int):
            d = np.load(os.path.join(snapshot_dir, frames[i]))
            P = d["pos_cm"]; rho = d["rho"]
            # lock FOV to the first frame by passing fov_cm
            img, extent = self._column_map_from_points(P, rho, r_cell,
                                                    axis=axis, nxy=nxy, fov_cm=fov_cm,
                                                    sigma_fac=sigma_fac, kappa=kappa,
                                                    to_NH=to_NH, mu_mass=mu_mass, mH=mH)
            im.set_data(np.log10(img + 1e-99))
            # keep extent fixed to extent0 for stable axes (omit next line if you want per-frame autos)
            im.set_extent(extent0)

            t_s = float(t_array[i]) if (t_array is not None and i < len(t_array)) else 0.0
            title.set_text(f"t = {t_s/year2s/1e6:.2f} Myr")

            _update_cloud_overlays(t_s)
            return [im, title] + list(ring_artists.values()) + (list(id_artists.values()) if show_ids else [])

        n_frames = len(frames)
        ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
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
