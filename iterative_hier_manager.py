# iterative_hier_manager.py  — bubble-up version
from __future__ import annotations
import os, json, uuid, warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
import excursion as exc
from consts_defaults import year2s, Gcgs as G

try:
    import generate_cloudrho0 as gcr  # build_cloud_baseline_fn
except Exception:
    gcr = None


def _shifted(baseline_fn: Callable[[float], dict], t_now: float, t_star: float):
    """Return f'(t) = f(t - (t_now - t_star)) so that f'(t_now) == f(t_star)."""
    shift = float(t_now) - float(t_star)
    def fprime(ts: float):
        return baseline_fn(float(ts) - shift)
    return fprime


def _robust_child_index(val, current_level: int, nlevels: int) -> Optional[int]:
    if val is None:
        return None
    try:
        i = int(val)
    except Exception:
        return None
    if 0 <= i < nlevels and i > current_level:
        return i
    return None

def _choose_volume_bit(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2))  # 0 or 1

def _rand_in_ball(radius: float, rng: np.random.Generator) -> np.ndarray:
    # draw uniform in 3D ball of given radius
    # radius * u^{1/3} with random unit direction
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    if n == 0:  
        v = np.array([1.0, 0.0, 0.0]); n = 1.0
    v = v / n
    u = rng.random()
    r = radius * (u ** (1.0/3.0))
    return r * v


@dataclass
class SnapShard:
    node_id: str
    dirpath: str
    parent_id: Optional[str] = None
    inherit_until_k: Optional[int] = None
    def meta_path(self): return os.path.join(self.dirpath, "meta.json")
    def write_meta(self):
        os.makedirs(self.dirpath, exist_ok=True)
        with open(self.meta_path(), "w") as f:
            json.dump(dict(node_id=self.node_id,
                           parent_id=self.parent_id,
                           inherit_until_k=self.inherit_until_k), f, indent=2)


@dataclass
class LevelEntry:
    level: int
    baseline_fn: Callable[[float], dict]
    baseline_info: Dict
    shard: SnapShard
    k_next: int = 0
    bit: Optional[int] = None         # 0/1 choice taken at this level relative to parent
    pos_cm: Optional[np.ndarray] = None  # center position at this level (3,)


class IterativeHierarchicalManager:
    """
    Iterative, smallest-scale-following manager with post-collapse bubble-up:

    - Push when a deeper unstable level appears.
    - Pop when the active level's baseline hits the free-fall threshold.
    - At each moment, only the active shell is pinned; all smaller scales OU-evolve.
    - Densities always normalized by grid.rho0; baseline is only for Lcut + timing.
    - Node snapshots store only post-divergence frames with parent linkage.
    """
    def __init__(self,
                 grid: Any,
                 dt_factor: float,
                 root_level: int,
                 root_baseline_fn: Callable[[float], dict],
                 save_root: str,
                 *,
                 seed: Optional[int] = None,
                 tff_threshold_yr: float = 1.0e4):
        self.grid = grid
        self.drfact = self.grid.dr
        self._assert_geometric_halving()
        self.dt_factor = float(dt_factor)
        self.seed = seed
        self.tff_thresh_s = float(tff_threshold_yr) * year2s

        # Trajectory (single state, reused across stack)
        self.traj = exc.trajectory(grid=self.grid, dt_factor=self.dt_factor,
                                   baseline_fn=root_baseline_fn, t0=0.0, seed=seed)
        
        # Root node & stack
        root_id = "node_root"
        root_dir = save_root

        self.save_root = save_root
        self._last_leaf = None

        shard = SnapShard(node_id=root_id, dirpath=root_dir, parent_id=None, inherit_until_k=None)
        shard.write_meta()

        # Align root baseline so collapse start (t_star) lands at t=0
        if gcr is not None:
            # Try to infer an aligned baseline with builder meta
            try:
                # If caller passed us a builder-made baseline, they probably know its info;
                # we can rebuild to get t_star for alignment = 0
                R0 = float(self.grid.rlevels[int(root_level)])
                btmp, info = gcr.build_cloud_baseline_fn(target_radius_cm=R0)
                t_star = float(info.get("t_star", 0.0))
                root_bfn = _shifted(root_baseline_fn, t_now=0.0, t_star=t_star)
                info0 = info
            except Exception:
                root_bfn = root_baseline_fn
                info0 = {}
        else:
            root_bfn = root_baseline_fn
            info0 = {}

        self.rng = np.random.default_rng(seed)
        root_pos = np.zeros(3, dtype=float)  # or pass in a starting pos
        self.stack: List[LevelEntry] = [LevelEntry(
            level=int(root_level),
            baseline_fn=root_bfn,
            baseline_info=info0,
            shard=shard,
            k_next=0,
            bit=None,
            pos_cm=root_pos,
        )]
        self.path_bits: List[int] = []        # from root to current leaf (bits for levels below root)
        self.pos_by_level: Dict[int, np.ndarray] = {int(root_level): root_pos.copy()}

        # Make sure the trajectory uses the active baseline
        self.traj.baseline_fn = self.stack[-1].baseline_fn
        self.t_now = 0.0

        self._last_leaf = self.stack[-1]

        self.active_index = int(root_level)   # authoritative pinned shell
        self.collapsed_levels: set[int] = set()


    
    @property
    def leaf(self) -> LevelEntry:
        """
        Backwards-compat alias for the 'current leaf' node.
        If the stack is empty (all levels collapsed and popped),
        returns the last active LevelEntry for compatibility.
        """
        if self.stack:
            return self.stack[-1]
        if self._last_leaf is not None:
            return self._last_leaf
        # fabricate a minimal stub if absolutely necessary
        raise AttributeError("No leaf available: stack empty and no last-leaf fallback recorded.")

    @property
    def leaf_dir(self) -> str:
        """Convenience: directory of the current (or last) leaf's shard."""
        try:
            return self.leaf.shard.dirpath
        except Exception:
            return self.save_root


    # --------- helpers ---------
    # in excursion.trajectory_grid.__init__ (after self.setup_grid / setup_grid_manual)
    def _assert_geometric_halving(self, tol=5e-3):
        target = 2.0 ** (-1.0/3.0)
        # if using a constant drfact
        if np.isscalar(getattr(self, 'drfact', None)):
            if not np.isclose(self.drfact, target, rtol=tol, atol=0):
                raise AssertionError(f"drfact={self.drfact:.6f} ≠ 2^(-1/3)≈{target:.6f}")
        else:
            # manual grid: check pairwise ratios
            ratios = self.grid.rlevels[1:] / self.grid.rlevels[:-1]
            if not np.allclose(ratios, target, rtol=tol, atol=0):
                raise AssertionError("Manual rlevels are not geometric with ratio 2^(-1/3).")


    def _build_baseline_for_level(self, level: int, t_now: float) -> Tuple[Callable[[float], dict], Dict]:
        R = float(self.grid.rlevels[level])
        if gcr is None or not hasattr(gcr, "build_cloud_baseline_fn"):
            warnings.warn("No generate_cloudrho0; reusing current baseline.")
            return self.stack[-1].baseline_fn, {}
        bfn, info = gcr.build_cloud_baseline_fn(target_radius_cm=R)
        t_star = float(info.get("t_star", 0.0))
        return _shifted(bfn, t_now=t_now, t_star=t_star), dict(info)

    def _active_level(self) -> int:
        return self.stack[-1].level

    def _check_ff_collapse(self, bfn: Callable[[float], dict], t: float) -> bool:
        try:
            rho0 = float(bfn(float(t)).get("rho0", np.nan))
            if not np.isfinite(rho0) or rho0 <= 0:
                return False
            tff = np.sqrt(3.0*np.pi / (32.0 * G * rho0))
            return bool(tff < self.tff_thresh_s)
        except Exception:
            return False
        

    def _save_snapshot(self, level_entry: LevelEntry, t: float):
        traj = self.traj
        base = level_entry.baseline_fn(float(t)) if level_entry.baseline_fn else {}
        Lcut_t = float(base.get("Lcut", np.inf))

        # filtered increments from the trajectory
        try:
            Dd_f = traj._filtered_Ddelta(traj.Ddelta, t_seconds=float(t), write_cache=False)
        except Exception:
            mask = (traj.grid.rlevels <= Lcut_t)
            Dd_f = np.zeros_like(traj.Ddelta, float); Dd_f[mask] = traj.Ddelta[mask]
        try:
            Dv_f = traj._filtered_Dv(traj.Dv, t_seconds=float(t), write_cache=False)
        except Exception:
            mask = (traj.grid.rlevels <= Lcut_t)
            Dv_f = np.zeros_like(traj.Dv, float); Dv_f[mask] = traj.Dv[mask]

        mu = np.asarray(traj.grid.mu_lnrho, float)
        rho0_ref = float(traj.grid.rho0)
        rho_filtered = rho0_ref * np.exp(np.cumsum(Dd_f) + mu)

        rho_crit = traj.grid.rhocs

        os.makedirs(level_entry.shard.dirpath, exist_ok=True)
        fname = f"snap_k{level_entry.k_next:05d}.npz"
        fpath = os.path.join(level_entry.shard.dirpath, fname)

        # existing arrays (rho_filtered, Ddelta_filt, etc.)
        # spatial and path metadata
        pos_active = level_entry.pos_cm if level_entry.pos_cm is not None else np.zeros(3, float)
        bits_arr   = np.array(self.path_bits, dtype=int)

        # for convenience, dump positions for all known levels (sparse)
        # pack as a (N,4): [level, x, y, z]
        if self.pos_by_level:
            lvls  = np.array(sorted(self.pos_by_level.keys()), dtype=int)
            poss  = np.vstack([self.pos_by_level[L] for L in lvls])
            pos_table = np.column_stack([lvls, poss])
        else:
            pos_table = np.zeros((0,4), float)

        np.savez_compressed(
            fpath,
            time_s=np.array([t], float),
            rlevels_cm=np.array(traj.grid.rlevels, float),
            mu_lnrho=mu,
            rho0_ref=np.array([rho0_ref], float),
            Ddelta_filt=Dd_f.astype(float),
            Dv_filt=Dv_f.astype(float),
            rho_filtered=rho_filtered.astype(float),
            baseline_Lcut=np.array([Lcut_t], float),
            path_bits=bits_arr,
            pos_active_cm=pos_active.astype(float),
            pos_table=np.array(pos_table, float),  # (N,4): level,x,y,z
            bit_at_level=np.array([-1 if level_entry.bit is None else int(level_entry.bit)], int),
            rho_crit=rho_crit,
            level_active=np.array([self.active_index], int),
        )


    # --------- main loop ---------

    def evolve(self,
               Tend_s: float,
               dt_s: float,
               dt_snap_s: float,
               *,
               verbose: bool = False):
        """
        Run with push-down on new instability and pop-up after active collapse.
        """
        t = float(self.t_now)
        next_snap = t

        # initial snapshot
        self._save_snapshot(self.stack[-1], t)
        self.stack[-1].k_next += 1
        next_snap += dt_snap_s

        while t < Tend_s:
            i_star = self.active_index

            # Evolve only subscales: indices >= i_star+1
            dt = min(dt_s, Tend_s - t)
            # Freeze the active shell by restoring its OU state after the step
            Dd_keep = float(self.traj.Ddelta[i_star])
            Dv_keep = self.traj.Dv[i_star].copy()
            self.traj.time_step(dt=dt, cloud_evolve=False) #ilevel=i_star+1, )
            t = float(self.traj.t)
            # restore active shell OU
            self.traj.Ddelta[i_star] = Dd_keep
            self.traj.Dv[i_star] = Dv_keep
            self.traj.delta = np.cumsum(self.traj.Ddelta)
            self.traj.v = np.cumsum(self.traj.Dv, axis=0)
            if len(self.traj.ts) == 0 or self.traj.ts[-1] < t:
                self.traj.add_trajectory()

            icol = None
            icol = self.traj.find_unstable(t,type='smallest')
            '''print(icol)
            import matplotlib.pyplot as plt
            plt.plot(self.traj.grid.rlevels, self.traj.density_profile(t_seconds=t, filtered=True))
            plt.plot(self.traj.grid.rlevels, self.traj.grid.rhocs, color='r')
            if not icol is None:
                plt.axvline(self.traj.grid.rlevels[icol], color='k')
            plt.axvline(self.traj.grid.rlevels[i_star], linestyle='dashed', color='yellow')
            plt.xscale('log')
            plt.yscale('log')
            plt.show()'''
            # (1) push: look for deeper instability
            '''try:
                icol = self.traj.find_unstable(t, type='smallest')
                import matplotlib.pyplot as plt
                plt.plot(self.traj.grid.rlevels, self.traj.density_profile(t_seconds=t, filtered=True))
                plt.plot(self.traj.grid.rlevels, self.traj.grid.rhocs, color='r')
                plt.axvline(self.traj.grid.rlevels[icol], color='k')
                plt.axvline(self.traj.grid.rlevels[i_star], linestyle='dashed', color='yellow')
                plt.xscale('log')
                plt.yscale('log')
                plt.show()
                
            except Exception as e:
                warnings.warn(f"find_collapse failed: {e}")'''
            icol_new = _robust_child_index(icol,
                                           current_level=i_star,
                                           nlevels=len(self.grid.rlevels))
            
            if icol_new is not None and icol_new > i_star:
                # create a new node that inherits up to current frame
                # finalize parent's inheritance window
                parent = self.stack[-1]
                parent.shard.inherit_until_k = parent.k_next - 1

                # choose volume bit and position for the child
                bit = _choose_volume_bit(self.rng)
                R_parent = float(self.grid.rlevels[parent.level])
                R_child  = float(self.grid.rlevels[int(icol_new)])
                # step radius: half the *difference* so the child stays comfortably inside
                R_step   = 0.5 * max(R_parent - R_child, 0.0)
                dpos     = _rand_in_ball(R_step, self.rng)
                
                pos_child = (parent.pos_cm if parent.pos_cm is not None else np.zeros(3)) + dpos

                # build child baseline aligned to now
                bfn_aligned, info = self._build_baseline_for_level(int(icol_new), t_now=t)

                # make shard and level entry
                new_id  = f"node_{uuid.uuid4()}"
                new_dir = os.path.join(os.path.dirname(parent.shard.dirpath), new_id)
                shard   = SnapShard(node_id=new_id, dirpath=new_dir,
                                    parent_id=parent.shard.node_id, inherit_until_k=None)
                shard.write_meta()

                child = LevelEntry(level=int(icol_new),
                                baseline_fn=bfn_aligned,
                                baseline_info=info,
                                shard=shard,
                                k_next=0,
                                bit=bit,
                                pos_cm=pos_child)
                self.stack.append(child)

                self.traj.baseline_fn = bfn_aligned

                self.active_index = int(icol_new)   # new pinned shell
                self._last_leaf = self.stack[-1]    # keep plotting alias fresh
                self.traj.icol = None               # clear stale detection from find_collapse

                self._last_leaf = self.stack[-1]

                # update path + position maps
                self.path_bits.append(bit)
                self.pos_by_level[int(icol_new)] = pos_child.copy()

                # snapshot marking the switch
                self._save_snapshot(self.stack[-1], t)
                self.stack[-1].k_next += 1
                next_snap = t + dt_snap_s
                continue

            active = self.stack[-1]
            # (2) pop: check active level collapse via baseline t_ff
            if self._check_ff_collapse(active.baseline_fn, t):
                # finalize current node inheritance
                active.shard.inherit_until_k = active.k_next - 1
                if verbose:
                    print(f"[pop]  t={t/year2s/1e6:.3f} Myr level {active.level} collapsed")

                # --- RESET subscales below the collapsed cloud ---
                i_collapsed = int(active.level)
                j0 = i_collapsed + 1
                if j0 < len(self.traj.Ddelta):
                    self.traj.Ddelta[j0:] = 0.0
                    self.traj.Dv[j0:, :] = 0.0
                    # rebuild cumulative state
                    self.traj.delta = np.cumsum(self.traj.Ddelta)
                    self.traj.v = np.cumsum(self.traj.Dv, axis=0)
                    # keep the time-series arrays coherent
                    if len(self.traj.ts) == 0 or self.traj.ts[-1] < t:
                        self.traj.add_trajectory()

                # pop to parent
                self.stack.pop()
                if not self.stack:
                    if verbose:
                        print("[done] no collapsing levels remain")
                    break

                # promote parent (restore exactly the parent's baseline)
                parent = self.stack[-1]
                self.traj.baseline_fn = parent.baseline_fn

                # optional: clear stale collapse detection to avoid immediate re-push
                try:
                    self.traj.icol = None
                except Exception:
                    pass

                # snapshot marking the promotion
                self._save_snapshot(parent, t)
                parent.k_next += 1
                next_snap = t + dt_snap_s
                continue
        
        self.t_now = t
        if verbose:
            print(f"[end] t={t/year2s/1e6:.3f} Myr; active level = {self._active_level()}")
