import numpy as np
from turbfuncs import *
from consts_defaults import *
import matplotlib.pyplot as plt
import excursion as exc
import cloud as cl
from scipy.ndimage import zoom
from scipy.special import erfinv
import pickle
import os, glob, re
import matplotlib.animation as animation
from scipy.spatial import cKDTree
from scipy.special import erfinv
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import healpy as hp
import time as time_module
import generate_cloudrho0 as gcr


pc2cm = 3.086e18  
year2s = 3.154e7 

plt.rc('text', usetex=True)

class MultiResolutionArray:
	def __init__(self, snapshot_dir='snapshots', imaxcoll=None, rmax=200., rspatial=30.0, rmin=0.3, dr=0.8,cells_per_level=100, baseline_fn=None):
		"""
		Initialize the MultiResolutionArray. If a file with the given filename exists,
		load the object from the file. Otherwise, initialize the object and save it.

		Args:
			grid (trajectory_spatial_grid): The grid object containing rlevels and physical properties.
			filename (str): File path to save or load the object.

		Returns:
			None
		"""
		self.cells_per_level = int(cells_per_level)

		self.baseline_fn = baseline_fn

		pc2cm = 3.086e18  # Example constant
		print("Defining spatial scales...")
		self.define_spatial_scales(rmax*pc2cm, rspatial*pc2cm, rmin*pc2cm, dr, cells_per_level)

		print("Initializing trajectory grid...")
		self.grid = exc.trajectory_grid(self.scales)

		print("Generating spatial grid...")
		self.generate_unstructured_levels()
		
		print(f"Initialized MultiResolutionArray with {len(self.scales)} resolutions.")


		self.snapshot_dir = snapshot_dir
		if imaxcoll is None:
			iall= np.arange(len(self.scales))
			imaxcoll = int(np.percentile(iall, 90.0))
		
		self.imaxcoll = imaxcoll
		self.t = 0.0


		# Create snapshot directory if it doesn't exist
		os.makedirs(self.snapshot_dir, exist_ok=True)

		#self.plot_cell_centers_3d()

	def _get_baseline(self, t_seconds):
		"""
		Returns (rho0_t, v0_t, Lcut_t) from the user-supplied baseline_fn, or defaults.
		- rho0_t: scalar (>0)
		- v0_t:   np.array shape (3,) velocity offset [same units as your vectors]
		- Lcut_t: float, cutoff scale [cm]; ignore all perturbation levels with scale > Lcut_t
		"""
		if callable(self.baseline_fn):
			out = self.baseline_fn(t_seconds)
			rho0_t = float(out.get("rho0", self.grid.rho0))
			v0_t   = np.asarray(out.get("v0", np.zeros(3, dtype=float)), dtype=float).reshape(3,)
			Lcut_t = float(out.get("Lcut", np.inf))
		else:
			rho0_t = float(self.grid.rho0)
			v0_t   = np.zeros(3, dtype=float)
			Lcut_t = np.inf
		return rho0_t, v0_t, Lcut_t
	
	def _active_level_masks(self, Lcut):
		"""
		Returns:
		super_mask   : list[bool] same length as self.super_scales
		spatial_mask : list[bool] same length as self.spatial_scales
		True means the level is INCLUDED in the sum.
		"""
		super_mask   = [ (s <= Lcut) for s in self.super_scales ] if hasattr(self, "super_scales") else []
		spatial_mask = [ (s <= Lcut) for s in self.spatial_scales ]
		return super_mask, spatial_mask

	def generate_unstructured_levels(self):
		"""
		For each spatial level ℓ:
		- Nℓ = fixed number of cells
		- <V_cell> = 4/3 π (spatial_scales[ℓ])^3  (treat spatial_scale as cell radius/scale)
		- Rℓ = spatial_scales[ℓ] * Nℓ^(1/3)
		- Sample Nℓ centers uniformly in sphere of radius Rℓ
		- Initialize scalar/vector perturbations with same stats as before
		"""

		self.level_positions = []   # list of (Nℓ, 3)
		self.level_scalar = []      # list of (Nℓ,)
		self.level_vector = []      # list of (Nℓ, 3)

		for i, dx in enumerate(self.spatial_scales):
			N = self.n_res[i]
			print(N, dx)
			R = dx * (N**(1.0/3.0))  # because N*<V_cell> = 4/3 π R^3 → R = dx * N^(1/3)

			pos = self._sample_points_in_sphere(N, R)

			# OU field parameters at turbulence level index spatial_level[i]
			ilev = self.spatial_level[i]
			DS  = self.grid.Delta_S[ilev]
			DSV = self.grid.Delta_Sv[ilev]

			# Match your Gaussian via inverse-erf of uniform
			u_s = np.random.uniform(size=N)
			scal = np.sqrt(2.0 * DS)  * erfinv(2.0*u_s - 1.0)                  # (N,)
			u_v = np.random.uniform(size=(N, 3))
			vec  = np.sqrt(2.0 * DSV) * erfinv(2.0*u_v - 1.0)                  # (N,3)

			self.level_positions.append(pos)
			self.level_scalar.append(scal.astype(np.float64))
			self.level_vector.append(vec.astype(np.float64))

		# Keep super-scales as before (one value per super level)
		self.super_resolutions  = [self.initialize_resolution(j, 1)[0] for j in range(len(self.super_scales))]
		self.super_resolutions_v= [self.initialize_resolution(j, 1)[1] for j in range(len(self.super_scales))]

	def _sample_points_in_sphere(self, N, R):
		# radius ~ U(0,1)^(1/3) to be uniform in volume
		u = np.random.uniform(size=N)
		r = R * u**(1.0/3.0)
		# isotropic directions
		cos_theta = np.random.uniform(-1.0, 1.0, size=N)
		phi = np.random.uniform(0.0, 2.0*np.pi, size=N)
		sin_theta = np.sqrt(1.0 - cos_theta**2)

		x = r * sin_theta * np.cos(phi)
		y = r * sin_theta * np.sin(phi)
		z = r * cos_theta
		return np.stack([x, y, z], axis=-1)  # (N,3)
	
	"""def define_spatial_scales(self, rmax, rspatial, rmin, dr, n0):
		# Define scales from rmax to rspatial reducing by dr
		scales = [rmax]
		super_scales = [rmax]
		ir =0
		while scales[-1] * dr > rspatial:
			scales.append(scales[-1] * dr)
			super_scales.append(scales[-1])
			ir+=1 
		scales.append(rspatial)  # Ensure rspatial is included
		self.super_scales = super_scales

		# Define n_res at rspatial
		n_res = [n0]

		# Generate spatial scales based on n_res
		rscale = rspatial
		self.spatial_scales = [rspatial/float(n0)]
		self.spatial_level = [ir]
		while rscale > rmin:
			next_n_res = max(int(n_res[-1] / dr), n_res[-1] + 1)  # Enforce rule
			n_res.append(next_n_res)
			rscale = rspatial *float(n0) / float(next_n_res)  # Compute next spatial scale
			scales.append(rscale)
			self.spatial_scales.append(rscale)
			ir+=1
			self.spatial_level.append(ir)

		self.scales = np.array(scales)
		self.spatial_scales = np.array(self.spatial_scales)
		self.n_res = n_res

		return self.scales"""

	def define_spatial_scales(self, rmax, rspatial, rmin, dr, n0):

		#Definitions: scales is the scale of interest 
		# 
		scales = [rmax]
		super_scales = [rmax]
		ir = 0
		while scales[-1] * dr > rspatial:
			scales.append(scales[-1] * dr)
			super_scales.append(scales[-1])
			ir += 1
		scales.append(rspatial)  # Ensure rspatial is included
		self.super_scales = super_scales  # these feed the turbulence model

		# Define n_res at rspatial
		n_res = [n0]
	
		# Generate spatial scales based on n_res
		rscale = rspatial
		self.spatial_scales = [rspatial] 
		self.spatial_level = [ir]

		while rscale > rmin:
			next_n_res = n0  # Enforce rule
			n_res.append(next_n_res)
			rscale = rscale * dr 
			scales.append(rscale)
			self.spatial_scales.append(rscale)
			ir+=1
			self.spatial_level.append(ir)

		self.scales = np.array(scales)
		self.spatial_scales = np.array(self.spatial_scales)
		self.n_res = n_res

		return self.scales


	def generate_unstructured_levels(self):
		"""
		For each spatial level l:
		- Nl = fixed number of cells
		- <V_cell> = 4/3 π (spatial_scales[l])^3  (treat spatial_scale as cell radius/scale)
		- Rl = spatial_scales[l] * Nl^(1/3)
		- Sample Nl centers uniformly in sphere of radius Rl
		- Initialize scalar/vector perturbations with same stats as before
		"""
		N_cells = self.n_res

		self.level_positions = []   # list of (Nℓ, 3)
		self.level_scalar = []      # list of (Nℓ,)
		self.level_vector = []      # list of (Nℓ, 3)

		for i, dx in enumerate(self.spatial_scales):
			N = N_cells[i]
			R = dx * (N**(1.0/3.0))  # because N*<V_cell> = 4/3 π R^3 → R = dx * N^(1/3)

			pos = self._sample_points_in_sphere(N, R)

			print(pos)

			# OU field parameters at turbulence level index spatial_level[i]
			ilev = self.spatial_level[i]
			DS  = self.grid.Delta_S[ilev]
			DSV = self.grid.Delta_Sv[ilev]

			# Match your Gaussian via inverse-erf of uniform
			u_s = np.random.uniform(size=N)
			scal = np.sqrt(2.0 * DS)  * erfinv(2.0*u_s - 1.0)                  # (N,)
			u_v = np.random.uniform(size=(N, 3))
			vec  = np.sqrt(2.0 * DSV) * erfinv(2.0*u_v - 1.0)                  # (N,3)

			self.level_positions.append(pos)
			self.level_scalar.append(scal.astype(np.float64))
			self.level_vector.append(vec.astype(np.float64))

		# Keep super-scales as before (one value per super level)
		self.super_resolutions  = [self.initialize_resolution(j, 1)[0] for j in range(len(self.super_scales))]
		self.super_resolutions_v= [self.initialize_resolution(j, 1)[1] for j in range(len(self.super_scales))]


	def level_cell_centers(self, i):
		"""
		Exact 1D centers for level i, centered at 0.
		Returns: np.ndarray of shape (N_i,)
		"""
		N = int(self.n_res[i])
		dx = float(self.spatial_scales[i])
		L  = float(self.level_length[i])
		return (np.arange(N) + 0.5) * dx - L / 2.0
	
	def initialize_resolution(self, ilevel, resolution_size):
		"""
		Initialize the values of a single resolution grid.

		Args:
			ilevel (int): The level index for the resolution.
			resolution_size (int): The size of the resolution grid.

		Returns:
			np.ndarray: The initialized resolution grid.
		"""
		# Retrieve Delta_S for this level
		DS = self.grid.Delta_S[ilevel]
		DSV = self.grid.Delta_Sv[ilevel]

		# Create a uniform random array
		u_delta = np.random.uniform(size=(resolution_size, resolution_size, resolution_size))
		# Create a uniform random array
		u_delta_v = np.random.uniform(size=(resolution_size, resolution_size, resolution_size, 3))

		# Transform the uniform distribution using the inverse error function
		initialized_grid = np.sqrt(2.0 * DS) * erfinv(2.0 * u_delta - 1.0)
		initialized_grid_v = np.sqrt(2.0 * DSV) * erfinv(2.0 * u_delta_v - 1.0)


		return initialized_grid, initialized_grid_v

	
	def update_resolutions(self, dt):

		# super-scales: unchanged
		for ilevel in range(len(self.super_resolutions)):
			exp_decay = np.exp(-dt / self.grid.tau_R[ilevel])

			Ddelta_new = self.super_resolutions[ilevel] * exp_decay
			Ddelta_new += np.random.normal(size=self.super_resolutions[ilevel].shape) * np.sqrt(
				self.grid.Delta_S[ilevel] * (1.0 - np.exp(-2.0 * dt / self.grid.tau_R[ilevel]))
			)
			self.super_resolutions[ilevel] = Ddelta_new

			Ddeltav_new = self.super_resolutions_v[ilevel] * exp_decay
			Ddeltav_new += np.random.normal(size=self.super_resolutions_v[ilevel].shape) * np.sqrt(
				self.grid.Delta_Sv[ilevel] * (1.0 - np.exp(-2.0 * dt / self.grid.tau_R[ilevel]))
			)
			self.super_resolutions_v[ilevel] = Ddeltav_new

		# unstructured levels
		for i in range(len(self.spatial_scales)):
			ilev = self.spatial_level[i]
			exp_decay = np.exp(-dt / self.grid.tau_R[ilev])

			s_old = self.level_scalar[i]
			v_old = self.level_vector[i]

			s_new = s_old * exp_decay + np.random.normal(size=s_old.shape) * np.sqrt(
				self.grid.Delta_S[ilev] * (1.0 - np.exp(-2.0 * dt / self.grid.tau_R[ilev]))
			)
			v_new = v_old * exp_decay + np.random.normal(size=v_old.shape) * np.sqrt(
				self.grid.Delta_Sv[ilev] * (1.0 - np.exp(-2.0 * dt / self.grid.tau_R[ilev]))
			)

			self.level_scalar[i] = s_new
			self.level_vector[i] = v_new

	

	def evolve(self, Tend, fraction_of_tau=0.1, dt_snap=0.2):
		"""
		Evolve the system until Tend, saving snapshots at dt_snap intervals.

		If a saved snapshot already extends to Tend, skip the run.

		Args:
			Tend (float): Maximum time in Myr.
			fraction_of_tau (float): Fraction of the minimum tau_R used for timestep.
			dt_snap (float): Time interval between snapshots (Myr).

		Returns:
			None
		"""
		# Convert to seconds
		Tend_sec = Tend * 1e6 * year2s
		dt_snap_sec = dt_snap * 1e6 * year2s
		dt = fraction_of_tau * np.min(self.grid.tau_R)

		print(Tend_sec/year2s/1e6, "MYR", dt_snap_sec/year2s/1e6, "Myr", dt/year2s/1e6, "MYR per step")

		# Check if we already have snapshots up to Tend
		snapshot_times_file = os.path.join(self.snapshot_dir, "snapshot_times.npy")
		if os.path.exists(snapshot_times_file):
			snapshot_times = np.load(snapshot_times_file)
			if len(snapshot_times) > 0 and snapshot_times[-1] >= Tend_sec:
				print(f"Snapshots already exist up to {Tend} Myr. Skipping evolution.")
				return
			
		snapshot_times = []
		t = 0.0
		last_snapshot_idx = -1
		print("Starting evolution from the beginning.")

		snapshot_idx = last_snapshot_idx + 1
		next_snapshot_time = t + dt_snap_sec

		while t < Tend_sec:
			self.update_resolutions(dt)
			t += dt
			self.t = t

			if t >= next_snapshot_time:
				#print('Saving...')
				self.save_snapshot(snapshot_idx, t)
				snapshot_times.append(t)
				np.save(snapshot_times_file, np.array(snapshot_times))  # Update time tracking
				snapshot_idx += 1
				next_snapshot_time += dt_snap_sec

		print(f"Evolution completed: Total time = {Tend} Myr")

	
	def save_snapshot(self, snapshot_idx, time,
                  rasterize_to_grid=True,
                  # equal-area spherical grid params:
                  Nr=128, r_min=None, r_max=None,
                  nside=16,
                  store_cartesian_coords=False,
                  coords_filename="sph_coords_equalarea.npz",
				  profile=False):
		"""
		If rasterize_to_grid:
		- writes equal-area spherical fields:
			snapshot_XXXX_lnrho_eq.npy  (Nr, npix)
			snapshot_XXXX_v_eq.npy      (Nr, npix, 3)
		- writes coords once to coords_filename with {r, theta, phi, U[, X,Y,Z], nside}
		Else:
		- writes unstructured point clouds per level as before.
		Prints timing diagnostics for each stage.
		"""
		t0_total = time_module 
		t_start_total = time_module.perf_counter()


		os.makedirs(self.snapshot_dir, exist_ok=True)
		times_filename = os.path.join(self.snapshot_dir, "snapshot_times.npy")

		if not rasterize_to_grid:
			# ---------------- Unstructured save ----------------
			t0 = time_module.perf_counter()
			density_filename  = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_scalar_points.npz")
			velocity_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_vector_points.npz")

			# Save scalar point cloud
			t1 = time_module.perf_counter()
			np.savez_compressed(
				density_filename,
				**{f"pos_{i}": self.level_positions[i] for i in range(len(self.level_positions))},
				**{f"sca_{i}": self.level_scalar[i]   for i in range(len(self.level_scalar))}
			)
			t2 = time_module.perf_counter()
			if profile:
				print(f"[save_snapshot] Unstructured scalar points saved to {os.path.basename(density_filename)} "
					f"in {t2 - t1:.3f}s")

			# Save vector point cloud
			t3 = time_module.perf_counter()
			np.savez_compressed(
				velocity_filename,
				**{f"pos_{i}": self.level_positions[i] for i in range(len(self.level_vector))},
				**{f"vec_{i}": self.level_vector[i]    for i in range(len(self.level_vector))}
			)
			t4 = time_module.perf_counter()
			if profile:
				print(f"[save_snapshot] Unstructured vector points saved to {os.path.basename(velocity_filename)} "
					f"in {t4 - t3:.3f}s")

				print(f"[save_snapshot] Unstructured stage total: {t4 - t0:.3f}s")

		else:
			# ---------------- Equal-area grid save ----------------
			# Ensure coord file exists
			coord_path = os.path.join(self.snapshot_dir, coords_filename)
			t_coord0 = time_module.perf_counter()
			if not os.path.exists(coord_path):
				self.save_equal_area_coords(self.snapshot_dir,
											Nr=Nr, r_min=r_min, r_max=r_max,
											nside=nside,
											store_cartesian=store_cartesian_coords,
											filename=coords_filename)
				t_coord1 = time_module.perf_counter()
				if profile:
					print(f"[save_snapshot] Built & saved coord file {os.path.basename(coord_path)} "
							f"in {t_coord1 - t_coord0:.3f}s")
			else:
				t_coord1 = time_module.perf_counter()
				if profile:
					print(f"[save_snapshot] Coord file {os.path.basename(coord_path)} already present "
					f"(checked in {t_coord1 - t_coord0:.3f}s)")

			# Evaluate on equal-area grid
			t_eval0 = time_module.perf_counter()
			lnrho_eq, v_eq = self.evaluate_to_equal_area(
				Nr=Nr, r_min=r_min, r_max=r_max, nside=nside,
				coords_path=coord_path, return_coords=False,
				t_seconds=time  
			)
			t_eval1 = time_module.perf_counter()
			if profile:
				print(f"[save_snapshot] evaluate_to_equal_area: {t_eval1 - t_eval0:.3f}s "
				f"(lnrho shape={lnrho_eq.shape}, v shape={v_eq.shape})")

			# Save arrays
			lnrho_path = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_lnrho_eq.npy")
			v_path     = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_v_eq.npy")

			t_save0 = time_module.perf_counter()
			np.save(lnrho_path, lnrho_eq)
			t_save1 = time_module.perf_counter()
			if profile:
				print(f"[save_snapshot] Saved {os.path.basename(lnrho_path)} in {t_save1 - t_save0:.3f}s")

			np.save(v_path, v_eq)
			t_save2 = time_module.perf_counter()
			if profile:
				print(f"[save_snapshot] Saved {os.path.basename(v_path)} in {t_save2 - t_save1:.3f}s")
				print(f"[save_snapshot] Rasterized stage total: {t_save2 - t_coord0:.3f}s")

		# ---------------- Append time array ----------------
		t_times0 = time_module.perf_counter()
		if not os.path.exists(times_filename):
			snapshot_times = np.array([time], dtype=float)
		else:
			snapshot_times = np.load(times_filename)
			snapshot_times = np.append(snapshot_times, float(time))
		np.save(times_filename, snapshot_times)
		t_times1 = time_module.perf_counter()
		if profile:
				print(f"[save_snapshot] Updated snapshot_times.npy in {t_times1 - t_times0:.3f}s")

		# ---------------- Total ----------------
		t_end_total = time_module.perf_counter()
		if profile:
				print(f"[save_snapshot] Snapshot {snapshot_idx} saved at time {time / year2s / 1e6:.2f} Myr "
					f"(total {t_end_total - t_start_total:.3f}s)")




	def plot_cell_centers_3d(self,
							levels=None,
							max_points_per_level=None,
							s=2.0,
							alpha=0.6,
							units='cm',
							elev=20,
							azim=35,
							save=None):
			"""
			Plot 3D locations of all (unstructured) cell centers, colored by spatial level.

			Args:
				levels (list[int] or None): Levels to include; default = all spatial levels.
				max_points_per_level (int or None): If set, randomly subsample up to this many
					points per level so plotting stays responsive.
				s (float): marker size for scatter.
				alpha (float): marker transparency.
				units ('cm'|'pc'): Axis/display units.
				elev, azim (float): 3D view angles.
				save (str or None): If given, save to this filepath; otherwise show().

			Notes:
				- Requires unstructured mode: self.level_positions populated (list of (Ni,3)).
				- Colors are assigned per level and shown in the legend.
			"""
			# Sanity checks
			if not hasattr(self, "level_positions"):
				raise RuntimeError("Unstructured levels not found. Generate with generate_unstructured_levels().")

			# Choose which levels to plot
			if levels is None:
				levels = list(range(len(self.level_positions)))

			# Unit conversion
			pc2cm = 3.086e18
			if units == 'pc':
				scale = 1.0 / pc2cm
				xlabel, ylabel, zlabel = 'x [pc]', 'y [pc]', 'z [pc]'
				L_show = getattr(self, "L_domain", None)
				if L_show is not None:
					L_show *= scale
			elif units == 'cm':
				scale = 1.0
				xlabel, ylabel, zlabel = 'x [cm]', 'y [cm]', 'z [cm]'
				L_show = getattr(self, "L_domain", None)
			else:
				raise ValueError("units must be 'cm' or 'pc'")

			# Colors per level
			cmap = plt.get_cmap('tab20')
			colors = {i: cmap(i % 20) for i in levels}

			fig = plt.figure(figsize=(8, 7))
			ax = fig.add_subplot(111, projection='3d')

			# Plot each level
			handles = []
			labels = []
			rng = np.random.default_rng()

			for i in levels:
				pos = self.level_positions[i]  # (Ni, 3)
				if pos is None or len(pos) == 0:
					continue

				Ni = pos.shape[0]
				print(Ni)
				if max_points_per_level is not None and Ni > max_points_per_level:
					idx = rng.choice(Ni, size=max_points_per_level, replace=False)
					pts = pos[idx]
					shown = max_points_per_level
				else:
					pts = pos
					shown = Ni

				ax.scatter(pts[:, 0]*scale, pts[:, 1]*scale, pts[:, 2]*scale,
						s=s, alpha=alpha, color=colors[i])

				handles.append(plt.Line2D([0],[0], marker='o', color='w',
										markerfacecolor=colors[i], markersize=6))
				labels.append(f"Level {i} (N={Ni}, shown={shown})")

			# Axes limits centered at 0: use global domain if available; otherwise data bounds
			if L_show is not None and np.isfinite(L_show):
				half = L_show / 2.0
				ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(-half, half)
			else:
				# fallback to data bounds
				all_pos = np.vstack([self.level_positions[i] for i in levels if len(self.level_positions[i]) > 0])
				all_pos *= scale
				mins = np.min(all_pos, axis=0)
				maxs = np.max(all_pos, axis=0)
				center = 0.5*(mins+maxs)
				radius = 0.55*np.max(maxs - mins)
				ax.set_xlim(center[0]-radius, center[0]+radius)
				ax.set_ylim(center[1]-radius, center[1]+radius)
				ax.set_zlim(center[2]-radius, center[2]+radius)

			ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
			ax.view_init(elev=elev, azim=azim)
			ax.set_box_aspect([1,1,1])
			ax.grid(True, ls=':', lw=0.5)

			# Legend
			ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0),
					borderaxespad=0., fontsize=9)

			plt.tight_layout()
			if save:
				plt.savefig(save, dpi=200, bbox_inches='tight')
				plt.close(fig)
			else:
				plt.show()

	def build_equal_area_spherical_grid(self,
                                    Nr=128,
                                    r_min=None, r_max=None,
                                    nside=8,
                                    store_cartesian=False):
		"""
		Construct a spherical grid with:
		- log-spaced radial centers r_c (length Nr)
		- HEALPix angular centers (equal-area), npix = 12*nside^2
		Returns:
		r_c : (Nr,)
		theta : (npix,)
		phi   : (npix,)
		(optionally) X, Y, Z : each (Nr, npix), if store_cartesian=True
		and always U : (npix, 3) unit direction vectors
		"""
		# radii (log)
		if r_max is None:
			r_max =np.amax(self.spatial_scales)
		if r_min is None:
			r_min = np.amin(self.spatial_scales)

		if not (r_min > 0 and r_min < r_max):
			raise ValueError(f"Invalid radii: r_min={r_min}, r_max={r_max}")

		r_edges = np.geomspace(r_min, r_max, Nr + 1)
		r_c = np.sqrt(r_edges[:-1] * r_edges[1:])  # centers (Nr,)

		# HEALPix directions (fixed for all r)
		npix = hp.nside2npix(nside)
		theta, phi = hp.pix2ang(nside, np.arange(npix))  # (npix,), (npix,)

		print(theta.shape, phi.shape, r_c.shape, npix)

		'''fig = plt.figure(figsize=(8, 7))
		ax = fig.add_subplot(111, projection='3d')

		# unit vectors for each pixel
		sin_th = np.sin(theta)
		ux = sin_th * np.cos(phi)
		uy = sin_th * np.sin(phi)
		uz = np.cos(theta)
		U = np.stack([ux, uy, uz], axis=-1)  # (npix, 3)

		
		ax.scatter(ux, uy, uz, s=1.0, alpha=0.5, color='gray', label='HEALPix directions')

	
		plt.show()'''

		# unit vectors for each pixel
		sin_th = np.sin(theta)
		ux = sin_th * np.cos(phi)
		uy = sin_th * np.sin(phi)
		uz = np.cos(theta)
		U = np.stack([ux, uy, uz], axis=-1)  # (npix, 3)

		if not store_cartesian:
			return r_c, theta, phi, U

		# Full Cartesian centers (Nr, npix) — if you want to save full arrays
		X = np.outer(r_c, ux)  # (Nr, npix)
		Y = np.outer(r_c, uy)
		Z = np.outer(r_c, uz)

		return r_c, theta, phi, U, X, Y, Z
	
	def save_equal_area_coords(self, snapshot_dir=None,
                           Nr=128, r_min=None, r_max=None,
                           nside=16, store_cartesian=False,
                           filename="sph_coords_equalarea.npz"):
		"""
		Save spherical grid coordinates for later reuse:
		- always saves: r (Nr,), theta (npix,), phi (npix,), U (npix,3), nside
		- optionally saves: X, Y, Z (each Nr x npix) if store_cartesian=True
		"""
		outdir = self.snapshot_dir if snapshot_dir is None else snapshot_dir
		os.makedirs(outdir, exist_ok=True)
		path = os.path.join(outdir, filename)

		if store_cartesian:
			r_c, theta, phi, U, X, Y, Z = self.build_equal_area_spherical_grid(
				Nr=Nr, r_min=r_min, r_max=r_max, nside=nside, store_cartesian=True
			)
			np.savez(path, r=r_c, theta=theta, phi=phi, U=U, X=X, Y=Y, Z=Z, nside=np.int32(nside))
		else:
			r_c, theta, phi, U = self.build_equal_area_spherical_grid(
				Nr=Nr, r_min=r_min, r_max=r_max, nside=nside, store_cartesian=False
			)
			np.savez(path, r=r_c, theta=theta, phi=phi, U=U, nside=np.int32(nside))

		return path

	def evaluate_to_equal_area(self,
                           Nr=128, r_min=None, r_max=None,
                           nside=16, coords_path=None,
                           return_coords=False,
                           profile=False,
                           batch_query=True,
                           t_seconds=None):
		"""
		Rasterize unstructured levels onto log-r + HEALPix-angle grid.

		Returns:
			lnrho_maps : (Nr, npix)
			v_maps     : (Nr, npix, 3)
			(optionally) r_c, theta, phi, U
			(if profile=True) also returns a stats dict as last item
		"""
		# -- baseline & active masks --
		if t_seconds is None:
			t_seconds = getattr(self, "t", 0.0)
		
		time = time_module
		t_all0 = time.perf_counter()
		stats = {"notes": "evaluate_to_equal_area timings", "batch_query": bool(batch_query)}

		rho0_t, v0_t, Lcut_t = self._get_baseline(t_seconds)
		super_mask, spatial_mask = self._active_level_masks(Lcut_t)

		


		# ---- Load/build coords ----
		t0 = time.perf_counter()
		if coords_path is None or not os.path.exists(coords_path):
			r_c, theta, phi, U = self.build_equal_area_spherical_grid(
				Nr=Nr, r_min=r_min, r_max=r_max, nside=nside, store_cartesian=False
			)
			stats["coords_source"] = "built"
		else:
			cc = np.load(coords_path)
			r_c, theta, phi, U = cc["r"], cc["theta"], cc["phi"], cc["U"]
			stats["coords_source"] = "loaded"
		stats["t_coords"] = time.perf_counter() - t0

		Nr = r_c.size
		npix = theta.size
		stats["Nr"] = int(Nr)
		stats["npix"] = int(npix)
		stats["num_levels"] = int(len(self.spatial_scales))

		# ---- Init outputs (+ super-scales) ----
		t1 = time.perf_counter()

		lnrho_maps = np.zeros((Nr, npix), dtype=np.float64)
		v_maps     = np.zeros((Nr, npix, 3), dtype=np.float64)

		if hasattr(self, 'super_resolutions') and super_mask:
			for j, keep in enumerate(super_mask):
				if keep:
					s0 = float(np.asarray(self.super_resolutions[j]).reshape(-1)[0])   # scalar
					lnrho_maps += s0
					
		if hasattr(self, 'super_resolutions_v') and super_mask:
			for j, keep in enumerate(super_mask):
				if keep:
					v0 = np.asarray(self.super_resolutions_v[j]).reshape(3,)           # (3,)
					v_maps     += v0
		stats["t_init"] = time.perf_counter() - t1

		# ---- Build KD-trees per level ----
		t2 = time.perf_counter()
		trees = []
		tree_times = []
		sizes = []
		for i in range(len(self.spatial_scales)):
			pos = self.level_positions[i]
			sizes.append(int(pos.shape[0]))
			if pos.size:
				tbi0 = time.perf_counter()
				trees.append(cKDTree(pos))
				tree_times.append(time.perf_counter() - tbi0)
			else:
				trees.append(None)
				tree_times.append(0.0)
		stats["t_tree_build_total"] = sum(tree_times)
		stats["t_tree_build_per_level"] = tree_times
		stats["points_per_level"] = sizes
		stats["num_active_levels"] = int(sum(t is not None for t in trees))

		# ---- Queries & accumulation ----
		if batch_query:
			# One big query per level: stack all radii points -> (Nr*npix, 3)
			tq0 = time.perf_counter()
			# Prebuild stacked points once
			# pts_stack[k] corresponds to linear index (ir*npix + p)
			pts_stack = (r_c[:, None, None] * U[None, :, :]).reshape(Nr * npix, 3)
			stats["t_build_pts_stack"] = time.perf_counter() - tq0

			query_times = []
			accum_times = []
			t_query_total = 0.0
			t_accum_total = 0.0

			for i, tree in enumerate(trees):
				if tree is None or not spatial_mask[i]:
					query_times.append(0.0)
					accum_times.append(0.0)
					continue
				tqi0 = time.perf_counter()
				try:
					_, idx = tree.query(pts_stack, k=1, workers=-1)
				except TypeError:
					_, idx = tree.query(pts_stack, k=1)
				tqi1 = time.perf_counter()
				query_times.append(tqi1 - tqi0)
				t_query_total += (tqi1 - tqi0)

				tai0 = time.perf_counter()
				# Gather and reshape
				s_vals = self.level_scalar[i][idx].reshape(Nr, npix)
				v_vals = self.level_vector[i][idx, :].reshape(Nr, npix, 3)
				lnrho_maps += s_vals
				v_maps     += v_vals
				tai1 = time.perf_counter()
				accum_times.append(tai1 - tai0)
				t_accum_total += (tai1 - tai0)

			stats["t_query_per_level"] = query_times
			stats["t_accum_per_level"] = accum_times
			stats["t_query_total"] = t_query_total
			stats["t_accum_total"] = t_accum_total

		else:
			# Query per radius (keeps memory small, more Python overhead)
			query_times = [0.0] * len(trees)
			accum_times = [0.0] * len(trees)
			t_loop0 = time.perf_counter()
			for ir, r in enumerate(r_c):
				pts = r * U  # (npix, 3)
				for i, tree in enumerate(trees):
					if tree is None or not spatial_mask[i]:
						continue
					tqi0 = time.perf_counter()
					try:
						_, idx = tree.query(pts, k=1, workers=-1)
					except TypeError:
						_, idx = tree.query(pts, k=1)
					tqi1 = time.perf_counter()
					query_times[i] += (tqi1 - tqi0)

					tai0 = time.perf_counter()
					lnrho_maps[ir, :] += self.level_scalar[i][idx]
					v_maps[ir, :, :]  += self.level_vector[i][idx, :]
					tai1 = time.perf_counter()
					accum_times[i] += (tai1 - tai0)
			stats["t_outer_loop"] = time.perf_counter() - t_loop0
			stats["t_query_per_level"] = query_times
			stats["t_accum_per_level"] = accum_times
			stats["t_query_total"] = sum(query_times)
			stats["t_accum_total"] = sum(accum_times)

		lnrho_maps += np.log(rho0_t)
		# add bulk velocity offset everywhere
		v_maps += v0_t  # broadcast to (Nr, npix, 3)

		# ---- Finish ----
		stats["t_total"] = time.perf_counter() - t_all0

		if profile:
			# Pretty print a compact report
			print("[evaluate_to_equal_area] profile:")
			print(f"  coords: {stats['t_coords']:.3f}s ({stats['coords_source']})")
			print(f"  init:   {stats['t_init']:.3f}s")
			print(f"  tree build total: {stats['t_tree_build_total']:.3f}s "
				f"(active levels={stats['num_active_levels']}/{stats['num_levels']})")
			for i, (tbuild, npts) in enumerate(zip(stats['t_tree_build_per_level'], stats['points_per_level'])):
				if tbuild > 0:
					print(f"    level {i:02d}: build {tbuild:.3f}s, points={npts}")
			if batch_query:
				print(f"  build pts stack: {stats.get('t_build_pts_stack', 0.0):.3f}s")
			print(f"  query total: {stats['t_query_total']:.3f}s")
			print(f"  accum total: {stats['t_accum_total']:.3f}s")
			print(f"  TOTAL: {stats['t_total']:.3f}s (Nr={Nr}, npix={npix})")

		if return_coords and profile:
			return lnrho_maps, v_maps, r_c, theta, phi, U, stats
		if return_coords:
			return lnrho_maps, v_maps, r_c, theta, phi, U
		if profile:
			return lnrho_maps, v_maps, stats
		return lnrho_maps, v_maps
	
	def create_equal_area_slice_video(
			self,
			snapshot_dir="snapshots",
			file_pattern="snapshot_*_lnrho_eq.npy",
			coords_file="sph_coords_equalarea.npz",
			output_filename="equalarea_slice.mp4",
			slice_type="r",                 # 'r' | 'theta' | 'phi'
			r_index=None, r_value=None,     # r_value in cm
			theta_value_deg=None,
			phi_value_deg=None,
			ang_tolerance_deg=1.0,
			fps=12,
			cmap="inferno",
			log_display=True,
			vmin=None, vmax=None,
			graticule=True,
			grat_dpar=30, grat_dmer=30,
		):
			# --------- load files & coords ---------
			files = sorted(glob.glob(os.path.join(snapshot_dir, file_pattern)))
			if not files:
				raise FileNotFoundError(f"No files matching {file_pattern} in {snapshot_dir}")

			coords_path = coords_file if not os.path.exists(os.path.join(snapshot_dir, coords_file)) \
									else os.path.join(snapshot_dir, coords_file)
			cc = np.load(coords_path)
			r_c   = cc["r"]                # (Nr,)
			nside = int(cc["nside"])
			npix  = hp.nside2npix(nside)
			Nr    = r_c.size
			if np.load(files[0]).shape != (Nr, npix):
				raise ValueError("Snapshot shape mismatch vs coords.")

			# --------- choose slice ---------
			def nearest_idx(arr, val): return int(np.argmin(np.abs(arr - val)))
			if slice_type.lower() != "r":
				raise NotImplementedError("This fix covers the HEALPix full-sky 'r' slice. (theta/phi panels unchanged.)")

			if r_index is None:
				r_index = (Nr // 2) if r_value is None else nearest_idx(r_c, float(r_value))
			j_r = int(r_index)

			# --------- stable log transform ---------
			LOG10E = 1.0/np.log(10.0)
			rho0 = float(self.grid.rho0)
			log10_rho0 = np.log10(rho0)
			def to_display(arr_ln):
				return (arr_ln*LOG10E) if log_display else (np.exp(arr_ln, dtype=np.float64))

			# --------- first frame & color scale ---------
			first = np.load(files[0])
			map0 = to_display(first[j_r, :])
			finite = np.isfinite(map0)
			if vmin is None or vmax is None:
				lo, hi = np.nanpercentile(map0[finite], [5, 95]) if np.any(finite) else (-1, 1)
				vmin = lo if vmin is None else vmin
				vmax = hi if vmax is None else vmax

			# --------- create figure + single axes + single colorbar ---------
			fig = plt.figure(figsize=(8, 6))
			# First call to mollview creates the proj axes and the ONLY colorbar
			hp.mollview(map0, fig=fig.number, min=vmin, max=vmax, cmap=cmap, title="", cbar=True, reuse_axes=False)
			ax = plt.gca()                             # <-- Healpy projection axes
			if graticule:
				hp.graticule(dpar=grat_dpar, dmer=grat_dmer, alpha=0.4)

			# remove any axis frame/ticks (healpy usually does this already)
			ax.axis('off')

			# --------- time labels ---------
			times_path = os.path.join(snapshot_dir, "snapshot_times.npy")
			t_array = np.load(times_path) if os.path.exists(times_path) else None
			year2s = 3.154e7
			idx_re = re.compile(r"snapshot_(\d+)", re.IGNORECASE)
			def parse_idx(path):
				m = idx_re.search(os.path.basename(path))
				return int(m.group(1)) if m else None

			# --------- per-frame update using projmap on the SAME axes ---------
			def update(frame_i):
				f = files[frame_i]
				arr = np.load(f)
				# time label
				snap_i = parse_idx(f)
				t_myr = None
				if t_array is not None:
					if snap_i is not None and 0 <= snap_i < len(t_array):
						t_myr = t_array[snap_i] / year2s / 1e6
					elif 0 <= frame_i < len(t_array):
						t_myr = t_array[frame_i] / year2s / 1e6

				# compute map for this frame
				map_i = to_display(arr[j_r, :])

				# CLEAR JUST THE PROJECTION AXES (not the figure!) to avoid overlays/shrinking
				ax.cla()
				ax.axis('off')  # keep axes off
				# Draw new data into the SAME axes; no new colorbar/axes are created
				ax.projmap(map_i, coord=None, vmin=vmin, vmax=vmax, cmap=cmap)
				if graticule:
					hp.graticule(dpar=grat_dpar, dmer=grat_dmer, alpha=0.4)
				# Title inside the map (healpy uses its own)
				ax.set_title(f"t = {t_myr:.2f} Myr" if t_myr is not None else os.path.basename(f))

				# nothing to return for blitting
				return []

			ani = animation.FuncAnimation(fig, update, frames=len(files), blit=False)
			ani.save(output_filename, writer="ffmpeg", fps=fps)
			plt.close(fig)
			print(f"Saved video to {output_filename}")



if __name__ == "__main__":

	#grid = exc.trajectory_grid(rmax=30.0 * pc2cm, rmin=0.2 * pc2cm, drfact=0.5)

	R_sfr = 2.0 * pc2cm  # SFR radius in cm

	baseline_fn, info = gcr.build_cloud_baseline_fn(
			target_radius_cm=R_sfr,
			rmin=0.01*pc2cm,
			rmax=10.0*h_*pc2cm,
			drfact=0.95,
			dt_myr=0.01,
			tmax_myr=10.0,
			seed=42
		)

	# Initialize the MultiResolutionArray§
	mra = MultiResolutionArray(rmax=200.0, rspatial=0.1, rmin=1e-5, dr=0.5, cells_per_level=50, baseline_fn=baseline_fn)
	# Evolve for 10 Myr, storing snapshots every 1 Myr
	mra.evolve(Tend=5.0, fraction_of_tau=0.1, dt_snap=0.05)

	# Create an MP4 video from the snapshots
	mra.create_equal_area_slice_video(output_filename="evolution.mp4", fps=10, vmin=-25, vmax=-22.)

				
