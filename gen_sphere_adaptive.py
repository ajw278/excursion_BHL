import numpy as np
from turbfuncs import *
from consts_defaults import *
import matplotlib.pyplot as plt
import excursion as exc
import cloud as cl
from scipy.ndimage import zoom
from scipy.special import erfinv
import pickle
import os
import matplotlib.animation as animation



plt.rc('text', usetex=True)

class MultiResolutionArray:
	def __init__(self, snapshot_dir='snapshots', imaxcoll=None, rmax=200., rspatial=30.0, rmin=0.3, dr=0.8, n0=1, max_cells_per_dim=256):
		"""
		Initialize the MultiResolutionArray. If a file with the given filename exists,
		load the object from the file. Otherwise, initialize the object and save it.

		Args:
			grid (trajectory_spatial_grid): The grid object containing rlevels and physical properties.
			filename (str): File path to save or load the object.

		Returns:
			None
		"""
		self.max_cells_per_dim = int(max_cells_per_dim)

		pc2cm = 3.086e18  # Example constant
		self.define_spatial_scales(rmax*pc2cm, rspatial*pc2cm, rmin*pc2cm, dr, n0)
		self.grid = exc.trajectory_grid(self.scales)
		print(self.grid.rlevels)
		self.generate_resolutions()
		print(f"Initialized MultiResolutionArray with {len(self.resolutions)} resolutions.")


		self.snapshot_dir = snapshot_dir
		if imaxcoll is None:
			iall= np.arange(len(self.resolutions))
			imaxcoll = int(np.percentile(iall, 90.0))
		
		self.imaxcoll = imaxcoll
		self.t = 0.0

		self.clouds =  []
		self.icollapsed = [np.zeros(self.resolutions[ir].shape) for ir in range(len(self.resolutions[:imaxcoll]))]

		# Create snapshot directory if it doesn't exist
		os.makedirs(self.snapshot_dir, exist_ok=True)


		print("Level  idx |   N   |    dx[pc]   |   L[pc]   | local")
		pc2cm = 3.086e18
		for i,(N,dx,L,loc) in enumerate(zip(self.n_res, self.spatial_scales, self.level_length, self.level_is_local)):
			print(f"{i:>7} | {N:>5} | {dx/pc2cm:>10.4g} | {L/pc2cm:>8.4g} | {loc}")

		self.plot_grid_cells_slices(levels=None, planes=('xy','xz','yz'),
                           s=1, alpha=0.4, figsize=(12, 4), save=None)

		exit()

	
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
		scales = [rmax]
		super_scales = [rmax]
		ir = 0
		while scales[-1] * dr > rspatial:
			scales.append(scales[-1] * dr)
			super_scales.append(scales[-1])
			ir += 1
		scales.append(rspatial)  # Ensure rspatial is included
		self.super_scales = super_scales  # these feed the turbulence model

		# Base/domain definitions
		# spatial_scales[i] will be dx_i
		n_res = [n0]
		rscale = rspatial  # this will be treated as dx_0 = rspatial / n0 below
		spatial_scales = [rspatial / float(n0)]  # dx_0
		spatial_level = [ir]  # index into the turbulence grid arrays

		# We’ll track the physical length each level actually covers
		level_length = [n_res[0] * spatial_scales[0]]  # = rspatial
		level_is_local = [False]  # base level covers full domain by construction

		# Grow resolution until rscale>rmin, as before
		while rscale > rmin:
			# desired number of cells if we kept covering the full domain
			next_n_res_desired = max(int(n_res[-1] / dr), n_res[-1] + 1)
			# dx for the next level if it *were* to cover the full domain
			next_dx = (rspatial * float(n0)) / float(next_n_res_desired)  # = L_domain / N_desired

			# cap N if needed
			if next_n_res_desired > self.max_cells_per_dim:
				next_n_res = self.max_cells_per_dim
				# if we cap N, we keep the dx implied by the cascade (next_dx)
				# which means the physical size shrinks:
				this_level_length = next_n_res * next_dx
				is_local = True
			else:
				next_n_res = next_n_res_desired
				this_level_length = rspatial  # still covers full domain
				is_local = False

			n_res.append(next_n_res)
			spatial_scales.append(next_dx)   # store dx_i
			level_length.append(this_level_length)
			ir += 1
			spatial_level.append(ir)
			scales.append(next_dx)
			level_is_local.append(is_local)

			# update rscale for the loop condition – it’s the *cell size* at this level
			rscale = next_dx

		self.scales = np.array(scales)                    # for exc.trajectory_grid
		self.spatial_scales = np.array(spatial_scales)    # dx per level
		self.n_res = n_res                                # N per level (after capping)
		self.spatial_level = spatial_level
		self.level_length = np.array(level_length)        # physical size each level covers
		self.level_is_local = np.array(level_is_local)

		# Convenience: global domain length from the base level
		self.L_domain = self.spatial_scales[0] * self.n_res[0]


		return self.scales


	def level_cell_centers(self, i):
		"""
		Exact 1D centers for level i, centered at 0.
		Returns: np.ndarray of shape (N_i,)
		"""
		N = int(self.n_res[i])
		dx = float(self.spatial_scales[i])
		L  = float(self.level_length[i])
		return (np.arange(N) + 0.5) * dx - L / 2.0

	def plot_grid_cells_slices(self, levels=None, planes=('xy','xz','yz'),
                           s=1, alpha=0.4, figsize=(12, 4), save=None):
		"""
		Plot the EXACT locations of grid-cell centers for each level on
		slices through the origin. No subsampling. Each slice uses the
		index closest to 0 along the orthogonal axis.

		Args:
			levels: list of level indices to plot (default: all)
			planes: any of ('xy','xz','yz')
			s: marker size
			alpha: marker alpha
			figsize: (W,H)
			save: filepath to save; if None, show().

		Notes:
			- For plane 'xy', we take z index nearest to 0, and plot all (x,y) centers.
			- Local (shrunk) levels appear as smaller, zero-centered grids.
			- Colors are per-level (consistent across subplots).
		"""
		if levels is None:
			levels = list(range(len(self.n_res)))
		planes = tuple(p.lower() for p in planes)
		valid = {'xy','xz','yz'}
		assert set(planes).issubset(valid), f"planes must be subset of {valid}"

		cmap = plt.get_cmap('tab10')
		colors = {i: cmap(k % 10) for k, i in enumerate(levels)}

		fig, axs = plt.subplots(1, len(planes), figsize=figsize, squeeze=False)
		axs = axs[0]

		# Global extent for consistent axes
		Lg = float(self.L_domain)
		ext = (-Lg/2.0, Lg/2.0)

		for ax, plane in zip(axs, planes):
			for i in levels:
				# exact 1D centers
				c = self.level_cell_centers(i)
				N = c.size

				# choose the index closest to 0 on the orthogonal axis
				k0 = int(np.argmin(np.abs(c)))

				if plane == 'xy':
					X, Y = np.meshgrid(c, c, indexing='ij')
					ax.scatter(X.ravel(), Y.ravel(), s=s, alpha=alpha, color=colors[i], label=f"lvl {i}")
					ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]')
					ax.set_xlim(ext); ax.set_ylim(ext)

				elif plane == 'xz':
					X, Z = np.meshgrid(c, c, indexing='ij')
					ax.scatter(X.ravel(), Z.ravel(), s=s, alpha=alpha, color=colors[i], label=f"lvl {i}")
					ax.set_xlabel('x [cm]'); ax.set_ylabel('z [cm]')
					ax.set_xlim(ext); ax.set_ylim(ext)

				elif plane == 'yz':
					Y, Z = np.meshgrid(c, c, indexing='ij')
					ax.scatter(Y.ravel(), Z.ravel(), s=s, alpha=alpha, color=colors[i], label=f"lvl {i}")
					ax.set_xlabel('y [cm]'); ax.set_ylabel('z [cm]')
					ax.set_xlim(ext); ax.set_ylim(ext)

			ax.set_aspect('equal', adjustable='box')
			ax.grid(True, ls=':', lw=0.5)

		# Single legend outside
		handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=colors[i],
							markersize=6, label=f"lvl {i}") for i in levels]
		axs[-1].legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0., fontsize=9)

		plt.tight_layout()
		if save:
			plt.savefig(save, dpi=200, bbox_inches='tight')
			plt.close(fig)
		else:
			plt.show()


	def generate_resolutions(self):
		"""
		Generate arrays corresponding to each resolution level.

		Returns:
			list of np.ndarray: Arrays for each level, covering the same spatial area at appropriate resolutions.
		"""
		super_resolutions = []
		super_resolutions_v = []

		print(f"Generating resolutions for {len(self.super_scales)} super resolutions and {len(self.spatial_scales)} spatial resolutions.")


		for ir, r in enumerate(self.super_scales):
			# Create and initialize the grid for this resolution level
			print(f"Initializing super resolution level {ir}")
			level_grid, level_grid_v = self.initialize_resolution(ir, 1)
			super_resolutions.append(level_grid)
			super_resolutions_v.append(level_grid_v)


		resolutions = []
		resolutions_v = []
		# Get the finest resolution (spanning the domain with rmax as the length scale)
		#finest_resolution_size = int(self.grid.rmax / self.grid.dr[-1])

		print(f"Generating spatial resolutions for {len(self.spatial_scales)} levels.")

		print(self.grid.Delta_S, self.grid.Delta_Sv, self.grid.tau_R, self.grid.rlevels)

		for ir, r in enumerate(self.spatial_scales):
			# Define the resolution for the current level based on the length scale
			print(f"Initializing spatial resolution level {ir} with size {self.n_res[ir]}x{self.n_res[ir]}x{self.n_res[ir]}")
			resolution_size = self.n_res[ir]
			
			# Create and initialize the grid for this resolution level
			level_grid, level_grid_v = self.initialize_resolution(self.spatial_level[ir], resolution_size)
			resolutions.append(level_grid)
			resolutions_v.append(level_grid_v)
		
		print(f"Generated {len(super_resolutions)} super resolutions and {len(resolutions)} spatial resolutions.")

		self.super_resolutions = super_resolutions
		self.super_resolutions_v = super_resolutions_v
		self.resolutions = resolutions
		self.resolutions_v = resolutions_v

		return None
	
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

	'''def evaluate_at_finest(self):
		"""
		Computes the sum of all levels at the finest resolution.

		Returns:
			np.ndarray: A 3D array with the sum of all resolutions, mapped to the finest grid.
		"""
		# Initialize the result array at the finest resolution
		total = np.zeros(self.resolutions[-1].shape, dtype=np.float64)
		total += np.sum(self.super_resolutions)


		total_v = np.zeros(self.resolutions_v[-1].shape, dtype=np.float64)
		total_v += np.sum(self.super_resolutions_v)

		for ir, level_array in enumerate(self.resolutions):
			# Compute the scaling factor for this level relative to the finest level
			factor = np.array(total.shape) / np.array(level_array.shape)

			# Upsample the current array to the finest resolution
			upsampled_array = zoom(level_array, factor, order=0)  

			# Add the upsampled array to the total
			total += upsampled_array

		
		for ir, level_array_v in enumerate(self.resolutions_v):
			# Compute the scaling factor for this level relative to the finest level
			factor = np.array(total_v.shape) / np.array(level_array_v.shape)

			# Upsample the current array to the finest resolution
			upsampled_array = zoom(level_array_v, factor, order=0)  

			# Add the upsampled array to the total
			total_v += upsampled_array

		return total, total_v'''
	
	def evaluate_at_finest(self):
		"""
		Sum all levels onto the finest full-domain grid.
		Local (shrunk) levels are embedded at the center with zero outside.
		"""
		# 1) pick the target level: finest that still spans full domain
		full_domain_idxs = np.where(~self.level_is_local)[0]
		if len(full_domain_idxs) == 0:
			# Fallback: if somehow every level is local, pick the coarsest as target
			i_target = 0
		else:
			i_target = full_domain_idxs[-1]

		N_target = self.n_res[i_target]
		dx_target = self.spatial_scales[i_target]
		L_target = self.level_length[i_target]  # should equal self.L_domain

		total = np.zeros((N_target, N_target, N_target), dtype=np.float64)
		total_v = np.zeros((N_target, N_target, N_target, 3), dtype=np.float64)

		# 2) Add super-scale contributions (same as before): they’re scalars (shape (1,1,1))
		total += np.sum(self.super_resolutions)
		total_v += np.sum(self.super_resolutions_v)

		# 3) Utility to upsample an array to a given integer size per axis with order=0
		def upsample_to(arr, out_shape):
			factors = np.array(out_shape) / np.array(arr.shape[:3], dtype=float)
			# zoom needs floats; order=0 for blocky nearest replication (as before)
			return zoom(arr, list(factors) + ([1] if arr.ndim == 4 else []), order=0)

		# 4) Loop levels and accumulate
		for i, arr in enumerate(self.resolutions):
			arr_v = self.resolutions_v[i]
			dx_i = self.spatial_scales[i]
			N_i = self.n_res[i]
			L_i = self.level_length[i]
			is_local = self.level_is_local[i]

			if not is_local:
				# Covers the full domain: upsample to the full target cube and add
				up = upsample_to(arr, (N_target, N_target, N_target))
				total += up

				up_v = upsample_to(arr_v, (N_target, N_target, N_target))
				total_v += up_v
			else:
				# Local level: compute how many target cells this level covers
				# number of target cells along one axis that match the physical size L_i
				n_cov = int(round(L_i / dx_target))
				n_cov = max(1, min(n_cov, N_target))  # clamp for safety

				# Upsample arr to its coverage size in target resolution
				up_local = upsample_to(arr, (n_cov, n_cov, n_cov))
				up_local_v = upsample_to(arr_v, (n_cov, n_cov, n_cov))

				# Compute centered placement indices in the target cube
				start = (N_target - n_cov) // 2
				end = start + n_cov
				slicer = np.s_[start:end, start:end, start:end]

				total[slicer] += up_local
				total_v[slicer] += up_local_v

		return total, total_v


	
	def update_resolutions(self, dt):
		"""
		Update the resolutions for a given time-step.

		Args:
			dt (float): Time-step value.

		Returns:
			None: Updates self.resolutions in place.
		"""

		#Go over the super resolution scales first (larger than grid)

		for ilevel in range(len(self.super_resolutions)):
			# Compute exponential decay term
			exp_decay = np.exp(-dt / self.grid.tau_R[ilevel])


			# Update the current resolution level
			Ddelta_new = self.super_resolutions[ilevel] * exp_decay
			random_term = np.random.normal(
				loc=0.0,
				scale=1.0,
				size=self.super_resolutions[ilevel].shape
			)
			Ddelta_new += random_term * np.sqrt(
				self.grid.Delta_S[ilevel] * (1.0 - np.exp(-2.0 * dt / self.grid.tau_R[ilevel]))
			)

			# Update the resolution level in place
			self.super_resolutions[ilevel] = Ddelta_new


			# Update the current resolution level
			Ddeltav_new = self.super_resolutions_v[ilevel] * exp_decay
			random_term_v = np.random.normal(
				loc=0.0,
				scale=1.0,
				size=self.super_resolutions_v[ilevel].shape
			)
			Ddeltav_new += random_term_v * np.sqrt(
				self.grid.Delta_Sv[ilevel] * (1.0 - np.exp(-2.0 * dt / self.grid.tau_R[ilevel]))
			)

			# Update the resolution level in place
			self.super_resolutions_v[ilevel] = Ddeltav_new


		#Now update over the grid-level spatial scales
		for ilevel in range(len(self.resolutions)):
			# Compute exponential decay term
			ilevel_spatial = self.spatial_level[ilevel]
			exp_decay = np.exp(-dt / self.grid.tau_R[ilevel_spatial])

			# Update the current resolution level
			Ddelta_new = self.resolutions[ilevel] * exp_decay
			Ddeltav_new = self.resolutions_v[ilevel] * exp_decay
			random_term = np.random.normal(
				loc=0.0,
				scale=1.0,
				size=self.resolutions[ilevel].shape
			)

			random_term_v = np.random.normal(
				loc=0.0,
				scale=1.0,
				size=self.resolutions_v[ilevel].shape
			)
			Ddelta_new += random_term * np.sqrt(
				self.grid.Delta_S[ilevel_spatial] * (1.0 - np.exp(-2.0 * dt / self.grid.tau_R[ilevel_spatial]))
			)

			Ddeltav_new += random_term_v * np.sqrt(
				self.grid.Delta_Sv[ilevel_spatial] * (1.0 - np.exp(-2.0 * dt / self.grid.tau_R[ilevel_spatial]))
			)

			# Update the resolution level in place
			self.resolutions[ilevel] = Ddelta_new
			self.resolutions_v[ilevel] = Ddeltav_new

	

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
			print(t / year2s / 1e6, "Myr", end="\r")
			self.update_resolutions(dt)
			t += dt
			self.t = t


			"""# Evolve clouds
			for cloud in self.clouds:
				if not cloud.dispersed:
					cloud.evolve(t, rejuvinate=False)

			self.find_collapse()"""

			if t >= next_snapshot_time:
				print('Saving...')
				self.save_snapshot(snapshot_idx, t)
				snapshot_times.append(t)
				np.save(snapshot_times_file, np.array(snapshot_times))  # Update time tracking
				snapshot_idx += 1
				next_snapshot_time += dt_snap_sec

		print(f"Evolution completed: Total time = {Tend} Myr")
	
	def save_snapshot(self, snapshot_idx, time):
		"""
		Save the finest resolution density and cloud properties.

		Args:
			snapshot_idx (int): Index of the snapshot.
			time (float): Time of the snapshot (seconds).

		Returns:
			None
		"""
		density_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_density.npy")
		velocity_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_velocity.npy")
		coords_filename = os.path.join(self.snapshot_dir, "snapshot_coords.npy")

		
		lnrho_norm, v =  self.evaluate_at_finest()
		# Compute finest density
		volume_density = self.grid.rho0*np.exp(lnrho_norm)

		# Save updated density
		np.save(density_filename, volume_density)
		np.save(velocity_filename, v)

		# Save the coordinate array (1D physical positions), only once
		if not os.path.exists(coords_filename):
			finest_level = len(self.resolutions) - 1
			N = self.resolutions[finest_level].shape[0]
			rmax = self.spatial_scales[0] * self.n_res[0]  # total physical length
			dx = rmax / N
			coords_1d = (np.arange(N) + 0.5) * dx - rmax / 2.0
			np.save(coords_filename, coords_1d)
			print(f"Saved grid coordinates to {coords_filename}")

		print(f"Snapshot {snapshot_idx} saved at time {time / year2s / 1e6:.2f} Myr")



	def create_video(self, output_filename="simulation.mp4", fps=10):
		"""
		Create an MP4 video using precomputed density snapshots.

		- Loads saved `snapshot_times.npy` for time labels.
		- Loads the finest-resolution density for each snapshot.
		- Optionally overlays cloud positions as circles.
		
		Args:
			output_filename (str): The name of the output MP4 file.
			fps (int): Frames per second for the video.
		
		Returns:
			None
		"""
		# Load snapshot indices
		snapshot_indices = sorted(set(
			int(f.split("_")[1]) for f in os.listdir(self.snapshot_dir) if f.endswith("_density.npy")
		))

		if not snapshot_indices:
			print("No precomputed densities found. Exiting video creation.")
			return

		# Load time array
		snapshot_times_file = os.path.join(self.snapshot_dir, "snapshot_times.npy")
		if os.path.exists(snapshot_times_file):
			snapshot_times = np.load(snapshot_times_file)
		else:
			snapshot_times = np.array([i for i in range(len(snapshot_indices))])  # Dummy time array

		print(f"Creating video from {len(snapshot_indices)} density snapshots...")

		# Load first snapshot to initialize figure
		first_density_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_indices[0]:04d}_density.npy")
		first_density = np.load(first_density_filename)

		# Select slice for visualization (midplane cut in z)
		first_surface_density =  self.compute_surface_density(first_density)/(mu_mass*mH)

		# Compute 10th-90th percentile for color normalization
		vmin, vmax = np.percentile(first_surface_density, [10, 90])
		vmin = 20.0
		vmax = 22.5

		# Setup figure
		fig, ax = plt.subplots(figsize=(8, 6))
		rmax  = self.spatial_scales[0]*self.n_res[0]
		print(f"rmax: {rmax/pc2cm:.2f} pc")
		extent = [-rmax / pc2cm /2., rmax / pc2cm/2.,-rmax / pc2cm /2., rmax / pc2cm/2.]
		im = ax.imshow(np.log10(first_surface_density.T), extent=extent, origin="lower", aspect="auto", cmap="hot", vmin=vmin, vmax=vmax)

		# Add colorbar
		cbar = plt.colorbar(im, ax=ax, label=r"log $N_{\mathrm{H}}$ [$\mathrm{cm^{-2}}$]")

		# Set labels
		ax.set_xlabel(r"$x \, [\mathrm{pc}]$")
		ax.set_ylabel(r"$y \, [\mathrm{pc}]$")
		time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, color="white", fontsize=12)

		# Function to update the animation
		def update(frame_idx):
			snapshot_idx = snapshot_indices[frame_idx]
			density_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_density.npy")
			cloud_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_clouds.npy")

			# Load density
			density = np.load(density_filename)
			surface_density = self.compute_surface_density(density)/(mu_mass*mH)  # Midplane slice


			im.set_array(np.log10(surface_density.T))

			# Clear previous circles
			for coll in ax.collections:
				coll.remove()
			[p.remove() for p in reversed(ax.patches)]

			# Load clouds for this snapshot (if available)
			if os.path.exists(cloud_filename):
				cloud_data = np.load(cloud_filename)
				positions = cloud_data[:3, :].T
				sizes = cloud_data[3, :]

				# Convert to PC units
				pos_pc = positions / pc2cm
				sizes_pc = sizes / pc2cm 

				# Overlay clouds
				for pos, size in zip(pos_pc, sizes_pc):
					circle = plt.Circle((pos[0], pos[1]), size, color="cyan", fill=False, linewidth=1)
					ax.add_patch(circle)

			# Update time label
			time_text.set_text(f"Time = {snapshot_times[frame_idx] / 1e6 / year2s:.2f} Myr")
			print(time_text)
			return [im, time_text]

		# Create the animation
		ani = animation.FuncAnimation(fig, update, frames=len(snapshot_indices), blit=False)

		# Save the animation as MP4
		ani.save(output_filename, writer="ffmpeg", fps=fps)
		plt.close(fig)
		print(f"Video saved as {output_filename}.")

	
	def get_spatial_coordinates(self, ilevel, index):
		"""
		Compute the spatial coordinates of cell centers given their indices at a specific resolution level.

		Args:
			ilevel (int): The resolution level index.
			index (tuple or np.ndarray): A tuple (i, j, k) for a single point or an array of shape (N, 3) for multiple points.

		Returns:
			np.ndarray: A tuple (x, y, z) for a single index or an array of shape (N, 3) for multiple indices.
		"""
		# Compute grid spacing at this level
		rmax  = self.spatial_scales[0]*self.n_res[0]
		N = self.n_res[ilevel]  # Number of cells along each dimension
		dx = rmax / N  # Grid spacing at this level

		# Convert to NumPy array if needed
		index = np.asarray(index)

		# Compute spatial coordinates using vectorized operations
		coords = (index + 0.5) * dx -  rmax/2.

		return coords

	
	def precompute_volume_densities(self):
		"""
		Precompute and save volume densities, overriding grid cells where clouds exist.
		"""
		snapshot_indices = sorted(set(
			int(f.split("_")[1]) for f in os.listdir(self.snapshot_dir) if f.endswith("_density.npy")
		))

		if not snapshot_indices:
			print("No snapshot files found.")
			return

		print(f"Precomputing volume densities for {len(snapshot_indices)} snapshots...")

		for snapshot_idx in snapshot_indices:
			density_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_density.npy")
			
			# Load density
			volume_density = self.grid.rho0*np.exp(np.load(density_filename))

	
	def compute_surface_density(self, volume_density):
		"""
		Compute the surface density from a precomputed volume density.

		Args:
			volume_density (np.ndarray): The 3D volume density array.

		Returns:
			np.ndarray: The 2D surface density array.
		"""
		dz = self.spatial_scales[0] / volume_density.shape[2]  # Cell depth
		surface_density = np.sum(volume_density, axis=2) * dz 
		return surface_density

# Assuming trajectory_spatial_grid is properly defined
pc2cm = 3.086e18  # Example constant
year2s = 3.154e7  # Seconds in a year
#grid = exc.trajectory_grid(rmax=30.0 * pc2cm, rmin=0.2 * pc2cm, drfact=0.5)


# Initialize the MultiResolutionArray§
mra = MultiResolutionArray(rmax=200.0, rspatial=1.0, rmin=0.001, dr=0.2)
# Evolve for 10 Myr, storing snapshots every 1 Myr
mra.evolve(Tend=10.0, fraction_of_tau=0.1, dt_snap=0.1)

# Create an MP4 video from the snapshots
mra.create_video(output_filename="evolution.mp4", fps=10)

			
