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
	def __init__(self, snapshot_dir='snapshots', imaxcoll=None, rmax=200., rspatial=30.0, rmin=0.3, dr=0.8, n0=1):
		"""
		Initialize the MultiResolutionArray. If a file with the given filename exists,
		load the object from the file. Otherwise, initialize the object and save it.

		Args:
			grid (trajectory_spatial_grid): The grid object containing rlevels and physical properties.
			filename (str): File path to save or load the object.

		Returns:
			None
		"""

		pc2cm = 3.086e18  # Example constant
		self.define_spatial_scales(rmax*pc2cm, rspatial*pc2cm, rmin*pc2cm, dr, n0)
		self.grid = exc.trajectory_grid(self.scales)
		self.generate_resolutions()

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

	
	def define_spatial_scales(self, rmax, rspatial, rmin, dr, n0):
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

		return self.scales



	def generate_resolutions(self):
		"""
		Generate arrays corresponding to each resolution level.

		Returns:
			list of np.ndarray: Arrays for each level, covering the same spatial area at appropriate resolutions.
		"""
		super_resolutions = []
		for ir, r in enumerate(self.super_scales):
			# Create and initialize the grid for this resolution level
			level_grid = self.initialize_resolution(ir, 1)
			super_resolutions.append(level_grid)


		resolutions = []
		# Get the finest resolution (spanning the domain with rmax as the length scale)
		#finest_resolution_size = int(self.grid.rmax / self.grid.dr[-1])

		for ir, r in enumerate(self.spatial_scales):
			# Define the resolution for the current level based on the length scale
			resolution_size = self.n_res[ir]
			
			# Create and initialize the grid for this resolution level
			level_grid = self.initialize_resolution(self.spatial_level[ir], resolution_size)
			resolutions.append(level_grid)

		self.super_resolutions = super_resolutions
		self.resolutions = resolutions

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

		# Create a uniform random array
		u_delta = np.random.uniform(size=(resolution_size, resolution_size, resolution_size))

		# Transform the uniform distribution using the inverse error function
		initialized_grid = np.sqrt(2.0 * DS) * erfinv(2.0 * u_delta - 1.0)

		return initialized_grid

	def evaluate_at_finest(self):
		"""
		Computes the sum of all levels at the finest resolution.

		Returns:
			np.ndarray: A 3D array with the sum of all resolutions, mapped to the finest grid.
		"""
		# Initialize the result array at the finest resolution
		total = np.zeros(self.resolutions[-1].shape, dtype=np.float64)
		total += np.sum(self.super_resolutions)

		for ir, level_array in enumerate(self.resolutions):
			# Compute the scaling factor for this level relative to the finest level
			factor = np.array(total.shape) / np.array(level_array.shape)

			# Upsample the current array to the finest resolution
			upsampled_array = zoom(level_array, factor, order=0)  

			# Add the upsampled array to the total
			total += upsampled_array

		return total
	
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


		#Now update over the grid-level spatial scales
		for ilevel in range(len(self.resolutions)):
			# Compute exponential decay term
			ilevel_spatial = self.spatial_level[ilevel]
			exp_decay = np.exp(-dt / self.grid.tau_R[ilevel_spatial])

			# Update the current resolution level
			Ddelta_new = self.resolutions[ilevel] * exp_decay
			random_term = np.random.normal(
				loc=0.0,
				scale=1.0,
				size=self.resolutions[ilevel].shape
			)
			Ddelta_new += random_term * np.sqrt(
				self.grid.Delta_S[ilevel_spatial] * (1.0 - np.exp(-2.0 * dt / self.grid.tau_R[ilevel_spatial]))
			)

			# Update the resolution level in place
			self.resolutions[ilevel] = Ddelta_new

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
			self.update_resolutions(dt)
			t += dt
			self.t = t

			# Evolve clouds
			for cloud in self.clouds:
				if not cloud.dispersed:
					cloud.evolve(t, rejuvinate=False)

			self.find_collapse()

			if t >= next_snapshot_time:
				print('Saving...')
				self.save_snapshot(snapshot_idx, t)
				snapshot_times.append(t)
				np.save(snapshot_times_file, np.array(snapshot_times))  # Update time tracking
				snapshot_idx += 1
				next_snapshot_time += dt_snap_sec

		print(f"Evolution completed: Total time = {Tend} Myr")

	def save_snapshot_old(self, snapshot_idx, time):
		"""
		Save the finest resolution density and cloud properties.

		Args:
			snapshot_idx (int): Index of the snapshot.
			time (float): Time of the snapshot (seconds).

		Returns:
			None
		"""
		density_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_density.npy")
		clouds_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_clouds.npy")

		# Compute finest density
		finest_density = self.evaluate_at_finest()

		# Save density
		np.save(density_filename, finest_density)

		# Save cloud properties
		if self.clouds:
			survclouds= [] 
			for i, cloud in enumerate(self.clouds):
				if not cloud.dispersed:
					survclouds.append(cloud)

			cloud_data = np.zeros((8, len(survclouds)))  # 4 properties: (pos, size, vel, density)
			for i, cloud in enumerate(survclouds):
				cloud_data[:3, i] = cloud.r  # Position
				cloud_data[3, i] = cloud.R    # Size
				cloud_data[4:7, i] = cloud.v
				cloud_data[7, i] = cloud.rho_med

			np.save(clouds_filename, cloud_data)

		print(f"Snapshot {snapshot_idx} saved at time {time / year2s / 1e6:.2f} Myr")

	
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
		clouds_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_clouds.npy")

		# Compute finest density
		volume_density = self.grid.rho0*np.exp(self.evaluate_at_finest())

		# Save cloud properties
		if self.clouds:
			survclouds= [] 
			for i, cloud in enumerate(self.clouds):
				if not cloud.dispersed:
					survclouds.append(cloud)

			cloud_data = np.zeros((8, len(survclouds)))  # 4 properties: (pos, size, vel, density)
			for i, cloud in enumerate(survclouds):
				cloud_data[:3, i] = cloud.r  # Position
				cloud_data[3, i] = cloud.R    # Size
				cloud_data[4:7, i] = cloud.v
				cloud_data[7, i] = cloud.rho_med

			np.save(clouds_filename, cloud_data)

		# Load clouds
		if os.path.exists(clouds_filename):
			cloud_data = np.load(clouds_filename)
			positions = cloud_data[:3, :].T
			radii = cloud_data[3, :]
			#velocities = cloud_data[4:7, :].T
			densities = cloud_data[7, :]

			# Create grid arrays
			grid_shape = volume_density.shape
			rmax = self.spatial_scales[0]
			dx_finest = 2 * rmax / grid_shape[0]
			x_vals = np.linspace(-rmax + dx_finest / 2, rmax - dx_finest / 2, grid_shape[0])
			y_vals = np.linspace(-rmax + dx_finest / 2, rmax - dx_finest / 2, grid_shape[1])
			z_vals = np.linspace(-rmax + dx_finest / 2, rmax - dx_finest / 2, grid_shape[2])
			X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

			# Override density where clouds exist
			for pos, R, rho in zip(positions, radii, densities):
				dist_squared = (X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2
				volume_density[dist_squared <= R**2] = rho

		# Save updated density
		np.save(density_filename, volume_density)

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
		rmax  = self.spatial_scales[0]
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

	def cloud_check(self, position, radius):
		"""
		Check if the given position and radius overlap with any existing clouds.
		
		Args:
			position (np.ndarray): The (x, y, z) position of the new potential cloud.
			radius (float): The radius of the new potential cloud.
		
		Returns:
			int: -1 if the position is within an existing cloud,
					index i if the new cloud overlaps with an existing cloud,
					-2 if no nearby clouds exist.
		"""
		for i, cloud in enumerate(self.clouds):
			if cloud.dispersed:
				continue
			
			position_i = cloud.r
			radius_i = cloud.R
			
			# Compute distance between cloud centers
			distance = np.linalg.norm(position - position_i)
			
			if distance <= radius_i:
				return -1  # Position is within an existing cloud
			elif distance <= radius_i + radius:
				return i  # Overlaps with an existing cloud
			
		return -2  # No nearby cloud detected


	def find_collapse(self, **kwargs):
		"""
		"""

		if not hasattr(self, 'last_formation_time'):
			# Initialize formation tracking array (set to -inf so all regions are initially "unlocked")
			self.next_formation_time = [np.full(res.shape, -np.inf) for res in self.resolutions[:self.imaxcoll]]

		delta_sum_ = None

		if delta_sum_ is None:
			delta_sum_ = np.zeros(self.resolutions[0].shape)

		delta_sum_ += np.sum(self.super_resolutions)
		print('Super resolution:', np.sum(self.super_resolutions))

		inew=0
		for ir_, level_array in enumerate(self.resolutions[:self.imaxcoll]):
			ir = self.spatial_level[ir_]
			#print(self.grid.rlevels[ir]/pc2cm)
			if ir_>1:
				delta_c = self.grid.delta_c[ir]
				factor =  np.array(level_array.shape)/np.array(delta_sum_.shape)
				upsampled_array = zoom(delta_sum_, factor, order=0, grid_mode=False)
				delta_sum = upsampled_array+level_array

				#Check for any collapse on larger scales
				#self.icollapsed[ir_][delta_sum>1e5] = 1

				#Identify new collapses
				icollapse = (delta_sum>delta_c) #&(self.icollapsed[ir_]==0)

				for coll_ind in np.swapaxes(np.where(icollapse), 0, 1):
					position = self.get_spatial_coordinates(ir_, coll_ind)
					t_current = self.t  # Current simulation time

					if t_current<self.next_formation_time[ir_][tuple(coll_ind)]:
						print('Turbulent scale cannot produce more SFRs')
						continue

					"""locked=False
					# Check if this cell (or finer cells) are locked
					for coarse_ir_ in range(2, ir_)[::-1]:  # Only check coarser scales
						coarse_ir = self.spatial_level[coarse_ir_]
						coarse_factor = np.array(self.resolutions[coarse_ir_].shape) / np.array(self.resolutions[ir_].shape)
						coarse_idx = (np.array(coll_ind) * coarse_factor).astype(int)
						print(coll_ind, coarse_idx, coarse_factor)
						#coarse_idx = np.clip((np.array(coll_ind) / coarse_factor).astype(int), 0, np.array(self.resolutions[coarse_ir].shape) - 1)

						# If a collapse happened recently at a coarser scale, block it
						if t_current - self.last_formation_time[coarse_ir_][tuple(coarse_idx)] < self.grid.tau_R[coarse_ir]:
							locked=True
							break  # Skip this cell, it's still in "lock" phase

					if locked:
						continue"""
					

					check = self.cloud_check(position, self.grid.rlevels[ir]/2.)
					if check!=-2:
						continue 
					
					# If allowed, form new cloud
					self.clouds.append(cl.bound_clump(**kwargs))
					inew +=1

					# Assign cloud properties
					velocity = 2.*np.random.normal(size=3) * km2cm
					self.clouds[-1].form_nontraj(ir, self.grid, position, velocity, t_current)
					print('New cloud:', self.grid.rlevels[ir]/pc2cm)

					# Record formation time at this level
					self.next_formation_time[ir_][tuple(coll_ind)] = t_current + self.grid.tau_R[ir]

					# **Propagate Lock to Finer Levels**
					for finer_ir_ in range(ir_ + 1, len(self.resolutions[:self.imaxcoll])):
						finer_grid_shape = self.resolutions[finer_ir_].shape
						finer_factor = np.array(finer_grid_shape) / np.array(self.resolutions[ir_].shape)

						# Compute corresponding finer grid indices
						finer_indices = (np.array(coll_ind) * finer_factor).astype(int)

						# Apply lock to finer grid cells using `tau_R[ir]`
						slices = tuple(slice(fi, fi + 1) for fi in finer_indices)  # Select the finer region
						self.next_formation_time[finer_ir_][slices] = t_current + self.grid.tau_R[ir]

				#self.icollapsed[ir_][icollapse] = 1

				#Add large number to collapsed delta in order to flag that smaller scale cells 
				#have already collapsed
				#delta_sum[icollapse] = 1e9
		print('Collapse check complete. %d new collapses.'%inew)


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
			clouds_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_clouds.npy")

			# Load density
			volume_density = self.grid.rho0*np.exp(np.load(density_filename))

			# Load clouds
			if os.path.exists(clouds_filename):
				cloud_data = np.load(clouds_filename)
				positions = cloud_data[:3, :].T
				radii = cloud_data[3, :]
				#velocities = cloud_data[4:7, :].T
				densities = cloud_data[7, :]

				# Create grid arrays
				grid_shape = volume_density.shape
				rmax = self.spatial_scales[0]
				dx_finest = rmax / grid_shape[0]
				x_vals = np.linspace(-rmax/2. + dx_finest / 2, rmax/2. - dx_finest / 2, grid_shape[0])
				y_vals = np.linspace(-rmax/2. + dx_finest / 2, rmax/2. - dx_finest / 2, grid_shape[1])
				z_vals = np.linspace(-rmax/2. + dx_finest / 2,rmax/2. - dx_finest / 2, grid_shape[2])
				X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

				# Override density where clouds exist
				for pos, R, rho in zip(positions, radii, densities):
					dist_squared = (X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2
					volume_density[dist_squared <= R**2] = rho

			# Save updated density
			np.save(density_filename, volume_density)
			print(f"Updated volume density: {density_filename}")

	
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
mra = MultiResolutionArray()

# Evolve for 10 Myr, storing snapshots every 1 Myr
mra.evolve(Tend=10.0, fraction_of_tau=0.1, dt_snap=0.1)

# Create an MP4 video from the snapshots
mra.create_video(output_filename="evolution.mp4", fps=10)

			
