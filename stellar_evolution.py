import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

def find_closest_mass_file(directory, target_mass, tol= 0.5):
	closest_mass = None
	closest_filename = None

	for filename in os.listdir(directory):
		if filename.endswith('.track.eep'):
			try:
				mass_str = filename.split('M')[0]
				mass = float(mass_str) / 100  # Convert to solar masses assuming filename like '00037M.track.eep'
				if closest_mass is None or abs(mass - target_mass) < abs(closest_mass - target_mass):
					closest_mass = mass
					closest_filename = filename
			except ValueError:
				continue

	
	if closest_mass is None:
		raise Warning('No stellar model file found.')
	if abs(closest_mass-target_mass)/target_mass>tol and target_mass>0.1:
		
		raise Warning('No stellar model within %d percent of specified initial mass found.'%(tol*100))
		

	return closest_filename, closest_mass

def extract_eep_data(filepath):
	# Define the columns based on the README
	columns = [
	'star_age',  'star_mass', 'log_L', 'log_Teff', 'log_R', 'log_g'
	]

	# Read the data
	data = pd.read_csv(filepath, delim_whitespace=True, comment='#', names=columns, usecols=[0, 1, 6, 11, 13, 14])

	# Convert log values to their actual values
	data['L'] = 10**data['log_L']
	data['Teff'] = 10**data['log_Teff']
	data['R'] = 10**data['log_R']

	return data
    
def interpolate_stellar_properties(data, target_age):
	# Ensure the data is sorted by age
	data = data.sort_values('star_age')

	# Interpolation functions
	interpolate_Teff = interp1d(data['star_age'], data['Teff'], kind='linear', fill_value="extrapolate")
	interpolate_log_g = interp1d(data['star_age'], data['log_g'], kind='linear', fill_value="extrapolate")
	interpolate_log_L = interp1d(data['star_age'], data['log_L'], kind='linear', fill_value="extrapolate")
	interpolate_R = interp1d(data['star_age'], data['R'], kind='linear', fill_value="extrapolate")
	interpolate_mass = interp1d(data['star_age'], data['star_mass'], kind='linear', fill_value="extrapolate")

	# Interpolate values at target age
	Teff = interpolate_Teff(target_age)
	log_g = interpolate_log_g(target_age)
	log_L = interpolate_log_L(target_age)
	R = interpolate_R(target_age)
	star_mass = interpolate_mass(target_age)

	return Teff, log_g, log_L, R, star_mass

def example_plot(directory, target_mass):
	closest_filename, closest_mass = find_closest_mass_file(directory, target_mass)

	print(f"Closest mass file: {closest_filename} with mass {closest_mass} M_sun")

	filepath = os.path.join(directory, closest_filename)
	eep_data = extract_eep_data(filepath)
	# Plotting the results
	import matplotlib.pyplot as plt
	plt.rc('text', usetex=True)
	plt.figure(figsize=(10, 6))

	plt.subplot(2, 2, 1)
	plt.plot(eep_data['star_age'], eep_data['Teff'])
	plt.xlabel('Age [years]')
	plt.ylabel('Effective Temperature [K]')
	plt.yscale('log')
	plt.xscale('log')

	plt.subplot(2, 2, 2)
	plt.plot(eep_data['star_age'], eep_data['log_g'])
	plt.xlabel('Age [years]')
	plt.ylabel('$\log g$')
	plt.xscale('log')

	plt.subplot(2, 2, 3)
	plt.plot(eep_data['star_age'], eep_data['star_mass'])
	plt.xlabel('Age [years]')
	plt.yscale('log')
	plt.xscale('log')
	plt.ylabel('Stellar Mass [$M_\odot$]')
	plt.subplot(2, 2, 4)
	plt.plot(eep_data['star_age'], eep_data['L'])
	plt.xlabel('Age (years)')
	plt.ylabel('Luminosity [$L_\odot$]')
	plt.yscale('log')
	plt.xscale('log')
	plt.tight_layout()
	plt.show()

"""def fetch_stellar_properties(minit, age_years, directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS'):
	closest_filename, closest_mass = find_closest_mass_file(directory, minit)
	
	filepath = os.path.join(directory, closest_filename)
	eep_data = extract_eep_data(filepath)

	Teff, log_g, log_L, R, star_mass = interpolate_stellar_properties(eep_data, age_years)

	return Teff, log_g, log_L, R, star_mass"""


def fetch_stellar_properties(minit, age_years,  directory='MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS'):
	if minit>1.4:
		closest_filename, closest_mass = find_closest_mass_file(directory, minit)
		
		filepath = os.path.join(directory, closest_filename)
		eep_data = extract_eep_data(filepath)

		Teff, log_g, log_L, R, star_mass = interpolate_stellar_properties(eep_data, age_years)

		return Teff, log_g, log_L, R, star_mass
	else:
	
		# Load the provided CSV file and skip initial header rows
		file_path = 'BHAC15_tracks.csv'
		df_clean = pd.read_csv(file_path, comment='!', delim_whitespace=True)

		# Add appropriate column headers based on the description given
		column_names = [
		    "M/Ms", "log_t", "Teff", "L/Ls", "g", "R/Rs", "log(Li/Li0)", "log_Tc", "log_ROc",
		    "Mrad", "Rrad", "k2conv", "k2rad"
		]
		df_clean.columns = column_names

		
		# Convert the mass column to numeric, forcing errors to NaN (which we will drop)
		df_clean["M/Ms"] = pd.to_numeric(df_clean["M/Ms"], errors='coerce')
		min_age = 10.**np.amin(df_clean["log_t"])
		min_mass = np.amin(df_clean["M/Ms"])

		# Drop rows where the mass column has NaN (i.e., non-numeric values were present)
		df_clean_numeric = df_clean.dropna(subset=["M/Ms"])

		
		# Filter the DataFrame for relevant columns and convert mass and age to log space
		df_clean_numeric['log_M/Ms'] = np.log10(df_clean_numeric['M/Ms'])


		# Prepare data for interpolation: extract mass, age, and the columns we want to interpolate
		points = np.array([df_clean_numeric['log_M/Ms'], df_clean_numeric['log_t']]).T
		teff_values = df_clean_numeric['Teff']
		logg_values = df_clean_numeric['g']
		logL_values = df_clean_numeric['L/Ls']
		radius_values = df_clean_numeric['R/Rs']

		log_mass_target = np.log10(minit)
		log_age_target = np.log10(age_years)

			# Check if target values are within data range
		min_log_mass, max_log_mass = np.min(points[:, 0]), np.max(points[:, 0])
		min_log_age, max_log_age = np.min(points[:, 1]), np.max(points[:, 1])

		if not (min_log_mass <= log_mass_target <= max_log_mass and min_log_age <= log_age_target <= max_log_age):
			interpolation_method = 'nearest'  # Use nearest neighbor if outside the data range
		else:
			interpolation_method = 'linear'  # Use linear interpolation if inside the range



		# Perform 2D interpolation using griddata in log space
		interpolated_teff = griddata(points, teff_values, (log_mass_target, log_age_target), method=interpolation_method)
		interpolated_logg = griddata(points, logg_values, (log_mass_target, log_age_target), method=interpolation_method)
		interpolated_logL = griddata(points, logL_values, (log_mass_target, log_age_target), method=interpolation_method)
		interpolated_radius = griddata(points, radius_values, (log_mass_target, log_age_target), method=interpolation_method)

		return interpolated_teff, interpolated_logg, interpolated_logL, interpolated_radius, minit

	
if __name__=='__main__':
	# Example usage
	directory = 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'  # Replace with the path to your directory
	target_mass = 10.0  # Replace with the target mass in solar masses
	example_plot(directory, target_mass)

