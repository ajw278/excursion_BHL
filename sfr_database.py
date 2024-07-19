import numpy as np
from consts_defaults import *
import matplotlib.pyplot as plt
from turbfuncs import *
import pickle
import copy
import shutil
import disc_evol as de

from matplotlib.collections import LineCollection
from lifelines import KaplanMeierFitter

from scipy.stats import gaussian_kde
import pandas as pd
from lifelines.utils import coalesce


plt.rc('text', usetex=True)

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#e41a1c','#984ea3', '#dede00', '#999999','cyan', 'teal']

"""
class sfr_database
Sample and save database of star forming regions
Compute the accretion histories and disc properties of star-disc systems and save them
This also includes plotting routines
"""
class sfr_database():
		def __init__(self, fname=None, tag=''):
			if fname is None:
				fname = 'sfr_database'
			if tag!='': 
				fname+='_'+tag
			self.fname=fname
			self.tag = tag
			if not self.load():
				self.nstars = 0
				self.nregions = 0
				self.region_list = []
				self.region_nsts = []
				self.region_msts = []

				self.save()

		def load(self):
			try:
				with open(self.fname + '.pkl', 'rb') as f:
					data = pickle.load(f)
					self.__dict__.update(data)
				return True
			except FileNotFoundError:
				return False

		def save(self):
			temp_fname = self.fname + '_new.pkl'
    
			# Serialize the object to a temporary pickle file
			with open(temp_fname, 'wb') as f:
				pickle.dump(self.__dict__, f)

			# If serialization is successful, move the temporary file to the original file name
			shutil.move(temp_fname, self.fname + '.pkl')
		
		def add_region(self, sfr):
			region_dict = {'t_forms': sfr.t_forms, 'msts': sfr.msts, 't_evols': sfr.t_evols, 'Mdot_accs': sfr.Mdot_accs,\
				'dv_local': sfr.dv_local, 'rho_local': sfr.rho_local,'R_accs': sfr.R_accs, 'Mcl': sfr.Mcl,\
				'vcl': sfr.vcl, 'Mstcl': sfr.Mstcl, 'tcl': sfr.tcl}
			
			self.region_list.append(region_dict)
			self.region_nsts.append(len(sfr.msts))
			self.region_msts.append(sfr.Mstcl[-1])
			self.nstars += len(sfr.msts)
			self.nregions += 1
			self.save()

		def region_stmasses(self):
			for ireg, mreg in enumerate(self.region_msts):
				print('Region %d : %.2lf Msol'%(ireg, mreg/Msol2g))
			

		def plot_region_mass(self, iregion=None):
			if iregion is None:
				iregion = np.random.choice(np.arange(len(self.region_list)))
			
			fig, ax = plt.subplots(figsize=(6., 4.))
			Mcl = self.region_list[iregion]['Mcl']
			tcl = self.region_list[iregion]['tcl']
			Mst = self.region_list[iregion]['Mstcl']

			plt.plot(tcl/Myr2s, Mcl/Msol2g, linewidth=1, label='Gas')
			plt.plot(tcl/Myr2s, Mst/Msol2g, linewidth=1, label='Stars')
			plt.yscale('log')
			plt.ylabel('Mass of region [$M_\odot$]')
			plt.xlabel('Time [Myr]')
			plt.legend(loc='best')
			ax.tick_params(which='both', top=True, bottom=True, left=True, right=True)
			plt.show()

			return iregion
		
		def draw_representative_sample(self, Nsample, mreglim=250.0):

			if 10*Nsample>len(self.region_list):
				print('Warning: the region list is not large enough for the requested sample size.')
			ntot = np.sum(self.region_nsts)
			if Nsample>ntot:
				raise Warning('Tried to select more stars than available.')

			mweight = np.array(self.region_msts, dtype=float)
			mweight/=Msol2g
			mweight[mweight>mreglim] = 0.0
			mweight /= np.sum(mweight)
			
			mreg = []
			nsel = 0
			prob_norm = float(Nsample)/float(ntot)

			self.istar_ss = []
			#self.region_ss = np.random.choice(np.arange(self.nregions), size=Nsample, replace=False, p=mweight ) 
			while nsel<Nsample:
				ireg = np.random.choice(np.arange(len(self.region_list)), p=mweight)
				ia = np.random.choice(np.arange(self.region_nsts[ireg]))
				ia_pair = np.array([ireg, ia])
				ids = map(id, self.istar_ss)
				if not id(ia_pair) in ids:
					self.istar_ss.append(ia_pair)
					nsel+=1
					mreg.append(self.region_msts[ireg])
			
			return np.array(self.istar_ss)


		def plot_mstar_hist(self):

			fig, ax= plt.subplots(figsize=(6.,4.))
			bins = np.logspace(0.0,3.0, 20)

			plt.hist(np.array(self.region_msts)/Msol2g, bins=bins, edgecolor='k', histtype='step')
			plt.xlabel('Stellar mass of region: $M_{*,\mathrm{cl}}$ [$M_\odot$]')
			plt.ylabel('Number of included regions')
			plt.xscale('log')
			plt.yscale('log')
			print(np.amax(np.array(self.region_msts)/Msol2g))
			plt.xlim([1.0, 1000.0])
			ax.tick_params(which='both', top=True, left=True, bottom=True, right=True)
			plt.savefig('stregion_mdist.pdf')
			plt.show()


		def plot_discfrac(self, mlim=mllim, mstlim=0.1, tag='', label='', rmlim=None):
			if not rmlim is None:
				
				iinc = np.array(getattr(self, 'mstevol'+tag))/Msol2g>mstlim
				

				fig, ax = plt.subplots(figsize=(6.,4.))
			
				for im, mlim in enumerate(rmlim):
					md = getattr(self, 'mdiscevol'+tag)[0][iinc]/Msol2g
					tde = np.array(getattr(self, 'tdiscevol'+tag))/Myr2s
					iab = md>mlim

					dfrac = np.sum(iab, axis=0)/float(len(md))
					plt.plot(tde, dfrac, color=CB_color_cycle[im], linewidth=1, label='$> %d \\times 10^{-5} \, M_\odot$'%(mlim/1e-5))

				plt.ylabel('Disc fraction ($M_\mathrm{disc} > %d \\times 10^{-5} \, M_\odot$) for $m_* > %.1lf \, M_\odot$'%(mlim/1e-5, mstlim))
				plt.xlabel('Age [Myr]')
				tsp = np.linspace(0., 8.)
				plt.plot(tsp, np.exp(-tsp/3.), color='b',linewidth=1, label='$\\tau_\mathrm{disc} = 3$ Myr')
				plt.plot(tsp, np.exp(-tsp/5.), color='r', linewidth=1, label='$\\tau_\mathrm{disc} = 5$ Myr')


				plt.ylim([0.,1.])
				plt.xlim([0., 8.])
				plt.legend(loc='best')
				ax.tick_params(which='both', top=True, right=True, bottom=True, left=True)
				plt.savefig('disc_fraction'+tag+label+'.pdf', bbox_inches='tight', format='pdf')
				plt.show()
			
			else:
				iinc = np.array(getattr(self, 'mstevol'+tag))/Msol2g>mstlim
				md = getattr(self, 'mdiscevol'+tag)[0][iinc]/Msol2g
				tde = np.array(getattr(self, 'tdiscevol'+tag))/Myr2s
				iab = md>mlim

				dfrac = np.sum(iab, axis=0)/float(len(md))

				fig, ax = plt.subplots(figsize=(6.,4.))
				plt.plot(tde, dfrac, color='k', linewidth=1)

				plt.ylabel('Disc fraction ($M_\mathrm{disc} > %d \\times 10^{-5} \, M_\odot$) for $m_* > %.1lf \, M_\odot$'%(mlim/1e-5, mstlim))
				plt.xlabel('Age [Myr]')
				tsp = np.linspace(0., 8.)
				plt.plot(tsp, np.exp(-tsp/3.), color='b',linewidth=1, label='$\\tau_\mathrm{disc} = 3$ Myr')
				plt.plot(tsp, np.exp(-tsp/5.), color='r', linewidth=1, label='$\\tau_\mathrm{disc} = 5$ Myr')


				plt.ylim([0.,1.])
				plt.xlim([0., 8.])
				plt.legend(loc='best')
				ax.tick_params(which='both', top=True, right=True, bottom=True, left=True)
				plt.savefig('disc_fraction'+tag+label+'.pdf', bbox_inches='tight', format='pdf')
				plt.show()
		
		def plot_discfrac_msplit(self, mlim=mllim, mstlim=0.1, tag='', mbins = [0.1, 0.3, 1.0, np.inf]):
			mst_all =np.array(getattr(self, 'mstevol'+tag))
			iinc = mst_all/Msol2g>mstlim
			mst_inc = mst_all[iinc]
			md = getattr(self, 'mdiscevol'+tag)[0][iinc]/Msol2g
			tde = np.array(getattr(self, 'tdiscevol'+tag))/Myr2s
			iab = md>mlim

			dfrac = np.sum(iab, axis=0)/float(len(md))

			fig, ax = plt.subplots(figsize=(6.,4.))
			plt.plot(tde, dfrac, color='k', linewidth=1, label='All discs')

			plt.ylabel('Disc fraction ($M_\mathrm{disc} > %d \\times 10^{-5} \, M_\odot$)'%(mllim/1e-5))
			
			for im in range(len(mbins)-1):
				ibin_ = (mst_inc/Msol2g<mbins[im+1])&(mst_inc/Msol2g>=mbins[im])
				ibin_ab= iab&ibin_[:,np.newaxis]
				dfrac_ = np.sum(ibin_ab, axis=0)/np.sum(ibin_, axis=0)
				if im<len(mbins)-2:
					plt.plot(tde, dfrac_, color=CB_color_cycle[im], linewidth=1, label= '$%.1lf \, M_\odot >m_* > %.1lf \, M_\odot$'%(mbins[im+1],mbins[im]))
				else:
					plt.plot(tde, dfrac_, color=CB_color_cycle[im], linewidth=1, label= '$m_* > %.1lf \, M_\odot$'%(mbins[im]))
			
			plt.xlabel('Age [Myr]')
			tsp = np.linspace(0., 8.)
			plt.plot(tsp, np.exp(-tsp/3.), color='b',linewidth=1, label='$\\tau_\mathrm{disc} = 3$ Myr')
			plt.plot(tsp, np.exp(-tsp/5.), color='r', linewidth=1, label='$\\tau_\mathrm{disc} = 5$ Myr')


			plt.ylim([0.,1.])
			plt.xlim([0., 8.])
			plt.legend(loc='best')
			ax.tick_params(which='both', top=True, right=True, bottom=True, left=True)
			plt.savefig('disc_fraction_msplit'+tag+'.pdf', bbox_inches='tight', format='pdf')
			plt.show()
			
		def plot_rplf(self, mlim=mllim,tplot = [1.0, 2., 3.0, 8.0], mstllim=0.1,mstulim=np.inf, idt=0, tag=''):
			m_star = np.array(getattr(self, 'mstevol'+tag))/Msol2g
			time = np.array(getattr(self, 'tdiscevol'+tag))/Myr2s

			iinc = (m_star>mstllim)&(m_star<mstulim)
			print(np.sum(iinc), len(iinc))
			mdisc = getattr(self, 'mdiscevol'+tag)[:, iinc,:]/Msol2g
			fracmdisc = getattr(self, 'rpfevol'+tag)[:, iinc, :]
			m_star = m_star[iinc]

			isurv = mdisc>mlim


			
			
			cmap = plt.cm.viridis  # Choose a colormap
			#normalize = plt.Normalize(vmin=np.min(np.log10(0.01)), vmax=np.max(np.log10(3.0))) 
			fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7, 4), gridspec_kw={'width_ratios': [5., 4.0]}, sharey='row')
			
			normalize = plt.Normalize(-6., -2.)
			iinc = np.random.choice(np.arange(len(m_star)), size=min(20, len(m_star)), replace=False)
			iall = np.arange(len(m_star))

			for im , m_ in enumerate(m_star):
				fm =fracmdisc[idt, im]
				fm[fm>1.] =1.
				fm[fm<0.] = 0.0
				if im in iinc:
					#axs[0].plot(time, fracmdisc[idt, im],   color=cmap(normalize(np.log10(mdisc[idt, im, :]/m_))), linewidth=1, alpha=0.2)
					
					points = np.array([time, fm]).T.reshape(-1, 1, 2)
					segments = np.concatenate([points[:-1], points[1:]], axis=1)

					lc = LineCollection(segments, cmap='viridis', norm=normalize)
					# Set the values used for colormapping
					lc.set_array(np.log10(mdisc[idt, im, :]/m_))
					lc.set_linewidth(1.0)
					line = axs[0].add_collection(lc)
				else:
					points = np.array([time, fm]).T.reshape(-1, 1, 2)

					
					segments = np.concatenate([points[:-1], points[1:]], axis=1)

					lc = LineCollection(segments, cmap='viridis', norm=normalize, alpha=0.1)
					# Set the values used for colormapping
					lc.set_array(np.log10(mdisc[idt, im, :]/m_))
					lc.set_linewidth(0.3)
					line = axs[0].add_collection(lc)
						
			db = 0.025
			bins = np.arange(-0.1, 1.1+db, db)
			for it, t in enumerate(tplot):
				itime = np.argmin(np.absolute(time - t))
				
				isurv_ = np.where(copy.copy(isurv)[idt].T[itime])[0]
				fmd_end = fracmdisc[idt, isurv_, itime]
				iab = fmd_end>=0.99999
				fmd_end[iab] = 0.99999
				axs[1].hist(fmd_end.flatten(), bins=bins, orientation='horizontal', density=True,cumulative=True, histtype='step', color='gray', edgecolor=CB_color_cycle[it], label='$%d$ Myr'%t)
			sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
			sm.set_array([])
			# Create a colorbar

			# Remove white space between plots
			plt.subplots_adjust(wspace=0.1, hspace=0, top=0.95)
			cbar = plt.colorbar(sm,ax=axs.ravel().tolist(),  label='log. Disc-star mass ratio: $\log M_\mathrm{disc}/m_*$', orientation='horizontal', location='top') #,fraction=0.05, pad=0.00, anchor=(0.5, 1.0))
			
			
			#ax.set_ylim([1e-2, 1.0])
			axs[0].set_xscale('log')

			axs[0].set_xlim([0.5, int(time[-1]+0.5)])
			
			axs[0].set_ylim([-0.05, 1.05])

			axs[1].set_xticks([0., 0.25, 0.5, 0.75, 1.0])
			axs[1].set_yticks(np.arange(0.,1.1, 0.1))
			axs[1].grid()
			axs[0].set_ylabel('Half-life replenishment fraction: $f_{M, 1/2}$')
			axs[0].set_xlabel('Time: $t$ [Myr]')
			axs[1].set_xlabel('Cum. frac.')
			axs[0].tick_params(which='both', right=True, left=True, top=True, bottom=True)
			axs[1].tick_params(which='both', right=True, left=True, top=True, bottom=True)
			axs[1].legend(loc=2, fontsize=8)
			plt.savefig('hlrf'+tag+'.png', bbox_inches='tight', format='png', dpi=500)
			plt.show()

			fig, ax = plt.subplots(figsize=(4.,4.))

			for im , m_ in enumerate(m_star):
			
				if im in iinc:
					plt.scatter(mdisc[idt, im, ::10], fracmdisc[idt, im,::10], color='k', s=1)
					plt.scatter(-mdisc[idt, im, ::10], -fracmdisc[idt, im,::10], color='r', s=1)
					plt.scatter(-mdisc[idt, im,::10], fracmdisc[idt, im,::10], color='b', s=1)
			plt.yscale('log')
			plt.xscale('log')
			plt.savefig('test.pdf', format='pdf')
			plt.show()
			
		
		def fetch_obs(self, region, ax=None, plot='md', color='k', s=2, shape='o', label=False):
			# Assuming you have loaded the data from a CSV file as you have shown in the initial part of your code
			df = pd.read_csv('PP7-Surveys_2022-10-19_PPVII_website.tsv', sep='\t')

			# Extracting data for plotting
			mdot = 10. ** np.asarray(df['logMacc_PPVII'][df['Region'] == region])
			mst = np.asarray(df['Mstar_PPVII'][df['Region'] == region])

			md =  np.asarray(df['Standardized_Mdust_Mearth'][df['Region']==region], dtype=str)
			md[md=='--'] = 'nan'
			
			notes_mdot  = np.array(df['notes_Macc_PPVII'][df['Region']==region], dtype='str')
			
			iupper_mdot = notes_mdot=='<'
			imeas_mdot = ~iupper_mdot
			rd = np.asarray(df['H20_R68_au_DR3'][df['Region']==region], dtype=str)
			rd[rd=='#VALUE!'] = 'nan'
			rd = np.array(rd, dtype=float)
			
			
			
			
			iupper = np.flatnonzero(np.core.defchararray.find(md,'<')!=-1)
			imeas = np.flatnonzero(np.core.defchararray.find(md,'<')==-1)
			
			md = np.char.strip(md, chars='<')
			for i in iupper:
				md[i] = np.char.strip(md[i], chars='<')
			md = np.asarray(md, dtype=float)*Mearth2Msol*1e2
			
			
			if not ax is None:
				if plot=='md':
					if label:
						ax.scatter(mst[imeas], md[imeas], color=color, marker=shape, s=s, label=region)
					else:
						ax.scatter(mst[imeas], md[imeas], color=color, marker=shape, s=s)
					ax.scatter(mst[iupper], md[iupper], color=color, marker='v', s=s)
				elif plot=='mdot':
					if label:
						ax.scatter(mst[imeas_mdot], mdot[imeas_mdot], color=color, marker=shape, s=s, label=region)
					else:
						ax.scatter(mst[imeas_mdot], mdot[imeas_mdot], color=color, marker=shape, s=s)
					ax.scatter(mst[iupper_mdot], mdot[iupper_mdot], color=color, marker='v', s=s)
				elif plot=='rd':
					if label:
						ax.scatter(mst, rd, color=color, marker=shape, s=s, label=region)
					else:
						ax.scatter(mst, rd, color=color, marker=shape, s=s)

			
			return mst, mdot, md, imeas, iupper, rd

		def create_dataframe_slicer(self, iloc, loc, timeline):
			if (loc is not None) and (iloc is not None):
				raise ValueError("Cannot set both loc and iloc in call to .plot().")
			user_did_not_specify_certain_indexes = (iloc is None) and (loc is None)
			user_submitted_slice = (
			slice(timeline.min(), timeline.max())
			if user_did_not_specify_certain_indexes
			else coalesce(loc, iloc)
			)

			get_method = "iloc" if iloc is not None else "loc"
			return lambda df: getattr(df, get_method)[user_submitted_slice]		

		def plot_mcdfs(self, tplot =[0.1, 0.3, 1.0, 3.0, 9.0], msrange = [0.5, 1.0], idt=0, mlim=mllim, tag=''):
			m_star = np.array(getattr(self, 'mstevol'+tag)) / Msol2g
			time = np.array( getattr(self,'tdiscevol'+tag)) / Myr2s
			
			mdisc = getattr(self, 'mdiscevol'+tag)[idt] / Msol2g

			imass = (m_star>msrange[0])&(m_star<msrange[1])
			mdsp = np.logspace(np.log10(mlim),np.log10(np.amax(mdisc)), 1000 )

			# Find the closest time snapshot for each tplot
			closest_times = [np.argmin(np.abs(time - t)) for t in tplot]

			regions = ['Taurus', 'USco']
			label = {'Taurus':'Taurus', 'Lupus':'Lupus', 'ChamI': 'Chameleon I', 'USco':'Upper Sco'}
			
			cmap = plt.cm.get_cmap('viridis')  # Define colormap
			norm = plt.Normalize(min(np.log10(tplot)), max(np.log10(tplot)))  # Normalize colormap based on tplot range
			scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

			colors = ['pink', 'k']
			plt.figure(figsize=(6, 5))
			for i in range(len(regions)):


				color=  colors[i]

				region_age = regions[i] 
				ms_obs, mdot_obs, md_obs, imeas_, iupper_, rd_ = self.fetch_obs(region_age,ax=None, plot=False)
				
				imrange = (ms_obs>=msrange[0])&(ms_obs<=msrange[1])&~np.isnan(md_obs)
				ms_obs = ms_obs[imrange]
				md_obs = md_obs[imrange]
				imeas_bool = np.zeros(len(mdot_obs), dtype=bool)
				imeas_bool[imeas_] = True 
				imeas_bool = imeas_bool[imrange]

				# Kaplan-Meier estimator for observed data
				kmf = KaplanMeierFitter()
				kmf.fit(md_obs, event_observed=imeas_bool)
				obs_cdf = kmf.survival_function_at_times(mdsp).values
				obs_lower, obs_upper = kmf.confidence_interval_.values.T
				#kmf.plot()
				#plt.show()
				plt.plot(mdsp, obs_cdf, color=color, linestyle='dashed', linewidth=1, label=label[region_age])
				mdobs_sort = np.sort(md_obs)
				xv0 = 0.0
				plt_x = []
				plt_yl = []
				plt_yu = []
				for iint in range(len(md_obs)+1):
					if iint ==len(md_obs):
						xv = xv0
					else:
						xv = mdobs_sort[iint]
					if iint == len(obs_lower):
						yvl0 = obs_lower[iint-1]
						yvu0 = obs_upper[iint-1]
					else:
						yvl0 = obs_lower[iint]
						yvu0 = obs_upper[iint]
					plt_x.append(xv0)
					plt_x.append(xv)
					plt_yl.append(yvl0)
					plt_yl.append(yvl0)
					plt_yu.append(yvu0)
					plt_yu.append(yvu0)
					xv0 = xv

				dataframe_slicer = self.create_dataframe_slicer(None, None, kmf.timeline)
				x = dataframe_slicer(kmf.confidence_interval_).index.values.astype(float)
				lower = dataframe_slicer(kmf.confidence_interval_.iloc[:, [0]]).values[:, 0]
				upper = dataframe_slicer(kmf.confidence_interval_.iloc[:, [1]]).values[:, 0]
				#plt.fill_between(plt_x, plt_yl, plt_yu, color=color, alpha=0.2, steps='steps-')
				plt.fill_between(x, lower, upper, color=color, alpha=0.2, step='post')
			
			for i, itime in enumerate(closest_times):

				t = tplot[i]

				iab_dmass = mdisc[:, itime] > mlim
				iss  =iab_dmass&imass
				md_mod = mdisc[iss, itime]

				# Calculate CDFs
				#obs_cdf = np.array([np.sum(md_obs > m) / len(md_obs) for m in mdsp])
				mod_cdf = np.array([np.sum(md_mod > m) / len(md_mod) for m in mdsp])
				color = scalar_map.to_rgba(np.log10(t))  # Set color based on time
				plt.plot(mdsp, mod_cdf, color=color, linestyle='solid', linewidth=1, label=f'Model: {t} Myr')


			plt.xscale('log')
			plt.xlabel('Mass: $M$ [$M_\odot$]')
			plt.ylabel('Fraction discs w/ mass $M_\mathrm{disc}>M$ ($%.1lf \, M_\odot <m_*<%.1lf M_\odot$ )'%(msrange[0], msrange[1]))
			plt.ylim([0, 1])
			plt.xlim([mlim, 0.1])
			plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)

			#cbar = plt.colorbar(scalar_map)
			#cbar.set_label('log Age of stars: $\log t$ [Myr]')
    
			plt.legend(loc='best', fontsize=8)
			plt.savefig('mdisc_cdf'+tag+'.pdf', bbox_inches='tight')
			plt.show()

			return None

		def plot_all(self, tplot=[0.5, 1.0, 2., 3.0, 5.0], idt=0, mlim=mllim, nlevs=20, tag=''):
			m_star = np.array(getattr(self, 'mstevol'+tag)) / Msol2g
			time = np.array( getattr(self,'tdiscevol'+tag)) / Myr2s
			
			mdisc = getattr(self, 'mdiscevol'+tag)[idt] / Msol2g
			mdotst = getattr(self, 'mdotevol'+tag)[idt] * year2s / Msol2g
			rdisc = getattr(self, 'rdiscevol'+tag)[idt] / au2cm
			
			levels = np.arange(0.0, 0.8, 0.05)

			# Find the closest time snapshot for each tplot
			closest_times = [np.argmin(np.abs(time - t)) for t in tplot]

			# Set up the figure and axes
			fig, axes = plt.subplots(3, len(tplot), sharex=True, sharey='row', figsize=(12, 10))

			msp = np.logspace(-2., 1.)
			log_xmin = -2
			log_xmax = 0.5
			
			bins = np.linspace(log_xmin-0.5, log_xmax+0.5, 15)
			log_ymin_dmass = -4.8
			log_ymax_dmass = -0.5

			regions = ['rOph',  'Taurus', 'Lupus', 'ChamI', 'USco']
			label = {'rOph': '$\\rho$ Oph', 'Taurus':'Taurus', 'Lupus':'Lupus', 'ChamI': 'Cham I', 'USco':'USco'}
			# Loop over each time snapshot in tplot
			for i, itime in enumerate(closest_times):
				# Disc Mass Plot
				
				region_age = regions[i] 
				
				
				ms_, mdot_, md_, imead_, iupper_, rd_ = self.fetch_obs(region_age,ax=None, plot='md', color='r', s=5, shape='o')
				
				idet = md_>1e-20
				ms_ = ms_[idet]
				md_ = md_[idet]
				pdf, be = np.histogram(np.log10(ms_), bins=bins, density=True)
				pdf = np.array(pdf, dtype='float')
				pdf /= np.sum(pdf*np.diff(be))
				bc = (be[1:]+be[:-1])/2.
				
				
				xx_dmass, yy_dmass = np.mgrid[log_xmin:log_xmax:100j, log_ymin_dmass:log_ymax_dmass:100j]
				positions_dmass = np.vstack([xx_dmass.ravel(), yy_dmass.ravel()])
				iab_dmass = mdisc[:, itime] > mlim
				log_m_star_dmass = np.log10(m_star[iab_dmass])
				log_mdisc_dmass = np.log10(mdisc[iab_dmass, itime])


				pdf_model, be = np.histogram(log_m_star_dmass, bins=bins, density=True)
				pdf_model = np.array(pdf_model, dtype='float')
				pdf_model /= np.sum(pdf_model*np.diff(be))
				weight_arr = pdf/(pdf_model+0.1)
				pdf_func = interpolate.interp1d(bc, weight_arr)

				weights = pdf_func(log_m_star_dmass)
				kde_dmass = gaussian_kde(np.vstack([log_m_star_dmass, log_mdisc_dmass]), weights=weights)
				zz_dmass = np.reshape(kde_dmass(positions_dmass).T, xx_dmass.shape)
				cf_dmass = axes[0, i].contourf(10. ** xx_dmass, 10. ** yy_dmass, zz_dmass, cmap='viridis', levels=nlevs)
				axes[0, i].set_title(f'{label[region_age]} vs. {time[itime]:.1f} Myr')
				if i==0:
					axes[0, i].set_ylabel('Disc mass: $M_\mathrm{disc}$ [$M_\odot$]')
				axes[0, i].set_yscale('log')
				axes[0, i].tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
				axes[0, i].tick_params(axis='y', which='both', left=True, right=True, labelleft=i == 0)
				axes[0, i].set_xlim([10. ** log_xmin, 10. ** log_xmax])
				axes[0, i].set_ylim([10. ** log_ymin_dmass, 10. ** log_ymax_dmass])

				# Stellar Accretion Rate Plot
				
				ms_, mdot_, md_, imead_, iupper_, rd_ = self.fetch_obs(region_age, ax=axes[0,i], plot='md', color='r', s=5, shape='o')
				
				idet = mdot_>1e-20
				ms_ = ms_[idet]
				mdot_ = mdot_[idet]
				pdf, be = np.histogram(np.log10(ms_), bins=bins, density=True)
				pdf = np.array(pdf, dtype='float')
				pdf /= np.sum(pdf*np.diff(be))
				bc = (be[1:]+be[:-1])/2.
				
				
				log_ymin_mdotst = -12.5
				log_ymax_mdotst = -6.0
				xx_mdotst, yy_mdotst = np.mgrid[log_xmin:log_xmax:100j, log_ymin_mdotst:log_ymax_mdotst:100j]
				positions_mdotst = np.vstack([xx_mdotst.ravel(), yy_mdotst.ravel()])
				iab_mdotst = mdisc[:, itime] > mlim
				log_m_star_mdotst = np.log10(m_star[iab_mdotst])
				log_mdot_mdotst = np.log10(mdotst[iab_mdotst, itime])
				pdf_model, be = np.histogram(log_m_star_mdotst, bins=bins, density=True)
				pdf_model = np.array(pdf_model, dtype='float')
				pdf_model /= np.sum(pdf_model*np.diff(be))
				weight_arr = pdf/(pdf_model+0.1)
				pdf_func = interpolate.interp1d(bc, weight_arr)
				weights = pdf_func(log_m_star_mdotst)


				kde_mdotst = gaussian_kde(np.vstack([log_m_star_mdotst, log_mdot_mdotst]), weights=weights)
				zz_mdotst = np.reshape(kde_mdotst(positions_mdotst).T, xx_mdotst.shape)
				cf_mdotst = axes[1, i].contourf(10. ** xx_mdotst, 10. ** yy_mdotst, zz_mdotst, cmap='viridis', levels=nlevs)
				if i==0:
					axes[1, i].set_ylabel('Stellar accretion rate: $\dot{M}_\mathrm{acc}$ [$M_\odot$]')
				axes[1, i].set_yscale('log')

				axes[1, i].tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
				axes[1, i].tick_params(axis='y', which='both', left=True, right=True, labelleft=i == 0)
				axes[1, i].set_xlim([10. ** log_xmin, 10. ** log_xmax])

				# Outer Disc Radius Plot
				
				ms_, mdot_, md_, imead_, iupper_, rd_ = self.fetch_obs(region_age, ax=axes[1,i], plot='mdot', color='r', s=5, shape='o')
				
				idet = rd_>1e-20
				ms_ = ms_[idet]
				rd_ = rd_[idet]
				pdf, be = np.histogram(np.log10(ms_), bins=bins, density=True)
				pdf = np.array(pdf, dtype='float')
				pdf /= np.sum(pdf*np.diff(be))
				bc = (be[1:]+be[:-1])/2.
				pdf_func = interpolate.interp1d(bc, pdf)
				
				
				log_ymin_rdisc = 0.0
				log_ymax_rdisc = 3.5
				xx_rdisc, yy_rdisc = np.mgrid[log_xmin:log_xmax:100j, log_ymin_rdisc:log_ymax_rdisc:100j]
				positions_rdisc = np.vstack([xx_rdisc.ravel(), yy_rdisc.ravel()])
				iab_rdisc = (mdisc[:, itime] > mlim)&(rdisc[:, itime]>5.0)
				log_m_star_rdisc = np.log10(m_star[iab_rdisc])
				log_rd_rdisc = np.log10(rdisc[iab_rdisc, itime])

				pdf_model, be = np.histogram(log_m_star_rdisc, bins=bins, density=True)
				pdf_model = np.array(pdf_model, dtype='float')
				pdf_model /= np.sum(pdf_model*np.diff(be))
				weight_arr = pdf/(pdf_model+0.1)
				pdf_func = interpolate.interp1d(bc, weight_arr)
				weights = pdf_func(log_m_star_rdisc)
				
				kde_rdisc = gaussian_kde(np.vstack([log_m_star_rdisc, log_rd_rdisc])) #, weights=weights)
				zz_rdisc = np.reshape(kde_rdisc(positions_rdisc).T, xx_rdisc.shape)
				cf_rdisc = axes[2, i].contourf(10. ** xx_rdisc, 10. ** yy_rdisc, zz_rdisc, cmap='viridis', levels=nlevs)
				
				if i==0:
					axes[2, i].set_ylabel('Accretion radius: $R_\mathrm{acc}$ [au]')
				axes[2, i].set_xlabel('Star mass: $m_*$ [$M_\odot$]')
				axes[2, i].set_yscale('log')
				axes[2, i].set_xscale('log')
				axes[2, i].plot(msp, 250.*msp**0.9, color='r', linewidth=1, linestyle='dashed', label='$R_\mathrm{CO} = 250 (m_*/1\,M_\odot)^{0.9}$ au')
				axes[2, i].tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
				axes[2, i].tick_params(axis='y', which='both', left=True, right=True, labelleft=i == 0)
				axes[2, i].set_xlim([10. ** log_xmin, 10. ** log_xmax])


				if i==0:
					#axes[0, i].legend(loc='best', fontsize=8)
					#axes[1, i].legend(loc='best', fontsize=8)
					axes[2, i].legend(loc=4, fontsize=7)
					

			# Add color bar to the last column
			"""fig.subplots_adjust(right=0.92)
			cbar_ax = fig.add_axes([0.95, 0.11, 0.02, 0.77])"""
			plt.subplots_adjust(wspace=0, hspace=0)
			#fig.colorbar(cf_rdisc, cax=cbar_ax, label='Model KDE')
			plt.savefig('discprops'+tag+'.pdf', bbox_inches='tight', format='pdf')
			plt.show()

			return None



		def calc_discevol(self, redraw=True, Nsample=1, maxtres=20000, minit=0.0, minitdisp=0.0, mlim=250., ptag='', wind=False, eps_wind=0.1):

			tag = ptag

			if not hasattr(self, 'tags'):
				self.tags  = [] 
			if minit>0.0:
				print(self.tags)
				tag += '_minit_%.2lf'%minit
				if minitdisp>0.0:
					tag += '_mdisp_%.2lf'%minitdisp
				print(tag)
			if wind:
				tag += '_wind'
			
			if not tag in self.tags:
				self.tags.append(tag)
			
			if not hasattr(self, 'idiscevol'+tag) or redraw:
				setattr(self, 'idiscevol'+tag, self.draw_representative_sample(Nsample, mreglim=mlim))

			setattr(self, 'wind'+tag, wind)
			setattr(self, 'eps_wind'+tag, eps_wind)
			
			if not hasattr(self, 'mdiscevol'+tag) or redraw:
				print( hasattr(self, 'mdiscevol'+tag), redraw)
				ta = []
				Mda = []
				Ra = []
				mst = []
				dva = []
				drhoa= []
				

				tmax_min = np.inf
				
				ide = getattr(self, 'idiscevol'+tag)
				for id in ide:
					ta.append(self.region_list[id[0]]['t_evols'][id[1]])
					Ra.append(self.region_list[id[0]]['R_accs'][id[1]])
					Mda.append(self.region_list[id[0]]['Mdot_accs'][id[1]])
					mst.append(self.region_list[id[0]]['msts'][id[1]])
					dva.append(self.region_list[id[0]]['dv_local'][id[1]]/1e5)
					drhoa.append(self.region_list[id[0]]['rho_local'][id[1]])

					if np.amax(self.region_list[id[0]]['t_evols'][id[1]])<tmax_min:
						teval_ = self.region_list[id[0]]['t_evols'][id[1]]
						tmax_min = teval_[-1]
					if tmax_min<8.*Myr2s:
						print(tmax_min/Myr2s)
						exit()
				while len(teval_)>maxtres:
					print('Down sampling teval:', len(teval_))
					teval_ = teval_[::2]

				dt_ = ['ln']

				setattr(self, 'tdiscevol'+tag, teval_)
				
				print('Running disc evolution calculation...')
				wind = getattr(self, 'wind'+tag)
				eps_wind = getattr(self, 'eps_wind'+tag)
				
				
				disc_mass, mdot_star, frac_tend, disc_radius, disc_vt, mdot_BHL, rho_BHL, dv_BHL = de.mdot_tacc(Mda, Ra, teval_, ta, dt_, mst, drhoa, dva, plot=False, mu=-.5, fM=minit, fM_disp=minitdisp,sigma=1.0, wind=wind, eps_wind=eps_wind)
					
				
				setattr(self, 'mdiscevol'+tag, disc_mass)
				setattr(self, 'mdotevol'+tag, mdot_star)
				setattr(self, 'rpfevol'+tag, frac_tend)
				setattr(self, 'mstevol'+tag, mst)
				setattr(self, 'vtevol'+tag, disc_vt)
				setattr(self, 'rdiscevol'+tag, disc_radius)
				setattr(self, 'mdotBHLevol'+tag, mdot_BHL)
				setattr(self, 'rhoBHLevol'+tag, rho_BHL)
				setattr(self, 'dvBHLevol'+tag, dv_BHL)

				self.save()
			
			return tag
				
		def plot_Next(self, tplot=[0.1, 0.3, 1.0, 3.0, 9.0], idt=0, cs=cs_*1e5, mlim=mllim,tag=''):
			time = np.array( getattr(self,'tdiscevol'+tag)) / Myr2s
			mdisc = getattr(self, 'mdiscevol'+tag)[idt] / Msol2g
			

			ide = getattr(self, 'idiscevol'+tag)
			Next = np.zeros((len(ide), len(time)))

			for iplt, i_ in enumerate(ide):
				ist = i_[1]
				ireg = i_[0]
				mstar_ = self.region_list[ireg]['msts'][ist]/Msol2g 
				ttmp_ = self.region_list[ireg]['t_evols'][ist]/Myr2s
				Ratmp_ = self.region_list[ireg]['R_accs'][ist]/au2cm
				Mdtmp_ = self.region_list[ireg]['Mdot_accs'][ist]*year2s/Msol2g
				Nexttmp_ = 2e19 * (Mdtmp_/1e-9)*((Ratmp_/250.)**-0.5) * (mstar_**-0.5)
				Next[iplt] = np.interp(time, ttmp_, Nexttmp_)

			# Find the closest time snapshot for each tplot
			closest_times = [np.argmin(np.abs(time - t)) for t in tplot]

			# Set up the figure and axes
			fig, ax = plt.subplots(figsize=(6, 4))
			Mbins = np.logspace(14., 23., 300)

			cmap = plt.cm.get_cmap('viridis')  # Define colormap
			norm = plt.Normalize(min(np.log10(tplot)), max(np.log10(tplot)))  # Normalize colormap based on tplot range
			scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


			# Loop over each time snapshot in tplot
			for i, itime in enumerate(closest_times):
				# Disc Mass Plot
				iab_dmass = mdisc[:, itime] > mlim
				Next_ = Next[iab_dmass, itime]
				t = tplot[i]
				color = scalar_map.to_rgba(np.log10(tplot[i]))  # Set color based on time
				mod_cdf = np.array([np.sum(Next_ < a_) / (len(Next_)+1e-2) for a_ in Mbins])
				plt.step(Mbins, mod_cdf, color=color, linestyle='solid', linewidth=1, label=f'Model: {t} Myr')

			plt.ylim([0.,1.])
			plt.axvline(1e20, color='r', linewidth=1, linestyle='dashed', label='Approx. $^{12}$CO dissociation')
			plt.xscale('log')
			plt.xlim([Mbins[0], Mbins[-1]])
			plt.ylabel('Cumulative distribution function')
			#plt.xlabel('Max. viscous alpha: $\log \\alpha_{\mathrm{SS}}$')
			plt.xlabel('External structure column density: $N_\mathrm{ext}$ [cm$^{-2}$]')

			plt.legend(loc='best', fontsize=8)

			ax.tick_params(which='both', left=True, right=True, top=True, bottom=True)
			plt.savefig('Next_evol'+tag+'.pdf', bbox_inches='tight', format='pdf')
			plt.show()

		
		def plot_tauacc(self, tplot=[0.1, 0.3, 1.0, 3.0, 9.0], idt=0, cs=cs_*1e5, mlim=mllim,tag=''):
			m_star = np.array(getattr(self, 'mstevol'+tag)) / Msol2g
			time = np.array( getattr(self,'tdiscevol'+tag)) / Myr2s
			mdisc = getattr(self, 'mdiscevol'+tag)[idt] / Msol2g
			mdotst = getattr(self, 'mdotevol'+tag)[idt] * year2s / Msol2g
			rdisc = getattr(self, 'rdiscevol'+tag)[idt] / au2cm
			vturb = getattr(self, 'vtevol'+tag)[idt] / 1e5 

			taud = 1E-6*mdisc/(mdotst+1e-30)
			# Find the closest time snapshot for each tplot
			closest_times = [np.argmin(np.abs(time - t)) for t in tplot]

			# Set up the figure and axes
			fig, ax = plt.subplots(figsize=(6, 4))
			Mbins = np.logspace(-1.5, 0.5, 300)

			cmap = plt.cm.get_cmap('viridis')  # Define colormap
			norm = plt.Normalize(min(np.log10(tplot)), max(np.log10(tplot)))  # Normalize colormap based on tplot range
			scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


			# Loop over each time snapshot in tplot
			for i, itime in enumerate(closest_times):
				# Disc Mass Plot
				iab_dmass = mdisc[:, itime] > mlim
				ms_ = m_star[iab_dmass]
				taud_ = taud[iab_dmass, itime]
				t = tplot[i]
				color = scalar_map.to_rgba(np.log10(tplot[i]))  # Set color based on time
				mod_cdf = np.array([np.sum(taud_ < a_) / (len(taud_)+1e-2) for a_ in Mbins])
				plt.step(Mbins, mod_cdf, color=color, linestyle='solid', linewidth=1, label=f'Model: {t} Myr')

			plt.ylim([0.,1.])
			plt.xscale('log')
			plt.xlim([Mbins[0], Mbins[-1]])
			plt.ylabel('Cumulative distribution function')
			#plt.xlabel('Max. viscous alpha: $\log \\alpha_{\mathrm{SS}}$')
			plt.xlabel('Accretion time-scale: $\\tau_\mathrm{acc}$ [Myr]')

			plt.legend(loc='best')

			ax.tick_params(which='both', left=True, right=True, top=True, bottom=True)
			plt.savefig('taud_evol'+tag+'.pdf', bbox_inches='tight', format='pdf')
			plt.show()

			
			return None


		def plot_vturb(self, tplot=[0.1, 0.3, 1.0, 3.0, 9.0], idt=0, cs=cs_*1e5, mlim=mllim,tag=''):
			m_star = np.array(getattr(self, 'mstevol'+tag)) / Msol2g
			time = np.array( getattr(self,'tdiscevol'+tag)) / Myr2s
			mdisc = getattr(self, 'mdiscevol'+tag)[idt] / Msol2g
			mdotst = getattr(self, 'mdotevol'+tag)[idt] * year2s / Msol2g
			rdisc = getattr(self, 'rdiscevol'+tag)[idt] / au2cm
			vturb = getattr(self, 'vtevol'+tag)[idt] / 1e5 

			Mach = vturb*1e5/cs
			alpha = Mach**2

			# Find the closest time snapshot for each tplot
			closest_times = [np.argmin(np.abs(time - t)) for t in tplot]

			# Set up the figure and axes
			fig, ax = plt.subplots(figsize=(6, 4))
			Mbins = np.linspace(-2.8, 0.5, 300)

			cmap = plt.cm.get_cmap('viridis')  # Define colormap
			norm = plt.Normalize(min(np.log10(tplot)), max(np.log10(tplot)))  # Normalize colormap based on tplot range
			scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


			# Loop over each time snapshot in tplot
			for i, itime in enumerate(closest_times):
				# Disc Mass Plot
				iab_dmass = mdisc[:, itime] > mlim
				ms_ = m_star[iab_dmass]
				Mach_ = Mach[iab_dmass, itime]
				print(Mach_)
				t = tplot[i]
				color = scalar_map.to_rgba(np.log10(tplot[i]))  # Set color based on time
				mod_cdf = np.array([np.sum(np.log10(Mach_) < a_) / (len(Mach_)+1e-2) for a_ in Mbins])
				plt.step(Mbins*2., mod_cdf, color=color, linestyle='solid', linewidth=1, label=f'Model: {t} Myr')

			dxtxt = 0.05
			plt.axvline(np.log10(0.0625), color='brown', linewidth=0.8)
			plt.text(np.log10(0.0625)+dxtxt, 0.1, 'DM Tau', color='brown', rotation=90)

			plt.axvline(np.log10(0.5**2), color='purple', linewidth=0.8)
			plt.text(np.log10(0.5**2)+dxtxt, 0.1, 'IM Lup', color='purple', rotation=90)

			xTW = np.log10(6e-3)
			arrow_height = 0.5  # Height of the arrow in data coordinates
			arrow_length = 0.3

			plt.axvline(xTW, color='orange', linewidth=0.8)
			plt.annotate('', xy=(xTW+dxtxt-arrow_length, arrow_height), xytext=(xTW+dxtxt, arrow_height),
			arrowprops=dict(arrowstyle='->', color='orange', lw=1), color='orange')
			plt.text(xTW+dxtxt, 0.1, 'TW Hya', color='orange', rotation=90)

			xHD163296 = np.log10(2.5e-3)
			plt.axvline(xHD163296, color='gray', linewidth=0.8)
			plt.annotate('', xy=(xHD163296+dxtxt-arrow_length, arrow_height), xytext=(xHD163296+dxtxt, arrow_height),
			arrowprops=dict(arrowstyle='->', color='gray', lw=1), color='gray')
			plt.text(xHD163296+dxtxt, 0.1, 'HD 163296', color='gray', rotation=90)


			xMWC480 = np.log10(1e-2)
			plt.axvline(xMWC480, color='red', linewidth=0.8)
			plt.annotate('', xy=(xMWC480+dxtxt-arrow_length, arrow_height), xytext=(xMWC480+dxtxt, arrow_height),
			arrowprops=dict(arrowstyle='->', color='red', lw=1), color='red')
			plt.text(xMWC480+dxtxt, 0.1, 'MWC 480', color='red', rotation=90)



			xV4046 = np.log10(2e-2)
			plt.axvline(xV4046, color='cyan', linewidth=0.8)
			plt.annotate('', xy=(xV4046+dxtxt-arrow_length, arrow_height), xytext=(xV4046+dxtxt, arrow_height),
			arrowprops=dict(arrowstyle='->', color='cyan', lw=1), color='cyan')
			plt.text(xV4046+dxtxt, 0.1, 'V4046 Sgr', color='cyan', rotation=90)

			plt.ylim([0.,1.])
			plt.xlim([2.*Mbins[0], 2.*Mbins[-1]])
			plt.ylabel('Cumulative distribution function')
			#plt.xlabel('Max. viscous alpha: $\log \\alpha_{\mathrm{SS}}$')
			plt.xlabel('Max. viscous alpha: $\log \\alpha_{\mathrm{SS}}$')

			plt.legend(loc='best')


			 # Add second x-axis
			ax2 = ax.twiny()
			ax2.set_xlabel('Max. turb. Mach number: $\log v_{\mathrm{t, disc}}/c_\mathrm{s}$')

			# Set tick locations and labels for the second x-axis
			tick_positions =  ax.get_xticks() # np.arange(Mbins[0], Mbins[-1], 1.0)
			ax2.set_xticks(tick_positions)
			ax2.set_xticklabels(['$%.1lf$'%(tick/2.) for tick in tick_positions])

			# Adjust position of the second x-axis to be on top
			ax2.xaxis.set_ticks_position('top')
			ax2.xaxis.set_label_position('top')

			ax.tick_params(which='both', left=True, right=True)
			plt.savefig('alpha_evol'+tag+'.pdf', bbox_inches='tight', format='pdf')
			plt.show()




			
			return None
			
		def plot_accretion_rates_wevap(self, Nsample=8, idt=0,  tag=''):
				
			nsamp_tot = len(getattr(self, 'mstevol'+tag))
			irands = np.random.choice(np.arange(nsamp_tot), size=Nsample)
			Mdplt = []
			vplt = []
			rhoplt = []
			mstar_plt = []
			tplt = np.array(getattr(self, 'tdiscevol'+tag))/Myr2s
			for irand in irands:
				mstar_plt.append(np.array(getattr(self, 'mstevol'+tag)[irand])/Msol2g)
				vplt.append(getattr(self, 'dvBHLevol'+tag)[idt][irand])
				Mdplt.append(getattr(self, 'mdotBHLevol'+tag)[idt][irand]*year2s/Msol2g)
				rhoplt.append(getattr(self, 'rhoBHLevol'+tag)[idt][irand])


			mstar_plt = np.array(mstar_plt).flatten()
			cmap = plt.cm.viridis  # Choose a colormap
			normalize = plt.Normalize(vmin=np.min(np.log10(mstar_plt)), vmax=np.max(np.log10(mstar_plt)))  # Normalize stellar mass for colormap

			# Create figure and axis objects
			fig, axs = plt.subplots(nrows=3, figsize=(5, 10),  sharex='col')

			# Plot relative velocity, density, and accretion rate over time
			for i in range(Nsample):
				axs[0].plot(tplt, vplt[i], color=cmap(normalize(np.log10(mstar_plt[i]))), linewidth=1)
			
			axs[0].set_yscale('log')
			axs[0].set_ylabel('Relative velocity: $\\Delta v_\mathrm{gas}$ [km s$^{-1}$]')
			axs[0].tick_params(which='both', axis='both', direction='inout', right=True, left=True, top=True, bottom=True)


			for i in range(Nsample):
				axs[1].plot(tplt, rhoplt[i], color=cmap(normalize(np.log10(mstar_plt[i]))), linewidth=1)

			Thigh_dense = {'L1495': [2616, 31.7], 'B213': [1095,13.7], 'L1521': [1584, 17.6], "HCl2" : [1513,  15.8], 'L1498': [373, 5.7],  'L1506': [491, 7.7]}
			ireg = 0
			for Treg in Thigh_dense:
				Mtmp = Thigh_dense[Treg][0]
				atmp = Thigh_dense[Treg][1]
				rhotmp = Mtmp/(4.*np.pi*0.333*(atmp/np.pi)**1.5)
				rhotmp *= Msol2g*(1./pc2cm)**3
				axs[1].axhline(rhotmp, color=CB_color_cycle[ireg], linewidth=1, linestyle='dotted')
				x0 = 1.0
				dx = 1.1
				axs[1].annotate('', xy=(x0+dx*ireg, rhotmp*2.0), xytext=(x0+dx*ireg, rhotmp*0.9),
				arrowprops=dict(arrowstyle='->', color=CB_color_cycle[ireg], lw=1), color=CB_color_cycle[ireg])
				axs[1].text(x0+dx*ireg, rhotmp*10.0, Treg, color=CB_color_cycle[ireg])

				ireg+=1
			

			#axs[1].axhline(6.46e-22, color='r', linewidth=1, linestyle='dashed', label='Mean high density regions in Taurus:\nGoldsmith et al. 2008')
			axs[1].set_yscale('log')
			axs[1].set_ylabel('Density: $\\rho_\mathrm{gas}$ [g cm$^{-3}$]')

			axs[1].tick_params(which='both', axis='both', direction='inout', right=True, left=True, top=True, bottom=True)
			#axs[1].legend(loc='best')
			for i in range(Nsample):
				axs[2].plot(tplt, Mdplt[i], color=cmap(normalize(np.log10(mstar_plt[i]))), linewidth=1)

			Mda =  getattr(self, 'mdotBHLevol'+tag)[idt]*year2s/Msol2g #np.zeros((len(iregs_avg), len(trange)))\
			print('Shape of Mda array:', Mda.shape)
			msta = np.array(getattr(self, 'mstevol'+tag))/Msol2g
			Mda /= msta[:, np.newaxis]**2

			tgrd = tplt[np.newaxis,:]*np.ones(Mda.shape)
			
			Mda_med = np.median(Mda, axis=0)
			Mda_mean = np.mean(Mda, axis=0)
			binsy = np.logspace(np.log10(3e-14), np.log10(5e-6), 20)
			binsx = np.linspace(0.0, 8.0, 25)
			axs[2].plot(tplt, Mda_med, color='r', linewidth=1, linestyle='solid', label='Median')
			axs[2].plot(tplt, Mda_mean, color='r', linewidth=1, linestyle='dashed', label='Mean')
			axs[2].hist2d(tgrd.flatten(), Mda.flatten(), bins=(binsx, binsy), cmap='gray_r')

			axs[2].set_ylabel('Norm. BHL acc.: $\dot{M}_\mathrm{BHL} \\cdot \left(\\frac{m_*}{1\, M_\odot}\\right)^{-2}$ [$M_\odot$ yr$^{-1}$]')
			axs[2].set_yscale('log')
			axs[2].set_xlim([0.0, 8.0])
			axs[2].set_xlabel('Time: $t$ [Myr]')
			axs[2].tick_params(which='both', axis='both', direction='inout', right=True, left=True, top=True, bottom=True)

			axs[2].legend(loc='best', fontsize=7)

			#axs[2].plot(tsp, Mdot_cc, color='k',  linewidth=1, linestyle='dashed', label='Cloud capture')

			#axs[2].legend(loc='best', fontsize=7)
			#axs[2, 1].axhline(np.mean(mdotacc[iplot]), linestyle='solid', linewidth=1, color='k')
			#axs[2, 1].axhline(np.median(mdotacc[iplot]), linestyle='dashed', linewidth=1, color='k')\

			#axs[2, 0].axhline(np.mean(mdotacc[iplot]), linestyle='solid', linewidth=1, color='k')
			#axs[2, 0].axhline(np.median(mdotacc[iplot]), linestyle='dashed', linewidth=1, color='k')

			sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
			sm.set_array([])
			# Create a colorbar

			axs[2].set_ylim([3e-14,5e-6])
			axs[1].set_ylim([3e-25, 3e-19])
			axs[0].set_ylim([0.03, 20.0])

			# Remove white space between plots
			plt.subplots_adjust(wspace=0.05, hspace=0, top=0.95)
			cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), orientation='horizontal', location='top',fraction=0.05, pad=0.00, anchor=(0.5, 1.0))
			cbar.set_label('log. Star mass: $m_*$ [$M_\odot$]')

			plt.savefig('mdot_tevol.pdf', bbox_inches='tight', format='pdf')
			# Adjust layout and show the plot
			#plt.tight_layout()
			plt.show()
		
		def plot_accretion_rates(self, Nsample=8, mlim =250.0):
			iregs, ists = self.draw_representative_sample(Nsample, mreglim=mlim).T[:]

			tplt = []
			Raplt = []
			Mdplt = []
			vplt = []
			rhoplt = []
			mstar_plt = []
			for iplt, ireg in enumerate(iregs):
				ist = ists[iplt]

				mstar_plt.append(self.region_list[ireg]['msts'][ist]/Msol2g) 
				tplt.append(self.region_list[ireg]['t_evols'][ist]/Myr2s)
				Raplt.append(self.region_list[ireg]['R_accs'][ist]/au2cm)
				Mdplt.append(self.region_list[ireg]['Mdot_accs'][ist]*year2s/Msol2g/(mstar_plt[-1]**2))
				vplt.append(self.region_list[ireg]['dv_local'][ist]/1e5)
				rhoplt.append(self.region_list[ireg]['rho_local'][ist])

			mstar_plt = np.array(mstar_plt).flatten()
			cmap = plt.cm.viridis  # Choose a colormap
			normalize = plt.Normalize(vmin=np.min(np.log10(mstar_plt)), vmax=np.max(np.log10(mstar_plt)))  # Normalize stellar mass for colormap

			# Create figure and axis objects
			fig, axs = plt.subplots(nrows=3, figsize=(5, 10),  sharex='col')

			# Plot relative velocity, density, and accretion rate over time
			for i in range(Nsample):
				axs[0].plot(tplt[i], vplt[i], color=cmap(normalize(np.log10(mstar_plt[i]))), linewidth=1)
			
			axs[0].set_yscale('log')
			axs[0].set_ylabel('Relative velocity: $\\Delta v_\mathrm{gas}$ [km s$^{-1}$]')
			axs[0].tick_params(which='both', axis='both', direction='inout', right=True, left=True, top=True, bottom=True)


			for i in range(Nsample):
				axs[1].plot(tplt[i], rhoplt[i], color=cmap(normalize(np.log10(mstar_plt[i]))), linewidth=1)


			arrow_height = 0.5  # Height of the arrow in data coordinates
			arrow_length = 0.3
			Thigh_dense = {'L1495': [2616, 31.7], 'B213': [1095,13.7], 'L1521': [1584, 17.6], "HCl2" : [1513,  15.8], 'L1498': [373, 5.7],  'L1506': [491, 7.7]}
			ireg = 0
			for Treg in Thigh_dense:
				Mtmp = Thigh_dense[Treg][0]
				atmp = Thigh_dense[Treg][1]
				rhotmp = Mtmp/(4.*np.pi*0.333*(atmp/np.pi)**1.5)
				rhotmp *= Msol2g*(1./pc2cm)**3
				print(rhotmp)
				axs[1].axhline(rhotmp, color=CB_color_cycle[ireg], linewidth=1, linestyle='dotted')
				x0 = 1.0
				dx = 1.1
				axs[1].annotate('', xy=(x0+dx*ireg, rhotmp*2.0), xytext=(x0+dx*ireg, rhotmp*0.9),
				arrowprops=dict(arrowstyle='->', color=CB_color_cycle[ireg], lw=1), color=CB_color_cycle[ireg])
				axs[1].text(x0+dx*ireg, rhotmp*10.0, Treg, color=CB_color_cycle[ireg])

				ireg+=1
			

			#axs[1].axhline(6.46e-22, color='r', linewidth=1, linestyle='dashed', label='Mean high density regions in Taurus:\nGoldsmith et al. 2008')
			axs[1].set_yscale('log')
			axs[1].set_ylabel('Density: $\\rho_\mathrm{gas}$ [g cm$^{-3}$]')

			axs[1].tick_params(which='both', axis='both', direction='inout', right=True, left=True, top=True, bottom=True)
			#axs[1].legend(loc='best')
			for i in range(Nsample):
				axs[2].plot(tplt[i], Mdplt[i], color=cmap(normalize(np.log10(mstar_plt[i]))), linewidth=1)


			iregs_avg, ists_avg = self.draw_representative_sample(1000, mreglim=mlim).T[:]
			trange = np.linspace(0.0, 8.0, 500)
			Mda = np.zeros((len(iregs_avg), len(trange)))
			for iplt, ireg in enumerate(iregs_avg):
				ist = ists_avg[iplt]
				Mda_tmp = self.region_list[ireg]['Mdot_accs'][ist]*year2s/Msol2g
				t_tmp = self.region_list[ireg]['t_evols'][ist]/Myr2s
				mst_tmp = self.region_list[ireg]['msts'][ist]/Msol2g
				Mda[iplt] = np.interp(trange, t_tmp, Mda_tmp/mst_tmp/mst_tmp)

			Mda_med = np.median(Mda, axis=0)
			Mda_mean = np.mean(Mda, axis=0)
			binsy = np.logspace(np.log10(3e-14), np.log10(5e-6), 20)
			binsx = np.linspace(trange[0], trange[-1], 25)
			tgrid = trange[np.newaxis, :]*np.ones(Mda.shape)
			axs[2].plot(trange, Mda_med, color='r', linewidth=1, linestyle='solid', label='Median')
			axs[2].plot(trange, Mda_mean, color='r', linewidth=1, linestyle='dashed', label='Mean')
			axs[2].hist2d(tgrid.flatten(), Mda.flatten(), bins=(binsx, binsy), cmap='gray_r')

			axs[2].set_ylabel('Norm. BHL acc.: $\dot{M}_\mathrm{BHL} \\cdot \left(\\frac{m_*}{1\, M_\odot}\\right)^{-2}$ [$M_\odot$ yr$^{-1}$]')
			axs[2].set_yscale('log')
			axs[2].set_xlim([0.0, 8.0])
			axs[2].set_xlabel('Time: $t$ [Myr]')
			axs[2].tick_params(which='both', axis='both', direction='inout', right=True, left=True, top=True, bottom=True)

			axs[2].legend(loc='best', fontsize=7)

			#axs[2].plot(tsp, Mdot_cc, color='k',  linewidth=1, linestyle='dashed', label='Cloud capture')

			#axs[2].legend(loc='best', fontsize=7)
			#axs[2, 1].axhline(np.mean(mdotacc[iplot]), linestyle='solid', linewidth=1, color='k')
			#axs[2, 1].axhline(np.median(mdotacc[iplot]), linestyle='dashed', linewidth=1, color='k')\

			#axs[2, 0].axhline(np.mean(mdotacc[iplot]), linestyle='solid', linewidth=1, color='k')
			#axs[2, 0].axhline(np.median(mdotacc[iplot]), linestyle='dashed', linewidth=1, color='k')

			sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
			sm.set_array([])
			# Create a colorbar

			axs[2].set_ylim([3e-14,5e-6])
			axs[1].set_ylim([3e-25, 3e-19])
			axs[0].set_ylim([0.03, 20.0])

			# Remove white space between plots
			plt.subplots_adjust(wspace=0.05, hspace=0, top=0.95)
			cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), orientation='horizontal', location='top',fraction=0.05, pad=0.00, anchor=(0.5, 1.0))
			cbar.set_label('log. Star mass: $m_*$ [$M_\odot$]')

			plt.savefig('mdot_tevol.pdf', bbox_inches='tight', format='pdf')
			# Adjust layout and show the plot
			#plt.tight_layout()
			plt.show()
		



