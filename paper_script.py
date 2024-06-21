import excursion
		


if __name__=='__main__':

	#Draw 10^5 random GMCs and plot the GMC mass function
	#excursion.GMC_MF(Nsample=100000)

	#Genererate density histories for at least 10000 stars in at least 500 different regions
	sfdb = excursion.get_density_history(Nregions=300, Nstars=5000)


	#For a representative sample of 500 stars, calculate the disc evolution
	tag = sfdb.calc_discevol(redraw=False, Nsample=500, minit=0.0, minitdisp=0.0, mlim=250., ptag='m250_')
	print('Tag:', tag)

	#Plot the BHL accretion rate histories
	sfdb.plot_accretion_rates_wevap(ptag='m250_', minit=0.0)

	#Plot the BHL accretion rate histories
	#sfdb.plot_accretion_rates()

	#Make the plots for the paper
	sfdb.plot_all(tag=tag)
	sfdb.plot_rplf(tag=tag)
	sfdb.plot_vturb(tag=tag)
	sfdb.plot_Next(tag=tag)


	#Test results for a substantial initial disc mass
	tag = sfdb.calc_discevol(redraw=False, Nsample=500, minit=0.01, minitdisp=1.0, ptag='m250_')

	sfdb.plot_all(tag=tag)
	sfdb.plot_rplf(tag=tag)

