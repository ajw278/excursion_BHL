import excursion
import wind_limits as wl	


if __name__=='__main__':

	#Draw 10^5 random GMCs and plot the GMC mass function
	#excursion.GMC_MF(Nsample=10000)
	#exit()
	#

	#Genererate density histories for at least 10000 stars in at least 500 different regions
	sfdb = excursion.get_density_history(GMCmin=2.0, GMCmax=1000.0, Nregions=1000, Nstars=10000, tag='_ll1')

	#wl.construct_grid()


	print('\n\n *******Lower SFR mass*******\n\n')
	#For a representative sample of 500 stars, calculate the disc evolution
	tag = sfdb.calc_discevol(redraw=False, Nsample=500, minit=0.0, minitdisp=0.0, mlim=250., ptag='m250_'+sfdb.tag+'_', wind=False, eps_wind=0.1)
	print('Tag:', tag)
	
	#Plot the BHL accretion rate histories
	sfdb.plot_accretion_rates_wevap(tag=tag)

	sfdb.plot_discfrac_msplit(tag=tag)
	#Plot the BHL accretion rate histories
	#sfdb.plot_accretion_rates()

	#Make the plots for the paper
	sfdb.plot_discfrac(tag=tag)
	sfdb.plot_rplf(tag=tag)
	sfdb.plot_all(tag=tag)
	sfdb.plot_vturb(tag=tag)
	sfdb.plot_Next(tag=tag)


	#Genererate density histories for at least 10000 stars in at least 500 different regions
	sfdb = excursion.get_density_history(GMCmin=10.0, GMCmax=1000.0, Nregions=300, Nstars=5000, tag='')

	print('\n\n *******Fiducial*******\n\n')

	#For a representative sample of 500 stars, calculate the disc evolution
	tag = sfdb.calc_discevol(redraw=False, Nsample=500, minit=0.0, minitdisp=0.0, mlim=250., ptag='m250_', wind=False, eps_wind=0.1)
	print('Tag:', tag)
	
	
	#Plot the BHL accretion rate histories
	sfdb.plot_accretion_rates_wevap(tag=tag)

	sfdb.plot_discfrac_msplit(tag=tag)
	#Plot the BHL accretion rate histories
	#sfdb.plot_accretion_rates()

	#Make the plots for the paper
	sfdb.plot_discfrac(tag=tag)
	sfdb.plot_rplf(tag=tag)
	sfdb.plot_all(tag=tag)
	sfdb.plot_vturb(tag=tag)
	sfdb.plot_Next(tag=tag)



	print('\n\n *******Wind, 0 initial disc mass*******\n\n')


	#Test results for a substantial initial disc mass
	tag = sfdb.calc_discevol(redraw=False, Nsample=500, minit=0.0, minitdisp=0.0, ptag='m250_', wind=True, eps_wind=0.1)

	sfdb.plot_accretion_rates_wevap(tag=tag)
	sfdb.plot_all(tag=tag)
	sfdb.plot_rplf(tag=tag)

	print('\n\n *******IC disc vary, no wind *******\n\n')
	

	#Test results for a substantial initial disc mass
	tag = sfdb.calc_discevol(redraw=False, Nsample=500, minit=0.01, minitdisp=1.0, ptag='m250_', wind=False, eps_wind=0.1)

	sfdb.plot_accretion_rates_wevap(tag=tag)
	sfdb.plot_all(tag=tag)
	sfdb.plot_rplf(tag=tag)
