from epidemiccore_w import Epidemic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import matplotlib
from numpy.random import RandomState

rand = RandomState()

import os as os

Root_Folder = os.getcwd()
Plot_Folder = os.path.join(Root_Folder,'plots')


def fig_save(fig, Plot_Folder, fname):
    fig.savefig(os.path.join (Plot_Folder, fname),dpi=300)
    fig.savefig (os.path.join (Plot_Folder, fname + "." + 'pdf'), format='pdf')
    # fig.savefig (os.path.join (Plot_Folder, fname + "." + 'svg'), format='svg')
    fig.savefig (os.path.join (Plot_Folder, fname + "." + 'eps'), format='eps', dpi=1000)


'''
# USA data
'''
usa_file = 'US0316.csv'
df_usa = pd.read_csv(usa_file)

# get daily counts
df_usa_inf = df_usa['cum_confirm'].diff().abs()
df_usa_inf[0] = df_usa['cum_confirm'].iloc[0]
df_usa['daily_confirm'] = df_usa_inf

inf_count = df_usa['cum_confirm'].values
death_count = df_usa['cum_dead'].values
cure_count = df_usa['cum_heal'].values

# get daily counts recovery
df_usa_cure = df_usa['cum_heal'].diff().abs()
df_usa_cure[0] = df_usa['cum_heal'].iloc[0]
df_usa['recovery'] = df_usa_cure

fig_usa = Epidemic.Plot_Data(
	inf_count=inf_count,
	death_count=death_count,
	cure_count=cure_count,
	daily_confirm=df_usa['daily_confirm'].values,
	daily_cure=df_usa['recovery'].values,
	t=np.arange(1,df_usa.shape[0]+1),
	legend='left',
	scale_density=1.0,
	location='United States of America'
)
fname = 'usa'
fig_save(fig_usa,Plot_Folder,fname)



# generate infection times through uniformly distributing throughout each day
infection_data = list(i+rand.uniform() for i,y in enumerate(df_usa['daily_confirm'].values) for z in range(y.astype(int)))
df = pd.DataFrame(infection_data,index=range(len(infection_data)),columns=['infection'])

# get daily counts recovery
df_usa_cure = df_usa['cum_heal'].diff().abs()
df_usa_cure[0] = df_usa['cum_heal'].iloc[0]
df_usa['recovery'] = df_usa_cure

# generate recovery times through uniformly distributing throughout each day
recovery_data = list(i+rand.uniform() for i,y in enumerate(df_usa['recovery'].values) for z in range(y.astype(int)))
df_recovery = pd.DataFrame(recovery_data,index=range(len(recovery_data)),columns=['recovery'])

#file = '20200302_covid_transformed_data.csv'
bounds = [(0.1,1),(0.1,1),(1E-9,1E-1)]
#df = pd.read_csv(file)

N = min(2000,df_usa['cum_confirm'].iloc[-1])
n = 5
plot_T = 150 # show system through end of epidemic

epiT = Epidemic(file_or_df=df,bounds=bounds,abc=(0.4,0.6,1E-8),plot_T=plot_T)
epiT.fit(N=N) # use all the data

fi = epiT.Plot_FI_abc()
fname = 'FIusa'
fig_save(fi,Plot_Folder,fname)
#fi.savefig(os.path.join (Plot_Folder, fname),dpi=300)
#fi.savefig (os.path.join (Plot_Folder, fname + "." + 'pdf'), format='pdf')
## fig.savefig (os.path.join (Plot_Folder, fname + "." + 'svg'), format='svg')
#fi.savefig (os.path.join (Plot_Folder, fname + "." + 'eps'), format='eps', dpi=1000)

norescale = epiT.Plot_FI_integrand_ab(rescale=True)
fname = 'norescaleusa'
fig_save(norescale,Plot_Folder,fname)
# norescale.savefig(os.path.join (Plot_Folder, fname),dpi=300)
# norescale.savefig (os.path.join (Plot_Folder, fname + "." + 'pdf'), format='pdf')
# # fig.savefig (os.path.join (Plot_Folder, fname + "." + 'svg'), format='svg')
# norescale.savefig (os.path.join (Plot_Folder, fname + "." + 'eps'), format='eps', dpi=1000)

rescale = epiT.Plot_FI_integrand_ab(rescale=False)
fname = 'rescaleusa'
fig_save(rescale,Plot_Folder,fname)
# rescale.savefig(os.path.join (Plot_Folder, fname),dpi=300)
# rescale.savefig (os.path.join (Plot_Folder, fname + "." + 'pdf'), format='pdf')
# # fig.savefig (os.path.join (Plot_Folder, fname + "." + 'svg'), format='svg')
# rescale.savefig (os.path.join (Plot_Folder, fname + "." + 'eps'), format='eps', dpi=1000)


fig_recovery = epiT.estimate_gamma(df_recovery=df_recovery,N=N,x0=(0.1,-5),bounds=[(1.0/25,1.0/5),(-10,0)],approach='offset')
fname = 'recoveryusa'
fig_save(fig_recovery,Plot_Folder,fname)

epiT.simulate_and_fit(N=N,n=n)

fig_density = epiT.plot_density_fit()
fname = 'Tfinaldensityusa'
fig_save(fig_density,Plot_Folder,fname)
# fig_density.savefig(os.path.join (Plot_Folder, fname),dpi=300)
# fig_density.savefig (os.path.join (Plot_Folder, fname + "." + 'pdf'), format='pdf')
# # fig.savefig (os.path.join (Plot_Folder, fname + "." + 'svg'), format='svg')
# fig_density.savefig (os.path.join (Plot_Folder, fname + "." + 'eps'), format='eps', dpi=1000)

fig_inf_curve, fig_inf_T = epiT.plot_infections()
fname = 'Tfinalinfectionsusa'
fig_save(fig_inf_T,Plot_Folder,fname)


fig_dropout = epiT.plot_dropout(pt=epiT.delta/epiT.a)
fname = 'Tfinaldropoutusa'
fig_save(fig_dropout,Plot_Folder,fname)
# fig_inf_T.savefig('../natcommfigures/Tfinalinfectionsusa',dpi=300)
# fig_dropout.savefig('../natcommfigures/Tfinaldropoutusa',dpi=300)


fig_a, fig_b, fig_c, fig_R0, fig_rho, fig_n, fig_sT, fig_kinfty, fig_sinfty, fig_invrho = epiT.get_histograms()
fname = 'Tfinalausa'
fig_save(fig_a,Plot_Folder,fname)
#fig_a.savefig('../natcommfigures/Tfinalausa',dpi=300)

fname = 'Tfinalbusa'
fig_save(fig_b,Plot_Folder,fname)
#fig_b.savefig('../natcommfigures/Tfinalbusa',dpi=300)

fname = 'Tfinalcusa'
fig_save(fig_c,Plot_Folder,fname)
#fig_c.savefig('../natcommfigures/Tfinalcusa',dpi=300)

fname = 'TfinalR0usa'
fig_save(fig_R0,Plot_Folder,fname)
#fig_R0.savefig('../natcommfigures/TfinalR0usa',dpi=300)

fname = 'Tfinalrhousa'
fig_save(fig_rho,Plot_Folder,fname)
#fig_rho.savefig('../natcommfigures/Tfinalrhousa',dpi=300)

fname = 'Tfinalnusa'
fig_save(fig_n,Plot_Folder,fname)
#fig_n.savefig('../natcommfigures/Tfinalnusa',dpi=300)

fname = 'TfinalsTusa'
fig_save(fig_sT,Plot_Folder,fname)
#fig_sT.savefig('../natcommfigures/TfinalsTusa',dpi=300)

fname = 'Tfinalkinftyusa'
fig_save(fig_kinfty,Plot_Folder,fname)
#fig_kinfty.savefig('../natcommfigures/Tfinalkinftyusa',dpi=300)

fname = 'Tfinalsinftyusa'
fig_save(fig_sinfty,Plot_Folder,fname)
#fig_sinfty.savefig('../natcommfigures/Tfinalsinftyusa',dpi=300)

fname = 'Tfinalinvrhousa'
fig_save(fig_invrho,Plot_Folder,fname)
#fig_invrho.savefig('../natcommfigures/Tfinalinvrhousa',dpi=300)


'''
# generate figures for fit at days 20, 30
'''
fig_combined_survival = Epidemic.plot_survival_fits([epiT])
fname = 'survivalsusa'
fig_save(fig_combined_survival, Plot_Folder,fname)
#fig_combined_survival.savefig('../natcommfigures/survivalsusa',dpi=300)


fig_combined_infection = Epidemic.plot_infection_fits([epiT])
fname = 'infectionsusa'
fig_save(fig_combined_infection, Plot_Folder,fname)
#fig_combined_infection.savefig('../natcommfigures/infectionsusa',dpi=300)


'''
# show optimizer results
'''

print("T=final boundary pts", epiT.number_boundary_samples, "T=final interior pts", epiT.number_interior_samples)
epiT.summary()
plt.show(block=True)
