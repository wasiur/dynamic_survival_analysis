## Run python DSA.py

'''
This python code performs dynamic survival analysis.
'''

from numpy.random import RandomState
import time

rand = RandomState()

from epidemiccore_w import *
import os as os

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

today = pd.to_datetime('today')

root_folder = os.getcwd()
data_folder = os.path.join(root_folder,'data')

fname = 'dsa_dict' + today.strftime("%m%d") + '.pkl'
dsa_dict = pickle.load(open(os.path.join(data_folder,fname), "rb"))
datafile = dsa_dict['datafile']
plot_folder = dsa_dict['plot_folder']
location = dsa_dict['location']
last_date = dsa_dict['last_date']
estimate_gamma = dsa_dict['estimate_gamma']
ifsmooth = dsa_dict['ifsmooth']
ifMPI = dsa_dict['ifMPI']



n_remove = (today - last_date).days
print('Removing last %s days' % n_remove)

plot_folder = os.path.join(root_folder,plot_folder)

if not(os.path.exists(plot_folder)):
    os.system('mkdir %s' %plot_folder)

df_ohio_full = pd.read_csv(os.path.join(data_folder,datafile), parse_dates=["time"])

df_ohio = df_ohio_full.drop(df_ohio_full.tail(n_remove).index)
print(df_ohio)

print(df_ohio["time"].max())

day0 = df_ohio["time"].min()
print(day0)
today = pd.to_datetime('today')

## smoothing counts
df_ohio["rolling_mean"] = df_ohio.daily_confirm.rolling(window=3).mean()
df_ohio["rolling_mean"] = df_ohio.apply(lambda dd: dd.daily_confirm if np.isnan(dd.rolling_mean)
                                                   else dd.rolling_mean, axis=1)

if ifsmooth:
    print('Generating infection times by uniformly distributing throughout each day from smoothed daily counts\n')
    infection_data = list(
        i + rand.uniform() for i, y in enumerate(df_ohio['rolling_mean'].values) for z in range(y.astype(int)))
    df = pd.DataFrame(infection_data, index=range(len(infection_data)), columns=['infection'])
else:
    print('Generating infection times by uniformly distributing throughout each day from actual daily counts\n')
    infection_data = list(
        i + rand.uniform() for i, y in enumerate(df_ohio['daily_confirm'].values) for z in range(y.astype(int)))
    df = pd.DataFrame(infection_data,index=range(len(infection_data)),columns=['infection'])


if estimate_gamma:
    print('Generating recovery times by uniformly distributing throughout each day')
    recovery_data = list(
        i + rand.uniform() for i, y in enumerate(df_ohio['recovery'].values + df_ohio['deaths'].values) for z in
        range(y.astype(int)))
    df_recovery = pd.DataFrame(recovery_data, index=range(len(recovery_data)), columns=['recovery'])


bounds = [(0.1,1),(0.1,1),(1E-9,1E-1)]

N = min(2000,df_ohio['cum_confirm'].iloc[-1])
n = 1000
plot_T = 150 # show system through end of epidemic
import pickle
st = time.time()

epiT = Epidemic(file_or_df=df,bounds=bounds,abc=(0.4, 0.6, 1E-6),plot_T=plot_T)
epiT.fit(N=N) # use all the data



if ifMPI:
    epiT.simulate_and_fit_parallel(N=N, n=n)
else:
    epiT.simulate_and_fit(N=N, n=n)

print("Total time Simulate and Fit %s" % (time.time() - st))

print('Plotting density fit\n')
fig_density = epiT.plot_density_fit()
fname = 'Tfinaldensity' + today.strftime("%m%d")
fig_save(fig_density,plot_folder,fname)


fig_inf_curve, fig_inf_T = epiT.plot_infections()
fname = 'Tfinalinfections' + today.strftime("%m%d")
fig_save(fig_inf_T,plot_folder,fname)

if estimate_gamma:
    print('Estimating gamma\n')
    fig_recovery = epiT.estimate_gamma(df_recovery = df_recovery, N=N, x0=(0.1, -5),
                                       bounds=[(1.0 / 25, 1.0 / 5), (-10, 0)], approach='offset')
    fname = 'recovery_' + today.strftime("%m%d")
    fig_save(fig_recovery, plot_folder, fname)

    fig_dropout = epiT.plot_dropout(pt=epiT.delta / epiT.a)
    fname = 'Tfinaldropout' + today.strftime("%m%d")
    fig_save(fig_dropout, plot_folder, fname)


fig_combined_infection = Epidemic.plot_infection_fits([epiT])
fname = 'infections' + today.strftime("%m%d")
fig_save(fig_combined_infection, plot_folder,fname)

'''
# show optimizer results
'''

print("T=final boundary pts", epiT.number_boundary_samples, "T=final interior pts", epiT.number_interior_samples)
# epiT.summary()
epiT.summary(ifSave = True, fname=os.path.join(plot_folder,'summary.tex'))
plt.show(block=True)

fname = location + '_epi_' + today.strftime("%m%d") + '.pkl'
with open(os.path.join (plot_folder, fname), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(epiT, output, -1)

fname = 'fitted_parms_' + today.strftime("%m%d") + '.csv'
fitted_parms = pd.DataFrame({'a': list(f.a for f in epiT.fits), 'b' : list(f.b for f in epiT.fits), 'c': list(f.c for f in epiT.fits)})
fitted_parms.to_csv(os.path.join(plot_folder,fname))

m = epiT.theta
s = [np.sqrt(epiT.var_a()), np.sqrt(epiT.var_b()), np.sqrt(epiT.var_c())]
cov = epiT.cov_abc()

nDays = 150
dates = pd.DataFrame({'d':[day0 + pd.DateOffset(i) for i in np.arange(nDays)]})
fname = location + 'predictions_' + today.strftime("%m%d")
nSim = 2000
samples = parm_sample_correlated(m,cov,nSim)
predictions = epiT.predict(samples, df_ohio_full, nSim, dates, plot_folder, fname)

