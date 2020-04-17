## Run python DSA_Laplace.py

'''
This python code provides posterior samples of the parameters involved in the dynamic survival analysis model.
'''

from numpy.random import RandomState
import time
from scipy.stats import gamma, beta, uniform

rand = RandomState()

from epidemiccore_w import *
import os as os

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle




def draw_from_prior(p, nSample = None):
    if nSample == None:
        return [p[0].rvs(), p[1].rvs(), p[2].rvs()]
    else:
        sample = np.zeros((nSample, len(p)), dtype=np.float)
        for i in range(nSample):
            sample[i] = [p[0].rvs(), p[1].rvs(), p[2].rvs()]
        return sample

def laplace(df, theta):
    N = min(2000,np.size(df))
    epi = Epidemic(file_or_df=df, bounds=bounds, abc=tuple(theta), plot_T=plot_T)
    epi.fit(N=N)
    epi.simulate_and_fit(N=N, n=n)
    fitted_parms = pd.DataFrame(
        {'a': list(f.a for f in epi.fits),
         'b': list(f.b for f in epi.fits),
         'c': list(f.c for f in epi.fits),
         'kinfty': list(f.kinfty for f in epi.fits),
         'R0': list(f.R0 for f in epi.fits),
         'n': list(f.n for f in epi.fits),
         'tau': list(f.tau for f in epi.fits),
         'rho': list(f.rho for f in epi.fits)})
    return fitted_parms

today = pd.to_datetime('today')

root_folder = os.getcwd()
data_folder = os.path.join(root_folder,'data')

fname = 'laplace_dict' + today.strftime("%m%d") + '.pkl'
laplace_dict = pickle.load(open(os.path.join(data_folder,fname), "rb"))
datafile = laplace_dict['datafile']
plot_folder = laplace_dict['plot_folder']
location = laplace_dict['location']
last_date = laplace_dict['last_date']
estimate_gamma = laplace_dict['estimate_gamma']
ifsmooth = laplace_dict['ifsmooth']
ifMPI = laplace_dict['ifMPI']




plot_folder = os.path.join(root_folder,plot_folder)

if not(os.path.exists(plot_folder)):
    os.system('mkdir %s' %plot_folder)

df_ohio_full = pd.read_csv(os.path.join(data_folder,datafile), parse_dates=["time"])

last_date_on_file = df_ohio_full.time.max()

n_remove = (last_date_on_file - last_date).days
print('Removing last %s days' % n_remove)

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

pickle.dump(df,open("df","wb"),protocol=3)

if estimate_gamma:
    print('Generating recovery times by uniformly distributing throughout each day')
    recovery_data = list(
        i + rand.uniform() for i, y in enumerate(df_ohio['recovery'].values + df_ohio['deaths'].values) for z in
        range(y.astype(int)))
    df_recovery = pd.DataFrame(recovery_data, index=range(len(recovery_data)), columns=['recovery'])
    pickle.dump(df_recovery, open("df_recovery", "wb"), protocol=3)

plot_T = 150
N = min(2000,df_ohio['cum_confirm'].iloc[-1])
n = 5
bounds = [(0.3,0.50),(0.5,7.0),(1E-9,1E-3)]
if os.path.exists("bounds"):
    os.remove("bounds")
pickle.dump(bounds, open("bounds", "wb"), protocol=3)


# a_prior = expon(scale=0.4)
# b_prior = expon(scale=0.6)
#c_prior = expon(scale=1E-5)
a_prior = uniform(loc=bounds[0][0], scale=bounds[0][1])
b_prior = uniform(loc=bounds[1][0], scale=bounds[1][1])
c_prior = uniform(loc=bounds[2][0], scale=bounds[2][1])
p = [a_prior, b_prior, c_prior]
pickle.dump(p,open("p","wb"),protocol=3)

# a_prior = uniform(loc=bounds[0][0], scale=bounds[0][1])
# b_prior = uniform(loc=bounds[1][0], scale=bounds[1][1])
# c_prior = uniform(loc=bounds[2][0], scale=bounds[2][1])
# p = [a_prior, b_prior, c_prior]

if ifMPI:
    nPriorSamples = 100
else:
    nPriorSamples = 100

st = time.time()

thetas = draw_from_prior(p, nSample = nPriorSamples)
if os.path.exists("thetas"):
    os.remove("thetas")
pickle.dump(thetas, open("thetas", "wb"), protocol=3)

if ifMPI:
    os.system("mpiexec -n %s python parallel_Laplace.py" % THREADS)
    thetas_fitted = pickle.load(open("thetas_fitted", "rb"))
else:
    colNames = ['a', 'b', 'c', 'kinfty', 'R0', 'n', 'tau', 'rho']
    thetas_fitted = pd.DataFrame(columns=colNames)
    for i in range(nPriorSamples):
        temp = laplace(df, thetas[i])
        thetas_fitted = thetas_fitted.append(temp, ignore_index=True)

fname = location + '_fits_' + today.strftime("%m%d") + '.csv'
thetas_fitted.to_csv(os.path.join(plot_folder,fname), index=False)


print("Total time to draw posterior samples %s" % (time.time() - st))

temp = np.zeros((np.size(thetas,1), np.size(thetas_fitted,0)), dtype=np.float)
temp[0] = thetas_fitted.a.values
temp[1] = thetas_fitted.b.values
temp[2] = thetas_fitted.c.values
m = np.mean(thetas_fitted, axis=0)
s = np.std(thetas, axis=0)
theta = np.mean(temp,axis=1)
cov = np.cov(temp)
print(m)


epiT = Epidemic(file_or_df=df,bounds=bounds,abc=tuple(theta),plot_T=plot_T, p=p)
epiT.laplace_fit(N=N)

if estimate_gamma:
    fig_recovery = epiT.estimate_gamma(df_recovery=df_recovery, N=N, x0=(0.1, -5),
                                       bounds=[(1.0 / 25, 1.0 / 5), (-10, 0)], approach='offset')
    fname = location + '_recovery' + today.strftime("%m%d")
    fig_save(fig_recovery, plot_folder, fname)































