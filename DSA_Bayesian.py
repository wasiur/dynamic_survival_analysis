'''
This python script performs the Metropolis-Hastings algorithm to generate posterior samples of the parameters
involved in the dynamic survival analysis.

Warning: It still requires significant work. The chain seems to be horribly slow.
'''

from numpy.random import RandomState
import time

rand = RandomState()

from epidemiccore_w import *
from my_mh import *
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

THREADS = 20

today = pd.to_datetime('today')

root_folder = os.getcwd()
data_folder = os.path.join(root_folder,'data')

fname = 'mh_dict' + today.strftime("%m%d") + '.pkl'
mh_dict = pickle.load(open(os.path.join(data_folder,fname), "rb"))
datafile = mh_dict['datafile']
plot_folder = mh_dict['plot_folder']
location = mh_dict['location']
last_date = mh_dict['last_date']
estimate_gamma = mh_dict['estimate_gamma']
ifsmooth = mh_dict['ifsmooth']
ifMPI = mh_dict['ifMPI']
burn_in = np.int(mh_dict['burn_in'])
nChains = np.int(mh_dict['nChains'])

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


n_prior = 10
chain_length = 10**5

## set up prior
a = expon(scale=0.4)
b = expon(scale=0.6)
c = expon(scale=1E-6)
p = [a,b,c]
proposal = norm(scale=1E-1)
bayes = {'p': p,
         'proposal': proposal,
         'chain_length': chain_length,
         'burn_in': burn_in,
         'nChains': nChains}

fname = 'bayes' + today.strftime("%m%d")
with open(os.path.join (plot_folder, fname), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(bayes, output, -1)

N = min(2000,df_ohio['cum_confirm'].iloc[-1])
n = 1000
plot_T = 150 # show system through end of epidemic
bounds = [(0.1,1),(0.1,1),(1E-9,1E-1)]
import pickle

st = time.time()

epiT = Epidemic(file_or_df=df,bounds=bounds,abc=tuple(0.4, 0.6, 1E-6),plot_T=plot_T)
epiT.fit(N=N) # use all the data


fname = location + '_epi_' + today.strftime("%m%d") + '.pkl'
with open(os.path.join (plot_folder, fname), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(epiT, output, -1)


if ifMPI:
    os.system("mpiexec -n %s python parallel_mh.py" % THREADS)
    fname = "mh_chains" + today.strftime("%m%d")
    mh_chains = pickle.load(open(os.path.join(plot_folder, fname), "rb"))
else:
    temp = mh(epiT, p, proposal, burn_in, chain_length)
    res_a = temp[:, 0]
    res_b = temp[:, 1]
    res_c = temp[:, 2]
    mh_chains = {'a': res_a, 'b': res_b, 'c': res_c}
    fname = "mh_chains" + today.strftime("%m%d")
    pickle.dump(mh_chains, open(os.path.join(plot_folder, fname), "wb"), protocol=3)

a_chain = mh_chains["a"]
b_chain = mh_chains["b"]
c_chain = mh_chains["c"]








































































