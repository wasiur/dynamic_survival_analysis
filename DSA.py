
import os
import pandas as pd
from numpy.random import RandomState
rand = RandomState()
import pystan
from optparse import OptionParser

from dsacore import *
from mycolours import *
my_plot_configs()


def sample_correlated_asymptotic(m, cov):
    sample = np.random.multivariate_normal(m, cov)
    for i in range(len(m)):
        if not (sample[i] > 0):
            sample[i] = m[i]
    return sample

def parm_sample_correlated(m, cov, nSample=1):
    sample = np.zeros((nSample,len(m)), dtype=np.float)
    for i in range(nSample):
        sample[i] = sample_correlated_asymptotic(m, cov)
    return sample

def main():
    """
    Runs the Dynamic Survival Analysis (DSA) model using the method of Maximum Likelihood Estimation (MLE).
    The confidence intervals are generated using the bootstrap method.

    :return: estimates of the model parameters
    """
    usage = "Usage: python DSA.py -d <datafile>"
    parser = OptionParser(usage)
    parser.add_option("-d", "--data-file", action="store", type="string", dest="datafile",
                      help="Name of the data file.")
    parser.add_option("-l", "--location", action="store", type="string", dest="location",
                      default="Ohio", help="Name of the location.")
    parser.add_option("-m", "--mpi", action="store_false", dest="ifMPI",
                      default=True, help="Indicates whether to use MPI for parallelization.")
    parser.add_option("-o", "--output-folder", action="store", dest="output_folder",
                      default="plots_dsa", help="Name of the output folder")
    parser.add_option("-s", "--smooth", action="store_true", dest="ifsmooth",
                      default=False, help="Indicates whether the daily counts should be smoothed.")
    parser.add_option("-f", "--final-date", action="store", type="string", dest="last_date",
                      default=None, help="Last day of data to be used")
    parser.add_option("-r", "--estimate-recovery-parameters", action="store_false", dest="estimate_gamma",
                      default=True, help="Indicates the parameters of the recovery distribution will be estimated")
    parser.add_option("-N", action="store", dest="N",
                      default=2000, type="int",
                      help="Size of the random sample")
    parser.add_option("-T", "--T", action="store", type="float", dest="T",
                      help="End of observation time", default=150.0)
    parser.add_option("--day-zero", type="string", action="store",
                      dest="day0", default=None,
                      help="Date of onset of the epidemic")
    parser.add_option("--niter", action="store", type="int", default=500,
                      dest="niter",
                      help="Number of bootstraps for parameter estimation")
    parser.add_option("--nchains", action="store", type="int", default=4, dest="nchains",
                      help="Number of chains")
    parser.add_option("--threads", action="store", type="int",
                      default=40, help="Number of threads for MPI")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="Runs with default choices")


    (options, args) = parser.parse_args()

    root_folder = os.getcwd()
    data_folder = os.path.join(root_folder, 'data')

    if options.verbose:
        print('Entering verbose mode\n')
        datafile = "dummy.csv"
        print("DSA will be performed on ", datafile)
        df_full = pd.read_csv(os.path.join(data_folder, datafile), parse_dates=["time"])
        last_date_on_file = df_full.time.max()
        last_date = last_date_on_file
        print("No last date specified. Choosing the last date on the data file.\n")
        day0 = df_full.time.min()
        print("No day zero provided. Choosing the earliest onset date on the data file\n")
    elif options.datafile is None:
        parser.error("Please provide a data file")
    else:
        datafile = options.datafile
        df_full = pd.read_csv(os.path.join(data_folder, datafile), parse_dates=["time"])

    last_date_on_file = df_full.time.max()
    location = options.location
    ifMPI = options.ifMPI
    output_folder = options.output_folder
    ifsmooth = options.ifsmooth
    niter = options.niter
    threads = options.threads
    if options.last_date is None:
        last_date = last_date_on_file
    else:
        last_date = pd.to_datetime(options.last_date)

    if options.day0 is None:
        day0 = df_full.time.min()
    else:
        day0 = pd.to_datetime(options.day0)
    estimate_gamma = options.estimate_gamma
    N = options.N
    T = options.T

    plot_folder = os.path.join(root_folder, output_folder)
    if not (os.path.exists(plot_folder)):
        os.system('mkdir %s' % plot_folder)

    if (not 'daily_confirm' in df_full.columns) and (not 'cum_confirm' in df_full.columns):
        raise ValueError("Please provide at least one of the following: daily_confirm, cum_confirm")
    elif (not 'daily_confirm' in df_full.columns) and ('cum_confirm' in df_full.columns):
        df_inf = df_full['cum_confirm'].diff().abs()
        df_inf[0] = df_full['cum_confirm'].iloc[0]
        df_full['daily_confirm'] = df_inf
    elif ('daily_confirm' in df_full.columns) and (not 'cum_confirm' in df_full.columns):
        df_full["cum_confirm"] = df_full.daily_confirm.cumsum()

    n_remove = (last_date_on_file - last_date).days
    print('Removing last %s days' % n_remove)
    df1 = df_full.drop(df_full.tail(n_remove).index)
    print("Using data till %s", df1["time"].max())

    n_remove = (day0 - df_full.time.min()).days
    print('Removing first %s days' % n_remove)
    df_main = df1.loc[n_remove:]
    print(df_main)

    today = pd.to_datetime('today')

    if ifsmooth:
        ## smoothing counts
        df_main["rolling_mean"] = df_main.daily_confirm.rolling(window=3).mean()
        df_main["rolling_mean"] = df_main.apply(lambda dd: dd.daily_confirm if np.isnan(dd.rolling_mean)
        else dd.rolling_mean, axis=1)
        print('Generating infection times by uniformly distributing throughout each day from smoothed daily counts\n')
        infection_data = list(
            i + rand.uniform() for i, y in enumerate(df_main['rolling_mean'].values) for z in range(y.astype(int)))
        df = pd.DataFrame(infection_data, index=range(len(infection_data)), columns=['infection'])
    else:
        print('Generating infection times by uniformly distributing throughout each day from actual daily counts\n')
        infection_data = list(
            i + rand.uniform() for i, y in enumerate(df_main['daily_confirm'].values) for z in range(y.astype(int)))
        df = pd.DataFrame(infection_data, index=range(len(infection_data)), columns=['infection'])

    if estimate_gamma:
        print('Generating recovery times by uniformly distributing throughout each day')
        if (not 'recovery' in df_main.columns) and (not 'deaths' in df_main.columns) \
                and (not 'cum_heal' in df_main.columns) and (not 'cum_dead' in df_main.columns):
            raise ValueError('Please provide at least one of the following: recovery, deaths, cum_heal, cum_dead.\n')
        elif ('recovery' in df_main.columns) and (not 'deaths' in df_main.columns) and (not 'cum_dead' in df_main.columns):
            recovery_data = list(
                i + rand.uniform() for i, y in enumerate(df_main['recovery'].values) for z in
                range(y.astype(int)))
            print('Using only recovery counts.\n')
        elif ('cum_heal' in df_main.columns) and (not 'deaths' in df_main.columns) and (not 'cum_dead' in df_main.columns):
            # get daily recovery counts
            df_cure = df_main['cum_heal'].diff().abs()
            df_cure[0] = df_main['cum_heal'].iloc[0]
            df_main['recovery'] = df_cure
            recovery_data = list(
                i + rand.uniform() for i, y in enumerate(df_main['recovery'].values) for z in
                range(y.astype(int)))
            print('Using only recovery counts.\n')
        elif (not 'recovery' in df_main.columns) and ('deaths' in df_main.columns) and (not 'cum_heal' in df_main.columns):
            recovery_data = list(
                i + rand.uniform() for i, y in enumerate(df_main['deaths'].values) for z in
                range(y.astype(int)))
            print('Using only death counts.\n')
        elif ('cum_dead' in df_main.columns) and (not 'recovery' in df_main.columns) and (not 'cum_heal' in df_main.columns):
            # get daily death counts
            df_death = df_main['cum_dead'].diff().abs()
            df_death[0] = df_main['cum_dead'].iloc[0]
            df_main['deaths'] = df_death
            recovery_data = list(
                i + rand.uniform() for i, y in enumerate(df_main['deaths'].values) for z in
                range(y.astype(int)))
            print('Using only death counts.\n')
        else:
            recovery_data = list(
                i + rand.uniform() for i, y in enumerate(df_main['recovery'].values + df_main['deaths'].values) for z in
                range(y.astype(int)))
            print('Using both recovery and death counts.\n')
        df_recovery = pd.DataFrame(recovery_data, index=range(len(recovery_data)), columns=['recovery'])


    bounds = [(0.1, 2.0), (0.1, 2.0), (1e-9, 1e-3)]

    dsaobj = DSA(df=df, bounds=bounds)
    dsaobj.fit(N=N, laplace=False)
    if ifMPI:
        dsaobj.simulate_and_fit_parallel(N=N, n=niter, laplace=False)
    else:
        dsaobj.simulate_and_fit(N=N, n=niter, laplace=False)
    dsaobj.summary()

    ## posterior histograms
    figa, figb, figc, figR0, figrho, fign, figsT, figkinfty, figsinfty, figsinvrho = dsaobj.get_histograms()
    fname = location + 'posterior_hist_a_' + today.strftime("%m%d")
    fig_save(figa, plot_folder, fname)
    fname = location + 'posterior_hist_b_' + today.strftime("%m%d")
    fig_save(figb, plot_folder, fname)
    fname = location + 'posterior_hist_c_' + today.strftime("%m%d")
    fig_save(figc, plot_folder, fname)
    fname = location + 'posterior_hist_R0_' + today.strftime("%m%d")
    fig_save(figR0, plot_folder, fname)
    fname = location + 'posterior_hist_rho_' + today.strftime("%m%d")
    fig_save(figrho, plot_folder, fname)
    fname = location + 'posterior_hist_n_' + today.strftime("%m%d")
    fig_save(fign, plot_folder, fname)
    fname = location + 'posterior_hist_sT_' + today.strftime("%m%d")
    fig_save(figsT, plot_folder, fname)
    fname = location + 'posterior_hist_kinfty_' + today.strftime("%m%d")
    fig_save(figkinfty, plot_folder, fname)
    fname = location + 'posterior_hist_sinfty_' + today.strftime("%m%d")
    fig_save(figsinfty, plot_folder, fname)
    fname = location + 'posterior_hist_invrho_' + today.strftime("%m%d")
    fig_save(figsinvrho, plot_folder, fname)

    m = dsaobj.theta
    cov = dsaobj.cov_abr()
    nSim = 1000
    samples = parm_sample_correlated(m, cov, nSim)

    nDays = T
    dates = pd.DataFrame({'d': [day0 + pd.DateOffset(i) for i in np.arange(nDays)]})

    fig_a, fig_b, predictions = dsaobj.predict(samples, df=df_main, dates=dates, n0=min(df_main.cum_confirm),
                                               d0=df_main.daily_confirm.iloc[0],
                                               theta=dsaobj.theta)
    fname = location + 'predictions_' + today.strftime("%m%d")
    fig_save(fig_a, plot_folder, fname)
    fname = location + 'predictions_daily_new' + today.strftime("%m%d")
    fig_save(fig_b, plot_folder, fname)
    fname = location + 'predictions_' + today.strftime("%m%d") + '.csv'
    predictions.to_csv(os.path.join(plot_folder, fname), index=False)
    print('Predictions done.\n')

    fig_density = dsaobj.plot_density_fit_posterior(samples)
    fname = location + 'Tfinaldensity' + today.strftime("%m%d")
    fig_save(fig_density, plot_folder, fname)
    print('Density estimation done.\n')

    if estimate_gamma:
        dsaobj.estimate_gamma(df_recovery=df_recovery, N=N, x0=(0.1, -5),
                          bounds=[(1.0 / 25, 1.0 / 5), (-10, 0)], approach='offset')
        if ifMPI:
            pickle.dump(df_recovery, open("df_recovery", "wb"), protocol=3)
            fname = location + '_dsa_epi_' + today.strftime("%m%d") + '.pkl'
            with open(os.path.join(plot_folder, fname), 'wb') as output:  # Overwrites any existing file.
                pickle.dump(dsaobj, output, protocol=pickle.HIGHEST_PROTOCOL)
            commandstr = "mpiexec -n " + str(threads) + " python estimate_gamma_parallel.py -d " + fname + " -o " + output_folder + " -l " + location
            os.system(commandstr)

            fname = location + '_gammas_fitted_' + today.strftime("%m%d") + '.csv'
            gammas_fitted = pd.read_csv(os.path.join(plot_folder, fname))

            fname = location + 'posterior_hist_gamma_' + today.strftime("%m%d")
            fig_g = plt.figure()
            plt.hist(gammas_fitted.gamma.values, bins=50, density=True,
                 color=cyans['cyan3'].get_rgb())
            plt.xlabel('$\\gamma$')
            plt.ylabel('Density')
            sns.despine()
            fig_save(fig_g, plot_folder, fname)
        print('Estimation of recovery parameters done.\n')

    fname = location + '_dsa_epi_' + today.strftime("%m%d") + '.pkl'
    with open(os.path.join(plot_folder, fname), 'wb') as output:  # Overwrites any existing file.
        pickle.dump(dsaobj, output, protocol=pickle.HIGHEST_PROTOCOL)

    fname = os.path.join(plot_folder, location + '_fit_summary.tex')
    dsaobj.summary(ifSave=True, fname=fname)



if __name__ == "__main__":
    main()



