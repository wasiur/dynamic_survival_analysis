## Import packages
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import scipy as sc
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import stats
from scipy.stats import gamma, beta, norm, uniform, expon, gaussian_kde
from scipy.integrate import cumtrapz
from scipy.optimize import basinhopping, dual_annealing, shgo, differential_evolution
from tabulate import tabulate
from scipy.special import lambertw
from numpy.random import RandomState
rand = RandomState()

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import pystan

from mycolours import *
def my_plot_configs():
    plt.style.use('seaborn-bright')
    plt.rcParams["figure.frameon"] = False
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['axes.labelweight'] = 'bold'



def fig_save(fig, Plot_Folder, fname):
    # fig.savefig(os.path.join (Plot_Folder, fname),dpi=300)
    fig.savefig(os.path.join (Plot_Folder, fname + "." + 'pdf'),
                 format='pdf', transparent=True)


def euler1d(odefun, endpoints, ic=1.0):
    timepoints = np.linspace(endpoints[0], endpoints[1], 1000)
    stepsize = timepoints[1] - timepoints[0]
    sol = np.zeros(len(timepoints), dtype=np.float64)
    sol[0] = ic
    for i in range(1,len(timepoints)):
        t = timepoints[i-1]
        y = sol[i-1]
        sol[i] = sol[i-1] + np.float64(odefun(t,y)) * stepsize
    return timepoints, sol


class DSA():
    def __init__(self, df=None, a=0.4, b=0.6, rho=1e-6, parent=None, **kwargs):
        self.df = df
        self.a = a
        self.b = b
        self.rho = rho
        self.parent = parent
        self.T = np.ceil(self.df['infection'].max())
        if kwargs.get('bounds') is None:
            self.bounda = (0.1, 1.0)
            self.boundb = (0.1, 1.0)
            self.boundrho = (1e-9, 1e-3)
        else:
            self.bounda, self.boundb, self.boundrho = kwargs.get('bounds')
        self.bounds = [self.bounda, self.boundb, self.boundrho]
        if kwargs.get('priordist') is None:
            a_prior = uniform(loc=self.bounda[0],
                              scale=self.bounda[1] - self.bounda[0])
            b_prior = uniform(loc=self.boundb[0],
                              scale=self.boundb[1] - self.boundb[0])
            r_prior = uniform(loc=self.boundrho[0],
                              scale=self.boundrho[1] - self.boundrho[0])
            self.priordist = [a_prior, b_prior, r_prior]
        else:
            self.priordist = kwargs.get('priordist')

        # if kwargs.get('proposaldist') is None:
        #     self.proposaldist = lambda theta: self.priordist
        # else:
        #     self.proposaldist = kwargs.get('proposaldist')

        if kwargs.get('timepoints') is None:
            self.timepoints = np.linspace(0.0, self.T, 1000)
        else:
            self.timepoints = kwargs.get('timepoints')
        self.fits = None
        self.g = None
        self.offset = None
        self.data = None
        self.bounds_gamma = None
        if kwargs.get('mh_chains') is None:
            self.mh_chains = None
        else:
            self.mh_chains = kwargs.get('mh_chains')
        # if kwargs.get('stan_model') is None:
        #     self.stan_model = None
        # else:
        #     self.stan_model = kwargs.get('stan_model')
        # if kwargs.get('stan_fit') is None:
        #     self.stan_fit = None
        # else:
        #     self.stan_fit = kwargs.get('stan_fit')

    def proposaldist(self, theta):
        return self.priordist

    @classmethod
    def poisson_ode_fun(cls,t, S, a, b, rho):
        dsdt = [np.float64(- a * S * np.log(S, dtype=np.float64) - b * (S - S * S) - b * rho * S)]
        return dsdt

    @classmethod
    def data_likelihood(cls, df, theta):
        a, b, rho = theta
        S0 = [1.0]
        T = np.max(df.values)
        odefun = lambda t, S: DSA.poisson_ode_fun(t, S, a=a, b=b, rho=rho)
        timepoints, sol = euler1d(odefun=odefun, endpoints=[0.0, T])
        S = interp1d(timepoints, sol)
        smax = S(T)
        factor = 1 - smax
        Z = 1
        j = 0
        for x in df.values:
            s = S(x)
            if s > 0:
                z = (a * s * np.log(s, dtype=np.float64) + b * (s - s ** 2) + b * rho * s) / factor
                Z *= z
        return Z

    @classmethod
    def data_loglikelihood(cls, df, theta):
        a, b, rho = theta
        S0 = [1.0]
        T = np.max(df.values)
        odefun = lambda t, S: DSA.poisson_ode_fun(t, S, a=a, b=b, rho=rho)
        timepoints, sol = euler1d(odefun=odefun, endpoints=[0.0, T])
        S = interp1d(timepoints, sol)
        smax = S(T)
        factor = 1 - smax
        Z = 0
        j = 0
        for x in df.values:
            s = S(x)
            if s > 0:
                z = (a * s * np.log(s, dtype=np.float64) + b * (s - s ** 2) + b * rho * s) / factor
                if z > 0:
                    Z += np.log(z, dtype=np.float64)
                    j += 1
        return Z

    @classmethod
    def prior_density(cls, p, theta):
        a, b, rho = theta
        res = p[0].pdf(a) * p[1].pdf(b) * p[2].pdf(rho)
        return res

    @classmethod
    def log_prior(cls, p,  theta):
        a, b, rho = theta
        res = p[0].logpdf(a) + p[1].logpdf(b) + p[2].logpdf(rho)
        # res = p[0].pdf(a) * p[1].pdf(b) * p[2].pdf(rho)
        if res > 0:
            return res
        else:
            return -np.inf

    @classmethod
    def draw_from_prior(cls, p):
        return [p[0].rvs(), p[1].rvs(), p[2].rvs()]

    @classmethod
    def posterior_likelihood(cls, df, priordist, theta):
        dl = DSA.data_likelihood(df, theta=theta)
        pl = DSA.prior_density(p=priordist, theta=theta)
        return dl * pl

    @classmethod
    def neg_log_posterior(cls, df, priordist, theta):
        dl = DSA.data_loglikelihood(df, theta=theta)
        lp = DSA.log_prior(p=priordist, theta=theta)
        res = - dl - lp
        return res

    @classmethod
    def log_posterior(cls, df, priordist, theta):
        dl = DSA.data_loglikelihood(df, theta=theta)
        lp = DSA.log_prior(p=priordist, theta=theta)
        res = dl + lp
        return res

    @classmethod
    def draw_proposal(cls, proposal):
        ss = [proposal[0].rvs(), proposal[1].rvs(), proposal[2].rvs()]
        return ss

    @classmethod
    def proposal_loglikelihood(cls, proposal, theta):
        ss = [proposal[0].logpdf(theta[0]),
              proposal[1].logpdf(theta[1]),
              proposal[2].logpdf(theta[2])]
        return np.sum(ss)

    @classmethod
    def proposal_likelihood(cls, proposal, theta):
        ss = [proposal[0].pdf(theta[0]),
              proposal[1].pdf(theta[1]),
              proposal[2].pdf(theta[2])]
        return ss[0]*ss[1]*ss[2]

    @classmethod
    def accept_prob(cls, df, priordist, proposal, theta1, theta2):
        l1 = DSA.posterior_likelihood(df, priordist, theta1)
        l2 = DSA.posterior_likelihood(df, priordist, theta2)
        g21 = proposal(theta = theta2)
        p21 = DSA.proposal_likelihood(proposal=g21, theta=theta1)
        g12 = proposal(theta = theta1)
        p12 = DSA.proposal_likelihood(proposal=g12, theta=theta2)
        if l1*p12 > 0:
            a = min(1.0, (l2*p21)/(l1*p12))
        else:
            a = 1.0
        return a

    @classmethod
    def mh(cls, df, p, proposal, burnin=10 ** 3, chain_length=10 ** 4, thinning=10):
        chain = np.zeros((chain_length, 3), dtype=np.float)
        chain[0] = DSA.draw_from_prior(p)
        for i in np.arange(1, chain_length):
            current_state = chain[i-1]
            prop = proposal(theta=current_state)
            new = DSA.draw_proposal(prop)
            a = DSA.accept_prob(df=df, priordist=p, proposal=proposal, theta1=current_state, theta2=new)
            if uniform.rvs() < a:
                chain[i] = new
            else:
                chain[i] = chain[i - 1]
        after_burnin = chain[burnin:chain_length - 1]
        idx = np.arange(0, chain_length - burnin, thinning)
        return after_burnin[idx]


    @property
    def c(self):
        return self.b * self.rho

    @property
    def R0(self):
        return self.b / self.a

    @property
    def invrho(self):
        return 1.0/self.rho

    @property
    def delta(self):
        if self.g is not None:
            return self.a - self.g
        else:
            return 0

    @property
    def tau(self):
        return ((self.R0 + lambertw(-self.R0 * np.exp(-self.R0 * (1 + self.rho)))) / self.R0).real

    @property
    def kT(self):
        if self.parent is None:
            return self.df['infection'].shape[0]
        else:
            return self.parent.kT

    @property
    def rescale(self):
        return 1 - self.S(self.T)

    @property
    def n(self):
        return self.kT / self.rescale

    @property
    def sT(self):
        return self.n - self.kT

    @property
    def kinfty(self):
        return self.tau * self.n

    @property
    def sinfty(self):
        return (1 - self.tau) * self.n

    @property
    def theta(self):
        return [self.a, self.b, self.rho]

    @property
    def S(self):
        a, b, rho = self.theta
        odefun = lambda t, S: DSA.poisson_ode_fun(t, S, a=a, b=b, rho=rho)
        timepoints, sol = euler1d(odefun=odefun, endpoints=[0.0, self.T])
        S = interp1d(timepoints, sol)
        return S

    def var_T(self):
        if self.fits is not None:
            return np.var(list(f.T for f in self.fits))
        else:
            return 0.0

    def var_a(self):
        if self.fits is not None:
            return np.var(list(f.a for f in self.fits))
        else:
            return 0.0

    def var_b(self):
        if self.fits is not None:
            return np.var(list(f.b for f in self.fits))
        else:
            return 0.0

    def var_c(self):
        if self.fits is not None:
            return np.var(list(f.c for f in self.fits))
        else:
            return 0.0

    def cov_abc(self):
        if self.fits is not None:
            fitted_parms = np.zeros((len(self.theta), len(self.fits)), dtype=np.float64)
            fitted_parms[0] = list(f.a for f in self.fits)
            fitted_parms[1] = list(f.b for f in self.fits)
            fitted_parms[2] = list(f.c for f in self.fits)
            return np.cov(fitted_parms)
        else:
            return np.zeros((len(self.theta), len(self.fits)), dtype=np.float64)

    def cov_abr(self):
        if self.fits is not None:
            fitted_parms = np.zeros((len(self.theta), len(self.fits)), dtype=np.float64)
            fitted_parms[0] = list(f.a for f in self.fits)
            fitted_parms[1] = list(f.b for f in self.fits)
            fitted_parms[2] = list(f.rho for f in self.fits)
            return np.cov(fitted_parms)
        else:
            return np.zeros((len(self.theta), len(self.fits)), dtype=np.float64)

    def var_R0(self):
        if self.fits is not None:
            return np.var(list(f.R0 for f in self.fits))
        else:
            return 0.0

    def var_rho(self):
        if self.fits is not None:
            return np.var(list(f.rho for f in self.fits))
        else:
            return 0.0

    def var_invrho(self):
        if self.fits is not None:
            return np.var(list(f.invrho for f in self.fits))
        else:
            return 0.0

    def var_tau(self):
        if self.fits is not None:
            return np.var(list(f.tau for f in self.fits))
        else:
            return 0.0

    def var_n(self):
        if self.fits is not None:
            return np.var(list(f.n for f in self.fits))
        else:
            return 0.0

    def var_kT(self):
        if self.fits is not None:
            return np.var(list(f.kT for f in self.fits))
        else:
            return 0.0

    def var_sT(self):
        if self.fits is not None:
            return np.var(list(f.sT for f in self.fits))
        else:
            return 0.0

    def var_kinfty(self):
        if self.fits is not None:
            return np.var(list(f.kinfty for f in self.fits))
        else:
            return 0.0

    def var_sinfty(self):
        if self.fits is not None:
            return np.var(list(f.sinfty for f in self.fits))
        else:
            return 0.0

    def mean_T(self):
        if self.fits is not None:
            return np.mean(list(f.T for f in self.fits))
        else:
            return 0.0

    def mean_a(self):
        if self.fits is not None:
            return np.mean(list(f.a for f in self.fits))
        else:
            return 0.0

    def mean_b(self):
        if self.fits is not None:
            return np.mean(list(f.b for f in self.fits))
        else:
            return 0.0

    def mean_c(self):
        if self.fits is not None:
            return np.mean(list(f.c for f in self.fits))
        else:
            return 0.0

    def mean_R0(self):
        if self.fits is not None:
            return np.mean(list(f.R0 for f in self.fits))
        else:
            return self.R0

    def mean_rho(self):
        if self.fits is not None:
            return np.mean(list(f.rho for f in self.fits))
        else:
            return self.rho

    def mean_invrho(self):
        if self.fits is not None:
            return np.mean(list(f.invrho for f in self.fits))
        else:
            return self.invrho

    def mean_tau(self):
        if self.fits is not None:
            return np.mean(list(f.tau for f in self.fits))
        else:
            return self.tau

    def mean_n(self):
        if self.fits is not None:
            return np.mean(list(f.n for f in self.fits))
        else:
            return self.n

    def mean_kT(self):
        if self.fits is not None:
            return np.mean(list(f.kT for f in self.fits))
        else:
            return self.kT

    def mean_sT(self):
        if self.fits is not None:
            return np.mean(list(f.sT for f in self.fits))
        else:
            return self.sT

    def mean_kinfty(self):
        if self.fits is not None:
            return np.mean(list(f.kinfty for f in self.fits))
        else:
            return self.kinfty

    def mean_sinfty(self):
        if self.fits is not None:
            return np.mean(list(f.sinfty for f in self.fits))
        else:
            return self.sinfty

    def fit_till_success(self, objfun, x0, bounds,
                         ifglobal=True):
        if ifglobal:
            flag = True
            while flag:
                minobj = differential_evolution(objfun,
                              bounds=bounds)
                if minobj.success:
                    flag = False
                else:
                    x0 = DSA.draw_from_prior(p=self.priordist)
        else:
            flag = True
            while flag:
                minobj = minimize(objfun, x0=tuple(x0),
                              bounds=bounds,
                              options={'disp': False})
                if minobj.success:
                    flag = False
                else:
                    x0 = DSA.draw_from_prior(p=self.priordist)
        return minobj

    def fit(self, N=None, laplace=True, summary=False,
            ifSave=False, fname='summary.tex'):
        if N is None:
            self.data = self.df['infection']
        else:
            self.data = self.df['infection'].sample(N, replace=True)

        x0 = self.theta
        bounds = [self.bounda, self.boundb, self.boundrho]
        if laplace:
            objfun = lambda theta: DSA.neg_log_posterior(df=self.data, priordist=self.priordist, theta=theta)
            minobj = self.fit_till_success(objfun, x0=x0, bounds=bounds, ifglobal=True)
        else:
            objfun = lambda theta: -DSA.data_loglikelihood(self.data, theta=theta)
            minobj = self.fit_till_success(objfun, x0=x0, bounds=bounds, ifglobal=True)

        self.a, self.b, self.rho = minobj.x
        # print summary
        if summary:
            print(minobj)
            self.summary()
        # self.summary()
        if ifSave:
            self.summary(ifSave=True, fname=fname)
        return self

    def simulate_from_model(self, N):
        '''
        simulate N infections from the survival curve of the fitted model using inverse sampling
        '''
        unis = rand.uniform(low=self.S(self.T), high=1, size=N)
        vals = np.linspace(0, self.T, 10000)
        Ss = np.asarray(list(self.S(x) for x in vals))
        values = []
        for u in unis:
            i = np.argmax(Ss - u < 0)
            if i > 0 and i < vals.size - 1:
                '''
                point slope form to find zero
                '''
                y0 = Ss[i - 1] - u
                y1 = Ss[i] - u
                x0 = vals[i - 1]
                x1 = vals[i]
                m = (y1 - y0) / (x1 - x0)
                v = (m * x1 - y1) / m
            else:
                v = vals[i]
            values.append(v)
        df = pd.DataFrame(values, index=range(N), columns=['infection'])
        return df

    def simulate_and_fit(self, N, n, laplace=True):
        '''
        simulate N data from the model n times and refit
        '''
        fits = []
        for i in range(1, n + 1):
            df = self.simulate_from_model(N)
            bounds = [self.bounda, self.boundb, self.boundrho]
            a, b, rho = DSA.draw_from_prior(self.priordist)
            fit = DSA(df=df, a=a, b=b, rho=rho,
                      bounds=bounds, priordist=self.priordist,
                      parent=self).fit(laplace=laplace)
            fits.append(fit)
        self.fits = fits
        return self

    def simulate_and_fit_parallel(self, N, n, laplace=True, threads=40):
        epidemic_obj_arr = []
        total_time = 0
        for i in range(1, n + 1):
            # print("sample", i // 100)
            df = self.simulate_from_model(N)
            bounds = [self.bounda, self.boundb, self.boundrho]
            a, b, rho = DSA.draw_from_prior(self.priordist)
            st = time.time()
            epidemic_obj = DSA(df=df, a=a, b=b, rho=rho,
                               bounds=bounds, priordist=self.priordist,
                               parent=self)
            total_time += time.time() - st
            epidemic_obj_arr.append(epidemic_obj)
        import pickle
        st = time.time()
        if os.path.exists("epidemic_objects_array_fitted"):
            os.remove("epidemic_objects_array_fitted")
        pickle.dump(epidemic_obj_arr, open("epidemic_objects_array", "wb"), protocol=3)
        commandstr = "mpiexec -n " + str(threads) + " python dsa_parallel_fitting.py " + str(laplace)
        os.system(commandstr)
        # os.system("mpiexec -n %s python dsa_parallel_fitting.py %s" % (threads, laplace))
        epidemic_objects_array_fitted = pickle.load(open("epidemic_objects_array_fitted", "rb"))
        print("Total fit time %f" % ((time.time() - st) / 60.0))
        self.fits = epidemic_objects_array_fitted
        return self

    def bayesian_fit(self, N=None, niter=5000, nchains=4):
        if N is None:
            self.data = self.df['infection']
            N = self.df.size
        else:
            self.data = self.df['infection'].sample(N, replace=False)

        Tmax = self.data.max()
        ordered_infection_data = self.data.sort_values().tolist()

        pystan_data = dict(N=N,
                           Tmax=Tmax,
                           infectiontimes=ordered_infection_data,
                           t0=0.0)

        sm = pystan.StanModel(file='DSA.stan')
        fit = sm.sampling(data=pystan_data, iter=niter, chains=nchains)
        print(fit)
        res_a = fit.extract()['a']
        res_b = fit.extract()['b']
        res_rho = fit.extract()['rho']
        mh_chains = pd.DataFrame({'a': res_a, 'b': res_b, 'rho': res_rho})
        self.mh_chains = mh_chains
        self.a = np.mean(res_a)
        self.b = np.mean(res_b)
        self.rho = np.mean(res_rho)
        # self.stan_model = sm
        # self.stan_fit = fit
        # temp = DSA.mh(df=self.data,
        #               p=self.priordist,
        #               proposal=self.proposaldist,
        #               burnin=burnin,
        #               chain_length=chain_length,
        #               thinning=thinning)
        # res_a = temp[:, 0]
        # res_b = temp[:, 1]
        # res_rho = temp[:, 2]
        # mh_chains = pd.DataFrame({'a': res_a, 'b': res_b, 'rho': res_rho})
        # self.mh_chains = mh_chains
        fits = []
        l = len(res_a)
        for i in range(l):
            a = res_a[i]
            b = res_b[i]
            rho = res_rho[i]
            fit = DSA(df=self.df, a=a, b=b, rho=rho,
                      bounds=self.bounds, priordist=self.priordist,
                      parent=self)
            fits.append(fit)

        self.fits = fits
        return sm, fit

    def trace_plot(self):
        if self.mh_chains is None:
            self.bayesian_fit()
        a_chain = self.mh_chains["a"]
        b_chain = self.mh_chains["b"]
        rho_chain = self.mh_chains["rho"]
        my_plot_configs()

        fig_a = plt.figure(frameon=False)
        plt.plot(a_chain, color=cyans['cyan3'].get_rgb())
        plt.ylabel('$a$')
        sns.despine()
        # ax = plt.gca()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)

        fig_b = plt.figure(frameon=False)
        plt.plot(b_chain, color=cyans['cyan3'].get_rgb())
        plt.ylabel('$b$')
        sns.despine()
        # ax = plt.gca()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)

        fig_c = plt.figure(frameon=False)
        plt.plot(rho_chain, color=cyans['cyan3'].get_rgb())
        plt.ylabel('$\\rho$')
        sns.despine()
        # ax = plt.gca()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)

        return fig_a, fig_b, fig_c

    def add_fits(self, samples):
        fits = []
        l = np.size(samples, axis=0)
        for i in range(l):
            a, b, rho = samples[i]
            fit = DSA(df=self.df, a=a, b=b, rho=rho,
                      bounds=self.bounds, priordist=self.priordist,
                      parent=self)
            fits.append(fit)
        self.fits = fits
        return self

    def get_histograms(self):

        if self.fits is None:
            raise ValueError('\n Please run the inference model first.\n')

        my_plot_configs()
        figa = plt.figure()
        plt.hist(list(f.a for f in self.fits),
                 bins=50, density=True,
                 color=cyans['cyan3'].get_rgb()
                 )
        plt.xlabel('$a$')
        plt.ylabel('Density')
        sns.despine()
        # plt.title('$a$')

        figb = plt.figure()
        plt.hist(list(f.b for f in self.fits),
                 bins=50, density=True,
                 color=cyans['cyan3'].get_rgb()
                 )
        plt.xlabel('$b$')
        plt.ylabel('Density')
        sns.despine()
        # plt.title('$b$')

        figc = plt.figure()
        plt.hist(list(f.c for f in self.fits),
                 bins=50, density=True,
                 color=cyans['cyan3'].get_rgb()
                 )
        plt.xlabel('$c$')
        plt.ylabel('Density')
        sns.despine()
        # plt.title('$c$')

        figR0 = plt.figure()
        plt.hist(list(f.R0 for f in self.fits),
                 bins=50, density=True,
                 color=cyans['cyan3'].get_rgb()
                 )
        plt.xlabel('$R_0$')
        plt.ylabel('Density')
        sns.despine()
        # plt.title('$R_0$')

        figrho = plt.figure()
        plt.hist(list(f.rho for f in self.fits),
                 bins=50, density=True,
                 color=cyans['cyan3'].get_rgb()
                 )
        plt.xlabel('$\\rho$')
        plt.ylabel('Density')
        sns.despine()
        # plt.title('rho')

        fign, ax = plt.subplots()
        plt.hist(list(f.n for f in self.fits),
                 bins=50, density=True,
                 color=cyans['cyan3'].get_rgb()
                 )
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xlabel('$n$')
        plt.ylabel('Density')
        sns.despine()

        figsT, ax = plt.subplots()
        plt.hist(list(f.sT for f in self.fits),
                 bins=50, density=True,
                 color=cyans['cyan3'].get_rgb()
                 )
        # plt.title('$s_T$')
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xlabel('$s_T$')
        plt.ylabel('Density')
        sns.despine()

        figkinfty, ax = plt.subplots()
        plt.hist(list(f.kinfty for f in self.fits),
                 bins=50, density=True,
                 color=cyans['cyan3'].get_rgb()
                 )
        # plt.title('$k_\infty$')
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xlabel('$k_\infty$')
        plt.ylabel('Density')
        sns.despine()

        figsinfty, ax = plt.subplots()
        plt.hist(list(f.sinfty for f in self.fits),
                 bins=50, density=True,
                 color=cyans['cyan3'].get_rgb()
                 )
        plt.title('$s_\infty$')
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xlabel('$s_\infty$')
        plt.ylabel('Density')
        sns.despine()

        figsinvrho, ax = plt.subplots()
        plt.hist(list(f.invrho for f in self.fits),
                 bins=50, density=True,
                 color=cyans['cyan3'].get_rgb()
                 )
        # plt.title('1/rho')
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xlabel('$1/\\rho$')
        plt.ylabel('Density')
        sns.despine()

        return (figa, figb, figc, figR0, figrho, fign, figsT, figkinfty, figsinfty, figsinvrho)

    def posterior_histograms(self, samples):
        a_samples = samples[:,0]
        b_samples = samples[:,1]
        rho_samples = samples[:,2]
        my_plot_configs()

        fig_a = plt.figure()
        plt.hist(a_samples, bins=50, density=True,
                 color=cyans['cyan3'].get_rgb())
        plt.xlabel('$a$')
        plt.ylabel('Density')
        sns.despine()
        # ax = plt.gca()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)

        fig_b = plt.figure()
        plt.hist(b_samples, bins=50, density=True,
                 color=cyans['cyan3'].get_rgb())
        plt.xlabel('$b$')
        plt.ylabel('Density')
        sns.despine()
        # ax = plt.gca()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)

        fig_c = plt.figure()
        plt.hist(rho_samples, bins=50, density=True,
                 color=cyans['cyan3'].get_rgb())
        plt.xlabel('$\\rho$')
        plt.ylabel('Density')
        sns.despine()
        # ax = plt.gca()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)

        return fig_a, fig_b, fig_c

    def summary(self, ifSave=False, fname=None):

        headers = ['Parameter', 'Name', 'MLE', 'Mean', 'StdErr']
        table = [['T', 'final time', self.T, None if self.fits is None else self.mean_T(),
                  None if self.fits is None else np.sqrt(self.var_T())],
                 ["a", 'beta+gamma+delta', self.a, None if self.fits is None else self.mean_a(),
                  None if self.fits is None else np.sqrt(self.var_a())],
                 ["b", "beta*mu", self.b, None if self.fits is None else self.mean_b(),
                  None if self.fits is None else np.sqrt(self.var_b())],
                 ["c", "beta*mu*rho", self.c, None if self.fits is None else self.mean_c(),
                  None if self.fits is None else np.sqrt(self.var_c())],
                 ['R0', "R-naught", self.R0, None if self.fits is None else self.mean_R0(),
                  None if self.fits is None else np.sqrt(self.var_R0())],
                 ['rho', "initial fraction I", self.rho, None if self.fits is None else self.mean_rho(),
                  None if self.fits is None else np.sqrt(self.var_rho())],
                 ['tau', "epidemic size", self.tau, None if self.fits is None else self.mean_tau(),
                  None if self.fits is None else np.sqrt(self.var_tau())],
                 ['1-S(T)', "rescaling", self.rescale, None, None],
                 ['n', "#S+#I", self.n, None if self.fits is None else self.mean_n(),
                  None if self.fits is None else np.sqrt(self.var_n())],
                 ['kT', "#I(T)", self.kT, None if self.fits is None else self.mean_kT(),
                  None if self.fits is None else np.sqrt(self.var_kT())],
                 ['sT', "#S(T)", self.sT, None if self.fits is None else self.mean_sT(),
                  None if self.fits is None else np.sqrt(self.var_sT())],
                 ['kinfty', "#I(infty)", self.kinfty, None if self.fits is None else self.mean_kinfty(),
                  None if self.fits is None else np.sqrt(self.var_kinfty())],
                 ['sinfty', '#S(infty)', self.sinfty, None if self.fits is None else self.mean_sinfty(),
                  None if self.fits is None else np.sqrt(self.var_sinfty())],
                 ['1/rho', 'initial total population', self.invrho, None if self.fits is None else self.mean_invrho(),
                  None if self.fits is None else np.sqrt(self.var_invrho())],
                 ['gamma', 'recovery rate', None if self.g is None else self.g, None, None],
                 ['offset', 'shift parameter', None if self.offset is None else self.offset, None, None]]
        print(tabulate(table, headers=headers))

        # print(tabulate(table, headers=headers, tablefmt="html"))

        # print(tabulate(table,headers=headers,tablefmt="latex_booktabs"))

        if ifSave:
            str1 = '\\documentclass{article}\n \\usepackage{booktabs} \n \\begin{document}'
            str2 = '\\end{document}'
            if fname == None:
                fname = 'summary.tex'
            with open(fname, 'w') as outputfile:
                outputfile.write(str1 + tabulate(table, headers=headers, tablefmt="latex_booktabs") + str2)
        return self

    def predict(self, samples, df, dates, n0=1, d0=0, theta=None):
        nSamples = np.size(samples, axis=0)
        nDays = len(dates)
        time_points = np.arange(nDays)
        mean = np.zeros((nSamples, nDays), dtype=np.float)
        mean_daily = np.zeros((nSamples, nDays), dtype=np.float)
        if theta is not None:
            theta = np.mean(samples, axis=0)
        # theta = np.mean(samples, axis=0)
        # n = self.n
        my_plot_configs()
        fig_a = plt.figure()
        for i in range(nSamples):
            a, b, rho = samples[i]
            epi = DSA(df=self.df, a=a, b=b, rho=rho)
            n = epi.n
            odefun = lambda t, S: DSA.poisson_ode_fun(t, S, a=a, b=b, rho=rho)
            t, sol = euler1d(odefun=odefun, endpoints=[0.0, nDays + 1])
            S = interp1d(t, sol)
            mean[i] = np.asarray(list(n * (1 - S(x)) + n0 for x in time_points))
            # mean[i][0] = 1
            mean_daily[i] = np.append(d0, np.diff(mean[i]))
            l1, = plt.plot(dates['d'].dt.date, mean[i], '-', color=cyans['cyan1'].get_rgb(), lw=1, alpha=0.05)

        m_ = np.int64(np.ceil(np.mean(mean, axis=0)))
        l = np.int64(np.ceil(np.quantile(mean, q=0.025, axis=0)))
        h = np.int64(np.ceil(np.quantile(mean, q=0.975, axis=0)))

        n = self.n
        a, b, rho = theta
        odefun = lambda t, S: DSA.poisson_ode_fun(t, S, a=self.a, b=self.b, rho=self.rho)
        t, sol = euler1d(odefun=odefun, endpoints=[0.0, nDays + 1])
        S = interp1d(t, sol)
        m = np.asarray(list(n * (1 - S(x)) + n0 for x in time_points))

        l2 = plt.plot(dates['d'].dt.date, m, '-', color=cyans['cyan5'].get_rgb(), lw=3)
        #     l2_, = plt.plot(dates['d'].dt.date, m_, '-', color=myColours['tud1d'].get_rgb(), lw=3, label='With mitigation')
        l3 = plt.plot(dates['d'].dt.date, l, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l4 = plt.plot(dates['d'].dt.date, h, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l5 = plt.fill_between(dates['d'].dt.date, l, h, alpha=.1, color=cyans['cyan1'].get_rgb())

        # l6 = plt.axvline(x=df['time'].max(), color=blues['blue1'].get_rgb(), linestyle='--')
        l6 = plt.axvline(x=df['time'][self.T - 1], color=blues['blue1'].get_rgb(), linestyle='--')
        l7 = plt.plot(df['time'].values, df['cum_confirm'].values, '-', color=maroons['maroon3'].get_rgb(),
                      lw=2)
        plt.xlabel('Dates')
        plt.ylabel('Cumulative infections')
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()

        fig_b = plt.figure()
        for i in range(nSamples):
            # l1nd = plt.plot(dates['d'].dt.date, mean_nd_daily[i], '-', color=myColours['tud7b'].get_rgb(), lw=1, alpha=0.05)
            l1 = plt.plot(dates['d'].dt.date, mean_daily[i], '-', color=cyans['cyan3'].get_rgb(), lw=1, alpha=0.05)
        m_daily = np.append(m[0], np.diff(m))
        # m_nd_daily = np.append(m_nd[0], np.diff(m_nd))
        # l2nd, = plt.plot(dates['d'].dt.date, m_nd_daily, '-', color=myColours['tud7d'].get_rgb(), lw=3, label='Without mitigation')
        l2, = plt.plot(dates['d'].dt.date, m_daily, '-', color=cyans['cyan5'].get_rgb(), lw=3,
                       label='With mitigation')
        # l6 = plt.axvline(x=df['time'].max(), color=myColours['tud7d'].get_rgb(), linestyle='--')
        l6 = plt.axvline(x=df['time'][self.T - 1], color=maroons['maroon3'].get_rgb(), linestyle='-')
        # l7 = plt.plot(df_ohio['time'].values, df_ohio['daily_confirm'].values, color=myColours['tud11d'].get_rgb(), lw=3)
        plt.ylabel('Daily new infections')
        plt.xlabel('Dates')
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()
        # plt.legend(handles=[l2nd, l2])
        # fig_save(fig, Plot_Folder, fname_)
        # m[0] = 1
        # m_nd[0] = 1

        my_dict = {}
        my_dict['Dates'] = dates['d']
        my_dict['Mean'] = m
        my_dict['High'] = h
        my_dict['Low'] = l
        my_dict = pd.DataFrame(my_dict)
        # my_dict.to_csv(os.path.join(Plot_Folder, fname + '.csv'), index=False)
        return fig_a, fig_b, my_dict

    def compute_density(self, theta):
        a, b, rho = theta
        odefun = lambda t, S: DSA.poisson_ode_fun(t, S, a=a, b=b, rho=rho)
        t, sol = euler1d(odefun=odefun, endpoints=[0.0, self.T + 1])
        S = interp1d(t, sol)
        out = []
        ST = S(self.T)
        for x in self.timepoints:
            Sx = S(x)
            out.append((a * Sx * np.log(Sx) + b * (Sx - Sx ** 2) + b * rho * Sx) / (1 - ST))
        return out

    def plot_density_fit_posterior(self, samples):
        nSamples = np.size(samples, axis=0)
        Ds = np.zeros((nSamples, len(self.timepoints)), dtype=np.float)
        for idx in range(nSamples):
            Ds[idx] = self.compute_density(samples[idx])
        Dslow = np.quantile(Ds, q=0.025, axis=0)
        Dshigh = np.quantile(Ds, q=0.975, axis=0)
        Dmean = np.mean(Ds, axis=0)
        fig = plt.figure()
        plt.plot(self.timepoints, Dmean, '-', color=cyans['cyan3'].get_rgb(), lw=3)
        plt.plot(self.timepoints, Dslow, '--', color=cyans['cyan3'].get_rgb(), lw=1)
        plt.plot(self.timepoints, Dshigh, '--', color=cyans['cyan3'].get_rgb(), lw=1)
#        plt.axvline(x=self.T, color=junglegreen['green3'].get_rgb(), linestyle='-')

        mirrored_data = (2 * self.T - self.df['infection'].values).tolist()
        combined_data = self.df['infection'].values.tolist() + mirrored_data
        dense = gaussian_kde(combined_data)
        denseval = list(dense(x) * 2 for x in self.timepoints)
        plt.plot(self.timepoints, denseval, '-', color=maroons['maroon3'].get_rgb(), lw=3)
        plt.fill_between(self.timepoints, Dslow, Dshigh, alpha=.3, color=maroons['maroon3'].get_rgb())
        # plt.legend()
        plt.ylabel('$-S_t/(1-S_T)$')
        plt.xlabel('t')
        c = cumtrapz(Dmean, self.timepoints)
        ind = np.argmax(c >= 0.001)
        plt.xlim((self.timepoints[ind], self.timepoints[-1] + 1))
        sns.despine()
        return fig

    # density
    def density_nuA(self, u):
        S = self.Scure
        s = S(u)
        ST = S(self.Tcure)
        return (self.a * s * np.log(s) + self.b * (s - s ** 2) + self.b * self.rho * s) / (1 - ST)

    # density
    def density_gamma(self, t, gamma):
        return expon.pdf(t, scale=1 / gamma)

    # integral nu_A Q
    def int_nuAQ(self, t, gamma):
        l = int(max(np.ceil(100.0 / (self.T / t)), 2))
        ttt = np.linspace(0, t, l)
        integrand = []
        for u in ttt:
            result = self.density_nuA(u) * self.density_gamma(t - u, gamma=gamma)  # expon.pdf(t-u,scale=1/gamma)
            integrand.append(result)
        integrand = np.asarray(integrand)
        return np.trapz(integrand, ttt)

    # improper mixture recovery density
    def density_recovery(self, t, gamma):
        t = max(t, self.Tcure / 100)
        return (1.0 / (1 + self.rho) * self.int_nuAQ(t, gamma) + self.rho / (1 + self.rho) * self.density_gamma(t,
                                                                                                                gamma))

    # normalization of the improper mixture density
    def proper(self, gamma, offset):
        s = np.linspace(self.Tcure / 100, self.Tcure, 100).tolist()
        vals = [0] + list(self.density_recovery(u + offset, gamma) for u in s)
        s = [0] + s
        A = np.trapz(vals, s)
        return A

    def proper_density_recovery(self, t, gamma, prop=None, offset=0):
        if prop is None:
            return self.density_recovery(t + offset, gamma) / self.proper(gamma, offset=offset)
        else:
            return self.density_recovery(t + offset, gamma) / prop

    def negloglikelihood_gammaoffset(self, theta):
        gamma, offset = theta
        bound_g, bound_o = self.bounds_gamma
        if np.isclose(gamma, bound_g[0]) or np.isclose(gamma, bound_g[1]) or np.isclose(offset,
                                                                                        bound_o[0]) or np.isclose(
                offset, bound_o[1]):
            lk = 1E6
            # print("gamma=", gamma, "offset=", offset, "-LOGLIKE=", lk)
            return lk
        result = []
        prop = self.proper(gamma, offset=offset)
        for u in self.datacure:
            r = -np.log(self.proper_density_recovery(u, gamma, prop=prop, offset=offset))
            result.append(r)
        lk = np.sum(result)
        # print("gamma=", gamma, "offset=", offset, "-LOGLIKE=", lk)
        return lk

    def estimate_gamma(self, df_recovery, N, x0, bounds, approach='gamma'):
        gamma, offset = self.estimate_gamma_sample(self.theta, df_recovery, N, x0, bounds, approach=approach)
        self.g = gamma
        self.offset = offset
        return self

    def estimate_gamma_sample(self, sample, df_recovery, N, x0, bounds, approach='gamma'):
        S0 = [1.0]
        t = self.T
        if approach == 'offset':
            t += bounds[1][1]
        a, b, rho = sample
        odefun = lambda t, S: DSA.poisson_ode_fun(t, S, a=a, b=b, rho=rho)
        tt, sol = euler1d(odefun=odefun, endpoints=[0.0, t])
        S = interp1d(tt, sol)

        self.Scure = S
        self.Tcure = np.ceil(df_recovery['recovery'].max())
        self.datacure = df_recovery['recovery'].sample(N, replace=True).values

        if approach == 'offset':
            self.bounds_gamma = bounds
            gamma, offset = minimize(
                self.negloglikelihood_gammaoffset,
                x0=x0,
                bounds=self.bounds_gamma,
                options={'disp': False, 'maxiter': 3}
            ).x
            return gamma, offset
        elif approach == 'prior':
            self.bounds_gamma = bounds
            offset = 0
            alpha, beta = minimize(
                self.negloglikelihood_alphabeta,
                x0=x0,
                bounds=self.bounds_gamma,
                options={'disp': False, 'maxiter': 3}
            ).x
            gamma = alpha * beta
        elif approach == 'gamma':
            self.offset = 0
            self.bounds_gamma = bounds
            gamma, = minimize(
                self.negloglikelihood_gamma,
                x0=x0,
                bounds=self.bounds_gamma,
                options={'disp': False, 'maxiter': 3}
            ).x
        return gamma, offset


















