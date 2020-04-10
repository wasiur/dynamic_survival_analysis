import os as os
import pystan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import sys
from scipy.interpolate import interp1d, Rbf
from scipy.optimize import minimize, newton
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.special import lambertw
from scipy.integrate import cumtrapz
from numpy.random import RandomState
import matplotlib
import seaborn as sns
sns.set_style("whitegrid")
from scipy.stats import binom, norm, gaussian_kde, expon, gamma as Gamma
import matplotlib; # matplotlib.use('TkAgg')
from tabulate import tabulate
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

from tudColours import *

rand = RandomState()
THREADS = 40

def fig_save(fig, Plot_Folder, fname):
    fig.savefig(os.path.join (Plot_Folder, fname),dpi=300)
    fig.savefig (os.path.join (Plot_Folder, fname + "." + 'pdf'), format='pdf')


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

class Epidemic(object):

	@classmethod 
	def Plot_Data(cls,t,inf_count,death_count,cure_count,daily_confirm,daily_cure,legend,scale_density=1.0,location=''):
		'''
		Plot cumulative curves of infection, death, and cure. Overlay daily counts of infected and cured.
		'''

		def make_patch_spines_invisible(ax):
		    ax.set_frame_on(True)
		    ax.patch.set_visible(False)
		    for sp in ax.spines.values():
		        sp.set_visible(False)

		fig,host = plt.subplots()
		fig.subplots_adjust(right=0.75)

		par = host.twinx()
		par2 = host.twinx()

		# Offset the right spine of par2.  The ticks and label have already been
		# placed on the right by twinx above.
		par2.spines["right"].set_position(("axes", 1.2))
		# Having been created by twinx, par2 has its frame off, so the line of its
		# detached spine is invisible.  First, activate the frame but make the patch
		# and spines invisible.
		make_patch_spines_invisible(par2)
		# Second, show the right spine.
		par2.spines["right"].set_visible(True)

		host.set_xlim(t.min(),t.max()+1)
		host.set_ylim(0,(inf_count[-1]+death_count[-1]+cure_count[-1])/scale_density)

		host.set_xlabel("Day")
		host.set_ylabel("Cumulative Count")
		par.set_ylabel("Daily Count")
		par2.set_ylabel("Daily Count")
		plt.title(location)

		p1 = host.bar(t,inf_count,color='blue',alpha=0.7)
		p2 = host.bar(t,death_count,bottom=inf_count,color='red',alpha=0.7)
		p3 = host.bar(t,cure_count,bottom=inf_count+death_count,color='orange',alpha=0.7)
		p4, = par.plot(t,daily_confirm,color='cyan',lw=3)
		p5, = par2.plot(t,daily_cure,color='magenta',lw=3,ls='-.')
		
		host.legend((p1, p2, p3, p4, p5), ('Infected', 'Died','Recovered','Daily Infected','Daily Recovered'), loc='upper '+legend)
		par.set_ylim(0,daily_confirm.max()/scale_density)
		par2.set_ylim(0,daily_cure.max()/scale_density)
		host.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
		par.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
		par2.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
		
		par.yaxis.label.set_color(p4.get_color())
		par2.yaxis.label.set_color(p5.get_color())

		tkw = dict(size=4, width=1.5)
		host.tick_params(axis='y', **tkw)
		par.tick_params(axis='y', **tkw)
		par2.tick_params(axis='y', **tkw)
		host.tick_params(axis='x', **tkw)

		return fig

	@classmethod
	def FI_expon(cls,lamb):
		'''
		example calculation of Fisher information for the exponential distribution using finite difference
		'''

		def d_expon(x,T):
			d = 1E-7
			f1 = expon.logpdf(x,scale=1/(lamb+d))-np.log(expon.cdf(T,scale=1/(lamb+d)))
			f2 = expon.logpdf(x,scale=1/lamb)-np.log(expon.cdf(T,scale=1/lamb))
			return (f1-f2)/d

		t = np.linspace(0.1,expon.ppf(1-1E-5,scale=1/lamb),100)
		result = []
		for s in t:
			tt = np.linspace(0,s,1000)
			f = list(d_expon(x,s)**2*expon.pdf(x,scale=1/lamb)/expon.cdf(s,scale=1/lamb) for x in tt)
			result.append(np.trapz(f,tt))
		plt.figure()
		plt.plot(t,result,'b-')
		plt.show(block=True)
		return

	@classmethod
	def Deriv_S(cls, S, t, a, b, c):
		'''
		system definition
		parameters: a, b, c
		'''
		S = S[0]
		dsdt = [-a*S*np.log(S + 1E-20)-b*(S-S**2)-c*S]
		return dsdt

	def Fisher_information_a(self,T):
		'''
		compute Fisher information of a for final time T using finite differences
		'''
		l = int(np.ceil(1000.0/(self.T/T)))
		ts = np.linspace(0,T,l)
		
		S0 = [1]
		a,b,c = self.theta

		sol = odeint(Epidemic.Deriv_S, S0, ts, args=(a,b,c))
		S = interp1d(ts,sol[:,0],kind='linear')

		d = 1E-7
		sol = odeint(Epidemic.Deriv_S, S0, ts, args=(a+d,b,c))
		Sdelta = interp1d(ts,sol[:,0],kind='linear')

		def loglike(t):
			s = S(t)
			sT = S(T)
			return np.log((a*s*np.log(s)+b*(s-s**2)+c*s)/(1-sT))		

		def loglike_delta(t):
			s = Sdelta(t)
			sT = Sdelta(T)
			return np.log(((a+d)*s*np.log(s)+b*(s-s**2)+c*s)/(1-sT))

		def deriv_t(t):
			f1 = loglike_delta(t)
			f2 = loglike(t)
			return (f1-f2)/d

		fxn = []
		ds = []
		dens = []
		for t in ts:
			density = np.exp(loglike(t))
			dens.append(density)
			f = deriv_t(t)**2
			ds.append(f)
			fxn.append(f*density)

		return np.trapz(fxn,ts)

	def Fisher_information_a_t(self,T):
		'''
		compute Fisher information curve in time for a
		'''
		dt = T*1.0/100
		t = np.linspace(dt,T,100)
		return t, np.asarray(list(self.Fisher_information_a(s) for s in t))

	def Fisher_information_integrand_a(self,T,rescale=True):
		'''
		compute area of the Fisher information integrand to final time T using finite differences
		'''
		l = int(np.ceil(1000.0/(self.T/T)))
		ts = np.linspace(0,T,l)
		
		S0 = [1]
		a,b,c = self.theta

		sol = odeint(Epidemic.Deriv_S, S0, ts, args=(a,b,c))
		S = interp1d(ts,sol[:,0],kind='linear')

		d = 1E-7
		sol = odeint(Epidemic.Deriv_S, S0, ts, args=(a+d,b,c))
		Sdelta = interp1d(ts,sol[:,0],kind='linear')

		def loglike(t):
			s = S(t)
			sT = S(T) if rescale else 0
			return np.log((a*s*np.log(s)+b*(s-s**2)+c*s)/(1-sT))		

		def loglike_delta(t):
			s = Sdelta(t)
			sT = Sdelta(T) if rescale else 0
			return np.log(((a+d)*s*np.log(s)+b*(s-s**2)+c*s)/(1-sT))

		def deriv_t(t):
			f1 = loglike_delta(t)
			f2 = loglike(t)
			return (f1-f2)/d

		ds = []
		for t in ts:
			f = deriv_t(t)**2
			ds.append(f)

		return np.trapz(ds,ts)

	def Fisher_information_integrand_a_t(self,T,rescale=True):
		'''
		compute Fisher information curve in time for a
		'''
		dt = T*1.0/100
		t = np.linspace(dt,T,100)
		return t, np.asarray(list(self.Fisher_information_integrand_a(s,rescale=rescale) for s in t))

	def Plot_FI_a(self):
		'''
		plot Fisher information curve of a in time
		'''
		fig = plt.figure()
		t, fi = self.Fisher_information_a_t(self.plot_T)
		plt.plot(t,fi,'c-')
		return fig

	def Fisher_information_b(self,T):
		'''
		compute Fisher information of b for final time T using finite differences
		'''
		l = np.int64(np.ceil(1000.0/(self.T/T)))
		ts = np.linspace(0,T,l)
		
		S0 = [1]
		a,b,c = self.theta

		sol = odeint(Epidemic.Deriv_S, S0, ts, args=(a,b,c))
		S = interp1d(ts,sol[:,0],kind='linear')

		d = 1E-7
		sol = odeint(Epidemic.Deriv_S, S0, ts, args=(a,b+d,c))
		Sdelta = interp1d(ts,sol[:,0],kind='linear')

		def like(t):
			s = S(t)
			sT = S(T)
			return np.log((a*s*np.log(s)+b*(s-s**2)+c*s)/(1-sT))

		def like_delta(t):
			s = Sdelta(t)
			sT = Sdelta(T)
			return np.log((a*s*np.log(s)+(b+d)*(s-s**2)+c*s)/(1-sT))

		def deriv_t(t):
			f1 = like_delta(t)
			f2 = like(t)
			return (f1-f2)/d

		fxn = []
		ds = []
		for t in ts:
			density = np.exp(like(t))
			f = deriv_t(t)**2
			ds.append(f)
			fxn.append(f*density)
		return np.trapz(fxn,ts)

	def Fisher_information_b_t(self,T):
		'''
		compute Fisher information curve in time for b
		'''
		dt = T*1.0/100
		t = np.linspace(dt,T,100)
		return t, np.asarray(list(self.Fisher_information_b(s) for s in t))

	def Plot_FI_b(self):
		'''
		plot Fisher information curve of b in time
		'''
		fig = plt.figure()
		t, fi = self.Fisher_information_b_t(T=plot_T)
		plt.plot(t,fi,'m-')
		return fig

	def Fisher_information_integrand_b(self,T,rescale=True):
		'''
		compute Fisher information of a for final time T using finite differences
		'''
		l = int(np.ceil(1000.0/(self.T/T)))
		ts = np.linspace(0,T,l)
		
		S0 = [1]
		a,b,c = self.theta

		sol = odeint(Epidemic.Deriv_S, S0, ts, args=(a,b,c))
		S = interp1d(ts,sol[:,0],kind='linear')

		d = 1E-7
		sol = odeint(Epidemic.Deriv_S, S0, ts, args=(a,b+d,c))
		Sdelta = interp1d(ts,sol[:,0],kind='linear')

		def loglike(t):
			s = S(t)
			sT = S(T) if rescale else 0
			return np.log((a*s*np.log(s)+b*(s-s**2)+c*s)/(1-sT))		

		def loglike_delta(t):
			s = Sdelta(t)
			sT = Sdelta(T) if rescale else 0
			return np.log((a*s*np.log(s)+(b+d)*(s-s**2)+c*s)/(1-sT))

		def deriv_t(t):
			f1 = loglike_delta(t)
			f2 = loglike(t)
			return (f1-f2)/d

		ds = []
		for t in ts:
			f = deriv_t(t)**2
			ds.append(f)

		return np.trapz(ds,ts)

	def Fisher_information_integrand_b_t(self,T,rescale=True):
		'''
		compute Fisher information curve in time for a
		'''
		dt = T*1.0/100
		t = np.linspace(dt,T,100)
		return t, np.asarray(list(self.Fisher_information_integrand_b(s,rescale=rescale) for s in t))

	def Fisher_information_c(self,T):
		'''
		compute Fisher information of c for final time T using finite differences
		'''
		l = int(np.ceil(1000.0/(self.T/T)))
		ts = np.linspace(0,T,l)
		
		S0 = [1]
		a,b,c = self.theta

		sol = odeint(Epidemic.Deriv_S, S0, ts, args=(a,b,c))
		S = interp1d(ts,sol[:,0],kind='linear')

		d = 1E-8
		sol = odeint(Epidemic.Deriv_S, S0, ts, args=(a,b,c+d))
		Sdelta = interp1d(ts,sol[:,0],kind='linear')

		def like(t):
			s = S(t)
			sT = S(T)
			return np.log((a*s*np.log(s)+b*(s-s**2)+c*s)/(1-sT))

		def like_delta(t):
			s = Sdelta(t)
			sT = Sdelta(T)
			return np.log((a*s*np.log(s)+b*(s-s**2)+(c+d)*s)/(1-sT))

		def deriv_t(t):
			f1 = like_delta(t)
			f2 = like(t)
			return (f1-f2)/d

		fxn = []
		ds = []
		for t in ts:
			density = np.exp(like(t))
			f = deriv_t(t)**2
			ds.append(f)
			fxn.append(f*density)
		return np.trapz(fxn,ts)

	def Fisher_information_c_t(self,T):
		'''
		compute Fisher information curve in time for b
		'''
		dt = T*1.0/100
		t = np.linspace(dt,T,100)
		return t, np.asarray(list(self.Fisher_information_c(s) for s in t))

	def Plot_FI_c(self):
		'''
		plot Fisher information curve of b in time
		'''
		fig, ax = plt.subplots()
		t, fi = self.Fisher_information_c_t(T=self.plot_T)
		plt.plot(t,fi,'c-',lw=3,label='c')
		plt.legend(loc='upper left')
		plt.xlabel('t')
		plt.ylabel('Fisher information')
		plt.axvline(x=self.T,color='b',linestyle='-')
		ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
		return fig

	def Plot_FI_ab(self):
		'''
		plot Fisher information curves of a and b in time
		'''
		fig, ax = plt.subplots()
		ta, fia = self.Fisher_information_a_t(T=self.plot_T)
		tb, fib = self.Fisher_information_b_t(T=self.plot_T)
		plt.plot(ta,fia,'c-',lw=3,label='a')
		plt.plot(tb,fib,'m-',lw=3,label='b')
		plt.legend(loc='upper left')
		plt.xlabel('t')
		plt.ylabel('Fisher information')
		plt.axvline(x=self.T,color='b',linestyle='-')
		ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

		return fig

	def Plot_FI_abc(self):
		'''
		Plot cumulative curves of infection, death, and cure. Overlay daily counts of infected and cured.
		'''

		ta, fia = self.Fisher_information_a_t(T=self.plot_T)
		tb, fib = self.Fisher_information_b_t(T=self.plot_T)
		tc, fic = self.Fisher_information_c_t(T=self.plot_T)

		def make_patch_spines_invisible(ax):
		    ax.set_frame_on(True)
		    ax.patch.set_visible(False)
		    for sp in ax.spines.values():
		        sp.set_visible(False)

		fig,host = plt.subplots()
		fig.subplots_adjust(right=0.75)

		par = host.twinx()

		host.set_xlim(0,self.plot_T)
		host.set_xlabel("t")
		host.set_ylabel("Fisher information")
		par.set_ylabel("Fisher information")

		p1, = host.plot(ta,fia,color='cyan',lw=3)
		p2, = host.plot(tb,fib,color='magenta',lw=3)
		p3, = par.plot(tc,fic,color='green',lw=3)

		plt.axvline(x=self.T,color='b',linestyle='-')
		
		host.legend((p1, p2, p3), ('a', 'b','c'), loc='upper left')
		host.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
		par.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

		par.yaxis.label.set_color(p3.get_color())

		tkw = dict(size=4, width=1.5)
		host.tick_params(axis='x', **tkw)
		host.tick_params(axis='y', **tkw)
		par.tick_params(axis='y', **tkw)

		return fig

	def Plot_FI_integrand_ab(self,rescale=True):
		'''
		plot Fisher information curves of a and b in time
		'''
		fig, ax = plt.subplots()
		ta, fia = self.Fisher_information_integrand_a_t(T=self.plot_T,rescale=rescale)
		tb, fib = self.Fisher_information_integrand_b_t(T=self.plot_T,rescale=rescale)
		plt.plot(ta,fia,'c-',lw=3,label='a')
		plt.plot(tb,fib,'m-',lw=3,label='b')
		plt.legend(loc='upper left')
		plt.xlabel('t')
		plt.ylabel('Area')
		plt.axvline(x=self.T,color='b',linestyle='-')
		ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

		return fig

	@classmethod
	def Plot_S(cls,theta,T):
		'''
		plot survival function and survival density
		theta = parameters of model = list of length 3 = [a,b,c]
		T = final time
		'''
		S0 = [1]
		tt = np.linspace(0,T,1000)
		sol = odeint(Epidemic.Deriv_S, S0, tt, args=tuple(theta))
		S = interp1d(tt,sol[:,0])

		smax = S(T)
		factor = 1 - smax
		a,b,c = theta
		like = np.asarray(list((a*S(t)*np.log(S(t))+b*(S(t)-S(t)**2)+c*S(t))/factor for t in tt))
		R0 = b/a
		rho = c/b
		tau = ((R0 + lambertw(-R0*np.exp(-R0*(1+rho))))/R0).real
		
		fig = plt.figure()
		plt.plot(tt,1-sol,'b-',label='Cumulative Function',lw=2)
		plt.plot(tt,like,'b--',label='Survival Density',lw=2)
		plt.xlabel('t')
		plt.axhline(y=tau,color='red',linestyle='-',label='Final Epidemic Size')
		plt.legend(loc='upper left')
		plt.axis([0,T,0,1.2])

		return fig

	@classmethod
	def Plot_R(cls,theta,T):
		'''
		plot recovery function and density
		theta = parameters of model = list of length 3 = [a,b,c]
		T = final time
		'''

		S0 = [1]
		tt = np.linspace(0,T+1,1000) # add one for the offset in this example
		a,b,c,gamma = theta
		sol = odeint(Epidemic.Deriv_S, S0, tt, args=(a,b,c))
		S = interp1d(tt,sol[:,0])
		rho = c/b
		T = np.float(T)

		# density 
		def density_nuA(u):
			s = S(u)
			ST = S(T)
			return (a*s*np.log(s)+b*(s-s**2)+c*s)/(1-ST)

		# density
		def density_gamma(t,gamma):
			return expon.pdf(t,scale=1/gamma)

		# integral nu_A Q
		def int_nuAQ(t,gamma):
			l = int(np.ceil(100.0/(T/t)))
			ttt = np.linspace(0,t,l)
			integrand = []
			for u in ttt:
				result = density_nuA(u)*expon.pdf(t-u,scale=1/gamma)
				integrand.append(result)
			integrand = np.asarray(integrand)
			return np.trapz(integrand,ttt)

		# improper mixture recovery density
		def density_recovery(t,gamma):
			t = max(t,T/100)
			return (1.0/(1+rho)*int_nuAQ(t,gamma)+rho/(1+rho)*density_gamma(t,gamma))

		# normalization of the improper mixture density
		def proper(gamma,offset):
			s = np.linspace(T/100,T,100).tolist()
			vals = [0]+list(density_recovery(u+offset,gamma) for u in s)
			s = [0]+s
			A = np.trapz(vals,s)
			return A

		# finally define the proper density for recoveries
		def proper_density_recovery(t,gamma,prop=None,offset=0):
			if prop is None:
				return density_recovery(t+offset,gamma)/proper(gamma,offset=offset)
			else:
				return density_recovery(t+offset,gamma)/prop

		s = np.linspace(T/100,T,100)
		like = np.asarray(list(proper_density_recovery(u,gamma,offset=0) for u in s))
		cum = cumtrapz(like,s,initial=0)
		like1 = np.asarray(list(proper_density_recovery(u,gamma,offset=-1) for u in s))
		like2 = np.asarray(list(proper_density_recovery(u,gamma,offset=1) for u in s))
		fig = plt.figure()
		plt.plot(s,cum,'b-',label='Cumulative Distribution Function',lw=2)
		plt.plot(s,like,'b--',label='Recovery Density 0 shift',lw=2)
		plt.plot(s,like1,'c--',label='Recovery Density -1 shift',lw=2)
		plt.plot(s,like2,'m--',label='Recovery Density +1 shift',lw=2)
		plt.xlabel('t')
		plt.legend(loc='upper left')
		plt.axis([0,T,0,1.2])

		return fig

	# density 
	def density_nuA(self,u):
		S = self.Scure
		s = S(u)
		ST = S(self.Tcure)
		return (self.a*s*np.log(s)+self.b*(s-s**2)+self.c*s)/(1-ST)

	# density
	def density_gamma(self,t,gamma):
		return expon.pdf(t,scale=1/gamma)

	# integral nu_A Q
	def int_nuAQ(self,t,gamma):
		l = int(max(np.ceil(100.0/(self.T/t)),2))
		ttt = np.linspace(0,t,l)
		integrand = []
		for u in ttt:
			result = self.density_nuA(u)*self.density_gamma(t-u,gamma=gamma)#expon.pdf(t-u,scale=1/gamma)
			integrand.append(result)
		integrand = np.asarray(integrand)
		return np.trapz(integrand,ttt)

	# improper mixture recovery density
	def density_recovery(self,t,gamma):
		t = max(t,self.Tcure/100)
		return (1.0/(1+self.rho)*self.int_nuAQ(t,gamma)+self.rho/(1+self.rho)*self.density_gamma(t,gamma))

	# normalization of the improper mixture density
	def proper(self,gamma,offset):
		s = np.linspace(self.Tcure/100,self.Tcure,100).tolist()
		vals = [0]+list(self.density_recovery(u+offset,gamma) for u in s)
		s = [0]+s
		A = np.trapz(vals,s)
		return A

	# finally define the proper density for recoveries
	def proper_density_recovery(self,t,gamma,prop=None,offset=0):
		if prop is None:
			return self.density_recovery(t+offset,gamma)/self.proper(gamma,offset=offset)
		else:
			return self.density_recovery(t+offset,gamma)/prop

	# negative log-likelihood of the mixture density
	def negloglikelihood_gammaoffset(self,theta):

		gamma,offset = theta 
		bound_g, bound_o = self.bounds_gamma

		if np.isclose(gamma,bound_g[0]) or np.isclose(gamma,bound_g[1]) or np.isclose(offset,bound_o[0]) or np.isclose(offset,bound_o[1]):
			lk = 1E6
			print("gamma=", gamma, "offset=", offset, "-LOGLIKE=", lk)
			return lk

		result = []
		prop = self.proper(gamma,offset=offset)
		for u in self.datacure:
			r = -np.log(self.proper_density_recovery(u,gamma,prop=prop,offset=offset))
			result.append(r)
		lk = np.sum(result)
		print("gamma=", gamma, "offset=", offset, "-LOGLIKE=", lk)
		return lk

	def negloglikelihood_gamma(self,theta):

		gamma, = theta
		bound_g, = self.bounds_gamma 

		offset = 0
		if np.isclose(gamma,bound_g[0]) or np.isclose(gamma,bound_g[1]):
			lk = 1E6
			print("gamma=", gamma, "offset=", offset, "-LOGLIKE=", lk)
			return lk

		result = []
		prop = self.proper(gamma,offset=offset)
		for u in self.datacure:
			r = -np.log(self.proper_density_recovery(u,gamma,prop=prop,offset=offset))
			result.append(r)
		lk = np.sum(result)
		print("gamma=", gamma, "offset=", offset, "-LOGLIKE=", lk)
		return lk

	def negloglikelihood_alphabeta(self,theta):

		alpha,beta = theta
		bound_a, bound_b = self.bounds_gamma
		gamma = alpha*beta 
		offset = 0

		if np.isclose(alpha,bound_a[0]) or np.isclose(alpha,bound_a[1]) or np.isclose(beta,bound_b[0]) or np.isclose(beta,bound_b[1]):
			lk = 1E6
			print("alpha=", alpha, "beta=", beta, "gamma=", gamma, "offset=", offset, "-LOGLIKE=", lk)
			return lk

		result = []
		prop = self.proper(gamma,offset=offset)
		for u in self.datacure:
			r = -np.log(self.proper_density_recovery(u,gamma,prop=prop,offset=offset))
			result.append(r)
		lk = np.sum(result)
		
		print("alpha=", alpha, "beta=", beta, "gamma=", gamma, "offset=", offset, "-LOGLIKE=", lk)
		return lk

	def estimate_gamma(self,df_recovery,N,x0,bounds,approach='gamma'):

		'''
		estimate gamma from the recovery data using N random samples, initial condition for gamma can be supplied
		'''

		# solve the system over the relevant timespan
		S0 = [1.0]
		t = self.plot_T
		if approach == 'offset':
			t += bounds[1][1]
		tt = np.linspace(0,t,1000)
		sol = odeint(Epidemic.Deriv_S, S0, tt, args=tuple(self.theta))
		S = interp1d(tt,sol[:,0])
		self.Scure = S
		self.Tcure = np.ceil(df_recovery['recovery'].max())
		self.datacure = df_recovery['recovery'].sample(N,replace=True).values

		print("naive-estimate", 1.0/(df_recovery['recovery'].median() - self.df['infection'].median()))

		# if an offset is specified, then optimize both gamma and the offset
		# otherwise only optimize gamma
		if approach == 'offset':
			self.bounds_gamma = bounds
			# perform optimization
			self.gamma, self.offset = minimize(
				self.negloglikelihood_gammaoffset, 
				x0=x0, 
				bounds=self.bounds_gamma,
				options={'disp': True, 'maxiter':3}
			).x
		elif approach == 'prior':
			# perform optimization
			self.bounds_gamma = bounds
			self.offset = 0
			# perform optimization
			self.alpha, self.beta = minimize(
				self.negloglikelihood_alphabeta, 
				x0=x0, 
				bounds=self.bounds_gamma,
				options={'disp': True, 'maxiter':3}
			).x
			self.gamma = self.alpha*self.beta
		elif approach == 'gamma':
			self.offset = 0
			self.bounds_gamma = bounds 
			# perform optimization
			self.gamma, = minimize(
				self.negloglikelihood_gamma, 
				x0=x0, 
				bounds=self.bounds_gamma,
				options={'disp': True, 'maxiter':3}
			).x

		# plot data and theoretical curve
		fig = plt.figure()
		s = np.linspace(self.Tcure/100,self.Tcure,100).tolist()
		vals = np.asarray([0]+list(self.proper_density_recovery(u,self.gamma,offset=self.offset) for u in s))
		s = np.asarray([0]+s)
		cumdensity_model = cumtrapz(vals,s,initial=0)
		cumdensity_model /= cumdensity_model[-1]

		mirrored_data = (2*self.Tcure-df_recovery['recovery'].values).tolist()
		combined_data = df_recovery['recovery'].values.tolist() + mirrored_data
		dense = gaussian_kde(combined_data)
		denseval = [0]+list(dense(u)*2 for u in s[1:])
		cumdensity_data = cumtrapz(denseval,s,initial=0)
		cumdensity_data /= cumdensity_data[-1]

		ind1 = np.argmax(cumdensity_model>0.001)
		ind2 = np.argmax(cumdensity_model>0.999)
		ind3 = np.argmax(cumdensity_data>0.001)
		ind4 = np.argmax(cumdensity_data>0.999)
		low = min(ind1,ind3)
		high = max(ind2,ind4)

		plt.plot(s[low:high+1],vals[low:high+1],'r-',lw=3)
		plt.plot(s[low:high+1],denseval[low:high+1],'b-',lw=3)
		plt.xlabel('t')
		plt.ylabel('Probability')
		plt.axvline(x=self.Tcure,color='b',linestyle='-')

		return fig

	def Fisher_information_gamma(self,T):
		'''
		compute Fisher information of gamma for final time T using finite differences
		'''
		l = int(np.ceil(100.0/(self.Tcure/T)))
		ts = np.linspace(0,T,l)
		
		S0 = [1]
		a,b,c = self.theta
		gamma = self.gamma
		d = 1E-7

		def like(t):
			return np.log(self.proper_density_recovery(t,gamma,offset=0,prop=None))

		def like_delta(t):
			return np.log(self.proper_density_recovery(t,gamma+d,offset=0,prop=None))

		def deriv_t(t):
			f1 = like_delta(t)
			f2 = like(t)
			return (f1-f2)/d

		fxn = []
		ds = []
		for t in ts:
			density = np.exp(like(t))
			f = deriv_t(t)**2
			ds.append(f)
			fxn.append(f*density)
		return np.trapz(fxn,ts)

	def Fisher_information_gamma_t(self,T):
		'''
		compute Fisher information curve in time for b
		'''
		dt = T*1.0/100
		t = np.linspace(dt,T,100)
		return t, np.asarray(list(self.Fisher_information_gamma(s) for s in t))

	def Plot_FI_gamma(self):
		'''
		plot Fisher information curve of b in time
		'''
		fig,ax = plt.subplots()
		t, fi = self.Fisher_information_gamma_t(T=self.plot_T)
		plt.plot(t,fi,'c-',lw=3)
		plt.legend(loc='upper left')
		plt.xlabel('t')
		plt.ylabel('Fisher information')
		plt.axvline(x=self.Tcure,color='b',linestyle='-')
		ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
		return fig

	def objfxn(self,abc):
		'''
		negative log-likelihood of model given data
		'''

		a,b,c=abc
		S0 = [1.0]
		#t = np.linspace(0,self.data.max(),1000)
		sol = odeint(Epidemic.Deriv_S, S0, self.t, args=(a, b, c))
		S = interp1d(self.t,sol[:,0])

		smax = S(self.T)
		factor = 1 - smax

		Z = 0
		j = 0
		for x in self.data.values:
			s = S(x)
			if s > 0:
				z = (a*s*np.log(s)+b*(s-s**2)+c*s)/factor
				if z > 0:
					Z += -np.log(z)
					j += 1
		k = self.data.shape[0]
		Z = Z/j*k if j > 0 else 1E6 # big penalty if no points satisfy positive constraint
		# big penalty for hitting boundary
		if np.isclose(a,self.bounda[0]) or np.isclose(a,self.bounda[1]) or np.isclose(b,self.boundb[0]) or np.isclose(b,self.boundb[1]) or np.isclose(c,self.boundc[0]) or np.isclose(c,self.boundc[1]):
			Z = 1E6
		# print("theta=", abc, "-LogLike=", Z, "#okay=", j, "#points=", self.data.shape[0])
		return Z

	def density(self,theta):
		'''
		evaluate the density of the fit
		'''
		S0 = [1]
		sol = odeint(Epidemic.Deriv_S, S0, self.t, args=tuple(self.theta))
		S = interp1d(self.t,sol[:,0])
		a,b,c = self.theta
		out = []
		ST = self.S(self.T)
		for x in self.t:
			Sx = self.S(x)
			out.append((a*Sx*np.log(Sx)+b*(Sx-Sx**2)+c*Sx)/(1-ST))
		return out

	def get_df(self,file_or_df):

		if isinstance(file_or_df,pd.DataFrame):
			df = file_or_df
		else:
			df = pd.read_csv(file_or_df)

		if df.shape[1] == 1:
			df.columns = ['infection']
		elif df.shape[1] == 2:
			df.columns = ['infection','recovery']

		df = df.sort_values('infection')

		return df

	def __init__(self, file_or_df, bounds=[], plot_T=None,abc=None,parent=None):

		if file_or_df is None and plot_T is None and abc is None:
			raise ValueError('if file_or_df is None, then plot_T and abc must be provided')

		if file_or_df is not None:
			self.df = self.get_df(file_or_df)
		else:
			self.df = None

		if abc is None:
			self.a = 0.3
			self.b = 0.7
			self.c = 1E-5
		else:
			self.a, self.b, self.c = abc

		if len(bounds) == 3:
			self.bounda, self.boundb, self.boundc = bounds
		else:
			self.bounda = (0.1,1)
			self.boundb = (0.1,1)
			self.boundc = (1E-8,1E-3)

		self.stda = self.stdb = self.stdc = None
		self.data = self.datacure = None
		self.ecdf = self.survivaldata = None
		self.fits = None
		self.gamma = None
		self.alpha = self.beta = None
		self.bounds_gamma = None

		if file_or_df is not None:
			self.T = np.ceil(self.df['infection'].max())
		else:
			self.T = np.ceil(plot_T)

		if plot_T is None:
			self.plot_T = self.T
		else:
			self.plot_T = np.ceil(plot_T)

		self.t = np.linspace(0,self.T,1000)

		S0 = [1]
		self.sol = odeint(Epidemic.Deriv_S, S0, self.t, args=tuple(self.theta))
		self.S = interp1d(self.t,self.sol[:,0],kind='linear')
		self.Scure = None
		self.parent = parent
		self.offset = None
		self.Tcure = None

	@property 
	def R0(self):
		return self.b/self.a 

	@property 
	def rho(self):
		return self.c/self.b

	@property 
	def invrho(self):
		return self.b/self.c

	@property 
	def delta(self):
		if self.gamma is not None:
			return self.a - self.gamma
		else:
			return 0

	@property 
	def tau(self):
		return ((self.R0 + lambertw(-self.R0*np.exp(-self.R0*(1+self.rho))))/self.R0).real

	@property 
	def kT(self):
		if self.parent is None:
			return self.df['infection'].shape[0]
		else:
			return self.parent.kT

	@property 
	def rescale(self):
		return 1-self.S(self.T)

	@property 
	def n(self):
		return self.kT/self.rescale

	@property 
	def sT(self):
		return self.n-self.kT

	@property 
	def kinfty(self):
		return self.tau*self.n

	@property 
	def sinfty(self):
		return (1-self.tau)*self.n

	@property 
	def theta(self):
		return [self.a, self.b, self.c]

	def var_T(self):
		if self.fits is not None:
			return np.var(list(f.T for f in self.fits))
		else:
			return 0

	def var_a(self):
		if self.fits is not None:
			return np.var(list(f.a for f in self.fits))
		else:
			return 0

	def var_b(self):
		if self.fits is not None:
			return np.var(list(f.b for f in self.fits))
		else:
			return 0

	def var_c(self):
		if self.fits is not None:
			return np.var(list(f.c for f in self.fits))
		else:
			return 0

	def cov_abc(self):
		if self.fits is not None:
			fitted_parms = np.zeros((len(self.theta), len(self.fits)), dtype=np.float)
			fitted_parms[0] = list(f.a for f in self.fits)
			fitted_parms[1] = list(f.b for f in self.fits)
			fitted_parms[2] = list(f.c for f in self.fits)
			return np.cov(fitted_parms)
		else:
			return np.zeros((len(self.theta), len(self.fits)), dtype=np.float)

	def var_R0(self):
		if self.fits is not None:
			return np.var(list(f.R0 for f in self.fits))
		else:
			return 0

	def var_rho(self):
		if self.fits is not None:
			return np.var(list(f.rho for f in self.fits))
		else:
			return 0

	def var_invrho(self):
		if self.fits is not None:
			return np.var(list(f.invrho for f in self.fits))
		else:
			return 0

	def var_tau(self):
		if self.fits is not None:
			return np.var(list(f.tau for f in self.fits))
		else:
			return 0

	def var_n(self):
		if self.fits is not None:
			return np.var(list(f.n for f in self.fits))
		else:
			return 0

	def var_kT(self):
		if self.fits is not None:
			return np.var(list(f.kT for f in self.fits))
		else:
			return 0

	def var_sT(self):
		if self.fits is not None:
			return np.var(list(f.sT for f in self.fits))
		else:
			return 0

	def var_kinfty(self):
		if self.fits is not None:
			return np.var(list(f.kinfty for f in self.fits))
		else:
			return 0

	def var_sinfty(self):
		if self.fits is not None:
			return np.var(list(f.sinfty for f in self.fits))
		else:
			return 0

	def mean_T(self):
		if self.fits is not None:
			return np.mean(list(f.T for f in self.fits))
		else:
			return 0

	def mean_a(self):
		if self.fits is not None:
			return np.mean(list(f.a for f in self.fits))
		else:
			return 0

	def mean_b(self):
		if self.fits is not None:
			return np.mean(list(f.b for f in self.fits))
		else:
			return 0

	def mean_c(self):
		if self.fits is not None:
			return np.mean(list(f.c for f in self.fits))
		else:
			return 0

	def mean_R0(self):
		if self.fits is not None:
			return np.mean(list(f.R0 for f in self.fits))
		else:
			return 0

	def mean_rho(self):
		if self.fits is not None:
			return np.mean(list(f.rho for f in self.fits))
		else:
			return 0

	def mean_invrho(self):
		if self.fits is not None:
			return np.mean(list(f.invrho for f in self.fits))
		else:
			return 0

	def mean_tau(self):
		if self.fits is not None:
			return np.mean(list(f.tau for f in self.fits))
		else:
			return 0

	def mean_n(self):
		if self.fits is not None:
			return np.mean(list(f.n for f in self.fits))
		else:
			return 0

	def mean_kT(self):
		if self.fits is not None:
			return np.mean(list(f.kT for f in self.fits))
		else:
			return 0

	def mean_sT(self):
		if self.fits is not None:
			return np.mean(list(f.sT for f in self.fits))
		else:
			return 0

	def mean_kinfty(self):
		if self.fits is not None:
			return np.mean(list(f.kinfty for f in self.fits))
		else:
			return 0

	def mean_sinfty(self):
		if self.fits is not None:
			return np.mean(list(f.sinfty for f in self.fits))
		else:
			return 0

	def dropout_benefit(self):
		z = np.linspace(0.01, 0.99, 100)
		Out = []
		for i, th in enumerate(self.get_theta()):
			out = []
			for t in z:
				R0d = th[1] / (th[0] * t)
				rhod = th[2] / th[1]
				taunodrop = ((R0d + lambertw(-R0d * np.exp(-R0d * (1 + rhod)))) / R0d).real
				out.append(taunodrop)
			Out.append(np.asarray(out))
		Out = np.asarray(Out)
		change = (self.tau - Out) / Out * 100
		pt = self.delta/self.a
		idx = sum(1 - z > pt)
		temp = np.mean(change, axis=0)
		self.dropout_benefit = temp[idx]



	def summary(self, ifSave = False, fname = None):

		headers=['Parameter','Name','MLE','Mean','StdErr']
		table = [['T', 'final time', self.T, None if self.fits is None else self.mean_T(), None if self.fits is None else np.sqrt(self.var_T())],
			["a",'beta+gamma+delta',self.a,None if self.fits is None else self.mean_a(),self.stda if self.fits is None else np.sqrt(self.var_a())],
			["b","beta*mu", self.b,None if self.fits is None else self.mean_b(),self.stdb if self.fits is None else np.sqrt(self.var_b())],
			["c","beta*mu*rho", self.c,None if self.fits is None else self.mean_c(),self.stdc if self.fits is None else np.sqrt(self.var_c())],
			['R0',"R-naught", self.R0,None if self.fits is None else self.mean_R0(),None if self.fits is None else np.sqrt(self.var_R0())],
			['rho',"initial fraction I", self.rho,None if self.fits is None else self.mean_rho(), None if self.fits is None else np.sqrt(self.var_rho())],
			['tau',"epidemic size", self.tau, None if self.fits is None else self.mean_tau(), None if self.fits is None else np.sqrt(self.var_tau())],
			['1-S(T)',"rescaling",self.rescale,None,None],
			['n', "#S+#I", self.n, None if self.fits is None else self.mean_n(),  None if self.fits is None else np.sqrt(self.var_n())],
			['kT',"#I(T)", self.kT,None if self.fits is None else self.mean_kT(), None if self.fits is None else np.sqrt(self.var_kT())],
			['sT',"#S(T)", self.sT,None if self.fits is None else self.mean_sT(), None if self.fits is None else np.sqrt(self.var_sT())],
			['kinfty',"#I(infty)", self.kinfty, None if self.fits is None else self.mean_kinfty(), None if self.fits is None else np.sqrt(self.var_kinfty())],
			['sinfty','#S(infty)', self.sinfty, None if self.fits is None else self.mean_sinfty(), None if self.fits is None else np.sqrt(self.var_sinfty())],
			['1/rho','initial total population',self.invrho,None if self.fits is None else self.mean_invrho(), None if self.fits is None else np.sqrt(self.var_invrho())],
			['gamma','recovery rate',None if self.gamma is None else self.gamma,None,None],
			['offset','shift parameter',None if self.offset is None else self.offset,None,None]]
		print(tabulate(table,headers=headers))

		#print(tabulate(table, headers=headers, tablefmt="html"))

		# print(tabulate(table,headers=headers,tablefmt="latex_booktabs"))

		if ifSave:
			str1 = '\\documentclass{article}\n \\usepackage{booktabs} \n \\begin{document}'
			str2 = '\\end{document}'
			if fname == None:
				fname = 'summary.tex'
			with open(fname, 'w') as outputfile:
				outputfile.write(str1 + tabulate(table, headers=headers, tablefmt="latex_booktabs") + str2)
		return self

	@property
	def number_boundary_samples(self):
		if self.fits is not None:
			return len(self.fits) - len(list(self.get_theta()))
		else:
			return None

	@property
	def number_interior_samples(self):
		if self.fits is not None:
			return len(list(self.get_theta()))
		else:
			return None

	def get_theta(self):
		if self.fits is None:
			for i in range(1000):
				yield self.theta + np.asarray([rand.normal(0,self.stda),rand.normal(0,self.stdb),rand.normal(0,self.stdc)])
		else:
			for fit in self.fits:
				# skip boundary values
				if not np.isclose(fit.a,self.bounda[0]) and not np.isclose(fit.a,self.bounda[1]) and not np.isclose(fit.b,self.boundb[0]) and not np.isclose(fit.b,self.boundb[1]) and not np.isclose(fit.c,self.boundc[0]) and not np.isclose(fit.c,self.boundc[1]):
					yield np.asarray([fit.a,fit.b,fit.c])

	def plot_density_fit(self):

		Ds = np.asarray(list(self.density(th) for th in self.get_theta()))
		Dslow = np.percentile(Ds,2.5,axis=0)
		Dshigh = np.percentile(Ds,97.5,axis=0)
		Dmean = self.density(self.theta)

		fig = plt.figure()
		plt.plot(self.t,Dmean,'r-',color='red',lw=3)
		plt.plot(self.t,Dslow,'r--',lw=1)
		plt.plot(self.t,Dshigh,'r--',lw=1)
		plt.axvline(x=self.T,color='b',linestyle='-')

		mirrored_data = (2*self.T-self.df['infection'].values).tolist()
		combined_data = self.df['infection'].values.tolist() + mirrored_data
		dense = gaussian_kde(combined_data)
		denseval = list(dense(x)*2 for x in self.t)
		plt.plot(self.t,denseval,'b-',color='blue',lw=3)
		plt.fill_between(self.t, Dslow, Dshigh, alpha=.3, color='red')
		plt.legend()
		plt.ylabel('$-S_t/(1-S_T)$')
		plt.xlabel('t')
		c = cumtrapz(Dmean,self.t)
		ind = np.argmax(c>=0.001)
		plt.xlim((self.t[ind],self.t[-1]+1))

		return fig

	def plot_survival_fit(self):

		'''
		plot model fit with CIs against data
		'''

		S0 = [1]
		tt = np.linspace(0,self.plot_T*1.5,1000)
		Ss = np.asarray(list(odeint(Epidemic.Deriv_S, S0, tt, args=tuple(th))[:,0] for th in self.get_theta()))
		Sslow = np.percentile(Ss,2.5,axis=0)
		Sshigh = np.percentile(Ss,97.5,axis=0)
		fig = plt.figure()
		plt.plot(tt,Sslow,'r--',lw=1)
		plt.plot(tt,Sshigh,'r--',lw=1)
		plt.fill_between(tt, Sslow, Sshigh, alpha=.3, color='red')
		sol = odeint(Epidemic.Deriv_S, S0, tt, args=tuple(self.theta))
		plt.plot(tt,sol,'r-',color='red',lw=3)
		plt.axvline(x=self.T,color='b',linestyle='-')
		if self.df is not None:
			plt.plot(self.df['infection'].values, self.survivaldata, 'b-', color='blue', lw=3)
		plt.legend()
		plt.ylabel('$S_t$')
		plt.xlabel('t')
		plt.axis([0,self.plot_T*1.5,0,1.2])

		return fig

	@classmethod
	def plot_survival_fits(cls,epidemics):

		'''
		plot model fit with CIs against data
		'''
		fig = plt.figure()
		S0 = [1]
		for epi in epidemics:
			tt = np.linspace(0,epi.plot_T,1000)
			Ss = np.asarray(list(odeint(Epidemic.Deriv_S, S0, tt, args=tuple(th))[:,0] for th in epi.get_theta()))
			Sslow = np.percentile(Ss,2.5,axis=0)
			Sshigh = np.percentile(Ss,97.5,axis=0)
			plt.plot(tt,Sslow,'r--',lw=1)
			plt.plot(tt,Sshigh,'r--',lw=1)
			plt.fill_between(tt, Sslow, Sshigh, alpha=.3, color='red')
			sol = odeint(Epidemic.Deriv_S, S0, tt, args=tuple(epi.theta))
			plt.plot(tt,sol,'r-',color='red',lw=3)
			plt.axvline(x=epi.T,color='b',linestyle='-')
		for epi in epidemics:	
			plt.plot(epi.df['infection'].values, epi.survivaldata, 'b-', color='blue', lw=3)
		plt.ylabel('$S_t$')
		plt.xlabel('t')
		plt.axis([0,epi.plot_T,0,1.2])

		return fig

	def plot_infections(self):

		tt = np.linspace(0,self.T*1.25,1000)
		S0 = [1]
		sol = odeint(Epidemic.Deriv_S, S0, tt, args=tuple(self.theta))
		S = interp1d(tt,sol[:,0])

		'''
		plot of infections vs empirical
		'''

		fig1 = plt.figure()
		mean = np.asarray(list(self.n*(1-S(x)) for x in tt[1:]))
		low = np.asarray(list(binom.ppf(0.025,np.ceil(self.n),1-S(x)) for x in tt[1:]))
		high = np.asarray(list(binom.ppf(0.975,np.ceil(self.n),1-S(x)) for x in tt[1:]))
		plt.plot(tt[1:],mean,'r-',color='red',lw=3)
		plt.plot(tt[1:],low,'r--',lw=1)
		plt.plot(tt[1:],high,'r--',lw=1)
		plt.axvline(x=self.T,color='b',linestyle='-')
		#plt.ylim([0,80000])
		plt.fill_between(tt[1:], low, high, alpha=.3, color='red')
		counts = np.cumsum(np.ones(self.df['infection'].shape[0]))
		plt.plot(self.df['infection'].values,counts,color='b',lw=3)
		plt.ylabel('$K_A$')
		plt.xlabel('t')

		'''
		stem plot of infection pdf at time T
		'''

		fig2, ax = plt.subplots()
		low = binom.ppf(0.001,int(self.n)+1,1-S(self.T))
		high = binom.ppf(0.999,int(self.n)+1,1-S(self.T))
		grid = np.arange(low,high+1)
		pdfs = list(binom.pmf(x,int(self.n)+1,1-S(self.T)) for x in grid)
		plt.stem(grid,pdfs)
		plt.xlabel('$K_A$')
		plt.ylabel('Probability')
		plt.title('Distribution of $K_A$ at time $T$: $A=(0,T]$')
		plt.axvline(x=self.df.shape[0],color='r',linestyle='-')
		ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

		return (fig1, fig2)

	@classmethod
	def plot_infection_fits(cls,epidemics):

		fig, ax = plt.subplots()

		maxes = []
		for epi in epidemics:

			tt = np.linspace(0,epi.plot_T*1.25,1000)
			S0 = [1]
			sol = odeint(Epidemic.Deriv_S, S0, tt, args=tuple(epi.theta))
			S = interp1d(tt,sol[:,0])
			mean = np.asarray(list(epi.n*(1-S(x)) for x in tt[1:]))
			low = np.asarray(list(binom.ppf(0.025,int(epi.n)+1,1-S(x)) for x in tt[1:]))
			high = np.asarray(list(binom.ppf(0.975,int(epi.n)+1,1-S(x)) for x in tt[1:]))
			plt.plot(tt[1:],mean,'r-',color='red',lw=3)
			plt.plot(tt[1:],low,'r--',lw=1)
			plt.plot(tt[1:],high,'r--',lw=1)
			plt.axvline(x=epi.T,color='b',linestyle='-')
			plt.fill_between(tt[1:], low, high, alpha=.3, color='red')
			counts = np.cumsum(np.ones(epi.df['infection'].shape[0]))
			plt.plot(epi.df['infection'].values,counts,color='b',lw=3)
			plt.annotate('T='+str(np.ceil(epi.T)),xy=(epi.plot_T*1.25-20,mean[-1]))
			maxes.append(mean[-1])

		ax.get_yaxis().set_major_formatter(
			matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
		plt.ylabel('$K_A$')
		plt.xlabel('t')
		plt.ylim((0,max(maxes)+1000))

		return fig

	def plot_dropout(self,pt=None):

		z = np.linspace(0.01,0.99,100)
		Out = []
		for i,th in enumerate(self.get_theta()):
			out = []
			for t in z:
				R0d = th[1]/(th[0]*t)
				rhod = th[2]/th[1]
				taunodrop = ((R0d + lambertw(-R0d*np.exp(-R0d*(1+rhod))))/R0d).real
				out.append(taunodrop)
			Out.append(np.asarray(out))
		Out = np.asarray(Out)
		change = (self.tau - Out)/Out*100
		fig = plt.figure()
		plt.plot(1-z,np.mean(change,axis=0),'c-',lw=3)
		plt.plot(1-z,np.percentile(change,2.5,axis=0),'c--',lw=1)
		plt.plot(1-z,np.percentile(change,97.5,axis=0),'c--',lw=1)
		plt.fill_between(1-z, np.percentile(change,2.5,axis=0), np.percentile(change,97.5,axis=0), alpha=.3, color='cyan')
		plt.xlabel('$\delta/a$')
		plt.ylabel('percent change in epidemic size')
		if pt is not None:
			plt.axvline(x=pt,color='r',linestyle='-')
			idx = sum(1 - z > pt)
			temp = np.mean(change,axis=0)
			self.dropout_benefit = temp[idx]
		return fig

	def fit(self,N=None, summary = False, ifSave = False, fname='summary.tex'):

		'''
		N = number of (random) samples to use for fit
		'''

		if N is None:
			self.data = self.df['infection']
		else: 
			self.data = self.df['infection'].sample(N,replace=True)

		# fit data
		self.a, self.b, self.c = minimize(
			self.objfxn, 
			x0=tuple(self.theta), 
			bounds=[self.bounda, self.boundb, self.boundc],
			options={'disp': False}
		).x

		# compute standard errors of parameters
		self._std()

		# for the parameter fit, get interpolator and save
		S0 = [1]
		self.sol = odeint(Epidemic.Deriv_S, S0, self.t, args=tuple(self.theta))
		self.S = interp1d(self.t,self.sol[:,0],kind='linear')

		# get empirical distribution of data
		self.ecdf = ECDF(self.df['infection'].values)
		self.survivaldata = np.asarray(list(1-self.ecdf(x)*(1-self.S(self.T)) for x in self.df['infection'].values))

		# print summary
		if summary:
			self.summary()
		# self.summary()
		if ifSave:
			self.summary(ifSave=True, fname=fname)


		return self

	def simulate_from_model(self,N):
		'''
		simulate N infections from the survival curve of the fitted model using inverse sampling
		'''
		unis = rand.uniform(low=self.S(self.T),high=1,size=N)

		vals = np.linspace(0,self.T,10000)
		Ss = np.asarray(list(self.S(x) for x in vals))
		values = []
		for u in unis:
			i = np.argmax(Ss-u<0)
			if i>0 and i<vals.size-1:
				'''
				point slope form to find zero
				'''
				y0 = Ss[i-1]-u
				y1 = Ss[i]-u
				x0 = vals[i-1]
				x1 = vals[i]
				m = (y1-y0)/(x1-x0)
				v = (m*x1-y1)/m
			else:
				v = vals[i]
			values.append(v)

		df = pd.DataFrame(values,index=range(N),columns=['infection'])

		return df

	def simulate_and_fit(self,N,n):
		'''
		simulate N data from the model n times and refit
		'''
		fits = []
		for i in range(1,n+1):
			# print("sample", i)
			df = self.simulate_from_model(N)
			fit = Epidemic(file_or_df=df,bounds=[self.bounda, self.boundb, self.boundc],abc=self.theta,parent=self).fit()
			fits.append(fit)
		self.fits = fits

		return self

	def simulate_and_fit_parallel(self, N, n):
		'''
        simulate N data from the model n times and refit
        '''
		epidemic_obj_arr = []
		total_time = 0
		import time

		for i in range(1, n + 1):
			# print("sample", i // 100)
			df = self.simulate_from_model(N)
			st = time.time()
			epidemic_obj = Epidemic(file_or_df=df, bounds=[self.bounda, self.boundb, self.boundc], abc=self.theta,
									parent=self)
			total_time += time.time() - st
			epidemic_obj_arr.append(epidemic_obj)

		import pickle

		st = time.time()
		if os.path.exists("epidemic_objects_array_fitted"):
			os.remove("epidemic_objects_array_fitted")
		pickle.dump(epidemic_obj_arr, open("epidemic_objects_array", "wb"), protocol=3)
		os.system("mpiexec -n %s python parallel_epidemic.py" % THREADS)
		epidemic_objects_array_fitted = pickle.load(open("epidemic_objects_array_fitted", "rb"))
		print("Total fit time %f" % ((time.time() - st) / 60.0))

		self.fits = epidemic_objects_array_fitted
		return self

	def simulate_and_fit_parallel_laplace(self, N, n, rank):
		'''
        simulate N data from the model n times and refit
        '''
		epidemic_obj_arr = []
		total_time = 0
		import time

		for i in range(1, n + 1):
			# print("sample", i)
			df = self.simulate_from_model(N)
			st = time.time()
			epidemic_obj = Epidemic(file_or_df=df, bounds=[self.bounda, self.boundb, self.boundc], abc=self.theta,
									parent=self)
			total_time += time.time() - st
			epidemic_obj_arr.append(epidemic_obj)

		import pickle

		st = time.time()
		THREADS = 4
		# if os.path.exists("epidemic_objects_array_fitted_%s" % rank):
		# 	os.remove("epidemic_objects_array_fitted_%s" % rank)
		pickle.dump(epidemic_obj_arr, open("epidemic_objects_array_%s" % rank, "wb"), protocol=3)
		os.system("mpiexec -n %s python parallel_epidemic_laplace.py %s" % (THREADS, rank))
		epidemic_objects_array_fitted = pickle.load(open("epidemic_objects_array_fitted_%s" % rank, "rb"))
		print("Total fit time %f" % ((time.time() - st) / 60.0))

		self.fits = epidemic_objects_array_fitted

		return self

	@classmethod
	def n_estimates(cls, theta, df):
		nDays = len(df.time.values)
		time_points = np.arange(nDays)
		S0 = [1]
		sol = odeint(Epidemic.Deriv_S, S0, time_points, args=tuple(theta))
		S = interp1d(time_points, sol[:, 0])
		mean = np.asarray(list((1 - S(x)) for x in time_points))
		res = np.divide(df["cum_confirm"], mean)
		res.pop(0)
		return res

	def get_histograms(self):

		theta = list(self.get_theta())
		R0s = list(b/a for a,b,c in theta)
		rhos = list(c/b for a,b,c in theta)

		figa = plt.figure()
		plt.hist(list(a for a,b,c in theta))
		plt.title('a')

		figb = plt.figure()
		plt.hist(list(b for a,b,c in theta))
		plt.title('b')

		figc = plt.figure()
		plt.hist(list(c for a,b,c in theta))
		plt.title('c')

		figR0 = plt.figure()
		plt.hist(R0s)
		plt.title('$R_0$')

		figrho = plt.figure()
		plt.hist(rhos)
		plt.title('rho')

		fign, ax = plt.subplots()
		plt.hist(list(f.n for f in self.fits))
		plt.title('n')
		ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

		figsT, ax = plt.subplots()
		plt.hist(list(f.sT for f in self.fits))
		plt.title('$s_T$')
		ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

		figkinfty, ax = plt.subplots()
		plt.hist(list(f.kinfty for f in self.fits))
		plt.title('$k_\infty$')
		ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

		figsinfty, ax = plt.subplots()
		plt.hist(list(f.sinfty for f in self.fits))
		plt.title('$s_\infty$')
		ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

		figsinvrho, ax = plt.subplots()
		plt.hist(list(f.invrho for f in self.fits))
		plt.title('1/rho')
		ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))


		return (figa, figb, figc, figR0, figrho, fign, figsT, figkinfty, figsinfty, figsinvrho)

	def _std(self):

		"""
		compute standard error of parameters as inverse Hessian of negative log-likelihood using finite difference
		"""

		#print opt.hess_inv.todense()
		d = 1E-7
		theta = self.theta

		Theta = np.copy(theta)
		Theta[0] = Theta[0] + d
		aa1 = self.objfxn(tuple(Theta))
		Theta = np.copy(theta)
		Theta[0] = Theta[0] - d
		aa2 = self.objfxn(tuple(Theta))
		aa3 = self.objfxn(tuple(theta))

		self.stda = 1/np.sqrt((aa1 - 2*aa3 + aa2)/d**2)

		Theta = np.copy(theta)
		Theta[1] = Theta[1] + d
		bb1 = self.objfxn(tuple(Theta))
		Theta = np.copy(theta)
		Theta[1] = Theta[1] - d
		bb2 = self.objfxn(tuple(Theta))
		bb3 = self.objfxn(tuple(theta))

		self.stdb = 1/np.sqrt((bb1 - 2*bb3 + bb2)/d**2)

		d = 1E-9
		Theta = np.copy(theta)
		Theta[2] = Theta[2] + d
		cc1 = self.objfxn(tuple(Theta))
		Theta = np.copy(theta)
		Theta[2] = Theta[2] - d
		cc2 = self.objfxn(tuple(Theta))
		cc3 = self.objfxn(tuple(theta))

		self.stdc = 1/np.sqrt((cc1 - 2*cc3 + cc2)/d**2)

		return self

	def predict(self, samples, df, N, dates, plot_folder, fname):
		nDays = len(dates)
		time_points = np.arange(nDays)
		mean = np.zeros((N, nDays), dtype=np.float)
		mean_daily = np.zeros((N, nDays), dtype=np.float)
		theta = np.mean(samples, axis=0)
		n = np.mean(Epidemic.n_estimates(tuple(theta), df))
		fig = plt.figure()
		for i in range(N):
			S0 = [1]
			sol = odeint(Epidemic.Deriv_S, S0, time_points, args=tuple(samples[i]))
			S = interp1d(time_points, sol[:, 0])
			mean[i] = np.asarray(list(n * (1 - S(x)) for x in time_points))
			mean[i][0] = 1
			mean_daily[i] = np.append(mean[i][0], np.diff(mean[i]))
			l1, = plt.plot(dates['d'].dt.date, mean[i], '-', color=myColours['tud2b'].get_rgb(), lw=1, alpha=0.05)

		m_ = np.int64(np.ceil(np.mean(mean, axis=0)))
		l = np.int64(np.ceil(np.quantile(mean, q=0.025, axis=0)))
		h = np.int64(np.ceil(np.quantile(mean, q=0.975, axis=0)))

		S0 = [1]
		n = np.mean(Epidemic.n_estimates(tuple(theta), df))
		sol = odeint(Epidemic.Deriv_S, S0, time_points, args=tuple(theta))
		S = interp1d(time_points, sol[:, 0])
		m = np.asarray(list(n * (1 - S(x)) for x in time_points))
		l2 = plt.plot(dates['d'].dt.date, m, '-', color=myColours['tud1d'].get_rgb(), lw=3)
		l3 = plt.plot(dates['d'].dt.date, l, '--', color=myColours['tud1d'].get_rgb(), lw=1)
		l4 = plt.plot(dates['d'].dt.date, h, '--', color=myColours['tud1d'].get_rgb(), lw=1)
		l5 = plt.fill_between(dates['d'].dt.date, l, h, alpha=.1, color=myColours['tud1a'].get_rgb())

		l6 = plt.axvline(x=df['time'].max(), color=myColours['tud7d'].get_rgb(), linestyle='--')
		l6 = plt.axvline(x=df['time'][self.T - 1], color=myColours['tud7a'].get_rgb(), linestyle='-')
		l7 = plt.plot(df['time'].values, df['cum_confirm'].values, '-', color=myColours['tud7d'].get_rgb(),
					  lw=3)
		plt.xlabel('Dates')
		plt.ylabel('Cumulative infections')
		fig_save(fig, plot_folder, fname)

		fname_ = fname + '_daily_new'
		fig = plt.figure()
		for i in range(N):
			l1 = plt.plot(dates['d'].dt.date, mean_daily[i], '-', color=myColours['tud2b'].get_rgb(), lw=1, alpha=0.05)
		m_daily = np.append(m[0], np.diff(m))
		l2, = plt.plot(dates['d'].dt.date, m_daily, '-', color=myColours['tud1d'].get_rgb(), lw=3,
					   label='With mitigation')
		l6 = plt.axvline(x=df['time'].max(), color=myColours['tud7d'].get_rgb(), linestyle='--')
		l6 = plt.axvline(x=df['time'][self.T - 1], color=myColours['tud7a'].get_rgb(), linestyle='-')
		plt.ylabel('Daily new infections')
		plt.xlabel('Dates')
		fig_save(fig, plot_folder, fname_)
		m[0] = 1
		my_dict = {}
		my_dict['Dates'] = dates['d']
		my_dict['Mean'] = m
		my_dict['High'] = h
		my_dict['Low'] = l
		my_dict = pd.DataFrame(my_dict)
		my_dict.to_csv(os.path.join(plot_folder, fname + '.csv'), index=False)
		return my_dict

