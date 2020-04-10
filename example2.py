'''
This example performs the Laplace approximation on a dummy data. 
'''
import os as os
import numpy as np
import pandas as pd

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

root_folder = os.getcwd()
data_folder = os.path.join(root_folder,'data')

today = pd.to_datetime('today')
location = 'Ohio'
datafile = 'dummy.csv'
plot_folder =  'plots'
estimate_gamma = False
ifMPI = False
ifsmooth = False
last_date = pd.to_datetime(np.datetime64('2020-03-22 00:00:00'))

laplace_dict = {'datafile': datafile,
                'location': 'Ohio',
                'plot_folder': plot_folder,
                'ifMPI': ifMPI,
                'last_date': last_date,
                'estimate_gamma': estimate_gamma,
                'ifsmooth': ifsmooth}


fname = 'laplace_dict' + today.strftime("%m%d") + '.pkl'
f = open(os.path.join(data_folder,fname),"wb")
pickle.dump(laplace_dict,f)
f.close()

os.system('python DSA_Laplace.py')




