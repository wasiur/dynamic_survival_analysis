'''
This example fits the dynamic survival analysis model to the count data available from a repository maintained by the New York Times.
The link to the repository is available here:

https://github.com/nytimes/covid-19-data

In this example, we use the data from the state of Ohio.
'''

import numpy as np
import pandas as pd
import os as os
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

root_folder = os.getcwd()
data_folder = os.path.join(root_folder,'data')

today = pd.to_datetime('today')
location = 'Ohio'
datafile = 'oh_040820.csv'
plot_folder =  'plots'
estimate_gamma = False
ifMPI = False
ifsmooth = False
last_date = pd.to_datetime(np.datetime64('2020-04-07 00:00:00'))


dsa_dict = {'datafile': datafile,
            'location': 'Ohio',
            'plot_folder': plot_folder,
            'last_date': last_date,
            'estimate_gamma': estimate_gamma,
            'ifsmooth': ifsmooth,
            'ifMPI': ifMPI}

laplace_dict = {'datafile': datafile,
                'location': 'Ohio',
                'plot_folder': plot_folder,
                'ifMPI': ifMPI,
                'last_date': last_date,
                'estimate_gamma': estimate_gamma,
                'ifsmooth': ifsmooth}



fname = 'dsa_dict' + today.strftime("%m%d") +  '.pkl'
f = open(os.path.join(data_folder,fname),"wb")
pickle.dump(dsa_dict,f)
f.close()

fname = 'laplace_dict' + today.strftime("%m%d") + '.pkl'
f = open(os.path.join(data_folder,fname),"wb")
pickle.dump(laplace_dict,f)
f.close()

os.system('python DSA.py')
