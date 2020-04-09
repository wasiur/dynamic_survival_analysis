# Dynamic Survival Analysis

This repository provides a Python implementation of the dynamic survival analysis method.

**This is primarily based on a package (available [here](https://github.com/calebbastian/epidemic)) developed by Caleb Deen Bastian, Princeton University.** I also acknowledge Saket Gurukar, who helped with the parallelization of some of the routines.

If you have questions, comments, criticisms, or corrections, please email me at [khudabukhsh.2@osu.edu](mailto:khudabukhsh.2@osu.edu).

## Installation
1. Please make sure you have Python (version 3.6.x and above). If you do not have Python, we recommend installing it from Anaconda (link [here](https://www.anaconda.com/distribution/)).
2. You can download our package either by hitting download or by cloning our repository. Cloning can be done by running the following command
```bash
git clone https://github.com/wasiur/dynamic_survival_analysis.git
```
from your terminal.

3. Our implementation depends on a number of packages. In order for the parallelization to run smoothly, we recommend installing the following python environment "dynamic_survival_analysis". This is included in the file _environment.yml_. If you are using Anaconda (recommended), the environment can be installed by running
```bash
conda env create -f environment.yml
```
In order to check if the environment is now available, run
```bash
conda env list
```
4. Activate the environment "dynamic_survival_analysis" by running
```bash
source activate dynamic_survival_analysis
```
from your terminal.

## Data preparation
The input data to the model should have following seven columns:

time  | daily_confirm | recovery | deaths |	cum_confirm |	cum_heal |	cum_dead
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
2020-03-01 |	2 |	0 |	0	| 2	| 0 |	0
2020-03-02 |	8 |	1 |	0 |	10 |	1 |	0 |
. | . | . | . | . | . | .
. | . | . | . | . | . | .
. | . | . | . | . | . | .
2020-06-05 |	46 |	13 |	21	 | 63291	| 1200	| 1037

The metadata to the main Python scripts is passed by creating the following Python dictionaries:
```python
import numpy as np
import pandas as pd
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

location = 'Ohio'  
datafile = 'epi_data.csv'
plot_folder =  'plots'
estimate_gamma = True
ifMPI = True
last_date = pd.to_datetime(np.datetime64('2020-06-06 00:00:00'))
dsa_dict = {'datafile': datafile,
            'location': 'Ohio',
            'plot_folder': plot_folder,
            'last_date': last_date,
            'estimate_gamma': estimate_gamma}

laplace_dict = {'datafile': datafile,
                'location': 'Ohio',
                'plot_folder': plot_folder,
                'ifMPI': 'true',
                'last_date': last_date,
                'estimate_gamma': estimate_gamma}

fname = 'dsa_dict' + '.pkl'
f = open(fname,"wb")
pickle.dump(dsa_dict,f)
f.close()

fname = 'laplace_dict' + '.pkl'
f = open(fname,"wb")
pickle.dump(laplace_dict,f)
f.close()
```
Description of the variables:
Name | Description
--------- | ---------
datafile | Name of file containing the daily counts. It should be a ```.csv``` file.
location | Name of the location. This name will be added to the output files.
plot_folder | The name of the output folder. All output figures and tables will be stored in this folder.
estimate_gamma | Binary variable indicating whether to estimate the recovery rate. By default, it is set to ```True```.
last_date | Last date of data to be considered for modelling purposes.
ifMPI | Binary variable indicating whether to use ```MPI``` for parallelization. By default, it is set to ```True```.


If recovery information is not available, the model can be still run by explicitly setting the variable ```estimate_gamma = False```.

We used COVID-19 data published by the New York Times to inform our model. The repository can be accessed [here](https://github.com/nytimes/covid-19-data).

## Running the dynamic survival analysis model
1. Prepare the ```.pkl``` files for the main python scripts. Instruction given above.

2. The model with a strong prior on the parameters, which could be estimated from other epidemics that can be considered similar, can be run by invoking
```bash
python DSA.py
```
from the terminal or by running
```python
runfile('DSA.py')
```
from within Python.

3. The semi-Bayesian Laplace approximation to the posterior distribution of the parameters can be carried out by running the following command
```bash
python DSA_Laplace.py
```
from the terminal or by running
```Python
runfile('DSA_Laplace.py')
```
from within Python.
