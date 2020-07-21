# Dynamic Survival Analysis

This repository provides a Python implementation of the dynamic survival analysis method. A brief description of the methodology can be found in this [white paper](https://idi.osu.edu/assets/pdfs/covid_response_white_paper.pdf). Prof. Greg Rempała gave a couple of public talks on this model. You can watch his [MBI](https://mbi.osu.edu/) seminar talks here: [link to his first talk](https://video.mbi.ohio-state.edu/video/player/?id=4888&title=Mathematical+Models+of+Epidemics%3A+Tracking+Coronavirus+using+Dynamic+Survival+Analysis) and [link to his second talk](https://video.mbi.ohio-state.edu/video/player/?id=4891&title=Mathematics+of+Modeling+a+Pandemic%3A+The+Journey+Continues).


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
conda activate dynamic_survival_analysis
```
or 
```bash
source activate dynamic_survival_analysis
```
from your terminal.

## Data preparation
A typical input data to the model should have following seven columns:

time  | daily_confirm | recovery | deaths |	cum_confirm |	cum_heal |	cum_dead
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
2020-03-01 |	2 |	0 |	0	| 2	| 0 |	0
2020-03-02 |	8 |	1 |	0 |	10 |	1 |	0 |
. | . | . | . | . | . | .
. | . | . | . | . | . | .
. | . | . | . | . | . | .
2020-06-05 |	46 |	13 |	21	 | 63291	| 1200	| 1037


At least one of ```daily_confirm``` and ```cum_confirm``` must be present. If the parameters corresponding to the recovery distribution need to be estimated, at least of the four ```recovery```, ```deaths```, ```cum_heal```, and ```cum_dead``` must be present in the dataset. 


If no recovery information is available, the model can be still run by explicitly providing the ```-r``` option.

We used COVID-19 data published by the New York Times to inform our model. The repository can be accessed [here](https://github.com/nytimes/covid-19-data).

## Running the dynamic survival analysis model
The python scripts allow a number of options. The most important option is ```-d ```, which is used to pass the name of the data file to the python script. If no dataset is present, the model can be run on dummy data by providing the ```-v``` or ```--verbose``` option, which makes the script enter a verbose mode. If neither ```-d``` nor ```-v``` is provided, the script will throw an error. 

Fore more details on the options provided, run ```python DSA.py -h``` or ```python DSA.py --help```. For instance, a run of ```python DSA_Bayesian.py -h``` yields 
```bash
Usage: python DSA_Bayesian.py -d <datafile>

Options:
  -h, --help            show this help message and exit
  -d DATAFILE, --data-file=DATAFILE
                        Name of the data file.
  -l LOCATION, --location=LOCATION
                        Name of the location.
  -m, --mpi             Indicates whether to use MPI for parallelization.
  -o OUTPUT_FOLDER, --output-folder=OUTPUT_FOLDER
                        Name of the output folder
  -s, --smooth          Indicates whether the daily counts should be smoothed.
  -f LAST_DATE, --final-date=LAST_DATE
                        Last day of data to be used
  -r, --estimate-recovery-parameters
                        Indicates the parameters of the recovery distribution
                        will be estimated
  -N N                  Size of the random sample
  -T T, --T=T           End of observation time
  --day-zero=DAY0       Date of onset of the epidemic
  --niter=NITER         Number of iterations of the MCMC
  --threads=THREADS     Number of threads for MPI
  -v, --verbose         Runs with default choices
```


The easiest way to run our model is to open one of the Jupyter notebooks and run the cells. Please modify the commands as needed.

Alternatively, perform the following: 
1. (Recommended) The Bayesian model can be run by invoking 
```bash
python DSA_Bayesian.py -d <datafile>
```
from the terminal. 

2. The maximum likelihood based DSA model can be run by invoking
```bash
python DSA.py -d <datafile>
```
from the terminal.

3. The semi-Bayesian Laplace approximation to the posterior distribution of the parameters can be carried out by running the following command
```bash
python DSA_Laplace.py -d <datafile>
```
from the terminal.

## Examples
We provide two examples. 
1. The first example extracts count data from a repository maintained by the New York Times. This example fits the Bayesian DSA model. 

2. The second example works on a dummy data set and runs the basic DSA model. 
