
# Dynamic Survival Analysis

This repository provides a Python implementation of the dynamic survival analysis method. A brief description of the methodology can be found in this [white paper](https://idi.osu.edu/assets/pdfs/covid_response_white_paper.pdf). Prof. Greg Rempała gave a couple of public talks on this model. You can watch his [MBI](https://mbi.osu.edu/) seminar talks here: [link to his first talk](https://video.mbi.ohio-state.edu/video/player/?id=4888&title=Mathematical+Models+of+Epidemics%3A+Tracking+Coronavirus+using+Dynamic+Survival+Analysis) and [link to his second talk](https://video.mbi.ohio-state.edu/video/player/?id=4891&title=Mathematics+of+Modeling+a+Pandemic%3A+The+Journey+Continues).


**This is primarily based on a package (available [here](https://github.com/calebbastian/epidemic)) developed by Caleb Deen Bastian, Princeton University.** I also acknowledge Saket Gurukar, who helped with the parallelization of some of the routines.


If you have questions, comments, criticisms, or corrections, please email me at [khudabukhsh.2@osu.edu](mailto:khudabukhsh.2@osu.edu).

## Installation and setup
1. **Get files.** You can download our package either by hitting download or by cloning our repository. Cloning can be done by running the following command
	```bash
	git clone https://github.com/wasiur/dynamic_survival_analysis.git
	```
	from your terminal. Alternatively, you can download this git repository and unzip it at the destination folder.
 2. **Get dependencies.** Our implementation depends on a number of packages. In order for the parallelization to run smoothly, we recommend installing the following python environment "dynamic_survival_analysis". 
 
	**Way 1: Using Python to deploy. (Linux)** The Python environment  (version 3.6.x and above) should be installed sequentially, the required list of Python package can be obtained from the file [_dsa.yml_](https://github.com/wasiur/dynamic_survival_analysis/blob/master/dsa.yml). It is recommended to install each package with 
	```bash
	python -m pip install --upgrade mymodulename
	```
	to avoid the error mentioned [here](https://stackoverflow.com/questions/15052206/python-pip-install-module-is-not-found-how-to-link-python-to-pip-location). This is the most reliable way to deploy on Linux platform.
	
	Since the mingw on Windows is not fully compatible, if you must use Windows platform, please use way 2 below.
	
	**Way 2: Using Anaconda to deploy. (Windows/macOS)** The Anaconda environment is included in the file _dsa.yml_ (however, the file _environment.yml_ contains the specific environment we use for macOS platform). If you are using Anaconda (4.5 or above is recommended), the environment can be installed by running
	```bash
	conda env create -f dsa.yml
	```
	or you can use "Import" function in the anaconda-navigator interface. In order to check if the environment is now available, run
	```bash
	conda env list
	```
	After you activate the dsa environment,  you would see your bash prompt with '(dsa)' like
	```bash
	(dsa) [root]$ 
	```
	***Troubleshooting:*** 
	If you encounter *ResolvePackageNotFound* error, please see [here](https://github.com/datitran/object_detector_app/issues/41)  and [there](https://github.com/conda/conda/issues/9611).	
	If you encounter conda-yml error, please see [here](https://stackoverflow.com/questions/55554431/conda-fails-to-create-environment-from-yml). In a newer version of Anaconda (especially on Windows/Linux platform), try 
	```bash
	conda config --set restore_free_channel true
    ```
	Upon successfully deployment of the Anaconda environment, you will see following lines:
	```bash
	Preparing transaction: done
	Verifying transaction: done
	Executing transaction: done
	```
	Activate the environment "dynamic_survival_analysis" by running
	```bash
	conda activate dynamic_survival_analysis
	```
	from your terminal.
3. **Testing.** After deployment, we can test a minimal example
	```bash
	python DSA_Bayesian.py -d dummy.csv
	```
	You would expect output in the folder "_plots_dsa_bayesian_" after a patient wait (>30 minutes).



## Data preparation
### Real data format

A typical input data as a .csv file acceptable to the model should have following seven columns:

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

### Simulated data format
We can use [_SEIR2HMC_rev.r_](https://github.com/wasiur/dynamic_survival_analysis/blob/master/cluster/date%20prepartion/SEIR2HMC_rev.r) to convert SIR (or SEIR with slight modification) format datasets into acceptable csv file.


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

## Distributed Computation
We provide cluster sample script that applies to [TORQUE](https://en.wikipedia.org/wiki/TORQUE) based cluster at [this folder](https://github.com/wasiur/dynamic_survival_analysis/tree/master/cluster). A typical cluster script would look like
```bash
#!/bin/bash
#PBS -j oe 
#PBS -m abe
#PBS -M abc@def.ghi

#COMPILER MODULE
module load intel/16.0.3 

#FIRST-TIME ENVIRONMENT SETUP
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh

#SETUP CONDA BASH
source /home/username/miniconda3/etc/profile.d/conda.sh

#CREATE CONDA ENVIRONMENT IF FIRST-TIME
#conda env create -f /home/username/dynamic_survival_analysis/dsa.yml

#ACTIVATE DSA ENVIRONMENT
conda activate dsa
cd /home/username/dynamic_survival_analysis/

#RUN DSA ANALYSIS
#'-m False' disables the MPI, if your nodes support MPI, you can change it to '-m True'.
python /home/username/dynamic_survival_analysis/DSA_Bayesian.py -d HMC_CM_DataPoisson_35_Weibull_125_05_Gamma_45_05_nS1000_nI10.csv -o HMC_CM_DataPoisson_35_Weibull_125_05_Gamma_45_05_nS1000_nI10 -m False 
ls

```

Save these commands as a single pbs job script [DSA.pbs](https://github.com/wasiur/dynamic_survival_analysis/blob/master/cluster/bash%20prepartion/DSA.pbs). To submit the job, use following sample qsub command.
```bash
qsub -l walltime=240:00:00,nodes=1:ppn=4,mem=48GB /home/username/dynamic_survival_analysis/DSA.pbs
```
More details are available at [TORQUE documents](http://docs.adaptivecomputing.com/torque/4-1-3/Content/topics/12-appendices/commandsOverview.htm).

