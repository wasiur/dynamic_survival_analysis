# Dynamic Survival Analysis

This repository provides a Python implementation of the dynamic survival analysis method. **This is primarily based on a package (available [here](https://github.com/calebbastian/epidemic)) developed by Caleb Deen Bastian, Princeton University.** I acknowledge the help of Saket Gurukar, the Ohio State University, to parallelize some of the routines.

## Installation
1. Please make sure you have Python (version 3.6.x and above). If you do not have Python, we recommend installing it from Anaconda (link [here](https://www.anaconda.com/distribution/)).
2. You can download our package either by hitting download or by cloning our repository. Cloning can be done by running the following command
```bash
git clone https://github.com/wasiur/dynamic_survival_analysis.git
```
from your terminal. Please provide userid and password when prompted.

3. Our implementation depends on a number of packages. In order for the parallelization to run smoothly, we recommend installing the following python _environment_ "dynamic_survival_analysis". This is included in the file _environment.yml_. If you are using Anaconda (recommended), the environment can be installed by running
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

## Running the dynamic survival analysis model
