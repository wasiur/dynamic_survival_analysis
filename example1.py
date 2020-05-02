'''
This example fits the dynamic survival analysis model to the count data available from a repository maintained by the New York Times.
The link to the repository is available here:

https://github.com/nytimes/covid-19-data

In this example, we use the data from the state of Ohio.
'''

import os
location = 'Ohio'
datafile = 'oh_040820.csv'
output_folder = 'plots_bayesian'
last_date = '2020-04-07'

command_string = 'time python DSA_Bayesian.py -r -d ' + datafile + ' -l ' + location + ' -o ' + output_folder + ' --final-date=' + last_date
os.system(command_string)

