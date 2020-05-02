'''
This example performs the Laplace approximation on a dummy data. 
'''

import os
location = 'Ohio'
datafile = 'dummy.csv'
output_folder = 'plots'
last_date = '2020-03-22'

command_string = 'time python DSA.py -r -d ' + datafile + ' -l ' + location + ' -o ' + output_folder + ' --final-date=' + last_date
os.system(command_string)


