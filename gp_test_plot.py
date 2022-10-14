import numpy as np
import matplotlib.pyplot as plt
import csv
import os

current_path = os.getcwd()
csv_file = open('data_analyse/GP_result.csv', 'r')
dict_file = csv.DictReader(csv_file)
csv.field_size_limit(100000000)
var_list = [var for var in dict_file.fieldnames if var != '']

for var in var_list:
    globals()['li_' + var] = []

for row in dict_file:
    for var in var_list:
        temp_value = float(row[var])
        eval('li_' + var + '.append(temp_value)')

for var in var_list:
    globals()['np_' + var] = np.array(eval('li_' + var)).reshape(-1, 1)


plt.figure()
plt.plot(np_t, np_mu, '*r', label='mu')
plt.plot(np_t, np_prediction_state, 'b', label='data_set')
plt.title('gaussian process')
plt.legend()
plt.show()