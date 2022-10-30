import csv
import numpy as np
import ast

file_name='text_rank_results_225.csv'

with open(file_name) as f:
    reader = csv.DictReader(f)

    results_dict = [row for row in reader] 
    results = [np.mean(list(ast.literal_eval(row['results']).values())) for row in results_dict]
    i = np.argmax(results)
    print(results_dict[i])
