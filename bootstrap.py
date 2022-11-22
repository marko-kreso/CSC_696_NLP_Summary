import json
from pathlib import Path
import numpy as np
import pandas as pd


def main(data, max_iter):
    test = data.groupby(['article_index', 'model']).mean()
    print(type(test))
    print(test.columns)
    print(test.loc[0])
    num_articles = np.unique(test.index.get_level_values('article_index'))
    num_better = [0,0]
    for i in range(max_iter):
        #Randomly sample with replacement
        sample = test.unstack().sample(len(num_articles),replace=True).stack()
        results = sample.groupby(['model']).mean()
        print(results)
        num_better[0] += sum(np.array(results.loc['text-BART'][:]) >= np.array(results.loc['base-BART'][:]))
        num_better[1] += sum(np.array(results.loc['text-BART'][:]) >= np.array(results.loc['text-rank'][:]))
        assert(False)

    print(num_better)
    print(np.array(num_better)/(max_iter*4))




if __name__ == '__main__':
    path = Path('./BOOT_EVAL_200')
    info = json.load(path.open('r'))

    new_data = list()

    i = 0
    for data in info['data']:
        for k,v in data.items():
            test = dict(v)
            test['model'] = k 
            test['article_index']= i
            new_data.append(test)
        i += 1

    main(pd.json_normalize(new_data), 1000)