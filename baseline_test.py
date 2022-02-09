import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit
from sklearn import metrics

import multiprocessing

import pickle

import dataloader
import method
import ResultManager
import config

def run(X,y):
    batch_size = 64
    epochs = 20
    inp_dim = X.shape[1]
    model = method.MLModels(inp_dim, batch_size, epochs)
    model.fit(X, y)
    return model

def main():
    # df, features, Y = dataloader.load_multi_lungpancreasliver_primarymetastasis()
    # df = pd.concat([df,Y],axis=1)
    # df.to_csv('input/baseline_experiment.csv',sep=',',index=False)

    df = pd.read_csv('input/baseline_experiment.csv',sep=',')
    features = dataloader.load_feature_names()
    data = df[df['Cohort'].isin(['FAMDm','MSKm'])]
    
    X = data[features].values
    Y = data[config.outcome_colnames_1]
    
    result = {}    
    for label in Y.columns:
        print("{} classification".format(label))
        y = Y[label].values
        if len(np.unique(y)) == 1:
            continue
        result[label] = run(X,y)

    with open(config.baseline_result_out_filename, 'wb') as outp:
        pickle.dump(result, outp)

if __name__ == '__main__':
    main()
    
    
