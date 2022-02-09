import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

import pickle

import config
import dataloader
import method


df, features = dataloader.load_data()


def plot_variants_sum():
    df[features].sum(axis=1)    

def plot_explained_ratio(cohort_list):
    data = df[df['Cohort'].isin(cohort_list)]
    pca = PCA(svd_solver='randomized')
    pca.fit(data[features].values)
    explained_ratio = pca.explained_variance_ratio_

    y = np.cumsum(explained_ratio)
    x = range(len(explained_ratio))

def load_result():
    with open(config.baseline_result_out_filename, 'rb') as inp:
        result=pickle.load(inp)

        
    
if __name__ == '__main__':
    plot_explained_ratio(['FMADm','MSKm'])
    plot_explained_ratio(['FMADpr','FMADm'])
    plot_explained_ratio(['FMADpr','MSKpr'])