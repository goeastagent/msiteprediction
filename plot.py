import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os

from sklearn.decomposition import PCA

import pickle

import config
import dataloader
import method


df, features = dataloader.load_data()

def check_availability():
    if not os.path.isdir(config.image_out_dir):
        os.mkdir(config.image_out_dir)

def plot_variance_density():
    variance = df[features].var(axis=0)
    sns.kdeplot(variance)
    plt.savefig(config.image_out_dir + 'variance_density')
    
def plot_explained_ratio(cohort_list):
    data = df[df['Cohort'].isin(cohort_list)]
    pca = PCA(svd_solver='arpack')
    pca.fit(data[features].values)
    explained_ratio = pca.explained_variance_ratio_

    y = np.cumsum(explained_ratio)
    x = range(len(explained_ratio))
    plt.plot(x,y)
    plt.title(cohort_list)
    plt.savefig(config.image_out_dir + '_'.join(cohort_list))

def load_result():
    with open(config.baseline_result_out_filename, 'rb') as inp:
        result=pickle.load(inp)
    return result


def plot_performance():
    result = load_result()

    plot_df = pd.DataFrame()
    for label in result.keys():
        for method in result[label].val.keys():
            result[label].val[method].cv_result_
            r = pd.DataFrame(result[label].val).mean(axis=0).to_frame().T
            
            plot_df = pd.concat([plot_df, r], axis=0)
    plot_df['label'] = result.keys()
    plot_df.plot(x='label',kind='bar',stacked=False)
    plt.ylim([0,.8])
    plt.tight_layout()
    plt.savefig(config.image_out_dir + 'balanced_accuracy')
    
if __name__ == '__main__':
    check_availability()
    
    plot_explained_ratio(['FMADm','MSKm'])
    plot_explained_ratio(['FMADpr','FMADm'])
    plot_explained_ratio(['FMADpr','MSKpr'])
    plot_explained_ratio(['MSKpr','MSKm'])

    plot_variance_density()
