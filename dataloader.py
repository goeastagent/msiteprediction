import numpy as np
import pandas as pd

import config


def load_data():
    data = pd.read_csv('data/threem_allpr_matrix.csv',sep=',')
    features = load_feature_names()
    return data, features

def load_feature_names():
    features = pd.read_csv('feature_set_noheader.csv',header=None).iloc[:,0].tolist()
    return features

def load_multi_lungpancreasliver_primarymetastasis():
    df, features = load_data()    
    primary_list = config.outcome_1['primary']
    metas_list = config.outcome_1['metastasis']

    phenotype_matrix = {}
    # primary
    for primary_cancer in primary_list:
        phenotype_matrix['P:'+primary_cancer] = (df['Psite'] == primary_cancer).astype(int)
    # metastasis
    for metas_cancer in metas_list:
        phenotype_matrix['M:'+metas_cancer] = (df['Msite'] == metas_cancer).astype(int)

    phenotype_matrix = pd.DataFrame(phenotype_matrix)
    return df, features, phenotype_matrix
