import numpy as np


non_features = ['Cohort', 'Sample', 'Mutcnt', 'Psite', 'Ptype', 'Msite', 'Sex', 'Depth','CancerType', 'CancerType_Detail']
outcome_1 = {'primary': ['Lung, NOS', 'Pancreas, NOS','Lung','Pancreas','Liver', 'Prostate gland', 'Prostate'],
             'metastasis': ['Lung, NOS','Pancreas, NOS', 'Lung', 'Liver', 'Prostate gland']}
outcome_colnames_1 = ['P:Lung, NOS', 'P:Pancreas, NOS', 'P:Lung', 'P:Pancreas', 'P:Liver','P:Prostate gland', 'P:Prostate', 'M:Lung, NOS', 'M:Pancreas, NOS', 'M:Lung', 'M:Liver', 'M:Prostate gland' ]

baseline_result_out_filename = 'baseline_result.pkl'

parameters= {'lasso': {'C': np.arange(0.01, 10, 0.1)},
             'ridge': {'C': np.arange(0.01, 10, 0.1)},
             'svm': {'kernel': ['poly','rbf','sigmoid','linear'], 'gamma':'auto'}}
