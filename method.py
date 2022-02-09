from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit, GridSearchCV, cross_val_score


import keras
import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Dense, GRU, Input, Lambda,Dropout, LSTM
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.utils import to_categorical

import config


class DefaultModel(object):
    def __init__(self, input_dim):
        inp = Input(shape=(input_dim,))
        outp = self._build_model1(inp)
        model = Model(inp, outp)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model = model
        
    def _build_model1(self, inp):
        z = Dense(40, activation='relu')(inp)
        z = Dropout(.2)(z)
        z = Dense(40, activation='relu')(z)
        z = Dropout(0.2)(z)
        z = Dense(30, activation='relu')(z)
        z = Dropout(.2)(z)
        z = Dense(20, activation='relu')(z)
        z = Dense(2, activation='softmax')(z)
        return z

    def _build_model2(self, inp):
        z = Dense(500, activation='tanh')(inp)
        z = Dense(100, activation='relu')(z)
        z = Dense(2, activation='softmax')(z)
        return z

    def fit(self, X, y, batch_size, epochs):
        y = to_categorical(y)
        self.model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='auto',verbose=0)])
        #self.model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1)


    def predict(self, X):
        return self.model.predict(X)


class MLModels(object):
    def __init__(self, inp_dim, batch_size, epochs):
        models = {}
        models['logit'] = LogisticRegression()
        models['lasso'] = LogisticRegression(penalty='l1', C=1, solver='liblinear')
        models['ridge'] = LogisticRegression(penalty='l2', C=1)
        models['rf'] = RandomForestClassifier()
        #models['gb'] = GradientBoostingClassifier()
        models['svm'] = svm.SVC()
        models['dense'] = DefaultModel(inp_dim)
        
        self.models=models
        self.val = {}
        #self.model_names = ['logit','lasso','ridge','rf','gb','svm', 'dense']
        self.model_names=['logit','lasso','ridge']
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, x_train, y_train):
        for model_name in self.model_names:
            print("{} fitting...".format(model_name))
            self.fit_(model_name, x_train, y_train)

    def fit_(self, model_name, X, y):
        model = self.models[model_name]

        if model_name == 'logit':
            scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=StratifiedKFold(n_splits=5), n_jobs=10)
            self.val[model_name] = scores
        elif model_name in ['lasso','ridge']:
            clf = GridSearchCV(model, config.parameters[model_name], cv=StratifiedKFold(n_splits=5), verbose=1, n_jobs=10, scoring='balanced_accuracy')
            clf.fit(X,y)
            self.val[model_name] = clf
        # elif model_name == 'rf':
        #     pass
        # elif model_name == 'gb':
        #     pass
        # elif model_name == 'svm':
        #     pass

    # def evaluate_(self, model_name, X, y):
    #     pass
        
    # def evaluate(self, x_test, y_test):
    #     pass
    #     for model_name in self.model_names:
    #         if model_name == 'dense':
    #             y_pred = self.models[model_name].predict(x_test)[:,1]
    #         else:
    #             y_pred = self.models[model_name].predict_proba(x_test)[:,1]

    #         fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    #         auprc = metrics.average_precision_score(y_test, y_pred)
    #         auroc = metrics.auc(fpr, tpr)
    #         self.auroc[model_name]=auroc
    #         self.auprc[model_name]=auprc
            
    def __str__(self):
        output = ''
        for model_name in self.model_names:
            output += '{}: auroc({}), auprc({})\n'.format(model_name, self.auroc[model_name], self.auprc[model_name])
        return output
