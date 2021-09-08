#%% 
import sys
from pathlib import Path
sys.path.insert(0, str(Path('__file__').resolve()))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import pandas as pd
import warnings
import copy

from lyra._base.data import water
from lyra._base.utils import bootfold, coefficient, show_accuracy, evaluate_classif, writeout, get_table_cols
from lyra._base.error import AttributeDoesNotExist

#%%
warnings.filterwarnings("ignore")

class Classifier:
    def __init__(self, model, X, y, epoch=100, batch=0.75, metric='f1'):
        self.__model, self.__X, self.__y = model, X, y
        self.__feat_names_ = list(X.columns)
        self.__num_obs_ = len(y)
        self.__predicted = self.__metrics = self.__confusion = \
        self.__full = self.__coef = self.__partial = self.__table = None
        self.__full = self.__model.fit(X, y)
        self.__partial = self.__partial_run(epoch, batch, metric)
        self.__tabulate()
        self.__evaluate_coef()
    def partial_fit(self, epoch=100, batch=0.75, metric='f1'):
        self.__partial = self.__partial_run(epoch, batch, metric)
        self.__tabulate()
        self.__evaluate_coef()    
    def predict(self, X_test, y_test):
        self.__predicted = self.__model.predict(X_test)
        self.__confusion = confusion_matrix(y_test, self.__predicted)
        self.__metrics = evaluate_classif(y_test, self.__predicted)
    def __partial_run(self, epoch, batch, metric):
        if batch < 1: batch = round(batch*self.__num_obs_)
        trX, trY, tsX, tsY = bootfold(epoch, batch, self.__X, self.__y)
        assert(len(trX) == len(trY) == len(tsX) == len(tsY))
        run = []
        for i in tqdm(range(len(trX)), file=sys.stdout):
            with writeout():
                self.__model.fit(trX[i], trY[i])
                pred = self.__model.predict(tsX[i])
                show_accuracy(epoch, i, metric, tsY[i], pred)
                cm = confusion_matrix(tsY[i], pred)
                cf = self.__model.coef_[0].tolist()
                met = evaluate_classif(tsY[i], pred)
                run.append({'cm':cm,'coef':cf,'metric':met})
        return run
    def __evaluate_coef(self):
        eval = []
        for i in range(len(self.__partial[0]['coef'])):
            estimates = [j['coef'][i] for j in self.__partial]
            if self.__full is not None:
                eval.append(coefficient(estimates, self.__full.coef_[0][i]))
        self.__coef = pd.DataFrame(eval)
        self.__coef.insert(0,'feat-names',self.__feat_names_)
    def __tabulate(self):
        self.__table = None
        data = []
        for run in self.__partial:
            entry = []
            cm, met, cf = run['cm'], run['metric'], run['coef']
            entry.extend(cm.ravel())
            entry.extend([float(met['f1']), float(met['accuracy']), 
                          float(met['roc_auc']), float(met['precision']), 
                          float(met['bal_accuracy'])])
            entry.extend(cf)
            data.append(entry)
        col_names = copy.deepcopy(get_table_cols('binary'))
        col_names.extend(self.__feat_names_)
        self.__table = pd.DataFrame(data, columns=col_names)
    @property
    def full_(self):
        if self.__full is None:
            raise AttributeDoesNotExist('.fit(..., partial=False)')
        return self.__full
    @property
    def coef_(self):
        if self.__coef is None:
            raise AttributeDoesNotExist('.fit()')
        return self.__coef
    @property
    def table_(self):
        if self.__table is None:
            raise AttributeDoesNotExist('.fit()')
        return self.__table
    @property
    def predicted_(self):
        if self.__predicted is None:
            raise AttributeDoesNotExist('.predict()')
        return self.__predicted
    @property
    def metrics_(self):
        if self.__metrics is None:
            raise AttributeDoesNotExist('.predict()')
        return self.__metrics
    @property
    def confusion_(self):
        if self.__confusion is None:
            raise AttributeDoesNotExist('.predict()')
        return self.__confusion
