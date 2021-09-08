## Classification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score, precision_score
## Regression
from sklearn.metrics import  mean_squared_error, r2_score, explained_variance_score, max_error, mean_absolute_error

import pandas as pd

METRICS = {
    # Classification
    'accuracy': accuracy_score, 
    'f1': f1_score, 
    'roc_auc': roc_auc_score, 
    'bal_accuracy': balanced_accuracy_score, 
    'precision': precision_score,
    # Regression
    'mse': mean_squared_error, 
    'r2': r2_score, 
    'exp_var': explained_variance_score, 
    'max_err': max_error, 
    'mae': mean_absolute_error
}

TABLE_COL = {
    'binary' : ['TN', 'FP', 'FN', 'TP', 'F1', 'Accuracy', 'ROC/AUC', 'Precision', 'Bal_Accuracy']
}

CLASSIF = { 'accuracy', 'f1','roc_auc', 'bal_accuracy', 'precision'}

def show_accuracy(epoch, run, metric, true, pred):
    score = METRICS[metric]
    print("{:<1} {:<3} {:<17} {:<3} {:<1} {:<1} {:<1}".format(metric, 'score:',
                         round(score(true, pred),7),'- epoch',run+1,'of',epoch))
    
def evaluate_classif(y_true, y_pred):
    eval = {}
    for i in CLASSIF:
        eval[i] = float(METRICS[i](y_true, y_pred))
    return pd.DataFrame(eval, index=[0])

def get_table_cols(type):
    return(TABLE_COL[type])

