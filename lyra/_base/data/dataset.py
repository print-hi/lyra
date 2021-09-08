#%%
import os, sys
import pandas as pd
import numpy as np 

def get_path():
    path = os.path.abspath(os.path.dirname(''))
    if path[-4:] == 'lyra': return 'lyra/_data/'
    elif path[-8:] == 'lyra/lyra': return '_data/'
    elif path[-5:] == '_data': return ''
    elif path[-5:] == 'usage': return path[:-5] + 'lyra/_data/'
    elif path[-5:] == 'tests': return path[:-5] + 'lyra/_data/'
    else: raise(ImportError)
    
def water():
    data = pd.read_csv(get_path() + "water.csv")
    features = list(data.columns)
    features.remove('Potability')
    return data.drop(columns=['Potability']), data.drop(columns=features), None

def linearData(n_obs=1000, n_var=25, sparse=10, classif=False, multi_dist=False):
    X = []
    y = []
    
    n_noise = n_var % 3 + 3
    n_var = n_var - n_noise
    rng = np.random.default_rng(seed=43)
    params = rng.uniform(1,sparse,n_var)

    if multi_dist:
        norm = []
        exp = []
        poi = []
        noise = []
        for i in range(int((n_var)/3)):
            norm.append(rng.normal(params[i], sparse*params[n_var-(i+1)], n_obs))
            exp.append(rng.exponential(sparse*params[i], n_obs))
            poi.append(rng.poisson(sparse*params[i], n_obs))
        for i in range(n_noise):
            noise.append(rng.uniform(-sparse, sparse, n_obs))
        X.extend(norm)
        X.extend(exp)
        X.extend(poi)
        X.extend(noise)
    else:
        norm = []
        noise = []
        for i in range(int((n_var))):
            norm.append(rng.normal(params[i], sparse*params[n_var-(i+1)], n_obs))
        for i in range(n_noise):
            noise.append(rng.uniform(-sparse, sparse, n_obs))
        X.extend(norm)
        X.extend(noise)
    
    coef = rng.uniform(-sparse, sparse, n_var)
    for i in range(n_obs):
        val = 0
        for j in range(n_var):
            val += coef[j]*X[j][i]
        y.append(val)
    
    threshold = None
    if classif:
        threshold = rng.uniform(min(y), max(y), 1)
        for i in y:
            if i - threshold < 0: i = 0
            else: i = 1
        
    return pd.DataFrame(X).T.rename(columns = lambda x: 'X'+str(x)), \
           pd.DataFrame(y).rename(columns = lambda x: 'y'), \
           {'coef': coef, 'threshold': threshold}