from sklearn.utils import resample
from statistics import mean, stdev
from scipy.stats import norm

## Bootstrap / K-Fold combination
def bootfold(epoch, batch, X, y):
    ind = list(range(len(y)))
    train_X, train_y, test_X, test_y = [], [], [], []
    replace = False
    if(batch > len(y)):
        replace = True
    for i in range(epoch):
        tr = resample(ind, replace=replace, n_samples=batch, random_state=i)
        tr.sort()
        ts = [j for j in ind if j not in tr]
        
        train_X.append(X.loc[tr,])
        train_y.append(y.loc[tr,])
        test_X.append(X.loc[ts,])
        test_y.append(y.loc[ts,])

        ind = [j for j in ind if j not in tr]
        
        if(len(ind) < 2*batch): ind = list(range(len(y)))
        
    return train_X, train_y, test_X, test_y

def split(X, y, batch):
    ind = list(range(len(y)))
    train_X, train_y, test_X, test_y = [], [], [], []
    replace = False
    if batch < 1:
        batch = round(len(y)*batch)
    tr = resample(ind, replace=False, n_samples=batch, random_state=43)
    tr.sort()
    ts = [j for j in ind if j not in tr]
        
    return X.loc[tr,].reset_index(drop=True), X.loc[ts,].reset_index(drop=True), y.loc[tr,].reset_index(drop=True), y.loc[ts,].reset_index(drop=True)

def coefficient(values, estimate=0):
    mu = mean(values)
    sd = stdev(values)
    z = (mu-estimate)/sd
    p = norm.sf(abs(z))*2 
    
    return {'estimate': estimate, 'mean':mu, 'sd':sd, 'z-stat':z, 'p-value':p}
    
