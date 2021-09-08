<img src="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/LYRA-y.svg"> 

# its kinda like keras *but worse* (＾◡＾)っ ♡

Python package for machine learning models - backend implementation in pure C++. GLM is the only available model at the moment; currently fits models up to twice as fast as other popular libraries such as scikit-learn and statsmodel while retaining accuracy levels (only tested on balanced, medium-sized datasets so far).

# Installation

```bash
Installation has not been tested on other OS. 

Optimised implementation with complete suite of models coming soon.

If you still wish to install, proceed at your own risk.
```

### Dependencies

- Python (>= 3.7)
- scikit-learn (>= 0.20)
- NumPy (>= 1.14.6)
- SciPy (>= 1.1.0)
- joblib (>= 0.11)
- threadpoolctl (>= 2.0.0)
- tqdm (>= 4.62.0)

### User installation

Install using `pip`:
```bash
pip install -U lyra
```
or `conda`:
```bash
conda install -c conda-forge lyra
```
See https://lyra/installation/gethelp
    
> Haha sike, we haven't deployed anything yet

-------------------------------------------

## Algorithm Implementation
### Generalised Linear Model

> Equations will not be visible for light-mode users as LaTex renders in white font, reload on dark-mode to see loss functions.

GLM assumes that observations come from a distribution from the exponential dispersion family/model. Distributions from the exponential dispersion family/model can be shown to have pmf/pdf of the form: &nbsp;
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20%5Ccolor%7BWhite%7D%20%5Ctextit%7Bf%7D%28y_i%3B%20%5Ctheta_i%29%20%3D%20%5Cexp%20%5Cleft%5B%20%5Cdfrac%7By_i%20%5Ctheta_i%20-%20%5Ctextit%7Bb%7D%28%5Ctheta_i%29%7D%7B%5Ctextit%7Ba%7D_i%28%5Cphi%29%7D%20&plus;%20c%28y_i%3B%20%5Cphi%29%20%5Cright%5D" />
</p> <br />
To build the spread around the linear model, we can then make use of various differentiable transformations (injective) using the following relationship: <br/><br/>
<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20%5Ccolor%7BWhite%7D%20%5Ceta_i%20%3D%20g%28%5Cmu_i%29%20%3D%20g%28%5Cboldsymbol%7B%5Cmathbf%7Bx%7D%7D_i%27%20%7B%5Cboldsymbol%20%5Cbeta%7D%29%20%3D%20g%28%5Cmathbb%7BE%7D%5B%7B%5Cmathit%7BY_i%7D%7D%5D%29" />
</p> <br/>
To fit the model, iteratively reweighted least squares can be used with the following adjusted dependent variable: <br/><br/>
<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20%5Ccolor%7BWhite%7D%20%5Cmathnormal%7Bz%7D_i%20%3D%20%5Ceta_i%20&plus;%20%28%7B%5Cmathrm%7By%7D%7D_i%20-%20%5Cmu_i%29%20%5Cdfrac%7Bd%20%5Ceta_i%7D%7Bd%5Cmu_i%7D" />
</p> 

> When canonical links are used, the Hessian matrix concide so that the Fisher scoring method and Newton-Raphson method reduce to the same algorithm (McCullagh, P. and Nelder, J. A. (1989). Generalized Linear Models, Vol. 37 of Monographs on Statistics and Applied Probability, 2 edn, Chapman and Hall, London)

<br/>

Thereafter, we need to calculate the iterative weights, given by the diagonal matrix with entries: <br/><br/>
<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20%5Ccolor%7BWhite%7D%20%5Ctextit%7Bw%7D_i%20%3D%20%5Cdfrac%7B%5Cphi%20%5Ccdot%20b%5E%7B%27%27%7D%28%5Ctheta_i%29%7D%7B%5Ctextit%7Ba%7D_i%28%5Cphi%29%7D%20%5Ccdot%20%28%5Cdfrac%7Bd%20%5Ceta_i%7D%7Bd%5Cmu_i%7D%29%5E%7B-2%7D" />
</p><br/>

This leaves us with the following minimisation task: <br/><br/>
<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Clarge%20%5Ccolor%7BWhite%7D%5Cboldsymbol%20%5Cbeta%20%3D%20%28%5Cboldsymbol%7B%5Cmathrm%7BX%7D%7D%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5Cmathrm%7BW%7D%7D%20%5Cboldsymbol%7B%5Cmathrm%7BX%7D%7D%29%5E%7B-1%7D%20%5Cboldsymbol%7B%5Cmathrm%7BX%7D%7D%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5Cmathrm%7BW%7D%7D%20%5Cboldsymbol%7B%5Cmathrm%7Bz%7D%7D" />
</p><br/>

> To speed up calculations, we can make use of matrix decompositions. This is vital for optimisation as calculating inversions for large matricies is very costly, and instead we can aim to inverse triangle matricies and make use of the diagonalisation in the weight matrix. 

<br/>

##### Cholesky Factorization 

<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Clarge%20%5Ccolor%7BWhite%7D%20%7B%5Cmathcal%7BR%7D%7D%5E%7B%5Ctop%7D%7B%5Cmathcal%7BR%7D%7D%20%3A%3D%20%5Cboldsymbol%7B%5Cmathrm%7BX%7D%7D%5E%7B%5Ctop%7D%5Cboldsymbol%7B%5Cmathrm%7BW%7D%7D%5Cboldsymbol%7B%5Cmathrm%7BX%7D%7D%20%5Clongrightarrow%20%5Cboldsymbol%20%5Cbeta%20%3D%20%7B%5Cmathcal%7BR%7D%7D%5E%7B-1%7D%7B%5Cmathcal%7BR%7D%7D%20%5E%7B-%5Ctop%7D%20%5Cboldsymbol%7B%5Cmathrm%7BX%7D%7D%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5Cmathrm%7BW%7D%7D%20%5Cboldsymbol%7B%5Cmathrm%7Bz%7D%7D" />
</p>

##### QR Decomposition

<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Clarge%20%5Ccolor%7BWhite%7D%20%7B%5Cmathcal%7BQ%7D%7D%7B%5Cmathcal%7BR%7D%7D%20%3A%3D%20%5Cboldsymbol%7B%5Cmathrm%7BW%7D%7D%5E%7B%5Cfrac%7B1%7D%7B2%7D%7D%5Cboldsymbol%7B%5Cmathrm%7BX%7D%7D%20%5Clongrightarrow%20%5Cboldsymbol%20%5Cbeta%20%3D%20%7B%5Cmathcal%7BR%7D%7D%5E%7B-1%7D%7B%5Cmathcal%7BQ%7D%7D%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5Cmathrm%7BW%7D%7D%20%5Cboldsymbol%7B%5Cmathrm%7Bz%7D%7D" />
</p>

##### Column-Pivoting

<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Clarge%20%5Ccolor%7BWhite%7D%20%7B%5Cmathcal%7BQ%7D%7D%7B%5Cmathcal%7BR%7D%7D%7B%5Cmathcal%7BP%7D%7D%20%3A%3D%20%5Cboldsymbol%7B%5Cmathrm%7BX%7D%7D%5Cboldsymbol%7B%5Cmathrm%7BW%7D%7D%5E%7B%5Cfrac%7B1%7D%7B2%7D%7D%20%5Clongrightarrow%20%5Cboldsymbol%20%5Cbeta%20%3D%20%7B%5Cmathcal%7BP%7D%7D%7B%5Cmathcal%7BR%7D%7D%5E%7B-1%7D%7B%5Cmathcal%7BQ%7D%7D%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5Cmathrm%7BW%7D%7D%20%5Cboldsymbol%7B%5Cmathrm%7Bz%7D%7D" />
</p>

-------------------------------------------

## Machine Learning Interface
### [Classifier](https://github.com/print-hi/lyra/blob/usage/usage/classifier.ipynb) <a name="classifier"></a>

#### Class Parameters

| Parameters | Type | Description |
| ---- | --- | --- |
| *model*  | *list of \<Model\>*  | List of learning models |
| *X*  | *pd.DataFrame*  | Matrix of input values |
| *y*  | *pd.Dataframe, list*   | Vector of binary outcomes |

#### Class Attributes

| Attributes | Type | Description |
| ---- | --- | --- |
| *coef_*  | *list*    | Data on coefficients |
| *full_*  | *\<Model\>*   | Fitted full model |
| *partial_*  | *list.\<dict\>*   | Results from partial runs |
| *table_*  | *pd.Dataframe*   | Tabulated results of partial runs |
| *predicted_*  | *list*  | Predicted values of last predicted X_test |
| *metrics_*  | *pd.Dataframe*  | Evaluation metrics of last predicted X_test |
| *confusion_*  | *numpy.ndarray*  | Confusion matrix of last predicted X_test |

#### Class Methods
Predict and evalute a set of inputs using best model from most recent \<Classifier\>.fit() call. 

```python     
<Classifier>.predict(X_test, y_test)
```
> Related Attributes: predicted_, metrics_, confusion_

Fit model, estimate distribution of parameter estimates, estimate accuracy metrics, calculate statistical significance.

```python 
<Classifier>.partial_fit(epoch, batch, metric) 
```
> Related Attributes: coef_, partial_, table_ 


#### [Example of Usage: Classifier](https://github.com/print-hi/lyra/blob/usage/usage/classifier.ipynb)

-------------------------------------------

## More Information
### General Parameters <a name="params"></a>

| Parameters | Type | Default | Description |
| ---- | --- | --- |  --- |
| *X_train*  | *pd.DataFrame* | *X*     | Matrix of input values|
| *y_train*  | *pd.Dataframe, list* | *y*   | Vector of binary outcomes |
| *X_test*  | *pd.DataFrame*   | - | Matrix of input values|
| *y_test*  | *pd.Dataframe, list*   | - | Vector of binary outcomes |
| *epoch*  | *int* | *100*    | Number of runs |
| *batch*  | *int, float* | *0.75*   | Number of samples per run |
| *metric*  | *str* | *'f1'*   | Evaluation metric |

### Usage
### Generalised ML Analysis
#### [Classifier](https://github.com/print-hi/lyra/blob/usage/usage/classifier.ipynb) <a name="classifier"></a>
