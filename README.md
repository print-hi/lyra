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

## Algorithm Implementation 
### Generalised Linear Model

GLM assumes that observations come from a distribution from the exponential dispersion family/model. Distributions from the exponential dispersion family/model can be shown to have pmf/pdf of the form: &nbsp;
<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/dark-1.svg">
        <img src="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/light-1.svg" />
    </picture>
</p> 

> **Dark-Mode Users:** [Please click here if rendered in black text!](https://github.com/print-hi/lyra-v.0.1/blob/main/RM-NIGHT.md#user-installation)

<br />
To build the spread around the linear model, we can then make use of various differentiable transformations (injective) using the following relationship: <br/><br/>
<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/dark-2.svg">
        <img src="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/light-2.svg" />
    </picture>
</p> 
<br/>
To fit the model, iteratively reweighted least squares can be used with the following adjusted dependent variable: <br/><br/>
<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/dark-3.svg">
        <img src="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/light-3.svg" />
    </picture>
</p> 

> When canonical links are used, the Hessian matrix concide so that the Fisher scoring method and Newton-Raphson method reduce to the same algorithm (McCullagh, P. and Nelder, J. A. (1989). Generalized Linear Models, Vol. 37 of Monographs on Statistics and Applied Probability, 2 edn, Chapman and Hall, London)

<br/>

Thereafter, we need to calculate the iterative weights, given by the diagonal matrix with entries: <br/><br/>
<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/dark-4.svg">
        <img src="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/light-4.svg" />
    </picture>
</p> 

This leaves us with the following update rule: <br/><br/>
<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/dark-5.svg">
        <img src="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/light-5.svg" />
    </picture>
</p> 

> To speed up calculations, we can make use of matrix decompositions. This is vital for optimisation as calculating inversions for large matricies is very costly, and instead we can aim to inverse triangle matricies and make use of the diagonalisation in the weight matrix. 

<br/>

##### Cholesky Factorization 

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/dark-6.svg">
        <img src="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/light-6.svg" />
    </picture>
</p> 

##### QR Decomposition

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/dark-7.svg">
        <img src="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/light-7.svg" />
    </picture>
</p> 

##### Column-Pivoting

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/dark-8.svg">
        <img src="https://github.com/print-hi/lyra-v.0.1/blob/main/lib/svg/light-8.svg" />
    </picture>
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
