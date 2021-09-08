import sys, os
from pathlib import Path
import unittest

path = os.path.abspath(os.path.dirname(''))
if path[-4:] == 'lyra': sys.path.insert(0, str(Path('__file__').resolve().parent))
elif path[-5:] == 'tests': sys.path.insert(0, str(Path('__file__').resolve().parent.parent))

from lyra.__core.cmath import factorial_c

print(factorial_c(5))

from lyra.__core.descent import optimize, vectorize
import numpy as np

n = 10
arr = np.arange(0, 100, 1, np.int32)

from ctypes import c_double, POINTER

def npToC(np_arr):
    arr = (len(np_arr) * c_double) ()
    for i in range(len(np_arr)):
        arr[i] = np_arr[i]
    return arr


print(optimize(0, arr))
