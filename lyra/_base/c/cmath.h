#include <Python.h>

int fastfactorial(int n);

static PyObject* factorial(PyObject* self, PyObject* args){
    int n;
    if (!PyArg_ParseTuple(args,"i",&n))
        return NULL;
    int result = fastfactorial(n);
    return Py_BuildValue("i",result);
}

static PyMethodDef mainMethods[] = {
    {"factorial_c",factorial,METH_VARARGS,"Calculate the factorial of n"},
    {NULL,NULL,0,NULL}
};

static PyModuleDef cmath = {
    PyModuleDef_HEAD_INIT,
    "cmath","Factorial Calculation",
    -1,
    mainMethods
};

PyMODINIT_FUNC PyInit_cmath(void){
    return PyModule_Create(&cmath);
}