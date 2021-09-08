#ifndef FUNC_H
#define FUNC_H

#include "descent.h"
#include <autodiff/reverse/var.hpp>

// General Model for linear methods: AX where A = [a,b,..]^T, X = [x_0,x_1,..]
struct Paraboloid {
    Precision f(const Vector& X) {
        Precision res = 0;
        for(auto x : X) res += x*x;
        return res;
    }
    Precision operator()(const Vector& X) {
        return f(X);
    }
    autodiff::var hess_f(const autodiff::ArrayXvar& X) {
        autodiff::var res = 0;
        for(auto x : X) res += x*x;
        return res; 
    }
};
 
struct Sum {
    Precision f(const Vector& X) {
        Precision res = 0;
        for(auto x : X) res += x;
        return res;
    }
    Precision operator()(const Vector& X) {
        return f(X);
    }
    autodiff::var hess_f(const autodiff::ArrayXvar& X) {
        autodiff::var res = 0;
        for(auto x : X) res += x*x;
        return res; 
    }
};

struct Prod {
    Precision f(const Vector& X) {
        Precision res = 1;
        for(auto x : X) res *= x;
        return res;
    }
    Precision operator()(const Vector& X) {
        return f(X);
    }
    autodiff::var hess_f(const autodiff::ArrayXvar& X) {
        autodiff::var res = 0;
        for(auto x : X) res += x*x;
        return res; 
    }
};

#endif

