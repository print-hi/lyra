#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <iomanip>

#include <Eigen/Geometry>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include "optimizer.h"
#include "test_func.h"

#define print(x) std::cout << x << std::endl

int factorial_cpp(int a){
    int f = 1;
    for(int i = 1;i<=a; i++){f*=i;};
    return f;
}

template<typename Objective> 
class Optimizer {
public:
    // Constructor 
    Optimizer()
        : max_iter(50000), min_step(0.1), max_step(0.2), delta(1e-07), 
          tol(1e-14), iters_obj(0), iters_opt(0), iters_jac(0), iters_hes(0)
        {   }
    
    // Seting custom parameters
    Optimizer(Iter max_iter, Precision min_step, Precision max_step, 
                    Precision delta, Precision tol)
        : max_iter(max_iter), min_step(min_step), max_step(max_step), 
          delta(delta), tol(tol), iters_obj(0), iters_opt(0), iters_jac(0), 
          iters_hes(0)
        {   }

    // Destructor
    ~Optimizer() 
        {   }

    // Functionality to be added later
    void SetBounds(const Vector lower, const Vector upper) {
        lowerbound = lower; upperbound = upper; 
    }

    // Callable function to optimize problem
    Result Optimize(Vector startPoint, Solver sol) {
        iters_opt++;
        dim = startPoint.rows();
        switch(sol) {
            case gd: {
                return GradientDescent(startPoint);
            }    
            case sgd: {
                return GradientDescent(startPoint);
            }
            case bfgs: {
                return GradientDescent(startPoint);
            }
        }
    }

protected:
    // Read-able Optimisation parameters
    int dim; 
    Iter max_iter;
    Precision min_step, max_step, delta, tol;
    Vector lowerbound, upperbound;

    // Compute Jacobian Vector at current location
    Vector ComputeJacobian() {

        Vector del_X = curr_X; Vector grad = curr_X;

        // Compute partial derivates
        for(int i = 0; i < dim; i++) {

            del_X = curr_X; 
            del_X(i) += delta;

            grad(i) = (objective(del_X)-curr_y)/delta;
            iters_obj++;
        }

        iters_jac++; 
        return grad;
    }

    // Compute Hessian Matrix at current location
    Matrix ComputeHessian() {

        Vector_h x(dim);
        for(int i = 0; i < dim; i++) x(i) = curr_X(i);

        DetVar u = objective.hess_f(x); 
        Matrix H = hessian(u, x);
        
        iters_hes++; 
        return H;
    }

private:
    // Private Optimisation variables
    Vector curr_X;
    Precision curr_y; 
    Objective objective;
    Iter iters_obj, iters_opt, iters_jac, iters_hes;

    ///////////////////////////////////////////////////////////////////////////
    ////// Gradient Descent Implementation
    Result GradientDescent(Vector startPoint) {

        curr_X = startPoint; curr_y = objective(startPoint);

        Vector grad; Iter iter = 0; Precision size = tol + 1.5;
        
        // Iterate until one of the stopping conditions has been satisfied
        for(; (size > tol) && (iter < max_iter); iter++, size = grad.norm()) {
            // Calculate gradient vector
            grad = ComputeJacobian(); 

            // Update current position
            curr_X -= (grad * min_step); curr_y = objective(curr_X);
        }
        bool conv = true;
        if((size > tol) && !(iter < max_iter)) conv = false;

        Result res(conv, curr_X, curr_y, size, iters_obj, iters_opt, iters_jac, 
                   iters_hes);
        return res;
    }

};


// ============================   PYTHON  API   ============================= //

Result optimize(Function func, Solver sol, Vector start, Iter max_iter = 50000, 
                Precision min_step = 0.1,  Precision max_step = 0.2, 
                Precision delta = 1e-07, Precision tol = 1e-13) {
    switch(func) {
        case paraboloid: {
            Optimizer<Paraboloid> para(max_iter, min_step, max_step, delta, tol);
            Result r_para = para.Optimize(start, sol);
            return r_para;
        }    
        case sum: {
            Optimizer<Sum> sum(max_iter, min_step, max_step, delta, tol);
            Result r_s = sum.Optimize(start, sol);
            return r_s;
        }
        case prod: {
            Optimizer<Prod> prod(max_iter, min_step, max_step, delta, tol);
            Result r_p = prod.Optimize(start, sol);
            return r_p;
        }
    }
}

// ==============================   TIMEOPT   =============================== //
void time(Function f, Vector start, int n_runs) {


    auto r = optimize(f, gd, start);
    r.out();

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    double e_T, e_T2;
    double diff;

    for(int i = 0; i < n_runs; i++) {
        t0 = std::chrono::high_resolution_clock::now();
        // =====================================================================
        r = optimize(f, gd, start);
        // =====================================================================
        t1 = std::chrono::high_resolution_clock::now();

        diff = (std::chrono::
                duration_cast<std::chrono::microseconds>((t1-t0))).count();
        e_T += diff/n_runs;
        e_T2 += std::pow(diff,2)/n_runs;
    }

    std::cout << std::setprecision(5)
    << "Time (\xC2\xB5s): " << e_T 
    << "\n        +- " 
    << std::pow((e_T2-e_T*e_T), 0.5) << "\n"
    << "Time (ms): " << e_T/1000  
    << "\n        +- " 
    << std::pow((e_T2-e_T*e_T), 0.5)/1000 << "\n"
    << "Time  (s): " << e_T/1000000
    << "\n        +- " 
    << std::pow((e_T2-e_T*e_T), 0.5)/1000000
    << "\n" << "======================="  << std::endl; 
    
}

int main(){
    Vector X(7); X << 5,5,6,7,8,9,10;

    time(paraboloid, X, 1000);
}