#ifndef DESCENT_H
#define DESCENT_H

// Note - To save memory, toogle double -> float & long int -> int 
// !!!! - Restraints on accuracy and max iterations will change accordingly 
typedef double                                          Precision;
typedef autodiff::var                                   DetVar;
typedef Eigen::Matrix<Precision, Eigen::Dynamic, 1>     Vector;
typedef Eigen::Matrix<autodiff::var, Eigen::Dynamic, 1> Vector_h;
typedef Eigen::MatrixXd                                 Matrix;
typedef long int                                        Iter;

enum Solver {
    gd, sgd, bfgs
};

enum Function {
    paraboloid, sum, prod
};

struct Result {
    bool conv;
    Vector X;
    Precision y;
    Precision tol;
    Iter iters_obj, iters_opt, iters_jac, iters_hes;

    Result(bool conv, Vector X, Precision y, Precision tol, Iter iters_obj,
           Iter iters_opt, Iter iters_jac, Iter iters_hes)
        : conv(conv), X(X), y(y), tol(tol), iters_obj(iters_obj), 
          iters_opt(iters_opt), iters_jac(iters_jac), iters_hes(iters_hes)
        {   }

    void out() {
        std::cout 
        << "-----------------------" << "\n"
        << "------  Results  ------" << "\n"
        << "-----------------------" << "\n"
        << "Converged: " << (conv ? "True" : "False")  << "\n"
        << "-----------------------" << "\n"
        << "Tolerance: " << tol      << "\n"
        << "-----------------------" << "\n"
        << "Minim f_X: " << y        << "\n"
        << "-----------------------"
        << std::endl;
        for(int i = 0; i < X.rows(); i++) 
            std::cout << "Minim x_" << i << ": " << X(i) << std::endl;
        std::cout 
        << "-----------------------" << "\n"
        << "Iters_Obj: " << iters_obj << "\n"
        << "Iters_Opt: " << iters_opt << "\n"
        << "Iters_Jac: " << iters_jac << "\n"
        << "Iters_Hes: " << iters_hes << "\n"
        << "-----------------------" 
        << std::endl; 
    }
};

Result optimizer(Function func, Solver sol, Vector start, Iter max_iter, 
                Precision min_step,  Precision max_step, 
                Precision delta, Precision tol);


// #include <pybind11/pybind11.h>

// namespace py = pybind11;

// int factorial_cpp(int a);

// PYBIND11_MODULE(optimize, mod) {
//     mod.def("factorial_cpp", &factorial_cpp, "factorial algo");
//     mod.def("minimise", &optimizer, "optimize algo");
// }

#endif