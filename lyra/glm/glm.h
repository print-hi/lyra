#ifndef GLM_H
#define GLM_H

// Note - To save memory, toogle double -> float & long int -> int 
// !!!! - Restraints on accuracy and max iterations will change accordingly 
typedef double                                          Precision;
typedef autodiff::var                                   DetVar;
typedef Eigen::Matrix<Precision, Eigen::Dynamic, 1>     Vector;
typedef Eigen::Matrix<Precision, 1, Eigen::Dynamic>     Vector_row;
typedef Eigen::Matrix<autodiff::var, Eigen::Dynamic, 1> Vector_h;
typedef Eigen::MatrixXd                                 Matrix;
typedef long int                                        Iter;

enum Solver {
    Cholesky, QR
};

enum Link {
    IdentityLink, LogLink, LogitLink, ProbitLink, NegBinomLink, PowLink
};

enum Family {
    PoissonFamily, NormalFamily
};

struct Result {
    int conv;
    Vector X;
    Precision y;
    Precision tol;
    Iter iters;

    Result(int conv, Vector X, Precision y, Precision tol, Iter iters)
        : conv(conv), X(X), y(y), tol(tol), iters(iters)
        {   }

    void out() {
        std::cout 
        << "-----------------------" << "\n"
        << "------  Results  ------" << "\n"
        << "-----------------------" << "\n"
        << "Converged: " << conv  << "\n"
        << "-----------------------" << "\n"
        << "\u0394CoefNorm: " << tol      << "\n"
        << "-----------------------" << "\n"
        << "Neg LogLk: " << y        << "\n"
        << "-----------------------"
        << std::endl;
        for(int i = 0; i < X.rows(); i++) 
            std::cout << "Minim \u03B2_" << i << ": " << X(i) << std::endl;
        std::cout 
        << "-----------------------" << "\n"
        << "No. Iters: " << iters << "\n"
        << "-----------------------" 
        << std::endl; 
    }
};

Result pythonGLM(Family family, Matrix X, Vector y, Precision tol, Link link, Solver solver, 
               Iter max_iter, Iter batch, Iter n_runs);

// Result optimize(Family family, Link link, Solver sol, Matrix X,Vector y, int batch,
//                 Iter max_iter, Precision min_step,  
//                 Precision max_step, Precision delta, 
//                 Precision tol);

// #include <pybind11/pybind11.h>

// namespace py = pybind11;

// int factorial_cpp(int a);

// PYBIND11_MODULE(glm, mod) {
//     mod.def("glm", &glm, "optimize algo");
// }

#endif