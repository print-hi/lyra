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

#include "glm.h"
#include "math.h"
#include "family.h"

template<typename Family> 
class GLM {
public:
    
    // Constructor
    GLM(Iter max_iter, Precision tol)
        : max_iter(max_iter), tol(tol), iters(0)
        {   }

    // Destructor
    ~GLM() 
        {   }

    // Callable function to optimize problem
    Result Optimize(Matrix X, Vector y, Link link, Solver sol, int batch) {

        data = Data(X, y, batch); 

        n_obs = X.rows(); n_batch = batch; 

        curr_coefs = X.row(0); curr_weights = X.col(0);

        return IRLS(link, sol);
    }

protected:
    // Read-able Optimisation parameters
    Data data;    
    int n_obs, n_batch;
    Iter max_iter;
    Precision tol;

private:
    // Private Optimisation variables
    Family target;

    Iter iters;
    Precision norm;
    Vector curr_coefs;
    Matrix curr_weights;
    Eigen::LLT<Matrix> cholskey; 
    Eigen::ColPivHouseholderQR<Matrix> qr;

    ///////////////////////////////////////////////////////////////////////////
    ////// Iteratively Reweighted Least Squares Implementation
    Result IRLS(Link link, Solver sol) {

        Iter iter = 0; int conv_consec = 0;
        // Iterate until one of the stopping conditions has been satisfied
        for(; (conv_consec < n_obs/n_batch) && (iter < max_iter); iter++) {
            iters++;
            data.NextBatch();
            // Calculate gradient vector
            UpdateWeights(data.X, data.y, link, sol);

            // If batch learning is used, make sure that convergence is
            // consistent across entire dataset
            conv_consec = (norm < tol) ? conv_consec + 1 : 0;
        }

        Vector mu = target.mean((data.X * curr_coefs), link);
        Precision ll = target.negloglike(data.y, mu);

        Result res(conv_consec, curr_coefs, ll, norm, iters);

        return res;
    }

    // Update weights for one iteration 
    // set ---> B (p x 1) : B = (X'WX)^-1 x (X'Wz)
    //    where z (n x 1) : z_i = g(mu_i) + (y_i - mu_i) * g'(mu_i)  
    //          W (n x n) : w_i,i = p_i / (b''(theta) * g'(mu_i)**2),  
    //                   w_i,j = 0 for all i != j
    void UpdateWeights(Matrix X, Vector y, Link link, Solver sol) {

        // Calculate linear predictor, mean and adjusted dependent variable
        // Note: refer to class defs in family.h for description by family
        Vector lin_pred = X * curr_coefs;
        Vector mu = target.mean(lin_pred, link);
        Vector adj_y = target.adjExog(y, mu, lin_pred, link);

        // Update iterative weights
        curr_weights = target.getWeight(mu, link);

        // Cholesky Decomposition - R1 := inverse of R : R^T x R = X_T x W x X
        // Faster method w/ reduced numerical stability w.r.t QR 
        // -> (X^T x W x X)^-1 = R^(-1) x R^(-T)
        // -> B = R^(-1) x R^(-T) x X^T x W x z
        if(sol == Cholesky) {

            cholskey.compute(X.transpose()*DiagonalProd(X, curr_weights));

            Matrix R1 = MatrixBackSubstitution(cholskey.matrixU()); 

            Vector new_coefs = R1 * R1.transpose() * X.transpose() * 
                               DiagonalProd(adj_y, curr_weights);

            norm = (curr_coefs - new_coefs).norm();
            curr_coefs = new_coefs;

            return;
        }

        // QR Decomposition (DEFAULT) - Q, inv of R : QR = W^(1/2) x X
        // -> X^T = R^T x Q^T x W^(-1/2)
        // -> B = R^(-1) x Q^T x W(1/2) x z
        Vector rootW = (curr_weights.array().sqrt()).matrix();

        qr.compute(DiagonalProd(X, rootW));
 
        Matrix R_inv = qr.matrixR().template triangularView<Eigen::Upper>() ; 
        R_inv.conservativeResize(R_inv.cols(), R_inv.cols()); 
        R_inv = MatrixBackSubstitution(R_inv);

        Matrix Q = qr.householderQ() * Matrix::Identity(X.rows(), X.cols());

        Vector new_coefs = R_inv * Q.transpose() * DiagonalProd(adj_y, rootW);
        new_coefs = new_coefs.reverse();

        // Calculate magnitude of change and update weights
        norm = (curr_coefs - new_coefs).norm();
        curr_coefs = new_coefs;
    }
};

// ==============================   TIMEOPT   =============================== //

template<template<typename> class Model, typename Family> 
struct Time {

    Matrix X;
    Vector y;
    Link link;
    Solver solver;
    Precision tol;
    Iter max_iter, batch, n_runs;

    Time(Matrix X, Vector y, Link link, Solver solver, 
         Precision tol, Iter max_iter, Iter batch, Iter n_runs) 
        :  X(X), y(y), link(link), solver(solver),
           tol(tol), max_iter(max_iter), batch(batch), n_runs(n_runs)
        {    }

    void run() {

        Model<Family> glm(max_iter, tol);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto t1 = std::chrono::high_resolution_clock::now();
        double e_T, e_T2;
        double diff;

        Result res = glm.Optimize(X, y, link, solver, batch);
        res.out();

        for(int i = 0; i < n_runs; i++) {
            t0 = std::chrono::high_resolution_clock::now();
            // =====================================================================
            auto r = glm.Optimize(X, y, link, solver, batch);
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
        << std::endl;
    }

    ~Time() {
        std::cout
        << "======================="  << "\n" 
        << "byebye, have a good day "  << std::endl; 
    }
};

int main() {

    Eigen::MatrixXd X = openData("poi++.csv");
    Eigen::VectorXd y;

    split_Xy(X, y, 0);
    add_intercept(X);

    Time<GLM, Poisson> t(X, y, LogLink, Cholesky, 0.05, 10, X.rows()/4, 20);
    t.run();
}


Result pythonGLM(Family family, Matrix X, Vector y, Precision tol, Link link, Solver solver, 
               Iter max_iter, Iter batch, Iter n_runs) {
    switch(family){
        case PoissonFamily:{
            GLM<Poisson> Poi(max_iter, tol);
            Result res_p = Poi.Optimize(X, y, link, solver, batch);
            return res_p;
        }
        default: {
            GLM<Poisson> Norm(max_iter, tol);
            Result res_n = Norm.Optimize(X, y, link, solver, batch);
            return res_n;
        }
    }
}
