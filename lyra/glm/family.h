#ifndef FUNC_H
#define FUNC_H

#include<fstream>
#include <math.h>
#include <autodiff/reverse/var.hpp>

#include "glm.h"

struct Poisson {

    // Calculate negative log likelihood of fitted model
    Precision negloglike(const Vector& y, const Vector& mu) {

        Precision loglike = 0; 

        for(int i = 0; i < mu.rows(); i++) 
            loglike -= mu(i) - y(i)*std::log(mu(i));

        return loglike;
    }

    // Calculate mean-value parameter, mu = g^(-1) (X x B)
    Vector mean(const Vector& lin_pred, Link link) {

        if(link == IdentityLink)
            return lin_pred;

        return ((lin_pred.array()).exp()).matrix();
    }

    // z_i = g(mu_i) + (y_i - mu_i) * g'(mu_i) 
    //     = ln(mu_i) + y_i / mu_i - 1           - log link
    //     = y_i                                 - identity link
    Vector adjExog(const Vector& y, const Vector& mu,
                   const Vector& lin_pred, Link link) {

        if(link == IdentityLink)
            return y;

        Vector m_1 = lin_pred; m_1 = m_1.setOnes(lin_pred.rows());

        return (y.array() / mu.array()).matrix() + lin_pred - m_1;
    }

    // w_i,i = p_i / (b''(theta) * g'(mu_i)**2) 
    //       = mu_i                                - log link
    //       = 1/mu_i                              - identity link
    Vector getWeight(const Vector& mu, Link link) {

        if(link == IdentityLink)
            return ((mu.array()).inverse()).matrix();

        return mu;
    }

};

class Data {
public:
    int index, batch, n_obs;
    Matrix data_X; Vector data_y;

    Matrix X, X_t; Vector y;

    Data() 
        {   }

    Data(Matrix X, Vector y, int batch)
        : data_X(X), data_y(y), index(0), batch(batch), n_obs(X.rows())
        {   }

    ~Data() 
        {   }
    
    void NextBatch() {

        if(index + batch >= n_obs - 1) {
            X.colwise().reverse();
            index = 0;
        }

        y = data_y.block(index, 0, batch, 1);
        X = data_X.block(index, 0, batch, data_X.cols());  
        X_t = X.transpose();

        index += batch;
    }
};

Eigen::MatrixXd openData(std::string fileToOpen) {
    std::vector<double> matrixEntries;
 
    std::ifstream matrixDataFile(fileToOpen);
 
    std::string matrixRowString;
 
    std::string matrixEntry;
 
    int matrixRowNumber = 0;
 
 
    while (std::getline(matrixDataFile, matrixRowString)) {
        std::stringstream matrixRowStringStream(matrixRowString);
 
        while (std::getline(matrixRowStringStream, matrixEntry, ',')) {
            matrixEntries.push_back(stod(matrixEntry));   
        }
        matrixRowNumber++; 
    }
 
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 
                      Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, 
                                        matrixEntries.size() / matrixRowNumber);
 
}

void split_Xy(Eigen::MatrixXd& matrix, Eigen::VectorXd& vector, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    vector = matrix.col(colToRemove);

    if(colToRemove < numCols)
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

void add_intercept(Eigen::MatrixXd& matrix)
{
    Vector intercept = matrix.col(0);
    for(int i = 0; i < intercept.rows(); i++) intercept(i) = 1;

    matrix.conservativeResize(Eigen::NoChange, matrix.cols()+1);
    matrix.col(matrix.cols()-1) = intercept;
}

#endif