#include "glm.h"    

////////      my bag of maths tricks that make code go zoom zoom      ////////

// Calculate inverse of upper triangle matrix via back substitution
Matrix MatrixBackSubstitution(Matrix m) {
    
    Matrix inv = m; int dim = m.rows(); inv.setZero(dim, dim);
    Matrix Id = Matrix::Identity(dim, dim);
    Vector_row row_vec = m.row(0);

    Eigen::ArrayXd rec_i = (m.row(0)).array();
    for(int i = dim - 1; i > -1; i--) {
        rec_i = (Id.row(i)).array(); 
        for(int j = i+1; j < dim; j++) {
            rec_i -= (inv.row(j)).array() * m(i,j);
        }

        rec_i = rec_i / m(i, i);
        row_vec = rec_i.matrix();
        inv.row(i) = row_vec;
    }

    return inv;
}

// Flip square matrix along secondary diagonal
void TransposeSecondary(Matrix& m) {
    int n = m.cols();
    for(int i = 0; i < n - 1; i++)
        for(int j = 0; j < n-i-1; j++)  
            std::swap(m(i, j), m(n-j-1, n-i-1));
}

// Multiply Vector by Diaogonal Matrix i.e. v x D
// note: Only supply diagonal of A (as vector)
Vector DiagonalProd(Vector v, Vector D) {
    return (v.array().colwise() * D.array()).matrix();
}

// Multiply Matrix by Diaogonal Matrix i.e. M x D
// note: Only supply diagonal of A (as vector)
Matrix DiagonalProd(Matrix M, Vector D) {
    return (M.array().colwise() * D.array()).matrix();
}