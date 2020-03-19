#ifndef BATCHNORM_H
#define BATCHNORM_H

#include <armadillo>
class BatchNorm
{
private:
    //trainable gamma and beta values for shifting normalized data
    arma::rowvec feature_scales_mat;
    arma::rowvec feature_shifts_mat;

    arma::mat feature_scales_cube;
    arma::mat feature_shifts_cube;

    arma::cube * cube_data;
    arma::cube cube_data_copy;

    arma::mat * mat_data;
    arma::mat mat_data_copy;

    arma::rowvec feature_var_copy_mat;
    arma::mat feature_var_copy_cube;
    bool isCube;

    double LEARNING_RATE = 0.001;
public:
    BatchNorm(arma::cube * data);
    BatchNorm(arma::mat * data);
    ~BatchNorm();

    void set_data(arma::cube * data);
    void set_data(arma::mat * data);
    void normalize();

    arma::mat back_propagation(arma::mat);
    arma::cube back_propagation(arma::cube);
};

#endif