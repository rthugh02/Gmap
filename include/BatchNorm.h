#ifndef BATCHNORM_H
#define BATCHNORM_H

#include <armadillo>
class BatchNorm
{
private:
    //trainable gamma and beta values for shifting normalized data
    double scale = 1;
    double shift = 0;
    arma::cube * cube_data;
    arma::mat * mat_data;
    bool isCube;
public:
    BatchNorm(arma::cube * data);
    BatchNorm(arma::mat * data);
    ~BatchNorm();

    void set_data(arma::cube * data);
    void set_data(arma::mat * data);
    void normalize();
};

#endif