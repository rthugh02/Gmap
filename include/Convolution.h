#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <armadillo>
#include "BatchNorm.h"
class Convolution
{
private:
    arma::mat kernel;
    BatchNorm * batch_norm = NULL;
    arma::cube * data;
    arma::umat max_cells;

    unsigned int KERNEL_WIDTH;
    int DATA_ROWS;
    int INPUT_BATCH_SIZE;
    void maxpooling(); 
public:
    Convolution(arma::cube * data, int data_rows, int kernel_width);
    ~Convolution();

    void set_data(arma::cube *);
    void convolve(void (*)(arma::mat *, const char *));
    arma::cube back_propagation(arma::cube);
};

#endif
