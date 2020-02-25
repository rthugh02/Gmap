#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <armadillo>
#include "BatchNorm.h"
class Convolution
{
private:
    arma::mat kernel;
    BatchNorm * batch_norm1 = NULL;
    arma::cube * data;

    unsigned int KERNEL_WIDTH;
    int DATA_ROWS;
    int INPUT_BATCH_SIZE;
    void maxpooling(int); 
public:
    Convolution(arma::cube * data, int data_rows, int kernel_width, int batch_size);
    ~Convolution();

    void convolve(int, void (*)(arma::mat *, const char *));
};

#endif
