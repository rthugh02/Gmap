#ifndef DENSENETWORK_H
#define DENSENETWORK_H
#include <armadillo>
#include "BatchNorm.h"
class DenseNetwork
{
private:
    //dense layer weights
    arma::mat weights1;
    arma::mat weights2;
    arma::mat weights3;

    arma::mat pre_softmax;
    arma::mat pre_relu2;
    arma::mat pre_relu1;
    arma::mat batch_norm2_out;
    arma::mat batch_norm1_out;

    //batch normalization layers between the weights
    BatchNorm * batch_norm1 = NULL;
    BatchNorm * batch_norm2 = NULL;
    
    //batch data
    arma::mat data;

    void flatten_data(arma::cube * data);
    arma::mat update_weights_3(arma:: mat, arma::mat);
    arma::mat update_weights_2(arma::mat);
    arma::mat update_weights_1(arma::mat);
public:
    DenseNetwork(arma::cube * data);
    ~DenseNetwork();

    arma::mat calculate_output(void (*activation_func)(arma::mat *, const char *));
    void set_data(arma::cube * data);

    void back_propagation(arma::mat, arma::mat);
};


#endif