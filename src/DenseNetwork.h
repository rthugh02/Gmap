#ifndef DENSENETWORK_H
#define DENSENETWORK_H
#include <armadillo>
class DenseNetwork
{
private:
    //dense layer weights
    arma::mat weights1;
    arma::mat weights2;
    arma::mat weights3;

    //batch data
    arma::mat data;

    void flatten_data(arma::cube * data);
public:
    DenseNetwork(arma::cube * data);
    ~DenseNetwork();

    arma::mat calculate_output(void (*activation_func)(arma::mat *, const char *));
    void set_data(arma::cube * data);
};


#endif