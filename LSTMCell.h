#ifndef LSTMCELL_H
#define LSTMCELL_H

#include <armadillo>

class LSTMCell
{
    private:
    //weight matrices for each of the cells 
    arma::mat dataf_weights;
    arma::mat prevf_weights;
    arma::mat datai_weights;
    arma::mat previ_weights;
    arma::mat datao_weights;
    arma::mat prevo_weights;
    arma::mat datac_weights;
    arma::mat prevc_weights;
    public:
        LSTMCell(int, int, int);
        ~LSTMCell();
};

#endif