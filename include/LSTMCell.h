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

    std::vector<arma::mat> forgets = std::vector<arma::mat>();
    std::vector<arma::mat> inputs = std::vector<arma::mat>();
    std::vector<arma::mat> outputs = std::vector<arma::mat>();
    std::vector<arma::mat> cell_temps = std::vector<arma::mat>();
    std::vector<arma::mat> cell_states = std::vector<arma::mat>();
    std::vector<arma::mat> outs = std::vector<arma::mat>();

    //input data for the cell
    std::vector<arma::mat> sub_mats;
    int hidden_units;
    int batch_size;
    int features;

    double learning_rate = 0.001;

    void split_row_slice(arma::mat row_slice);
    void clear_gates();
    public:
        LSTMCell(arma::mat row_slice, int batch_size, int cols, int hidden_units);
        ~LSTMCell();

        void set_data(arma::mat row_slice, int batch_size);
        arma::mat calculate_output(void (*)(arma::mat *, const char *));

        arma::mat back_propagation(arma::mat, void (*)(arma::mat *, const char *));
};

#endif