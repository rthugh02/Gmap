#include "LSTMCell.h"
#include <armadillo>

LSTMCell::LSTMCell(arma::mat row_slice, int batch_size, int features, int hidden_units)
{
    std::random_device rd;
	
    this->hidden_units = hidden_units;
    this->batch_size = batch_size;
    this->features = features;

    split_row_slice(row_slice);
	//Uniform distribution of real numbers
	std::normal_distribution<double> distr(0, 1);
    std::mt19937 engine(rd());

    dataf_weights = arma::mat(features, hidden_units);
    dataf_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    prevf_weights = arma::mat(hidden_units, hidden_units);
    prevf_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    datai_weights = arma::mat(features, hidden_units);
    datai_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    previ_weights = arma::mat(hidden_units, hidden_units);
    previ_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    datao_weights = arma::mat(features, hidden_units);
    datao_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    prevo_weights = arma::mat(hidden_units, hidden_units);
    prevo_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    datac_weights = arma::mat(features, hidden_units);
    datac_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    prevc_weights = arma::mat(hidden_units, hidden_units);
    prevc_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
}

void LSTMCell::set_data(arma::mat row_slice)
{
    sub_mats.clear();
    split_row_slice(row_slice);
}

void LSTMCell::split_row_slice(arma::mat row_slice)
{
    for(arma::uword i = 0; i < row_slice.n_cols; i+=features)
    {
        auto temp = row_slice.submat(0, i, batch_size - 1, i + features - 1);
        sub_mats.emplace_back(temp);
    }
}

arma::mat LSTMCell::calculate_output(void (*activation_func)(arma::mat *, const char *))
{
    arma::mat out = arma::zeros<arma::mat>(batch_size, hidden_units);
    arma::mat cell_state = arma::zeros<arma::mat>(batch_size, hidden_units);
    for(int i = 0; i < hidden_units; i++)
    {
        arma::mat forget_gate = (sub_mats[i] * dataf_weights) + (out * prevf_weights);
        activation_func(&forget_gate, "sigmoid");

        arma::mat input_gate = (sub_mats[i] * datai_weights) + (out * previ_weights);
        activation_func(&input_gate, "sigmoid");

        arma::mat output_gate = (sub_mats[i] * datao_weights) + (out * prevo_weights);
        activation_func(&output_gate, "sigmoid");

        arma::mat cell_temp = (sub_mats[i] * datac_weights) + (out * prevc_weights);
        activation_func(&cell_temp, "tanh");

        cell_state = (forget_gate % cell_state) + (input_gate % cell_temp);
        
        arma::mat cell_state_tanh = cell_state;
        activation_func(&cell_state_tanh, "tanh");

        out = (output_gate % cell_state_tanh); 
    }

    return out;
}

arma::mat LSTMCell::back_propagation(arma::mat delta_error)
{
    
}

LSTMCell::~LSTMCell()
{

}