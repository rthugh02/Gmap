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
    clear_gates();
    split_row_slice(row_slice);
}

void LSTMCell::clear_gates()
{
    sub_mats.clear();
    forgets.clear();
    inputs.clear();
    outputs.clear();
    cell_temps.clear();
    cell_states.clear();
    outs.clear();
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
        forgets.push_back(forget_gate);

        arma::mat input_gate = (sub_mats[i] * datai_weights) + (out * previ_weights);
        activation_func(&input_gate, "sigmoid");
        inputs.push_back(input_gate);

        arma::mat output_gate = (sub_mats[i] * datao_weights) + (out * prevo_weights);
        activation_func(&output_gate, "sigmoid");
        outputs.push_back(output_gate);

        arma::mat cell_temp = (sub_mats[i] * datac_weights) + (out * prevc_weights);
        activation_func(&cell_temp, "tanh");
        cell_temps.push_back(cell_temp);

        cell_state = (forget_gate % cell_state) + (input_gate % cell_temp);
        cell_states.push_back(cell_state);

        arma::mat cell_state_tanh = cell_state;
        activation_func(&cell_state_tanh, "tanh");

        out = (output_gate % cell_state_tanh);
        outs.push_back(out); 
    }

    return out;
}

arma::mat LSTMCell::back_propagation(arma::mat delta_error, void (*activation_func)(arma::mat *, const char *))
{
    arma::mat delta_error_wr2_cell_in = arma::mat(batch_size, features*hidden_units);
    arma::mat delta_next_out = arma::zeros<arma::mat>(batch_size, hidden_units);
    arma::mat delta_t = delta_error;
    for(int i = hidden_units - 1; i > 0; i--)
    {
        arma::mat delta_out_t = delta_t + delta_next_out;

    }
    return delta_error_wr2_cell_in;
}

LSTMCell::~LSTMCell()
{

}