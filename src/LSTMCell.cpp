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
        //std::cout << "x sub t dims: " << temp.n_rows << " X " << temp.n_cols << std::endl;
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
    //https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9

    //running sum of gradients through time
    arma::mat dataf_gradient = arma::mat(features, hidden_units, arma::fill::zeros);
    arma::mat datai_gradient = arma::mat(features, hidden_units, arma::fill::zeros);
    arma::mat datao_gradient = arma::mat(features, hidden_units, arma::fill::zeros);
    arma::mat datac_gradient = arma::mat(features, hidden_units, arma::fill::zeros);

    arma::mat prevf_gradient = arma::mat(hidden_units, hidden_units, arma::fill::zeros);
    arma::mat previ_gradient = arma::mat(hidden_units, hidden_units, arma::fill::zeros);
    arma::mat prevo_gradient = arma::mat(hidden_units, hidden_units, arma::fill::zeros);
    arma::mat prevc_gradient = arma::mat(hidden_units, hidden_units, arma::fill::zeros);
    
    arma::mat deltaf_gate_next = arma::mat(batch_size, hidden_units, arma::fill::zeros);
    arma::mat deltai_gate_next = arma::mat(batch_size, hidden_units, arma::fill::zeros);
    arma::mat deltao_gate_next = arma::mat(batch_size, hidden_units, arma::fill::zeros);
    arma::mat deltac_gate_next = arma::mat(batch_size, hidden_units, arma::fill::zeros);

    arma::mat delta_error_wr2_cell_in = arma::mat(batch_size, features*hidden_units);
    arma::mat delta_state_next = arma::zeros<arma::mat>(batch_size, hidden_units);
    arma::mat forget_next = arma::zeros<arma::mat>(batch_size, hidden_units);
    arma::mat delta_t = delta_error;
    int count = 0;
    for(int i = hidden_units - 1; i >= 0; i--)
    {
        //calculating deltas of each gate

        arma::mat delta_out_t = delta_t;
        arma::mat cell_state_copy = cell_states[i];
        activation_func(&cell_states[i], "tanh2");
        arma::mat delta_state_t = delta_out_t % outs[i] % (1 - cell_states[i]) + (delta_state_next % forget_next);

        arma::mat delta_cell_temp_t = delta_state_t % inputs[i] % (1 - (cell_temps[i] % cell_temps[i]));
        arma::mat delta_input_t = delta_state_t % delta_cell_temp_t % inputs[i] % (1 - inputs[i]);
        arma::mat delta_forget_t = delta_state_t % (i == 0 ? arma::zeros<arma::mat>(batch_size, hidden_units) : cell_states[i])
        % forgets[i] % (1 - forgets[i]);

        activation_func(&cell_state_copy, "tanh");
        arma::mat delta_output_t = delta_out_t % cell_state_copy % outputs[i] % (1 - outputs[i]);

        //derivative of error with respect to input x at time t

        arma::mat delta_x_t = (delta_cell_temp_t * datac_weights.t()) + (delta_input_t * datai_weights.t()) + (delta_forget_t * dataf_weights.t()) + (delta_output_t * datao_weights.t());
        delta_t = (delta_cell_temp_t * prevc_weights.t()) + (delta_input_t * previ_weights.t()) + (delta_forget_t * prevf_weights.t()) + (delta_output_t * prevo_weights.t());

        delta_state_next = delta_state_t;
        forget_next = forgets[i];

        //transferring input x at time t to total delta matrix
        int start_col = (features * hidden_units - features) - (features * count);
        delta_error_wr2_cell_in.submat(0, start_col, batch_size-1, start_col + features-1) = delta_x_t;

        //summing gradients
        dataf_gradient += sub_mats[i].t() * delta_forget_t;
        datai_gradient += sub_mats[i].t() * delta_input_t;
        datao_gradient += sub_mats[i].t() * delta_output_t;
        datac_gradient += sub_mats[i].t() * delta_cell_temp_t;

        prevf_gradient += outs[i].t() * deltaf_gate_next;
        previ_gradient += outs[i].t() * deltai_gate_next;
        prevo_gradient += outs[i].t() * deltao_gate_next;
        prevc_gradient += outs[i].t() * deltac_gate_next;

        deltaf_gate_next = delta_forget_t;
        deltai_gate_next = delta_input_t;
        deltao_gate_next = delta_output_t;
        deltac_gate_next = delta_cell_temp_t;
    }








    return delta_error_wr2_cell_in;
}

LSTMCell::~LSTMCell()
{

}