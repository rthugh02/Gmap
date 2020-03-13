#include "DenseNetwork.h"
#include <armadillo>

DenseNetwork::DenseNetwork(arma::cube * data)
{
    flatten_data(data);
    
    //Random seed for initializing weights
	std::random_device rd;
	
	//Uniform distribution of real numbers
	std::normal_distribution<double> distr(0, 1);

	//dense layer weights
	weights1 = arma::mat(this->data.n_cols, 384);
	weights2 = arma::mat(384, 32);
	weights3 = arma::mat(32, 8);
	
	//mersenne twister engine for generating random values
	std::mt19937 engine(rd());
	//filling weight matrices with random values generated by mersenne engine by cell
	weights1.imbue( [&]() {return distr(engine) * 2 / (this->data.n_cols); } );
	engine.seed(rd());
	weights2.imbue( [&]() {return distr(engine) * 2 / (384); } );
	engine.seed(rd());
	weights3.imbue( [&]() {return distr(engine) * 2 / (32); } );
}

void DenseNetwork::set_data(arma::cube * data)
{
    flatten_data(data);
}

void DenseNetwork::flatten_data(arma::cube * data)
{
    //vectorizing slices and combining 
    this->data = arma::mat(data->n_slices, data->n_cols * data->n_rows);
    for(arma::uword i = 0; i < data->n_slices; i++)
    {
        this->data.row(i) = arma::vectorise(data->slice(i), 1);
    }
}

arma::mat DenseNetwork::calculate_output(void (*activation_func)(arma::mat *, const char *))
{
    arma::mat results = this->data * weights1;
    activation_func(&results, "relu");
    if(batch_norm1 == NULL)
        batch_norm1 = new BatchNorm(&results);
    else
        batch_norm1->set_data(&results);
    batch_norm1->normalize();
    
    results = results * weights2;
    activation_func(&results, "relu");
    if(batch_norm2 == NULL)
        batch_norm2 = new BatchNorm(&results);
    else
        batch_norm2->set_data(&results);
    batch_norm2->normalize();
    batch_norm2_out = results;

    results = results * weights3;
    pre_softmax = results;
    activation_func(&results, "softmax");

    return results;
}

void DenseNetwork::back_propagation(arma::mat predictions, arma::mat correct_output)
{
    //update order: weights3 -> -> weights2 -> batch_norm2 ->  weights1 -> batch_norm1 

    arma::mat delta_error_wr2_batchnorm2_out = update_weights_3(predictions, correct_output);
    
    arma::mat delta_error_wr2_batchnorm2_in =  batch_norm2->back_propagation(delta_error_wr2_batchnorm2_out);
}

arma::mat DenseNetwork::update_weights_3(arma::mat predictions, arma::mat correct_output)
{
    arma::mat predictions_copy = predictions;
    arma::mat output_copy = correct_output;

    //calculating the change in the error with respect to the prediction, derivative of the loss function
    predictions.transform([&](double val) {return 1 - val;});
    correct_output.transform([&](double val) {return 1 - val;});
    
    arma::mat delta_error_wr2_out = -((output_copy / predictions_copy) + (correct_output / predictions));
    pre_softmax.transform([&](double val) {return exp(val);});

    //change in predictions with respect to the input, derivative of softmax activation
    arma::colvec sum_of_ins = arma::sum(pre_softmax, 1);
    
    arma::mat copies_of_sums = arma::mat(sum_of_ins.n_rows, 8);
    copies_of_sums.each_col() = sum_of_ins;

    arma::mat delta_out_wr2_in = ((pre_softmax % (copies_of_sums - pre_softmax)) / (copies_of_sums % copies_of_sums));

    //change in input with respect to the weights, this is simply the input
    arma::mat delta_in_wr2_weights3 = batch_norm2_out;
    
    //applying chain rule and averaging results across the mini-batch to get gradient of weight3
    arma::mat temp_delta = delta_error_wr2_out % delta_out_wr2_in;
    
    arma::cube delta_error_wr2_weights3 = arma::cube(32, 8, delta_in_wr2_weights3.n_rows);

    for(arma::uword i = 0; i < delta_error_wr2_weights3.n_slices; i++)
    {
        delta_error_wr2_weights3.slice(i) = delta_in_wr2_weights3.row(i).t() * temp_delta.row(i);
    }

    arma::mat ret = temp_delta * weights3.t();
    arma::mat weights3_gradient = arma::mean(delta_error_wr2_weights3, 2);

    weights3 -= (weights3_gradient * 0.1);

    return ret;
}

DenseNetwork::~DenseNetwork()
{
}