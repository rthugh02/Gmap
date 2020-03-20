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
    this->data_cols = data->n_cols;
    this->data_rows = data->n_rows;
    this->data = arma::mat(data->n_slices, data->n_cols * data->n_rows);
    for(arma::uword i = 0; i < data->n_slices; i++)
    {
        this->data.row(i) = arma::vectorise(data->slice(i), 1);
    }
}

arma::mat DenseNetwork::calculate_output(void (*activation_func)(arma::mat *, const char *))
{
    arma::mat results = this->data * weights1;
    pre_relu1 = results;
    activation_func(&results, "leakyrelu");
    if(batch_norm1 == NULL)
        batch_norm1 = new BatchNorm(&results);
    else
        batch_norm1->set_data(&results);
    batch_norm1->normalize();
    batch_norm1_out = results;
    
    results = results * weights2;
    pre_relu2 = results;
    activation_func(&results, "leakyrelu");
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

arma::cube DenseNetwork::back_propagation(arma::mat predictions, arma::mat correct_output)
{
    //update order: weights3 -> batch_norm2 -> weights2 -> batchnorm1 -> weights1 
    
    arma::mat delta_error_wr2_batchnorm2_out = update_weights_3(predictions, correct_output);
    
    arma::mat delta_error_wr2_batchnorm2_in =  batch_norm2->back_propagation(delta_error_wr2_batchnorm2_out);

    arma::mat delta_error_wr2_batchnorm1_out = update_weights_2(delta_error_wr2_batchnorm2_in);

    arma::mat delta_error_wr2_batchnorm1_in = batch_norm1->back_propagation(delta_error_wr2_batchnorm1_out);

    arma::mat delta_error_wr2_dense_net_input = update_weights_1(delta_error_wr2_batchnorm1_in);   

    //reassembling delta error back into cube since input was flattened
    arma::cube ret = arma::cube(this->data_rows, this->data_cols, delta_error_wr2_dense_net_input.n_rows);
	for(arma::uword i = 0; i < delta_error_wr2_dense_net_input.n_rows; i++)
	{
		arma::mat reassembled_slice = arma::mat(ret.n_rows, ret.n_cols);
		for(arma::uword j = 0; j < ret.n_cols -1; j++)
		{
			reassembled_slice.row(j) = delta_error_wr2_dense_net_input.submat(i, j*ret.n_cols, i, j*ret.n_cols + ret.n_cols - 1);
		}
		ret.slice(i) = reassembled_slice;
	}

    return ret;
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
    
    arma::cube delta_error_wr2_weights3 = arma::cube(weights3.n_rows, weights3.n_cols, delta_in_wr2_weights3.n_rows);

    for(arma::uword i = 0; i < delta_error_wr2_weights3.n_slices; i++)
    {
        delta_error_wr2_weights3.slice(i) = delta_in_wr2_weights3.row(i).t() * temp_delta.row(i);
    }

    arma::mat ret = temp_delta * weights3.t();
    arma::mat weights3_gradient = arma::mean(delta_error_wr2_weights3, 2);

    weights3 -= (weights3_gradient * learning_rate);
    
    return ret;
}

arma::mat DenseNetwork::update_weights_2(arma::mat delta_error_wr2_out)
{
    //derivative of ReLu
    arma::mat delta_out_wr2_in = pre_relu2;
    
    delta_out_wr2_in.transform([&] (double val) { return val > 0 ? 1 : 0.01; });
    
    arma::mat delta_in_wr2_weights2 = batch_norm1_out;
    
    arma::cube delta_error_wr2_weights2 = arma::cube(weights2.n_rows, weights2.n_cols, delta_in_wr2_weights2.n_rows);
    arma::mat temp_delta = delta_error_wr2_out % delta_out_wr2_in; 

    for(arma::uword i = 0; i < delta_error_wr2_weights2.n_slices; i++)
    {
        delta_error_wr2_weights2.slice(i) =  delta_in_wr2_weights2.row(i).t() * temp_delta.row(i);
    }

    arma::mat ret = temp_delta * weights2.t();
    arma::mat weights2_gradient = arma::mean(delta_error_wr2_weights2, 2);

    weights2 -= (weights2_gradient * learning_rate);
   
    return ret;
}

arma::mat DenseNetwork::update_weights_1(arma::mat delta_error_wr2_out)
{
    //derivative of ReLu
    arma::mat delta_out_wr2_in = pre_relu1;
    
    delta_out_wr2_in.transform([&] (double val) { return val > 0 ? 1 : 0.01; });
    
    arma::mat delta_in_wr2_weights1 = this->data;
    
    arma::cube delta_error_wr2_weights1 = arma::cube(weights1.n_rows, weights1.n_cols, delta_in_wr2_weights1.n_rows);
    arma::mat temp_delta = delta_error_wr2_out % delta_out_wr2_in; 

    for(arma::uword i = 0; i < delta_error_wr2_weights1.n_slices; i++)
    {
        delta_error_wr2_weights1.slice(i) =  delta_in_wr2_weights1.row(i).t() * temp_delta.row(i);
    }

    arma::mat ret = temp_delta * weights1.t();
    arma::mat weights1_gradient = arma::mean(delta_error_wr2_weights1, 2);

    weights1 -= (weights1_gradient * learning_rate);

    return ret;
}

DenseNetwork::~DenseNetwork()
{
}