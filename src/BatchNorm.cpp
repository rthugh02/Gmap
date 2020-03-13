#include "BatchNorm.h"
#include <armadillo>

BatchNorm::BatchNorm(arma::cube * data)
{
    this->isCube = true;
    this->cube_data = data;
}

BatchNorm::BatchNorm(arma::mat * data)
{
    this->isCube = false;
    this->mat_data = data;
    this->feature_scales_mat = arma::rowvec(data->n_cols, arma::fill::ones);
    this->feature_shifts_mat = arma::rowvec(data->n_cols, arma::fill::zeros);
}

void BatchNorm::set_data(arma::cube * data)
{
    try
    {
        if(!this->isCube)
            throw std::exception();
    }
    catch(const std::exception& e)
    {
        std::cerr << "incorrect data format: expected matrix" << '\n';
    }

    this->cube_data = data;
     
}
void BatchNorm::set_data(arma::mat * data)
{
    try
    {
        if(this->isCube)
            throw std::exception();
    }
    catch(const std::exception& e)
    {
        std::cerr << "incorrect data format: expected cube" << '\n';
    }

    this->mat_data = data;
}
void BatchNorm::normalize()
{
    if(this->isCube)
    {
        arma::mat feature_means = arma::mean(*cube_data, 2);
	    arma::mat feature_variances = arma::sum(
		(cube_data->each_slice() - feature_means).transform([] (double val) { return val*val; } ), 2) / cube_data->n_rows; 
	
	    //denominator of normalization formula
	    feature_variances.transform([] (double val) { return sqrt(val + 0.0001); } );
	    //subtracting means for numerator
	    cube_data->each_slice() -= feature_means;
	    //normalized values
	    cube_data->each_slice() %= (1 / (feature_variances));
        cube_data_copy = *(cube_data);
        //apply gamma and beta scale and shift
       // cube_data->transform([&] (double val) { return (val * this->scale) + this->shift; } );
    }
    else
    {
        arma::rowvec feature_means = arma::mean(*mat_data, 0);
	    arma::rowvec feature_variances = arma::sum(
		(mat_data->each_row() - feature_means).transform([] (double val) { return val*val; } ), 0) / mat_data->n_rows; 
	
	    //denominator of normalization formula
	    feature_variances.transform([] (double val) { return sqrt(val + 0.0001); } );
	    //subtracting means for numerator
	    mat_data->each_row() -= feature_means;
	    //normalized values
	    mat_data->each_row() %= (1 / (feature_variances));
        mat_data_copy = *(mat_data);
        feature_var_copy = feature_variances;
        //apply gamma and beta scale and shift
        mat_data->each_row() %= this->feature_scales_mat;
        mat_data->each_row() += this->feature_shifts_mat;
    }
    
}

arma::mat BatchNorm::back_propagation(arma::mat delta_error)
{
    //https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html

    arma::rowvec delta_beta = arma::sum(delta_error, 0);
    arma::rowvec delta_gamma = arma::sum(delta_error % mat_data_copy, 0);

    int batch_size = delta_error.n_rows;
    arma:: mat delta_error_wr2_batchnorm_in = 
    ((batch_size * delta_error) - (arma::colvec(batch_size, arma::fill::ones) * delta_beta) - (mat_data_copy.each_row() % delta_gamma ));
    delta_error_wr2_batchnorm_in.each_row() %= ((double)(1/batch_size) * feature_scales_mat % (1 / feature_var_copy));
    

    feature_shifts_mat -= delta_beta;
    feature_scales_mat -= delta_gamma;
    
    std::cout << "delta error wr2 in dims: " << delta_error_wr2_batchnorm_in.n_rows << " X " << delta_error_wr2_batchnorm_in.n_cols << std::endl;
    return delta_error_wr2_batchnorm_in;
}

BatchNorm::~BatchNorm()
{
}