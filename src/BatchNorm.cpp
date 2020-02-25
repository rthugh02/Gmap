#include "BatchNorm.h"
#include <armadillo>

BatchNorm::BatchNorm(arma::cube * data)
{
    this->isCube = true;
    this->cube_data = data;
    this->shift = 0;
    this->scale = 1;
}

BatchNorm::BatchNorm(arma::mat * data)
{
    this->isCube = false;
    this->mat_data = data;
    this->shift = 0;
    this->scale = 1;
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
        //apply gamma and beta scale and shift
        cube_data->transform([&] (double val) { return (val * this->scale) + this->shift; } );
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
        //apply gamma and beta scale and shift
        mat_data->transform([&] (double val) { return (val * this->scale) + this->shift; } );
    }
    
}

BatchNorm::~BatchNorm()
{
}