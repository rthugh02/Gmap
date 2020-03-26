#include "InputBatch.h"
#include <armadillo>

InputBatch::InputBatch(arma::cube * input_matrix, arma::mat * genres)
{
    this->data = input_matrix;
    this->genres = genres;
}

InputBatch::InputBatch(arma::cube input_matrix, arma::mat genres)
{
    this->data = new arma::cube(input_matrix);
    this->genres = new arma::mat(genres);
}

void InputBatch::free()
{
    delete data;
    delete genres;
}

InputBatch::~InputBatch()
{
    
}