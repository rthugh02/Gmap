#include "InputBatch.h"
#include <armadillo>

InputBatch::InputBatch(arma::cube * input_matrix, arma::mat * genres)
{
    this->data = input_matrix;
    this->genres = genres;
}

void InputBatch::free()
{
    delete data;
    delete genres;
}

InputBatch::~InputBatch()
{
    
}