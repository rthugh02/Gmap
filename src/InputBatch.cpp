#include "InputBatch.h"
#include <armadillo>
#include <string>
#include <vector>

InputBatch::InputBatch(arma::mat * input_matrix, arma::mat * genres)
{
    data = input_matrix;
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