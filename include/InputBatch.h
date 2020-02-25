#ifndef INPUTBATCH_H
#define INPUTBATCH_H

#include <armadillo>
class InputBatch
{
  private:
    
  
  public:
    //cube of data. Each matrix slice represents one song
    arma::cube * data;
    //matrix of data. Each row represents the correct output vector corresponding with each slice in the batch. 
    arma::mat * genres;
    InputBatch(arma::cube * input_matrix, arma::mat * genres);
    void free();

    ~InputBatch();  

    
};

#endif