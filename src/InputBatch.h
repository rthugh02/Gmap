#ifndef INPUTBATCH_H
#define INPUTBATCH_H

#include <vector>
#include <string>
#include <armadillo>
class InputBatch
{
  private:
    
  
  public:
    arma::mat * genres;
    arma::mat * data;
    InputBatch(arma::mat *, arma::mat *);
    void free();

    ~InputBatch();  

    
};

#endif