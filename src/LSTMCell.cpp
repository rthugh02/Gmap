#include "LSTMCell.h"
#include <armadillo>

LSTMCell::LSTMCell(arma::mat row_slice, int batch_size, int cols, int hidden_units)
{
    std::random_device rd;
	this->row_slice = row_slice;
	//Uniform distribution of real numbers
	std::normal_distribution<double> distr(0, 1);
    std::mt19937 engine(rd());

    dataf_weights = arma::mat(batch_size, cols);
    dataf_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    prevf_weights = arma::mat(batch_size, hidden_units);
    prevf_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    datai_weights = arma::mat(batch_size, cols);
    datai_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    previ_weights = arma::mat(batch_size, hidden_units);
    previ_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    datao_weights = arma::mat(batch_size, cols);
    datao_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    prevo_weights = arma::mat(batch_size, hidden_units);
    prevo_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    datac_weights = arma::mat(batch_size, cols);
    datac_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
    prevc_weights = arma::mat(batch_size, hidden_units);
    prevc_weights.imbue( [&]() {return distr(engine) * 2 / (batch_size); } );
    engine.seed(rd());
}

LSTMCell::~LSTMCell()
{

}