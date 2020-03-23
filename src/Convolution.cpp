#include "Convolution.h"
#include <armadillo>
Convolution::Convolution(arma::cube * data, int data_rows, int kernel_width)
{
    this->data = data;
    this->DATA_ROWS = data_rows;
    this->KERNEL_WIDTH = kernel_width;
    this->INPUT_BATCH_SIZE = data->n_slices;

    std::random_device rd;
	
	std::normal_distribution<double> distr(0, 1);
	
	std::mt19937 engine(rd());

    this->kernel = arma::mat(data_rows, kernel_width);
    this->kernel.imbue( [&]() {return distr(engine) * 2 / (data_rows); } );
}

void Convolution::convolve(void (*activation_func)(arma::mat *, const char *))
{
    for(arma::uword slice = 0; slice < data->n_slices; slice++)
	{
		arma::mat convoluted_vectors(data->n_rows, data->n_cols);

		//calculate convoluted vectors centered around each column of the matrix
		for(unsigned int j = 0; j < data->n_cols; j++)
		{
			arma::mat sub_mat;
			
			//apply padding vectors on the left-most part of the X-axis
			if(j < ((KERNEL_WIDTH - 1) / 2))
			{
				unsigned int padding, index = 0;
				for(padding = j; padding < ((KERNEL_WIDTH - 1) / 2); padding++)
					sub_mat.insert_cols(index++,arma::colvec(data->n_rows, arma::fill::zeros));
				for(int remaining = 0; padding < KERNEL_WIDTH; padding++)
					sub_mat.insert_cols(index++, data->slice(slice).col(remaining++));
			}
			//apply padding vectors to right-most part of the X-axis
			else if(j >= data->n_cols - ((KERNEL_WIDTH - 1) / 2))
			{
				unsigned int index = 0;
				for(arma::uword adding = (j - ((KERNEL_WIDTH - 1) / 2)); adding < data->n_cols; adding++)
					sub_mat.insert_cols(index++, data->slice(slice).col(adding));
				for(; index < KERNEL_WIDTH; index++)
					sub_mat.insert_cols(index,arma::colvec(data->n_rows, arma::fill::zeros));
			}
			//no padding needed
			else
			{
				sub_mat = data->slice(slice).cols(j - ((KERNEL_WIDTH - 1) / 2), j + ((KERNEL_WIDTH - 1) / 2) );
			}

			convoluted_vectors.col(j) = arma::sum( (sub_mat % kernel) , 1);
		}
		data->slice(slice) = convoluted_vectors;
		activation_func(&data->slice(slice), "relu");
	}
    if(batch_norm == NULL)
        batch_norm = new BatchNorm(data);
    else
        batch_norm->set_data(data);
    batch_norm->normalize();
    maxpooling();
}

void Convolution::maxpooling()
{
	max_cells = arma::mat(data->n_rows, data->n_cols);
    arma::mat temp[INPUT_BATCH_SIZE];
	int new_column_size = 0;
	//for each song in the batch
	for(arma::uword slice = 0; slice < data->n_slices; slice++)
	{
		arma::mat pooled_song;
		int index = 0;
		//pooling based on step size for no overlap
		for(arma::uword j = 0; j < data->n_cols; j+= KERNEL_WIDTH)
		{
			if(j + KERNEL_WIDTH > data->n_cols - 1)
			{
				arma::mat step = data->slice(slice).submat(0, j, DATA_ROWS - 1, data->n_cols - 1);
				arma::colvec max_cell_vec = arma::max(step, 1);
				pooled_song.insert_cols(index++, max_cell_vec);
			}
				
			else
			{
				arma::mat step = data->slice(slice).submat(0, j, DATA_ROWS - 1, j + KERNEL_WIDTH - 1);
				arma::colvec max_cell_vec = arma::max(step, 1);
				pooled_song.insert_cols(index++, max_cell_vec);
			}					
		}
		//storing new column size and each pooled entry
		new_column_size = pooled_song.n_cols;
		temp[slice] = pooled_song;
	}
	//setting size of data post max-pooling
	data->set_size(DATA_ROWS, new_column_size, INPUT_BATCH_SIZE);
	
	//re-assigning data with calculated max-pools
	for(arma::uword i = 0; i < data->n_slices; i++)
		data->slice(i) = temp[i];
}

void Convolution::set_data(arma::cube * data)
{
    this->data = data;
	this->INPUT_BATCH_SIZE = data->n_slices;
}

arma::cube Convolution::back_propagation(arma::cube delta_error)
{
	//steps: maxpool -> batch norm -> convolving


}

Convolution::~Convolution()
{
}
