#include "Convolution.h"
#include <armadillo>
Convolution::Convolution(arma::cube * data, int data_rows, int kernel_width)
{
    this->data = data;
	this->data_copy = *(data);
    this->DATA_ROWS = data_rows;
    this->KERNEL_WIDTH = kernel_width;
    this->INPUT_BATCH_SIZE = data->n_slices;
	this->pre_relu = arma::cube(data->n_rows, data->n_cols, data->n_slices);

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
				{
					sub_mat.insert_cols(index++,arma::colvec(data->n_rows, arma::fill::zeros));
				}
					
				for(int remaining = 0; padding < KERNEL_WIDTH; padding++)
				{
					sub_mat.insert_cols(index++, data->slice(slice).col(remaining++));
				}
			}
			//apply padding vectors to right-most part of the X-axis
			else if(j >= data->n_cols - ((KERNEL_WIDTH - 1) / 2))
			{
				unsigned int index = 0;
				for(arma::uword adding = (j - ((KERNEL_WIDTH - 1) / 2)); adding < data->n_cols; adding++)
				{
					sub_mat.insert_cols(index++, data->slice(slice).col(adding));
				}
					
				for(; index < KERNEL_WIDTH; index++)
				{
					sub_mat.insert_cols(index,arma::colvec(data->n_rows, arma::fill::zeros));
				}
			}
			//no padding needed
			else
			{
				sub_mat = data->slice(slice).cols(j - ((KERNEL_WIDTH - 1) / 2), j + ((KERNEL_WIDTH - 1) / 2) );
			}

			convoluted_vectors.col(j) = arma::sum( (sub_mat % kernel) , 1);
		}
		data->slice(slice) = convoluted_vectors;
		pre_relu.slice(slice) = convoluted_vectors;
		activation_func(&data->slice(slice), "leakyrelu");
	}
    if(batch_norm == NULL)
        batch_norm = new BatchNorm(data);
    else
        batch_norm->set_data(data);
    batch_norm->normalize();
    maxpooling();
}

arma::cube Convolution::convolve_back_propagation(arma::cube delta_error)
{
	//output of convolution is passed through leakyRelu, calculating gradient of that:

	pre_relu.transform([&] (double val) { return val > 0 ? 1 : 0.01; });;

	arma::cube delta_error_wr2_convolve_out = pre_relu % delta_error;

	arma::cube delta_error_wr2_convolve_in = arma::cube(delta_error.n_rows, delta_error.n_cols, delta_error.n_slices, arma::fill::zeros);

	arma::mat delta_error_wr2_kernel = arma::mat(kernel.n_rows, kernel.n_cols, arma::fill::zeros);

	for(arma::uword slice = 0; slice < delta_error_wr2_convolve_in.n_slices; slice++)
	{
		for(arma::uword j = 0; j < data_copy.n_cols; j++)
		{
			arma::mat sub_mat_input;
			arma::mat sub_mat_delta = arma::mat(delta_error_wr2_convolve_out.n_rows, KERNEL_WIDTH);
			//apply padding vectors on the left-most part of the X-axis
			if(j < ((KERNEL_WIDTH - 1) / 2))
			{
				unsigned int padding, index = 0;
				for(padding = j; padding < ((KERNEL_WIDTH - 1) / 2); padding++)
				{
					sub_mat_input.insert_cols(index,arma::colvec(data_copy.n_rows, arma::fill::zeros));
					sub_mat_delta.col(index) = delta_error_wr2_convolve_out.slice(slice).col(j + ((KERNEL_WIDTH - 1) / 2) - index);
					index++;
				}
					
				for(int remaining = 0; padding < KERNEL_WIDTH; padding++)
				{
					sub_mat_input.insert_cols(index, data_copy.slice(slice).col(remaining++));
					//underflow of unsigned expression
					if(j + ((KERNEL_WIDTH - 1) / 2) - index < delta_error_wr2_convolve_out.n_cols - 1)
						sub_mat_delta.col(index) = delta_error_wr2_convolve_out.slice(slice).col(j + ((KERNEL_WIDTH - 1) / 2) - index);
					else
						sub_mat_delta.col(index) = arma::colvec(data_copy.n_rows, arma::fill::zeros);
					index++;
				}
					
			}
			//apply padding vectors to right-most part of the X-axis
			else if(j >= data_copy.n_cols - ((KERNEL_WIDTH - 1) / 2))
			{
				unsigned int index = 0;
				for(arma::uword adding = (j - ((KERNEL_WIDTH - 1) / 2)); adding < data_copy.n_cols; adding++)
				{
					sub_mat_input.insert_cols(index, data_copy.slice(slice).col(adding));
					
					if(j + ((KERNEL_WIDTH - 1) / 2) - index > data_copy.n_cols - 1)
						sub_mat_delta.col(index) = arma::colvec(data_copy.n_rows, arma::fill::zeros);
					else
						sub_mat_delta.col(index) = delta_error_wr2_convolve_out.slice(slice).col(j + ((KERNEL_WIDTH - 1) / 2) - index);
					index++;
				}

				for(; index < KERNEL_WIDTH; index++)
				{
					sub_mat_input.insert_cols(index,arma::colvec(data_copy.n_rows, arma::fill::zeros));
					sub_mat_delta.col(index) = delta_error_wr2_convolve_out.slice(slice).col(j + ((KERNEL_WIDTH - 1) / 2) - index);	
				}
					
			}
			//no padding needed
			else
			{
				sub_mat_input = data_copy.slice(slice).cols(j - ((KERNEL_WIDTH - 1) / 2), j + ((KERNEL_WIDTH - 1) / 2) );
				unsigned int index = 0; 
				for(arma::uword i = j + ((KERNEL_WIDTH - 1) / 2); i > j - ((KERNEL_WIDTH - 1) / 2); i--)
					sub_mat_delta.col(index++) = delta_error_wr2_convolve_out.slice(slice).col(i);

			}
			//calculating change in kernel weights with respect to each step as the kernel moves across
			delta_error_wr2_kernel += sub_mat_input.each_col() % delta_error_wr2_convolve_out.slice(slice).col(j);
			delta_error_wr2_convolve_in.slice(slice).col(j) = arma::sum(sub_mat_delta % kernel, 1); 
		}		
	}
	delta_error_wr2_kernel /= delta_error_wr2_convolve_in.n_slices;


	kernel -= (delta_error_wr2_kernel * learning_rate);

	return delta_error_wr2_convolve_in;
}

void Convolution::maxpooling()
{
	max_cells = arma::umat(data->n_rows, data->n_cols);
    arma::mat temp[INPUT_BATCH_SIZE];
	int new_column_size = 0;
	//for each song in the batch
	for(arma::uword slice = 0; slice < data->n_slices; slice++)
	{
		arma::mat pooled_song;
		int col_counter = 0;
		int index = 0;
		//pooling based on step size for no overlap
		for(arma::uword j = 0; j < data->n_cols; j+= KERNEL_WIDTH)
		{
			if(j + KERNEL_WIDTH > data->n_cols - 1)
			{
				arma::mat step = data->slice(slice).submat(0, j, DATA_ROWS - 1, data->n_cols - 1);
				arma::colvec max_cell_vec = arma::max(step, 1);
				pooled_song.insert_cols(index++, max_cell_vec);
				for(arma::uword k = 0; k < step.n_cols; k++)
					{
						arma::colvec step_copy = step.col(k);
						arma::umat result = (step_copy == max_cell_vec);	
						max_cells.col(col_counter++) = result;
					}
			}
			else
			{
				arma::mat step = data->slice(slice).submat(0, j, DATA_ROWS - 1, j + KERNEL_WIDTH - 1);
				arma::colvec max_cell_vec = arma::max(step, 1);
				pooled_song.insert_cols(index++, max_cell_vec);
				for(arma::uword k = 0; k < step.n_cols; k++)
					{
						arma::colvec step_copy = step.col(k);
						arma::umat result = (step_copy == max_cell_vec);
						max_cells.col(col_counter++) = result;
					}
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

arma::cube Convolution::max_pool_back_propagation(arma::cube delta_error)
{
	arma::cube delta_error_wr2_maxpool_in = arma::cube(max_cells.n_rows, max_cells.n_cols, delta_error.n_slices);

	for(arma::uword slice = 0; slice < delta_error_wr2_maxpool_in.n_slices; slice++)
	{
		int gradient_col = 0;
		for(arma::uword i = 0; i < delta_error_wr2_maxpool_in.n_cols; i+=KERNEL_WIDTH)
		{
			if(i + KERNEL_WIDTH - 1 < max_cells.n_cols)
			{
				for(arma::uword j = i; j < i + KERNEL_WIDTH; j++)
				{
					delta_error_wr2_maxpool_in.slice(slice).col(j) = max_cells.col(j) % delta_error.slice(slice).col(gradient_col);
				}
			}
			else
			{
				for(arma::uword j = i; i < max_cells.n_cols - 1; j++)
				{
					delta_error_wr2_maxpool_in.slice(slice).col(j) = max_cells.col(j) % delta_error.slice(slice).col(gradient_col);
				}
			}
			
		}
	}
	return delta_error_wr2_maxpool_in;
}

arma::cube Convolution::back_propagation(arma::cube delta_error)
{
	//steps: maxpool -> batch norm -> convolving

	arma::cube delta_error_wr2_maxpool_in = max_pool_back_propagation(delta_error);

	arma::cube delta_error_wr2_batchnorm_in = batch_norm->back_propagation(delta_error_wr2_maxpool_in);

	return convolve_back_propagation(delta_error_wr2_batchnorm_in);
}

void Convolution::set_data(arma::cube * data)
{
    this->data = data;
	this->INPUT_BATCH_SIZE = data->n_slices;
	this->pre_relu = arma::cube(data->n_rows, data->n_cols, data->n_slices);
}

Convolution::~Convolution()
{
}
