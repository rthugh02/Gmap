#include <iostream>
#include <stdlib.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <queue>
#include <random>
#include <thread>
#include <condition_variable>
#include <algorithm>
#include <armadillo>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "InputBatch.h"
#include "Convolution.h"
#include "LSTMCell.h"
#include "DenseNetwork.h"
#include "BatchNorm.h"
#include "Song.h"
/*
	TROUBLESHOOTING:

	-If can't open a library in application lib, run sudo ldconfig /home/rthugh02/Documents/Masters_Project/NN_Project/lib
	to refresh library cache

	TODO AND NOTES:

	Layers/steps to build:
	- Add bias to all needed layers
*/

//*****************************//
//*********FUNCTIONS***********//
//*****************************//

void convolution(arma::cube *);
void LSTM(arma::cube *);
arma::mat dense_layer(arma::cube *);
void max_pooling(arma::cube *, int);
void train(std::vector<std::string>);
void back_propagation(arma::mat, arma::mat);
void loss_and_accuracy(arma::mat, arma::mat);
void convert_data(std::vector<std::string>, int);
InputBatch * convert_validation_data(std::vector<std::string>);
arma::rowvec genre_to_output(const char *);
const char * output_to_genre(arma::rowvec);
arma::mat feed_forward(InputBatch *);
void activation_function(arma::mat *, const char *);

//*****************************//
//*********CONSTANTS***********//
//*****************************//

//Batch size of calculations that will be done for back propogation
const int INPUT_BATCH_SIZE = 50;
//Data dimensions
const int DATA_ROWS = 128;
const int DATA_ROW_LENGTH = 1294;
//convolution layers settings
const int KERNEL_WIDTH = 3;
//Output Neurons, there are 8 genre classifications
const int OUTPUT_COUNT = 8;
//number of threads to use for parsing files
const int THREAD_COUNT = 11;

const int LSTM_TIMESTEP = 16;

//*******************************//
//*********SHARED VARS***********//
//*******************************//

//keep running count of threads that are still parsing files so train thread knows when to stop
int unfinished_threads = THREAD_COUNT;
//mutex for thread safe enqueueing/dequeueing
std::mutex queue_mutex;
//mutex for thread safe decrementing of unfinished_threads
std::mutex finish_mutex;
//storing names of files for training
std::vector<std::string> training_files;
//names of files for validation
std::vector<std::string> validation_files;
//condition variable used by train thread to make the assigned thread block when queue is empty
std::condition_variable cond;
//queue shared by threads
std::queue<InputBatch *> input_queue;
//Convolution Layers with maxpooling
Convolution * convolution_layer1 = NULL;
Convolution * convolution_layer2 = NULL;
Convolution * convolution_layer3 = NULL;
//LSTM_cells for each mel-spec row
std::vector<LSTMCell> LSTM_cells;
BatchNorm * LSTM_batch_norm = NULL;
//Dense network layer
DenseNetwork * dense_network = NULL;

int main() 
{
	std::random_device rd;
	//threads that will parse song_data directory and generate input data for NN
	std::vector<std::thread> directory_threads;
	
	//getting all file paths and randomly shuffling them
	for(const auto & file : std::filesystem::directory_iterator("song_data/"))
		training_files.push_back(file.path().string());
	std::shuffle(std::begin(training_files), std::end(training_files), rd);
	
	int validation_size = 100;

	for(int i = 0; i < validation_size; i++)
	{
		validation_files.push_back(training_files[i]);
	}
	
	training_files.erase(training_files.begin(), training_files.begin() + validation_size);

	//thread dedicated to feed-forward/back-propogation of NN
	std::thread train_thread(train, validation_files);

	//Iterating through all song data that will be used as input
	int song_count = 0;
	std::vector<std::string> thread_tasks[THREAD_COUNT];

	for(auto file : training_files)
		thread_tasks[song_count++ % THREAD_COUNT].push_back(file);

	//launching threads for converting song_data files
	for(auto & task : thread_tasks)
		directory_threads.emplace_back(convert_data, task, INPUT_BATCH_SIZE);

	//detaching directory threads
	for(auto & thread : directory_threads)
		thread.detach();
		
	train_thread.join();
}

//Function for retreiving batches of song data from queue for feed-forward/backpropogation
void train(std::vector<std::string> validation_files)
{
	std::cout << "constructing validation set.." << std::endl;
	InputBatch * validation_batch = convert_validation_data(validation_files);
	int epoch = 0;
	while(epoch < 50)
	{
		//while there are directory threads still doing work
		while(unfinished_threads > 0) 
		{
			//if queue is empty block this thread to not waste CPU time polling 
			std::unique_lock<std::mutex> lock(queue_mutex);
			while (input_queue.empty())
				cond.wait(lock);

			InputBatch * next = input_queue.front();
			input_queue.pop();

			arma::mat predictions = feed_forward(next);
			back_propagation(predictions, *(next->genres));
			next->free();
			//std::cout << "item " << ++batch_count << " loss: " << loss << std::endl;
		}
		while(!input_queue.empty())
		{
			InputBatch * next = input_queue.front();
			input_queue.pop();

			arma::mat predictions = feed_forward(next);
			back_propagation(predictions, *(next->genres));
			next->free();
			//std::cout << "item " << ++batch_count << " loss: " << loss << std::endl;
		}
		//running validation after epoch completes
		InputBatch * validation_input = new InputBatch(*(validation_batch->data), *(validation_batch->genres));
		arma::mat validation_predictions = feed_forward(validation_input);
		std::cout << "validation loss and accuracy: " << std::endl;
		loss_and_accuracy(validation_predictions, *(validation_input->genres));
		validation_input->free();

		//resetting directory thread counting and shuffling files
		unfinished_threads = THREAD_COUNT;
		std::random_device rd;
		std::shuffle(std::begin(training_files), std::end(training_files), rd);

		//kicking off new threads for reading in song data
		std::vector<std::thread> directory_threads;
		std::vector<std::string> thread_tasks[THREAD_COUNT];
		int song_count = 0;
		for(auto file : training_files)
			thread_tasks[song_count++ % THREAD_COUNT].push_back(file);
		for(auto & task : thread_tasks)
			directory_threads.emplace_back(convert_data, task, INPUT_BATCH_SIZE);
		
		for(auto & thread : directory_threads)
			thread.detach();
		epoch++;
	}	
}

arma::mat feed_forward(InputBatch * input)
{
	//Process: convolution -> LSTM -> dense -> output
		
	convolution(input->data);
	
	LSTM(input->data);

	return dense_layer(input->data);
}

void convolution(arma::cube * data)
{
	//convolution layer 1
	if(convolution_layer1 == NULL)
		convolution_layer1 = new Convolution(data, DATA_ROWS, KERNEL_WIDTH);
	else
		convolution_layer1->set_data(data);
	convolution_layer1->convolve(activation_function);

	//convolution layer 2
	if(convolution_layer2 == NULL)
		convolution_layer2 = new Convolution(data, DATA_ROWS, KERNEL_WIDTH);
	else
		convolution_layer2->set_data(data);
	convolution_layer2->convolve(activation_function);

	//convolution layer 3
	if(convolution_layer3 == NULL)
		convolution_layer3 = new Convolution(data, DATA_ROWS, KERNEL_WIDTH);
	else
		convolution_layer3->set_data(data);
	convolution_layer3->convolve(activation_function);
}

/*
Notes on LSTM:

LSTM layers are not densely connected LSTM cells, each cell operates on itself in a loop
an LSTM layer accepts all the inputs from the layer leading into it
View the graph in this link for conceptual view: https://www.quora.com/In-LSTM-how-do-you-figure-out-what-size-the-weights-are-supposed-to-be

*/
void LSTM(arma::cube * data)
{
	arma::mat temp[DATA_ROWS];
	int split_column_width = data->n_cols / LSTM_TIMESTEP;
	for(int i = 0; i < DATA_ROWS; i++)
	{
		arma::mat row_slice = data->tube(i, 0, i, data->n_cols - 1);
		arma::inplace_trans(row_slice);
		if(LSTM_cells.size() < DATA_ROWS)
			LSTM_cells.emplace_back(row_slice, data->n_slices, split_column_width, LSTM_TIMESTEP);
		else
			LSTM_cells[i].set_data(row_slice, data->n_slices);
		temp[i] = LSTM_cells[i].calculate_output(activation_function);
	}
	data->set_size(DATA_ROWS, temp[0].n_cols, data->n_slices);
	for(int i = 0; i < DATA_ROWS; i++)
		data->tube(i, 0, i, data->n_cols - 1) = temp[i].t();

	if(LSTM_batch_norm == NULL)
		LSTM_batch_norm = new BatchNorm(data);
	else
		LSTM_batch_norm->set_data(data);
	LSTM_batch_norm->normalize();
}

arma::mat dense_layer(arma::cube * data)
{
	if(dense_network == NULL)
		dense_network = new DenseNetwork(data);
	else
		dense_network->set_data(data);
	return dense_network->calculate_output(activation_function);
	
}
int loss_count = 0;
void back_propagation(arma::mat predictions, arma::mat correct_output)
{
	//backwards through dense_network

	arma::cube delta_error_wr2_lstm_BN_out = dense_network->back_propagation(predictions, correct_output);

	//backwards through LSTM_batch_norm

	arma::cube delta_error_wr2_lstm_BN_in = LSTM_batch_norm->back_propagation(delta_error_wr2_lstm_BN_out);

	//backwards through LSTM
	
	arma::mat temp[DATA_ROWS];
	for(int i = 0; i < DATA_ROWS; i++)
	{
		arma::mat delta_error_row_slice = delta_error_wr2_lstm_BN_in.tube(i, 0, i, delta_error_wr2_lstm_BN_in.n_cols - 1);
		arma::inplace_trans(delta_error_row_slice);
		temp[i] = LSTM_cells[i].back_propagation(delta_error_row_slice, activation_function);
	}
	arma::cube delta_error_wr2_conv3_out = arma::cube(DATA_ROWS, temp[0].n_cols, predictions.n_rows);
	for(int i = 0; i < DATA_ROWS; i++)
		delta_error_wr2_conv3_out.tube(i, 0, i, delta_error_wr2_conv3_out.n_cols - 1) = temp[i].t();

	//backwards through convolution layers
	convolution_layer1->back_propagation(
		convolution_layer2->back_propagation(
			convolution_layer3->back_propagation(delta_error_wr2_conv3_out)));

	//predictions.row(0).print("prediction: ");
	//correct_output.row(0).print("label: ");
	loss_and_accuracy(predictions, correct_output);
}

void loss_and_accuracy(arma::mat predictions, arma::mat correct_output)
{
	//calculating accuracy
	arma::colvec genre_guess = arma::max(predictions, 1);
	arma::colvec guess_for_correct_genre = arma::max(predictions % correct_output, 1);
	int correct = arma::sum(genre_guess == guess_for_correct_genre);
	int total = predictions.n_rows;
	
	//calculating loss
	predictions.transform([&] (double val) { return log(val); });
	predictions %= correct_output;

	double loss = -arma::mean(
		arma::sum(predictions, 1)
	);

	std::cout << ++loss_count <<" loss: " << loss;
	std::cout << " accuracy: " << (double)correct / total << std::endl;
}

void activation_function(arma::mat * input, const char * function)
{
	if(strcmp("relu", function) == 0)
		input->transform([] (double val) { return std::max(0.0, val); } );
	else if(strcmp("leakyrelu", function) == 0)
		input->transform([] (double val) { return (val < 0) ? 0.01*val : val; } );
	else if(strcmp("sigmoid", function) == 0)
		input->transform([] (double val) { return (1 / (1 + exp(-val))); } );
	else if(strcmp("tanh", function) == 0)
		input->transform([] (double val) { return (exp(val) - exp(-val)) / (exp(val) + exp(-val)) ; } );
	else if(strcmp("tanh2", function) == 0)
		input->transform([] (double val) { double temp = (exp(val) - exp(-val)) / (exp(val) + exp(-val)) ; return temp * temp; } );
	else if(strcmp("softmax", function) == 0)
	{

		//getting max of each row into column vector
		arma::colvec row_max = arma::max(*input, 1);

		//expanding row max to be a matrix to perform max subtraction trick
		arma::mat max_subtractor(input->n_rows,0);
		for(arma::uword i = 0; i < input->n_cols; i++)
			max_subtractor.insert_cols(i, row_max);
		
		//subtracting max value of each row from each value in the row to prevent extremely large exponentiation
		*(input) -= max_subtractor;

		//exponentiating data to start softmax 
		input->transform([] (double val) { return exp(val); } );

		//summation of each exponentiated row
		arma::colvec row_summation = arma::sum(*input, 1);
		
		//creating matrix to perform Hadamard product to get softmax probabilities
		arma::mat sum_multiplier (input->n_rows, 0);
		for(arma::uword i = 0; i < input->n_cols; i++)
			sum_multiplier.insert_cols(i, row_summation);
		
		*(input) %= (1 / sum_multiplier);
	}
}

/*files assigned to thread are parsed and used to create matrix rows that are inserted into
matrices of row size INPUT_BATCH_SIZE. These matrices are then enqueued and consumed 
by the training thread.
*/
void convert_data(std::vector<std::string> files, int batch_size)
{
	//JSON parser
	rapidjson::Document document;
	int row_counter = 0;

	std::vector<arma::mat> song_buffer;
	std::vector<arma::rowvec> genre_buffer; 

	//closure for creating batch of song data and enqueueing 
	auto build_batch = [&]() {
			//song batch data
			arma::cube * input_batch = new arma::cube(DATA_ROWS, DATA_ROW_LENGTH, row_counter);
			arma::mat * correct_output = new arma::mat(0, OUTPUT_COUNT);
			row_counter = 0;
			for(auto it = song_buffer.begin(); it != song_buffer.end(); it++)
			{
				input_batch->slice(row_counter) = *it;
				correct_output->insert_rows(row_counter, genre_buffer[row_counter]);
				row_counter++;
			}
			//thread safe enqueuing and notify training thread there is work to do
			queue_mutex.lock();
			input_queue.push(new InputBatch(input_batch, correct_output));
			cond.notify_one();
			queue_mutex.unlock();

			song_buffer.clear();
			genre_buffer.clear();
			row_counter = 0;
	};
	for(const auto & file : files)
	{
		if(row_counter < batch_size)
		{
			//Reading song data into string
			std::ifstream f(file);
			std::string content( (std::istreambuf_iterator<char>(f) ),
        	            (std::istreambuf_iterator<char>()));

			//parsing json data 
			document.Parse(content.c_str());
			auto raw_data = document["data"].GetArray();
			auto genre = document["genre"].GetString();

			arma::mat spectogram_data(0, DATA_ROW_LENGTH);
			for(rapidjson::SizeType i = 0; i < raw_data.Size(); i++) 
    		{
				std::vector<double> temp;
				int inner_row_count = 0;
        		rapidjson::Value& row = raw_data[i];
        		for(rapidjson::SizeType j = 0; j < row.Size(); j++)
				{
					temp.push_back(row[j].GetDouble());
					inner_row_count++;
				}
				//adding missing time data to end if not long enough
				while(inner_row_count < DATA_ROW_LENGTH)
				{
					temp.push_back(0);
					inner_row_count++;
				}
				spectogram_data.insert_rows((int)i, arma::rowvec(temp));				
    		}

			song_buffer.push_back(spectogram_data);
			genre_buffer.push_back(genre_to_output(genre));
			row_counter++;
		}
		else
			build_batch();	
	}
	if(row_counter > 0)
		build_batch();
	
	finish_mutex.lock();
	unfinished_threads--;
	finish_mutex.unlock();
}

InputBatch * convert_validation_data(std::vector<std::string> validation_files)
{
	//JSON parser
	rapidjson::Document document;
	int row_counter = 0;

	std::vector<arma::mat> song_buffer;
	std::vector<arma::rowvec> genre_buffer; 
	InputBatch * ret = NULL;

	auto build_batch = [&]() {
			//song batch data
			arma::cube * input_batch = new arma::cube(DATA_ROWS, DATA_ROW_LENGTH, row_counter);
			arma::mat * correct_output = new arma::mat(0, OUTPUT_COUNT);
			row_counter = 0;
			for(auto it = song_buffer.begin(); it != song_buffer.end(); it++)
			{
				input_batch->slice(row_counter) = *it;
				correct_output->insert_rows(row_counter, genre_buffer[row_counter]);
				row_counter++;
			}
			ret = new InputBatch(input_batch, correct_output);
	};

	for(const auto & file : validation_files)
	{
		//Reading song data into string
		std::ifstream f(file);
		std::string content( (std::istreambuf_iterator<char>(f) ),
                    (std::istreambuf_iterator<char>()));
		//parsing json data 
		document.Parse(content.c_str());
		auto raw_data = document["data"].GetArray();
		auto genre = document["genre"].GetString();
		arma::mat spectogram_data(0, DATA_ROW_LENGTH);
		for(rapidjson::SizeType i = 0; i < raw_data.Size(); i++) 
    	{
			std::vector<double> temp;
			int inner_row_count = 0;
        	rapidjson::Value& row = raw_data[i];
        	for(rapidjson::SizeType j = 0; j < row.Size(); j++)
			{
				temp.push_back(row[j].GetDouble());
				inner_row_count++;
			}
			//adding missing time data to end if not long enough
			while(inner_row_count < DATA_ROW_LENGTH)
			{
				temp.push_back(0);
				inner_row_count++;
			}
			spectogram_data.insert_rows((int)i, arma::rowvec(temp));				
    	}
		song_buffer.push_back(spectogram_data);
		genre_buffer.push_back(genre_to_output(genre));
		row_counter++;	
	}
	build_batch();
	return ret;
}

/*
	mapping:
	
	Hip-Hop			1 0 0 0 0 0 0 0
	Pop				0 1 0 0 0 0 0 0
	Electronic		0 0 1 0 0 0 0 0
	Instrumental	0 0 0 1 0 0 0 0
	International	0 0 0 0 1 0 0 0
	Folk			0 0 0 0 0 1 0 0
	Experimental	0 0 0 0 0 0 1 0
	Rock			0 0 0 0 0 0 0 1
*/

arma::rowvec genre_to_output(const char * genre)
{
	if(strcmp("Hip-Hop", genre) == 0)
		return arma::rowvec("1 0 0 0 0 0 0 0");
	else if(strcmp("Pop", genre) == 0)
		return arma::rowvec("0 1 0 0 0 0 0 0");
	else if(strcmp("Electronic", genre) == 0)
		return arma::rowvec("0 0 1 0 0 0 0 0");
	else if(strcmp("Instrumental", genre) == 0)
		return arma::rowvec("0 0 0 1 0 0 0 0");
	else if(strcmp("International", genre) == 0)
		return arma::rowvec("0 0 0 0 1 0 0 0");
	else if(strcmp("Folk", genre) == 0)
		return arma::rowvec("0 0 0 0 0 1 0 0");
	else if(strcmp("Experimental", genre) == 0)
		return arma::rowvec("0 0 0 0 0 0 1 0");
	else //Rock
		return arma::rowvec("0 0 0 0 0 0 0 1");
}

const char * output_to_genre(arma::rowvec output)
{
	if(output[0] == 1)
		return "Hip-Hop";
	else if(output[1] == 1)
		return "Pop";
	else if(output[2] == 1)
		return "Electronic";	
	else if(output[3] == 1)
		return "Instrumental";	
	else if(output[4] == 1)
		return "International";	
	else if(output[5] == 1)
		return "Folk";	
	else if(output[6] == 1)
		return "Experimental";	
	else
		return "Rock";		
}
