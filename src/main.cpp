#include <iostream>
#include <stdlib.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <queue>
#include <random>
#include <thread>
#include <condition_variable>
#include <armadillo>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "InputBatch.h"
/*
	TROUBLESHOOTING:

	-If can't open a library in application lib, run sudo ldconfig /home/rthugh02/Documents/Masters_Project/NN_Project/lib
	to refresh library cache

	TODO AND NOTES:

	Layers/steps to build:
	
	1. 1D Convolution layer and Maxpooling
		review these links, will probably need to change out data is represented to best facilitate convolution process using matrix operations  
		- https://towardsdatascience.com/understanding-1d-and-3d-convolution-neural-network-keras-9d8f76e29610
		- https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/
	2. add gamma and beta to batch normalization
	3. LSTM 
	4. Add bias to feed-forward

*/

//*****************************//
//*********FUNCTIONS***********//
//*****************************//

void train(arma::mat *, arma::mat *, arma::mat *);
void convert_data(std::vector<std::string>);
arma::rowvec genre_to_output(const char *);
const char * output_to_genre(arma::rowvec);
void feed_forward(InputBatch *, arma::mat *, arma::mat *, arma::mat *);
void activation_function(arma::mat *, const char *);
void batch_normalization(arma::mat *);

//*****************************//
//*********CONSTANTS***********//
//*****************************//

//Batch size of calculations that will be done for back propogation
const int INPUT_BATCH_SIZE = 50;
//Input Neurons
const int INPUT_COUNT = 165632; //Max dimensions 128 X 1294
const int DATA_ROWS = 128;
const int DATA_ROW_LENGTH = 1294;
//Number of neurons in first hidden layer
const int LAYER_ONE_INPUT = 24;
//Number of neurons in second hidden layer
const int LAYER_TWO_INPUT = 12;
//Output Neurons, there are 8 genre classifications
const int OUTPUT_COUNT = 8;
//min(-) max(+) possible weight
const double MAX_WEIGHT = .25;
//number of threads to use for parsing files
const int THREAD_COUNT = 11;

//*******************************//
//*********SHARED VARS***********//
//*******************************//

//keep running count of threads that are still parsing files so train thread knows when to stop
int unfinished_threads = THREAD_COUNT;
//mutex for thread safe enqueueing/dequeueing
std::mutex queue_mutex;
//mutex for thread safe decrementing of unfinished_threads
std::mutex finish_mutex;
//condition variable used by train thread to make the assigned thread block when queue is empty
std::condition_variable cond;
//queue shared by threads
std::queue<InputBatch *> input_queue;


int main() {
	
	//Random seed for initializing weights
	std::random_device rd;
	
	//Uniform distribution of real numbers
	std::uniform_real_distribution<double> distr(-MAX_WEIGHT, MAX_WEIGHT);

	//weights matrix following Input neurons -> 1st hidden layer
	arma::mat * weights1 = new arma::mat(INPUT_COUNT, LAYER_ONE_INPUT);
	//weights matrix following 1st hidden layer -> 2nd hidden layer
	arma::mat * weights2 = new arma::mat(LAYER_ONE_INPUT, LAYER_TWO_INPUT);
	//weights matrix following 2nd hidden layer -> output neurons
	arma::mat * weights3 = new arma::mat(LAYER_TWO_INPUT, OUTPUT_COUNT);
	
	//mersenne twister engine for generating random values
	std::mt19937 engine(rd());

	//filling weight matrices with random values generated by mersenne engine by cell
	weights1->imbue( [&]() {return distr(engine); } );
	engine.seed(rd());
	weights2->imbue( [&]() {return distr(engine); } );
	engine.seed(rd());
	weights3->imbue( [&]() {return distr(engine); } );

	//threads that will parse song_data directory and generate input data for NN
	std::vector<std::thread> directory_threads;

	//thread dedicated to feed-forward/back-propogation of NN
	std::thread train_thread(train, weights1, weights2, weights3);
	
	//Iterating through all song data that will be used as input
	int song_count = 0;
	std::vector<std::string> thread_tasks[THREAD_COUNT];
	//assigning each file to a thread
	for(const auto & file : std::filesystem::directory_iterator("song_data/"))
		thread_tasks[song_count++ % THREAD_COUNT].push_back(file.path().string());
	
	//launching threads for converting song_data files
	for(auto & task : thread_tasks)
		directory_threads.emplace_back(convert_data, task);

	//joining threads
	for(auto & thread : directory_threads)
		thread.join();
	train_thread.join();
}

//Function for retreiving batches of song data from queue for feed-forward/backpropogation
void train(arma::mat * layer1, arma::mat * layer2, arma::mat * layer3)
{
	//while there are directory threads still doing work
	int batch_count = 0;
	while(unfinished_threads > 0) 
	{
		//if queue is empty block this thread so as not to waste CPU time polling 
		std::unique_lock<std::mutex> lock(queue_mutex);
		while (input_queue.empty())
			cond.wait(lock);
		
		InputBatch * next = input_queue.front();
		input_queue.pop();

		feed_forward(next, layer1, layer2, layer3);
		next->free();
		std::cout << "consuming item " << ++batch_count << " from queue" << std::endl;
	}
}

//TODO: add bias to this process
void feed_forward(InputBatch * input, arma::mat * layer1, arma::mat * layer2, arma::mat * layer3)
{
	
	//Process: 1D convolution -> ReLu -> Batch normalization -> maxpooling
	//batch_normalization(input->data);
	//*(input->data) = *(input->data) * *(layer1);
	//activation_function(input->data, "relu");
	//*(input->data) = *(input->data) * *(layer2);
	//activation_function(input->data, "relu");
	//*(input->data) = *(input->data) * *(layer3);
	//activation_function(input->data, "softmax");
}

//TODO: Make sure to review softmax code for correctness

void activation_function(arma::mat * input, const char * function)
{
	if(strcmp("relu", function) == 0)
		input->transform([] (double val) { return std::max(0.0, val); } );
	else if(strcmp("leakyrelu", function) == 0)
		input->transform([] (double val) { return (val < 0) ? 0.01*val : val; } );
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

		//input->print("submax:");

		//exponentiating data to start softmax 
		input->transform([] (double val) { return exp(val); } );

		//input->print("exponentiate:");
		//summation of each exponentiated row
		arma::colvec row_summation = arma::sum(*input, 1);
		
		//creating matrix to perform Hadamard product to get softmax probabilities
		arma::mat sum_multiplier (input->n_rows, 0);
		for(arma::uword i = 0; i < input->n_cols; i++)
			sum_multiplier.insert_cols(i, row_summation);
		

		*(input) %= (1 / sum_multiplier);

		//input->print("output:");
		
	}
}

//TODO: Add in gamma and beta scale/shift values that can be trained
//Normalization such that norm(val) = (val - mean) / sqrt(variance + er)
void batch_normalization(arma::mat * batch) 
{
	arma::rowvec feature_means = arma::sum(*batch, 0) / batch->n_rows;
	arma::rowvec feature_variances = arma::sum(
		(batch->each_row() - feature_means).transform([] (double val) { return val*val; } ), 0) / batch->n_rows; 
	
	//denominator of normalization formula
	feature_variances.transform([] (double val) { return sqrt(val + 0.0001); } );
	//subtracting means for numerator
	batch->each_row() -= feature_means;
	//normalized values
	batch->each_row() %= (1 / (feature_variances));
}

/*files assigned to thread are parsed and used to create matrix rows that are inserted into
matrices of row size INPUT_BATCH_SIZE. These matrices are then enqueued and consumed 
by the training thread.
*/
void convert_data(std::vector<std::string> files)
{
	//JSON parser
	rapidjson::Document document;
	int row_counter = 0;

	std::vector<arma::mat> song_buffer;
	std::vector<arma::rowvec> genre_buffer; 

	//closure for creating batch of song data and enqueueing 
	auto build_batch = [&]() {
			row_counter = 0;

			//song batch data
			arma::cube * input_batch = new arma::cube(DATA_ROWS, DATA_ROW_LENGTH, INPUT_BATCH_SIZE);
			arma::mat * correct_output = new arma::mat(0, OUTPUT_COUNT);
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
		if(row_counter < INPUT_BATCH_SIZE)
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
	
	finish_mutex.lock();
	unfinished_threads--;
	finish_mutex.unlock();
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
