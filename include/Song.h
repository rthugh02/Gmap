#ifndef SONG_H
#define SONG_H

#include <armadillo>
class Song
{
private:
    arma::mat song_data;
    arma::rowvec genre;
public:
    Song(arma::mat song_data, arma::rowvec genre);
    ~Song();
};

#endif