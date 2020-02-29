#include "Song.h"

Song::Song(arma::mat song_data, arma::rowvec genre)
{
    this->song_data = song_data;
    this->genre = genre;
}

Song::~Song()
{
}
