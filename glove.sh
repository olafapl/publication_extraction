#! /usr/bin/env sh

curr_dir=$(dirname $(readlink -f $0))
data_dir=$curr_dir/data/

if [ ! -d $data_dir ]; then
  mkdir $data_dir
fi

cd $data_dir
wget http://nlp.stanford.edu/data/glove.6B.zip \
&& unzip glove.6B.zip -d glove \
&& rm glove.6B.zip
