#! /usr/bin/env sh

curr_dir=$(dirname $(readlink -f $0))
data_dir=$curr_dir/data/

if [ ! -d $data_dir ]; then
  mkdir $data_dir
fi

cd $data_dir
wget http://www.iesl.cs.umass.edu/data/umass-citation.tar.gz \
&& mkdir umass -p && tar -xf umass-citation.tar.gz -C umass --strip-components 1 \
&& rm umass-citation.tar.gz
