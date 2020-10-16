#! /usr/bin/env sh

curr_dir=$(dirname $(readlink -f $0))
data_dir=$curr_dir/data/

if [ ! -d $data_dir ]; then
  mkdir $data_dir
fi

cd $data_dir
wget -O homepub-2500.zip https://www.dropbox.com/sh/u8e7eia0191n70t/AAD0gjZUW9e4ixCizdN4MqdMa/homepub-2500.zip?dl=0 \
&& unzip homepub-2500.zip
&& rm homepub-2500.zip
