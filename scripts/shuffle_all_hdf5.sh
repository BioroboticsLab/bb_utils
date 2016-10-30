#! /usr/bin/env bash

set -e

for h5file in `ls *.hdf5`; do
    without_ext="${h5file%%.*}"
    if [[ "$without_ext" != *shuffled ]]; then
        output=${without_ext}_shuffled.hdf5
        if [ ! -f $output ]; then
            shuffle_hdf5 --output ${without_ext}_shuffled.hdf5 $h5file
        fi
    fi
done