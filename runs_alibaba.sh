#!/bin/sh
for f in /home/eleclercq/Documents/CIFRE/code_cifre/data_alibaba/20n_batch_bench_bis/*;
    do
        [ -d $f ] && cd "$f" && cots --path .
    done;
