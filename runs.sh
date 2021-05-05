#!/bin/sh
for f in /home/eleclercq/Documents/CIFRE/code_cifre/data_aw/15_22-03_bench/*;
    do
        [ -d $f ] && cd "$f" && cots --path .
    done;