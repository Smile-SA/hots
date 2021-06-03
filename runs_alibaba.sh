#!/bin/sh
for f in /home/eleclercq/Documents/CIFRE/data/alibaba/alibaba18/18n/bench/*;
    do
        [ -d $f ] && cd "$f" && cots --path .
    done;
