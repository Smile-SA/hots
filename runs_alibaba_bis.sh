#!/bin/bash
path="/home/eleclercq/Documents/CIFRE/data/alibaba/alibaba18/18n/"
ks=( 2 3 4 5 6 7 8 9 10)
taus=( 25 50 70 100 150 200 300)

for k in "${ks[@]}"
do
    for tau in "${taus[@]}"
    do
        cots --path $path --k $k --tau $tau
    done
done
