#!/bin/bash

logfolder=logs
for i in `ls case*` 
do
  for j in 1 2 3
  do
    julia --project=.. benchmarks.jl $i $j > $logfolder/${i}_${j}.log
  done
done
