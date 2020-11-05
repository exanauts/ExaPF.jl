#!/bin/bash

logfolder=logs
for i in `ls ./test/data/paper` 
do
  for j in 1 2 3
  do
    total=`grep Newton $logfolder/${i}_${j}.log | tail -1 | awk '{ print $3 }' | sed 's/ms//'`
    autodiff=`grep Jacobian $logfolder/${i}_${j}.log | tail -1 | awk '{ print $3 }' | sed 's/ms//'`
    # var3=`bc <<< "scale=2; $total"`
    bicgstab=""
    preconditioner=""
    if [ $j -eq 3 ]
    then
        bicgstab=`grep BICGSTAB $logfolder/${i}_${j}.log | tail -1 | awk '{
        print $3 }' | sed 's/ms//'`
        preconditioner=`grep Preconditioner $logfolder/${i}_${j}.log | tail -1 |
          awk '{ print $3 }' | sed 's/ms//'`
        bicgstab=`bc -l <<< "$bicgstab/$total"`
        preconditioner=`bc -l <<< "$preconditioner/$total"`
    fi
    autodiff=`bc -l <<< "$autodiff/$total"`
    echo $total $autodiff $bicgstab $preconditioner

  done
done
