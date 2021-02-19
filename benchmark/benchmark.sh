#!/bin/bash

REV=`git rev-parse HEAD`
JULIA_BIN=julia
# CPU Solvers
for i in `ls case*` 
do
  for j in DirectSolver
  do
    $JULIA_BIN --project=.. benchmarks.jl $j CPU $i | tail -1 >> cpu_$REV.log
  done
done

# GPU Solver
for i in `ls case*` 
do
  for j in KrylovBICGSTAB DQGMRES BICGSTAB EigenBICGSTAB DirectSolver
  do
    $JULIA_BIN --project=.. benchmarks.jl $j CUDADevice $i | tail -1 >> gpu_$REV.log
  done
done
