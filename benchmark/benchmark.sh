#!/bin/bash

# CPU Solvers
for i in `ls case*` 
do
  for j in DirectSolver
  do
    julia --project=.. benchmarks.jl $j CPU $i | tail -1 >> cpu.log
  done
done

# GPU Solver
for i in `ls case*` 
do
  for j in KrylovBICGSTAB DQGMRES BICGSTAB EigenBICGSTAB DirectSolver
  do
    julia --project=.. benchmarks.jl $j CUDADevice $i | tail -1 >> gpu.log
  done
done
