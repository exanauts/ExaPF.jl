module ExaPF

# Standard library
using Printf
using LinearAlgebra
using SparseArrays

import CUDA
import CUDA.CUBLAS
import CUDA.CUSPARSE
import CUDA.CUSOLVER

import ForwardDiff
using KernelAbstractions
const KA = KernelAbstractions

import Base: show, get

include("architectures.jl")

# Templates
include("models.jl")

# Import submodules
include("autodiff.jl")
using .AutoDiff
include("LinearSolvers/LinearSolvers.jl")
using .LinearSolvers
include("PowerSystem/PowerSystem.jl")
using .PowerSystem

const PS = PowerSystem
const LS = LinearSolvers

# Polar formulation
include("Polar/polar.jl")

end
