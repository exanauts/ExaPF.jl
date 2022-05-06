module ExaPF

# Standard library
using Printf
using LinearAlgebra
using SparseArrays

import CUDA
import CUDA.CUSPARSE

import ForwardDiff
using KernelAbstractions
const KA = KernelAbstractions

import Base: show, get

export run_pf
export State, Control, AllVariables, PolarForm

# Export KernelAbstractions devices
export CPU

include("templates.jl")
include("utils.jl")

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

# CUDA extension
if CUDA.has_cuda()
    include("cuda_wrapper.jl")
end

end
