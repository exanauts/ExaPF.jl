module ExaPF

# Standard library
using Printf
using LinearAlgebra
using SparseArrays

import CUDA
import CUDA.CUBLAS
import CUDA.CUSPARSE
import CUDA.CUSOLVER

import AMDGPU

import ForwardDiff
using KernelAbstractions
const KA = KernelAbstractions
using TimerOutputs: @timeit, TimerOutput

import Base: show, get

const VERBOSE_LEVEL_HIGH = 3
const VERBOSE_LEVEL_MEDIUM = 2
const VERBOSE_LEVEL_LOW = 1
const VERBOSE_LEVEL_NONE = 0
const TIMER = TimerOutput()

include("utils.jl")
include("backends.jl")

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
