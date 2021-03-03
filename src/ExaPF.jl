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
import MathOptInterface
const MOI = MathOptInterface
using TimerOutputs: @timeit, TimerOutput

import Base: show, get

const VERBOSE_LEVEL_HIGH = 3
const VERBOSE_LEVEL_MEDIUM = 2
const VERBOSE_LEVEL_LOW = 1
const VERBOSE_LEVEL_NONE = 0
const TIMER = TimerOutput()

include("utils.jl")
include("architectures.jl")
# Templates
include("models.jl")

# Import submodules
include("autodiff.jl")
using .AutoDiff
include("indexes.jl")
using .IndexSet
include("LinearSolvers/LinearSolvers.jl")
using .LinearSolvers
include("parsers/parse_mat.jl")
using .ParseMAT
include("parsers/parse_psse.jl")
using .ParsePSSE
include("PowerSystem/PowerSystem.jl")
using .PowerSystem

const PS = PowerSystem

# Polar formulation
include("Polar/polar.jl")

# Evaluators
include("Evaluators/Evaluators.jl")

end
