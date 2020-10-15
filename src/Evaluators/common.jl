
abstract type AbstractADFactory end

struct ADFactory <: AbstractADFactory
    Jgₓ::AD.StateJacobianAD
    Jgᵤ::AD.DesignJacobianAD
    ∇f::AD.ObjectiveAD
    # Workaround before CUDA.jl 1.4: keep a cache for the transpose
    # matrix of the State Jacobian matrix, and update it inplace
    # when we need to solve the system Jₓ' \ y
    Jᵗ::Union{Nothing, AbstractMatrix}
end

abstract type AbstractPenalty end

Base.size(penal::AbstractPenalty) = penal.n

mutable struct QuadraticPenalty{T} <: AbstractPenalty
    n::Int
    coefs::T
    c♯::T
end
function QuadraticPenalty(n::Int; c₀=0.1)
    # Default coefficients
    coefs = c₀
    c♯ = 1e12
    return QuadraticPenalty(n, coefs, c♯)
end
function (penalty::QuadraticPenalty)(cx::AbstractVector)
    return 0.5 * penalty.coefs * dot(cx, cx)
end
function update!(penal::QuadraticPenalty, cx::AbstractVector, η)
    penal.coefs = min(η * penal.coefs, penal.c♯)
end
function gradient!(
    nlp::AbstractNLPEvaluator,
    grad::AbstractVector,
    penal::QuadraticPenalty,
    cons::Function,
    u::AbstractVector,
    cx::AbstractVector;
    start=1
)
    jtprod!(nlp, cons, grad, u, cx .* penal.coefs; start=start)
end

struct AugLagPenalty{T} <: AbstractPenalty
    coefs::AbstractVector{T}
    μ::AbstractVector{T}
end

