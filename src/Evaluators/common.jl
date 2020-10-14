
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

Base.size(penal::AbstractPenalty) = length(penal.coefs)

struct QuadraticPenalty{T} <: AbstractPenalty
    coefs::AbstractVector{T}
    η::T
    c♯::T
end
function QuadraticPenalty(nlp::AbstractNLPEvaluator, cons::Function; c₀=0.1)
    if !is_constraint(cons)
        error("Function $cons is not a valid constraint function")
    end
    n = size_constraint(nlp.model, cons)
    # Default coefficients
    coefs = similar(nlp.x, n)
    fill!(coefs, c₀)
    η = 10.0
    c♯ = 1e12
    return QuadraticPenalty(coefs, η, c♯)
end
function (penalty::QuadraticPenalty)(cx::AbstractVector)
    return 0.5 * dot(penalty.coefs, cx .* cx)
end
function update!(penal::QuadraticPenalty, cx::AbstractVector, η)
    penal.coefs .= min.(η .* penal.coefs, penal.c♯)
end
function gradient!(
    nlp::AbstractNLPEvaluator,
    grad::AbstractVector,
    penal::QuadraticPenalty,
    cons::Function,
    u::AbstractVector,
    cx::AbstractVector
)
    jtprod!(nlp, cons, grad, u, cx .* penal.coefs)
end

struct AugLagPenalty{T} <: AbstractPenalty
    coefs::AbstractVector{T}
    μ::AbstractVector{T}
end

