# Licensing
# This file is a direct port of :
# https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/master/src/linesearch/armijo_wolfe.jl
# Copyright (c) 2015-2019: Dominique Orban and Abel Soares Siqueira
# SolverTools.jl is licensed under the MPL version 2.0.
#
# See detailed license here:
# https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/master/LICENSE.md

"""
    t, good_grad, ht, nbk, nbW = armijo_wolfe(h, h₀, slope, g)

Performs a line search from `x` along the direction `d` as defined by the
`LineModel` ``h(t) = f(x + t d)``, where
`h₀ = h(0) = f(x)`, `slope = h'(0) = ∇f(x)ᵀd` and `g` is a vector that will be
overwritten with the gradient at various points. On exit, if `good_grad=true`,
`g` contains the gradient at the final step length.
The steplength is chosen trying to satisfy the Armijo and Wolfe conditions. The Armijo condition is
```math
h(t) ≤ h₀ + τ₀ t h'(0)
```
and the Wolfe condition is
```math
h'(t) ≤ τ₁ h'(0).
```
Initially the step is increased trying to satisfy the Wolfe condition.
Afterwards, only backtracking is performed in order to try to satisfy the Armijo condition.
The final steplength may only satisfy Armijo's condition.

The output is the following:
- t: the step length;
- good_grad: whether `g` is the gradient at `x + t * d`;
- ht: the model value at `t`, i.e., `f(x + t * d)`;
- nbk: the number of times the steplength was decreased to satisfy the Armijo condition, i.e., number of backtracks;
- nbW: the number of times the steplength was increased to satisfy the Wolfe condition.

The following keyword arguments can be provided:
- `t`: starting steplength (default `1`);
- `τ₀`: slope factor in the Armijo condition (default `max(1e-4, √ϵₘ)`);
- `τ₁`: slope factor in the Wolfe condition. It should satisfy `τ₁ > τ₀` (default `0.9999`);
- `bk_max`: maximum number of backtracks (default `10`);
- `bW_max`: maximum number of increases (default `5`);
- `verbose`: whether to print information (default `false`).
"""
function armijo_wolfe(obj,
                      grad!,
                      h₀ :: T,
                      slope :: T,
                      g :: Array{T,1};
                      t :: T=one(T),
                      τ₀ :: T=max(T(1.0e-4), sqrt(eps(T))),
                      τ₁ :: T=T(0.9999),
                      bk_max :: Int=10,
                      bW_max :: Int=5,
                      verbose :: Bool=false) where T <: AbstractFloat

    # Perform improved Armijo linesearch.
    nbk = 0
    nbW = 0

    # First try to increase t to satisfy loose Wolfe condition
    ht = obj(t)
    slope_t = grad!(t, g)
    while (slope_t < τ₁*slope) && (ht <= h₀ + τ₀ * t * slope) && (nbW < bW_max)
        t *= 5
        ht = obj(t)
        slope_t = grad!(t, g)
        nbW += 1
    end

    hgoal = h₀ + slope * t * τ₀;
    fact = -T(0.8)
    ϵ = eps(T)^T(3/5)

    # Enrich Armijo's condition with Hager & Zhang numerical trick
    Armijo = (ht <= hgoal) || ((ht <= h₀ + ϵ * abs(h₀)) && (slope_t <= fact * slope))
    good_grad = true
    while !Armijo && (nbk < bk_max)
        t *= T(0.4)
        ht = obj(t)
        hgoal = h₀ + slope * t * τ₀;

        # avoids unused grad! calls
        Armijo = false
        good_grad = false
        if ht <= hgoal
            Armijo = true
        elseif ht <= h₀ + ϵ * abs(h₀)
            slope_t = grad!(t, g)
            good_grad = true
            if slope_t <= fact * slope
                Armijo = true
            end
        end

        nbk += 1
    end

    verbose && @printf("  %4d %4d\n", nbk, nbW);

    return (t, good_grad, ht, nbk, nbW)
end

function projected_direction!(pₓ, p, x, x♭, x♯)
    for i in eachindex(x)
        if (x[i] == x♭[i]) && (p[i] < 0.0)
            pₓ[i] = 0
        elseif (x[i] == x♯[i]) && (p[i] > 0.0)
            pₓ[i] = 0
        else
            pₓ[i] = p[i]
        end
    end
end

function armijo_ls(nlp, uk::Vector{Float64}, f0, grad::Vector{Float64}; t0=1e-4)
    nᵤ = length(grad)
    s = copy(-grad)
    pₓ = copy(-grad)
    function Lalpha(alpha)
        w_ = uk .+ alpha .* s
        u_ = similar(w_)
        ExaPF.project!(u_, w_, nlp.inner.u_min, nlp.inner.u_max)
        ExaPF.update!(nlp, u_)
        return ExaPF.objective(nlp, u_)
    end
    function grad_Lalpha(alpha, g_)
        w_ = uk .+ alpha .* s
        u_ = similar(w_)
        ExaPF.project!(u_, w_, nlp.inner.u_min, nlp.inner.u_max)
        ExaPF.gradient!(nlp, g_, u_)
        projected_direction!(pₓ, s, u_, nlp.inner.u_min, nlp.inner.u_max)
        return dot(g_, pₓ)
    end
    slope = dot(s, grad)
    alpha, obj = armijo_wolfe(Lalpha, grad_Lalpha, f0, slope, s; t=t0)
    return alpha
end

