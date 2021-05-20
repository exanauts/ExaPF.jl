
@doc raw"""
    SlackEvaluator{Evaluator<:AbstractNLPEvaluator, T, VT} <: AbstractNLPEvaluator

Reformulate a problem with inequality constraints as an
equality constrained problem, by introducing a set of slack
variables.

### Description
A `SlackEvaluator` takes as input an original [`AbstractNLPEvaluator`](@ref),
subject to inequality constraints
```math
\begin{aligned}
       \min_{u \in \mathbb{R}^n} \quad & f(u)\\
\mathrm{s.t.} \quad & h^♭ ≤ h(u) ≤ h^♯,\\
                    & u^♭ ≤  u   ≤ u^♯.
\end{aligned}
```
The `SlackEvaluator` instance rewrites this problem with inequalities
as a new problem comprising only *equality constraints*, by introducing
$m$ slack variables $s_1, ⋯, s_m$. The new problem writes out
```math
\begin{aligned}
       \min_{u \in \mathbb{R}^n, s \in \mathbb{R}^m} \quad & f(u)\\
    \mathrm{s.t.} \quad & h(u) - s = 0 \\
                    & u^♭ ≤  u   ≤ u^♯, \\
                    & h^♭ ≤  s   ≤ h^♯.
\end{aligned}
```

### Attributes

- `inner::Evaluator`: original evaluator
- `s_min::VT`: stores lower bounds for slack variables
- `s_max::VT`: stores upper bounds for slack variables
- `nv::Int`: number of original variables
- `ns::Int`: number of slack variables

"""
mutable struct SlackEvaluator{Evaluator<:AbstractNLPEvaluator, T, VT, Bridge} <: AbstractNLPEvaluator
    inner::Evaluator
    s_min::VT
    s_max::VT
    nv::Int
    ns::Int
    bridge::Bridge
end
function SlackEvaluator(nlp::AbstractNLPEvaluator, device)
    if !is_constrained(nlp)
        error("Input problem must have inequality constraints")
    end
    nv, ns = n_variables(nlp), n_constraints(nlp)
    # Target device
    VT = isa(device, CPU) ? Vector{Float64} : CuVector{Float64}
    s_min, s_max = bounds(nlp, Constraints())
    s_min = s_min |> VT
    s_max = s_max |> VT

    # Deporting device
    if isa(nlp.model.device, CPU)
        VTD = Array{Float64, 1}
        MTD = Array{Float64, 2}
    else
        VTD = CUDA.CuArray{Float64, 1}
        MTD = CUDA.CuArray{Float64, 2}
    end
    bridge = BridgeDevice(nv, ns, VTD, MTD)
    return SlackEvaluator{typeof(nlp), eltype(s_min), VT, typeof(bridge)}(nlp, s_min, s_max, nv, ns, bridge)
end
function SlackEvaluator(
    datafile::String;
    device=KA.CPU(), options...
)
    nlp = ReducedSpaceEvaluator(datafile; device=device)
    return SlackEvaluator(nlp, device)
end

n_variables(nlp::SlackEvaluator) = nlp.nv + nlp.ns
n_constraints(nlp::SlackEvaluator) = n_constraints(nlp.inner)

constraints_type(::SlackEvaluator) = :equality
has_hessian(nlp::SlackEvaluator) = has_hessian(nlp.inner)

# Getters
get(nlp::SlackEvaluator, attr::AbstractNLPAttribute) = get(nlp.inner, attr)
get(nlp::SlackEvaluator, attr::AbstractVariable) = get(nlp.inner, attr)
get(nlp::SlackEvaluator, attr::PS.AbstractNetworkAttribute) = get(nlp.inner, attr)

# Setters
function setvalues!(nlp::SlackEvaluator, attr::PS.AbstractNetworkValues, values)
    setvalues!(nlp.inner, attr, values)
end

# Bounds
function bounds(nlp::SlackEvaluator{Ev, T, VT, B}, ::Variables) where {Ev, T, VT, B}
    u♭, u♯ = bounds(nlp.inner, Variables())
    return [u♭; nlp.s_min] |> VT , [u♯; nlp.s_max] |> VT
end
function bounds(nlp::SlackEvaluator{Ev, T, VT, B}, ::Constraints) where {Ev, T, VT, B}
    return (VT(zeros(nlp.ns)), VT(zeros(nlp.ns)))
end

function initial(nlp::SlackEvaluator{Ev, T, VT, B}) where {Ev, T, VT, B}
    u0 = initial(nlp.inner)
    update!(nlp.inner, u0)
    cons = constraint(nlp.inner, u0)
    return [u0; -cons] |> VT
end

function update!(nlp::SlackEvaluator, w)
    u = @view w[1:nlp.nv]
    copyto!(nlp.bridge.u, u)
    return update!(nlp.inner, nlp.bridge.u)
end

# f(x) = f₀(u)   , with x = (u, s)
function objective(nlp::SlackEvaluator, w)
    u = @view w[1:nlp.nv]
    return objective(nlp.inner, nlp.bridge.u)
end

# h(x) = h₀(u) - s
function constraint!(nlp::SlackEvaluator, cons, w)
    u = @view w[1:nlp.nv]
    s = @view w[nlp.nv+1:end]
    constraint!(nlp.inner, nlp.bridge.cons, nlp.bridge.u)
    copyto!(cons, nlp.bridge.cons)
    cons .-= s
    return
end

# Gradient
# ∇f = [ ∇f₀ ; 0 ]
function gradient!(nlp::SlackEvaluator, grad, w)
    # w.r.t. u
    u = @view w[1:nlp.nv]
    gu = @view grad[1:nlp.nv]
    gradient!(nlp.inner, nlp.bridge.g, nlp.bridge.u)
    copyto!(gu, nlp.bridge.g)
    # w.r.t. s
    gs = @view grad[nlp.nv+1:end]
    fill!(gs, 0.0)
    return
end

## Transpose Jacobian-vector product
# N.B.: constraints are specified as h(u) - s = 0
# J = [J₀  -I], so
# J' = [ J₀' ]
#      [ -I  ]
function jtprod!(nlp::SlackEvaluator, jv, w, v)
    # w.r.t. u
    u = @view w[1:nlp.nv]
    jvu = @view jv[1:nlp.nv]
    copyto!(nlp.bridge.v, v)
    jtprod!(nlp.inner, nlp.bridge.jv, nlp.bridge.u, nlp.bridge.v)
    copyto!(jvu, nlp.bridge.jv)
    # w.r.t. s
    jvs = @view jv[nlp.nv+1:end]
    jvs .= -v
    return
end

function jacobian!(nlp::SlackEvaluator, jac, w)
    fill!(jac, 0)
    u = @view w[1:nlp.nv]
    # w.r.t. u
    Jᵤ = @view jac[:, 1:nlp.nv]
    jacobian!(nlp.inner, Jᵤ, u)
    # w.r.t. s
    for i in 1:nlp.ns
        jac[i, i + nlp.nv] = -1.0
    end
end

function ojtprod!(nlp::SlackEvaluator, jv, w, σ, v)
    # w.r.t. u
    u = @view w[1:nlp.nv]
    jvu = @view jv[1:nlp.nv]
    copyto!(nlp.bridge.v, v)
    ojtprod!(nlp.inner, nlp.bridge.jv, nlp.bridge.u, σ, nlp.bridge.v)
    copyto!(jvu, nlp.bridge.jv)
    # w.r.t. s
    jvs = @view jv[nlp.nv+1:end]
    jvs .= -v
    return
end

# H = [ H₀   0 ]
#     [ 0    0 ]
function hessprod!(nlp::SlackEvaluator, hessvec, w, v)
    # w.r.t. u
    @views hessprod!(nlp.inner, hessvec[1:nlp.nv], w[1:nlp.nv], v[1:nlp.nv])
    # w.r.t. s
    hus = @view hessvec[nlp.nv+1:end]
    hus .= 0.0
    return
end

# J = [Jᵤ -I] , hence
# J' * J = [ Jᵤ' * Jᵤ    - Jᵤ']
#          [ - Jᵤ           I ]
function hessian_lagrangian_penalty_prod!(
    nlp::SlackEvaluator, hessvec, x, y, σ, v, ρ,
)
    @views begin
        u   = x[1:nlp.nv]
        vᵤ  = v[1:nlp.nv]
        vₛ  = v[nlp.nv+1:end]
        hvu = hessvec[1:nlp.nv]
        hvs = hessvec[nlp.nv+1:end]
    end
    fill!(hessvec, 0)
    u_buf = similar(u) ; fill!(u_buf, 0)
    y_buf = similar(y) ; fill!(y_buf, 0)
    # w.r.t. uu
    # ∇²L + ρ Jᵤ' * Jᵤ
    hessian_lagrangian_penalty_prod!(nlp.inner, hvu, u, y, σ, vᵤ, ρ)

    if !iszero(ρ)
        # w.r.t. us
        y_buf .= ρ .* vₛ
        # - Jᵤ' * vₛ
        jtprod!(nlp.inner, u_buf, u, y_buf)
        hvu .-=  u_buf
        # w.r.t. su
        jprod!(nlp.inner, y_buf, u, vᵤ)
        hvs .-= ρ .* y_buf
        # w.r.t. ss
        hvs .+= ρ .* vₛ
    end
    return
end

function hessian_lagrangian_penalty!(
    nlp::SlackEvaluator, H, x, y, σ, w,
)
    n = n_variables(nlp)
    @views begin
        u   = x[1:nlp.nv]
        Hᵤᵤ = H[1:nlp.nv, 1:nlp.nv]
        Hᵤᵥ = H[1:nlp.nv, 1+nlp.nv:n]
        Hᵥᵤ = H[1+nlp.nv:n, 1:nlp.nv]
        Hᵥᵥ = H[1+nlp.nv:n, 1+nlp.nv:n]
    end
    # w.r.t. uu
    # ∇²L + ρ Jᵤ' * Jᵤ
    hessian_lagrangian_penalty!(nlp.inner, Hᵤᵤ, u, y, σ, w)

    if !iszero(w)
        D = Diagonal(w)
        Jᵤ = similar(Hᵥᵤ) ; fill!(Jᵤ, 0.0)
        jacobian!(nlp.inner, Jᵤ, u)
        mul!(Hᵤᵥ, Jᵤ', -D)
        mul!(Hᵥᵤ, - D, Jᵤ)
        fill!(Hᵥᵥ, 0)
        @inbounds for i in 1:nlp.ns
            Hᵥᵥ[i, i] = w[i]
        end
    else
        fill!(Hᵤᵥ, 0)
        fill!(Hᵥᵤ, 0)
        fill!(Hᵥᵥ, 0)
    end
end

function batch_hessian_lagrangian_penalty!(
    nlp::SlackEvaluator, H, x, y, σ, w,
)
    n = n_variables(nlp)
    @views begin
        u   = x[1:nlp.nv]
        Hᵤᵤ = H[1:nlp.nv, 1:nlp.nv]
        Hᵤᵥ = H[1:nlp.nv, 1+nlp.nv:n]
        Hᵥᵤ = H[1+nlp.nv:n, 1:nlp.nv]
        Hᵥᵥ = H[1+nlp.nv:n, 1+nlp.nv:n]
    end
    # w.r.t. uu
    # ∇²L + ρ Jᵤ' * Jᵤ
    copyto!(nlp.bridge.w, w)
    copyto!(nlp.bridge.y, y)
    H = similar(Hᵤᵤ)
    batch_hessian_lagrangian_penalty!(nlp.inner, nlp.bridge.H, nlp.bridge.u, nlp.bridge.y, σ, nlp.bridge.w)
    copyto!(H, nlp.bridge.H)
    Hᵤᵤ .= H

    if !iszero(w)
        D = Diagonal(w)
        Jᵤ = similar(Hᵥᵤ) ; fill!(Jᵤ, 0.0)
        batch_jacobian!(nlp.inner, nlp.bridge.J, nlp.bridge.u)
        copyto!(Jᵤ, nlp.bridge.J)
        mul!(Hᵤᵥ, Jᵤ', -D)
        mul!(Hᵥᵤ, - D, Jᵤ)
        fill!(Hᵥᵥ, 0)
        @inbounds for i in 1:nlp.ns
            Hᵥᵥ[i, i] = w[i]
        end
    else
        fill!(Hᵤᵥ, 0)
        fill!(Hᵥᵤ, 0)
        fill!(Hᵥᵥ, 0)
    end
end

# TODO: return sparse sparsity pattern for bottom-left block
function hessian_structure(nlp::SlackEvaluator)
    n = n_variables(nlp)
    rows = Int[r for r in 1:n for c in 1:r]
    cols = Int[c for r in 1:n for c in 1:r]
    return rows, cols
end

function reset!(nlp::SlackEvaluator)
    reset!(nlp.inner)
end

# Utils function
function primal_infeasibility!(nlp::SlackEvaluator, cons, u)
    constraint!(nlp, cons, u) # Evaluate constraints
    return norm(cons, Inf)
end
function primal_infeasibility(nlp::SlackEvaluator, u)
    cons = similar(nlp.s_min) ; fill!(cons, 0)
    return primal_infeasibility!(nlp, cons, u)
end

