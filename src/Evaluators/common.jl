# Common interface for AbstractNLPEvaluator
#
function Base.show(io::IO, nlp::AbstractNLPEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    println(io, "A Evaluator object")
    println(io, "    * #vars: ", n)
    println(io, "    * #cons: ", m)
end

## Generic callbacks
function constraint(nlp::AbstractNLPEvaluator, x)
    cons = similar(x, n_constraints(nlp)) ; fill!(cons, 0)
    constraint!(nlp, cons, x)
    return cons
end

function gradient(nlp::AbstractNLPEvaluator, x)
    ∇f = similar(x) ; fill!(∇f, 0)
    gradient!(nlp, ∇f, x)
    return ∇f
end

function jacobian(nlp::AbstractNLPEvaluator, x)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    J = similar(x, m, n) ; fill!(J, 0)
    jacobian!(nlp, J, x)
    return J
end

# Default implementation of jprod!, using full Jacobian matrix
function jprod!(nlp::AbstractNLPEvaluator, jv, u, v)
    nᵤ = length(u)
    m  = n_constraints(nlp)
    @assert nᵤ == length(v)
    jac = jacobian(nlp, u)
    mul!(jv, jac, v)
    return
end

# Joint Objective Jacobian transpose vector product (default implementation)
function ojtprod!(nlp::AbstractNLPEvaluator, jv, u, σ, v)
    gradient!(nlp, jv, u)
    jv .*= σ  # scale gradient
    jtprod!(nlp, jv, u, v)
    return
end

# Common interface for Hessian
function hessian!(nlp::AbstractNLPEvaluator, hess, x)
    @assert has_hessian(nlp)
    n = n_variables(nlp)
    v = similar(x)
    @inbounds for i in 1:n
        hv = @view hess[:, i]
        fill!(v, 0)
        v[i] = 1.0
        hessprod!(nlp, hv, x, v)
    end
end

function hessian_lagrangian_penalty!(nlp::AbstractNLPEvaluator, hess, x, y, σ, D,)
    n = n_variables(nlp)
    v = similar(x)
    @inbounds for i in 1:n
        hv = @view hess[:, i]
        fill!(v, 0)
        v[i] = 1.0
        hessian_lagrangian_penalty_prod!(nlp, hv, x, y, σ, v, D)
    end
end

function batch_hessian!(nlp::AbstractNLPEvaluator, hess, x)
    @assert has_hessian(nlp)
    n = ExaPF.n_variables(nlp)
    ∇²f = nlp.hesslag.hess
    nbatch = size(nlp.hesslag.tmp_hv, 2)

    # Allocate memory
    v_cpu = zeros(n, nbatch)
    v = similar(x, n, nbatch)

    update_hessian!(nlp.model, ∇²f, nlp.buffer)

    N = div(n, nbatch, RoundDown)
    for i in 1:N
        # Init tangents on CPU
        fill!(v_cpu, 0.0)
        @inbounds for j in 1:nbatch
            v_cpu[j+(i-1)*nbatch, j] = 1.0
        end
        # Pass tangents to the device
        copyto!(v, v_cpu)

        hm = @view hess[:, nbatch * (i-1) + 1: nbatch * i]
        hessprod_!(nlp, hm, x, v)
    end

    # Last slice
    last_batch = n - N*nbatch
    if last_batch > 0
        fill!(v_cpu, 0.0)
        @inbounds for j in 1:nbatch
            v_cpu[n-nbatch+j, j] = 1.0
        end
        copyto!(v, v_cpu)

        hm = @view hess[:, (n - nbatch + 1) : n]
        hessprod_!(nlp, hm, x, v)
    end
end

function batch_hessian_lagrangian_penalty!(nlp::AbstractNLPEvaluator, hess, x, y, σ, w)
    @assert has_hessian(nlp)
    n = ExaPF.n_variables(nlp)
    ∇²f = nlp.hesslag.hess
    nbatch = size(nlp.hesslag.tmp_hv, 2)

    # Allocate memory
    v_cpu = zeros(n, nbatch)
    v = similar(x, n, nbatch)

    update_hessian!(nlp.model, ∇²f, nlp.buffer)

    N = div(n, nbatch, RoundDown)
    for i in 1:N
        # Init tangents on CPU
        fill!(v_cpu, 0.0)
        @inbounds for j in 1:nbatch
            v_cpu[j+(i-1)*nbatch, j] = 1.0
        end
        # Pass tangents to the device
        copyto!(v, v_cpu)

        hm = @view hess[:, nbatch * (i-1) + 1: nbatch * i]
        hessian_lagrangian_penalty_prod_!(nlp, hm, x, y, σ, v, w)
    end

    # Last slice
    last_batch = n - N*nbatch
    if last_batch > 0
        fill!(v_cpu, 0.0)
        @inbounds for j in 1:nbatch
            v_cpu[n-nbatch+j, j] = 1.0
        end
        copyto!(v, v_cpu)

        hm = @view hess[:, (n - nbatch + 1) : n]
        hessian_lagrangian_penalty_prod_!(nlp, hm, x, y, σ, v, w)
    end
end

function batch_jacobian!(nlp::AbstractNLPEvaluator, jac, x)
    @assert has_hessian(nlp)
    n = ExaPF.n_variables(nlp)
    ∇²f = nlp.hesslag.hess
    nbatch = n_batches(nlp.hesslag)

    # Allocate memory
    v_cpu = zeros(n, nbatch)
    v = similar(x, n, nbatch)

    N = div(n, nbatch, RoundDown)
    for i in 1:N
        # Init tangents on CPU
        fill!(v_cpu, 0.0)
        @inbounds for j in 1:nbatch
            v_cpu[j+(i-1)*nbatch, j] = 1.0
        end
        # Pass tangents to the device
        copyto!(v, v_cpu)

        jm = @view jac[:, nbatch * (i-1) + 1: nbatch * i]
        jprod_!(nlp, jm, x, v)
    end

    # Last slice
    last_batch = n - N*nbatch
    if last_batch > 0
        fill!(v_cpu, 0.0)
        @inbounds for j in 1:nbatch
            v_cpu[n-nbatch+j, j] = 1.0
        end
        copyto!(v, v_cpu)

        jm = @view jac[:, (n - nbatch + 1) : n]
        jprod_!(nlp, jm, x, v)
    end
end

function hessian(nlp::AbstractNLPEvaluator, x)
    n = n_variables(nlp)
    H = similar(x, n, n) ; fill!(H, 0)
    hessian!(nlp, H, x)
    return H
end

# Counters
abstract type AbstractCounter end

mutable struct NLPCounter <: AbstractCounter
    objective::Int
    gradient::Int
    hessian::Int
    jacobian::Int
    jtprod::Int
    hprod::Int
end
NLPCounter() = NLPCounter(0, 0, 0, 0, 0, 0)

function Base.empty!(c::NLPCounter)
    for attr in fieldnames(NLPCounter)
        setfield!(c, attr, 0)
    end
end

# Active set utils
function _check(val, val_min, val_max)
    violated_inf = findall(val .< val_min)
    violated_sup = findall(val .> val_max)
    n_inf = length(violated_inf)
    n_sup = length(violated_sup)
    err_inf = norm(val_min[violated_inf] .- val[violated_inf], Inf)
    err_sup = norm(val[violated_sup] .- val_max[violated_sup] , Inf)
    return (n_inf, err_inf, n_sup, err_sup)
end

function _inf_pr(nlp::AbstractNLPEvaluator, cons)
    (n_inf, err_inf, n_sup, err_sup) = _check(cons, nlp.g_min, nlp.g_max)
    return max(err_inf, err_sup)
end

# Scaler utils
abstract type AbstractScaler end

scale_factor(h, tol, η) = max(tol, η / max(1.0, h))

struct MaxScaler{T, VT} <: AbstractScaler
    scale_obj::T
    scale_cons::VT
    g_min::VT
    g_max::VT
end
function MaxScaler(g_min, g_max)
    @assert length(g_min) == length(g_max)
    m = length(g_min)
    sc = similar(g_min) ; fill!(sc, 1.0)
    return MaxScaler{eltype(g_min), typeof(g_min)}(1.0, sc, g_min, g_max)
end
function MaxScaler(nlp::AbstractNLPEvaluator, u0::AbstractVector;
                   η=100.0, tol=1e-8)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    conv = update!(nlp, u0)
    ∇g = similar(u0) ; fill!(∇g, 0)
    gradient!(nlp, ∇g, u0)
    s_obj = scale_factor(norm(∇g, Inf), tol, η)

    VT = typeof(u0)
    ∇c = xzeros(VT, n)
    s_cons = xzeros(VT, m)
    v = xzeros(VT, m)
    for i in eachindex(s_cons)
        fill!(v, 0.0)
        v[i] = 1.0
        jtprod!(nlp, ∇c, u0, v)
        s_cons[i] = scale_factor(norm(∇c, Inf), tol, η)
    end

    g♭, g♯ = bounds(nlp, Constraints())

    return MaxScaler{typeof(s_obj), typeof(s_cons)}(s_obj, s_cons, s_cons .* g♭, s_cons .* g♯)
end

struct BridgeDevice{VT, MT}
    u::VT
    g::VT
    cons::VT
    v::VT
    y::VT
    w::VT
    jv::VT
    J::MT
    H::MT
end

function BridgeDevice(n::Int, m::Int, VT, MT)
    BridgeDevice{VT, MT}(
        VT(undef, n),
        VT(undef, n),
        VT(undef, m),
        VT(undef, m),
        VT(undef, m),
        VT(undef, m),
        VT(undef, n),
        MT(undef, m, n),
        MT(undef, n, n),
    )
end

