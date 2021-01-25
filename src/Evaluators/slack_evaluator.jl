
"""
    SlackEvaluator{T} <: AbstractNLPEvaluator

Reformulate a problem with inequality constraints as an
equality constrained problem, by introducing a set of slack
variables.

"""
mutable struct SlackEvaluator{T} <: AbstractNLPEvaluator
    inner::AbstractNLPEvaluator
    s_min::AbstractVector{T}
    s_max::AbstractVector{T}
    nv::Int
    ns::Int
end
function SlackEvaluator(nlp::AbstractNLPEvaluator)
    if !is_constrained(nlp)
        error("Input problem must have inequality constraints")
    end
    nv, ns = n_variables(nlp), n_constraints(nlp)
    s_min, s_max = bounds(nlp, Constraints())
    return SlackEvaluator(nlp, s_min, s_max, nv, ns)
end
function SlackEvaluator(
    datafile::String;
    device=CPU(), options...
)
    nlp = ReducedSpaceEvaluator(datafile; device=device)
    return SlackEvaluator(nlp)
end

n_variables(nlp::SlackEvaluator) = nlp.nv + nlp.ns
n_constraints(nlp::SlackEvaluator) = n_constraints(nlp.inner)

# Getters
get(nlp::SlackEvaluator, attr::AbstractNLPAttribute) = get(nlp.inner, attr)
get(nlp::SlackEvaluator, attr::AbstractVariable) = get(nlp.inner, attr)
get(nlp::SlackEvaluator, attr::PS.AbstractNetworkAttribute) = get(nlp.inner, attr)

# Setters
function setvalues!(nlp::SlackEvaluator, attr::PS.AbstractNetworkValues, values)
    setvalues!(nlp.inner, attr, values)
end

# Bounds
function bounds(nlp::SlackEvaluator, ::Variables)
    u♭, u♯ = bounds(nlp.inner, Variables())
    return [u♭; nlp.s_min], [u♯; nlp.s_max]
end
bounds(nlp::SlackEvaluator, ::Constraints) = (zeros(nlp.ns), zeros(nlp.ns))

function initial(nlp::SlackEvaluator)
    u0 = initial(nlp.inner)
    s0 = copy(nlp.s_min)
    return [u0; s0]
end

function update!(nlp::SlackEvaluator, w)
    u = @view w[1:nlp.nv]
    s = @view w[nlp.nv+1:end]
    return update!(nlp.inner, u)
end

function objective(nlp::SlackEvaluator, w)
    u = @view w[1:nlp.nv]
    s = @view w[nlp.nv+1:end]
    return objective(nlp.inner, u)
end

function constraint!(nlp::SlackEvaluator, cons, w)
    u = @view w[1:nlp.nv]
    s = @view w[nlp.nv+1:end]
    constraint!(nlp.inner, cons, u)
    cons .-= s
    return
end

# Gradient
function reduced_gradient!(nlp::SlackEvaluator, grad, jvx, jvu, w)
    # w.r.t. u
    u = @view w[1:nlp.nv]
    gu = @view grad[1:nlp.nv]
    reduced_gradient!(nlp.inner, gv, jvx, jvu, u)
    # w.r.t. s
    gs = @view grad[nlp.nv+1:end]
    fill!(gs, 0.0)
end

function gradient_full!(nlp::SlackEvaluator, jvx, jvu, w)
    autodiff = get(nlp.inner, AutoDiffBackend())
    u = @view w[1:nlp.nv]
    gradient_full!(nlp.inner, jvx, jvu, u)
end

function gradient!(nlp::SlackEvaluator, grad, w)
    # w.r.t. u
    u = @view w[1:nlp.nv]
    gu = @view grad[1:nlp.nv]
    gradient!(nlp.inner, gu, u)
    # w.r.t. s
    gs = @view grad[nlp.nv+1:end]
    fill!(gs, 0.0)
    return nothing
end

## Transpose Jacobian-vector product
# N.B.: constraints are specified as h(u) - s = 0
function jtprod!(nlp::SlackEvaluator, jv, w, v)
    # w.r.t. u
    u = @view w[1:nlp.nv]
    jvu = @view jv[1:nlp.nv]
    jtprod!(nlp.inner, jvu, u, v)
    # w.r.t. s
    s = @view w[nlp.nv+1:end]
    jvs = @view jv[nlp.nv+1:end]
    jvs .-= v
end

function jtprod_full!(nlp::SlackEvaluator, jvx, jvu, w, v)
    u = @view w[1:nlp.nu]
    jtprod_full!(nlp.inner, jvx, jvu, u, v)
    # TODO
end

function reset!(nlp::SlackEvaluator)
    reset!(nlp.inner)
end

