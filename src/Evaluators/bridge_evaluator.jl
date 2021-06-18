
#=
    BridgeEvaluator
=#
struct BridgeDeviceArrays{VT, MT}
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

function BridgeDeviceArrays(n::Int, m::Int, VT, MT)
    BridgeDeviceArrays{VT, MT}(
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

struct BridgeDeviceEvaluator{Evaluator, VT, MT, DVT, DMT} <: AbstractNLPEvaluator
    inner::Evaluator
    bridge::BridgeDeviceArrays{DVT, DMT}
end
function BridgeDeviceEvaluator(nlp::AbstractNLPEvaluator, device)
    n, m = n_variables(nlp), n_constraints(nlp)
    # Deporting device
    VT = Array{Float64, 1}
    MT = Array{Float64, 2}
    if isa(nlp.model.device, CPU)
        VTD = Array{Float64, 1}
        MTD = Array{Float64, 2}
    else
        VTD = CUDA.CuArray{Float64, 1}
        MTD = CUDA.CuArray{Float64, 2}
    end
    bridge = BridgeDeviceArrays(n, m, VTD, MTD)
    return BridgeDeviceEvaluator{typeof(nlp), VT, MT, VTD, MTD}(nlp, bridge)
end
function BridgeDeviceEvaluator(case::String; device=KA.CPU())
    nlp = ReducedSpaceEvaluator(case)
    return BridgeDeviceEvaluator(nlp, device)
end

n_variables(nlp::BridgeDeviceEvaluator) = n_variables(nlp.inner)
n_constraints(nlp::BridgeDeviceEvaluator) = n_constraints(nlp.inner)
constraints_type(nlp::BridgeDeviceEvaluator) = constraints_type(nlp.inner)
has_hessian(nlp::BridgeDeviceEvaluator) = has_hessian(nlp.inner)
reset!(nlp::BridgeDeviceEvaluator) = reset!(nlp.inner)

# Getters
get(nlp::BridgeDeviceEvaluator, attr::AbstractNLPAttribute) = get(nlp.inner, attr)
get(nlp::BridgeDeviceEvaluator, attr::AbstractVariable) = get(nlp.inner, attr)
get(nlp::BridgeDeviceEvaluator, attr::PS.AbstractNetworkAttribute) = get(nlp.inner, attr)

function setvalues!(nlp::BridgeDeviceEvaluator, attr::PS.AbstractNetworkValues, values)
    setvalues!(nlp.inner, attr, values)
end

function bounds(nlp::BridgeDeviceEvaluator{Ev, VT, MT, DVT, DMT}, attr::AbstractNLPAttribute) where {Ev, VT, MT, DVT, DMT}
    b♭, b♯ = bounds(nlp.inner, attr)
    return b♭ |> VT, b♯ |> VT
end

function initial(nlp::BridgeDeviceEvaluator{Ev, VT, MT, DVT, DMT}) where {Ev, VT, MT, DVT, DMT}
    return initial(nlp.inner) |> VT
end

function update!(nlp::BridgeDeviceEvaluator, u)
    copyto!(nlp.bridge.u, u)
    return update!(nlp.inner, nlp.bridge.u)
end

objective(nlp::BridgeDeviceEvaluator, u) = objective(nlp.inner, nlp.bridge.u)

function constraint!(nlp::BridgeDeviceEvaluator, cons, u)
    constraint!(nlp.inner, nlp.bridge.cons, nlp.bridge.u)
    copyto!(cons, nlp.bridge.cons)
    return
end

function gradient!(nlp::BridgeDeviceEvaluator, grad, u)
    gradient!(nlp.inner, nlp.bridge.g, nlp.bridge.u)
    copyto!(grad, nlp.bridge.g)
    return
end

function jtprod!(nlp::BridgeDeviceEvaluator, jv, u, v)
    copyto!(nlp.bridge.v, v)
    jtprod!(nlp.inner, nlp.bridge.jv, nlp.bridge.u, nlp.bridge.v)
    copyto!(jv, nlp.bridge.jv)
    return
end

function jacobian!(nlp::BridgeDeviceEvaluator, jac, w)
    jacobian!(nlp.inner, nlp.bridge.J, nlp.bridge.u)
    copyto!(jac, nlp.bridge.J)
    return
end

function ojtprod!(nlp::BridgeDeviceEvaluator, jv, u, σ, v)
    copyto!(nlp.bridge.v, v)
    ojtprod!(nlp.inner, nlp.bridge.jv, nlp.bridge.u, σ, nlp.bridge.v)
    copyto!(jv, nlp.bridge.jv)
    return
end

function hessprod!(nlp::BridgeDeviceEvaluator, hv, u, v)
    copyto!(nlp.bridge.v, v)
    hessprod!(nlp.inner, nlp.bridge.jv, nlp.bridge.u, nlp.bridge.v)
    copyto!(hv, nlp.bridge.jv)
    return
end

function hessian!(nlp::BridgeDeviceEvaluator, H, u)
    hessian!(nlp.inner, nlp.bridge.H, nlp.bridge.u)
    copyto!(H, nlp.bridge.H)
    return
end

function hessian_lagrangian_penalty!(nlp::BridgeDeviceEvaluator, H, u, y, σ, w)
    copyto!(nlp.bridge.w, w)
    copyto!(nlp.bridge.y, y)
    hessian_lagrangian_penalty!(nlp.inner, nlp.bridge.H, nlp.bridge.u, nlp.bridge.y, σ, nlp.bridge.w)
    copyto!(H, nlp.bridge.H)
    return
end

