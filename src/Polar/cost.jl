# Adjoints needed in polar formulation
#

"""
    AdjointStackObjective{VT}

An object for storing the adjoint stack in the adjoint objective computation.

"""
struct AdjointStackObjective{VT}
    ∇fₓ::VT
    ∇fᵤ::VT
    ∂pg::VT
    ∂vm::VT
    ∂va::VT
    jvₓ::VT
    jvᵤ::VT
    ∂flow::VT
end

function cost_production(polar::PolarForm, pg)
    ngen = PS.get(polar, PS.NumberOfGenerators())
    coefs = polar.costs_coefficients
    c2 = @view coefs[:, 2]
    c3 = @view coefs[:, 3]
    c4 = @view coefs[:, 4]
    # Return quadratic cost
    # NB: this operation induces three allocations on the GPU,
    #     but is faster than writing the sum manually
    cost = sum(c2 .+ c3 .* pg .+ c4 .* pg.^2)
    return cost
end

function put(
    polar::PolarForm{T, IT, VT, MT},
    ::PS.Generators,
    ::PS.ActivePower,
    obj_autodiff::AdjointStackObjective,
    buffer::PolarNetworkState
) where {T, IT, VT, MT}

    index_pv = polar.indexing.index_pv
    pv_to_gen = polar.indexing.index_pv_to_gen

    adj_pg = obj_autodiff.∂pg
    adj_x = obj_autodiff.∇fₓ
    adj_u = obj_autodiff.∇fᵤ
    adj_vmag = obj_autodiff.∂vm
    adj_vang = obj_autodiff.∂va
    # TODO
    adj_pinj = similar(adj_vmag)  ; fill!(adj_pinj, 0.0)
    fill!(adj_vmag, 0.0)
    fill!(adj_vang, 0.0)

    # Adjoint w.r.t Slack nodes
    adjoint!(polar, active_power_constraints, adj_pg, buffer.pg, obj_autodiff, buffer)
    # Adjoint w.t.t. PV nodes
    adj_pinj[index_pv] .= adj_pg[pv_to_gen]

    # Adjoint w.r.t. x and u
    fill!(adj_x, 0.0)
    fill!(adj_u, 0.0)
    adjoint_transfer!(polar, adj_u, adj_x, adj_vmag, adj_vang, adj_pinj, nothing)

    return
end

function ∂cost(polar::PolarForm, ∂obj::AdjointStackObjective, buffer::PolarNetworkState)
    pg = buffer.pg
    coefs = polar.costs_coefficients
    c3 = @view coefs[:, 3]
    c4 = @view coefs[:, 4]
    # Return adjoint of quadratic cost
    ∂obj.∂pg .= c3 .+ 2.0 .* c4 .* pg
    put(polar, PS.Generators(), PS.ActivePower(), ∂obj, buffer)
    return
end

function hessian_cost(polar::PolarForm, buffer::PolarNetworkState)
    coefs = polar.costs_coefficients
    c3 = @view coefs[:, 3]
    c4 = @view coefs[:, 4]
    # Change ordering from pg to [ref; pv]
    pv2gen = polar.indexing.index_pv_to_gen
    ref2gen = polar.indexing.index_ref_to_gen
    ∂pg = (c3 .+ 2.0 .* c4 .* buffer.pg)[ref2gen]
    ∂²pg = 2.0 .* c4[[ref2gen; pv2gen]]

    # Evaluate Hess-vec of objective function f(x, u)
    ∂₂P = active_power_hessian(polar, buffer)::FullSpaceHessian{SparseMatrixCSC{Float64, Int}}

    ∂Pₓ = matpower_jacobian(polar, active_power_constraints, State(), buffer)::SparseMatrixCSC{Float64, Int}
    ∂Pᵤ = matpower_jacobian(polar, active_power_constraints, Control(), buffer)::SparseMatrixCSC{Float64, Int}

    D = Diagonal(∂²pg)
    ∇²fₓₓ = ∂pg .* ∂₂P.xx + ∂Pₓ' * D * ∂Pₓ
    ∇²fᵤᵤ = ∂pg .* ∂₂P.uu + ∂Pᵤ' * D * ∂Pᵤ
    ∇²fₓᵤ = ∂pg .* ∂₂P.xu + ∂Pᵤ' * D * ∂Pₓ

    return FullSpaceHessian(∇²fₓₓ, ∇²fₓᵤ, ∇²fᵤᵤ)
end

