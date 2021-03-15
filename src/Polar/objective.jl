# Adjoints needed in polar formulation
#

function cost_production(polar::PolarForm, pg)
    ngen = PS.get(polar, PS.NumberOfGenerators())
    coefs = polar.costs_coefficients
    c2 = @view coefs[:, 2]
    c3 = @view coefs[:, 3]
    c4 = @view coefs[:, 4]
    # Return quadratic cost
    # NB: this operation induces three allocations on the GPU,
    #     but is faster than writing the sum manually
    return sum(c2 .+ c3 .* pg .+ c4 .* pg.^2)
end

function objective(polar::PolarForm, buffer::PolarNetworkState)
    return cost_production(polar, buffer.pg)
end

function adjoint_cost!(polar::PolarForm, ∂f, pg)
    coefs = polar.costs_coefficients
    c3 = @view coefs[:, 3]
    c4 = @view coefs[:, 4]
    # Return adjoint of quadratic cost
    ∂f .= c3 .+ 2.0 .* c4 .* pg
    return
end

function hessian_cost!(polar, ∂²f)
    coefs = polar.costs_coefficients
    c4 = @view coefs[:, 4]
    ∂²f .= 2.0 .* c4
    return
end

function put(
    polar::PolarForm{T, IT, VT, MT},
    ::PS.Generators,
    ::PS.ActivePower,
    obj_autodiff::AdjointStackObjective,
    buffer::PolarNetworkState
) where {T, IT, VT, MT}

    index_pv = polar.indexing.index_pv
    pv2gen = polar.indexing.index_pv_to_gen
    ref2gen = polar.indexing.index_ref_to_gen

    adj_pg = obj_autodiff.∂pg
    adj_x = obj_autodiff.∇fₓ
    adj_u = obj_autodiff.∇fᵤ
    adj_vmag = obj_autodiff.∂vm
    adj_vang = obj_autodiff.∂va
    adj_pinj = obj_autodiff.∂pinj

    # Adjoint w.r.t Slack nodes
    adjoint!(polar, active_power_constraints, view(adj_pg, ref2gen), buffer.pg, obj_autodiff, buffer)
    # Adjoint w.r.t. PV nodes
    adj_pinj[index_pv] .= @view adj_pg[pv2gen]

    # Adjoint w.r.t. x and u
    fill!(adj_x, 0.0)
    fill!(adj_u, 0.0)
    adjoint_transfer!(polar, adj_u, adj_x, adj_vmag, adj_vang, adj_pinj, nothing)

    return
end

function gradient_objective!(polar::PolarForm, ∂obj::AdjointStackObjective, buffer::PolarNetworkState)
    adjoint_cost!(polar, ∂obj.∂pg, buffer.pg)
    put(polar, PS.Generators(), PS.ActivePower(), ∂obj, buffer)
    return
end

function hessian_prod_objective!(
    polar::PolarForm,
    ∇²f::AutoDiff.Hessian, ∇f::AdjointStackObjective,
    hv::AbstractVector,
    ∂²f::AbstractVector, ∂f::AbstractVector,
    buffer::PolarNetworkState,
    tgt::AbstractVector,
)
    nx = get(polar, NumberOfState())
    nu = get(polar, NumberOfControl())
    # Indexing of generators
    pv2gen = polar.indexing.index_pv_to_gen ; npv = length(pv2gen)
    ref2gen = polar.indexing.index_ref_to_gen ; nref = length(ref2gen)

    # Remember that
    # ```math
    # ∂f = (∂f / ∂pg) * ∂pg
    # ```
    # Using the chain-rule, we get
    # ```math
    # ∂²f = (∂f / ∂pg) * ∂²pg + ∂pg' * (∂²f / ∂²pg) * ∂pg
    # ```

    ## Step 1: evaluate (∂f / ∂pg) * ∂²pg
    ∂pg_ref = @view ∂f[ref2gen]
    AutoDiff.adj_hessian_prod!(polar, ∇²f, hv, buffer, ∂pg_ref, tgt)

    ## Step 2: evaluate ∂pg' * (∂²f / ∂²pg) * ∂pg
    # Compute adjoint w.r.t. ∂pg_ref
    ∇f.∂pg .= 0.0
    ∇f.∂pg[ref2gen] .= 1.0
    put(polar, PS.Generators(), PS.ActivePower(), ∇f, buffer)
    @views begin
        tx = tgt[1:nx]
        tu = tgt[1+nx:nx+nu]
        tpg = tgt[nx+nu-npv+1:end]

        ∇pgₓ = ∇f.∇fₓ
        ∇pgᵤ = ∇f.∇fᵤ

        scale_x = dot(∇pgₓ, ∂²f[ref2gen[1]], tx)
        scale_u = dot(∇pgᵤ, ∂²f[ref2gen[1]], tu)
        # Contribution of slack node
        hv[1:nx]       .+= (scale_x + scale_u) .* ∇pgₓ
        hv[1+nx:nx+nu] .+= (scale_x + scale_u) .* ∇pgᵤ
        # Contribution of PV nodes (only through power generation)
        hv[nx+nu-npv+1:end] .+= ∂²f[pv2gen] .* tpg
    end

    return nothing
end

#=
    Special function to compute Hessian of ProxAL's objective.
    This avoid to reuse intermediate results, for efficiency.

    This function collect the contribution of the state and
    the control (in `hv_xu`) and the contribution of the slack
    variable (`hv_s`).

    For ProxAL, we have:
    H = [ H_xx  H_ux  J_x' ]
        [ H_xu  H_uu  J_u' ]
        [ J_x   J_u   ρ I  ]

    so, if `v = [v_x; v_u; v_s]`, we get

    H * v = [ H_xx v_x  +   H_ux v_u  +  J_x' v_s ]
            [ H_xu v_x  +   H_uu v_u  +  J_u' v_s ]
            [  J_x v_x  +    J_u v_u  +   ρ I     ]

=#
function hessian_prod_objective_proxal!(
    polar::PolarForm,
    ∇²f::AutoDiff.Hessian, ∇f::AdjointStackObjective,
    hv_xu::AbstractVector,
    hv_s::AbstractVector,
    ∂²f::AbstractVector, ∂f::AbstractVector,
    buffer::PolarNetworkState,
    tgt::AbstractVector,
    vs::AbstractVector,
    ρ::Float64,
    has_slack::Bool,
)
    nx = get(polar, NumberOfState())
    nu = get(polar, NumberOfControl())
    ngen = get(polar, PS.NumberOfGenerators())

    # Indexing of generators
    pv2gen = polar.indexing.index_pv_to_gen ; npv = length(pv2gen)
    ref2gen = polar.indexing.index_ref_to_gen ; nref = length(ref2gen)

    # Split tangent wrt x part and u part
    tx = @view tgt[1:nx]
    tu = @view tgt[1+nx:nx+nu]

    #= BLOCK UU
        [ H_xx v_x  +   H_ux v_u ]
        [ H_xu v_x  +   H_uu v_u ]
    =#
    ## Step 1: evaluate (∂f / ∂pg) * ∂²pg
    ∂pg_ref = @view ∂f[ref2gen]
    AutoDiff.adj_hessian_prod!(polar, ∇²f, hv_xu, buffer, ∂pg_ref, tgt)

    ## Step 2: evaluate ∂pg' * (∂²f / ∂²pg) * ∂pg
    # Compute adjoint w.r.t. ∂pg_ref
    ∇f.∂pg .= 0.0
    ∇f.∂pg[ref2gen] .= 1.0
    put(polar, PS.Generators(), PS.ActivePower(), ∇f, buffer)

    ∇pgₓ = ∇f.∇fₓ
    ∇pgᵤ = ∇f.∇fᵤ

    @views begin
        tpg = tgt[nx+nu-npv+1:end]

        scale_x = dot(∇pgₓ, ∂²f[ref2gen[1]], tx)
        scale_u = dot(∇pgᵤ, ∂²f[ref2gen[1]], tu)
        # Contribution of slack node
        hv_xu[1:nx]       .+= (scale_x + scale_u) .* ∇pgₓ
        hv_xu[1+nx:nx+nu] .+= (scale_x + scale_u) .* ∇pgᵤ
        # Contribution of PV nodes (only through power generation)
        hv_xu[nx+nu-npv+1:end] .+= ∂²f[pv2gen] .* tpg
    end

    if has_slack
        @views begin
            #= BLOCK SU
                [  J_x' v_s ]
                [  J_u' v_s ]
                [  J_x v_x + J_u v_u ]
            =#
            # 1. Contribution of slack node
            hv_s[ref2gen] .+= - ρ * (dot(∇pgᵤ, tu) + dot(∇pgₓ, tx))
            hv_xu[1:nx] .-= ρ * ∇pgₓ .* vs[ref2gen]
            hv_xu[1+nx:nx+nu] .-= ρ * ∇pgᵤ .* vs[ref2gen]

            # 2. Contribution of PV node
            hv_s[pv2gen] .-= ρ .* tu[nref+npv+1:end]
            hv_xu[nx+nref+npv+1:end] .-= ρ .* vs[pv2gen]

            #= BLOCK SS
                 ρ I
            =#
            # Hessian w.r.t. slack
            hv_s .+= ρ .* vs
        end
    end

    return nothing
end

