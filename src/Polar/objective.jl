is_constraint(::typeof(cost_production)) = true
size_constraint(polar::PolarForm, ::typeof(cost_production)) = 1

function pullback_objective(polar::PolarForm)
    return AutoDiff.TapeMemory(
        cost_production,
        AdjointStackObjective(polar),
        nothing,
    )
end

@inline quadratic_cost(pg, c0, c1, c2) = c0 + c1 * pg + c2 * pg^2
@inline adj_quadratic_cost(pg, c0, c1, c2) = c1 + 2.0 * c2 * pg

KA.@kernel function cost_production_kernel!(
    costs, pg, @Const(vmag), @Const(vang), @Const(pinj), @Const(pload),
    @Const(c0), @Const(c1), @Const(c2),
    @Const(pv), @Const(ref), @Const(pv_to_gen), @Const(ref_to_gen),
    @Const(ybus_re_nzval), @Const(ybus_re_colptr), @Const(ybus_re_rowval),
    @Const(ybus_im_nzval),
)
    i, j = @index(Global, NTuple)
    npv = length(pv)
    nref = length(ref)
    # Evaluate active power at PV nodes
    if i <= npv
        bus = pv[i]
        i_gen = pv_to_gen[i]
        pg[i_gen, j] = pinj[bus, j] + pload[bus]
    # Evaluate active power at slack nodes
    elseif i <= npv + nref
        i_ = i - npv
        bus = ref[i_]
        i_gen = ref_to_gen[i_]
        inj = 0.0
        @inbounds for c in ybus_re_colptr[bus]:ybus_re_colptr[bus+1]-1
            to = ybus_re_rowval[c]
            aij = vang[bus, j] - vang[to, j]
            # f_re = a * cos + b * sin
            # f_im = a * sin - b * cos
            coef_cos = vmag[bus, j]*vmag[to, j]*ybus_re_nzval[c]
            coef_sin = vmag[bus, j]*vmag[to, j]*ybus_im_nzval[c]
            cos_val = cos(aij)
            sin_val = sin(aij)
            inj += coef_cos * cos_val + coef_sin * sin_val
        end
        pg[i_gen, j] = inj + pload[bus]
    end

    costs[i_gen, j] = quadratic_cost(pg[i_gen, j], c0[i_gen], c1[i_gen], c2[i_gen])
end

KA.@kernel function adj_cost_production_kernel!(
    adj_costs,
    @Const(vmag), adj_vmag, @Const(vang), adj_vang, @Const(pinj), adj_pinj, @Const(pload),
    @Const(c0), @Const(c1), @Const(c2),
    @Const(pv), @Const(ref), @Const(pv_to_gen), @Const(ref_to_gen),
    @Const(ybus_re_nzval), @Const(ybus_re_colptr), @Const(ybus_re_rowval), @Const(ybus_im_nzval),
)
    i, j = @index(Global, NTuple)
    npv = length(pv)
    nref = length(ref)
    if i <= npv
        bus = pv[i]
        i_gen = pv_to_gen[i]
        pg = pinj[bus, j] + pload[bus]
        adj_pinj[bus, j] = adj_costs[1] * adj_quadratic_cost(pg, c0[i_gen], c1[i_gen], c2[i_gen])
    # Evaluate active power at slack nodes
    elseif i <= npv + nref
        i_ = i - npv
        fr = ref[i_]
        i_gen = ref_to_gen[i_]

        inj = 0.0
        @inbounds for c in ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1
            to = ybus_re_rowval[c]
            aij = vang[fr, j] - vang[to, j]
            # f_re = a * cos + b * sin
            # f_im = a * sin - b * cos
            coef_cos = vmag[fr, j]*vmag[to, j]*ybus_re_nzval[c]
            coef_sin = vmag[fr, j]*vmag[to, j]*ybus_im_nzval[c]
            cos_val = cos(aij)
            sin_val = sin(aij)
            inj += coef_cos * cos_val + coef_sin * sin_val
        end
        pg = inj + pload[fr]

        adj_inj = adj_costs[1] * adj_quadratic_cost(pg, c0[i_gen], c1[i_gen], c2[i_gen])
        adj_pinj[fr, j] = adj_inj
        @inbounds for c in ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1
            to = ybus_re_rowval[c]
            aij = vang[fr, j] - vang[to, j]
            # f_re = a * cos + b * sin
            # f_im = a * sin - b * cos
            coef_cos = vmag[fr, j]*vmag[to, j]*ybus_re_nzval[c]
            coef_sin = vmag[fr, j]*vmag[to, j]*ybus_im_nzval[c]
            cosθ = cos(aij)
            sinθ = sin(aij)

            adj_coef_cos = cosθ  * adj_inj
            adj_cos_val  = coef_cos * adj_inj
            adj_coef_sin = sinθ  * adj_inj
            adj_sin_val  = coef_sin * adj_inj

            adj_aij =   cosθ * adj_sin_val
            adj_aij -=  sinθ * adj_cos_val

            adj_vmag[fr, j] += vmag[to, j] * ybus_re_nzval[c] * adj_coef_cos
            adj_vmag[to, j] += vmag[fr, j] * ybus_re_nzval[c] * adj_coef_cos
            adj_vmag[fr, j] += vmag[to, j] * ybus_im_nzval[c] * adj_coef_sin
            adj_vmag[to, j] += vmag[fr, j] * ybus_im_nzval[c] * adj_coef_sin

            adj_vang[fr, j] += adj_aij
            adj_vang[to, j] -= adj_aij
        end
    end
end

# Adjoints needed in polar formulation
function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    pg, ∂cost,
    vm, ∂vm,
    va, ∂va,
    pnet, ∂pnet,
    pload, qload,
) where {F<:typeof(cost_production), S, I}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    index_pv = polar.indexing.index_pv
    index_ref = polar.indexing.index_ref
    pv2gen = polar.indexing.index_pv_to_gen
    ref2gen = polar.indexing.index_ref_to_gen

    coefs = polar.costs_coefficients
    c0 = @view coefs[:, 2]
    c1 = @view coefs[:, 3]
    c2 = @view coefs[:, 4]

    ngen = get(polar, PS.NumberOfGenerators())
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    fill!(∂vm, 0.0)
    fill!(∂va, 0.0)
    fill!(∂pnet, 0.0)
    ev = adj_cost_production_kernel!(polar.device)(
        ∂cost,
        vm, ∂vm,
        va, ∂va,
        pnet, ∂pnet, pload,
        c0, c1, c2,
        index_pv, index_ref, pv2gen, ref2gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval, ybus_im.nzval,
        ndrange=(ngen, size(∂vm, 2)),
        dependencies=Event(polar.device)
    )
    wait(ev)
    return
end

function cost_production(polar::PolarForm, buffer::PolarNetworkState)
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    pv2gen = polar.indexing.index_pv_to_gen
    ref2gen = polar.indexing.index_ref_to_gen
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    ngen = PS.get(polar, PS.NumberOfGenerators())
    coefs = polar.costs_coefficients
    c0 = @view coefs[:, 2]
    c1 = @view coefs[:, 3]
    c2 = @view coefs[:, 4]
    costs = similar(buffer.pg)

    ev = cost_production_kernel!(polar.device)(
        costs, buffer.pgen,
        buffer.vmag, buffer.vang, buffer.pnet, buffer.pload,
        c0, c1, c2,
        pv, ref, pv2gen, ref2gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval, ybus_im.nzval,
        ndrange=(ngen, size(buffer.pinj, 2)),
        dependencies=Event(polar.device)
    )
    wait(ev)
    return sum(costs)
end

function gradient_objective!(polar::PolarForm, ∂obj::AutoDiff.TapeMemory, buffer::PolarNetworkState)
    ∂pg = ∂obj.stack.∂pg
    obj_autodiff = ∂obj.stack
    adj_pg = obj_autodiff.∂pg
    adj_x = obj_autodiff.∇fₓ
    adj_u = obj_autodiff.∇fᵤ
    adj_vmag = obj_autodiff.∂vm
    adj_vang = obj_autodiff.∂va
    adj_pinj = obj_autodiff.∂pinj

    # Adjoint of active power generation
    adjoint!(polar, ∂obj,
        buffer.pgen, 1.0,
        buffer.vmag, adj_vmag,
        buffer.vang, adj_vang,
        buffer.pnet, adj_pinj,
        buffer.pload, buffer.qload,
    )

    # Adjoint w.r.t. x and u
    fill!(adj_x, 0.0)
    fill!(adj_u, 0.0)
    adjoint_transfer!(polar, adj_u, adj_x, adj_vmag, adj_vang, adj_pinj)
    return
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
    ∇²f::AutoDiff.Hessian, adj_obj::AutoDiff.TapeMemory,
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

    # Stack
    ∇f = adj_obj.stack
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
    put(polar, PS.Generators(), PS.ActivePower(), adj_obj, buffer)

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

