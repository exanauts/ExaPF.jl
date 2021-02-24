is_constraint(::typeof(voltage_magnitude_constraints)) = true

# We add constraint only on vmag_pq
function voltage_magnitude_constraints(polar::PolarForm, cons, buffer)
    index_pq = polar.indexing.index_pq
    cons .= @view buffer.vmag[index_pq]
    return
end

function size_constraint(polar::PolarForm, ::typeof(voltage_magnitude_constraints))
    return PS.get(polar.network, PS.NumberOfPQBuses())
end

function bounds(polar::PolarForm, ::typeof(voltage_magnitude_constraints))
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    fr_ = npq + npv + 1
    to_ = 2*npq + npv
    return polar.x_min[fr_:to_], polar.x_max[fr_:to_]
end

# State Jacobian: Jx_i = [0, ..., 1, ... 0] where
function jacobian(polar::PolarForm, ::typeof(voltage_magnitude_constraints), i_cons::Int, ∂jac, buffer)
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    fr_ = npq + npv

    # Adjoint / State
    fill!(∂jac.∇fₓ, 0)
    ∂jac.∇fₓ[fr_ + i_cons] = 1.0
    # Adjoint / Control
    fill!(∂jac.∇fᵤ, 0)
end

function jacobian(polar::PolarForm, cons::typeof(voltage_magnitude_constraints), buffer)
    m = size_constraint(polar, cons)
    nᵤ = get(polar, NumberOfControl())
    nₓ = get(polar, NumberOfState())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    shift = npq + npv

    I = 1:m
    J = (shift+1):(shift+npq)
    V = ones(m)
    jx = sparse(I, J, V, m, nₓ)
    ju = spzeros(m, nᵤ)
    return FullSpaceJacobian(jx, ju)
end

function jtprod(polar::PolarForm, ::typeof(voltage_magnitude_constraints), ∂jac, buffer, v)
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    fr_ = npq + npv + 1
    # Adjoint / Control
    fill!(∂jac.∇fᵤ, 0)
    # Adjoint / State
    fill!(∂jac.∇fₓ, 0)
    ∂jac.∇fₓ[fr_:end] .= v
end

function hessian(polar::PolarForm, ::typeof(voltage_magnitude_constraints), buffer, λ)
    nu = get(polar, NumberOfControl())
    nx = get(polar, NumberOfState())
    return FullSpaceHessian(
        spzeros(nx, nx),
        spzeros(nu, nx),
        spzeros(nu, nu),
    )
end

