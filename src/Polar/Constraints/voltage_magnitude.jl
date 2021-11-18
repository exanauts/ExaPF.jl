is_constraint(::typeof(voltage_magnitude_constraints)) = true

is_linear(polar::PolarForm, ::typeof(voltage_magnitude_constraints)) = true

# We add constraint only on vmag_pq
function voltage_magnitude_constraints(polar::PolarForm, cons, vmag, vang, pnet, qnet, pload, qload)
    _, _, pq = index_buses_device(polar)
    cons .= @view vmag[pq]
    return
end
function voltage_magnitude_constraints(polar::PolarForm, cons, buffer)
    _, _, pq = index_buses_device(polar)
    cons .= @view buffer.vmag[pq]
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

function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    cons, ∂cons,
    vmag, ∂vmag,
    vang, ∂vang,
    pnet, ∂pnet,
    pload, qload,
) where {F<:typeof(voltage_magnitude_constraints), S, I}
    _, _, pq = index_buses_device(polar)
    ∂vmag[pq] .= ∂cons
end

function matpower_jacobian(
    polar::PolarForm,
    X::Union{State,Control},
    cons::typeof(voltage_magnitude_constraints),
    V,
)
    m = size_constraint(polar, cons)
    nᵤ = get(polar, NumberOfControl())
    nₓ = get(polar, NumberOfState())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    shift = npq + npv

    I = 1:m
    J = (shift+1):(shift+npq)
    V = ones(m)
    if isa(X, State)
        return sparse(I, J, V, m, nₓ)
    elseif isa(X, Control)
        return spzeros(m, nᵤ)
    end
end

function matpower_hessian(polar::PolarForm, ::typeof(voltage_magnitude_constraints), buffer, λ)
    nu = get(polar, NumberOfControl())
    nx = get(polar, NumberOfState())
    return FullSpaceHessian(
        spzeros(nx, nx),
        spzeros(nu, nx),
        spzeros(nu, nu),
    )
end
