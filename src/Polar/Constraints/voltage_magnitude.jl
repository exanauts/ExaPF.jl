is_constraint(::typeof(voltage_magnitude_constraints)) = true

is_linear(polar::PolarForm, ::typeof(voltage_magnitude_constraints)) = true

# We add constraint only on vmag_pq
function voltage_magnitude_constraints(polar::PolarForm, cons, vm, va, pinj, qinj)
    index_pq = polar.indexing.index_pq
    cons .= @view vm[index_pq]
    return
end
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

function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    cons, ∂cons,
    vm, ∂vm,
    va, ∂va,
    pinj, ∂pinj,
) where {F<:typeof(voltage_magnitude_constraints), S, I}
    index_pq = polar.indexing.index_pq
    ∂vm[index_pq] .= ∂cons
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
