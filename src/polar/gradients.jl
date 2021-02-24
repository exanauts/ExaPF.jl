struct FullSpaceJacobian{SpM}
    x::SpM
    u::SpM
end

function _matpower_residual_jacobian(V, Ybus)
    n = size(V, 1)
    Ibus = Ybus*V
    diagV       = sparse(1:n, 1:n, V, n, n)
    diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
    diagVnorm   = sparse(1:n, 1:n, V./abs.(V), n, n)

    dSbus_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
    dSbus_dVa = 1im * diagV * conj(diagIbus - Ybus * diagV)
    return (dSbus_dVm, dSbus_dVa)
end

# Jacobian Jₓ (from Matpower)
"""
    residual_jacobian(V, Ybus, pv, pq)

Compute the Jacobian w.r.t. the state `x` of the power
balance function [`power_balance`](@ref).

# Note
Code adapted from MATPOWER.
"""
function residual_jacobian(::State, V, Ybus, pv, pq, ref)
    # @warn("deprecated residual_jacobian")
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
    j12 = real(dSbus_dVm[[pv; pq], pq])
    j21 = imag(dSbus_dVa[pq, [pv; pq]])
    j22 = imag(dSbus_dVm[pq, pq])

    J = [j11 j12; j21 j22]
end

# Jacobian Jᵤ (from Matpower)
function residual_jacobian(::Control, V, Ybus, pv, pq, ref)
    # @warn("deprecated residual_jacobian")
    dSbus_dVm, _ = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVm[[pv; pq], [ref; pv; pv]])
    j21 = imag(dSbus_dVm[pq, [ref; pv; pv]])
    J = [j11; j21]
end

function residual_jacobian(A::Attr, polar::PolarForm) where {Attr <: Union{State, Control}}
    # @warn("deprecated residual_jacobian")
    pf = polar.network
    ref = polar.network.ref
    pv = polar.network.pv
    pq = polar.network.pq
    n = PS.get(pf, PS.NumberOfBuses())

    Y = pf.Ybus
    # Randomized inputs
    Vre = rand(n)
    Vim = rand(n)
    V = Vre .+ 1im .* Vim
    return residual_jacobian(A, V, Y, pv, pq, ref)
end
_sparsity_pattern(polar::PolarForm) = findnz(residual_jacobian(State(), polar))


# Jacobian wrt active power generation
function active_power_jacobian(::State, V, Ybus, pv, pq, ref)
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVa[ref, [pv; pq]])
    j12 = real(dSbus_dVm[ref, pq])
    J = [
        j11 j12
        spzeros(length(pv), length(pv) + 2 * length(pq))
    ]
end

function active_power_jacobian(::Control, V, Ybus, pv, pq, ref)
    ngen = length(pv) + length(ref)
    npv = length(pv)
    dSbus_dVm, _ = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVm[ref, [ref; pv]])
    j12 = sparse(I, npv, npv)
    return [
        j11 spzeros(length(ref), npv)
        spzeros(npv, ngen) j12
    ]
end

function active_power_jacobian(
    polar::PolarForm,
    r::AbstractVariable,
    buffer::PolarNetworkState,
)
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    V = buffer.vmag .* exp.(im .* buffer.vang)
    return active_power_jacobian(r, V, polar.network.Ybus, pv, pq, ref)
end

# Jacobian wrt active power generation
function reactive_power_jacobian(::State, V, Ybus, pv, pq, ref)
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    j11 = imag(dSbus_dVa[[ref; pv], [pv; pq]])
    j12 = imag(dSbus_dVm[[ref; pv], pq])
    return [j11 j12]
end

function reactive_power_jacobian(::Control, V, Ybus, pv, pq, ref)
    ngen = length(pv) + length(ref)
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    j11 = imag(dSbus_dVm[[ref; pv], [ref; pv]])
    return [j11 spzeros(ngen, length(pv))]
end

function AutoDiff.Jacobian(
    func, polar::PolarForm{T, VI, VT, MT}, structure, variable,
) where {T, VI, VT, MT}

    if isa(polar.device, CPU)
        SMT = SparseMatrixCSC{Float64,Int}
        A = Vector
    elseif isa(polar.device, CUDADevice)
        SMT = CUSPARSE.CuSparseMatrixCSR{Float64}
        A = CUDA.CuVector
    end

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    map = VI(structure.map)
    nmap = length(structure.map)

    # Sparsity pattern
    Vre = Float64[i for i in 1:nbus]
    Vim = Float64[i for i in nbus+1:2*nbus]
    V = Vre .+ 1im .* Vim
    Y = pf.Ybus
    J = structure.sparsity(variable, V, Y, pf.pv, pf.pq, pf.ref)

    # Coloring
    coloring = AutoDiff.SparseDiffTools.matrix_colors(J)
    ncolor = size(unique(coloring),1)

    # TODO: clean
    nx = 2 * nbus
    x = VT(zeros(Float64, nx))
    m = size(J, 1)

    J = convert(SMT, J)

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    t1sx = A{t1s{ncolor}}(x)
    t1sF = A{t1s{ncolor}}(zeros(Float64, m))

    t1sseeds = AutoDiff.init_seed(coloring, ncolor, nmap)

    # Move the seeds over to the GPU
    gput1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(t1sseeds)
    compressedJ = MT(zeros(Float64, ncolor, m))

    varx = view(x, map)
    t1svarx = view(t1sx, map)

    return AutoDiff.Jacobian{typeof(func), VI, VT, MT, SMT, typeof(gput1sseeds), typeof(t1sx), typeof(varx), typeof(t1svarx)}(
        func, variable, J, compressedJ, coloring,
        gput1sseeds, t1sF, x, t1sx, map, varx, t1svarx
    )
end

function (jac::AutoDiff.Jacobian)(polar::PolarForm, buffer)
    nbus = get(polar, PS.NumberOfBuses())
    type = jac.var
    if isa(type, State)
        jac.x[1:nbus] .= buffer.vmag
        jac.x[nbus+1:2*nbus] .= buffer.vang
        jac.t1sx .= jac.x
        jac.t1sF .= 0.0
    elseif isa(type, Control)
        jac.x[1:nbus] .= buffer.vmag
        jac.x[nbus+1:2*nbus] .= buffer.pinj
        jac.t1sx .= jac.x
        jac.t1sF .= 0.0
    else
        error("Unsupported Jacobian structure")
    end

    AutoDiff.seed!(jac.t1sseeds, jac.varx, jac.t1svarx, nbus)

    if isa(type, State)
        jac.func(
            polar,
            jac.t1sF,
            view(jac.t1sx, 1:nbus),
            view(jac.t1sx, nbus+1:2*nbus),
            buffer.pinj,
            buffer.qinj,
        )
    elseif isa(type, Control)
        jac.func(
            polar,
            jac.t1sF,
            view(jac.t1sx, 1:nbus),
            buffer.vang,
            view(jac.t1sx, nbus+1:2*nbus),
            buffer.qinj,
        )
    else
        error("Unsupported Jacobian structure")
    end

    AutoDiff.getpartials_kernel!(jac.compressedJ, jac.t1sF, nbus)
    AutoDiff.uncompress_kernel!(jac.J, jac.compressedJ, jac.coloring)
    return jac.J
end

