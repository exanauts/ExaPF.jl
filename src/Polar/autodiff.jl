
function AutoDiff.Jacobian(
    func, polar::PolarForm{T, VI, VT, MT}, vmap, variable,
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
    map = VI(vmap)
    nmap = length(vmap)

    # Sparsity pattern
    J = jacobian_sparsity(polar, func, variable)

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
    end

    AutoDiff.getpartials_kernel!(jac.compressedJ, jac.t1sF, nbus)
    AutoDiff.uncompress_kernel!(jac.J, jac.compressedJ, jac.coloring)
    return jac.J
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

