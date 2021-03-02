"""
    AutoDiff.Jacobian(polar, func::Function, variable::AbstractVariable)

Instantiate a Jacobian AD factory for constraint function
`func`, w.r.t. state ``x`` (if `variable=State()`) or control
``u`` (if `variable=Control()`).

The coloring is done using Jacobian's expressions from MATPOWER.

### Examples

```julia
julia> Jacx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
```
"""
function AutoDiff.Jacobian(
    polar::PolarForm{T, VI, VT, MT}, func, variable,
) where {T, VI, VT, MT}
    @assert is_constraint(func)

    if isa(polar.device, CPU)
        SMT = SparseMatrixCSC{Float64,Int}
        A = Vector
    elseif isa(polar.device, CUDADevice)
        SMT = CUSPARSE.CuSparseMatrixCSR{Float64}
        A = CUDA.CuVector
    end

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    if isa(variable, State)
        map = VI(polar.mapx)
    elseif isa(variable, Control)
        map = VI(polar.mapu)
    end

    nmap = length(map)

    # Sparsity pattern
    J = jacobian_sparsity(polar, func, variable)

    # Coloring
    coloring = AutoDiff.SparseDiffTools.matrix_colors(J)
    ncolor = size(unique(coloring),1)

    # TODO: clean
    nx = 2 * nbus
    x = VT(zeros(Float64, nx))
    m = size(J, 1)

    # Move Jacobian to the GPU
    J = convert(SMT, J)

    # Seedings
    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    t1sx = A{t1s{ncolor}}(x)
    t1sF = A{t1s{ncolor}}(zeros(Float64, m))
    t1sseeds = AutoDiff.init_seed(coloring, ncolor, nmap)

    # Move the seeds over to the device, if necessary
    gput1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(t1sseeds)
    compressedJ = MT(zeros(Float64, ncolor, m))

    # Views
    varx = view(x, map)
    t1svarx = view(t1sx, map)

    return AutoDiff.Jacobian{typeof(func), VI, VT, MT, SMT, typeof(gput1sseeds), typeof(t1sx), typeof(varx), typeof(t1svarx)}(
        func, variable, J, compressedJ, coloring,
        gput1sseeds, t1sF, x, t1sx, map, varx, t1svarx
    )
end

"""
    AutoDiff.jacobian!(polar::PolarForm, jac::AutoDiff.Jacobian, buffer)

Update the sparse Jacobian entries `jacobian.J` using AutoDiff.
No allocations are taking place in this function.

* `polar::PolarForm`: polar formulation, stores all parameters.
* `jac::AutoDiff.Jacobian`: AutoDiff Factory with Jacobian to update.
* `buffer::PolarNetworkState`: store current values for network's variables.

"""
function AutoDiff.jacobian!(polar::PolarForm, jac::AutoDiff.Jacobian, buffer)
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

## Utils
# Expression of Jacobians from MATPOWER
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

function _matpower_lineflow_jacobian(V, branches)
    nb = size(V, 1)
    f = branches.from_buses
    t = branches.to_buses
    nl = length(f)

    diagV     = sparse(1:nb, 1:nb, V, nb, nb)
    diagVnorm = sparse(1:nb, 1:nb, V./abs.(V), nb, nb)

    # Connection matrices
    Cf = sparse(1:nl, f, ones(nl), nl, nb)
    Ct = sparse(1:nl, t, ones(nl), nl, nb)

    i = [1:nl; 1:nl]
    Yf = sparse(i, [f; t], [branches.Yff; branches.Yft], nl, nb)
    Yt = sparse(i, [f; t], [branches.Ytf; branches.Ytt], nl, nb)

    If = Yf * V
    It = Yt * V

    diagCfV = sparse(1:nl, 1:nl, Cf * V, nl, nl)
    diagCtV = sparse(1:nl, 1:nl, Ct * V, nl, nl)

    Sf = diagCfV * conj(If)
    St = diagCtV * conj(It)

    diagIf = sparse(1:nl, 1:nl, If, nl, nl)
    diagIt = sparse(1:nl, 1:nl, It, nl, nl)

    dSf_dVm = conj(diagIf) * Cf * diagVnorm + diagCfV * conj(Yf * diagVnorm)
    dSf_dVa = 1im * (conj(diagIf) * Cf * diagV - diagCfV * conj(Yf * diagV))

    dSt_dVm = conj(diagIt) * Ct * diagVnorm + diagCtV * conj(Yt * diagVnorm)
    dSt_dVa = 1im * (conj(diagIt) * Ct * diagV - diagCtV * conj(Yt * diagV))

    return (Sf, St, dSf_dVm, dSf_dVa, dSt_dVm, dSt_dVa)
end

function _matpower_lineflow_power_jacobian(V, branches)
    nl = length(branches.from_buses)
    (Sf, St, dSf_dVm, dSf_dVa, dSt_dVm, dSt_dVa) = _matpower_lineflow_jacobian(V, branches)

    dSf = sparse(1:nl, 1:nl, Sf, nl, nl)
    dSt = sparse(1:nl, 1:nl, St, nl, nl)

    dHf_dVm = 2 * (real(dSf) * real(dSf_dVm) + imag(dSf) * imag(dSf_dVm))
    dHf_dVa = 2 * (real(dSf) * real(dSf_dVa) + imag(dSf) * imag(dSf_dVa))

    dHt_dVm = 2 * (real(dSt) * real(dSt_dVm) + imag(dSt) * imag(dSt_dVm))
    dHt_dVa = 2 * (real(dSt) * real(dSt_dVa) + imag(dSt) * imag(dSt_dVa))

    dH_dVm = [dHf_dVm; dHt_dVm]
    dH_dVa = [dHf_dVa; dHt_dVa]
    return dH_dVm, dH_dVa
end

