
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

function AutoDiff.Hessian(polar::PolarForm{T, VI, VT, MT}, func) where {T, VI, VT, MT}
    @assert is_constraint(func)

    if isa(polar.device, CPU)
        A = Vector
    elseif isa(polar.device, CUDADevice)
        A = CUDA.CuVector
    end

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    n_cons = size_constraint(polar, func)

    map = VI(polar.hessianstructure.map)
    nmap = length(map)

    x = VT(zeros(Float64, 4*nbus))

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    t1sx = A{t1s{1}}(x)
    t1sF = A{t1s{1}}(zeros(Float64, n_cons))
    t1sseeds = A{ForwardDiff.Partials{1,Float64}}(undef, nmap)
    varx = view(x, map)
    t1svarx = view(t1sx, map)
    VP = typeof(t1sseeds)
    VD = typeof(t1sx)
    return AutoDiff.Hessian{typeof(func), VI, VT, MT, Nothing, VP, VD, typeof(varx), typeof(t1svarx)}(
        func, t1sseeds, t1sF, x, t1sx, map, varx, t1svarx
    )
end

# λ' * H * v
function AutoDiff.adj_hessian_prod!(
    polar, H::AutoDiff.Hessian, buffer, λ, v,
)
    nbus = get(polar, PS.NumberOfBuses())
    x = H.x
    ntgt = length(v)
    t1sx = H.t1sx
    adj_t1sx = similar(t1sx)
    t1sF = H.t1sF
    adj_t1sF = similar(t1sF)
    # Move data
    x[1:nbus] .= buffer.vmag
    x[nbus+1:2*nbus] .= buffer.vang
    x[2*nbus+1:3*nbus] .= buffer.pinj
    x[3*nbus+1:4*nbus] .= buffer.qinj
    # Init dual variables
    t1sx .= H.x
    adj_t1sx .= 0.0
    t1sF .= 0.0
    adj_t1sF .= λ
    # Seeding
    nmap = length(H.map)

    # Init seed
    for i in 1:nmap
        H.t1sseeds[i] = ForwardDiff.Partials{1, Float64}(NTuple{1, Float64}(v[i]))
    end
    AutoDiff.seed!(H.t1sseeds, H.varx, H.t1svarx, nbus)

    adjoint!(
        polar, H.func,
        t1sF, adj_t1sF,
        view(t1sx, 1:nbus), view(adj_t1sx, 1:nbus), # vmag
        view(t1sx, nbus+1:2*nbus), view(adj_t1sx, nbus+1:2*nbus), # vang
        view(t1sx, 2*nbus+1:3*nbus), view(adj_t1sx, 2*nbus+1:3*nbus), # pinj
        view(t1sx, 3*nbus+1:4*nbus), view(adj_t1sx, 3*nbus+1:4*nbus), # qinj
    )

    # TODO, this is redundant
    ps = ForwardDiff.partials.(adj_t1sx[H.map])
    res = similar(v)
    res .= 0.0
    for i in 1:length(ps)
        res[i] = ps[i].values[1]
    end
    return res
end

## Utils
# TODO: find better naming
function init_autodiff_factory(polar::PolarForm{T, IT, VT, MT}, buffer::PolarNetworkState) where {T, IT, VT, MT}
    # Build the AutoDiff Jacobian structure
    statejacobian = AutoDiff.Jacobian(
        polar, power_balance, State(),
    )
    controljacobian = AutoDiff.Jacobian(
        polar, power_balance, Control(),
    )

    # Build the AutoDiff structure for the objective
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    nₓ = get(polar, NumberOfState())
    nᵤ = get(polar, NumberOfControl())
    Vm, Va, pbus, qbus = buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj
    ∇fₓ = xzeros(VT, nₓ)
    ∇fᵤ = xzeros(VT, nᵤ)
    adjoint_pg = similar(buffer.pg)
    adjoint_vm = similar(Vm)
    adjoint_va = similar(Va)
    # Build cache for Jacobian vector-product
    jvₓ = xzeros(VT, nₓ)
    jvᵤ = xzeros(VT, nᵤ)
    adjoint_flow = xzeros(VT, 2 * nbus)
    objectiveAD = AdjointStackObjective(∇fₓ, ∇fᵤ, adjoint_pg, adjoint_vm, adjoint_va, jvₓ, jvᵤ, adjoint_flow)

    return statejacobian, controljacobian, objectiveAD
end

