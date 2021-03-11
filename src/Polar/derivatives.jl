
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

    AutoDiff.seed!(jac.t1sseeds, jac.varx, jac.t1svarx)

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

    AutoDiff.getpartials_kernel!(jac.compressedJ, jac.t1sF)
    AutoDiff.uncompress_kernel!(jac.J, jac.compressedJ, jac.coloring)
    return jac.J
end

# Handle properly constant Jacobian case
function AutoDiff.ConstantJacobian(polar::PolarForm, func::Function, variable::Union{State,Control})
    @assert is_constraint(func)

    if isa(polar.device, CPU)
        SMT = SparseMatrixCSC{Float64,Int}
    elseif isa(polar.device, CUDADevice)
        SMT = CUSPARSE.CuSparseMatrixCSR{Float64}
    end

    nbus = get(polar, PS.NumberOfBuses())
    vmag = ones(nbus)
    vang = ones(nbus)
    V = vmag .* exp.(im .* vang)
    # Evaluate Jacobian with MATPOWER
    J = matpower_jacobian(polar, variable, func, V)
    # Move Jacobian to the GPU
    if isa(polar.device, CUDADevice) && iszero(J)
        J = 0.0
    else
        J = convert(SMT, J)
    end
    return AutoDiff.ConstantJacobian(J)
end

function AutoDiff.jacobian!(polar::PolarForm, jac::AutoDiff.ConstantJacobian, buffer)
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
    host_t1sseeds = Vector{ForwardDiff.Partials{1,Float64}}(undef, nmap)
    t1sseeds = A{ForwardDiff.Partials{1,Float64}}(undef, nmap)
    varx = view(x, map)
    t1svarx = view(t1sx, map)
    VHP = typeof(host_t1sseeds)
    VP = typeof(t1sseeds)
    VD = typeof(t1sx)
    adj_t1sx = similar(t1sx)
    adj_t1sF = similar(t1sF)
    return AutoDiff.Hessian{typeof(func), VI, VT, VHP, VP, VD, typeof(varx), typeof(t1svarx)}(
        func, host_t1sseeds, t1sseeds, t1sF, adj_t1sF, x, t1sx, adj_t1sx, map, varx, t1svarx
    )
end

# λ' * H * v
function AutoDiff.adj_hessian_prod!(
    polar, H::AutoDiff.Hessian, hv, buffer, λ, v,
)
    @assert length(hv) == length(v)
    nbus = get(polar, PS.NumberOfBuses())
    x = H.x
    ntgt = length(v)
    t1sx = H.t1sx
    adj_t1sx = H.∂t1sx
    t1sF = H.t1sF
    adj_t1sF = H.∂t1sF
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
    hostv = Array(v)
    @inbounds Threads.@threads for i in 1:nmap
        H.host_t1sseeds[i] = ForwardDiff.Partials{1, Float64}(NTuple{1, Float64}(hostv[i]))
    end
    copyto!(H.t1sseeds, H.host_t1sseeds)
    AutoDiff.seed!(H.t1sseeds, H.varx, H.t1svarx)

    adjoint!(
        polar, H.func,
        t1sF, adj_t1sF,
        view(t1sx, 1:nbus), view(adj_t1sx, 1:nbus),                   # vmag
        view(t1sx, nbus+1:2*nbus), view(adj_t1sx, nbus+1:2*nbus),     # vang
        view(t1sx, 2*nbus+1:3*nbus), view(adj_t1sx, 2*nbus+1:3*nbus), # pinj
        view(t1sx, 3*nbus+1:4*nbus), view(adj_t1sx, 3*nbus+1:4*nbus), # qinj
    )

    AutoDiff.getpartials_kernel!(hv, adj_t1sx, H.map)
    return nothing
end


# Adjoint's structure
"""
    AdjointStackObjective{VT}

An object for storing the adjoint stack in the adjoint objective computation.

"""
struct AdjointStackObjective{VT<:AbstractVector}
    ∇fₓ::VT
    ∇fᵤ::VT
    ∂pg::VT
    ∂vm::VT
    ∂va::VT
    jvₓ::VT
    jvᵤ::VT
end

function AdjointStackObjective(polar::PolarForm{T, VI, VT, MT}) where {T, VI, VT, MT}
    nbus = get(polar, PS.NumberOfBuses())
    return AdjointStackObjective{VT}(
        xzeros(VT, get(polar, NumberOfState())),
        xzeros(VT, get(polar, NumberOfControl())),
        xzeros(VT, get(polar, PS.NumberOfGenerators())),
        xzeros(VT, nbus),
        xzeros(VT, nbus),
        xzeros(VT, get(polar, NumberOfState())),
        xzeros(VT, get(polar, NumberOfControl())),
    )
end

struct AdjointPolar{VT<:AbstractVector} <: AutoDiff.AbstractAdjointStack
    ∂vm::VT
    ∂va::VT
    ∂pinj::VT
    ∂qinj::VT
    ∂x::VT
    ∂u::VT
end

function AdjointPolar{VT}(nx::Int, nu::Int, nbus::Int) where {VT}
    return AdjointPolar{VT}(
        xzeros(VT, nbus),
        xzeros(VT, nbus),
        xzeros(VT, nbus),
        xzeros(VT, nbus),
        xzeros(VT, nx),
        xzeros(VT, nu),
    )
end

# Stack constructor for each constraint
function AdjointPolar(polar::PolarForm{T, VI, VT, MT}) where {T, VI, VT, MT}
    nbus = get(polar, PS.NumberOfBuses())
    nx = get(polar, NumberOfState())
    nu = get(polar, NumberOfControl())
    return AdjointPolar{VT}(nx, nu, nbus)
end

struct JacobianStorage{Jacx, Jacu}
    Jx::Jacx
    Ju::Jacu
end

struct HessianStorage{VT,Hess1,Hess2}
    state::Hess1
    obj::Hess2
    constraints::Vector{AutoDiff.Hessian}
    # Adjoints
    z::VT
    ψ::VT
    tmp_tgt::VT
    tmp_hv::VT
end

function HessianStorage(polar::PolarForm{T, VI, VT, MT}) where {T, VI, VT, MT}
    nx = get(polar, NumberOfState())
    nu = get(polar, NumberOfControl())

    Hstate = AutoDiff.Hessian(polar, power_balance)
    Hobj = AutoDiff.Hessian(polar, active_power_constraints)
    Hcons = AutoDiff.Hessian[]

    z = xzeros(VT, nx)
    ψ = xzeros(VT, nx)

    tgt = xzeros(VT, nx+nu)
    hv = xzeros(VT, nx+nu)

    return HessianStorage{VT, typeof(Hstate), typeof(Hobj)}(Hstate, Hobj, Hcons, z, ψ, tgt, hv)
end

