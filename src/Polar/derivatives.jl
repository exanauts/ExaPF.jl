
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
    elseif isa(polar.device, GPU)
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
        copyto!(jac.x, 1, buffer.vmag, 1, nbus)
        copyto!(jac.x, nbus+1, buffer.vang, 1, nbus)
        jac.t1sx .= jac.x
        jac.t1sF .= 0.0
    elseif isa(type, Control)
        copyto!(jac.x, 1, buffer.vmag, 1, nbus)
        copyto!(jac.x, nbus+1, buffer.pnet, 1, nbus)
        jac.t1sx .= jac.x
        jac.t1sF .= 0.0
    end

    AutoDiff.seed!(jac.t1sseeds, jac.varx, jac.t1svarx, polar.device)

    if isa(type, State)
        jac.func(
            polar,
            jac.t1sF,
            view(jac.t1sx, 1:nbus),
            view(jac.t1sx, nbus+1:2*nbus),
            buffer.pnet,
            buffer.qnet,
            buffer.pload,
            buffer.qload,
        )
    elseif isa(type, Control)
        jac.func(
            polar,
            jac.t1sF,
            view(jac.t1sx, 1:nbus),
            buffer.vang,
            view(jac.t1sx, nbus+1:2*nbus),
            buffer.qnet,
            buffer.pload,
            buffer.qload,
        )
    end

    AutoDiff.getpartials_kernel!(jac.compressedJ, jac.t1sF, polar.device)
    AutoDiff.uncompress_kernel!(jac.J, jac.compressedJ, jac.coloring, polar.device)
    return jac.J
end

# Handle properly constant Jacobian case
function AutoDiff.ConstantJacobian(polar::PolarForm, func::Function, variable::Union{State,Control})
    @assert is_constraint(func)

    if isa(polar.device, CPU)
        SMT = SparseMatrixCSC{Float64,Int}
    elseif isa(polar.device, GPU)
        SMT = CUSPARSE.CuSparseMatrixCSR{Float64}
    end

    nbus = get(polar, PS.NumberOfBuses())
    vmag = ones(nbus)
    vang = ones(nbus)
    V = vmag .* exp.(im .* vang)
    # Evaluate Jacobian with MATPOWER
    J = matpower_jacobian(polar, variable, func, V)
    # Move Jacobian to the GPU
    if isa(polar.device, GPU) && iszero(J)
        # CUSPARSE does not support zero matrix. Return nothing instead.
        J = nothing
    else
        J = convert(SMT, J)
    end
    return AutoDiff.ConstantJacobian(J)
end

function AutoDiff.jacobian!(polar::PolarForm, jac::AutoDiff.ConstantJacobian, buffer)
    return jac.J
end


function AutoDiff.Hessian(polar::PolarForm{T, VI, VT, MT}, func; tape=nothing) where {T, VI, VT, MT}
    @assert is_constraint(func)

    if isa(polar.device, CPU)
        A = Vector
    elseif isa(polar.device, GPU)
        A = CUDA.CuVector
    end

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    n_cons = size_constraint(polar, func)

    map = VI(polar.hessianstructure.map)
    nmap = length(map)

    x = VT(zeros(Float64, 3*nbus))

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
    if isnothing(tape)
        buffer = AutoDiff.TapeMemory(polar, func, VD; with_stack=false)
    else
        buffer = tape
    end
    return AutoDiff.Hessian(
        func, host_t1sseeds, t1sseeds, x, t1sF, adj_t1sF, t1sx, adj_t1sx, map, varx, t1svarx, buffer,
    )
end

function _init_seed_hessian!(dest, tmp, v::AbstractArray, nmap)
    @inbounds for i in 1:nmap
        dest[i] = ForwardDiff.Partials{1, Float64}(NTuple{1, Float64}(v[i]))
    end
    return
end
function _init_seed_hessian!(dest, tmp, v::CUDA.CuArray, nmap)
    hostv = Array(v)
    @inbounds Threads.@threads for i in 1:nmap
        tmp[i] = ForwardDiff.Partials{1, Float64}(NTuple{1, Float64}(hostv[i]))
    end
    copyto!(dest, tmp)
    return
end

function update_hessian!(polar::PolarForm, H::AutoDiff.Hessian, buffer)
    nbatch = size(H.t1sx, 2)
    nbus = get(polar, PS.NumberOfBuses())

    # Move data
    copyto!(H.x,        1, buffer.vmag, 1, nbus)
    copyto!(H.x,   nbus+1, buffer.vang, 1, nbus)
    copyto!(H.x, 2*nbus+1, buffer.pnet, 1, nbus)
    @inbounds for i in 1:nbatch
        H.t1sx[:, i] .= H.x
    end
    return
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
    copyto!(x, 1, buffer.vmag, 1, nbus)
    copyto!(x, nbus+1, buffer.vang, 1, nbus)
    copyto!(x, 2*nbus+1, buffer.pnet, 1, nbus)
    # Init dual variables
    t1sx .= H.x
    adj_t1sx .= 0.0
    adj_t1sF .= λ
    # Seeding
    nmap = length(H.map)

    # Init seed
    _init_seed_hessian!(H.t1sseeds, H.host_t1sseeds, v, nmap)
    AutoDiff.seed!(H.t1sseeds, H.varx, H.t1svarx, polar.device)

    adjoint!(
        polar, H.buffer,
        t1sF, adj_t1sF,
        view(t1sx, 1:nbus), view(adj_t1sx, 1:nbus),                   # vmag
        view(t1sx, nbus+1:2*nbus), view(adj_t1sx, nbus+1:2*nbus),     # vang
        view(t1sx, 2*nbus+1:3*nbus), view(adj_t1sx, 2*nbus+1:3*nbus), # pnet
        buffer.pload, buffer.qload,
    )

    AutoDiff.getpartials_kernel!(hv, adj_t1sx, H.map, polar.device)
    return nothing
end

function AutoDiff.adj_hessian_prod!(
    polar, H::AutoDiff.ConstantHessian, hv, buffer, λ, v,
)
    copyto!(hv, H.hv)
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
    ∂pinj::VT
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
        xzeros(VT, nbus),
        xzeros(VT, get(polar, NumberOfState())),
        xzeros(VT, get(polar, NumberOfControl())),
    )
end

# Adjoint's stack for Polar
struct AdjointPolar{VT} <: AutoDiff.AbstractAdjointStack{VT}
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

function reset!(adj::AdjointPolar)
    adj.∂vm .= 0.0
    adj.∂va .= 0.0
    adj.∂pinj .= 0.0
    adj.∂x .= 0.0
    adj.∂u .= 0.0
end

# Stack constructor for each constraint
function AdjointPolar(polar::PolarForm{T, VI, VT, MT}) where {T, VI, VT, MT}
    nbus = get(polar, PS.NumberOfBuses())
    nx = get(polar, NumberOfState())
    nu = get(polar, NumberOfControl())
    return AdjointPolar{VT}(nx, nu, nbus)
end


struct FullSpaceJacobian{Jacx,Jacu}
    x::Jacx
    u::Jacu
end

struct FullSpaceHessian{SpMT}
    xx::SpMT
    xu::SpMT
    uu::SpMT
end

#=
    JACOBIAN
=#
struct ConstraintsJacobianStorage{SpMT}
    Jx::SpMT
    Ju::SpMT
    constraints_ad::Vector{FullSpaceJacobian}
end

function ConstraintsJacobianStorage(polar::PolarForm{T, VI, VT, MT}, constraints::Vector{Function}) where {T, VI, VT, MT}
    if isa(polar.device, CPU)
        SpMT = SparseMatrixCSC{Float64, Int}
    elseif isa(polar.device, GPU)
        SpMT = CUSPARSE.CuSparseMatrixCSR{Float64}
    end

    SparseCPU = SparseMatrixCSC{Float64, Int}

    Jx = SparseCPU[]
    Ju = SparseCPU[]
    # Build global Jacobian on the CPU
    for cons in constraints
        push!(Jx, jacobian_sparsity(polar, cons, State()))
        push!(Ju, jacobian_sparsity(polar, cons, Control()))
    end
    gJx = convert(SpMT, vcat(Jx...))
    gJu = convert(SpMT, vcat(Ju...))

    # Build AD
    cons_ad = FullSpaceJacobian[]
    for cons in constraints
        jac_ad_x = _build_jacobian(polar, cons, State())
        jac_ad_u = _build_jacobian(polar, cons, Control())
        push!(cons_ad, FullSpaceJacobian(jac_ad_x, jac_ad_u))
    end

    return ConstraintsJacobianStorage{SpMT}(
        gJx,
        gJu,
        cons_ad,
    )
end

function update_full_jacobian!(
    polar::PolarForm,
    cons_jac::ConstraintsJacobianStorage{SpMT},
    buffer::PolarNetworkState
) where {SpMT}
    shift = 0
    for ad in cons_jac.constraints_ad
        # Update Jacobian
        Jx = AutoDiff.jacobian!(polar, ad.x, buffer)::SpMT
        Ju = AutoDiff.jacobian!(polar, ad.u, buffer)::Union{Nothing, SpMT}
        # Copy back results
        _transfer_sparse!(cons_jac.Jx, Jx, shift, polar.device)
        if !isnothing(Ju)
            _transfer_sparse!(cons_jac.Ju, Ju, shift, polar.device)
        end

        shift += size(Jx, 1)
    end
    return
end

#=
    HESSIAN
=#
# Small utils to compute the factorization for batch Hessian algorithm
function _batch_hessian_factorization(J::AbstractSparseMatrix, nbatch)
    lufac = LS.exa_factorize(J)
    if isnothing(lufac)
        error("Unable to find a factorization routine for type $(typeof(J))")
    end
    return (lufac, lufac')
end

abstract type AbstractHessianStorage end

struct HessianLagrangian{VT,Hess,Fac1,Fac2} <: AbstractHessianStorage
    hess::Hess
    # Adjoints
    y::VT
    z::VT
    ψ::VT
    tmp_tgt::VT
    tmp_hv::VT
    lu::Fac1
    adjlu::Fac2
end
function HessianLagrangian(polar::PolarForm{T, VI, VT, MT}, J::AbstractSparseMatrix) where {T, VI, VT, MT}
    lu1, lu2 = _batch_hessian_factorization(J, 1)
    nx, nu = get(polar, NumberOfState()), get(polar, NumberOfControl())
    m = size_constraint(polar, network_operations)
    H = AutoDiff.Hessian(polar, network_operations)
    y = VT(undef, m)
    z = VT(undef, nx)
    ψ = VT(undef, nx)
    tgt = VT(undef, nx+nu)
    hv = VT(undef, nx+nu)
    return HessianLagrangian(H, y, z, ψ, tgt, hv, lu1, lu2)
end
n_batches(hlag::HessianLagrangian) = 1

struct BatchHessianLagrangian{MT,Hess,Fac1,Fac2} <: AbstractHessianStorage
    nbatch::Int
    hess::Hess
    # Adjoints
    y::MT
    z::MT
    ψ::MT
    tmp_tgt::MT
    tmp_hv::MT
    lu::Fac1
    adjlu::Fac2
end
function BatchHessianLagrangian(polar::PolarForm{T, VI, VT, MT}, J, nbatch) where {T, VI, VT, MT}
    lu1, lu2 = _batch_hessian_factorization(J, nbatch)
    nx, nu = get(polar, NumberOfState()), get(polar, NumberOfControl())
    m = size_constraint(polar, network_operations)
    H = BatchHessian(polar, network_operations, nbatch)
    y   = MT(undef, m, 1)  # adjoint is the same for all batches
    z   = MT(undef, nx, nbatch)
    ψ   = MT(undef, nx, nbatch)
    tgt = MT(undef, nx+nu, nbatch)
    hv  = MT(undef, nx+nu, nbatch)
    return BatchHessianLagrangian(nbatch, H, y, z, ψ, tgt, hv, lu1, lu2)
end
n_batches(hlag::BatchHessianLagrangian) = hlag.nbatch


function update_factorization!(hlag::AbstractHessianStorage, J::AbstractSparseMatrix)
    LinearAlgebra.lu!(hlag.lu, J)
    return
end

function update_factorization!(hlag::AbstractHessianStorage, J::CUSPARSE.CuSparseMatrixCSR)
    LinearAlgebra.lu!(hlag.lu, J)
    ∇gₓᵀ = CUSPARSE.CuSparseMatrixCSC(J)
    LinearAlgebra.lu!(hlag.adjlu, ∇gₓᵀ)
    return
end

