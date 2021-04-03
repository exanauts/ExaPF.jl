
# Adjoint
function batch_adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    cons, ∂cons,
    vm, ∂vm,
    va, ∂va,
    pinj, ∂pinj,
) where {F<:typeof(power_balance), S, I}
    nbus = get(polar, PS.NumberOfBuses())
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    qinj = polar.reactive_load
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    fill!(pbm.intermediate.∂edge_vm_fr , 0.0)
    fill!(pbm.intermediate.∂edge_vm_to , 0.0)
    fill!(pbm.intermediate.∂edge_va_fr , 0.0)
    fill!(pbm.intermediate.∂edge_va_to , 0.0)

    batch_adj_residual_polar!(
        cons, ∂cons,
        vm, ∂vm,
        va, ∂va,
        ybus_re, ybus_im, polar.topology.sortperm,
        pinj, ∂pinj, qinj,
        pbm.intermediate.∂edge_vm_fr,
        pbm.intermediate.∂edge_vm_to,
        pbm.intermediate.∂edge_va_fr,
        pbm.intermediate.∂edge_va_to,
        pv, pq, nbus,
    )
end

function batch_hessian(polar::PolarForm{T, VI, VT, MT}, func, nbatch) where {T, VI, VT, MT}
    @assert is_constraint(func)

    if isa(polar.device, CPU)
        A = Vector
        MMT = Matrix
    elseif isa(polar.device, CUDADevice)
        A = CUDA.CuVector
        MMT = CuMatrix
    end

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    n_cons = size_constraint(polar, func)

    map = VI(polar.hessianstructure.map)
    nmap = length(map)

    x = VT(zeros(Float64, 3*nbus))

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    t1sx = MMT{t1s{1}}(zeros(Float64, nbatch, 3*nbus))
    t1sF = MMT{t1s{1}}(zeros(Float64, nbatch, n_cons))
    host_t1sseeds = MMT{ForwardDiff.Partials{1,Float64}}(undef, nbatch, nmap)
    t1sseeds = MMT{ForwardDiff.Partials{1,Float64}}(undef, nbatch, nmap)
    varx = view(x, map)
    t1svarx = view(t1sx, :, map)
    VHP = typeof(host_t1sseeds)
    VP = typeof(t1sseeds)
    VD = typeof(t1sx)
    adj_t1sx = MMT{t1s{1}}(zeros(Float64, nbatch, 3 * nbus))
    adj_t1sF = A{t1s{1}}(zeros(Float64, n_cons))
    buffer = batch_tape(polar, func, nbatch, typeof(adj_t1sx))
    return AutoDiff.Hessian(
        func, host_t1sseeds, t1sseeds, x, t1sF, adj_t1sF, t1sx, adj_t1sx, map, varx, t1svarx, buffer,
    )
end

function batch_stack(polar::PolarForm{T, VI, VT, MT}, nbatch::Int) where {T, VI, VT, MT}
    nbus = get(polar, PS.NumberOfBuses())
    return AdjointPolar{MT}(
        MT(undef, nbatch, nbus),
        MT(undef, nbatch, nbus),
        MT(undef, nbatch, nbus),
        MT(undef, nbatch, nbus),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
    )
end

function batch_tape(
    polar::PolarForm, func, nbatch, MD,
)
    nnz = length(polar.topology.ybus_im.nzval)
    intermediate = (
        ∂edge_vm_fr = MD(undef, nbatch, nnz),
        ∂edge_va_fr = MD(undef, nbatch, nnz),
        ∂edge_vm_to = MD(undef, nbatch, nnz),
        ∂edge_va_to = MD(undef, nbatch, nnz),
    )
    return AutoDiff.TapeMemory(func, nothing, intermediate)
end

function batch_init_seed_hessian!(dest, tmp, v::AbstractArray, nmap)
    nbatch = size(dest, 1)
    @inbounds for i in 1:nmap
        for j in 1:nbatch
            dest[j, i] = ForwardDiff.Partials{1, Float64}(NTuple{1, Float64}(v[j, i]))
        end
    end
    return
end

function batch_init_seed_hessian!(dest, tmp, v::CUDA.CuArray, nmap)
    hostv = Array(v)
    @inbounds Threads.@threads for i in 1:nmap
        tmp[i] = ForwardDiff.Partials{1, Float64}(NTuple{1, Float64}(hostv[i]))
    end
    copyto!(dest, tmp)
    return
end

function batch_adj_hessian_prod!(
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
    nbatch = size(adj_t1sx, 1)
    # Move data
    copyto!(x, 1, buffer.vmag, 1, nbus)
    copyto!(x, nbus+1, buffer.vang, 1, nbus)
    copyto!(x, 2*nbus+1, buffer.pinj, 1, nbus)
    # Init dual variables
    #
    for i in 1:nbatch
        t1sx[i, :] .= H.x
    end

    adj_t1sx .= 0.0
    t1sF .= 0.0
    adj_t1sF .= λ
    # Seeding
    nmap = length(H.map)

    # Init seed
    batch_init_seed_hessian!(H.t1sseeds, H.host_t1sseeds, v, nmap)
    BatchAutoDiff.seed!(H.t1sseeds, H.varx, H.t1svarx)

    batch_adjoint!(
        polar, H.buffer,
        t1sF, adj_t1sF,
        view(t1sx, :, 1:nbus), view(adj_t1sx, :, 1:nbus),                   # vmag
        view(t1sx, :, nbus+1:2*nbus), view(adj_t1sx, :, nbus+1:2*nbus),     # vang
        view(t1sx, :, 2*nbus+1:3*nbus), view(adj_t1sx, :, 2*nbus+1:3*nbus), # pinj
    )

    BatchAutoDiff.getpartials_kernel!(hv, adj_t1sx, H.map)
    return nothing
end

