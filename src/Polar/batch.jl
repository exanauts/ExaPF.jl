
# Adjoint
function batch_adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    cons, ∂cons,
    vm, ∂vm,
    va, ∂va,
    pnet, ∂pnet,
    pload, qload,
) where {F<:typeof(power_balance), S, I}
    nbus = get(polar, PS.NumberOfBuses())
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    fill!(pbm.intermediate.∂edge_vm_fr , 0.0)
    fill!(pbm.intermediate.∂edge_vm_to , 0.0)
    fill!(pbm.intermediate.∂edge_va_fr , 0.0)
    fill!(pbm.intermediate.∂edge_va_to , 0.0)

    adj_residual_polar!(
        cons, ∂cons,
        vm, ∂vm,
        va, ∂va,
        ybus_re, ybus_im, polar.topology.sortperm,
        pnet, ∂pnet, pload, qload,
        pbm.intermediate.∂edge_vm_fr,
        pbm.intermediate.∂edge_vm_to,
        pbm.intermediate.∂edge_va_fr,
        pbm.intermediate.∂edge_va_to,
        pv, pq, nbus, polar.device
    )
end

function BatchHessian(polar::PolarForm{T, VI, VT, MT}, func, nbatch) where {T, VI, VT, MT}
    @assert is_constraint(func)

    if isa(polar.device, CPU)
        A = Vector
        MMT = Matrix
    elseif isa(polar.device, GPU)
        A = CUDA.CuVector
        MMT = CUDA.CuMatrix
    end

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    n_cons = size_constraint(polar, func)

    map = VI(polar.hessianstructure.map)
    nmap = length(map)

    x = VT(zeros(Float64, 3*nbus))

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    t1sx = MMT{t1s{1}}(zeros(Float64, 3*nbus, nbatch))
    t1sF = MMT{t1s{1}}(zeros(Float64, n_cons, nbatch))
    host_t1sseeds = MMT{ForwardDiff.Partials{1,Float64}}(undef, nmap, nbatch)
    t1sseeds = MMT{ForwardDiff.Partials{1,Float64}}(undef, nmap, nbatch)
    varx = view(x, map)
    t1svarx = view(t1sx, map, :)
    VHP = typeof(host_t1sseeds)
    VP = typeof(t1sseeds)
    VD = typeof(t1sx)
    adj_t1sx = MMT{t1s{1}}(zeros(Float64, 3 * nbus, nbatch))
    adj_t1sF = A{t1s{1}}(zeros(Float64, n_cons))
    buffer = batch_tape(polar, func, nbatch, typeof(adj_t1sx))
    return AutoDiff.Hessian(
        func, host_t1sseeds, t1sseeds, x, t1sF, adj_t1sF, t1sx, adj_t1sx, map, varx, t1svarx, buffer,
    )
end

# Batch buffers
function batch_buffer(polar::PolarForm{T, VI, VT, MT}, nbatch::Int) where {T, VI, VT, MT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    nstates = get(polar, NumberOfState())
    gen2bus = polar.indexing.index_generators
    h_gen2bus = polar.indexing.index_generators |> Array
    buffer =  PolarNetworkState{VI,MT}(
        MT(undef, nbus, nbatch),
        MT(undef, nbus, nbatch),
        MT(undef, nbus, nbatch),
        MT(undef, nbus, nbatch),
        MT(undef, ngen, nbatch),
        MT(undef, ngen, nbatch),
        MT(undef, nbus, nbatch),
        MT(undef, nbus, nbatch),
        MT(undef, nstates, nbatch),
        MT(undef, nstates, nbatch),
        gen2bus,
    )

    # Init
    pbus = zeros(nbus)
    qbus = zeros(nbus)
    vmag = abs.(polar.network.vbus)
    vang = angle.(polar.network.vbus)
    pd = get(polar.network, PS.ActiveLoad())
    qd = get(polar.network, PS.ReactiveLoad())
    pg = get(polar.network, PS.ActivePower())
    qg = get(polar.network, PS.ReactivePower())
    pbus[h_gen2bus] .= pg
    qbus[h_gen2bus] .= qg

    for i in 1:nbatch
        copyto!(buffer.vmag, nbus * (i-1) + 1, vmag, 1, nbus)
        copyto!(buffer.vang, nbus * (i-1) + 1, vang, 1, nbus)
        copyto!(buffer.pnet, nbus * (i-1) + 1, pbus, 1, nbus)
        copyto!(buffer.qnet, nbus * (i-1) + 1, qbus, 1, nbus)
        copyto!(buffer.pgen,   ngen * (i-1) + 1,   pg, 1, ngen)
        copyto!(buffer.qgen,   ngen * (i-1) + 1,   qg, 1, ngen)
        copyto!(buffer.pload,  nbus * (i-1) + 1,   pd, 1, nbus)
        copyto!(buffer.qload,  nbus * (i-1) + 1,   qd, 1, nbus)
    end

    return buffer
end

function batch_stack(polar::PolarForm{T, VI, VT, MT}, nbatch::Int) where {T, VI, VT, MT}
    nbus = get(polar, PS.NumberOfBuses())
    return AdjointPolar{MT}(
        MT(undef, nbus, nbatch),
        MT(undef, nbus, nbatch),
        MT(undef, nbus, nbatch),
        MT(undef, nbus, nbatch),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
    )
end

function batch_tape(
    polar::PolarForm, func, nbatch, MD,
)
    nnz = length(polar.topology.ybus_im.nzval)
    intermediate = (
        ∂edge_vm_fr = MD(undef, nnz, nbatch),
        ∂edge_va_fr = MD(undef, nnz, nbatch),
        ∂edge_vm_to = MD(undef, nnz, nbatch),
        ∂edge_va_to = MD(undef, nnz, nbatch),
    )
    return AutoDiff.TapeMemory(func, nothing, intermediate)
end

function update!(polar::PolarForm, H::AutoDiff.Hessian, buffer)
    x = H.x
    t1sx = H.t1sx
    nbatch = size(t1sx, 2)
    nbus = get(polar, PS.NumberOfBuses())

    # Move data
    copyto!(x, 1, buffer.vmag, 1, nbus)
    copyto!(x, nbus+1, buffer.vang, 1, nbus)
    copyto!(x, 2*nbus+1, buffer.pnet, 1, nbus)

    @inbounds for i in 1:nbatch
        t1sx[:, i] .= H.x
    end
end

function batch_adj_hessian_prod!(
    polar, H::AutoDiff.Hessian, hv, buffer, λ, v,
)
    @assert length(hv) == length(v)
    device = polar.device
    nbus = get(polar, PS.NumberOfBuses())
    x = H.x
    ntgt = length(v)
    t1sx = H.t1sx
    adj_t1sx = H.∂t1sx
    t1sF = H.t1sF
    adj_t1sF = H.∂t1sF
    nbatch = size(adj_t1sx, 2)
    # Init dual variables
    adj_t1sx .= 0.0
    t1sF .= 0.0
    adj_t1sF .= λ
    # Seeding
    nmap = length(H.map)

    # Init seed
    AutoDiff.batch_init_seed_hessian!(H.t1sseeds, H.host_t1sseeds, v, nmap, device)
    AutoDiff.batch_seed_hessian!(H.t1sseeds, H.varx, H.t1svarx, device)

    batch_adjoint!(
        polar, H.buffer,
        t1sF, adj_t1sF,
        view(t1sx, 1:nbus, :), view(adj_t1sx, 1:nbus, :),                   # vmag
        view(t1sx, nbus+1:2*nbus, :), view(adj_t1sx, nbus+1:2*nbus, :),     # vang
        view(t1sx, 2*nbus+1:3*nbus, :), view(adj_t1sx, 2*nbus+1:3*nbus, :), # pinj
        buffer.pload, buffer.qload,
    )

    AutoDiff.batch_partials_hessian!(hv, adj_t1sx, H.map, device)
    return nothing
end

function BatchJacobian(
    polar::PolarForm{T, VI, VT, MT}, func, variable, nbatch,
) where {T, VI, VT, MT}
    @assert is_constraint(func)
    device = polar.device

    if isa(device, CPU)
        SMT = SparseMatrixCSC{Float64,Int}
        A = Array
    elseif isa(device, GPU)
        SMT = CUSPARSE.CuSparseMatrixCSR{Float64}
        A = CUDA.CuArray
    end

    # Tensor type
    TT = A{T, 3}

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

    nx = 2 * nbus
    x = MT(zeros(Float64, nx, nbatch))
    m = size(J, 1)

    # Move Jacobian to the GPU
    if isa(polar.device, CPU)
        Js = SMT[J for i in 1:nbatch]
    else
        Js = BatchCuSparseMatrixCSR(J, nbatch)
    end

    # Seedings
    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    t1sx = A{t1s{ncolor}}(x)
    t1sF = A{t1s{ncolor}}(zeros(Float64, m, nbatch))
    t1sseeds = AutoDiff.init_seed(coloring, ncolor, nmap)

    # Move the seeds over to the device, if necessary
    gput1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(t1sseeds)
    compressedJ = TT(zeros(Float64, ncolor, m, nbatch))

    # Views
    varx = view(x, map, :)
    t1svarx = view(t1sx, map, :)

    return AutoDiff.Jacobian{typeof(func), VI, MT, TT, typeof(Js), typeof(gput1sseeds), typeof(t1sx), typeof(varx), typeof(t1svarx)}(
        func, variable, Js, compressedJ, coloring,
        gput1sseeds, t1sF, x, t1sx, map, varx, t1svarx
    )
end

function batch_jacobian!(polar::PolarForm, jac::AutoDiff.Jacobian, buffer)
    device = polar.device
    nbus = get(polar, PS.NumberOfBuses())
    type = jac.var
    nbatch = size(jac.x, 2)
    if isa(type, State)
        for i in 1:nbatch
            f = (i-1) * nbus
            copyto!(jac.x, 1 + 2 * f, buffer.vmag, 1 + f, nbus)
            copyto!(jac.x, nbus + 2 * f + 1, buffer.vang, 1 + f, nbus)
        end
        jac.t1sx .= jac.x
        jac.t1sF .= 0.0
    elseif isa(type, Control)
        copyto!(jac.x, 1, buffer.vmag, 1, nbus)
        copyto!(jac.x, nbus+1, buffer.pnet, 1, nbus)
        jac.t1sx .= jac.x
        jac.t1sF .= 0.0
    end

    AutoDiff.batch_seed_jacobian!(jac.t1sseeds, jac.varx, jac.t1svarx, device)

    if isa(type, State)
        jac.func(
            polar,
            jac.t1sF,
            view(jac.t1sx, 1:nbus, :),
            view(jac.t1sx, nbus+1:2*nbus, :),
            buffer.pnet, buffer.qnet,
            buffer.pload, buffer.qload,
        )
    elseif isa(type, Control)
        jac.func(
            polar,
            jac.t1sF,
            view(jac.t1sx, 1:nbus, :),
            buffer.vang,
            view(jac.t1sx, nbus+1:2*nbus, :), buffer.qnet,
            buffer.pload, buffer.qload,
        )
    end

    AutoDiff.batch_partials_jacobian!(jac.compressedJ, jac.t1sF, device)
    AutoDiff.batch_uncompress!(jac.J, jac.compressedJ, jac.coloring, device)
    return jac.J
end

