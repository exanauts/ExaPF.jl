
function jacobian!(
    jac::AutoDiff.AbstractJacobian, stack,
)
    # init
    AutoDiff.set_value!(jac, stack.input)
    jac.t1sF .= 0.0
    # forward pass
    jac.func(jac.t1sF, jac.stack)
    # extract partials
    AutoDiff.partials!(jac)
    return jac.J
end

function set_params!(jac::AutoDiff.AbstractJacobian, stack::AbstractNetworkStack)
    copyto!(jac.stack.params, stack.params)
end

struct Jacobian{Model, Func, Stack, VD, SMT, VI} <: AutoDiff.AbstractJacobian
    model::Model
    func::Func
    map::VI
    stack::Stack
    coloring::VI
    ncolors::Int
    t1sF::VD
    J::SMT
end

function Base.show(io::IO, jacobian::Jacobian)
    println(io, "A AutoDiff Jacobian for $(jacobian.func)")
    print(io, "Number of Jacobian colors: ", jacobian.ncolors)
end

Base.size(jac::Jacobian, n::Int) = size(jac.J, n)

# Coloring
function _jacobian_sparsity(polar::PolarForm, func::AutoDiff.AbstractExpression)
    nbus = get(polar, PS.NumberOfBuses())
    v = PS.voltage(polar.network) .+ 0.01 .* rand(ComplexF64, nbus)
    return matpower_jacobian(polar, func, v)
end

function _get_jacobian_colors(polar::PolarForm, func::AutoDiff.AbstractExpression, map::Vector{Int})
    # Sparsity pattern
    J = _jacobian_sparsity(polar, func)
    Jsub = J[:, map]
    # Coloring
    colors = AutoDiff.SparseDiffTools.matrix_colors(Jsub)
    return (Jsub, colors)
end

function Jacobian(
    polar::PolarForm{T, VI, VT, MT},
    func::AutoDiff.AbstractExpression,
    map::Vector{Int},
) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func)

    map_device = map |> VI

    J_host, coloring = _get_jacobian_colors(polar, func, map)
    ncolors = length(unique(coloring))

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    VD = A{t1s{ncolors}}

    J = J_host |> SMT

    # Structures
    stack = NetworkStack(nbus, ngen, nlines, VT, VD)
    init!(polar, stack)
    t1sF = zeros(Float64, n_cons) |> VD

    coloring = coloring |> VI

    jac = Jacobian(
        polar, func, map_device, stack, coloring, ncolors, t1sF, J,
    )

    # seed
    AutoDiff.seed_coloring!(jac, coloring)

    return jac
end
Jacobian(polar::PolarForm, func::AutoDiff.AbstractExpression, x::AbstractVariable) = Jacobian(polar, func, mapping(polar, x))

function Jacobian(
    polar::PolarForm{T, VI, VT, MT},
    func::AutoDiff.AbstractExpression,
    var::AbstractVariable,
    k::Int,
) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    # Generate mappings
    map = mapping(polar, var)
    blk_map = mapping(polar, var, k)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func) * k

    J_host, coloring = _get_jacobian_colors(polar, func, map)
    ncolors = length(unique(coloring))

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    VD = A{t1s{ncolors}}

    if AutoDiff.has_multiple_expressions(func)
        slices = AutoDiff.get_slices(func)
        shuf = [0; cumsum(slices)]
        jacs_shuf = [J_host[1+shuf[i]:shuf[i+1], :] for i in 1:length(shuf)-1]
        J = vcat([repeat(j, k) for j in jacs_shuf]...) |> SMT
    else
        J = repeat(J_host, k) |> SMT
    end
    coloring = repeat(coloring, k) |> VI

    # Structures
    stack = BlockNetworkStack(k, nbus, ngen, nlines, VT, VD)
    init!(polar, stack)
    t1sF = zeros(Float64, n_cons) |> VD

    map_device = blk_map |> VI

    jac = Jacobian(
        polar, func, map_device, stack, coloring, ncolors, t1sF, J,
    )

    # seed
    AutoDiff.seed_coloring!(jac, coloring)
    return jac
end



struct BlockJacobian{Model, Func, Stack, VD, SMT, VI} <: AutoDiff.AbstractJacobian
    model::Model
    func::Func
    map::VI
    stack::Stack
    coloring::VI
    ncolors::Int
    t1sF::VD
    J::SMT
    n::Int
    nblocks::Int
end

function BlockJacobian(
    polar::PolarForm{T, VI, VT, MT},
    func::AutoDiff.AbstractExpression,
    var::AbstractVariable,
    k::Int,
) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    # Generate mappings
    map = mapping(polar, var)
    blk_map = mapping(polar, var, k)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func) * k

    J_host, coloring = _get_jacobian_colors(polar, func, map)
    ncolors = length(unique(coloring))

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    VD = A{t1s{ncolors}}

    if AutoDiff.has_multiple_expressions(func)
        error("BlockJacobian does not support MultiExpressions.")
    end

    J = blockdiag([J_host for _ in 1:k]...) |> SMT
    coloring = repeat(coloring, k) |> VI

    # Structures
    stack = BlockNetworkStack(k, nbus, ngen, nlines, VT, VD)
    init!(polar, stack)
    t1sF = zeros(Float64, n_cons) |> VD

    map_device = blk_map |> VI

    jac = BlockJacobian(
        polar, func, map_device, stack, coloring, ncolors, t1sF, J, length(func), k,
    )

    # seed
    AutoDiff.seed_coloring!(jac, coloring)
    return jac
end

@kernel function _block_partials_csc_kernel!(J_colptr, J_rowval, J_nzval, duals, coloring, nzval)
    # CSC is column oriented: nmap is equal to number of columns
    i, k = @index(Global, NTuple)
    shift_nz = (k - 1) * nzval
    for j in J_colptr[i]:J_colptr[i+1]-1
        J_nzval[j + shift_nz] = duals[coloring[i]+1, J_rowval[j + shift_nz]]
    end
end

# Adapt partials extraction for block structure
function AutoDiff.partials!(jac::BlockJacobian)
    J = jac.J
    N = jac.ncolors
    T = eltype(J)
    duals = jac.t1sF
    device = jac.model.device
    coloring = jac.coloring
    n = length(duals)
    nzval = div(nnz(jac.J), jac.nblocks)

    duals_ = reshape(reinterpret(T, duals), N+1, n)

    ndrange = (jac.n, jac.nblocks)
    ev = _block_partials_csc_kernel!(device)(
        J.colptr, J.rowval, J.nzval, duals_, coloring, nzval;
        ndrange=ndrange, dependencies=Event(device),
    )
    wait(ev)
end


struct ArrowheadJacobian{Model, Func, Stack, VD, SMT, VI} <: AutoDiff.AbstractJacobian
    model::Model
    func::Func
    map::VI
    stack::Stack
    coloring::VI
    ncolors::Int
    t1sF::VD
    J::SMT
    nx::Int
    nu::Int
    nblocks::Int
    block_id::VI
end

function jacobian_arrowhead_sparsity(J, block_id, nx, nu, nblocks)
    i_jac, j_jac, _ = findnz(J)
    i_coo, j_coo = Int[], Int[]

    idk = 1 # start with first constraints
    for (i, j) in zip(i_jac, j_jac)
        k = block_id[i]
        ## / x
        if j <= nx
            push!(i_coo, i)
            push!(j_coo, j + (k-1) * nx)
        ## / u
        elseif j <= nx + nu
            push!(i_coo, i)
            push!(j_coo, j + (nblocks-1) * nx)
        end
    end
    return i_coo, j_coo
end

function ArrowheadJacobian(
    polar::PolarForm{T, VI, VT, MT},
    func::AutoDiff.AbstractExpression,
    k::Int,
) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    nx = number(polar, State())
    nu = number(polar, Control())
    # Generate mappings
    map = mapping(polar, AllVariables())
    blk_map = mapping(polar, AllVariables(), k)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func) * k

    J_host, coloring = _get_jacobian_colors(polar, func, map)
    ncolors = length(unique(coloring))

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    VD = A{t1s{ncolors}}

    if AutoDiff.has_multiple_expressions(func)
        slices = AutoDiff.get_slices(func)
        cumslices = cumsum(slices)
        shuf = [0; cumslices]
        cumslices .*= k
        jacs_shuf = [J_host[1+shuf[i]:shuf[i+1], :] for i in 1:length(shuf)-1]
        J_blk = vcat([repeat(j, k) for j in jacs_shuf]...) |> SMT
        block_id = Int[]
        idk = 1
        for i in 1:n_cons
            if i > cumslices[idk]
                idk += 1
            end
            shift = (idk > 1) ? cumslices[idk-1] : 0
            # Which scenario?
            b = div(i - shift - 1, slices[idk]) + 1
            @assert 1 <= b <= k
            push!(block_id, b)
        end
    else
        J_blk = repeat(J_host, k) |> SMT
        m = length(func)
        block_id = vcat([fill(_id, m) for _id in 1:k]...)
    end

    i_coo, j_coo = jacobian_arrowhead_sparsity(J_blk, block_id, nx, nu, k)
    J = sparse(i_coo, j_coo, ones(length(i_coo))) |> SMT

    coloring = repeat(coloring, k) |> VI

    # Structures
    stack = BlockNetworkStack(k, nbus, ngen, nlines, VT, VD)
    init!(polar, stack)
    t1sF = zeros(Float64, n_cons) |> VD

    map_device = blk_map |> VI

    jac = ArrowheadJacobian(
        polar, func, map_device, stack, coloring, ncolors, t1sF, J, nx, nu, k, block_id,
    )

    # seed
    AutoDiff.seed_coloring!(jac, coloring)
    return jac
end

@kernel function _arrowhead_partials_csc_kernel!(J_colptr, J_rowval, J_nzval, duals, coloring, nx, nu, nblock)
    j = @index(Global, Linear)
    if j <= nblock * nx
        jk = (j-1) % nx + 1
    else
        jk = j - nblock * nx + nx
    end

    for c in J_colptr[j]:J_colptr[j+1]-1
        i = J_rowval[c]
        J_nzval[c] = duals[coloring[jk]+1, i]
    end
end

# Adapt partials extraction for block structure
function AutoDiff.partials!(jac::ArrowheadJacobian)
    J = jac.J
    N = jac.ncolors
    T = eltype(J)
    duals = jac.t1sF
    device = jac.model.device
    coloring = jac.coloring
    n = length(duals)
    duals_ = reshape(reinterpret(T, duals), N+1, n)

    ndrange = (size(J, 2), )
    ev = _arrowhead_partials_csc_kernel!(device)(
        J.colptr, J.rowval, J.nzval, duals_, coloring, jac.nx, jac.nu, jac.nblocks;
        ndrange=ndrange, dependencies=Event(device),
    )
    wait(ev)
end

