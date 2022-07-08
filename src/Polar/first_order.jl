
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
function _jacobian_sparsity(polar::AbstractPolarFormulation, func::AutoDiff.AbstractExpression)
    nbus = get(polar, PS.NumberOfBuses())
    v = PS.voltage(polar.network) .+ 0.01 .* rand(ComplexF64, nbus)
    return matpower_jacobian(polar, func, v)
end

function _get_jacobian_colors(polar::AbstractPolarFormulation, func::AutoDiff.AbstractExpression, map::Vector{Int})
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
    polar::BlockPolarForm{T, VI, VT, MT},
    func::AutoDiff.AbstractExpression,
    X::AbstractVariable,
) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    k = nblocks(polar)
    nx = number(polar, State())
    nu = number(polar, Control())
    if isa(X, Control)
        nu = nu
        nx = 0
    elseif isa(X, State)
        nu = 0
        nx = nx
    elseif isa(X, AllVariables)
        nu = nu
        nx = nx
    end
    ntot = nx * k + nu
    # Generate mappings
    map = mapping(polar, X)
    blk_map = mapping(polar, X, k)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func)

    J_host, coloring = _get_jacobian_colors(polar, func, map)
    ncolors = length(unique(coloring))

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    VD = A{t1s{ncolors}}

    if AutoDiff.has_multiple_expressions(func)
        slices = AutoDiff.get_slices(func)
        cumslices = cumsum(slices)
        shuf = convert(Vector{Int}, [0; cumslices] ./ k)
        # cumslices .*= k
        jacs_shuf = [J_host[1+shuf[i]:shuf[i+1], :] for i in 1:length(shuf)-1]
        i_jac = Int[]
        j_jac = Int[]
        shift = 0
        for J in jacs_shuf
            i, j, _ = findnz(J)
            for kk in 1:k
                append!(i_jac, i .+ shift)
                append!(j_jac, j)
                shift += size(J, 1)
            end
        end
        nnzJ = length(i_jac)
        J_blk = sparse(i_jac, j_jac, ones(nnzJ))
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
        J_blk = repeat(J_host, k)
        m = div(length(func), k)
        block_id = vcat([fill(_id, m) for _id in 1:k]...)
    end

    i_coo, j_coo = jacobian_arrowhead_sparsity(J_blk, block_id, nx, nu, k)
    J = sparse(i_coo, j_coo, ones(length(i_coo)), n_cons, ntot) |> SMT

    coloring = repeat(coloring, k) |> VI

    # Structures
    stack = BlockNetworkStack(k, nbus, ngen, nlines, VT, VD)
    init!(polar, stack)
    t1sF = zeros(Float64, n_cons) |> VD

    map_device = blk_map |> VI
    block_id = block_id |> VI

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

@kernel function _arrowhead_partials_csr_kernel!(J_rowptr, J_colval, J_nzval, duals, coloring, nx, nu, nblock)
    i = @index(Global, Linear)

    for c in J_rowptr[i]:J_rowptr[i+1]-1
        j = J_colval[c]
        if j <= nblock * nx
            jk = (j-1) % nx + 1
        else
            jk = j - nblock * nx + nx
        end
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

    if isa(J, SparseMatrixCSC)
        ndrange = (size(J, 2), )
        ev = _arrowhead_partials_csc_kernel!(device)(
            J.colptr, J.rowval, J.nzval, duals_, coloring, jac.nx, jac.nu, jac.nblocks;
            ndrange=ndrange, dependencies=Event(device),
        )
    elseif isa(J, CuSparseMatrixCSR)
        ndrange = (size(J, 1), )
        ev = _arrowhead_partials_csr_kernel!(device)(
            J.rowPtr, J.colVal, J.nzVal, duals_, coloring, jac.nx, jac.nu, jac.nblocks;
            ndrange=ndrange, dependencies=Event(device),
        )
    end
    wait(ev)
end

