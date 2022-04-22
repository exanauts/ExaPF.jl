function set_params!(hess::AutoDiff.AbstractFullHessian, stack::AbstractNetworkStack)
    copyto!(hess.stack.params, stack.params)
    copyto!(hess.∂stack.params, stack.params)
end

function hessian!(
    H::AutoDiff.AbstractFullHessian, stack, λ,
)
    # init
    AutoDiff.set_value!(H, stack.input)
    empty!(H.∂stack)
    H.∂t1sF .= λ
    # forward pass
    H.func(H.t1sF, H.stack)
    # forward-over-reverse pass
    adjoint!(H.func, H.∂stack, H.stack, H.∂t1sF)
    # extract partials
    AutoDiff.partials!(H)
    return H.H
end


struct HessianProd{Model, Func, VT, VD, VI, Buff} <: AutoDiff.AbstractHessianProd
    model::Model
    func::Func
    map::VI
    stack::NetworkStack{VT, VD}
    ∂stack::NetworkStack{VT, VD}
    t1sF::VD
    ∂t1sF::VD
    buffer::Buff
end

function HessianProd(polar::PolarForm{T, VI, VT, MT}, func::AutoDiff.AbstractExpression, map::Vector{Int}) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func)

    nmap = length(map)
    map_device = map |> VI

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    VD = A{t1s{1}}

    stack = NetworkStack(nbus, ngen, nlines, VT, VD)
    init!(polar, stack)

    ∂stack = NetworkStack(nbus, ngen, nlines, VT, VD)

    t1sF = zeros(Float64, n_cons) |> VD
    adj_t1sF = similar(t1sF)

    intermediate = nothing
    return HessianProd(
        polar, func, map_device, stack, ∂stack, t1sF, adj_t1sF,
        intermediate,
    )
end

function hprod!(
    H::HessianProd, hv, stack, λ, v,
)
    @assert length(hv) == length(v)

    # Init dual variables
    H.stack.input .= stack.input
    empty!(H.∂stack)
    H.∂t1sF .= λ

    # Seeding
    AutoDiff.seed!(H, v)
    # Forward
    H.func(H.t1sF, H.stack)
    # Forward-over-Reverse
    adjoint!(H.func, H.∂stack, H.stack, H.∂t1sF)

    AutoDiff.getpartials_kernel!(hv, H)
    return
end

function _hessian_sparsity(polar::PolarForm, func)
    m = length(func)
    nbus = get(polar, PS.NumberOfBuses())
    Vre = Float64[i for i in 1:nbus]
    Vim = Float64[i for i in nbus+1:2*nbus]
    V = Vre .+ im .* Vim
    y = rand(m)
    return matpower_hessian(polar, func, V, y)
end

struct FullHessian{Model, Func, Stack, VD, SMT, VI} <: AutoDiff.AbstractFullHessian
    model::Model
    func::Func
    map::VI
    stack::Stack
    ∂stack::Stack
    coloring::VI
    ncolors::Int
    t1sF::VD
    ∂t1sF::VD
    H::SMT
end

function _get_hessian_colors(polar::PolarForm, func::AutoDiff.AbstractExpression, map::Vector{Int})
    H = _hessian_sparsity(polar, func)::SparseMatrixCSC
    Hsub = H[map, map] # reorder
    colors = AutoDiff.SparseDiffTools.matrix_colors(Hsub)
    return (Hsub, colors)
end

function FullHessian(polar::PolarForm{T, VI, VT, MT}, func::AutoDiff.AbstractExpression, map::Vector{Int}) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func)

    nmap = length(map)
    map_device = map |> VI

    H_host, coloring = _get_hessian_colors(polar, func, map)
    ncolors = length(unique(coloring))
    VD = A{ForwardDiff.Dual{Nothing, Float64, ncolors}}

    H = H_host |> SMT

    # Structures
    stack = NetworkStack(nbus, ngen, nlines, VT, VD)
    init!(polar, stack)

    ∂stack = NetworkStack(nbus, ngen, nlines, VT, VD)
    t1sF = zeros(Float64, n_cons) |> VD
    adj_t1sF = similar(t1sF)

    coloring = coloring |> VI

    hess = FullHessian(
        polar, func, map_device, stack, ∂stack, coloring, ncolors, t1sF, adj_t1sF,
        H,
    )

    # seed
    AutoDiff.seed_coloring!(hess, coloring)

    return hess
end

struct ArrowheadHessian{Model, Func, Stack, VD, SMT, VI} <: AutoDiff.AbstractFullHessian
    model::Model
    func::Func
    map::VI
    stack::Stack
    ∂stack::Stack
    coloring::VI
    ncolors::Int
    t1sF::VD
    ∂t1sF::VD
    H::SMT
    nx::Int
    nu::Int
    nblocks::Int
    vartype::VI
end

function hessian_arrowhead_sparsity(H, nx, nu, nscen)
    i_hess, j_hess, _ = findnz(H)
    nnzh = length(i_hess)
    i_coo, j_coo = Int[], Int[]
    idk = 1
    for (i, j) in zip(i_hess, j_hess)
        # Get current scenario
        k = div(i - 1, nx+nu) + 1
        @assert 1 <= k <= nscen
        ik = (i-1) % (nx + nu) + 1
        jk = j
        ## xx
        if (ik <= nx) && (jk <= nx)
            push!(i_coo, ik + (k-1) * nx)
            push!(j_coo, jk + (k-1) * nx)
        ## xu
        elseif (nx < ik <= nx+nu) && (jk <= nx)
            push!(i_coo, ik + (nscen-1) * nx)
            push!(j_coo, jk + (k-1) * nx)
        ## ux
        elseif (ik <= nx) && (nx < jk <= nx+nu)
            push!(i_coo, ik + (k-1) * nx)
            push!(j_coo, jk + (nscen-1) * nx)
        ## uu
        elseif (nx < ik <= nx+nu) && (nx < jk <= nx+nu)
            push!(i_coo, ik + (nscen-1) * nx)
            push!(j_coo, jk + (nscen-1) * nx)
        end
        idk += 1
    end
    return i_coo, j_coo
end

function ArrowheadHessian(
    polar::PolarForm{T, VI, VT, MT},
    func::AutoDiff.AbstractExpression,
    X::AbstractVariable,
    k::Int,
) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)
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

    map = mapping(polar, X)
    blk_map = mapping(polar, X, k)

    varid = Int[]
    for i in 1:k
        for j in 1:nx
            push!(varid, 0)
        end
        for j in 1:nu
            push!(varid, 1)
        end
    end

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func) * k

    nmap = length(map)

    H_host, coloring = _get_hessian_colors(polar, func, map)
    ncolors = length(unique(coloring))
    VD = A{ForwardDiff.Dual{Nothing, Float64, ncolors}}

    i_hess, j_hess = Int[], Int[]
    i, j, _ = findnz(H_host)
    shift = 0
    for kk in 1:k
        append!(i_hess, i .+ shift)
        append!(j_hess, j)
        shift += size(H_host, 1)
    end

    nnzh = length(i_hess)
    H_blk = sparse(i_hess, j_hess, ones(nnzh))
    i_coo, j_coo = hessian_arrowhead_sparsity(H_blk, nx, nu, k)
    H = sparse(i_coo, j_coo, ones(length(i_coo)), ntot, ntot) |> SMT

    # Structures
    stack = BlockNetworkStack(k, nbus, ngen, nlines, VT, VD)
    init!(polar, stack)

    ∂stack = BlockNetworkStack(k, nbus, ngen, nlines, VT, VD)
    t1sF = zeros(Float64, n_cons) |> VD
    adj_t1sF = similar(t1sF)

    coloring = repeat(coloring, k) |> VI

    map_device = blk_map |> VI
    varid = varid |> VI

    hess = ArrowheadHessian(
        polar, func, map_device, stack, ∂stack,
        coloring, ncolors, t1sF, adj_t1sF, H,
        nx, nu, k, varid,
    )
    # seed
    AutoDiff.seed_coloring!(hess, coloring)

    return hess
end

@kernel function _arrowhead_hess_partials_csc_kernel!(
    J_colptr, J_rowval, J_nzval, duals, map, coloring, vartype, nblocks, nx, nu,
)
    j = @index(Global, Linear)

    shift = nblocks * nx

    for c in J_colptr[j]:J_colptr[j+1]-1
        i = J_rowval[c]
        # xx
        if (1 <= i <= shift) && (1 <= j <= shift)
            k = div(i-1, nx) + 1
            ik = (i-1) % nx + 1 + (k-1) * (nx+nu)
            jk = (j-1) % nx + 1
            J_nzval[c] = duals[coloring[jk]+1, map[ik]]
        # xu
        elseif (1 <= i <= shift) && (shift + 1 <= j <= shift + nu)
            k = div(i-1, nx) + 1
            ik = (i-1) % nx + 1 + (k-1) * (nx+nu)
            jk = j - shift + nx
            J_nzval[c] = duals[coloring[jk]+1, map[ik]]
        # ux
        elseif (shift + 1 <= i <= shift + nu) && (1 <= j <= shift)
            k = div(j-1, nx) + 1
            ik = i - shift + (k-1) * (nx+nu) + nx
            jk = (j-1) % nx + 1
            J_nzval[c] = duals[coloring[jk]+1, map[ik]]
        # uu
        elseif (shift < i <= shift + nu) && (shift < j <= shift + nu)
            J_nzval[c] = 0.0
            ik = i - shift + nx
            jk = j - shift + nx
            for k in 1:nblocks
                ℓ = (k-1) * (nx+nu) + ik
                @assert vartype[ℓ] == 1
                J_nzval[c] += duals[coloring[jk]+1, map[ℓ]]
            end
        end
    end
end

@kernel function _arrowhead_hess_partials_csr_kernel!(
    J_rowptr, J_colval, J_nzval, duals, map, coloring, vartype, nblocks, nx, nu,
)
    i = @index(Global, Linear)

    shift = nblocks * nx

    for c in J_rowptr[i]:J_rowptr[i+1]-1
        j = J_colval[c]
        # xx
        if (1 <= i <= shift) && (1 <= j <= shift)
            k = div(i-1, nx) + 1
            ik = (i-1) % nx + 1 + (k-1) * (nx+nu)
            jk = (j-1) % nx + 1
            J_nzval[c] = duals[coloring[jk]+1, map[ik]]
        # xu
        elseif (1 <= i <= shift) && (shift + 1 <= j <= shift + nu)
            k = div(i-1, nx) + 1
            ik = (i-1) % nx + 1 + (k-1) * (nx+nu)
            jk = j - shift + nx
            J_nzval[c] = duals[coloring[jk]+1, map[ik]]
        # ux
        elseif (shift + 1 <= i <= shift + nu) && (1 <= j <= shift)
            k = div(j-1, nx) + 1
            ik = i - shift + (k-1) * (nx+nu) + nx
            jk = (j-1) % nx + 1
            J_nzval[c] = duals[coloring[jk]+1, map[ik]]
        # uu
        elseif (shift < i <= shift + nu) && (shift < j <= shift + nu)
            J_nzval[c] = 0.0
            ik = i - shift + nx
            jk = j - shift + nx
            for k in 1:nblocks
                ℓ = (k-1) * (nx+nu) + ik
                @assert vartype[ℓ] == 1
                J_nzval[c] += duals[coloring[jk]+1, map[ℓ]]
            end
        end
    end
end

function AutoDiff.partials!(hess::ArrowheadHessian)
    H = hess.H
    N = hess.ncolors
    T = eltype(H)
    duals = hess.∂stack.input
    device = hess.model.device
    coloring = hess.coloring
    n = length(duals)
    duals_ = reshape(reinterpret(T, duals), N+1, n)

    if isa(H, SparseMatrixCSC)
        ndrange = (size(H, 2), )
        ev = _arrowhead_hess_partials_csc_kernel!(device)(
            H.colptr, H.rowval, H.nzval, duals_, hess.map, coloring, hess.vartype, hess.nblocks, hess.nx, hess.nu;
            ndrange=ndrange, dependencies=Event(device),
        )
    elseif isa(H, CuSparseMatrixCSR)
        ndrange = (size(H, 1), )
        ev = _arrowhead_hess_partials_csr_kernel!(device)(
            H.rowPtr, H.colVal, H.nzVal, duals_, hess.map, coloring, hess.vartype, hess.nblocks, hess.nx, hess.nu;
            ndrange=ndrange, dependencies=Event(device),
        )
    end
    wait(ev)
end

