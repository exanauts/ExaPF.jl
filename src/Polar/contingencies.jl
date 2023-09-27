

abstract type AbstractContingency end

struct LineContingency <: AbstractContingency
    line_id::Int
end

struct GeneratorContingency <: AbstractContingency
    gen_id::Int
end

# Modify the definition of functions in presence of contingencies

function PowerFlowBalance(
    polar::AbstractPolarFormulation{T, VI, VT, MT},
    contingencies::Vector{LineContingency},
) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.device)
    k = nblocks(polar)

    # Check we have enough blocks to represent each contingency + base case
    @assert k == length(contingencies) + 1

    pf = polar.network
    ngen = pf.ngen
    nbus = pf.nbus
    gen = pf.gen2bus
    npq, npv = length(pf.pq), length(pf.pv)

    # Assemble matrices for loads
    Cd_tot = spdiagm(nbus, nbus, ones(nbus))               # Identity matrix
    Cdp = [Cd_tot[[pf.pv ; pf.pq], :]; spzeros(npq, nbus)] # active part
    Cdq = [spzeros(npq+npv, nbus) ; Cd_tot[pf.pq, :]]      # reactive part
    Cdp = _blockdiag(Cdp, k)
    Cdq = _blockdiag(Cdq, k)

    # Assemble matrices for generators
    Cg_tot = sparse(gen, 1:ngen, ones(ngen), nbus, ngen)
    Cg = -[Cg_tot[pf.pv, :] ; spzeros(2*npq, ngen)]
    Cg  = _blockdiag(Cg, k)

    # Assemble matrices for power flow
    ## Base case
    M_b = PS.get_basis_matrix(polar.network)
    M_ = [-M_b[[pf.pv; pf.pq; nbus .+ pf.pq], :]]
    ## Contingencies
    for contingency in contingencies
        M_c = PS.get_basis_matrix(polar.network; remove_line=contingency.line_id)
        push!(M_, -M_c[[pf.pv; pf.pq; nbus .+ pf.pq], :])
    end
    M = blockdiag(M_...)

    return PowerFlowBalance{VT, SMT}(M, Cg, Cdp, Cdq)
end

function PowerGenerationBounds(
    polar::AbstractPolarFormulation{T, VI, VT, MT},
    contingencies::Vector{LineContingency},
) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.device)
    k = nblocks(polar)

    # Check we have enough blocks to represent each contingency + base case
    @assert k == length(contingencies) + 1

    pf = polar.network
    nbus = pf.nbus
    ns = length(pf.ref) + length(pf.pv)

    # Assemble matrices for loads
    Cd_tot = spdiagm(nbus, nbus, ones(nbus))
    Cdp = [Cd_tot[pf.ref, :] ; spzeros(ns, nbus)]
    Cdq = [spzeros(length(pf.ref), nbus) ; Cd_tot[[pf.ref ; pf.pv], :]]
    Cdp = _blockdiag(Cdp, k)
    Cdq = _blockdiag(Cdq, k)

    # Assemble matrices for power flow
    ## Base case
    M_b = PS.get_basis_matrix(pf)
    M_ = [-M_b[[pf.ref; nbus .+ pf.ref; nbus .+ pf.pv], :]]
    ## Contingencies
    for contingency in contingencies
        M_c = PS.get_basis_matrix(pf; remove_line=contingency.line_id)
        push!(M_, -M_c[[pf.ref; nbus .+ pf.ref; nbus .+ pf.pv], :])
    end
    M = blockdiag(M_...)

    return PowerGenerationBounds{VT, SMT}(M, Cdp, Cdq)
end

function LineFlows(
    polar::AbstractPolarFormulation{T,VI,VT,MT},
    contingencies::Vector{LineContingency},
) where {T,VI,VT,MT}
    SMT = default_sparse_matrix(polar.device)
    nlines = get(polar, PS.NumberOfLines())
    ## Base case
    Lfp, Lfq, Ltp, Ltq = PS.get_line_flow_matrices(polar.network)
    Lfp_ = [Lfp]
    Lfq_ = [Lfq]
    Ltp_ = [Ltp]
    Ltq_ = [Ltq]
    ## Contingencies
    for contingency in contingencies
        Lfp_c, Lfq_c, Ltp_c, Ltq_c = PS.get_line_flow_matrices(polar.network; remove_line=contingency.line_id)
        push!(Lfp_, Lfp_c)
        push!(Lfq_, Lfq_c)
        push!(Ltp_, Ltp_c)
        push!(Ltq_, Ltq_c)
    end

    return LineFlows{VT,SMT}(
        nlines,
        blockdiag(Lfp_...),
        blockdiag(Lfq_...),
        blockdiag(Ltp_...),
        blockdiag(Ltq_...),
        polar.device,
    )
end

function PowerFlowRecourse(
    polar::PolarFormRecourse{T, VI, VT, MT},
    contingencies::Vector{LineContingency};
    epsilon=1e-2,
    alpha=nothing,
) where {T, VI, VT, MT}
    @assert polar.ncustoms > 0
    SMT = default_sparse_matrix(polar.device)
    k = nblocks(polar)
    @assert k == length(contingencies) + 1

    pf = polar.network
    ngen = pf.ngen
    nbus = pf.nbus
    gen = pf.gen2bus
    npv = length(pf.pv)
    npq = length(pf.pq)
    @assert npv + npq + 1 == nbus

    # Assemble matrices for loads
    Cd_tot = spdiagm(nbus, nbus, ones(nbus)) # Identity matrix
    Cdp = [Cd_tot[[pf.ref; pf.pv ; pf.pq], :]; spzeros(npq, nbus)]
    Cdq = [spzeros(nbus, nbus) ; Cd_tot[pf.pq, :]]
    Cdp = _blockdiag(Cdp, k)
    Cdq = _blockdiag(Cdq, k)

    # Assemble matrices for generators
    Cg_tot = sparse(gen, 1:ngen, ones(ngen), nbus, ngen)
    Cg = -[Cg_tot[[pf.ref; pf.pv], :] ; spzeros(2*npq, ngen)]
    Cg  = _blockdiag(Cg, k)

    # Assemble matrices for power flow
    ## Base case
    M_b = PS.get_basis_matrix(polar.network)
    M_ = [-M_b[[pf.ref; pf.pv; pf.pq; nbus .+ pf.pq], :]]
    ## Contingencies
    for contingency in contingencies
        M_c = PS.get_basis_matrix(polar.network; remove_line=contingency.line_id)
        push!(M_, -M_c[[pf.ref; pf.pv; pf.pq; nbus .+ pf.pq], :])
    end
    M = blockdiag(M_...)

    # Response ratio (by default dispatch recourse evenly)
    if isnothing(alpha)
        alpha = ones(ngen) ./ ngen
    end
    @assert length(alpha) == ngen
    # Bounds
    _pgmin, _pgmax = PS.bounds(polar.network, PS.Generators(), PS.ActivePower())
    pgmin = repeat(_pgmin, k)
    pgmax = repeat(_pgmax, k)

    return PowerFlowRecourse{VT, SMT}(M, Cg, Cdp, Cdq, pgmin, pgmax, alpha, epsilon)
end

function ReactivePowerBounds(
    polar::PolarFormRecourse{T, VI, VT, MT},
    contingencies::Vector{LineContingency},
) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.device)
    k = nblocks(polar)
    @assert k == length(contingencies) + 1

    pf = polar.network
    nbus = pf.nbus
    ns = length(pf.ref) + length(pf.pv)

    # Assemble matrices for loads
    Cd_tot = spdiagm(nbus, nbus, ones(nbus)) # Identity matrix
    Cdq = Cd_tot[[pf.ref ; pf.pv], :]
    Cdq = _blockdiag(Cdq, nblocks(polar))

    # Assemble matrices for power flow
    M_b = PS.get_basis_matrix(pf)
    M_ = [-M_b[[nbus .+ pf.ref; nbus .+ pf.pv], :]]
    for contingency in contingencies
        M_c = PS.get_basis_matrix(pf; remove_line=contingency.line_id)
        push!(M_, -M_c[[nbus .+ pf.ref; nbus .+ pf.pv], :])
    end
    M = blockdiag(M_...)

    return ReactivePowerBounds{VT, SMT}(M, Cdq)
end

