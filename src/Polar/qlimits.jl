"""
    Q Limit Enforcement for Power Flow

This module implements reactive power limit enforcement using the PandaPower approach:
1. Run standard power flow (no limits)
2. Check for Q limit violations at PV buses
3. Convert violated PV buses to PQ buses (fix Q at the limit)
4. Re-run power flow with modified bus types
5. Repeat until no violations or max iterations reached
"""

#=============================================================================
    Data Structures
=============================================================================#

"""
    QLimitStatus

Status of a generator that violated its reactive power limit.

# Fields
- `gen_idx::Int`: Generator index
- `bus_idx::Int`: Bus index where the generator is connected
- `q_limit::Float64`: The Q limit value that was hit (Qmin or Qmax) in per-unit
- `limit_type::Symbol`: Either `:upper` (Qmax) or `:lower` (Qmin)
"""
struct QLimitStatus
    gen_idx::Int
    bus_idx::Int
    q_limit::Float64
    limit_type::Symbol
end

function Base.show(io::IO, s::QLimitStatus)
    print(io, "QLimitStatus(gen=$(s.gen_idx), bus=$(s.bus_idx), ",
          "Q=$(round(s.q_limit, digits=4)), type=$(s.limit_type))")
end

"""
    QLimitEnforcementResult

Result of Q limit enforcement during power flow.

# Fields
- `converged::Bool`: Whether the Q-limited power flow converged
- `n_outer_iterations::Int`: Number of outer loop iterations (bus type modifications)
- `n_total_pf_iterations::Int`: Total Newton-Raphson iterations across all solves
- `violated_generators::Vector{QLimitStatus}`: Generators that hit their Q limits
- `final_q_values::Vector{Float64}`: Final reactive power output for all generators
"""
struct QLimitEnforcementResult
    converged::Bool
    n_outer_iterations::Int
    n_total_pf_iterations::Int
    violated_generators::Vector{QLimitStatus}
    final_q_values::Vector{Float64}
end

function Base.show(io::IO, r::QLimitEnforcementResult)
    print(io, "QLimitEnforcementResult(converged=$(r.converged), ",
          "outer_iters=$(r.n_outer_iterations), ",
          "total_pf_iters=$(r.n_total_pf_iterations), ",
          "n_violations=$(length(r.violated_generators)))")
end

"""
    BatchedQLimitResult

Result of Q limit enforcement for batched (multi-scenario) power flow.

# Fields
- `converged::Vector{Bool}`: Per-scenario convergence status
- `results::Vector{QLimitEnforcementResult}`: Per-scenario Q limit enforcement results
"""
struct BatchedQLimitResult
    converged::Vector{Bool}
    results::Vector{QLimitEnforcementResult}
end

function Base.show(io::IO, r::BatchedQLimitResult)
    n_converged = sum(r.converged)
    n_total = length(r.converged)
    print(io, "BatchedQLimitResult(converged=$(n_converged)/$(n_total) scenarios)")
end

#=============================================================================
    Core Functions
=============================================================================#

"""
    compute_generator_reactive_power(polar::AbstractPolarFormulation, stack::NetworkStack)

Compute the reactive power output for each generator after power flow solution.

# Arguments
- `polar`: The polar formulation
- `stack`: The network stack containing the solution

# Returns
- `Vector{Float64}`: Reactive power output for each generator in per-unit
"""
function compute_generator_reactive_power(
    polar::AbstractPolarFormulation,
    stack::AbstractNetworkStack
)
    pf = polar.network
    nbus = pf.nbus
    ngen = pf.ngen

    # Get voltage solution (transfer to CPU if on GPU)
    vmag = Array(stack.vmag)
    vang = Array(stack.vang)
    V = vmag .* exp.(im .* vang)

    # Compute Q injection at all buses: Q = imag(V * conj(Y * V))
    Ybus = pf.Ybus
    S_bus = V .* conj.(Ybus * V)
    Q_bus = imag.(S_bus)

    # Get reactive load at each bus (transfer to CPU if on GPU)
    qload = Array(stack.qload)

    # Q_gen at bus = Q_injection + Q_load (power flow sign convention)
    # Q_injection is negative of what flows into the network
    Q_gen_at_bus = Q_bus .+ qload

    # Map bus reactive power to generators
    gen2bus = pf.gen2bus
    q_min, q_max = PS.bounds(pf, PS.Generators(), PS.ReactivePower())

    qgen = zeros(Float64, ngen)

    # Group generators by bus
    for b in unique(gen2bus)
        gens_at_bus = findall(isequal(b), gen2bus)
        if length(gens_at_bus) == 1
            qgen[gens_at_bus[1]] = Q_gen_at_bus[b]
        else
            # Multiple generators at bus: distribute proportionally based on Q range
            q_ranges = [q_max[g] - q_min[g] for g in gens_at_bus]
            total_range = sum(q_ranges)
            if total_range > 0
                for (i, g) in enumerate(gens_at_bus)
                    qgen[g] = Q_gen_at_bus[b] * q_ranges[i] / total_range
                end
            else
                # Equal distribution if all ranges are zero
                for g in gens_at_bus
                    qgen[g] = Q_gen_at_bus[b] / length(gens_at_bus)
                end
            end
        end
    end

    return qgen
end

"""
    compute_bus_reactive_power(polar::AbstractPolarFormulation, stack::NetworkStack)

Compute the reactive power injection at each bus after power flow solution.

# Arguments
- `polar`: The polar formulation
- `stack`: The network stack containing the solution

# Returns
- `Vector{Float64}`: Reactive power injection at each bus in per-unit
"""
function compute_bus_reactive_power(
    polar::AbstractPolarFormulation,
    stack::AbstractNetworkStack
)
    pf = polar.network

    # Get voltage solution (transfer to CPU if on GPU)
    vmag = Array(stack.vmag)
    vang = Array(stack.vang)
    V = vmag .* exp.(im .* vang)

    # Compute Q injection at all buses: Q = imag(V * conj(Y * V))
    S_bus = V .* conj.(pf.Ybus * V)
    return imag.(S_bus)
end

"""
    check_q_violations(polar, stack, bustype; tol=1e-6)

Check which generators violate their reactive power limits.

# Arguments
- `polar`: The polar formulation
- `stack`: The network stack containing the solution
- `bustype`: Current bus type vector (may differ from network's original)
- `tol::Float64=1e-6`: Tolerance for violation detection

# Returns
- `Vector{QLimitStatus}`: List of violated generators with their limit status
"""
function check_q_violations(
    polar::AbstractPolarFormulation,
    stack::AbstractNetworkStack,
    bustype::Vector{Int};
    tol::Float64=1e-6
)
    pf = polar.network
    ngen = pf.ngen

    qgen = compute_generator_reactive_power(polar, stack)
    q_min, q_max = PS.bounds(pf, PS.Generators(), PS.ReactivePower())

    violations = QLimitStatus[]

    for g in 1:ngen
        bus_idx = pf.gen2bus[g]
        # Only check PV and REF buses (not already converted to PQ)
        if bustype[bus_idx] != PS.PQ_BUS_TYPE
            if qgen[g] > q_max[g] + tol
                push!(violations, QLimitStatus(g, bus_idx, q_max[g], :upper))
            elseif qgen[g] < q_min[g] - tol
                push!(violations, QLimitStatus(g, bus_idx, q_min[g], :lower))
            end
        end
    end

    return violations
end

"""
    network_to_data(network::PS.PowerNetwork)

Convert a PowerNetwork back to a data dictionary format.
"""
function network_to_data(network::PS.PowerNetwork)
    data = Dict{String, Array}()
    data["bus"] = copy(network.buses)
    data["branch"] = copy(network.branches)
    data["gen"] = copy(network.generators)
    data["baseMVA"] = Float64[network.baseMVA]
    if !isnothing(network.costs)
        data["cost"] = copy(network.costs)
    end
    return data
end

"""
    modify_bus_types(network::PS.PowerNetwork, pv_to_pq::Vector{Int}, q_fixed::Dict{Int,Float64})

Create a modified PowerNetwork with specified PV buses converted to PQ.

# Arguments
- `network`: Original PowerNetwork
- `pv_to_pq`: List of bus indices to convert from PV to PQ
- `q_fixed`: Dict mapping bus index to fixed Q injection (generator Q at limit)

# Returns
- `PS.PowerNetwork`: New network with modified bus types and loads
"""
function modify_bus_types(
    network::PS.PowerNetwork,
    pv_to_pq::Vector{Int},
    q_fixed::Dict{Int,Float64}
)
    # Convert network to data dictionary
    data = network_to_data(network)

    # Get bus indices
    BUS_I, BUS_TYPE, PD, QD = PS.IndexSet.idx_bus()[1:4]

    # Modify bus types
    for bus in pv_to_pq
        if data["bus"][bus, BUS_TYPE] != PS.REF_BUS_TYPE
            data["bus"][bus, BUS_TYPE] = PS.PQ_BUS_TYPE
        else
            @warn "Cannot convert reference bus $bus to PQ - skipping"
        end
    end

    # Modify reactive demand to fix Q at limits
    # When a PV bus becomes PQ, the generator Q is fixed
    # Qd_new = Qd_old - Q_gen_fixed (in MVA)
    for (bus, q_val) in q_fixed
        if data["bus"][bus, BUS_TYPE] == PS.PQ_BUS_TYPE
            data["bus"][bus, QD] -= q_val * network.baseMVA
        end
    end

    return PS.PowerNetwork(data)
end
