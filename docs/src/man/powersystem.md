```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
    const PS = ExaPF.PowerSystem
end
DocTestFilters = [r"ExaPF"]
```

# PowerSystem

The main goal of ExaPF.jl is the solution of optimization problems for electrical power systems in the steady state. The first step in this process is the creation of an object that describes the physics and topology of the power system which ultimately will be mapped into an abstract mathematical optimization problem. In this section we briefly review the power system in the steady state and describe the tools to create and examine power systems in ExaPF.jl.

We usually load the `PowerSystem` system submodule with the alias `PS`:
```julia-repl
julia> PS = ExaPF.PowerSystem

```

## Description

The electrical power system is represented as a linear, lumped network which has to satisfy the Kirchhoff laws:

```math
    \bm{i} = \bm{Y}\bm{v} \,,
```

where $\bm{i}, \bm{v} \in \mathbb{C}^{N_B}$ are the current and voltage
vectors associated to the system and $\bm{Y} \in \mathbb{C}^{N_B \times N_B}$
is the admittance matrix. These equations are often rewritten in terms of apparent powers:

```math
    \bm{s} = \bm{p} + j\bm{q} = \textit{diag}(\bm{v^*}) \bm{Y}\bm{v}
```

Using polar representation of the voltage vector, such as $\bm{v} = |v|e^{j \theta}$,
each bus $i=1, \cdots, N_B$  must satisfy the power balance equations:

```math
\begin{aligned}
    p_i &= v_i \sum_{j}^{n} v_j (g_{ij}\cos{(\theta_i - \theta_j)} + b_{ij}\sin{(\theta_i - \theta_j})) \,, \\
    q_i &= v_i \sum_{j}^{n} v_j (g_{ij}\sin{(\theta_i - \theta_j)} - b_{ij}\cos{(\theta_i - \theta_j})) \,.
\end{aligned}
```

where each bus $i$ has variables $p_i, q_i, v_i, \theta_i$ and the topology
of the network is defined by a non-negative value of the admittance between
two buses $i$ and $j$, $y_{ij} = g_{ij} + ib_{ij}$.

## The PowerNetwork object

Currently we can create a [`PS.PowerNetwork`](@ref) object by parsing a MATPOWER data file.

```@setup artifacts
using ExaPF
datafile = "case9.m"
ExaPF.PowerSystem.load_case(datafile)
```

```jldoctests
julia> datafile = "case9.m";

julia> ps = PS.load_case(datafile)
PowerNetwork object with:
    Buses: 9 (Slack: 1. PV: 2. PQ: 6)
    Generators: 3.

```

Then, using multiple dispatch, we have defined a set of abstract data types and getter functions which allow us to retrieve information from the PowerNetwork object

```jldoctests
julia> PS.get(ps, PS.NumberOfPQBuses())
6

julia> PS.get(ps, PS.NumberOfPVBuses())
2

julia> PS.get(ps, PS.NumberOfSlackBuses())
1
```
