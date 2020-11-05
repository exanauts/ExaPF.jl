```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
    const PS = ExaPF.PowerSystem
end
DocTestFilters = [r"ExaPF"]
```

# PowerSystem

The main goal of ExaPF.jl is the solution of optimization problems for electrical power systems in the steady state. The first step in this process is the creation of an object that describes the physics and topology of the power system which ultimatelly will be mapped into an abstract mathematical optimization problem. In this section we briefly review the power system in the steady state and describe the tools to create and examine power systems in ExaPF.jl.

## Description

The electrical power system is represented as a linear, lumped network which has to satisfy the Kirchoff laws:


```math
    \bm{i} = \bm{Y}\bm{v} \,,
```

where $\bm{i}, \bm{v} \in \mathbb{C}^{N_B}$ are the current and voltage
vectors associated to the system and $\bm{Y} \in \mathbb{C}^{N_B \times N_B}$
is the admittance matrix. These equations are often rewritten in terms of aparent powers:


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

Currently we can create a PowerNetwork object by parsing a MatPower datafile.

```julia-repl
julia> ps = PowerSystem.PowerNetwork("test/case9.m", 1)
```

If we print the object, we will obtain bus information and initial voltage and power that we read from the datafile.

```julia-repl
julia> println(ps)
Power Network characteristics:
	Buses: 9. Slack: 1. PV: 2. PQ: 6
	Generators: 3.
	==============================================
	BUS 	 TYPE 	 VMAG 	 VANG 	 P 	 Q
	==============================================
	1 	  3 	 1.000	0.00	0.000	0.000
	2 	  2 	 1.000	0.00	1.630	0.000
	3 	  2 	 1.000	0.00	0.850	0.000
	4 	  1 	 1.000	0.00	0.000	0.000
	5 	  1 	 1.000	0.00	-0.900	-0.300
	6 	  1 	 1.000	0.00	0.000	0.000
	7 	  1 	 1.000	0.00	-1.000	-0.350
	8 	  1 	 1.000	0.00	0.000	0.000
	9 	  1 	 1.000	0.00	-1.250	-0.500
```

then, using multiple dispatch, we have defined a set of abstract datatypes and getter functions which allow us to retrieve information from the PowerNetwork object

```julia-repl
julia> PowerSystem.get(ps, PowerSystem.NumberOfPQBuses())
6
julia> PowerSystem.get(ps, PowerSystem.NumberOfPVBuses())
2
julia> PowerSystem.get(ps, PowerSystem.NumberOfSlackBuses())
1
```
