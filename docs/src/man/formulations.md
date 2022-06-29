
```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
    using LazyArtifacts
    import ExaPF: AutoDiff
end
```

# Formulations
ExaPF's formalism is based on a vectorized formulation of the
power flow problem, as introduced in [Lee, Turitsyn, Molzahn, Roald (2020)](https://arxiv.org/abs/1906.09483). Throughout this page, we will refer to this formulation as **LTMR2020**.
It is equivalent to the classical polar formulation of the OPF.

In what follows, we denote by $v \in \mathbb{R}^{n_b}$ the
voltage magnitudes, $\theta \in \mathbb{R}^{n_b}$ the voltage
angles and $p_g, q_g \in \mathbb{R}^{n_g}$ the active and
reactive power generations. The active and reactive loads are denoted
respectively by $p_d, q_d \in \mathbb{R}^{n_b}$.

## Power flow model

The idea is to factorize all nonlinearities inside a basis function
depending both on the voltage magnitudes $v$ and voltage angles
$\theta$, such that $\psi: \mathbb{R}^{n_b} \times \mathbb{R}^{n_b} \to \mathbb{R}^{2n_\ell + n_b}$.
If we introduce the intermediate expressions
```math
    \psi_\ell^C(v, \theta) = v^f  v^t  \cos(\theta_f - \theta_t) \quad \forall \ell = 1, \cdots, n_\ell \\
    \psi_\ell^S(v, \theta) = v^f  v^t  \sin(\theta_f - \theta_t) \quad \forall \ell = 1, \cdots, n_\ell \\
    \psi_k(v, \theta) = v_k^2 \quad \forall k = 1, \cdots, n_b
```
the basis $\psi$ is defined as
```math
    \psi(v, \theta) = [\psi_\ell^C(v, \theta)^\top ~ \psi_\ell^S(v, \theta)^\top ~ \psi_k(v, \theta)^\top ] \, .
```


The basis $\psi$ encodes all the nonlinearities in the problem. For instance,
the power flow equations rewrite directly as
```math
    \begin{bmatrix}
    C_g p_g - p_d \\
    C_g q_g - q_d
    \end{bmatrix}
    +
    \underbrace{
    \begin{bmatrix}
    - \hat{G}^c & - \hat{B}^s & -G^d \\
     \hat{B}^c & - \hat{G}^s & B^d
    \end{bmatrix}
    }_{M}
    \psi(v, \theta)
    = 0
```
with $C_g \in \mathbb{R}^{n_b \times n_g}$ the bus-generators
incidence matrix, and the matrices $B, G$ defined from the admittance
matrix $Y_b$ of the network.

Similarly, the line flows rewrite
```math
    \begin{bmatrix}
    s_p^f \\ s_q^f
    \end{bmatrix}
    =
    \overbrace{
    \begin{bmatrix}
    G_{ft} & B_{ft} & G_{ff} C_f^\top \\
    -B_{ft} & G_{ft} & -B_{ff} C_f^\top
    \end{bmatrix}
    }^{L_{line}^f}
    \psi(v, \theta) \\
    \begin{bmatrix}
    s_p^t \\ s_q^t
    \end{bmatrix}
    =
    \underbrace{
    \begin{bmatrix}
    G_{tf} & B_{tf} & G_{tt} C_t^\top \\
    -B_{tf} & G_{tf} & -B_{tt} C_t^\top
    \end{bmatrix}
    }_{L_{line}^t}
    \psi(v, \theta)
```
with $C_f \in \mathbb{R}^{n_b \times n_\ell}$ the bus-from incidence
matrix and $C_t \in \mathbb{R}^{n_b \times n_\ell}$ the bus-to incidence
matrix. Then, the line flows constraints write directly with the quadratic expressions:
```math
    (s_p^f)^2 + (s_q^f)^2 \leq (s^{max})^2 \quad \, ,
    (s_p^t)^2 + (s_q^t)^2 \leq (s^{max})^2 \quad \, .
```


### Why is this model advantageous?
Implementing the model **LTMR2020** is not difficult
once the basis function $\psi$ has been defined. Indeed,
if we select a subset of the power flow equations (as usual, associated
to the active injections at PV nodes, and active and reactive injections
at PQ nodes), we get
```math
    C_{eq} p_g + M_{eq} \psi + \tau = 0
```
with $C_{eq}$ defined from the bus-generator incidence matrix $C_g$, $M_{eq}$
a subset of the matrix $M$, $\tau$ a constant depending on the loads in
the problem. Note that $C_{eq}$ and $M_{eq}$ are sparse matrices, so the
expression can be implemented efficiently with sparse linear algebra
operations (2 SpMV operations, 2 vector additions).
The same holds true for the line flow constraints,
evaluated with 2 SpMV operations:
```math
    s^f = L_{line}^f \psi \, , \quad
    s^t = L_{line}^t \psi \, .
```

In ExaPF, all nonlinear expressions are written
as linear operations depending on the nonlinear basis
$\psi$. By doing so, all the unstructured sparsity of the power flow
problem is directly handled inside the sparse linear algebra library
(`cusparse` on CUDA GPU, `SuiteSparse` on the CPU).

In what follows, we detail the implementation of the **LTMR2020** model
in ExaPF.


## How to instantiate the inputs?

We have implemented the LTMR2020 model in ExaPF, both on the CPU
and on CUDA GPU. All the operations have been rewritten in a
vectorized fashion. Every model depends on *inputs* we propagate
forward with *functions*. In ExaPF, the inputs will be specified
in a `NetworkStack <: AbstractStack`. The functions will be implemented
as [`AutoDiff.AbstractExpression`](@ref).

### Specifying inputs in `NetworkStack`
Our three inputs are $(v, \theta, p_g) \in \mathbb{R}^{2n_b + n_g}$ (voltage magnitude, voltage
angle, power generations).
The basis $\psi$ is considered as an intermediate expression.

We store all inputs in a [`NetworkStack`](@ref) structure:
```julia
struct NetworkStack{VT} <: AbstractStack
    input::VT
    vmag::VT # voltage magnitudes (view)
    vang::VT # voltage angles (view)
    pgen::VT # active power generations (view)
    ψ::VT    # nonlinear basis ψ(vmag, vang)
end
```

All the inputs are specified in the vector `input`. The three vectors
`vmag`, `vang` and `pgen` are views porting on `input`, and are defined
mostly for convenience. By convention the vector `input` is ordered
as `[vmag; vang; pgen]`:
```jldoctests
# Define dimension of the problem
julia> nbus, ngen, nlines = 3, 2, 4;

julia> stack = ExaPF.NetworkStack(nbus, ngen, nlines, Vector{Float64}, Vector{Float64})
8-elements NetworkStack{Vector{Float64}}

julia> stack.input
8-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0

julia> stack.vmag .= 1;

julia> stack.vang .= 2;

julia> stack.pgen .= 3;

julia> stack.input
8-element Vector{Float64}:
 1.0
 1.0
 1.0
 2.0
 2.0
 2.0
 3.0
 3.0
```

The basis vector `ψ` is an intermediate expression,
whose values depend on the inputs.


### Defining a state and a control

In the reduced space method, we have to split the variables
in a *state* $x$ and a *control* $u$. By default, we define
```math
    x = (\theta_{pv}, \theta_{pq}, v_{pq}) \, , \quad
    x = (v_{ref}, v_{pv}, p_{g,genpv}) \,.
```
and the control, and was not flexible. In the new implementation,
we define the state and the control as two *mappings* porting
on the vector `stack.input` (which itself stores all the inputs in
the problem):
```jldoctests
julia> nbus, ngen, nlines = 4, 3, 4;

julia> stack = ExaPF.NetworkStack(nbus, ngen, nlines, Vector{Float64}, Vector{Float64});

julia> stack.input .= 1:length(stack.input); # index array input

julia> ref, pv, pq, genpv = [1], [2], [3, 4], [2, 3];

julia> mapx = [nbus .+ pv; nbus .+ pq; pq];

julia> mapu = [ref; pv; 2*nbus .+ genpv];

julia> x = @view stack.input[mapx]
5-element view(::Vector{Float64}, [6, 7, 8, 3, 4]) with eltype Float64:
 6.0
 7.0
 8.0
 3.0
 4.0

julia> u = @view stack.input[mapu]
4-element view(::Vector{Float64}, [1, 2, 10, 11]) with eltype Float64:
  1.0
  2.0
 10.0
 11.0
```

By doing so, the values of the state and the control are directly
stored inside the `NetworkStack` structure, avoiding to duplicate values
in the memory.


## How to manipulate the expressions?

ExaPF implements the different functions required to implement
the optimal power flow problem with the polar formulation:
- [`PowerFlowBalance`](@ref): power flow balance equations
- [`PowerGenerationBounds`](@ref): bounds on the power generation
- [`LineFlows`](@ref): line flow constraints
- [`CostFunction`](@ref): quadratic cost function

Each function follows the LTMR2020 model and depends on
the basis function $$\psi(v, \theta)$$, here implemented in
the [`PolarBasis`](@ref) function.

We demonstrate how to use the different functions on the `case9`
instance. The procedure remains the same for all power network.
```jldoctests interface
julia> polar = ExaPF.load_polar("case9.m");

julia> stack = ExaPF.NetworkStack(polar);

```

!!! note
    All the code presented below is agnostic with regards
    to the specific device (`CPU`, `CUDADevice`...) we are using.
    By default, ExaPF computes the expressions on the CPU.
    Deporting the computation on a `CUDADevice` simply
    translates to instantiate the [`PolarForm`](@ref) structure
    on the GPU: `polar = PolarForm("case9.m", CUDADevice())`.

### Interface

All functions are following [`AutoDiff.AbstractExpression`](@ref)'s interface.
The structure of the network is specified by the [`PolarForm`](@ref)
we pass as an argument in the constructor. For instance,
we build a new [`PolarBasis`](@ref) expression associated to `case9`
directly as
```jldoctests interface
julia> basis = ExaPF.PolarBasis(polar)
PolarBasis (AbstractExpression)

```
Each expression as a given dimension, given by
```jldoctests interface
julia> length(basis)
27

```
In ExaPF, the inputs and the parameters are stored
inside a [`NetworkStack`](@ref) structure. Evaluating the basis $$\psi$$
naturally translates to
```jldoctests interface
julia> basis(stack)
27-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 0.0
 ⋮
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
```
This function call allocates a vector `psi` with 27 elements and evaluates
the basis associated to the LTMR2020 model. To avoid unnecessary allocation,
one can preallocate the vector `psi`:
```jldoctests interface
julia> psi = zeros(length(basis)) ;

julia> basis(psi, stack);

```

### Compose expressions together

In the LTMR2020 model, the polar basis $$\psi(v, \theta)$$ depends only
on the voltage magnitudes and the voltage angles. However, this is not
the case for the other functions (power flow balance, line flows, ...),
which all depends on the basis $$\psi(v, \theta)$$.

In ExaPF, one has to build manually the vectorized expression tree associated
to the power flow model. Luckily, evaluating the LTMR2020 simply amounts
to compose functions together with the polar basis $$\psi(v, \theta)$$.
ExaPF overloads the function `∘` to compose functions with a [`PolarBasis`](@ref)
instance. The power flow balance can be evaluated as
```jldoctests interface
julia> pflow = ExaPF.PowerFlowBalance(polar) ∘ basis;

```
which returns a [`ComposedExpressions`](@ref) structure.

The function `pflow` follows the same API, as any regular
[`AutoDiff.AbstractExpression`](@ref).
```jldoctests interface
julia> n_balance = length(pflow)
14

julia> pflow(stack) # evaluate the power flow balance
14-element Vector{Float64}:
 -1.63
 -0.85
  0.0
  0.9000000000000004
  0.0
  1.0
  0.0
  1.2499999999999998
 -0.1670000000000016
  0.04200000000000159
 -0.28349999999999653
  0.17099999999999937
 -0.22749999999999915
  0.2590000000000039

```
When we evaluate a [`ComposedExpressions`](@ref), ExaPF
first computes the basis $$\psi(v, \theta)$$ inside `stack.psi`,
and then ExaPF uses the values in `stack.psi` to evaluate the
final result.

The procedure remains the same if one wants to evaluate the
[`LineFlows`](@ref) or the [`PowerGenerationBounds`](@ref).
For instance, evaluating the line flows amounts to
```jldoctests interface
julia> line_flows = ExaPF.LineFlows(polar) ∘ basis;

julia> line_flows(stack)
18-element Vector{Float64}:
 0.0
 0.006241000000000099
 0.0320410000000001
 0.0
 0.010920249999999961
 0.005550250000000068
 0.0
 0.02340899999999987
 0.007743999999999858
 0.0
 0.006241000000000099
 0.0320410000000001
 0.0
 0.010920249999999961
 0.005550250000000068
 0.0
 0.02340899999999987
 0.007743999999999858

```

