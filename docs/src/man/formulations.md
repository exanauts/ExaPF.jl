
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


## Implementation

We have implemented the LTMR2020 model in ExaPF, both on the CPU
and on CUDA GPU. All the operations have been rewritten in a
vectorized fashion. Every model depends on *inputs* we propagate
forward with *functions*. In ExaPF, the inputs will be specified
in a `NetworkStack <: AbstractStack`. The functions will be implemented
as `AbstractExpressions`.

### Specifying inputs in `NetworkStack`
Our three inputs are $(v, \theta, p_g) \in \mathbb{R}^{2n_b + n_g}$ (voltage magnitude, voltage
angle, power generations).
The basis $\psi$ is considered as an intermediate expression.

We store all inputs in a `NetworkStack` structure:
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
```julia
# Define dimension of the problem
julia> nbus, ngen, nlines = 3, 2, 4
# Instantiate stack
julia> stack = ExaPF.NetworkStack(nbus, ngen, nlines, Vector{Float64});

# Look at values in input
julia> stack.input'
1×8 adjoint(::Vector{Float64}) with eltype Float64:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

# Modify values in views
julia> stack.vmag .= 1
julia> stack.vang .= 2
julia> stack.pgen .= 3
# Look again at the values in input:
julia> stack.input'
1×8 adjoint(::Vector{Float64}) with eltype Float64:
 1.0  1.0  1.0  2.0  2.0  2.0  3.0  3.0
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
```julia
# Define dimension of the problem
julia> nbus, ngen, nlines = 4, 3, 4
# Instantiate stack
julia> stack = ExaPF.NetworkStack(nbus, ngen, nlines, Vector{Float64});
# Define pv, pq, genpv
julia> ref, pv, pq, genpv = [1], [2], [3, 4], [2, 3]
# Define state as a mapping on stack.input
# Remember that ordering of input is [vmag, vang, pgen]!
julia> mapx = [nbus .+ pv; nbus .+ pq; pq]
julia> mapu = [ref; pv; genpv]
# Load values for state and control
julia> x = @view stack.input[mapx]
julia> u = @view stack.input[mapu]
```

By doing so, the values of the state and the control are directly
stored inside the `NetworkStack` structure, avoiding to duplicate values
in the memory.
