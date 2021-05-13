export PolarForm, bounds, powerflow
export State, Control, Parameters, NumberOfState, NumberOfControl

"""
    AbstractStructure

The user may specify a mapping to the single input vector `x` for AD.

"""
abstract type AbstractStructure end

"""
    AbstractFormulation

Second layer of the package, implementing the interface between
the first layer (the topology of the network) and the
third layer (implementing the callbacks for the optimization solver).

"""
abstract type AbstractFormulation end

"""
    AbstractFormAttribute

Attributes attached to an `AbstractFormulation`.
"""
abstract type AbstractFormAttribute end

"Number of states attached to a particular formulation."
struct NumberOfState <: AbstractFormAttribute end

"Number of controls attached to a particular formulation."
struct NumberOfControl <: AbstractFormAttribute end

"""
    AbstractVariable

Variables corresponding to a particular formulation.
"""
abstract type AbstractVariable end

"""
    State <: AbstractVariable

All variables ``x`` depending on the variables `Control` ``u`` through
a non-linear equation ``g(x, u) = 0``.

"""
struct State <: AbstractVariable end

"""
    Control <: AbstractVariable

Independent variables ``u`` used in the reduced-space
formulation.

"""
struct Control <: AbstractVariable end

"""
    PhysicalState <: AbstractVariable

All physical variables describing the current physical state
of the underlying network.

`PhysicalState` variables are encoded in a `AbstractNetworkBuffer`,
storing all the physical values needed to describe the current
state of the network.

"""
struct PhysicalState <: AbstractVariable end

# Templates
"""
    get(form::AbstractFormulation, attr::AbstractFormAttribute)

Return value of attribute `attr` attached to the particular
formulation `form`.

## Examples

```julia
get(form, NumberOfState())
get(form, NumberOfControl())

```
"""
function get end

"""
    setvalues!(form::AbstractFormulation, attr::PS.AbstractNetworkAttribute, values)

Update inplace the attribute's values specified by `attr`.

## Examples

```julia
setvalues!(form, ActiveLoad(), new_ploads)
setvalues!(form, ReactiveLoad(), new_qloads)

```
"""
function setvalues! end

"""
    bounds(form::AbstractFormulation, var::AbstractVariable)

Return the bounds attached to the variable `var`.

    bounds(form::AbstractFormulation, func::Function)

Return a tuple of vectors `(lb, ub)` specifying the admissible range
of the constraints specified by the function `cons_func`.

## Examples

```julia
u_min, u_max = bounds(form, Control())
h_min, h_max = bounds(form, reactive_power_constraints)

```
"""
function bounds end

"""
    initial(form::AbstractFormulation, var::AbstractVariable)

Return an initial position for the variable `var`.

## Examples

```julia
u₀ = initial(form, Control())
x₀ = initial(form, State())

```
"""
function initial end

"""
    powerflow(form::AbstractFormulation,
              algo::AbstractNonLinearSolver;
              kwargs...)

    powerflow(form::AbstractFormulation,
              jacobian::AutoDiff.Jacobian,
              buffer::AbstractNetworkBuffer,
              algo::AbstractNonLinearSolver;
              kwargs...) where VT <: AbstractVector

Solve the power flow equations ``g(x, u) = 0`` w.r.t. the state ``x``,
using the algorithm specified in `algo` ([`NewtonRaphson`](@ref) by default).
The initial state ``x`` is specified inside
`buffer`. The object `buffer` is modified inplace in the function.

The algorithm stops when a tolerance `tol` or a maximum number of
iterations `maxiter` is reached (these parameters being specified
in the object `algo`).

## Notes
If only the arguments `form` and `algo` are specified to the function,
then the Jacobian `jacobian` and the cache `buffer` are inferred
from the object `form`.

## Arguments

* `form::AbstractFormulation`: formulation of the power flow equation
* `jacobian::AutoDiff.Jacobian`: Jacobian
* `buffer::AbstractNetworkBuffer`: buffer storing current state `x` and control `u`
* `algo::AbstractNonLinearSolver`: non-linear solver. Currently only `NewtonRaphson` is being implemented.

## Optional arguments

* `linear_solver::AbstractLinearSolver` (default `DirectSolver()`): solver to solve the linear systems ``J x = y`` arising at each iteration of the Newton-Raphson algorithm.

"""
function powerflow end

# Cost function
"""
    cost_production(form::AbstractFormulation, buffer::AbstractNetworkBuffer)::Float64

Get operational cost.
"""
function cost_production end

"""
    cost_penalty_ramping_constraints(form::AbstractFormulation, buffer::AbstractNetworkBuffer, params...)::Float64

Get operational cost, including a quadratic penalty penalizing the ramping
constraints w.r.t. a given reference.
"""
function cost_penalty_ramping_constraints end

function network_operations end

# Generic constraints

"""
    voltage_magnitude_constraints(form::AbstractFormulation, cons::AbstractVector, buffer::AbstractNetworkBuffer)

Bounds the voltage magnitudes at PQ nodes:
```math
v_{pq}^♭ ≤ v_{pq} ≤ v_{pq}^♯ .
```
The result is stored inplace, inside `cons`.

## Note
The constraints on the voltage magnitudes at PV nodes ``v_{pv}``
are taken into account when bounding the control ``u``.
"""
function voltage_magnitude_constraints end

"""
    active_power_constraints(form::AbstractFormulation, cons::AbstractVector, buffer::AbstractNetworkBuffer)

Evaluate the constraints on the **active power production** at the generators
that are not already taken into account in the bound constraints.
```math
p_g^♭ ≤ p_g ≤ p_g^♯  .
```

The result is stored inplace, inside the vector `cons`.
"""
function active_power_constraints end

"""
    reactive_power_constraints(form::AbstractFormulation, cons::AbstractVector, buffer::AbstractNetworkBuffer)

Evaluate the constraints on the **reactive power production** at the generators:
```math
q_g^♭ ≤ q_g ≤ q_g^♯  .
```
The result is stored inplace, inside the vector `cons`.
"""
function reactive_power_constraints end

"""
    flow_constraints(form::AbstractFormulation, cons::AbstractVector, buffer::AbstractNetworkBuffer)

Evaluate the thermal limit constraints porting on the lines of the network.
The result is stored inplace, inside the vector `cons`.
"""
function flow_constraints end


@doc raw"""
    power_balance(form::AbstractFormulation, cons::AbstractVector, buffer::AbstractNetworkBuffer)

Evaluate the power balance in the network:
```math
g(x, u) = 0 ,
```
corresponding to the balance equations
```math
\begin{aligned}
    p_i &= v_i \sum_{j}^{n} v_j (g_{ij}\cos{(\theta_i - \theta_j)} + b_{ij}\sin{(\theta_i - \theta_j})) \,, &
    ∀ i ∈ \{PV, PQ\} \\
    q_i &= v_i \sum_{j}^{n} v_j (g_{ij}\sin{(\theta_i - \theta_j)} - b_{ij}\cos{(\theta_i - \theta_j})) \,. &
    ∀ i ∈ \{PQ\} \\
\end{aligned}
```

The result is stored inplace, inside the vector `cons`.
"""
function power_balance end

function bus_power_injection end

# Interface for the constraints
"""
    size_constraint(cons_func::Function)::Bool
Return whether the function `cons_func` is a supported constraint
in the powerflow model.
"""
function is_constraint end

"""
    size_constraint(form::AbstractFormulation, cons_func::Function)::Int

Get number of constraints specified by the function `cons_func`
in the formulation `form`.
"""
function size_constraint end

"""
    adjoint!(form::AbstractFormulation, pbm::AutoDiff.TapeMemory, adj_h, h, buffer)

Return the adjoint w.r.t. the variables of the network (voltage magnitudes
and angles, power injection) for the constraint stored inside the [`AutoDiff.TapeMemory`](@ref)
object `pbm`. The results are stored directly inside the stack stored
inside `pbm`.
"""
function adjoint! end

"""
    jacobian_transpose_product!(form::AbstractFormulation, pbm::AutoDiff.TapeMemory, buffer, v)

Return the two transpose-Jacobian vector product ``(Jᵤ^⊤ v, Jₓ^⊤ v)``  w.r.t. the
control ``u`` and the state ``x``. Store the two resulting vectors directly inside
the [`AutoDiff.TapeMemory`](@ref) `pbm`.

"""
function jacobian_transpose_product! end

"""
    matpower_jacobian(form::AbstractFormulation, X::Union{State,Control}, cons_func::Function, V::Vector{Complex})
    matpower_jacobian(form::AbstractFormulation, X::Union{State,Control}, cons_func::Function, buffer::AbstractNetworkBuffer)

For the constraint `cons_func`, return the expression of the Jacobian ``J``
w.r.t. the state or the control (depending on the argument `X`),
as given by MATPOWER.
"""
function matpower_jacobian end

@doc raw"""
    matpower_hessian(form::AbstractFormulation, cons_func::Function, buffer::AbstractNetworkBuffer, λ::AbstractVector)

For constraint `cons_func`, return the three matrices ``(λ^⊤ H_{xx},
λ^⊤ H_{xu},λ^⊤ H_{uu})`` storing the product of the Hessian tensor ``H`` with the vector ``\lambda``.
The expressions of the Hessian matrices are given by MATPOWER.

"""
function matpower_hessian end

"""
    jacobian_sparsity(form::AbstractFormulation, cons_func::Function, X::Union{State,Control})

For the constraint `cons_func`, return the sparsity pattern of the Jacobian ``J``
w.r.t. the state or the control (depending on the argument `X`).

"""
function jacobian_sparsity end

