export PolarForm, get, bounds, powerflow
export State, Control, Parameters, NumberOfState, NumberOfControl


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

All variables `x` depending on the variables `Control` `u` through
a non-linear equation `g(x, u) = 0`.

"""
struct State <: AbstractVariable end

"""
    Control <: AbstractVariable

Implement the independent variables used in the reduced-space
formulation.

"""
struct Control <: AbstractVariable end

"""
    Parameters <: AbstractVariable

Constant parameters.

"""
struct Parameters <: AbstractVariable end

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
    bounds(form::AbstractFormulation, attr::AbstractFormAttribute)

Return the bounds attached to a particular attribute.

    bounds(form::AbstractFormulation, attr::AbstractFormAttribute)

Return the upper and lower bounds attached to a given constraint
functional.

## Examples

```julia
u_min, u_max = bounds(form, Control())
h_min, h_max = bounds(form, power_constraints)

```
"""
function bounds end

"""
    initial(form::AbstractFormulation, attr::AbstractFormAttribute)

Return an initial position for the attribute `attr`.

## Examples

```julia
u₀ = initial(form, Control())
x₀ = initial(form, State())

```
"""
function initial end

"""
    powerflow(form::AbstractFormulation,
              jacobian::AD.StateJacobianAD,
              x::VT,
              u::VT,
              p::VT;
              kwargs...) where VT <: AbstractVector

Solve the power flow equations `g(x, u) = 0` w.r.t. the state `x`,
using a Newton-Raphson algorithm.
The power flow equations are specified in the formulation `form`.
The solution `x(u)` depends on the control `u` passed in input.

The algorithm stops when a tolerance `tol` or a maximum number of
irations `maxiter` are reached.

## Arguments

* `form::AbstractFormulation`: formulation of the power flow equation
* `x::AbstractVector`: initial state (the value of `x` is kept constant)
* `u::AbstractVector`: fixed control (the control is determined by the formulation used)
* `p::AbstractVector`: fixed parameters (also dependent on the formulation used)

## Optional arguments

* `tol::Float64` (default `1e-7`): tolerance of the Newton-Raphson algorithm.
* `maxiter::Int` (default `20`): maximum number of iterations.
* `verbose_level::Int` (default `O`, max value: `3`): verbose level

"""
function powerflow end

"""
    power_balance(polar::PolarForm, x::VT, u::VT, p::VT) where {VT<:AbstractVector}

Get power balance at buses, depending on the state `x` and the control `u`.

"""
function power_balance end

# Generic constraints
"""
    state_constraints(polar::PolarForm, g::VT, x::VT, u::VT, p::VT) where {VT<:AbstractVector}

Evaluate the constraints porting on the state `x`, as a
function of `x` and `u`. The result is stored inplace, inside `g`.
"""
function state_constraints end

"""
    power_constraints(polar::PolarForm, g::VT, x::VT, u::VT, p::VT) where {VT<:AbstractVector}

Evaluate the constraints on the **power production** that are not taken into
account in

* the box constraints on the control `u`
* the box constraints on the state `x` (implemented in `state_constraints`)

The result is stored inplace, inside `g`.
"""
function power_constraints end

"""
    thermal_limit_constraints(polar::PolarForm, g::VT, x::VT, u::VT, p::VT) where {VT<:AbstractVector}

Evaluate the thermal limit constraints porting on the lines of the network.

The result is stored inplace, inside the vector `g`.
"""
function thermal_limit_constraints end

# Polar formulation
include("polar.jl")
