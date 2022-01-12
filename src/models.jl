export PolarForm, bounds, powerflow
export State, Control, Parameters, NumberOfState, NumberOfControl

"""
    AbstractFormulation

Second layer of the package, implementing the interface between
the first layer (the topology of the network) and the
third layer (implementing the callbacks for the optimization solver).

"""
abstract type AbstractFormulation end

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
    bounds(form::AbstractFormulation, var::AbstractVariable)

Return the bounds attached to the variable `var`.

    bounds(form::AbstractFormulation, func::AbstractExpression)

Return a tuple of vectors `(lb, ub)` specifying the admissible range
of the constraints specified by the function `cons_func`.

## Examples

```julia
u_min, u_max = bounds(form, Control())
h_min, h_max = bounds(form, reactive_power_constraints)

```
"""
function bounds end

