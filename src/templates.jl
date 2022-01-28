
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
the non-linear equation ``g(x, u) = 0``.

"""
struct State <: AbstractVariable end

"""
    Control <: AbstractVariable

Independent variables ``u`` used in the reduced-space formulation.

"""
struct Control <: AbstractVariable end

