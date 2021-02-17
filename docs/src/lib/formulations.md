```@meta
CurrentModule = ExaPF
```

# Formulations

## Description

```@docs
AbstractFormulation

```

## API Reference

### Variables

```@docs
AbstractVariable
State
Control
PhysicalState

```

Get default values attached to a given variable:
```@docs
initial

```

### Attributes

```@docs
AbstractFormAttribute
NumberOfState
NumberOfControl

```
`ExaPF` extends `Base.get` to query the different attributes
of a model:
```@docs
get

```

### Powerflow solver

```@docs
powerflow
NewtonRaphson

power_balance!

```

### Costs

```@docs
cost_production
```

### Constraints

Current supported constraints are:
```@docs
state_constraints
power_constraints
flow_constraints

```

These functions allow to query constraints' attributes:
```@docs
is_constraint
size_constraint
bounds

```

### Utils

To ease the integration, the following functions have been
imported from MATPOWER. Note that these functions work
exclusively on the CPU.

```@docs
power_balance
residual_jacobian

```

```@docs
get_power_injection
get_react_injection
```

