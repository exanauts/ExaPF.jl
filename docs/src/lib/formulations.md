```@meta
CurrentModule = ExaPF
```

# Formulations

## Description

```@docs
AbstractFormulation

```

## Powerflow solver

```@docs
powerflow
NewtonRaphson

```

## Constraints

Current supported constraints are:
```@docs
voltage_magnitude_constraints
active_power_constraints
reactive_power_constraints
flow_constraints
power_balance

```

These functions allow to query constraints' attributes:
```@docs
is_constraint
size_constraint
bounds

```

ExaPF implements special functions to compute the derivatives
of each constraints:
```@docs
adjoint!
jacobian_transpose_product!
matpower_jacobian
matpower_hessian
jacobian_sparsity
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
The associated setter is implemented with `setvalues!`:
```@docs
setvalues!
```

### Costs

```@docs
cost_production
```
