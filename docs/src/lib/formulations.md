```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
end
DocTestFilters = [r"ExaPF"]
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
Parameters

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

### Powerflow solver

```@docs
powerflow

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
thermal_limit_constraints

```

These functions allow to query constraints' attributes:
```@docs
is_constraint
size_constraint
bounds

```

