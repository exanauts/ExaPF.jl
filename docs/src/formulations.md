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


### Powerflow solver

```@docs
powerflow

```

### Constraints

```@docs
state_constraints
power_constraints
thermal_limit_constraints

```

Admissible range for variables and constraints:
```@docs
bounds
```

### Costs


### Attributes

```@docs
AbstractFormAttribute
NumberOfState
NumberOfControl

```
