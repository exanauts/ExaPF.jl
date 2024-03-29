```@meta
CurrentModule = ExaPF
const PS = ExaPF.PowerSystem
DocTestSetup = quote
    using ExaPF
end
```


# Polar formulation

## Generic templates

```@docs
AbstractVariable
AbstractFormulation
State
Control

```

## Structure and variables
```@docs
PolarForm
BlockPolarForm
load_polar
NetworkStack
init!

```

The state and the control are defined as mapping:
```@docs
mapping

```

## Powerflow solver

```@docs
run_pf
nlsolve!
NewtonRaphson

```

## Constraints

The different parts of the polar formulation are
implemented in the following `AbstractExpression`:

```@docs
PolarBasis
PowerFlowBalance
VoltageMagnitudeBounds
PowerGenerationBounds
LineFlows

```

## Objective

The production costs is given in the `AbstractExpression` `CostFunction`:
```@docs
CostFunction
```

# Composition of expressions

The different expressions can be combined together
in several different ways.
```@docs
MultiExpressions
ComposedExpressions
```

