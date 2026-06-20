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
PowerFlowProblem
run_pf
solve!
nlsolve!
NewtonRaphson
get_active_load
get_reactive_load
set_active_load!
set_reactive_load!
get_voltage_magnitude
get_voltage_angle
get_solution

```

## Reactive Power Limit Enforcement

ExaPF supports enforcing generator reactive power (Q) limits during power flow.
When enabled, the solver iteratively converts PV buses that violate their Q limits
to PQ buses, fixing the reactive power at the limit value.

### Q Limit Data Structures
```@docs
QLimitStatus
QLimitEnforcementResult
BatchedQLimitResult

```

### Q Limit Functions
```@docs
run_pf_with_qlim
run_pf_batched_with_qlim
compute_generator_reactive_power
compute_bus_reactive_power
check_q_violations

```

### Q Limit Accessor Functions
```@docs
get_reactive_power_limits
get_generator_reactive_power
get_qlimit_result
get_violated_generators
is_qlimit_converged
get_bus_reactive_power
get_generators_at_limit

```

## Constraints

The different parts of the polar formulation are
implemented in the following `AbstractExpression`:

```@docs
Basis
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

