```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
    const PS = ExaPF.PowerSystem
end
DocTestFilters = [r"ExaPF"]
```

# PowerSystem

## Description

```@docs
PS.AbstractPowerSystem
PS.PowerNetwork
```

## API Reference

### Network elements

```@docs
PS.AbstractNetworkElement
```

List of elements:

```@docs
PS.Buses
PS.Lines
PS.Generator
```

### Network attributes

```@docs
PS.AbstractNetworkAttribute
```

Function for getting attributes from a network:
```@docs
PS.get
```

List of attributes:
```@docs
PS.NumberOfBuses
PS.NumberOfLines
PS.NumberOfGenerators
PS.NumberOfPVBuses
PS.NumberOfPQBuses
PS.NumberOfSlackBuses
PS.BaseMVA
```

### Network values

```@docs
PS.AbstractNetworkValues
```

List of values:
```@docs
PS.VoltageMagnitude
PS.VoltageAngle
PS.ActivePower
PS.ReactivePower

```

Function to get the range of a given value:
```@docs
PS.bounds
```
