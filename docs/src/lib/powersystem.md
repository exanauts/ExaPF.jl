```@meta
CurrentModule = ExaPF.PowerSystem
```

# PowerSystem

## Description

```@docs
AbstractPowerSystem
PowerNetwork
```

## API Reference

### Network elements

```@docs
AbstractNetworkElement
```

List of elements:

```@docs
Buses
Lines
Generator
```

### Network attributes

```@docs
AbstractNetworkAttribute
```

Function for getting attributes from a network:
```@docs
get
```

List of attributes:
```@docs
NumberOfBuses
NumberOfLines
NumberOfGenerators
NumberOfPVBuses
NumberOfPQBuses
NumberOfSlackBuses
BaseMVA
```

### Network values

```@docs
AbstractNetworkValues
```

List of values:
```@docs
VoltageMagnitude
VoltageAngle
ActivePower
ReactivePower

```

Function to get the range of a given value:
```@docs
bounds
```
