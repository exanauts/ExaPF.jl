```@meta
CurrentModule = ExaPF.PowerSystem
```

# PowerSystem

## Description

```@docs
AbstractPowerSystem
PowerNetwork
load_case
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
Generators
```

### Network attributes

```@docs
AbstractNetworkAttribute
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
BusAdmittanceMatrix
```

Query the indexing of the different elements in a given network:
```@docs
PVIndexes
PQIndexes
SlackIndexes
GeneratorIndexes

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
ActiveLoad
ReactiveLoad

```

Function to get the range of a given value:
```@docs
bounds
```
