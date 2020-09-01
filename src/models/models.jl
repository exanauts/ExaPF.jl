export PolarForm, get, bounds, powerflow
export State, Control, Parameters, NumberOfState, NumberOfControl


abstract type AbstractFormulation end

abstract type AbstractFormAttribute end
struct NumberOfState <: AbstractFormAttribute end
struct NumberOfControl <: AbstractFormAttribute end

abstract type AbstractVariable end

struct State <: AbstractVariable end
struct Control <: AbstractVariable end
struct Parameters <: AbstractVariable end

# Templates
# TODO: add documentation
function get end

function bounds end

function power_balance end

function initial end

function powerflow end

# Generic constraints
function state_constraints end
function power_constraints end
function thermal_limit_constraints end

# Polar formulation
include("polar.jl")
