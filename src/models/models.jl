

abstract type AbstractFormulation end

abstract type AbstractFormAttribute end
struct NumberOfState <: AbstractFormAttribute end
struct NumberOfControl <: AbstractFormAttribute end

abstract type AbstractVariable end

struct State <: AbstractVariable end
struct Control <: AbstractVariable end
struct Parameters <: AbstractVariable end

# Templates
function get end

function bounds end

function power_balance end

function initial end

function powerflow end

#
include("polar.jl")
