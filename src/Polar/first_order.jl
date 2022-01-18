
struct Jacobian{Model, Func, VD, SMT, MT, VI, VP}
    model::Model
    func::Func
    map::VI
    stack::NetworkStack{VD}
    compressedJ::MT
    coloring::VI
    t1sseeds::VP
    t1sF::VD
    J::SMT
end

function Base.show(io::IO, jacobian::Jacobian)
    println(io, "A AutoDiff Jacobian for $(jacobian.func)")
    ncolor = size(jacobian.compressedJ, 1)
    print(io, "Number of Jacobian colors: ", ncolor)
end

Base.size(jac::Jacobian, n::Int) = size(jac.J, n)

# Coloring
function jacobian_sparsity(polar::PolarForm, func::AbstractExpression)
    nbus = get(polar, PS.NumberOfBuses())
    v = polar.network.vbus .+ 0.01 .* rand(ComplexF64, nbus)
    return matpower_jacobian(polar, func, v)
end

function get_jacobian_colors(polar::PolarForm, func::AbstractExpression, map::Vector{Int})
    # Sparsity pattern
    J = jacobian_sparsity(polar, func)
    Jsub = J[:, map]
    # Coloring
    colors = AutoDiff.SparseDiffTools.matrix_colors(Jsub)
    return (Jsub, colors)
end

function Jacobian(polar::PolarForm{T, VI, VT, MT}, func::AbstractExpression, map::Vector{Int}) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func)

    nmap = length(map)
    map_device = map |> VI

    J_host, coloring = get_jacobian_colors(polar, func, map)
    ncolor = size(unique(coloring),1)

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    VD = A{t1s{ncolor}}

    J = J_host |> SMT

    # Structures
    stack = NetworkStack(nbus, ngen, nlines, VD)
    t1sF = zeros(Float64, n_cons) |> VD

    # Seedings
    t1sseeds = AutoDiff.init_seed(coloring, ncolor, nmap)

    # Move the seeds over to the device, if necessary
    gput1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(t1sseeds)
    compressedJ = MT(zeros(Float64, ncolor, n_cons))
    coloring = coloring |> VI

    return Jacobian(
        polar, func, map_device, stack, compressedJ, coloring, gput1sseeds, t1sF, J,
    )
end

function jacobian!(
    jac::Jacobian, stack,
)
    # init
    jac.stack.input .= stack.input
    jac.t1sF .= 0.0
    # seed
    AutoDiff.seed!(jac.stack, stack, jac.t1sseeds, jac.map, jac.model.device)
    # forward pass
    jac.func(jac.t1sF, jac.stack)
    # uncompress
    AutoDiff.partials_jac!(jac.compressedJ, jac.t1sF, jac.model.device)
    AutoDiff.uncompress_kernel!(jac.J, jac.compressedJ, jac.coloring, jac.model.device)
    return jac.J
end

