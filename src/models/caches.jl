abstract type AbstractBuffer end
abstract type AbstractNetworkBuffer <: AbstractBuffer end

"Store indexing on target device"
struct IndexingCache{IVT} <: AbstractBuffer
    index_pv::IVT
    index_pq::IVT
    index_ref::IVT
    index_generators::IVT
    index_pv_to_gen::IVT
    index_ref_to_gen::IVT
end

"Store current state of the network, in term of physical values."
struct PolarNetworkState{VT} <: AbstractNetworkBuffer
    vmag::VT # nb
    vang::VT # nb
    pinj::VT # nb
    qinj::VT # nb
    pg::VT # ng
    qg::VT # ng
    balance::VT
    dx::VT
end

setvalues!(buf::PolarNetworkState, ::PS.VoltageMagnitude, values) = copyto!(buf.vmag, values)
setvalues!(buf::PolarNetworkState, ::PS.VoltageAngle, values) = copyto!(buf.vang, values)
setvalues!(buf::PolarNetworkState, ::PS.ActivePower, values) = copyto!(buf.pg, values)
setvalues!(buf::PolarNetworkState, ::PS.ReactivePower, values) = copyto!(buf.qg, values)

"Store topology of the network on target device."
struct NetworkTopology{VTI, VTD}
    # Bus admittance matrix
    ybus_re::Spmat{VTI, VTD} # nb x nb
    ybus_im::Spmat{VTI, VTD} # nb x nb
    # Branches admittance matrix
    ## Real part
    yff_re::VTD # nl
    yft_re::VTD # nl
    ytf_re::VTD # nl
    ytt_re::VTD # nl
    ## Imag part
    yff_im::VTD # nl
    yft_im::VTD # nl
    ytf_im::VTD # nl
    ytt_im::VTD # nl
    # Correspondence
    f_buses::VTI # nl
    t_buses::VTI # nl
end

function NetworkTopology{VTI, VTD}(pf::PS.PowerNetwork) where {VTI, VTD}
    ybus_re, ybus_im = Spmat{VTI, VTD}(pf.Ybus)
    lines = pf.lines
    yff_re = real.(lines.Yff) |> VTD
    yft_re = real.(lines.Yft) |> VTD
    ytf_re = real.(lines.Ytf) |> VTD
    ytt_re = real.(lines.Ytt) |> VTD

    yff_im = imag.(lines.Yff) |> VTD
    yft_im = imag.(lines.Yft) |> VTD
    ytf_im = imag.(lines.Ytf) |> VTD
    ytt_im = imag.(lines.Ytt) |> VTD

    f = lines.from_buses |> VTI
    t = lines.to_buses   |> VTI

    return NetworkTopology(
        ybus_re, ybus_im,
        yff_re, yft_re, ytf_re, ytt_re,
        yff_im, yft_im, ytf_im, ytt_im,
        f, t,
    )
end

get(net::NetworkTopology, ::PS.BusAdmittanceMatrix) = (net.ybus_re, net.ybus_im)

