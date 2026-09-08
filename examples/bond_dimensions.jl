
import GaussianFermions as gf
using ITensorMPS
using Statistics: median
using ITensors: dim, flux, svd

let
    N = 20
    Nf = N ÷ 2
    t = 1.0
    trunc_cutoff = 1.0e-8

    H = gf.GaussianOperator(N)
    for j in 1:(N - 1)
        H += -t, "C†", j, "C", j + 1
        H += -t, "C†", j + 1, "C", j
    end

    E0, ϕ0 = gf.ground_state(H; Nf)
    @show E0

    entanglement = zeros(N-1)
    χs = zeros(Int,N-1)
    for j=1:(N-1)
        region_A = 1:j

        entanglement[j] = gf.entanglement(ϕ0, 1:j)

        χs[j], trunc_error = gf.bond_dimension(ϕ0, 1:j, trunc_cutoff)
    end


    #
    # Check with ITensorMPS DMRG
    #

    h = OpSum()
    for j in 1:(N - 1)
        h += -t, "Cdag", j, "C", j + 1
        h += -t, "Cdag", j + 1, "C", j
    end
    sites = siteinds("Fermion", N; conserve_qns = true)
    H = MPO(h, sites)

    ψ0 = MPS(sites, [isodd(j) ? "1" : "0" for j in 1:N])

    nsweeps = 20
    maxdim = [10, 20, 40, 80, 160, 320]
    cutoff = 1.0e-12
    E0, ψ = dmrg(H, ψ0; nsweeps, cutoff, maxdim)

    ψtrunc = truncate(ψ; cutoff = trunc_cutoff)

    χs_dmrg = linkdims(ψtrunc)


    #
    # Report
    #
    println("Bond dimension GaussianState:")
    println(χs)
    @show median(χs)

    println("Bond dimension DMRG:")
    println(χs_dmrg)
    @show median(χs_dmrg)

    return nothing
end
