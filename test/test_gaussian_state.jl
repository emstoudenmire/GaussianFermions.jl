import GaussianFermions as gf
using LinearAlgebra: eigvals, norm
using Test

include("utilities/hamiltonians.jl")

@testset "Ground State" begin
    # Test ground state of a 1D chain
    N = 10
    H = fermion_chain_h(N)
    Nf = N ÷ 2
    E0, ϕ0 = gf.ground_state(H; Nf)
    @test gf.expect(H, ϕ0) ≈ E0

    # 1D electron chain
    H = electron_chain_h(N)
    Nf = N ÷ 2
    E0, ϕ0 = gf.ground_state(H; Nf)
    @test gf.expect(H, ϕ0) ≈ E0

    # Test for 2D square lattice
    H_graph = square_lattice_h(4)
    Nsites = length(gf.labels(H_graph))
    Nf_graph = Nsites ÷ 2
    E0_graph, ϕ0_graph = gf.ground_state(H_graph; Nf = Nf_graph)
    @test gf.expect(H_graph, ϕ0_graph) ≈ E0_graph
    @test gf.labels(ϕ0_graph) == gf.labels(H_graph)
end

@testset "Entanglement and Bond Dimension" begin
    N = 10
    H = fermion_chain_h(N)
    E0, ϕ0 = gf.ground_state(H; Nf = N ÷ 2)

    region_A = 1:(N ÷ 2)
    @test gf.entanglement(ϕ0, region_A) > 0

    cutoff = 1.0e-7
    χ, trunc_err = gf.bond_dimension(ϕ0, region_A, cutoff)
    @test trunc_err < cutoff
    @test 10 < χ < 30

    # The reduced occupations are the eigenvalues of the reduced correlation
    # matrix, whichever of the two equivalent matrices was diagonalized: a region
    # smaller than the orbital count (1:3) and one larger than it (4:10).
    for region in (1:3, 4:N)
        ν = gf.reduced_occupations(ϕ0, region)
        ν_direct = real(eigvals(Matrix(gf.correlation_matrix(ϕ0; labels = region))))
        @test sort(filter(x -> x > 1.0e-12, ν)) ≈ sort(filter(x -> x > 1.0e-12, ν_direct))
        @test gf.entanglement(ν) ≈ gf.entanglement(ϕ0, region)
        @test gf.bond_dimension(ν, cutoff) == gf.bond_dimension(ϕ0, region, cutoff)
    end
    # A pure state is equally entangled on either side of a cut.
    @test gf.entanglement(ϕ0, 1:3) ≈ gf.entanglement(ϕ0, 4:N)
    @test first(gf.bond_dimension(ϕ0, 1:3, cutoff)) ==
        first(gf.bond_dimension(ϕ0, 4:N, cutoff))

    # Mixed states go through the same helper.
    ϕβ = gf.thermal_state(H, 2.0)
    ν_direct = real(eigvals(Matrix(gf.correlation_matrix(ϕβ; labels = 4:N))))
    @test sort(gf.reduced_occupations(ϕβ, 4:N)) ≈ sort(ν_direct)
end
