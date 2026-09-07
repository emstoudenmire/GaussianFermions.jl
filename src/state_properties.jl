import LinearAlgebra as la

"""
    correlation_matrix(ϕ::GaussianState; labels=labels(ϕ))

Compute the single-particle correlation matrix ``C_{ij} = \\langle c^\\dagger_i c_j \\rangle``
for the state `ϕ`, given by ``C_{ij} = \\sum_n \\bar{d}^n_i \\, \\eta_n \\, d^j_n``
where ``d^j_n`` are the orbitals and ``\\eta_n`` the occupancy values.

For a pure state (all ``\\eta_n \\in \\{0,1\\}``), the correlation matrix is a
projector: ``C^2 = C``.

If `labels` is given, only the submatrix for those labels is returned.

# Example

```julia
import GaussianFermions as gf

H = gf.GaussianOperator(4)
for j in 1:3
    H = gf.add_hop(H, j, j + 1, -1.0)
end
_, ϕ = gf.ground_state(H; Nf = 2)
C = gf.correlation_matrix(ϕ)
```
"""
function correlation_matrix(ϕ::GaussianState; labels = labels(ϕ))
    orbs = Matrix(orbitals(ϕ)[labels, :])
    C = trace(ϕ) * (conj.(orbs) * la.Diagonal(occupancy(ϕ)) * transpose(orbs))
    return NamedArray(C, (labels, labels), ("Labels", "Labels"))
end

"""
    density(ϕ::GaussianState; labels=labels(ϕ))

Return a vector of site occupation numbers ``\\langle n_i \\rangle`` (the diagonal
of the correlation matrix).
These are computed relative to the normalized state, so as ⟨ϕ|n̂_i|ϕ⟩/⟨ϕ|ϕ⟩.

# Example

```julia
import GaussianFermions as gf

H = gf.GaussianOperator(4)
for j in 1:3
    H = gf.add_hop(H, j, j + 1, -1.0)
end
_, ϕ = gf.ground_state(H; Nf = 2)
gf.density(ϕ)
```
"""
function density(ϕ::GaussianState; kws...)
    return la.diag(correlation_matrix(ϕ; kws...)) / trace(ϕ)
end

"""
    nparticles(ϕ::GaussianState)

Return the number of particles in the state `ϕ`.
"""
function nparticles(ϕ::GaussianState; tol = 1.0e-3)
    ispure(ϕ) || error("nparticles currently only defined for pure Gaussian states")
    tot_density = sum(density(ϕ))
    npart = round(Int, tot_density)
    if abs(npart - tot_density) > tol
        error("State does not have an integer number of particles")
    end
    return npart
end

function inner(ϕ::GaussianState, ψ::GaussianState; tol = 1.0e-6)
    if !(ispure(ϕ) && ispure(ψ))
        error("`inner` currently implemented for pure states only")
    end
    M = orbitals(ϕ)' * orbitals(ψ)
    ϕset = findall(ν -> isapprox(1.0, ν; atol = tol), occupancy(ϕ))
    ψset = findall(ν -> isapprox(1.0, ν; atol = tol), occupancy(ψ))
    return la.det(M[ϕset, ψset]) * norm(ϕ) * norm(ψ)
end

"""
    entanglement(ϕ::GaussianState, labels)

Compute the von Neumann entanglement entropy of the subsystem defined by `labels`.

The entropy is obtained from the eigenvalues ``\\nu_k`` of the reduced correlation
matrix ``C_{AA}`` (the block of the correlation matrix restricted to `labels`):

```math
S = -\\sum_k \\bigl[\\nu_k \\ln \\nu_k + (1 - \\nu_k) \\ln(1 - \\nu_k)\\bigr]
```

Eigenvalues near 0 or 1 correspond to orbitals fully empty or full within the
subsystem and do not contribute to entanglement.

# Example

```julia
import GaussianFermions as gf

H = gf.GaussianOperator(10)
for j in 1:9
    H = gf.add_hop(H, j, j + 1, -1.0)
end
_, ϕ = gf.ground_state(H; Nf = 5)
gf.entanglement(ϕ, 1:5)
```
"""
function entanglement(ϕ::GaussianState, labels)
    C = correlation_matrix(ϕ; labels)
    occs, _ = la.eigen(C)
    Svn = 0.0
    @assert la.norm(imag(occs)) < 1.0e-8
    for ν in real(occs)
        if real(ν) > 0.0
            Svn += -ν * log(ν)
        end
        if real(ν) < 1.0
            Svn += -(1 - ν) * log(1 - ν)
        end
    end
    return Svn
end

"""
    bond_dimension(ϕ::GaussianState, labels, cutoff::Real)

Compute the matrix product state (MPS) bond dimension needed to
represent the state `ϕ` bipartitioned into modes given by `labels`
(the labels on one side of the cut) and the complementary modes
to an accuracy given by the `cutoff`.

The bond dimension is determined by discarding the smallest
density matrix eigenvalues such that their sum is below `cutoff`.

Returns a tuple of the bond dimension and the truncation error
incurred by truncating to this bond dimension.

# Example

```julia
import GaussianFermions as gf

N = 10
H = gf.GaussianOperator(N)
for j in 1:(N - 1)
    H += -1, "C†", j, "C", j+1
    H += -1, "C†", j+1, "C", j
end
E0, ϕ0 = gf.ground_state(H; Nf = 5)
region_labels = 1:(N ÷ 2)
gf.bond_dimension(ϕ0, region_labels, 1e-7)
```
"""
function bond_dimension(ϕ::GaussianState, labels, cutoff::Real; perturbation_strength=0.0)
    C = correlation_matrix(ϕ; labels)

    if !iszero(perturbation_strength)
        eps = cutoff*perturbation_strength
        M = randn(size(C)...)
        C = C + eps*(M'*M)
    end

    occs, _ = la.eigen(C)
    occs = real(occs)
    occs = sort(occs; by = ν -> abs(ν - 1 / 2))
    n = length(occs)

    # Build spectrum top down
    # by doubling size of eigs
    eigs = [1.0]
    truncerr = 0.0
    for j in 1:n
        νj = occs[j]
        eigs = vcat(eigs .* (1 - νj), eigs .* νj)
        if length(eigs) > 2^18
            error(
                @sprintf(
                    "Excessively large bond dimension > 2^18 in `bond_dimension`. Please use a larger cutoff (cutoff was %.4E).",
                    cutoff
                )
            )
        end
        eigs = sort(eigs; rev = true)
        remainder = 1.0
        for r in (j + 1):n
            remainder *= max(occs[r], 1 - occs[r])
        end
        truncerr = 1 - remainder * sum(eigs)
        (truncerr < cutoff) && break
    end
    χ = length(eigs)

    # Refine bond dimension
    while (truncerr + eigs[χ] < cutoff) && (χ > 1)
        truncerr += eigs[χ]
        χ -= 1
    end

    return χ, truncerr
end
