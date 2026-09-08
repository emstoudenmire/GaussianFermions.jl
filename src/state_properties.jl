import LinearAlgebra as la
using DataStructures: BinaryMaxHeap

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
    reduced_occupations(ϕ::GaussianState, labels) -> Vector{Float64}

The eigenvalues ``\\nu_k`` of the reduced correlation matrix ``C_{AA}`` on the
subsystem `labels`: the occupations of its natural orbitals, which fix every
entanglement property of a Gaussian state. They are clamped to ``[0, 1]``,
where rounding can otherwise put them a hair outside.

For ``x`` labels and ``N`` orbitals the ``x \\times x`` matrix
``C_{AA} = t \\bar{\\Phi}_A \\eta \\Phi_A^T`` and the ``N \\times N`` matrix
``t \\sqrt{\\eta} \\overline{\\Phi_A^\\dagger \\Phi_A} \\sqrt{\\eta}`` share their
nonzero spectrum, so the smaller one is diagonalized; the eigenvalues dropped
that way are exactly zero and contribute to nothing.
"""
function reduced_occupations(ϕ::GaussianState, labels)
    Φ = Matrix(orbitals(ϕ)[labels, :])
    η = la.Diagonal(occupancy(ϕ))
    M = if size(Φ, 1) <= size(Φ, 2)
        conj.(Φ) * η * transpose(Φ)
    else
        sqrt(η) * conj.(Φ' * Φ) * sqrt(η)
    end
    return clamp.(la.eigvals(la.Hermitian(trace(ϕ) * M)), 0.0, 1.0)
end

"""
    entanglement(ϕ::GaussianState, labels)
    entanglement(ν::AbstractVector{<:Real})

Compute the von Neumann entanglement entropy of the subsystem defined by `labels`,
or directly from the eigenvalues `ν` of its reduced correlation matrix as returned
by [`reduced_occupations`](@ref) -- which lets one diagonalization serve both this
and [`bond_dimension`](@ref).

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
entanglement(ϕ::GaussianState, labels) = entanglement(reduced_occupations(ϕ, labels))
entanglement(ν::AbstractVector{<:Real}) = sum(binary_entropy, ν; init = 0.0)

binary_entropy(ν) = (ν > 0 ? -ν * log(ν) : 0.0) + (ν < 1 ? -(1 - ν) * log(1 - ν) : 0.0)

"""
    bond_dimension(ϕ::GaussianState, labels, cutoff::Real)
    bond_dimension(ν::AbstractVector{<:Real}, cutoff::Real)

Compute the matrix product state (MPS) bond dimension needed to
represent the state `ϕ` bipartitioned into modes given by `labels`
(the labels on one side of the cut) and the complementary modes
to an accuracy given by the `cutoff`: the smallest number of
Schmidt weights that can be kept with the discarded ones summing
to less than `cutoff`. The second form takes the eigenvalues `ν`
of the reduced correlation matrix directly, as returned by
[`reduced_occupations`](@ref).

Returns a tuple of the bond dimension and the truncation error
incurred by truncating to this bond dimension.

Each Schmidt weight is a product over the natural orbitals of the
subsystem, orbital `k` contributing either its dominant branch
`max(νₖ, 1 - νₖ)` or its minor one `min(νₖ, 1 - νₖ)`. The weights
are generated in descending order by a best-first search from the
all-dominant state, so only the ones kept are ever visited.

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
function bond_dimension(ϕ::GaussianState, labels, cutoff::Real)
    return bond_dimension(reduced_occupations(ϕ, labels), cutoff)
end

function bond_dimension(ν::AbstractVector{<:Real}, cutoff::Real)
    # Flipping orbital k to its minor branch multiplies the weight by the ratio of
    # its minor to its dominant branch. Orbitals that cannot be flipped at all are
    # dropped, and the rest sorted so that each flip costs at least as much as the
    # one before it.
    ratio = sort(filter(>(0), [min(x, 1 - x) / max(x, 1 - x) for x in ν]); rev = true)
    n = length(ratio)

    # A state is (weight, k) with k its highest flipped orbital, 0 for none. Its
    # children flip k + 1 in addition to k (extend) or instead of k (shift). Every
    # set of flipped orbitals descends from the all-dominant state by exactly one
    # sequence of these moves, and neither move increases the weight, so a max-heap
    # of the frontier yields the weights in descending order.
    heap = BinaryMaxHeap([(prod(max.(ν, 1 .- ν); init = 1.0), 0)])
    χ, kept = 0, 0.0
    while 1 - kept >= cutoff && !isempty(heap)
        weight, k = pop!(heap)
        χ += 1
        kept += weight
        k < n && push!(heap, (weight * ratio[k + 1], k + 1))
        0 < k < n && push!(heap, (weight * ratio[k + 1] / ratio[k], k + 1))
    end
    return χ, 1 - kept
end
