# Background

This page summarizes the key ideas behind Gaussian fermion states and connects
them to the functions provided by GaussianFermions.jl. For full derivations, see
the detailed notes in `docs/notes/gaussian_fermions.pdf` in the repository.

## Gaussian States (Slater Determinants)

A Gaussian state, also known as a Slater determinant, is a many-body fermionic
state built by filling a set of single-particle orbitals:

```math
|\Psi\rangle = \hat{d}^\dagger_{1} \hat{d}^\dagger_{2} \cdots \hat{d}^\dagger_{N_f} |0\rangle
```

where each ``\hat{d}^\dagger_m = \sum_j d^j_m \, \hat{c}^\dagger_j`` creates a
fermion in the orbital defined by the column vector ``d^j_m``. Here ``\hat{c}^\dagger_j``
creates a fermion with label ``j``. One can think of the label ``j`` as a site of a lattice,
a combined site-spin label, or a more abstract label such as a momentum or vertex of a graph.

In GaussianFermions.jl, a Gaussian state is represented by a
[`GaussianState`](@ref GaussianFermions.GaussianState), which stores the orbital
matrix ``d^j_m`` (accessed via [`orbitals`](@ref GaussianFermions.orbitals)) and
an occupancy vector ``\eta_m`` (accessed via [`occupancy`](@ref GaussianFermions.occupancy))
indicating which orbitals are occupied.

## Mode Labels

The indices labelling the fermionic modes are flexible. They can be:

- **Integers** `1, 2, 3, …` for spinless fermions on a chain.
- **Tuples** `(1,1), (1,2), …` for fermions on a two-dimensional lattice.
- **Spin labels** using [`Up(j)`](@ref GaussianFermions.Up) and [`Dn(j)`](@ref GaussianFermions.Dn) for
  spinful (e.g. electron) systems, where the label encodes both site and spin.

This convention mirrors the physics: all derivations in the Gaussian fermion
formalism apply equally to spinless and spinful fermions in any spatial dimension,
with the understanding that the indices ``i, j`` run over all degrees of freedom
(site, spin, etc.).

## Quadratic Operators

A [`GaussianOperator`](@ref GaussianFermions.GaussianOperator) represents a
quadratic (non-interacting) fermion operator of the form

```math
\hat{\mathcal{O}} = \sum_{i,j} O_{ij}\, \hat{c}^\dagger_i \hat{c}_j
```

where ``O_{ij}`` is the matrix of coefficients (accessed via
[`matrix_elements`](@ref GaussianFermions.matrix_elements)). Important special
cases include Hamiltonians with hopping terms built via [`add_hop`](@ref GaussianFermions.add_hop).

## The Correlation Matrix

The central quantity for Gaussian states is the single-particle correlation matrix

```math
C_{ij} = \langle \hat{c}^\dagger_i \hat{c}_j \rangle = \sum_n \bar{d}^n_i \, \eta_n \, d^j_n
```

computed by [`correlation_matrix`](@ref GaussianFermions.correlation_matrix).
For a pure state (all ``\eta_n \in \{0,1\}``), the correlation matrix is a
projector: ``C^2 = C``. All physical properties of a Gaussian state can be
extracted from ``C``.

The diagonal elements give the site occupation numbers
``\langle n_i \rangle = C_{ii}``, computed by
[`density`](@ref GaussianFermions.density).

## Expectation Values

The expectation value of any quadratic operator ``\hat{\mathcal{O}}`` in a
Gaussian state is given by a simple trace:

```math
\langle \hat{\mathcal{O}} \rangle = \mathrm{Tr}[O \, C]
```

This is computed by [`expect`](@ref GaussianFermions.expect).

## Ground States and Energy Levels

A quadratic Hamiltonian ``\hat{H} = \sum_{ij} h_{ij}\, \hat{c}^\dagger_i \hat{c}_j``
is diagonalized by finding the eigenstates of the hopping matrix ``h_{ij}``:

```math
h_{ij} = \sum_n \phi^i_n \, \epsilon_n \, \bar{\phi}^n_j
```

The many-body eigenstates are formed by filling subsets of these single-particle
levels, and their energies are simply the sums of the filled single-particle
energies:

```math
E = \sum_n \eta_n \, \epsilon_n
```

The ground state of ``N_f`` fermions fills the ``N_f`` lowest-energy levels.
This is computed by [`ground_state`](@ref GaussianFermions.ground_state), which
calls [`energies_states`](@ref GaussianFermions.energies_states) to diagonalize
the hopping matrix.

## Time Evolution

Under a quadratic Hamiltonian, Gaussian states remain Gaussian for all time.
The orbitals evolve by the single-particle propagator:

```math
d^j_m(t) = \sum_{j'} g^j_{j'}(t)\, d^{j'}_m \qquad \text{where} \qquad g(t) = e^{-i h t}
```

Equivalently, the correlation matrix evolves as

```math
C(t) = g(t)\, C(0)\, g(t)^\dagger
```

The function [`time_evolve`](@ref GaussianFermions.time_evolve) implements this
for both states (Schrödinger picture) and operators (Heisenberg picture).

## Entanglement Entropy

For a bipartition of the system into regions ``A`` and ``B``, the entanglement
entropy can be computed entirely from the eigenvalues ``\nu_k`` of the reduced
correlation matrix ``C_{AA}`` (the block of ``C`` restricted to region ``A``):

```math
S = -\sum_k \bigl[\nu_k \ln \nu_k + (1 - \nu_k) \ln(1 - \nu_k)\bigr]
```

Eigenvalues near 0 or 1 correspond to orbitals that are fully empty or fully
occupied within the subsystem and do not contribute to entanglement.
Only "active" modes with ``\nu_k`` away from 0 and 1 contribute.

This is computed by [`entanglement`](@ref GaussianFermions.entanglement).
The related function [`bond_dimension`](@ref GaussianFermions.bond_dimension) uses
the same eigenvalues to compute the MPS bond dimension needed to represent the
state across the given bipartition to a given truncation error.

## Further Reading

Detailed derivations of all the results above, including the fermionic vector
space formalism, commutation relations, Green's functions, and the Schmidt
decomposition of Gaussian states, can be found in the notes bundled with the
repository at `docs/notes/gaussian_fermions.pdf`.
