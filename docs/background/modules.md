# How the modules fit together

pythTQC takes a model along one path, from its Hamiltonian to its invariants.
Each step is a call of one public module. `realized_group` returns the space
group the Hamiltonian obeys, with the representation of the group on the
orbitals. `hsp_table` returns the multiplicities of the little-group
representations a set of bands carries at every high-symmetry point, its
symmetry data vector. `decompose`, `symmetry_indicator` and `srsi_of_bands`
read that symmetry data vector. `srsi_of_model` reads the model's orbitals, and
`splitting` reads both.

## The path a model takes

```mermaid
flowchart TD
    model(["a PythTB<br/>tight-binding model"])
    declared[/"what the crystal is: typat=,<br/>characters=, environment=,<br/>spin=, precision="/]
    tables[("the tables of the<br/>irreptables package")]
    invariants[("the tables of the stable<br/>real-space invariants")]

    groups["<b>groups</b><br/>realized_group:<br/>the space group<br/>the Hamiltonian obeys,<br/>and the group's tables"]
    spin["<b>spin</b><br/>spin_basis: which orbitals<br/>are one spinful orbital's<br/>spin states"]
    symmetrization["<b>symmetrization</b><br/>the Hamiltonian<br/>projected onto its<br/>symmetric part"]
    symmetry["<b>symmetry</b><br/>each operation's matrix,<br/>and the atomic-limit table"]
    sewing["<b>sewing</b><br/>hsp_table: the bands'<br/>little-group representations<br/>at every high-symmetry point"]
    ebr["<b>ebr</b><br/>decompose,<br/>symmetry_indicator"]
    srsi["<b>srsi</b><br/>srsi_of_bands,<br/>srsi_of_model, splitting"]

    model --> groups
    declared --> groups
    tables --> groups
    model -. "spin in the<br/>orbital index" .-> spin -. "spin=" .-> groups
    groups <-. "H symmetric only<br/>within the tolerance" .-> symmetrization
    groups -- "builds r.rep" --> symmetry
    groups -- "r" --> sewing
    sewing -- "t" --> ebr
    sewing -- "t" --> srsi
    groups -- "r.group" --> ebr
    groups -- "r.group, r" --> srsi
    invariants --> srsi
```

Each box is a module, with the calls it holds. Each call after `realized_group`
takes what it returned, `r`, or a record read off it. `hsp_table(r, bands=n)`
returns `t`, the table of the little-group representations the bands carry.
`decompose(t)`, `symmetry_indicator(t)` and `srsi_of_bands(t)` take the table
alone, which carries its group. `srsi_of_model(r)` and `splitting(r, bands=n)`
take `r` whole. The tables of the group itself are read from `r.group`:
`high_symmetry_points(r.group)`, `little_group`, `k_grid`, `corep_table` and
`ebr_data`.

A dotted arrow is a step `realized_group` takes only where a declaration or the
model asks for it. With `spin=`, it reads which orbitals of a Wannier90 spinor
import are the spin states of one spinful orbital. `precision=` sets the
tolerance within which an operation is judged to hold, for a numerical model.
With no precision declared, the model is read as exact, and the tolerance is
`1e-8` times the largest entry of `H(k)`, the mean on-site energy subtracted.
Where the Hamiltonian is symmetric only within the tolerance, `realized_group`
projects it onto its symmetric part.

## Module by module

Each chart is one module. A rectangle is a call the module holds, or a field
read off a record. A box with double sides is a record a call returns. A
rounded box is anything else a call takes or returns: a model, a plain value,
or a record another module returned. A cylinder is a table the package ships or
reads. An arrow labeled with a field, such as `.wyckoff_orbits`, reads that
field of the record. A call that cannot complete returns a `Refusal` in place
of its record, and the charts leave it out. [Returned
records](../reference/records.md) lists the fields of every record.

### `groups`

The space group a Hamiltonian obeys, and the tables of that group, written in
the model's own cell, the one its pythtb Lattice specifies
([What group does my Hamiltonian have?](../guide/group.md)).

```mermaid
flowchart TD
    g_model(["a PythTB model, and what the crystal is declared to be"])
    g_realized["realized_group(model, typat=..., characters=..., environment=..., spin=..., precision=...)"]
    g_r[["r, a RealizedGroup: the space group the Hamiltonian obeys, with the Wyckoff position of each orbital and the representation of the group on the orbitals"]]

    g_model --> g_realized --> g_r
    g_r --> g_fields
    g_fields -- "r.group" --> g_tables

    subgraph g_fields ["the records a RealizedGroup holds"]
        direction LR
        g_group[["r.group, GroupData: the space group, written in the model's own cell"]]
        g_orbits[["r.wyckoff_orbits, AtomicLimitTable: the orbitals by Wyckoff orbit, and the site irreps each carries, from the symmetry module"]]
        g_validated[["r.numerical.validated, ValidatedGroup: whether the Hamiltonian obeys each operation of the group, as measured"]]
        g_rep[["r.rep, SpaceGroupRep: the group's representation on the orbitals, from the symmetry module"]]
    end

    subgraph g_tables ["calls on r.group"]
        direction LR
        g_hsp["high_symmetry_points(r.group)"] --> g_hsp_out[["HighSymmetryPoints: the tabulated high-symmetry points, in the model's reduced coordinates"]]
        g_lg["little_group(r.group, k)"] --> g_lg_out[["LittleGroup: the operations that leave k invariant up to a reciprocal lattice vector"]]
        g_kg["k_grid(r.group, n)"] --> g_kg_out[["KGrid: a Γ-centered mesh that respects the group, with the high-symmetry points it holds"]]
        g_ops["operations(r.group)"] --> g_ops_out[["OperationTable: one operation for each coset of the lattice translations, by index, with its Seitz symbol"]]
        g_sites["wyckoff_sites_tables(r.group, positions)"] --> g_sites_out[["SiteReference, one per site: its Wyckoff position in the setting of the irreptables package"]]
        g_ct["corep_table(r.group, k)"] --> g_ct_out(["the character table of the little group at k"])
        g_ed["ebr_data(r.group)"] --> g_ed_out[["EBRData: every elementary band representation of the group"]]
        g_seitz["operation_index(r.group, seitz)"] --> g_seitz_out(["an operation's index from its Seitz symbol"])
    end
```

Two calls of the module stand apart from the path. `stack_presentation(model)`
returns a layer repeated along its normal with no hopping between the copies.
The stack is a crystal with the layer's bands at every `k_z`. `tables_status()`
says whether the character tables of the irreptables package are in place, and
whether ortools is installed.

### `symmetry`

The model's representation of its space group, and the atomic-limit table of
its orbitals ([How do I say what my orbitals are?](../guide/orbitals.md)).

```mermaid
flowchart TD
    s_realized(["r = realized_group(model), from the groups module"])
    s_rep[["r.rep, SpaceGroupRep: each operation's matrix on the orbitals"]]
    s_table[["r.wyckoff_orbits, AtomicLimitTable: each Wyckoff orbit of orbitals and the site irreps it carries, a row per Wyckoff orbit"]]

    s_realized -- "builds r.rep" --> s_rep
    s_realized -- ".wyckoff_orbits" --> s_table
    s_table --> s_parts
    s_rep --> s_calls
    s_parts ~~~ s_calls

    subgraph s_parts ["what a row of the AtomicLimitTable holds"]
        direction LR
        s_row[["[i], WyckoffOrbit: one Wyckoff orbit's orbitals, the site irreps they carry, and whether those were declared, fixed by the Hamiltonian, or chosen by the lowest-angular-momentum rule"]]
        s_br[[".band_representation, BandRepresentation: the band representation the orbit induces"]]
        s_mult([".site_multiplets: each site's multiplets, the sets of its orbitals that the operations fixing the site mix"])
    end

    subgraph s_calls ["calls on r.rep"]
        direction LR
        s_bloch["r.rep.bloch_operator(ind, k)"] --> s_bloch_out(["the Bloch operator of operation ind at k"])
        s_sew["r.rep.sewing_operator(ind, k)"] --> s_sew_out(["the Bloch operator of an operation that leaves k invariant, folded back to k"])
        s_little["r.rep.little_group(k), in_little_group(ind, k), transform_k(ind, k)"] --> s_little_out(["which operations leave k invariant, and where each operation carries k"])
        s_cov["r.rep.covariance(kpoints)"] --> s_cov_out(["for each operation, the largest modulus of an entry of D H(k) D† − H(Rk) over the k-points, D its matrix and H(k) the Bloch Hamiltonian"])
    end
```

### `spin`

The spin of a model whose spin states are listed as separate orbitals, as in a
Wannier90 spinor import
([How does pythTQC read the spin of my model?](../guide/spin.md)).

```mermaid
flowchart TD
    p_spinful(["a spinful PythTB model"])
    p_import(["a model whose spin states are listed as separate orbitals, as in a Wannier90 spinor import"])
    p_present["spinless_presentation(model)"]
    p_basis["spin_basis(model, order=..., pairs=...)"]
    p_sb[["SpinBasis: which orbitals are the spin up and down states of one spinful orbital"]]
    p_op["spin_operator(spin, n)"]
    p_op_out(["the spin along n̂, n̂·σ in units of ħ/2, on each pair of orbitals a SpinBasis names"])
    p_realized(["realized_group(model, spin=...), from the groups module"])

    p_spinful --> p_present
    p_present --> p_import
    p_present --> p_sb
    p_import --> p_basis --> p_sb
    p_sb -- "spin" --> p_op --> p_op_out
    p_sb -- "spin=" --> p_realized
```

`spin=` takes `"interleaved"`, `"blocked"`, a mapping of the arguments of
`spin_basis`, or a `SpinBasis`. `realized_group` calls `spin_basis` itself
where it is not handed a `SpinBasis`, and keeps the basis on `r.spin`.

### `symmetrization`

A Hamiltonian symmetric only within the tolerance, projected onto its symmetric
part ([How do I measure a numerical or Wannier90
model?](../guide/numerical.md)).

```mermaid
flowchart TD
    y_r(["r = realized_group(model, precision=p), from the groups module"])

    y_r --> y_fields

    subgraph y_fields ["fields of r"]
        direction LR
        y_positions["r.numerical.positions"] --> y_moved[["Positions: the orbital centers, moved onto the nearest positions the group carries exactly"]]
        y_symmetrization["r.numerical.symmetrization"] --> y_sym[["Symmetrization: how far H is from its symmetric part, and what holds only within the tolerance"]]
    end
```

`position_tol=` declares how far an orbital center may miss the position the
group puts it at and still be moved onto it, in reduced coordinates.
`realized_group` moves the orbital centers where `precision=` or
`position_tol=` is declared. It keeps the move on `r.numerical.positions`. It
projects the Hamiltonian onto its symmetric part where an operation or a site's
multiplet holds only within the tolerance, or where the hoppings couple
separate parts of the crystal or tell identical copies apart only below it. It
reports the projection on `r.numerical.symmetrization`. Each field is `None`
where its step was not taken.

### `sewing`

The little-group representations a set of bands carries at the high-symmetry
points, read off its eigenvectors
([Which little-group representations do these bands carry?](../guide/labels.md)).

```mermaid
flowchart TD
    w_rep(["r, from realized_group"])
    w_next(["decompose, symmetry_indicator and srsi_of_bands, which read t"])

    w_rep --> w_calls
    w_calls -- "t, from hsp_table" --> w_next
    w_next ~~~ w_tools

    subgraph w_calls ["calls on r"]
        direction LR
        w_ht["hsp_table(r, bands=n)"] --> w_HT[["t, an HSPTable: the multiplicities of the little-group representations the bands carry at every high-symmetry point, their symmetry data vector"]]
        w_cc["coreps_at(r, k, bands=n)"] --> w_CC[["CorepsAt: the little-group representations the bands carry at one high-symmetry point, multiplet by multiplet"]]
        w_ch["characters(r, k, bands=n)"] --> w_CT[["BandCharacters: the trace of each unitary operation of the little group over each degenerate multiplet of the bands at k"]]
        w_sm["sewing_matrix(r, operation, k, bands=n)"] --> w_SM[["SewingMatrix: the matrix B(g,k) of one operation on the bands, at a k it leaves invariant"]]
    end

    subgraph w_tools ["calls on a spectrum"]
        direction LR
        w_solve["solve_at(model, k)"] --> w_solve_out(["the energies and eigenvectors at k, from one diagonalization"])
        w_mult["multiplets(energies, degeneracy_tol)"] --> w_mult_out(["which bands are one degenerate level, at the energy tolerance degeneracy_tol"])
        w_fmt["format_coreps(coreps)"] --> w_fmt_out(["the multiplicities of the little-group representations, on one line"])
    end
```

`t.multiplets` holds, at each point, the rows `coreps_at` returns there: each
degenerate multiplet with the little-group representations it carries.

### `ebr`

Whether a set of bands could be an atomic limit, and its symmetry-indicator
class ([Could these bands be an atomic limit?](../guide/atomic-limits.md),
[Which symmetry-indicator class do these bands
occupy?](../guide/indicators.md)).

```mermaid
flowchart TD
    e_t(["t, the HSPTable hsp_table returned, carrying its group and the site irreps its labels rest on"])

    e_t --> e_calls
    e_calls ~~~ e_other

    subgraph e_calls ["calls on t"]
        direction LR
        e_dec["decompose(t)"] --> e_D[["EBRDecomposition: the sums of elementary band representations whose symmetry data vector is t's, or none"]]
        e_si["symmetry_indicator(t)"] --> e_SI[["SymmetryIndicator: the class of t's symmetry data vector among the band structures modulo the atomic limits"]]
        e_rc["regular_completion(t, bands=n)"] --> e_RC[["RegularCompletion: the bands that would complete t to copies of the regular band representation, and the orbitals whose band representation they would be"]]
    end

    subgraph e_other ["calls on r.group alone, or on two tables"]
        direction LR
        e_ipr["in_plane_restriction(r.group)"] --> e_R[["Restriction: the little-group representations at the points measured, and for each one at the other points, the measured ones that occur as often as it in every elementary band representation"]]
        e_rows["ebr_rows(r.group)"] --> e_rows_out(["each elementary band representation of the group as a symmetry data vector"])
        e_sub["subtract(t_a, t_b)"] --> e_sub_out(["the symmetry data vector of the bands in t_a and not in t_b"])
    end
```

An HSPTable names the little-group representations of the group it was read
in. It carries that group as `t.group`, and the site irrep of each orbital with
whether each was declared, fixed by the Hamiltonian, or chosen by the
lowest-angular-momentum rule. The names of its little-group representations
depend on these site irreps. So each call on it takes the table alone. A
decomposition returns those site irreps with its sums. A table handed in with
`r` beside it raises `TypeError`. A symmetry data vector written out by hand is
passed with the `r` whose group names its labels:
`decompose(symmetry_data, r)`. For a layer, the table carries the orbitals'
heights, and a decomposition keeps only the elementary band representations
at Wyckoff positions within the layer. An `EBRDecomposition` and a
`SymmetryIndicator` hold the `Restriction` they were read with on
`.numerical.restriction`.

### `srsi`

The stable real-space invariants: their tables, their values on a set of bands
and on a model, and whether the bands on the two sides of a gap can both be
atomic insulators ([What do the stable real-space invariants say about these
bands?](../guide/srsi.md)). The invariants and their tables are those of
Hwang *et al.*, [*Stable real-space invariants and topology beyond symmetry
indicators*, Nat. Commun. **17**, 8564
(2026)](https://doi.org/10.1038/s41467-026-74844-w).

```mermaid
flowchart TD
    i_t(["t, the HSPTable hsp_table returned"])
    i_r(["r = realized_group(model)"])
    i_data[("the tables of the stable<br/>real-space invariants")]
    i_tables["srsi_tables(r.group)"]
    i_ST[["SRSITables: the stable real-space invariants of one grey space group, as Hwang et al. tabulate them"]]

    i_t --> i_band_calls
    i_r --> i_band_calls
    i_band_calls ~~~ i_tables
    i_data --> i_tables --> i_ST
    i_ST --> i_table_calls

    subgraph i_band_calls ["calls on a set of bands, or on a model"]
        direction LR
        i_bands["srsi_of_bands(t)"] --> i_B[["BandSRSIs: the integer-valued stable real-space invariants of the bands t counts"]]
        i_model["srsi_of_model(r)"] --> i_M[["ModelSRSIs: the stable real-space invariants of the model, a row for each Wyckoff orbit"]]
        i_split["splitting(r, bands=n)"] --> i_S[["Splitting: whether the bands asked for and the model's other bands can both be atomic insulators"]]
    end

    subgraph i_table_calls ["calls and fields of an SRSITables"]
        direction LR
        i_nsit["nsit_criterion(tables, theta_total, theta_Z_v, theta_Z_c)"] --> i_S2[["Splitting: whether the bands below and above a gap can both be atomic insulators, from the invariants of all the bands, theta_total, and the integer-valued ones of each side, theta_Z_v and theta_Z_c"]]
        i_ar["atomic_realizations(tables, theta_Z)"] --> i_RZ[["Realizations: every atomic insulator with the integer-valued invariants theta_Z, each a set of site irreps at maximal Wyckoff positions"]]
        i_pairs["tables.stably_equivalent"] --> i_SP[["StablePair, one per pair: two site irreps at one maximal Wyckoff position that no stable real-space invariant distinguishes"]]
        i_of["tables.theta(p)"] --> i_of_out[["Invariants: the stable real-space invariants of a site-symmetry vector p, by name"]]
    end
```

`nsit_criterion` is the test `splitting` makes, run on invariants already in
hand. A `Splitting` holds each side's `Realizations` on `.valence` and
`.conduction`. A `BandSRSIs` holds the `SymmetryIndicator` measured with it on
`.indicator`.

## Beneath the path

Every module of the package, public and private, from the ground up. When a
module is loaded, it imports only modules above it in this table.

| module | what it holds |
|---|---|
| `refusal` | `Refusal`, returned in place of a value a call cannot measure; `RefusalError`, a refusal raised as an exception by `Refusal.raise_()`; `InternalError`, raised where the package disagrees with itself; and `MagneticScopeWarning`, issued where time reversal is measured broken |
| `_tolerance` | what a measurement of crystalline symmetry judges noise against |
| `_sitechars` | the character table of a site's symmetry group, computed from the measured group, and the site irrep each orbital shorthand such as `px` carries |
| `_bloch` | `H(k)` of a tight-binding model, summed from the hoppings of each lattice vector |
| `spin` | the spin of a model whose spin states are listed as separate orbitals, as in a Wannier90 spinor import, and the spin operator on it |
| `_text` | the words the records' printouts share: a coordinate, a k-point, a character, a count of operations, a set of orbitals, and a sum of elementary band representations |
| `_spacegroup` | the space group of a cell, found from its positions and written in the model's own cell, with the records every later measurement is written on |
| `_seitz` | the operations of a group by index, each with its Seitz symbol, written in the conventional cell used by the irreptables package, and its name, written in the Cartesian axes in which the pythtb Lattice is specified |
| `_tables` | the tables of the irreptables package in the model's own cell: the high-symmetry points, the little group of a k-point, the character table of its little-group representations, the elementary band representations, the indicator group, and a mesh that respects the group |
| `_subgroups` | the subgroups of a group inside a set of its operations, and the tie rule that picks one where several of the largest order qualify |
| `_detector` | every matrix `X` with `X H(k) = H(Rk) X` on a grid of `k`, the candidates for an operation's matrix |
| `_matrices` | the arithmetic of the operations' matrices: their composition, their phase convention, the Bloch operator of an operation, and the irreducible representations of a group with a factor system |
| `_grammar` | the grammar of `characters=` and `site_multiplets=`, read where a call starts and before anything is measured |
| `_centers` | the orbital centers against the group's action on them: where the operations carry each site, how far the centers miss, and the least move onto positions the operations carry exactly |
| `_site_names` | site irreps named by the tables of the stable real-space invariants |
| `_naming` | site irreps named from the tables of the irreptables package, each by the elementary band representation it induces from its Wyckoff position |
| `symmetry` | the model's representation of its space group, and the atomic-limit table of its orbitals |
| `_sectors` | the sectors of the states, each a set of states the hoppings couple to nothing else, taken with its identical copies, read off the matrices that commute with `H(k)` at every `k` and act within sites |
| `_multiplets` | each site's multiplets, the sets of its orbitals that the operations fixing the site mix, and the block of every operation on each multiplet |
| `_declaration` | what a declaration states, read on the multiplets the Hamiltonian measures: each site's declared site irreps, and one verdict per operation |
| `_measure` | whether a Hamiltonian obeys each operation of a group, as measured |
| `_layers` | a layer's presentations, its slab and its decoupled stack; its two environments, on a substrate and free-standing; and the comparison of the groups the two give |
| `sewing` | the little-group representations a set of bands carries at the high-symmetry points, read off its eigenvectors |
| `_smith` | the integer arithmetic of a group's matrix of elementary band representations: its Smith normal form, and the compatibility relations |
| `_srsitables` | the tables of the stable real-space invariants of Hwang et al., as shipped, with the paper's names for the symmetry indicators |
| `ebr` | whether bands could be an atomic limit, and their symmetry-indicator class |
| `srsi` | the stable real-space invariants: the tables, their values on bands and models, and whether the bands on the two sides of a gap can both be atomic insulators |
| `_labeling` | the site irreps the Hamiltonian leaves open, chosen by the lowest-angular-momentum rule, and the atomic-limit table read off the choice |
| `_build` | the representation, built on what the measurement of each operation found: one matrix per operation on the model's states |
| `symmetrization` | a Hamiltonian symmetric only within the tolerance, projected onto its symmetric part |
| `_descent` | the group of the cell lowered to the group the Hamiltonian obeys |
| `_realized` | the space group a Hamiltonian obeys, and the record `realized_group` returns |
| `groups` | the space group a Hamiltonian obeys, and the tables of that group |
| `models` | tight-binding crystals to measure on, each chosen to make one question sharp |
| `utils` | `save_figure`, which writes the examples' figures so that a rerun writes the same file |

| outside the package | what it gives |
|---|---|
| `pythtb` | the models, and the hoppings `H(k)` is summed from |
| `irrep`, `irreptables` | the space groups, their little-group representations, their elementary band representations and their symmetry-indicator tables |
| `ortools` | the integer solver that finds every sum of elementary band representations |
| `spglib` | the space group of a cell from its positions, and each site's Wyckoff letter |
| `sympy` | the coordinates of the Wyckoff positions, read as expressions |
