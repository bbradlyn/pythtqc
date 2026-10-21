# How the modules fit together

The package is one path from a model to its invariants, and each step is one
module's. `realized_group` reads the space group the Hamiltonian obeys and builds
its representation; `hsp_table` reads the coreps a set of bands carries at every
high-symmetry point; and `decompose`, `symmetry_indicator` and the stable
real-space invariants read what that content says.

## The path a model takes

```mermaid
flowchart TD
    model(["a PythTB<br/>tight-binding model"])
    declared[/"what the crystal is: typat=,<br/>characters=, environment=,<br/>spin=, precision="/]
    tables[("irrep's tables")]
    invariants[("the tables of the stable<br/>real-space invariants")]

    groups["<b>groups</b><br/>realized_group:<br/>the space group<br/>the Hamiltonian obeys,<br/>and the group's tables"]
    spin["<b>spin</b><br/>spin_basis: which orbitals<br/>are one Wannier function's<br/>spin states"]
    symmetrization["<b>symmetrization</b><br/>the Hamiltonian<br/>projected onto its<br/>symmetric part"]
    symmetry["<b>symmetry</b><br/>each operation's matrix,<br/>and the atomic-limit table"]
    sewing["<b>sewing</b><br/>hsp_table: the bands'<br/>coreps at every<br/>high-symmetry point"]
    ebr["<b>ebr</b><br/>decompose,<br/>symmetry_indicator"]
    srsi["<b>srsi</b><br/>srsi_of_bands,<br/>srsi_of_model, splitting"]

    model --> groups
    declared --> groups
    tables --> groups
    model -. "spin in the<br/>orbital index" .-> spin -. "spin=" .-> groups
    groups <-. "precision=" .-> symmetrization
    groups -- "builds r.rep" --> symmetry
    symmetry -- "r.rep" --> sewing
    sewing -- "t" --> ebr
    sewing -- "t" --> srsi
    groups -- "r.group" --> ebr
    groups -- "r.group, r" --> srsi
    invariants --> srsi
```

Each box is a module, with the calls it holds. Every call after `realized_group`
takes what it returned, `r`: `hsp_table(r.rep, bands=n)` returns the table `t` of
the bands' coreps, which `decompose`, `symmetry_indicator` and `srsi_of_bands`
each read with the group, as `decompose(t, r.group, rep=r.rep)`, and
`srsi_of_model(r)` and `splitting(r, bands=n)` take the record whole. The group's
own tables are read from `r.group`: `high_symmetry_points(r.group)`,
`little_group`, `k_grid`, `corep_table` and `ebr_data`.

A dotted arrow is a step `realized_group` takes where a declaration asks for it:
with `spin=`, it reads which orbitals of a wannier90 spinor import are the spin
states of one Wannier function; with `precision=`, it projects the Hamiltonian
onto its symmetric part, where an operation holds only within that precision.

## Module by module

Each chart is one module. A rectangle is a call it holds, a box with double
sides a record a call returns, and a rounded box anything else a call takes or
returns: a model, a plain value, or a record another module returned. An arrow
labelled with a field, such as `.group`, reads that field of the record. A call
that cannot complete returns a `Refusal` in its record's place, which the charts
leave out; every record is read field by field in the
[records reference](../reference/records.md).

### `groups`

The space group a Hamiltonian obeys, and that group's tables in the model's own
cell ([what group does my Hamiltonian have?](../guide/group.md)).

```mermaid
flowchart TD
    g_model(["a PythTB model, and what the crystal is declared to be"])
    g_realized["realized_group(model, typat=..., characters=..., environment=..., spin=..., precision=...)"]
    g_r[["r, a RealizedGroup: the space group the Hamiltonian obeys, with each orbital's Wyckoff position and the representation that labels the bands"]]

    g_model --> g_realized --> g_r
    g_r --> g_fields
    g_fields -- "r.group" --> g_tables

    subgraph g_fields ["the records a RealizedGroup holds"]
        direction LR
        g_group[["r.group, GroupData: the space group, written in the model's own cell"]]
        g_assignment[["r.assignment, OrbitalAssignment: where the orbitals sit, and how the operations move their sites"]]
        g_validated[["r.validated, ValidatedGroup: the Hamiltonian's verdict on each operation, and how well each holds"]]
        g_rep[["r.rep, SpaceGroupRep: the group's representation on the orbitals, from the symmetry module"]]
    end

    subgraph g_tables ["calls on r.group"]
        direction LR
        g_hsp["high_symmetry_points(r.group)"] --> g_hsp_out[["HighSymmetryPoints: the tabulated high-symmetry points, in the model's reduced coordinates"]]
        g_lg["little_group(r.group, k)"] --> g_lg_out[["LittleGroup: the operations that leave k invariant up to a reciprocal lattice vector"]]
        g_kg["k_grid(r.group, n)"] --> g_kg_out[["KGrid: a Γ-centred mesh that respects the group, with the high-symmetry points it holds"]]
        g_ops["operations(r.group)"] --> g_ops_out[["OperationTable: every operation of the group by index, with its Seitz symbol"]]
        g_sites["wyckoff_sites_tables(r.group, positions)"] --> g_sites_out[["SiteReference, one per site: its Wyckoff position in the tables' setting"]]
        g_ct["corep_table(r.group, k)"] --> g_ct_out(["the character table of the little group at k"])
        g_ed["ebr_data(r.group)"] --> g_ed_out(["every elementary band representation of the group"])
        g_seitz["seitz(r.group, ind), operation_index(r.group, symbol)"] --> g_seitz_out(["an operation's Seitz symbol from its index, and its index from the symbol"])
    end
```

Two calls stand apart: `stack_presentation(model)` returns a layer repeated
along its normal with no hopping between the copies, a crystal with the layer's
bands at every `k_z`; and `tables_status()` says whether irrep's character tables
and ortools are installed.

### `symmetry`

The model's representation of its space group, and the atomic-limit table of its
orbitals ([how do I say what my orbitals are?](../guide/orbitals.md)).

```mermaid
flowchart TD
    s_realized(["r = realized_group(model), from the groups module"])
    s_rep[["r.rep, SpaceGroupRep: each operation's matrix on the orbitals"]]
    s_table[["AtomicLimitTable: each orbital's Wyckoff position and the site irrep it carries, a row per orbital"]]
    s_mult[["SiteMultiplets: the multiplets the Hamiltonian binds each site's orbitals into"]]

    s_realized -- "builds r.rep" --> s_rep
    s_rep -- ".characters" --> s_table
    s_rep -- ".multiplets" --> s_mult
    s_table --> s_parts
    s_rep --> s_calls
    s_parts ~~~ s_calls

    subgraph s_parts ["what an AtomicLimitTable holds"]
        direction LR
        s_row[["[i], AtomicLimitRow: orbital i's Wyckoff position, its site irrep, and how the irrep is known"]]
        s_br[[".band_representations, BandRepresentation: the band representation each orbit of orbitals induces"]]
        s_ms[[".multiplicity_spaces, MultiplicitySpace: a pseudospin the Hamiltonian conserves on an orbit of orbitals, and the site content assigned its copies"]]
    end

    subgraph s_calls ["calls on r.rep"]
        direction LR
        s_bloch["r.rep.bloch_operator(ind, k)"] --> s_bloch_out(["the Bloch operator of operation ind at k"])
        s_sew["r.rep.sewing_operator(ind, k)"] --> s_sew_out(["the Bloch operator of an operation that leaves k invariant, folded back to k"])
        s_little["r.rep.little_group(k), in_little_group(ind, k), transform_k(ind, k), k_action(ind)"] --> s_little_out(["which operations leave k invariant, and where each operation carries k"])
        s_cov["r.rep.covariance(k_pts)"] --> s_cov_out(["for each operation, the largest ‖D H(k) D† − H(Rk)‖ over the k-points, H(k) the Bloch Hamiltonian"])
    end
```

### `spin`

The spin a spinless model carries in its orbital index, as a wannier90 spinor
import does ([how does pythtqc read the spin of my model?](../guide/spin.md)).

```mermaid
flowchart TD
    p_spinful(["a spinful PythTB model"])
    p_import(["a spinless model whose orbital index carries spin, as a wannier90 spinor import's does"])
    p_present["spinless_presentation(model)"]
    p_basis["spin_basis(model, order=..., pairs=...)"]
    p_sb[["SpinBasis: which orbitals are the spin up and down states of one Wannier function"]]
    p_op["spin_operator(spin, n)"]
    p_op_out(["the spin along n̂, n̂·σ on each pair of orbitals a SpinBasis names"])
    p_realized(["realized_group(model, spin=...), from the groups module"])

    p_spinful --> p_present
    p_present --> p_import
    p_present --> p_sb
    p_import --> p_basis --> p_sb
    p_sb -- "spin" --> p_op --> p_op_out
    p_sb -- "spin=" --> p_realized
```

Given `spin="interleaved"`, `"blocked"` or the pairs, `realized_group` calls
`spin_basis` itself, and the representation keeps the basis on `r.rep.spin`.

### `symmetrization`

A model symmetric only to its declared precision, made exactly symmetric
([how do I measure a numerical or wannier90 model?](../guide/numerical.md)).

```mermaid
flowchart TD
    y_r(["r = realized_group(model), from the groups module"])

    y_r --> y_calls

    subgraph y_calls ["calls on the model and r"]
        direction LR
        y_classify["classify_operations(r.validated, tol)"] --> y_classes(["each operation's class against the tolerance: exact, near or broken"])
        y_positions["symmetrize_positions(model, r.group, r.assignment)"] --> y_moved(["the model with its orbital centres moved onto the nearest positions the group carries"])
        y_symmetrize["symmetrize(model, r.group, r.assignment, r.validated)"] --> y_exact(["the model nearest the one handed in that obeys the group exactly"])
        y_symmetrize --> y_sym[["Symmetrization: how far the model was moved onto its symmetric part, and why"]]
    end
```

`realized_group` takes all three steps itself where `precision=` asks, and keeps
the centres' correction on `r.positions` and the projection on
`r.symmetrization`.

### `sewing`

The coreps a set of bands carries at the high-symmetry points, read off its
eigenvectors ([which coreps do these bands carry?](../guide/labels.md)).

```mermaid
flowchart TD
    w_rep(["r.rep, from realized_group"])
    w_next(["decompose, symmetry_indicator and srsi_of_bands, which read t"])

    w_rep --> w_calls
    w_calls -- "t, from hsp_table" --> w_next
    w_next ~~~ w_tools

    subgraph w_calls ["calls on r.rep"]
        direction LR
        w_ht["hsp_table(r.rep, bands=n)"] --> w_HT[["t, an HSPTable: the coreps the bands carry at every high-symmetry point, their symmetry vector"]]
        w_cc["corep_content(r.rep, k, bands=n)"] --> w_CC[["CorepContent: the coreps the bands carry at one high-symmetry point, multiplet by multiplet"]]
        w_ch["characters(r.rep, k, bands=n)"] --> w_CT[["CharacterTable: the trace of each operation of the little group over each degenerate multiplet of the bands at k"]]
        w_sm["sewing_matrix(r.rep, op, k, bands=n)"] --> w_SM[["SewingMatrix: the matrix of one operation on the bands at a k it leaves invariant, B(g,k)"]]
    end

    subgraph w_tools ["calls on a spectrum"]
        direction LR
        w_solve["solve_at(model, k)"] --> w_solve_out(["the energies and eigenvectors at k, from one diagonalization"])
        w_mult["multiplets(energies, tol)"] --> w_mult_out(["which bands are one degenerate level, at the energy tolerance tol"])
        w_fmt["format_counts(counts)"] --> w_fmt_out(["the coreps and their multiplicities on one line, 2×-GM6 + -GM7"])
    end
```

An `HSPTable` holds the `CorepContent` read at each point on `.records`, and a
`CorepContent` the `CharacterTable` its labels were read from on `.table`.

### `ebr`

Whether a set of bands could be an atomic limit, and its symmetry-indicator class
([could these bands be an atomic limit?](../guide/atomic-limits.md),
[which symmetry-indicator class do these bands occupy?](../guide/indicators.md)).

```mermaid
flowchart TD
    e_t(["t, the HSPTable hsp_table returned"])
    e_r(["r.group and r.rep: the group and the representation t was read in"])

    e_t --> e_calls
    e_r --> e_calls
    e_calls ~~~ e_other

    subgraph e_calls ["calls on t, r.group and r.rep"]
        direction LR
        e_dec["decompose(t, r.group, rep=r.rep)"] --> e_D[["EBRDecomposition: every sum of elementary band representations that carries t's coreps, or none"]]
        e_si["symmetry_indicator(t, r.group, rep=r.rep)"] --> e_SI[["SymmetryIndicator: the class of t's symmetry vector among the band structures modulo the atomic limits"]]
        e_cc["completeness_check(t, r.group)"] --> e_CR[["CompletenessReport: whether t's coreps are a sum of elementary band representations, and how many bands each sum holds"]]
        e_rc["regular_completion(t, r.group, rep=r.rep)"] --> e_RC[["RegularCompletion: the bands that would complete t to whole copies of the regular band representation, and the orbitals they would be"]]
    end

    subgraph e_other ["calls on r.group alone, or on two tables"]
        direction LR
        e_ipr["in_plane_restriction(r.group)"] --> e_R[["Restriction: which coreps of the elementary band representations a layer measures"]]
        e_rows["ebr_rows(r.group)"] --> e_rows_out(["each elementary band representation of the group as a symmetry vector"])
        e_sub["subtract(t_a, t_b)"] --> e_sub_out(["the coreps of the bands in t_a and not in t_b"])
    end
```

`t` and `r.group` come from one `realized_group`: a table's labels name the
coreps of the group it was read in, and the table carries that representation as
`t.rep`, so a group or a `rep=` handed in beside it that is not the table's own
raises `ValueError`. `rep=r.rep` says how each site irrep is known
and, on a layer, which Wyckoff positions lie within it. An `EBRDecomposition` and
a `SymmetryIndicator` hold the `Restriction` they were read with on
`.restriction`, and a decomposition holds what it does not certify on `.caveat`,
a `WannierCaveat`.

### `srsi`

The stable real-space invariants: their tables, their values on a set of bands
and on a model, and the no-go criterion for a gapped splitting
([what do the stable real-space invariants say about these bands?](../guide/srsi.md)).

```mermaid
flowchart TD
    i_t(["t, the HSPTable hsp_table returned"])
    i_r(["r = realized_group(model)"])
    i_data[("the tables of the stable<br/>real-space invariants")]
    i_tables["srsi_tables(r.group)"]
    i_ST[["SRSITables: the stable real-space invariants of one grey space group, as the paper tabulates them"]]

    i_t --> i_band_calls
    i_r --> i_band_calls
    i_band_calls ~~~ i_tables
    i_data --> i_tables --> i_ST
    i_ST --> i_table_calls

    subgraph i_band_calls ["calls on a set of bands, or on a model"]
        direction LR
        i_bands["srsi_of_bands(t, r.group, rep=r.rep)"] --> i_B[["BandSRSIs: the ℤ-valued stable real-space invariants of the bands t counts"]]
        i_model["srsi_of_model(r)"] --> i_M[["ModelSRSIs: the model's stable real-space invariants, read off its orbitals"]]
        i_split["splitting(r, bands=n)"] --> i_S[["Splitting: whether the bands asked for and the model's other bands can both be atomic insulators"]]
    end

    subgraph i_table_calls ["calls on an SRSITables"]
        direction LR
        i_nsit["nsit_criterion(tables, θ_total, θ_Z_v, θ_Z_c)"] --> i_S2[["Splitting: whether the bands below and above a gap can both be atomic insulators, from their invariants alone"]]
        i_ar["atomic_realizations(tables, θ_Z)"] --> i_RZ[["Realizations: every atomic insulator with ℤ-valued invariants θ_Z, each a set of orbitals at maximal Wyckoff positions"]]
        i_pairs["stably_equivalent_pairs(tables)"] --> i_SP[["StablePair, one per pair: two site irreps at one maximal Wyckoff position that no stable real-space invariant distinguishes"]]
        i_of["srsi_of(tables, p), srsi_of_columns(tables, p)"] --> i_of_out(["(θ_Z, θ_Zn), the invariants of a configuration p of orbitals"])
    end
```

`nsit_criterion` is the test `splitting` makes, run on invariants already in
hand. A `Splitting` holds each side's `Realizations` on
`.realizations_v` and `.realizations_c`; a `BandSRSIs` the `SymmetryIndicator`
measured with it on `.indicator`; and a `ModelSRSIs` the `SRSITables` it was read
with on `.tables`.

## Beneath the path

Every module of the package, from the ground up: a module reads only the modules
above it in this table, the public ones among them.

| module | what it holds |
|---|---|
| `refusal` | `Refusal`, returned in place of a value a call cannot measure, and `InternalError`, raised where the package disagrees with itself |
| `_tolerance` | what a measurement judges noise against, resolved once |
| `_sitechars` | the characters of a site's symmetry group |
| `_bloch` | the Bloch Hamiltonian `H(k)`, summed from the model's hoppings |
| `spin` | which orbitals of a spinless model are one Wannier function's spin states, and the spin operator on them |
| `_text` | the words the records' printouts share: a coordinate, a k-point, a character, a count of operations, a sum of elementary band representations |
| `_spacegroup` | the space group of a cell, found on its point set and written in the model's own cell, with the records every measurement is written on |
| `_seitz` | each operation of a group by its index, with its Seitz symbol in the tables' setting and its name in the model's frame |
| `_tables` | irrep's tables in the model's own cell: the high-symmetry points, the little groups, the corep tables, the elementary band representations and the indicator group |
| `_subgroups` | the subgroups inside the operations the Hamiltonian keeps, and the tie rule that picks one where several of one order qualify |
| `_detector` | the eigen-sift detector: every matrix `X` with `X H(k) = H(Rk) X` on a grid of `k` |
| `_matrices` | the arithmetic of the operations' matrices: their composition, their phase, the Bloch operator, the members of a solution space inside given blocks |
| `_grammar` | the grammar of `characters=` and `multiplets=`, read where a call starts |
| `_centres` | the orbital centres against the group's action: where the operations put them, and how far they miss |
| `_naming` | site irreps named off the tables, each by the elementary band representation it induces |
| `symmetry` | the representation's records: each operation's matrix on the model's states, the atomic-limit table and each orbital's row |
| `_sectors` | the sectors the conserved matrices acting within sites split the states into |
| `_multiplets` | the multiplets the Hamiltonian binds each site's orbitals into, and the block of every operation on each |
| `_declaration` | what a declaration states, read on the measured multiplets, and the verdict per operation it gives |
| `_layers` | a layer's slab and its decoupled stack, the substrate under every site, and the check that the two environments agree |
| `sewing` | the coreps a set of bands carries at every high-symmetry point, read off its eigenvectors |
| `_smith` | the Smith normal form of the elementary band representations' matrix, and the compatibility relations read off it |
| `_srsitables` | the tables of the stable real-space invariants as shipped, and the paper's names for the symmetry indicators |
| `ebr` | whether a content could be an atomic limit, and its symmetry-indicator class |
| `srsi` | the stable real-space invariants of a set of bands and of a model, and the no-go criterion on a gapped splitting |
| `_labelling` | the labelling the Hamiltonian leaves open, chosen once by the lowest-angular-momentum rule, and the atomic-limit table read off it |
| `_measure` | `H(k)` on the detection grid and one solution space per k-action: the verdict on each operation of a group |
| `_build` | the representation, built once on what the verdict stage measured |
| `symmetrization` | a model symmetric only to its declared precision, made exactly symmetric |
| `_descent` | the unitary descent: one verdict per operation of the cell, and the cell lowered to the operations that hold |
| `_realized` | the space group a Hamiltonian obeys, measured round after round until it settles, and the record `realized_group` returns |
| `groups` | the space group a Hamiltonian obeys and the group's tables in the model's cell: `realized_group`, and the calls on `r.group` |
| `models` | the crystals the pages measure |
| `utils` | `save_figure`, which writes the examples' figures |

| outside the package | what it gives |
|---|---|
| `pythtb` | the models, and the hoppings the Bloch Hamiltonian is summed from |
| `irrep`, `irreptables` | the space groups, their coreps, their elementary band representations and their symmetry-indicator tables |
| `ortools` | the integer solver that finds every sum of elementary band representations |
| `spglib`, `sympy` | each site's Wyckoff letter, and the Wyckoff positions' coordinates read as expressions |
