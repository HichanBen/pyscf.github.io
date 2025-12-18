#!/usr/bin/env python
"""
Geometry optimization for a copper complex using PySCF.

This example builds a Cu complex from user-supplied Cartesian coordinates,
sets up an unrestricted KS-DFT calculation with the B3LYP functional, and
performs a geometry optimization through :func:`pyscf.geomopt.optimize`.

The molecule has an odd number of electrons (185). If the spin multiplicity is
omitted or set inconsistently, PySCF raises the familiar
``Electron number ... and spin ... are not consistent`` error. The helper below
computes the electron count and automatically picks a spin value with the
correct parity (doublet for these coordinates) unless you override it
explicitly.

For production work, consider using a larger basis on the transition metal
(e.g. ``def2-tzvp`` for Cu) while keeping lighter atoms in a smaller basis
(e.g. ``def2-svp``) to control cost. For illustration, this example assigns
``def2-svp`` uniformly.
"""

from typing import Optional

from pyscf import dft, gto
from pyscf.geomopt import optimize


def _electron_count(atom_block: str, charge: int) -> int:
    """Return the total number of electrons implied by the atom block."""

    electrons = 0
    for line in atom_block.strip().splitlines():
        tokens = line.split()
        if not tokens:
            continue
        electrons += gto.charge(tokens[0])
    return electrons - charge


def build_cu_complex(spin: Optional[int] = None, charge: int = 0):
    """Construct the molecule from user-supplied Cartesian coordinates.

    Parameters
    ----------
    spin
        ``2S = N_alpha - N_beta``. If ``None`` (default), the spin is inferred
        to have the correct parity for the electron count implied by the
        coordinates and charge. Override only if you intentionally change the
        oxidation state.
    charge
        Total molecular charge. Keep at ``0`` unless you modify the complex to
        represent a charged species.
    """

    atom_coords = """
    C  -6.48761  -4.97242   1.48230
    C  -7.93727  -4.98602   0.94199
    H  -8.63742  -5.45562   1.66343
    H  -7.98292  -5.57699  -0.00369
    H  -8.30189  -3.96377   0.71400
    C  -6.08494  -6.43841   1.75677
    H  -6.79668  -6.90212   2.47714
    H  -5.07090  -6.50674   2.20062
    H  -6.10603  -7.04331   0.82404
    C  -5.57115  -4.30739   0.40727
    C  -4.09466  -4.36089   0.79896
    O  -3.21209  -4.93698  -0.06507
    H  -2.26810  -4.96945   0.14337
    O  -3.69987  -3.91890   1.84571
    N  -6.01231  -2.91451   0.17743
    H  -5.67484  -4.85544  -0.55208
    S  -6.34630  -4.02050   3.07487
    C  -5.52051   0.92555   2.46932
    C  -5.36110   2.35886   1.89397
    H  -5.77444   2.43361   0.86449
    H  -4.29831   2.67674   1.86222
    H  -5.90928   3.09786   2.52328
    C  -4.92834   0.91160   3.88738
    H  -3.85195   1.19773   3.83542
    H  -4.99420  -0.10280   4.32871
    H  -5.41240   1.64820   4.55757
    C  -7.05510   0.54715   2.36663
    C  -7.93117   1.41086   3.25636
    O  -7.82656   1.34945   4.47366
    N  -7.33659  -0.86298   2.65689
    H  -7.35549   0.71830   1.30860
    O  -8.84776   2.23713   2.71054
    H  -9.41545   2.79163   3.26665
    Cu -5.94420  -2.02580   1.91951
    S  -4.47973  -0.26743   1.51396
    H  -8.30389  -1.11923   2.35558
    H  -5.35116  -2.45939  -0.49326
    """

    nelec = _electron_count(atom_coords, charge)
    inferred_spin = nelec % 2
    spin = inferred_spin if spin is None else spin

    if (nelec + spin) % 2 != 0:
        raise ValueError(
            "Spin must have the same parity as the electron count "
            f"(electrons={nelec}, spin provided={spin}). "
            f"Suggested spin for these coordinates: {inferred_spin}."
        )

    mol = gto.M(
        atom=atom_coords,
        unit="Angstrom",
        spin=spin,  # 2S = N_alpha - N_beta
        charge=charge,
        basis={
            "Cu": "def2-svp",  # use def2-tzvp for Cu in production
            "default": "def2-svp",
        },
    )
    return mol


def optimize_cu_complex(spin: Optional[int] = None, charge: int = 0):
    """Set up a DFT calculation and perform geometry optimization.

    By default, ``spin`` is inferred from the electron count and therefore
    matches the provided coordinates (an open-shell doublet). Override only if
    you adjust the electron count (e.g., by adding/removing hydrogens) or wish
    to model a different oxidation state or charge.
    """

    mol = build_cu_complex(spin=spin, charge=charge)
    mf = dft.UKS(mol)
    mf.xc = "b3lyp"
    mf.kernel()

    mol_opt = optimize(mf)
    print("Optimized geometry (Angstrom):")
    for sym, coord in zip(
        mol_opt.atom_symbol(), mol_opt.atom_coords(unit="Angstrom")
    ):
        x, y, z = coord
        print(f"{sym:>2s}  {x: .5f}  {y: .5f}  {z: .5f}")


if __name__ == "__main__":
    optimize_cu_complex()
