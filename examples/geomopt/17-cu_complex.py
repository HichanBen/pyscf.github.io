from pyscf import gto, dft

geom = """
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

# ---- choose these carefully ----
charge = 0
spin   = 1   # spin = (Nalpha - Nbeta). For a doublet, spin=1. If singlet, spin=0.

mol = gto.M(
    atom=geom,
    unit='Angstrom',
    charge=charge,
    spin=spin,
    # Good starter basis for organometallic:
    basis={'default': 'def2-SVP', 'Cu': 'def2-TZVP'},
    # If you later decide to use an ECP for Cu, we can switch basis/ecp accordingly.
    verbose=4
)

mf = dft.UKS(mol) if spin != 0 else dft.RKS(mol)
mf.xc = 'B3LYP'          # common DFT functional; alternatives: 'PBE0', 'TPSSh', 'M06'
mf.grids.level = 3       # integration grid quality (3-5 typical)
mf.max_cycle = 200

e_scf = mf.kernel()
print("SCF Energy (Eh):", e_scf)

# ---- stability analysis ----
# If unstable, PySCF can return a better wavefunction guess.
stable_info = mf.stability()
# stable_info is typically (mo_coeff, mo_occ, ...) depending on method
new_mf = mf.newton()  # more robust convergence (optional)
new_mf.kernel()
print("After Newton Energy (Eh):", new_mf.e_tot)
