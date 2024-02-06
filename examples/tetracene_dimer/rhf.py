from functools import reduce
import numpy as np
import scipy

import pyscf
from pyscf import fci
from pyscf import gto, scf, ao2mo, lo, tdscf, cc


def tda_denisty_matrix(td, state_id):
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the density
    matrix (in AO basis) of the excited states
    '''
    cis_t1 = td.xy[state_id][0]
    dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

    # The ground state density matrix in mo_basis
    mf = td._scf
    dm = np.diag(mf.mo_occ)

    # Add CIS contribution
    nocc = cis_t1.shape[0]
    # Note that dm_oo and dm_vv correspond to spin-up contribution. "*2" to
    # include the spin-down contribution
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm

mol = gto.Mole()
mol.atom = '''
C          3.86480       -1.02720        1.27180
C          3.60170       -0.34020        0.07040
C          2.93000       -1.06360        2.30730
C          4.54410       -0.28950       -0.97720
C          2.32170        0.33090       -0.09430
C          3.20030       -1.73210        3.54060
C          1.65210       -0.39240        2.14450
C          4.27330        0.38740       -2.16700
C          2.05960        1.01870       -1.29550
C          1.37970        0.28170        0.95340
C          2.27260       -1.73270        4.55160
C          0.71620       -0.41900        3.22490
C          2.99590        1.05910       -2.32920
C          5.20900        0.41800       -3.24760
C          1.01870       -1.06880        4.39470
C          2.72630        1.73410       -3.55840
C          4.90670        1.07300       -4.41500
C          3.65410        1.73950       -4.56900
C         -3.86650        1.01870       -1.29550
C         -2.93020        1.05910       -2.32920
C         -3.60440        0.33090       -0.09430
C         -3.19980        1.73410       -3.55840
C         -1.65280        0.38740       -2.16700
C         -2.32430       -0.34020        0.07040
C         -4.54630        0.28170        0.95340
C         -2.27200        1.73950       -4.56900
C         -0.71710        0.41800       -3.24760
C         -1.38200       -0.28950       -0.97720
C         -2.06130       -1.02720        1.27180
C         -4.27400       -0.39240        2.14450
C         -1.01930        1.07300       -4.41500
C         -2.99610       -1.06360        2.30730
C         -5.20980       -0.41900        3.22490
C         -2.72580       -1.73210        3.54060
C         -4.90730       -1.06880        4.39470
C         -3.65350       -1.73270        4.55160
H          4.82300       -1.53290        1.39770
H          5.49910       -0.80290       -0.85660
H          4.15900       -2.23700        3.66390
H          1.10180        1.52560       -1.42170
H          0.42460        0.79440        0.83100
H          2.50000       -2.24040        5.48840
H         -0.23700        0.09640        3.10140
H          6.16210       -0.09790       -3.12730
H          0.29870       -1.07700        5.21470
H          1.76850        2.24120       -3.67870
H          5.62580        1.08320       -5.23570
H          3.42730        2.25190       -5.50340
H         -4.82430        1.52560       -1.42170
H         -4.15760        2.24120       -3.67870
H         -5.50150        0.79440        0.83100
H         -2.49880        2.25190       -5.50340
H          0.23610       -0.09790       -3.12730
H         -0.42700       -0.80290       -0.85660
H         -1.10300       -1.53290        1.39770
H         -0.30030        1.08320       -5.23570
H         -6.16310        0.09640        3.10140
H         -1.76710       -2.23700        3.66390
H         -5.62740       -1.07700        5.21470
H         -3.42610       -2.24040        5.48840
'''


mol.basis = '6-31g*'
mol.spin = 0
mol.build()

#mf = scf.ROHF(mol).x2c()
mf = scf.RHF(mol)
mf.verbose = 4
mf.get_init_guess(mol, key='minao')
mf.conv_tol = 1e-9
#mf.level_shift = .1
#mf.diis_start_cycle = 4
#mf.diis_space = 10
mf.run(max_cycle=200)


n_triplets = 2
n_singlets = 2

avg_rdm1 = mf.make_rdm1()


# compute singlets
mytd = tdscf.TDA(mf)
mytd.singlet = True 
mytd = mytd.run(nstates=n_singlets)
mytd.analyze()
for i in range(mytd.nroots):
    avg_rdm1 += tda_denisty_matrix(mytd, i)

# compute triplets 
mytd = tdscf.TDA(mf)
mytd.singlet = False 
mytd = mytd.run(nstates=n_triplets)
mytd.analyze()
for i in range(mytd.nroots):
    avg_rdm1 += tda_denisty_matrix(mytd, i)

# normalize
avg_rdm1 = avg_rdm1 / (n_singlets + n_triplets + 1)


S = mf.get_ovlp()
F = mf.get_fock()
np.save("fock_mat18", F)
np.save("overlap_mat18", S)
np.save("density_mat18", mf.make_rdm1())
np.save("rhf_mo_coeffs18", mf.mo_coeff)
np.save("cis_sa_density_mat18", avg_rdm1)


