from functools import reduce
import numpy as np
import scipy

import pyscf
from pyscf import fci, tools
from pyscf import gto, scf, ao2mo, lo, tdscf, cc
from orbitalpartitioning import *

def get_natural_orbital_active_space(rdm, S, thresh=.01):
   

    Ssqrt = scipy.linalg.sqrtm((S+S.T)/2.0)
    Sinvsqrt = scipy.linalg.inv(Ssqrt)

    print(" Number of electrons found %12.8f" %np.trace(S@rdm))

    Dtot = Ssqrt.T @ rdm @ Ssqrt
    #Dtot = Ssqrt.T @ ( da + db) @ Ssqrt
    D_evals, D_evecs = np.linalg.eigh((Dtot+Dtot.T)/2.0)

    sorted_list = np.argsort(D_evals)[::-1]
    D_evals = D_evals[sorted_list]
    D_evecs = D_evecs[:,sorted_list]

    act_list = []
    doc_list = []


    for idx,n in enumerate(D_evals):
        print(" %4i = %12.8f" %(idx,n),end="")
        if n < 2.0 - thresh:
            if n > thresh:
                act_list.append(idx)
                print(" Active")
            else:
                print(" Virt")
        else:
            doc_list.append(idx)
            print(" DOcc")

    print(" Number of active orbitals: ", len(act_list))
    print(" Number of doc    orbitals: ", len(doc_list))

    D_evecs = Sinvsqrt @ D_evecs
    Cdoc = D_evecs[:, doc_list]
    Cact = D_evecs[:, act_list]
    return Cdoc, Cact 





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

np.save("xyz.npy", mol.atom)

mol.basis = '6-31g*'
mol.spin = 0
mol.build()

mf = scf.RHF(mol).density_fit()
mf.verbose = 4
mf.get_init_guess(mol, key='minao')
mf.conv_tol = 1e-9

# load precomputed data
C = np.load("./rhf_mo_coeffs18.npy")
avg_rdm1 = np.load("./cis_sa_density_mat18.npy")

S = mf.get_ovlp()

print(avg_rdm1.shape)
print(S.shape)

print(" Number of electrons found %12.8f" %np.trace(S@avg_rdm1))

Cdoc, Cact = get_natural_orbital_active_space(avg_rdm1, S, thresh=.008)
print(Cdoc.shape)
print(Cact.shape)

# localize
Cact = pyscf.lo.PM(mol).kernel(Cact, verbose=4);
pyscf.tools.molden.from_mo(mol, "Cact.molden", Cact)

# Pseudo canonicalize fragments
frag1 = [0,3,5,7,8,11]
frag2 = [1,2,4,6,9,10]
Cfrags = []
Cfrags.append(Cact[:,frag1])
Cfrags.append(Cact[:,frag2])


# Pseudo canonicalize fragments
F = np.load("./fock_mat18.npy")
Cact = canonicalize(Cfrags, F)
np.save("Cfrags", Cact)

Cact = np.hstack(Cact)
pyscf.tools.molden.from_mo(mol, "Cact12_pseudo_can.molden", Cact)
# First get the density from the doubly occupied orbitals 
# to include in our effective 1 body operator
d1_embed = 2 * Cdoc @ Cdoc.T

h0 = pyscf.gto.mole.energy_nuc(mol)
h  = pyscf.scf.hf.get_hcore(mol)
j, k = pyscf.scf.hf.get_jk(mol, d1_embed, hermi=1)

h0 += np.trace(d1_embed @ ( h + .5*j - .25*k))

# Rotate 1electron terms to active space
h = Cact.T @ h @ Cact
j = Cact.T @ j @ Cact;
k = Cact.T @ k @ Cact;

h1 = h + j - .5*k;

# form 2e integrals in active space
nact = h.shape[0]
h2 = pyscf.ao2mo.kernel(mol, Cact, aosym="s4", compact=False)
h2.shape = (nact, nact, nact, nact)

np.save("integrals_h0_12", h0)
np.save("integrals_h1_12", h1)
np.save("integrals_h2_12", h2)
np.save("mo_coeffs_act_12", Cact)
np.save("mo_coeffs_doc_12", Cdoc)



