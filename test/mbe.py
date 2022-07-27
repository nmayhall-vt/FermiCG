import numpy as np
import pyscf
import copy as cp
from pyscf import gto

body1_mol = []
body2_mol = []
body3_mol = []


mol = gto.Mole()
atom = [
        ["X",   (-1.0,       0.0,        0.0)],   
        ["X",   ( 1.0,       0.0,        0.0)],   
        ["X",   ( 0.0,      -1.0,        0.0)],   
        ["X",   ( 0.0,       1.0,        0.0)]   
        ]

basis = "aug-cc-pvdz"
mol.atom = atom
mol.basis={"X":gto.basis.load(basis, "He"), "He":basis}
mol.build()
S = mol.intor('int1e_ovlp_sph')

for i in range(len(atom)):
    mol = gto.Mole()
    atomi = cp.deepcopy(atom)
    atomi[i] = ("He", atom[i][1])
    mol.atom = atomi
    mol.basis={"X":gto.basis.load(basis, "He"), "He":basis}
    mol.build()
    body1_mol.append(((i,),mol))

for i in range(len(atom)):
    for j in range(i+1,len(atom)):
        mol = gto.Mole()
        atomi = cp.deepcopy(atom)
        atomi[i] = ("He", atom[i][1])
        atomi[j] = ("He", atom[j][1])
        mol.atom = atomi
        mol.basis={"X":gto.basis.load(basis, "He"), "He":basis}
        mol.build()
        body2_mol.append(((i,j),mol))

for i in range(len(atom)):
    for j in range(i+1,len(atom)):
        for k in range(j+1,len(atom)):
            mol = gto.Mole()
            atomi = cp.deepcopy(atom)
            atomi[i] = ("He", atom[i][1])
            atomi[j] = ("He", atom[j][1])
            atomi[k] = ("He", atom[k][1])
            mol.atom = atomi
            mol.basis={"X":gto.basis.load(basis, "He"), "He":basis}
            mol.build()
            body3_mol.append(((i,j,k),mol))

    

results = {} 

etot = 0.0

n_results = 3

for ind,mol in body1_mol:
    mf = mol.HF().run()
    myci = mf.CISD().run()
    print('RCISD tot energy        ', myci.e_tot)
    print('RCISD correlation energy', myci.e_corr)
    rdm1 = myci.make_rdm1()
    rdm2 = myci.make_rdm2()

    rdm1 = mf.mo_coeff @ rdm1 @ mf.mo_coeff.T
    i = ind[0]

    e = myci.e_tot

    #print(mol.atom)
    #print(mol.ao_labels())
    #print(rdm1)
    results[(i,)] = [e, rdm1, rdm2]



print(results.keys())
for ind,mol in body2_mol:
    mf = mol.HF().run()
    myci = mf.CISD().run()
    print('RCISD correlation energy', myci.e_corr)
    rdm1 = myci.make_rdm1()
    rdm2 = myci.make_rdm2()
    
    rdm1 = mf.mo_coeff @ rdm1 @ mf.mo_coeff.T
    
    i = ind[0]
    j = ind[1]

    results_curr  = [myci.e_tot, rdm1, rdm2]
    results_curr = [results_curr[x] - results[(i,)][x] for x in range(n_results)]
    results_curr = [results_curr[x] - results[(j,)][x] for x in range(n_results)]
    
    results[(i,j)] = results_curr 

for ind,mol in body3_mol:
    mf = mol.HF().run()
    myci = mf.CISD().run()
    print('RCISD correlation energy', myci.e_corr)
    rdm1 = myci.make_rdm1()
    rdm2 = myci.make_rdm2()
    
    rdm1 = mf.mo_coeff @ rdm1 @ mf.mo_coeff.T
    
    i = ind[0]
    j = ind[1]
    k = ind[2]

    results_curr  = [myci.e_tot, rdm1, rdm2]
    results_curr = [results_curr[x] - results[(i,)][x] for x in range(n_results)]
    results_curr = [results_curr[x] - results[(j,)][x] for x in range(n_results)]
    results_curr = [results_curr[x] - results[(k,)][x] for x in range(n_results)]
    results_curr = [results_curr[x] - results[(i,j)][x] for x in range(n_results)]
    results_curr = [results_curr[x] - results[(i,k)][x] for x in range(n_results)]
    results_curr = [results_curr[x] - results[(j,k)][x] for x in range(n_results)]
    
    results[(i,j,k)] = results_curr 


for f_ind,f_res in results.items():
    print("%12s %12.8f %12.8f"%(f_ind, f_res[0], np.trace(f_res[1] @ S)))




