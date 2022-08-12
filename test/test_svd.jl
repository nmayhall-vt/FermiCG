using LinearAlgebra
using FermiCG
using Printf
using Test
using Arpack

using NPZ


atoms = []
push!(atoms,Atom(1,"H",[0, 0, 0]))
push!(atoms,Atom(2,"H",[0, 1,-1]))
push!(atoms,Atom(3,"H",[0, 1, 1]))
push!(atoms,Atom(4,"H",[0, 2, 0]))
push!(atoms,Atom(5,"H",[0, 4, 0]))
push!(atoms,Atom(6,"H",[0, 5,-1]))
push!(atoms,Atom(7,"H",[0, 5, 1]))
push!(atoms,Atom(8,"H",[0, 6, 0]))
#basis = "6-31g"
basis = "sto-3g"

na = 4
nb = 4
frag = 4

mol     = Molecule(0,1,atoms,basis)
mf = FermiCG.pyscf_do_scf(mol)
nbas = size(mf.mo_coeff)[1]
ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,na,nb)
# @printf(" FCI Energy: %12.8f\n", e_fci)

FermiCG.pyscf_write_molden(mol,mf.mo_coeff,filename="scf.molden")

C = mf.mo_coeff
Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
S = FermiCG.get_ovlp(mf)
U =  C' * S * Cl
println(" Build Integrals")
flush(stdout)
ints = FermiCG.orbital_rotation(ints,U)
println(" done.")
flush(stdout)


norb = size(ints.h1)[1]
ansatz = FCIAnsatz(norb, na, nb)

display(ansatz)

Hmat = build_H_matrix(ansatz)
EIG = eigen(Hmat)
v = EIG.vectors
e = EIG.values

v = v[:,1]



basis = svd_state(v,ansatz,frag,norb-frag,1e-6)
