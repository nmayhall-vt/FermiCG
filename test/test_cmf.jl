using NPZ
using JSON
using Random
using LinearAlgebra
using FermiCG


atoms = []
push!(atoms,Atom(1,"H",[0,0,0]))
push!(atoms,Atom(2,"H",[0,0,1]))
push!(atoms,Atom(3,"H",[0,0,2]))
push!(atoms,Atom(4,"H",[0,0,3]))
push!(atoms,Atom(5,"H",[0,0,4]))
push!(atoms,Atom(6,"H",[0,0,5]))

mol     = Molecule(0,1,atoms)
mf = FermiCG.pyscf_do_scf(mol,"6-31g")

FermiCG.pyscf_write_molden(mol,"6-31g",mf.mo_coeff,filename="scf.molden")

Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf.mol)
FermiCG.pyscf_write_molden(mol,"6-31g",Cl,filename="lowdin.molden")

Cl = FermiCG.localize(mf.mo_coeff,"boys",mf.mol)
FermiCG.pyscf_write_molden(mol,"6-31g",Cl,filename="boys.molden")
