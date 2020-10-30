using NPZ
using JSON
using Random
using LinearAlgebra
using FermiCG


atoms = []
push!(atoms,Atom(1,"H",[0,0,0]))
push!(atoms,Atom(2,"H",[0,0,1]))
push!(atoms,Atom(3,"H",[0,0,5]))
push!(atoms,Atom(4,"H",[0,0,6]))
push!(atoms,Atom(5,"H",[0,0,10]))
push!(atoms,Atom(6,"H",[0,0,11]))

mol     = Molecule(0,1,atoms)
mf = FermiCG.pyscf_do_scf(mol,"6-31g")
ints = FermiCG.pyscf_build_ints(mf.mol,mf.mo_coeff);

FermiCG.pyscf_write_molden(mol,"6-31g",mf.mo_coeff,filename="scf.molden")

# Cl = FermiCG.localize(mf.mo_coeff,"boys", mf)
# FermiCG.pyscf_write_molden(mol,"6-31g",Cl,filename="boys.molden")
C = mf.mo_coeff
Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
FermiCG.pyscf_write_molden(mol,"6-31g",Cl,filename="lowdin.molden")
S = FermiCG.get_ovlp(mf)
U =  C' * S * Cl
ints = FermiCG.orbital_rotation(ints,U)
# ints = FermiCG.pyscf_build_ints(mf.mol,Cl);

clusters    = [(1:4),(5:8),(9:12)]
init_fspace = [(1,1),(1,1),(1,1)]

clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)

for ci in clusters
    display(ints.h1)
    ints_i = FermiCG.subset(ints,ci.orb_list)
    display(ints_i)
end
