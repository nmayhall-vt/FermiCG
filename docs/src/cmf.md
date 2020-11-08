# Cluster Mean-Field Calculation (CMF)
In this example, we cluster a sequence of H<sub>2</sub> molecules
and solve them self-consistently


### First create a molecule
```julia
using FermiCG

atoms = []
push!(atoms,Atom(1,"H",[0,0,0]))
push!(atoms,Atom(2,"H",[0,0,1]))
push!(atoms,Atom(3,"H",[0,0,2]))
push!(atoms,Atom(4,"H",[0,0,3]))
push!(atoms,Atom(5,"H",[0,0,4]))
push!(atoms,Atom(6,"H",[0,0,5]))
basis = "sto-3g"
```

Now create a PySCF object for creating integrals,
and run FCI with 3 alpha and 3 beta electrons
```julia
mol  = Molecule(0,1,atoms)
mf   = FermiCG.pyscf_do_scf(mol,basis)
ints = FermiCG.pyscf_build_ints(mf.mol,mf.mo_coeff);

na = 3
nb = 3
e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,na,nb)
C = mf.mo_coeff
rdm_mf = C[:,1:2] * C[:,1:2]'
```

Localize the orbitals and print to molden file for viewing
```julia
Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
FermiCG.pyscf_write_molden(mol,basis,Cl,filename="lowdin.molden")
S = FermiCG.get_ovlp(mf)
U =  C' * S * Cl
```

Rotate the integrals to this new localized basis
```julia
ints = FermiCG.orbital_rotation(ints,U)
```

Now we need to specify a clustering
```julia
clusters    = [(1:2),(3:4),(5:6)]
init_fspace = [(1,1),(1,1),(1,1)]

clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)

rdm1 = zeros(size(ints.h1))
rdm1a = rdm_mf*.5
rdm1b = rdm_mf*.5

```

Now run the orbital optimization and dump the resultinging orbitals 
```julia
U = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, verbose=0, gconv=1e-6)

C_cmf = Cl*U

FermiCG.pyscf_write_molden(mol,basis,C_cmf,filename="cmf.molden")
```
