# Coupled Mean-Field Calculation (without orbital optimization)
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
```

Now we need to specify a clustering
