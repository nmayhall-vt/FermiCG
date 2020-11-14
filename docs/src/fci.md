# Simple (and slow) FCI Calculation (FCI)
In this example, we cluster a sequence of H<sub>2</sub> molecules


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
Create an FCIProblem object containing problem data
```
norbs = size(ints.h1)[1]

problem = StringCI.FCIProblem(norbs, 4, 4)
```

Now run the CI code. This seems to be about 10x slower than pyscf at the moment,
and only uses lanczos instead of a preconditioned solver 
```
StringCI.run_fci(ints, problem)
```
