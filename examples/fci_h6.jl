using LinearAlgebra
using FermiCG
using Printf
using Test
using LinearMaps
using Arpack

using Profile 

atoms = []
push!(atoms,Atom(1,"H",[0,0,0]))
push!(atoms,Atom(2,"H",[0,0,1]))
push!(atoms,Atom(3,"H",[0,0,2]))
push!(atoms,Atom(4,"H",[0,0,3]))
push!(atoms,Atom(5,"H",[0,0,4]))
push!(atoms,Atom(6,"H",[0,0,5]))
push!(atoms,Atom(7,"H",[0,0,6]))
push!(atoms,Atom(8,"H",[0,0,7]))
#push!(atoms,Atom(9,"H",[0,0,8]))
#push!(atoms,Atom(10,"H",[0,0,9]))
#push!(atoms,Atom(11,"H",[0,0,10]))
#push!(atoms,Atom(12,"H",[0,0,11]))
#basis = "6-31g"
basis = "sto-3g"

mol     = Molecule(0,1,atoms)
mf = FermiCG.pyscf_do_scf(mol,basis)
ints = FermiCG.pyscf_build_ints(mf.mol,mf.mo_coeff);
@time e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,4,4)
# @printf(" FCI Energy: %12.8f\n", e_fci)


norbs = size(ints.h1)[1]

problem = StringCI.FCIProblem(norbs, 4, 4)

display(problem)

v0 = zeros(problem.dim,1)
v0[1] = 1

e, v = StringCI.run_fci(ints, problem, v0=v0)
e, v = StringCI.run_fci(ints, problem, v0=v)
    
ket_a = StringCI.DeterminantString(problem.no, problem.na)
ket_b = StringCI.DeterminantString(problem.no, problem.nb)

v = reshape(v,ket_a.max,ket_b.max)
F = svd(v)
N = 50
v0 = F.U[:,1:N] * Diagonal(F.S[1:N]) * F.Vt[1:N,:]
v0 = reshape(v0,ket_a.max*ket_b.max)
println(v0'*v0)
v0 = v0/norm(v0)
e, v = StringCI.run_fci(ints, problem, v0=v0)
