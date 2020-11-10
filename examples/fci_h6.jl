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
#basis = "6-31g"
basis = "sto-3g"

mol     = Molecule(0,1,atoms)
mf = FermiCG.pyscf_do_scf(mol,basis)
ints = FermiCG.pyscf_build_ints(mf.mol,mf.mo_coeff);
e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,3,3)
# @printf(" FCI Energy: %12.8f\n", e_fci)


norbs = size(ints.h1)[1]
ket = DeterminantString(norbs, 3)
print(ket)
Helpers.get_nchk(6,3)
ket.config[1] = 5
FermiCG.display(ket)
display(ket.config)

problem = FCIProblem(6, 3, 3)

print(" Compute spin_diagonal terms\n")
@time Hdiag_a = FermiCG.precompute_spin_diag_terms(ints,problem,problem.na)
Hdiag_b = FermiCG.precompute_spin_diag_terms(ints,problem,problem.nb)
print(" done\n")
print(" Kron them")
Hdiag_a = kron(Matrix(1.0I, problem.dimb, problem.dimb), Hdiag_a)
@time Hdiag_b = kron(Hdiag_b, Matrix(1.0I, problem.dima, problem.dima))
print(" done\n")

Hmap = FermiCG.get_map(ints, problem, Hdiag_a, Hdiag_b)

v = zeros(problem.dim,1)
v[1] = 1

#Hmat = .5*(Hmat + transpose(Hmat))
@time e,v = eigs(Hmap, v0=v[:,1], nev = 1, which=:SR)
e = real(e)
for ei in e
    @printf(" Energy: %12.8f\n",ei+ints.h0)
end

#v = zeros(problem.dim,1)
#v[1] = 1
#
#println(v'*v)
#
#sig = Hmap*v
#display(sig'*sig)
##    
##ket_a = DeterminantString(problem.no, problem.na)
##ket_b = DeterminantString(problem.no, problem.nb)
##
##lookup_a = FermiCG.fill_ca_lookup(ket_a)
##lookup_b = FermiCG.fill_ca_lookup(ket_b)
##
##sigma = Array
##function mymatvec!(sigma,vec)
##    sig = FermiCG.compute_ab_terms(vec, ints, problem)
##    sig += Hdiag_a * vec
##    sig += Hdiag_b * vec
##    return sig
##end
##
##mymatvec!(sigma, v)
###mymap = LinearMap(mymatvec!, problem.dim, problem.dim)
###mymap = LinearMap(mymatvec!, problem.dim, problem.dim; issymmetric=true, ishermitian=true)
