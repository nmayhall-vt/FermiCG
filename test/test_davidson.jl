using LinearAlgebra
using FermiCG
using Printf
using Test
using LinearMaps
using Arpack
using Random
using Profile 

#@testset "davidson" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,0,1]))
    push!(atoms,Atom(3,"H",[1,0,2]))
    push!(atoms,Atom(4,"H",[1,0,3]))
    push!(atoms,Atom(5,"H",[2,0,4]))
    push!(atoms,Atom(6,"H",[2,0,5]))
    push!(atoms,Atom(7,"H",[3,0,6]))
    push!(atoms,Atom(8,"H",[3,0,7]))
    push!(atoms,Atom(9,"H",[0,0,8]))
    push!(atoms,Atom(10,"H",[0,0,9]))
    #push!(atoms,Atom(11,"H",[0,0,10]))
    #push!(atoms,Atom(12,"H",[0,0,11]))
    basis = "6-31g"
    basis = "sto-3g"

    mol     = Molecule(0,1,atoms,basis)
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));

    na = 5
    nb = 5

    e_mf = mf.e_tot - mf.energy_nuc()
    if 1==1
        @printf(" Mean-field energy %12.8f", e_mf)
        @time e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,na,nb)
        # @printf(" FCI Energy: %12.8f\n", e_fci)
    end

    norbs = size(ints.h1)[1]

    problem = StringCI.FCIProblem(norbs, na, nb)
    display(problem)
    nr = 1
    v0 = rand(problem.dim,nr)
    v0[:,1] .= 0
    v0[1,1] = 1
    v0 = v0 * inv(sqrt(v0'*v0))

    Hmap = StringCI.get_map(ints, problem)
    Random.seed!(3);
    A = Diagonal(rand(20)) + .0001*rand(20,20)
    A = A'+A


    #davidson = FermiCG.Davidson(A,max_iter=400, nroots=nr, tol=1e-5)
    davidson = FermiCG.Davidson(Hmap,v0=v0,max_iter=40, nroots=nr, tol=1e-5)
    Adiag = StringCI.compute_fock_diagonal(problem,mf.mo_energy, e_mf)
    #FermiCG.solve(davidson)
    @printf(" Now iterate: \n")
    flush(stdout)
    @time FermiCG.solve(davidson, Adiag=Adiag)
    #FermiCG.solve(davidson, Adiag=Diagonal(A))
#end
