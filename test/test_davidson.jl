using LinearAlgebra
using FermiCG
using Printf
using Test
using LinearMaps
using Arpack
using Random
using Profile 

if true 
@testset "davidson" begin
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
    #push!(atoms,Atom(11,"H",[0,0,12]))
    #push!(atoms,Atom(12,"H",[0,0,13]))
    basis = "6-31g"
    basis = "sto-3g"

    mol     = Molecule(0,1,atoms,basis)
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));

    na = 5
    nb = 5
    nr = 1

    e_mf = mf.e_tot - mf.energy_nuc()
    #if 1==1
    @printf(" Mean-field energy %12.8f", e_mf)
    @time e_fci, d1_fci, d2_fci, v_pyscf = FermiCG.pyscf_fci(ints,na,nb,conv_tol=1e-10)
    #@time e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,na,nb,nroots=nr)
    # @printf(" FCI Energy: %12.8f\n", e_fci)
    #end

    norbs = size(ints.h1)[1]

    problem = StringCI.FCIProblem(norbs, na, nb)
    display(problem)
    v0 = rand(problem.dim,nr)
    v0[:,1] .= 0
    v0[1,1] = 1
    v0 = v0 * inv(sqrt(v0'*v0))

    Hmap = StringCI.get_map(ints, problem)
    Random.seed!(3);
    A = Diagonal(rand(20)) + .0001*rand(20,20)
    A = A'+A


    #davidson = FermiCG.Davidson(A,max_iter=400, nroots=nr, tol=1e-5)
    davidson = FermiCG.Davidson(Hmap,v0=v0,max_iter=80, max_ss_vecs=40, nroots=nr, tol=1e-5)
    Adiag = StringCI.compute_fock_diagonal(problem,mf.mo_energy, e_mf)
    #FermiCG.solve(davidson)
    @printf(" Now iterate: \n")
    flush(stdout)
    #@time FermiCG.iteration(davidson, Adiag=Adiag, iprint=2)
    @time e,v = FermiCG.solve(davidson, Adiag=Adiag);

    @test isapprox(e[1], e_fci, atol=1e-10)
    #@profilehtml FermiCG.solve(davidson, Adiag=Adiag)
    #FermiCG.solve(davidson, Adiag=Diagonal(A))
    

    if 1==1
        problem = StringCI.FCIProblem(norbs, 4, 5)
        e, v = StringCI.do_fci(problem, ints, 1, tol=1e-12);
        rdma, rdmb = StringCI.compute_1rdm(problem, v[:,1], v[:,1]);
        ee, d1a, d1b, d2, ci = FermiCG.pyscf_fci(ints, problem.na, problem.nb);
        display(rdma-d1a)
        display(rdmb-d1b)
        @test isapprox(e[1], ee, atol=1e-10)
        @test isapprox(norm(rdma-d1a), 0, atol=1e-5)
        @test isapprox(norm(rdmb-d1b), 0, atol=1e-5)
    end
        
    #rdm1a, rdm1b, rdm2aa, rdm2bb = StringCI.compute_1rdm(problem, v[:,1], v[:,1]);

end
end


@testset "davidson_rand" begin

    N = 1000
    nr = 8
    A = Diagonal(rand(N)) + .01*rand(N,N)
    A = A'+A


    davidson = FermiCG.Davidson(A,max_iter=400, nroots=nr, tol=1e-6)
    #@time e,v = FermiCG.solve(davidson, Adiag=diag(A));
    @time e,v = FermiCG.solve(davidson);

    @time eref, vref = Arpack.eigs(A, nev=nr, which=:SR)

    @test isapprox(e, eref, atol=1e-10)
    println(size(v), size(vref))
end
