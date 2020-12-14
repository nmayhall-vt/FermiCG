using LinearAlgebra
using FermiCG
using Printf
using Test
using LinearMaps
using Arpack
using Random
using Profile 

#@testset "fci_matvec" begin
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
    push!(atoms,Atom(11,"H",[0,0,10]))
    push!(atoms,Atom(12,"H",[0,0,11]))
    push!(atoms,Atom(13,"H",[0,0,12]))
    push!(atoms,Atom(14,"H",[0,0,13]))
    basis = "6-31g"
    basis = "sto-3g"

    mol     = Molecule(0,1,atoms,basis)
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));

    na = 7
    nb = 7

    e_mf = mf.e_tot - mf.energy_nuc()
    if 1==0
        @printf(" Mean-field energy %12.8f", e_mf)
        @time e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,na,na)
        # @printf(" FCI Energy: %12.8f\n", e_fci)
    end

    norbs = size(ints.h1)[1]

    problem = StringCI.FCIProblem(norbs, 1, 1)
    display(problem)
    nr = 1
    v0 = rand(problem.dim,nr)
    #v0[:,1] .= 0
    v0[1,1] = 1
    v0 = v0 * inv(sqrt(v0'*v0))

    Hmap = StringCI.get_map(ints, problem)
    Random.seed!(3);
    A = Diagonal(rand(20)) + .0001*rand(20,20)
    A = A'+A

    function test_matvec(v,prb,n)
        ket_a = FermiCG.StringCI.DeterminantString(prb.no, prb.na)
        ket_b = FermiCG.StringCI.DeterminantString(prb.no, prb.nb)

        lookup_a = FermiCG.StringCI.fill_ca_lookup2(ket_a)
        lookup_b = FermiCG.StringCI.fill_ca_lookup2(ket_b)

        s = similar(v)
        for i=1:n
            println(i)
            flush(stdout)
            v = reshape(v,ket_a.max,ket_b.max, 1)
            @time s = FermiCG.StringCI.compute_ab_terms2(v,ints,prb,lookup_a, lookup_b)
            #@profilehtml s = FermiCG.StringCI.compute_ab_terms2(v,ints,prb,lookup_a, lookup_b)
            #@profilehtml s = FermiCG.StringCI.compute_ss_terms2(v,ints,prb,lookup_a, lookup_b)
            #s = Matrix(H*v)
        end
    end

    test_matvec(v0,problem,1)
    
    problem = StringCI.FCIProblem(norbs, na, nb)
    display(problem)
    nr = 1
    v0 = rand(problem.dim,nr)
    #v0[:,1] .= 0
    v0[1,1] = 1
    v0 = v0 * inv(sqrt(v0'*v0))

    Random.seed!(3);
    A = Diagonal(rand(20)) + .0001*rand(20,20)
    A = A'+A
    @time test_matvec(v0,problem,2)

#end

