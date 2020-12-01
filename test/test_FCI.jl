using LinearAlgebra
using FermiCG
using Printf
using Test
using Arpack


@testset "fci" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,0,1]))
    push!(atoms,Atom(3,"H",[0,0,2]))
    push!(atoms,Atom(4,"H",[0,0,3]))
    push!(atoms,Atom(5,"H",[0,0,4]))
    push!(atoms,Atom(6,"H",[0,0,5]))
    #basis = "6-31g"
    basis = "sto-3g"

    mol     = Molecule(0,1,atoms,basis)
    mf = FermiCG.pyscf_do_scf(mol,basis)
    ints = FermiCG.pyscf_build_ints(mf.mol,mf.mo_coeff);
    e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,3,3)
    @printf(" FCI Energy: %12.8f\n", e_fci)

    n_elec_a = 3
    n_elec_b = 3

    norb = size(ints.h1)[1]
    problem = FCIProblem(norb, n_elec_a, n_elec_b)


    display(problem)

    @time Hmat = FermiCG.build_H_matrix(ints, problem)
    @time e,v = eigs(Hmat, nev = 10, which=:SR)
    e = real(e)
    for ei in e
        @printf(" Energy: %12.8f\n",ei+ints.h0)
    end
    @test isapprox(e[1], e_fci , atol=1e-10)
end
