using LinearAlgebra
using FermiCG
using Printf
using Test
using Arpack


@testset "fci" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,0,1]))
    #basis = "6-31g"
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
    

    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,1,1)
    @printf(" FCI Energy: %12.8f\n", e_fci)

    n_elec_a = 1
    n_elec_b = 1

    norb = size(ints.h1)[1]
    problem = FermiCG.StringCI.FCIProblem(norb, n_elec_a, n_elec_b)


    display(problem)

    @time Hmat = FermiCG.StringCI.build_H_matrix(ints, problem)
    Ssq = FermiCG.StringCI.build_S2_matrix(problem)
    print("Ssq")
    print(Ssq)
    @time e,v = eigs(Hmat+0.01*Ssq, nev = 10, which=:SR)
    e = diag(v'*Hmat*v)
    ss = diag(v'*Ssq*v)

    e = real(e)
    for (i,ei) in enumerate(e)
        @printf(" Energy: %12.8f   <S2>: %6.4f\n",ei+ints.h0,ss[i])
    end
    @test isapprox(e[1], e_fci , atol=1e-10)
end

