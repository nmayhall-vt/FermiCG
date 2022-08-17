using LinearAlgebra
using FermiCG
using Printf
using Test
using Arpack
using ActiveSpaceSolvers

@testset "S^2" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,0,1]))
    #push!(atoms,Atom(3,"H",[0,0,2]))
    #push!(atoms,Atom(4,"H",[0,0,3]))
    #basis = "6-31g"
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
    
    n_elec_a = 1
    n_elec_b = 1

    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,1,1)
    @printf(" FCI Energy: %12.8f\n", e_fci)


    norb = size(ints.h1)[1]
    ansatz = FCIAnsatz(norb, n_elec_a, n_elec_b)


    display(ansatz)

    @time Hmat = build_H_matrix(ints, ansatz)
    Ssq = build_S2_matrix(ansatz)
    println("Ssq")
    println(Ssq)
    F = eigen(Hmat+0.01*Ssq)
    e = F.values
    v = F.vectors
    e = diag(v'*Hmat*v)
    ss = diag(v'*Ssq*v)

    e = real(e)
    for (i,ei) in enumerate(e)
        @printf(" Energy: %12.8f   <S2>: %12.8f\n",ei+ints.h0,ss[i])
    end
    @test isapprox(e[1], e_fci+ints.h0 , atol=1e-10)
end

