using NPZ
using JSON
using Random
using LinearAlgebra
using FermiCG

@testset "hamiltonian stuff" begin
    atoms = []
    push!(atoms,Atom(1,"H",[0,0,0]))
    push!(atoms,Atom(2,"H",[0,0,1]))
    push!(atoms,Atom(3,"H",[0,0,2]))
    push!(atoms,Atom(4,"H",[0,0,3]))

    mol     = Molecule(0,1,atoms)
    mf = FermiCG.pyscf_do_scf(mol,"6-31g")
    @test isapprox(mf.e_tot, -2.16024391299511, atol=1e-10)
    ints = FermiCG.pyscf_build_ints(mf.mol,mf.mo_coeff);
    e,d,dim = FermiCG.pyscf_fci(ints,2,2)
    @test isapprox(e, -2.2251145788392828, atol=1e-10)
end
