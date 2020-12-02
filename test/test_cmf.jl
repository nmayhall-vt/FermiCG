using LinearAlgebra
using FermiCG
using Printf
using Test

@testset "cmf" begin
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
    mf = FermiCG.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    e_fci, d1_fci, d2_fci = FermiCG.pyscf_fci(ints,3,3)
    # @printf(" FCI Energy: %12.8f\n", e_fci)

    function compute_npairs(d2)
        npairs = 0
        for i in 1:size(d2)[1]
            for j in 1:size(d2)[1]
                npairs += d2[i,i,j,j]
            end
        end
        println(" NPairs:", round(npairs,digits=3))
    end

    compute_npairs(d2_fci)

    FermiCG.pyscf_write_molden(mol,mf.mo_coeff,filename="scf.molden")

    # Cl = FermiCG.localize(mf.mo_coeff,"boys", mf)
    # FermiCG.pyscf_write_molden(mol,Cl,filename="boys.molden")
    C = mf.mo_coeff
    rdm_mf = C[:,1:2] * C[:,1:2]'
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Build Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    clusters    = [(1:2),(3:4),(5:6)]
    #clusters    = [(1:4),(5:8),(9:12)]
    init_fspace = [(1,1),(1,1),(1,1)]

    clusters = [Cluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))
    rdm1a = rdm_mf*.5
    rdm1b = rdm_mf*.5
    
    # test in core method
    f1 = FermiCG.cmf_ci_iteration(ints, clusters, rdm1a, rdm1b, init_fspace, verbose=1)
    @test isapprox(f1[1], -2.876651063218, atol=1e-10)
    
    # test on the fly integral method
    f2 = FermiCG.cmf_ci_iteration(mol, Cl, rdm1a, rdm1b, clusters, init_fspace, verbose=1)
    @test isapprox(f2[1], -2.876651063218, atol=1e-10)

    e_cmf, U = FermiCG.cmf_oo(ints, clusters, init_fspace, rdm1, verbose=0, gconv=1e-6)

    @test isapprox(e_cmf, -3.205983033016, atol=1e-10)
    #FermiCG.pyscf_write_molden(mol,C_cmf,filename="cmf.molden")
end
